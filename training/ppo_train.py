from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from server.domain.legal_actions import ActionCandidate
from server.domain.seats import SEATS
from server.services.snapshots import SeatSnapshot
from training.encode import encode_action, encode_observation
from training.model import build_candidate_actor_critic, pair_feature_dim, require_torch
from training.rollout import RolloutDecision, RolloutResult, collect_rollout, discounted_returns


@dataclass(frozen=True, slots=True)
class PpoConfig:
    output_path: Path
    rollout_seeds: tuple[str, ...] = ("ppo-seed-0",)
    updates: int = 1
    epochs_per_update: int = 1
    max_deals_per_seed: int = 1
    max_steps_per_seed: int = 20_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    hidden_dim: int = 256
    dropout: float = 0.1
    seed: int = 1
    device: str | None = None


@dataclass(frozen=True, slots=True)
class PpoSummary:
    updates: int
    transitions: int
    final_loss: float
    output_path: Path


def train_self_play_ppo(config: PpoConfig) -> PpoSummary:
    torch = require_torch()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    observation_dim, action_dim = _initial_dimensions(config.rollout_seeds[0])
    model = build_candidate_actor_critic(
        pair_feature_dim(observation_dim, action_dim),
        observation_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    final_loss = 0.0
    total_transitions = 0

    for update_index in range(config.updates):
        policy = TorchRolloutPolicy(torch, model, device)
        transitions = []
        for seed in config.rollout_seeds:
            rollout = collect_rollout(
                {seat: policy for seat in SEATS},
                seed=f"{seed}:update:{update_index}",
                max_deals=config.max_deals_per_seed,
                max_steps=config.max_steps_per_seed,
            )
            if rollout.stopped_reason not in {"max_deals", "match_complete"}:
                raise RuntimeError(f"rollout stopped unexpectedly: {rollout.stopped_reason}")
            transitions.extend(rollout.transitions)
        total_transitions += len(transitions)
        returns = discounted_returns(tuple(transitions), gamma=config.gamma)
        for _epoch in range(config.epochs_per_update):
            total_loss = 0.0
            for transition, target_return in zip(transitions, returns, strict=True):
                loss = _ppo_loss(torch, model, transition, target_return, config, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
            final_loss = total_loss / max(len(transitions), 1)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "updates": config.updates,
            "transitions": total_transitions,
            "final_loss": final_loss,
        },
        config.output_path,
    )
    return PpoSummary(config.updates, total_transitions, final_loss, config.output_path)


class TorchRolloutPolicy:
    def __init__(self, torch, model, device) -> None:
        self.torch = torch
        self.model = model
        self.device = device

    def choose_decision(self, snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...]) -> RolloutDecision:
        with self.torch.no_grad():
            logits, value = _policy_logits_and_value(self.torch, self.model, snapshot, actions, self.device)
            distribution = self.torch.distributions.Categorical(logits=logits)
            index = int(distribution.sample().item())
            log_prob = float(distribution.log_prob(self.torch.tensor(index, device=self.device)).detach().cpu())
            value_item = float(value.detach().cpu())
        return RolloutDecision(actions[index], index, log_prob=log_prob, value=value_item)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rollout_seeds = tuple(args.seed or ("ppo-seed-0",))
    summary = train_self_play_ppo(
        PpoConfig(
            output_path=Path(args.output),
            rollout_seeds=rollout_seeds,
            updates=args.updates,
            epochs_per_update=args.epochs_per_update,
            max_deals_per_seed=args.max_deals,
            max_steps_per_seed=args.max_steps,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            clip_epsilon=args.clip_epsilon,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            seed=args.torch_seed,
            device=args.device,
        )
    )
    print(
        f"ppo updates={summary.updates} transitions={summary.transitions} "
        f"loss={summary.final_loss:.4f}; wrote {summary.output_path}",
        flush=True,
    )
    return 0


def _ppo_loss(torch, model, transition, target_return: float, config: PpoConfig, device):
    logits, value = _transition_logits_and_value(torch, model, transition, device)
    log_probs = torch.log_softmax(logits, dim=0)
    probs = torch.softmax(logits, dim=0)
    action_index = torch.tensor(transition.action_index, dtype=torch.long, device=device)
    new_log_prob = log_probs[action_index]
    old_log_prob = torch.tensor(transition.old_log_prob, dtype=torch.float32, device=device)
    target = torch.tensor(target_return, dtype=torch.float32, device=device)
    advantage = target - torch.tensor(transition.value, dtype=torch.float32, device=device)
    ratio = torch.exp(new_log_prob - old_log_prob)
    clipped = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
    policy_loss = -torch.minimum(ratio * advantage, clipped * advantage)
    value_loss = (value - target).pow(2)
    entropy = -(probs * log_probs).sum()
    return policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy


def _initial_dimensions(seed: str) -> tuple[int, int]:
    from training.env import GuandanTrainingEnv

    env = GuandanTrainingEnv()
    env.reset(seed=seed)
    actor = env.current_actor()
    if actor is None:
        raise RuntimeError("initial training environment has no actor")
    snapshot = env.observe(actor)
    actions = env.legal_actions(actor)
    if not actions:
        raise RuntimeError("initial training environment has no legal actions")
    observation_dim = len(encode_observation(snapshot).values)
    action_dim = len(encode_action(actions[0], snapshot).values)
    return observation_dim, action_dim


def _policy_logits_and_value(torch, model, snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...], device):
    observation = torch.tensor(encode_observation(snapshot).values, dtype=torch.float32, device=device)
    action_values = [encode_action(action, snapshot).values for action in actions]
    action_features = torch.tensor(action_values, dtype=torch.float32, device=device)
    observations = observation.expand(action_features.shape[0], -1)
    pair_features = torch.cat((observations, action_features), dim=1)
    return model.policy_logits(pair_features), model.value(observation)


def _transition_logits_and_value(torch, model, transition, device):
    observation = torch.tensor(transition.observation_values, dtype=torch.float32, device=device)
    action_features = torch.tensor(transition.candidate_values, dtype=torch.float32, device=device)
    observations = observation.expand(action_features.shape[0], -1)
    pair_features = torch.cat((observations, action_features), dim=1)
    return model.policy_logits(pair_features), model.value(observation)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Guandan candidate policy with self-play PPO.")
    parser.add_argument("output", help="Output checkpoint path.")
    parser.add_argument("--seed", action="append", help="Rollout seed. Can be repeated.")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--epochs-per-update", type=int, default=1)
    parser.add_argument("--max-deals", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--torch-seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
