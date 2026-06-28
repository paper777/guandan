from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

from server.domain.legal_actions import ActionCandidate
from server.domain.seats import SEATS
from server.services.snapshots import SeatSnapshot
from training.encode import ENCODING_SCHEMA_VERSION, encode_action, encode_observation, encoding_schema, validate_encoding_schema
from training.model import build_candidate_actor_critic, pair_feature_dim, require_torch
from training.rollout import RolloutDecision, RolloutResult, RolloutTransition, collect_rollout


@dataclass(frozen=True, slots=True)
class PpoConfig:
    output_path: Path
    init_policy_path: Path | None = None
    rollout_seeds: tuple[str, ...] = ("ppo-seed-0",)
    updates: int = 1
    epochs_per_update: int = 3
    max_deals_per_seed: int = 1
    max_steps_per_seed: int = 20_000
    learning_rate: float = 1e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.1
    entropy_coef: float = 0.003
    value_coef: float = 0.5
    batch_size: int = 256
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    normalize_advantages: bool = True
    hidden_dim: int = 256
    dropout: float = 0.0
    seed: int = 1
    device: str | None = None
    log_updates: bool = True


@dataclass(frozen=True, slots=True)
class PpoSummary:
    updates: int
    transitions: int
    final_loss: float
    output_path: Path
    init_policy_path: Path | None = None
    init_checkpoint_kind: str | None = None


@dataclass(frozen=True, slots=True)
class RolloutMetrics:
    completed_deals: int
    steps: int
    stopped_reasons: dict[str, int]


def train_self_play_ppo(config: PpoConfig) -> PpoSummary:
    torch = require_torch()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    schema_version = _schema_version_for_training(torch, config.init_policy_path, device)
    observation_dim, action_dim = _initial_dimensions(config.rollout_seeds[0], schema_version=schema_version)
    model = build_candidate_actor_critic(
        pair_feature_dim(observation_dim, action_dim),
        observation_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    init_checkpoint_kind: str | None = None
    if config.init_policy_path is not None:
        init_checkpoint_kind = _initialize_model_from_checkpoint(
            torch,
            model,
            config.init_policy_path,
            observation_dim,
            action_dim,
            config.hidden_dim,
            device,
            schema_version,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    final_loss = 0.0
    total_transitions = 0
    if config.log_updates:
        init_note = (
            f" init_checkpoint={config.init_policy_path} kind={init_checkpoint_kind}"
            if config.init_policy_path
            else ""
        )
        print(
            f"ppo config device={device} seeds={len(config.rollout_seeds)} updates={config.updates} "
            f"epochs_per_update={config.epochs_per_update} max_deals={config.max_deals_per_seed} "
            f"max_steps={config.max_steps_per_seed} batch_size={config.batch_size} lr={config.learning_rate:g} "
            f"gamma={config.gamma:g} gae_lambda={config.gae_lambda:g} clip={config.clip_epsilon:g} "
            f"entropy_coef={config.entropy_coef:g} target_kl={config.target_kl:g}{init_note}",
            flush=True,
        )

    for update_index in range(config.updates):
        update_started_at = time.perf_counter()
        rollout_started_at = update_started_at
        policy = TorchRolloutPolicy(torch, model, device, schema_version)
        transitions: list[RolloutTransition] = []
        rollouts: list[RolloutResult] = []
        for seed in config.rollout_seeds:
            rollout = collect_rollout(
                {seat: policy for seat in SEATS},
                seed=f"{seed}:update:{update_index}",
                max_deals=config.max_deals_per_seed,
                max_steps=config.max_steps_per_seed,
            )
            if rollout.stopped_reason not in {"max_deals", "match_complete"}:
                raise RuntimeError(f"rollout stopped unexpectedly: {rollout.stopped_reason}")
            rollouts.append(rollout)
            transitions.extend(rollout.transitions)
        rollout_elapsed = time.perf_counter() - rollout_started_at
        total_transitions += len(transitions)
        returns, advantages = _gae_returns_and_advantages(
            tuple(transitions),
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            normalize=config.normalize_advantages,
        )
        mean_return = sum(returns) / len(returns) if returns else 0.0
        if config.log_updates:
            rollout_metrics = _rollout_metrics(rollouts)
            print(
                f"ppo update={update_index + 1}/{config.updates} rollout "
                f"seeds={len(rollouts)} transitions={len(transitions)} "
                f"deals={rollout_metrics.completed_deals} steps={rollout_metrics.steps} "
                f"stops={_format_stop_counts(rollout_metrics.stopped_reasons)} "
                f"mean_return={mean_return:.4f} elapsed={rollout_elapsed:.2f}s",
                flush=True,
            )
        mean_kl = 0.0
        train_elapsed = 0.0
        if transitions:
            model.train()
            indices = list(range(len(transitions)))
            train_started_at = time.perf_counter()
            for epoch_index in range(config.epochs_per_update):
                random.Random(f"{config.seed}:{update_index}:{epoch_index}").shuffle(indices)
                total_loss = 0.0
                total_kl = 0.0
                batches = 0
                for batch_indices in _iter_batches(indices, config.batch_size):
                    batch_transitions = tuple(transitions[index] for index in batch_indices)
                    batch_returns = tuple(returns[index] for index in batch_indices)
                    batch_advantages = tuple(advantages[index] for index in batch_indices)
                    loss, approx_kl = _ppo_batch_loss(
                        torch,
                        model,
                        batch_transitions,
                        batch_returns,
                        batch_advantages,
                        config,
                        device,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    if config.max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    total_loss += float(loss.detach().cpu())
                    total_kl += approx_kl
                    batches += 1
                final_loss = total_loss / max(batches, 1)
                mean_kl = total_kl / max(batches, 1)
                early_stop = config.target_kl > 0.0 and mean_kl > config.target_kl
                if config.log_updates:
                    stop_note = " early_stop=target_kl" if early_stop else ""
                    print(
                        f"ppo update={update_index + 1}/{config.updates} "
                        f"epoch={epoch_index + 1}/{config.epochs_per_update} "
                        f"batches={batches} loss={final_loss:.4f} kl={mean_kl:.5f}{stop_note}",
                        flush=True,
                    )
                if early_stop:
                    break
            train_elapsed = time.perf_counter() - train_started_at
        if config.log_updates:
            update_elapsed = time.perf_counter() - update_started_at
            print(
                f"ppo update={update_index + 1}/{config.updates} done "
                f"transitions={len(transitions)} loss={final_loss:.4f} "
                f"mean_return={mean_return:.4f} kl={mean_kl:.5f} "
                f"total_transitions={total_transitions} train_elapsed={train_elapsed:.2f}s "
                f"elapsed={update_elapsed:.2f}s",
                flush=True,
            )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "encoding_schema": encoding_schema(schema_version),
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "updates": config.updates,
            "transitions": total_transitions,
            "final_loss": final_loss,
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "clip_epsilon": config.clip_epsilon,
            "entropy_coef": config.entropy_coef,
            "value_coef": config.value_coef,
            "batch_size": config.batch_size,
            "max_grad_norm": config.max_grad_norm,
            "target_kl": config.target_kl,
            "normalize_advantages": config.normalize_advantages,
            "init_policy_path": str(config.init_policy_path) if config.init_policy_path else None,
            "init_checkpoint_kind": init_checkpoint_kind,
        },
        config.output_path,
    )
    return PpoSummary(
        config.updates,
        total_transitions,
        final_loss,
        config.output_path,
        config.init_policy_path,
        init_checkpoint_kind,
    )


class TorchRolloutPolicy:
    def __init__(self, torch, model, device, schema_version: str = ENCODING_SCHEMA_VERSION) -> None:
        self.torch = torch
        self.model = model
        self.device = device
        self.schema_version = schema_version

    def choose_decision(self, snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...]) -> RolloutDecision:
        was_training = self.model.training
        self.model.eval()
        try:
            with self.torch.no_grad():
                logits, value = _policy_logits_and_value(
                    self.torch,
                    self.model,
                    snapshot,
                    actions,
                    self.device,
                    self.schema_version,
                )
                distribution = self.torch.distributions.Categorical(logits=logits)
                index = int(distribution.sample().item())
                log_prob = float(distribution.log_prob(self.torch.tensor(index, device=self.device)).detach().cpu())
                value_item = float(value.detach().cpu())
        finally:
            if was_training:
                self.model.train()
        return RolloutDecision(actions[index], index, log_prob=log_prob, value=value_item)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        rollout_seeds = _rollout_seeds_from_args(args.seed, args.seed_count)
    except ValueError as exc:
        parser.error(str(exc))
    summary = train_self_play_ppo(
        PpoConfig(
            output_path=Path(args.output),
            init_policy_path=Path(args.init_policy) if args.init_policy else None,
            rollout_seeds=rollout_seeds,
            updates=args.updates,
            epochs_per_update=args.epochs_per_update,
            max_deals_per_seed=args.max_deals,
            max_steps_per_seed=args.max_steps,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            batch_size=args.batch_size,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            normalize_advantages=not args.no_normalize_advantages,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            seed=args.torch_seed,
            device=args.device,
            log_updates=not args.quiet,
        )
    )
    init_note = (
        f" init_checkpoint={summary.init_policy_path} kind={summary.init_checkpoint_kind}"
        if summary.init_policy_path
        else ""
    )
    print(
        f"ppo updates={summary.updates} transitions={summary.transitions} "
        f"loss={summary.final_loss:.4f}{init_note}; wrote {summary.output_path}",
        flush=True,
    )
    return 0


def _initialize_model_from_checkpoint(
    torch,
    model,
    checkpoint_path: Path,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
    device,
    schema_version: str,
) -> str:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    kind, model_state = _initial_model_state_from_checkpoint(
        checkpoint,
        observation_dim,
        action_dim,
        hidden_dim,
        schema_version=schema_version,
    )
    if kind == "ppo":
        model.load_state_dict(model_state)
    else:
        model.policy_net.load_state_dict(model_state)
    return kind


def _policy_state_from_bc_checkpoint(
    checkpoint: object,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
) -> dict[str, object]:
    kind, model_state = _initial_model_state_from_checkpoint(
        checkpoint,
        observation_dim,
        action_dim,
        hidden_dim,
        schema_version=ENCODING_SCHEMA_VERSION,
    )
    if kind != "bc":
        raise ValueError("checkpoint is not a BC ranker checkpoint")
    return model_state


def _initial_model_state_from_checkpoint(
    checkpoint: object,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
    *,
    schema_version: str | None = None,
) -> tuple[str, dict[str, object]]:
    if not isinstance(checkpoint, dict):
        raise ValueError("initial checkpoint must be a dictionary")
    _validate_checkpoint_dim(checkpoint, "observation_dim", observation_dim)
    _validate_checkpoint_dim(checkpoint, "action_dim", action_dim)
    _validate_checkpoint_dim(checkpoint, "hidden_dim", hidden_dim)
    validate_encoding_schema(checkpoint, schema_version=schema_version)
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("initial checkpoint is missing model_state")
    if any(str(key).startswith("policy_net.") for key in model_state):
        return "ppo", model_state
    policy_state = _bc_ranker_state_to_policy_state(model_state)
    if policy_state:
        return "bc", policy_state
    raise ValueError("initial checkpoint model_state is neither PPO actor-critic nor BC ranker")


def _validate_checkpoint_dim(checkpoint: dict[str, object], key: str, expected: int) -> None:
    actual = checkpoint.get(key)
    if actual != expected:
        raise ValueError(f"initial checkpoint {key}={actual!r} does not match PPO {key}={expected!r}")


def _bc_ranker_state_to_policy_state(model_state: dict[str, object]) -> dict[str, object]:
    return {
        str(key).removeprefix("net."): value
        for key, value in model_state.items()
        if str(key).startswith("net.")
    }


def _rollout_metrics(rollouts: list[RolloutResult]) -> RolloutMetrics:
    stopped_reasons: dict[str, int] = {}
    for rollout in rollouts:
        stopped_reasons[rollout.stopped_reason] = stopped_reasons.get(rollout.stopped_reason, 0) + 1
    return RolloutMetrics(
        completed_deals=sum(rollout.completed_deals for rollout in rollouts),
        steps=sum(rollout.steps for rollout in rollouts),
        stopped_reasons=stopped_reasons,
    )


def _format_stop_counts(stopped_reasons: dict[str, int]) -> str:
    if not stopped_reasons:
        return "none"
    return ",".join(f"{reason}:{count}" for reason, count in sorted(stopped_reasons.items()))


def _ppo_batch_loss(
    torch,
    model,
    transitions: tuple[RolloutTransition, ...],
    returns: tuple[float, ...],
    advantages: tuple[float, ...],
    config: PpoConfig,
    device,
):
    logits, values, mask = _transition_batch_logits_and_values(torch, model, transitions, device)
    log_probs = torch.log_softmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)
    action_indices = torch.tensor(
        [transition.action_index for transition in transitions],
        dtype=torch.long,
        device=device,
    )
    new_log_probs = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
    old_log_probs = torch.tensor(
        [transition.old_log_prob for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    target_returns = torch.tensor(returns, dtype=torch.float32, device=device)
    advantage_values = torch.tensor(advantages, dtype=torch.float32, device=device)
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
    policy_loss = -torch.minimum(ratio * advantage_values, clipped * advantage_values).mean()
    value_loss = (values - target_returns).pow(2).mean()
    entropy = -(probs * log_probs.masked_fill(~mask, 0.0)).sum(dim=1).mean()
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
    approx_kl = ((ratio - 1.0) - log_ratio).mean().detach()
    return loss, float(approx_kl.cpu())


def _gae_returns_and_advantages(
    transitions: tuple[RolloutTransition, ...],
    *,
    gamma: float,
    gae_lambda: float,
    normalize: bool,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    returns = [0.0 for _ in transitions]
    advantages = [0.0 for _ in transitions]
    next_advantage_by_seat = {seat.value: 0.0 for seat in SEATS}
    next_value_by_seat = {seat.value: 0.0 for seat in SEATS}
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        if transition.done:
            next_advantage = 0.0
            next_value = 0.0
        else:
            next_advantage = next_advantage_by_seat[transition.seat]
            next_value = next_value_by_seat[transition.seat]
        delta = transition.reward + gamma * next_value - transition.value
        advantage = delta + gamma * gae_lambda * next_advantage
        advantages[index] = advantage
        returns[index] = advantage + transition.value
        next_advantage_by_seat[transition.seat] = advantage
        next_value_by_seat[transition.seat] = transition.value
    if normalize and len(advantages) > 1:
        mean = sum(advantages) / len(advantages)
        variance = sum((advantage - mean) ** 2 for advantage in advantages) / len(advantages)
        std = variance**0.5
        if std > 1e-8:
            advantages = [(advantage - mean) / std for advantage in advantages]
    return tuple(returns), tuple(advantages)


def _iter_batches(indices: list[int], batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _schema_version_for_training(torch, checkpoint_path: Path | None, device) -> str:
    if checkpoint_path is None:
        return ENCODING_SCHEMA_VERSION
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("initial checkpoint must be a dictionary")
    return validate_encoding_schema(checkpoint)


def _initial_dimensions(seed: str, *, schema_version: str = ENCODING_SCHEMA_VERSION) -> tuple[int, int]:
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
    observation_dim = len(encode_observation(snapshot, schema_version=schema_version).values)
    action_dim = len(encode_action(actions[0], snapshot, schema_version=schema_version).values)
    return observation_dim, action_dim


def _policy_logits_and_value(
    torch,
    model,
    snapshot: SeatSnapshot,
    actions: tuple[ActionCandidate, ...],
    device,
    schema_version: str = ENCODING_SCHEMA_VERSION,
):
    observation = torch.tensor(
        encode_observation(snapshot, schema_version=schema_version).values,
        dtype=torch.float32,
        device=device,
    )
    action_values = [encode_action(action, snapshot, schema_version=schema_version).values for action in actions]
    action_features = torch.tensor(action_values, dtype=torch.float32, device=device)
    observations = observation.expand(action_features.shape[0], -1)
    pair_features = torch.cat((observations, action_features), dim=1)
    return model.policy_logits(pair_features), model.value(observation)


def _transition_batch_logits_and_values(
    torch,
    model,
    transitions: tuple[RolloutTransition, ...],
    device,
):
    observations = torch.tensor(
        [transition.observation_values for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    counts = torch.tensor(
        [len(transition.candidate_values) for transition in transitions],
        dtype=torch.long,
        device=device,
    )
    flat_actions = [
        candidate_values
        for transition in transitions
        for candidate_values in transition.candidate_values
    ]
    action_features = torch.tensor(flat_actions, dtype=torch.float32, device=device)
    pair_observations = observations.repeat_interleave(counts, dim=0)
    flat_logits = model.policy_logits(torch.cat((pair_observations, action_features), dim=1))
    max_candidates = int(counts.max().item())
    logits = torch.full((len(transitions), max_candidates), -torch.inf, dtype=torch.float32, device=device)
    offset = 0
    for row, count_item in enumerate(counts.tolist()):
        logits[row, :count_item] = flat_logits[offset : offset + count_item]
        offset += count_item
    candidate_positions = torch.arange(max_candidates, device=device).unsqueeze(0)
    mask = candidate_positions < counts.unsqueeze(1)
    return logits, model.value(observations), mask


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Guandan candidate policy with self-play PPO.")
    parser.add_argument("output", help="Output checkpoint path.")
    parser.add_argument(
        "--init-policy",
        "--init-model",
        dest="init_policy",
        help="BC ranker or trained PPO actor-critic checkpoint used to initialize PPO training.",
    )
    parser.add_argument("--seed", action="append", help="Rollout seed. Can be repeated.")
    parser.add_argument("--seed-count", type=int, help="Generate rollout seeds ppo-seed-0..N-1.")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--epochs-per-update", type=int, default=3)
    parser.add_argument("--max-deals", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--no-normalize-advantages", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--torch-seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true", help="Disable training progress output.")
    return parser


def _rollout_seeds_from_args(seed_args: list[str] | None, seed_count: int | None) -> tuple[str, ...]:
    if seed_args and seed_count is not None:
        raise ValueError("--seed-count cannot be combined with --seed")
    if seed_args:
        return tuple(seed_args)
    if seed_count is None:
        return ("ppo-seed-0",)
    if seed_count <= 0:
        raise ValueError("--seed-count must be positive")
    return tuple(f"ppo-seed-{index}" for index in range(seed_count))


if __name__ == "__main__":
    raise SystemExit(main())
