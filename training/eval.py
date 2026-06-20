from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from server.domain.legal_actions import ActionCandidate
from server.domain.seats import SEATS, Seat
from server.domain.state import DealResult, MatchPhase
from server.services.snapshots import SeatSnapshot
from training.env import GuandanTrainingEnv
from training.heuristic import HeuristicPolicy


class Policy(Protocol):
    def choose_action(self, snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...]) -> ActionCandidate:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    completed_deals: int
    steps: int
    rewards: dict[Seat, float]
    final_phase: MatchPhase
    final_result: DealResult | None
    stopped_reason: str

    @property
    def match_complete(self) -> bool:
        return self.final_phase == MatchPhase.MATCH_COMPLETE


def evaluate_match(
    policies: Mapping[Seat, Policy],
    *,
    seed: str | int | bytes | None = None,
    max_steps: int = 20_000,
    max_deals: int | None = None,
) -> EvaluationResult:
    env = GuandanTrainingEnv()
    env.reset(seed=seed)
    rewards = {seat: 0.0 for seat in SEATS}
    steps = 0
    completed_deals = 0
    stopped_reason = "match_complete"

    while env.state.phase != MatchPhase.MATCH_COMPLETE:
        if steps >= max_steps:
            stopped_reason = "max_steps"
            break

        if env.state.phase == MatchPhase.DEAL_COMPLETE:
            completed_deals += 1
            if max_deals is not None and completed_deals >= max_deals:
                stopped_reason = "max_deals"
                break
            next_seed = _next_deal_seed(seed, completed_deals + 1)
            step = env.start_next_deal(seed=next_seed)
            if step.rejection is not None:
                stopped_reason = f"rejected:{step.rejection.code.value}"
                break
            continue

        actor = env.current_actor()
        if actor is None:
            stopped_reason = f"no_actor:{env.state.phase.value}"
            break

        actions = env.legal_actions(actor)
        snapshot = env.observe(actor)
        action = policies[actor].choose_action(snapshot, actions)
        step = env.step(actor, action)
        steps += 1
        for seat, reward in step.rewards.items():
            rewards[seat] += reward
        if step.rejection is not None:
            stopped_reason = f"rejected:{step.rejection.code.value}"
            break

    if env.state.phase == MatchPhase.MATCH_COMPLETE and env.state.last_deal_result is not None:
        completed_deals += 1

    return EvaluationResult(
        completed_deals=completed_deals,
        steps=steps,
        rewards=rewards,
        final_phase=env.state.phase,
        final_result=env.state.last_deal_result,
        stopped_reason=stopped_reason,
    )


def evaluate_heuristic_match(
    *,
    seed: str | int | bytes | None = None,
    max_steps: int = 20_000,
    max_deals: int | None = None,
) -> EvaluationResult:
    policy = HeuristicPolicy()
    return evaluate_match(
        {seat: policy for seat in SEATS},
        seed=seed,
        max_steps=max_steps,
        max_deals=max_deals,
    )


def _next_deal_seed(seed: str | int | bytes | None, deal_number: int) -> str:
    return f"{seed!r}:deal:{deal_number}"
