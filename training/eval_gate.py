from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from npc.rl_agent.model_loader import RlAgentConfig, RlModelLoader
from server.domain.hand_types import HandType
from server.domain.legal_actions import ActionCandidate
from server.domain.seats import SEATS, Seat, Team, team_for_seat
from server.domain.state import DealResult, MatchPhase
from server.services.snapshots import SeatSnapshot
from training.env import GuandanTrainingEnv
from training.heuristic import HeuristicPolicy


@dataclass(frozen=True, slots=True)
class EvalGateSummary:
    opponent: str
    seeds: int
    deals: int
    win_rate: float
    average_advance_count: float
    double_down_rate: float
    pass_rate: float
    bomb_rate: float
    illegal_action_rate: float
    stopped_reasons: dict[str, int]

    def to_json(self) -> dict[str, object]:
        return {
            "opponent": self.opponent,
            "seeds": self.seeds,
            "deals": self.deals,
            "win_rate": self.win_rate,
            "average_advance_count": self.average_advance_count,
            "double_down_rate": self.double_down_rate,
            "pass_rate": self.pass_rate,
            "bomb_rate": self.bomb_rate,
            "illegal_action_rate": self.illegal_action_rate,
            "stopped_reasons": dict(self.stopped_reasons),
        }


class CheckpointPolicy:
    def __init__(self, checkpoint_path: Path, *, device: str | None = None) -> None:
        self.loader = RlModelLoader(RlAgentConfig(model_path=checkpoint_path, device=device))
        self.fallback = HeuristicPolicy()

    def choose_action(self, snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...]) -> ActionCandidate:
        selected = self.loader.choose_action(snapshot, actions)
        return selected if selected is not None else self.fallback.choose_action(snapshot, actions)


class DummyTrainingPolicy:
    def choose_action(self, snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...]) -> ActionCandidate:
        if not actions:
            raise ValueError("dummy policy requires at least one legal action")
        if snapshot.legal_action == "play_or_pass":
            return next((action for action in actions if action.kind == "pass"), actions[0])
        if snapshot.legal_action == "lead":
            plays = tuple(action for action in actions if action.kind == "play_cards")
            return min(
                plays or actions,
                key=lambda action: (action.length, action.primary_rank.value if action.primary_rank else "", action.card_ids),
            )
        if snapshot.legal_action == "tribute":
            return actions[0]
        if snapshot.legal_action == "return_tribute":
            return actions[0]
        return actions[0]


def run_checkpoint_eval_gate(
    checkpoint_path: Path,
    *,
    previous_checkpoint_path: Path | None = None,
    seed_count: int = 4,
    max_deals: int = 1,
    max_steps: int = 20_000,
    device: str | None = None,
) -> dict[str, object]:
    candidate = CheckpointPolicy(checkpoint_path, device=device)
    opponents: list[tuple[str, object]] = [
        ("dummy", DummyTrainingPolicy()),
        ("heuristic", HeuristicPolicy()),
    ]
    if previous_checkpoint_path is not None and previous_checkpoint_path.exists():
        opponents.append(("previous", CheckpointPolicy(previous_checkpoint_path, device=device)))
    seeds = tuple(f"eval-gate-{index}" for index in range(seed_count))
    summaries = [
        evaluate_matchup(
            candidate,
            opponent,
            opponent_name=name,
            seeds=seeds,
            max_deals=max_deals,
            max_steps=max_steps,
        )
        for name, opponent in opponents
    ]
    return {
        "checkpoint": str(checkpoint_path),
        "previous_checkpoint": str(previous_checkpoint_path) if previous_checkpoint_path else None,
        "seed_count": seed_count,
        "max_deals": max_deals,
        "max_steps": max_steps,
        "summaries": [summary.to_json() for summary in summaries],
    }


def evaluate_matchup(
    candidate_policy,
    opponent_policy,
    *,
    opponent_name: str,
    seeds: tuple[str, ...],
    max_deals: int,
    max_steps: int,
) -> EvalGateSummary:
    totals = _EvalTotals()
    stopped_reasons: dict[str, int] = {}
    for seed in seeds:
        for candidate_team in (Team.EAST_WEST, Team.SOUTH_NORTH):
            reason = _evaluate_one(
                candidate_policy,
                opponent_policy,
                candidate_team=candidate_team,
                seed=f"{seed}:{candidate_team.value}",
                max_deals=max_deals,
                max_steps=max_steps,
                totals=totals,
            )
            stopped_reasons[reason] = stopped_reasons.get(reason, 0) + 1
    deals = max(totals.deals, 1)
    actions = max(totals.actions, 1)
    return EvalGateSummary(
        opponent=opponent_name,
        seeds=len(seeds),
        deals=totals.deals,
        win_rate=totals.wins / deals,
        average_advance_count=totals.advance_count / deals,
        double_down_rate=totals.double_downs / deals,
        pass_rate=totals.passes / actions,
        bomb_rate=totals.bombs / actions,
        illegal_action_rate=totals.illegal_actions / actions,
        stopped_reasons=stopped_reasons,
    )


@dataclass(slots=True)
class _EvalTotals:
    deals: int = 0
    wins: int = 0
    advance_count: int = 0
    double_downs: int = 0
    actions: int = 0
    passes: int = 0
    bombs: int = 0
    illegal_actions: int = 0


def _evaluate_one(
    candidate_policy,
    opponent_policy,
    *,
    candidate_team: Team,
    seed: str,
    max_deals: int,
    max_steps: int,
    totals: _EvalTotals,
) -> str:
    env = GuandanTrainingEnv()
    env.reset(seed=seed)
    completed_deals = 0
    steps = 0
    seen_result: DealResult | None = None
    while env.state.phase != MatchPhase.MATCH_COMPLETE:
        if steps >= max_steps:
            return "max_steps"
        if env.state.phase == MatchPhase.DEAL_COMPLETE:
            completed_deals += 1
            if completed_deals >= max_deals:
                return "max_deals"
            step = env.start_next_deal(seed=f"{seed}:deal:{completed_deals + 1}")
            if step.rejection is not None:
                totals.illegal_actions += 1
                return f"rejected:{step.rejection.code.value}"
            continue
        actor = env.current_actor()
        if actor is None:
            return f"no_actor:{env.state.phase.value}"
        snapshot = env.observe(actor)
        actions = env.legal_actions(actor)
        policy = candidate_policy if team_for_seat(actor) == candidate_team else opponent_policy
        action = policy.choose_action(snapshot, actions)
        if team_for_seat(actor) == candidate_team:
            _record_candidate_action(totals, action)
        step = env.step(actor, action)
        steps += 1
        if step.rejection is not None:
            totals.illegal_actions += 1
            return f"rejected:{step.rejection.code.value}"
        if step.deal_complete and step.state.last_deal_result is not None and step.state.last_deal_result != seen_result:
            seen_result = step.state.last_deal_result
            _record_deal_result(totals, seen_result, candidate_team)
    if env.state.last_deal_result is not None and env.state.last_deal_result != seen_result:
        _record_deal_result(totals, env.state.last_deal_result, candidate_team)
    return "match_complete"


def _record_candidate_action(totals: _EvalTotals, action: ActionCandidate) -> None:
    totals.actions += 1
    if action.kind == "pass":
        totals.passes += 1
    if action.hand_type in {HandType.BOMB, HandType.STRAIGHT_FLUSH, HandType.FOUR_JOKERS}:
        totals.bombs += 1


def _record_deal_result(totals: _EvalTotals, result: DealResult, candidate_team: Team) -> None:
    totals.deals += 1
    if result.winning_team == candidate_team:
        totals.wins += 1
        totals.advance_count += result.advance_count
        if result.advance_count == 3:
            totals.double_downs += 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_checkpoint_eval_gate(
        Path(args.checkpoint),
        previous_checkpoint_path=Path(args.previous_checkpoint) if args.previous_checkpoint else None,
        seed_count=args.seed_count,
        max_deals=args.max_deals,
        max_steps=args.max_steps,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Guandan checkpoint against fixed policy gates.")
    parser.add_argument("checkpoint")
    parser.add_argument("--previous-checkpoint")
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--max-deals", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--device", default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
