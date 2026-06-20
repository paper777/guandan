from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from server.domain.legal_actions import ActionCandidate
from server.domain.seats import SEATS, Seat
from server.domain.state import MatchPhase
from training.encode import encode_action, encode_observation
from training.env import GuandanTrainingEnv
from training.heuristic import HeuristicPolicy


DATASET_VERSION = 1


@dataclass(frozen=True, slots=True)
class BcSample:
    version: int
    seed: str
    deal_id: int
    event_seq: int
    seat: str
    legal_action: str | None
    observation_names: tuple[str, ...]
    observation_values: tuple[float, ...]
    action_names: tuple[str, ...]
    candidate_values: tuple[tuple[float, ...], ...]
    candidate_payloads: tuple[dict[str, object], ...]
    chosen_index: int
    chosen_payload: dict[str, object]

    def to_json(self) -> dict[str, object]:
        return {
            "version": self.version,
            "seed": self.seed,
            "deal_id": self.deal_id,
            "event_seq": self.event_seq,
            "seat": self.seat,
            "legal_action": self.legal_action,
            "observation_names": list(self.observation_names),
            "observation_values": list(self.observation_values),
            "action_names": list(self.action_names),
            "candidate_values": [list(values) for values in self.candidate_values],
            "candidate_payloads": list(self.candidate_payloads),
            "chosen_index": self.chosen_index,
            "chosen_payload": self.chosen_payload,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "BcSample":
        return cls(
            version=int(payload["version"]),
            seed=str(payload["seed"]),
            deal_id=int(payload["deal_id"]),
            event_seq=int(payload["event_seq"]),
            seat=str(payload["seat"]),
            legal_action=_optional_string(payload.get("legal_action")),
            observation_names=tuple(str(name) for name in _list(payload["observation_names"])),
            observation_values=tuple(float(value) for value in _list(payload["observation_values"])),
            action_names=tuple(str(name) for name in _list(payload["action_names"])),
            candidate_values=tuple(
                tuple(float(value) for value in _list(values))
                for values in _list(payload["candidate_values"])
            ),
            candidate_payloads=tuple(
                dict(item) for item in _list(payload["candidate_payloads"]) if isinstance(item, dict)
            ),
            chosen_index=int(payload["chosen_index"]),
            chosen_payload=dict(payload["chosen_payload"]) if isinstance(payload["chosen_payload"], dict) else {},
        )


@dataclass(frozen=True, slots=True)
class CollectionResult:
    samples: tuple[BcSample, ...]
    completed_deals: int
    steps: int
    stopped_reason: str


def collect_heuristic_samples(
    seeds: Iterable[str | int | bytes | None],
    *,
    max_deals_per_seed: int = 1,
    max_steps_per_seed: int = 20_000,
) -> CollectionResult:
    all_samples: list[BcSample] = []
    completed_deals = 0
    steps = 0
    stopped_reason = "complete"
    for seed in seeds:
        result = collect_match_samples(
            seed=seed,
            max_deals=max_deals_per_seed,
            max_steps=max_steps_per_seed,
        )
        all_samples.extend(result.samples)
        completed_deals += result.completed_deals
        steps += result.steps
        stopped_reason = result.stopped_reason
        if result.stopped_reason not in {"max_deals", "match_complete"}:
            break
    return CollectionResult(tuple(all_samples), completed_deals, steps, stopped_reason)


def collect_match_samples(
    *,
    seed: str | int | bytes | None,
    max_deals: int = 1,
    max_steps: int = 20_000,
    policy: HeuristicPolicy | None = None,
) -> CollectionResult:
    env = GuandanTrainingEnv()
    env.reset(seed=seed)
    teacher = policy or HeuristicPolicy()
    samples: list[BcSample] = []
    completed_deals = 0
    steps = 0
    stopped_reason = "match_complete"
    seed_label = repr(seed)

    while env.state.phase != MatchPhase.MATCH_COMPLETE:
        if steps >= max_steps:
            stopped_reason = "max_steps"
            break

        if env.state.phase == MatchPhase.DEAL_COMPLETE:
            completed_deals += 1
            if completed_deals >= max_deals:
                stopped_reason = "max_deals"
                break
            step = env.start_next_deal(seed=f"{seed_label}:deal:{completed_deals + 1}")
            if step.rejection is not None:
                stopped_reason = f"rejected:{step.rejection.code.value}"
                break
            continue

        actor = env.current_actor()
        if actor is None:
            stopped_reason = f"no_actor:{env.state.phase.value}"
            break

        snapshot = env.observe(actor)
        actions = env.legal_actions(actor)
        chosen = teacher.choose_action(snapshot, actions)
        samples.append(_sample_from_decision(seed_label, snapshot.public.deal_id, actor, snapshot.public.event_seq, snapshot, actions, chosen))
        step = env.step(actor, chosen)
        steps += 1
        if step.rejection is not None:
            stopped_reason = f"rejected:{step.rejection.code.value}"
            break

    if env.state.phase == MatchPhase.MATCH_COMPLETE and env.state.last_deal_result is not None:
        completed_deals += 1

    return CollectionResult(tuple(samples), completed_deals, steps, stopped_reason)


def write_jsonl(samples: Iterable[BcSample], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as output:
        for sample in samples:
            output.write(json.dumps(sample.to_json(), sort_keys=True, separators=(",", ":")))
            output.write("\n")
            count += 1
    return count


def read_jsonl(path: str | Path, *, limit: int | None = None) -> tuple[BcSample, ...]:
    samples: list[BcSample] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            if limit is not None and len(samples) >= limit:
                break
            if line.strip():
                samples.append(BcSample.from_json(json.loads(line)))
    return tuple(samples)


def collect_to_jsonl(
    path: str | Path,
    seeds: Iterable[str | int | bytes | None],
    *,
    max_deals_per_seed: int = 1,
    max_steps_per_seed: int = 20_000,
) -> CollectionResult:
    result = collect_heuristic_samples(
        seeds,
        max_deals_per_seed=max_deals_per_seed,
        max_steps_per_seed=max_steps_per_seed,
    )
    write_jsonl(result.samples, path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    seeds = tuple(args.seed or (f"bc-seed-{index}" for index in range(args.seed_count)))
    result = collect_to_jsonl(
        args.output,
        seeds,
        max_deals_per_seed=args.max_deals,
        max_steps_per_seed=args.max_steps,
    )
    print(
        f"wrote {len(result.samples)} samples from {result.completed_deals} deals "
        f"and {result.steps} steps to {args.output}; stopped_reason={result.stopped_reason}",
        flush=True,
    )
    return 0


def _sample_from_decision(
    seed: str,
    deal_id: int,
    seat: Seat,
    event_seq: int,
    snapshot,
    actions: tuple[ActionCandidate, ...],
    chosen: ActionCandidate,
) -> BcSample:
    observation = encode_observation(snapshot)
    action_vectors = tuple(encode_action(action, snapshot) for action in actions)
    chosen_index = actions.index(chosen)
    action_names = action_vectors[0].names if action_vectors else ()
    return BcSample(
        version=DATASET_VERSION,
        seed=seed,
        deal_id=deal_id,
        event_seq=event_seq,
        seat=seat.value,
        legal_action=snapshot.legal_action,
        observation_names=observation.names,
        observation_values=observation.values,
        action_names=action_names,
        candidate_values=tuple(vector.values for vector in action_vectors),
        candidate_payloads=tuple(action.to_payload() for action in actions),
        chosen_index=chosen_index,
        chosen_payload=chosen.to_payload(),
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Guandan heuristic behavior cloning samples.")
    parser.add_argument("output", help="Output JSONL path.")
    parser.add_argument("--seed", action="append", help="Seed to collect. Can be provided multiple times.")
    parser.add_argument("--seed-count", type=int, default=1, help="Generated seed count when --seed is omitted.")
    parser.add_argument("--max-deals", type=int, default=1, help="Deals per seed.")
    parser.add_argument("--max-steps", type=int, default=20_000, help="Maximum reducer steps per seed.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
