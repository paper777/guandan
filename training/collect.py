from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Iterator
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
SeedValue = str | int | bytes | None
SUCCESS_STOP_REASONS = {"max_deals", "match_complete"}


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

    def to_json(self, *, compact: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": self.version,
            "seed": self.seed,
            "deal_id": self.deal_id,
            "event_seq": self.event_seq,
            "seat": self.seat,
            "legal_action": self.legal_action,
            "observation_values": list(self.observation_values),
            "candidate_values": [list(values) for values in self.candidate_values],
            "chosen_index": self.chosen_index,
        }
        if compact:
            chosen_kind = _optional_string(self.chosen_payload.get("type"))
            if chosen_kind is not None:
                payload["chosen_kind"] = chosen_kind
            return payload
        payload.update(
            {
                "observation_names": list(self.observation_names),
                "action_names": list(self.action_names),
                "candidate_payloads": list(self.candidate_payloads),
                "chosen_payload": self.chosen_payload,
            }
        )
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "BcSample":
        return cls(
            version=int(payload["version"]),
            seed=str(payload["seed"]),
            deal_id=int(payload["deal_id"]),
            event_seq=int(payload["event_seq"]),
            seat=str(payload["seat"]),
            legal_action=_optional_string(payload.get("legal_action")),
            observation_names=tuple(str(name) for name in _list(payload.get("observation_names"))),
            observation_values=tuple(float(value) for value in _list(payload["observation_values"])),
            action_names=tuple(str(name) for name in _list(payload.get("action_names"))),
            candidate_values=tuple(
                tuple(float(value) for value in _list(values))
                for values in _list(payload["candidate_values"])
            ),
            candidate_payloads=tuple(
                dict(item) for item in _list(payload.get("candidate_payloads")) if isinstance(item, dict)
            ),
            chosen_index=int(payload["chosen_index"]),
            chosen_payload=_chosen_payload_from_json(payload),
        )


@dataclass(frozen=True, slots=True)
class CollectionResult:
    samples: tuple[BcSample, ...]
    completed_deals: int
    steps: int
    stopped_reason: str
    sample_count: int | None = None

    @property
    def total_samples(self) -> int:
        return len(self.samples) if self.sample_count is None else self.sample_count


@dataclass(frozen=True, slots=True)
class SeedCollectionResult:
    index: int
    seed: SeedValue
    result: CollectionResult


@dataclass(frozen=True, slots=True)
class ShardCollectionResult:
    index: int
    seed: SeedValue
    path: str
    sample_count: int
    completed_deals: int
    steps: int
    stopped_reason: str


def collect_heuristic_samples(
    seeds: Iterable[SeedValue],
    *,
    max_deals_per_seed: int = 1,
    max_steps_per_seed: int = 20_000,
    workers: int = 1,
) -> CollectionResult:
    seed_values = tuple(seeds)
    seed_results = _collect_seed_results(
        seed_values,
        max_deals_per_seed=max_deals_per_seed,
        max_steps_per_seed=max_steps_per_seed,
        workers=workers,
    )
    return _combine_seed_results(seed_results)


def _collect_seed_results(
    seeds: tuple[SeedValue, ...],
    *,
    max_deals_per_seed: int,
    max_steps_per_seed: int,
    workers: int,
) -> tuple[SeedCollectionResult, ...]:
    _validate_workers(workers)
    for seed in seeds:
        print(f"collect for seed: {seed}", flush=True)
    if workers == 1 or len(seeds) <= 1:
        return tuple(
            _collect_seed_job((index, seed, max_deals_per_seed, max_steps_per_seed))
            for index, seed in enumerate(seeds)
        )

    jobs = tuple(
        (index, seed, max_deals_per_seed, max_steps_per_seed)
        for index, seed in enumerate(seeds)
    )
    with _process_pool(max_workers=min(workers, len(jobs))) as executor:
        return tuple(executor.map(_collect_seed_job, jobs, chunksize=1))


def _collect_seed_job(job: tuple[int, SeedValue, int, int]) -> SeedCollectionResult:
    index, seed, max_deals, max_steps = job
    result = collect_match_samples(
        seed=seed,
        max_deals=max_deals,
        max_steps=max_steps,
    )
    return SeedCollectionResult(index=index, seed=seed, result=result)


def _combine_seed_results(seed_results: tuple[SeedCollectionResult, ...]) -> CollectionResult:
    all_samples: list[BcSample] = []
    completed_deals = 0
    steps = 0
    stopped_reason = "complete"
    for seed_result in sorted(seed_results, key=lambda item: item.index):
        result = seed_result.result
        all_samples.extend(result.samples)
        completed_deals += result.completed_deals
        steps += result.steps
        stopped_reason = result.stopped_reason
        if result.stopped_reason not in SUCCESS_STOP_REASONS:
            break
    return CollectionResult(tuple(all_samples), completed_deals, steps, stopped_reason, sample_count=len(all_samples))


def collect_match_samples(
    *,
    seed: SeedValue,
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
        samples.append(
            _sample_from_decision(
                seed_label,
                snapshot.public.deal_id,
                actor,
                snapshot.public.event_seq,
                snapshot,
                actions,
                chosen,
            )
        )
        step = env.step(actor, chosen)
        steps += 1
        if step.rejection is not None:
            stopped_reason = f"rejected:{step.rejection.code.value}"
            break

    if env.state.phase == MatchPhase.MATCH_COMPLETE and env.state.last_deal_result is not None:
        completed_deals += 1

    return CollectionResult(tuple(samples), completed_deals, steps, stopped_reason, sample_count=len(samples))


def write_jsonl(samples: Iterable[BcSample], path: str | Path, *, compact: bool = False) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _open_text(target, "w") as output:
        for sample in samples:
            output.write(json.dumps(sample.to_json(compact=compact), sort_keys=True, separators=(",", ":")))
            output.write("\n")
            count += 1
    return count


def read_jsonl(path: str | Path, *, limit: int | None = None) -> tuple[BcSample, ...]:
    return tuple(iter_jsonl(path, limit=limit))


def iter_jsonl(path: str | Path, *, limit: int | None = None) -> Iterator[BcSample]:
    count = 0
    with _open_text(Path(path), "r") as source:
        for line in source:
            if limit is not None and count >= limit:
                break
            if line.strip():
                count += 1
                yield BcSample.from_json(json.loads(line))


def collect_to_jsonl(
    path: str | Path,
    seeds: Iterable[SeedValue],
    *,
    max_deals_per_seed: int = 1,
    max_steps_per_seed: int = 20_000,
    workers: int = 1,
    compact: bool = False,
) -> CollectionResult:
    seed_values = tuple(seeds)
    if workers > 1 and len(seed_values) > 1:
        return _collect_to_jsonl_parallel(
            path,
            seed_values,
            max_deals_per_seed=max_deals_per_seed,
            max_steps_per_seed=max_steps_per_seed,
            workers=workers,
            compact=compact,
        )

    result = collect_heuristic_samples(
        seed_values,
        max_deals_per_seed=max_deals_per_seed,
        max_steps_per_seed=max_steps_per_seed,
        workers=workers,
    )
    write_jsonl(result.samples, path, compact=compact)
    return result


def _collect_to_jsonl_parallel(
    path: str | Path,
    seeds: tuple[SeedValue, ...],
    *,
    max_deals_per_seed: int,
    max_steps_per_seed: int,
    workers: int,
    compact: bool,
) -> CollectionResult:
    _validate_workers(workers)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        print(f"collect for seed: {seed}", flush=True)

    with tempfile.TemporaryDirectory(prefix="guandan-bc-") as shard_dir:
        jobs = tuple(
            (index, seed, max_deals_per_seed, max_steps_per_seed, shard_dir, compact)
            for index, seed in enumerate(seeds)
        )
        with _process_pool(max_workers=min(workers, len(jobs))) as executor:
            shard_results = tuple(executor.map(_collect_seed_to_jsonl_job, jobs, chunksize=1))
        return _merge_seed_shards(target, shard_results)


def _collect_seed_to_jsonl_job(job: tuple[int, SeedValue, int, int, str, bool]) -> ShardCollectionResult:
    index, seed, max_deals, max_steps, shard_dir, compact = job
    result = collect_match_samples(seed=seed, max_deals=max_deals, max_steps=max_steps)
    shard_path = Path(shard_dir) / f"seed-{index:06d}.jsonl"
    sample_count = write_jsonl(result.samples, shard_path, compact=compact)
    return ShardCollectionResult(
        index=index,
        seed=seed,
        path=str(shard_path),
        sample_count=sample_count,
        completed_deals=result.completed_deals,
        steps=result.steps,
        stopped_reason=result.stopped_reason,
    )


def _merge_seed_shards(path: Path, shard_results: tuple[ShardCollectionResult, ...]) -> CollectionResult:
    completed_deals = 0
    steps = 0
    sample_count = 0
    stopped_reason = "complete"
    with _open_text(path, "w") as output:
        for shard in sorted(shard_results, key=lambda item: item.index):
            _copy_text_file(Path(shard.path), output)
            completed_deals += shard.completed_deals
            steps += shard.steps
            sample_count += shard.sample_count
            stopped_reason = shard.stopped_reason
            if shard.stopped_reason not in SUCCESS_STOP_REASONS:
                break
    return CollectionResult((), completed_deals, steps, stopped_reason, sample_count=sample_count)


def _copy_text_file(source: Path, output) -> None:
    with _open_text(source, "r") as input_file:
        shutil.copyfileobj(input_file, output)


def _validate_workers(workers: int) -> None:
    if workers < 1:
        raise ValueError("workers must be at least 1")


def _process_pool(*, max_workers: int) -> ProcessPoolExecutor:
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:
        return ProcessPoolExecutor(max_workers=max_workers)
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=context)


def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        text_mode = mode if "t" in mode else f"{mode}t"
        return gzip.open(path, text_mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    seeds = tuple(args.seed or (f"bc-seed-{index}" for index in range(args.seed_count)))
    result = collect_to_jsonl(
        args.output,
        seeds,
        max_deals_per_seed=args.max_deals,
        max_steps_per_seed=args.max_steps,
        workers=args.workers,
        compact=args.compact,
    )
    print(
        f"wrote {result.total_samples} samples from {result.completed_deals} deals "
        f"and {result.steps} steps to {args.output}; stopped_reason={result.stopped_reason}; "
        f"workers={args.workers}; compact={args.compact}",
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


def _chosen_payload_from_json(payload: dict[str, object]) -> dict[str, object]:
    chosen_payload = payload.get("chosen_payload")
    if isinstance(chosen_payload, dict):
        return dict(chosen_payload)
    chosen_kind = _optional_string(payload.get("chosen_kind"))
    return {"type": chosen_kind} if chosen_kind is not None else {}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Guandan heuristic behavior cloning samples.")
    parser.add_argument("output", help="Output JSONL path.")
    parser.add_argument("--seed", action="append", help="Seed to collect. Can be provided multiple times.")
    parser.add_argument("--seed-count", type=int, default=1, help="Generated seed count when --seed is omitted.")
    parser.add_argument("--max-deals", type=int, default=1, help="Deals per seed.")
    parser.add_argument("--max-steps", type=int, default=20_000, help="Maximum reducer steps per seed.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel seed workers.")
    parser.add_argument("--compact", action="store_true", help="Omit debug names and action payloads from samples.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
