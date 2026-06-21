from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.collect import BcSample, iter_jsonl
from training.model import require_torch


CACHE_FORMAT = "guandan-bc-tensor-shards"
CACHE_VERSION = 1
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class TensorCache:
    cache_dir: Path
    manifest: dict[str, Any]
    built: bool = False


def ensure_tensor_cache(
    torch,
    dataset_path: Path,
    cache_dir: Path,
    *,
    limit: int | None = None,
    shard_size: int = 2048,
    force: bool = False,
) -> TensorCache:
    if not force and is_tensor_cache_dir(cache_dir):
        cache = load_tensor_cache(cache_dir)
        if _manifest_matches_source(cache.manifest, dataset_path, limit=limit):
            return cache
    return build_tensor_cache(
        torch,
        dataset_path,
        cache_dir,
        limit=limit,
        shard_size=shard_size,
    )


def build_tensor_cache(
    torch,
    dataset_path: Path,
    cache_dir: Path,
    *,
    limit: int | None = None,
    shard_size: int = 2048,
) -> TensorCache:
    if shard_size < 1:
        raise ValueError("shard_size must be at least 1")
    source = _source_metadata(dataset_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _clear_cache_files(cache_dir)

    seed_vocab: list[str] = []
    seed_ids: dict[str, int] = {}
    legal_action_vocab: list[str] = []
    legal_action_ids: dict[str, int] = {}
    chosen_kind_vocab: list[str] = []
    chosen_kind_ids: dict[str, int] = {}
    candidate_count_vocab: list[str] = []
    candidate_count_ids: dict[str, int] = {}

    seed_counts: dict[str, int] = {}
    observation_dim: int | None = None
    action_dim: int | None = None
    observation_names: tuple[str, ...] = ()
    action_names: tuple[str, ...] = ()
    shard_infos: list[dict[str, int | str]] = []
    builder: _ShardBuilder | None = None
    started = time.perf_counter()

    for sample_index, sample in enumerate(iter_jsonl(dataset_path, limit=limit)):
        if observation_dim is None or action_dim is None:
            observation_dim = len(sample.observation_values)
            action_dim = _sample_action_dim(sample)
            observation_names = sample.observation_names
            action_names = sample.action_names
            builder = _ShardBuilder(torch, action_dim=action_dim)
        assert builder is not None
        _validate_sample_dimensions(sample, observation_dim, action_dim, sample_index)
        seed_counts[sample.seed] = seed_counts.get(sample.seed, 0) + 1
        builder.add(
            sample,
            seed_id=_vocab_id(seed_vocab, seed_ids, sample.seed),
            legal_action_id=_vocab_id(legal_action_vocab, legal_action_ids, _legal_action_category(sample)),
            chosen_kind_id=_vocab_id(chosen_kind_vocab, chosen_kind_ids, _chosen_kind_category(sample)),
            candidate_count_id=_vocab_id(
                candidate_count_vocab,
                candidate_count_ids,
                _candidate_count_category(sample),
            ),
        )
        if builder.sample_count >= shard_size:
            shard_infos.append(builder.flush(cache_dir, len(shard_infos)))

    if observation_dim is None or action_dim is None:
        raise ValueError("dataset contains no behavior cloning samples")
    assert builder is not None
    final_shard = builder.flush(cache_dir, len(shard_infos))
    if final_shard is not None:
        shard_infos.append(final_shard)

    manifest: dict[str, Any] = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "source": source,
        "limit": limit,
        "shard_size": shard_size,
        "samples": sum(seed_counts.values()),
        "seed_counts": seed_counts,
        "seed_vocab": seed_vocab,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "observation_names": list(observation_names),
        "action_names": list(action_names),
        "legal_action_vocab": legal_action_vocab,
        "chosen_kind_vocab": chosen_kind_vocab,
        "candidate_count_vocab": candidate_count_vocab,
        "shards": shard_infos,
        "build_seconds": time.perf_counter() - started,
    }
    _write_manifest(cache_dir, manifest)
    return TensorCache(cache_dir=cache_dir, manifest=manifest, built=True)


def is_tensor_cache_dir(path: Path) -> bool:
    return (path / MANIFEST_NAME).is_file()


def load_tensor_cache(cache_dir: Path) -> TensorCache:
    manifest_path = cache_dir / MANIFEST_NAME
    with manifest_path.open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    if manifest.get("format") != CACHE_FORMAT or int(manifest.get("version", 0)) != CACHE_VERSION:
        raise ValueError(f"{manifest_path} is not a supported BC tensor cache manifest")
    return TensorCache(cache_dir=cache_dir, manifest=manifest)


def load_cache_shard(torch, cache: TensorCache, shard_info: dict[str, object]) -> dict[str, object]:
    return torch.load(cache.cache_dir / str(shard_info["path"]), map_location="cpu")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    torch = require_torch()
    cache = build_tensor_cache(
        torch,
        Path(args.dataset),
        Path(args.cache_dir),
        limit=args.limit,
        shard_size=args.shard_size,
    )
    print(
        f"cached {cache.manifest['samples']} samples into {len(cache.manifest['shards'])} shards "
        f"at {cache.cache_dir}; build_seconds={cache.manifest['build_seconds']:.1f}",
        flush=True,
    )
    return 0


class _ShardBuilder:
    def __init__(self, torch, *, action_dim: int) -> None:
        self.torch = torch
        self.action_dim = action_dim
        self.observations: list[tuple[float, ...]] = []
        self.candidate_values: list[tuple[float, ...]] = []
        self.candidate_offsets: list[int] = [0]
        self.chosen_indices: list[int] = []
        self.seed_ids: list[int] = []
        self.legal_action_ids: list[int] = []
        self.chosen_kind_ids: list[int] = []
        self.candidate_count_ids: list[int] = []

    @property
    def sample_count(self) -> int:
        return len(self.observations)

    def add(
        self,
        sample: BcSample,
        *,
        seed_id: int,
        legal_action_id: int,
        chosen_kind_id: int,
        candidate_count_id: int,
    ) -> None:
        self.observations.append(sample.observation_values)
        self.candidate_values.extend(sample.candidate_values)
        self.candidate_offsets.append(self.candidate_offsets[-1] + len(sample.candidate_values))
        self.chosen_indices.append(sample.chosen_index)
        self.seed_ids.append(seed_id)
        self.legal_action_ids.append(legal_action_id)
        self.chosen_kind_ids.append(chosen_kind_id)
        self.candidate_count_ids.append(candidate_count_id)

    def flush(self, cache_dir: Path, shard_index: int) -> dict[str, int | str] | None:
        if self.sample_count == 0:
            return None
        path = cache_dir / f"shard-{shard_index:06d}.pt"
        candidate_tensor = (
            self.torch.tensor(self.candidate_values, dtype=self.torch.float32)
            if self.candidate_values
            else self.torch.empty((0, self.action_dim), dtype=self.torch.float32)
        )
        payload = {
            "version": CACHE_VERSION,
            "observations": self.torch.tensor(self.observations, dtype=self.torch.float32),
            "candidate_values": candidate_tensor,
            "candidate_offsets": self.torch.tensor(self.candidate_offsets, dtype=self.torch.int64),
            "chosen_indices": self.torch.tensor(self.chosen_indices, dtype=self.torch.int64),
            "seed_ids": self.torch.tensor(self.seed_ids, dtype=self.torch.int32),
            "legal_action_ids": self.torch.tensor(self.legal_action_ids, dtype=self.torch.int16),
            "chosen_kind_ids": self.torch.tensor(self.chosen_kind_ids, dtype=self.torch.int16),
            "candidate_count_ids": self.torch.tensor(self.candidate_count_ids, dtype=self.torch.int16),
        }
        self.torch.save(payload, path)
        shard_info = {
            "path": path.name,
            "samples": self.sample_count,
            "candidates": len(self.candidate_values),
        }
        self._reset()
        return shard_info

    def _reset(self) -> None:
        self.observations.clear()
        self.candidate_values.clear()
        self.candidate_offsets = [0]
        self.chosen_indices.clear()
        self.seed_ids.clear()
        self.legal_action_ids.clear()
        self.chosen_kind_ids.clear()
        self.candidate_count_ids.clear()


def _source_metadata(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _manifest_matches_source(manifest: dict[str, Any], dataset_path: Path, *, limit: int | None) -> bool:
    try:
        source = _source_metadata(dataset_path)
    except FileNotFoundError:
        return False
    return manifest.get("source") == source and manifest.get("limit") == limit


def _write_manifest(cache_dir: Path, manifest: dict[str, Any]) -> None:
    tmp_path = cache_dir / f"{MANIFEST_NAME}.tmp"
    with tmp_path.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
    tmp_path.replace(cache_dir / MANIFEST_NAME)


def _clear_cache_files(cache_dir: Path) -> None:
    for shard in cache_dir.glob("shard-*.pt"):
        shard.unlink()
    for path in (cache_dir / MANIFEST_NAME, cache_dir / f"{MANIFEST_NAME}.tmp"):
        if path.exists():
            path.unlink()


def _vocab_id(vocab: list[str], lookup: dict[str, int], value: str) -> int:
    existing = lookup.get(value)
    if existing is not None:
        return existing
    index = len(vocab)
    lookup[value] = index
    vocab.append(value)
    return index


def _sample_action_dim(sample: BcSample) -> int:
    if sample.action_names:
        return len(sample.action_names)
    if sample.candidate_values:
        return len(sample.candidate_values[0])
    raise ValueError("sample has no action features")


def _validate_sample_dimensions(sample: BcSample, observation_dim: int, action_dim: int, index: int) -> None:
    if len(sample.observation_values) != observation_dim:
        raise ValueError(f"sample {index} has inconsistent observation dimension")
    if sample.action_names and len(sample.action_names) != action_dim:
        raise ValueError(f"sample {index} has inconsistent action feature names")
    if not sample.candidate_values:
        raise ValueError(f"sample {index} has no candidates")
    if not 0 <= sample.chosen_index < len(sample.candidate_values):
        raise ValueError(f"sample {index} chosen_index is out of range")
    for candidate in sample.candidate_values:
        if len(candidate) != action_dim:
            raise ValueError(f"sample {index} has inconsistent action dimension")


def _legal_action_category(sample: BcSample) -> str:
    return sample.legal_action or "unknown"


def _chosen_kind_category(sample: BcSample) -> str:
    chosen_kind = sample.chosen_payload.get("type")
    return chosen_kind if isinstance(chosen_kind, str) else "unknown"


def _candidate_count_category(sample: BcSample) -> str:
    count = len(sample.candidate_values)
    if count <= 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 16:
        return "5-16"
    if count <= 64:
        return "17-64"
    if count <= 256:
        return "65-256"
    return "257+"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build tensor shards for Guandan BC training.")
    parser.add_argument("dataset", help="Input JSONL or JSONL.GZ dataset from training.collect.")
    parser.add_argument("cache_dir", help="Output cache directory.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=2048)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
