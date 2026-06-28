from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.bc_cache import (
    TensorCache,
    ensure_tensor_cache,
    is_tensor_cache_dir,
    load_cache_shard,
    load_tensor_cache,
)
from training.collect import BcSample, iter_jsonl
from training.encode import ENCODING_SCHEMA_VERSION, LEGACY_ENCODING_SCHEMA_VERSION, encoding_schema
from training.model import build_candidate_ranker, pair_feature_dim, require_torch


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    dataset_path: Path
    output_path: Path
    epochs: int = 3
    learning_rate: float = 1e-3
    hidden_dim: int = 256
    dropout: float = 0.1
    validation_fraction: float = 0.1
    shuffle_buffer_size: int = 2048
    batch_size: int = 64
    cache_dir: Path | None = None
    cache_shard_size: int = 2048
    rebuild_cache: bool = False
    log_epochs: bool = True
    limit: int | None = None
    seed: int = 1
    device: str | None = None


@dataclass(frozen=True, slots=True)
class MetricSummary:
    samples: int
    loss: float
    accuracy: float

    def to_json(self) -> dict[str, int | float]:
        return {"samples": self.samples, "loss": self.loss, "accuracy": self.accuracy}


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    overall: MetricSummary
    by_legal_action: dict[str, MetricSummary]
    by_chosen_kind: dict[str, MetricSummary]
    by_candidate_count: dict[str, MetricSummary]

    def to_json(self) -> dict[str, object]:
        return {
            "overall": self.overall.to_json(),
            "by_legal_action": _metrics_to_json(self.by_legal_action),
            "by_chosen_kind": _metrics_to_json(self.by_chosen_kind),
            "by_candidate_count": _metrics_to_json(self.by_candidate_count),
        }


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    samples: int
    train_samples: int
    validation_samples: int
    epochs: int
    final_loss: float
    final_accuracy: float
    validation_loss: float | None
    validation_accuracy: float | None
    best_epoch: int
    output_path: Path


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    samples: int
    seed_counts: dict[str, int]
    observation_dim: int
    action_dim: int
    observation_names: tuple[str, ...]
    action_names: tuple[str, ...]
    encoding_schema: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    validation_seeds: frozenset[str]
    train_samples: int
    validation_samples: int


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    info: DatasetInfo
    cache: TensorCache | None = None


def train_behavior_clone(config: TrainingConfig) -> TrainingSummary:
    torch = require_torch()
    dataset = _prepare_training_dataset(torch, config)
    dataset_info = dataset.info
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    split = _split_dataset_by_seed(
        dataset_info.seed_counts,
        validation_fraction=config.validation_fraction,
        rng=rng,
    )

    model = build_candidate_ranker(
        pair_feature_dim(dataset_info.observation_dim, dataset_info.action_dim),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    final_loss = 0.0
    final_accuracy = 0.0
    final_train_eval = _empty_evaluation()
    final_validation_eval: EvaluationSummary | None = None
    best_epoch = 0
    best_validation_loss = float("inf")
    best_validation_accuracy: float | None = None
    best_model_state = None
    validation_seed_ids = _cache_validation_seed_ids(dataset.cache, split.validation_seeds) if dataset.cache else frozenset()
    for epoch_index in range(config.epochs):
        epoch_started = time.perf_counter()
        if dataset.cache is None:
            final_train_eval = _train_epoch_streaming(
                torch,
                model,
                config.dataset_path,
                limit=config.limit,
                validation_seeds=split.validation_seeds,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                rng=rng,
                shuffle_buffer_size=config.shuffle_buffer_size,
            )
        else:
            final_train_eval = _train_epoch_cached(
                torch,
                model,
                dataset.cache,
                validation_seed_ids=validation_seed_ids,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                rng=rng,
                batch_size=config.batch_size,
            )
        final_loss = final_train_eval.overall.loss
        final_accuracy = final_train_eval.overall.accuracy
        if split.validation_samples:
            if dataset.cache is None:
                final_validation_eval = _evaluate_streaming_samples(
                    torch,
                    model,
                    config.dataset_path,
                    limit=config.limit,
                    validation_seeds=split.validation_seeds,
                    validation=True,
                    loss_fn=loss_fn,
                    device=device,
                )
            else:
                final_validation_eval = _evaluate_cached_samples(
                    torch,
                    model,
                    dataset.cache,
                    validation_seed_ids=validation_seed_ids,
                    validation=True,
                    loss_fn=loss_fn,
                    device=device,
                    batch_size=config.batch_size,
                )
            validation_loss = final_validation_eval.overall.loss
            if validation_loss < best_validation_loss:
                best_epoch = epoch_index + 1
                best_validation_loss = validation_loss
                best_validation_accuracy = final_validation_eval.overall.accuracy
                best_model_state = _state_dict_cpu_clone(torch, model)
        if config.log_epochs:
            print(
                _format_epoch_progress(
                    epoch=epoch_index + 1,
                    epochs=config.epochs,
                    train_eval=final_train_eval,
                    validation_eval=final_validation_eval,
                    best_epoch=best_epoch,
                    best_validation_loss=best_validation_loss if final_validation_eval else None,
                    seconds=time.perf_counter() - epoch_started,
                ),
                flush=True,
            )

    if dataset.cache is None:
        final_train_eval = _evaluate_streaming_samples(
            torch,
            model,
            config.dataset_path,
            limit=config.limit,
            validation_seeds=split.validation_seeds,
            validation=False,
            loss_fn=loss_fn,
            device=device,
        )
    else:
        final_train_eval = _evaluate_cached_samples(
            torch,
            model,
            dataset.cache,
            validation_seed_ids=validation_seed_ids,
            validation=False,
            loss_fn=loss_fn,
            device=device,
            batch_size=config.batch_size,
        )
    final_loss = final_train_eval.overall.loss
    final_accuracy = final_train_eval.overall.accuracy
    if split.validation_samples and final_validation_eval is None:
        if dataset.cache is None:
            final_validation_eval = _evaluate_streaming_samples(
                torch,
                model,
                config.dataset_path,
                limit=config.limit,
                validation_seeds=split.validation_seeds,
                validation=True,
                loss_fn=loss_fn,
                device=device,
            )
        else:
            final_validation_eval = _evaluate_cached_samples(
                torch,
                model,
                dataset.cache,
                validation_seed_ids=validation_seed_ids,
                validation=True,
                loss_fn=loss_fn,
                device=device,
                batch_size=config.batch_size,
            )
        best_validation_loss = final_validation_eval.overall.loss
        best_validation_accuracy = final_validation_eval.overall.accuracy

    if best_model_state is None:
        best_epoch = config.epochs
        best_model_state = model.state_dict()

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_model_state,
            "observation_names": dataset_info.observation_names,
            "action_names": dataset_info.action_names,
            "encoding_schema": dataset_info.encoding_schema,
            "observation_dim": dataset_info.observation_dim,
            "action_dim": dataset_info.action_dim,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "samples": dataset_info.samples,
            "train_samples": split.train_samples,
            "validation_samples": split.validation_samples,
            "epochs": config.epochs,
            "final_loss": final_loss,
            "final_accuracy": final_accuracy,
            "validation_loss": final_validation_eval.overall.loss if final_validation_eval else None,
            "validation_accuracy": final_validation_eval.overall.accuracy if final_validation_eval else None,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss if final_validation_eval else None,
            "best_validation_accuracy": best_validation_accuracy,
            "train_metrics": final_train_eval.to_json(),
            "validation_metrics": final_validation_eval.to_json() if final_validation_eval else None,
            "validation_fraction": config.validation_fraction,
            "dataset_cache": str(dataset.cache.cache_dir) if dataset.cache else None,
            "batch_size": config.batch_size,
        },
        config.output_path,
    )
    return TrainingSummary(
        samples=dataset_info.samples,
        train_samples=split.train_samples,
        validation_samples=split.validation_samples,
        epochs=config.epochs,
        final_loss=final_loss,
        final_accuracy=final_accuracy,
        validation_loss=final_validation_eval.overall.loss if final_validation_eval else None,
        validation_accuracy=final_validation_eval.overall.accuracy if final_validation_eval else None,
        best_epoch=best_epoch,
        output_path=config.output_path,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = train_behavior_clone(
        TrainingConfig(
            dataset_path=Path(args.dataset),
            output_path=Path(args.output),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            validation_fraction=args.validation_fraction,
            shuffle_buffer_size=args.shuffle_buffer_size,
            batch_size=args.batch_size,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            cache_shard_size=args.cache_shard_size,
            rebuild_cache=args.rebuild_cache,
            log_epochs=not args.quiet,
            limit=args.limit,
            seed=args.seed,
            device=args.device,
        )
    )
    validation_note = (
        f" validation_samples={summary.validation_samples} "
        f"validation_loss={summary.validation_loss:.4f} validation_accuracy={summary.validation_accuracy:.3f} "
        if summary.validation_loss is not None and summary.validation_accuracy is not None
        else " validation_samples=0 "
    )
    print(
        f"trained {summary.train_samples}/{summary.samples} train samples for {summary.epochs} epochs; "
        f"loss={summary.final_loss:.4f} accuracy={summary.final_accuracy:.3f};"
        f"{validation_note}best_epoch={summary.best_epoch}; "
        f"wrote {summary.output_path}",
        flush=True,
    )
    return 0


def _sample_logits(torch, model, sample: BcSample, device):
    observation = torch.tensor(sample.observation_values, dtype=torch.float32, device=device)
    actions = torch.tensor(sample.candidate_values, dtype=torch.float32, device=device)
    observations = observation.expand(actions.shape[0], -1)
    pair_features = torch.cat((observations, actions), dim=1)
    return model(pair_features)


def _prepare_training_dataset(torch, config: TrainingConfig) -> TrainingDataset:
    if is_tensor_cache_dir(config.dataset_path):
        cache = load_tensor_cache(config.dataset_path)
        return TrainingDataset(info=_dataset_info_from_cache(cache), cache=cache)
    if config.cache_dir is not None:
        if config.log_epochs:
            print(f"preparing tensor cache at {config.cache_dir}", flush=True)
        cache = ensure_tensor_cache(
            torch,
            config.dataset_path,
            config.cache_dir,
            limit=config.limit,
            shard_size=config.cache_shard_size,
            force=config.rebuild_cache,
        )
        if config.log_epochs:
            action = "built" if cache.built else "using"
            print(
                f"{action} tensor cache {cache.cache_dir}; "
                f"samples={cache.manifest['samples']} shards={len(cache.manifest['shards'])}",
                flush=True,
            )
        return TrainingDataset(info=_dataset_info_from_cache(cache), cache=cache)
    return TrainingDataset(info=_inspect_dataset(config.dataset_path, limit=config.limit))


def _dataset_info_from_cache(cache: TensorCache) -> DatasetInfo:
    manifest = cache.manifest
    seed_counts_obj = manifest.get("seed_counts")
    if not isinstance(seed_counts_obj, dict):
        raise ValueError("tensor cache manifest is missing seed_counts")
    observation_dim = int(manifest["observation_dim"])
    action_dim = int(manifest["action_dim"])
    observation_names = tuple(str(name) for name in _manifest_list(manifest, "observation_names"))
    action_names = tuple(str(name) for name in _manifest_list(manifest, "action_names"))
    return DatasetInfo(
        samples=int(manifest["samples"]),
        seed_counts={str(seed): int(count) for seed, count in seed_counts_obj.items()},
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_names=observation_names,
        action_names=action_names,
        encoding_schema=_known_encoding_schema(observation_names, action_names, observation_dim, action_dim),
    )


def _cache_validation_seed_ids(cache: TensorCache | None, validation_seeds: frozenset[str]) -> frozenset[int]:
    if cache is None or not validation_seeds:
        return frozenset()
    seed_to_id = {str(seed): index for index, seed in enumerate(_manifest_list(cache.manifest, "seed_vocab"))}
    return frozenset(seed_to_id[seed] for seed in validation_seeds if seed in seed_to_id)


def _train_epoch_cached(
    torch,
    model,
    cache: TensorCache,
    *,
    validation_seed_ids: frozenset[int],
    loss_fn,
    optimizer,
    device,
    rng: random.Random,
    batch_size: int,
) -> EvaluationSummary:
    _validate_batch_size(batch_size)
    accumulator = _CachedEvaluationAccumulator(cache)
    shard_infos = list(_cache_shard_infos(cache))
    rng.shuffle(shard_infos)
    model.train()
    for shard_info in shard_infos:
        shard_cpu = load_cache_shard(torch, cache, shard_info)
        indices = _cached_split_indices(shard_cpu, validation_seed_ids=validation_seed_ids, validation=False)
        rng.shuffle(indices)
        shard = _move_cached_feature_tensors(shard_cpu, device)
        for batch_indices in _iter_batches(indices, batch_size):
            logits, targets = _cached_batch_logits_and_targets(torch, model, shard, shard_cpu, batch_indices, device)
            per_sample_losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
            loss = per_sample_losses.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predictions = torch.argmax(logits, dim=1).detach().cpu().tolist()
            losses = per_sample_losses.detach().cpu().tolist()
            target_values = targets.detach().cpu().tolist()
            for row, index in enumerate(batch_indices):
                correct = int(predictions[row] == target_values[row])
                accumulator.add(shard_cpu, index, float(losses[row]), correct)
    return accumulator.summary()


def _evaluate_cached_samples(
    torch,
    model,
    cache: TensorCache,
    *,
    validation_seed_ids: frozenset[int],
    validation: bool,
    loss_fn,
    device,
    batch_size: int,
) -> EvaluationSummary:
    _validate_batch_size(batch_size)
    accumulator = _CachedEvaluationAccumulator(cache)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for shard_info in _cache_shard_infos(cache):
            shard_cpu = load_cache_shard(torch, cache, shard_info)
            indices = _cached_split_indices(shard_cpu, validation_seed_ids=validation_seed_ids, validation=validation)
            shard = _move_cached_feature_tensors(shard_cpu, device)
            for batch_indices in _iter_batches(indices, batch_size):
                logits, targets = _cached_batch_logits_and_targets(torch, model, shard, shard_cpu, batch_indices, device)
                per_sample_losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
                predictions = torch.argmax(logits, dim=1).detach().cpu().tolist()
                losses = per_sample_losses.detach().cpu().tolist()
                target_values = targets.detach().cpu().tolist()
                for row, index in enumerate(batch_indices):
                    correct = int(predictions[row] == target_values[row])
                    accumulator.add(shard_cpu, index, float(losses[row]), correct)
    if was_training:
        model.train()
    return accumulator.summary()


def _cached_sample_logits(torch, model, shard: dict[str, object], index: int):
    observations = shard["observations"]
    candidate_values = shard["candidate_values"]
    candidate_offsets = shard["candidate_offsets"]
    start = int(candidate_offsets[index].item())
    end = int(candidate_offsets[index + 1].item())
    observation = observations[index]
    actions = candidate_values[start:end]
    pair_features = torch.cat((observation.expand(actions.shape[0], -1), actions), dim=1)
    return model(pair_features)


def _cached_batch_logits_and_targets(
    torch,
    model,
    shard: dict[str, object],
    shard_cpu: dict[str, object],
    indices: list[int],
    device,
):
    if not indices:
        raise ValueError("batch contains no sample indices")
    observations = shard["observations"]
    candidate_values = shard["candidate_values"]
    candidate_offsets = shard_cpu["candidate_offsets"]
    chosen_indices = shard_cpu["chosen_indices"]
    index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    batch_observations = observations.index_select(0, index_tensor)
    ranges = [
        (int(candidate_offsets[index].item()), int(candidate_offsets[index + 1].item()))
        for index in indices
    ]
    counts = [end - start for start, end in ranges]
    if any(count <= 0 for count in counts):
        raise ValueError("cached sample contains no candidates")
    batch_actions = torch.cat([candidate_values[start:end] for start, end in ranges], dim=0)
    counts_tensor = torch.tensor(counts, dtype=torch.long, device=device)
    pair_features = torch.cat((batch_observations.repeat_interleave(counts_tensor, dim=0), batch_actions), dim=1)
    flat_logits = model(pair_features)
    max_candidates = max(counts)
    logits = torch.full(
        (len(indices), max_candidates),
        -torch.inf,
        dtype=flat_logits.dtype,
        device=device,
    )
    cursor = 0
    for row, count in enumerate(counts):
        logits[row, :count] = flat_logits[cursor : cursor + count]
        cursor += count
    targets = torch.tensor(
        [int(chosen_indices[index].item()) for index in indices],
        dtype=torch.long,
        device=device,
    )
    return logits, targets


def _move_cached_feature_tensors(shard: dict[str, object], device) -> dict[str, object]:
    if device.type == "cpu":
        return shard
    moved = dict(shard)
    moved["observations"] = shard["observations"].to(device)
    moved["candidate_values"] = shard["candidate_values"].to(device)
    return moved


def _cached_split_indices(
    shard: dict[str, object],
    *,
    validation_seed_ids: frozenset[int],
    validation: bool,
) -> list[int]:
    sample_count = int(shard["observations"].shape[0])
    if not validation_seed_ids:
        return list(range(sample_count)) if not validation else []
    seed_ids = shard["seed_ids"].tolist()
    return [index for index, seed_id in enumerate(seed_ids) if (int(seed_id) in validation_seed_ids) == validation]


def _cache_shard_infos(cache: TensorCache) -> tuple[dict[str, object], ...]:
    shards = cache.manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("tensor cache manifest is missing shards")
    return tuple(dict(shard) for shard in shards if isinstance(shard, dict))


def _iter_batches(indices: list[int], batch_size: int) -> Iterable[list[int]]:
    _validate_batch_size(batch_size)
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _validate_batch_size(batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")


def _train_epoch_streaming(
    torch,
    model,
    path: Path,
    *,
    limit: int | None,
    validation_seeds: frozenset[str],
    loss_fn,
    optimizer,
    device,
    rng: random.Random,
    shuffle_buffer_size: int,
) -> EvaluationSummary:
    accumulator = _EvaluationAccumulator()
    model.train()
    for sample in _iter_training_samples(
        path,
        limit=limit,
        validation_seeds=validation_seeds,
        rng=rng,
        shuffle_buffer_size=shuffle_buffer_size,
    ):
        logits = _sample_logits(torch, model, sample, device)
        target = torch.tensor([sample.chosen_index], dtype=torch.long, device=device)
        loss = loss_fn(logits.unsqueeze(0), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct = int(torch.argmax(logits).item() == sample.chosen_index)
        accumulator.add(sample, float(loss.detach().cpu()), correct)
    return accumulator.summary()


def _evaluate_streaming_samples(
    torch,
    model,
    path: Path,
    *,
    limit: int | None,
    validation_seeds: frozenset[str],
    validation: bool,
    loss_fn,
    device,
) -> EvaluationSummary:
    return _evaluate_sample_iterable(
        torch,
        model,
        _iter_split_samples(path, limit=limit, validation_seeds=validation_seeds, validation=validation),
        loss_fn,
        device,
    )


def _evaluate_samples(torch, model, samples: tuple[BcSample, ...], loss_fn, device) -> EvaluationSummary:
    return _evaluate_sample_iterable(torch, model, samples, loss_fn, device)


def _evaluate_sample_iterable(torch, model, samples: Iterable[BcSample], loss_fn, device) -> EvaluationSummary:
    accumulator = _EvaluationAccumulator()
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for sample in samples:
            logits = _sample_logits(torch, model, sample, device)
            target = torch.tensor([sample.chosen_index], dtype=torch.long, device=device)
            loss = float(loss_fn(logits.unsqueeze(0), target).detach().cpu())
            correct = int(torch.argmax(logits).item() == sample.chosen_index)
            accumulator.add(sample, loss, correct)
    if was_training:
        model.train()
    return accumulator.summary()


def _iter_training_samples(
    path: Path,
    *,
    limit: int | None,
    validation_seeds: frozenset[str],
    rng: random.Random,
    shuffle_buffer_size: int,
) -> Iterable[BcSample]:
    if shuffle_buffer_size < 0:
        raise ValueError("shuffle_buffer_size must be non-negative")
    if shuffle_buffer_size <= 1:
        yield from _iter_split_samples(path, limit=limit, validation_seeds=validation_seeds, validation=False)
        return

    buffer: list[BcSample] = []
    for sample in _iter_split_samples(path, limit=limit, validation_seeds=validation_seeds, validation=False):
        buffer.append(sample)
        if len(buffer) >= shuffle_buffer_size:
            rng.shuffle(buffer)
            yield from buffer
            buffer.clear()
    if buffer:
        rng.shuffle(buffer)
        yield from buffer


def _iter_split_samples(
    path: Path,
    *,
    limit: int | None,
    validation_seeds: frozenset[str],
    validation: bool,
) -> Iterable[BcSample]:
    for sample in iter_jsonl(path, limit=limit):
        if (sample.seed in validation_seeds) == validation:
            yield sample


def _inspect_dataset(path: Path, *, limit: int | None) -> DatasetInfo:
    seed_counts: dict[str, int] = defaultdict(int)
    observation_dim: int | None = None
    action_dim: int | None = None
    observation_names: tuple[str, ...] = ()
    action_names: tuple[str, ...] = ()

    for index, sample in enumerate(iter_jsonl(path, limit=limit)):
        if observation_dim is None or action_dim is None:
            observation_dim = len(sample.observation_values)
            action_dim = _sample_action_dim(sample)
            observation_names = sample.observation_names
            action_names = sample.action_names
        _validate_sample_dimensions(sample, observation_dim, action_dim, index)
        seed_counts[sample.seed] += 1

    if observation_dim is None or action_dim is None:
        raise ValueError("dataset contains no behavior cloning samples")

    return DatasetInfo(
        samples=sum(seed_counts.values()),
        seed_counts=dict(seed_counts),
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_names=observation_names,
        action_names=action_names,
        encoding_schema=_known_encoding_schema(observation_names, action_names, observation_dim, action_dim),
    )


def _known_encoding_schema(
    observation_names: tuple[str, ...],
    action_names: tuple[str, ...],
    observation_dim: int,
    action_dim: int,
) -> dict[str, object] | None:
    for version in (ENCODING_SCHEMA_VERSION, LEGACY_ENCODING_SCHEMA_VERSION):
        schema = encoding_schema(version)
        schema_observation_names = tuple(str(name) for name in schema["observation_names"])
        schema_action_names = tuple(str(name) for name in schema["action_names"])
        if observation_names and action_names:
            if observation_names == schema_observation_names and action_names == schema_action_names:
                return schema
        elif len(schema_observation_names) == observation_dim and len(schema_action_names) == action_dim:
            return schema
    return None


def _validate_dimensions(samples: tuple[BcSample, ...], observation_dim: int, action_dim: int) -> None:
    for index, sample in enumerate(samples):
        _validate_sample_dimensions(sample, observation_dim, action_dim, index)


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


def _sample_action_dim(sample: BcSample) -> int:
    if sample.action_names:
        return len(sample.action_names)
    if sample.candidate_values:
        return len(sample.candidate_values[0])
    raise ValueError("sample has no action features")


def _split_samples_by_seed(
    samples: tuple[BcSample, ...],
    *,
    validation_fraction: float,
    rng: random.Random,
) -> tuple[tuple[BcSample, ...], tuple[BcSample, ...]]:
    validation_seeds = _choose_validation_seeds({sample.seed for sample in samples}, validation_fraction, rng)
    train_samples = tuple(sample for sample in samples if sample.seed not in validation_seeds)
    validation_samples = tuple(sample for sample in samples if sample.seed in validation_seeds)
    return train_samples, validation_samples


def _split_dataset_by_seed(
    seed_counts: dict[str, int],
    *,
    validation_fraction: float,
    rng: random.Random,
) -> DatasetSplit:
    validation_seeds = _choose_validation_seeds(seed_counts.keys(), validation_fraction, rng)
    validation_samples = sum(count for seed, count in seed_counts.items() if seed in validation_seeds)
    train_samples = sum(count for seed, count in seed_counts.items() if seed not in validation_seeds)
    return DatasetSplit(
        validation_seeds=validation_seeds,
        train_samples=train_samples,
        validation_samples=validation_samples,
    )


def _choose_validation_seeds(
    seeds: Iterable[str],
    validation_fraction: float,
    rng: random.Random,
) -> frozenset[str]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0.0, 1.0)")
    sorted_seeds = sorted(seeds)
    if validation_fraction == 0.0 or len(sorted_seeds) < 2:
        return frozenset()
    shuffled = list(sorted_seeds)
    rng.shuffle(shuffled)
    validation_count = min(max(1, round(len(shuffled) * validation_fraction)), len(shuffled) - 1)
    return frozenset(shuffled[:validation_count])


@dataclass(slots=True)
class _MetricAccumulator:
    samples: int = 0
    total_loss: float = 0.0
    correct: int = 0

    def add(self, loss: float, correct: int) -> None:
        self.samples += 1
        self.total_loss += loss
        self.correct += correct

    def summary(self) -> MetricSummary:
        if self.samples == 0:
            return MetricSummary(samples=0, loss=0.0, accuracy=0.0)
        return MetricSummary(
            samples=self.samples,
            loss=self.total_loss / self.samples,
            accuracy=self.correct / self.samples,
        )


@dataclass(slots=True)
class _EvaluationAccumulator:
    overall: _MetricAccumulator = field(default_factory=_MetricAccumulator)
    by_legal_action: dict[str, _MetricAccumulator] = field(default_factory=lambda: defaultdict(_MetricAccumulator))
    by_chosen_kind: dict[str, _MetricAccumulator] = field(default_factory=lambda: defaultdict(_MetricAccumulator))
    by_candidate_count: dict[str, _MetricAccumulator] = field(default_factory=lambda: defaultdict(_MetricAccumulator))

    def add(self, sample: BcSample, loss: float, correct: int) -> None:
        self.overall.add(loss, correct)
        self.by_legal_action[_legal_action_category(sample)].add(loss, correct)
        self.by_chosen_kind[_chosen_kind_category(sample)].add(loss, correct)
        self.by_candidate_count[_candidate_count_category(sample)].add(loss, correct)

    def summary(self) -> EvaluationSummary:
        return EvaluationSummary(
            overall=self.overall.summary(),
            by_legal_action=_summarize_metrics(self.by_legal_action),
            by_chosen_kind=_summarize_metrics(self.by_chosen_kind),
            by_candidate_count=_summarize_metrics(self.by_candidate_count),
        )


@dataclass(slots=True)
class _CachedEvaluationAccumulator:
    cache: TensorCache
    overall: _MetricAccumulator = field(default_factory=_MetricAccumulator)
    by_legal_action: dict[str, _MetricAccumulator] = field(default_factory=lambda: defaultdict(_MetricAccumulator))
    by_chosen_kind: dict[str, _MetricAccumulator] = field(default_factory=lambda: defaultdict(_MetricAccumulator))
    by_candidate_count: dict[str, _MetricAccumulator] = field(default_factory=lambda: defaultdict(_MetricAccumulator))

    def add(self, shard: dict[str, object], index: int, loss: float, correct: int) -> None:
        self.overall.add(loss, correct)
        self.by_legal_action[_cached_category(self.cache, "legal_action_vocab", shard, "legal_action_ids", index)].add(
            loss,
            correct,
        )
        self.by_chosen_kind[_cached_category(self.cache, "chosen_kind_vocab", shard, "chosen_kind_ids", index)].add(
            loss,
            correct,
        )
        self.by_candidate_count[
            _cached_category(self.cache, "candidate_count_vocab", shard, "candidate_count_ids", index)
        ].add(loss, correct)

    def summary(self) -> EvaluationSummary:
        return EvaluationSummary(
            overall=self.overall.summary(),
            by_legal_action=_summarize_metrics(self.by_legal_action),
            by_chosen_kind=_summarize_metrics(self.by_chosen_kind),
            by_candidate_count=_summarize_metrics(self.by_candidate_count),
        )


def _cached_category(
    cache: TensorCache,
    vocab_name: str,
    shard: dict[str, object],
    tensor_name: str,
    index: int,
) -> str:
    vocab = _manifest_list(cache.manifest, vocab_name)
    category_id = int(shard[tensor_name][index].item())
    if 0 <= category_id < len(vocab):
        return str(vocab[category_id])
    return "unknown"


def _empty_evaluation() -> EvaluationSummary:
    empty = MetricSummary(samples=0, loss=0.0, accuracy=0.0)
    return EvaluationSummary(overall=empty, by_legal_action={}, by_chosen_kind={}, by_candidate_count={})


def _summarize_metrics(metrics: dict[str, _MetricAccumulator]) -> dict[str, MetricSummary]:
    return {key: accumulator.summary() for key, accumulator in sorted(metrics.items())}


def _metrics_to_json(metrics: dict[str, MetricSummary]) -> dict[str, dict[str, int | float]]:
    return {key: value.to_json() for key, value in metrics.items()}


def _manifest_list(manifest: dict[str, Any], key: str) -> list[object]:
    value = manifest.get(key)
    return value if isinstance(value, list) else []


def _format_epoch_progress(
    *,
    epoch: int,
    epochs: int,
    train_eval: EvaluationSummary,
    validation_eval: EvaluationSummary | None,
    best_epoch: int,
    best_validation_loss: float | None,
    seconds: float,
) -> str:
    train = train_eval.overall
    fields = [
        f"epoch={epoch}/{epochs}",
        f"train_samples={train.samples}",
        f"train_loss={train.loss:.4f}",
        f"train_accuracy={train.accuracy:.3f}",
    ]
    if validation_eval is None:
        fields.append("validation_samples=0")
    else:
        validation = validation_eval.overall
        fields.extend(
            [
                f"validation_samples={validation.samples}",
                f"validation_loss={validation.loss:.4f}",
                f"validation_accuracy={validation.accuracy:.3f}",
                f"best_epoch={best_epoch}",
            ]
        )
        if best_validation_loss is not None:
            fields.append(f"best_validation_loss={best_validation_loss:.4f}")
    fields.append(f"seconds={seconds:.1f}")
    return " ".join(fields)


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


def _state_dict_cpu_clone(torch, model) -> dict[str, object]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Guandan behavior cloning candidate ranker.")
    parser.add_argument("dataset", help="Input JSONL dataset from training.collect.")
    parser.add_argument("output", help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--shuffle-buffer-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for tensor-cache training.")
    parser.add_argument("--cache-dir", default=None, help="Tensor shard cache directory to build or reuse.")
    parser.add_argument("--cache-shard-size", type=int, default=2048)
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild --cache-dir even when it is current.")
    parser.add_argument("--quiet", action="store_true", help="Disable per-epoch progress output.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
