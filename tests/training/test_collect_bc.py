from __future__ import annotations

import tempfile
import unittest
import random
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from training.bc_train import (
    EvaluationSummary,
    MetricSummary,
    TrainingConfig,
    _candidate_count_category,
    _chosen_kind_category,
    _format_epoch_progress,
    _inspect_dataset,
    _iter_training_samples,
    _split_dataset_by_seed,
    _split_samples_by_seed,
    _validate_dimensions,
    train_behavior_clone,
)
from training.bc_cache import build_tensor_cache, load_cache_shard
from training.collect import BcSample, collect_heuristic_samples, main as collect_main, read_jsonl, write_jsonl
from training.model import pair_feature_dim, require_torch


class BehaviorCloningCollectionTests(unittest.TestCase):
    def test_collect_heuristic_samples_records_candidates_and_choice(self) -> None:
        result = collect_heuristic_samples(["1"], max_deals_per_seed=1, max_steps_per_seed=1_000)

        self.assertEqual(result.completed_deals, 1)
        self.assertGreater(len(result.samples), 0)
        self.assertEqual(result.total_samples, len(result.samples))
        sample = result.samples[0]
        self.assertEqual(sample.version, 1)
        self.assertEqual(len(sample.observation_names), len(sample.observation_values))
        self.assertGreater(len(sample.candidate_values), 0)
        self.assertEqual(sample.candidate_payloads[sample.chosen_index], sample.chosen_payload)
        self.assertEqual(len(sample.action_names), len(sample.candidate_values[0]))

    def test_collect_heuristic_samples_can_parallelize_by_seed(self) -> None:
        with _inline_process_pool():
            result = collect_heuristic_samples(
                ["1", "bench"],
                max_deals_per_seed=1,
                max_steps_per_seed=1_000,
                workers=2,
            )

        self.assertEqual(result.completed_deals, 2)
        self.assertGreater(len(result.samples), 0)
        self.assertEqual(result.total_samples, len(result.samples))
        self.assertEqual(result.samples[0].seed, "'1'")
        self.assertIn("'bench'", {sample.seed for sample in result.samples})

    def test_jsonl_round_trip_preserves_sample(self) -> None:
        result = collect_heuristic_samples(["1"], max_deals_per_seed=1, max_steps_per_seed=1_000)
        sample = result.samples[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"

            count = write_jsonl((sample,), path)
            loaded = read_jsonl(path)

        self.assertEqual(count, 1)
        self.assertEqual(loaded, (sample,))

    def test_compact_jsonl_round_trip_keeps_training_features(self) -> None:
        result = collect_heuristic_samples(["1"], max_deals_per_seed=1, max_steps_per_seed=1_000)
        sample = result.samples[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "compact-samples.jsonl.gz"

            count = write_jsonl((sample,), path, compact=True)
            loaded = read_jsonl(path)

        self.assertEqual(count, 1)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].observation_values, sample.observation_values)
        self.assertEqual(loaded[0].candidate_values, sample.candidate_values)
        self.assertEqual(loaded[0].chosen_index, sample.chosen_index)
        self.assertEqual(loaded[0].observation_names, ())
        self.assertEqual(loaded[0].action_names, ())
        self.assertEqual(loaded[0].candidate_payloads, ())
        self.assertEqual(loaded[0].chosen_payload.get("type"), sample.chosen_payload.get("type"))

    def test_collect_cli_writes_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cli-samples.jsonl"

            with redirect_stdout(StringIO()):
                exit_code = collect_main([str(path), "--seed", "1", "--max-deals", "1"])

            loaded = read_jsonl(path)
        self.assertEqual(exit_code, 0)
        self.assertGreater(len(loaded), 0)

    def test_collect_cli_parallel_workers_write_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parallel-cli-samples.jsonl"

            with redirect_stdout(StringIO()), _inline_process_pool():
                exit_code = collect_main(
                    [
                        str(path),
                        "--seed",
                        "1",
                        "--seed",
                        "bench",
                        "--max-deals",
                        "1",
                        "--workers",
                        "2",
                    ]
                )

            loaded = read_jsonl(path)
        self.assertEqual(exit_code, 0)
        self.assertGreater(len(loaded), 0)
        self.assertEqual(loaded[0].seed, "'1'")
        self.assertIn("'bench'", {sample.seed for sample in loaded})

    def test_collect_rejects_zero_workers(self) -> None:
        with self.assertRaises(ValueError):
            collect_heuristic_samples(["1"], workers=0)

    def test_bc_dimension_validation_accepts_collected_samples(self) -> None:
        result = collect_heuristic_samples(["1"], max_deals_per_seed=1, max_steps_per_seed=1_000)
        sample = result.samples[0]

        _validate_dimensions((sample,), len(sample.observation_values), len(sample.action_names))

    def test_bc_dimension_validation_accepts_compact_sample_without_names(self) -> None:
        result = collect_heuristic_samples(["1"], max_deals_per_seed=1, max_steps_per_seed=1_000)
        sample = BcSample.from_json(result.samples[0].to_json(compact=True))

        _validate_dimensions((sample,), len(sample.observation_values), len(sample.candidate_values[0]))

    def test_bc_dimension_validation_rejects_bad_chosen_index(self) -> None:
        result = collect_heuristic_samples(["1"], max_deals_per_seed=1, max_steps_per_seed=1_000)
        sample = result.samples[0]
        broken = BcSample(
            version=sample.version,
            seed=sample.seed,
            deal_id=sample.deal_id,
            event_seq=sample.event_seq,
            seat=sample.seat,
            legal_action=sample.legal_action,
            observation_names=sample.observation_names,
            observation_values=sample.observation_values,
            action_names=sample.action_names,
            candidate_values=sample.candidate_values,
            candidate_payloads=sample.candidate_payloads,
            chosen_index=len(sample.candidate_values),
            chosen_payload=sample.chosen_payload,
        )

        with self.assertRaises(ValueError):
            _validate_dimensions((broken,), len(sample.observation_values), len(sample.action_names))

    def test_pair_feature_dim_is_observation_plus_action_dim(self) -> None:
        self.assertEqual(pair_feature_dim(3, 5), 8)

    def test_validation_split_uses_whole_seeds(self) -> None:
        samples = (
            _minimal_sample("a", 0),
            _minimal_sample("a", 1),
            _minimal_sample("b", 0),
            _minimal_sample("c", 0),
        )

        train, validation = _split_samples_by_seed(samples, validation_fraction=0.34, rng=random.Random(1))

        self.assertGreater(len(validation), 0)
        self.assertLess(len(validation), len(samples))
        validation_seeds = {sample.seed for sample in validation}
        self.assertTrue(validation_seeds.isdisjoint({sample.seed for sample in train}))

    def test_bc_streaming_metadata_and_training_iter_use_seed_split(self) -> None:
        samples = (
            _minimal_sample("a", 0),
            _minimal_sample("a", 1),
            _minimal_sample("b", 0),
            _minimal_sample("c", 1),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl.gz"
            write_jsonl(samples, path, compact=True)

            info = _inspect_dataset(path, limit=None)
            split = _split_dataset_by_seed(info.seed_counts, validation_fraction=0.34, rng=random.Random(1))
            train_samples = tuple(
                _iter_training_samples(
                    path,
                    limit=None,
                    validation_seeds=split.validation_seeds,
                    rng=random.Random(1),
                    shuffle_buffer_size=2,
                )
            )

        self.assertEqual(info.samples, 4)
        self.assertEqual(info.observation_dim, 1)
        self.assertEqual(info.action_dim, 1)
        self.assertGreater(split.validation_samples, 0)
        self.assertEqual(len(train_samples), split.train_samples)
        self.assertTrue(all(sample.seed not in split.validation_seeds for sample in train_samples))

    def test_bc_tensor_cache_builds_shards_and_trains(self) -> None:
        try:
            torch = require_torch()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        samples = (
            _minimal_sample("a", 0),
            _minimal_sample("a", 1),
            _minimal_sample("b", 0),
            _minimal_sample("c", 1),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "samples.jsonl.gz"
            cache_dir = root / "samples.bc-cache"
            output_path = root / "bc.pt"
            write_jsonl(samples, dataset_path, compact=True)

            cache = build_tensor_cache(torch, dataset_path, cache_dir, shard_size=2)
            shard = load_cache_shard(torch, cache, cache.manifest["shards"][0])
            summary = train_behavior_clone(
                TrainingConfig(
                    dataset_path=dataset_path,
                    output_path=output_path,
                    epochs=1,
                    validation_fraction=0.34,
                    cache_dir=cache_dir,
                    cache_shard_size=2,
                    batch_size=2,
                    log_epochs=False,
                    device="cpu",
                )
            )
            output_exists = output_path.exists()

        self.assertEqual(cache.manifest["samples"], 4)
        self.assertEqual(len(cache.manifest["shards"]), 2)
        self.assertEqual(tuple(shard["observations"].shape), (2, 1))
        self.assertEqual(tuple(shard["candidate_values"].shape), (4, 1))
        self.assertEqual(summary.samples, 4)
        self.assertGreater(summary.train_samples, 0)
        self.assertTrue(output_exists)

    def test_bc_metric_categories_support_compact_samples(self) -> None:
        sample = BcSample.from_json(_minimal_sample("a", 0).to_json(compact=True))

        self.assertEqual(_chosen_kind_category(sample), "play_cards")
        self.assertEqual(_candidate_count_category(sample), "2-4")

    def test_bc_epoch_progress_includes_train_and_validation_metrics(self) -> None:
        train_eval = _evaluation(samples=90, loss=0.4567, accuracy=0.8123)
        validation_eval = _evaluation(samples=10, loss=0.5678, accuracy=0.7)

        line = _format_epoch_progress(
            epoch=2,
            epochs=10,
            train_eval=train_eval,
            validation_eval=validation_eval,
            best_epoch=1,
            best_validation_loss=0.5,
            seconds=12.34,
        )

        self.assertIn("epoch=2/10", line)
        self.assertIn("train_samples=90", line)
        self.assertIn("train_loss=0.4567", line)
        self.assertIn("train_accuracy=0.812", line)
        self.assertIn("validation_samples=10", line)
        self.assertIn("validation_loss=0.5678", line)
        self.assertIn("validation_accuracy=0.700", line)
        self.assertIn("best_epoch=1", line)
        self.assertIn("best_validation_loss=0.5000", line)
        self.assertIn("seconds=12.3", line)


class _InlineExecutor:
    def __init__(self, *, max_workers: int) -> None:
        self.max_workers = max_workers

    def __enter__(self) -> "_InlineExecutor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def map(self, fn, jobs, *, chunksize: int = 1):
        return map(fn, jobs)


def _inline_process_pool():
    return patch("training.collect._process_pool", lambda *, max_workers: _InlineExecutor(max_workers=max_workers))


def _minimal_sample(seed: str, chosen_index: int) -> BcSample:
    return BcSample(
        version=1,
        seed=seed,
        deal_id=1,
        event_seq=1,
        seat="E",
        legal_action="lead",
        observation_names=(),
        observation_values=(0.0,),
        action_names=(),
        candidate_values=((0.0,), (1.0,)),
        candidate_payloads=(),
        chosen_index=chosen_index,
        chosen_payload={"type": "play_cards"},
    )


def _evaluation(*, samples: int, loss: float, accuracy: float) -> EvaluationSummary:
    return EvaluationSummary(
        overall=MetricSummary(samples=samples, loss=loss, accuracy=accuracy),
        by_legal_action={},
        by_chosen_kind={},
        by_candidate_count={},
    )


if __name__ == "__main__":
    unittest.main()
