from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from training.bc_train import _validate_dimensions
from training.collect import BcSample, collect_heuristic_samples, main as collect_main, read_jsonl, write_jsonl
from training.model import pair_feature_dim


class BehaviorCloningCollectionTests(unittest.TestCase):
    def test_collect_heuristic_samples_records_candidates_and_choice(self) -> None:
        result = collect_heuristic_samples(["1"], max_deals_per_seed=1, max_steps_per_seed=1_000)

        self.assertEqual(result.completed_deals, 1)
        self.assertGreater(len(result.samples), 0)
        sample = result.samples[0]
        self.assertEqual(sample.version, 1)
        self.assertEqual(len(sample.observation_names), len(sample.observation_values))
        self.assertGreater(len(sample.candidate_values), 0)
        self.assertEqual(sample.candidate_payloads[sample.chosen_index], sample.chosen_payload)
        self.assertEqual(len(sample.action_names), len(sample.candidate_values[0]))

    def test_jsonl_round_trip_preserves_sample(self) -> None:
        result = collect_heuristic_samples(["1"], max_deals_per_seed=1, max_steps_per_seed=1_000)
        sample = result.samples[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"

            count = write_jsonl((sample,), path)
            loaded = read_jsonl(path)

        self.assertEqual(count, 1)
        self.assertEqual(loaded, (sample,))

    def test_collect_cli_writes_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cli-samples.jsonl"

            with redirect_stdout(StringIO()):
                exit_code = collect_main([str(path), "--seed", "1", "--max-deals", "1"])

            loaded = read_jsonl(path)
        self.assertEqual(exit_code, 0)
        self.assertGreater(len(loaded), 0)

    def test_bc_dimension_validation_accepts_collected_samples(self) -> None:
        result = collect_heuristic_samples(["1"], max_deals_per_seed=1, max_steps_per_seed=1_000)
        sample = result.samples[0]

        _validate_dimensions((sample,), len(sample.observation_values), len(sample.action_names))

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


if __name__ == "__main__":
    unittest.main()
