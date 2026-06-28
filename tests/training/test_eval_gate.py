from __future__ import annotations

import unittest

from training.eval_gate import DummyTrainingPolicy, evaluate_matchup
from training.heuristic import HeuristicPolicy


class EvalGateTests(unittest.TestCase):
    def test_eval_gate_reports_fixed_policy_metrics(self) -> None:
        summary = evaluate_matchup(
            HeuristicPolicy(),
            DummyTrainingPolicy(),
            opponent_name="dummy",
            seeds=("gate-seed",),
            max_deals=1,
            max_steps=4,
        )

        self.assertEqual(summary.opponent, "dummy")
        self.assertGreaterEqual(summary.deals, 0)
        self.assertGreaterEqual(summary.win_rate, 0.0)
        self.assertLessEqual(summary.win_rate, 1.0)
        self.assertIn("max_steps", summary.stopped_reasons)


if __name__ == "__main__":
    unittest.main()
