from __future__ import annotations

import unittest

from guandan.cli import create_started_bot_table, play_one_bot_action, run_cli
from guandan.domain.state import MatchPhase


class CliTests(unittest.TestCase):
    def test_snapshot_command_outputs_started_table(self) -> None:
        result = run_cli(["snapshot", "--seed", "test-seed"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Table: cli-table", result.output)
        self.assertIn("Phase: PLAYING", result.output)
        self.assertIn("E: Bot E (27 cards)", result.output)

    def test_demo_command_outputs_events(self) -> None:
        result = run_cli(["demo", "--seed", "test-seed", "--turns", "3"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("played", result.output)
        self.assertIn("Table: cli-table", result.output)

    def test_play_one_bot_action_advances_state(self) -> None:
        actor = create_started_bot_table(seed="test-seed")
        before = actor.state.event_seq

        result = play_one_bot_action(actor)

        self.assertIsNone(result.rejection)
        self.assertEqual(actor.state.phase, MatchPhase.PLAYING)
        self.assertGreater(actor.state.event_seq, before)


if __name__ == "__main__":
    unittest.main()
