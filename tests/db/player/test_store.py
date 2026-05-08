from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from db.player.store import (
    DEFAULT_PLAYER_PROFILES,
    load_player_database,
    profile_assignments,
    record_profile_result,
)


class PlayerDatabaseTests(unittest.TestCase):
    def test_missing_database_uses_default_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database = load_player_database(Path(tmp) / "missing")

        self.assertEqual([profile.display_name for profile in database.profiles], ["Ming", "Jade", "River", "Atlas"])

    def test_loads_player_directory_and_preserves_split_files_on_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "players.json").write_text(
                json.dumps({"players": ["also"]}),
                encoding="utf-8",
            )
            player_dir = root / "also"
            player_dir.mkdir()
            (player_dir / "profile.json").write_text(
                json.dumps(
                    {
                        "display_name": "also",
                        "kind": "llm",
                        "favorite_color": "green",
                    }
                ),
                encoding="utf-8",
            )
            (player_dir / "llm_config.json").write_text(
                json.dumps({"provider_name": "codex-cli"}),
                encoding="utf-8",
            )
            (player_dir / "statistics.json").write_text(
                json.dumps({"score": 9}),
                encoding="utf-8",
            )

            database = load_player_database(root)
            database.save()
            saved_profile = json.loads((player_dir / "profile.json").read_text(encoding="utf-8"))
            saved_llm_config = json.loads((player_dir / "llm_config.json").read_text(encoding="utf-8"))
            saved_statistics = json.loads((player_dir / "statistics.json").read_text(encoding="utf-8"))
            saved_index = json.loads((root / "players.json").read_text(encoding="utf-8"))
            created_files = {path.name for path in player_dir.iterdir()}

        profile = database.profiles[0]
        self.assertEqual(profile.preferred_seat, "E")
        self.assertEqual(profile.llm_config.provider_name, "codex-cli")
        self.assertEqual(profile.statistics.score, 9)
        self.assertNotIn("seat", saved_profile)
        self.assertEqual(saved_profile["favorite_color"], "green")
        self.assertEqual(saved_llm_config["provider_name"], "codex-cli")
        self.assertEqual(saved_statistics["score"], 9)
        self.assertEqual(saved_index["players"], ["also"])
        self.assertEqual(
            created_files,
            {"actions.json", "llm_config.json", "memory.json", "profile.json", "statistics.json"},
        )

    def test_index_rejects_seat_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "players.json"
            path.write_text(
                json.dumps({"players": [{"directory": "South", "seat": "S"}]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must not contain seat"):
                load_player_database(path)

    def test_profile_assignment_honors_exclusions_and_fills_defaults(self) -> None:
        assignments = profile_assignments(
            [DEFAULT_PLAYER_PROFILES[0]],
            ("E", "S"),
            shuffle_seed=1,
            exclude_profile_keys={"Jade"},
        )

        self.assertEqual([profile.profile_key for profile, _ in assignments], ["Ming", "River"])
        self.assertEqual({seat for _, seat in assignments}, {"E", "S"})

    def test_record_profile_result_updates_deal_and_match_statistics(self) -> None:
        profile = DEFAULT_PLAYER_PROFILES[0]

        profile = record_profile_result(profile, kind="deal", won=True, score_delta=3)
        profile = record_profile_result(profile, kind="match", won=False)

        self.assertEqual(profile.statistics.deal_count, 1)
        self.assertEqual(profile.statistics.deal_wins, 1)
        self.assertEqual(profile.statistics.deal_win_rate, 1.0)
        self.assertEqual(profile.statistics.score, 3)
        self.assertEqual(profile.statistics.match_count, 1)
        self.assertEqual(profile.statistics.match_wins, 0)
        self.assertEqual(profile.statistics.match_win_rate, 0.0)

    def test_duplicate_profile_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "players.json").write_text(
                json.dumps(
                    {
                        "players": [
                            "One",
                            "Two",
                        ]
                    }
                ),
                encoding="utf-8",
            )
            for name in ("One", "Two"):
                player_dir = root / name
                player_dir.mkdir()
                (player_dir / "profile.json").write_text(
                    json.dumps({"id": "same", "display_name": name, "kind": "dummy"}),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(ValueError, "duplicate player profile"):
                load_player_database(root)


if __name__ == "__main__":
    unittest.main()
