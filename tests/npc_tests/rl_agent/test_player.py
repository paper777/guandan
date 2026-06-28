from __future__ import annotations

import unittest
from pathlib import Path

from client.types import ActionRequest
from db.player import Player
from npc.rl_agent import RlAgentConfig, RlAgentPlayer
from npc.rl_agent.player import seat_snapshot_from_request


class RlAgentPlayerTests(unittest.TestCase):
    def test_rl_agent_is_player(self) -> None:
        self.assertIsInstance(RlAgentPlayer(), Player)

    def test_falls_back_to_heuristic_when_checkpoint_is_missing(self) -> None:
        action = RlAgentPlayer(RlAgentConfig(model_path=Path("/tmp/guandan-missing-rl-model.pt"))).choose_action(
            _lead_request()
        )

        self.assertEqual(action, {"type": "play_cards", "card_ids": ["D1-S-3"], "declared_type": "single"})

    def test_uses_model_selected_candidate_when_available(self) -> None:
        action = RlAgentPlayer(model_loader=_LastActionModel()).choose_action(_lead_request())

        self.assertEqual(action, {"type": "play_cards", "card_ids": ["D1-S-A"], "declared_type": "single"})

    def test_falls_back_when_model_raises(self) -> None:
        action = RlAgentPlayer(model_loader=_FailingModel()).choose_action(_lead_request())

        self.assertEqual(action, {"type": "play_cards", "card_ids": ["D1-S-3"], "declared_type": "single"})

    def test_builds_seat_snapshot_from_action_request(self) -> None:
        snapshot = seat_snapshot_from_request(_lead_request())

        self.assertEqual(snapshot.seat.value, "E")
        self.assertEqual(snapshot.hand, ("D1-S-A", "D1-S-3"))
        self.assertEqual(snapshot.public.hand_counts[snapshot.seat], 2)
        self.assertNotIn("other_hand", snapshot.public.current_trick or {})

    def test_builds_public_played_counts_and_pass_count_from_action_request(self) -> None:
        request = _lead_request()
        request.snapshot["public"]["current_trick"] = {"pass_count": 2}
        request.snapshot["public"]["played_card_counts"] = {"S-3": 1, "BJ": 2}

        snapshot = seat_snapshot_from_request(request)

        self.assertEqual((snapshot.public.current_trick or {}).get("pass_count"), 2)
        self.assertEqual(snapshot.public.played_card_counts, {"S-3": 1, "BJ": 2})


class _LastActionModel:
    def choose_action(self, snapshot, actions):
        return actions[-1]


class _FailingModel:
    def choose_action(self, snapshot, actions):
        raise RuntimeError("model failed")


def _lead_request() -> ActionRequest:
    return ActionRequest(
        "r-1",
        {"kind": "lead", "current_level": "2"},
        {
            "table_id": "table-1",
            "seat": "E",
            "hand": ["D1-S-A", "D1-S-3"],
            "public": {
                "table_id": "table-1",
                "phase": "PLAYING",
                "current_turn": "E",
                "acting_seat": "E",
                "current_level": "2",
                "level_by_team": {"EW": "2", "SN": "2"},
                "hand_counts": {"E": 2, "S": 2, "W": 2, "N": 2},
                "finish_order": [],
                "event_seq": 1,
            },
        },
    )


if __name__ == "__main__":
    unittest.main()
