from __future__ import annotations

import unittest

from client.types import ActionRequest
from db.player import Player
from npc.dummy_bot import DummyBotPlayer


class DummyBotPlayerTests(unittest.TestCase):
    def test_dummy_bot_is_player(self) -> None:
        self.assertIsInstance(DummyBotPlayer(), Player)

    def test_passes_when_not_leading(self) -> None:
        action = DummyBotPlayer().choose_action(
            ActionRequest("r-1", {"kind": "play_or_pass"}, {"hand": ["D1-S-3"]})
        )

        self.assertEqual(action, {"type": "pass"})

    def test_leads_lowest_card_when_forced(self) -> None:
        action = DummyBotPlayer().choose_action(
            ActionRequest("r-1", {"kind": "lead", "current_level": "2"}, {"hand": ["D1-S-A", "D1-S-3"]})
        )

        self.assertEqual(action, {"type": "play_cards", "card_ids": ["D1-S-3"]})

    def test_pays_highest_eligible_tribute(self) -> None:
        action = DummyBotPlayer().choose_action(
            ActionRequest(
                "r-1",
                {"kind": "tribute", "current_level": "2"},
                {"hand": ["D1-H-2", "D1-S-A", "D1-S-3"]},
            )
        )

        self.assertEqual(action, {"type": "submit_tribute", "card_id": "D1-S-A"})

    def test_uses_server_provided_legal_return_choices(self) -> None:
        action = DummyBotPlayer().choose_action(
            ActionRequest(
                "r-1",
                {"kind": "return_tribute", "current_level": "2", "eligible_card_ids": ["D1-S-10"]},
                {"hand": ["D1-S-10", "D1-S-A"]},
            )
        )

        self.assertEqual(action, {"type": "return_tribute", "card_id": "D1-S-10"})


if __name__ == "__main__":
    unittest.main()
