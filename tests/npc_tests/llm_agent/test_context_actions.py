from __future__ import annotations

import unittest

from client.types import ActionRequest
from npc.llm_agent.actions import validate_action
from npc.llm_agent.context import AgentRequestContext


class AgentRequestContextTests(unittest.TestCase):
    def test_builds_table_context_from_prompt_and_public_snapshot(self) -> None:
        context = AgentRequestContext.from_request(
            ActionRequest(
                "r-1",
                {"kind": "lead", "current_level": "3"},
                {
                    "seat": "S",
                    "hand": ["D1-S-3"],
                    "public": {
                        "deal_id": 2,
                        "phase": "PLAYING",
                        "event_seq": 9,
                        "current_level": "2",
                        "current_turn": "S",
                        "hand_counts": {"E": 27, "S": 1, "W": 27, "N": 1},
                    },
                },
            )
        )

        self.assertEqual(context.seat, "S")
        self.assertEqual(context.hand, ("D1-S-3",))
        self.assertEqual(context.current_level, "3")
        self.assertEqual(context.table_context["partner"], "N")
        self.assertEqual(context.table_context["opponents"], ["E", "W"])
        self.assertEqual(context.table_context["deal_id"], 2)
        self.assertEqual(context.table_context["phase"], "PLAYING")
        self.assertEqual(context.table_context["event_seq"], 9)
        self.assertEqual(context.table_context["current_level"], "3")


class ActionValidationTests(unittest.TestCase):
    def test_validates_normalized_play_cards_action(self) -> None:
        context = AgentRequestContext.from_request(
            ActionRequest("r-1", {"kind": "lead"}, {"hand": ["D1-S-3", "D1-S-4"]})
        )

        self.assertEqual(
            validate_action({"type": "play_cards", "card_ids": ["D1-S-3"], "declared_type": "single"}, context),
            {"type": "play_cards", "card_ids": ["D1-S-3"], "declared_type": "single"},
        )

    def test_rejects_wrong_prompt_kind_duplicate_cards_and_cards_outside_hand(self) -> None:
        tribute_context = AgentRequestContext.from_request(
            ActionRequest("r-1", {"kind": "tribute"}, {"hand": ["D1-S-3"]})
        )
        lead_context = AgentRequestContext.from_request(
            ActionRequest("r-2", {"kind": "lead"}, {"hand": ["D1-S-3"]})
        )

        self.assertIsNone(validate_action({"type": "pass"}, lead_context))
        self.assertIsNone(validate_action({"type": "play_cards", "card_ids": ["D1-S-3"]}, tribute_context))
        self.assertIsNone(validate_action({"type": "play_cards", "card_ids": ["D1-S-3", "D1-S-3"]}, lead_context))
        self.assertIsNone(validate_action({"type": "play_cards", "card_ids": ["D1-S-4"]}, lead_context))


if __name__ == "__main__":
    unittest.main()
