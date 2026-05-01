from __future__ import annotations

import unittest

from client.api import ActionRequest
from npc.llm_agent.card_player import CardPlayerAdvisor


class CardPlayerAdvisorTests(unittest.TestCase):
    def test_lead_candidates_include_low_single_and_structures(self) -> None:
        advice = CardPlayerAdvisor().advise(
            ActionRequest(
                "r-1",
                {"kind": "lead", "current_level": "2"},
                {"hand": ["D1-S-3", "D2-S-3", "D1-H-4", "D1-C-4", "D2-D-4"]},
            )
        )

        self.assertEqual(advice["recommended_action"], {"type": "play_cards", "card_ids": ["D1-S-3"]})
        actions = [candidate["action"] for candidate in advice["candidates"]]
        self.assertIn({"type": "play_cards", "card_ids": ["D1-S-3"]}, actions)
        self.assertIn({"type": "play_cards", "card_ids": ["D1-S-3", "D2-S-3"], "declared_type": "pair"}, actions)
        self.assertIn(
            {"type": "play_cards", "card_ids": ["D1-C-4", "D1-H-4", "D2-D-4"], "declared_type": "three_of_a_kind"},
            actions,
        )

    def test_play_or_pass_recommends_lowest_beating_candidate_when_trick_is_known(self) -> None:
        advice = CardPlayerAdvisor().advise(
            ActionRequest(
                "r-1",
                {
                    "kind": "play_or_pass",
                    "current_level": "2",
                    "current_trick": {"card_ids": ["D1-S-3"], "hand_type": "single", "last_play_seat": "E"},
                },
                {"hand": ["D1-S-4", "D1-S-A"]},
            )
        )

        self.assertEqual(advice["recommended_action"], {"type": "play_cards", "card_ids": ["D1-S-4"], "declared_type": "single"})
        self.assertEqual(advice["candidates"][-1]["action"], {"type": "pass"})

    def test_play_or_pass_recommends_pass_when_no_known_beating_candidate(self) -> None:
        advice = CardPlayerAdvisor().advise(
            ActionRequest(
                "r-1",
                {
                    "kind": "play_or_pass",
                    "current_level": "2",
                    "current_trick": {"card_ids": ["D1-S-A"], "hand_type": "single", "last_play_seat": "E"},
                },
                {"hand": ["D1-S-3"]},
            )
        )

        self.assertEqual(advice["recommended_action"], {"type": "pass"})

    def test_tribute_excludes_red_heart_level_card(self) -> None:
        advice = CardPlayerAdvisor().advise(
            ActionRequest(
                "r-1",
                {"kind": "tribute", "current_level": "2"},
                {"hand": ["D1-H-2", "D1-S-A", "D1-S-3"]},
            )
        )

        self.assertEqual(advice["recommended_action"], {"type": "submit_tribute", "card_id": "D1-S-A"})


if __name__ == "__main__":
    unittest.main()
