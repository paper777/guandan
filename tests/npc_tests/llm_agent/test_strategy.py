from __future__ import annotations

import unittest
import tempfile

from client.types import ActionRequest
from npc.llm_agent import LlmAgentConfig
from npc.llm_agent import LlmAgentPlayer


class StrategyContextTests(unittest.TestCase):
    def test_strong_hand_exposes_objective_features_without_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = CapturingProvider()
            player = LlmAgentPlayer(LlmAgentConfig(storage_dir=tmp, seat="S"), provider=provider)

            player.choose_action(
                ActionRequest(
                    "r-1",
                    {"kind": "lead", "current_level": "2"},
                    {
                        "seat": "S",
                        "hand": ["D1-BJ", "D2-BJ", "D1-S-A", "D2-S-A"],
                        "public": {"hand_counts": {"E": 20, "S": 4, "W": 20, "N": 20}, "current_level": "2"},
                    },
                )
            )

            self.assertNotIn("role_estimate", provider.prompts[0]["strategy_context"])
            self.assertEqual(provider.prompts[0]["strategy_context"]["hand_features"]["control_card_count"], 4)

    def test_partner_near_finish_exposes_pressure_without_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = CapturingProvider()
            player = LlmAgentPlayer(LlmAgentConfig(storage_dir=tmp, seat="S"), provider=provider)

            player.choose_action(
                ActionRequest(
                    "r-1",
                    {"kind": "play_or_pass", "current_level": "2"},
                    {
                        "seat": "S",
                        "hand": ["D1-S-3", "D1-H-4", "D1-C-5"],
                        "public": {"hand_counts": {"E": 20, "S": 3, "W": 20, "N": 6}, "current_level": "2"},
                    },
                )
            )

            strategy = provider.prompts[0]["strategy_context"]
            self.assertNotIn("role_estimate", strategy)
            self.assertTrue(strategy["pressure"]["partner_near_finish"])


class CapturingProvider:
    def __init__(self) -> None:
        self.prompts = []

    def choose_action(self, prompt):
        self.prompts.append(prompt)
        if prompt["table_context"]["prompt_kind"] == "play_or_pass":
            return {"type": "pass", "thinking": "Support by passing."}
        return {"type": "play_cards", "card_ids": [prompt["snapshot"]["hand"][0]], "thinking": "Lead one."}


if __name__ == "__main__":
    unittest.main()
