from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from npc.llm_agent.memory import MemoryAgent
from npc.llm_agent.prompts import (
    MEMORY_RULE_CONTEXT,
    MEMORY_PLAYER_ANALYSIS_PROMPT,
    MEMORY_TECHNIQUE_COMPACTION_PROMPT,
    MEMORY_TECHNIQUE_SUMMARY_PROMPT,
)
from npc.llm_agent.storage import JsonMemoryStore


class FakeMemoryProvider:
    def __init__(self):
        self.calls = []

    def complete_memory(self, *, system_prompt, context, max_output_tokens=None):
        self.calls.append((system_prompt, context, max_output_tokens))
        if system_prompt == MEMORY_TECHNIQUE_SUMMARY_PROMPT:
            return {"summary": "Use precise tempo transfer.", "techniques": ["Lead low to partner when safe."]}
        if system_prompt == MEMORY_PLAYER_ANALYSIS_PROMPT:
            return {
                "players": {
                    "Jade": {
                        "latest_seat": "S",
                        "personality": "aggressive",
                        "playing_style": "sprints from short hands",
                        "evidence": "Finished first after a low lead.",
                        "confidence": "medium",
                    }
                }
            }
        if system_prompt == MEMORY_TECHNIQUE_COMPACTION_PROMPT:
            return {
                "team_coordination": ["Let partner keep tempo when they are short."],
                "bomb_usage": [],
                "offensive_card_formation": [],
                "defensive_card_formation": [],
                "combo_removal": [],
                "others": [],
            }
        return {}


class LlmMemoryTests(unittest.TestCase):
    def test_memory_store_migrates_legacy_skills_to_techniques(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            path.write_text(json.dumps({"skills": ["Old skill"], "player_profiles": []}), encoding="utf-8")
            store = JsonMemoryStore(path, player_name="Jade", seat="S")

            memory = store.load()

            self.assertNotIn("skills", memory)
            self.assertEqual(memory["techniques"]["level1"][0]["summary"], "Old skill")
            self.assertEqual(memory["techniques"]["level2"]["team_coordination"], [])
            self.assertEqual(memory["player_profiles"], {})

    def test_memory_agent_summarizes_profiles_and_compacts_by_category(self) -> None:
        provider = FakeMemoryProvider()
        agent = MemoryAgent(provider, compaction_char_limit=10, max_output_tokens=321)
        memory = {
            "techniques": {
                "level1": [{"summary": "x" * 20, "techniques": ["x" * 20]}],
                "level2": {
                    "team_coordination": [],
                    "bomb_usage": [],
                    "offensive_card_formation": [],
                    "defensive_card_formation": [],
                    "combo_removal": [],
                    "others": [],
                },
            },
            "player_profiles": {},
        }
        events = [
            {
                "seq": 9,
                "type": "DealEnded",
                "payload": {"finish_order": ["S", "N", "E", "W"], "winning_team": "SN"},
            }
        ]

        agent.process_deal(
            memory,
            recent_actions=[{"kind": "observed_action", "actor_name": "Jade"}],
            events=events,
            players_by_seat={"S": "Jade", "W": "River"},
            observer_name="Jade",
        )

        self.assertEqual([call[2] for call in provider.calls], [321, 321, 321])
        self.assertEqual(provider.calls[0][1]["players_by_seat"]["S"], "Jade")
        self.assertEqual(provider.calls[0][1]["rule_context"], MEMORY_RULE_CONTEXT)
        self.assertEqual(provider.calls[1][1]["rule_context"], MEMORY_RULE_CONTEXT)
        self.assertEqual(provider.calls[2][1]["rule_context"], MEMORY_RULE_CONTEXT)
        self.assertEqual(memory["techniques"]["level1"], [])
        self.assertEqual(
            memory["techniques"]["level2"]["team_coordination"],
            ["Let partner keep tempo when they are short."],
        )
        self.assertEqual(memory["player_profiles"]["Jade"]["latest_seat"], "S")
        self.assertNotIn("last_deal_seq", memory["player_profiles"]["Jade"])
        self.assertEqual(memory["last_memory_deal_seq"], 9)


if __name__ == "__main__":
    unittest.main()
