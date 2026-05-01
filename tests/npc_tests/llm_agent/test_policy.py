from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from client.api import ActionRequest
from npc.broker.broker import NpcBroker
from npc.common.player import Player
from npc.llm_agent import LlmAgentConfig, LlmAgentPlayer, LlmAgentPolicy
from npc.llm_agent.prompts import SYSTEM_PROMPT
from npc.llm_agent.skills import CARD_RECORDER_SKILL


class InvalidProvider:
    def choose_action(self, prompt):
        return {
            "type": "play_cards",
            "card_ids": ["D1-S-A"],
            "thinking": "This intentionally chooses a card that is not in hand.",
        }


class StaticProvider:
    def __init__(self, action):
        self.action = action
        self.prompts = []

    def choose_action(self, prompt):
        self.prompts.append(prompt)
        return dict(self.action)


class FakeClient:
    def __init__(self):
        self.calls = []

    def seat_snapshot(self, table_id, seat, controller_id):
        self.calls.append(("seat_snapshot", table_id, seat, controller_id))
        return {
            "public": {
                "phase": "PLAYING",
                "current_turn": seat,
                "event_seq": 4,
                "current_level": "2",
                "hand_counts": {"E": 27, "S": 1, "W": 1, "N": 27},
            },
            "seat": seat,
            "hand": ["D1-S-3"],
            "legal_action": "lead",
        }

    def play_cards(self, table_id, seat, controller_id, card_ids):
        self.calls.append(("play_cards", table_id, seat, controller_id, card_ids))
        return {
            "event_seq": 5,
            "events": [
                {
                    "seq": 5,
                    "type": "CardsPlayed",
                    "payload": {
                        "seat": seat,
                        "card_ids": list(card_ids),
                        "hand_type": "single",
                        "remaining_count": 0,
                    },
                }
            ],
        }

    def pass_turn(self, table_id, seat, controller_id):
        self.calls.append(("pass_turn", table_id, seat, controller_id))
        return {"event_seq": 5, "events": [{"seq": 5, "type": "PlayerPassed", "payload": {"seat": seat}}]}

    def submit_tribute(self, table_id, seat, controller_id, card_id):
        self.calls.append(("submit_tribute", table_id, seat, controller_id, card_id))
        return {"event_seq": 5, "events": []}

    def return_tribute(self, table_id, seat, controller_id, card_id):
        self.calls.append(("return_tribute", table_id, seat, controller_id, card_id))
        return {"event_seq": 5, "events": []}


class LlmAgentPolicyTests(unittest.TestCase):
    def test_policy_alias_points_to_player_class(self) -> None:
        self.assertIs(LlmAgentPolicy, LlmAgentPlayer)
        self.assertIsInstance(LlmAgentPolicy(), Player)

    def test_provider_prompt_includes_central_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = StaticProvider({"type": "play_cards", "card_ids": ["D1-S-3"], "thinking": "Lead low."})
            policy = LlmAgentPlayer(
                LlmAgentConfig(storage_dir=tmp, seat="S", personality="aggressive"),
                provider=provider,
            )

            policy.choose_action(_lead_request("S"))

            self.assertEqual(provider.prompts[-1]["system_prompt"], SYSTEM_PROMPT)
            self.assertIn("Tribute giver", provider.prompts[-1]["system_prompt"])
            self.assertIn("Never assume hidden cards", provider.prompts[-1]["system_prompt"])
            self.assertEqual(provider.prompts[-1]["table_context"]["partner"], "N")
            self.assertEqual(provider.prompts[-1]["table_context"]["opponents"], ["E", "W"])
            self.assertEqual(provider.prompts[-1]["strategy_context"]["role_estimate"], "balanced")
            self.assertEqual(provider.prompts[-1]["personality"]["type"], "aggressive")
            self.assertEqual(provider.prompts[-1]["strategy_context"]["personality"]["type"], "aggressive")
            self.assertEqual(provider.prompts[-1]["personality"]["risk_tolerance"], "high")
            self.assertEqual(
                provider.prompts[-1]["card_player"]["recommended_action"],
                {"type": "play_cards", "card_ids": ["D1-S-3"]},
            )
            self.assertIn("Bombs are tempo tools", provider.prompts[-1]["system_prompt"])
            self.assertIn("card_player", provider.prompts[-1]["system_prompt"])
            self.assertIn(CARD_RECORDER_SKILL, provider.prompts[-1]["skills"])

    def test_play_or_pass_prompt_includes_beating_card_player_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = StaticProvider({"type": "play_cards", "card_ids": ["D1-S-4"], "thinking": "Beat cheaply."})
            policy = LlmAgentPlayer(LlmAgentConfig(storage_dir=tmp, seat="S"), provider=provider)

            action = policy.choose_action(
                ActionRequest(
                    "r-1",
                    {
                        "kind": "play_or_pass",
                        "current_level": "2",
                        "current_trick": {"card_ids": ["D1-S-3"], "hand_type": "single", "last_play_seat": "E"},
                    },
                    {
                        "seat": "S",
                        "hand": ["D1-S-4", "D1-S-A"],
                        "public": {
                            "current_level": "2",
                            "current_turn": "S",
                            "current_trick": {"card_ids": ["D1-S-3"], "hand_type": "single", "last_play_seat": "E"},
                        },
                    },
                )
            )

            self.assertEqual(action["type"], "play_cards")
            self.assertEqual(action["card_ids"], ["D1-S-4"])
            self.assertEqual(
                provider.prompts[-1]["card_player"]["recommended_action"],
                {"type": "play_cards", "card_ids": ["D1-S-4"], "declared_type": "single"},
            )

    def test_explicit_player_storage_paths_are_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "custom-memory.json"
            action_path = Path(tmp) / "custom-actions.json"
            policy = LlmAgentPolicy(
                LlmAgentConfig(
                    player_name="South Agent",
                    memory_path=memory_path,
                    action_log_path=action_path,
                )
            )

            action = policy.choose_action(_lead_request("S"))

            self.assertEqual(action["type"], "play_cards")
            self.assertTrue(memory_path.exists())
            self.assertTrue(action_path.exists())
            self.assertEqual(_read_json(memory_path)["player_name"], "South Agent")

    def test_default_storage_paths_are_isolated_by_seat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policies = {
                seat: LlmAgentPolicy(LlmAgentConfig(storage_dir=tmp, seat=seat))
                for seat in ("S", "W", "N")
            }

            for seat, policy in policies.items():
                policy.choose_action(_lead_request(seat))

            paths = [policy.storage_paths[seat] for seat, policy in policies.items()]
            self.assertEqual(len(set(paths)), 3)
            for seat, (memory_path, action_path) in zip(("S", "W", "N"), paths, strict=True):
                self.assertEqual(memory_path, Path(tmp) / seat / "memory.json")
                self.assertEqual(action_path, Path(tmp) / seat / "actions.json")
                self.assertEqual(_read_json(memory_path)["seat"], seat)

    def test_invalid_provider_action_falls_back_and_records_thinking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            action_path = Path(tmp) / "actions.json"
            policy = LlmAgentPolicy(
                LlmAgentConfig(action_log_path=action_path, memory_path=Path(tmp) / "memory.json"),
                provider=InvalidProvider(),
            )

            action = policy.choose_action(_lead_request("S"))

            self.assertEqual(action["type"], "play_cards")
            self.assertEqual(action["card_ids"], ["D1-S-3"])
            self.assertIn("thinking", action)
            entries = _read_json(action_path)
            self.assertTrue(entries[0]["fallback_used"])
            self.assertIn("card-player fallback", entries[0]["thinking"])

    def test_provider_prompt_uses_only_that_player_recent_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            south_provider = StaticProvider(
                {
                    "type": "play_cards",
                    "card_ids": ["D1-S-3"],
                    "thinking": "Lead one low card.",
                    "memory_updates": {"skills": ["South skill"]},
                }
            )
            west_provider = StaticProvider(
                {
                    "type": "play_cards",
                    "card_ids": ["D1-S-3"],
                    "thinking": "Lead one low card.",
                    "memory_updates": {"skills": ["West skill"]},
                }
            )
            south = LlmAgentPolicy(LlmAgentConfig(storage_dir=tmp, seat="S"), provider=south_provider)
            west = LlmAgentPolicy(LlmAgentConfig(storage_dir=tmp, seat="W"), provider=west_provider)

            south.choose_action(_lead_request("S"))
            west.choose_action(_lead_request("W"))
            south.choose_action(_lead_request("S"))

            self.assertEqual(south_provider.prompts[-1]["recent_actions"][0]["seat"], "S")
            self.assertNotIn("West skill", _read_json(Path(tmp) / "S" / "memory.json")["skills"])
            self.assertNotIn("South skill", _read_json(Path(tmp) / "W" / "memory.json")["skills"])

    def test_broker_notifies_all_llm_agents_after_each_submitted_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broker = NpcBroker(FakeClient(), "table-1")
            south = LlmAgentPolicy(LlmAgentConfig(storage_dir=tmp, seat="S"))
            west = LlmAgentPolicy(LlmAgentConfig(storage_dir=tmp, seat="W"))
            broker.add_seat("S", south, "South Agent").controller_id = "c-S"
            broker.add_seat("W", west, "West Agent").controller_id = "c-W"

            actions = broker.poll_once("S")

            self.assertEqual(actions[0]["type"], "play_cards")
            west_log = _read_json(Path(tmp) / "W" / "actions.json")
            observed = [entry for entry in west_log if entry["kind"] == "observed_action"]
            self.assertEqual(observed[0]["actor_seat"], "S")
            self.assertEqual(observed[0]["response_events"][0]["type"], "CardsPlayed")

    def test_action_log_does_not_store_opponent_private_hands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            action_path = Path(tmp) / "actions.json"
            policy = LlmAgentPolicy(
                LlmAgentConfig(memory_path=Path(tmp) / "memory.json", action_log_path=action_path)
            )

            policy.choose_action(_lead_request("S"))

            entry = _read_json(action_path)[0]
            self.assertEqual(entry["snapshot"]["hand"], ["D1-S-3"])
            self.assertNotIn("hands", json.dumps(entry))


def _lead_request(seat: str) -> ActionRequest:
    return ActionRequest(
        request_id=f"r-{seat}",
        prompt={"kind": "lead", "current_level": "2"},
        snapshot={
            "table_id": "table-1",
            "seat": seat,
            "hand": ["D1-S-3"],
            "public": {
                "phase": "PLAYING",
                "event_seq": 4,
                "current_level": "2",
                "current_turn": seat,
                "hand_counts": {"E": 27, "S": 1, "W": 1, "N": 27},
            },
        },
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
