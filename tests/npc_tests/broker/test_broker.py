from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from npc.broker.broker import NpcBroker, load_default_player_profiles
from npc.common.player import Player
from npc.dummy_bot.policy import DummyBotPolicy
from npc.llm_agent import LlmAgentPlayer


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def join_agent(self, table_id, seat, display_name):
        self.calls.append(("join_agent", table_id, seat, display_name))
        return {"player_id": f"p-{seat}", "controller_id": f"c-{seat}"}

    def ready(self, table_id, seat, controller_id):
        self.calls.append(("ready", table_id, seat, controller_id))
        return {}

    def seat_snapshot(self, table_id, seat, controller_id):
        self.calls.append(("seat_snapshot", table_id, seat, controller_id))
        return {
            "public": {"current_turn": seat, "event_seq": 4},
            "seat": seat,
            "hand": ["D1-S-3"],
            "legal_action": "lead",
        }

    def play_cards(self, table_id, seat, controller_id, card_ids):
        self.calls.append(("play_cards", table_id, seat, controller_id, card_ids))
        return {}

    def pass_turn(self, table_id, seat, controller_id):
        self.calls.append(("pass_turn", table_id, seat, controller_id))
        return {}

    def submit_tribute(self, table_id, seat, controller_id, card_id):
        self.calls.append(("submit_tribute", table_id, seat, controller_id, card_id))
        return {}

    def return_tribute(self, table_id, seat, controller_id, card_id):
        self.calls.append(("return_tribute", table_id, seat, controller_id, card_id))
        return {}


class NpcBrokerTests(unittest.TestCase):
    def test_default_mixed_lineup_uses_named_profiles(self) -> None:
        broker = NpcBroker(FakeClient(), "table-1")

        seats = broker.add_default_players(("E", "S", "W", "N"), lineup="mixed")

        self.assertEqual([seat.display_name for seat in seats], ["Ming", "Jade", "River", "Atlas"])
        self.assertIsInstance(broker.seats["E"].policy, DummyBotPolicy)
        self.assertIsInstance(broker.seats["S"].policy, LlmAgentPlayer)
        self.assertIsInstance(broker.seats["W"].policy, LlmAgentPlayer)
        self.assertIsInstance(broker.seats["N"].policy, LlmAgentPlayer)
        self.assertIsInstance(broker.seats["E"].policy, Player)
        self.assertEqual(broker.seats["S"].policy.config.personality, "aggressive")
        self.assertEqual(broker.seats["W"].policy.config.personality, "balanced")
        self.assertEqual(broker.seats["N"].policy.config.personality, "defensive")

    def test_default_lineups_can_force_all_dummy_or_all_llm(self) -> None:
        dummy_broker = NpcBroker(FakeClient(), "table-1")
        llm_broker = NpcBroker(FakeClient(), "table-1")

        dummy_broker.add_default_players(("S", "W"), lineup="dummy")
        llm_broker.add_default_players(("S", "W"), lineup="llm")

        self.assertTrue(all(isinstance(seat.policy, DummyBotPolicy) for seat in dummy_broker.seats.values()))
        self.assertTrue(all(isinstance(seat.policy, LlmAgentPlayer) for seat in llm_broker.seats.values()))

    def test_default_profiles_can_be_loaded_from_data_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "default_players.json"
            path.write_text(
                json.dumps(
                    {
                        "players": [
                            {"seat": "S", "display_name": "South Custom", "kind": "dummy"},
                            {
                                "seat": "W",
                                "display_name": "West Custom",
                                "kind": "llm",
                                "personality": "defensive",
                                "provider_name": "codex-cli",
                                "codex_binary": "codex-test",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            profiles = load_default_player_profiles(path)
            broker = NpcBroker(FakeClient(), "table-1")
            seats = broker.add_default_players(("S", "W"), lineup="mixed", config_path=path)

        self.assertEqual(profiles[1].display_name, "South Custom")
        self.assertEqual(profiles[2].display_name, "West Custom")
        self.assertEqual(profiles[2].personality, "defensive")
        self.assertEqual([seat.display_name for seat in seats], ["South Custom", "West Custom"])
        self.assertIsInstance(broker.seats["S"].policy, DummyBotPolicy)
        self.assertIsInstance(broker.seats["W"].policy, LlmAgentPlayer)
        self.assertEqual(profiles[2].provider_name, "codex-cli")
        self.assertIsNone(profiles[2].model_name)
        self.assertEqual(profiles[2].codex_binary, "codex-test")
        self.assertEqual(broker.seats["W"].policy.config.model_name, "gpt-5.2")
        self.assertEqual(broker.seats["W"].policy.config.timeout_seconds, 120.0)
        self.assertEqual(broker.seats["W"].policy.config.personality, "defensive")

    def test_join_ready_and_poll_submit_are_owned_by_broker(self) -> None:
        client = FakeClient()
        broker = NpcBroker(client, "table-1")
        broker.add_seat("S", DummyBotPolicy(), "Dummy S")

        broker.join_and_ready_all()
        actions = broker.poll_once()

        self.assertEqual(actions, [{"type": "play_cards", "card_ids": ["D1-S-3"]}])
        self.assertEqual(
            client.calls,
            [
                ("join_agent", "table-1", "S", "Dummy S"),
                ("ready", "table-1", "S", "c-S"),
                ("seat_snapshot", "table-1", "S", "c-S"),
                ("play_cards", "table-1", "S", "c-S", ("D1-S-3",)),
            ],
        )

    def test_poll_uses_acting_seat_for_tribute_prompt(self) -> None:
        class TributeClient(FakeClient):
            def seat_snapshot(self, table_id, seat, controller_id):
                self.calls.append(("seat_snapshot", table_id, seat, controller_id))
                return {
                    "public": {"current_turn": "E", "acting_seat": seat, "event_seq": 9, "current_level": "2"},
                    "seat": seat,
                    "hand": ["D1-H-2", "D1-S-A", "D1-S-3"],
                    "legal_action": "tribute",
                }

        client = TributeClient()
        broker = NpcBroker(client, "table-1")
        seat = broker.add_seat("S", DummyBotPolicy(), "Dummy S")
        seat.controller_id = "c-S"

        actions = broker.poll_once()

        self.assertEqual(actions, [{"type": "submit_tribute", "card_id": "D1-S-A"}])
        self.assertEqual(
            client.calls,
            [
                ("seat_snapshot", "table-1", "S", "c-S"),
                ("submit_tribute", "table-1", "S", "c-S", "D1-S-A"),
            ],
        )

    def test_targeted_poll_only_submits_for_requested_seat(self) -> None:
        client = FakeClient()
        broker = NpcBroker(client, "table-1")
        south = broker.add_seat("S", DummyBotPolicy(), "Dummy S")
        west = broker.add_seat("W", DummyBotPolicy(), "Dummy W")
        south.controller_id = "c-S"
        west.controller_id = "c-W"

        actions = broker.poll_once("W")

        self.assertEqual(actions, [{"type": "play_cards", "card_ids": ["D1-S-3"]}])
        self.assertEqual(
            client.calls,
            [
                ("seat_snapshot", "table-1", "W", "c-W"),
                ("play_cards", "table-1", "W", "c-W", ("D1-S-3",)),
            ],
        )

    def test_prompt_includes_current_trick_for_play_or_pass(self) -> None:
        class CapturingPolicy:
            def __init__(self):
                self.requests = []

            def choose_action(self, request):
                self.requests.append(request)
                return {"type": "play_cards", "card_ids": ["D1-S-4"]}

        class TrickClient(FakeClient):
            def seat_snapshot(self, table_id, seat, controller_id):
                self.calls.append(("seat_snapshot", table_id, seat, controller_id))
                return {
                    "public": {
                        "current_turn": seat,
                        "event_seq": 4,
                        "current_level": "2",
                        "current_trick": {"card_ids": ["D1-S-3"], "hand_type": "single", "last_play_seat": "E"},
                    },
                    "seat": seat,
                    "hand": ["D1-S-4"],
                    "legal_action": "play_or_pass",
                }

        client = TrickClient()
        policy = CapturingPolicy()
        broker = NpcBroker(client, "table-1")
        seat = broker.add_seat("S", policy, "NPC S")
        seat.controller_id = "c-S"

        broker.poll_once("S")

        self.assertEqual(policy.requests[0].prompt["current_trick"]["card_ids"], ["D1-S-3"])


if __name__ == "__main__":
    unittest.main()
