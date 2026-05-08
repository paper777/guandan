from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from client.broker import NpcBroker
from db.player import Player, load_player_profiles
from npc.dummy_bot import DummyBotPlayer
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
    def test_mixed_lineup_uses_named_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broker = NpcBroker(FakeClient(), "table-1", player_db_path=Path(tmp) / "players.json")

            seats = broker.add_players(("E", "S", "W", "N"), lineup="mixed", shuffle_seed=0)

        self.assertEqual([seat.display_name for seat in seats], ["Ming", "Jade", "River", "Atlas"])
        self.assertEqual([seat.seat for seat in seats], ["W", "E", "S", "N"])
        self.assertEqual([seat.profile_seat for seat in seats], ["E", "S", "W", "N"])
        self.assertIsInstance(broker.seats["W"].policy, DummyBotPlayer)
        self.assertIsInstance(broker.seats["E"].policy, LlmAgentPlayer)
        self.assertIsInstance(broker.seats["S"].policy, LlmAgentPlayer)
        self.assertIsInstance(broker.seats["N"].policy, LlmAgentPlayer)
        self.assertIsInstance(broker.seats["W"].policy, Player)
        self.assertEqual(broker.seats["E"].policy.config.personality, "aggressive")
        self.assertEqual(broker.seats["E"].policy.config.seat, "E")
        self.assertEqual(broker.seats["S"].policy.config.personality, "balanced")
        self.assertEqual(broker.seats["S"].policy.config.seat, "S")
        self.assertEqual(broker.seats["N"].policy.config.personality, "defensive")
        self.assertEqual(broker.seats["N"].policy.config.seat, "N")

    def test_lineups_can_force_all_dummy_or_all_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            player_db_path = Path(tmp) / "players.json"
            dummy_broker = NpcBroker(FakeClient(), "table-1", player_db_path=player_db_path)
            llm_broker = NpcBroker(FakeClient(), "table-1", player_db_path=player_db_path)

            dummy_broker.add_players(("S", "W"), lineup="dummy", shuffle_seed=0)
            llm_broker.add_players(("S", "W"), lineup="llm", shuffle_seed=0)

        self.assertTrue(all(isinstance(seat.policy, DummyBotPlayer) for seat in dummy_broker.seats.values()))
        self.assertTrue(all(isinstance(seat.policy, LlmAgentPlayer) for seat in llm_broker.seats.values()))

    def test_player_profiles_can_be_loaded_from_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "players.json").write_text(
                json.dumps(
                    {
                        "players": [
                            "South-Custom",
                            "West-Custom",
                        ]
                    }
                ),
                encoding="utf-8",
            )
            south_dir = path / "South-Custom"
            south_dir.mkdir()
            (south_dir / "profile.json").write_text(
                json.dumps({"display_name": "South Custom", "kind": "dummy"}),
                encoding="utf-8",
            )
            west_dir = path / "West-Custom"
            west_dir.mkdir()
            (west_dir / "profile.json").write_text(
                json.dumps(
                    {
                        "display_name": "West Custom",
                        "kind": "llm",
                        "personality": "defensive",
                    }
                ),
                encoding="utf-8",
            )
            (west_dir / "llm_config.json").write_text(
                json.dumps(
                    {
                        "provider_name": "codex-cli",
                        "codex_binary": "codex-test",
                        "memory_compaction_char_limit": 123,
                        "memory_recent_deal_scan_limit": 45,
                        "memory_max_output_tokens": 678,
                    }
                ),
                encoding="utf-8",
            )
            (west_dir / "statistics.json").write_text(json.dumps({"score": 7}), encoding="utf-8")

            profiles = load_player_profiles(path)
            broker = NpcBroker(FakeClient(), "table-1", player_db_path=path)
            seats = broker.add_players(("S", "W"), lineup="mixed", shuffle_seed=1)

        self.assertEqual(profiles[0].display_name, "South Custom")
        self.assertEqual(profiles[1].display_name, "West Custom")
        self.assertEqual(profiles[1].personality, "defensive")
        self.assertEqual([seat.display_name for seat in seats], ["West Custom", "South Custom"])
        self.assertEqual([seat.seat for seat in seats], ["W", "S"])
        self.assertIsInstance(broker.seats["S"].policy, DummyBotPlayer)
        self.assertIsInstance(broker.seats["W"].policy, LlmAgentPlayer)
        self.assertEqual(profiles[1].provider_name, "codex-cli")
        self.assertEqual(profiles[1].score, 7)
        self.assertIsNone(profiles[1].model_name)
        self.assertEqual(profiles[1].codex_binary, "codex-test")
        self.assertEqual(broker.seats["W"].policy.config.model_name, "gpt-5.2")
        self.assertEqual(broker.seats["W"].policy.config.timeout_seconds, 120.0)
        self.assertEqual(broker.seats["W"].policy.config.personality, "defensive")
        self.assertEqual(broker.seats["W"].policy.config.memory_compaction_char_limit, 123)
        self.assertEqual(broker.seats["W"].policy.config.memory_recent_deal_scan_limit, 45)
        self.assertEqual(broker.seats["W"].policy.config.memory_max_output_tokens, 678)

    def test_result_events_update_player_database_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "players.json").write_text(
                json.dumps(
                    {
                        "players": [
                            "South",
                            "West",
                        ]
                    }
                ),
                encoding="utf-8",
            )
            south_dir = path / "South"
            south_dir.mkdir()
            (south_dir / "profile.json").write_text(
                json.dumps({"display_name": "South", "kind": "dummy"}),
                encoding="utf-8",
            )
            west_dir = path / "West"
            west_dir.mkdir()
            (west_dir / "profile.json").write_text(
                json.dumps({"display_name": "West", "kind": "dummy"}),
                encoding="utf-8",
            )
            (west_dir / "statistics.json").write_text(
                json.dumps({"deal_count": 4, "deal_wins": 2, "score": 5}),
                encoding="utf-8",
            )
            broker = NpcBroker(FakeClient(), "table-1", player_db_path=path)
            seats = broker.add_players(("S", "W"), lineup="mixed", shuffle_seed=1)

            response = {
                "events": [
                    {"seq": 10, "type": "DealEnded", "payload": {"winning_team": "SN"}},
                    {"seq": 11, "type": "MatchEnded", "payload": {"winning_team": "SN"}},
                    {"seq": 10, "type": "DealEnded", "payload": {"winning_team": "SN"}},
                ],
                "snapshot": {"level_by_team": {"EW": "K", "SN": "2"}},
            }
            broker._notify_action_observers(seats[0], {"type": "pass"}, response)

            players = {
                player_dir.name: json.loads((player_dir / "statistics.json").read_text(encoding="utf-8"))
                for player_dir in path.iterdir()
                if player_dir.is_dir()
            }

        self.assertEqual(players["South"]["deal_count"], 1)
        self.assertEqual(players["South"]["deal_wins"], 1)
        self.assertEqual(players["South"]["deal_win_rate"], 1.0)
        self.assertEqual(players["South"]["score"], 2)
        self.assertEqual(players["South"]["match_count"], 1)
        self.assertEqual(players["South"]["match_wins"], 1)
        self.assertEqual(players["South"]["match_win_rate"], 1.0)
        self.assertEqual(players["West"]["deal_count"], 5)
        self.assertEqual(players["West"]["deal_wins"], 2)
        self.assertEqual(players["West"]["deal_win_rate"], 0.4)
        self.assertEqual(players["West"]["score"], 3)
        self.assertEqual(players["West"]["match_count"], 1)
        self.assertEqual(players["West"]["match_wins"], 0)
        self.assertEqual(players["West"]["match_win_rate"], 0.0)

    def test_join_ready_and_poll_submit_are_owned_by_broker(self) -> None:
        client = FakeClient()
        broker = NpcBroker(client, "table-1")
        broker.add_seat("S", DummyBotPlayer(), "Dummy S")

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
        seat = broker.add_seat("S", DummyBotPlayer(), "Dummy S")
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
        south = broker.add_seat("S", DummyBotPlayer(), "Dummy S")
        west = broker.add_seat("W", DummyBotPlayer(), "Dummy W")
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

    def test_broker_passes_player_names_to_policy_and_observers(self) -> None:
        class NamedTableClient(FakeClient):
            def seat_snapshot(self, table_id, seat, controller_id):
                snapshot = super().seat_snapshot(table_id, seat, controller_id)
                snapshot["public"]["seats"] = _named_seats()
                return snapshot

            def play_cards(self, table_id, seat, controller_id, card_ids):
                response = super().play_cards(table_id, seat, controller_id, card_ids)
                response["snapshot"] = {"seats": _named_seats()}
                return response

        class CapturingPolicy:
            def __init__(self):
                self.requests = []
                self.observations = []

            def choose_action(self, request):
                self.requests.append(request)
                return {"type": "play_cards", "card_ids": ["D1-S-3"]}

            def observe_action(self, observation):
                self.observations.append(observation)

        def _named_seats():
            return {
                "E": {"display_name": "Human East", "kind": "human"},
                "S": {"display_name": "Jade", "kind": "agent"},
                "W": {"display_name": "River", "kind": "agent"},
                "N": {"display_name": "Human North", "kind": "human"},
            }

        client = NamedTableClient()
        south_policy = CapturingPolicy()
        west_policy = CapturingPolicy()
        broker = NpcBroker(client, "table-1")
        south = broker.add_seat("S", south_policy, "Jade")
        west = broker.add_seat("W", west_policy, "River")
        south.controller_id = "c-S"
        west.controller_id = "c-W"

        broker.poll_once("S")

        self.assertEqual(
            south_policy.requests[0].snapshot["players_by_seat"],
            {"E": "Human East", "S": "Jade", "W": "River", "N": "Human North"},
        )
        self.assertEqual(west_policy.observations[0]["actor_name"], "Jade")
        self.assertEqual(west_policy.observations[0]["observer_name"], "River")
        self.assertEqual(
            west_policy.observations[0]["players_by_seat"],
            {"E": "Human East", "S": "Jade", "W": "River", "N": "Human North"},
        )

    def test_rotate_seats_after_match_moves_whole_broker_bundles(self) -> None:
        client = FakeClient()
        broker = NpcBroker(client, "table-1")
        south = broker.add_seat("S", DummyBotPlayer(), "South")
        west = broker.add_seat("W", DummyBotPlayer(), "West")
        north = broker.add_seat("N", DummyBotPlayer(), "North")
        for seat in (south, west, north):
            seat.player_id = f"p-{seat.seat}"
            seat.controller_id = f"c-{seat.seat}"

        seat_map = broker.rotate_seats_after_match(shuffle_seed=2)

        self.assertNotEqual(seat_map, {"S": "S", "W": "W", "N": "N"})
        self.assertEqual(set(broker.seats), {"S", "W", "N"})
        self.assertIs(broker.seats[seat_map["S"]], south)
        self.assertIs(broker.seats[seat_map["W"]], west)
        self.assertIs(broker.seats[seat_map["N"]], north)
        self.assertTrue(all(seat.player_id == "" and seat.controller_id == "" for seat in broker.seats.values()))


if __name__ == "__main__":
    unittest.main()
