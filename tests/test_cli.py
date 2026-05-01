from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from client.api import GuandanClientError
from client.cli import (
    CliSession,
    drive_bot_turns,
    format_card_id,
    format_command_response,
    format_friend_mark,
    format_hand,
    format_npc_metadata,
    format_public_snapshot,
    format_seat_snapshot,
    resolve_card_inputs,
    run_cli,
)


class FakeClient:
    def __init__(self) -> None:
        self.table_id = "table-1"
        self.seats: dict[str, dict[str, str]] = {}
        self.phase = "WAITING_FOR_PLAYERS"
        self.current_turn: str | None = None
        self.event_seq = 0
        self.calls: list[tuple] = []
        self.hands = {
            "E": ["D1-H-4", "D1-C-3", "D2-S-3"],
            "S": ["D1-H-4"],
            "W": ["D1-C-4"],
            "N": ["D1-D-4"],
        }

    def create_table(self):
        self.calls.append(("create_table",))
        return {"table_id": self.table_id}

    def table_snapshot(self, table_id):
        self.calls.append(("table_snapshot", table_id))
        return self._snapshot()

    def join_human(self, table_id, seat, *, player_id=None, controller_id=None, display_name=None):
        self.calls.append(("join_human", table_id, seat, player_id, controller_id, display_name))
        self.seats[seat] = {"display_name": display_name or player_id, "kind": "human"}
        return {"controller_id": controller_id, "snapshot": self._snapshot()}

    def join_local_bot(self, table_id, seat, *, player_id=None, controller_id=None, display_name=None):
        self.calls.append(("join_local_bot", table_id, seat, player_id, controller_id, display_name))
        self.seats[seat] = {"display_name": display_name or player_id, "kind": "bot"}
        return {"controller_id": controller_id, "snapshot": self._snapshot()}

    def join_agent(self, table_id, seat, display_name):
        self.calls.append(("join_agent", table_id, seat, display_name))
        self.seats[seat] = {"display_name": display_name, "kind": "agent"}
        return {"player_id": f"agent-{seat}", "controller_id": f"agent-controller-{seat}"}

    def ready(self, table_id, seat, controller_id):
        self.calls.append(("ready", table_id, seat, controller_id))
        return {"events": [], "snapshot": self._snapshot()}

    def start(self, table_id, *, seed=None):
        self.calls.append(("start", table_id, seed))
        self.phase = "PLAYING"
        self.current_turn = "E"
        return {"events": [{"seq": 1, "type": "MatchStarted", "payload": {"table_id": table_id}}], "snapshot": self._snapshot()}

    def seat_snapshot(self, table_id, seat, controller_id):
        self.calls.append(("seat_snapshot", table_id, seat, controller_id))
        legal_action = None
        if seat == self.current_turn:
            legal_action = "lead" if seat == "E" else "play_or_pass"
        return {
            "public": self._snapshot(),
            "seat": seat,
            "hand": list(self.hands[seat]),
            "legal_action": legal_action,
        }

    def play_cards(self, table_id, seat, controller_id, card_ids, *, declared_type=None):
        self.calls.append(("play_cards", table_id, seat, controller_id, card_ids, declared_type))
        for card_id in card_ids:
            self.hands[seat].remove(card_id)
        self.current_turn = {"E": "S", "S": "W", "W": "N", "N": "E"}[seat]
        self.event_seq += 1
        return {
            "events": [{"seq": self.event_seq, "type": "CardsPlayed", "payload": {"seat": seat, "hand_type": "single", "card_ids": list(card_ids)}}],
            "snapshot": self._snapshot(),
        }

    def pass_turn(self, table_id, seat, controller_id):
        self.calls.append(("pass_turn", table_id, seat, controller_id))
        self.current_turn = {"E": "S", "S": "W", "W": "N", "N": "E"}[seat]
        self.event_seq += 1
        return {
            "events": [{"seq": self.event_seq, "type": "PlayerPassed", "payload": {"seat": seat}}],
            "snapshot": self._snapshot(),
        }

    def _snapshot(self):
        return {
            "table_id": self.table_id,
            "phase": self.phase,
            "seats": self.seats,
            "hand_counts": {seat: len(hand) for seat, hand in self.hands.items()},
            "current_turn": self.current_turn,
            "finish_order": [],
            "event_seq": self.event_seq,
        }


class CliTests(unittest.TestCase):
    def test_default_play_creates_human_and_three_mixed_broker_agents(self) -> None:
        client = FakeClient()

        result = run_cli([], input_fn=lambda prompt: "quit", client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Table table-1 | You are E", result.output)
        self.assertIn("PLAYING | Seat E | Turn E", result.output)
        self.assertIn("Players: E human-E 3 | S Jade 1 [", result.output)
        self.assertIn("W(F) River 1 [", result.output)
        self.assertIn("N Atlas 1 [", result.output)
        self.assertIn("Hand: ♠️ 3  ♣️ 3  ♥️ 4", result.output)
        self.assertIn(("join_human", "table-1", "E", "human-E", "human-controller-E", "human-E"), client.calls)
        self.assertIn(("join_agent", "table-1", "S", "Jade"), client.calls)
        self.assertIn(("join_agent", "table-1", "W", "River"), client.calls)
        self.assertIn(("join_agent", "table-1", "N", "Atlas"), client.calls)
        self.assertNotIn(("join_local_bot", "table-1", "S", "bot-S", "bot-controller-S", "Bot S"), client.calls)
        self.assertIn(("start", "table-1", "cli-demo"), client.calls)

    def test_human_play_readable_card_label_then_drives_bot_passes(self) -> None:
        client = FakeClient()
        commands = iter(["play C3", "quit"])

        result = run_cli(["--npc-lineup", "dummy"], input_fn=lambda prompt: next(commands), client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertIn(("play_cards", "table-1", "E", "human-controller-E", ("D1-C-3",), None), client.calls)
        self.assertIn(("pass_turn", "table-1", "S", "agent-controller-S"), client.calls)
        self.assertIn(("pass_turn", "table-1", "W", "agent-controller-W"), client.calls)
        self.assertIn(("pass_turn", "table-1", "N", "agent-controller-N"), client.calls)
        self.assertIn("1: E played single [♣️ 3]", result.output)
        self.assertIn("2: S passed", result.output)
        self.assertIn("3: W passed", result.output)
        self.assertIn("4: N passed", result.output)

    def test_human_timeout_refreshes_before_processing_input_and_prints_bot_actions(self) -> None:
        client = FakeClient()
        commands = iter(["", "quit"])
        timed_out = False

        def input_after_timeout(prompt):
            nonlocal timed_out
            if not timed_out and client.current_turn == "E":
                timed_out = True
                client.current_turn = "S"
                client.event_seq += 1
            return next(commands)

        result = run_cli(["--npc-lineup", "dummy"], input_fn=input_after_timeout, client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertIn(("pass_turn", "table-1", "S", "agent-controller-S"), client.calls)
        self.assertIn(("pass_turn", "table-1", "W", "agent-controller-W"), client.calls)
        self.assertIn(("pass_turn", "table-1", "N", "agent-controller-N"), client.calls)
        self.assertIn("2: S passed", result.output)
        self.assertIn("3: W passed", result.output)
        self.assertIn("4: N passed", result.output)

    def test_human_input_deadline_refreshes_without_command(self) -> None:
        client = FakeClient()
        timed_out = False

        def input_timeout(prompt):
            nonlocal timed_out
            if not timed_out and client.current_turn == "E":
                timed_out = True
                client.current_turn = "S"
                client.event_seq += 1
                return None
            return "quit"

        result = run_cli(["--npc-lineup", "dummy"], input_fn=input_timeout, client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertIn(("pass_turn", "table-1", "S", "agent-controller-S"), client.calls)
        self.assertIn(("pass_turn", "table-1", "W", "agent-controller-W"), client.calls)
        self.assertIn(("pass_turn", "table-1", "N", "agent-controller-N"), client.calls)
        self.assertIn("2: S passed", result.output)
        self.assertIn("3: W passed", result.output)
        self.assertIn("4: N passed", result.output)

    def test_cli_dummy_lineup_uses_named_dummy_players(self) -> None:
        client = FakeClient()

        result = run_cli(["--npc-lineup", "dummy"], input_fn=lambda prompt: "quit", client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertIn(("join_agent", "table-1", "S", "Jade"), client.calls)
        self.assertIn(("join_agent", "table-1", "W", "River"), client.calls)
        self.assertIn(("join_agent", "table-1", "N", "Atlas"), client.calls)

    def test_cli_llm_lineup_uses_named_llm_players(self) -> None:
        client = FakeClient()

        result = run_cli(["--npc-lineup", "llm"], input_fn=lambda prompt: "quit", client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertIn(("join_agent", "table-1", "S", "Jade"), client.calls)
        self.assertIn(("join_agent", "table-1", "W", "River"), client.calls)
        self.assertIn(("join_agent", "table-1", "N", "Atlas"), client.calls)

    def test_cli_uses_custom_npc_player_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "players.json"
            path.write_text(
                json.dumps(
                    {
                        "players": [
                            {"seat": "S", "display_name": "South Config", "kind": "dummy"},
                            {"seat": "W", "display_name": "West Config", "kind": "dummy"},
                            {"seat": "N", "display_name": "North Config", "kind": "dummy"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            client = FakeClient()

            result = run_cli(
                ["--npc-player-config", str(path), "--npc-lineup", "mixed"],
                input_fn=lambda prompt: "quit",
                client=client,
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn(("join_agent", "table-1", "S", "South Config"), client.calls)
        self.assertIn(("join_agent", "table-1", "W", "West Config"), client.calls)
        self.assertIn(("join_agent", "table-1", "N", "North Config"), client.calls)

    def test_card_formatter_hides_first_deck_and_uses_suit_emoji(self) -> None:
        self.assertEqual(format_card_id("D1-S-3"), "♠️ 3")
        self.assertEqual(format_card_id("D2-H-10"), "♥️ 10")
        self.assertEqual(format_card_id("D1-SJ"), "🃏SJ")

    def test_hand_formatter_sorts_by_number_then_suit(self) -> None:
        self.assertEqual(
            format_hand(["D1-C-3", "D1-H-2", "D1-S-3", "D2-D-2", "D1-BJ", "D1-SJ"]),
            "♥️ 2  ♦️ 2  ♠️ 3  ♣️ 3  🃏SJ  🃏BJ",
        )

    def test_numeric_card_input_uses_sorted_hand_order(self) -> None:
        self.assertEqual(
            resolve_card_inputs(["1", "3"], {"hand": ["D1-C-3", "D1-H-2", "D1-S-3"]}),
            ("D1-H-2", "D1-C-3"),
        )

    def test_readable_card_input_resolves_against_hand(self) -> None:
        self.assertEqual(
            resolve_card_inputs(["S3", "♥2", "SJ"], {"hand": ["D1-SJ", "D1-H-2", "D2-S-3"]}),
            ("D2-S-3", "D1-H-2", "D1-SJ"),
        )

    def test_repeated_readable_card_input_uses_distinct_physical_cards(self) -> None:
        self.assertEqual(
            resolve_card_inputs(["S3", "S3"], {"hand": ["D2-S-3", "D1-S-3"]}),
            ("D1-S-3", "D2-S-3"),
        )

    def test_bot_turn_race_refreshes_instead_of_printing_not_your_turn(self) -> None:
        class RaceBroker:
            seats = {"S": object()}

            def poll_once_results(self, seat):
                raise GuandanClientError(400, "NOT_YOUR_TURN", {"rejection": {"code": "NOT_YOUR_TURN"}})

        client = FakeClient()
        client.phase = "PLAYING"
        client.current_turn = "S"
        output = []

        snapshot = drive_bot_turns(
            client,
            CliSession("table-1", "E", "human-controller-E", RaceBroker(), {}),
            client._snapshot(),
            output.append,
            4,
        )

        self.assertEqual(snapshot["current_turn"], "S")
        self.assertEqual(output, [])

    def test_format_npc_metadata_shows_llm_provider_and_model(self) -> None:
        class Policy:
            config = type(
                "Config",
                (),
                {"provider_name": "codex-cli", "model_name": "gpt-5.2"},
            )()

        self.assertEqual(format_npc_metadata(Policy()), "codex-cli/gpt-5.2")

    def test_public_snapshot_shows_timer_when_deadline_is_present(self) -> None:
        output = format_public_snapshot(
            {
                "table_id": "table-1",
                "phase": "PLAYING",
                "event_seq": 1,
                "current_turn": "E",
                "action_deadline_epoch_ms": 9_999_999_999_999,
                "seats": {},
                "hand_counts": {},
                "finish_order": [],
            }
        )

        self.assertIn("Timer:", output)
        self.assertIn("PLAYING | Turn E", output)

    def test_seat_snapshot_merges_header_and_seats(self) -> None:
        output = format_seat_snapshot(
            {
                "public": {
                    "table_id": "table-1",
                    "phase": "PLAYING",
                    "event_seq": 1,
                    "current_turn": "E",
                    "seats": {"E": {"display_name": "East"}, "S": {"display_name": "South"}},
                    "hand_counts": {"E": 3, "S": 4, "W": 0, "N": 0},
                    "finish_order": [],
                },
                "seat": "E",
                "legal_action": "lead",
                "hand": ["D1-H-4", "D1-C-3", "D2-S-3"],
            }
        )

        self.assertIn("PLAYING | Seat E | Turn E", output)
        self.assertIn("Players: E East 3 | S South 4 | W(F) - 0 | N - 0", output)
        self.assertNotIn("Your seat:", output)

    def test_friend_mark_identifies_partner_for_viewer_seat(self) -> None:
        self.assertEqual(format_friend_mark("W", "E"), "(F)")
        self.assertEqual(format_friend_mark("S", "E"), "")

    def test_command_response_omits_action_prompted_events(self) -> None:
        output = format_command_response(
            {
                "events": [
                    {
                        "seq": 1,
                        "type": "CardsPlayed",
                        "payload": {"seat": "E", "hand_type": "single", "card_ids": ["D1-S-3"]},
                    },
                    {
                        "seq": 2,
                        "type": "ActionPrompted",
                        "payload": {"seat": "S", "kind": "play_or_pass"},
                    },
                ]
            }
        )

        self.assertEqual(output, "1: E played single [♠️ 3]\n")
        self.assertNotIn("ActionPrompted", output)


if __name__ == "__main__":
    unittest.main()
