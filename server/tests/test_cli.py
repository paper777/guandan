from __future__ import annotations

import unittest

from guandan.cli import format_card_id, format_public_snapshot, run_cli


class FakeClient:
    def __init__(self) -> None:
        self.table_id = "table-1"
        self.seats: dict[str, dict[str, str]] = {}
        self.phase = "WAITING_FOR_PLAYERS"
        self.current_turn: str | None = None
        self.event_seq = 0
        self.calls: list[tuple] = []
        self.hands = {
            "E": ["D1-S-3", "D2-S-3"],
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
    def test_default_play_creates_human_and_three_bots(self) -> None:
        client = FakeClient()

        result = run_cli([], input_fn=lambda prompt: "quit", client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Connected to table table-1 as seat E.", result.output)
        self.assertIn("Hand: 1: ♠️ 3  2: ♠️ 3", result.output)
        self.assertIn(("join_human", "table-1", "E", "human-E", "human-controller-E", "human-E"), client.calls)
        self.assertIn(("join_local_bot", "table-1", "S", "bot-S", "bot-controller-S", "Bot S"), client.calls)
        self.assertIn(("join_local_bot", "table-1", "W", "bot-W", "bot-controller-W", "Bot W"), client.calls)
        self.assertIn(("join_local_bot", "table-1", "N", "bot-N", "bot-controller-N", "Bot N"), client.calls)
        self.assertIn(("start", "table-1", "cli-demo"), client.calls)

    def test_human_play_then_drives_bot_passes(self) -> None:
        client = FakeClient()
        commands = iter(["play 1", "quit"])

        result = run_cli([], input_fn=lambda prompt: next(commands), client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertIn(("play_cards", "table-1", "E", "human-controller-E", ("D1-S-3",), None), client.calls)
        self.assertIn(("pass_turn", "table-1", "S", "bot-controller-S"), client.calls)
        self.assertIn(("pass_turn", "table-1", "W", "bot-controller-W"), client.calls)
        self.assertIn(("pass_turn", "table-1", "N", "bot-controller-N"), client.calls)

    def test_card_formatter_hides_first_deck_and_uses_suit_emoji(self) -> None:
        self.assertEqual(format_card_id("D1-S-3"), "♠️ 3")
        self.assertEqual(format_card_id("D2-H-10"), "♥️ 10")
        self.assertEqual(format_card_id("D1-SJ"), "🃏SJ")

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


if __name__ == "__main__":
    unittest.main()
