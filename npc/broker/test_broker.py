from __future__ import annotations

import unittest

from npc.broker.broker import NpcBroker
from npc.dummy_bot.policy import DummyBotPolicy


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


if __name__ == "__main__":
    unittest.main()
