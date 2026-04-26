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


if __name__ == "__main__":
    unittest.main()
