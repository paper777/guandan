from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from guandan.domain.commands import JoinTable
from guandan.domain.controllers import ControllerCapability, ControllerKind, ControllerRef, PlayerKind, PlayerRef
from guandan.domain.seats import Seat
from guandan.persistence.sqlite_store import SQLiteEventStore
from guandan.services.table_actor import TableActor


class TableActorTests(unittest.TestCase):
    def test_dispatch_persists_events_and_replays_idempotent_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteEventStore(Path(tmp) / "events.db")
            actor = TableActor("table-1", event_store=store)
            player = PlayerRef("p-E", "E", PlayerKind.HUMAN)
            controller = ControllerRef(
                "c-E",
                ControllerKind.HUMAN_WS,
                Seat.EAST,
                "p-E",
                frozenset({ControllerCapability.PLAY, ControllerCapability.OBSERVE_PRIVATE}),
            )
            command = JoinTable(player, controller, Seat.EAST)

            first = actor.dispatch(command, controller_id="c-E", request_id="r-1")
            second = actor.dispatch(command, controller_id="c-E", request_id="r-1")
            store.close()

        self.assertIsNone(first.rejection)
        self.assertTrue(second.replayed)
        self.assertEqual([event.seq for event in first.events], [1])
        self.assertEqual(second.events, first.events)


if __name__ == "__main__":
    unittest.main()
