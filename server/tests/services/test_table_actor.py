from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from guandan.domain.commands import JoinTable, Ready, StartMatch
from guandan.domain.controllers import ControllerCapability, ControllerKind, ControllerRef, PlayerKind, PlayerRef
from guandan.domain.seats import SEATS, Seat
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

    def test_dispatch_persists_multi_event_start_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteEventStore(Path(tmp) / "events.db")
            actor = TableActor("table-1", event_store=store)
            for seat in SEATS:
                actor.dispatch(JoinTable(player(seat), controller(seat), seat))
            for seat in SEATS:
                actor.dispatch(Ready(controller(seat).id, seat))

            result = actor.dispatch(StartMatch(seed="fixed-seed"), controller_id="c-E", request_id="start-1")
            loaded = store.load_events(actor.match_id)
            store.close()

        self.assertIsNone(result.rejection)
        self.assertEqual([event.type for event in result.events], ["MatchStarted", "DealStarted", "CardsDealt"])
        self.assertEqual([event.seq for event in result.events], [9, 10, 11])
        self.assertEqual([event.seq for event in loaded[-3:]], [9, 10, 11])
        self.assertEqual(len(loaded[-1].payload["hands"][Seat.EAST.value]), 27)

    def test_initializes_state_from_existing_event_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteEventStore(Path(tmp) / "events.db")
            actor = TableActor("table-1", match_id="match-1", event_store=store)
            for seat in SEATS:
                actor.dispatch(JoinTable(player(seat), controller(seat), seat))
            for seat in SEATS:
                actor.dispatch(Ready(controller(seat).id, seat))
            actor.dispatch(StartMatch(seed="fixed-seed"))

            restored = TableActor("table-1", match_id="match-1", event_store=store)
            store.close()

        self.assertEqual(restored.state, actor.state)

    def test_async_dispatch_serializes_commands(self) -> None:
        async def run() -> TableActor:
            actor = TableActor("table-1")
            await asyncio.gather(
                *[actor.dispatch_async(JoinTable(player(seat), controller(seat), seat)) for seat in SEATS]
            )
            return actor

        actor = asyncio.run(run())

        self.assertEqual(actor.state.event_seq, 4)
        self.assertEqual(set(actor.state.seats), set(SEATS))

def player(seat: Seat) -> PlayerRef:
    return PlayerRef(f"p-{seat.value}", seat.value, PlayerKind.HUMAN)


def controller(seat: Seat) -> ControllerRef:
    return ControllerRef(
        f"c-{seat.value}",
        ControllerKind.HUMAN_WS,
        seat,
        f"p-{seat.value}",
        frozenset(
            {
                ControllerCapability.PLAY,
                ControllerCapability.OBSERVE_PUBLIC,
                ControllerCapability.OBSERVE_PRIVATE,
            }
        ),
    )


if __name__ == "__main__":
    unittest.main()
