from __future__ import annotations

import unittest

from guandan.domain.commands import JoinTable, Pass, PlayCards, Ready, StartMatch
from guandan.domain.controllers import ControllerCapability, ControllerKind, ControllerRef, PlayerKind, PlayerRef
from guandan.domain.events import Event
from guandan.domain.seats import SEATS, Seat
from guandan.services.replay import rebuild_state_from_events
from guandan.services.table_actor import TableActor


class ReplayTests(unittest.TestCase):
    def test_rebuilds_started_match_from_events(self) -> None:
        actor, events = _started_actor_and_events()

        rebuilt = rebuild_state_from_events(actor.table_id, tuple(events))

        self.assertEqual(rebuilt, actor.state)

    def test_rebuilds_play_pass_trick_from_events(self) -> None:
        actor, events = _started_actor_and_events()
        assert actor.state.deal is not None
        east_card = actor.state.deal.hand_for(Seat.EAST)[0]
        events.extend(actor.dispatch(PlayCards(controller(Seat.EAST).id, Seat.EAST, (east_card,))).events)
        for seat in (Seat.NORTH, Seat.WEST, Seat.SOUTH):
            events.extend(actor.dispatch(Pass(controller(seat).id, seat)).events)

        rebuilt = rebuild_state_from_events(actor.table_id, tuple(events))

        self.assertEqual(rebuilt, actor.state)


def _started_actor_and_events() -> tuple[TableActor, list[Event]]:
    actor = TableActor("table-1")
    events: list[Event] = []
    for seat in SEATS:
        events.extend(actor.dispatch(JoinTable(player(seat), controller(seat), seat)).events)
    for seat in SEATS:
        events.extend(actor.dispatch(Ready(controller(seat).id, seat)).events)
    events.extend(actor.dispatch(StartMatch(seed="fixed-seed")).events)
    return actor, events


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
