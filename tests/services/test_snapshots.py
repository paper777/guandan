from __future__ import annotations

import unittest

from server.domain.commands import JoinTable, Ready, StartMatch
from server.domain.controllers import ControllerCapability, ControllerKind, ControllerRef, PlayerKind, PlayerRef
from server.domain.reducer import reduce_command
from server.domain.seats import SEATS, Seat
from server.domain.state import MatchState
from server.services.snapshots import public_snapshot, seat_snapshot


def make_state() -> MatchState:
    state = MatchState(table_id="table-1")
    capabilities = frozenset(
        {
            ControllerCapability.PLAY,
            ControllerCapability.OBSERVE_PUBLIC,
            ControllerCapability.OBSERVE_PRIVATE,
        }
    )
    for seat in SEATS:
        player = PlayerRef(id=f"p-{seat.value}", display_name=seat.value, kind=PlayerKind.BOT)
        controller = ControllerRef(
            id=f"c-{seat.value}",
            kind=ControllerKind.LOCAL_BOT,
            seat=seat,
            player_id=player.id,
            capabilities=capabilities,
        )
        result = reduce_command(state, JoinTable(player, controller, seat))
        assert result.rejection is None
        state = result.state
    for seat in SEATS:
        result = reduce_command(state, Ready(f"c-{seat.value}", seat))
        assert result.rejection is None
        state = result.state
    result = reduce_command(state, StartMatch(seed="fixed-seed"))
    assert result.rejection is None
    return result.state


class SnapshotTests(unittest.TestCase):
    def test_public_snapshot_exposes_counts_not_private_hands(self) -> None:
        state = make_state()

        snapshot = public_snapshot(state)

        self.assertEqual(snapshot.hand_counts[Seat.EAST], 27)
        self.assertFalse(hasattr(snapshot, "hands"))

    def test_seat_snapshot_exposes_only_attached_seat_hand(self) -> None:
        state = make_state()
        assert state.deal is not None

        east = seat_snapshot(state, Seat.EAST, "c-E")

        self.assertEqual(east.hand, state.deal.hand_for(Seat.EAST))
        self.assertNotEqual(east.hand, state.deal.hand_for(Seat.SOUTH))

    def test_seat_snapshot_rejects_wrong_controller(self) -> None:
        state = make_state()

        with self.assertRaises(PermissionError):
            seat_snapshot(state, Seat.EAST, "c-S")


if __name__ == "__main__":
    unittest.main()
