from __future__ import annotations

import unittest
from dataclasses import replace

from server.domain.commands import JoinTable, Ready, StartMatch
from server.domain.controllers import ControllerCapability, ControllerKind, ControllerRef, PlayerKind, PlayerRef
from server.domain.reducer import reduce_command
from server.domain.seats import SEATS, Seat
from server.domain.state import DealState, MatchPhase, MatchState, TributeObligation, TributeState, TrickState
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
        self.assertEqual(snapshot.level_by_team, state.scores.level_by_team)
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

    def test_tribute_snapshot_exposes_only_current_seat_legal_choices(self) -> None:
        state = make_state()
        assert state.deal is not None
        obligation = TributeObligation(giver=Seat.EAST, receiver=Seat.SOUTH)
        deal = replace(
            state.deal,
            hands={
                Seat.EAST: ("D1-S-A", "D1-H-2", "D1-S-3"),
                Seat.SOUTH: ("D1-C-3",),
                Seat.WEST: ("D1-C-4",),
                Seat.NORTH: ("D1-C-5",),
            },
            turn=Seat.EAST,
            tribute=TributeState(obligations=(obligation,), leader_after=Seat.EAST),
        )
        state = replace(state, phase=MatchPhase.TRIBUTE, current_level=state.current_level, deal=deal)

        snapshot = seat_snapshot(state, Seat.EAST, "c-E")

        self.assertEqual(snapshot.legal_action, "tribute")
        self.assertEqual(snapshot.eligible_card_ids, ("D1-S-A",))
        self.assertEqual(snapshot.tribute_to, Seat.SOUTH)
        self.assertIsNone(snapshot.tribute_from)

    def test_tribute_snapshot_exposes_one_highest_eligible_card_when_tied(self) -> None:
        state = make_state()
        assert state.deal is not None
        obligation = TributeObligation(giver=Seat.EAST, receiver=Seat.SOUTH)
        deal = replace(
            state.deal,
            hands={
                Seat.EAST: ("D1-S-A", "D2-S-A", "D1-S-3"),
                Seat.SOUTH: ("D1-C-3",),
                Seat.WEST: ("D1-C-4",),
                Seat.NORTH: ("D1-C-5",),
            },
            turn=Seat.EAST,
            tribute=TributeState(obligations=(obligation,), leader_after=Seat.EAST),
        )
        state = replace(state, phase=MatchPhase.TRIBUTE, deal=deal)

        snapshot = seat_snapshot(state, Seat.EAST, "c-E")

        self.assertEqual(snapshot.legal_action, "tribute")
        self.assertEqual(snapshot.eligible_card_ids, ("D1-S-A",))

    def test_return_snapshot_exposes_low_card_constraint_for_partner_return(self) -> None:
        state = make_state()
        assert state.deal is not None
        obligation = TributeObligation(giver=Seat.EAST, receiver=Seat.WEST, tribute_card_id="D1-S-A")
        deal = DealState(
            hands={
                Seat.EAST: ("D1-S-3",),
                Seat.SOUTH: ("D1-C-3",),
                Seat.WEST: ("D1-S-A", "D1-S-10", "D1-S-J"),
                Seat.NORTH: ("D1-C-5",),
            },
            active_seats=frozenset(SEATS),
            finish_order=(),
            leader=Seat.EAST,
            turn=Seat.WEST,
            current_trick=TrickState(lead_seat=Seat.EAST),
            tribute=TributeState(obligations=(obligation,), leader_after=Seat.EAST),
        )
        state = replace(state, phase=MatchPhase.TRIBUTE, deal=deal)

        snapshot = seat_snapshot(state, Seat.WEST, "c-W")

        self.assertEqual(snapshot.legal_action, "return_tribute")
        self.assertEqual(snapshot.eligible_card_ids, ("D1-S-10",))
        self.assertEqual(snapshot.tribute_from, Seat.EAST)
        self.assertTrue(snapshot.return_rank_at_most_ten)

    def test_return_snapshot_exposes_low_cards_for_non_partner_return(self) -> None:
        state = make_state()
        assert state.deal is not None
        obligation = TributeObligation(giver=Seat.EAST, receiver=Seat.SOUTH, tribute_card_id="D1-S-A")
        deal = DealState(
            hands={
                Seat.EAST: ("D1-S-3",),
                Seat.SOUTH: ("D1-S-A", "D1-S-10", "D1-S-9"),
                Seat.WEST: ("D1-C-4",),
                Seat.NORTH: ("D1-C-5",),
            },
            active_seats=frozenset(SEATS),
            finish_order=(),
            leader=Seat.EAST,
            turn=Seat.SOUTH,
            current_trick=TrickState(lead_seat=Seat.EAST),
            tribute=TributeState(obligations=(obligation,), leader_after=Seat.EAST),
        )
        state = replace(state, phase=MatchPhase.TRIBUTE, deal=deal)

        snapshot = seat_snapshot(state, Seat.SOUTH, "c-S")

        self.assertEqual(snapshot.legal_action, "return_tribute")
        self.assertEqual(snapshot.eligible_card_ids, ("D1-S-10", "D1-S-9"))
        self.assertFalse(snapshot.return_rank_at_most_ten)


if __name__ == "__main__":
    unittest.main()
