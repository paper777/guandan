from __future__ import annotations

import unittest

from guandan.controllers.simple_policy import SimpleBotPolicy
from guandan.domain.commands import PlayCards
from guandan.domain.seats import Seat
from guandan.domain.state import MatchPhase
from guandan.services.snapshots import PublicTableSnapshot, SeatSnapshot


class SimpleBotPolicyTests(unittest.TestCase):
    def test_leads_first_card_when_turn(self) -> None:
        public = PublicTableSnapshot(
            table_id="table-1",
            phase=MatchPhase.PLAYING,
            seats={},
            hand_counts={Seat.EAST: 1},
            current_turn=Seat.EAST,
            finish_order=(),
            event_seq=1,
        )
        snapshot = SeatSnapshot(public=public, seat=Seat.EAST, hand=("D1-S-3",), legal_action="act")

        action = SimpleBotPolicy("c-E", Seat.EAST).choose_action(snapshot)

        self.assertIsInstance(action, PlayCards)
        self.assertEqual(action.card_ids, ("D1-S-3",))


if __name__ == "__main__":
    unittest.main()
