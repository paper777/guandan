from __future__ import annotations

import unittest

from server.controllers.external_agent import ExternalAgentClient
from server.domain.seats import Seat
from server.domain.state import MatchPhase
from server.services.snapshots import PublicTableSnapshot, SeatSnapshot


class ExternalAgentClientTests(unittest.TestCase):
    def test_build_payload_filters_to_seat_private_hand(self) -> None:
        public = PublicTableSnapshot(
            table_id="table-1",
            deal_id=1,
            phase=MatchPhase.PLAYING,
            seats={},
            hand_counts={Seat.EAST: 2, Seat.SOUTH: 27},
            current_turn=Seat.EAST,
            finish_order=(),
            event_seq=12,
        )
        snapshot = SeatSnapshot(public=public, seat=Seat.EAST, hand=("D1-S-3", "D2-S-3"), legal_action="act")

        payload = ExternalAgentClient("http://agent").build_payload("r-1", snapshot, {"kind": "lead"})

        self.assertEqual(payload["snapshot"]["hand"], ["D1-S-3", "D2-S-3"])
        self.assertNotIn("hands", payload["snapshot"]["public_state"])
        self.assertEqual(payload["prompt"]["kind"], "lead")


if __name__ == "__main__":
    unittest.main()
