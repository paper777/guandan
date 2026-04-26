from __future__ import annotations

from dataclasses import dataclass

from guandan.domain.commands import Pass, PlayCards
from guandan.domain.seats import Seat
from guandan.services.snapshots import SeatSnapshot


@dataclass(frozen=True, slots=True)
class SimpleBotPolicy:
    controller_id: str
    seat: Seat

    def choose_action(self, snapshot: SeatSnapshot) -> PlayCards | Pass:
        if snapshot.public.current_turn != self.seat:
            return Pass(controller_id=self.controller_id, seat=self.seat)
        if not snapshot.hand:
            return Pass(controller_id=self.controller_id, seat=self.seat)
        # Baseline policy: lead the first card; otherwise pass. Stronger policy can use legal-action hints later.
        return PlayCards(controller_id=self.controller_id, seat=self.seat, card_ids=(snapshot.hand[0],))
