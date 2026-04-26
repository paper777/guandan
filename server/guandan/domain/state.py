from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from guandan.domain.cards import Rank
from guandan.domain.controllers import ControllerRef, PlayerRef
from guandan.domain.hand_types import PlayedHand
from guandan.domain.seats import SEATS, Seat, Team


class MatchPhase(StrEnum):
    WAITING_FOR_PLAYERS = "WAITING_FOR_PLAYERS"
    READY_CHECK = "READY_CHECK"
    DEALING = "DEALING"
    TRIBUTE = "TRIBUTE"
    PLAYING = "PLAYING"
    DEAL_COMPLETE = "DEAL_COMPLETE"
    MATCH_COMPLETE = "MATCH_COMPLETE"
    ABORTED = "ABORTED"


@dataclass(frozen=True, slots=True)
class TrickState:
    lead_seat: Seat
    last_play: PlayedHand | None = None
    last_play_seat: Seat | None = None
    pass_count: int = 0


@dataclass(frozen=True, slots=True)
class DealState:
    hands: dict[Seat, tuple[str, ...]]
    active_seats: frozenset[Seat]
    finish_order: tuple[Seat, ...]
    leader: Seat
    turn: Seat
    current_trick: TrickState
    tribute: TributeState | None = None
    report_10_done: frozenset[Seat] = field(default_factory=frozenset)

    def hand_for(self, seat: Seat) -> tuple[str, ...]:
        return self.hands.get(seat, ())


@dataclass(frozen=True, slots=True)
class DealResult:
    finish_order: tuple[Seat, ...]
    winning_team: Team
    advance_count: int
    previous_level: Rank
    next_level: Rank
    match_complete: bool


@dataclass(frozen=True, slots=True)
class TributeObligation:
    giver: Seat
    receiver: Seat
    tribute_card_id: str | None = None
    return_card_id: str | None = None


@dataclass(frozen=True, slots=True)
class TributeState:
    obligations: tuple[TributeObligation, ...]
    leader_after: Seat
    resisted: bool = False

    @property
    def complete(self) -> bool:
        return all(item.tribute_card_id is not None and item.return_card_id is not None for item in self.obligations)


@dataclass(frozen=True, slots=True)
class ScoreState:
    level_by_team: dict[Team, Rank] = field(
        default_factory=lambda: {Team.EAST_WEST: Rank.TWO, Team.SOUTH_NORTH: Rank.TWO}
    )


@dataclass(frozen=True, slots=True)
class MatchState:
    table_id: str
    phase: MatchPhase = MatchPhase.WAITING_FOR_PLAYERS
    current_level: Rank = Rank.TWO
    seats: dict[Seat, PlayerRef] = field(default_factory=dict)
    controllers: dict[Seat, ControllerRef] = field(default_factory=dict)
    ready_seats: frozenset[Seat] = field(default_factory=frozenset)
    deal: DealState | None = None
    last_deal_result: DealResult | None = None
    scores: ScoreState = field(default_factory=ScoreState)
    event_seq: int = 0

    @property
    def is_full(self) -> bool:
        return all(seat in self.seats and seat in self.controllers for seat in SEATS)

    def bump_seq(self, count: int = 1) -> MatchState:
        return replace(self, event_seq=self.event_seq + count)
