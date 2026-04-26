from __future__ import annotations

from dataclasses import dataclass

from guandan.domain.controllers import ControllerCapability
from guandan.domain.seats import Seat
from guandan.domain.state import MatchPhase, MatchState


@dataclass(frozen=True, slots=True)
class PublicPlayer:
    player_id: str
    display_name: str
    kind: str
    controlled: bool


@dataclass(frozen=True, slots=True)
class PublicTableSnapshot:
    table_id: str
    phase: MatchPhase
    seats: dict[Seat, PublicPlayer]
    hand_counts: dict[Seat, int]
    current_turn: Seat | None
    finish_order: tuple[Seat, ...]
    event_seq: int


@dataclass(frozen=True, slots=True)
class SeatSnapshot:
    public: PublicTableSnapshot
    seat: Seat
    hand: tuple[str, ...]
    legal_action: str | None


def public_snapshot(state: MatchState) -> PublicTableSnapshot:
    hand_counts: dict[Seat, int] = {}
    current_turn: Seat | None = None
    finish_order: tuple[Seat, ...] = ()
    if state.deal is not None:
        hand_counts = {seat: len(hand) for seat, hand in state.deal.hands.items()}
        current_turn = state.deal.turn
        finish_order = state.deal.finish_order
    seats = {
        seat: PublicPlayer(
            player_id=player.id,
            display_name=player.display_name,
            kind=player.kind.value,
            controlled=seat in state.controllers,
        )
        for seat, player in state.seats.items()
    }
    return PublicTableSnapshot(
        table_id=state.table_id,
        phase=state.phase,
        seats=seats,
        hand_counts=hand_counts,
        current_turn=current_turn,
        finish_order=finish_order,
        event_seq=state.event_seq,
    )


def seat_snapshot(state: MatchState, seat: Seat, controller_id: str) -> SeatSnapshot:
    controller = state.controllers.get(seat)
    if controller is None or controller.id != controller_id:
        raise PermissionError("controller is not attached to that seat")
    if not controller.can(ControllerCapability.OBSERVE_PRIVATE):
        raise PermissionError("controller cannot observe private seat state")
    hand = state.deal.hand_for(seat) if state.deal is not None else ()
    legal_action = None
    if state.deal is not None and state.deal.turn == seat and controller.can(ControllerCapability.PLAY):
        legal_action = "lead" if state.deal.current_trick.last_play is None else "play_or_pass"
    return SeatSnapshot(public=public_snapshot(state), seat=seat, hand=hand, legal_action=legal_action)
