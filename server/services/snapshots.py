from __future__ import annotations

from dataclasses import dataclass

from server.domain.cards import Rank
from server.domain.controllers import ControllerCapability
from server.domain.seats import Seat
from server.domain.state import MatchPhase, MatchState


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
    current_level: Rank = Rank.TWO
    action_deadline_epoch_ms: int | None = None
    action_timeout_seconds: int = 45
    acting_seat: Seat | None = None


@dataclass(frozen=True, slots=True)
class SeatSnapshot:
    public: PublicTableSnapshot
    seat: Seat
    hand: tuple[str, ...]
    legal_action: str | None


def public_snapshot(
    state: MatchState,
    *,
    action_deadline_epoch_ms: int | None = None,
    action_timeout_seconds: int = 45,
    acting_seat: Seat | None = None,
) -> PublicTableSnapshot:
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
        current_level=state.current_level,
        finish_order=finish_order,
        event_seq=state.event_seq,
        action_deadline_epoch_ms=action_deadline_epoch_ms,
        action_timeout_seconds=action_timeout_seconds,
        acting_seat=acting_seat,
    )


def seat_snapshot(
    state: MatchState,
    seat: Seat,
    controller_id: str,
    *,
    action_deadline_epoch_ms: int | None = None,
    action_timeout_seconds: int = 45,
    acting_seat: Seat | None = None,
) -> SeatSnapshot:
    controller = state.controllers.get(seat)
    if controller is None or controller.id != controller_id:
        raise PermissionError("controller is not attached to that seat")
    if not controller.can(ControllerCapability.OBSERVE_PRIVATE):
        raise PermissionError("controller cannot observe private seat state")
    hand = state.deal.hand_for(seat) if state.deal is not None else ()
    legal_action = None
    if state.deal is not None and controller.can(ControllerCapability.PLAY):
        if state.phase == MatchPhase.PLAYING and state.deal.turn == seat:
            legal_action = "lead" if state.deal.current_trick.last_play is None else "play_or_pass"
        elif state.phase == MatchPhase.TRIBUTE and state.deal.tribute is not None:
            for obligation in state.deal.tribute.obligations:
                if obligation.giver == seat and obligation.tribute_card_id is None:
                    legal_action = "tribute"
                    break
            if legal_action is None:
                for obligation in state.deal.tribute.obligations:
                    if (
                        obligation.receiver == seat
                        and obligation.tribute_card_id is not None
                        and obligation.return_card_id is None
                    ):
                        legal_action = "return_tribute"
                        break
    return SeatSnapshot(
        public=public_snapshot(
            state,
            action_deadline_epoch_ms=action_deadline_epoch_ms,
            action_timeout_seconds=action_timeout_seconds,
            acting_seat=acting_seat,
        ),
        seat=seat,
        hand=hand,
        legal_action=legal_action,
    )
