from __future__ import annotations

from dataclasses import dataclass, field

from server.domain.cards import CARD_BY_ID, Rank, is_red_heart_level_card
from server.domain.comparator import RankContext
from server.domain.controllers import ControllerCapability
from server.domain.hand_types import PlayedHand
from server.domain.seats import Seat, Team
from server.domain.state import MatchPhase, MatchState, TributeObligation
from server.services.table_config import DEFAULT_ACTION_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class PublicPlayer:
    player_id: str
    display_name: str
    kind: str
    controlled: bool


@dataclass(frozen=True, slots=True)
class PublicTableSnapshot:
    table_id: str
    deal_id: int
    phase: MatchPhase
    seats: dict[Seat, PublicPlayer]
    hand_counts: dict[Seat, int]
    current_turn: Seat | None
    finish_order: tuple[Seat, ...]
    event_seq: int
    current_level: Rank = Rank.TWO
    level_by_team: dict[Team, Rank] = field(
        default_factory=lambda: {Team.EAST_WEST: Rank.TWO, Team.SOUTH_NORTH: Rank.TWO}
    )
    action_deadline_epoch_ms: int | None = None
    action_timeout_seconds: int = DEFAULT_ACTION_TIMEOUT_SECONDS
    acting_seat: Seat | None = None
    current_trick: dict[str, object] | None = None
    played_card_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SeatSnapshot:
    public: PublicTableSnapshot
    seat: Seat
    hand: tuple[str, ...]
    legal_action: str | None
    eligible_card_ids: tuple[str, ...] = ()
    tribute_from: Seat | None = None
    tribute_to: Seat | None = None
    return_rank_at_most_ten: bool = False


def public_snapshot(
    state: MatchState,
    *,
    deal_id: int = 0,
    action_deadline_epoch_ms: int | None = None,
    action_timeout_seconds: int = DEFAULT_ACTION_TIMEOUT_SECONDS,
    acting_seat: Seat | None = None,
) -> PublicTableSnapshot:
    hand_counts: dict[Seat, int] = {}
    current_turn: Seat | None = None
    finish_order: tuple[Seat, ...] = ()
    if state.deal is not None:
        hand_counts = {seat: len(hand) for seat, hand in state.deal.hands.items()}
        current_turn = state.deal.turn
        finish_order = state.deal.finish_order
        current_trick = _public_trick(
            state.deal.current_trick.last_play,
            state.deal.current_trick.last_play_seat,
            state.deal.current_trick.pass_count,
        )
        played_card_counts = _played_card_counts(state.deal.played_card_ids)
    else:
        current_trick = None
        played_card_counts = {}
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
        deal_id=deal_id,
        phase=state.phase,
        seats=seats,
        hand_counts=hand_counts,
        current_turn=current_turn,
        current_level=state.current_level,
        level_by_team=dict(state.scores.level_by_team),
        finish_order=finish_order,
        event_seq=state.event_seq,
        action_deadline_epoch_ms=action_deadline_epoch_ms,
        action_timeout_seconds=action_timeout_seconds,
        acting_seat=acting_seat,
        current_trick=current_trick,
        played_card_counts=played_card_counts,
    )


def seat_snapshot(
    state: MatchState,
    seat: Seat,
    controller_id: str,
    *,
    deal_id: int = 0,
    action_deadline_epoch_ms: int | None = None,
    action_timeout_seconds: int = DEFAULT_ACTION_TIMEOUT_SECONDS,
    acting_seat: Seat | None = None,
) -> SeatSnapshot:
    controller = state.controllers.get(seat)
    if controller is None or controller.id != controller_id:
        raise PermissionError("controller is not attached to that seat")
    if not controller.can(ControllerCapability.OBSERVE_PRIVATE):
        raise PermissionError("controller cannot observe private seat state")
    hand = state.deal.hand_for(seat) if state.deal is not None else ()
    legal_action = None
    eligible_card_ids: tuple[str, ...] = ()
    tribute_from: Seat | None = None
    tribute_to: Seat | None = None
    return_rank_at_most_ten = False
    if state.deal is not None and controller.can(ControllerCapability.PLAY):
        if state.phase == MatchPhase.PLAYING and state.deal.turn == seat:
            legal_action = "lead" if state.deal.current_trick.last_play is None else "play_or_pass"
        elif state.phase == MatchPhase.TRIBUTE and state.deal.tribute is not None:
            for obligation in state.deal.tribute.obligations:
                if obligation.giver == seat and obligation.tribute_card_id is None:
                    legal_action = "tribute"
                    eligible_card_ids = _highest_eligible_tribute_card(hand, state.current_level)
                    tribute_to = obligation.receiver
                    break
            if legal_action is None:
                for obligation in state.deal.tribute.obligations:
                    if (
                        obligation.receiver == seat
                        and obligation.tribute_card_id is not None
                        and obligation.return_card_id is None
                    ):
                        legal_action = "return_tribute"
                        eligible_card_ids = _eligible_return_cards(hand, obligation)
                        tribute_from = obligation.giver
                        return_rank_at_most_ten = _return_rank_at_most_ten(obligation)
                        break
    return SeatSnapshot(
        public=public_snapshot(
            state,
            deal_id=deal_id,
            action_deadline_epoch_ms=action_deadline_epoch_ms,
            action_timeout_seconds=action_timeout_seconds,
            acting_seat=acting_seat,
        ),
        seat=seat,
        hand=hand,
        legal_action=legal_action,
        eligible_card_ids=eligible_card_ids,
        tribute_from=tribute_from,
        tribute_to=tribute_to,
        return_rank_at_most_ten=return_rank_at_most_ten,
    )


def _public_trick(last_play: PlayedHand | None, last_play_seat: Seat | None, pass_count: int) -> dict[str, object] | None:
    if last_play is None or last_play_seat is None:
        return {"pass_count": pass_count} if pass_count else None
    return {
        "last_play_seat": last_play_seat.value,
        "card_ids": list(last_play.card_ids),
        "hand_type": last_play.type.value,
        "primary_rank": last_play.primary_rank.value,
        "length": last_play.length,
        "pass_count": pass_count,
    }


def _played_card_counts(card_ids: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for card_id in card_ids:
        face = _face(card_id)
        counts[face] = counts.get(face, 0) + 1
    return counts


def _face(card_id: str) -> str:
    card = CARD_BY_ID[card_id]
    if card.suit is None:
        return card.rank.value
    return f"{card.suit.value}-{card.rank.value}"


def _highest_eligible_tribute_card(card_ids: tuple[str, ...], level: Rank) -> tuple[str, ...]:
    ctx = RankContext(level)
    eligible = tuple(
        CARD_BY_ID[card_id] for card_id in card_ids if not is_red_heart_level_card(CARD_BY_ID[card_id], level)
    )
    if not eligible:
        return ()
    max_value = max(ctx.rank_value(card.rank) for card in eligible)
    for card in eligible:
        if ctx.rank_value(card.rank) == max_value:
            return (card.id,)
    return ()


def _eligible_return_cards(card_ids: tuple[str, ...], obligation: TributeObligation) -> tuple[str, ...]:
    return tuple(card_id for card_id in card_ids if _rank_at_most_ten(card_id))


def _return_rank_at_most_ten(obligation: TributeObligation) -> bool:
    return _partner_for_seat(obligation.receiver) == obligation.giver


def _partner_for_seat(seat: Seat) -> Seat:
    return {
        Seat.EAST: Seat.WEST,
        Seat.WEST: Seat.EAST,
        Seat.SOUTH: Seat.NORTH,
        Seat.NORTH: Seat.SOUTH,
    }[seat]


def _rank_at_most_ten(card_id: str) -> bool:
    rank = CARD_BY_ID[card_id].rank
    return rank.value in {"2", "3", "4", "5", "6", "7", "8", "9", "10"}
