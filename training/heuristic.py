from __future__ import annotations

from dataclasses import dataclass

from server.domain.cards import CARD_BY_ID
from server.domain.comparator import RankContext
from server.domain.hand_types import HandType
from server.domain.legal_actions import ActionCandidate
from server.domain.seats import SEATS, Seat, team_for_seat
from server.services.snapshots import SeatSnapshot


@dataclass(frozen=True, slots=True)
class HeuristicPolicy:
    """Deterministic baseline policy for bootstrapping and evaluation."""

    name: str = "heuristic"

    def choose_action(self, snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...]) -> ActionCandidate:
        if not actions:
            raise ValueError("heuristic policy requires at least one legal action")

        non_play = [action for action in actions if action.kind != "play_cards"]
        plays = [action for action in actions if action.kind == "play_cards"]
        if not plays:
            return _choose_non_play(snapshot, tuple(non_play))

        finishers = [action for action in plays if len(action.card_ids) == len(snapshot.hand)]
        if finishers:
            return _best_finisher(snapshot, tuple(finishers))

        if snapshot.legal_action == "lead":
            return _choose_lead(snapshot, tuple(plays))

        if _last_play_was_partner(snapshot):
            return _pass_action(actions) or _cheapest_play(snapshot, tuple(plays))

        non_bombs = tuple(action for action in plays if not _is_bomb_like(action))
        if non_bombs:
            return _cheapest_play(snapshot, non_bombs)

        if _opponent_is_dangerous(snapshot):
            return _cheapest_play(snapshot, tuple(plays))

        return _pass_action(actions) or _cheapest_play(snapshot, tuple(plays))


def _choose_non_play(snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...]) -> ActionCandidate:
    if not actions:
        raise ValueError("no legal non-play actions")
    if actions[0].kind == "submit_tribute":
        return max(actions, key=lambda action: (_single_card_value(snapshot, action), action.card_ids))
    if actions[0].kind == "return_tribute":
        return min(actions, key=lambda action: (_single_card_value(snapshot, action), action.card_ids))
    return actions[0]


def _choose_lead(snapshot: SeatSnapshot, plays: tuple[ActionCandidate, ...]) -> ActionCandidate:
    ordinary = tuple(action for action in plays if not _is_bomb_like(action))
    if ordinary:
        return max(
            ordinary,
            key=lambda action: (
                action.length,
                -_action_rank_value(snapshot, action),
                _type_preference(action),
                tuple(reversed(action.card_ids)),
            ),
        )
    return _cheapest_play(snapshot, plays)


def _best_finisher(snapshot: SeatSnapshot, plays: tuple[ActionCandidate, ...]) -> ActionCandidate:
    non_bombs = tuple(action for action in plays if not _is_bomb_like(action))
    pool = non_bombs or plays
    return min(pool, key=lambda action: (_is_bomb_like(action), _action_rank_value(snapshot, action), action.card_ids))


def _cheapest_play(snapshot: SeatSnapshot, plays: tuple[ActionCandidate, ...]) -> ActionCandidate:
    return min(
        plays,
        key=lambda action: (
            _is_bomb_like(action),
            _action_rank_value(snapshot, action),
            action.length,
            action.card_ids,
        ),
    )


def _pass_action(actions: tuple[ActionCandidate, ...]) -> ActionCandidate | None:
    return next((action for action in actions if action.kind == "pass"), None)


def _last_play_was_partner(snapshot: SeatSnapshot) -> bool:
    trick = snapshot.public.current_trick or {}
    raw_seat = trick.get("last_play_seat")
    if not isinstance(raw_seat, str):
        return False
    try:
        last_seat = Seat(raw_seat)
    except ValueError:
        return False
    return last_seat != snapshot.seat and team_for_seat(last_seat) == team_for_seat(snapshot.seat)


def _opponent_is_dangerous(snapshot: SeatSnapshot) -> bool:
    counts = snapshot.public.hand_counts
    own_team = team_for_seat(snapshot.seat)
    return any(
        team_for_seat(seat) != own_team and 0 < counts.get(seat, 0) <= 2
        for seat in SEATS
    )


def _is_bomb_like(action: ActionCandidate) -> bool:
    return action.hand_type in {HandType.BOMB, HandType.STRAIGHT_FLUSH, HandType.FOUR_JOKERS}


def _action_rank_value(snapshot: SeatSnapshot, action: ActionCandidate) -> int:
    if action.primary_rank is None:
        return -1
    return RankContext(snapshot.public.current_level).rank_value(action.primary_rank)


def _single_card_value(snapshot: SeatSnapshot, action: ActionCandidate) -> int:
    card = CARD_BY_ID[action.card_ids[0]]
    return RankContext(snapshot.public.current_level).rank_value(card.rank)


def _type_preference(action: ActionCandidate) -> int:
    order = {
        HandType.THREE_PAIR_RUN: 5,
        HandType.TRIPLE_RUN: 4,
        HandType.STRAIGHT: 3,
        HandType.FULL_HOUSE: 2,
        HandType.THREE_OF_A_KIND: 1,
        HandType.PAIR: 0,
        HandType.SINGLE: -1,
    }
    return order.get(action.hand_type, -2)
