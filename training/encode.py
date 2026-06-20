from __future__ import annotations

from dataclasses import dataclass

from server.domain.cards import CARD_BY_ID, STANDARD_RANKS, Rank, Suit, is_red_heart_level_card
from server.domain.hand_types import HandType
from server.domain.legal_actions import ActionCandidate
from server.domain.seats import SEATS, Seat, Team
from server.domain.state import MatchPhase
from server.services.snapshots import SeatSnapshot


CARD_FACES: tuple[str, ...] = tuple(
    f"{suit.value}-{rank.value}" for suit in Suit for rank in STANDARD_RANKS
) + (Rank.SMALL_JOKER.value, Rank.BIG_JOKER.value)
ACTION_KINDS: tuple[str, ...] = ("pass", "play_cards", "submit_tribute", "return_tribute")
LEGAL_ACTIONS: tuple[str, ...] = ("lead", "play_or_pass", "tribute", "return_tribute")


@dataclass(frozen=True, slots=True)
class EncodedVector:
    names: tuple[str, ...]
    values: tuple[float, ...]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=True))


def encode_observation(snapshot: SeatSnapshot) -> EncodedVector:
    public = snapshot.public
    names: list[str] = []
    values: list[float] = []

    _extend_one_hot(names, values, "seat", [seat.value for seat in SEATS], snapshot.seat.value)
    _extend_one_hot(names, values, "phase", [phase.value for phase in MatchPhase], public.phase.value)
    _extend_one_hot(names, values, "current_level", [rank.value for rank in STANDARD_RANKS], public.current_level.value)
    _extend_one_hot(
        names,
        values,
        "current_turn",
        [seat.value for seat in SEATS],
        public.current_turn.value if public.current_turn is not None else None,
    )
    _extend_one_hot(
        names,
        values,
        "acting_seat",
        [seat.value for seat in SEATS],
        public.acting_seat.value if public.acting_seat is not None else None,
    )
    _extend_one_hot(names, values, "legal_action", LEGAL_ACTIONS, snapshot.legal_action)
    _extend_one_hot(
        names,
        values,
        "tribute_from",
        [seat.value for seat in SEATS],
        snapshot.tribute_from.value if snapshot.tribute_from is not None else None,
    )
    _extend_one_hot(
        names,
        values,
        "tribute_to",
        [seat.value for seat in SEATS],
        snapshot.tribute_to.value if snapshot.tribute_to is not None else None,
    )

    for team in Team:
        names.append(f"team_level/{team.value}")
        values.append(_standard_rank_index(public.level_by_team[team]) / (len(STANDARD_RANKS) - 1))

    hand_counts = public.hand_counts
    for seat in SEATS:
        names.append(f"hand_count/{seat.value}")
        values.append(hand_counts.get(seat, 0) / 27.0)

    finish_positions = {seat: index + 1 for index, seat in enumerate(public.finish_order)}
    for seat in SEATS:
        names.append(f"finish_position/{seat.value}")
        values.append(finish_positions.get(seat, 0) / 4.0)

    own_counts = _face_counts(snapshot.hand)
    for face in CARD_FACES:
        names.append(f"hand_face/{face}")
        values.append(own_counts.get(face, 0) / 2.0)

    trick = public.current_trick or {}
    _extend_one_hot(names, values, "trick_last_seat", [seat.value for seat in SEATS], _string_or_none(trick.get("last_play_seat")))
    _extend_one_hot(
        names,
        values,
        "trick_hand_type",
        [hand_type.value for hand_type in HandType],
        _string_or_none(trick.get("hand_type")),
    )
    _extend_one_hot(
        names,
        values,
        "trick_primary_rank",
        [rank.value for rank in Rank],
        _string_or_none(trick.get("primary_rank")),
    )
    names.append("trick_length")
    values.append(float(trick.get("length") or 0) / 8.0)
    names.append("return_rank_at_most_ten")
    values.append(1.0 if snapshot.return_rank_at_most_ten else 0.0)
    return EncodedVector(tuple(names), tuple(values))


def encode_action(action: ActionCandidate, snapshot: SeatSnapshot) -> EncodedVector:
    names: list[str] = []
    values: list[float] = []
    level = snapshot.public.current_level

    _extend_one_hot(names, values, "action_kind", ACTION_KINDS, action.kind)
    _extend_one_hot(
        names,
        values,
        "action_hand_type",
        [hand_type.value for hand_type in HandType],
        action.hand_type.value if action.hand_type is not None else None,
    )
    _extend_one_hot(
        names,
        values,
        "action_primary_rank",
        [rank.value for rank in Rank],
        action.primary_rank.value if action.primary_rank is not None else None,
    )

    counts = _face_counts(action.card_ids)
    for face in CARD_FACES:
        names.append(f"action_face/{face}")
        values.append(counts.get(face, 0) / 2.0)

    names.append("action_length")
    values.append(action.length / 8.0)
    names.append("action_declared")
    values.append(1.0 if action.declared_type is not None else 0.0)
    names.append("action_bomb_like")
    values.append(1.0 if action.hand_type in {HandType.BOMB, HandType.STRAIGHT_FLUSH, HandType.FOUR_JOKERS} else 0.0)
    names.append("action_uses_wild")
    values.append(1.0 if any(is_red_heart_level_card(CARD_BY_ID[card_id], level) for card_id in action.card_ids) else 0.0)
    names.append("remaining_after_action")
    values.append(max(len(snapshot.hand) - len(action.card_ids), 0) / 27.0)
    return EncodedVector(tuple(names), tuple(values))


def _extend_one_hot(
    names: list[str],
    values: list[float],
    prefix: str,
    choices: tuple[str, ...] | list[str],
    active: str | None,
) -> None:
    for choice in choices:
        names.append(f"{prefix}/{choice}")
        values.append(1.0 if active == choice else 0.0)


def _face_counts(card_ids: tuple[str, ...]) -> dict[str, int]:
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


def _standard_rank_index(rank: Rank) -> int:
    return STANDARD_RANKS.index(rank)


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None
