from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from server.domain.cards import CARD_BY_ID, STANDARD_RANKS, Rank, Suit, is_red_heart_level_card
from server.domain.comparator import RankContext
from server.domain.hand_types import HandType
from server.domain.legal_actions import ActionCandidate
from server.domain.seats import SEATS, Seat, Team, team_for_seat
from server.domain.state import MatchPhase, MatchState
from server.services.snapshots import PublicTableSnapshot, SeatSnapshot


CARD_FACES: tuple[str, ...] = tuple(
    f"{suit.value}-{rank.value}" for suit in Suit for rank in STANDARD_RANKS
) + (Rank.SMALL_JOKER.value, Rank.BIG_JOKER.value)
ACTION_KINDS: tuple[str, ...] = ("pass", "play_cards", "submit_tribute", "return_tribute")
LEGAL_ACTIONS: tuple[str, ...] = ("lead", "play_or_pass", "tribute", "return_tribute")
ENCODING_SCHEMA_VERSION = "v2"
LEGACY_ENCODING_SCHEMA_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class EncodedVector:
    names: tuple[str, ...]
    values: tuple[float, ...]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=True))


def encode_observation(snapshot: SeatSnapshot, *, schema_version: str | None = None) -> EncodedVector:
    schema_version = _resolve_schema_version(schema_version)
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

    if schema_version != LEGACY_ENCODING_SCHEMA_VERSION:
        played_counts = public.played_card_counts
        for face in CARD_FACES:
            names.append(f"played_face/{face}")
            values.append(played_counts.get(face, 0) / 2.0)

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
    if schema_version != LEGACY_ENCODING_SCHEMA_VERSION:
        names.append("trick_pass_count")
        values.append(float(trick.get("pass_count") or 0) / 3.0)
    names.append("return_rank_at_most_ten")
    values.append(1.0 if snapshot.return_rank_at_most_ten else 0.0)
    return EncodedVector(tuple(names), tuple(values))


def encode_action(action: ActionCandidate, snapshot: SeatSnapshot, *, schema_version: str | None = None) -> EncodedVector:
    schema_version = _resolve_schema_version(schema_version)
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
    if schema_version != LEGACY_ENCODING_SCHEMA_VERSION:
        _append_action_context_features(names, values, action, snapshot)
    return EncodedVector(tuple(names), tuple(values))


def encoding_schema(schema_version: str | None = None) -> dict[str, object]:
    schema_version = _resolve_schema_version(schema_version)
    snapshot = _schema_snapshot()
    action = ActionCandidate(
        kind="play_cards",
        card_ids=("D1-H-2",),
        declared_type="single",
        hand_type=HandType.SINGLE,
        primary_rank=Rank.TWO,
        length=1,
    )
    observation_names = encode_observation(snapshot, schema_version=schema_version).names
    action_names = encode_action(action, snapshot, schema_version=schema_version).names
    payload = {
        "version": schema_version,
        "observation_names": list(observation_names),
        "action_names": list(action_names),
        "normalization": "guandan-fixed-v1",
    }
    payload["hash"] = _schema_hash(payload)
    return payload


def validate_encoding_schema(checkpoint: dict[str, object], *, schema_version: str | None = None) -> str:
    checkpoint_schema = checkpoint.get("encoding_schema")
    if isinstance(checkpoint_schema, dict):
        version = str(schema_version or checkpoint_schema.get("version") or ENCODING_SCHEMA_VERSION)
        expected = encoding_schema(version)
        checkpoint_hash = checkpoint_schema.get("hash")
        if checkpoint_hash != expected["hash"]:
            raise ValueError(
                f"encoding schema mismatch: checkpoint={checkpoint_hash!r} runtime={expected['hash']!r}"
            )
        return version
    observation_dim = int(checkpoint.get("observation_dim", -1))
    action_dim = int(checkpoint.get("action_dim", -1))
    for candidate_version in (LEGACY_ENCODING_SCHEMA_VERSION, ENCODING_SCHEMA_VERSION):
        candidate = encoding_schema(candidate_version)
        if (
            len(candidate["observation_names"]) == observation_dim
            and len(candidate["action_names"]) == action_dim
        ):
            return candidate_version
    return schema_version or "custom"


def encode_critic_observation(
    state: MatchState,
    actor: Seat,
    *,
    schema_version: str | None = None,
) -> EncodedVector:
    _resolve_schema_version(schema_version)
    names: list[str] = []
    values: list[float] = []
    deal = state.deal

    _extend_one_hot(names, values, "critic_actor", [seat.value for seat in SEATS], actor.value)
    _extend_one_hot(names, values, "critic_phase", [phase.value for phase in MatchPhase], state.phase.value)
    _extend_one_hot(names, values, "critic_current_level", [rank.value for rank in STANDARD_RANKS], state.current_level.value)
    _extend_one_hot(
        names,
        values,
        "critic_current_turn",
        [seat.value for seat in SEATS],
        deal.turn.value if deal is not None else None,
    )
    for team in Team:
        names.append(f"critic_team_level/{team.value}")
        values.append(_standard_rank_index(state.scores.level_by_team[team]) / (len(STANDARD_RANKS) - 1))
    finish_positions = {seat: index + 1 for index, seat in enumerate(deal.finish_order)} if deal is not None else {}
    active_seats = deal.active_seats if deal is not None else frozenset()
    for seat in SEATS:
        hand = deal.hand_for(seat) if deal is not None else ()
        names.append(f"critic_active/{seat.value}")
        values.append(1.0 if seat in active_seats else 0.0)
        names.append(f"critic_hand_count/{seat.value}")
        values.append(len(hand) / 27.0)
        names.append(f"critic_finish_position/{seat.value}")
        values.append(finish_positions.get(seat, 0) / 4.0)
        counts = _face_counts(hand)
        for face in CARD_FACES:
            names.append(f"critic_hand_face/{seat.value}/{face}")
            values.append(counts.get(face, 0) / 2.0)

    played_counts = _face_counts(deal.played_card_ids) if deal is not None else {}
    for face in CARD_FACES:
        names.append(f"critic_played_face/{face}")
        values.append(played_counts.get(face, 0) / 2.0)

    trick = deal.current_trick if deal is not None else None
    _extend_one_hot(
        names,
        values,
        "critic_trick_last_seat",
        [seat.value for seat in SEATS],
        trick.last_play_seat.value if trick is not None and trick.last_play_seat is not None else None,
    )
    _extend_one_hot(
        names,
        values,
        "critic_trick_hand_type",
        [hand_type.value for hand_type in HandType],
        trick.last_play.type.value if trick is not None and trick.last_play is not None else None,
    )
    _extend_one_hot(
        names,
        values,
        "critic_trick_primary_rank",
        [rank.value for rank in Rank],
        trick.last_play.primary_rank.value if trick is not None and trick.last_play is not None else None,
    )
    names.append("critic_trick_length")
    values.append((trick.last_play.length if trick is not None and trick.last_play is not None else 0) / 8.0)
    names.append("critic_trick_pass_count")
    values.append((trick.pass_count if trick is not None else 0) / 3.0)
    return EncodedVector(tuple(names), tuple(values))


def _append_action_context_features(
    names: list[str],
    values: list[float],
    action: ActionCandidate,
    snapshot: SeatSnapshot,
) -> None:
    relation = _last_play_relation(snapshot)
    opponent_danger = _opponent_is_dangerous(snapshot)
    names.append("action_beats_partner")
    values.append(1.0 if action.kind == "play_cards" and relation == "partner" else 0.0)
    names.append("action_beats_opponent")
    values.append(1.0 if action.kind == "play_cards" and relation == "opponent" else 0.0)
    names.append("action_finishes_hand")
    values.append(1.0 if action.kind == "play_cards" and len(action.card_ids) == len(snapshot.hand) else 0.0)
    names.append("action_opponent_danger")
    values.append(1.0 if opponent_danger else 0.0)
    names.append("action_rank_margin")
    values.append(_rank_margin(action, snapshot))
    names.append("action_breaks_bomb")
    values.append(1.0 if _breaks_bomb(action, snapshot) else 0.0)
    names.append("action_breaks_sequence")
    values.append(1.0 if _breaks_sequence(action, snapshot) else 0.0)
    names.append("action_breaks_pair_run")
    values.append(1.0 if _breaks_pair_run(action, snapshot) else 0.0)


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


def _resolve_schema_version(schema_version: str | None) -> str:
    version = schema_version or os.environ.get("GUANDAN_ENCODING_SCHEMA") or ENCODING_SCHEMA_VERSION
    if version not in {LEGACY_ENCODING_SCHEMA_VERSION, ENCODING_SCHEMA_VERSION}:
        raise ValueError(f"unsupported encoding schema version: {version}")
    return version


def _schema_hash(payload: dict[str, object]) -> str:
    normalized = json.dumps(
        {key: value for key, value in payload.items() if key != "hash"},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _schema_snapshot() -> SeatSnapshot:
    public = PublicTableSnapshot(
        table_id="schema",
        deal_id=1,
        phase=MatchPhase.PLAYING,
        seats={},
        hand_counts={seat: 1 for seat in SEATS},
        current_turn=Seat.EAST,
        finish_order=(),
        event_seq=1,
        current_level=Rank.TWO,
        level_by_team={Team.EAST_WEST: Rank.TWO, Team.SOUTH_NORTH: Rank.TWO},
        acting_seat=Seat.EAST,
        current_trick={
            "last_play_seat": Seat.SOUTH.value,
            "card_ids": ["D1-S-3"],
            "hand_type": HandType.SINGLE.value,
            "primary_rank": Rank.THREE.value,
            "length": 1,
            "pass_count": 1,
        },
        played_card_counts={face: 0 for face in CARD_FACES},
    )
    return SeatSnapshot(
        public=public,
        seat=Seat.EAST,
        hand=("D1-H-2",),
        legal_action="play_or_pass",
    )


def _last_play_relation(snapshot: SeatSnapshot) -> str | None:
    trick = snapshot.public.current_trick or {}
    raw_seat = trick.get("last_play_seat")
    if not isinstance(raw_seat, str):
        return None
    try:
        seat = Seat(raw_seat)
    except ValueError:
        return None
    if seat == snapshot.seat:
        return "self"
    return "partner" if team_for_seat(seat) == team_for_seat(snapshot.seat) else "opponent"


def _opponent_is_dangerous(snapshot: SeatSnapshot) -> bool:
    own_team = team_for_seat(snapshot.seat)
    return any(
        team_for_seat(seat) != own_team and 0 < snapshot.public.hand_counts.get(seat, 0) <= 2
        for seat in SEATS
    )


def _rank_margin(action: ActionCandidate, snapshot: SeatSnapshot) -> float:
    if action.kind != "play_cards" or action.primary_rank is None:
        return 0.0
    trick = snapshot.public.current_trick or {}
    raw_rank = _string_or_none(trick.get("primary_rank"))
    if raw_rank is None:
        return 0.0
    try:
        current_rank = Rank(raw_rank)
    except ValueError:
        return 0.0
    ctx = RankContext(snapshot.public.current_level)
    if action.hand_type in {HandType.BOMB, HandType.STRAIGHT_FLUSH, HandType.FOUR_JOKERS} and trick.get("hand_type") not in {
        HandType.BOMB.value,
        HandType.STRAIGHT_FLUSH.value,
        HandType.FOUR_JOKERS.value,
    }:
        return 1.0
    return _clamp((ctx.rank_value(action.primary_rank) - ctx.rank_value(current_rank)) / 15.0, -1.0, 1.0)


def _breaks_bomb(action: ActionCandidate, snapshot: SeatSnapshot) -> bool:
    if action.kind != "play_cards" or action.hand_type in {HandType.BOMB, HandType.STRAIGHT_FLUSH, HandType.FOUR_JOKERS}:
        return False
    selected = set(action.card_ids)
    ranks = {rank for rank, count in _rank_counts(snapshot.hand).items() if rank not in {Rank.SMALL_JOKER, Rank.BIG_JOKER} and count >= 4}
    if any(CARD_BY_ID[card_id].rank in ranks for card_id in selected):
        return True
    jokers = {card_id for card_id in snapshot.hand if CARD_BY_ID[card_id].is_joker}
    return len(jokers) == 4 and not selected.isdisjoint(jokers)


def _breaks_sequence(action: ActionCandidate, snapshot: SeatSnapshot) -> bool:
    if action.kind != "play_cards" or action.hand_type in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        return False
    selected_ranks = {CARD_BY_ID[card_id].rank for card_id in action.card_ids}
    hand_ranks = {CARD_BY_ID[card_id].rank for card_id in snapshot.hand if CARD_BY_ID[card_id].rank in STANDARD_RANKS}
    for start in range(0, len(tuple(STANDARD_RANKS)) - 4):
        window = set(STANDARD_RANKS[start : start + 5])
        if window.issubset(hand_ranks) and not selected_ranks.isdisjoint(window):
            return True
    return False


def _breaks_pair_run(action: ActionCandidate, snapshot: SeatSnapshot) -> bool:
    if action.kind != "play_cards" or action.hand_type in {
        HandType.PAIR,
        HandType.THREE_PAIR_RUN,
        HandType.TRIPLE_RUN,
        HandType.FULL_HOUSE,
    }:
        return False
    pair_ranks = {rank for rank, count in _rank_counts(snapshot.hand).items() if count >= 2}
    return any(CARD_BY_ID[card_id].rank in pair_ranks for card_id in action.card_ids)


def _rank_counts(card_ids: tuple[str, ...]) -> dict[Rank, int]:
    counts: dict[Rank, int] = {}
    for card_id in card_ids:
        rank = CARD_BY_ID[card_id].rank
        counts[rank] = counts.get(rank, 0) + 1
    return counts


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
