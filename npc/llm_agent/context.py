from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from client.api import ActionRequest, JsonObject
from server.domain.cards import CARD_BY_ID, Rank


RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
VALID_SEATS = {"E", "S", "W", "N"}


@dataclass(frozen=True, slots=True)
class AgentRequestContext:
    """Shared, derived request facts for LLM player collaborators."""

    request: ActionRequest
    seat: str | None
    hand: tuple[str, ...]
    public: JsonObject
    prompt_kind: object
    current_level: str
    current_trick: JsonObject | None
    table_context: JsonObject

    @classmethod
    def from_request(cls, request: ActionRequest) -> AgentRequestContext:
        seat = seat_from_request(request)
        public = request.snapshot.get("public")
        public_snapshot = public if isinstance(public, dict) else {}
        current_level = str(request.prompt.get("current_level") or public_snapshot.get("current_level") or "2")
        current_trick = _current_trick(request, public_snapshot)
        table_context = {
            "seat": seat,
            "team": team_for_seat(seat) if seat in VALID_SEATS else None,
            "partner": partner_for_seat(seat),
            "opponents": list(opponents_for_seat(seat)),
            "prompt_kind": request.prompt.get("kind"),
            "current_level": request.prompt.get("current_level") or public_snapshot.get("current_level"),
            "current_turn": public_snapshot.get("current_turn"),
            "acting_seat": public_snapshot.get("acting_seat") or public_snapshot.get("current_turn"),
            "hand_counts": public_snapshot.get("hand_counts", {}),
            "finish_order": public_snapshot.get("finish_order", []),
        }
        return cls(
            request=request,
            seat=seat,
            hand=tuple(str(card_id) for card_id in request.snapshot.get("hand", [])),
            public=public_snapshot,
            prompt_kind=request.prompt.get("kind"),
            current_level=current_level,
            current_trick=current_trick,
            table_context=table_context,
        )


def seat_from_request(request: ActionRequest) -> str | None:
    seat = request.snapshot.get("seat")
    return str(seat) if seat is not None else None


def snapshot_value(request: ActionRequest, key: str) -> object:
    return request.snapshot.get(key)


def public_value(request: ActionRequest, key: str) -> object:
    public = request.snapshot.get("public")
    return public.get(key) if isinstance(public, dict) else None


def team_for_seat(seat: str | None) -> str:
    return "EW" if seat in {"E", "W"} else "SN"


def partner_for_seat(seat: str | None) -> str | None:
    return {"E": "W", "W": "E", "S": "N", "N": "S"}.get(seat or "")


def opponents_for_seat(seat: str | None) -> tuple[str, ...]:
    team = team_for_seat(seat) if seat in VALID_SEATS else None
    if team == "EW":
        return ("S", "N")
    if team == "SN":
        return ("E", "W")
    return ()


def rank_enum(rank: str) -> Rank:
    return Rank(rank) if rank in Rank._value2member_map_ else Rank.TWO


def rank(card_id: str) -> str:
    card = CARD_BY_ID.get(card_id)
    if card is not None:
        return card.rank.value
    parts = card_id.split("-")
    return parts[-1] if parts else card_id


def suit(card_id: str) -> str | None:
    card = CARD_BY_ID.get(card_id)
    if card is not None:
        return card.suit.value if card.suit is not None else None
    parts = card_id.split("-")
    return parts[1] if len(parts) == 3 else None


def rank_value(card_rank: str, level: str) -> int:
    if card_rank == "BJ":
        return 15
    if card_rank == "SJ":
        return 14
    if card_rank == level:
        return 13
    return RANKS.index(card_rank) if card_rank in RANKS else -1


def is_red_heart_level_card(card_id: str, level: str) -> bool:
    return suit(card_id) == "H" and rank(card_id) == level


def rank_at_most_ten(card_id: str) -> bool:
    return rank(card_id) in {"2", "3", "4", "5", "6", "7", "8", "9", "10"}


def cards_by_rank(hand: tuple[str, ...]) -> dict[str, list[str]]:
    by_rank: dict[str, list[str]] = defaultdict(list)
    for card_id in hand:
        by_rank[rank(card_id)].append(card_id)
    return {card_rank: sorted(card_ids) for card_rank, card_ids in by_rank.items()}


def lowest_card_id(card_ids: tuple[str, ...], level: str) -> str:
    return min(card_ids, key=lambda card_id: (rank_value(rank(card_id), level), card_id))


def highest_eligible_tribute_card(card_ids: tuple[str, ...], level: str) -> str | None:
    eligible = tuple(card_id for card_id in card_ids if not is_red_heart_level_card(card_id, level))
    if not eligible:
        return None
    return max(eligible, key=lambda card_id: (rank_value(rank(card_id), level), card_id))


def safe_snapshot(snapshot: JsonObject) -> JsonObject:
    public = snapshot.get("public")
    return {
        "table_id": snapshot.get("table_id"),
        "seat": snapshot.get("seat"),
        "hand": list(snapshot.get("hand", [])),
        "public": public if isinstance(public, dict) else {},
    }


def _current_trick(request: ActionRequest, public: JsonObject) -> JsonObject | None:
    prompt_trick = request.prompt.get("current_trick")
    if isinstance(prompt_trick, dict):
        return prompt_trick
    public_trick = public.get("current_trick")
    return public_trick if isinstance(public_trick, dict) else None
