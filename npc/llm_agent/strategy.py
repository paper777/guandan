from __future__ import annotations

from collections import Counter

from client.api import ActionRequest, JsonObject
from server.domain.cards import CARD_BY_ID, Rank


RANK_STRENGTH = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "10": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "A": 12,
    "SJ": 14,
    "BJ": 15,
}


def build_strategy_context(request: ActionRequest, table_context: JsonObject) -> JsonObject:
    hand = tuple(str(card_id) for card_id in request.snapshot.get("hand", []))
    counts = _rank_counts(hand)
    singles = sum(1 for count in counts.values() if count == 1)
    bombs = _bomb_count(counts)
    controls = _control_count(hand, str(table_context.get("current_level") or "2"))
    hand_count = len(hand)
    partner = table_context.get("partner")
    opponents = table_context.get("opponents", [])
    hand_counts = table_context.get("hand_counts", {})
    partner_count = _seat_count(hand_counts, partner)
    opponent_counts = [
        count
        for count in (_seat_count(hand_counts, seat) for seat in opponents if isinstance(seat, str))
        if count is not None
    ]
    low_opponent_count = min(opponent_counts) if opponent_counts else None
    role = _role_for(hand_count, bombs, controls, singles, partner_count)

    return {
        "role_estimate": role,
        "hand_features": {
            "card_count": hand_count,
            "rank_count": len(counts),
            "single_rank_count": singles,
            "pair_or_triple_rank_count": sum(1 for count in counts.values() if count in {2, 3}),
            "bomb_count": bombs,
            "control_card_count": controls,
            "red_heart_level_count": _red_heart_level_count(hand, str(table_context.get("current_level") or "2")),
        },
        "pressure": {
            "partner_card_count": partner_count,
            "lowest_opponent_card_count": low_opponent_count,
            "endgame_defense": low_opponent_count is not None and low_opponent_count <= 10,
            "partner_near_finish": partner_count is not None and partner_count <= 10,
        },
        "priorities": _priorities_for(role),
    }


def _rank_counts(card_ids: tuple[str, ...]) -> Counter[str]:
    return Counter(_rank(card_id) for card_id in card_ids)


def _bomb_count(counts: Counter[str]) -> int:
    return sum(1 for count in counts.values() if count >= 4)


def _control_count(card_ids: tuple[str, ...], level: str) -> int:
    return sum(1 for card_id in card_ids if _is_control_card(card_id, level))


def _red_heart_level_count(card_ids: tuple[str, ...], level: str) -> int:
    return sum(1 for card_id in card_ids if _suit(card_id) == "H" and _rank(card_id) == level)


def _is_control_card(card_id: str, level: str) -> bool:
    rank = _rank(card_id)
    if rank in {"SJ", "BJ"}:
        return True
    return rank == level or RANK_STRENGTH.get(rank, -1) >= RANK_STRENGTH["A"]


def _rank(card_id: str) -> str:
    card = CARD_BY_ID.get(card_id)
    if card is not None:
        return card.rank.value
    parts = card_id.split("-")
    return parts[-1] if parts else card_id


def _suit(card_id: str) -> str | None:
    card = CARD_BY_ID.get(card_id)
    if card is not None:
        return card.suit.value if card.suit is not None else None
    parts = card_id.split("-")
    return parts[1] if len(parts) == 3 else None


def _seat_count(hand_counts: object, seat: object) -> int | None:
    if not isinstance(hand_counts, dict) or not isinstance(seat, str):
        return None
    value = hand_counts.get(seat)
    return int(value) if isinstance(value, int) else None


def _role_for(
    hand_count: int,
    bombs: int,
    controls: int,
    singles: int,
    partner_count: int | None,
) -> str:
    if partner_count is not None and partner_count <= 10:
        return "support_partner_finish"
    if bombs >= 2 or controls >= 4 or (hand_count <= 10 and controls >= 2):
        return "primary_attacker"
    if singles >= max(5, hand_count // 3) and bombs == 0:
        return "support_guard"
    return "balanced"


def _priorities_for(role: str) -> list[str]:
    common = [
        "Decide role before choosing cards.",
        "Preserve strong structures unless breaking them clearly reduces effective turns or directly helps partner finish.",
        "Treat bombs as tempo tools: use them to regain/deny control only when follow-up is available or endgame defense requires it.",
        "Do not split bombs as routine play; consider splitting only for weak four-bombs in fragmented hands or to send partner out.",
    ]
    if role == "primary_attacker":
        return [
            "Compress your own effective turns while keeping at least one control path.",
            "Use partner cooperation opportunities, but do not give away winning tempo unnecessarily.",
            *common,
        ]
    if role == "support_partner_finish":
        return [
            "Prioritize returning tempo to partner and blocking opponents over personal fast exit.",
            "Lead or respond with shapes that partner can likely use and opponents are less likely to continue.",
            *common,
        ]
    if role == "support_guard":
        return [
            "Act as guard: block opponent runs, preserve control cards, and feed partner when possible.",
            "Avoid spending the last control card unless it prevents an opponent from finishing or gives partner tempo.",
            *common,
        ]
    return [
        "Balance personal turn reduction with partner tempo and opponent blocking.",
        "Prefer legal plays that keep pairs/runs/triples intact over isolated single-card exits.",
        *common,
    ]
