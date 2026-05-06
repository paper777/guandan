from __future__ import annotations

from collections import Counter

from client.api import JsonObject
from npc.llm_agent.context import AgentRequestContext, rank, rank_value, suit


def build_strategy_context(context: AgentRequestContext) -> JsonObject:
    hand = context.hand
    counts = _rank_counts(hand)
    singles = sum(1 for count in counts.values() if count == 1)
    bombs = _bomb_count(counts)
    controls = _control_count(hand, context.current_level)
    hand_count = len(hand)
    partner = context.table_context.get("partner")
    opponents = context.table_context.get("opponents", [])
    hand_counts = context.table_context.get("hand_counts", {})
    partner_count = _seat_count(hand_counts, partner)
    opponent_counts = [
        count
        for count in (_seat_count(hand_counts, seat) for seat in opponents if isinstance(seat, str))
        if count is not None
    ]
    low_opponent_count = min(opponent_counts) if opponent_counts else None

    return {
        "hand_features": {
            "card_count": hand_count,
            "rank_count": len(counts),
            "single_rank_count": singles,
            "pair_or_triple_rank_count": sum(1 for count in counts.values() if count in {2, 3}),
            "bomb_count": bombs,
            "control_card_count": controls,
            "red_heart_level_count": _red_heart_level_count(hand, context.current_level),
        },
        "pressure": {
            "partner_card_count": partner_count,
            "lowest_opponent_card_count": low_opponent_count,
            "endgame_defense": low_opponent_count is not None and low_opponent_count <= 10,
            "partner_near_finish": partner_count is not None and partner_count <= 10,
        },
    }


def _rank_counts(card_ids: tuple[str, ...]) -> Counter[str]:
    return Counter(rank(card_id) for card_id in card_ids)


def _bomb_count(counts: Counter[str]) -> int:
    return sum(1 for count in counts.values() if count >= 4)


def _control_count(card_ids: tuple[str, ...], level: str) -> int:
    return sum(1 for card_id in card_ids if _is_control_card(card_id, level))


def _red_heart_level_count(card_ids: tuple[str, ...], level: str) -> int:
    return sum(1 for card_id in card_ids if suit(card_id) == "H" and rank(card_id) == level)


def _is_control_card(card_id: str, level: str) -> bool:
    card_rank = rank(card_id)
    if card_rank in {"SJ", "BJ"}:
        return True
    return card_rank == level or rank_value(card_rank, level) >= rank_value("A", level)


def _seat_count(hand_counts: object, seat: object) -> int | None:
    if not isinstance(hand_counts, dict) or not isinstance(seat, str):
        return None
    value = hand_counts.get(seat)
    return int(value) if isinstance(value, int) else None

