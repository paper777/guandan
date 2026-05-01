from __future__ import annotations

from client.api import ActionRequest, JsonObject
from npc.common.player import Player


RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")


class DummyBotPolicy(Player):
    """Minimal policy: pass when possible, lead the lowest card when forced."""

    def choose_action(self, request: ActionRequest) -> JsonObject:
        prompt_kind = request.prompt.get("kind")
        level = str(request.prompt.get("current_level", "2"))
        hand = tuple(str(card_id) for card_id in request.snapshot.get("hand", []))

        if prompt_kind == "play_or_pass":
            return {"type": "pass"}
        if prompt_kind == "lead":
            if not hand:
                return {"type": "error", "message": "cannot lead with an empty hand"}
            return {"type": "play_cards", "card_ids": [_lowest_card_id(hand, level)]}
        if prompt_kind == "tribute":
            card_id = _highest_eligible_tribute_card(hand, level)
            if card_id is None:
                return {"type": "error", "message": "no eligible tribute card"}
            return {"type": "submit_tribute", "card_id": card_id}
        if prompt_kind == "return_tribute":
            if request.prompt.get("return_rank_at_most_ten", False):
                hand = tuple(card_id for card_id in hand if _rank_at_most_ten(card_id))
            if not hand:
                return {"type": "error", "message": "no eligible return card"}
            return {"type": "return_tribute", "card_id": _lowest_card_id(hand, level)}
        return {"type": "pass"}


def _lowest_card_id(card_ids: tuple[str, ...], level: str) -> str:
    return min(card_ids, key=lambda card_id: (_rank_value(_rank(card_id), level), card_id))


def _highest_eligible_tribute_card(card_ids: tuple[str, ...], level: str) -> str | None:
    eligible = tuple(card_id for card_id in card_ids if not _is_red_heart_level_card(card_id, level))
    if not eligible:
        return None
    return max(eligible, key=lambda card_id: (_rank_value(_rank(card_id), level), card_id))


def _rank(card_id: str) -> str:
    parts = card_id.split("-")
    return parts[-1] if parts else card_id


def _suit(card_id: str) -> str | None:
    parts = card_id.split("-")
    return parts[1] if len(parts) == 3 else None


def _rank_value(rank: str, level: str) -> int:
    if rank == "BJ":
        return 15
    if rank == "SJ":
        return 14
    if rank == level:
        return 13
    return RANKS.index(rank) if rank in RANKS else -1


def _is_red_heart_level_card(card_id: str, level: str) -> bool:
    return _suit(card_id) == "H" and _rank(card_id) == level


def _rank_at_most_ten(card_id: str) -> bool:
    return _rank(card_id) in {"2", "3", "4", "5", "6", "7", "8", "9", "10"}
