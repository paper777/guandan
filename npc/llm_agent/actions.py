from __future__ import annotations

from client.types import JsonObject
from npc.llm_agent.context import AgentRequestContext


def validate_action(action: JsonObject, context: AgentRequestContext) -> JsonObject | None:
    """Normalize a provider/advisor action if it is valid for this request."""

    action_type = action.get("type")
    prompt_kind = context.prompt_kind
    hand = set(context.hand)

    if action_type == "pass":
        return {"type": "pass"} if prompt_kind == "play_or_pass" else None
    if action_type == "play_cards":
        if prompt_kind not in {"lead", "play_or_pass"}:
            return None
        card_ids = tuple(str(card_id) for card_id in action.get("card_ids", []))
        if not card_ids or len(set(card_ids)) != len(card_ids) or not set(card_ids).issubset(hand):
            return None
        normalized: JsonObject = {"type": "play_cards", "card_ids": list(card_ids)}
        declared_type = action.get("declared_type")
        if declared_type is not None:
            normalized["declared_type"] = str(declared_type)
        return normalized
    if action_type == "submit_tribute":
        if prompt_kind != "tribute":
            return None
        card_id = str(action.get("card_id", ""))
        return {"type": "submit_tribute", "card_id": card_id} if card_id in hand else None
    if action_type == "return_tribute":
        if prompt_kind != "return_tribute":
            return None
        card_id = str(action.get("card_id", ""))
        return {"type": "return_tribute", "card_id": card_id} if card_id in hand else None
    return None
