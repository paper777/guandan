from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from client.api import ActionRequest, JsonObject
from npc.llm_agent.context import (
    AgentRequestContext,
    cards_by_rank,
    highest_eligible_tribute_card,
    lowest_card_id,
    rank,
    rank_at_most_ten,
    rank_enum,
    rank_value,
)
from server.domain.cards import resolve_cards
from server.domain.comparator import can_beat
from server.domain.hand_types import HandType, PlayedHand, parse_hand


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    """One legal action option plus the policy reason for considering it."""

    action: JsonObject
    reason: str
    priority: int

    def to_json(self) -> JsonObject:
        return {"action": self.action, "reason": self.reason, "priority": self.priority}


@dataclass(frozen=True, slots=True)
class ActionAdvice:
    """Advisor output consumed both by model prompts and policy fallback."""

    legal_action: object
    recommended_action: JsonObject
    candidates: tuple[ActionCandidate, ...]
    notes: tuple[str, ...]

    def to_json(self) -> JsonObject:
        return {
            "legal_action": self.legal_action,
            "recommended_action": self.recommended_action,
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "notes": list(self.notes),
        }


class ActionAdvisor:
    """Deterministic advisor that gives the LLM concrete legal policy options."""

    def advise(self, request: ActionRequest, strategy_context: JsonObject | None = None) -> JsonObject:
        return self.build_advice(AgentRequestContext.from_request(request), strategy_context).to_json()

    def build_advice(
        self,
        context: AgentRequestContext | ActionRequest,
        strategy_context: JsonObject | None = None,
    ) -> ActionAdvice:
        if isinstance(context, ActionRequest):
            context = AgentRequestContext.from_request(context)
        candidates = tuple(_candidate_actions(context))
        recommended = candidates[0].action if candidates else {"type": "pass"}
        return ActionAdvice(
            legal_action=context.prompt_kind,
            recommended_action=recommended,
            candidates=candidates,
            notes=tuple(_notes_for(strategy_context or {})),
        )


def _candidate_actions(context: AgentRequestContext) -> list[ActionCandidate]:
    if context.prompt_kind == "play_or_pass":
        return [
            *_beat_candidates(context),
            ActionCandidate(
                action={"type": "pass"},
                reason="Pass is legal, but prefer a low beating candidate when it preserves structure or wins useful tempo.",
                priority=99,
            ),
        ]
    if context.prompt_kind == "lead":
        return _lead_candidates(context.hand, context.current_level)
    if context.prompt_kind == "tribute":
        card_id = highest_eligible_tribute_card(context.hand, context.current_level)
        if card_id is None:
            return []
        return [
            ActionCandidate(
                action={"type": "submit_tribute", "card_id": card_id},
                reason="Tribute should submit the highest eligible non-red-heart-level card.",
                priority=0,
            )
        ]
    if context.prompt_kind == "return_tribute":
        eligible = context.hand
        if context.request.prompt.get("return_rank_at_most_ten", False):
            eligible = tuple(card_id for card_id in context.hand if rank_at_most_ten(card_id))
        if not eligible:
            return []
        return [
            ActionCandidate(
                action={"type": "return_tribute", "card_id": lowest_card_id(eligible, context.current_level)},
                reason="Return the lowest eligible card to preserve stronger structures and controls.",
                priority=0,
            )
        ]
    return []


def _lead_candidates(hand: tuple[str, ...], level: str) -> list[ActionCandidate]:
    if not hand:
        return []
    candidates = [
        ActionCandidate(
            action={"type": "play_cards", "card_ids": [lowest_card_id(hand, level)]},
            reason="Lowest single lead preserves stronger structures and controls.",
            priority=0,
        )
    ]
    by_rank = cards_by_rank(hand)
    for size, label, priority in ((2, "pair", 1), (3, "three_of_a_kind", 2), (4, "bomb", 8)):
        group = _lowest_group(by_rank, size, level)
        if group is None:
            continue
        reason = (
            "Bomb lead is usually low priority; use only when strategy_context says tempo or endgame defense requires it."
            if size >= 4
            else f"Lowest {label} lead can preserve singles while testing structure."
        )
        candidates.append(ActionCandidate(_action_for_group(group, label), reason, priority))
    return sorted(candidates, key=lambda candidate: candidate.priority)


def _beat_candidates(context: AgentRequestContext) -> list[ActionCandidate]:
    current = context.current_trick
    if current is None:
        return _lead_candidates(context.hand, context.current_level)
    current_hand = _parse_current_hand(current, context.current_level)
    if current_hand is None:
        return []

    candidates: list[ActionCandidate] = []
    target_type = current_hand.type.value
    target_size = len(current_hand.card_ids)
    for group in combinations(context.hand, target_size):
        candidate = _candidate_for_group(group, context.current_level, declared_type=target_type)
        if candidate is None or not can_beat(candidate, current_hand, rank_enum(context.current_level)):
            continue
        candidates.append(
            ActionCandidate(
                action=_action_for_group(group, candidate.type.value),
                reason=f"Lowest available {candidate.type.value} that beats the current trick.",
                priority=_response_priority(candidate.type.value, group, context.current_level),
            )
        )
    candidates.extend(_bomb_response_candidates(context.hand, context.current_level, current_hand))
    return _dedupe_candidates(candidates)[:8]


def _notes_for(strategy_context: JsonObject) -> list[str]:
    notes = [
        "Prefer a candidate action unless there is a clear strategic reason to choose another valid action.",
        "Never choose cards outside snapshot.hand.",
    ]
    role = strategy_context.get("role_estimate")
    if role == "support_partner_finish":
        notes.append("Favor candidates that return tempo to partner or block opponents over personal fast exit.")
    if role == "primary_attacker":
        notes.append("Favor candidates that reduce effective turns while preserving a follow-up control path.")
    return notes


def _parse_current_hand(current: JsonObject, level: str) -> PlayedHand | None:
    card_ids = current.get("card_ids")
    if not isinstance(card_ids, list):
        return None
    declared_type = current.get("hand_type")
    try:
        return parse_hand(
            resolve_cards(str(card_id) for card_id in card_ids),
            str(declared_type) if declared_type is not None else None,
            level=rank_enum(level),
        )
    except ValueError:
        return None


def _candidate_for_group(group: tuple[str, ...], level: str, declared_type: str | None = None) -> PlayedHand | None:
    try:
        return parse_hand(resolve_cards(group), declared_type, level=rank_enum(level))
    except ValueError:
        return None


def _bomb_response_candidates(hand: tuple[str, ...], level: str, current_hand: PlayedHand) -> list[ActionCandidate]:
    by_rank = cards_by_rank(hand)
    candidates: list[ActionCandidate] = []
    for card_ids in by_rank.values():
        if len(card_ids) < 4:
            continue
        group = tuple(card_ids)
        candidate = _candidate_for_group(group, level, declared_type=HandType.BOMB.value)
        if candidate is None or not can_beat(candidate, current_hand, rank_enum(level)):
            continue
        candidates.append(
            ActionCandidate(
                action=_action_for_group(group, candidate.type.value),
                reason="Bomb can beat the current trick; use only for tempo, endgame defense, or partner delivery.",
                priority=_response_priority(candidate.type.value, group, level),
            )
        )
    jokers = tuple(card_id for card_id in hand if rank(card_id) in {"SJ", "BJ"})
    if len(jokers) == 4:
        candidate = _candidate_for_group(jokers, level, declared_type=HandType.FOUR_JOKERS.value)
        if candidate is not None and can_beat(candidate, current_hand, rank_enum(level)):
            candidates.append(
                ActionCandidate(
                    action=_action_for_group(jokers, candidate.type.value),
                    reason="Four jokers beat all other hands; reserve for decisive tempo or defense.",
                    priority=50,
                )
            )
    return candidates


def _action_for_group(group: tuple[str, ...], declared_type: str) -> JsonObject:
    action: JsonObject = {"type": "play_cards", "card_ids": list(group)}
    if declared_type:
        action["declared_type"] = declared_type
    return action


def _response_priority(declared_type: str, group: tuple[str, ...], level: str) -> int:
    bomb_like = {HandType.BOMB.value, HandType.FOUR_JOKERS.value, HandType.STRAIGHT_FLUSH.value}
    base = 40 if declared_type in bomb_like else 0
    return base + rank_value(rank(group[0]), level)


def _dedupe_candidates(candidates: list[ActionCandidate]) -> list[ActionCandidate]:
    deduped: dict[tuple[str, tuple[str, ...], str], ActionCandidate] = {}
    for candidate in sorted(candidates, key=lambda item: item.priority):
        key = (
            str(candidate.action.get("type")),
            tuple(str(card_id) for card_id in candidate.action.get("card_ids", [])),
            str(candidate.action.get("declared_type", "")),
        )
        deduped.setdefault(key, candidate)
    return list(deduped.values())


def _lowest_group(by_rank: dict[str, list[str]], size: int, level: str) -> tuple[str, ...] | None:
    eligible = [tuple(card_ids[:size]) for card_ids in by_rank.values() if len(card_ids) >= size]
    if not eligible:
        return None
    return min(eligible, key=lambda group: (rank_value(rank(group[0]), level), group))
