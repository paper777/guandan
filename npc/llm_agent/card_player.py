from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from client.api import ActionRequest, JsonObject
from server.domain.cards import CARD_BY_ID, Rank, resolve_cards
from server.domain.comparator import can_beat
from server.domain.hand_types import HandType, PlayedHand, parse_hand


RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")


@dataclass(frozen=True, slots=True)
class CardPlayerCandidate:
    """One legal action option plus the policy reason for considering it."""

    action: JsonObject
    reason: str
    priority: int

    def to_json(self) -> JsonObject:
        return {"action": self.action, "reason": self.reason, "priority": self.priority}


@dataclass(frozen=True, slots=True)
class CardPlayerAdvice:
    """Advisor output consumed both by model prompts and policy fallback."""

    legal_action: object
    recommended_action: JsonObject
    candidates: tuple[CardPlayerCandidate, ...]
    notes: tuple[str, ...]

    def to_json(self) -> JsonObject:
        return {
            "legal_action": self.legal_action,
            "recommended_action": self.recommended_action,
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "notes": list(self.notes),
        }


class CardPlayerAdvisor:
    """Deterministic advisor that gives the LLM concrete legal policy options."""

    def advise(self, request: ActionRequest, strategy_context: JsonObject | None = None) -> JsonObject:
        return self.build_advice(request, strategy_context).to_json()

    def build_advice(self, request: ActionRequest, strategy_context: JsonObject | None = None) -> CardPlayerAdvice:
        kind = request.prompt.get("kind")
        level = str(request.prompt.get("current_level") or _public_value(request, "current_level") or "2")
        hand = tuple(str(card_id) for card_id in request.snapshot.get("hand", []))
        candidates = tuple(_candidate_actions(kind, hand, level, request))
        recommended = candidates[0].action if candidates else {"type": "pass"}
        return CardPlayerAdvice(
            legal_action=kind,
            recommended_action=recommended,
            candidates=candidates,
            notes=tuple(_notes_for(strategy_context or {})),
        )


def _candidate_actions(kind: object, hand: tuple[str, ...], level: str, request: ActionRequest) -> list[CardPlayerCandidate]:
    if kind == "play_or_pass":
        return [
            *_beat_candidates(hand, level, request),
            CardPlayerCandidate(
                action={"type": "pass"},
                reason="Pass is legal, but prefer a low beating candidate when it preserves structure or wins useful tempo.",
                priority=99,
            ),
        ]
    if kind == "lead":
        return _lead_candidates(hand, level)
    if kind == "tribute":
        card_id = _highest_eligible_tribute_card(hand, level)
        if card_id is None:
            return []
        return [
            CardPlayerCandidate(
                action={"type": "submit_tribute", "card_id": card_id},
                reason="Tribute should submit the highest eligible non-red-heart-level card.",
                priority=0,
            )
        ]
    if kind == "return_tribute":
        eligible = hand
        if request.prompt.get("return_rank_at_most_ten", False):
            eligible = tuple(card_id for card_id in hand if _rank_at_most_ten(card_id))
        if not eligible:
            return []
        return [
            CardPlayerCandidate(
                action={"type": "return_tribute", "card_id": _lowest_card_id(eligible, level)},
                reason="Return the lowest eligible card to preserve stronger structures and controls.",
                priority=0,
            )
        ]
    return []


def _lead_candidates(hand: tuple[str, ...], level: str) -> list[CardPlayerCandidate]:
    if not hand:
        return []
    candidates = [
        CardPlayerCandidate(
            action={"type": "play_cards", "card_ids": [_lowest_card_id(hand, level)]},
            reason="Lowest single lead preserves stronger structures and controls.",
            priority=0,
        )
    ]
    by_rank = _cards_by_rank(hand)
    for size, label, priority in ((2, "pair", 1), (3, "three_of_a_kind", 2), (4, "bomb", 8)):
        group = _lowest_group(by_rank, size, level)
        if group is None:
            continue
        reason = (
            "Bomb lead is usually low priority; use only when strategy_context says tempo or endgame defense requires it."
            if size >= 4
            else f"Lowest {label} lead can preserve singles while testing structure."
        )
        candidates.append(CardPlayerCandidate(_action_for_group(group, label), reason, priority))
    return sorted(candidates, key=lambda candidate: candidate.priority)


def _beat_candidates(hand: tuple[str, ...], level: str, request: ActionRequest) -> list[CardPlayerCandidate]:
    current = _current_trick(request)
    if current is None:
        return _lead_candidates(hand, level)
    current_hand = _parse_current_hand(current, level)
    if current_hand is None:
        return []

    candidates: list[CardPlayerCandidate] = []
    target_type = current_hand.type.value
    target_size = len(current_hand.card_ids)
    for group in combinations(hand, target_size):
        candidate = _candidate_for_group(group, level, declared_type=target_type)
        if candidate is None or not can_beat(candidate, current_hand, _rank_enum(level)):
            continue
        candidates.append(
            CardPlayerCandidate(
                action=_action_for_group(group, candidate.type.value),
                reason=f"Lowest available {candidate.type.value} that beats the current trick.",
                priority=_response_priority(candidate.type.value, group, level),
            )
        )
    candidates.extend(_bomb_response_candidates(hand, level, current_hand))
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


def _current_trick(request: ActionRequest) -> JsonObject | None:
    prompt_trick = request.prompt.get("current_trick")
    if isinstance(prompt_trick, dict):
        return prompt_trick
    public = request.snapshot.get("public")
    if isinstance(public, dict) and isinstance(public.get("current_trick"), dict):
        return public["current_trick"]
    return None


def _parse_current_hand(current: JsonObject, level: str) -> PlayedHand | None:
    card_ids = current.get("card_ids")
    if not isinstance(card_ids, list):
        return None
    declared_type = current.get("hand_type")
    try:
        return parse_hand(
            resolve_cards(str(card_id) for card_id in card_ids),
            str(declared_type) if declared_type is not None else None,
            level=_rank_enum(level),
        )
    except ValueError:
        return None


def _candidate_for_group(group: tuple[str, ...], level: str, declared_type: str | None = None) -> PlayedHand | None:
    try:
        return parse_hand(resolve_cards(group), declared_type, level=_rank_enum(level))
    except ValueError:
        return None


def _bomb_response_candidates(hand: tuple[str, ...], level: str, current_hand: PlayedHand) -> list[CardPlayerCandidate]:
    by_rank = _cards_by_rank(hand)
    candidates: list[CardPlayerCandidate] = []
    for card_ids in by_rank.values():
        if len(card_ids) < 4:
            continue
        group = tuple(card_ids)
        candidate = _candidate_for_group(group, level, declared_type=HandType.BOMB.value)
        if candidate is None or not can_beat(candidate, current_hand, _rank_enum(level)):
            continue
        candidates.append(
            CardPlayerCandidate(
                action=_action_for_group(group, candidate.type.value),
                reason="Bomb can beat the current trick; use only for tempo, endgame defense, or partner delivery.",
                priority=_response_priority(candidate.type.value, group, level),
            )
        )
    jokers = tuple(card_id for card_id in hand if _rank(card_id) in {"SJ", "BJ"})
    if len(jokers) == 4:
        candidate = _candidate_for_group(jokers, level, declared_type=HandType.FOUR_JOKERS.value)
        if candidate is not None and can_beat(candidate, current_hand, _rank_enum(level)):
            candidates.append(
                CardPlayerCandidate(
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
    return base + _rank_value(_rank(group[0]), level)


def _dedupe_candidates(candidates: list[CardPlayerCandidate]) -> list[CardPlayerCandidate]:
    deduped: dict[tuple[str, tuple[str, ...], str], CardPlayerCandidate] = {}
    for candidate in sorted(candidates, key=lambda item: item.priority):
        key = (
            str(candidate.action.get("type")),
            tuple(str(card_id) for card_id in candidate.action.get("card_ids", [])),
            str(candidate.action.get("declared_type", "")),
        )
        deduped.setdefault(key, candidate)
    return list(deduped.values())


def _rank_enum(rank: str) -> Rank:
    return Rank(rank) if rank in Rank._value2member_map_ else Rank.TWO


def _cards_by_rank(hand: tuple[str, ...]) -> dict[str, list[str]]:
    by_rank: dict[str, list[str]] = defaultdict(list)
    for card_id in hand:
        by_rank[_rank(card_id)].append(card_id)
    return {rank: sorted(card_ids) for rank, card_ids in by_rank.items()}


def _lowest_group(by_rank: dict[str, list[str]], size: int, level: str) -> tuple[str, ...] | None:
    eligible = [tuple(card_ids[:size]) for card_ids in by_rank.values() if len(card_ids) >= size]
    if not eligible:
        return None
    return min(eligible, key=lambda group: (_rank_value(_rank(group[0]), level), group))


def _lowest_card_id(card_ids: tuple[str, ...], level: str) -> str:
    return min(card_ids, key=lambda card_id: (_rank_value(_rank(card_id), level), card_id))


def _highest_eligible_tribute_card(card_ids: tuple[str, ...], level: str) -> str | None:
    eligible = tuple(card_id for card_id in card_ids if not _is_red_heart_level_card(card_id, level))
    if not eligible:
        return None
    return max(eligible, key=lambda card_id: (_rank_value(_rank(card_id), level), card_id))


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


def _public_value(request: ActionRequest, key: str) -> object:
    public = request.snapshot.get("public")
    return public.get(key) if isinstance(public, dict) else None
