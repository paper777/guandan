from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from guandan.domain.cards import STANDARD_RANKS, Card, Rank, Suit, is_red_heart_level_card


class HandType(StrEnum):
    SINGLE = "single"
    PAIR = "pair"
    THREE_OF_A_KIND = "three_of_a_kind"
    THREE_PAIR_RUN = "three_pair_run"
    TRIPLE_RUN = "triple_run"
    FULL_HOUSE = "full_house"
    STRAIGHT = "straight"
    STRAIGHT_FLUSH = "straight_flush"
    BOMB = "bomb"
    FOUR_JOKERS = "four_jokers"


class AmbiguousHandError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PlayedHand:
    card_ids: tuple[str, ...]
    type: HandType
    primary_rank: Rank
    length: int
    wild_assignments: tuple[tuple[str, Suit, Rank], ...] = ()

    @property
    def is_bomb_like(self) -> bool:
        return self.type in {HandType.BOMB, HandType.STRAIGHT_FLUSH, HandType.FOUR_JOKERS}


SEQUENCE_RANKS: tuple[Rank, ...] = (
    Rank.ACE,
    Rank.TWO,
    Rank.THREE,
    Rank.FOUR,
    Rank.FIVE,
    Rank.SIX,
    Rank.SEVEN,
    Rank.EIGHT,
    Rank.NINE,
    Rank.TEN,
    Rank.JACK,
    Rank.QUEEN,
    Rank.KING,
    Rank.ACE,
)


def parse_hand(cards: tuple[Card, ...], declared_type: str | None = None, level: Rank = Rank.TWO) -> PlayedHand:
    if not cards:
        raise ValueError("empty play is not a hand")
    card_ids = tuple(card.id for card in cards)
    candidates = _parse_with_wildcards(cards, card_ids, level)
    if declared_type is not None:
        candidates = [hand for hand in candidates if hand.type.value == declared_type]
    if not candidates:
        raise ValueError("cards do not form a supported Guandan hand")
    if len(candidates) > 1 and declared_type is not None:
        return max(candidates, key=lambda hand: _rank_sort_value(hand.primary_rank, level))
    if len(candidates) > 1 and declared_type is None:
        raise AmbiguousHandError("hand is ambiguous; declared_type is required")
    return candidates[0]


def _rank_sort_value(rank: Rank, level: Rank) -> int:
    if rank == Rank.BIG_JOKER:
        return 15
    if rank == Rank.SMALL_JOKER:
        return 14
    if rank == level:
        return 13
    return STANDARD_RANKS.index(rank)


def _parse_with_wildcards(cards: tuple[Card, ...], card_ids: tuple[str, ...], level: Rank) -> list[PlayedHand]:
    natural = _parse_candidates(cards, card_ids, wild_assignments=())
    wild_indexes = [index for index, card in enumerate(cards) if is_red_heart_level_card(card, level)]
    if not wild_indexes or len(cards) == 1:
        return natural

    candidates = list(natural)
    replacements = tuple((suit, rank) for suit in Suit for rank in STANDARD_RANKS)
    for assignment_values in product(replacements, repeat=len(wild_indexes)):
        expanded = list(cards)
        assignments: list[tuple[str, Suit, Rank]] = []
        for card_index, (suit, rank) in zip(wild_indexes, assignment_values, strict=True):
            wild = cards[card_index]
            expanded[card_index] = Card(id=wild.id, deck=wild.deck, suit=suit, rank=rank)
            assignments.append((wild.id, suit, rank))
        candidates.extend(_parse_candidates(tuple(expanded), card_ids, wild_assignments=tuple(assignments)))

    deduped: dict[tuple[HandType, Rank, int, tuple[tuple[str, Suit, Rank], ...]], PlayedHand] = {}
    for candidate in candidates:
        key = (candidate.type, candidate.primary_rank, candidate.length, candidate.wild_assignments)
        deduped[key] = candidate
    return list(deduped.values())


def _parse_candidates(
    cards: tuple[Card, ...],
    card_ids: tuple[str, ...],
    wild_assignments: tuple[tuple[str, Suit, Rank], ...],
) -> list[PlayedHand]:
    ranks = [card.rank for card in cards]
    counts = Counter(ranks)
    candidates: list[PlayedHand] = []

    if len(cards) == 1:
        candidates.append(PlayedHand(card_ids, HandType.SINGLE, ranks[0], 1, wild_assignments))

    if len(cards) == 2 and len(counts) == 1:
        candidates.append(PlayedHand(card_ids, HandType.PAIR, ranks[0], 2, wild_assignments))

    if len(cards) == 3 and len(counts) == 1:
        candidates.append(PlayedHand(card_ids, HandType.THREE_OF_A_KIND, ranks[0], 3, wild_assignments))

    if _is_four_jokers(cards):
        candidates.append(PlayedHand(card_ids, HandType.FOUR_JOKERS, Rank.BIG_JOKER, 4, wild_assignments))

    if len(cards) >= 4 and len(counts) == 1:
        candidates.append(PlayedHand(card_ids, HandType.BOMB, ranks[0], len(cards), wild_assignments))

    if len(cards) == 5:
        full_house_rank = _full_house_primary(counts)
        if full_house_rank is not None:
            candidates.append(PlayedHand(card_ids, HandType.FULL_HOUSE, full_house_rank, 5, wild_assignments))

        straight_high = _straight_high_rank(cards)
        if straight_high is not None:
            candidates.append(PlayedHand(card_ids, HandType.STRAIGHT, straight_high, 5, wild_assignments))
            if _same_suit(cards):
                candidates.append(PlayedHand(card_ids, HandType.STRAIGHT_FLUSH, straight_high, 5, wild_assignments))

    if len(cards) == 6:
        pair_run_high = _same_count_run_high(counts, count=2, run_length=3)
        if pair_run_high is not None:
            candidates.append(PlayedHand(card_ids, HandType.THREE_PAIR_RUN, pair_run_high, 6, wild_assignments))

        triple_run_high = _same_count_run_high(counts, count=3, run_length=2)
        if triple_run_high is not None:
            candidates.append(PlayedHand(card_ids, HandType.TRIPLE_RUN, triple_run_high, 6, wild_assignments))

    return candidates


def _is_four_jokers(cards: tuple[Card, ...]) -> bool:
    ranks = Counter(card.rank for card in cards)
    return len(cards) == 4 and ranks == Counter({Rank.SMALL_JOKER: 2, Rank.BIG_JOKER: 2})


def _full_house_primary(counts: Counter[Rank]) -> Rank | None:
    if sorted(counts.values()) != [2, 3]:
        return None
    for rank, count in counts.items():
        if count == 3:
            return rank
    return None


def _straight_high_rank(cards: tuple[Card, ...]) -> Rank | None:
    if any(card.is_joker for card in cards):
        return None
    ranks = tuple(card.rank for card in cards)
    if len(set(ranks)) != 5:
        return None
    for index in range(0, len(SEQUENCE_RANKS) - 4):
        window = SEQUENCE_RANKS[index : index + 5]
        if set(ranks) == set(window):
            return window[-1]
    return None


def _same_suit(cards: tuple[Card, ...]) -> bool:
    suits = {card.suit for card in cards}
    return len(suits) == 1 and None not in suits


def _same_count_run_high(counts: Counter[Rank], count: int, run_length: int) -> Rank | None:
    if len(counts) != run_length or any(value != count for value in counts.values()):
        return None
    ranks = tuple(counts)
    if any(rank in {Rank.SMALL_JOKER, Rank.BIG_JOKER} for rank in ranks):
        return None
    for index in range(0, len(SEQUENCE_RANKS) - run_length + 1):
        window = SEQUENCE_RANKS[index : index + run_length]
        if len(set(window)) == run_length and set(ranks) == set(window):
            return window[-1]
    return None
