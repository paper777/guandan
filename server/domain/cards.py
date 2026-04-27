from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from random import Random
from secrets import token_bytes
from typing import Iterable


class Suit(StrEnum):
    SPADE = "S"
    HEART = "H"
    DIAMOND = "D"
    CLUB = "C"


class Rank(StrEnum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"
    SMALL_JOKER = "SJ"
    BIG_JOKER = "BJ"


STANDARD_RANKS: tuple[Rank, ...] = (
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

LEVEL_RANKS: tuple[Rank, ...] = STANDARD_RANKS


@dataclass(frozen=True, slots=True)
class Card:
    id: str
    deck: int
    suit: Suit | None
    rank: Rank

    @property
    def is_joker(self) -> bool:
        return self.rank in {Rank.SMALL_JOKER, Rank.BIG_JOKER}


def build_deck() -> tuple[Card, ...]:
    cards: list[Card] = []
    for deck in (1, 2):
        for suit in Suit:
            for rank in STANDARD_RANKS:
                cards.append(Card(id=f"D{deck}-{suit.value}-{rank.value}", deck=deck, suit=suit, rank=rank))
        cards.append(Card(id=f"D{deck}-{Rank.SMALL_JOKER.value}", deck=deck, suit=None, rank=Rank.SMALL_JOKER))
        cards.append(Card(id=f"D{deck}-{Rank.BIG_JOKER.value}", deck=deck, suit=None, rank=Rank.BIG_JOKER))
    return tuple(cards)


DECK: tuple[Card, ...] = build_deck()
CARD_BY_ID: dict[str, Card] = {card.id: card for card in DECK}


def shuffled_deck(seed: str | int | bytes | None = None) -> tuple[Card, ...]:
    deck = list(DECK)
    Random(token_bytes(32) if seed is None else seed).shuffle(deck)
    return tuple(deck)


def deal_cards(seed: str | int | bytes | None = None, player_count: int = 4) -> tuple[tuple[str, ...], ...]:
    deck = shuffled_deck(seed)
    if len(deck) % player_count != 0:
        raise ValueError("deck cannot be evenly dealt")
    hands = [[] for _ in range(player_count)]
    for index, card in enumerate(deck):
        hands[index % player_count].append(card.id)
    return tuple(tuple(hand) for hand in hands)


def resolve_cards(card_ids: Iterable[str]) -> tuple[Card, ...]:
    cards: list[Card] = []
    for card_id in card_ids:
        try:
            cards.append(CARD_BY_ID[card_id])
        except KeyError as exc:
            raise ValueError(f"unknown card id: {card_id}") from exc
    return tuple(cards)


def is_red_heart_level_card(card: Card, level: Rank) -> bool:
    return card.suit == Suit.HEART and card.rank == level
