from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Mapping

from server.domain.cards import CARD_BY_ID, DECK, Card
from server.domain.seats import SEATS, Seat


@dataclass(frozen=True, slots=True)
class MatchCards:
    match_id: str
    match_finished: bool = False
    seen_cards: dict[Seat, tuple[Card, ...]] = field(
        default_factory=lambda: {seat: () for seat in SEATS}
    )
    unseen_cards: tuple[Card, ...] = DECK


class CardRecorderStore(ABC):
    @abstractmethod
    def persist(self, match: MatchCards) -> None:
        """Persist the latest state for one match."""

    @abstractmethod
    def load(self) -> Mapping[str, MatchCards]:
        """Load all persisted matches keyed by match id."""


class InMemoryCardRecorderStore(CardRecorderStore):
    def __init__(self, matches: Mapping[str, MatchCards] | None = None) -> None:
        self._matches = dict(matches or {})

    def persist(self, match: MatchCards) -> None:
        self._matches[match.match_id] = match

    def load(self) -> Mapping[str, MatchCards]:
        return dict(self._matches)


class CardRecorder:
    def __init__(self, store: CardRecorderStore | None = None) -> None:
        self.store = store or InMemoryCardRecorderStore()
        self.matches = dict(self.store.load())
        self.current_match = self._find_current_match()

    def start_match(self, match_id: str) -> bool:
        if self.current_match is not None and not self.current_match.match_finished:
            return False
        if match_id in self.matches:
            if not self.matches[match_id].match_finished:
                self.current_match = self.matches[match_id]
            return False

        match = MatchCards(match_id=match_id)
        self.matches[match_id] = match
        self.current_match = match
        self.store.persist(match)
        return True

    def finish_match(self) -> None:
        match = self._require_current_match()
        finished = replace(match, match_finished=True)
        self.matches[finished.match_id] = finished
        self.current_match = finished
        self.store.persist(finished)

    def turn(self, seat: Seat, cards: tuple[Card, ...]) -> None:
        match = self._require_current_match()
        if not cards:
            return
        normalized_seat = Seat(seat)
        self._validate_cards(cards, match)

        seen_cards = dict(match.seen_cards)
        seen_cards[normalized_seat] = seen_cards.get(normalized_seat, ()) + cards
        played_ids = {card.id for card in cards}
        updated = replace(
            match,
            seen_cards=seen_cards,
            unseen_cards=tuple(card for card in match.unseen_cards if card.id not in played_ids),
        )
        self.matches[updated.match_id] = updated
        self.current_match = updated
        self.store.persist(updated)

    def _find_current_match(self) -> MatchCards | None:
        unfinished = tuple(match for match in self.matches.values() if not match.match_finished)
        if len(unfinished) > 1:
            raise ValueError("loaded card recorder state has multiple unfinished matches")
        return unfinished[0] if unfinished else None

    def _require_current_match(self) -> MatchCards:
        if self.current_match is None or self.current_match.match_finished:
            raise RuntimeError("no active match")
        return self.current_match

    def _validate_cards(self, cards: tuple[Card, ...], match: MatchCards) -> None:
        card_ids = tuple(card.id for card in cards)
        if len(set(card_ids)) != len(card_ids):
            raise ValueError("turn contains duplicate cards")
        unknown = [card_id for card_id in card_ids if card_id not in CARD_BY_ID]
        if unknown:
            raise ValueError(f"unknown card id: {unknown[0]}")
        unseen_ids = {card.id for card in match.unseen_cards}
        already_seen = [card_id for card_id in card_ids if card_id not in unseen_ids]
        if already_seen:
            raise ValueError(f"card has already been seen: {already_seen[0]}")
