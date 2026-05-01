from __future__ import annotations

import unittest

from server.domain.cards import CARD_BY_ID, Card, Rank, Suit
from server.domain.seats import Seat
from tools.card_recorder import CardRecorder, InMemoryCardRecorderStore


class CardRecorderTests(unittest.TestCase):
    def test_start_match_records_seen_and_unseen_cards(self) -> None:
        recorder = CardRecorder()

        self.assertTrue(recorder.start_match("match-1"))
        recorder.turn(Seat.EAST, (CARD_BY_ID["D1-S-3"], CARD_BY_ID["D2-H-4"]))

        match = recorder.current_match
        self.assertIsNotNone(match)
        self.assertEqual(match.seen_cards[Seat.EAST], (CARD_BY_ID["D1-S-3"], CARD_BY_ID["D2-H-4"]))
        self.assertNotIn(CARD_BY_ID["D1-S-3"], match.unseen_cards)
        self.assertEqual(len(match.unseen_cards), 106)

    def test_start_match_rejects_second_active_match(self) -> None:
        recorder = CardRecorder()

        self.assertTrue(recorder.start_match("match-1"))

        self.assertFalse(recorder.start_match("match-2"))
        self.assertEqual(recorder.current_match.match_id, "match-1")

    def test_finish_match_allows_next_match(self) -> None:
        recorder = CardRecorder()
        recorder.start_match("match-1")

        recorder.finish_match()

        self.assertTrue(recorder.start_match("match-2"))
        self.assertEqual(recorder.current_match.match_id, "match-2")

    def test_finished_match_id_cannot_be_reused(self) -> None:
        recorder = CardRecorder()
        recorder.start_match("match-1")
        recorder.finish_match()

        self.assertFalse(recorder.start_match("match-1"))
        self.assertTrue(recorder.matches["match-1"].match_finished)

    def test_turn_requires_active_match_even_when_empty(self) -> None:
        recorder = CardRecorder()

        with self.assertRaisesRegex(RuntimeError, "no active match"):
            recorder.turn(Seat.EAST, ())

    def test_rejects_duplicate_or_already_seen_cards(self) -> None:
        recorder = CardRecorder()
        recorder.start_match("match-1")
        card = CARD_BY_ID["D1-S-3"]

        with self.assertRaisesRegex(ValueError, "duplicate cards"):
            recorder.turn(Seat.EAST, (card, card))

        recorder.turn(Seat.EAST, (card,))
        with self.assertRaisesRegex(ValueError, "already been seen"):
            recorder.turn(Seat.SOUTH, (card,))

    def test_rejects_unknown_cards(self) -> None:
        recorder = CardRecorder()
        recorder.start_match("match-1")
        unknown = Card(id="missing", deck=1, suit=Suit.SPADE, rank=Rank.THREE)

        with self.assertRaisesRegex(ValueError, "unknown card id: missing"):
            recorder.turn(Seat.EAST, (unknown,))

    def test_loads_unfinished_match_from_store(self) -> None:
        store = InMemoryCardRecorderStore()
        recorder = CardRecorder(store)
        recorder.start_match("match-1")
        recorder.turn(Seat.EAST, (CARD_BY_ID["D1-S-3"],))

        reloaded = CardRecorder(store)

        self.assertEqual(reloaded.current_match.match_id, "match-1")
        self.assertEqual(reloaded.current_match.seen_cards[Seat.EAST], (CARD_BY_ID["D1-S-3"],))


if __name__ == "__main__":
    unittest.main()
