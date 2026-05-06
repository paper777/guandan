from __future__ import annotations

import unittest

from server.domain.cards import CARD_BY_ID
from server.domain.cards import Rank
from server.domain.hand_types import HandType, parse_hand


def cards(*ids: str):
    return tuple(CARD_BY_ID[card_id] for card_id in ids)


class HandTypeTests(unittest.TestCase):
    def test_parse_ordinary_hand_types(self) -> None:
        cases = [
            (("D1-S-3",), HandType.SINGLE),
            (("D1-S-3", "D2-S-3"), HandType.PAIR),
            (("D1-S-3", "D2-S-3", "D1-H-3"), HandType.THREE_OF_A_KIND),
            (("D1-S-3", "D2-S-3", "D1-H-4", "D2-H-4", "D1-C-5", "D2-C-5"), HandType.THREE_PAIR_RUN),
            (("D1-S-3", "D2-S-3", "D1-H-3", "D1-S-4", "D2-S-4", "D1-H-4"), HandType.TRIPLE_RUN),
            (("D1-S-3", "D2-S-3", "D1-H-3", "D1-S-4", "D2-S-4"), HandType.FULL_HOUSE),
            (("D1-S-3", "D1-H-4", "D1-C-5", "D1-D-6", "D2-S-7"), HandType.STRAIGHT),
            (("D1-S-3", "D1-S-4", "D1-S-5", "D1-S-6", "D1-S-7"), HandType.STRAIGHT_FLUSH),
            (("D1-S-3", "D2-S-3", "D1-H-3", "D2-H-3"), HandType.BOMB),
            (("D1-SJ", "D2-SJ", "D1-BJ", "D2-BJ"), HandType.FOUR_JOKERS),
        ]

        for card_ids, expected in cases:
            with self.subTest(expected=expected):
                hand = parse_hand(cards(*card_ids), declared_type=expected.value)
                self.assertEqual(hand.type, expected)

    def test_rejects_bad_straight_wrap(self) -> None:
        with self.assertRaises(ValueError):
            parse_hand(cards("D1-S-J", "D1-H-Q", "D1-C-K", "D1-D-A", "D2-S-2"))

    def test_requires_declaration_for_straight_flush_ambiguity(self) -> None:
        same_suit_straight = cards("D1-S-3", "D1-S-4", "D1-S-5", "D1-S-6", "D1-S-7")

        with self.assertRaises(ValueError):
            parse_hand(same_suit_straight)

        self.assertEqual(parse_hand(same_suit_straight, "straight").type, HandType.STRAIGHT)
        self.assertEqual(parse_hand(same_suit_straight, "straight_flush").type, HandType.STRAIGHT_FLUSH)

    def test_red_heart_level_card_can_complete_pair(self) -> None:
        hand = parse_hand(cards("D1-S-8", "D1-H-2"), "pair", level=Rank.TWO)

        self.assertEqual(hand.type, HandType.PAIR)
        self.assertEqual(hand.primary_rank, Rank.EIGHT)
        self.assertEqual(hand.wild_assignments[0][0], "D1-H-2")

    def test_red_heart_level_card_can_complete_declared_hand_shapes(self) -> None:
        cases = [
            (
                ("D1-S-8", "D2-S-8", "D1-H-2"),
                HandType.THREE_OF_A_KIND,
                Rank.EIGHT,
            ),
            (
                ("D1-S-K", "D2-S-K", "D1-H-2", "D1-S-6", "D2-S-6"),
                HandType.FULL_HOUSE,
                Rank.KING,
            ),
            (
                ("D1-S-3", "D1-H-2", "D1-C-5", "D1-D-6", "D2-S-7"),
                HandType.STRAIGHT,
                Rank.SEVEN,
            ),
            (
                ("D1-S-7", "D1-S-8", "D1-S-9", "D1-S-J", "D1-H-2"),
                HandType.STRAIGHT_FLUSH,
                Rank.JACK,
            ),
            (
                ("D1-S-6", "D2-S-6", "D1-C-6", "D1-H-2"),
                HandType.BOMB,
                Rank.SIX,
            ),
            (
                ("D1-S-9", "D1-H-9", "D1-D-9", "D1-C-9", "D1-H-2", "D2-H-2"),
                HandType.BOMB,
                Rank.NINE,
            ),
            (
                ("D1-S-3", "D2-S-3", "D1-S-4", "D1-H-2", "D1-S-5", "D2-S-5"),
                HandType.THREE_PAIR_RUN,
                Rank.FIVE,
            ),
            (
                ("D1-S-6", "D1-D-6", "D1-H-2", "D1-S-7", "D1-D-7", "D1-H-7"),
                HandType.TRIPLE_RUN,
                Rank.SEVEN,
            ),
        ]

        for card_ids, expected_type, expected_rank in cases:
            with self.subTest(expected_type=expected_type):
                hand = parse_hand(cards(*card_ids), expected_type.value, level=Rank.TWO)

                self.assertEqual(hand.type, expected_type)
                self.assertEqual(hand.primary_rank, expected_rank)
                self.assertTrue(hand.wild_assignments)

    def test_two_red_heart_level_cards_can_complete_declared_hand_shapes(self) -> None:
        cases = [
            (
                ("D1-S-8", "D1-H-2", "D2-H-2"),
                HandType.THREE_OF_A_KIND,
                Rank.EIGHT,
            ),
            (
                ("D1-S-K", "D1-H-2", "D2-H-2", "D1-S-6", "D2-S-6"),
                HandType.FULL_HOUSE,
                Rank.KING,
            ),
            (
                ("D1-S-3", "D1-H-2", "D2-H-2", "D1-D-6", "D2-S-7"),
                HandType.STRAIGHT,
                Rank.SEVEN,
            ),
            (
                ("D1-S-7", "D1-S-8", "D1-S-10", "D1-H-2", "D2-H-2"),
                HandType.STRAIGHT_FLUSH,
                Rank.JACK,
            ),
            (
                ("D1-S-6", "D2-S-6", "D1-H-2", "D2-H-2"),
                HandType.BOMB,
                Rank.SIX,
            ),
            (
                ("D1-S-3", "D2-S-3", "D1-S-4", "D2-S-4", "D1-H-2", "D2-H-2"),
                HandType.THREE_PAIR_RUN,
                Rank.FIVE,
            ),
            (
                ("D1-S-6", "D1-D-6", "D1-H-2", "D1-S-7", "D1-D-7", "D2-H-2"),
                HandType.TRIPLE_RUN,
                Rank.SEVEN,
            ),
        ]

        for card_ids, expected_type, expected_rank in cases:
            with self.subTest(expected_type=expected_type):
                hand = parse_hand(cards(*card_ids), expected_type.value, level=Rank.TWO)

                self.assertEqual(hand.type, expected_type)
                self.assertEqual(hand.primary_rank, expected_rank)
                self.assertEqual(len(hand.wild_assignments), 2)

    def test_two_red_heart_level_cards_can_be_played_as_level_pair(self) -> None:
        hand = parse_hand(cards("D1-H-2", "D2-H-2"), "pair", level=Rank.TWO)

        self.assertEqual(hand.type, HandType.PAIR)
        self.assertEqual(hand.primary_rank, Rank.TWO)
        self.assertEqual(hand.wild_assignments, ())

    def test_equivalent_wildcard_assignments_do_not_require_declaration(self) -> None:
        hand = parse_hand(
            cards("D1-H-2", "D1-H-4", "D1-D-4", "D1-C-4"),
            level=Rank.TWO,
        )

        self.assertEqual(hand.type, HandType.BOMB)
        self.assertEqual(hand.primary_rank, Rank.FOUR)

    def test_wildcard_straight_flush_still_requires_declaration(self) -> None:
        same_suit_straight = cards("D1-S-7", "D1-S-8", "D1-S-9", "D1-S-J", "D1-H-2")

        with self.assertRaises(ValueError):
            parse_hand(same_suit_straight, level=Rank.TWO)

        self.assertEqual(parse_hand(same_suit_straight, "straight", level=Rank.TWO).type, HandType.STRAIGHT)
        self.assertEqual(
            parse_hand(same_suit_straight, "straight_flush", level=Rank.TWO).type,
            HandType.STRAIGHT_FLUSH,
        )

    def test_red_heart_level_card_cannot_represent_joker(self) -> None:
        with self.assertRaises(ValueError):
            parse_hand(cards("D1-SJ", "D2-SJ", "D1-BJ", "D1-H-2"), "four_jokers", level=Rank.TWO)

    def test_non_level_heart_card_is_not_wild(self) -> None:
        with self.assertRaises(ValueError):
            parse_hand(cards("D1-S-8", "D1-H-3"), "pair", level=Rank.TWO)


if __name__ == "__main__":
    unittest.main()
