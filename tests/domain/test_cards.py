from __future__ import annotations

import unittest

from server.domain.cards import CARD_BY_ID, DECK, Rank, Suit, deal_cards, is_red_heart_level_card, resolve_cards


class CardTests(unittest.TestCase):
    def test_deck_has_108_unique_cards(self) -> None:
        self.assertEqual(len(DECK), 108)
        self.assertEqual(len({card.id for card in DECK}), 108)
        self.assertEqual(len(CARD_BY_ID), 108)

    def test_deal_gives_four_27_card_hands_without_loss(self) -> None:
        hands = deal_cards("fixed-seed")

        self.assertEqual(len(hands), 4)
        self.assertEqual([len(hand) for hand in hands], [27, 27, 27, 27])
        dealt = [card_id for hand in hands for card_id in hand]
        self.assertEqual(len(dealt), 108)
        self.assertEqual(set(dealt), set(CARD_BY_ID))

    def test_resolve_rejects_unknown_card_id(self) -> None:
        with self.assertRaises(ValueError):
            resolve_cards(["missing"])

    def test_red_heart_level_card_detection(self) -> None:
        card = CARD_BY_ID["D1-H-2"]

        self.assertTrue(is_red_heart_level_card(card, Rank.TWO))
        self.assertFalse(is_red_heart_level_card(card, Rank.THREE))
        self.assertEqual(card.suit, Suit.HEART)


if __name__ == "__main__":
    unittest.main()
