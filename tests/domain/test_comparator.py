from __future__ import annotations

import unittest

from server.domain.cards import CARD_BY_ID, Rank
from server.domain.comparator import RankContext, can_beat
from server.domain.hand_types import HandType, parse_hand


def hand(*ids: str, declared_type: str | None = None, level: Rank = Rank.TWO):
    return parse_hand(tuple(CARD_BY_ID[card_id] for card_id in ids), declared_type, level=level)


class ComparatorTests(unittest.TestCase):
    def test_level_rank_beats_ace_but_not_jokers(self) -> None:
        ctx = RankContext(Rank.TWO)

        self.assertGreater(ctx.rank_value(Rank.TWO), ctx.rank_value(Rank.ACE))
        self.assertGreater(ctx.rank_value(Rank.SMALL_JOKER), ctx.rank_value(Rank.TWO))
        self.assertGreater(ctx.rank_value(Rank.BIG_JOKER), ctx.rank_value(Rank.SMALL_JOKER))

    def test_same_type_must_be_higher(self) -> None:
        low = hand("D1-S-3")
        high = hand("D1-S-4")

        self.assertTrue(can_beat(high, low, Rank.TWO))
        self.assertFalse(can_beat(low, high, Rank.TWO))

    def test_connected_runs_compare_by_sequence_rank_not_level_rank(self) -> None:
        low_triple_run = hand(
            "D1-S-5",
            "D2-S-5",
            "D1-H-5",
            "D1-S-6",
            "D2-S-6",
            "D1-H-6",
            declared_type=HandType.TRIPLE_RUN.value,
            level=Rank.SIX,
        )
        high_triple_run = hand(
            "D1-S-7",
            "D2-S-7",
            "D1-H-7",
            "D1-S-8",
            "D2-S-8",
            "D1-H-8",
            declared_type=HandType.TRIPLE_RUN.value,
            level=Rank.SIX,
        )
        low_pair_run = hand(
            "D1-S-4",
            "D2-S-4",
            "D1-S-5",
            "D2-S-5",
            "D1-S-6",
            "D1-H-6",
            declared_type=HandType.THREE_PAIR_RUN.value,
            level=Rank.SIX,
        )
        high_pair_run = hand(
            "D1-S-7",
            "D2-S-7",
            "D1-S-8",
            "D2-S-8",
            "D1-S-9",
            "D2-S-9",
            declared_type=HandType.THREE_PAIR_RUN.value,
            level=Rank.SIX,
        )

        self.assertEqual(low_triple_run.primary_rank, Rank.SIX)
        self.assertEqual(low_pair_run.primary_rank, Rank.SIX)
        self.assertFalse(can_beat(low_triple_run, high_triple_run, Rank.SIX))
        self.assertTrue(can_beat(high_triple_run, low_triple_run, Rank.SIX))
        self.assertFalse(can_beat(low_pair_run, high_pair_run, Rank.SIX))
        self.assertTrue(can_beat(high_pair_run, low_pair_run, Rank.SIX))

    def test_bomb_beats_ordinary_hand(self) -> None:
        single = hand("D1-S-A")
        bomb = hand("D1-S-3", "D2-S-3", "D1-H-3", "D2-H-3")

        self.assertTrue(can_beat(bomb, single, Rank.TWO))
        self.assertFalse(can_beat(single, bomb, Rank.TWO))

    def test_bomb_hierarchy(self) -> None:
        four_aces = hand("D1-S-A", "D2-S-A", "D1-H-A", "D2-H-A")
        five_threes = hand("D1-S-3", "D2-S-3", "D1-H-3", "D2-H-3", "D1-C-3")
        straight_flush = hand("D1-S-3", "D1-S-4", "D1-S-5", "D1-S-6", "D1-S-7", declared_type="straight_flush")
        six_threes = hand("D1-S-3", "D2-S-3", "D1-H-3", "D2-H-3", "D1-C-3", "D2-C-3")
        four_jokers = hand("D1-SJ", "D2-SJ", "D1-BJ", "D2-BJ")

        self.assertTrue(can_beat(five_threes, four_aces, Rank.TWO))
        self.assertTrue(can_beat(straight_flush, five_threes, Rank.TWO))
        self.assertTrue(can_beat(six_threes, straight_flush, Rank.TWO))
        self.assertTrue(can_beat(four_jokers, six_threes, Rank.TWO))


if __name__ == "__main__":
    unittest.main()
