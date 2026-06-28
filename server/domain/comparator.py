from __future__ import annotations

from dataclasses import dataclass

from server.domain.cards import Rank, STANDARD_RANKS
from server.domain.hand_types import HandType, PlayedHand


@dataclass(frozen=True, slots=True)
class RankContext:
    level: Rank

    def rank_value(self, rank: Rank) -> int:
        if rank == Rank.BIG_JOKER:
            return 15
        if rank == Rank.SMALL_JOKER:
            return 14
        if rank == self.level:
            return 13
        base_order = {rank_value: index for index, rank_value in enumerate(STANDARD_RANKS)}
        return base_order[rank]

    def sequence_value(self, rank: Rank) -> int:
        return STANDARD_RANKS.index(rank)


def can_beat(candidate: PlayedHand, current: PlayedHand | None, level: Rank) -> bool:
    if current is None:
        return True
    ctx = RankContext(level)

    if candidate.type == HandType.FOUR_JOKERS:
        return current.type != HandType.FOUR_JOKERS
    if current.type == HandType.FOUR_JOKERS:
        return False

    if candidate.is_bomb_like or current.is_bomb_like:
        return _bomb_like_can_beat(candidate, current, ctx)

    if candidate.type != current.type or candidate.length != current.length:
        return False
    return _primary_rank_value(candidate, ctx) > _primary_rank_value(current, ctx)


def _bomb_like_can_beat(candidate: PlayedHand, current: PlayedHand, ctx: RankContext) -> bool:
    if not candidate.is_bomb_like:
        return False
    if not current.is_bomb_like:
        return True

    if candidate.type == HandType.STRAIGHT_FLUSH and current.type == HandType.BOMB:
        if current.length <= 5:
            return True
        return False

    if candidate.type == HandType.BOMB and current.type == HandType.STRAIGHT_FLUSH:
        return candidate.length >= 6

    if candidate.type == HandType.STRAIGHT_FLUSH and current.type == HandType.STRAIGHT_FLUSH:
        return _primary_rank_value(candidate, ctx) > _primary_rank_value(current, ctx)

    if candidate.type == HandType.BOMB and current.type == HandType.BOMB:
        if candidate.length != current.length:
            return candidate.length > current.length
        return ctx.rank_value(candidate.primary_rank) > ctx.rank_value(current.primary_rank)

    return False


def _primary_rank_value(hand: PlayedHand, ctx: RankContext) -> int:
    if hand.type in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH, HandType.THREE_PAIR_RUN, HandType.TRIPLE_RUN}:
        return ctx.sequence_value(hand.primary_rank)
    return ctx.rank_value(hand.primary_rank)
