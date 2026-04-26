from __future__ import annotations

from dataclasses import dataclass

from guandan.domain.cards import Rank, STANDARD_RANKS
from guandan.domain.hand_types import HandType, PlayedHand


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
    return ctx.rank_value(candidate.primary_rank) > ctx.rank_value(current.primary_rank)


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
        return ctx.rank_value(candidate.primary_rank) > ctx.rank_value(current.primary_rank)

    if candidate.type == HandType.BOMB and current.type == HandType.BOMB:
        if candidate.length != current.length:
            return candidate.length > current.length
        return ctx.rank_value(candidate.primary_rank) > ctx.rank_value(current.primary_rank)

    return False
