from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Literal

from server.domain.cards import CARD_BY_ID, STANDARD_RANKS, Rank, is_red_heart_level_card, resolve_cards
from server.domain.commands import Pass, PlayCards, ReturnTribute, SubmitTribute
from server.domain.comparator import RankContext, can_beat
from server.domain.hand_types import SEQUENCE_RANKS, HandType, PlayedHand, parse_hand
from server.domain.seats import Seat, team_for_seat
from server.domain.state import MatchPhase, MatchState, TributeObligation


ActionKind = Literal["play_cards", "pass", "submit_tribute", "return_tribute"]
CommandAction = PlayCards | Pass | SubmitTribute | ReturnTribute


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    kind: ActionKind
    card_ids: tuple[str, ...] = ()
    declared_type: str | None = None
    hand_type: HandType | None = None
    primary_rank: Rank | None = None
    length: int = 0

    def to_command(self, controller_id: str, seat: Seat) -> CommandAction:
        if self.kind == "pass":
            return Pass(controller_id, seat)
        if self.kind == "submit_tribute":
            return SubmitTribute(controller_id, seat, self._single_card_id())
        if self.kind == "return_tribute":
            return ReturnTribute(controller_id, seat, self._single_card_id())
        return PlayCards(controller_id, seat, self.card_ids, declared_type=self.declared_type)

    def to_payload(self) -> dict[str, object]:
        if self.kind == "pass":
            return {"type": "pass"}
        if self.kind in {"submit_tribute", "return_tribute"}:
            return {"type": self.kind, "card_id": self._single_card_id()}
        payload: dict[str, object] = {"type": "play_cards", "card_ids": list(self.card_ids)}
        if self.declared_type is not None:
            payload["declared_type"] = self.declared_type
        return payload

    def _single_card_id(self) -> str:
        if len(self.card_ids) != 1:
            raise ValueError(f"{self.kind} action requires exactly one card")
        return self.card_ids[0]


def legal_actions_for_state(state: MatchState, seat: Seat) -> tuple[ActionCandidate, ...]:
    if state.deal is None:
        return ()
    if state.phase == MatchPhase.PLAYING and state.deal.turn == seat:
        return _playing_actions(state, seat)
    if state.phase == MatchPhase.TRIBUTE and state.deal.tribute is not None and state.deal.turn == seat:
        return _tribute_actions(state, seat)
    return ()


def _playing_actions(state: MatchState, seat: Seat) -> tuple[ActionCandidate, ...]:
    assert state.deal is not None
    hand = state.deal.hand_for(seat)
    current = state.deal.current_trick.last_play
    actions: list[ActionCandidate] = []
    if current is not None:
        actions.append(ActionCandidate(kind="pass"))
    actions.extend(_play_actions(hand, state.current_level, current))
    return _sort_actions(actions)


def _tribute_actions(state: MatchState, seat: Seat) -> tuple[ActionCandidate, ...]:
    assert state.deal is not None and state.deal.tribute is not None
    tribute = state.deal.tribute
    hand = state.deal.hand_for(seat)

    obligation = _pending_tribute_obligation(tribute.obligations, seat)
    if obligation is not None:
        return tuple(
            ActionCandidate(kind="submit_tribute", card_ids=(card_id,), length=1)
            for card_id in _highest_eligible_tribute_cards(hand, state.current_level)
        )

    obligation = _pending_return_obligation(tribute.obligations, seat)
    if obligation is not None:
        return tuple(
            ActionCandidate(kind="return_tribute", card_ids=(card_id,), length=1)
            for card_id in _eligible_return_cards(hand, obligation)
        )

    return ()


def _play_actions(
    hand: tuple[str, ...],
    level: Rank,
    current: PlayedHand | None,
) -> tuple[ActionCandidate, ...]:
    candidates: dict[tuple[ActionKind, tuple[str, ...], str | None], ActionCandidate] = {}

    for card_ids, hand_type in _candidate_card_groups(hand, level):
        candidate = _candidate_for_cards(card_ids, hand_type, level, current)
        if candidate is not None:
            candidates[(candidate.kind, candidate.card_ids, candidate.declared_type)] = candidate

    return _sort_actions(candidates.values())


def _candidate_card_groups(hand: tuple[str, ...], level: Rank) -> tuple[tuple[tuple[str, ...], HandType], ...]:
    groups: list[tuple[tuple[str, ...], HandType]] = []
    by_rank = _cards_by_rank(hand)
    wilds = tuple(card_id for card_id in hand if is_red_heart_level_card(CARD_BY_ID[card_id], level))

    groups.extend((card_ids, HandType.SINGLE) for card_ids in _single_groups(hand))
    groups.extend((card_ids, HandType.PAIR) for card_ids in _same_rank_groups(by_rank, wilds, level, 2))
    groups.extend((card_ids, HandType.THREE_OF_A_KIND) for card_ids in _same_rank_groups(by_rank, wilds, level, 3))
    groups.extend((card_ids, HandType.BOMB) for card_ids in _bomb_groups(by_rank, wilds, level))
    groups.extend((card_ids, HandType.FOUR_JOKERS) for card_ids in _four_joker_groups(by_rank))
    groups.extend((card_ids, HandType.FULL_HOUSE) for card_ids in _full_house_groups(by_rank, wilds, level))
    groups.extend((card_ids, HandType.STRAIGHT) for card_ids in _straight_groups(by_rank, wilds, level))
    groups.extend((card_ids, HandType.STRAIGHT_FLUSH) for card_ids in _straight_flush_groups(hand, wilds))
    groups.extend((card_ids, HandType.THREE_PAIR_RUN) for card_ids in _same_count_run_groups(by_rank, wilds, level, 2, 3))
    groups.extend((card_ids, HandType.TRIPLE_RUN) for card_ids in _same_count_run_groups(by_rank, wilds, level, 3, 2))

    deduped: dict[tuple[tuple[str, ...], HandType], tuple[tuple[str, ...], HandType]] = {}
    for card_ids, hand_type in groups:
        deduped[(card_ids, hand_type)] = (card_ids, hand_type)
    return tuple(deduped.values())


def _single_groups(hand: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple((card_id,) for card_id in hand)


def _cards_by_rank(hand: tuple[str, ...]) -> dict[Rank, tuple[str, ...]]:
    groups: dict[Rank, list[str]] = {rank: [] for rank in Rank}
    for card_id in hand:
        groups[CARD_BY_ID[card_id].rank].append(card_id)
    return {rank: tuple(card_ids) for rank, card_ids in groups.items()}


def _same_rank_groups(
    by_rank: dict[Rank, tuple[str, ...]],
    wilds: tuple[str, ...],
    level: Rank,
    size: int,
) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    for rank in Rank:
        if rank in {Rank.SMALL_JOKER, Rank.BIG_JOKER}:
            pool = by_rank[rank]
        elif rank == level:
            pool = by_rank[rank]
        else:
            pool = _unique_cards((*by_rank[rank], *wilds))
        if len(pool) >= size:
            groups.extend(tuple(combo) for combo in combinations(pool, size))
    return _dedupe_card_groups(groups)


def _bomb_groups(
    by_rank: dict[Rank, tuple[str, ...]],
    wilds: tuple[str, ...],
    level: Rank,
) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    for rank in STANDARD_RANKS:
        pool = by_rank[rank] if rank == level else _unique_cards((*by_rank[rank], *wilds))
        for size in range(4, len(pool) + 1):
            groups.extend(tuple(combo) for combo in combinations(pool, size))
    return _dedupe_card_groups(groups)


def _four_joker_groups(by_rank: dict[Rank, tuple[str, ...]]) -> tuple[tuple[str, ...], ...]:
    jokers = (*by_rank[Rank.SMALL_JOKER], *by_rank[Rank.BIG_JOKER])
    if len(jokers) != 4:
        return ()
    return (jokers,)


def _full_house_groups(
    by_rank: dict[Rank, tuple[str, ...]],
    wilds: tuple[str, ...],
    level: Rank,
) -> tuple[tuple[str, ...], ...]:
    triples_by_rank = {rank: _rank_group_options(by_rank, wilds, level, rank, 3) for rank in STANDARD_RANKS}
    pairs_by_rank = {rank: _rank_group_options(by_rank, wilds, level, rank, 2) for rank in Rank}
    groups: list[tuple[str, ...]] = []
    for triple_rank, triples in triples_by_rank.items():
        for pair_rank, pairs in pairs_by_rank.items():
            if pair_rank == triple_rank:
                continue
            for triple in triples:
                triple_set = set(triple)
                for pair in pairs:
                    if triple_set.isdisjoint(pair):
                        groups.append((*triple, *pair))
    return _dedupe_card_groups(groups)


def _straight_groups(
    by_rank: dict[Rank, tuple[str, ...]],
    wilds: tuple[str, ...],
    level: Rank,
) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    for window in _sequence_windows(5):
        options = [_straight_rank_options(by_rank, wilds, level, rank) for rank in window]
        groups.extend(_one_from_each(options))
    return _dedupe_card_groups(groups)


def _straight_flush_groups(
    hand: tuple[str, ...],
    wilds: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    for suit in ("S", "H", "D", "C"):
        by_rank = _cards_by_rank_for_suit(hand, suit)
        for window in _sequence_windows(5):
            options = [_straight_flush_rank_options(by_rank, wilds, rank) for rank in window]
            groups.extend(_one_from_each(options))
    return _dedupe_card_groups(groups)


def _same_count_run_groups(
    by_rank: dict[Rank, tuple[str, ...]],
    wilds: tuple[str, ...],
    level: Rank,
    count: int,
    run_length: int,
) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    for window in _sequence_windows(run_length):
        options = [_rank_group_options(by_rank, wilds, level, rank, count) for rank in window]
        for selected in _one_group_from_each(options):
            card_ids: list[str] = []
            for group in selected:
                card_ids.extend(group)
            groups.append(tuple(card_ids))
    return _dedupe_card_groups(groups)


def _rank_group_options(
    by_rank: dict[Rank, tuple[str, ...]],
    wilds: tuple[str, ...],
    level: Rank,
    rank: Rank,
    size: int,
) -> tuple[tuple[str, ...], ...]:
    if rank in {Rank.SMALL_JOKER, Rank.BIG_JOKER}:
        pool = by_rank[rank]
    elif rank == level:
        pool = by_rank[rank]
    else:
        pool = _unique_cards((*by_rank[rank], *wilds))
    if len(pool) < size:
        return ()
    return tuple(tuple(combo) for combo in combinations(pool, size))


def _straight_rank_options(
    by_rank: dict[Rank, tuple[str, ...]],
    wilds: tuple[str, ...],
    level: Rank,
    rank: Rank,
) -> tuple[str, ...]:
    return by_rank[rank] if rank == level else _unique_cards((*by_rank[rank], *wilds))


def _straight_flush_rank_options(
    by_rank: dict[Rank, tuple[str, ...]],
    wilds: tuple[str, ...],
    rank: Rank,
) -> tuple[str, ...]:
    return _unique_cards((*by_rank[rank], *wilds))


def _cards_by_rank_for_suit(hand: tuple[str, ...], suit: str) -> dict[Rank, tuple[str, ...]]:
    groups: dict[Rank, list[str]] = {rank: [] for rank in STANDARD_RANKS}
    for card_id in hand:
        card = CARD_BY_ID[card_id]
        if card.suit == suit and card.rank in STANDARD_RANKS:
            groups[card.rank].append(card_id)
    return {rank: tuple(card_ids) for rank, card_ids in groups.items()}


def _sequence_windows(length: int) -> tuple[tuple[Rank, ...], ...]:
    return tuple(
        window
        for window in (SEQUENCE_RANKS[index : index + length] for index in range(0, len(SEQUENCE_RANKS) - length + 1))
        if len(set(window)) == length
    )


def _one_from_each(options_by_rank: list[tuple[str, ...]]) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []

    def visit(index: int, selected: list[str], used: set[str]) -> None:
        if index == len(options_by_rank):
            groups.append(tuple(selected))
            return
        for card_id in options_by_rank[index]:
            if card_id in used:
                continue
            selected.append(card_id)
            used.add(card_id)
            visit(index + 1, selected, used)
            used.remove(card_id)
            selected.pop()

    visit(0, [], set())
    return tuple(groups)


def _one_group_from_each(options_by_rank: list[tuple[tuple[str, ...], ...]]) -> tuple[tuple[tuple[str, ...], ...], ...]:
    groups: list[tuple[tuple[str, ...], ...]] = []

    def visit(index: int, selected: list[tuple[str, ...]], used: set[str]) -> None:
        if index == len(options_by_rank):
            groups.append(tuple(selected))
            return
        for option in options_by_rank[index]:
            option_set = set(option)
            if not used.isdisjoint(option_set):
                continue
            selected.append(option)
            used.update(option_set)
            visit(index + 1, selected, used)
            used.difference_update(option_set)
            selected.pop()

    visit(0, [], set())
    return tuple(groups)


def _unique_cards(card_ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(card_ids))


def _dedupe_card_groups(groups: Iterable[tuple[str, ...]]) -> tuple[tuple[str, ...], ...]:
    deduped: dict[frozenset[str], tuple[str, ...]] = {}
    for group in groups:
        deduped.setdefault(frozenset(group), group)
    return tuple(deduped.values())


def _candidate_for_cards(
    card_ids: tuple[str, ...],
    declared_type: HandType,
    level: Rank,
    current: PlayedHand | None,
) -> ActionCandidate | None:
    try:
        played = parse_hand(resolve_cards(card_ids), declared_type.value, level=level)
    except ValueError:
        return None
    if not can_beat(played, current, level):
        return None
    return ActionCandidate(
        kind="play_cards",
        card_ids=tuple(card_ids),
        declared_type=played.type.value,
        hand_type=played.type,
        primary_rank=played.primary_rank,
        length=played.length,
    )


def _pending_tribute_obligation(
    obligations: tuple[TributeObligation, ...],
    seat: Seat,
) -> TributeObligation | None:
    for obligation in obligations:
        if obligation.giver == seat and obligation.tribute_card_id is None:
            return obligation
    return None


def _pending_return_obligation(
    obligations: tuple[TributeObligation, ...],
    seat: Seat,
) -> TributeObligation | None:
    for obligation in obligations:
        if obligation.receiver == seat and obligation.tribute_card_id is not None and obligation.return_card_id is None:
            return obligation
    return None


def _highest_eligible_tribute_cards(card_ids: tuple[str, ...], level: Rank) -> tuple[str, ...]:
    ctx = RankContext(level)
    eligible = tuple(
        CARD_BY_ID[card_id] for card_id in card_ids if not is_red_heart_level_card(CARD_BY_ID[card_id], level)
    )
    if not eligible:
        return ()
    max_value = max(ctx.rank_value(card.rank) for card in eligible)
    return tuple(card.id for card in eligible if ctx.rank_value(card.rank) == max_value)


def _eligible_return_cards(card_ids: tuple[str, ...], obligation: TributeObligation) -> tuple[str, ...]:
    if team_for_seat(obligation.giver) != team_for_seat(obligation.receiver):
        return card_ids
    return tuple(card_id for card_id in card_ids if _rank_at_most_ten(card_id))


def _rank_at_most_ten(card_id: str) -> bool:
    return CARD_BY_ID[card_id].rank in {
        Rank.TWO,
        Rank.THREE,
        Rank.FOUR,
        Rank.FIVE,
        Rank.SIX,
        Rank.SEVEN,
        Rank.EIGHT,
        Rank.NINE,
        Rank.TEN,
    }


def _sort_actions(actions: Iterable[ActionCandidate]) -> tuple[ActionCandidate, ...]:
    kind_order = {"pass": 0, "submit_tribute": 1, "return_tribute": 2, "play_cards": 3}
    type_order = {hand_type: index for index, hand_type in enumerate(HandType)}
    rank_order = {rank: RankContext(Rank.TWO).rank_value(rank) for rank in Rank}
    return tuple(
        sorted(
            actions,
            key=lambda action: (
                kind_order[action.kind],
                action.length,
                type_order.get(action.hand_type, -1),
                rank_order.get(action.primary_rank, -1),
                action.card_ids,
                action.declared_type or "",
            ),
        )
    )
