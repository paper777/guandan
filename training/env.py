from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from server.domain.cards import CARD_BY_ID, Rank, is_red_heart_level_card
from server.domain.commands import Command, JoinTable, Ready, StartMatch
from server.domain.comparator import RankContext
from server.domain.controllers import ControllerCapability, ControllerKind, ControllerRef, PlayerKind, PlayerRef
from server.domain.events import CommandRejected, Event
from server.domain.hand_types import SEQUENCE_RANKS, HandType
from server.domain.legal_actions import ActionCandidate, CommandAction, legal_actions_for_state
from server.domain.reducer import reduce_command
from server.domain.seats import SEATS, Seat, Team, team_for_seat
from server.domain.state import DealResult, MatchPhase, MatchState
from server.services.snapshots import SeatSnapshot, seat_snapshot


HAND_STRENGTH_REWARD_WEIGHT = 0.35
MIN_HAND_STRENGTH_REWARD_MULTIPLIER = 0.65
MAX_HAND_STRENGTH_REWARD_MULTIPLIER = 1.35


@dataclass(frozen=True, slots=True)
class EnvStep:
    state: MatchState
    events: tuple[Event, ...]
    rewards: dict[Seat, float]
    rejection: CommandRejected | None = None

    @property
    def terminated(self) -> bool:
        return self.state.phase == MatchPhase.MATCH_COMPLETE

    @property
    def deal_complete(self) -> bool:
        return self.state.phase in {MatchPhase.DEAL_COMPLETE, MatchPhase.MATCH_COMPLETE}


@dataclass(frozen=True, slots=True)
class InitialHandProfile:
    control_score: float
    regularity_score: float
    estimated_turns: int

    @property
    def strength_score(self) -> float:
        return _clamp(0.65 * self.control_score + 0.35 * self.regularity_score, 0.0, 1.0)


class GuandanTrainingEnv:
    """In-process training environment backed by the authoritative reducer."""

    def __init__(self, *, table_id: str = "training-table", reward_shaping_weight: float = 0.0) -> None:
        self.table_id = table_id
        self.state = MatchState(table_id=table_id)
        self.controller_ids = {seat: f"training-controller-{seat.value}" for seat in SEATS}
        self._deal_start_hands: dict[Seat, tuple[str, ...]] = {}
        self.reward_shaping_weight = reward_shaping_weight

    def reset(self, seed: str | int | bytes | None = None) -> MatchState:
        self.state = MatchState(table_id=self.table_id)
        self._deal_start_hands = {}
        for seat in SEATS:
            self._apply(JoinTable(_training_player(seat), _training_controller(seat, self.controller_ids[seat]), seat))
        for seat in SEATS:
            self._apply(Ready(self.controller_ids[seat], seat))
        self._apply(StartMatch(seed=seed))
        self._record_deal_start_hands()
        return self.state

    def observe(self, seat: Seat) -> SeatSnapshot:
        return seat_snapshot(self.state, seat, self.controller_ids[seat])

    def legal_actions(self, seat: Seat) -> tuple[ActionCandidate, ...]:
        return legal_actions_for_state(self.state, seat)

    def current_actor(self) -> Seat | None:
        if self.state.deal is None:
            return None
        if self.state.phase not in {MatchPhase.PLAYING, MatchPhase.TRIBUTE}:
            return None
        return self.state.deal.turn

    def step(self, seat: Seat, action: ActionCandidate | CommandAction) -> EnvStep:
        command = action.to_command(self.controller_ids[seat], seat) if isinstance(action, ActionCandidate) else action
        shaping = (
            _action_shaping_rewards(self.state, seat, action, self.reward_shaping_weight)
            if isinstance(action, ActionCandidate)
            else {seat: 0.0 for seat in SEATS}
        )
        previous_result = self.state.last_deal_result
        deal_level = self.state.current_level
        result = reduce_command(self.state, command)
        self.state = result.state
        rewards = _rewards_for_transition(
            previous_result,
            self.state.last_deal_result,
            initial_hands=self._deal_start_hands,
            level=deal_level,
        )
        if result.rejection is None and any(value != 0.0 for value in shaping.values()):
            rewards = {seat_item: rewards[seat_item] + shaping[seat_item] for seat_item in SEATS}
        return EnvStep(state=self.state, events=result.events, rewards=rewards, rejection=result.rejection)

    def start_next_deal(self, seed: str | int | bytes | None = None) -> EnvStep:
        previous_result = self.state.last_deal_result
        deal_level = self.state.current_level
        result = reduce_command(self.state, StartMatch(seed=seed))
        self.state = result.state
        rewards = _rewards_for_transition(
            previous_result,
            self.state.last_deal_result,
            initial_hands=self._deal_start_hands,
            level=deal_level,
        )
        if result.rejection is None:
            self._record_deal_start_hands()
        return EnvStep(state=self.state, events=result.events, rewards=rewards, rejection=result.rejection)

    def terminal_result(self) -> DealResult | None:
        return self.state.last_deal_result if self.state.phase == MatchPhase.MATCH_COMPLETE else None

    def _apply(self, command: Command) -> tuple[Event, ...]:
        result = reduce_command(self.state, command)
        if result.rejection is not None:
            raise RuntimeError(f"training env setup command rejected: {result.rejection.code}: {result.rejection.message}")
        self.state = result.state
        return result.events

    def _record_deal_start_hands(self) -> None:
        self._deal_start_hands = dict(self.state.deal.hands) if self.state.deal is not None else {}


def _training_player(seat: Seat) -> PlayerRef:
    return PlayerRef(
        id=f"training-player-{seat.value}",
        display_name=f"Training {seat.value}",
        kind=PlayerKind.BOT,
    )


def _training_controller(seat: Seat, controller_id: str) -> ControllerRef:
    return ControllerRef(
        id=controller_id,
        kind=ControllerKind.LOCAL_BOT,
        seat=seat,
        player_id=f"training-player-{seat.value}",
        capabilities=frozenset(
            {
                ControllerCapability.PLAY,
                ControllerCapability.OBSERVE_PUBLIC,
                ControllerCapability.OBSERVE_PRIVATE,
                ControllerCapability.AUTO_READY,
                ControllerCapability.DEBUG_FULL_STATE,
            }
        ),
    )


def _rewards_for_transition(
    previous: DealResult | None,
    current: DealResult | None,
    *,
    initial_hands: dict[Seat, tuple[str, ...]] | None = None,
    level: Rank = Rank.TWO,
) -> dict[Seat, float]:
    rewards = {seat: 0.0 for seat in SEATS}
    if current is None or current == previous:
        return rewards
    reward_multiplier = _reward_multiplier_for_result(current, initial_hands or {}, level)
    deal_reward = (current.advance_count / 3.0) * reward_multiplier
    _add_team_reward(rewards, current.winning_team, deal_reward)
    _add_team_reward(rewards, _opposing_team(current.winning_team), -deal_reward)
    if current.match_complete:
        _add_team_reward(rewards, current.winning_team, reward_multiplier)
        _add_team_reward(rewards, _opposing_team(current.winning_team), -reward_multiplier)
    return rewards


def _action_shaping_rewards(
    state: MatchState,
    seat: Seat,
    action: ActionCandidate,
    weight: float,
) -> dict[Seat, float]:
    rewards = {seat_item: 0.0 for seat_item in SEATS}
    if weight <= 0.0 or state.deal is None:
        return rewards
    value = 0.0
    if action.kind == "play_cards":
        hand = state.deal.hand_for(seat)
        finishes = len(action.card_ids) == len(hand)
        opponent_danger = _has_dangerous_opponent(state, seat)
        last_play_seat = state.deal.current_trick.last_play_seat
        if finishes:
            value += 0.08
        if last_play_seat is not None and team_for_seat(last_play_seat) != team_for_seat(seat) and opponent_danger:
            value += 0.05
        if (
            action.hand_type in {HandType.BOMB, HandType.STRAIGHT_FLUSH, HandType.FOUR_JOKERS}
            and not finishes
            and not opponent_danger
        ):
            value -= 0.04
    elif action.kind == "pass":
        last_play_seat = state.deal.current_trick.last_play_seat
        if last_play_seat is not None and team_for_seat(last_play_seat) == team_for_seat(seat):
            value += 0.02
        elif last_play_seat is not None and _has_dangerous_opponent(state, seat):
            value -= 0.04
    rewards[seat] = value * weight
    return rewards


def _has_dangerous_opponent(state: MatchState, seat: Seat) -> bool:
    if state.deal is None:
        return False
    own_team = team_for_seat(seat)
    return any(
        team_for_seat(other) != own_team and 0 < len(state.deal.hand_for(other)) <= 2
        for other in SEATS
    )


def _reward_multiplier_for_result(result: DealResult, initial_hands: dict[Seat, tuple[str, ...]], level: Rank) -> float:
    if any(seat not in initial_hands for seat in SEATS):
        return 1.0
    winning_strength = _team_initial_strength(result.winning_team, initial_hands, level)
    losing_strength = _team_initial_strength(_opposing_team(result.winning_team), initial_hands, level)
    relative_advantage = winning_strength - losing_strength
    return _clamp(
        1.0 - HAND_STRENGTH_REWARD_WEIGHT * relative_advantage,
        MIN_HAND_STRENGTH_REWARD_MULTIPLIER,
        MAX_HAND_STRENGTH_REWARD_MULTIPLIER,
    )


def _team_initial_strength(team: Team, initial_hands: dict[Seat, tuple[str, ...]], level: Rank) -> float:
    profiles = [
        _initial_hand_profile(initial_hands[seat], level)
        for seat in SEATS
        if team_for_seat(seat) == team
    ]
    return sum(profile.strength_score for profile in profiles) / len(profiles)


def _initial_hand_profile(hand: tuple[str, ...], level: Rank) -> InitialHandProfile:
    estimated_turns = _estimated_turn_count(hand)
    return InitialHandProfile(
        control_score=_control_score(hand, level),
        regularity_score=_regularity_score(estimated_turns),
        estimated_turns=estimated_turns,
    )


def _control_score(hand: tuple[str, ...], level: Rank) -> float:
    cards = [CARD_BY_ID[card_id] for card_id in hand]
    ctx = RankContext(level)
    joker_score = sum(1.2 if card.rank == Rank.BIG_JOKER else 1.0 for card in cards if card.is_joker)
    high_card_score = sum(max(ctx.rank_value(card.rank) - 9, 0) / 6.0 for card in cards if not card.is_joker)
    bomb_score = _bomb_control_score(hand, level)
    return _clamp((0.9 * joker_score + 0.35 * high_card_score + 1.3 * bomb_score) / 8.0, 0.0, 1.0)


def _bomb_control_score(hand: tuple[str, ...], level: Rank) -> float:
    cards = [CARD_BY_ID[card_id] for card_id in hand]
    ranks = Counter(card.rank for card in cards)
    wild_count = sum(1 for card in cards if is_red_heart_level_card(card, level))
    score = 0.0
    if ranks[Rank.SMALL_JOKER] >= 2 and ranks[Rank.BIG_JOKER] >= 2:
        score += 1.5
    for rank in Rank:
        if rank in {Rank.SMALL_JOKER, Rank.BIG_JOKER}:
            continue
        usable = ranks[rank] if rank == level else ranks[rank] + wild_count
        if usable >= 4:
            score += 1.0 + 0.2 * (usable - 4)
    return score


def _regularity_score(estimated_turns: int) -> float:
    return _clamp((18.0 - estimated_turns) / 10.0, 0.0, 1.0)


def _estimated_turn_count(hand: tuple[str, ...]) -> int:
    ranks = Counter(CARD_BY_ID[card_id].rank for card_id in hand)
    turns = 0

    if ranks[Rank.SMALL_JOKER] >= 2 and ranks[Rank.BIG_JOKER] >= 2:
        ranks[Rank.SMALL_JOKER] -= 2
        ranks[Rank.BIG_JOKER] -= 2
        turns += 1

    for rank, count in tuple(ranks.items()):
        if rank not in {Rank.SMALL_JOKER, Rank.BIG_JOKER} and count >= 4:
            ranks[rank] = 0
            turns += 1

    turns += _remove_rank_runs(ranks, per_rank=3, run_length=2)
    turns += _remove_rank_runs(ranks, per_rank=2, run_length=3)
    turns += _remove_rank_runs(ranks, per_rank=1, run_length=5)

    while True:
        triple_rank = next((rank for rank, count in ranks.items() if count >= 3), None)
        pair_rank = next((rank for rank, count in ranks.items() if rank != triple_rank and count >= 2), None)
        if triple_rank is None or pair_rank is None:
            break
        ranks[triple_rank] -= 3
        ranks[pair_rank] -= 2
        turns += 1

    for rank, count in tuple(ranks.items()):
        if rank in {Rank.SMALL_JOKER, Rank.BIG_JOKER}:
            continue
        while ranks[rank] >= 3:
            ranks[rank] -= 3
            turns += 1
        while ranks[rank] >= 2:
            ranks[rank] -= 2
            turns += 1

    return turns + sum(count for count in ranks.values() if count > 0)


def _remove_rank_runs(ranks: Counter[Rank], *, per_rank: int, run_length: int) -> int:
    removed = 0
    while True:
        window = _first_rank_run(ranks, per_rank=per_rank, run_length=run_length)
        if window is None:
            return removed
        for rank in window:
            ranks[rank] -= per_rank
        removed += 1


def _first_rank_run(ranks: Counter[Rank], *, per_rank: int, run_length: int) -> tuple[Rank, ...] | None:
    for start in range(0, len(SEQUENCE_RANKS) - run_length + 1):
        window = SEQUENCE_RANKS[start : start + run_length]
        if len(set(window)) == run_length and all(ranks[rank] >= per_rank for rank in window):
            return window
    return None


def _add_team_reward(rewards: dict[Seat, float], team: Team, value: float) -> None:
    for seat in SEATS:
        if team_for_seat(seat) == team:
            rewards[seat] += value


def _opposing_team(team: Team) -> Team:
    return Team.SOUTH_NORTH if team == Team.EAST_WEST else Team.EAST_WEST


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
