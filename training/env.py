from __future__ import annotations

from dataclasses import dataclass

from server.domain.commands import Command, JoinTable, Ready, StartMatch
from server.domain.controllers import ControllerCapability, ControllerKind, ControllerRef, PlayerKind, PlayerRef
from server.domain.events import CommandRejected, Event
from server.domain.legal_actions import ActionCandidate, CommandAction, legal_actions_for_state
from server.domain.reducer import reduce_command
from server.domain.seats import SEATS, Seat, Team, team_for_seat
from server.domain.state import DealResult, MatchPhase, MatchState
from server.services.snapshots import SeatSnapshot, seat_snapshot


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


class GuandanTrainingEnv:
    """In-process training environment backed by the authoritative reducer."""

    def __init__(self, *, table_id: str = "training-table") -> None:
        self.table_id = table_id
        self.state = MatchState(table_id=table_id)
        self.controller_ids = {seat: f"training-controller-{seat.value}" for seat in SEATS}

    def reset(self, seed: str | int | bytes | None = None) -> MatchState:
        self.state = MatchState(table_id=self.table_id)
        for seat in SEATS:
            self._apply(JoinTable(_training_player(seat), _training_controller(seat, self.controller_ids[seat]), seat))
        for seat in SEATS:
            self._apply(Ready(self.controller_ids[seat], seat))
        self._apply(StartMatch(seed=seed))
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
        previous_result = self.state.last_deal_result
        result = reduce_command(self.state, command)
        self.state = result.state
        rewards = _rewards_for_transition(previous_result, self.state.last_deal_result)
        return EnvStep(state=self.state, events=result.events, rewards=rewards, rejection=result.rejection)

    def start_next_deal(self, seed: str | int | bytes | None = None) -> EnvStep:
        previous_result = self.state.last_deal_result
        result = reduce_command(self.state, StartMatch(seed=seed))
        self.state = result.state
        rewards = _rewards_for_transition(previous_result, self.state.last_deal_result)
        return EnvStep(state=self.state, events=result.events, rewards=rewards, rejection=result.rejection)

    def terminal_result(self) -> DealResult | None:
        return self.state.last_deal_result if self.state.phase == MatchPhase.MATCH_COMPLETE else None

    def _apply(self, command: Command) -> tuple[Event, ...]:
        result = reduce_command(self.state, command)
        if result.rejection is not None:
            raise RuntimeError(f"training env setup command rejected: {result.rejection.code}: {result.rejection.message}")
        self.state = result.state
        return result.events


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


def _rewards_for_transition(previous: DealResult | None, current: DealResult | None) -> dict[Seat, float]:
    rewards = {seat: 0.0 for seat in SEATS}
    if current is None or current == previous:
        return rewards
    deal_reward = current.advance_count / 3.0
    _add_team_reward(rewards, current.winning_team, deal_reward)
    _add_team_reward(rewards, _opposing_team(current.winning_team), -deal_reward)
    if current.match_complete:
        _add_team_reward(rewards, current.winning_team, 1.0)
        _add_team_reward(rewards, _opposing_team(current.winning_team), -1.0)
    return rewards


def _add_team_reward(rewards: dict[Seat, float], team: Team, value: float) -> None:
    for seat in SEATS:
        if team_for_seat(seat) == team:
            rewards[seat] += value


def _opposing_team(team: Team) -> Team:
    return Team.SOUTH_NORTH if team == Team.EAST_WEST else Team.EAST_WEST
