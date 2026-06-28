from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Protocol

from server.domain.legal_actions import ActionCandidate
from server.domain.seats import SEATS, Seat
from server.domain.state import MatchPhase
from server.services.snapshots import SeatSnapshot
from training.encode import ENCODING_SCHEMA_VERSION, encode_action, encode_critic_observation, encode_observation
from training.env import GuandanTrainingEnv
from training.heuristic import HeuristicPolicy


@dataclass(frozen=True, slots=True)
class RolloutDecision:
    action: ActionCandidate
    action_index: int
    log_prob: float = 0.0
    value: float = 0.0


class RolloutPolicy(Protocol):
    def choose_decision(self, snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...]) -> RolloutDecision:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RolloutTransition:
    seed: str
    deal_id: int
    event_seq: int
    seat: str
    observation_names: tuple[str, ...]
    observation_values: tuple[float, ...]
    action_names: tuple[str, ...]
    candidate_values: tuple[tuple[float, ...], ...]
    candidate_payloads: tuple[dict[str, object], ...]
    action_index: int
    action_payload: dict[str, object]
    old_log_prob: float
    value: float
    reward: float = 0.0
    done: bool = False
    critic_observation_values: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class RolloutResult:
    transitions: tuple[RolloutTransition, ...]
    completed_deals: int
    steps: int
    stopped_reason: str


@dataclass(frozen=True, slots=True)
class HeuristicRolloutPolicy:
    policy: HeuristicPolicy = HeuristicPolicy()

    def choose_decision(self, snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...]) -> RolloutDecision:
        action = self.policy.choose_action(snapshot, actions)
        return RolloutDecision(action=action, action_index=actions.index(action), log_prob=0.0, value=0.0)


def collect_rollout(
    policies: Mapping[Seat, RolloutPolicy],
    *,
    seed: str | int | bytes | None,
    max_deals: int = 1,
    max_steps: int = 20_000,
    record_seats: frozenset[Seat] | None = None,
    reward_shaping_weight: float = 0.0,
) -> RolloutResult:
    env = GuandanTrainingEnv(reward_shaping_weight=reward_shaping_weight)
    env.reset(seed=seed)
    transitions: list[RolloutTransition] = []
    latest_by_seat: dict[Seat, int] = {}
    completed_deals = 0
    steps = 0
    stopped_reason = "match_complete"
    seed_label = repr(seed)

    while env.state.phase != MatchPhase.MATCH_COMPLETE:
        if steps >= max_steps:
            stopped_reason = "max_steps"
            break

        if env.state.phase == MatchPhase.DEAL_COMPLETE:
            completed_deals += 1
            if completed_deals >= max_deals:
                stopped_reason = "max_deals"
                break
            step = env.start_next_deal(seed=f"{seed_label}:deal:{completed_deals + 1}")
            latest_by_seat = {}
            if step.rejection is not None:
                stopped_reason = f"rejected:{step.rejection.code.value}"
                break
            continue

        actor = env.current_actor()
        if actor is None:
            stopped_reason = f"no_actor:{env.state.phase.value}"
            break

        snapshot = env.observe(actor)
        actions = env.legal_actions(actor)
        policy = policies[actor]
        if hasattr(policy, "choose_decision_with_state"):
            decision = policy.choose_decision_with_state(snapshot, actions, env.state)
        else:
            decision = policy.choose_decision(snapshot, actions)
        schema_version = str(getattr(policy, "schema_version", ENCODING_SCHEMA_VERSION))
        if record_seats is None or actor in record_seats:
            critic_values = (
                encode_critic_observation(env.state, actor, schema_version=schema_version).values
                if getattr(policy, "centralized_critic", False)
                else ()
            )
            transitions.append(
                _transition_from_decision(
                    seed_label,
                    snapshot,
                    actions,
                    decision,
                    schema_version=schema_version,
                    critic_observation_values=critic_values,
                )
            )
            latest_by_seat[actor] = len(transitions) - 1

        step = env.step(actor, decision.action)
        steps += 1
        if step.rejection is not None:
            stopped_reason = f"rejected:{step.rejection.code.value}"
            break
        if any(reward != 0.0 for reward in step.rewards.values()) or step.deal_complete:
            _apply_rewards(transitions, latest_by_seat, step.rewards, done=step.deal_complete)

    if env.state.phase == MatchPhase.MATCH_COMPLETE and env.state.last_deal_result is not None:
        completed_deals += 1

    return RolloutResult(tuple(transitions), completed_deals, steps, stopped_reason)


def collect_heuristic_rollout(
    *,
    seed: str | int | bytes | None,
    max_deals: int = 1,
    max_steps: int = 20_000,
    reward_shaping_weight: float = 0.0,
) -> RolloutResult:
    policy = HeuristicRolloutPolicy()
    return collect_rollout(
        {seat: policy for seat in SEATS},
        seed=seed,
        max_deals=max_deals,
        max_steps=max_steps,
        reward_shaping_weight=reward_shaping_weight,
    )


def discounted_returns(transitions: tuple[RolloutTransition, ...], *, gamma: float = 0.99) -> tuple[float, ...]:
    returns = [0.0 for _ in transitions]
    running_by_seat = {seat.value: 0.0 for seat in SEATS}
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        if transition.done:
            running_by_seat[transition.seat] = 0.0
        running = transition.reward + gamma * running_by_seat[transition.seat]
        running_by_seat[transition.seat] = running
        returns[index] = running
    return tuple(returns)


def _transition_from_decision(
    seed: str,
    snapshot: SeatSnapshot,
    actions: tuple[ActionCandidate, ...],
    decision: RolloutDecision,
    *,
    schema_version: str = ENCODING_SCHEMA_VERSION,
    critic_observation_values: tuple[float, ...] = (),
) -> RolloutTransition:
    observation = encode_observation(snapshot, schema_version=schema_version)
    action_vectors = tuple(encode_action(action, snapshot, schema_version=schema_version) for action in actions)
    action_names = action_vectors[0].names if action_vectors else ()
    return RolloutTransition(
        seed=seed,
        deal_id=snapshot.public.deal_id,
        event_seq=snapshot.public.event_seq,
        seat=snapshot.seat.value,
        observation_names=observation.names,
        observation_values=observation.values,
        action_names=action_names,
        candidate_values=tuple(vector.values for vector in action_vectors),
        candidate_payloads=tuple(action.to_payload() for action in actions),
        action_index=decision.action_index,
        action_payload=decision.action.to_payload(),
        old_log_prob=decision.log_prob,
        value=decision.value,
        critic_observation_values=critic_observation_values,
    )


def _apply_rewards(
    transitions: list[RolloutTransition],
    latest_by_seat: dict[Seat, int],
    rewards: dict[Seat, float],
    *,
    done: bool,
) -> None:
    for seat, reward in rewards.items():
        index = latest_by_seat.get(seat)
        if index is None:
            continue
        current = transitions[index]
        transitions[index] = replace(current, reward=current.reward + reward, done=current.done or done)
