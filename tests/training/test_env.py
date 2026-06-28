from __future__ import annotations

import unittest
from dataclasses import replace

from server.domain.cards import Rank
from server.domain.legal_actions import ActionCandidate
from server.domain.seats import SEATS, Seat, Team
from server.domain.state import DealResult, DealState, MatchPhase, TrickState
from training.env import (
    GuandanTrainingEnv,
    _initial_hand_profile,
    _reward_multiplier_for_result,
    _reward_multipliers_for_result,
    _rewards_for_transition,
)


class GuandanTrainingEnvTests(unittest.TestCase):
    def test_reset_is_deterministic_for_seed(self) -> None:
        first = GuandanTrainingEnv()
        second = GuandanTrainingEnv()

        first.reset(seed="fixed-training-seed")
        second.reset(seed="fixed-training-seed")

        assert first.state.deal is not None
        assert second.state.deal is not None
        self.assertEqual(first.state.phase, MatchPhase.PLAYING)
        self.assertEqual(first.current_actor(), Seat.EAST)
        self.assertEqual(first.state.deal.hands, second.state.deal.hands)

    def test_observe_returns_private_snapshot_for_one_seat(self) -> None:
        env = GuandanTrainingEnv()
        env.reset(seed="snapshot-seed")
        assert env.state.deal is not None

        snapshot = env.observe(Seat.SOUTH)

        self.assertEqual(snapshot.seat, Seat.SOUTH)
        self.assertEqual(snapshot.hand, env.state.deal.hand_for(Seat.SOUTH))
        self.assertEqual(snapshot.public.hand_counts[Seat.SOUTH], 27)
        self.assertFalse(hasattr(snapshot, "hands"))

    def test_step_accepts_legal_action_candidate(self) -> None:
        env = GuandanTrainingEnv()
        env.reset(seed="step-seed")
        actor = env.current_actor()
        assert actor is not None
        action = next(action for action in env.legal_actions(actor) if action.kind == "play_cards")

        step = env.step(actor, action)

        self.assertIsNone(step.rejection)
        self.assertTrue(step.events)
        self.assertEqual(step.rewards, {seat: 0.0 for seat in SEATS})

    def test_deal_completion_emits_team_rewards(self) -> None:
        env = GuandanTrainingEnv()
        env.reset(seed="reward-seed")
        env.state = _one_card_double_down_state(env)

        step = env.step(Seat.EAST, _action_by_cards(env, Seat.EAST, ("D1-S-A",)))
        self.assertIsNone(step.rejection)
        for seat in (Seat.NORTH, Seat.WEST, Seat.SOUTH):
            step = env.step(seat, env.legal_actions(seat)[0])
            self.assertIsNone(step.rejection)
        step = env.step(Seat.WEST, _action_by_cards(env, Seat.WEST, ("D1-S-K",)))

        self.assertIsNone(step.rejection)
        self.assertEqual(step.state.phase, MatchPhase.DEAL_COMPLETE)
        assert step.state.last_deal_result is not None
        expected = _reward_multiplier_for_result(step.state.last_deal_result, env._deal_start_hands, Rank.TWO)
        self.assertAlmostEqual(step.rewards[Seat.EAST], expected)
        self.assertAlmostEqual(step.rewards[Seat.WEST], expected)
        self.assertAlmostEqual(step.rewards[Seat.SOUTH], -expected)
        self.assertAlmostEqual(step.rewards[Seat.NORTH], -expected)

    def test_initial_hand_profile_scores_controls_and_regularity(self) -> None:
        strong = (
            "D1-BJ",
            "D2-BJ",
            "D1-SJ",
            "D2-SJ",
            "D1-S-A",
            "D2-S-A",
            "D1-H-A",
            "D2-H-A",
        )
        weak = (
            "D1-S-3",
            "D1-H-4",
            "D1-C-6",
            "D1-D-8",
            "D2-S-10",
            "D2-H-J",
            "D2-C-Q",
            "D2-D-K",
        )

        strong_profile = _initial_hand_profile(strong, Rank.TWO)
        weak_profile = _initial_hand_profile(weak, Rank.TWO)

        self.assertGreater(strong_profile.control_score, weak_profile.control_score)
        self.assertLess(strong_profile.estimated_turns, weak_profile.estimated_turns)
        self.assertGreater(strong_profile.strength_score, weak_profile.strength_score)

    def test_reward_shaping_can_reward_finishing_action(self) -> None:
        env = GuandanTrainingEnv(reward_shaping_weight=1.0)
        env.reset(seed="shaping-seed")
        env.state = _one_card_not_complete_state(env)

        step = env.step(Seat.EAST, _action_by_cards(env, Seat.EAST, ("D1-S-A",)))

        self.assertIsNone(step.rejection)
        self.assertAlmostEqual(step.rewards[Seat.EAST], 0.08)

    def test_reward_multiplier_discounts_strong_winners_and_rewards_upsets(self) -> None:
        initial_hands = _strong_east_west_initial_hands()
        strong_win = _reward_multiplier_for_result(
            _deal_result(Team.EAST_WEST),
            initial_hands,
            Rank.TWO,
        )
        weak_win = _reward_multiplier_for_result(
            _deal_result(Team.SOUTH_NORTH),
            initial_hands,
            Rank.TWO,
        )

        self.assertLess(strong_win, 1.0)
        self.assertGreater(weak_win, 1.0)
        self.assertLess(strong_win, weak_win)

    def test_reward_multipliers_penalize_strong_losing_team_more_than_upset_reward(self) -> None:
        initial_hands = _strong_east_west_initial_hands()
        result = _deal_result(Team.SOUTH_NORTH)

        multipliers = _reward_multipliers_for_result(result, initial_hands, Rank.TWO)
        rewards = _rewards_for_transition(None, result, initial_hands=initial_hands, level=Rank.TWO)

        self.assertGreater(multipliers.winner, 1.0)
        self.assertGreater(multipliers.loser, multipliers.winner)
        self.assertAlmostEqual(rewards[Seat.SOUTH], multipliers.winner)
        self.assertAlmostEqual(rewards[Seat.NORTH], multipliers.winner)
        self.assertAlmostEqual(rewards[Seat.EAST], -multipliers.loser)
        self.assertAlmostEqual(rewards[Seat.WEST], -multipliers.loser)

    def test_start_next_deal_after_deal_complete(self) -> None:
        env = GuandanTrainingEnv()
        env.reset(seed="next-deal-seed")
        env.state = _one_card_double_down_state(env)
        env.step(Seat.EAST, _action_by_cards(env, Seat.EAST, ("D1-S-A",)))
        for seat in (Seat.NORTH, Seat.WEST, Seat.SOUTH):
            env.step(seat, env.legal_actions(seat)[0])
        env.step(Seat.WEST, _action_by_cards(env, Seat.WEST, ("D1-S-K",)))

        step = env.start_next_deal(seed="after-double-down")

        self.assertIsNone(step.rejection)
        self.assertIn(step.state.phase, {MatchPhase.TRIBUTE, MatchPhase.PLAYING})
        self.assertIsNotNone(env.current_actor())


def _one_card_double_down_state(env: GuandanTrainingEnv):
    assert env.state.deal is not None
    hands = {
        Seat.EAST: ("D1-S-A",),
        Seat.SOUTH: ("D1-S-4",),
        Seat.WEST: ("D1-S-K",),
        Seat.NORTH: ("D1-S-3",),
    }
    env._deal_start_hands = hands
    deal = DealState(
        hands=hands,
        active_seats=frozenset(SEATS),
        finish_order=(),
        leader=Seat.EAST,
        turn=Seat.EAST,
        current_trick=TrickState(lead_seat=Seat.EAST),
    )
    return replace(env.state, phase=MatchPhase.PLAYING, deal=deal)


def _one_card_not_complete_state(env: GuandanTrainingEnv):
    assert env.state.deal is not None
    hands = {
        Seat.EAST: ("D1-S-A",),
        Seat.SOUTH: ("D1-S-4", "D1-H-4"),
        Seat.WEST: ("D1-S-K", "D1-H-K"),
        Seat.NORTH: ("D1-S-3", "D1-H-3"),
    }
    env._deal_start_hands = hands
    deal = DealState(
        hands=hands,
        active_seats=frozenset(SEATS),
        finish_order=(),
        leader=Seat.EAST,
        turn=Seat.EAST,
        current_trick=TrickState(lead_seat=Seat.EAST),
    )
    return replace(env.state, phase=MatchPhase.PLAYING, deal=deal)


def _action_by_cards(env: GuandanTrainingEnv, seat: Seat, card_ids: tuple[str, ...]) -> ActionCandidate:
    for action in env.legal_actions(seat):
        if action.card_ids == card_ids:
            return action
    raise AssertionError(f"no legal action for {seat.value}: {card_ids}")


def _strong_east_west_initial_hands() -> dict[Seat, tuple[str, ...]]:
    return {
        Seat.EAST: (
            "D1-BJ",
            "D2-BJ",
            "D1-SJ",
            "D2-SJ",
            "D1-S-A",
            "D2-S-A",
            "D1-H-A",
            "D2-H-A",
        ),
        Seat.WEST: (
            "D1-S-K",
            "D2-S-K",
            "D1-H-K",
            "D2-H-K",
            "D1-S-Q",
            "D2-S-Q",
            "D1-H-Q",
            "D2-H-Q",
        ),
        Seat.SOUTH: (
            "D1-S-3",
            "D1-H-4",
            "D1-C-6",
            "D1-D-8",
            "D2-S-10",
            "D2-H-J",
            "D2-C-Q",
            "D2-D-K",
        ),
        Seat.NORTH: (
            "D2-S-3",
            "D2-H-4",
            "D2-C-6",
            "D2-D-8",
            "D1-S-10",
            "D1-H-J",
            "D1-C-Q",
            "D1-D-K",
        ),
    }


def _deal_result(winning_team: Team) -> DealResult:
    finish_order = (
        (Seat.EAST, Seat.WEST, Seat.SOUTH, Seat.NORTH)
        if winning_team == Team.EAST_WEST
        else (Seat.SOUTH, Seat.NORTH, Seat.EAST, Seat.WEST)
    )
    return DealResult(
        finish_order=finish_order,
        winning_team=winning_team,
        advance_count=3,
        previous_level=Rank.TWO,
        next_level=Rank.FIVE,
        match_complete=False,
    )


if __name__ == "__main__":
    unittest.main()
