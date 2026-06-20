from __future__ import annotations

import unittest
from dataclasses import replace

from server.domain.legal_actions import ActionCandidate
from server.domain.seats import SEATS, Seat
from server.domain.state import DealState, MatchPhase, TrickState
from training.env import GuandanTrainingEnv


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
        self.assertEqual(step.rewards[Seat.EAST], 1.0)
        self.assertEqual(step.rewards[Seat.WEST], 1.0)
        self.assertEqual(step.rewards[Seat.SOUTH], -1.0)
        self.assertEqual(step.rewards[Seat.NORTH], -1.0)

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


if __name__ == "__main__":
    unittest.main()
