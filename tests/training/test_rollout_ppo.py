from __future__ import annotations

import unittest

from training.ppo_train import PpoConfig, _initial_dimensions, _parser
from training.rollout import RolloutTransition, collect_heuristic_rollout, discounted_returns


class RolloutTests(unittest.TestCase):
    def test_collect_heuristic_rollout_records_rewards_and_done_flags(self) -> None:
        result = collect_heuristic_rollout(seed="1", max_deals=1, max_steps=1_000)

        self.assertEqual(result.stopped_reason, "max_deals")
        self.assertEqual(result.completed_deals, 1)
        self.assertGreater(len(result.transitions), 0)
        self.assertTrue(any(transition.done for transition in result.transitions))
        self.assertTrue(any(transition.reward != 0.0 for transition in result.transitions))
        first = result.transitions[0]
        self.assertEqual(first.candidate_payloads[first.action_index], first.action_payload)

    def test_discounted_returns_are_computed_per_seat(self) -> None:
        transitions = (
            _transition("E", reward=0.0),
            _transition("S", reward=0.0),
            _transition("E", reward=1.0, done=True),
            _transition("S", reward=-1.0, done=True),
        )

        returns = discounted_returns(transitions, gamma=0.5)

        self.assertEqual(returns, (0.5, -0.5, 1.0, -1.0))


class PpoScaffoldTests(unittest.TestCase):
    def test_initial_dimensions_come_from_training_environment(self) -> None:
        observation_dim, action_dim = _initial_dimensions("1")

        self.assertGreater(observation_dim, 0)
        self.assertGreater(action_dim, 0)

    def test_parser_uses_default_seed_when_seed_is_omitted(self) -> None:
        args = _parser().parse_args(["model.pt"])

        config = PpoConfig(output_path=args.output, rollout_seeds=tuple(args.seed or ("ppo-seed-0",)))

        self.assertEqual(config.rollout_seeds, ("ppo-seed-0",))

    def test_parser_uses_explicit_repeated_seeds(self) -> None:
        args = _parser().parse_args(["model.pt", "--seed", "a", "--seed", "b"])

        self.assertEqual(tuple(args.seed), ("a", "b"))


def _transition(seat: str, *, reward: float, done: bool = False) -> RolloutTransition:
    return RolloutTransition(
        seed="seed",
        deal_id=1,
        event_seq=1,
        seat=seat,
        observation_names=("obs",),
        observation_values=(0.0,),
        action_names=("act",),
        candidate_values=((0.0,),),
        candidate_payloads=({"type": "pass"},),
        action_index=0,
        action_payload={"type": "pass"},
        old_log_prob=0.0,
        value=0.0,
        reward=reward,
        done=done,
    )


if __name__ == "__main__":
    unittest.main()
