from __future__ import annotations

import unittest

from training.ppo_train import (
    PpoConfig,
    _format_stop_counts,
    _gae_returns_and_advantages,
    _initial_dimensions,
    _initial_model_state_from_checkpoint,
    _iter_batches,
    _parser,
    _policy_state_from_bc_checkpoint,
    _rollout_metrics,
    _rollout_seeds_from_args,
)
from training.rollout import RolloutResult, RolloutTransition, collect_heuristic_rollout, discounted_returns


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

        config = PpoConfig(
            output_path=args.output,
            rollout_seeds=_rollout_seeds_from_args(args.seed, args.seed_count),
        )

        self.assertEqual(config.rollout_seeds, ("ppo-seed-0",))

    def test_parser_uses_explicit_repeated_seeds(self) -> None:
        args = _parser().parse_args(["model.pt", "--seed", "a", "--seed", "b"])

        self.assertEqual(_rollout_seeds_from_args(args.seed, args.seed_count), ("a", "b"))

    def test_parser_generates_seed_count_seeds(self) -> None:
        args = _parser().parse_args(["model.pt", "--seed-count", "3"])

        self.assertEqual(
            _rollout_seeds_from_args(args.seed, args.seed_count),
            ("ppo-seed-0", "ppo-seed-1", "ppo-seed-2"),
        )

    def test_parser_rejects_seed_count_with_explicit_seed(self) -> None:
        with self.assertRaises(ValueError):
            _rollout_seeds_from_args(["a"], 2)

    def test_parser_uses_bc_warm_start_ppo_defaults(self) -> None:
        args = _parser().parse_args(["model.pt"])

        self.assertEqual(args.epochs_per_update, 3)
        self.assertEqual(args.batch_size, 256)
        self.assertAlmostEqual(args.learning_rate, 1e-4)
        self.assertAlmostEqual(args.gamma, 0.995)
        self.assertAlmostEqual(args.gae_lambda, 0.95)
        self.assertAlmostEqual(args.clip_epsilon, 0.1)
        self.assertAlmostEqual(args.entropy_coef, 0.003)
        self.assertAlmostEqual(args.max_grad_norm, 0.5)
        self.assertAlmostEqual(args.target_kl, 0.03)
        self.assertAlmostEqual(args.dropout, 0.0)

    def test_parser_accepts_bc_policy_initialization_checkpoint(self) -> None:
        args = _parser().parse_args(["model.pt", "--init-policy", "bc.pt"])

        self.assertEqual(args.init_policy, "bc.pt")

    def test_parser_accepts_trained_model_initialization_alias(self) -> None:
        args = _parser().parse_args(["model.pt", "--init-model", "ppo.pt"])

        self.assertEqual(args.init_policy, "ppo.pt")

    def test_gae_returns_and_advantages_follow_seat_trajectory(self) -> None:
        transitions = (
            _transition("E", reward=0.0, value=0.2),
            _transition("E", reward=1.0, value=0.1, done=True),
        )

        returns, advantages = _gae_returns_and_advantages(
            transitions,
            gamma=1.0,
            gae_lambda=1.0,
            normalize=False,
        )

        self.assertEqual(returns, (1.0, 1.0))
        self.assertAlmostEqual(advantages[0], 0.8)
        self.assertAlmostEqual(advantages[1], 0.9)

    def test_iter_batches_yields_tail_batch(self) -> None:
        self.assertEqual(
            list(_iter_batches([0, 1, 2, 3, 4], 2)),
            [[0, 1], [2, 3], [4]],
        )

    def test_rollout_metrics_summarize_progress_output_fields(self) -> None:
        rollouts = [
            RolloutResult((), completed_deals=2, steps=10, stopped_reason="max_deals"),
            RolloutResult((), completed_deals=1, steps=7, stopped_reason="match_complete"),
            RolloutResult((), completed_deals=2, steps=9, stopped_reason="max_deals"),
        ]

        metrics = _rollout_metrics(rollouts)

        self.assertEqual(metrics.completed_deals, 5)
        self.assertEqual(metrics.steps, 26)
        self.assertEqual(_format_stop_counts(metrics.stopped_reasons), "match_complete:1,max_deals:2")

    def test_bc_checkpoint_policy_state_maps_ranker_net_to_actor_policy(self) -> None:
        checkpoint = {
            "observation_dim": 3,
            "action_dim": 2,
            "hidden_dim": 4,
            "model_state": {
                "net.0.weight": "w0",
                "net.0.bias": "b0",
                "ignored": "value",
            },
        }

        policy_state = _policy_state_from_bc_checkpoint(checkpoint, 3, 2, 4)

        self.assertEqual(policy_state, {"0.weight": "w0", "0.bias": "b0"})

    def test_initial_checkpoint_state_accepts_trained_ppo_actor_critic(self) -> None:
        checkpoint = {
            "observation_dim": 3,
            "action_dim": 2,
            "hidden_dim": 4,
            "model_state": {
                "policy_net.0.weight": "pw0",
                "policy_net.0.bias": "pb0",
                "value_net.0.weight": "vw0",
                "value_net.0.bias": "vb0",
            },
        }

        kind, model_state = _initial_model_state_from_checkpoint(checkpoint, 3, 2, 4)

        self.assertEqual(kind, "ppo")
        self.assertEqual(model_state, checkpoint["model_state"])

    def test_initial_checkpoint_state_maps_bc_ranker_for_bootstrap(self) -> None:
        checkpoint = {
            "observation_dim": 3,
            "action_dim": 2,
            "hidden_dim": 4,
            "model_state": {
                "net.0.weight": "w0",
                "net.0.bias": "b0",
            },
        }

        kind, model_state = _initial_model_state_from_checkpoint(checkpoint, 3, 2, 4)

        self.assertEqual(kind, "bc")
        self.assertEqual(model_state, {"0.weight": "w0", "0.bias": "b0"})

    def test_bc_checkpoint_policy_state_rejects_dimension_mismatch(self) -> None:
        checkpoint = {
            "observation_dim": 99,
            "action_dim": 2,
            "hidden_dim": 4,
            "model_state": {"net.0.weight": "w0"},
        }

        with self.assertRaises(ValueError):
            _policy_state_from_bc_checkpoint(checkpoint, 3, 2, 4)


def _transition(seat: str, *, reward: float, done: bool = False, value: float = 0.0) -> RolloutTransition:
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
        value=value,
        reward=reward,
        done=done,
    )


if __name__ == "__main__":
    unittest.main()
