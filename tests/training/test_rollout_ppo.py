from __future__ import annotations

import random
import unittest
from pathlib import Path

from server.domain.seats import SEATS
from training.encode import encoding_schema
from training.ppo_train import (
    PolicyInferenceProfile,
    PpoConfig,
    _candidate_count_bucket,
    _format_policy_inference_profile,
    _format_rollout_profile,
    _format_stop_counts,
    _gae_returns_and_advantages,
    _initial_dimensions,
    _initial_model_state_from_checkpoint,
    _iter_batches,
    _iter_candidate_bucket_batches,
    _linear_decay,
    _parser,
    _policy_state_from_bc_checkpoint,
    _rollout_jobs_for_update,
    _rollout_metrics,
    _rollout_process_jobs_for_update,
    _rollout_seeds_from_args,
)
from training.rollout import RolloutProfile, RolloutResult, RolloutTransition, collect_heuristic_rollout, discounted_returns


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
        observation_dim, action_dim, critic_dim, critic_names = _initial_dimensions("1")

        self.assertGreater(observation_dim, 0)
        self.assertGreater(action_dim, 0)
        self.assertGreater(critic_dim, observation_dim)
        self.assertIn("critic_actor/E", critic_names)

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
        self.assertEqual(args.opponent_pool, "self")
        self.assertEqual(args.rollout_workers, 1)
        self.assertFalse(args.no_candidate_bucket_batches)
        self.assertAlmostEqual(args.reward_shaping_start, 0.02)

    def test_parser_accepts_bc_policy_initialization_checkpoint(self) -> None:
        args = _parser().parse_args(["model.pt", "--init-policy", "bc.pt"])

        self.assertEqual(args.init_policy, "bc.pt")

    def test_parser_accepts_late_optimization_flags(self) -> None:
        args = _parser().parse_args([
            "model.pt",
              "--opponent-pool",
              "self,heuristic,dummy,previous",
              "--opponent-checkpoint",
              "history.pt",
              "--rollout-workers",
              "2",
              "--rollout-processes",
              "3",
              "--inference-batch-size",
              "8",
              "--inference-batch-wait-ms",
              "2.5",
              "--no-candidate-bucket-batches",
              "--reward-shaping-start",
              "0.05",
              "--reward-shaping-end",
              "0.01",
          ])

        self.assertEqual(args.opponent_pool, "self,heuristic,dummy,previous")
        self.assertEqual(args.opponent_checkpoint, ["history.pt"])
        self.assertEqual(args.rollout_workers, 2)
        self.assertEqual(args.rollout_processes, 3)
        self.assertEqual(args.inference_batch_size, 8)
        self.assertAlmostEqual(args.inference_batch_wait_ms, 2.5)
        self.assertTrue(args.no_candidate_bucket_batches)
        self.assertAlmostEqual(args.reward_shaping_start, 0.05)
        self.assertAlmostEqual(args.reward_shaping_end, 0.01)

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

    def test_candidate_count_bucket_uses_power_of_two_bounds(self) -> None:
        self.assertEqual(_candidate_count_bucket(0), 0)
        self.assertEqual(_candidate_count_bucket(1), 1)
        self.assertEqual(_candidate_count_bucket(2), 2)
        self.assertEqual(_candidate_count_bucket(3), 4)
        self.assertEqual(_candidate_count_bucket(8), 8)
        self.assertEqual(_candidate_count_bucket(9), 16)

    def test_candidate_bucket_batches_keep_similar_padding_widths(self) -> None:
        candidate_counts = [1, 2, 3, 4, 5, 8, 9, 12]
        transitions = [
            _transition("E", reward=0.0, candidate_count=candidate_count)
            for candidate_count in candidate_counts
        ]

        batches = list(
            _iter_candidate_bucket_batches(
                transitions,
                list(range(len(transitions))),
                2,
                random.Random(3),
            )
        )

        flattened = [index for batch in batches for index in batch]
        self.assertCountEqual(flattened, range(len(transitions)))
        for batch in batches:
            self.assertLessEqual(len(batch), 2)
            bucket_keys = {
                _candidate_count_bucket(len(transitions[index].candidate_values))
                for index in batch
            }
            self.assertEqual(len(bucket_keys), 1)

    def test_rollout_metrics_summarize_progress_output_fields(self) -> None:
        rollouts = [
            RolloutResult(
                (),
                completed_deals=2,
                steps=10,
                stopped_reason="max_deals",
                profile=RolloutProfile(decisions=2, recorded_transitions=1, candidate_count_total=6, candidate_count_max=4),
            ),
            RolloutResult(
                (),
                completed_deals=1,
                steps=7,
                stopped_reason="match_complete",
                profile=RolloutProfile(
                    decisions=3,
                    recorded_transitions=2,
                    candidate_count_total=15,
                    candidate_count_max=8,
                    encoded_transition_reuses=2,
                ),
            ),
            RolloutResult((), completed_deals=2, steps=9, stopped_reason="max_deals"),
        ]

        metrics = _rollout_metrics(rollouts)

        self.assertEqual(metrics.completed_deals, 5)
        self.assertEqual(metrics.steps, 26)
        self.assertEqual(_format_stop_counts(metrics.stopped_reasons), "match_complete:1,max_deals:2")
        self.assertEqual(metrics.profile.decisions, 5)
        self.assertEqual(metrics.profile.recorded_transitions, 3)
        self.assertEqual(metrics.profile.candidate_count_total, 21)
        self.assertEqual(metrics.profile.candidate_count_max, 8)
        self.assertIn("encoded_reuse=2/3", _format_rollout_profile(metrics.profile))

    def test_policy_inference_profile_formats_batch_stats(self) -> None:
        profile = PolicyInferenceProfile(requests=9, batches=3, max_batch_size=4, inference_seconds=1.25)

        formatted = _format_policy_inference_profile(profile)

        self.assertIn("requests=9", formatted)
        self.assertIn("avg_batch=3.0", formatted)
        self.assertEqual(_format_policy_inference_profile(PolicyInferenceProfile()), "direct")

    def test_opponent_pool_expands_seed_jobs_by_candidate_team(self) -> None:
        jobs = _rollout_jobs_for_update(
            PpoConfig(
                output_path=Path("model.pt"),
                rollout_seeds=("seed",),
                opponent_pool=("heuristic,dummy",),
            ),
            current_policy=object(),
            update_index=0,
            device_name="cpu",
            reward_shaping_weight=0.0,
        )

        self.assertEqual(len(jobs), 4)
        self.assertTrue(all(job.record_seats is not None for job in jobs))

    def test_process_opponent_pool_expands_to_picklable_specs(self) -> None:
        jobs = _rollout_process_jobs_for_update(
            PpoConfig(
                output_path=Path("model.pt"),
                init_policy_path=Path("previous.pt"),
                rollout_seeds=("seed",),
                opponent_pool=("self,heuristic,previous",),
            ),
            update_index=0,
            device_name="cuda",
            reward_shaping_weight=0.0,
            schema_version="v2",
        )

        self.assertEqual(len(jobs), 5)
        self.assertEqual(jobs[0].opponent_name, "self")
        self.assertIsNone(jobs[0].record_seats)
        self.assertEqual(jobs[0].current_policy_seats, frozenset(SEATS))
        self.assertEqual(jobs[-1].opponent_name, "previous")
        self.assertEqual(jobs[-1].opponent_checkpoint_path, Path("previous.pt"))
        self.assertEqual(jobs[-1].opponent_device_name, "cpu")
        self.assertTrue(jobs[-1].centralized_critic)

    def test_reward_shaping_weight_decays_linearly(self) -> None:
        self.assertAlmostEqual(_linear_decay(0.05, 0.01, 0, 5), 0.05)
        self.assertAlmostEqual(_linear_decay(0.05, 0.01, 4, 5), 0.01)
        self.assertAlmostEqual(_linear_decay(0.05, 0.01, 2, 5), 0.03)

    def test_bc_checkpoint_policy_state_maps_ranker_net_to_actor_policy(self) -> None:
        observation_dim, action_dim = _schema_dims()
        checkpoint = {
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "hidden_dim": 4,
            "encoding_schema": encoding_schema(),
            "model_architecture": "dual_tower_v1",
            "model_state": {
                "policy_net.state_encoder.0.weight": "sw0",
                "policy_net.action_encoder.0.weight": "aw0",
                "ignored": "value",
            },
        }

        policy_state = _policy_state_from_bc_checkpoint(checkpoint, observation_dim, action_dim, 4)

        self.assertEqual(policy_state, {"state_encoder.0.weight": "sw0", "action_encoder.0.weight": "aw0"})

    def test_initial_checkpoint_state_accepts_trained_ppo_actor_critic(self) -> None:
        observation_dim, action_dim = _schema_dims()
        checkpoint = {
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "critic_observation_dim": 5,
            "hidden_dim": 4,
            "encoding_schema": encoding_schema(),
            "model_architecture": "dual_tower_v1",
            "centralized_critic": True,
            "model_state": {
                "policy_net.0.weight": "pw0",
                "policy_net.0.bias": "pb0",
                "value_net.0.weight": "vw0",
                "value_net.0.bias": "vb0",
            },
        }

        kind, model_state = _initial_model_state_from_checkpoint(
            checkpoint,
            observation_dim,
            action_dim,
            4,
            value_input_dim=5,
        )

        self.assertEqual(kind, "ppo")
        self.assertEqual(model_state, checkpoint["model_state"])

    def test_initial_checkpoint_state_maps_bc_ranker_for_bootstrap(self) -> None:
        observation_dim, action_dim = _schema_dims()
        checkpoint = {
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "hidden_dim": 4,
            "encoding_schema": encoding_schema(),
            "model_architecture": "dual_tower_v1",
            "model_state": {
                "policy_net.state_encoder.0.weight": "sw0",
                "policy_net.action_encoder.0.weight": "aw0",
            },
        }

        kind, model_state = _initial_model_state_from_checkpoint(checkpoint, observation_dim, action_dim, 4)

        self.assertEqual(kind, "bc")
        self.assertEqual(model_state, {"state_encoder.0.weight": "sw0", "action_encoder.0.weight": "aw0"})

    def test_initial_checkpoint_state_maps_dual_tower_bc_ranker_for_bootstrap(self) -> None:
        observation_dim, action_dim = _schema_dims()
        checkpoint = {
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "hidden_dim": 4,
            "encoding_schema": encoding_schema(),
            "model_architecture": "dual_tower_v1",
            "model_state": {
                "policy_net.state_encoder.0.weight": "sw0",
                "policy_net.action_encoder.0.weight": "aw0",
            },
        }

        kind, model_state = _initial_model_state_from_checkpoint(
            checkpoint,
            observation_dim,
            action_dim,
            4,
        )

        self.assertEqual(kind, "bc")
        self.assertEqual(
            model_state,
            {"state_encoder.0.weight": "sw0", "action_encoder.0.weight": "aw0"},
        )

    def test_bc_checkpoint_policy_state_rejects_dimension_mismatch(self) -> None:
        observation_dim, action_dim = _schema_dims()
        checkpoint = {
            "observation_dim": 99,
            "action_dim": action_dim,
            "hidden_dim": 4,
            "encoding_schema": encoding_schema(),
            "model_architecture": "dual_tower_v1",
            "model_state": {"policy_net.state_encoder.0.weight": "sw0"},
        }

        with self.assertRaises(ValueError):
            _policy_state_from_bc_checkpoint(checkpoint, observation_dim, action_dim, 4)


def _transition(
    seat: str,
    *,
    reward: float,
    done: bool = False,
    value: float = 0.0,
    candidate_count: int = 1,
) -> RolloutTransition:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    candidate_values = tuple((float(index),) for index in range(candidate_count))
    candidate_payloads = tuple({"type": "pass", "index": index} for index in range(candidate_count))
    return RolloutTransition(
        seed="seed",
        deal_id=1,
        event_seq=1,
        seat=seat,
        observation_names=("obs",),
        observation_values=(0.0,),
        action_names=("act",),
        candidate_values=candidate_values,
        candidate_payloads=candidate_payloads,
        action_index=0,
        action_payload=candidate_payloads[0],
        old_log_prob=0.0,
        value=value,
        reward=reward,
        done=done,
    )


def _schema_dims() -> tuple[int, int]:
    schema = encoding_schema()
    return len(schema["observation_names"]), len(schema["action_names"])


if __name__ == "__main__":
    unittest.main()
