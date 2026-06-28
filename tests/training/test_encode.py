from __future__ import annotations

import unittest
from dataclasses import replace

from server.domain.seats import Seat
from training.encode import CARD_FACES, encode_action, encode_critic_observation, encode_observation, encoding_schema
from training.env import GuandanTrainingEnv


class EncodingTests(unittest.TestCase):
    def test_observation_encoding_has_stable_named_features(self) -> None:
        env = GuandanTrainingEnv()
        env.reset(seed="encode-seed")

        encoded = encode_observation(env.observe(Seat.EAST))

        self.assertEqual(len(encoded.names), len(encoded.values))
        self.assertIn("seat/E", encoded.names)
        self.assertIn("hand_face/S-3", encoded.names)
        self.assertIn("played_face/S-3", encoded.names)
        self.assertIn("trick_pass_count", encoded.names)
        self.assertEqual(len([name for name in encoded.names if name.startswith("hand_face/")]), len(CARD_FACES))

    def test_encoding_schema_is_current_v2_only(self) -> None:
        current = encoding_schema("v2")

        self.assertEqual(current["version"], "v2")
        self.assertIn("played_face/S-3", current["observation_names"])
        self.assertIn("action_beats_opponent", current["action_names"])
        with self.assertRaises(ValueError):
            encoding_schema("v1")

    def test_observation_encoding_does_not_depend_on_opponent_card_identities(self) -> None:
        env = GuandanTrainingEnv()
        env.reset(seed="privacy-seed")
        before = encode_observation(env.observe(Seat.EAST))
        assert env.state.deal is not None
        hands = dict(env.state.deal.hands)
        hands[Seat.SOUTH] = tuple(reversed(hands[Seat.WEST]))
        env.state = replace(env.state, deal=replace(env.state.deal, hands=hands))

        after = encode_observation(env.observe(Seat.EAST))

        self.assertEqual(before, after)

    def test_critic_encoding_can_use_training_private_state(self) -> None:
        env = GuandanTrainingEnv()
        env.reset(seed="critic-seed")
        before = encode_critic_observation(env.state, Seat.EAST)
        assert env.state.deal is not None
        hands = dict(env.state.deal.hands)
        hands[Seat.SOUTH] = tuple(reversed(hands[Seat.WEST]))
        env.state = replace(env.state, deal=replace(env.state.deal, hands=hands))

        after = encode_critic_observation(env.state, Seat.EAST)

        self.assertEqual(len(before.names), len(before.values))
        self.assertIn("critic_hand_face/S/S-3", before.names)
        self.assertNotEqual(before, after)

    def test_action_encoding_marks_length_and_remaining_count(self) -> None:
        env = GuandanTrainingEnv()
        env.reset(seed="action-encode-seed")
        snapshot = env.observe(Seat.EAST)
        action = next(action for action in env.legal_actions(Seat.EAST) if action.kind == "play_cards")

        encoded = encode_action(action, snapshot).as_dict()

        self.assertEqual(encoded["action_length"], action.length / 8.0)
        self.assertEqual(encoded["remaining_after_action"], (len(snapshot.hand) - len(action.card_ids)) / 27.0)
        self.assertIn("action_finishes_hand", encoded)
        self.assertIn("action_rank_margin", encoded)


if __name__ == "__main__":
    unittest.main()
