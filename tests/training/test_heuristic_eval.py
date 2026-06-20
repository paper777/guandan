from __future__ import annotations

import unittest
from dataclasses import replace

from server.domain.cards import CARD_BY_ID
from server.domain.hand_types import parse_hand
from server.domain.seats import SEATS, Seat
from server.domain.state import DealState, MatchPhase, TrickState
from training.env import GuandanTrainingEnv
from training.eval import evaluate_heuristic_match
from training.heuristic import HeuristicPolicy


class HeuristicPolicyTests(unittest.TestCase):
    def test_lead_prefers_longer_ordinary_play_over_single(self) -> None:
        env = GuandanTrainingEnv()
        env.reset(seed="heuristic-lead")
        env.state = _state_with_hands(
            env,
            turn=Seat.EAST,
            hands={
                Seat.EAST: ("D1-S-3", "D2-S-3", "D1-S-4"),
                Seat.SOUTH: ("D1-S-5",),
                Seat.WEST: ("D1-S-6",),
                Seat.NORTH: ("D1-S-7",),
            },
        )

        policy = HeuristicPolicy()
        action = policy.choose_action(env.observe(Seat.EAST), env.legal_actions(Seat.EAST))

        self.assertEqual(action.card_ids, ("D1-S-3", "D2-S-3"))

    def test_follow_passes_when_partner_is_winning_trick(self) -> None:
        env = GuandanTrainingEnv()
        env.reset(seed="heuristic-partner")
        last_play = parse_hand(tuple(CARD_BY_ID[card_id] for card_id in ("D1-S-3",)))
        env.state = _state_with_hands(
            env,
            turn=Seat.EAST,
            hands={
                Seat.EAST: ("D1-S-4", "D1-S-8"),
                Seat.SOUTH: ("D1-S-5",),
                Seat.WEST: ("D1-S-3",),
                Seat.NORTH: ("D1-S-6",),
            },
            trick=TrickState(lead_seat=Seat.WEST, last_play=last_play, last_play_seat=Seat.WEST),
        )

        policy = HeuristicPolicy()
        action = policy.choose_action(env.observe(Seat.EAST), env.legal_actions(Seat.EAST))

        self.assertEqual(action.kind, "pass")


class EvaluationTests(unittest.TestCase):
    def test_heuristic_evaluation_runs_one_deal_without_rejection(self) -> None:
        result = evaluate_heuristic_match(seed="1", max_deals=1, max_steps=1_000)

        self.assertEqual(result.stopped_reason, "max_deals")
        self.assertEqual(result.completed_deals, 1)
        self.assertGreater(result.steps, 0)
        self.assertEqual(result.final_phase, MatchPhase.DEAL_COMPLETE)


def _state_with_hands(
    env: GuandanTrainingEnv,
    *,
    turn: Seat,
    hands: dict[Seat, tuple[str, ...]],
    trick: TrickState | None = None,
):
    all_hands = {
        Seat.EAST: ("D1-S-3",),
        Seat.SOUTH: ("D1-S-4",),
        Seat.WEST: ("D1-S-5",),
        Seat.NORTH: ("D1-S-6",),
    }
    all_hands.update(hands)
    deal = DealState(
        hands=all_hands,
        active_seats=frozenset(SEATS),
        finish_order=(),
        leader=turn,
        turn=turn,
        current_trick=trick or TrickState(lead_seat=turn),
    )
    return replace(env.state, phase=MatchPhase.PLAYING, deal=deal)


if __name__ == "__main__":
    unittest.main()
