from __future__ import annotations

import unittest

from server.domain.cards import CARD_BY_ID, Rank
from server.domain.commands import Pass
from server.domain.controllers import ControllerCapability, ControllerKind, ControllerRef, PlayerKind, PlayerRef
from server.domain.hand_types import HandType, parse_hand
from server.domain.legal_actions import (
    ActionCandidate,
    candidate_generation_cache_info,
    legal_actions_for_snapshot,
    legal_actions_for_state,
)
from server.domain.reducer import reduce_command
from server.domain.seats import SEATS, Seat
from server.domain.state import DealState, MatchPhase, MatchState, TributeObligation, TributeState, TrickState
from server.services.snapshots import SeatSnapshot, public_snapshot


def player(seat: Seat) -> PlayerRef:
    return PlayerRef(id=f"player-{seat.value}", display_name=f"Player {seat.value}", kind=PlayerKind.BOT)


def controller(seat: Seat) -> ControllerRef:
    return ControllerRef(
        id=f"controller-{seat.value}",
        kind=ControllerKind.LOCAL_BOT,
        seat=seat,
        player_id=f"player-{seat.value}",
        capabilities=frozenset(
            {
                ControllerCapability.PLAY,
                ControllerCapability.OBSERVE_PUBLIC,
                ControllerCapability.OBSERVE_PRIVATE,
            }
        ),
    )


def cards(*ids: str):
    return tuple(CARD_BY_ID[card_id] for card_id in ids)


def make_state(
    *,
    phase: MatchPhase = MatchPhase.PLAYING,
    turn: Seat = Seat.EAST,
    hands: dict[Seat, tuple[str, ...]] | None = None,
    current_trick: TrickState | None = None,
    tribute: TributeState | None = None,
    current_level: Rank = Rank.TWO,
) -> MatchState:
    all_hands = {
        Seat.EAST: ("D1-S-3",),
        Seat.SOUTH: ("D1-S-4",),
        Seat.WEST: ("D1-S-5",),
        Seat.NORTH: ("D1-S-6",),
    }
    if hands is not None:
        all_hands.update(hands)
    deal = DealState(
        hands=all_hands,
        active_seats=frozenset(SEATS),
        finish_order=(),
        leader=turn,
        turn=turn,
        current_trick=current_trick or TrickState(lead_seat=turn),
        tribute=tribute,
    )
    return MatchState(
        table_id="table-1",
        phase=phase,
        current_level=current_level,
        seats={seat: player(seat) for seat in SEATS},
        controllers={seat: controller(seat) for seat in SEATS},
        ready_seats=frozenset(SEATS),
        deal=deal,
    )


class LegalActionTests(unittest.TestCase):
    def test_lead_includes_declared_straight_flush(self) -> None:
        straight_flush = ("D1-S-3", "D1-S-4", "D1-S-5", "D1-S-6", "D1-S-7")
        state = make_state(hands={Seat.EAST: straight_flush})

        actions = legal_actions_for_state(state, Seat.EAST)

        self.assertIn(
            ActionCandidate(
                kind="play_cards",
                card_ids=straight_flush,
                declared_type=HandType.STRAIGHT_FLUSH.value,
                hand_type=HandType.STRAIGHT_FLUSH,
                primary_rank=Rank.SEVEN,
                length=5,
            ),
            actions,
        )
        self._assert_reducer_accepts_all(state, Seat.EAST, actions)

    def test_follow_includes_pass_and_only_beating_plays(self) -> None:
        current = parse_hand(cards("D1-S-A"))
        state = make_state(
            turn=Seat.NORTH,
            current_trick=TrickState(lead_seat=Seat.EAST, last_play=current, last_play_seat=Seat.EAST),
            hands={
                Seat.NORTH: (
                    "D1-S-3",
                    "D1-BJ",
                    "D1-S-4",
                    "D2-S-4",
                    "D1-H-4",
                    "D2-H-4",
                )
            },
        )

        actions = legal_actions_for_state(state, Seat.NORTH)

        self.assertIsInstance(actions[0].to_command(controller(Seat.NORTH).id, Seat.NORTH), Pass)
        self.assertIn(("D1-BJ",), [action.card_ids for action in actions])
        self.assertNotIn(("D1-S-3",), [action.card_ids for action in actions])
        self.assertIn(HandType.BOMB, [action.hand_type for action in actions])
        self._assert_reducer_accepts_all(state, Seat.NORTH, actions)

    def test_lead_includes_large_bomb(self) -> None:
        seven_card_bomb = (
            "D1-S-3",
            "D1-H-3",
            "D1-D-3",
            "D1-C-3",
            "D2-S-3",
            "D2-H-3",
            "D2-D-3",
        )
        state = make_state(hands={Seat.EAST: seven_card_bomb})

        actions = legal_actions_for_state(state, Seat.EAST)

        self.assertIn(
            ActionCandidate(
                kind="play_cards",
                card_ids=seven_card_bomb,
                declared_type=HandType.BOMB.value,
                hand_type=HandType.BOMB,
                primary_rank=Rank.THREE,
                length=7,
            ),
            actions,
        )
        self._assert_reducer_accepts_all(state, Seat.EAST, actions)

    def test_tribute_actions_use_highest_eligible_cards(self) -> None:
        obligation = TributeObligation(giver=Seat.SOUTH, receiver=Seat.EAST)
        state = make_state(
            phase=MatchPhase.TRIBUTE,
            turn=Seat.SOUTH,
            tribute=TributeState(obligations=(obligation,), leader_after=Seat.SOUTH),
            hands={Seat.SOUTH: ("D1-S-A", "D2-S-A", "D1-H-2", "D1-S-K")},
        )

        actions = legal_actions_for_state(state, Seat.SOUTH)

        self.assertEqual([action.to_payload() for action in actions], [
            {"type": "submit_tribute", "card_id": "D1-S-A"},
            {"type": "submit_tribute", "card_id": "D2-S-A"},
        ])
        self._assert_reducer_accepts_all(state, Seat.SOUTH, actions)

    def test_partner_return_actions_are_limited_to_ten_or_lower(self) -> None:
        obligation = TributeObligation(giver=Seat.EAST, receiver=Seat.WEST, tribute_card_id="D1-S-A")
        state = make_state(
            phase=MatchPhase.TRIBUTE,
            turn=Seat.WEST,
            tribute=TributeState(obligations=(obligation,), leader_after=Seat.EAST),
            hands={Seat.WEST: ("D1-S-A", "D1-S-10", "D1-S-9")},
        )

        actions = legal_actions_for_state(state, Seat.WEST)

        self.assertEqual([action.to_payload() for action in actions], [
            {"type": "return_tribute", "card_id": "D1-S-10"},
            {"type": "return_tribute", "card_id": "D1-S-9"},
        ])
        self._assert_reducer_accepts_all(state, Seat.WEST, actions)

    def test_opponent_return_actions_include_all_owned_cards(self) -> None:
        obligation = TributeObligation(giver=Seat.EAST, receiver=Seat.SOUTH, tribute_card_id="D1-S-A")
        state = make_state(
            phase=MatchPhase.TRIBUTE,
            turn=Seat.SOUTH,
            tribute=TributeState(obligations=(obligation,), leader_after=Seat.EAST),
            hands={Seat.SOUTH: ("D1-S-A", "D1-S-10")},
        )

        actions = legal_actions_for_state(state, Seat.SOUTH)

        self.assertEqual([action.to_payload() for action in actions], [
            {"type": "return_tribute", "card_id": "D1-S-A"},
            {"type": "return_tribute", "card_id": "D1-S-10"},
        ])
        self._assert_reducer_accepts_all(state, Seat.SOUTH, actions)

    def test_seat_without_turn_has_no_actions(self) -> None:
        state = make_state(turn=Seat.EAST)

        self.assertEqual(legal_actions_for_state(state, Seat.SOUTH), ())

    def test_candidate_generation_reuses_cached_card_groups(self) -> None:
        state = make_state(hands={Seat.EAST: ("D1-S-3", "D1-H-3", "D1-S-4", "D1-H-4", "D1-S-5")})

        legal_actions_for_state(state, Seat.EAST)
        before = candidate_generation_cache_info()
        legal_actions_for_state(state, Seat.EAST)
        after = candidate_generation_cache_info()

        self.assertGreater(after["hits"], before["hits"])

    def test_snapshot_play_or_pass_uses_public_trick_and_private_hand(self) -> None:
        current = parse_hand(cards("D1-S-A"))
        state = make_state(
            turn=Seat.NORTH,
            current_trick=TrickState(lead_seat=Seat.EAST, last_play=current, last_play_seat=Seat.EAST),
            hands={Seat.NORTH: ("D1-S-3", "D1-BJ")},
        )
        snapshot = SeatSnapshot(
            public=public_snapshot(state, acting_seat=Seat.NORTH),
            seat=Seat.NORTH,
            hand=("D1-S-3", "D1-BJ"),
            legal_action="play_or_pass",
        )

        actions = legal_actions_for_snapshot(snapshot)

        self.assertEqual(actions[0].to_payload(), {"type": "pass"})
        self.assertIn(("D1-BJ",), [action.card_ids for action in actions])
        self.assertNotIn(("D1-S-3",), [action.card_ids for action in actions])

    def _assert_reducer_accepts_all(
        self,
        state: MatchState,
        seat: Seat,
        actions: tuple[ActionCandidate, ...],
    ) -> None:
        for action in actions:
            with self.subTest(action=action):
                result = reduce_command(state, action.to_command(controller(seat).id, seat))
                self.assertIsNone(result.rejection)


if __name__ == "__main__":
    unittest.main()
