from __future__ import annotations

import unittest
from dataclasses import replace

from server.domain.commands import JoinTable, Pass, PlayCards, Ready, ReturnTribute, StartMatch, SubmitTribute
from server.domain.controllers import ControllerCapability, ControllerKind, ControllerRef, PlayerKind, PlayerRef
from server.domain.events import RejectCode
from server.domain.reducer import reduce_command
from server.domain.cards import Rank
from server.domain.seats import SEATS, Seat, Team
from server.domain.state import DealResult, MatchPhase, MatchState, ScoreState


def player(seat: Seat) -> PlayerRef:
    return PlayerRef(id=f"player-{seat.value}", display_name=f"Player {seat.value}", kind=PlayerKind.HUMAN)


def controller(seat: Seat) -> ControllerRef:
    return ControllerRef(
        id=f"controller-{seat.value}",
        kind=ControllerKind.HUMAN_WS,
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


def started_state() -> MatchState:
    state = MatchState(table_id="table-1")
    for seat in SEATS:
        result = reduce_command(state, JoinTable(player(seat), controller(seat), seat))
        assert result.rejection is None
        state = result.state
    for seat in SEATS:
        result = reduce_command(state, Ready(controller(seat).id, seat))
        assert result.rejection is None
        state = result.state
    result = reduce_command(state, StartMatch(seed="fixed-seed"))
    assert result.rejection is None
    return result.state


class ReducerTests(unittest.TestCase):
    def test_start_match_deals_cards_and_enters_playing(self) -> None:
        state = MatchState(table_id="table-1")
        for seat in SEATS:
            result = reduce_command(state, JoinTable(player(seat), controller(seat), seat))
            assert result.rejection is None
            state = result.state
        for seat in SEATS:
            result = reduce_command(state, Ready(controller(seat).id, seat))
            assert result.rejection is None
            state = result.state
        result = reduce_command(state, StartMatch(seed="fixed-seed"))
        self.assertIsNone(result.rejection)
        state = result.state

        self.assertEqual(state.phase, MatchPhase.PLAYING)
        self.assertIsNotNone(state.deal)
        assert state.deal is not None
        self.assertEqual([len(state.deal.hand_for(seat)) for seat in SEATS], [27, 27, 27, 27])
        self.assertEqual(state.deal.turn, Seat.EAST)
        self.assertEqual([event.type for event in result.events], ["MatchStarted", "DealStarted", "CardsDealt"])
        self.assertEqual(set(result.events[-1].payload["hands"]), {seat.value for seat in SEATS})

    def test_start_match_after_match_complete_resets_levels_for_new_match(self) -> None:
        state = started_state()
        state = replace(
            state,
            phase=MatchPhase.MATCH_COMPLETE,
            current_level=Rank.ACE,
            last_deal_result=DealResult(
                finish_order=(Seat.EAST, Seat.WEST, Seat.SOUTH, Seat.NORTH),
                winning_team=Team.EAST_WEST,
                advance_count=1,
                previous_level=Rank.ACE,
                next_level=Rank.ACE,
                match_complete=True,
            ),
            scores=ScoreState(level_by_team={Team.EAST_WEST: Rank.ACE, Team.SOUTH_NORTH: Rank.SEVEN}),
        )
        for seat in SEATS:
            result = reduce_command(state, JoinTable(player(seat), controller(seat), seat))
            self.assertIsNone(result.rejection)
            state = result.state
            result = reduce_command(state, Ready(controller(seat).id, seat))
            self.assertIsNone(result.rejection)
            state = result.state

        result = reduce_command(state, StartMatch(seed="next-match"))

        self.assertIsNone(result.rejection)
        self.assertEqual(result.state.phase, MatchPhase.PLAYING)
        self.assertEqual(result.state.current_level, Rank.TWO)
        self.assertEqual(result.state.scores.level_by_team[Team.EAST_WEST], Rank.TWO)
        self.assertEqual(result.state.scores.level_by_team[Team.SOUTH_NORTH], Rank.TWO)
        self.assertIsNone(result.state.last_deal_result)
        self.assertEqual([event.type for event in result.events], ["MatchStarted", "DealStarted", "CardsDealt"])
        self.assertEqual(len(result.events[-1].payload["hands"][Seat.EAST.value]), 27)

    def test_rejects_play_from_unattached_controller(self) -> None:
        state = started_state()
        assert state.deal is not None
        card_id = state.deal.hand_for(Seat.EAST)[0]

        result = reduce_command(state, PlayCards("wrong", Seat.EAST, (card_id,)))

        self.assertIsNotNone(result.rejection)
        assert result.rejection is not None
        self.assertEqual(result.rejection.code, RejectCode.CONTROLLER_NOT_ATTACHED)

    def test_rejects_card_not_owned(self) -> None:
        state = started_state()
        assert state.deal is not None
        south_card = state.deal.hand_for(Seat.SOUTH)[0]

        result = reduce_command(state, PlayCards(controller(Seat.EAST).id, Seat.EAST, (south_card,)))

        self.assertIsNotNone(result.rejection)
        assert result.rejection is not None
        self.assertEqual(result.rejection.code, RejectCode.CARD_NOT_OWNED)

    def test_play_then_pass_advances_and_ends_trick(self) -> None:
        state = started_state()
        assert state.deal is not None
        east_card = state.deal.hand_for(Seat.EAST)[0]

        result = reduce_command(state, PlayCards(controller(Seat.EAST).id, Seat.EAST, (east_card,)))
        self.assertIsNone(result.rejection)
        state = result.state

        for seat in (Seat.NORTH, Seat.WEST, Seat.SOUTH):
            result = reduce_command(state, Pass(controller(seat).id, seat))
            self.assertIsNone(result.rejection)
            state = result.state

        assert state.deal is not None
        self.assertEqual(state.deal.turn, Seat.EAST)
        self.assertIsNone(state.deal.current_trick.last_play_seat)
        self.assertEqual(result.events[-1].type, "TrickEnded")
        self.assertEqual([event.seq for event in result.events], list(range(result.events[0].seq, result.events[-1].seq + 1)))

    def test_play_emits_ten_card_report_once(self) -> None:
        state = started_state()
        state = self._give_hands(
            state,
            {
                Seat.EAST: ("D1-S-A", "D1-S-3"),
                Seat.NORTH: ("D1-S-4",),
                Seat.WEST: ("D1-S-5",),
                Seat.SOUTH: ("D1-S-6",),
            },
        )

        result = reduce_command(state, PlayCards(controller(Seat.EAST).id, Seat.EAST, ("D1-S-A",)))

        self.assertIsNone(result.rejection)
        self.assertIn("TenCardReport", [event.type for event in result.events])
        self.assertEqual([event.seq for event in result.events], list(range(result.events[0].seq, result.events[-1].seq + 1)))
        state = result.state
        assert state.deal is not None
        self.assertIn(Seat.EAST, state.deal.report_10_done)

    def test_play_emits_player_finished(self) -> None:
        state = started_state()
        state = self._give_hands(
            state,
            {
                Seat.EAST: ("D1-S-A",),
                Seat.NORTH: ("D1-S-4",),
                Seat.WEST: ("D1-S-5",),
                Seat.SOUTH: ("D1-S-6",),
            },
        )

        result = reduce_command(state, PlayCards(controller(Seat.EAST).id, Seat.EAST, ("D1-S-A",)))

        self.assertIsNone(result.rejection)
        self.assertEqual([event.type for event in result.events], ["CardsPlayed", "TenCardReport", "PlayerFinished"])
        self.assertEqual(result.events[-1].payload["position"], 1)

    def test_rejects_ambiguous_hand_without_declaration(self) -> None:
        state = started_state()
        state = self._give_hands(
            state,
            {
                Seat.EAST: ("D1-S-3", "D1-S-4", "D1-S-5", "D1-S-6", "D1-S-7"),
                Seat.NORTH: ("D1-S-8",),
                Seat.WEST: ("D1-S-9",),
                Seat.SOUTH: ("D1-S-10",),
            },
        )

        result = reduce_command(
            state,
            PlayCards(
                controller(Seat.EAST).id,
                Seat.EAST,
                ("D1-S-3", "D1-S-4", "D1-S-5", "D1-S-6", "D1-S-7"),
            ),
        )

        self.assertIsNotNone(result.rejection)
        assert result.rejection is not None
        self.assertEqual(result.rejection.code, RejectCode.AMBIGUOUS_WILD_CARD_DECLARATION)

    def test_finished_player_final_trick_lead_borrows_to_partner(self) -> None:
        state = started_state()
        assert state.deal is not None
        state = self._give_hands(
            state,
            {
                Seat.EAST: ("D1-S-A",),
                Seat.NORTH: ("D1-S-3",),
                Seat.WEST: ("D1-S-4",),
                Seat.SOUTH: ("D1-S-5",),
            },
        )

        result = reduce_command(state, PlayCards(controller(Seat.EAST).id, Seat.EAST, ("D1-S-A",)))
        self.assertIsNone(result.rejection)
        state = result.state
        for seat in (Seat.NORTH, Seat.WEST, Seat.SOUTH):
            result = reduce_command(state, Pass(controller(seat).id, seat))
            self.assertIsNone(result.rejection)
            state = result.state

        assert state.deal is not None
        self.assertEqual(state.deal.turn, Seat.WEST)

    def test_double_down_completes_deal_and_advances_three_levels(self) -> None:
        state = started_state()
        state = self._give_hands(
            state,
            {
                Seat.EAST: ("D1-S-A",),
                Seat.NORTH: ("D1-S-3",),
                Seat.WEST: ("D1-S-K",),
                Seat.SOUTH: ("D1-S-4",),
            },
        )

        result = reduce_command(state, PlayCards(controller(Seat.EAST).id, Seat.EAST, ("D1-S-A",)))
        self.assertIsNone(result.rejection)
        state = result.state
        for seat in (Seat.NORTH, Seat.WEST, Seat.SOUTH):
            state = reduce_command(state, Pass(controller(seat).id, seat)).state

        result = reduce_command(state, PlayCards(controller(Seat.WEST).id, Seat.WEST, ("D1-S-K",)))

        self.assertIsNone(result.rejection)
        state = result.state
        self.assertEqual(state.phase, MatchPhase.DEAL_COMPLETE)
        self.assertIsNotNone(state.last_deal_result)
        assert state.last_deal_result is not None
        self.assertEqual(state.last_deal_result.finish_order[:2], (Seat.EAST, Seat.WEST))
        self.assertEqual(state.scores.level_by_team[state.last_deal_result.winning_team], Rank.FIVE)

    def test_next_deal_enters_tribute_and_completes_exchange(self) -> None:
        state = self._completed_normal_deal_state()

        result = reduce_command(state, StartMatch(seed="tribute-seed"))

        self.assertIsNone(result.rejection)
        state = result.state
        self.assertEqual(state.phase, MatchPhase.TRIBUTE)
        assert state.deal is not None and state.deal.tribute is not None
        obligation = state.deal.tribute.obligations[0]
        self.assertEqual(state.deal.turn, obligation.giver)
        # Use the reducer's validation by finding an accepted highest card.
        for candidate in state.deal.hand_for(obligation.giver):
            paid = reduce_command(state, SubmitTribute(controller(obligation.giver).id, obligation.giver, candidate))
            if paid.rejection is None:
                state = paid.state
                break
        else:
            self.fail("no tribute card accepted")

        assert state.deal is not None and state.deal.tribute is not None
        self.assertEqual(state.deal.turn, obligation.receiver)
        receiver = obligation.receiver
        return_card = next(card_id for card_id in state.deal.hand_for(receiver) if not card_id.endswith("-BJ"))
        result = reduce_command(state, ReturnTribute(controller(receiver).id, receiver, return_card))

        self.assertIsNone(result.rejection)
        state = result.state
        self.assertEqual(state.phase, MatchPhase.PLAYING)
        assert state.deal is not None
        self.assertIsNone(state.deal.tribute)
        self.assertEqual(state.deal.turn, obligation.giver)

    def test_rejects_invalid_hand_type(self) -> None:
        state = started_state()
        assert state.deal is not None
        hand = state.deal.hand_for(Seat.EAST)

        result = reduce_command(state, PlayCards(controller(Seat.EAST).id, Seat.EAST, (hand[0], hand[1])))

        if result.rejection is None:
            self.skipTest("seed produced a legal opening pair")
        self.assertEqual(result.rejection.code, RejectCode.INVALID_HAND_TYPE)

    def test_rejects_following_play_that_does_not_beat_current_hand(self) -> None:
        state = started_state()
        assert state.deal is not None
        state = self._give_hands(
            state,
            {
                Seat.EAST: ("D1-S-A",),
                Seat.NORTH: ("D1-S-3",),
                Seat.WEST: ("D1-S-4",),
                Seat.SOUTH: ("D1-S-5",),
            },
        )

        result = reduce_command(state, PlayCards(controller(Seat.EAST).id, Seat.EAST, ("D1-S-A",)))
        self.assertIsNone(result.rejection)
        state = result.state

        result = reduce_command(state, PlayCards(controller(Seat.NORTH).id, Seat.NORTH, ("D1-S-3",)))

        self.assertIsNotNone(result.rejection)
        assert result.rejection is not None
        self.assertEqual(result.rejection.code, RejectCode.DOES_NOT_BEAT_CURRENT_HAND)

    def _give_hands(self, state: MatchState, hands: dict[Seat, tuple[str, ...]]) -> MatchState:
        from dataclasses import replace

        assert state.deal is not None
        return replace(state, deal=replace(state.deal, hands=hands))

    def _completed_normal_deal_state(self) -> MatchState:
        from dataclasses import replace

        state = started_state()
        result = DealResult(
            finish_order=(Seat.EAST, Seat.NORTH, Seat.WEST, Seat.SOUTH),
            winning_team=Team.EAST_WEST,
            advance_count=2,
            previous_level=Rank.TWO,
            next_level=Rank.FOUR,
            match_complete=False,
        )
        return replace(
            state,
            phase=MatchPhase.DEAL_COMPLETE,
            last_deal_result=result,
            current_level=Rank.FOUR,
        )


if __name__ == "__main__":
    unittest.main()
