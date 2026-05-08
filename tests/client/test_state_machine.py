from __future__ import annotations

import unittest
from types import SimpleNamespace

from client.session import Session
from client.state_machine import StateMachine, trigger_role_observers
from client.types import SeatMember, SeatRole, Table


class RotatingBroker:
    def __init__(self) -> None:
        self.seats = {}
        self.rotations = 0

    def rotate_seats_after_match(self):
        self.rotations += 1
        return {"E": "S", "S": "E"}

    def join_and_ready_all(self):
        return None


class StateMachineTransitionTests(unittest.TestCase):
    def test_records_deal_and_multiple_match_transitions(self) -> None:
        table = Table("table-1")
        broker = RotatingBroker()
        session = Session("table-1", "E", "human-controller-E", broker, {}, table=table)
        table.members_for("E").player = SeatMember(SeatRole.PLAYER, "Human", is_human=True)
        table.members_for("S").player = SeatMember(SeatRole.PLAYER, "Bot")
        output: list[str] = []
        machine = StateMachine(
            args=SimpleNamespace(max_bot_actions=4),
            client=object(),
            session=session,
            input_fn=lambda prompt: "quit",
            emit=output.append,
        )

        machine._record_table_transitions(
            {"table_id": "table-1", "phase": "PLAYING", "deal_id": 1, "current_level": "2"}
        )
        machine._record_table_transitions(
            {"table_id": "table-1", "phase": "DEAL_COMPLETE", "deal_id": 1, "current_level": "2"}
        )
        machine._record_table_transitions({"table_id": "table-1", "phase": "MATCH_COMPLETE", "deal_id": 1})
        machine._record_table_transitions(
            {"table_id": "table-1", "phase": "PLAYING", "deal_id": 1, "current_level": "3"}
        )
        machine._record_table_transitions({"table_id": "table-1", "phase": "MATCH_COMPLETE", "deal_id": 1})

        self.assertEqual([match.match_id for match in table.history_matches], ["table-1-match-1", "table-1-match-2"])
        self.assertEqual(broker.rotations, 2)
        self.assertEqual(session.human_seat, "E")
        self.assertIn("Match 1 started.", output)
        self.assertIn("Deal 1 complete.", output)
        self.assertIn("Match 2 complete.", output)
        self.assertTrue(any(line.startswith("Seat roles rotated for next match:") for line in output))

    def test_match_complete_rotates_rejoins_and_starts_next_match(self) -> None:
        class BrokerSeat:
            def __init__(self, seat: str, display_name: str) -> None:
                self.seat = seat
                self.display_name = display_name
                self.controller_id = ""
                self.profile_key = display_name
                self.policy = object()

        class RestartBroker:
            def __init__(self) -> None:
                self.seats = {"S": BrokerSeat("S", "Bot")}
                self.joined = 0

            def rotate_seats_after_match(self):
                seat = self.seats.pop("S")
                seat.seat = "E"
                seat.controller_id = ""
                self.seats["E"] = seat
                return {"E": "S", "S": "E"}

            def join_and_ready_all(self):
                self.joined += 1
                self.seats["E"].controller_id = "agent-controller-E"

        class Client:
            def __init__(self) -> None:
                self.calls: list[tuple] = []

            def join_human(self, table_id, seat, *, player_id=None, controller_id=None, display_name=None):
                self.calls.append(("join_human", table_id, seat, player_id, controller_id, display_name))
                return {"controller_id": controller_id}

            def ready(self, table_id, seat, controller_id):
                self.calls.append(("ready", table_id, seat, controller_id))
                return {}

            def start(self, table_id):
                self.calls.append(("start", table_id))
                return {
                    "events": [{"seq": 11, "type": "MatchStarted", "payload": {"table_id": table_id}}],
                    "snapshot": {
                        "table_id": table_id,
                        "phase": "PLAYING",
                        "event_seq": 11,
                        "deal_id": 1,
                        "current_turn": None,
                        "level_by_team": {"EW": "2", "SN": "2"},
                    },
                }

            def table_snapshot(self, table_id):
                return {
                    "table_id": table_id,
                    "phase": "PLAYING",
                    "event_seq": 11,
                    "deal_id": 1,
                    "current_turn": None,
                }

        table = Table("table-1")
        broker = RestartBroker()
        session = Session("table-1", "E", "human-controller-E", broker, {}, table=table)
        table.members_for("E").player = SeatMember(
            SeatRole.PLAYER,
            "Human",
            controller_id="human-controller-E",
            is_human=True,
        )
        table.members_for("S").player = SeatMember(SeatRole.PLAYER, "Bot")
        output: list[str] = []
        client = Client()
        machine = StateMachine(
            args=SimpleNamespace(max_bot_actions=4),
            client=client,
            session=session,
            input_fn=lambda prompt: "quit",
            emit=output.append,
        )

        next_snapshot = machine._start_and_drive_next_match(
            {
                "table_id": "table-1",
                "phase": "MATCH_COMPLETE",
                "event_seq": 10,
                "deal_id": 7,
                "current_level": "A",
                "level_by_team": {"EW": "A", "SN": "7"},
            }
        )

        self.assertEqual(next_snapshot["phase"], "PLAYING")
        self.assertEqual(session.human_seat, "S")
        self.assertEqual(session.human_controller_id, "human-controller-S")
        self.assertEqual(broker.joined, 1)
        self.assertIn(("join_human", "table-1", "S", "human-S", "human-controller-S", "Human"), client.calls)
        self.assertIn(("ready", "table-1", "S", "human-controller-S"), client.calls)
        self.assertIn(("start", "table-1"), client.calls)
        self.assertIn("Match 1 complete.", output)
        self.assertIn("Match 2 started.", output)

    def test_role_observers_run_gossiper_then_witnesses_after_player_action(self) -> None:
        observed: list[tuple[str, str]] = []

        class ObserverPolicy:
            def __init__(self, name: str) -> None:
                self.name = name

            def choose_action(self, request):
                return {"type": "pass"}

            def observe_action(self, observation):
                observed.append((self.name, observation["observer_role"]))

        table = Table("table-1")
        members = table.members_for("E")
        members.player = SeatMember(SeatRole.PLAYER, "Player")
        members.gossiper = SeatMember(SeatRole.GOSSIPER, "Gossiper", policy=ObserverPolicy("gossiper"))
        members.witnesses.append(SeatMember(SeatRole.WITNESS, "Witness 1", policy=ObserverPolicy("witness-1")))
        members.witnesses.append(SeatMember(SeatRole.WITNESS, "Witness 2", policy=ObserverPolicy("witness-2")))
        session = Session("table-1", "E", "human-controller-E", RotatingBroker(), {}, table=table)

        trigger_role_observers(
            session,
            "E",
            {"type": "pass"},
            {"events": [], "snapshot": {"seats": {"E": {"display_name": "Player"}}}},
        )

        self.assertEqual(
            observed,
            [("gossiper", "gossiper"), ("witness-1", "witness"), ("witness-2", "witness")],
        )


if __name__ == "__main__":
    unittest.main()
