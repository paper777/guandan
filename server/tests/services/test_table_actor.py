from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from guandan.domain.commands import JoinTable, PlayCards, Ready, StartMatch
from guandan.domain.controllers import ControllerCapability, ControllerKind, ControllerRef, PlayerKind, PlayerRef
from guandan.domain.seats import SEATS, Seat
from guandan.persistence.sqlite_store import SQLiteEventStore
from guandan.services.table_actor import TableActor
from guandan.services.table_config import TableConfig


class TableActorTests(unittest.TestCase):
    def test_dispatch_persists_events_and_replays_idempotent_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteEventStore(Path(tmp) / "events.db")
            actor = TableActor("table-1", event_store=store)
            player = PlayerRef("p-E", "E", PlayerKind.HUMAN)
            controller = ControllerRef(
                "c-E",
                ControllerKind.HUMAN_WS,
                Seat.EAST,
                "p-E",
                frozenset({ControllerCapability.PLAY, ControllerCapability.OBSERVE_PRIVATE}),
            )
            command = JoinTable(player, controller, Seat.EAST)

            first = actor.dispatch(command, controller_id="c-E", request_id="r-1")
            second = actor.dispatch(command, controller_id="c-E", request_id="r-1")
            store.close()

        self.assertIsNone(first.rejection)
        self.assertTrue(second.replayed)
        self.assertEqual([event.seq for event in first.events], [1])
        self.assertEqual(second.events, first.events)

    def test_dispatch_persists_multi_event_start_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteEventStore(Path(tmp) / "events.db")
            actor = TableActor("table-1", event_store=store)
            for seat in SEATS:
                actor.dispatch(JoinTable(player(seat), controller(seat), seat))
            for seat in SEATS:
                actor.dispatch(Ready(controller(seat).id, seat))

            result = actor.dispatch(StartMatch(seed="fixed-seed"), controller_id="c-E", request_id="start-1")
            loaded = store.load_events(actor.match_id)
            store.close()

        self.assertIsNone(result.rejection)
        self.assertEqual(
            [event.type for event in result.events],
            ["MatchStarted", "DealStarted", "CardsDealt", "ActionPrompted"],
        )
        self.assertEqual([event.seq for event in result.events], [9, 10, 11, 12])
        self.assertEqual([event.seq for event in loaded[-4:]], [9, 10, 11, 12])
        self.assertEqual(len(loaded[-2].payload["hands"][Seat.EAST.value]), 27)
        self.assertEqual(loaded[-1].payload["timeout_seconds"], 45)

    def test_initializes_state_from_existing_event_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteEventStore(Path(tmp) / "events.db")
            actor = TableActor("table-1", match_id="match-1", event_store=store)
            for seat in SEATS:
                actor.dispatch(JoinTable(player(seat), controller(seat), seat))
            for seat in SEATS:
                actor.dispatch(Ready(controller(seat).id, seat))
            actor.dispatch(StartMatch(seed="fixed-seed"))

            restored = TableActor("table-1", match_id="match-1", event_store=store)
            store.close()

        self.assertEqual(restored.state, actor.state)

    def test_async_dispatch_serializes_commands(self) -> None:
        async def run() -> TableActor:
            actor = TableActor("table-1")
            await asyncio.gather(
                *[actor.dispatch_async(JoinTable(player(seat), controller(seat), seat)) for seat in SEATS]
            )
            return actor

        actor = asyncio.run(run())

        self.assertEqual(actor.state.event_seq, 4)
        self.assertEqual(set(actor.state.seats), set(SEATS))

    def test_table_config_defaults_to_45_second_timeout(self) -> None:
        actor = TableActor("table-1")

        self.assertEqual(actor.config.action_timeout_seconds, 45)

    def test_start_match_schedules_action_prompt_deadline(self) -> None:
        actor = started_actor_sync()

        snapshot = actor.public_snapshot()

        self.assertIsNotNone(actor.active_prompt)
        self.assertEqual(actor.active_prompt.seat, Seat.EAST)
        self.assertEqual(actor.active_prompt.kind, "lead")
        self.assertEqual(snapshot.acting_seat, Seat.EAST)
        self.assertEqual(snapshot.action_timeout_seconds, 45)
        self.assertEqual(snapshot.action_deadline_epoch_ms, actor.active_prompt.deadline_epoch_ms)

    def test_human_command_resets_prompt_deadline(self) -> None:
        async def run() -> tuple[str, str, tuple[str, ...]]:
            actor = await started_actor_async(clock=StepClock())
            first_prompt_id = actor.active_prompt.prompt_id
            assert actor.state.deal is not None
            card_id = actor.state.deal.hand_for(Seat.EAST)[0]

            result = await actor.dispatch_async(PlayCards("c-E", Seat.EAST, (card_id,)))

            assert actor.active_prompt is not None
            return first_prompt_id, actor.active_prompt.prompt_id, tuple(event.type for event in result.events)

        first_prompt_id, next_prompt_id, event_types = asyncio.run(run())

        self.assertNotEqual(first_prompt_id, next_prompt_id)
        self.assertEqual(event_types[-1], "ActionPrompted")

    def test_timeout_auto_plays_smallest_card_when_leading(self) -> None:
        async def run() -> tuple[TableActor, tuple[str, ...]]:
            sleeper = ManualSleeper()
            actor = await started_actor_async(clock=StepClock(), sleeper=sleeper)
            await asyncio.sleep(0)
            assert actor.state.deal is not None
            starting_east_hand = actor.state.deal.hand_for(Seat.EAST)

            sleeper.fire_next()
            await asyncio.sleep(0)
            await asyncio.sleep(0)

            assert actor.last_timeout_result is not None
            return actor, starting_east_hand

        actor, starting_east_hand = asyncio.run(run())

        self.assertIsNotNone(actor.last_timeout_result)
        assert actor.last_timeout_result is not None
        event_types = [event.type for event in actor.last_timeout_result.events]
        self.assertIn("ActionTimedOut", event_types)
        self.assertIn("CardsPlayed", event_types)
        self.assertIn("TimeoutFallbackApplied", event_types)
        self.assertEqual(actor.state.deal.turn, Seat.NORTH)
        self.assertLess(len(actor.state.deal.hand_for(Seat.EAST)), len(starting_east_hand))

    def test_local_bot_auto_acts_when_prompted(self) -> None:
        async def run() -> TableActor:
            actor = TableActor("table-1")
            await actor.dispatch_async(JoinTable(player(Seat.EAST), local_bot_controller(Seat.EAST), Seat.EAST))
            for seat in (Seat.NORTH, Seat.WEST, Seat.SOUTH):
                await actor.dispatch_async(JoinTable(player(seat), controller(seat), seat))
            for seat in SEATS:
                await actor.dispatch_async(Ready(actor.state.controllers[seat].id, seat))
            await actor.dispatch_async(StartMatch(seed="fixed-seed"))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return actor

        actor = asyncio.run(run())

        self.assertIsNotNone(actor.last_npc_result)
        assert actor.last_npc_result is not None
        self.assertIsNone(actor.last_npc_result.rejection)
        self.assertIn("CardsPlayed", [event.type for event in actor.last_npc_result.events])
        self.assertEqual(actor.state.deal.turn, Seat.NORTH)
        self.assertEqual(len(actor.state.deal.hand_for(Seat.EAST)), 26)

def player(seat: Seat) -> PlayerRef:
    return PlayerRef(f"p-{seat.value}", seat.value, PlayerKind.HUMAN)


def controller(seat: Seat) -> ControllerRef:
    return ControllerRef(
        f"c-{seat.value}",
        ControllerKind.HUMAN_WS,
        seat,
        f"p-{seat.value}",
        frozenset(
            {
                ControllerCapability.PLAY,
                ControllerCapability.OBSERVE_PUBLIC,
                ControllerCapability.OBSERVE_PRIVATE,
            }
        ),
    )


def local_bot_controller(seat: Seat) -> ControllerRef:
    return ControllerRef(
        f"bot-c-{seat.value}",
        ControllerKind.LOCAL_BOT,
        seat,
        f"p-{seat.value}",
        frozenset(
            {
                ControllerCapability.PLAY,
                ControllerCapability.OBSERVE_PUBLIC,
                ControllerCapability.OBSERVE_PRIVATE,
                ControllerCapability.AUTO_READY,
            }
        ),
    )


def started_actor_sync() -> TableActor:
    actor = TableActor("table-1")
    for seat in SEATS:
        actor.dispatch(JoinTable(player(seat), controller(seat), seat))
    for seat in SEATS:
        actor.dispatch(Ready(controller(seat).id, seat))
    actor.dispatch(StartMatch(seed="fixed-seed"))
    return actor


async def started_actor_async(
    *,
    clock=None,
    sleeper=None,
) -> TableActor:
    actor = TableActor("table-1", config=TableConfig(), clock=clock, sleeper=sleeper)
    for seat in SEATS:
        await actor.dispatch_async(JoinTable(player(seat), controller(seat), seat))
    for seat in SEATS:
        await actor.dispatch_async(Ready(controller(seat).id, seat))
    await actor.dispatch_async(StartMatch(seed="fixed-seed"))
    return actor


class StepClock:
    def __init__(self) -> None:
        self.value = 1_700_000_000_000

    def __call__(self) -> int:
        self.value += 100
        return self.value


class ManualSleeper:
    def __init__(self) -> None:
        self.futures = []

    async def __call__(self, delay: float) -> None:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.futures.append(future)
        await future

    def fire_next(self) -> None:
        if not self.futures:
            raise AssertionError("no sleeper is waiting")
        self.futures.pop(0).set_result(None)


if __name__ == "__main__":
    unittest.main()
