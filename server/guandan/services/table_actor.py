from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from guandan.domain.cards import CARD_BY_ID, Rank, is_red_heart_level_card
from guandan.domain.commands import Command, Pass, PlayCards, ReturnTribute, SubmitTribute
from guandan.domain.comparator import RankContext
from guandan.domain.controllers import ControllerKind
from guandan.domain.events import CommandRejected, Event, RejectCode
from guandan.domain.reducer import reduce_command
from guandan.domain.seats import Seat, partner_for_seat
from guandan.domain.state import MatchPhase, MatchState
from guandan.persistence.sqlite_store import SQLiteEventStore
from guandan.services.replay import rebuild_state_from_events
from guandan.services.snapshots import PublicTableSnapshot, SeatSnapshot, public_snapshot, seat_snapshot
from guandan.services.table_config import TableConfig, TimeoutFallback
from npc.common.client import ActionRequest, JsonObject
from npc.dummy_bot.policy import DummyBotPolicy


@dataclass(frozen=True, slots=True)
class ActorResult:
    events: tuple[Event, ...]
    rejection: CommandRejected | None = None
    replayed: bool = False


PromptKind = Literal["lead", "play_or_pass", "tribute", "return_tribute"]
Clock = Callable[[], int]
Sleeper = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class PromptRequirement:
    seat: Seat
    kind: PromptKind
    state_seq: int

    @property
    def prompt_id(self) -> str:
        return f"{self.state_seq}:{self.seat.value}:{self.kind}"


@dataclass(frozen=True, slots=True)
class ActivePrompt:
    prompt_id: str
    seat: Seat
    kind: PromptKind
    started_epoch_ms: int
    deadline_epoch_ms: int
    state_seq: int


class TableActor:
    def __init__(
        self,
        table_id: str,
        match_id: str | None = None,
        event_store: SQLiteEventStore | None = None,
        config: TableConfig | None = None,
        clock: Clock | None = None,
        sleeper: Sleeper | None = None,
    ) -> None:
        self.table_id = table_id
        self.match_id = match_id or table_id
        self.state = MatchState(table_id=table_id)
        self.event_store = event_store
        self.config = config or TableConfig()
        self._clock = clock or _system_epoch_ms
        self._sleeper = sleeper or asyncio.sleep
        self._lock = asyncio.Lock()
        self._active_prompt: ActivePrompt | None = None
        self._timeout_task: asyncio.Task[None] | None = None
        self._npc_task: asyncio.Task[None] | None = None
        self.last_timeout_result: ActorResult | None = None
        self.last_npc_result: ActorResult | None = None
        if self.event_store is not None:
            self.event_store.create_match(self.match_id, table_id)
            events = self.event_store.load_events(self.match_id)
            if events:
                self.state = rebuild_state_from_events(table_id, events)
        self._refresh_prompt(schedule_timeout=False, emit_prompt_event=False)

    @property
    def active_prompt(self) -> ActivePrompt | None:
        return self._active_prompt

    def public_snapshot(self) -> PublicTableSnapshot:
        return public_snapshot(
            self.state,
            action_deadline_epoch_ms=self._active_prompt.deadline_epoch_ms if self._active_prompt else None,
            action_timeout_seconds=self.config.action_timeout_seconds,
            acting_seat=self._active_prompt.seat if self._active_prompt else None,
        )

    def seat_snapshot(self, seat: Seat, controller_id: str) -> SeatSnapshot:
        return seat_snapshot(
            self.state,
            seat,
            controller_id,
            action_deadline_epoch_ms=self._active_prompt.deadline_epoch_ms if self._active_prompt else None,
            action_timeout_seconds=self.config.action_timeout_seconds,
            acting_seat=self._active_prompt.seat if self._active_prompt else None,
        )

    async def dispatch_async(
        self,
        command: Command,
        *,
        controller_id: str | None = None,
        request_id: str | None = None,
    ) -> ActorResult:
        async with self._lock:
            return self._dispatch_locked(
                command,
                controller_id=controller_id,
                request_id=request_id,
                schedule_timeout=True,
            )

    def dispatch(
        self,
        command: Command,
        *,
        controller_id: str | None = None,
        request_id: str | None = None,
    ) -> ActorResult:
        return self._dispatch_locked(
            command,
            controller_id=controller_id,
            request_id=request_id,
            schedule_timeout=False,
        )

    def close(self) -> None:
        self._cancel_timeout_task()
        self._cancel_npc_task()

    def _dispatch_locked(
        self,
        command: Command,
        *,
        controller_id: str | None,
        request_id: str | None,
        schedule_timeout: bool,
    ) -> ActorResult:
        if self.event_store is not None and controller_id is not None and request_id is not None:
            existing = self.event_store.find_idempotency(self.match_id, controller_id, request_id)
            if existing is not None:
                first_seq, last_seq = existing
                events = tuple(
                    event for event in self.event_store.load_events(self.match_id) if first_seq <= event.seq <= last_seq
                )
                return ActorResult(events=events, replayed=True)

        result = reduce_command(self.state, command)
        if result.rejection is not None:
            return ActorResult(events=(), rejection=result.rejection)

        self.state = result.state
        events = (*result.events, *self._refresh_prompt(schedule_timeout=schedule_timeout, emit_prompt_event=True))
        if self.event_store is not None and events:
            self.event_store.append_events(self.match_id, events)
            if controller_id is not None and request_id is not None:
                self.event_store.record_idempotency(
                    self.match_id,
                    controller_id,
                    request_id,
                    events[0].seq,
                    events[-1].seq,
                )
        return ActorResult(events=events)

    def _refresh_prompt(self, *, schedule_timeout: bool, emit_prompt_event: bool) -> tuple[Event, ...]:
        requirement = _prompt_requirement(self.state)
        if requirement is None:
            self._active_prompt = None
            self._cancel_timeout_task()
            self._cancel_npc_task()
            return ()

        if self._active_prompt is not None and self._active_prompt.prompt_id == requirement.prompt_id:
            if schedule_timeout and self._timeout_task is None:
                self._schedule_timeout_task()
            if schedule_timeout and self._npc_task is None:
                self._schedule_npc_task()
            return ()

        self._cancel_timeout_task()
        self._cancel_npc_task()
        started = self._clock()
        deadline = started + (self.config.action_timeout_seconds * 1000)
        self._active_prompt = ActivePrompt(
            prompt_id=requirement.prompt_id,
            seat=requirement.seat,
            kind=requirement.kind,
            started_epoch_ms=started,
            deadline_epoch_ms=deadline,
            state_seq=requirement.state_seq,
        )
        if schedule_timeout:
            self._schedule_timeout_task()
            self._schedule_npc_task()
        if not emit_prompt_event:
            return ()
        return (self._service_event("ActionPrompted", self._prompt_payload(self._active_prompt)),)

    def _schedule_timeout_task(self) -> None:
        if self._active_prompt is None:
            return
        prompt_id = self._active_prompt.prompt_id
        delay = max(0.0, (self._active_prompt.deadline_epoch_ms - self._clock()) / 1000)
        self._timeout_task = asyncio.create_task(self._timeout_after(prompt_id, delay))

    def _cancel_timeout_task(self) -> None:
        if self._timeout_task is not None and not self._timeout_task.done():
            self._timeout_task.cancel()
        self._timeout_task = None

    def _schedule_npc_task(self) -> None:
        if self._active_prompt is None:
            return
        controller = self.state.controllers.get(self._active_prompt.seat)
        if controller is None or controller.kind != ControllerKind.LOCAL_BOT:
            return
        self._npc_task = asyncio.create_task(self._run_npc_after(self._active_prompt.prompt_id))

    def _cancel_npc_task(self) -> None:
        current = _current_task_or_none()
        if self._npc_task is not None and not self._npc_task.done() and self._npc_task is not current:
            self._npc_task.cancel()
        self._npc_task = None

    async def _timeout_after(self, prompt_id: str, delay: float) -> None:
        try:
            await self._sleeper(delay)
        except asyncio.CancelledError:
            return
        async with self._lock:
            if self._timeout_task is asyncio.current_task():
                self._timeout_task = None
            self.last_timeout_result = self._apply_timeout_locked(prompt_id)

    async def _run_npc_after(self, prompt_id: str) -> None:
        await asyncio.sleep(0)
        async with self._lock:
            if self._npc_task is asyncio.current_task():
                self._npc_task = None
            prompt = self._active_prompt
            if prompt is None or prompt.prompt_id != prompt_id:
                self.last_npc_result = ActorResult(events=())
                return
            controller = self.state.controllers.get(prompt.seat)
            if controller is None or controller.kind != ControllerKind.LOCAL_BOT:
                self.last_npc_result = ActorResult(events=())
                return
            command = self._npc_command(prompt, controller.id)
            if command is None:
                self.last_npc_result = ActorResult(events=())
                return
            self.last_npc_result = self._dispatch_locked(
                command,
                controller_id=controller.id,
                request_id=f"npc:{prompt.prompt_id}",
                schedule_timeout=True,
            )

    def _apply_timeout_locked(self, prompt_id: str) -> ActorResult:
        prompt = self._active_prompt
        if prompt is None or prompt.prompt_id != prompt_id:
            return ActorResult(events=())

        timeout_event = self._service_event(
            "ActionTimedOut",
            {
                "seat": prompt.seat.value,
                "kind": prompt.kind,
                "deadline_epoch_ms": prompt.deadline_epoch_ms,
                "prompt_id": prompt.prompt_id,
            },
        )
        command = self._fallback_command(prompt)
        if command is None:
            result = ActorResult(
                events=(timeout_event,),
                rejection=CommandRejected(RejectCode.ACTION_TIMEOUT, "timeout fallback could not be built"),
            )
            if self.event_store is not None:
                self.event_store.append_events(self.match_id, result.events)
            return result

        fallback_result = reduce_command(self.state, command)
        if fallback_result.rejection is not None:
            result = ActorResult(events=(timeout_event,), rejection=fallback_result.rejection)
            if self.event_store is not None:
                self.event_store.append_events(self.match_id, result.events)
            return result

        self.state = fallback_result.state
        fallback_events = fallback_result.events
        applied_event = self._service_event(
            "TimeoutFallbackApplied",
            {
                "seat": prompt.seat.value,
                "kind": prompt.kind,
                "fallback": self.config.timeout_fallback.value,
                "command_type": type(command).__name__,
                "event_seq_range": [fallback_events[0].seq, fallback_events[-1].seq] if fallback_events else [],
                "prompt_id": prompt.prompt_id,
            },
        )
        prompt_events = self._refresh_prompt(schedule_timeout=True, emit_prompt_event=True)
        events = (timeout_event, *fallback_events, applied_event, *prompt_events)
        if self.event_store is not None and events:
            self.event_store.append_events(self.match_id, events)
        return ActorResult(events=events)

    def _fallback_command(self, prompt: ActivePrompt) -> Command | None:
        if self.config.timeout_fallback != TimeoutFallback.AUTO_PASS:
            return None
        controller = self.state.controllers.get(prompt.seat)
        if controller is None:
            return None
        if prompt.kind == "play_or_pass":
            return Pass(controller.id, prompt.seat)
        if prompt.kind == "lead":
            card_id = _smallest_card_id(self.state, prompt.seat)
            return PlayCards(controller.id, prompt.seat, (card_id,)) if card_id is not None else None
        if prompt.kind == "tribute":
            card_id = _highest_eligible_tribute_card(self.state, prompt.seat)
            return SubmitTribute(controller.id, prompt.seat, card_id) if card_id is not None else None
        if prompt.kind == "return_tribute":
            card_id = _smallest_return_card(self.state, prompt.seat)
            return ReturnTribute(controller.id, prompt.seat, card_id) if card_id is not None else None
        return None

    def _npc_command(self, prompt: ActivePrompt, controller_id: str) -> Command | None:
        snapshot = self.seat_snapshot(prompt.seat, controller_id)
        action = DummyBotPolicy().choose_action(
            ActionRequest(
                request_id=prompt.prompt_id,
                prompt={
                    "kind": prompt.kind,
                    "current_level": self.state.current_level.value,
                    "return_rank_at_most_ten": _return_to_partner_required(self.state, prompt.seat),
                },
                snapshot={
                    "table_id": snapshot.public.table_id,
                    "seat": prompt.seat.value,
                    "hand": list(snapshot.hand),
                    "public": {
                        "event_seq": snapshot.public.event_seq,
                        "current_turn": snapshot.public.current_turn.value if snapshot.public.current_turn else None,
                    },
                },
            )
        )
        return _command_from_npc_action(action, controller_id, prompt.seat)

    def _service_event(self, event_type: str, payload: dict[str, object]) -> Event:
        next_seq = self.state.event_seq + 1
        self.state = self.state.bump_seq()
        return Event(next_seq, event_type, payload)

    def _prompt_payload(self, prompt: ActivePrompt) -> dict[str, object]:
        return {
            "seat": prompt.seat.value,
            "kind": prompt.kind,
            "deadline_epoch_ms": prompt.deadline_epoch_ms,
            "timeout_seconds": self.config.action_timeout_seconds,
            "prompt_id": prompt.prompt_id,
        }


def _system_epoch_ms() -> int:
    return int(time.time() * 1000)


def _current_task_or_none() -> asyncio.Task[object] | None:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def _prompt_requirement(state: MatchState) -> PromptRequirement | None:
    if state.deal is None:
        return None
    if state.phase == MatchPhase.PLAYING:
        kind: PromptKind = "lead" if state.deal.current_trick.last_play is None else "play_or_pass"
        return PromptRequirement(state.deal.turn, kind, state.event_seq)
    if state.phase == MatchPhase.TRIBUTE and state.deal.tribute is not None:
        for obligation in state.deal.tribute.obligations:
            if obligation.tribute_card_id is None:
                return PromptRequirement(obligation.giver, "tribute", state.event_seq)
        for obligation in state.deal.tribute.obligations:
            if obligation.return_card_id is None:
                return PromptRequirement(obligation.receiver, "return_tribute", state.event_seq)
    return None


def _smallest_card_id(state: MatchState, seat: Seat) -> str | None:
    if state.deal is None:
        return None
    return _min_card(state.deal.hand_for(seat), state.current_level)


def _highest_eligible_tribute_card(state: MatchState, seat: Seat) -> str | None:
    if state.deal is None:
        return None
    hand = state.deal.hand_for(seat)
    eligible = [
        card_id for card_id in hand if not is_red_heart_level_card(CARD_BY_ID[card_id], state.current_level)
    ]
    if not eligible:
        return None
    ctx = RankContext(state.current_level)
    return max(eligible, key=lambda card_id: (ctx.rank_value(CARD_BY_ID[card_id].rank), card_id))


def _smallest_return_card(state: MatchState, seat: Seat) -> str | None:
    if state.deal is None or state.deal.tribute is None:
        return None
    obligation = next(
        (
            item
            for item in state.deal.tribute.obligations
            if item.receiver == seat and item.tribute_card_id is not None and item.return_card_id is None
        ),
        None,
    )
    if obligation is None:
        return None
    hand = state.deal.hand_for(seat)
    if partner_for_seat(seat) == obligation.giver:
        hand = tuple(card_id for card_id in hand if _rank_at_most_ten(card_id))
    return _min_card(hand, state.current_level)


def _return_to_partner_required(state: MatchState, seat: Seat) -> bool:
    if state.deal is None or state.deal.tribute is None:
        return False
    obligation = next(
        (
            item
            for item in state.deal.tribute.obligations
            if item.receiver == seat and item.tribute_card_id is not None and item.return_card_id is None
        ),
        None,
    )
    return obligation is not None and partner_for_seat(seat) == obligation.giver


def _command_from_npc_action(action: JsonObject, controller_id: str, seat: Seat) -> Command | None:
    action_type = action.get("type")
    if action_type == "pass":
        return Pass(controller_id, seat)
    if action_type == "play_cards":
        card_ids = tuple(str(card_id) for card_id in action.get("card_ids", ()))
        return PlayCards(controller_id, seat, card_ids) if card_ids else None
    if action_type == "submit_tribute":
        card_id = action.get("card_id")
        return SubmitTribute(controller_id, seat, str(card_id)) if card_id is not None else None
    if action_type == "return_tribute":
        card_id = action.get("card_id")
        return ReturnTribute(controller_id, seat, str(card_id)) if card_id is not None else None
    return None


def _min_card(card_ids: tuple[str, ...], level: Rank) -> str | None:
    if not card_ids:
        return None
    ctx = RankContext(level)
    return min(card_ids, key=lambda card_id: (ctx.rank_value(CARD_BY_ID[card_id].rank), card_id))


def _rank_at_most_ten(card_id: str) -> bool:
    return CARD_BY_ID[card_id].rank in {
        Rank.TWO,
        Rank.THREE,
        Rank.FOUR,
        Rank.FIVE,
        Rank.SIX,
        Rank.SEVEN,
        Rank.EIGHT,
        Rank.NINE,
        Rank.TEN,
    }
