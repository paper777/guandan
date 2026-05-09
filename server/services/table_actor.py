from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Literal

from common.log import deadline_remaining_ms, trace_event
from server.domain.cards import CARD_BY_ID, Rank, is_red_heart_level_card
from server.domain.commands import Command, Pass, PlayCards, ReturnTribute, StartMatch, SubmitTribute
from server.domain.comparator import RankContext
from server.domain.events import CommandRejected, Event, RejectCode
from server.domain.reducer import reduce_command
from server.domain.seats import Seat, partner_for_seat
from server.domain.state import MatchPhase, MatchState
from server.persistence.sqlite_store import SQLiteEventStore
from server.services.replay import rebuild_state_from_events
from server.services.snapshots import PublicTableSnapshot, SeatSnapshot, public_snapshot, seat_snapshot
from server.services.table_config import TableConfig, TimeoutFallback


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
        self.last_timeout_result: ActorResult | None = None
        self._deal_number = 0
        if self.event_store is not None:
            self.event_store.create_match(self.match_id, table_id)
            events = self.event_store.load_events(self.match_id)
            if events:
                self.state = rebuild_state_from_events(table_id, events)
                self._deal_number = _count_deals_started(events)
        self._refresh_prompt(schedule_timeout=False, emit_prompt_event=False)

    @property
    def active_prompt(self) -> ActivePrompt | None:
        return self._active_prompt

    def public_snapshot(self) -> PublicTableSnapshot:
        return public_snapshot(
            self.state,
            deal_id=self._deal_number,
            action_deadline_epoch_ms=self._active_prompt.deadline_epoch_ms if self._active_prompt else None,
            action_timeout_seconds=self.config.action_timeout_seconds,
            acting_seat=self._active_prompt.seat if self._active_prompt else None,
        )

    def seat_snapshot(self, seat: Seat, controller_id: str) -> SeatSnapshot:
        return seat_snapshot(
            self.state,
            seat,
            controller_id,
            deal_id=self._deal_number,
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

        effective_command = self._server_command(command)
        result = reduce_command(self.state, effective_command)
        if result.rejection is not None:
            return ActorResult(events=(), rejection=result.rejection)

        self.state = result.state
        events = (*result.events, *self._refresh_prompt(schedule_timeout=schedule_timeout, emit_prompt_event=True))
        self._deal_number += sum(1 for event in events if event.type == "DealStarted")
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

    def _server_command(self, command: Command) -> Command:
        if isinstance(command, StartMatch):
            return replace(command, seed=self.config.seed_for_deal(self._deal_number + 1))
        return command

    def _refresh_prompt(self, *, schedule_timeout: bool, emit_prompt_event: bool) -> tuple[Event, ...]:
        requirement = _prompt_requirement(self.state)
        if requirement is None:
            self._active_prompt = None
            self._cancel_timeout_task()
            return ()

        if self._active_prompt is not None and self._active_prompt.prompt_id == requirement.prompt_id:
            if schedule_timeout and self._timeout_task is None:
                self._schedule_timeout_task()
            return ()

        self._cancel_timeout_task()
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
        if not emit_prompt_event:
            return ()
        trace_event(
            "server.action_prompted",
            table_id=self.table_id,
            match_id=self.match_id,
            seat=self._active_prompt.seat.value,
            kind=self._active_prompt.kind,
            prompt_id=self._active_prompt.prompt_id,
            state_seq=self._active_prompt.state_seq,
            started_epoch_ms=self._active_prompt.started_epoch_ms,
            deadline_epoch_ms=self._active_prompt.deadline_epoch_ms,
            timeout_seconds=self.config.action_timeout_seconds,
        )
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

    async def _timeout_after(self, prompt_id: str, delay: float) -> None:
        try:
            await self._sleeper(delay)
        except asyncio.CancelledError:
            return
        async with self._lock:
            if self._timeout_task is asyncio.current_task():
                self._timeout_task = None
            self.last_timeout_result = self._apply_timeout_locked(prompt_id)

    def _apply_timeout_locked(self, prompt_id: str) -> ActorResult:
        prompt = self._active_prompt
        if prompt is None or prompt.prompt_id != prompt_id:
            return ActorResult(events=())
        trace_event(
            "server.action_timed_out",
            table_id=self.table_id,
            match_id=self.match_id,
            seat=prompt.seat.value,
            kind=prompt.kind,
            prompt_id=prompt.prompt_id,
            state_seq=prompt.state_seq,
            deadline_epoch_ms=prompt.deadline_epoch_ms,
            deadline_remaining_ms=deadline_remaining_ms(prompt.deadline_epoch_ms, now_epoch_ms=self._clock()),
            timeout_fallback=self.config.timeout_fallback.value,
        )

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
            trace_event(
                "server.timeout_fallback_failed",
                table_id=self.table_id,
                match_id=self.match_id,
                seat=prompt.seat.value,
                kind=prompt.kind,
                prompt_id=prompt.prompt_id,
                reason="fallback command could not be built",
            )
            result = ActorResult(
                events=(timeout_event,),
                rejection=CommandRejected(RejectCode.ACTION_TIMEOUT, "timeout fallback could not be built"),
            )
            if self.event_store is not None:
                self.event_store.append_events(self.match_id, result.events)
            return result

        fallback_result = reduce_command(self.state, command)
        if fallback_result.rejection is not None:
            trace_event(
                "server.timeout_fallback_rejected",
                table_id=self.table_id,
                match_id=self.match_id,
                seat=prompt.seat.value,
                kind=prompt.kind,
                prompt_id=prompt.prompt_id,
                command_type=type(command).__name__,
                rejection={
                    "code": fallback_result.rejection.code.value,
                    "message": fallback_result.rejection.message,
                },
            )
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
        trace_event(
            "server.timeout_fallback_applied",
            table_id=self.table_id,
            match_id=self.match_id,
            seat=prompt.seat.value,
            kind=prompt.kind,
            prompt_id=prompt.prompt_id,
            command_type=type(command).__name__,
            fallback_event_types=[event.type for event in fallback_events],
            next_prompt=self._prompt_payload(self._active_prompt) if self._active_prompt is not None else None,
        )
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


def _count_deals_started(events: tuple[Event, ...]) -> int:
    return sum(1 for event in events if event.type == "DealStarted")


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
