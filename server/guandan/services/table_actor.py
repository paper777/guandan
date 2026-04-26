from __future__ import annotations

import asyncio
from dataclasses import dataclass

from guandan.domain.commands import Command
from guandan.domain.events import CommandRejected, Event
from guandan.domain.reducer import reduce_command
from guandan.domain.state import MatchState
from guandan.persistence.sqlite_store import SQLiteEventStore
from guandan.services.replay import rebuild_state_from_events


@dataclass(frozen=True, slots=True)
class ActorResult:
    events: tuple[Event, ...]
    rejection: CommandRejected | None = None
    replayed: bool = False


class TableActor:
    def __init__(
        self,
        table_id: str,
        match_id: str | None = None,
        event_store: SQLiteEventStore | None = None,
    ) -> None:
        self.table_id = table_id
        self.match_id = match_id or table_id
        self.state = MatchState(table_id=table_id)
        self.event_store = event_store
        self._lock = asyncio.Lock()
        if self.event_store is not None:
            self.event_store.create_match(self.match_id, table_id)
            events = self.event_store.load_events(self.match_id)
            if events:
                self.state = rebuild_state_from_events(table_id, events)

    async def dispatch_async(
        self,
        command: Command,
        *,
        controller_id: str | None = None,
        request_id: str | None = None,
    ) -> ActorResult:
        async with self._lock:
            return self.dispatch(command, controller_id=controller_id, request_id=request_id)

    def dispatch(
        self,
        command: Command,
        *,
        controller_id: str | None = None,
        request_id: str | None = None,
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
        if self.event_store is not None and result.events:
            self.event_store.append_events(self.match_id, result.events)
            if controller_id is not None and request_id is not None:
                self.event_store.record_idempotency(
                    self.match_id,
                    controller_id,
                    request_id,
                    result.events[0].seq,
                    result.events[-1].seq,
                )
        return ActorResult(events=result.events)
