from __future__ import annotations

import argparse
import time
from enum import StrEnum

from client.api import GuandanClientError, GuandanHttpClient, JsonObject
from client.cli_app.commands import read_command, submit_human_command
from client.cli_app.render import (
    client_error_code,
    format_client_error,
    format_command_response,
    format_public_snapshot,
    format_seat_snapshot,
    format_timeout_fallback,
    help_text,
)
from client.cli_app.session import CliSession
from client.cli_app.types import InputFn, OutputFn


ACTIVE_PHASES = {"PLAYING", "TRIBUTE"}
TERMINAL_PHASES = {"MATCH_COMPLETE", "ABORTED"}


class CliMachineState(StrEnum):
    BOT_TURN = "BOT_TURN"
    DEAL_COMPLETE = "DEAL_COMPLETE"
    HUMAN_TURN = "HUMAN_TURN"
    FINISHED = "FINISHED"
    WAITING = "WAITING"


class CliStateMachine:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        client: GuandanHttpClient,
        session: CliSession,
        input_fn: InputFn,
        emit: OutputFn,
    ) -> None:
        self.args = args
        self.client = client
        self.session = session
        self.input_fn = input_fn
        self.emit = emit
        self.deal_number = 1

    def run(self, public_snapshot: JsonObject) -> JsonObject:
        current = drive_bot_turns(self.client, self.session, public_snapshot, self.emit, self.args.max_bot_actions)
        while True:
            state = self._machine_state(current)
            if state == CliMachineState.FINISHED:
                self.emit(format_public_snapshot(current, viewer_seat=self.session.human_seat, npc_metadata=self.session.npc_metadata).rstrip())
                return current
            if state == CliMachineState.DEAL_COMPLETE:
                current = self._start_next_deal()
                self.emit(format_public_snapshot(current, viewer_seat=self.session.human_seat, npc_metadata=self.session.npc_metadata).rstrip())
                current = drive_bot_turns(self.client, self.session, current, self.emit, self.args.max_bot_actions)
                continue
            if state == CliMachineState.BOT_TURN:
                updated = drive_bot_turns(self.client, self.session, current, self.emit, self.args.max_bot_actions)
                if updated == current:
                    current = wait_for_timeout_resolution(self.client, current, self.emit)
                    if updated == current:
                        self.emit(f"Waiting for {snapshot_acting_seat(current)}.")
                        return current
                    continue
                current = updated
                continue
            if state == CliMachineState.HUMAN_TURN:
                next_snapshot = self._handle_human_turn()
                if next_snapshot is None:
                    return current
                current = next_snapshot
                continue
            updated = wait_for_timeout_resolution(self.client, current, self.emit)
            if updated == current:
                self.emit(f"Waiting for {snapshot_acting_seat(current) or current.get('phase', '-')}.")
                return current
            current = updated
            continue

    def _machine_state(self, snapshot: JsonObject) -> CliMachineState:
        phase = snapshot.get("phase")
        if phase in TERMINAL_PHASES:
            return CliMachineState.FINISHED
        if phase == "DEAL_COMPLETE":
            return CliMachineState.DEAL_COMPLETE
        if phase in ACTIVE_PHASES:
            acting_seat = snapshot_acting_seat(snapshot)
            if acting_seat == self.session.human_seat:
                return CliMachineState.HUMAN_TURN
            if isinstance(acting_seat, str) and acting_seat in self.session.bot_broker.seats:
                return CliMachineState.BOT_TURN
        return CliMachineState.WAITING

    def _start_next_deal(self) -> JsonObject:
        self.deal_number += 1
        response = self.client.start(self.session.table_id, seed=deal_seed(self.args.seed, self.deal_number))
        self.emit(format_command_response(response).rstrip())
        return response.get("snapshot") or self.client.table_snapshot(self.session.table_id)

    def _handle_human_turn(self) -> JsonObject | None:
        seat_snapshot = self.client.seat_snapshot(
            self.session.table_id,
            self.session.human_seat,
            self.session.human_controller_id,
        )
        self.emit(format_seat_snapshot(seat_snapshot, npc_metadata=self.session.npc_metadata).rstrip())
        try:
            raw_command = read_command(self.input_fn, "guandan> ", input_deadline_epoch_ms(seat_snapshot))
        except EOFError:
            self.emit("Quit.")
            return None

        latest_snapshot = self.client.table_snapshot(self.session.table_id)
        if latest_snapshot.get("phase") not in ACTIVE_PHASES:
            return latest_snapshot
        if raw_command is None:
            return refresh_after_input_timeout(
                self.client,
                self.session,
                latest_snapshot,
                self.emit,
                self.args.max_bot_actions,
                before_snapshot=seat_snapshot.get("public") if isinstance(seat_snapshot.get("public"), dict) else None,
                kind=str(seat_snapshot.get("legal_action") or "") or None,
            )
        if _human_turn_elapsed(latest_snapshot, self.session):
            return drive_bot_turns(self.client, self.session, latest_snapshot, self.emit, self.args.max_bot_actions)

        command = raw_command.strip()
        if not command:
            return latest_snapshot
        if command in {"quit", "exit"}:
            self.emit("Quit.")
            return None
        if command == "help":
            self.emit(help_text().rstrip())
            return latest_snapshot
        if command == "hand":
            self.emit(format_seat_snapshot(seat_snapshot, npc_metadata=self.session.npc_metadata).rstrip())
            return latest_snapshot
        if command == "table":
            refreshed = self.client.table_snapshot(self.session.table_id)
            self.emit(format_public_snapshot(refreshed, viewer_seat=self.session.human_seat, npc_metadata=self.session.npc_metadata).rstrip())
            return refreshed

        try:
            response = submit_human_command(self.client, self.session, command, seat_snapshot)
        except GuandanClientError as exc:
            self.emit(format_client_error(exc))
            return self.client.table_snapshot(self.session.table_id)
        self.emit(format_command_response(response).rstrip())
        public_snapshot = response.get("snapshot") or self.client.table_snapshot(self.session.table_id)
        return drive_bot_turns(self.client, self.session, public_snapshot, self.emit, self.args.max_bot_actions)


def deal_seed(base_seed: object, deal_number: int) -> object:
    if deal_number <= 1:
        return base_seed
    return f"{base_seed}:deal-{deal_number}"


def drive_bot_turns(
    client: GuandanHttpClient,
    session: CliSession,
    public_snapshot: JsonObject,
    emit: OutputFn,
    max_actions: int,
) -> JsonObject:
    current = public_snapshot
    actions = 0
    while current.get("phase") in ACTIVE_PHASES:
        seat = snapshot_acting_seat(current)
        if not isinstance(seat, str) or seat not in session.bot_broker.seats:
            return current
        if actions >= max_actions:
            emit("Stopped automatic bot play after reaching the safety limit.")
            return current
        try:
            submitted = session.bot_broker.poll_once_results(seat)
        except GuandanClientError as exc:
            if client_error_code(exc) == "NOT_YOUR_TURN":
                return client.table_snapshot(session.table_id)
            emit(format_client_error(exc))
            return client.table_snapshot(session.table_id)
        if not submitted:
            return client.table_snapshot(session.table_id)
        for result in submitted:
            emit(format_command_response(result.response).rstrip())
        current = client.table_snapshot(session.table_id)
        actions += len(submitted)
    return current


def _human_turn_elapsed(snapshot: JsonObject, session: CliSession) -> bool:
    return snapshot.get("phase") in ACTIVE_PHASES and snapshot_acting_seat(snapshot) != session.human_seat


def refresh_after_input_timeout(
    client: GuandanHttpClient,
    session: CliSession,
    snapshot: JsonObject,
    emit: OutputFn,
    max_bot_actions: int,
    *,
    before_snapshot: JsonObject | None = None,
    kind: str | None = None,
) -> JsonObject:
    current = snapshot
    for attempt in range(20):
        if current.get("phase") not in ACTIVE_PHASES:
            return current
        if _human_turn_elapsed(current, session):
            if before_snapshot is not None and _snapshot_changed_by_timeout(before_snapshot, current):
                emit(format_timeout_fallback(before_snapshot, current, kind=kind))
            return drive_bot_turns(client, session, current, emit, max_bot_actions)
        if attempt < 19:
            time.sleep(0.1)
            current = client.table_snapshot(session.table_id)
    if before_snapshot is not None:
        resolved = wait_for_timeout_resolution(client, before_snapshot, emit, kind=kind)
        if resolved != before_snapshot:
            return drive_bot_turns(client, session, resolved, emit, max_bot_actions)
    return current


def wait_for_timeout_resolution(
    client: GuandanHttpClient,
    before: JsonObject,
    emit: OutputFn,
    *,
    kind: str | None = None,
) -> JsonObject:
    deadline = before.get("action_deadline_epoch_ms")
    if not isinstance(deadline, int):
        return before
    sleep_seconds = max(0.0, (deadline - int(time.time() * 1000)) / 1000)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    table_id = str(before.get("table_id", ""))
    current = before
    for attempt in range(20):
        current = client.table_snapshot(table_id)
        if _snapshot_changed_by_timeout(before, current):
            emit(format_timeout_fallback(before, current, kind=kind))
            return current
        if attempt < 19:
            time.sleep(0.1)
    return current


def _snapshot_changed_by_timeout(before: JsonObject, after: JsonObject) -> bool:
    if after.get("event_seq") != before.get("event_seq"):
        return True
    if after.get("phase") != before.get("phase"):
        return True
    if snapshot_acting_seat(after) != snapshot_acting_seat(before):
        return True
    if after.get("current_trick") != before.get("current_trick"):
        return True
    return False


def snapshot_acting_seat(snapshot: JsonObject) -> str | None:
    seat = snapshot.get("acting_seat") or snapshot.get("current_turn")
    return seat if isinstance(seat, str) else None


def input_deadline_epoch_ms(seat_snapshot: JsonObject) -> int | None:
    public = seat_snapshot.get("public")
    if not isinstance(public, dict):
        return None
    deadline = public.get("action_deadline_epoch_ms")
    return deadline if isinstance(deadline, int) else None
