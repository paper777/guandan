from __future__ import annotations

import argparse
import time
from enum import StrEnum

from client.http_client import GuandanClientError, GuandanHttpClient
from client.types import ActionRequest, JsonObject, SeatMember, SeatRole
from client.tui.commands import command_card_ids, read_command, submit_human_command
from client.tui.render import (
    client_error_code,
    format_card_list,
    format_client_error,
    format_command_response,
    format_npc_metadata,
    format_public_snapshot,
    format_seat_snapshot,
    format_timeout_fallback,
    help_text,
)
from client.session import Session
from client.tui.types import InputFn, OutputFn
from common.log import debug_event, error_event, trace_event


ACTIVE_PHASES = {"PLAYING", "TRIBUTE"}
TERMINAL_PHASES = {"ABORTED"}
TIMEOUT_POLL_ATTEMPTS = 20
TIMEOUT_POLL_INTERVAL_SECONDS = 0.1
_COMMAND_NOT_HANDLED = object()


class MachineState(StrEnum):
    BOT_TURN = "BOT_TURN"
    DEAL_COMPLETE = "DEAL_COMPLETE"
    MATCH_COMPLETE = "MATCH_COMPLETE"
    HUMAN_TURN = "HUMAN_TURN"
    FINISHED = "FINISHED"
    WAITING = "WAITING"


class StateMachine:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        client: GuandanHttpClient,
        session: Session,
        input_fn: InputFn,
        emit: OutputFn,
    ) -> None:
        self.args = args
        self.client = client
        self.session = session
        self.input_fn = input_fn
        self.emit = emit
        self.deal_number = 1
        self._last_recorded_event_seq: int | None = None

    def run(self, public_snapshot: JsonObject) -> JsonObject:
        trace_event("client.state_machine.run_started", **_state_machine_log_fields(self.session, public_snapshot))
        try:
            current = self._drive_bot_turns(public_snapshot)
            while True:
                state = self._machine_state(current)
                if state == MachineState.FINISHED:
                    finished = self._finish(current)
                    debug_event(
                        "client.state_machine.run_completed",
                        reason="finished",
                        **_state_machine_log_fields(self.session, finished),
                    )
                    return finished
                if state == MachineState.DEAL_COMPLETE:
                    current = self._start_and_drive_next_deal()
                    continue
                if state == MachineState.MATCH_COMPLETE:
                    current = self._start_and_drive_next_match(current)
                    continue
                if state == MachineState.BOT_TURN:
                    updated = self._drive_bot_turns(current)
                    if updated == current:
                        resolved = self._wait_for_timeout(current, waiting_label=str(snapshot_acting_seat(current)))
                        if resolved == current:
                            debug_event(
                                "client.state_machine.run_completed",
                                reason="bot_waiting",
                                **_state_machine_log_fields(self.session, current),
                            )
                            return current
                        current = resolved
                        continue
                    current = updated
                    continue
                if state == MachineState.HUMAN_TURN:
                    next_snapshot = self._handle_human_turn()
                    if next_snapshot is None:
                        debug_event(
                            "client.state_machine.run_completed",
                            reason="human_exit",
                            **_state_machine_log_fields(self.session, current),
                        )
                        return current
                    current = next_snapshot
                    continue
                waiting_label = str(snapshot_acting_seat(current) or current.get("phase", "-"))
                updated = self._wait_for_timeout(current, waiting_label=waiting_label)
                if updated == current:
                    debug_event(
                        "client.state_machine.run_completed",
                        reason="waiting",
                        **_state_machine_log_fields(self.session, current),
                    )
                    return current
                current = updated
        except Exception as exc:
            error_event(
                "client.state_machine.run_failed",
                **_state_machine_log_fields(self.session, public_snapshot),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise

    def _machine_state(self, snapshot: JsonObject) -> MachineState:
        phase = snapshot.get("phase")
        if phase in TERMINAL_PHASES:
            return MachineState.FINISHED
        if phase == "MATCH_COMPLETE":
            return MachineState.MATCH_COMPLETE
        if phase == "DEAL_COMPLETE":
            return MachineState.DEAL_COMPLETE
        if phase in ACTIVE_PHASES:
            acting_seat = snapshot_acting_seat(snapshot)
            if self.session.watches_llm_player and self._is_broker_seat(acting_seat):
                return MachineState.BOT_TURN
            if acting_seat == self.session.human_seat:
                return MachineState.HUMAN_TURN
            if self._is_broker_seat(acting_seat):
                return MachineState.BOT_TURN
        return MachineState.WAITING

    def _is_broker_seat(self, seat: object) -> bool:
        return isinstance(seat, str) and seat in self.session.bot_broker.seats

    def _drive_bot_turns(self, public_snapshot: JsonObject) -> JsonObject:
        current = drive_bot_turns(
            self.client,
            self.session,
            public_snapshot,
            self.emit,
            self.args.max_bot_actions,
            watch_private_seat=self.session.watched_private_seat,
        )
        self._record_table_transitions(current)
        return current

    def _finish(self, snapshot: JsonObject) -> JsonObject:
        self._record_table_transitions(snapshot)
        self._emit_table(snapshot)
        return snapshot

    def _start_and_drive_next_deal(self) -> JsonObject:
        self._record_table_transitions(self.client.table_snapshot(self.session.table_id))
        current = self._start_next_deal()
        self._record_table_transitions(current)
        self._emit_table(current)
        return self._drive_bot_turns(current)

    def _start_and_drive_next_match(self, snapshot: JsonObject) -> JsonObject:
        self._record_table_transitions(snapshot)
        self._prepare_next_match_seats()
        current = self._start_next_match()
        self._record_table_transitions(current)
        self._emit_table(current)
        return self._drive_bot_turns(current)

    def _emit_table(self, snapshot: JsonObject) -> None:
        self.emit(
            format_public_snapshot(
                snapshot,
                viewer_seat=self.session.human_seat,
                npc_metadata=self.session.npc_metadata,
            ).rstrip()
        )

    def _wait_for_timeout(self, snapshot: JsonObject, *, waiting_label: str) -> JsonObject:
        updated = wait_for_timeout_resolution(self.client, snapshot, self.emit)
        if updated == snapshot:
            self.emit(f"Waiting for {waiting_label}.")
        return updated

    def _start_next_deal(self) -> JsonObject:
        self.deal_number += 1
        response = self.client.start(self.session.table_id)
        self.emit(format_command_response(response).rstrip())
        return response.get("snapshot") or self.client.table_snapshot(self.session.table_id)

    def _start_next_match(self) -> JsonObject:
        self.deal_number = 1
        response = self.client.start(self.session.table_id)
        self.emit(format_command_response(response).rstrip())
        return response.get("snapshot") or self.client.table_snapshot(self.session.table_id)

    def _record_table_transitions(self, snapshot: JsonObject) -> None:
        event_seq = snapshot.get("event_seq")
        if isinstance(event_seq, int) and event_seq == self._last_recorded_event_seq:
            return
        for transition in self.session.table.record_snapshot(snapshot):
            self.emit(transition.message)
            if transition.kind == "match_complete":
                self._rotate_seats_after_match()
        if isinstance(event_seq, int):
            self._last_recorded_event_seq = event_seq

    def _rotate_seats_after_match(self) -> None:
        seat_map = self.session.bot_broker.rotate_seats_after_match()
        self.session.rotate_seat_members(seat_map)
        self.emit(_format_seat_rotation(seat_map))

    def _prepare_next_match_seats(self) -> None:
        if self.session.player_mode == "human":
            self._join_human_player_for_next_match()
        self.session.bot_broker.join_and_ready_all()
        self._sync_broker_role_members()
        if self.session.watches_llm_player and self.session.human_seat in self.session.bot_broker.seats:
            self.session.human_controller_id = self.session.bot_broker.seats[self.session.human_seat].controller_id
        self._refresh_npc_metadata()

    def _join_human_player_for_next_match(self) -> None:
        members = self.session.table.members_for(self.session.human_seat)
        human = members.player if members.player is not None and members.player.is_human else None
        if human is None:
            return
        response = self.client.join_human(
            self.session.table_id,
            self.session.human_seat,
            player_id=f"human-{self.session.human_seat}",
            controller_id=f"human-controller-{self.session.human_seat}",
            display_name=human.display_name,
        )
        controller_id = str(response.get("controller_id") or f"human-controller-{self.session.human_seat}")
        self.session.human_controller_id = controller_id
        human.controller_id = controller_id
        self.client.ready(self.session.table_id, self.session.human_seat, controller_id)

    def _sync_broker_role_members(self) -> None:
        for broker_seat in self.session.bot_broker.seats.values():
            member = self.session.table.members_for(broker_seat.seat).player
            if member is None:
                continue
            member.controller_id = broker_seat.controller_id
            member.profile_key = broker_seat.profile_key

    def _refresh_npc_metadata(self) -> None:
        self.session.npc_metadata = {
            seat: format_npc_metadata(broker_seat.policy)
            for seat, broker_seat in self.session.bot_broker.seats.items()
        }

    def _handle_human_turn(self) -> JsonObject | None:
        seat_snapshot, raw_command = self._read_human_command()
        latest_snapshot = self.client.table_snapshot(self.session.table_id)
        if latest_snapshot.get("phase") not in ACTIVE_PHASES:
            return latest_snapshot
        if raw_command is None:
            return self._refresh_after_input_timeout(latest_snapshot, seat_snapshot)
        if _human_turn_elapsed(latest_snapshot, self.session):
            return self._drive_bot_turns(latest_snapshot)

        command = raw_command.strip()
        if not command:
            return latest_snapshot
        meta_response = self._handle_meta_command(command, seat_snapshot)
        if meta_response is not _COMMAND_NOT_HANDLED:
            return meta_response
        return self._submit_human_turn(command, seat_snapshot)

    def _read_human_command(self) -> tuple[JsonObject, str | None]:
        seat_snapshot = self.client.seat_snapshot(
            self.session.table_id,
            self.session.human_seat,
            self.session.human_controller_id,
        )
        self.emit(format_seat_snapshot(seat_snapshot, npc_metadata=self.session.npc_metadata).rstrip())
        self._emit_gossiper_advice(seat_snapshot)
        try:
            raw_command = read_command(self.input_fn, "guandan> ", input_deadline_epoch_ms(seat_snapshot))
        except EOFError:
            return seat_snapshot, "quit"
        return seat_snapshot, raw_command

    def _emit_gossiper_advice(self, seat_snapshot: JsonObject) -> None:
        gossiper = self.session.table.members_for(self.session.human_seat).gossiper
        if gossiper is None or gossiper.policy is None:
            return
        try:
            action = gossiper.policy.choose_action(_action_request_from_seat_snapshot(self.session, seat_snapshot))
        except Exception as exc:  # pragma: no cover - defensive CLI boundary
            self.emit(f"Advice from {gossiper.display_name}: unavailable ({exc})")
            return
        self.emit(_format_advice(gossiper, action))

    def _refresh_after_input_timeout(self, latest_snapshot: JsonObject, seat_snapshot: JsonObject) -> JsonObject:
        return refresh_after_input_timeout(
            self.client,
            self.session,
            latest_snapshot,
            self.emit,
            self.args.max_bot_actions,
            before_snapshot=_public_snapshot_from_seat_snapshot(seat_snapshot),
            kind=str(seat_snapshot.get("legal_action") or "") or None,
        )

    def _handle_meta_command(self, command: str, seat_snapshot: JsonObject) -> JsonObject | object | None:
        if command in {"quit", "exit"}:
            self.emit("Quit.")
            return None
        if command == "help":
            self.emit(help_text().rstrip())
            return self.client.table_snapshot(self.session.table_id)
        if command == "hand":
            self.emit(format_seat_snapshot(seat_snapshot, npc_metadata=self.session.npc_metadata).rstrip())
            return self.client.table_snapshot(self.session.table_id)
        if command == "table":
            refreshed = self.client.table_snapshot(self.session.table_id)
            self.emit(
                format_public_snapshot(
                    refreshed,
                    viewer_seat=self.session.human_seat,
                    npc_metadata=self.session.npc_metadata,
                ).rstrip()
            )
            return refreshed
        return _COMMAND_NOT_HANDLED

    def _submit_human_turn(self, command: str, seat_snapshot: JsonObject) -> JsonObject:
        try:
            response = submit_human_command(self.client, self.session, command, seat_snapshot)
        except GuandanClientError as exc:
            error_event(
                "client.state_machine.human_command_failed",
                table_id=self.session.table_id,
                seat=self.session.human_seat,
                command=command_action(command, seat_snapshot),
                status=exc.status,
                error_payload=exc.payload,
            )
            self.emit(format_client_error(exc))
            rejected_cards = rejected_command_cards(command, seat_snapshot, exc)
            if rejected_cards:
                self.emit(f"Rejected cards: {format_card_list(rejected_cards)}")
            return self.client.table_snapshot(self.session.table_id)
        self.emit(format_command_response(response).rstrip())
        debug_event(
            "client.state_machine.human_command_completed",
            table_id=self.session.table_id,
            seat=self.session.human_seat,
            command=command_action(command, seat_snapshot),
            event_seq=response.get("event_seq"),
        )
        trigger_role_observers(self.session, self.session.human_seat, command_action(command, seat_snapshot), response)
        public_snapshot = response.get("snapshot") or self.client.table_snapshot(self.session.table_id)
        return self._drive_bot_turns(public_snapshot)


def drive_bot_turns(
    client: GuandanHttpClient,
    session: Session,
    public_snapshot: JsonObject,
    emit: OutputFn,
    max_actions: int,
    *,
    watch_private_seat: str | None = None,
) -> JsonObject:
    current = public_snapshot
    actions = 0
    while current.get("phase") in ACTIVE_PHASES:
        seat = snapshot_acting_seat(current)
        if not isinstance(seat, str) or seat not in session.bot_broker.seats:
            debug_event(
                "client.state_machine.bot_drive_completed",
                reason="no_broker_seat",
                submitted_actions=actions,
                **_state_machine_log_fields(session, current),
            )
            return current
        if actions >= max_actions:
            emit("Stopped automatic bot play after reaching the safety limit.")
            debug_event(
                "client.state_machine.bot_drive_completed",
                reason="safety_limit",
                submitted_actions=actions,
                **_state_machine_log_fields(session, current),
            )
            return current
        try:
            if seat == watch_private_seat:
                seat_snapshot = client.seat_snapshot(session.table_id, seat, session.human_controller_id)
                emit(format_seat_snapshot(seat_snapshot, npc_metadata=session.npc_metadata).rstrip())
            submitted = session.bot_broker.poll_once_results(seat)
        except GuandanClientError as exc:
            error_event(
                "client.state_machine.bot_drive_failed",
                seat=seat,
                status=exc.status,
                error_payload=exc.payload,
                **_state_machine_log_fields(session, current),
            )
            if client_error_code(exc) == "NOT_YOUR_TURN":
                return client.table_snapshot(session.table_id)
            emit(format_client_error(exc))
            rejected_cards = rejected_broker_action_cards(exc)
            if rejected_cards:
                emit(f"Rejected cards: {format_card_list(rejected_cards)}")
            return client.table_snapshot(session.table_id)
        if not submitted:
            debug_event(
                "client.state_machine.bot_drive_completed",
                reason="no_action",
                seat=seat,
                submitted_actions=actions,
                **_state_machine_log_fields(session, current),
            )
            return client.table_snapshot(session.table_id)
        for result in submitted:
            emit(format_command_response(result.response).rstrip())
            trigger_role_observers(session, seat, result.action, result.response)
        current = client.table_snapshot(session.table_id)
        actions += len(submitted)
    debug_event(
        "client.state_machine.bot_drive_completed",
        reason="inactive_phase",
        submitted_actions=actions,
        **_state_machine_log_fields(session, current),
    )
    return current


def trigger_role_observers(session: Session, actor_seat: str, action: JsonObject, response: JsonObject) -> None:
    seat_members = session.table.seats.get(actor_seat)
    if seat_members is None:
        return
    observation = _role_observation(session, actor_seat, action, response)
    for member in seat_members.trigger_order():
        if member.role == SeatRole.PLAYER:
            continue
        observe_action = getattr(member.policy, "observe_action", None)
        if observe_action is not None:
            observe_action({**observation, "observer_name": member.display_name, "observer_role": member.role.value})


def command_action(command: str, seat_snapshot: JsonObject) -> JsonObject:
    parts = command.split()
    if not parts:
        return {}
    action = parts[0]
    if action == "pass":
        return {"type": "pass"}
    if action == "play":
        return {"type": "play_cards", "card_ids": list(command_card_ids(command, seat_snapshot))}
    if action == "tribute":
        card_ids = command_card_ids(command, seat_snapshot)
        return {"type": "submit_tribute", "card_id": card_ids[0] if card_ids else ""}
    if action in {"return", "return_tribute"}:
        card_ids = command_card_ids(command, seat_snapshot)
        return {"type": "return_tribute", "card_id": card_ids[0] if card_ids else ""}
    return {"type": action}


def _action_request_from_seat_snapshot(session: Session, seat_snapshot: JsonObject) -> ActionRequest:
    public = _public_snapshot_from_seat_snapshot(seat_snapshot) or {}
    seat = str(seat_snapshot.get("seat") or session.human_seat)
    return ActionRequest(
        request_id=f"{session.table_id}:{seat}:gossiper:{public.get('event_seq', 0)}",
        prompt={
            "kind": seat_snapshot.get("legal_action"),
            "current_level": public.get("current_level", "2"),
            "current_trick": public.get("current_trick"),
            "eligible_card_ids": list(seat_snapshot.get("eligible_card_ids", [])),
            "tribute_from": seat_snapshot.get("tribute_from"),
            "tribute_to": seat_snapshot.get("tribute_to"),
            "return_rank_at_most_ten": bool(seat_snapshot.get("return_rank_at_most_ten", False)),
        },
        snapshot={
            "table_id": session.table_id,
            "seat": seat,
            "hand": list(seat_snapshot.get("hand", [])),
            "players_by_seat": _players_by_seat(session, public),
            "public": public,
        },
    )


def _format_advice(gossiper: SeatMember, action: JsonObject) -> str:
    action_type = action.get("type")
    if action_type == "play_cards":
        return f"Advice from {gossiper.display_name}: play {format_card_list(action.get('card_ids', []))}"
    if action_type == "pass":
        return f"Advice from {gossiper.display_name}: pass"
    if action_type == "submit_tribute":
        return f"Advice from {gossiper.display_name}: tribute {format_card_list([action.get('card_id')])}"
    if action_type == "return_tribute":
        return f"Advice from {gossiper.display_name}: return {format_card_list([action.get('card_id')])}"
    message = action.get("message")
    if isinstance(message, str) and message:
        return f"Advice from {gossiper.display_name}: {message}"
    return f"Advice from {gossiper.display_name}: {action_type or 'no action'}"


def _role_observation(session: Session, actor_seat: str, action: JsonObject, response: JsonObject) -> JsonObject:
    events = response.get("events", [])
    event_list = events if isinstance(events, list) else []
    snapshot = response.get("snapshot")
    public = snapshot if isinstance(snapshot, dict) else {}
    return {
        "table_id": session.table_id,
        "actor_seat": actor_seat,
        "actor_name": _actor_name(session, actor_seat),
        "players_by_seat": _players_by_seat(session, public),
        "deal_id": public.get("deal_id"),
        "action": action,
        "events": event_list,
        "event_seq": response.get("event_seq"),
    }


def _actor_name(session: Session, actor_seat: str) -> str:
    members = session.table.seats.get(actor_seat)
    if members is not None and members.player is not None:
        return members.player.display_name
    return actor_seat


def _players_by_seat(session: Session, public_snapshot: JsonObject) -> JsonObject:
    players: JsonObject = {}
    seats = public_snapshot.get("seats")
    if isinstance(seats, dict):
        for raw_seat, raw_player in seats.items():
            if not isinstance(raw_player, dict):
                continue
            display_name = str(raw_player.get("display_name") or "").strip()
            if display_name:
                players[str(raw_seat)] = display_name
    for seat, members in session.table.seats.items():
        if members.player is not None:
            players.setdefault(seat, members.player.display_name)
    return players


def _format_seat_rotation(seat_map: dict[str, str]) -> str:
    changes = [f"{old}->{new}" for old, new in seat_map.items() if old != new]
    if not changes:
        return "Seat roles stay in place for the next match."
    return "Seat roles rotated for next match: " + ", ".join(changes)


def rejected_command_cards(command: str, seat_snapshot: JsonObject, error: GuandanClientError) -> tuple[str, ...]:
    if error.status != 400:
        return ()
    try:
        return command_card_ids(command, seat_snapshot)
    except GuandanClientError:
        return ()


def rejected_broker_action_cards(error: GuandanClientError) -> tuple[str, ...]:
    if error.status != 400:
        return ()
    card_ids = error.payload.get("rejected_card_ids")
    if not isinstance(card_ids, list):
        return ()
    return tuple(str(card_id) for card_id in card_ids)


def _human_turn_elapsed(snapshot: JsonObject, session: Session) -> bool:
    return snapshot.get("phase") in ACTIVE_PHASES and snapshot_acting_seat(snapshot) != session.human_seat


def _public_snapshot_from_seat_snapshot(seat_snapshot: JsonObject) -> JsonObject | None:
    public = seat_snapshot.get("public")
    return public if isinstance(public, dict) else None


def refresh_after_input_timeout(
    client: GuandanHttpClient,
    session: Session,
    snapshot: JsonObject,
    emit: OutputFn,
    max_bot_actions: int,
    *,
    before_snapshot: JsonObject | None = None,
    kind: str | None = None,
) -> JsonObject:
    current = snapshot
    for attempt in range(TIMEOUT_POLL_ATTEMPTS):
        if current.get("phase") not in ACTIVE_PHASES:
            return current
        if _human_turn_elapsed(current, session):
            if before_snapshot is not None and _snapshot_changed_by_timeout(before_snapshot, current):
                emit(format_timeout_fallback(before_snapshot, current, kind=kind))
            return drive_bot_turns(client, session, current, emit, max_bot_actions)
        if attempt < TIMEOUT_POLL_ATTEMPTS - 1:
            time.sleep(TIMEOUT_POLL_INTERVAL_SECONDS)
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
    for attempt in range(TIMEOUT_POLL_ATTEMPTS):
        current = client.table_snapshot(table_id)
        if _snapshot_changed_by_timeout(before, current):
            emit(format_timeout_fallback(before, current, kind=kind))
            debug_event(
                "client.state_machine.timeout_wait_completed",
                reason="resolved",
                table_id=table_id,
                event_seq=current.get("event_seq"),
                acting_seat=snapshot_acting_seat(current),
                phase=current.get("phase"),
            )
            return current
        if attempt < TIMEOUT_POLL_ATTEMPTS - 1:
            time.sleep(TIMEOUT_POLL_INTERVAL_SECONDS)
    debug_event(
        "client.state_machine.timeout_wait_completed",
        reason="unresolved",
        table_id=table_id,
        event_seq=current.get("event_seq"),
        acting_seat=snapshot_acting_seat(current),
        phase=current.get("phase"),
    )
    return current


def _state_machine_log_fields(session: Session, snapshot: JsonObject) -> JsonObject:
    return {
        "table_id": session.table_id,
        "human_seat": session.human_seat,
        "phase": snapshot.get("phase"),
        "deal_id": snapshot.get("deal_id"),
        "event_seq": snapshot.get("event_seq"),
        "acting_seat": snapshot_acting_seat(snapshot),
        "bot_seats": sorted(session.bot_broker.seats),
    }


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
