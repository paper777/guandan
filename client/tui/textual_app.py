from __future__ import annotations

import argparse
import time
from typing import Iterable

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static

from client.http_client import GuandanClientError, GuandanHttpClient
from client.session import Session, prepare_default_table
from client.state_machine import (
    ACTIVE_PHASES,
    TERMINAL_PHASES,
    command_action,
    drive_bot_turns,
    safe_command_action,
    snapshot_acting_seat,
    trigger_role_observers,
)
from client.tui.render import client_error_code, format_client_error, format_npc_metadata
from client.tui.types import Result
from client.tui.view_model import (
    ActionRow,
    action_availability,
    action_rows_from_response,
    card_views,
    selected_card_ids_in_hand_order,
    system_action_row,
    table_view,
)
from client.types import JsonObject
from common.log import debug_event, error_event


MAX_HAND_CARDS = 36
ACTIVE_ACTIONS = {"lead", "play_or_pass", "tribute", "return_tribute"}
MIN_UI_WIDTH = 80
HAND_GRID_COLUMNS = 8
ACTION_GRID_COLUMNS = 6
CARD_BUTTON_WIDTH = 10
ACTION_BUTTON_WIDTH = 13
FEED_COLUMN_WIDTHS = {
    "Seq": 4,
    "Seat": 4,
    "Action": 14,
    "Cards": 32,
    "Detail": 16,
}


class GuandanTextualApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #status {
        height: 2;
        padding: 0 1;
    }

    #body {
        height: 1fr;
    }

    #table-pane {
        width: 100%;
        height: auto;
    }

    #feed-pane {
        width: 100%;
        height: 10;
        min-height: 8;
    }

    #board {
        width: 100%;
        height: 12;
        layout: grid;
        grid-size: 3 3;
        grid-gutter: 1 1;
        padding: 0 1 1 1;
    }

    .seat {
        height: 3;
        border: round $surface-lighten-1;
        padding: 0 1;
    }

    .seat-active {
        border: heavy $accent;
    }

    #seat-n {
        column-span: 3;
    }

    #seat-w {
        column-span: 1;
    }

    #trick {
        column-span: 1;
        border: round $primary;
        padding: 0 1;
    }

    #seat-e {
        column-span: 1;
    }

    #seat-s {
        column-span: 3;
    }

    #hand-title {
        height: 2;
        padding: 0 1;
    }

    #hand {
        width: 100%;
        height: auto;
        layout: grid;
        grid-size: 8;
        grid-gutter: 0 0;
        padding: 0 1 1 1;
    }

    .card-button {
        width: 100%;
        min-width: 0;
        height: 1;
        margin: 0;
        background: $surface;
        text-align: center;
    }

    .selected-card {
        background: $accent;
        color: $text;
        text-style: bold;
    }

    .ineligible-card {
        opacity: 40%;
    }

    #actions {
        width: 100%;
        height: auto;
        layout: grid;
        grid-size: 6;
        grid-gutter: 0 0;
        padding: 0 1;
    }

    .action-button {
        width: 100%;
        min-width: 0;
        height: 1;
        margin: 0;
        background: $surface;
        text-align: center;
    }

    .disabled-control {
        opacity: 45%;
    }

    #feed {
        height: 1fr;
        width: 100%;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("c", "clear_selection", "Clear"),
        Binding("enter", "play_selected", "Submit"),
        Binding("space", "pass_turn", "Pass"),
    ]

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        client: GuandanHttpClient | None = None,
    ) -> None:
        super().__init__()
        self.args = args
        self.client = client
        self.session: Session | None = None
        self.public_snapshot: JsonObject = {}
        self.seat_snapshot: JsonObject | None = None
        self.selected_ids: set[str] = set()
        self.card_by_slot: dict[int, str] = {}
        self.busy = False
        self._result = Result(0, "")
        self._last_recorded_event_seq: int | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Preparing table...", id="status")
        with VerticalScroll(id="body"):
            with Vertical(id="table-pane"):
                with Container(id="board"):
                    yield Static("", id="seat-n", classes="seat")
                    yield Static("", id="seat-w", classes="seat")
                    yield Static("", id="trick")
                    yield Static("", id="seat-e", classes="seat")
                    yield Static("", id="seat-s", classes="seat")
                yield Static("", id="hand-title")
                with Container(id="hand"):
                    for index in range(MAX_HAND_CARDS):
                        yield Static("", id=f"card-{index}", classes="card-button")
                with Container(id="actions"):
                    yield Static("Play", id="play-action", classes="action-button disabled-control")
                    yield Static("Pass", id="pass-action", classes="action-button disabled-control")
                    yield Static("Tribute", id="tribute-action", classes="action-button disabled-control")
                    yield Static("Return", id="return-action", classes="action-button disabled-control")
                    yield Static("Clear", id="clear-action", classes="action-button disabled-control")
                    yield Static("Refresh", id="refresh-action", classes="action-button")
            with Vertical(id="feed-pane"):
                yield DataTable(id="feed")
        yield Footer()

    def on_mount(self) -> None:
        feed = self.query_one("#feed", DataTable)
        feed.cursor_type = "row"
        for label, width in FEED_COLUMN_WIDTHS.items():
            feed.add_column(label, width=width)
        self.set_interval(1.0, self._tick)
        self._set_busy("Preparing table...")
        self.run_worker(self._setup_table, thread=True, exclusive=True, name="setup-table")

    def _setup_table(self) -> None:
        try:
            client = self.client or GuandanHttpClient(base_url=self.args.base_url)
            session, public_snapshot = prepare_default_table(client, self.args)
        except Exception as exc:  # pragma: no cover - defensive TUI boundary
            self.call_from_thread(self._handle_fatal_error, exc)
            return
        self.call_from_thread(self._handle_setup_complete, client, session, public_snapshot)

    def _handle_setup_complete(
        self,
        client: GuandanHttpClient,
        session: Session,
        public_snapshot: JsonObject,
    ) -> None:
        self.client = client
        self.session = session
        seat_label = "Watching" if session.player_mode == "llm" else "You are"
        self._append_row(system_action_row(f"Table {session.table_id} | {seat_label} {session.human_seat}"))
        self._render_public_snapshot(public_snapshot)
        self.run_worker(lambda: self._advance_game(public_snapshot), thread=True, exclusive=True, name="advance-game")

    def _advance_game(self, public_snapshot: JsonObject) -> None:
        session = self._session()
        client = self._client()
        current = public_snapshot
        self.call_from_thread(self._set_busy, "Syncing table...")
        try:
            while True:
                self._record_table_transitions(current)
                phase = current.get("phase")
                if phase in TERMINAL_PHASES:
                    self.call_from_thread(self._handle_public_update, current, None, "Table finished.")
                    return
                if phase == "DEAL_COMPLETE":
                    current = self._start_next_deal()
                    continue
                if phase == "MATCH_COMPLETE":
                    current = self._start_next_match()
                    continue
                if phase in ACTIVE_PHASES:
                    acting = snapshot_acting_seat(current)
                    if self._is_broker_seat(acting):
                        updated = drive_bot_turns(
                            client,
                            session,
                            current,
                            self._thread_bot_emit,
                            self.args.max_bot_actions,
                            watch_private_seat=session.watched_private_seat,
                            response_hook=self._thread_response,
                        )
                        self._record_table_transitions(updated)
                        if updated != current:
                            current = updated
                            continue
                        current = updated
                    if snapshot_acting_seat(current) == session.human_seat and not session.watches_llm_player:
                        seat_snapshot = client.seat_snapshot(
                            session.table_id,
                            session.human_seat,
                            session.human_controller_id,
                        )
                        self._thread_advice(seat_snapshot)
                        self.call_from_thread(self._handle_public_update, current, seat_snapshot, "")
                        return
                self.call_from_thread(self._handle_public_update, current, None, self._waiting_message(current))
                return
        except Exception as exc:  # pragma: no cover - defensive TUI boundary
            error_event(
                "client.textual_tui.advance_failed",
                table_id=session.table_id,
                human_seat=session.human_seat,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            self.call_from_thread(self._handle_nonfatal_error, exc)

    def _start_next_deal(self) -> JsonObject:
        session = self._session()
        response = self._client().start(session.table_id)
        self._thread_response(response)
        return response.get("snapshot") or self._client().table_snapshot(session.table_id)

    def _start_next_match(self) -> JsonObject:
        session = self._session()
        if session.player_mode == "human":
            self._join_human_player_for_next_match()
        session.bot_broker.join_and_ready_all()
        self._sync_broker_role_members()
        if session.watches_llm_player and session.human_seat in session.bot_broker.seats:
            session.human_controller_id = session.bot_broker.seats[session.human_seat].controller_id
        self._refresh_npc_metadata()
        response = self._client().start(session.table_id)
        self._thread_response(response)
        return response.get("snapshot") or self._client().table_snapshot(session.table_id)

    def _submit_action(self, kind: str) -> None:
        if self.busy:
            return
        self._set_busy("Submitting action...")
        self.run_worker(lambda: self._submit_action_worker(kind), thread=True, exclusive=True, name="submit-action")

    def _submit_action_worker(self, kind: str) -> None:
        session = self._session()
        client = self._client()
        seat_snapshot = self.seat_snapshot or {}
        selected = selected_card_ids_in_hand_order(seat_snapshot.get("hand", ()), self.selected_ids)
        try:
            if kind == "play":
                response = client.play_cards(
                    session.table_id,
                    session.human_seat,
                    session.human_controller_id,
                    selected,
                )
                command = "play " + " ".join(selected)
            elif kind == "pass":
                response = client.pass_turn(session.table_id, session.human_seat, session.human_controller_id)
                command = "pass"
            elif kind == "tribute":
                response = client.submit_tribute(
                    session.table_id,
                    session.human_seat,
                    session.human_controller_id,
                    selected[0],
                )
                command = f"tribute {selected[0]}"
            elif kind == "return":
                response = client.return_tribute(
                    session.table_id,
                    session.human_seat,
                    session.human_controller_id,
                    selected[0],
                )
                command = f"return {selected[0]}"
            else:
                raise GuandanClientError(None, f"unsupported TUI action: {kind}")
        except GuandanClientError as exc:
            error_event(
                "client.textual_tui.human_command_failed",
                table_id=session.table_id,
                seat=session.human_seat,
                command=safe_command_action(_command_for_error(kind, selected), seat_snapshot),
                status=exc.status,
                error_payload=exc.payload,
            )
            self.call_from_thread(self._handle_nonfatal_error, exc)
            return
        self._thread_response(response)
        debug_event(
            "client.textual_tui.human_command_completed",
            table_id=session.table_id,
            seat=session.human_seat,
            command=command_action(command, seat_snapshot),
            event_seq=response.get("event_seq"),
        )
        trigger_role_observers(session, session.human_seat, command_action(command, seat_snapshot), response)
        public_snapshot = response.get("snapshot") or client.table_snapshot(session.table_id)
        self.call_from_thread(self._clear_selection)
        self._advance_game(public_snapshot)

    def on_click(self, event: events.Click) -> None:
        widget_id = event.widget.id if event.widget is not None else ""
        if not widget_id:
            return
        if widget_id.startswith("card-"):
            self._toggle_card(widget_id)
            event.stop()
            return
        availability = action_availability(self.seat_snapshot, self.selected_ids)
        if widget_id == "play-action":
            self.action_play_selected()
            event.stop()
            return
        if widget_id == "pass-action" and availability.can_pass:
            self.action_pass_turn()
            event.stop()
            return
        if widget_id == "tribute-action" and availability.can_submit_tribute:
            self._submit_action("tribute")
            event.stop()
            return
        if widget_id == "return-action" and availability.can_return_tribute:
            self._submit_action("return")
            event.stop()
            return
        if widget_id == "clear-action":
            self.action_clear_selection()
            event.stop()
            return
        if widget_id == "refresh-action":
            self.action_refresh()
            event.stop()

    def action_play_selected(self) -> None:
        availability = action_availability(self.seat_snapshot, self.selected_ids)
        if availability.can_play:
            self._submit_action("play")
        elif availability.can_submit_tribute:
            self._submit_action("tribute")
        elif availability.can_return_tribute:
            self._submit_action("return")
        elif availability.reason:
            self._append_row(system_action_row(availability.reason))

    def action_pass_turn(self) -> None:
        if action_availability(self.seat_snapshot, self.selected_ids).can_pass:
            self._submit_action("pass")

    def action_clear_selection(self) -> None:
        self._clear_selection()

    def action_refresh(self) -> None:
        if self.busy or self.session is None:
            return
        self._set_busy("Refreshing...")
        self.run_worker(self._refresh_worker, thread=True, exclusive=True, name="refresh")

    def _refresh_worker(self) -> None:
        session = self._session()
        try:
            snapshot = self._client().table_snapshot(session.table_id)
        except Exception as exc:  # pragma: no cover - defensive TUI boundary
            self.call_from_thread(self._handle_nonfatal_error, exc)
            return
        self._advance_game(snapshot)

    def _tick(self) -> None:
        if self.public_snapshot:
            self._render_status()
        if self.busy or not self.public_snapshot:
            return
        deadline = self.public_snapshot.get("action_deadline_epoch_ms")
        if isinstance(deadline, int) and deadline <= int(time.time() * 1000):
            self.action_refresh()

    def _toggle_card(self, button_id: str) -> None:
        if self.busy:
            return
        try:
            slot = int(button_id.removeprefix("card-"))
        except ValueError:
            return
        card_id = self.card_by_slot.get(slot)
        if card_id is None:
            return
        seat_snapshot = self.seat_snapshot or {}
        eligible = seat_snapshot.get("eligible_card_ids")
        if isinstance(eligible, (list, tuple)) and eligible and card_id not in set(eligible):
            return
        if card_id in self.selected_ids:
            self.selected_ids.remove(card_id)
        else:
            self.selected_ids.add(card_id)
        self._render_hand()

    def _handle_public_update(
        self,
        public_snapshot: JsonObject,
        seat_snapshot: JsonObject | None,
        message: str,
    ) -> None:
        self.public_snapshot = public_snapshot
        self.seat_snapshot = seat_snapshot
        if seat_snapshot is not None:
            current_hand = set(str(card_id) for card_id in seat_snapshot.get("hand", ()) if card_id)
            self.selected_ids.intersection_update(current_hand)
        else:
            self.selected_ids.clear()
        self._render_public_snapshot(public_snapshot)
        self._render_hand()
        self._set_busy("")
        if message:
            self._append_row(system_action_row(message))

    def _render_public_snapshot(self, snapshot: JsonObject) -> None:
        self.public_snapshot = snapshot
        self._render_status()
        self._render_board()

    def _render_status(self) -> None:
        session = self.session
        view = table_view(
            self.public_snapshot,
            viewer_seat=session.human_seat if session else None,
            npc_metadata=session.npc_metadata if session else None,
        )
        selected_count = len(self.selected_ids)
        busy = " | Busy" if self.busy else ""
        timer = f" | Timer {view.timer}" if view.timer else ""
        seat = session.human_seat if session else "-"
        status = (
            f"{view.table_id} | {view.phase} | You {seat} | Turn {view.turn} "
            f"| Act {view.acting_seat} | Deal {view.deal_id} | {view.level_summary}{timer}"
            f" | Sel {selected_count}{busy}"
        )
        self.query_one("#status", Static).update(status)

    def _render_board(self) -> None:
        session = self.session
        view = table_view(
            self.public_snapshot,
            viewer_seat=session.human_seat if session else None,
            npc_metadata=session.npc_metadata if session else None,
        )
        for seat in view.seats:
            widget = self.query_one(f"#seat-{seat.seat.lower()}", Static)
            widget.set_class(seat.is_acting, "seat-active")
            widget.update(_seat_renderable(seat))
        self.query_one("#trick", Static).update(_trick_renderable(view.trick.summary))

    def _render_hand(self) -> None:
        snapshot = self.seat_snapshot or {}
        legal_action = snapshot.get("legal_action") or "-"
        eligible = snapshot.get("eligible_card_ids")
        eligible_ids = eligible if isinstance(eligible, (list, tuple)) else ()
        views = card_views(snapshot.get("hand", ()), selected_ids=self.selected_ids, eligible_card_ids=eligible_ids)
        self.card_by_slot = {index: view.card_id for index, view in enumerate(views)}
        for slot in range(MAX_HAND_CARDS):
            button = self.query_one(f"#card-{slot}", Static)
            view = views[slot] if slot < len(views) else None
            if view is None:
                button.display = False
                button.update("")
                continue
            button.display = True
            button.update(_card_cell_label(view.index, view.label))
            button.set_class(self.busy or not view.eligible, "disabled-control")
            button.set_class(view.selected, "selected-card")
            button.set_class(not view.eligible, "ineligible-card")
        availability = action_availability(self.seat_snapshot, self.selected_ids)
        title = f"Hand | Action {legal_action}"
        if availability.reason:
            title += f" | {availability.reason}"
        self.query_one("#hand-title", Static).update(title)
        self._set_action_enabled("play-action", availability.can_play)
        self._set_action_enabled("pass-action", availability.can_pass)
        self._set_action_enabled("tribute-action", availability.can_submit_tribute)
        self._set_action_enabled("return-action", availability.can_return_tribute)
        self._set_action_enabled("clear-action", bool(self.selected_ids))
        self._set_action_enabled("refresh-action", not self.busy)

    def _set_action_enabled(self, widget_id: str, enabled: bool) -> None:
        self.query_one(f"#{widget_id}", Static).set_class(self.busy or not enabled, "disabled-control")

    def _clear_selection(self) -> None:
        self.selected_ids.clear()
        self._render_hand()

    def _set_busy(self, message: str) -> None:
        self.busy = bool(message)
        if message:
            self.query_one("#hand-title", Static).update(message)
        self._render_status()
        self._render_hand()

    def _append_row(self, row: ActionRow) -> None:
        feed = self.query_one("#feed", DataTable)
        feed.add_row(
            _fit_cell(row.seq, FEED_COLUMN_WIDTHS["Seq"]),
            _fit_cell(row.seat, FEED_COLUMN_WIDTHS["Seat"]),
            _fit_cell(row.action, FEED_COLUMN_WIDTHS["Action"]),
            _fit_cell(row.cards, FEED_COLUMN_WIDTHS["Cards"]),
            _fit_cell(row.detail, FEED_COLUMN_WIDTHS["Detail"]),
        )
        feed.move_cursor(row=feed.row_count - 1)

    def _append_rows(self, rows: Iterable[ActionRow]) -> None:
        for row in rows:
            self._append_row(row)

    def _thread_emit(self, message: str) -> None:
        for line in message.splitlines():
            if line.strip():
                self.call_from_thread(self._append_row, system_action_row(line.strip()))

    def _thread_bot_emit(self, message: str) -> None:
        for line in message.splitlines():
            stripped = line.strip()
            if stripped and not _looks_like_command_event_line(stripped) and stripped != "No events.":
                self.call_from_thread(self._append_row, system_action_row(stripped))

    def _thread_response(self, response: JsonObject) -> None:
        rows = action_rows_from_response(response)
        if rows:
            self.call_from_thread(self._append_rows, rows)

    def _thread_advice(self, seat_snapshot: JsonObject) -> None:
        session = self._session()
        gossiper = session.table.members_for(session.human_seat).gossiper
        if gossiper is None or gossiper.policy is None:
            return
        try:
            from client.state_machine import _action_request_from_seat_snapshot, _format_advice

            action = gossiper.policy.choose_action(_action_request_from_seat_snapshot(session, seat_snapshot))
            self.call_from_thread(self._append_row, system_action_row(_format_advice(gossiper, action)))
        except Exception as exc:  # pragma: no cover - defensive TUI boundary
            self.call_from_thread(
                self._append_row,
                system_action_row(f"Advice from {gossiper.display_name}: unavailable ({exc})"),
            )

    def _record_table_transitions(self, snapshot: JsonObject) -> None:
        event_seq = snapshot.get("event_seq")
        if isinstance(event_seq, int) and event_seq == self._last_recorded_event_seq:
            return
        session = self._session()
        for transition in session.table.record_snapshot(snapshot):
            self.call_from_thread(self._append_row, system_action_row(transition.message))
            if transition.kind == "match_complete":
                seat_map = session.bot_broker.rotate_seats_after_match()
                session.rotate_seat_members(seat_map)
                self.call_from_thread(self._append_row, system_action_row(_format_seat_rotation(seat_map)))
        if isinstance(event_seq, int):
            self._last_recorded_event_seq = event_seq

    def _join_human_player_for_next_match(self) -> None:
        session = self._session()
        members = session.table.members_for(session.human_seat)
        human = members.player if members.player is not None and members.player.is_human else None
        if human is None:
            return
        response = self._client().join_human(
            session.table_id,
            session.human_seat,
            player_id=f"human-{session.human_seat}",
            controller_id=f"human-controller-{session.human_seat}",
            display_name=human.display_name,
        )
        controller_id = str(response.get("controller_id") or f"human-controller-{session.human_seat}")
        session.human_controller_id = controller_id
        human.controller_id = controller_id
        self._client().ready(session.table_id, session.human_seat, controller_id)

    def _sync_broker_role_members(self) -> None:
        session = self._session()
        for broker_seat in session.bot_broker.seats.values():
            member = session.table.members_for(broker_seat.seat).player
            if member is None:
                continue
            member.controller_id = broker_seat.controller_id
            member.profile_key = broker_seat.profile_key

    def _refresh_npc_metadata(self) -> None:
        session = self._session()
        session.npc_metadata = {
            seat: format_npc_metadata(broker_seat.policy)
            for seat, broker_seat in session.bot_broker.seats.items()
        }

    def _is_broker_seat(self, seat: object) -> bool:
        session = self._session()
        return isinstance(seat, str) and seat in session.bot_broker.seats

    def _waiting_message(self, snapshot: JsonObject) -> str:
        if snapshot.get("phase") in ACTIVE_ACTIONS:
            return ""
        waiting = snapshot_acting_seat(snapshot) or str(snapshot.get("phase") or "-")
        return f"Waiting for {waiting}."

    def _handle_nonfatal_error(self, exc: Exception) -> None:
        if isinstance(exc, GuandanClientError):
            self._append_row(system_action_row(format_client_error(exc)))
            if client_error_code(exc):
                self._append_row(system_action_row(f"Server rejection: {client_error_code(exc)}"))
        else:
            self._append_row(system_action_row(f"Error: {exc}"))
        self._set_busy("")
        if self.session is not None:
            self.action_refresh()

    def _handle_fatal_error(self, exc: Exception) -> None:
        self._result = Result(1, format_client_error(exc) if isinstance(exc, GuandanClientError) else f"Error: {exc}")
        self._append_row(system_action_row(self._result.output))
        self._set_busy("")

    def _client(self) -> GuandanHttpClient:
        if self.client is None:
            raise RuntimeError("client is not initialized")
        return self.client

    def _session(self) -> Session:
        if self.session is None:
            raise RuntimeError("session is not initialized")
        return self.session


def run_textual_play(args: argparse.Namespace, *, client: GuandanHttpClient | None = None) -> Result:
    app = GuandanTextualApp(args=args, client=client)
    result = app.run()
    return result if isinstance(result, Result) else app._result


def _seat_renderable(seat) -> Text:
    flags: list[str] = []
    if seat.is_viewer:
        flags.append("YOU")
    if seat.is_partner:
        flags.append("PARTNER")
    if seat.is_acting:
        flags.append("ACT")
    if seat.is_finished:
        flags.append("DONE")
    metadata = f" {_fit_cell(seat.metadata, 10)}" if seat.metadata else ""
    flag_text = f" {'/'.join(flags)}" if flags else ""
    return Text.assemble(
        (seat.seat, "bold cyan"),
        " ",
        (_fit_cell(seat.display_name or "-", 16), "bold"),
        " ",
        (str(seat.hand_count), "green"),
        " cards",
        (metadata, "dim"),
        (flag_text, "yellow"),
    )


def _trick_renderable(summary: str) -> Text:
    return Text.assemble(("Trick ", "bold blue"), _fit_cell(summary, 46))


def _card_cell_label(index: int, label: str) -> str:
    return _fit_cell(f"{index}:{label}", CARD_BUTTON_WIDTH)


def _command_for_error(kind: str, selected: tuple[str, ...]) -> str:
    if kind == "pass":
        return "pass"
    if kind == "tribute":
        return "tribute " + " ".join(selected)
    if kind == "return":
        return "return " + " ".join(selected)
    return "play " + " ".join(selected)


def _format_seat_rotation(seat_map: dict[str, str]) -> str:
    changes = [f"{old}->{new}" for old, new in seat_map.items() if old != new]
    if not changes:
        return "Seat roles stay in place for the next match."
    return "Seat roles rotated for next match: " + ", ".join(changes)


def _looks_like_command_event_line(value: str) -> bool:
    head, separator, _tail = value.partition(":")
    return bool(separator) and head.isdecimal()


def hand_columns_for_width(width: int) -> int:
    return HAND_GRID_COLUMNS if width >= MIN_UI_WIDTH else max(1, width // CARD_BUTTON_WIDTH)


def action_columns_for_width(width: int) -> int:
    return ACTION_GRID_COLUMNS if width >= MIN_UI_WIDTH else max(1, width // ACTION_BUTTON_WIDTH)


def feed_table_width() -> int:
    return sum(FEED_COLUMN_WIDTHS.values())


def _fit_cell(value: str, width: int) -> str:
    text = str(value)
    if width <= 3:
        return text[:width]
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."
