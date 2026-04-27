from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from client.api import GuandanClientError, GuandanHttpClient, JsonObject
from server.domain.seats import SEATS


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
SUIT_EMOJI = {
    "S": "♠️ ",
    "H": "♥️ ",
    "D": "♦️ ",
    "C": "♣️ ",
}


@dataclass(frozen=True, slots=True)
class CliResult:
    exit_code: int
    output: str


@dataclass(frozen=True, slots=True)
class CliSession:
    table_id: str
    human_seat: str
    human_controller_id: str
    bot_controller_ids: dict[str, str]


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def main(argv: Sequence[str] | None = None) -> int:
    result = run_cli(sys.argv[1:] if argv is None else argv, input_fn=input, output_fn=sys.stdout.write)
    return result.exit_code


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    input_fn: InputFn | None = None,
    output_fn: OutputFn | None = None,
    client: GuandanHttpClient | None = None,
) -> CliResult:
    parser = _build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    output: list[str] = []

    def emit(text: str = "") -> None:
        line = f"{text}\n"
        output.append(line)
        if output_fn is not None:
            output_fn(line)

    try:
        active_client = client or GuandanHttpClient(base_url=args.base_url)
        if args.command == "snapshot":
            emit(format_public_snapshot(active_client.table_snapshot(args.table_id)).rstrip())
            return CliResult(0, "".join(output))
        input_reader = input_fn or input
        return _run_play(args, active_client, input_reader, emit, output)
    except GuandanClientError as exc:
        emit(format_client_error(exc))
        return CliResult(1, "".join(output))


def _build_parser() -> argparse.ArgumentParser:
    default_base_url = os.environ.get("GUANDAN_BASE_URL", DEFAULT_BASE_URL)
    parser = argparse.ArgumentParser(prog="guandan-cli", description="Interactive Guandan HTTP client.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    play = subparsers.add_parser("play", help="Create or join a table and play interactively.")
    _add_connection_args(play, default_base_url)
    play.add_argument("--table-id", help="Existing table to join. Creates a table when omitted.")
    play.add_argument("--seat", choices=[seat.value for seat in SEATS], default="E", help="Human seat.")
    play.add_argument("--player-id", help="Human player ID. Defaults to human-<seat>.")
    play.add_argument("--controller-id", help="Human controller ID. Defaults to human-controller-<seat>.")
    play.add_argument("--display-name", help="Human display name. Defaults to the player ID.")
    play.add_argument("--seed", default="cli-demo", help="Deal seed used when starting a new match.")
    play.add_argument(
        "--max-bot-actions",
        type=int,
        default=128,
        help="Safety limit for consecutive automatic bot turns.",
    )

    snapshot = subparsers.add_parser("snapshot", help="Print a public table snapshot.")
    _add_connection_args(snapshot, default_base_url)
    snapshot.add_argument("table_id")
    return parser


def _add_connection_args(parser: argparse.ArgumentParser, default_base_url: str) -> None:
    parser.add_argument("--base-url", default=default_base_url, help="Guandan server base URL.")


def _normalize_argv(argv: Sequence[str] | None) -> list[str]:
    values = list(argv or [])
    if values and values[0] in {"play", "snapshot"}:
        return values
    return ["play", *values]


def _run_play(
    args: argparse.Namespace,
    client: GuandanHttpClient,
    input_fn: InputFn,
    emit: OutputFn,
    output: list[str],
) -> CliResult:
    session, public_snapshot = prepare_default_table(client, args)
    emit(f"Connected to table {session.table_id} as seat {session.human_seat}.")
    emit(format_public_snapshot(public_snapshot).rstrip())
    public_snapshot = drive_bot_turns(client, session, public_snapshot, emit, args.max_bot_actions)

    while public_snapshot.get("phase") == "PLAYING":
        current_turn = public_snapshot.get("current_turn")
        if current_turn != session.human_seat:
            updated = drive_bot_turns(client, session, public_snapshot, emit, args.max_bot_actions)
            if updated == public_snapshot:
                emit(f"Waiting for seat {current_turn}.")
                break
            public_snapshot = updated
            continue

        seat_snapshot = client.seat_snapshot(session.table_id, session.human_seat, session.human_controller_id)
        emit(format_seat_snapshot(seat_snapshot).rstrip())
        try:
            raw_command = input_fn("guandan> ")
        except EOFError:
            emit("Quit.")
            break
        command = raw_command.strip()
        if not command:
            continue
        if command in {"quit", "exit"}:
            emit("Quit.")
            break
        if command == "help":
            emit(help_text().rstrip())
            continue
        if command == "hand":
            emit(format_seat_snapshot(seat_snapshot).rstrip())
            continue
        if command == "table":
            public_snapshot = client.table_snapshot(session.table_id)
            emit(format_public_snapshot(public_snapshot).rstrip())
            continue
        try:
            response = submit_human_command(client, session, command, seat_snapshot)
        except GuandanClientError as exc:
            emit(format_client_error(exc))
            public_snapshot = client.table_snapshot(session.table_id)
            continue
        emit(format_command_response(response).rstrip())
        public_snapshot = response.get("snapshot") or client.table_snapshot(session.table_id)
        public_snapshot = drive_bot_turns(client, session, public_snapshot, emit, args.max_bot_actions)

    if public_snapshot.get("phase") != "PLAYING":
        emit(format_public_snapshot(public_snapshot).rstrip())
    return CliResult(0, "".join(output))


def prepare_default_table(client: GuandanHttpClient, args: argparse.Namespace) -> tuple[CliSession, JsonObject]:
    table_id = args.table_id or str(client.create_table()["table_id"])
    public_snapshot = client.table_snapshot(table_id)
    human_seat = str(args.seat)
    human_player_id = args.player_id or f"human-{human_seat}"
    human_controller_id = args.controller_id or f"human-controller-{human_seat}"
    seats = public_snapshot.get("seats", {})
    if human_seat not in seats:
        response = client.join_human(
            table_id,
            human_seat,
            player_id=human_player_id,
            controller_id=human_controller_id,
            display_name=args.display_name or human_player_id,
        )
        human_controller_id = str(response.get("controller_id", human_controller_id))
        public_snapshot = response.get("snapshot") or client.table_snapshot(table_id)

    bot_controller_ids: dict[str, str] = {}
    seats = public_snapshot.get("seats", {})
    for seat in [seat.value for seat in SEATS if seat.value != human_seat]:
        if seat in seats:
            player = seats.get(seat)
            if isinstance(player, dict) and player.get("kind") == "bot":
                bot_controller_ids[seat] = f"bot-controller-{seat}"
            continue
        controller_id = f"bot-controller-{seat}"
        response = client.join_local_bot(
            table_id,
            seat,
            player_id=f"bot-{seat}",
            controller_id=controller_id,
            display_name=f"Bot {seat}",
        )
        bot_controller_ids[seat] = str(response.get("controller_id", controller_id))
        public_snapshot = response.get("snapshot") or client.table_snapshot(table_id)
        seats = public_snapshot.get("seats", {})

    for seat, controller_id in {human_seat: human_controller_id, **bot_controller_ids}.items():
        client.ready(table_id, seat, controller_id)

    public_snapshot = client.table_snapshot(table_id)
    if public_snapshot.get("phase") != "PLAYING":
        response = client.start(table_id, seed=args.seed)
        public_snapshot = response.get("snapshot") or client.table_snapshot(table_id)
    return CliSession(table_id, human_seat, human_controller_id, bot_controller_ids), public_snapshot


def drive_bot_turns(
    client: GuandanHttpClient,
    session: CliSession,
    public_snapshot: JsonObject,
    emit: OutputFn,
    max_actions: int,
) -> JsonObject:
    current = public_snapshot
    actions = 0
    while current.get("phase") == "PLAYING":
        seat = current.get("current_turn")
        if not isinstance(seat, str) or seat not in session.bot_controller_ids:
            return current
        if actions >= max_actions:
            emit("Stopped automatic bot play after reaching the safety limit.")
            return current
        controller_id = session.bot_controller_ids[seat]
        snapshot = client.seat_snapshot(session.table_id, seat, controller_id)
        try:
            if snapshot.get("legal_action") == "lead" and snapshot.get("hand"):
                response = client.play_cards(session.table_id, seat, controller_id, (str(snapshot["hand"][0]),))
            else:
                response = client.pass_turn(session.table_id, seat, controller_id)
        except GuandanClientError as exc:
            emit(format_client_error(exc))
            return client.table_snapshot(session.table_id)
        emit(format_command_response(response).rstrip())
        current = response.get("snapshot") or client.table_snapshot(session.table_id)
        actions += 1
    return current


def submit_human_command(
    client: GuandanHttpClient,
    session: CliSession,
    command: str,
    seat_snapshot: JsonObject | None = None,
) -> JsonObject:
    parts = command.split()
    action = parts[0]
    if action == "play" and len(parts) > 1:
        return client.play_cards(
            session.table_id,
            session.human_seat,
            session.human_controller_id,
            resolve_card_inputs(parts[1:], seat_snapshot),
        )
    if action == "pass" and len(parts) == 1:
        return client.pass_turn(session.table_id, session.human_seat, session.human_controller_id)
    raise GuandanClientError(None, f"unsupported command: {command}")


def format_public_snapshot(snapshot: JsonObject) -> str:
    seats = snapshot.get("seats", {})
    hand_counts = snapshot.get("hand_counts", {})
    lines = [
        f"Table: {snapshot.get('table_id', '-')}",
        f"Phase: {snapshot.get('phase', '-')}",
        f"Seq: {snapshot.get('event_seq', '-')}",
        f"Turn: {snapshot.get('current_turn') or '-'}",
    ]
    timer = format_timer(snapshot)
    if timer is not None:
        lines.append(f"Timer: {timer}")
    lines.append("Seats:")
    for seat in [seat.value for seat in SEATS]:
        player = seats.get(seat)
        name = player.get("display_name", "-") if isinstance(player, dict) else "-"
        count = hand_counts.get(seat, 0)
        lines.append(f"  {seat}: {name} ({count} cards)")
    finish_order = snapshot.get("finish_order") or ()
    if finish_order:
        lines.append("Finish: " + " ".join(str(seat) for seat in finish_order))
    return "\n".join(lines) + "\n"


def format_seat_snapshot(snapshot: JsonObject) -> str:
    lines = [format_public_snapshot(snapshot["public"]).rstrip()]
    lines.append(f"Your seat: {snapshot.get('seat')}")
    lines.append(f"Legal action: {snapshot.get('legal_action') or '-'}")
    lines.append("Hand: " + format_hand(snapshot.get("hand", ())))
    return "\n".join(lines) + "\n"


def format_command_response(response: JsonObject) -> str:
    events = response.get("events", [])
    if not events:
        return "No events.\n"
    return "\n".join(format_event(event) for event in events) + "\n"


def format_event(event: JsonObject) -> str:
    payload = event.get("payload", {})
    seat = payload.get("seat")
    event_type = event.get("type")
    seq = event.get("seq")
    if event_type == "CardsPlayed":
        cards = format_card_list(payload.get("card_ids", ()))
        return f"{seq}: {seat} played {payload.get('hand_type')} {cards}"
    if event_type == "PlayerPassed":
        return f"{seq}: {seat} passed"
    return f"{seq}: {event_type} {payload}"


def help_text() -> str:
    return "\n".join(
        (
            "Commands:",
            "  play <hand-number-or-card-id> [<hand-number-or-card-id>...]",
            "  pass",
            "  hand",
            "  table",
            "  help",
            "  quit",
        )
    ) + "\n"


def format_client_error(error: GuandanClientError) -> str:
    status = f"HTTP {error.status}: " if error.status is not None else ""
    return f"Error: {status}{error}"


def format_timer(snapshot: JsonObject) -> str | None:
    deadline = snapshot.get("action_deadline_epoch_ms")
    if not isinstance(deadline, int):
        return None
    remaining = max(0, int((deadline - int(time.time() * 1000) + 999) / 1000))
    return f"{remaining}s remaining"


def format_hand(card_ids: object) -> str:
    if not isinstance(card_ids, (list, tuple)):
        return ""
    return "  ".join(f"{index}: {format_card_id(str(card_id))}" for index, card_id in enumerate(card_ids, start=1))


def format_card_list(card_ids: object) -> str:
    if not isinstance(card_ids, (list, tuple)):
        return str(card_ids)
    return "[" + ", ".join(format_card_id(str(card_id)) for card_id in card_ids) + "]"


def format_card_id(card_id: str) -> str:
    parts = card_id.split("-")
    deck_label = ""
    if len(parts) == 2 and parts[0].startswith("D"):
        # deck_label = "" if parts[0] == "D1" else f"{parts[0]} "
        joker = f"🃏{parts[1]}" if parts[1] in {"SJ", "BJ"} else parts[1]
        return f"{deck_label}{joker}"
    if len(parts) == 3 and parts[0].startswith("D"):
        # deck_label = "" if parts[0] == "D1" else f"{parts[0]} "
        suit = SUIT_EMOJI.get(parts[1], parts[1])
        return f"{deck_label}{suit}{parts[2]}"
    return card_id


def resolve_card_inputs(tokens: list[str], snapshot: JsonObject | None) -> tuple[str, ...]:
    hand = snapshot.get("hand", ()) if snapshot is not None else ()
    resolved: list[str] = []
    for token in tokens:
        if token.isdecimal() and isinstance(hand, (list, tuple)):
            index = int(token) - 1
            if index < 0 or index >= len(hand):
                raise GuandanClientError(None, f"hand index out of range: {token}")
            resolved.append(str(hand[index]))
        else:
            resolved.append(token)
    return tuple(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
