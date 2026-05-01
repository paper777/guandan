from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from client.api import GuandanClientError, GuandanHttpClient, JsonObject
from npc.broker.broker import NPC_LINEUPS, NpcBroker
from server.domain.seats import SEATS


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
SUIT_EMOJI = {
    "S": "♠️ ",
    "H": "♥️ ",
    "D": "♦️ ",
    "C": "♣️ ",
}
SUIT_INPUT_ALIASES = {
    "S": "S",
    "SPADE": "S",
    "SPADES": "S",
    "♠": "S",
    "♠️": "S",
    "H": "H",
    "HEART": "H",
    "HEARTS": "H",
    "♥": "H",
    "♥️": "H",
    "D": "D",
    "DIAMOND": "D",
    "DIAMONDS": "D",
    "♦": "D",
    "♦️": "D",
    "C": "C",
    "CLUB": "C",
    "CLUBS": "C",
    "♣": "C",
    "♣️": "C",
}
RANK_SORT_ORDER = {
    rank: index
    for index, rank in enumerate(("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "SJ", "BJ"))
}
SUIT_SORT_ORDER = {suit: index for index, suit in enumerate(("S", "H", "D", "C"))}


@dataclass(frozen=True, slots=True)
class CliResult:
    exit_code: int
    output: str


@dataclass(frozen=True, slots=True)
class CliSession:
    table_id: str
    human_seat: str
    human_controller_id: str
    bot_broker: NpcBroker
    npc_metadata: dict[str, str]


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
    play.add_argument(
        "--npc-lineup",
        choices=NPC_LINEUPS,
        default="mixed",
        help="NPC lineup for broker-controlled seats.",
    )
    play.add_argument(
        "--npc-player-config",
        help="JSON file for default NPC player names and kinds.",
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
    emit(f"Table {session.table_id} | You are {session.human_seat}")
    emit(format_public_snapshot(public_snapshot, npc_metadata=session.npc_metadata).rstrip())
    public_snapshot = drive_bot_turns(client, session, public_snapshot, emit, args.max_bot_actions)

    while public_snapshot.get("phase") == "PLAYING":
        acting_seat = snapshot_acting_seat(public_snapshot)
        if acting_seat != session.human_seat:
            updated = drive_bot_turns(client, session, public_snapshot, emit, args.max_bot_actions)
            if updated == public_snapshot:
                emit(f"Waiting for {acting_seat}.")
                break
            public_snapshot = updated
            continue

        seat_snapshot = client.seat_snapshot(session.table_id, session.human_seat, session.human_controller_id)
        emit(format_seat_snapshot(seat_snapshot, npc_metadata=session.npc_metadata).rstrip())
        try:
            raw_command = input_fn("guandan> ")
        except EOFError:
            emit("Quit.")
            break
        latest_snapshot = client.table_snapshot(session.table_id)
        if latest_snapshot.get("phase") != "PLAYING":
            public_snapshot = latest_snapshot
            continue
        if _human_turn_elapsed(latest_snapshot, session):
            public_snapshot = drive_bot_turns(client, session, latest_snapshot, emit, args.max_bot_actions)
            continue
        public_snapshot = latest_snapshot
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
            emit(format_seat_snapshot(seat_snapshot, npc_metadata=session.npc_metadata).rstrip())
            continue
        if command == "table":
            public_snapshot = client.table_snapshot(session.table_id)
            emit(format_public_snapshot(public_snapshot, npc_metadata=session.npc_metadata).rstrip())
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
        emit(format_public_snapshot(public_snapshot, npc_metadata=session.npc_metadata).rstrip())
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

    bot_broker = NpcBroker(client, table_id)
    seats = public_snapshot.get("seats", {})
    broker_seats = bot_broker.add_default_players(
        [seat.value for seat in SEATS if seat.value != human_seat and seat.value not in seats],
        lineup=args.npc_lineup,
        config_path=args.npc_player_config,
    )
    npc_metadata = {broker_seat.seat: format_npc_metadata(broker_seat.policy) for broker_seat in broker_seats}

    client.ready(table_id, human_seat, human_controller_id)
    bot_broker.join_and_ready_all()

    public_snapshot = client.table_snapshot(table_id)
    if public_snapshot.get("phase") != "PLAYING":
        response = client.start(table_id, seed=args.seed)
        public_snapshot = response.get("snapshot") or client.table_snapshot(table_id)
    return CliSession(table_id, human_seat, human_controller_id, bot_broker, npc_metadata), public_snapshot


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
    return snapshot.get("phase") == "PLAYING" and snapshot_acting_seat(snapshot) != session.human_seat


def snapshot_acting_seat(snapshot: JsonObject) -> str | None:
    seat = snapshot.get("acting_seat") or snapshot.get("current_turn")
    return seat if isinstance(seat, str) else None


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


def format_public_snapshot(
    snapshot: JsonObject,
    *,
    viewer_seat: object = None,
    npc_metadata: dict[str, str] | None = None,
) -> str:
    seats = snapshot.get("seats", {})
    hand_counts = snapshot.get("hand_counts", {})
    header = f"{snapshot.get('phase', '-')}"
    if viewer_seat is not None:
        header += f" | Seat {viewer_seat}"
    header += f" | Turn {snapshot.get('current_turn') or '-'}"
    lines = [header]
    timer = format_timer(snapshot)
    if timer is not None:
        lines.append(f"Timer: {timer}")
    metadata = npc_metadata or {}
    lines.append("Players: " + " | ".join(format_seat_summary(seat.value, seats, hand_counts, metadata) for seat in SEATS))
    finish_order = snapshot.get("finish_order") or ()
    if finish_order:
        lines.append("Finish: " + " ".join(str(seat) for seat in finish_order))
    return "\n".join(lines) + "\n"


def format_seat_snapshot(snapshot: JsonObject, *, npc_metadata: dict[str, str] | None = None) -> str:
    lines = [format_public_snapshot(snapshot["public"], viewer_seat=snapshot.get("seat"), npc_metadata=npc_metadata).rstrip()]
    lines.append(f"Action: {snapshot.get('legal_action') or '-'}")
    lines.append("Hand: " + format_hand(snapshot.get("hand", ())))
    return "\n".join(lines) + "\n"


def format_seat_summary(seat: str, seats: object, hand_counts: object, npc_metadata: dict[str, str] | None = None) -> str:
    player = seats.get(seat) if isinstance(seats, dict) else None
    name = player.get("display_name", "-") if isinstance(player, dict) else "-"
    count = hand_counts.get(seat, 0) if isinstance(hand_counts, dict) else 0
    metadata = (npc_metadata or {}).get(seat)
    suffix = f" [{metadata}]" if metadata else ""
    return f"{seat} {name} {count}{suffix}"


def format_npc_metadata(policy: object) -> str:
    config = getattr(policy, "config", None)
    if config is None:
        return ""
    provider = str(getattr(config, "provider_name", "") or "").strip()
    model = str(getattr(config, "model_name", "") or "").strip()
    if provider and model:
        return f"{provider}/{model}"
    return provider or model


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
            "  play <card-label-or-id> [<card-label-or-id>...]",
            "    examples: play S3, play H10 C10, play SJ",
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


def client_error_code(error: GuandanClientError) -> str | None:
    rejection = error.payload.get("rejection")
    if not isinstance(rejection, dict):
        return None
    code = rejection.get("code")
    return code if isinstance(code, str) else None


def format_timer(snapshot: JsonObject) -> str | None:
    deadline = snapshot.get("action_deadline_epoch_ms")
    if not isinstance(deadline, int):
        return None
    remaining = max(0, int((deadline - int(time.time() * 1000) + 999) / 1000))
    return f"{remaining}s remaining"


def format_hand(card_ids: object) -> str:
    if not isinstance(card_ids, (list, tuple)):
        return ""
    return "  ".join(format_card_id(card_id) for card_id in sort_card_ids(card_ids))


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


def sort_card_ids(card_ids: object) -> tuple[str, ...]:
    if not isinstance(card_ids, (list, tuple)):
        return ()
    return tuple(sorted((str(card_id) for card_id in card_ids), key=card_sort_key))


def card_sort_key(card_id: str) -> tuple[int, int, int, str]:
    parts = card_id.split("-")
    if len(parts) == 3 and parts[0].startswith("D"):
        return (RANK_SORT_ORDER.get(parts[2], 99), SUIT_SORT_ORDER.get(parts[1], 99), _deck_sort_value(parts[0]), card_id)
    if len(parts) == 2 and parts[0].startswith("D"):
        return (RANK_SORT_ORDER.get(parts[1], 99), 99, _deck_sort_value(parts[0]), card_id)
    return (99, 99, 99, card_id)


def _deck_sort_value(deck: str) -> int:
    value = deck.removeprefix("D")
    return int(value) if value.isdecimal() else 99


def resolve_card_inputs(tokens: list[str], snapshot: JsonObject | None) -> tuple[str, ...]:
    hand = sort_card_ids(snapshot.get("hand", ())) if snapshot is not None else ()
    resolved: list[str] = []
    used: set[str] = set()
    for token in tokens:
        if token.isdecimal():
            index = int(token) - 1
            if index < 0 or index >= len(hand):
                raise GuandanClientError(None, f"hand index out of range: {token}")
            card_id = str(hand[index])
            if card_id in used:
                raise GuandanClientError(None, f"card already selected: {format_card_id(card_id)}")
            resolved.append(card_id)
            used.add(card_id)
        else:
            card_id = resolve_card_label(token, hand, used)
            resolved.append(card_id)
            used.add(card_id)
    return tuple(resolved)


def resolve_card_label(token: str, hand: tuple[str, ...], used: set[str]) -> str:
    if token in hand:
        if token in used:
            raise GuandanClientError(None, f"card already selected: {format_card_id(token)}")
        return token
    if not hand:
        return token
    label = normalized_card_label(token)
    if label is None:
        return token
    for card_id in hand:
        if card_id in used:
            continue
        if normalized_card_label(card_id) == label:
            return card_id
    raise GuandanClientError(None, f"card not in hand: {token}")


def normalized_card_label(value: str) -> tuple[str | None, str] | None:
    raw = (
        value.strip()
        .upper()
        .replace("\ufe0f", "")
        .replace("🃏", "")
    )
    parts = raw.replace("_", "-").replace(":", "-").split("-")
    if len(parts) == 2 and parts[0].startswith("D"):
        rank = normalize_rank(parts[1])
        return (None, rank) if rank in {"SJ", "BJ"} else None
    if len(parts) == 3 and parts[0].startswith("D"):
        suit = SUIT_INPUT_ALIASES.get(parts[1])
        rank = normalize_rank(parts[2])
        return (suit, rank) if suit is not None and rank in RANK_SORT_ORDER else None

    token = raw.replace("-", "").replace("_", "").replace(":", "").replace(" ", "")
    if not token:
        return None
    if token in {"SJ", "SMALLJOKER", "SMALL"}:
        return (None, "SJ")
    if token in {"BJ", "BIGJOKER", "BIG"}:
        return (None, "BJ")
    for suit_alias in sorted(SUIT_INPUT_ALIASES, key=len, reverse=True):
        if token.startswith(suit_alias):
            suit = SUIT_INPUT_ALIASES[suit_alias]
            rank = normalize_rank(token[len(suit_alias) :])
            return (suit, rank) if rank in RANK_SORT_ORDER else None
    return None


def normalize_rank(value: str) -> str:
    rank = value.upper()
    if rank == "T":
        return "10"
    return rank


if __name__ == "__main__":
    raise SystemExit(main())
