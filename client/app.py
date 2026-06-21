from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from client.http_client import GuandanClientError, GuandanHttpClient
from client.tui.render import format_client_error, format_public_snapshot
from client.session import prepare_default_table
from client.state_machine import StateMachine
from client.tui.types import Result, InputFn, OutputFn
from db.player import NPC_LINEUPS


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def main(argv: Sequence[str] | None = None) -> int:
    result = run_cli(sys.argv[1:] if argv is None else argv, input_fn=input, output_fn=sys.stdout.write)
    return result.exit_code


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    input_fn: InputFn | None = None,
    output_fn: OutputFn | None = None,
    client: GuandanHttpClient | None = None,
) -> Result:
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
            return Result(0, "".join(output))
        input_reader = input_fn or input
        session, public_snapshot = prepare_default_table(active_client, args)
        seat_label = "Watching" if session.player_mode == "llm" else "You are"
        emit(f"Table {session.table_id} | {seat_label} {session.human_seat}")
        emit(format_public_snapshot(public_snapshot, viewer_seat=session.human_seat, npc_metadata=session.npc_metadata).rstrip())
        machine = StateMachine(
            args=args,
            client=active_client,
            session=session,
            input_fn=input_reader,
            emit=emit,
        )
        machine.run(public_snapshot)
        return Result(0, "".join(output))
    except GuandanClientError as exc:
        emit(format_client_error(exc))
        return Result(1, "".join(output))


def _build_parser() -> argparse.ArgumentParser:
    default_base_url = os.environ.get("GUANDAN_BASE_URL", DEFAULT_BASE_URL)
    parser = argparse.ArgumentParser(prog="guandan-cli", description="Interactive Guandan HTTP client.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    play = subparsers.add_parser("play", help="Create or join a table and play interactively.")
    _add_connection_args(play, default_base_url)
    play.add_argument("--table-id", help="Existing table to join. Creates a table when omitted.")
    play.add_argument("--player-id", help="Human player ID. Defaults to human-<seat>.")
    play.add_argument("--controller-id", help="Human controller ID. Defaults to human-controller-<seat>.")
    play.add_argument("--display-name", help="Human display name. Defaults to the OS login name.")
    play.add_argument(
        "--player-mode",
        dest="player_mode",
        choices=("human", "llm"),
        default="human",
        help="Who controls the selected seat. Use llm to watch an LLM agent play that seat.",
    )
    play.add_argument(
        "--gossiper-mode",
        choices=("none", "llm"),
        default="none",
        help="Optional advisor role on the selected seat.",
    )
    play.add_argument(
        "--max-bot-actions",
        type=int,
        default=128,
        help="Safety limit for consecutive automatic bot turns.",
    )
    play.add_argument(
        "--npc-lineup",
        choices=NPC_LINEUPS,
        default="rl",
        help="NPC lineup for broker-controlled seats.",
    )
    play.add_argument(
        "--npc-player-config",
        help="Player storage directory for NPC profiles, stats, memory, and actions.",
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
