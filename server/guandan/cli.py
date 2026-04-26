from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

from guandan.domain.commands import JoinTable, Pass, PlayCards, Ready, StartMatch
from guandan.domain.controllers import ControllerCapability, ControllerKind, ControllerRef, PlayerKind, PlayerRef
from guandan.domain.events import Event
from guandan.domain.seats import SEATS, Seat
from guandan.domain.state import MatchPhase, MatchState
from guandan.services.snapshots import public_snapshot, seat_snapshot
from guandan.services.table_actor import ActorResult, TableActor


BOT_CAPABILITIES = frozenset(
    {
        ControllerCapability.PLAY,
        ControllerCapability.OBSERVE_PUBLIC,
        ControllerCapability.OBSERVE_PRIVATE,
        ControllerCapability.AUTO_READY,
    }
)


@dataclass(frozen=True, slots=True)
class CliResult:
    exit_code: int
    output: str


def main(argv: Sequence[str] | None = None) -> int:
    result = run_cli(argv)
    print(result.output, end="")
    return result.exit_code


def run_cli(argv: Sequence[str] | None = None) -> CliResult:
    parser = argparse.ArgumentParser(prog="guandan-cli", description="Local Guandan server CLI client.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Create a local bot table and play a few automatic turns.")
    demo.add_argument("--seed", default="cli-demo", help="Deterministic deal seed.")
    demo.add_argument("--turns", type=int, default=8, help="Maximum automatic bot actions to attempt.")

    snapshot = subparsers.add_parser("snapshot", help="Create a local bot table and print its starting snapshot.")
    snapshot.add_argument("--seed", default="cli-demo", help="Deterministic deal seed.")

    args = parser.parse_args(argv)
    if args.command == "snapshot":
        actor = create_started_bot_table(seed=args.seed)
        return CliResult(0, format_snapshot(actor.state))
    if args.command == "demo":
        actor = create_started_bot_table(seed=args.seed)
        lines = [format_snapshot(actor.state).rstrip(), ""]
        for _ in range(args.turns):
            if actor.state.phase != MatchPhase.PLAYING:
                break
            result = play_one_bot_action(actor)
            lines.extend(format_actor_result(result))
            if result.rejection is not None:
                break
        lines.append(format_snapshot(actor.state).rstrip())
        lines.append("")
        return CliResult(0, "\n".join(lines))
    return CliResult(2, "unknown command\n")


def create_started_bot_table(seed: str = "cli-demo") -> TableActor:
    actor = TableActor(table_id="cli-table")
    for seat in SEATS:
        player = PlayerRef(id=f"bot-{seat.value}", display_name=f"Bot {seat.value}", kind=PlayerKind.BOT)
        controller = bot_controller(seat)
        _must_dispatch(actor, JoinTable(player=player, controller=controller, requested_seat=seat))
    for seat in SEATS:
        _must_dispatch(actor, Ready(controller_id=bot_controller(seat).id, seat=seat))
    _must_dispatch(actor, StartMatch(seed=seed))
    return actor


def bot_controller(seat: Seat) -> ControllerRef:
    return ControllerRef(
        id=f"bot-controller-{seat.value}",
        kind=ControllerKind.LOCAL_BOT,
        seat=seat,
        player_id=f"bot-{seat.value}",
        capabilities=BOT_CAPABILITIES,
    )


def play_one_bot_action(actor: TableActor) -> ActorResult:
    state = actor.state
    if state.deal is None:
        raise RuntimeError("table has not started")
    seat = state.deal.turn
    controller_id = bot_controller(seat).id
    snapshot = seat_snapshot(state, seat, controller_id)
    if state.deal.current_trick.last_play is None and snapshot.hand:
        command = PlayCards(controller_id=controller_id, seat=seat, card_ids=(snapshot.hand[0],))
    else:
        command = Pass(controller_id=controller_id, seat=seat)
    return actor.dispatch(command, controller_id=controller_id)


def format_snapshot(state: MatchState) -> str:
    snapshot = public_snapshot(state)
    lines = [
        f"Table: {snapshot.table_id}",
        f"Phase: {snapshot.phase.value}",
        f"Seq: {snapshot.event_seq}",
        f"Turn: {snapshot.current_turn.value if snapshot.current_turn else '-'}",
        "Seats:",
    ]
    for seat in SEATS:
        player = snapshot.seats.get(seat)
        count = snapshot.hand_counts.get(seat, 0)
        name = player.display_name if player else "-"
        lines.append(f"  {seat.value}: {name} ({count} cards)")
    if snapshot.finish_order:
        lines.append("Finish: " + " ".join(seat.value for seat in snapshot.finish_order))
    return "\n".join(lines) + "\n"


def format_actor_result(result: ActorResult) -> list[str]:
    if result.rejection is not None:
        return [f"Rejected: {result.rejection.code.value} - {result.rejection.message}"]
    return [format_event(event) for event in result.events]


def format_event(event: Event) -> str:
    seat = event.payload.get("seat")
    if event.type == "CardsPlayed":
        return f"{event.seq}: {seat} played {event.payload.get('hand_type')} {event.payload.get('card_ids')}"
    if event.type == "PlayerPassed":
        return f"{event.seq}: {seat} passed"
    return f"{event.seq}: {event.type} {event.payload}"


def _must_dispatch(actor: TableActor, command) -> None:
    result = actor.dispatch(command)
    if result.rejection is not None:
        raise RuntimeError(f"{result.rejection.code.value}: {result.rejection.message}")


if __name__ == "__main__":
    raise SystemExit(main())
