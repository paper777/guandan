from __future__ import annotations

import argparse
from dataclasses import dataclass

from client.api import GuandanHttpClient, JsonObject
from client.cli_app.render import format_npc_metadata
from npc.broker.broker import NpcBroker
from server.domain.seats import SEATS


@dataclass(frozen=True, slots=True)
class CliSession:
    table_id: str
    human_seat: str
    human_controller_id: str
    bot_broker: NpcBroker
    npc_metadata: dict[str, str]


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
    if public_snapshot.get("phase") not in {"PLAYING", "TRIBUTE", "DEAL_COMPLETE", "MATCH_COMPLETE"}:
        response = client.start(table_id, seed=args.seed)
        public_snapshot = response.get("snapshot") or client.table_snapshot(table_id)
    return CliSession(table_id, human_seat, human_controller_id, bot_broker, npc_metadata), public_snapshot
