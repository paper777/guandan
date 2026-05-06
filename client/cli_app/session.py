from __future__ import annotations

import argparse
import getpass
from dataclasses import dataclass, replace
from pathlib import Path

from client.api import GuandanHttpClient, JsonObject
from client.cli_app.render import format_npc_metadata
from npc.broker.broker import (
    BrokerSeat,
    NpcBroker,
    PlayerProfile,
    load_player_database,
    player_for_profile,
)
from server.domain.seats import SEATS


@dataclass(frozen=True, slots=True)
class CliSession:
    table_id: str
    human_seat: str
    human_controller_id: str
    bot_broker: NpcBroker
    npc_metadata: dict[str, str]
    player_mode: str = "human"


def prepare_default_table(client: GuandanHttpClient, args: argparse.Namespace) -> tuple[CliSession, JsonObject]:
    table_id = args.table_id or str(client.create_table()["table_id"])
    public_snapshot = client.table_snapshot(table_id)
    human_seat = str(args.seat)
    human_player_id = args.player_id or f"human-{human_seat}"
    human_controller_id = args.controller_id or f"human-controller-{human_seat}"
    human_display_name = args.display_name or getpass.getuser()
    player_mode = str(getattr(args, "player_mode", "human"))
    storage_dir = _npc_storage_dir(args.npc_player_config)
    seats = public_snapshot.get("seats", {})
    bot_broker = NpcBroker(client, table_id, player_db_path=args.npc_player_config, storage_dir=storage_dir)

    if player_mode == "llm":
        watched_seat = _add_watched_llm_seat(
            bot_broker,
            human_seat,
            display_name=human_display_name,
            existing_seats=seats,
            controller_id=human_controller_id,
            player_id=human_player_id,
            config_path=args.npc_player_config,
            storage_dir=storage_dir,
        )
        if watched_seat.controller_id:
            human_controller_id = watched_seat.controller_id
    elif human_seat not in seats:
        response = client.join_human(
            table_id,
            human_seat,
            player_id=human_player_id,
            controller_id=human_controller_id,
            display_name=human_display_name,
        )
        human_controller_id = str(response.get("controller_id", human_controller_id))
        public_snapshot = response.get("snapshot") or client.table_snapshot(table_id)
        seats = public_snapshot.get("seats", {})

    bot_broker.add_players(
        [seat.value for seat in SEATS if seat.value != human_seat and seat.value not in seats],
        lineup=args.npc_lineup,
        storage_dir=storage_dir,
    )

    if player_mode == "human":
        client.ready(table_id, human_seat, human_controller_id)
    bot_broker.join_and_ready_all()
    if player_mode == "llm":
        human_controller_id = bot_broker.seats[human_seat].controller_id

    npc_metadata = {
        broker_seat.seat: format_npc_metadata(broker_seat.policy)
        for broker_seat in bot_broker.seats.values()
    }

    public_snapshot = client.table_snapshot(table_id)
    if public_snapshot.get("phase") not in {"PLAYING", "TRIBUTE", "DEAL_COMPLETE", "MATCH_COMPLETE"}:
        response = client.start(table_id)
        public_snapshot = response.get("snapshot") or client.table_snapshot(table_id)
    return (
        CliSession(table_id, human_seat, human_controller_id, bot_broker, npc_metadata, player_mode),
        public_snapshot,
    )


def _add_watched_llm_seat(
    bot_broker: NpcBroker,
    seat: str,
    *,
    display_name: str,
    existing_seats: object,
    controller_id: str,
    player_id: str,
    config_path: str | None,
    storage_dir: Path,
) -> BrokerSeat:
    profile = _llm_profile_for_seat(
        seat,
        display_name=display_name,
        config_path=config_path,
        storage_dir=storage_dir,
    )
    broker_seat = bot_broker.add_seat(seat, player_for_profile(profile, "mixed", storage_dir), profile.display_name)
    if isinstance(existing_seats, dict) and seat in existing_seats:
        broker_seat.player_id = player_id
        broker_seat.controller_id = controller_id
    return broker_seat


def _llm_profile_for_seat(
    seat: str,
    *,
    display_name: str,
    config_path: str | None,
    storage_dir: Path,
) -> PlayerProfile:
    profiles = load_player_database(config_path or storage_dir / "players.json").profiles
    base = next(
        (profile for profile in profiles if profile.seat == seat),
        PlayerProfile(seat, display_name, "llm"),
    )
    return replace(base, display_name=display_name, kind="llm")


def _npc_storage_dir(config_path: str | None) -> Path:
    if config_path:
        return Path(config_path).expanduser().parent
    return Path("data")
