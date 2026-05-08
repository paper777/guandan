from __future__ import annotations

import argparse
import getpass
from dataclasses import dataclass, field, replace
from pathlib import Path
from random import SystemRandom

from client.http_client import GuandanHttpClient
from client.types import JsonObject, SeatMember, SeatRole, Table
from client.tui.render import format_npc_metadata
from client.broker import BrokerSeat, NpcBroker
from db.player import PlayerProfile, load_player_database, player_for_profile
from server.domain.seats import SEATS


STARTED_PHASES = {"PLAYING", "TRIBUTE", "DEAL_COMPLETE", "MATCH_COMPLETE"}


@dataclass(slots=True)
class Session:
    table_id: str
    human_seat: str
    human_controller_id: str
    bot_broker: NpcBroker
    npc_metadata: dict[str, str]
    player_mode: str = "human"
    table: Table = field(default_factory=lambda: Table(""))

    @property
    def watches_llm_player(self) -> bool:
        return self.player_mode == "llm"

    @property
    def watched_private_seat(self) -> str | None:
        return self.human_seat if self.watches_llm_player else None

    def rotate_seat_members(self, seat_map: dict[str, str]) -> None:
        self.table.rotate_seat_members(seat_map)
        self.human_seat = seat_map.get(self.human_seat, self.human_seat)


def prepare_default_table(client: GuandanHttpClient, args: argparse.Namespace) -> tuple[Session, JsonObject]:
    table_id, public_snapshot = _load_or_create_table(client, args.table_id)
    role_table = Table(table_id)
    human_display_name = args.display_name or getpass.getuser()
    player_mode = str(getattr(args, "player_mode", "human"))
    gossiper_mode = str(getattr(args, "gossiper_mode", "none"))
    storage_dir = _npc_storage_dir(args.npc_player_config)
    occupied_seats = _snapshot_seats(public_snapshot)
    human_seat = _choose_available_seat(occupied_seats)
    human_player_id = args.player_id or f"human-{human_seat}"
    human_controller_id = args.controller_id or f"human-controller-{human_seat}"
    bot_broker = NpcBroker(client, table_id, player_db_path=args.npc_player_config, storage_dir=storage_dir)

    human_controller_id, watched_profile_key, public_snapshot = _attach_user_seat(
        client,
        bot_broker,
        table_id,
        human_seat,
        human_display_name=human_display_name,
        player_mode=player_mode,
        human_player_id=human_player_id,
        human_controller_id=human_controller_id,
        occupied_seats=occupied_seats,
        config_path=args.npc_player_config,
        storage_dir=storage_dir,
    )
    occupied_seats = _snapshot_seats(public_snapshot)
    _add_human_role(role_table, human_seat, human_display_name, human_controller_id, player_mode)

    _add_remaining_npc_players(
        bot_broker,
        human_seat,
        occupied_seats,
        lineup=args.npc_lineup,
        storage_dir=storage_dir,
        watched_profile_key=watched_profile_key,
    )
    human_controller_id = _ready_players(client, bot_broker, table_id, human_seat, human_controller_id, player_mode)
    _add_broker_player_roles(role_table, bot_broker)
    _add_gossiper_role(
        role_table,
        human_seat,
        gossiper_mode=gossiper_mode,
        config_path=args.npc_player_config,
        storage_dir=storage_dir,
    )
    npc_metadata = _npc_metadata(bot_broker)
    public_snapshot = _ensure_match_started(client, table_id)

    return (
        Session(table_id, human_seat, human_controller_id, bot_broker, npc_metadata, player_mode, role_table),
        public_snapshot,
    )


def _load_or_create_table(client: GuandanHttpClient, table_id: str | None) -> tuple[str, JsonObject]:
    selected_table_id = table_id or str(client.create_table()["table_id"])
    return selected_table_id, client.table_snapshot(selected_table_id)


def _attach_user_seat(
    client: GuandanHttpClient,
    bot_broker: NpcBroker,
    table_id: str,
    human_seat: str,
    *,
    human_display_name: str,
    player_mode: str,
    human_player_id: str,
    human_controller_id: str,
    occupied_seats: dict[object, object],
    config_path: str | None,
    storage_dir: Path,
) -> tuple[str, str | None, JsonObject]:
    if player_mode == "llm":
        watched_seat = _add_watched_llm_seat(
            bot_broker,
            human_seat,
            display_name=human_display_name,
            existing_seats=occupied_seats,
            controller_id=human_controller_id,
            player_id=human_player_id,
            config_path=config_path,
            storage_dir=storage_dir,
        )
        controller_id = watched_seat.controller_id or human_controller_id
        return controller_id, watched_seat.profile_key, client.table_snapshot(table_id)

    if human_seat in occupied_seats:
        return human_controller_id, None, client.table_snapshot(table_id)

    response = client.join_human(
        table_id,
        human_seat,
        player_id=human_player_id,
        controller_id=human_controller_id,
        display_name=human_display_name,
    )
    controller_id = str(response.get("controller_id", human_controller_id))
    return controller_id, None, response.get("snapshot") or client.table_snapshot(table_id)


def _add_remaining_npc_players(
    bot_broker: NpcBroker,
    human_seat: str,
    occupied_seats: dict[object, object],
    *,
    lineup: str,
    storage_dir: Path,
    watched_profile_key: str | None,
) -> None:
    npc_seats = [seat.value for seat in SEATS if seat.value != human_seat and seat.value not in occupied_seats]
    bot_broker.add_players(
        npc_seats,
        lineup=lineup,
        storage_dir=storage_dir,
        exclude_profile_keys={watched_profile_key} if watched_profile_key else None,
    )


def _add_human_role(
    table: Table,
    seat: str,
    display_name: str,
    controller_id: str,
    player_mode: str,
) -> None:
    member = SeatMember(
        role=SeatRole.WITNESS if player_mode == "llm" else SeatRole.PLAYER,
        display_name=display_name,
        controller_id=controller_id,
        is_human=True,
    )
    seat_members = table.members_for(seat)
    if player_mode == "llm":
        seat_members.witnesses.append(member)
    else:
        seat_members.player = member


def _add_broker_player_roles(table: Table, bot_broker: NpcBroker) -> None:
    for broker_seat in bot_broker.seats.values():
        table.members_for(broker_seat.seat).player = SeatMember(
            role=SeatRole.PLAYER,
            display_name=broker_seat.display_name,
            policy=broker_seat.policy,
            controller_id=broker_seat.controller_id,
            profile_key=broker_seat.profile_key,
        )


def _add_gossiper_role(
    table: Table,
    seat: str,
    *,
    gossiper_mode: str,
    config_path: str | None,
    storage_dir: Path,
) -> None:
    if gossiper_mode != "llm":
        return
    profile = _llm_gossiper_profile_for_seat(seat, config_path=config_path, storage_dir=storage_dir)
    table.members_for(seat).gossiper = SeatMember(
        role=SeatRole.GOSSIPER,
        display_name=profile.display_name,
        policy=player_for_profile(profile, "mixed", storage_dir),
        profile_key=profile.profile_key,
    )


def _ready_players(
    client: GuandanHttpClient,
    bot_broker: NpcBroker,
    table_id: str,
    human_seat: str,
    human_controller_id: str,
    player_mode: str,
) -> str:
    if player_mode == "human":
        client.ready(table_id, human_seat, human_controller_id)
    bot_broker.join_and_ready_all()
    if player_mode == "llm":
        return bot_broker.seats[human_seat].controller_id
    return human_controller_id


def _npc_metadata(bot_broker: NpcBroker) -> dict[str, str]:
    return {
        broker_seat.seat: format_npc_metadata(broker_seat.policy)
        for broker_seat in bot_broker.seats.values()
    }


def _ensure_match_started(client: GuandanHttpClient, table_id: str) -> JsonObject:
    public_snapshot = client.table_snapshot(table_id)
    if public_snapshot.get("phase") in STARTED_PHASES:
        return public_snapshot
    response = client.start(table_id)
    return response.get("snapshot") or client.table_snapshot(table_id)


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
    broker_seat = bot_broker.add_seat(
        seat,
        player_for_profile(profile, "mixed", storage_dir),
        profile.display_name,
        profile_seat=profile.preferred_seat,
        profile_key=profile.profile_key or profile.display_name,
    )
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
    profiles = load_player_database(config_path or storage_dir).profiles
    base = next(
        (profile for profile in profiles if profile.display_name == display_name or profile.preferred_seat == seat),
        PlayerProfile(display_name, "llm", profile_key=display_name, preferred_seat=seat),
    )
    return replace(
        base,
        preferred_seat=seat,
        display_name=display_name,
        kind="llm",
        profile_key=base.profile_key or display_name,
    )


def _llm_gossiper_profile_for_seat(
    seat: str,
    *,
    config_path: str | None,
    storage_dir: Path,
) -> PlayerProfile:
    profiles = load_player_database(config_path or storage_dir).profiles
    base = next((profile for profile in profiles if profile.kind == "llm"), None)
    if base is None:
        return PlayerProfile(f"LLM Gossiper {seat}", "llm", profile_key=f"gossiper-{seat}", preferred_seat=seat)
    return replace(
        base,
        preferred_seat=seat,
        display_name=f"{base.display_name} Gossiper",
        kind="llm",
        profile_key=f"{base.profile_key or base.display_name}-gossiper-{seat}",
    )


def _npc_storage_dir(config_path: str | None) -> Path:
    if config_path:
        path = Path(config_path).expanduser()
        return path.parent if path.suffix == ".json" else path
    return Path("data")


def _snapshot_seats(snapshot: JsonObject) -> dict[object, object]:
    seats = snapshot.get("seats")
    return seats if isinstance(seats, dict) else {}


def _choose_available_seat(seats: object) -> str:
    occupied = set(str(seat) for seat in seats) if isinstance(seats, dict) else set()
    available = [seat.value for seat in SEATS if seat.value not in occupied]
    if not available:
        raise ValueError("no available seats")
    return SystemRandom().choice(available)
