from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from client.api import ActionRequest, GuandanNpcClient, JsonObject, NpcPolicy
from npc.dummy_bot.policy import DummyBotPolicy
from npc.llm_agent import LlmAgentConfig, LlmAgentPlayer


SEATS = ("E", "S", "W", "N")
NPC_LINEUPS = ("mixed", "dummy", "llm")


@dataclass(slots=True)
class BrokerSeat:
    seat: str
    policy: NpcPolicy
    display_name: str
    player_id: str = ""
    controller_id: str = ""


@dataclass(frozen=True, slots=True)
class BrokerActionResult:
    action: JsonObject
    response: JsonObject


@dataclass(frozen=True, slots=True)
class DefaultPlayerProfile:
    seat: str
    display_name: str
    kind: str
    personality: str = "balanced"
    provider_name: str | None = None
    model_name: str | None = None
    api_base_url: str | None = None
    timeout_seconds: float | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    codex_binary: str | None = None
    codex_working_dir: str | Path | None = None


DEFAULT_PLAYER_PROFILES = (
    DefaultPlayerProfile("E", "Ming", "dummy", "balanced"),
    DefaultPlayerProfile("S", "Jade", "llm", "aggressive"),
    DefaultPlayerProfile("W", "River", "llm", "balanced"),
    DefaultPlayerProfile("N", "Atlas", "llm", "defensive"),
)
DEFAULT_PLAYER_CONFIG_PATH = Path("data/default_players.json")


class NpcBroker:
    def __init__(self, client: GuandanNpcClient, table_id: str) -> None:
        self.client = client
        self.table_id = table_id
        self.seats: dict[str, BrokerSeat] = {}

    def add_seat(self, seat: str, policy: NpcPolicy, display_name: str | None = None) -> BrokerSeat:
        broker_seat = BrokerSeat(seat=seat, policy=policy, display_name=display_name or f"NPC {seat}")
        self.seats[seat] = broker_seat
        return broker_seat

    def add_default_players(
        self,
        seats: tuple[str, ...] | list[str] | None = None,
        *,
        lineup: str = "mixed",
        storage_dir: str | Path = Path("data"),
        config_path: str | Path | None = None,
    ) -> list[BrokerSeat]:
        selected = tuple(seats or SEATS)
        profiles = load_default_player_profiles(config_path or Path(storage_dir) / "default_players.json")
        return [
            self.add_seat(profile.seat, _player_for_profile(profile, lineup, storage_dir), profile.display_name)
            for profile in profiles
            if profile.seat in selected
        ]

    def join_and_ready_all(self) -> None:
        for broker_seat in self.seats.values():
            response = self.client.join_agent(self.table_id, broker_seat.seat, broker_seat.display_name)
            broker_seat.player_id = str(response.get("player_id", ""))
            broker_seat.controller_id = str(response.get("controller_id", ""))
            self.client.ready(self.table_id, broker_seat.seat, broker_seat.controller_id)

    def poll_once(self, seat: str | None = None) -> list[JsonObject]:
        return [result.action for result in self.poll_once_results(seat)]

    def poll_once_results(self, seat: str | None = None) -> list[BrokerActionResult]:
        results: list[BrokerActionResult] = []
        for broker_seat in self.seats.values():
            if seat is not None and broker_seat.seat != seat:
                continue
            if broker_seat.controller_id == "":
                continue
            snapshot = self.client.seat_snapshot(self.table_id, broker_seat.seat, broker_seat.controller_id)
            public = snapshot.get("public", {})
            legal_action = snapshot.get("legal_action")
            acting_seat = public.get("acting_seat") or public.get("current_turn")
            if legal_action is None or acting_seat != broker_seat.seat:
                continue

            action = broker_seat.policy.choose_action(
                ActionRequest(
                    request_id=f"{self.table_id}:{broker_seat.seat}:{public.get('event_seq', 0)}",
                    prompt={
                        "kind": legal_action,
                        "current_level": public.get("current_level", "2"),
                        "current_trick": public.get("current_trick"),
                    },
                    snapshot={
                        "table_id": self.table_id,
                        "seat": broker_seat.seat,
                        "hand": list(snapshot.get("hand", [])),
                        "public": public,
                    },
                )
            )
            response = self._submit_action(broker_seat, action)
            self._notify_action_observers(broker_seat, action, response)
            results.append(BrokerActionResult(action=action, response=response))
        return results

    def run_forever(self, *, interval_seconds: float = 0.5) -> None:
        while True:
            self.poll_once()
            time.sleep(interval_seconds)

    def _submit_action(self, broker_seat: BrokerSeat, action: JsonObject) -> JsonObject:
        action_type = action.get("type")
        if action_type == "pass":
            return self.client.pass_turn(self.table_id, broker_seat.seat, broker_seat.controller_id)
        if action_type == "play_cards":
            card_ids = tuple(str(card_id) for card_id in action.get("card_ids", []))
            declared_type = action.get("declared_type")
            if declared_type is None:
                return self.client.play_cards(self.table_id, broker_seat.seat, broker_seat.controller_id, card_ids)
            return self.client.play_cards(
                self.table_id,
                broker_seat.seat,
                broker_seat.controller_id,
                card_ids,
                declared_type=str(declared_type),
            )
        if action_type == "submit_tribute":
            return self.client.submit_tribute(
                self.table_id,
                broker_seat.seat,
                broker_seat.controller_id,
                str(action["card_id"]),
            )
        if action_type == "return_tribute":
            return self.client.return_tribute(
                self.table_id,
                broker_seat.seat,
                broker_seat.controller_id,
                str(action["card_id"]),
            )
        raise ValueError(f"unsupported NPC action: {action_type}")

    def _notify_action_observers(self, actor: BrokerSeat, action: JsonObject, response: JsonObject) -> None:
        events = response.get("events", [])
        event_seq = response.get("event_seq")
        for observer in self.seats.values():
            observe_action = getattr(observer.policy, "observe_action", None)
            if observe_action is None:
                continue
            observe_action(
                {
                    "table_id": self.table_id,
                    "observer_seat": observer.seat,
                    "actor_seat": actor.seat,
                    "action": action,
                    "events": events if isinstance(events, list) else [],
                    "event_seq": event_seq,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NPCs against a Guandan table.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--table-id", required=True)
    parser.add_argument("--seats", default="S,W,N", help="Comma-separated seats to control.")
    parser.add_argument("--lineup", choices=NPC_LINEUPS, default="mixed", help="Default NPC player lineup.")
    parser.add_argument(
        "--player-config",
        default=str(DEFAULT_PLAYER_CONFIG_PATH),
        help="JSON file for default NPC player names and kinds.",
    )
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--start", action="store_true", help="Try to start the match after joining and readying NPCs.")
    args = parser.parse_args()

    broker = NpcBroker(GuandanNpcClient(args.server_url), args.table_id)
    broker.add_default_players(_parse_seats(args.seats), lineup=args.lineup, config_path=args.player_config)
    broker.join_and_ready_all()
    if args.start:
        broker.client.start(args.table_id)
    broker.run_forever(interval_seconds=args.interval)


def _parse_seats(raw: str) -> tuple[str, ...]:
    seats = tuple(item.strip().upper() for item in raw.split(",") if item.strip())
    invalid = [seat for seat in seats if seat not in SEATS]
    if invalid:
        raise ValueError(f"invalid seat(s): {', '.join(invalid)}")
    return seats


def _player_for_profile(profile: DefaultPlayerProfile, lineup: str, storage_dir: str | Path) -> NpcPolicy:
    if lineup not in NPC_LINEUPS:
        raise ValueError(f"unsupported NPC lineup: {lineup}")
    kind = profile.kind if lineup == "mixed" else lineup
    if kind == "dummy":
        return DummyBotPolicy()
    provider_name = profile.provider_name or "deterministic"
    return LlmAgentPlayer(
        LlmAgentConfig(
            player_name=profile.display_name,
            seat=profile.seat,
            personality=profile.personality,
            storage_dir=storage_dir,
            provider_name=provider_name,
            model_name=profile.model_name or _default_model_for_provider(provider_name),
            api_base_url=profile.api_base_url,
            timeout_seconds=profile.timeout_seconds or _default_timeout_for_provider(provider_name),
            temperature=profile.temperature if profile.temperature is not None else 0.2,
            max_output_tokens=profile.max_output_tokens or 800,
            codex_binary=profile.codex_binary or "codex",
            codex_working_dir=profile.codex_working_dir,
        )
    )


def load_default_player_profiles(path: str | Path = DEFAULT_PLAYER_CONFIG_PATH) -> tuple[DefaultPlayerProfile, ...]:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return DEFAULT_PLAYER_PROFILES
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid default player config JSON: {config_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"default player config must be a JSON object: {config_path}")
    players = raw.get("players")
    if not isinstance(players, list):
        raise ValueError(f"default player config must contain a players list: {config_path}")

    by_seat = {profile.seat: profile for profile in DEFAULT_PLAYER_PROFILES}
    for item in players:
        if not isinstance(item, dict):
            raise ValueError("default player config entries must be objects")
        seat = str(item.get("seat", "")).upper()
        if seat not in SEATS:
            raise ValueError(f"invalid default player seat: {seat}")
        display_name = str(item.get("display_name") or item.get("name") or by_seat[seat].display_name)
        kind = str(item.get("kind") or by_seat[seat].kind).lower()
        if kind not in {"dummy", "llm"}:
            raise ValueError(f"invalid default player kind for {seat}: {kind}")
        by_seat[seat] = DefaultPlayerProfile(
            seat=seat,
            display_name=display_name,
            kind=kind,
            personality=str(item.get("personality") or by_seat[seat].personality),
            provider_name=_optional_str(item.get("provider_name")),
            model_name=_optional_str(item.get("model_name")),
            api_base_url=_optional_str(item.get("api_base_url")),
            timeout_seconds=_optional_float(item.get("timeout_seconds")),
            temperature=_optional_float(item.get("temperature")),
            max_output_tokens=_optional_int(item.get("max_output_tokens")),
            codex_binary=_optional_str(item.get("codex_binary")),
            codex_working_dir=_optional_str(item.get("codex_working_dir")),
        )
    return tuple(by_seat[seat] for seat in SEATS)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _default_model_for_provider(provider_name: str) -> str:
    if provider_name in {"codex-cli", "codex_signed_in", "codex-signed-in"}:
        return "gpt-5.2"
    return "deterministic-guandan-v1"


def _default_timeout_for_provider(provider_name: str) -> float:
    if provider_name in {"codex-cli", "codex_signed_in", "codex-signed-in"}:
        return 120.0
    return 3.0


if __name__ == "__main__":
    main()
