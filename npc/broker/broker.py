from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from client.api import ActionRequest, GuandanClientError, GuandanNpcClient, JsonObject, NpcPolicy
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
class PlayerProfile:
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
    memory_compaction_char_limit: int | None = None
    memory_recent_deal_scan_limit: int | None = None
    memory_max_output_tokens: int | None = None
    codex_binary: str | None = None
    codex_working_dir: str | Path | None = None
    deal_count: int = 0
    deal_wins: int = 0
    deal_win_rate: float = 0.0
    match_count: int = 0
    match_wins: int = 0
    match_win_rate: float = 0.0
    extra: dict[str, object] = field(default_factory=dict)


PLAYER_PROFILES = (
    PlayerProfile("E", "Ming", "dummy", "balanced"),
    PlayerProfile("S", "Jade", "llm", "aggressive"),
    PlayerProfile("W", "River", "llm", "balanced"),
    PlayerProfile("N", "Atlas", "llm", "defensive"),
)
PLAYER_DATABASE_PATH = Path("data/players.json")


@dataclass(slots=True)
class PlayerDatabase:
    path: Path
    profiles: list[PlayerProfile]

    def profile_for_seat(self, seat: str) -> PlayerProfile | None:
        return next((profile for profile in self.profiles if profile.seat == seat), None)

    def replace_profile(self, updated: PlayerProfile) -> None:
        self.profiles = [updated if profile.seat == updated.seat else profile for profile in self.profiles]

    def save(self) -> None:
        payload = {"players": [_profile_to_json(profile) for profile in self.profiles]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


_PROFILE_KEYS = {
    "seat",
    "display_name",
    "name",
    "kind",
    "personality",
    "provider_name",
    "model_name",
    "api_base_url",
    "timeout_seconds",
    "temperature",
    "max_output_tokens",
    "memory_compaction_char_limit",
    "memory_recent_deal_scan_limit",
    "memory_max_output_tokens",
    "codex_binary",
    "codex_working_dir",
    "deal_count",
    "deal_wins",
    "deal_win_rate",
    "match_count",
    "match_wins",
    "match_win_rate",
}


class NpcBroker:
    def __init__(
        self,
        client: GuandanNpcClient,
        table_id: str,
        *,
        player_db_path: str | Path | None = None,
        storage_dir: str | Path = Path("data"),
    ) -> None:
        self.client = client
        self.table_id = table_id
        self.seats: dict[str, BrokerSeat] = {}
        self.player_database = load_player_database(player_db_path or Path(storage_dir) / PLAYER_DATABASE_PATH.name)
        self._recorded_result_event_seqs: set[tuple[str, int]] = set()

    def add_seat(self, seat: str, policy: NpcPolicy, display_name: str | None = None) -> BrokerSeat:
        broker_seat = BrokerSeat(seat=seat, policy=policy, display_name=display_name or f"NPC {seat}")
        self.seats[seat] = broker_seat
        return broker_seat

    def add_players(
        self,
        seats: tuple[str, ...] | list[str] | None = None,
        *,
        lineup: str = "mixed",
        storage_dir: str | Path = Path("data"),
        config_path: str | Path | None = None,
    ) -> list[BrokerSeat]:
        selected = tuple(seats or SEATS)
        if config_path is not None:
            self.player_database = load_player_database(config_path)
        return [
            self.add_seat(profile.seat, _player_for_profile(profile, lineup, storage_dir), profile.display_name)
            for profile in self.player_database.profiles
            if profile.seat in selected
        ]

    def join_and_ready_all(self) -> None:
        for broker_seat in self.seats.values():
            if broker_seat.controller_id == "":
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
            players_by_seat = self._players_by_seat(public)

            action = broker_seat.policy.choose_action(
                ActionRequest(
                    request_id=f"{self.table_id}:{broker_seat.seat}:{public.get('event_seq', 0)}",
                    prompt={
                        "kind": legal_action,
                        "current_level": public.get("current_level", "2"),
                        "current_trick": public.get("current_trick"),
                        "legal_card_ids": list(snapshot.get("legal_card_ids", [])),
                        "tribute_from": snapshot.get("tribute_from"),
                        "tribute_to": snapshot.get("tribute_to"),
                        "return_rank_at_most_ten": bool(snapshot.get("return_rank_at_most_ten", False)),
                    },
                    snapshot={
                        "table_id": self.table_id,
                        "seat": broker_seat.seat,
                        "hand": list(snapshot.get("hand", [])),
                        "legal_card_ids": list(snapshot.get("legal_card_ids", [])),
                        "players_by_seat": players_by_seat,
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
        try:
            return self._submit_action_unchecked(broker_seat, action)
        except GuandanClientError as exc:
            _attach_rejected_action(exc, action)
            raise

    def _submit_action_unchecked(self, broker_seat: BrokerSeat, action: JsonObject) -> JsonObject:
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
        event_list = events if isinstance(events, list) else []
        self._record_result_events(event_list)
        event_seq = response.get("event_seq")
        response_snapshot = response.get("snapshot")
        players_by_seat = self._players_by_seat(response_snapshot if isinstance(response_snapshot, dict) else None)
        for observer in self.seats.values():
            observe_action = getattr(observer.policy, "observe_action", None)
            if observe_action is None:
                continue
            observe_action(
                {
                    "table_id": self.table_id,
                    "observer_seat": observer.seat,
                    "observer_name": observer.display_name,
                    "actor_seat": actor.seat,
                    "actor_name": actor.display_name,
                    "players_by_seat": players_by_seat,
                    "action": action,
                    "events": event_list,
                    "event_seq": event_seq,
                }
            )

    def _record_result_events(self, events: list[object]) -> None:
        changed = False
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type") or "")
            if event_type not in {"DealEnded", "MatchEnded"}:
                continue
            seq = event.get("seq")
            if isinstance(seq, int):
                dedupe_key = (event_type, seq)
                if dedupe_key in self._recorded_result_event_seqs:
                    continue
                self._recorded_result_event_seqs.add(dedupe_key)
            payload = event.get("payload")
            winning_team = str(payload.get("winning_team") or "") if isinstance(payload, dict) else ""
            if winning_team not in {"EW", "SN"}:
                continue
            for seat in self.seats:
                profile = self.player_database.profile_for_seat(seat)
                if profile is None:
                    continue
                if event_type == "DealEnded":
                    updated = _record_profile_result(
                        profile,
                        count_field="deal_count",
                        wins_field="deal_wins",
                        rate_field="deal_win_rate",
                        won=_team_for_seat(seat) == winning_team,
                    )
                else:
                    updated = _record_profile_result(
                        profile,
                        count_field="match_count",
                        wins_field="match_wins",
                        rate_field="match_win_rate",
                        won=_team_for_seat(seat) == winning_team,
                    )
                self.player_database.replace_profile(updated)
                changed = True
        if changed:
            self.player_database.save()

    def _players_by_seat(self, public_snapshot: JsonObject | None = None) -> JsonObject:
        players: JsonObject = {}
        seats = public_snapshot.get("seats") if isinstance(public_snapshot, dict) else None
        if isinstance(seats, dict):
            for raw_seat, raw_player in seats.items():
                if not isinstance(raw_player, dict):
                    continue
                name = str(raw_player.get("display_name") or "").strip()
                if name:
                    players[str(raw_seat)] = name
        for seat, broker_seat in self.seats.items():
            players.setdefault(seat, broker_seat.display_name)
        return players


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NPCs against a Guandan table.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--table-id", required=True)
    parser.add_argument("--seats", default="S,W,N", help="Comma-separated seats to control.")
    parser.add_argument("--lineup", choices=NPC_LINEUPS, default="mixed", help="Default NPC player lineup.")
    parser.add_argument(
        "--player-config",
        default=str(PLAYER_DATABASE_PATH),
        help="JSON player database for NPC profiles and stats.",
    )
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--start", action="store_true", help="Try to start the match after joining and readying NPCs.")
    args = parser.parse_args()

    broker = NpcBroker(GuandanNpcClient(args.server_url), args.table_id, player_db_path=args.player_config)
    broker.add_players(_parse_seats(args.seats), lineup=args.lineup)
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


def _player_for_profile(profile: PlayerProfile, lineup: str, storage_dir: str | Path) -> NpcPolicy:
    return player_for_profile(profile, lineup, storage_dir)


def player_for_profile(profile: PlayerProfile, lineup: str, storage_dir: str | Path = Path("data")) -> NpcPolicy:
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
            memory_compaction_char_limit=(
                profile.memory_compaction_char_limit if profile.memory_compaction_char_limit is not None else 16000
            ),
            memory_recent_deal_scan_limit=(
                profile.memory_recent_deal_scan_limit if profile.memory_recent_deal_scan_limit is not None else 200
            ),
            memory_max_output_tokens=(
                profile.memory_max_output_tokens if profile.memory_max_output_tokens is not None else 1200
            ),
            codex_binary=profile.codex_binary or "codex",
            codex_working_dir=profile.codex_working_dir,
        )
    )


def _attach_rejected_action(error: GuandanClientError, action: JsonObject) -> None:
    if error.status != 400:
        return
    card_ids = _action_card_ids(action)
    if not card_ids:
        return
    error.payload["rejected_action"] = dict(action)
    error.payload["rejected_card_ids"] = list(card_ids)


def _action_card_ids(action: JsonObject) -> tuple[str, ...]:
    action_type = action.get("type")
    if action_type == "play_cards":
        return tuple(str(card_id) for card_id in action.get("card_ids", []))
    if action_type in {"submit_tribute", "return_tribute"}:
        card_id = action.get("card_id")
        return (str(card_id),) if card_id is not None else ()
    return ()


def _profile_to_json(profile: PlayerProfile) -> JsonObject:
    payload: JsonObject = dict(profile.extra)
    payload["seat"] = profile.seat
    payload["display_name"] = profile.display_name
    payload["kind"] = profile.kind
    payload["personality"] = profile.personality
    for key in (
        "provider_name",
        "model_name",
        "api_base_url",
        "timeout_seconds",
        "temperature",
        "max_output_tokens",
        "memory_compaction_char_limit",
        "memory_recent_deal_scan_limit",
        "memory_max_output_tokens",
        "codex_binary",
        "codex_working_dir",
    ):
        value = getattr(profile, key)
        if value is not None:
            payload[key] = str(value) if isinstance(value, Path) else value
    payload["deal_count"] = profile.deal_count
    payload["deal_wins"] = profile.deal_wins
    payload["deal_win_rate"] = profile.deal_win_rate
    payload["match_count"] = profile.match_count
    payload["match_wins"] = profile.match_wins
    payload["match_win_rate"] = profile.match_win_rate
    return payload


def _record_profile_result(
    profile: PlayerProfile,
    *,
    count_field: str,
    wins_field: str,
    rate_field: str,
    won: bool,
) -> PlayerProfile:
    count = int(getattr(profile, count_field)) + 1
    wins = int(getattr(profile, wins_field)) + (1 if won else 0)
    return replace(profile, **{count_field: count, wins_field: wins, rate_field: wins / count})


def _team_for_seat(seat: str) -> str:
    if seat in {"E", "W"}:
        return "EW"
    return "SN"


def load_player_profiles(path: str | Path = PLAYER_DATABASE_PATH) -> tuple[PlayerProfile, ...]:
    return tuple(load_player_database(path).profiles)


def load_player_database(path: str | Path = PLAYER_DATABASE_PATH) -> PlayerDatabase:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return PlayerDatabase(config_path, list(PLAYER_PROFILES))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid player database JSON: {config_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"player database must be a JSON object: {config_path}")
    players = raw.get("players")
    if not isinstance(players, list):
        raise ValueError(f"player database must contain a players list: {config_path}")

    fallback_by_seat = {profile.seat: profile for profile in PLAYER_PROFILES}
    profiles: list[PlayerProfile] = []
    seen: set[str] = set()
    for item in players:
        if not isinstance(item, dict):
            raise ValueError("player database entries must be objects")
        seat = str(item.get("seat", "")).upper()
        if seat not in SEATS:
            raise ValueError(f"invalid player seat: {seat}")
        if seat in seen:
            raise ValueError(f"duplicate player seat: {seat}")
        seen.add(seat)
        fallback = fallback_by_seat[seat]
        display_name = str(item.get("display_name") or item.get("name") or fallback.display_name)
        kind = str(item.get("kind") or fallback.kind).lower()
        if kind not in {"dummy", "llm"}:
            raise ValueError(f"invalid player kind for {seat}: {kind}")
        profiles.append(
            PlayerProfile(
                seat=seat,
                display_name=display_name,
                kind=kind,
                personality=str(item.get("personality") or fallback.personality),
                provider_name=_optional_str(item.get("provider_name")),
                model_name=_optional_str(item.get("model_name")),
                api_base_url=_optional_str(item.get("api_base_url")),
                timeout_seconds=_optional_float(item.get("timeout_seconds")),
                temperature=_optional_float(item.get("temperature")),
                max_output_tokens=_optional_int(item.get("max_output_tokens")),
                memory_compaction_char_limit=_optional_int(item.get("memory_compaction_char_limit")),
                memory_recent_deal_scan_limit=_optional_int(item.get("memory_recent_deal_scan_limit")),
                memory_max_output_tokens=_optional_int(item.get("memory_max_output_tokens")),
                codex_binary=_optional_str(item.get("codex_binary")),
                codex_working_dir=_optional_str(item.get("codex_working_dir")),
                deal_count=_stat_int(item.get("deal_count")),
                deal_wins=_stat_int(item.get("deal_wins")),
                deal_win_rate=_stat_float(item.get("deal_win_rate")),
                match_count=_stat_int(item.get("match_count")),
                match_wins=_stat_int(item.get("match_wins")),
                match_win_rate=_stat_float(item.get("match_win_rate")),
                extra={str(key): value for key, value in item.items() if key not in _PROFILE_KEYS},
            )
        )
    return PlayerDatabase(config_path, profiles)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stat_int(value: object) -> int:
    if value is None:
        return 0
    return int(value)


def _stat_float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


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
