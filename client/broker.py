from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from random import Random, SystemRandom

from client.http_client import GuandanClientError, GuandanHttpClient
from client.types import ActionRequest, JsonObject
from common.log import debug_event, deadline_fields, deadline_remaining_ms, elapsed_ms, error_event, trace_event
from db.player.types import Player
from db.player import (
    NPC_LINEUPS,
    PLAYER_DATABASE_PATH,
    PlayerDatabase,
    assigned_profile,
    deal_score_delta,
    load_player_database,
    player_for_profile,
    profile_assignments,
    record_profile_result,
    team_for_seat,
)


SEATS = ("E", "S", "W", "N")


@dataclass(slots=True)
class BrokerSeat:
    seat: str
    policy: Player
    display_name: str
    profile_seat: str = ""
    profile_key: str = ""
    player_id: str = ""
    controller_id: str = ""


@dataclass(frozen=True, slots=True)
class BrokerActionResult:
    action: JsonObject
    response: JsonObject


class NpcBroker:
    def __init__(
        self,
        client: GuandanHttpClient,
        table_id: str,
        *,
        player_db_path: str | Path | None = None,
        storage_dir: str | Path = Path("data"),
    ) -> None:
        self.client = client
        self.table_id = table_id
        self.seats: dict[str, BrokerSeat] = {}
        self.player_database: PlayerDatabase = load_player_database(player_db_path or storage_dir)
        self._recorded_result_event_seqs: set[tuple[str, int]] = set()
        self.last_seat_rotation: dict[str, str] = {}

    def add_seat(
        self,
        seat: str,
        policy: Player,
        display_name: str | None = None,
        *,
        profile_seat: str | None = None,
        profile_key: str | None = None,
    ) -> BrokerSeat:
        broker_seat = BrokerSeat(
            seat=seat,
            policy=policy,
            display_name=display_name or f"NPC {seat}",
            profile_seat=profile_seat or seat,
            profile_key=profile_key or display_name or seat,
        )
        self.seats[seat] = broker_seat
        return broker_seat

    def add_players(
        self,
        seats: tuple[str, ...] | list[str] | None = None,
        *,
        lineup: str = "rl",
        storage_dir: str | Path = Path("data"),
        config_path: str | Path | None = None,
        shuffle_seed: object = None,
        exclude_profile_keys: set[str] | frozenset[str] | None = None,
    ) -> list[BrokerSeat]:
        selected = tuple(seats or SEATS)
        if config_path is not None:
            self.player_database = load_player_database(config_path)
        assignments = profile_assignments(
            self.player_database.profiles,
            selected,
            shuffle_seed=shuffle_seed,
            exclude_profile_keys=exclude_profile_keys,
        )
        broker_seats: list[BrokerSeat] = []
        for profile, seat in assignments:
            runtime_profile = assigned_profile(profile, seat)
            broker_seats.append(
                self.add_seat(
                    seat,
                    player_for_profile(runtime_profile, lineup, storage_dir),
                    profile.display_name,
                    profile_seat=profile.preferred_seat,
                    profile_key=profile.profile_key,
                )
            )
        return broker_seats

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
            acting_seat = public.get("acting_seat") or public.get("current_turn") if isinstance(public, dict) else None
            if legal_action is None or acting_seat != broker_seat.seat:
                continue
            players_by_seat = self._players_by_seat(public if isinstance(public, dict) else None)
            request_id = f"{self.table_id}:{broker_seat.seat}:{public.get('event_seq', 0)}"
            deadline_epoch_ms = public.get("action_deadline_epoch_ms")
            trace_event(
                "broker.action_requested",
                table_id=self.table_id,
                seat=broker_seat.seat,
                display_name=broker_seat.display_name,
                profile_key=broker_seat.profile_key,
                policy=type(broker_seat.policy).__name__,
                request_id=request_id,
                legal_action=legal_action,
                event_seq=public.get("event_seq"),
                current_turn=public.get("current_turn"),
                acting_seat=acting_seat,
                **deadline_fields(deadline_epoch_ms),
            )

            request = ActionRequest(
                request_id=request_id,
                prompt={
                    "kind": legal_action,
                    "current_level": public.get("current_level", "2"),
                    "current_trick": public.get("current_trick"),
                    "eligible_card_ids": list(snapshot.get("eligible_card_ids", [])),
                    "tribute_from": snapshot.get("tribute_from"),
                    "tribute_to": snapshot.get("tribute_to"),
                    "return_rank_at_most_ten": bool(snapshot.get("return_rank_at_most_ten", False)),
                },
                snapshot={
                    "table_id": self.table_id,
                    "seat": broker_seat.seat,
                    "hand": list(snapshot.get("hand", [])),
                    "players_by_seat": players_by_seat,
                    "public": public,
                },
            )
            started = time.perf_counter()
            try:
                action = broker_seat.policy.choose_action(request)
            except Exception as exc:
                error_event(
                    "broker.action_failed",
                    table_id=self.table_id,
                    seat=broker_seat.seat,
                    request_id=request_id,
                    duration_ms=elapsed_ms(started),
                    deadline_remaining_ms=deadline_remaining_ms(deadline_epoch_ms),
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
                raise
            debug_event(
                "broker.action_selected",
                table_id=self.table_id,
                seat=broker_seat.seat,
                request_id=request_id,
                action=action,
                duration_ms=elapsed_ms(started),
                deadline_remaining_ms=deadline_remaining_ms(deadline_epoch_ms),
            )
            response = self._submit_action(broker_seat, action)
            self._notify_action_observers(broker_seat, action, response)
            results.append(BrokerActionResult(action=action, response=response))
        return results

    def run_forever(self, *, interval_seconds: float = 0.5) -> None:
        while True:
            self.poll_once()
            time.sleep(interval_seconds)

    def rotate_seats_after_match(self, *, shuffle_seed: object = None) -> dict[str, str]:
        seats = list(self.seats)
        if len(seats) < 2:
            self.last_seat_rotation = {seat: seat for seat in seats}
            return dict(self.last_seat_rotation)

        shuffled = list(seats)
        rng = SystemRandom() if shuffle_seed is None else Random(shuffle_seed)
        for _ in range(8):
            rng.shuffle(shuffled)
            if shuffled != seats:
                break
        if shuffled == seats:
            shuffled = [*seats[1:], seats[0]]

        seat_map = dict(zip(seats, shuffled, strict=True))
        rotated: dict[str, BrokerSeat] = {}
        for old_seat, broker_seat in list(self.seats.items()):
            new_seat = seat_map[old_seat]
            broker_seat.seat = new_seat
            broker_seat.player_id = ""
            broker_seat.controller_id = ""
            rotated[new_seat] = broker_seat
        self.seats = rotated
        self.last_seat_rotation = dict(seat_map)
        return dict(seat_map)

    def _submit_action(self, broker_seat: BrokerSeat, action: JsonObject) -> JsonObject:
        started = time.perf_counter()
        try:
            response = self._submit_action_unchecked(broker_seat, action)
        except GuandanClientError as exc:
            _attach_rejected_action(exc, action)
            error_event(
                "broker.action_rejected",
                table_id=self.table_id,
                seat=broker_seat.seat,
                action=action,
                status=exc.status,
                duration_ms=elapsed_ms(started),
                error_payload=exc.payload,
                latest_snapshot=self._latest_public_snapshot(),
            )
            raise
        debug_event(
            "broker.action_accepted",
            table_id=self.table_id,
            seat=broker_seat.seat,
            action=action,
            duration_ms=elapsed_ms(started),
            event_seq=response.get("event_seq"),
        )
        return response

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
        event_seq = response.get("event_seq")
        response_snapshot = response.get("snapshot")
        snapshot = response_snapshot if isinstance(response_snapshot, dict) else None
        self._record_result_events(event_list, snapshot)
        players_by_seat = self._players_by_seat(snapshot)
        deal_id = snapshot.get("deal_id") if isinstance(snapshot, dict) else None
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
                    "deal_id": deal_id,
                    "action": action,
                    "events": event_list,
                    "event_seq": event_seq,
                }
            )

    def _record_result_events(self, events: list[object], snapshot: JsonObject | None = None) -> None:
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
            score_delta = deal_score_delta(winning_team, snapshot, payload) if event_type == "DealEnded" else 0
            for broker_seat in self.seats.values():
                profile = self.player_database.profile_for_key(broker_seat.profile_key)
                if profile is None:
                    continue
                won = team_for_seat(broker_seat.seat) == winning_team
                updated = record_profile_result(
                    profile,
                    kind="deal" if event_type == "DealEnded" else "match",
                    won=won,
                    score_delta=score_delta if won else -score_delta,
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

    def _latest_public_snapshot(self) -> JsonObject | None:
        try:
            snapshot = self.client.table_snapshot(self.table_id)
        except Exception:
            return None
        if not isinstance(snapshot, dict):
            return None
        return {
            "phase": snapshot.get("phase"),
            "event_seq": snapshot.get("event_seq"),
            "deal_id": snapshot.get("deal_id"),
            "current_turn": snapshot.get("current_turn"),
            "acting_seat": snapshot.get("acting_seat"),
            "current_trick": snapshot.get("current_trick"),
            "hand_counts": snapshot.get("hand_counts"),
            "action_deadline_epoch_ms": snapshot.get("action_deadline_epoch_ms"),
            "action_timeout_seconds": snapshot.get("action_timeout_seconds"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NPCs against a Guandan table.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--table-id", required=True)
    parser.add_argument("--seats", default="E,S,W,N", help="Comma-separated seats to control.")
    parser.add_argument("--lineup", choices=NPC_LINEUPS, default="rl", help="Default NPC player lineup.")
    parser.add_argument(
        "--player-config",
        default=str(PLAYER_DATABASE_PATH),
        help="Player storage directory for NPC profiles, stats, memory, and actions.",
    )
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--start", action="store_true", help="Try to start the match after joining and readying NPCs.")
    args = parser.parse_args()

    broker = NpcBroker(GuandanHttpClient(args.server_url), args.table_id, player_db_path=args.player_config)
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


if __name__ == "__main__":
    main()
