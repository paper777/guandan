from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from client.api import ActionRequest, GuandanNpcClient, JsonObject, NpcPolicy
from npc.dummy_bot.policy import DummyBotPolicy


SEATS = ("E", "S", "W", "N")


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


class NpcBroker:
    def __init__(self, client: GuandanNpcClient, table_id: str) -> None:
        self.client = client
        self.table_id = table_id
        self.seats: dict[str, BrokerSeat] = {}

    def add_seat(self, seat: str, policy: NpcPolicy, display_name: str | None = None) -> BrokerSeat:
        broker_seat = BrokerSeat(seat=seat, policy=policy, display_name=display_name or f"NPC {seat}")
        self.seats[seat] = broker_seat
        return broker_seat

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
            return self.client.play_cards(self.table_id, broker_seat.seat, broker_seat.controller_id, card_ids)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NPCs against a Guandan table.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--table-id", required=True)
    parser.add_argument("--seats", default="S,W,N", help="Comma-separated seats to control.")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--start", action="store_true", help="Try to start the match after joining and readying NPCs.")
    args = parser.parse_args()

    broker = NpcBroker(GuandanNpcClient(args.server_url), args.table_id)
    for seat in _parse_seats(args.seats):
        broker.add_seat(seat, DummyBotPolicy(), display_name=f"Dummy {seat}")
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


if __name__ == "__main__":
    main()
