from __future__ import annotations

import asyncio
import json
import unittest
from urllib.parse import urlsplit

from guandan.app.main import TABLES, app


async def call_app(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    messages = []
    request_body = json.dumps(body or {}).encode()
    parsed = urlsplit(path)

    async def receive():
        return {"type": "http.request", "body": request_body, "more_body": False}

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": parsed.path,
            "raw_path": parsed.path.encode(),
            "query_string": parsed.query.encode(),
            "headers": [(b"content-type", b"application/json")],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        },
        receive,
        send,
    )
    status = next(message["status"] for message in messages if message["type"] == "http.response.start")
    body = next(message["body"] for message in messages if message["type"] == "http.response.body")
    return status, json.loads(body.decode())


async def call_ws(path: str, inbound: list[dict] | None = None) -> list[dict]:
    messages = []
    inbound_messages = [{"type": "websocket.connect"}, *(inbound or []), {"type": "websocket.disconnect", "code": 1000}]

    async def receive():
        return inbound_messages.pop(0)

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "scheme": "ws",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "subprotocols": [],
        },
        receive,
        send,
    )
    return messages


class AppTests(unittest.TestCase):
    def setUp(self) -> None:
        TABLES.clear()

    def test_health(self) -> None:
        status, body = asyncio.run(call_app("GET", "/health"))

        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

    def test_create_and_get_table_snapshot(self) -> None:
        status, body = asyncio.run(call_app("POST", "/tables"))
        self.assertEqual(status, 201)
        table_id = body["table_id"]

        status, body = asyncio.run(call_app("GET", f"/tables/{table_id}"))

        self.assertEqual(status, 200)
        self.assertEqual(body["table_id"], table_id)

    def test_join_ready_and_start_table_via_http(self) -> None:
        status, body = asyncio.run(call_app("POST", "/tables"))
        self.assertEqual(status, 201)
        table_id = body["table_id"]

        for seat in ("E", "S", "W", "N"):
            status, body = asyncio.run(
                call_app(
                    "POST",
                    f"/tables/{table_id}/join-human",
                    {"seat": seat, "player_id": f"p-{seat}", "controller_id": f"c-{seat}"},
                )
            )
            self.assertEqual(status, 200)
            self.assertEqual(body["controller_id"], f"c-{seat}")

        for seat in ("E", "S", "W", "N"):
            status, _ = asyncio.run(
                call_app("POST", f"/tables/{table_id}/ready", {"seat": seat, "controller_id": f"c-{seat}"})
            )
            self.assertEqual(status, 200)

        status, body = asyncio.run(call_app("POST", f"/tables/{table_id}/start", {"seed": "fixed-seed"}))

        self.assertEqual(status, 200)
        self.assertEqual([event["type"] for event in body["events"]], ["MatchStarted", "DealStarted", "CardsDealt"])
        self.assertEqual(body["snapshot"]["phase"], "PLAYING")

    def test_private_seat_snapshot_requires_attached_controller(self) -> None:
        status, body = asyncio.run(call_app("POST", "/tables"))
        self.assertEqual(status, 201)
        table_id = body["table_id"]

        for seat in ("E", "S", "W", "N"):
            status, _ = asyncio.run(
                call_app(
                    "POST",
                    f"/tables/{table_id}/join-human",
                    {"seat": seat, "player_id": f"p-{seat}", "controller_id": f"c-{seat}"},
                )
            )
            self.assertEqual(status, 200)
            status, _ = asyncio.run(
                call_app("POST", f"/tables/{table_id}/ready", {"seat": seat, "controller_id": f"c-{seat}"})
            )
            self.assertEqual(status, 200)
        status, _ = asyncio.run(call_app("POST", f"/tables/{table_id}/start", {"seed": "fixed-seed"}))
        self.assertEqual(status, 200)

        status, body = asyncio.run(call_app("GET", f"/tables/{table_id}/seats/E/snapshot?controller_id=c-E"))
        self.assertEqual(status, 200)
        self.assertEqual(body["seat"], "E")
        self.assertEqual(len(body["hand"]), 27)
        self.assertEqual(body["legal_action"], "lead")

        status, body = asyncio.run(call_app("GET", f"/tables/{table_id}/seats/E/snapshot?controller_id=c-S"))
        self.assertEqual(status, 400)

    def test_http_rejection_returns_code(self) -> None:
        status, body = asyncio.run(call_app("POST", "/tables"))
        table_id = body["table_id"]

        status, body = asyncio.run(
            call_app("POST", f"/tables/{table_id}/ready", {"seat": "E", "controller_id": "missing"})
        )

        self.assertEqual(status, 400)
        self.assertEqual(body["rejection"]["code"], "CONTROLLER_NOT_ATTACHED")

    def test_websocket_sends_initial_snapshot_and_handles_snapshot_request(self) -> None:
        status, body = asyncio.run(call_app("POST", "/tables"))
        table_id = body["table_id"]

        messages = asyncio.run(
            call_ws("/ws/tables/" + table_id, [{"type": "websocket.receive", "text": '{"type": "snapshot"}'}])
        )

        self.assertEqual(messages[0]["type"], "websocket.accept")
        initial = json.loads(messages[1]["text"])
        response = json.loads(messages[2]["text"])
        self.assertEqual(initial["type"], "snapshot")
        self.assertEqual(response["status"], 200)
        self.assertEqual(response["payload"]["snapshot"]["table_id"], table_id)

    def test_websocket_rejects_unknown_table(self) -> None:
        messages = asyncio.run(call_ws("/ws/tables/missing"))

        self.assertEqual(messages[0]["type"], "websocket.close")
        self.assertEqual(messages[0]["code"], 1008)


if __name__ == "__main__":
    unittest.main()
