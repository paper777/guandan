from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from server.app.main import TABLES, app


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
        self.assertEqual(body["action_timeout_seconds"], 180)
        self.assertEqual(body["timeout_fallback"], "auto_pass")

        status, body = asyncio.run(call_app("GET", f"/tables/{table_id}"))

        self.assertEqual(status, 200)
        self.assertEqual(body["table_id"], table_id)
        self.assertEqual(body["action_timeout_seconds"], 180)

    def test_create_table_accepts_custom_timeout(self) -> None:
        status, body = asyncio.run(call_app("POST", "/tables", {"action_timeout_seconds": 60}))

        self.assertEqual(status, 201)
        self.assertEqual(body["action_timeout_seconds"], 60)

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

        status, body = asyncio.run(call_app("POST", f"/tables/{table_id}/start"))

        self.assertEqual(status, 200)
        self.assertEqual(
            [event["type"] for event in body["events"]],
            ["MatchStarted", "DealStarted", "CardsDealt", "ActionPrompted"],
        )
        cards_dealt = body["events"][2]
        self.assertNotIn("hands", cards_dealt["payload"])
        self.assertEqual(cards_dealt["payload"]["hand_counts"], {"E": 27, "N": 27, "S": 27, "W": 27})
        self.assertEqual(body["snapshot"]["phase"], "PLAYING")
        self.assertEqual(body["snapshot"]["deal_id"], 1)
        self.assertEqual(body["snapshot"]["acting_seat"], "E")
        self.assertIsNotNone(body["snapshot"]["action_deadline_epoch_ms"])

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
        status, _ = asyncio.run(call_app("POST", f"/tables/{table_id}/start"))
        self.assertEqual(status, 200)

        status, body = asyncio.run(call_app("GET", f"/tables/{table_id}/seats/E/snapshot?controller_id=c-E"))
        self.assertEqual(status, 200)
        self.assertEqual(body["seat"], "E")
        self.assertEqual(len(body["hand"]), 27)
        self.assertEqual(body["legal_action"], "lead")
        self.assertEqual(body["eligible_card_ids"], [])
        self.assertEqual(body["public"]["deal_id"], 1)
        self.assertEqual(body["public"]["acting_seat"], "E")
        self.assertIsNotNone(body["public"]["action_deadline_epoch_ms"])

        status, body = asyncio.run(call_app("GET", f"/tables/{table_id}/seats/E/snapshot?controller_id=c-S"))
        self.assertEqual(status, 400)

    def test_audit_log_records_request_response_with_private_fields_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.jsonl"
            with patch.dict(
                "os.environ",
                {"GUANDAN_AUDIT_LOG_PATH": str(audit_path), "GUANDAN_AUDIT_LOG_ENABLED": "1"},
            ):
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
                status, _ = asyncio.run(call_app("POST", f"/tables/{table_id}/start"))
                self.assertEqual(status, 200)
                status, _ = asyncio.run(call_app("GET", f"/tables/{table_id}/seats/E/snapshot?controller_id=c-E"))
                self.assertEqual(status, 200)

            entries = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]

        snapshot_entry = entries[-1]
        self.assertEqual(snapshot_entry["request"]["method"], "GET")
        self.assertEqual(snapshot_entry["request"]["path"], f"/tables/{table_id}/seats/E/snapshot")
        self.assertEqual(snapshot_entry["request"]["query"], {"controller_id": "<redacted>"})
        self.assertEqual(snapshot_entry["response"]["status"], 200)
        self.assertEqual(snapshot_entry["response"]["body"]["hand"], "<redacted>")

    def test_start_rejects_client_seed_payload(self) -> None:
        status, body = asyncio.run(call_app("POST", "/tables"))
        table_id = body["table_id"]

        status, _ = asyncio.run(call_app("POST", f"/tables/{table_id}/start", {"seed": "fixed-seed"}))

        self.assertIn(status, {400, 422})

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
