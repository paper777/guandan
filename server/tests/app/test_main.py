from __future__ import annotations

import asyncio
import json
import unittest

from guandan.app.main import TABLES, app


async def call_app(method: str, path: str) -> tuple[int, dict]:
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await app({"type": "http", "method": method, "path": path}, receive, send)
    status = next(message["status"] for message in messages if message["type"] == "http.response.start")
    body = next(message["body"] for message in messages if message["type"] == "http.response.body")
    return status, json.loads(body.decode())


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


if __name__ == "__main__":
    unittest.main()
