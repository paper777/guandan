from __future__ import annotations

import asyncio
import unittest
from urllib.parse import urlsplit

from server.app.main import app


async def call_raw(path: str) -> tuple[int, dict[str, str], bytes]:
    messages = []
    parsed = urlsplit(path)

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": parsed.path,
            "raw_path": parsed.path.encode(),
            "query_string": parsed.query.encode(),
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        },
        receive,
        send,
    )
    start = next(message for message in messages if message["type"] == "http.response.start")
    bodies = [message.get("body", b"") for message in messages if message["type"] == "http.response.body"]
    headers = {key.decode().lower(): value.decode() for key, value in start.get("headers", [])}
    return int(start["status"]), headers, b"".join(bodies)


class WebUiTests(unittest.TestCase):
    def test_root_redirects_to_ui(self) -> None:
        status, headers, body = asyncio.run(call_raw("/"))

        self.assertEqual(status, 307)
        self.assertEqual(headers["location"], "/ui/")
        self.assertEqual(body, b"")

    def test_ui_index_served(self) -> None:
        status, headers, body = asyncio.run(call_raw("/ui/"))

        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["content-type"])
        self.assertIn(b"Guandan Showdown", body)
        self.assertIn(b"control-panel", body)
        self.assertIn(b"bottom-strip control-panel", body)
        self.assertIn(b"score-panel", body)
        self.assertNotIn(b"topbar", body)
        self.assertNotIn(b"panel-section control-panel", body)

    def test_ui_javascript_served(self) -> None:
        status, headers, body = asyncio.run(call_raw("/ui/app.js"))

        self.assertEqual(status, 200)
        self.assertIn("javascript", headers["content-type"])
        self.assertIn(b"quickStart", body)
        self.assertIn(b"human-seat-row", body)
        self.assertIn(b"--played-card-count", body)

    def test_ui_styles_make_cards_and_player_info_readable(self) -> None:
        status, headers, body = asyncio.run(call_raw("/ui/styles.css"))

        self.assertEqual(status, 200)
        self.assertIn("text/css", headers["content-type"])
        self.assertIn(b"--human-card-width: 96px", body)
        self.assertIn(b"grid-template-columns: repeat(var(--card-count, 1)", body)
        self.assertIn(b".human-seat-row", body)
        self.assertIn(b".compact-hand .card-shell", body)
        self.assertIn(b"--played-card-width: 84px", body)
        self.assertIn(b"width: min(520px, 100%)", body)
        self.assertIn(b"overflow-wrap: anywhere", body)
        self.assertNotIn(b"--played-card-width: 72px", body)
        self.assertNotIn(b"--human-card-gap", body)

    def test_table_actions_do_not_render_old_selection_controls(self) -> None:
        status, _, body = asyncio.run(call_raw("/ui/app.js"))

        self.assertEqual(status, 200)
        self.assertIn(b"renderSeatPlayed", body)
        self.assertIn(b"table-action-dock", body)
        self.assertNotIn(b"declaredType", body)
        self.assertNotIn(b"selected-chip", body)

    def test_visual_seat_mapping_is_counter_clockwise(self) -> None:
        status, _, body = asyncio.run(call_raw("/ui/app.js"))

        self.assertEqual(status, 200)
        self.assertIn(b'return ["bottom", "right", "top", "left"][relative];', body)

    def test_unknown_ui_asset_returns_404(self) -> None:
        status, _, _ = asyncio.run(call_raw("/ui/missing.js"))

        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
