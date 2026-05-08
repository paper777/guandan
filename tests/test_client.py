from __future__ import annotations

import unittest

from client.http_client import GuandanHttpClient


class ClientTests(unittest.TestCase):
    def test_create_table_sends_timeout_config(self) -> None:
        calls = []

        def transport(method, path, body, query):
            calls.append((method, path, body, query))
            return {"table_id": "table-1"}

        client = GuandanHttpClient(transport=transport)

        client.create_table(action_timeout_seconds=60, timeout_fallback="auto_pass")

        self.assertEqual(
            calls,
            [("POST", "/tables", {"action_timeout_seconds": 60, "timeout_fallback": "auto_pass"}, None)],
        )

    def test_join_human_sends_expected_request(self) -> None:
        calls = []

        def transport(method, path, body, query):
            calls.append((method, path, body, query))
            return {"controller_id": "c-E"}

        client = GuandanHttpClient(base_url="http://example.test", transport=transport)

        response = client.join_human(
            "table-1",
            "E",
            player_id="p-E",
            controller_id="c-E",
            display_name="East",
        )

        self.assertEqual(response["controller_id"], "c-E")
        self.assertEqual(
            calls,
            [
                (
                    "POST",
                    "/tables/table-1/join-human",
                    {"seat": "E", "player_id": "p-E", "controller_id": "c-E", "display_name": "East"},
                    None,
                )
            ],
        )

    def test_seat_snapshot_sends_controller_query(self) -> None:
        calls = []

        def transport(method, path, body, query):
            calls.append((method, path, body, query))
            return {"seat": "E", "hand": []}

        client = GuandanHttpClient(transport=transport)

        client.seat_snapshot("table-1", "E", "c-E")

        self.assertEqual(
            calls,
            [("GET", "/tables/table-1/seats/E/snapshot", None, {"controller_id": "c-E"})],
        )

    def test_join_agent_sends_expected_request(self) -> None:
        calls = []

        def transport(method, path, body, query):
            calls.append((method, path, body, query))
            return {"controller_id": "agent-S"}

        client = GuandanHttpClient(transport=transport)

        response = client.join_agent("table-1", "S", "Dummy S")

        self.assertEqual(response["controller_id"], "agent-S")
        self.assertEqual(
            calls,
            [
                (
                    "POST",
                    "/tables/table-1/join-agent",
                    {"seat": "S", "display_name": "Dummy S"},
                    None,
                )
            ],
        )

    def test_start_sends_no_seed_payload(self) -> None:
        calls = []

        def transport(method, path, body, query):
            calls.append((method, path, body, query))
            return {"event_seq": 1}

        client = GuandanHttpClient(transport=transport)

        client.start("table-1")

        self.assertEqual(calls, [("POST", "/tables/table-1/start", {}, None)])


if __name__ == "__main__":
    unittest.main()
