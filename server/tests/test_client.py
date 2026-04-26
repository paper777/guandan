from __future__ import annotations

import unittest

from guandan.client import GuandanHttpClient


class ClientTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
