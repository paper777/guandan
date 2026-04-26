from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from guandan.domain.events import Event
from guandan.persistence.sqlite_store import SQLiteEventStore


class SQLiteEventStoreTests(unittest.TestCase):
    def test_append_and_load_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteEventStore(Path(tmp) / "events.db")
            store.create_match("match-1", "table-1")

            store.append_events(
                "match-1",
                (
                    Event(seq=1, type="MatchStarted", payload={"table_id": "table-1"}),
                    Event(seq=2, type="DealStarted", payload={"leader": "E"}),
                ),
            )

            loaded = store.load_events("match-1")
            store.close()

        self.assertEqual([event.seq for event in loaded], [1, 2])
        self.assertEqual(loaded[1].payload["leader"], "E")

    def test_idempotency_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteEventStore(Path(tmp) / "events.db")
            store.create_match("match-1", "table-1")
            store.record_idempotency("match-1", "controller-1", "request-1", 10, 12)

            found = store.find_idempotency("match-1", "controller-1", "request-1")
            missing = store.find_idempotency("match-1", "controller-1", "missing")
            store.close()

        self.assertEqual(found, (10, 12))
        self.assertIsNone(missing)


if __name__ == "__main__":
    unittest.main()
