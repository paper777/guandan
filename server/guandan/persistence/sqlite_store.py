from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from guandan.domain.events import Event


class SQLiteEventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.configure()
        self.init_schema()

    def close(self) -> None:
        self.connection.close()

    def configure(self) -> None:
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 5000")

    def init_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS matches (
              id TEXT PRIMARY KEY,
              table_id TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS match_events (
              match_id TEXT NOT NULL,
              seq INTEGER NOT NULL,
              type TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (match_id, seq),
              FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS idempotency_keys (
              match_id TEXT NOT NULL,
              controller_id TEXT NOT NULL,
              request_id TEXT NOT NULL,
              first_seq INTEGER NOT NULL,
              last_seq INTEGER NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (match_id, controller_id, request_id),
              FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE
            );
            """
        )
        self.connection.commit()

    def create_match(self, match_id: str, table_id: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO matches (id, table_id) VALUES (?, ?)",
                (match_id, table_id),
            )

    def append_events(self, match_id: str, events: Iterable[Event]) -> None:
        rows = [(match_id, event.seq, event.type, json.dumps(event.payload, sort_keys=True)) for event in events]
        if not rows:
            return
        with self.connection:
            self.connection.executemany(
                "INSERT INTO match_events (match_id, seq, type, payload_json) VALUES (?, ?, ?, ?)",
                rows,
            )

    def load_events(self, match_id: str) -> tuple[Event, ...]:
        rows = self.connection.execute(
            "SELECT seq, type, payload_json FROM match_events WHERE match_id = ? ORDER BY seq ASC",
            (match_id,),
        ).fetchall()
        return tuple(Event(seq=row["seq"], type=row["type"], payload=json.loads(row["payload_json"])) for row in rows)

    def record_idempotency(
        self,
        match_id: str,
        controller_id: str,
        request_id: str,
        first_seq: int,
        last_seq: int,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO idempotency_keys (match_id, controller_id, request_id, first_seq, last_seq)
                VALUES (?, ?, ?, ?, ?)
                """,
                (match_id, controller_id, request_id, first_seq, last_seq),
            )

    def find_idempotency(self, match_id: str, controller_id: str, request_id: str) -> tuple[int, int] | None:
        row = self.connection.execute(
            """
            SELECT first_seq, last_seq FROM idempotency_keys
            WHERE match_id = ? AND controller_id = ? AND request_id = ?
            """,
            (match_id, controller_id, request_id),
        ).fetchone()
        if row is None:
            return None
        return row["first_seq"], row["last_seq"]
