from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common.log import (
    AUDIT_LOG_ENABLED_ENV,
    AUDIT_LOG_PATH_ENV,
    DEFAULT_AUDIT_LOG_PATH,
    DEFAULT_LOG_DIR,
    DEFAULT_TRACE_LOG_PATH,
    TRACE_LOG_ENABLED_ENV,
    TRACE_LOG_PATH_ENV,
    audit_log_path,
    deadline_remaining_ms,
    make_audit_entry,
    redact_trace_payload,
    trace_event,
    trace_log_path,
    write_audit_entry,
)


class TraceLogTests(unittest.TestCase):
    def test_default_log_paths_live_under_data_log(self) -> None:
        with patch.dict(os.environ, {TRACE_LOG_PATH_ENV: "", AUDIT_LOG_PATH_ENV: ""}, clear=False):
            self.assertEqual(DEFAULT_LOG_DIR, Path("data/log"))
            self.assertEqual(DEFAULT_TRACE_LOG_PATH, Path("data/log/guandan_trace.jsonl"))
            self.assertEqual(DEFAULT_AUDIT_LOG_PATH, Path("data/log/server_audit.jsonl"))
            self.assertEqual(trace_log_path(), DEFAULT_TRACE_LOG_PATH)
            self.assertEqual(audit_log_path(), DEFAULT_AUDIT_LOG_PATH)

    def test_trace_event_writes_jsonl_and_redacts_private_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            with patch.dict(os.environ, {TRACE_LOG_PATH_ENV: str(path), TRACE_LOG_ENABLED_ENV: "1"}):
                trace_event(
                    "test.event",
                    table_id="table-1",
                    controller_id="controller-1",
                    action={"type": "play_cards", "card_ids": ["D1-S-3"]},
                )

            entry = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(entry["event"], "test.event")
            self.assertEqual(entry["level"], "trace")
            self.assertEqual(entry["controller_id"], "<redacted>")
            self.assertEqual(entry["action"]["card_ids"], "<redacted>")
            self.assertEqual(entry["table_id"], "table-1")

    def test_trace_event_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            with patch.dict(os.environ, {TRACE_LOG_PATH_ENV: str(path), TRACE_LOG_ENABLED_ENV: "0"}):
                trace_event("test.disabled")

            self.assertFalse(path.exists())

    def test_deadline_remaining_ms_reports_negative_after_deadline(self) -> None:
        self.assertEqual(deadline_remaining_ms(900, now_epoch_ms=1000), -100)
        self.assertIsNone(deadline_remaining_ms("900", now_epoch_ms=1000))

    def test_redact_trace_payload_handles_nested_values(self) -> None:
        self.assertEqual(
            redact_trace_payload({"outer": [{"hand": ["D1-S-3"], "safe": "ok"}]}),
            {"outer": [{"hand": "<redacted>", "safe": "ok"}]},
        )

    def test_audit_entry_helpers_live_in_common_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            with patch.dict(os.environ, {AUDIT_LOG_PATH_ENV: str(path), AUDIT_LOG_ENABLED_ENV: "1"}):
                entry = make_audit_entry(
                    method="GET",
                    path="/tables/table-1/seats/E/snapshot",
                    query="controller_id=c-E",
                    status=200,
                    started_at=0.0,
                    request_body={},
                    response_body={"hand": ["D1-S-3"], "public": {"event_seq": 1}},
                    client=("testclient", 50000),
                )
                write_audit_entry(entry)

            written = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(written["client"], "testclient")
            self.assertEqual(written["request"]["query"], {"controller_id": "<redacted>"})
            self.assertEqual(written["response"]["body"]["hand"], "<redacted>")


if __name__ == "__main__":
    unittest.main()
