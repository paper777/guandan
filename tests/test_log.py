from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common.log import (
    TRACE_LOG_ENABLED_ENV,
    TRACE_LOG_PATH_ENV,
    deadline_remaining_ms,
    redact_trace_payload,
    trace_event,
)


class TraceLogTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
