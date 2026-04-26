from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any

from guandan.services.snapshots import SeatSnapshot


@dataclass(frozen=True, slots=True)
class ExternalAgentClient:
    agent_url: str
    shared_secret: str | None = None
    timeout_seconds: float = 3.0

    def build_payload(self, request_id: str, snapshot: SeatSnapshot, prompt: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocol_version": "1",
            "request_id": request_id,
            "deadline_ms": int(self.timeout_seconds * 1000),
            "snapshot": {
                "table_id": snapshot.public.table_id,
                "seat": snapshot.seat.value,
                "seq": snapshot.public.event_seq,
                "phase": snapshot.public.phase.value,
                "hand": list(snapshot.hand),
                "public_state": {
                    "current_turn": snapshot.public.current_turn.value if snapshot.public.current_turn else None,
                    "finish_order": [seat.value for seat in snapshot.public.finish_order],
                },
            },
            "prompt": prompt,
        }

    def request_action(self, request_id: str, snapshot: SeatSnapshot, prompt: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(self.build_payload(request_id, snapshot, prompt)).encode()
        headers = {"content-type": "application/json"}
        if self.shared_secret is not None:
            headers["x-guandan-agent-secret"] = self.shared_secret
        request = urllib.request.Request(self.agent_url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode())
