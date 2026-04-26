from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from npc.common.client import ActionRequest, JsonObject, NpcPolicy


def run_policy_server(policy: NpcPolicy, *, host: str = "127.0.0.1", port: int = 9001) -> None:
    handler = _handler_for(policy)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"npc policy server listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _handler_for(policy: NpcPolicy) -> type[BaseHTTPRequestHandler]:
    class PolicyHandler(BaseHTTPRequestHandler):
        server_version = "GuandanNpcPolicy/0.1"

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length).decode() or "{}")
                if not isinstance(payload, dict):
                    raise ValueError("payload must be a JSON object")
                response = policy.choose_action(ActionRequest.from_payload(payload))
                status = 200
            except Exception as exc:  # pragma: no cover - defensive HTTP boundary
                response = {"type": "error", "message": str(exc)}
                status = 400
            self._send_json(status, response)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"ok": True, "service": "guandan-npc-policy"})
                return
            self._send_json(404, {"error": "not found"})

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, payload: JsonObject) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return PolicyHandler
