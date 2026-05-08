from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from client.types import JsonObject

Transport = Callable[[str, str, JsonObject | None, JsonObject | None], JsonObject]


class GuandanClientError(RuntimeError):
    def __init__(self, status: int | None, message: str, payload: JsonObject | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload or {}

@dataclass(slots=True)
class JsonHttpClient:
    base_url: str
    timeout: float = 10.0
    transport: Transport | None = None

    def request(
        self,
        method: str,
        path: str,
        body: JsonObject | None = None,
        *,
        query: JsonObject | None = None,
    ) -> JsonObject:
        return _request_json(self.base_url, self.timeout, self.transport, method, path, body, query)


@dataclass(slots=True)
class GuandanHttpClient:
    base_url: str = "http://127.0.0.1:8000"
    timeout: float = 10.0
    transport: Transport | None = None

    def create_table(
        self,
        *,
        action_timeout_seconds: int | None = None,
        timeout_fallback: str | None = None,
    ) -> JsonObject:
        return self._request(
            "POST",
            "/tables",
            _without_none(
                {
                    "action_timeout_seconds": action_timeout_seconds,
                    "timeout_fallback": timeout_fallback,
                }
            ),
        )

    def list_tables(self) -> JsonObject:
        return self._request("GET", "/tables")

    def table_snapshot(self, table_id: str) -> JsonObject:
        return self._request("GET", f"/tables/{table_id}")

    def seat_snapshot(self, table_id: str, seat: str, controller_id: str) -> JsonObject:
        return self._request(
            "GET",
            f"/tables/{table_id}/seats/{seat}/snapshot",
            query={"controller_id": controller_id},
        )

    def join_human(
        self,
        table_id: str,
        seat: str,
        *,
        player_id: str | None = None,
        controller_id: str | None = None,
        display_name: str | None = None,
    ) -> JsonObject:
        return self._join(
            table_id,
            "join-human",
            seat,
            player_id=player_id,
            controller_id=controller_id,
            display_name=display_name,
        )

    def join_local_bot(
        self,
        table_id: str,
        seat: str,
        *,
        player_id: str | None = None,
        controller_id: str | None = None,
        display_name: str | None = None,
    ) -> JsonObject:
        return self._join(
            table_id,
            "join-local-bot",
            seat,
            player_id=player_id,
            controller_id=controller_id,
            display_name=display_name,
        )

    def join_agent(
        self,
        table_id: str,
        seat: str,
        display_name: str,
        *,
        player_id: str | None = None,
        controller_id: str | None = None,
        agent_url: str | None = None,
        timeout_ms: int | None = None,
        timeout_fallback: str | None = None,
        shared_secret: str | None = None,
    ) -> JsonObject:
        return self._request(
            "POST",
            f"/tables/{table_id}/join-agent",
            _without_none(
                {
                    "seat": seat,
                    "player_id": player_id,
                    "controller_id": controller_id,
                    "display_name": display_name,
                    "agent_url": agent_url,
                    "timeout_ms": timeout_ms,
                    "timeout_fallback": timeout_fallback,
                    "shared_secret": shared_secret,
                }
            ),
        )

    def ready(self, table_id: str, seat: str, controller_id: str) -> JsonObject:
        return self._request("POST", f"/tables/{table_id}/ready", {"seat": seat, "controller_id": controller_id})

    def start(self, table_id: str) -> JsonObject:
        return self._request("POST", f"/tables/{table_id}/start", {})

    def play_cards(
        self,
        table_id: str,
        seat: str,
        controller_id: str,
        card_ids: tuple[str, ...],
        *,
        declared_type: str | None = None,
    ) -> JsonObject:
        return self._request(
            "POST",
            f"/tables/{table_id}/play",
            _without_none(
                {
                    "seat": seat,
                    "controller_id": controller_id,
                    "card_ids": list(card_ids),
                    "declared_type": declared_type,
                }
            ),
        )

    def pass_turn(self, table_id: str, seat: str, controller_id: str) -> JsonObject:
        return self._request("POST", f"/tables/{table_id}/pass", {"seat": seat, "controller_id": controller_id})

    def submit_tribute(self, table_id: str, seat: str, controller_id: str, card_id: str) -> JsonObject:
        return self._request(
            "POST",
            f"/tables/{table_id}/tribute",
            {"seat": seat, "controller_id": controller_id, "card_id": card_id},
        )

    def return_tribute(self, table_id: str, seat: str, controller_id: str, card_id: str) -> JsonObject:
        return self._request(
            "POST",
            f"/tables/{table_id}/return-tribute",
            {"seat": seat, "controller_id": controller_id, "card_id": card_id},
        )

    def _join(
        self,
        table_id: str,
        action: str,
        seat: str,
        *,
        player_id: str | None,
        controller_id: str | None,
        display_name: str | None,
    ) -> JsonObject:
        return self._request(
            "POST",
            f"/tables/{table_id}/{action}",
            _without_none(
                {
                    "seat": seat,
                    "player_id": player_id,
                    "controller_id": controller_id,
                    "display_name": display_name,
                }
            ),
        )

    def _request(
        self,
        method: str,
        path: str,
        body: JsonObject | None = None,
        *,
        query: JsonObject | None = None,
    ) -> JsonObject:
        return _request_json(self.base_url, self.timeout, self.transport, method, path, body, query)


def _request_json(
    base_url: str,
    timeout: float,
    transport: Transport | None,
    method: str,
    path: str,
    body: JsonObject | None = None,
    query: JsonObject | None = None,
) -> JsonObject:
    if transport is not None:
        return transport(method, path, body, query)

    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    if query:
        url = f"{url}?{urlencode(query)}"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return _decode_response(response.read())
    except HTTPError as exc:
        payload = _decode_response(exc.read())
        raise GuandanClientError(exc.code, _error_message(payload), payload) from exc
    except URLError as exc:
        raise GuandanClientError(None, f"could not connect to Guandan server: {exc.reason}") from exc


def _decode_response(raw: bytes) -> JsonObject:
    if not raw:
        return {}
    value = json.loads(raw.decode())
    if not isinstance(value, dict):
        raise GuandanClientError(None, "server returned a non-object JSON response")
    return value


def _error_message(payload: JsonObject) -> str:
    if "rejection" in payload and isinstance(payload["rejection"], dict):
        rejection = payload["rejection"]
        code = rejection.get("code", "rejected")
        message = rejection.get("message", "")
        return f"{code}: {message}".rstrip()
    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    error = payload.get("error")
    if isinstance(error, str):
        return error
    return "Guandan server request failed"


def _without_none(payload: JsonObject) -> JsonObject:
    return {key: value for key, value in payload.items() if value is not None}
