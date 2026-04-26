from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ActionRequest:
    request_id: str
    prompt: JsonObject
    snapshot: JsonObject

    @classmethod
    def from_payload(cls, payload: JsonObject) -> ActionRequest:
        return cls(
            request_id=str(payload.get("request_id", "")),
            prompt=_dict(payload.get("prompt")),
            snapshot=_dict(payload.get("snapshot")),
        )


class NpcPolicy(Protocol):
    def choose_action(self, request: ActionRequest) -> JsonObject:
        """Return a command-like action for the broker or HTTP agent server."""


class NpcClientError(RuntimeError):
    def __init__(self, status: int | None, message: str, payload: JsonObject | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload or {}


@dataclass(slots=True)
class JsonHttpClient:
    base_url: str
    timeout: float = 10.0

    def request(
        self,
        method: str,
        path: str,
        body: JsonObject | None = None,
        *,
        query: JsonObject | None = None,
    ) -> JsonObject:
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        url = urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))
        if query:
            url = f"{url}?{urlencode(query)}"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return _decode_response(response.read())
        except HTTPError as exc:
            payload = _decode_response(exc.read())
            raise NpcClientError(exc.code, _error_message(payload), payload) from exc
        except URLError as exc:
            raise NpcClientError(None, f"could not connect: {exc.reason}") from exc


@dataclass(slots=True)
class GuandanNpcClient:
    base_url: str = "http://127.0.0.1:8000"
    timeout: float = 10.0

    def __post_init__(self) -> None:
        self._http = JsonHttpClient(self.base_url, timeout=self.timeout)

    def create_table(self) -> JsonObject:
        return self._http.request("POST", "/tables", {})

    def table_snapshot(self, table_id: str) -> JsonObject:
        return self._http.request("GET", f"/tables/{table_id}")

    def seat_snapshot(self, table_id: str, seat: str, controller_id: str) -> JsonObject:
        return self._http.request(
            "GET",
            f"/tables/{table_id}/seats/{seat}/snapshot",
            query={"controller_id": controller_id},
        )

    def join_agent(self, table_id: str, seat: str, display_name: str) -> JsonObject:
        return self._http.request(
            "POST",
            f"/tables/{table_id}/join-agent",
            {"seat": seat, "display_name": display_name},
        )

    def ready(self, table_id: str, seat: str, controller_id: str) -> JsonObject:
        return self._http.request("POST", f"/tables/{table_id}/ready", {"seat": seat, "controller_id": controller_id})

    def start(self, table_id: str) -> JsonObject:
        return self._http.request("POST", f"/tables/{table_id}/start", {})

    def play_cards(self, table_id: str, seat: str, controller_id: str, card_ids: tuple[str, ...]) -> JsonObject:
        return self._http.request(
            "POST",
            f"/tables/{table_id}/play",
            {"seat": seat, "controller_id": controller_id, "card_ids": list(card_ids)},
        )

    def pass_turn(self, table_id: str, seat: str, controller_id: str) -> JsonObject:
        return self._http.request("POST", f"/tables/{table_id}/pass", {"seat": seat, "controller_id": controller_id})


def _decode_response(raw: bytes) -> JsonObject:
    if not raw:
        return {}
    value = json.loads(raw.decode())
    if not isinstance(value, dict):
        raise NpcClientError(None, "server returned a non-object JSON response")
    return value


def _error_message(payload: JsonObject) -> str:
    if "rejection" in payload and isinstance(payload["rejection"], dict):
        rejection = payload["rejection"]
        return f"{rejection.get('code', 'rejected')}: {rejection.get('message', '')}".rstrip()
    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    error = payload.get("error")
    if isinstance(error, str):
        return error
    return "request failed"


def _dict(value: Any) -> JsonObject:
    return value if isinstance(value, dict) else {}
