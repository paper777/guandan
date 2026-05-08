from __future__ import annotations

import json
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

from client.types import JsonObject
from common.log import elapsed_ms, trace_event


HttpTransport = Callable[[str, JsonObject, JsonObject, float], JsonObject]
CommandRunner = Callable[[list[str], str, float, Path | None], str]


@dataclass(frozen=True, slots=True)
class ModelRequest:
    system_prompt: str
    user_prompt: str
    model: str
    temperature: float = 0.2
    timeout_seconds: float = 3.0
    max_output_tokens: int = 800


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str
    raw: JsonObject


class ModelClient(ABC):
    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return model text for a structured prompt request."""


class CodexCliModelClient(ModelClient):
    """Model client that uses the signed-in Codex CLI instead of an API key."""

    def __init__(
        self,
        *,
        codex_binary: str = "codex",
        working_dir: str | Path | None = None,
        runner: CommandRunner | None = None,
    ) -> None:
        self.codex_binary = codex_binary
        self.working_dir = Path(working_dir) if working_dir is not None else None
        self.runner = runner or _run_codex_exec

    def complete(self, request: ModelRequest) -> ModelResponse:
        prompt = "\n\n".join(
            (
                request.system_prompt,
                "Return only the final JSON object for this NPC action.",
                request.user_prompt,
            )
        )
        command = [
            self.codex_binary,
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--model",
            request.model,
            "-",
        ]
        content = self.runner(command, prompt, request.timeout_seconds, self.working_dir)
        return ModelResponse(content=content, raw={"provider": "codex-cli", "content": content})


class OpenAIResponsesModelClient(ModelClient):
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1/responses",
        transport: HttpTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.transport = transport or _post_json

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload: JsonObject = {
            "model": request.model,
            "instructions": request.system_prompt,
            "input": request.user_prompt,
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
        }
        raw = self.transport(
            self.base_url,
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            payload,
            request.timeout_seconds,
        )
        return ModelResponse(content=_openai_text(raw), raw=raw)


class ClaudeMessagesModelClient(ModelClient):
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.anthropic.com/v1/messages",
        transport: HttpTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.transport = transport or _post_json

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload: JsonObject = {
            "model": request.model,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": request.user_prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        raw = self.transport(
            self.base_url,
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload,
            request.timeout_seconds,
        )
        return ModelResponse(content=_claude_text(raw), raw=raw)


class DoubaoChatModelClient(ModelClient):
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        transport: HttpTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.transport = transport or _post_json

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload: JsonObject = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        raw = self.transport(
            self.base_url,
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            payload,
            request.timeout_seconds,
        )
        return ModelResponse(content=_chat_completion_text(raw), raw=raw)


def _post_json(url: str, headers: JsonObject, payload: JsonObject, timeout_seconds: float) -> JsonObject:
    started = time.perf_counter()
    trace_event(
        "llm_model.http_started",
        url=url,
        model=payload.get("model"),
        timeout_seconds=timeout_seconds,
    )
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={str(key): str(value) for key, value in headers.items()},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = json.loads(response.read().decode())
    except Exception as exc:
        trace_event(
            "llm_model.http_failed",
            url=url,
            model=payload.get("model"),
            timeout_seconds=timeout_seconds,
            duration_ms=elapsed_ms(started),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    if not isinstance(raw, dict):
        raise ValueError("model API returned a non-object JSON response")
    trace_event(
        "llm_model.http_completed",
        url=url,
        model=payload.get("model"),
        timeout_seconds=timeout_seconds,
        duration_ms=elapsed_ms(started),
    )
    return raw


def _run_codex_exec(command: list[str], prompt: str, timeout_seconds: float, working_dir: Path | None) -> str:
    started = time.perf_counter()
    trace_event(
        "llm_model.codex_exec_started",
        command=command[:6],
        model=_command_model(command),
        timeout_seconds=timeout_seconds,
        working_dir=str(working_dir) if working_dir is not None else None,
        prompt_chars=len(prompt),
    )
    with tempfile.NamedTemporaryFile(prefix="guandan-codex-", suffix=".txt", delete=False) as tmp:
        output_path = Path(tmp.name)
    try:
        command_with_output = [*command[:-1], "--output-last-message", str(output_path), command[-1]]
        completed = subprocess.run(
            command_with_output,
            input=prompt,
            text=True,
            capture_output=True,
            cwd=working_dir,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or f"codex exited {completed.returncode}"
            trace_event(
                "llm_model.codex_exec_failed",
                model=_command_model(command),
                timeout_seconds=timeout_seconds,
                duration_ms=elapsed_ms(started),
                returncode=completed.returncode,
                stderr_chars=len(completed.stderr or ""),
                stdout_chars=len(completed.stdout or ""),
                error={"type": "RuntimeError", "message": message},
            )
            raise RuntimeError(message)
        content = output_path.read_text(encoding="utf-8").strip()
        result = content or completed.stdout.strip()
        trace_event(
            "llm_model.codex_exec_completed",
            model=_command_model(command),
            timeout_seconds=timeout_seconds,
            duration_ms=elapsed_ms(started),
            content_chars=len(result),
        )
        return result
    except subprocess.TimeoutExpired as exc:
        trace_event(
            "llm_model.codex_exec_failed",
            model=_command_model(command),
            timeout_seconds=timeout_seconds,
            duration_ms=elapsed_ms(started),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    finally:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass


def _openai_text(raw: JsonObject) -> str:
    output_text = raw.get("output_text")
    if isinstance(output_text, str):
        return output_text
    texts: list[str] = []
    output = raw.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
    return "\n".join(texts)


def _claude_text(raw: JsonObject) -> str:
    content = raw.get("content")
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(part["text"])
        for part in content
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )


def _chat_completion_text(raw: JsonObject) -> str:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return ""


def parse_json_object(text: str) -> JsonObject:
    value: Any = json.loads(_strip_json_fences(text))
    if not isinstance(value, dict):
        raise ValueError("model returned a non-object JSON value")
    return value


def _command_model(command: list[str]) -> str | None:
    try:
        index = command.index("--model")
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def _strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
