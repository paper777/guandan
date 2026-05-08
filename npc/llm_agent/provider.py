from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from client.types import ActionRequest, JsonObject
from common.log import trace_event
from npc.dummy_bot.player import DummyBotPlayer
from npc.llm_agent.models import (
    ClaudeMessagesModelClient,
    CodexCliModelClient,
    DoubaoChatModelClient,
    ModelClient,
    ModelRequest,
    OpenAIResponsesModelClient,
    parse_json_object,
)
from npc.llm_agent.prompts import SYSTEM_PROMPT, build_user_prompt, prompt_context


class LlmActionProvider(Protocol):
    def choose_action(self, prompt: JsonObject) -> JsonObject:
        """Return structured action JSON with optional LLM-authored diagnostics."""


class DeterministicLlmProvider:
    """Dependency-free provider used for tests and local dry runs."""

    def __init__(self) -> None:
        self._fallback = DummyBotPlayer()

    def choose_action(self, prompt: JsonObject) -> JsonObject:
        action_prompt = _dict(prompt.get("prompt")) or _prompt_from_table_context(_dict(prompt.get("table_context")))
        request = ActionRequest(
            request_id=str(prompt.get("request_id", "")),
            prompt=action_prompt,
            snapshot=_dict(prompt.get("snapshot")),
        )
        return self._fallback.choose_action(request)


class ModelBackedLlmProvider:
    """LLM action provider backed by an external model API client."""

    def __init__(
        self,
        model_client: ModelClient,
        *,
        model_name: str,
        temperature: float = 0.2,
        timeout_seconds: float = 3.0,
        max_output_tokens: int = 800,
        audit_log_path: str | Path | None = None,
    ) -> None:
        self.model_client = model_client
        self.model_name = model_name
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.audit_log_path = Path(audit_log_path) if audit_log_path is not None else None

    def choose_action(self, prompt: JsonObject) -> JsonObject:
        response = self._complete_with_audit(
            "action",
            ModelRequest(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(prompt_context(prompt)),
                model=self.model_name,
                temperature=self.temperature,
                timeout_seconds=self.timeout_seconds,
                max_output_tokens=self.max_output_tokens,
            ),
            metadata={"request_id": prompt.get("request_id")},
        )
        return parse_json_object(response.content)

    def complete_memory(
        self,
        *,
        system_prompt: str,
        context: JsonObject,
        max_output_tokens: int | None = None,
    ) -> JsonObject:
        response = self._complete_with_audit(
            "memory",
            ModelRequest(
                system_prompt=system_prompt,
                user_prompt=build_user_prompt(context),
                model=self.model_name,
                temperature=self.temperature,
                timeout_seconds=self.timeout_seconds,
                max_output_tokens=max_output_tokens or self.max_output_tokens,
            ),
            metadata={},
        )
        return parse_json_object(response.content)

    def _complete_with_audit(self, purpose: str, request: ModelRequest, *, metadata: JsonObject) -> object:
        started_at = _utc_now()
        start = time.perf_counter()
        trace_event(
            "llm_provider.completion_started",
            purpose=purpose,
            metadata=metadata,
            provider=type(self.model_client).__name__,
            model=request.model,
            timeout_seconds=request.timeout_seconds,
            max_output_tokens=request.max_output_tokens,
        )
        try:
            response = self.model_client.complete(request)
        except Exception as exc:
            completed_at = _utc_now()
            duration_ms = _elapsed_ms(start)
            trace_event(
                "llm_provider.completion_failed",
                purpose=purpose,
                metadata=metadata,
                provider=type(self.model_client).__name__,
                model=request.model,
                timeout_seconds=request.timeout_seconds,
                duration_ms=duration_ms,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            self._append_audit_entry(
                purpose,
                request,
                metadata=metadata,
                timing=_timing(started_at, completed_at, duration_ms),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise
        completed_at = _utc_now()
        duration_ms = _elapsed_ms(start)
        trace_event(
            "llm_provider.completion_completed",
            purpose=purpose,
            metadata=metadata,
            provider=type(self.model_client).__name__,
            model=request.model,
            timeout_seconds=request.timeout_seconds,
            duration_ms=duration_ms,
            content_chars=len(getattr(response, "content", "") or ""),
        )
        self._append_audit_entry(
            purpose,
            request,
            metadata=metadata,
            timing=_timing(started_at, completed_at, duration_ms),
            response={
                "content": getattr(response, "content", ""),
                "raw": getattr(response, "raw", None),
            },
        )
        return response

    def _append_audit_entry(
        self,
        purpose: str,
        request: ModelRequest,
        *,
        metadata: JsonObject,
        timing: JsonObject,
        response: JsonObject | None = None,
        error: JsonObject | None = None,
    ) -> None:
        if self.audit_log_path is None:
            return
        entry: JsonObject = {
            "timestamp": timing["completed_at"],
            "purpose": purpose,
            "metadata": metadata,
            "timing": timing,
            "provider": type(self.model_client).__name__,
            "request": {
                "model": request.model,
                "temperature": request.temperature,
                "timeout_seconds": request.timeout_seconds,
                "max_output_tokens": request.max_output_tokens,
                "system_prompt": request.system_prompt,
                "user_prompt": request.user_prompt,
            },
        }
        if response is not None:
            entry["response"] = response
        if error is not None:
            entry["error"] = error
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def provider_from_config(config: object) -> LlmActionProvider:
    provider_name = str(getattr(config, "provider_name", "deterministic")).lower()
    if provider_name == "deterministic":
        return DeterministicLlmProvider()
    model_name = str(getattr(config, "model_name", ""))
    temperature = float(getattr(config, "temperature", 0.2))
    timeout_seconds = float(getattr(config, "timeout_seconds", 3.0))
    max_output_tokens = int(getattr(config, "max_output_tokens", 800))
    api_base_url = getattr(config, "api_base_url", None)

    if provider_name in {"codex-cli", "codex_signed_in", "codex-signed-in"}:
        if not model_name or model_name == "deterministic-guandan-v1":
            model_name = "gpt-5.2"
        if timeout_seconds == 3.0:
            timeout_seconds = 120.0
        client = CodexCliModelClient(
            codex_binary=str(getattr(config, "codex_binary", "codex")),
            working_dir=getattr(config, "codex_working_dir", None),
        )
    elif provider_name in {"openai", "codex"}:
        api_key = _api_key_for(config, provider_name)
        client = OpenAIResponsesModelClient(
            api_key,
            base_url=str(api_base_url) if api_base_url else "https://api.openai.com/v1/responses",
        )
    elif provider_name == "claude":
        api_key = _api_key_for(config, provider_name)
        client = ClaudeMessagesModelClient(
            api_key,
            base_url=str(api_base_url) if api_base_url else "https://api.anthropic.com/v1/messages",
        )
    elif provider_name == "doubao":
        api_key = _api_key_for(config, provider_name)
        client = DoubaoChatModelClient(
            api_key,
            base_url=str(api_base_url) if api_base_url else "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        )
    else:
        raise ValueError(f"unsupported LLM provider: {provider_name}")
    return ModelBackedLlmProvider(
        client,
        model_name=model_name,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_output_tokens=max_output_tokens,
        audit_log_path=getattr(config, "resolved_audit_log_path", lambda: None)(),
    )


def _dict(value: object) -> JsonObject:
    return value if isinstance(value, dict) else {}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _timing(started_at: str, completed_at: str, duration_ms: float) -> JsonObject:
    return {
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": duration_ms,
    }


def _prompt_from_table_context(table_context: JsonObject) -> JsonObject:
    prompt: JsonObject = {}
    kind = table_context.get("prompt_kind")
    if kind is not None:
        prompt["kind"] = kind
    for key in ("current_level", "current_trick", "tribute_from", "tribute_to", "return_rank_at_most_ten"):
        value = table_context.get(key)
        if value is not None:
            prompt[key] = value
    return prompt


def _api_key_for(config: object, provider_name: str) -> str:
    configured = getattr(config, "api_key", None)
    if configured:
        return str(configured)
    env_by_provider = {
        "openai": "OPENAI_API_KEY",
        "codex": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "doubao": "DOUBAO_API_KEY",
    }
    env_name = env_by_provider.get(provider_name, "")
    value = os.environ.get(env_name)
    if not value:
        raise ValueError(f"{env_name} is required for LLM provider {provider_name}")
    return value
