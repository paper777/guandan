from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from client.types import ActionRequest, JsonObject
from common.log import debug_event, error_event, trace_event
from npc.dummy_bot.player import DummyBotPlayer
from npc.llm_agent.config import ResolvedModelSettings
from npc.llm_agent.models import (
    ClaudeMessagesModelClient,
    CodexCliModelClient,
    DoubaoChatModelClient,
    GlmChatModelClient,
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
        timeout_seconds: float = 40.0,
        max_output_tokens: int = 800,
        fast_model: ResolvedModelSettings | None = None,
        pro_model: ResolvedModelSettings | None = None,
        memory_model: ResolvedModelSettings | None = None,
        audit_log_path: str | Path | None = None,
    ) -> None:
        self.model_client = model_client
        default_model = ResolvedModelSettings(
            role="fast",
            provider_name=type(model_client).__name__,
            api_base_url=None,
            model_name=model_name,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            model_reasoning_effort=None,
        )
        self.fast_model = _minimum_timeout(_with_role(fast_model or default_model, "fast"))
        self.pro_model = _minimum_timeout(_with_role(pro_model or self.fast_model, "pro"))
        self.memory_model = _minimum_timeout(_with_role(memory_model or self.pro_model, "memory"))
        self.model_name = self.fast_model.model_name
        self.temperature = self.fast_model.temperature
        self.timeout_seconds = self.fast_model.timeout_seconds
        self.max_output_tokens = self.fast_model.max_output_tokens
        self.audit_log_path = Path(audit_log_path) if audit_log_path is not None else None

    def choose_action(self, prompt: JsonObject) -> JsonObject:
        model = self._action_model(prompt)
        response = self._complete_with_audit(
            "action",
            ModelRequest(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(prompt_context(prompt)),
                model=model.model_name,
                temperature=model.temperature,
                timeout_seconds=model.timeout_seconds,
                max_output_tokens=model.max_output_tokens,
                model_reasoning_effort=model.model_reasoning_effort,
            ),
            metadata={"request_id": prompt.get("request_id"), "model_role": model.role},
        )
        return parse_json_object(response.content)

    def complete_memory(
        self,
        *,
        system_prompt: str,
        context: JsonObject,
        max_output_tokens: int | None = None,
    ) -> JsonObject:
        model = self.memory_model
        response = self._complete_with_audit(
            "memory",
            ModelRequest(
                system_prompt=system_prompt,
                user_prompt=build_user_prompt(context),
                model=model.model_name,
                temperature=model.temperature,
                timeout_seconds=model.timeout_seconds,
                max_output_tokens=max_output_tokens or model.max_output_tokens,
                model_reasoning_effort=model.model_reasoning_effort,
            ),
            metadata={"model_role": model.role},
        )
        return parse_json_object(response.content)

    def _action_model(self, prompt: JsonObject) -> ResolvedModelSettings:
        explicit_model = prompt.get("model")
        if isinstance(explicit_model, dict):
            role = explicit_model.get("role")
            if role == "pro":
                return self.pro_model
            if role == "fast":
                return self.fast_model
        return self.pro_model if _pressure_prompt(prompt) else self.fast_model

    def _complete_with_audit(self, purpose: str, request: ModelRequest, *, metadata: JsonObject) -> object:
        started_at = _utc_now()
        start = time.perf_counter()
        trace_event(
            "llm_provider.completion_started",
            purpose=purpose,
            metadata=metadata,
            provider=type(self.model_client).__name__,
            model=request.model,
            model_role=metadata.get("model_role"),
            timeout_seconds=request.timeout_seconds,
            max_output_tokens=request.max_output_tokens,
            model_reasoning_effort=request.model_reasoning_effort,
            request=_request_log_fields(request),
        )
        try:
            response = self.model_client.complete(request)
        except Exception as exc:
            completed_at = _utc_now()
            duration_ms = _elapsed_ms(start)
            error_event(
                "llm_provider.completion_failed",
                purpose=purpose,
                metadata=metadata,
                provider=type(self.model_client).__name__,
                model=request.model,
                model_role=metadata.get("model_role"),
                timeout_seconds=request.timeout_seconds,
                duration_ms=duration_ms,
                request=_request_log_fields(request),
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
        debug_event(
            "llm_provider.completion_completed",
            purpose=purpose,
            metadata=metadata,
            provider=type(self.model_client).__name__,
            model=request.model,
            model_role=metadata.get("model_role"),
            timeout_seconds=request.timeout_seconds,
            duration_ms=duration_ms,
            request=_request_log_fields(request),
            response=_response_log_fields(response),
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
                "model_role": metadata.get("model_role"),
                "temperature": request.temperature,
                "timeout_seconds": request.timeout_seconds,
                "max_output_tokens": request.max_output_tokens,
                "model_reasoning_effort": request.model_reasoning_effort,
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
    fast_model = _resolved_config_model(config, "fast", "deterministic-guandan-v1")
    provider_name = str(fast_model.provider_name or "deterministic").lower()
    if provider_name == "deterministic":
        return DeterministicLlmProvider()
    default_model_name = fast_model.model_name
    api_base_url = fast_model.api_base_url or getattr(config, "api_base_url", None)

    if provider_name in {"codex-cli", "codex_signed_in", "codex-signed-in"}:
        if not default_model_name or default_model_name == "deterministic-guandan-v1":
            default_model_name = "gpt-5.2"
            fast_model = _replace_model_name(fast_model, default_model_name)
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
    elif provider_name in {"glm", "bigmodel", "zhipu", "zhipuai"}:
        if not default_model_name or default_model_name == "deterministic-guandan-v1":
            default_model_name = "glm-5.1"
            fast_model = _replace_model_name(fast_model, default_model_name)
        api_key = _api_key_for(config, provider_name)
        client = GlmChatModelClient(
            api_key,
            base_url=str(api_base_url) if api_base_url else "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        )
    else:
        raise ValueError(f"unsupported LLM provider: {provider_name}")
    pro_model = _resolved_config_model(config, "pro", default_model_name)
    memory_model = _resolved_config_model(config, "memory", default_model_name)
    return ModelBackedLlmProvider(
        client,
        model_name=fast_model.model_name,
        temperature=fast_model.temperature,
        timeout_seconds=fast_model.timeout_seconds,
        max_output_tokens=fast_model.max_output_tokens,
        fast_model=fast_model,
        pro_model=pro_model,
        memory_model=memory_model,
        audit_log_path=getattr(config, "resolved_audit_log_path", lambda: None)(),
    )


def _resolved_config_model(config: object, role: str, default_model_name: str) -> ResolvedModelSettings:
    resolve_model = getattr(config, "resolved_model", None)
    if callable(resolve_model):
        model = resolve_model(role)
        model_name = model.model_name
        if not model_name or model_name == "deterministic-guandan-v1":
            model_name = default_model_name
        return _minimum_timeout(
            ResolvedModelSettings(
                role=role,
                provider_name=model.provider_name,
                api_base_url=model.api_base_url,
                model_name=model_name,
                temperature=model.temperature,
                timeout_seconds=model.timeout_seconds,
                max_output_tokens=model.max_output_tokens,
                model_reasoning_effort=model.model_reasoning_effort,
            )
        )
    return _minimum_timeout(
        ResolvedModelSettings(
            role=role,
            provider_name=str(getattr(config, "provider_name", "deterministic")),
            api_base_url=getattr(config, "api_base_url", None),
            model_name=default_model_name,
            temperature=float(getattr(config, "temperature", 0.2)),
            timeout_seconds=float(getattr(config, "timeout_seconds", 40.0)),
            max_output_tokens=int(getattr(config, "max_output_tokens", 800)),
            model_reasoning_effort=None,
        )
    )


def _minimum_timeout(model: ResolvedModelSettings) -> ResolvedModelSettings:
    timeout_seconds = model.timeout_seconds
    if timeout_seconds < 40.0:
        timeout_seconds = 40.0
    return ResolvedModelSettings(
        role=model.role,
        provider_name=model.provider_name,
        api_base_url=model.api_base_url,
        model_name=model.model_name,
        temperature=model.temperature,
        timeout_seconds=timeout_seconds,
        max_output_tokens=model.max_output_tokens,
        model_reasoning_effort=model.model_reasoning_effort,
    )


def _with_role(model: ResolvedModelSettings, role: str) -> ResolvedModelSettings:
    return ResolvedModelSettings(
        role=role,
        provider_name=model.provider_name,
        api_base_url=model.api_base_url,
        model_name=model.model_name,
        temperature=model.temperature,
        timeout_seconds=model.timeout_seconds,
        max_output_tokens=model.max_output_tokens,
        model_reasoning_effort=model.model_reasoning_effort,
    )


def _replace_model_name(model: ResolvedModelSettings, model_name: str) -> ResolvedModelSettings:
    return ResolvedModelSettings(
        role=model.role,
        provider_name=model.provider_name,
        api_base_url=model.api_base_url,
        model_name=model_name,
        temperature=model.temperature,
        timeout_seconds=model.timeout_seconds,
        max_output_tokens=model.max_output_tokens,
        model_reasoning_effort=model.model_reasoning_effort,
    )


def _pressure_prompt(prompt: JsonObject) -> bool:
    strategy_context = _dict(prompt.get("strategy_context"))
    pressure = _dict(strategy_context.get("pressure"))
    if pressure.get("endgame_defense") is True or pressure.get("partner_near_finish") is True:
        return True
    table_context = _dict(prompt.get("table_context"))
    if str(table_context.get("current_level") or "").upper() == "A":
        return True
    level_by_team = table_context.get("level_by_team")
    if isinstance(level_by_team, dict):
        return any(str(level).upper() == "A" for level in level_by_team.values())
    return False


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


def _request_log_fields(request: ModelRequest) -> JsonObject:
    return {
        "model": request.model,
        "temperature": request.temperature,
        "timeout_seconds": request.timeout_seconds,
        "max_output_tokens": request.max_output_tokens,
        "model_reasoning_effort": request.model_reasoning_effort,
        "system_prompt_chars": len(request.system_prompt),
        "user_prompt_chars": len(request.user_prompt),
    }


def _response_log_fields(response: object) -> JsonObject:
    raw = getattr(response, "raw", None)
    return {
        "content_chars": len(getattr(response, "content", "") or ""),
        "raw_keys": sorted(str(key) for key in raw) if isinstance(raw, dict) else None,
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
        "openai": ("OPENAI_API_KEY",),
        "codex": ("OPENAI_API_KEY",),
        "claude": ("ANTHROPIC_API_KEY",),
        "doubao": ("DOUBAO_API_KEY",),
        "glm": ("BIGMODEL_API_KEY", "ZHIPUAI_API_KEY", "GLM_API_KEY"),
        "bigmodel": ("BIGMODEL_API_KEY", "ZHIPUAI_API_KEY", "GLM_API_KEY"),
        "zhipu": ("BIGMODEL_API_KEY", "ZHIPUAI_API_KEY", "GLM_API_KEY"),
        "zhipuai": ("BIGMODEL_API_KEY", "ZHIPUAI_API_KEY", "GLM_API_KEY"),
    }
    env_names = env_by_provider.get(provider_name, ())
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    env_display = " or ".join(env_names)
    if not env_display:
        env_display = "api_key"
    raise ValueError(f"{env_display} is required for LLM provider {provider_name}")
