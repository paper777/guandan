from __future__ import annotations

import os
from typing import Protocol

from client.api import ActionRequest, JsonObject
from npc.dummy_bot.policy import DummyBotPolicy
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
        self._fallback = DummyBotPolicy()

    def choose_action(self, prompt: JsonObject) -> JsonObject:
        request = ActionRequest(
            request_id=str(prompt.get("request_id", "")),
            prompt=_dict(prompt.get("prompt")),
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
    ) -> None:
        self.model_client = model_client
        self.model_name = model_name
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens

    def choose_action(self, prompt: JsonObject) -> JsonObject:
        response = self.model_client.complete(
            ModelRequest(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(prompt_context(prompt)),
                model=self.model_name,
                temperature=self.temperature,
                timeout_seconds=self.timeout_seconds,
                max_output_tokens=self.max_output_tokens,
            )
        )
        return parse_json_object(response.content)

    def complete_memory(
        self,
        *,
        system_prompt: str,
        context: JsonObject,
        max_output_tokens: int | None = None,
    ) -> JsonObject:
        response = self.model_client.complete(
            ModelRequest(
                system_prompt=system_prompt,
                user_prompt=build_user_prompt(context),
                model=self.model_name,
                temperature=self.temperature,
                timeout_seconds=self.timeout_seconds,
                max_output_tokens=max_output_tokens or self.max_output_tokens,
            )
        )
        return parse_json_object(response.content)


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
    )


def _dict(value: object) -> JsonObject:
    return value if isinstance(value, dict) else {}


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
