from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from npc.llm_agent.models import (
    ClaudeMessagesModelClient,
    CodexCliModelClient,
    DoubaoChatModelClient,
    GlmChatModelClient,
    ModelRequest,
    OpenAIResponsesModelClient,
)
from npc.llm_agent.provider import ModelBackedLlmProvider, provider_from_config
from npc.llm_agent.config import LlmAgentConfig, ModelSettings, ResolvedModelSettings
from npc.llm_agent.prompts import MEMORY_TECHNIQUE_SUMMARY_PROMPT


class CapturingTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, url, headers, payload, timeout_seconds):
        self.calls.append((url, headers, payload, timeout_seconds))
        return self.response


class ModelAdapterTests(unittest.TestCase):
    def test_openai_responses_adapter_builds_payload_and_extracts_text(self) -> None:
        transport = CapturingTransport({"output_text": '{"type":"pass"}'})
        client = OpenAIResponsesModelClient("key", transport=transport)

        response = client.complete(_request())

        self.assertEqual(response.content, '{"type":"pass"}')
        url, headers, payload, timeout_seconds = transport.calls[0]
        self.assertEqual(url, "https://api.openai.com/v1/responses")
        self.assertEqual(headers["Authorization"], "Bearer key")
        self.assertEqual(payload["instructions"], "system")
        self.assertEqual(payload["input"], "user")
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertEqual(timeout_seconds, 4.0)

    def test_claude_messages_adapter_builds_payload_and_extracts_text(self) -> None:
        transport = CapturingTransport({"content": [{"type": "text", "text": '{"type":"pass"}'}]})
        client = ClaudeMessagesModelClient("key", transport=transport)

        response = client.complete(_request())

        self.assertEqual(response.content, '{"type":"pass"}')
        _, headers, payload, _ = transport.calls[0]
        self.assertEqual(headers["x-api-key"], "key")
        self.assertEqual(payload["system"], "system")
        self.assertEqual(payload["messages"][0]["content"], "user")

    def test_doubao_adapter_builds_chat_completion_payload_and_extracts_text(self) -> None:
        transport = CapturingTransport({"choices": [{"message": {"content": '{"type":"pass"}'}}]})
        client = DoubaoChatModelClient("key", transport=transport)

        response = client.complete(_request())

        self.assertEqual(response.content, '{"type":"pass"}')
        _, headers, payload, _ = transport.calls[0]
        self.assertEqual(headers["Authorization"], "Bearer key")
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")

    def test_glm_adapter_builds_chat_completion_payload_and_extracts_text(self) -> None:
        transport = CapturingTransport({"choices": [{"message": {"content": '{"type":"pass"}'}}]})
        client = GlmChatModelClient("key", transport=transport)

        response = client.complete(_request())

        self.assertEqual(response.content, '{"type":"pass"}')
        url, headers, payload, timeout_seconds = transport.calls[0]
        self.assertEqual(url, "https://open.bigmodel.cn/api/paas/v4/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer key")
        self.assertEqual(payload["model"], "model-a")
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["max_tokens"], 120)
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(timeout_seconds, 4.0)

    def test_codex_cli_adapter_uses_signed_in_cli_without_api_key(self) -> None:
        calls = []

        def runner(command, prompt, timeout_seconds, working_dir):
            calls.append((command, prompt, timeout_seconds, working_dir))
            return '{"type":"pass","thinking":"Use signed-in Codex."}'

        client = CodexCliModelClient(codex_binary="codex-test", working_dir="/tmp/project", runner=runner)

        response = client.complete(_request())

        self.assertEqual(response.content, '{"type":"pass","thinking":"Use signed-in Codex."}')
        command, prompt, timeout_seconds, working_dir = calls[0]
        self.assertEqual(command[:2], ["codex-test", "exec"])
        self.assertIn("--ephemeral", command)
        self.assertIn("--model", command)
        self.assertIn("model-a", command)
        self.assertIn("-c", command)
        self.assertIn('model_reasoning_effort="low"', command)
        self.assertIn("Return only the final JSON object", prompt)
        self.assertEqual(timeout_seconds, 4.0)
        self.assertEqual(str(working_dir), "/tmp/project")

    def test_model_backed_provider_parses_json_action(self) -> None:
        class FakeModel:
            def __init__(self):
                self.requests = []

            def complete(self, request):
                self.requests.append(request)
                return type("Response", (), {"content": '{"type":"pass","thinking":"Wait."}'})()

        model = FakeModel()
        provider = ModelBackedLlmProvider(model, model_name="model-a")

        action = provider.choose_action({"table_context": {"prompt_kind": "play_or_pass"}, "snapshot": {"hand": []}})

        self.assertEqual(action["type"], "pass")
        self.assertIn("Guandan MASTER level player", model.requests[0].system_prompt)
        self.assertIn("strategy_context", model.requests[0].user_prompt)
        self.assertNotIn('"prompt"', model.requests[0].user_prompt)
        self.assertNotIn("card_player", model.requests[0].user_prompt)
        self.assertNotIn('"skills"', model.requests[0].user_prompt)
        self.assertEqual(model.requests[0].model, "model-a")

    def test_model_backed_provider_routes_action_models_by_pressure(self) -> None:
        class FakeModel:
            def __init__(self):
                self.requests = []

            def complete(self, request):
                self.requests.append(request)
                return type("Response", (), {"content": '{"type":"pass"}'})()

        model = FakeModel()
        provider = ModelBackedLlmProvider(
            model,
            model_name="fast-model",
            fast_model=_resolved("fast", "fast-model", reasoning_effort="low"),
            pro_model=_resolved("pro", "pro-model", reasoning_effort="high"),
        )

        provider.choose_action({"table_context": {"prompt_kind": "play_or_pass"}, "strategy_context": {}})
        provider.choose_action(
            {
                "table_context": {"prompt_kind": "play_or_pass"},
                "strategy_context": {"pressure": {"endgame_defense": True}},
            }
        )
        provider.choose_action({"table_context": {"prompt_kind": "lead", "level_by_team": {"EW": "A"}}})

        self.assertEqual([request.model for request in model.requests], ["fast-model", "pro-model", "pro-model"])
        self.assertEqual(
            [request.model_reasoning_effort for request in model.requests],
            ["low", "high", "high"],
        )

    def test_model_backed_provider_runs_memory_prompt_with_memory_model(self) -> None:
        class FakeModel:
            def __init__(self):
                self.requests = []

            def complete(self, request):
                self.requests.append(request)
                return type("Response", (), {"content": '{"summary":"Learned.","techniques":["Keep tempo."]}'})()

        model = FakeModel()
        provider = ModelBackedLlmProvider(
            model,
            model_name="model-a",
            max_output_tokens=800,
            memory_model=_resolved("memory", "memory-model", max_output_tokens=1400),
        )

        result = provider.complete_memory(
            system_prompt=MEMORY_TECHNIQUE_SUMMARY_PROMPT,
            context={"deal_events": []},
            max_output_tokens=1200,
        )

        self.assertEqual(result["summary"], "Learned.")
        self.assertEqual(model.requests[0].system_prompt, MEMORY_TECHNIQUE_SUMMARY_PROMPT)
        self.assertIn("deal_events", model.requests[0].user_prompt)
        self.assertEqual(model.requests[0].model, "memory-model")
        self.assertEqual(model.requests[0].max_output_tokens, 1200)

    def test_model_backed_provider_audits_action_and_memory_completions(self) -> None:
        class FakeModel:
            def __init__(self):
                self.requests = []

            def complete(self, request):
                self.requests.append(request)
                if "deal_events" in request.user_prompt:
                    content = '{"summary":"Learned.","techniques":[]}'
                else:
                    content = '{"type":"pass"}'
                return type("Response", (), {"content": content, "raw": {"content": content}})()

        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "llm_completions.jsonl"
            provider = ModelBackedLlmProvider(FakeModel(), model_name="model-a", audit_log_path=audit_path)

            provider.choose_action(
                {"request_id": "r-1", "table_context": {"prompt_kind": "play_or_pass"}, "snapshot": {"hand": []}}
            )
            provider.complete_memory(
                system_prompt=MEMORY_TECHNIQUE_SUMMARY_PROMPT,
                context={"deal_events": []},
            )

            entries = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([entry["purpose"] for entry in entries], ["action", "memory"])
        self.assertEqual(entries[0]["metadata"]["request_id"], "r-1")
        self.assertEqual(entries[0]["request"]["model"], "model-a")
        self.assertEqual(entries[0]["request"]["model_role"], "fast")
        self.assertIn("started_at", entries[0]["timing"])
        self.assertIn("completed_at", entries[0]["timing"])
        self.assertIsInstance(entries[0]["timing"]["duration_ms"], float)
        self.assertGreaterEqual(entries[0]["timing"]["duration_ms"], 0.0)
        self.assertEqual(entries[0]["timestamp"], entries[0]["timing"]["completed_at"])
        self.assertIn("Guandan MASTER level player", entries[0]["request"]["system_prompt"])
        self.assertEqual(entries[0]["response"]["content"], '{"type":"pass"}')
        self.assertEqual(entries[1]["request"]["system_prompt"], MEMORY_TECHNIQUE_SUMMARY_PROMPT)
        self.assertEqual(entries[1]["request"]["model_role"], "memory")
        self.assertIn("duration_ms", entries[1]["timing"])
        self.assertEqual(entries[1]["response"]["raw"]["content"], '{"summary":"Learned.","techniques":[]}')

    def test_model_backed_provider_audits_failed_completion(self) -> None:
        class FailingModel:
            def complete(self, request):
                raise RuntimeError("model unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "llm_completions.jsonl"
            provider = ModelBackedLlmProvider(FailingModel(), model_name="model-a", audit_log_path=audit_path)

            with self.assertRaises(RuntimeError):
                provider.choose_action(
                    {"request_id": "r-1", "table_context": {"prompt_kind": "lead"}, "snapshot": {"hand": []}}
                )

            entry = json.loads(audit_path.read_text(encoding="utf-8"))

        self.assertEqual(entry["purpose"], "action")
        self.assertIn("timing", entry)
        self.assertGreaterEqual(entry["timing"]["duration_ms"], 0.0)
        self.assertEqual(entry["error"]["type"], "RuntimeError")
        self.assertEqual(entry["error"]["message"], "model unavailable")

    def test_provider_factory_allows_codex_cli_without_api_key(self) -> None:
        provider = provider_from_config(
            LlmAgentConfig(
                codex_binary="codex-test",
                play_fast=ModelSettings(provider_name="codex-cli", model_name="fast-model"),
                play_pro=ModelSettings(model_name="pro-model", model_reasoning_effort="high"),
                memory_model=ModelSettings(model_name="memory-model", max_output_tokens=1400),
            )
        )

        self.assertIsInstance(provider, ModelBackedLlmProvider)
        self.assertIsInstance(provider.model_client, CodexCliModelClient)
        self.assertEqual(provider.model_name, "fast-model")
        self.assertEqual(provider.fast_model.model_name, "fast-model")
        self.assertEqual(provider.pro_model.model_name, "pro-model")
        self.assertEqual(provider.pro_model.model_reasoning_effort, "high")
        self.assertEqual(provider.memory_model.model_name, "memory-model")
        self.assertEqual(provider.memory_model.max_output_tokens, 1400)
        self.assertEqual(provider.timeout_seconds, 40.0)
        self.assertEqual(provider.audit_log_path, Path("data/log") / "llm_completions.jsonl")

    def test_provider_factory_builds_glm_provider(self) -> None:
        provider = provider_from_config(LlmAgentConfig(play_fast=ModelSettings(provider_name="glm"), api_key="key"))

        self.assertIsInstance(provider, ModelBackedLlmProvider)
        self.assertIsInstance(provider.model_client, GlmChatModelClient)
        self.assertEqual(provider.model_name, "glm-5.1")

    def test_provider_factory_defaults_memory_model_to_pro_model(self) -> None:
        provider = provider_from_config(
            LlmAgentConfig(
                play_fast=ModelSettings(provider_name="codex-cli", model_name="fast-model"),
                play_pro=ModelSettings(model_name="pro-model"),
            )
        )

        self.assertEqual(provider.memory_model.model_name, "pro-model")
        self.assertEqual(provider.memory_model.role, "memory")


def _request() -> ModelRequest:
    return ModelRequest(
        system_prompt="system",
        user_prompt="user",
        model="model-a",
        temperature=0.4,
        timeout_seconds=4.0,
        max_output_tokens=120,
        model_reasoning_effort="low",
    )


def _resolved(
    role: str,
    model_name: str,
    *,
    max_output_tokens: int = 800,
    reasoning_effort: str | None = None,
) -> ResolvedModelSettings:
    return ResolvedModelSettings(
        role=role,
        provider_name="test",
        api_base_url=None,
        model_name=model_name,
        temperature=0.2,
        timeout_seconds=40.0,
        max_output_tokens=max_output_tokens,
        model_reasoning_effort=reasoning_effort,
    )


if __name__ == "__main__":
    unittest.main()
