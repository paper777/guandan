# LLM Agent Player

`npc.llm_agent` provides a broker-compatible Guandan NPC policy with per-player filesystem memory.

Example:

```python
from npc.llm_agent import LlmAgentConfig, LlmAgentPolicy

broker.add_seat(
    "S",
    LlmAgentPolicy(LlmAgentConfig(player_name="South Agent", seat="S", storage_dir="npc/llm_agent/data")),
    display_name="South Agent",
)
```

Each `LlmAgentPolicy` instance owns its own memory, action log, and provider context. By default, storage is namespaced by seat:

- `npc/llm_agent/data/S/memory.json`
- `npc/llm_agent/data/S/actions.json`

Both paths are configurable with `memory_path` and `action_log_path`.

The default provider is deterministic and dependency-free. It returns legal conservative actions plus a concise `thinking` rationale, then stores that rationale with the decision log. Custom providers can implement `choose_action(prompt) -> dict` and return normal broker actions such as `play_cards`, `pass`, `submit_tribute`, or `return_tribute`.

`LlmAgentPlayer` includes a deterministic card-player advisor. Each model prompt receives `card_player.recommended_action` and `card_player.candidates`, so the LLM can choose from concrete policy options instead of reasoning only from raw cards. If model output is invalid, the agent falls back to the card-player recommendation before using the conservative dummy bot fallback.

LLM players also receive a `personality` profile in the prompt. Supported defaults are `aggressive`, `balanced`, and `defensive`; the profile influences risk tolerance, tempo bias, bomb usage, passing bias, and structure preservation while keeping all actions legal.

Default broker players are configured in `npc/llm_agent/data/default_players.json`. To use the signed-in Codex CLI instead of an API key, set an LLM player's provider to `codex-cli`:

```json
{
  "seat": "S",
  "display_name": "Jade",
  "kind": "llm",
  "personality": "aggressive",
  "provider_name": "codex-cli",
  "model_name": "gpt-5.2"
}
```

This runs `codex exec` through the local signed-in Codex session. API-key providers remain available with `provider_name` values such as `openai`, `claude`, and `doubao`.
