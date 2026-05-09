# LLM Agent Player

`npc.llm_agent` provides a broker-compatible Guandan NPC policy with per-player filesystem memory.

Example:

```python
from npc.llm_agent import LlmAgentConfig, LlmAgentPolicy

broker.add_seat(
    "S",
    LlmAgentPolicy(LlmAgentConfig(player_name="South Agent", seat="S", storage_dir="data")),
    display_name="South Agent",
)
```

Each `LlmAgentPolicy` instance owns its own memory, action log, memory sub-agent, and provider context. By default, storage is namespaced by player name so memory follows the player if their seat changes:

- `data/South-Agent/memory.json`
- `data/South-Agent/actions.json`

Both paths are configurable with `memory_path` and `action_log_path`.

The default provider is deterministic and dependency-free. It returns legal conservative broker actions for local dry runs. Custom providers can implement `choose_action(prompt) -> dict` and return normal broker actions such as `play_cards`, `pass`, `submit_tribute`, or `return_tribute`.

Model providers may include diagnostic fields such as `thinking`, `role`, `candidates`, and `recommended_action`. These fields are authored by the LLM and written to the decision log only; `LlmAgentPlayer.choose_action()` returns only the validated broker action. If model output is invalid, the agent falls back directly to the conservative dummy bot player.

Memory stores long-term lessons under `techniques`, not `skills`. `techniques.level1` keeps recent per-deal technique summaries. When the serialized L1 content exceeds `memory_compaction_char_limit` (default `16000` characters), the memory sub-agent compacts it into `techniques.level2` categories:

- `team_coordination`
- `bomb_usage`
- `offensive_card_formation`
- `defensive_card_formation`
- `combo_removal`
- `others`

After each deal ends, the same memory sub-agent also updates `player_profiles` by player display name, not by seat. Each profile keeps the latest known seat plus inferred personality and playing style from public observations. Player memory does not store score counters; player database statistics remain in each player's `statistics.json`.

LLM players also receive a `personality` profile in the prompt. Supported defaults are `aggressive`, `balanced`, and `defensive`; the profile influences risk tolerance, tempo bias, bomb usage, passing bias, and structure preservation while keeping all actions legal.

Broker players are indexed by `data/players.json`, with one directory per player under `data/`. Seat assignment is runtime state and is not stored in `players.json` or in a player's `profile.json`:

```json
{
  "players": [
    "Jade"
  ]
}
```

To use the signed-in Codex CLI instead of an API key, set an LLM player's provider to `codex-cli` in that player's `llm_config.json`:

```json
{
  "provider_name": "codex-cli",
  "play": {
    "fast": {"model_name": "gpt-5.4-mini", "model_reasoning_effort": "low"},
    "pro": {"model_name": "gpt-5.4", "model_reasoning_effort": "high"}
  },
  "memory": {
    "model_name": "gpt-5.4",
    "compaction_char_limit": 16000,
    "recent_deal_scan_limit": 200,
    "max_output_tokens": 1200
  }
}
```

`play.fast` is used for normal action decisions. `play.pro` is used when the prompt has endgame pressure, partner-near-finish pressure, or ace-level stakes. `memory` is used for deal summaries, player analysis, and technique compaction. For GPT/Codex providers, `model_reasoning_effort` is forwarded to the model client. The older flat keys (`model_name`, `temperature`, `timeout_seconds`, `max_output_tokens`, and `memory_*` limits) are still accepted as aliases.

Each player directory has this layout:

```text
data/Jade/
  actions.json
  llm_config.json
  memory.json
  profile.json
  statistics.json
```

This runs `codex exec` through the local signed-in Codex session. API-key providers remain available with `provider_name` values such as `openai`, `claude`, `doubao`, and `glm`. The GLM provider uses BigModel chat completions by default:

```json
{
  "provider_name": "glm",
  "play": {
    "fast": {"model_name": "glm-5.1"},
    "pro": {"model_name": "glm-5.1"}
  }
}
```

Set `BIGMODEL_API_KEY` for GLM, or provide `api_key` directly in the LLM config.
