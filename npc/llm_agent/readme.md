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

Model providers may include diagnostic fields such as `thinking`, `role`, `candidates`, and `recommended_action`. These fields are authored by the LLM and written to the decision log only; `LlmAgentPlayer.choose_action()` returns only the validated broker action. If model output is invalid, the agent falls back directly to the conservative dummy bot policy.

Memory stores long-term lessons under `techniques`, not `skills`. `techniques.level1` keeps recent per-deal technique summaries. When the serialized L1 content exceeds `memory_compaction_char_limit` (default `16000` characters), the memory sub-agent compacts it into `techniques.level2` categories:

- `team_coordination`
- `bomb_usage`
- `offensive_card_formation`
- `defensive_card_formation`
- `combo_removal`
- `others`

After each deal ends, the same memory sub-agent also updates `player_profiles` by player display name, not by seat. Each profile keeps the latest known seat plus inferred personality and playing style from public observations.

LLM players also receive a `personality` profile in the prompt. Supported defaults are `aggressive`, `balanced`, and `defensive`; the profile influences risk tolerance, tempo bias, bomb usage, passing bias, and structure preservation while keeping all actions legal.

Broker players are configured in `data/players.json`. To use the signed-in Codex CLI instead of an API key, set an LLM player's provider to `codex-cli`:

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
