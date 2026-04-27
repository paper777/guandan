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
