# NPC Player Refactor Todo

## Summary

Refactor NPC players around a shared abstract `Player`, split the LLM agent into player, model, and prompt layers, integrate named default broker lineups for CLI play, and add a Codex skill for card recording.

## Todo

- [x] Add `npc/common/player.py` with abstract `Player`.
- [x] Make `DummyBotPolicy` inherit from `Player`.
- [x] Rename the LLM concrete class to `LlmAgentPlayer`.
- [x] Keep `LlmAgentPolicy = LlmAgentPlayer` for import compatibility.
- [x] Move all LLM prompt text into `npc/llm_agent/prompts.py`.
- [x] Explain Guandan seats, partners, turn roles, tribute roles, legal actions, and privacy constraints in the prompt text.
- [x] Add `npc/llm_agent/models.py` with `ModelClient`, `ModelRequest`, and `ModelResponse`.
- [x] Implement real stdlib HTTP adapters for OpenAI/Codex, Claude, and Doubao.
- [x] Keep deterministic LLM behavior available for offline tests and local dry runs.
- [x] Add broker helpers for default NPC lineups.
- [x] Define the mixed default lineup as `Ming` dummy bot plus `Jade`, `River`, and `Atlas` LLM agents.
- [x] Make the default lineup configurable through `npc/llm_agent/data/default_players.json`.
- [x] Support signed-in Codex CLI as `provider_name: "codex-cli"` without an API key.
- [x] Integrate a card-player advisor into `LlmAgentPlayer` prompts and fallback policy.
- [x] Add CLI `--npc-lineup` with `mixed`, `dummy`, and `llm` choices.
- [x] Create `skills/card-record/SKILL.md` for using `tools.card_recorder`.
- [x] Add tests for player inheritance, model adapters, prompts, broker lineups, CLI lineup selection, and skill metadata.
- [x] Run `uv run python -m unittest discover -s npc`.
- [x] Run `uv run python -m unittest discover -s tests`.

## Constraints

- Keep rule logic independent from web, SQLite, and wall-clock time.
- Do not add runtime dependencies for model adapters.
- Keep existing imports working, especially `from npc.llm_agent import LlmAgentPolicy`.
- Do not call external model APIs in tests.
