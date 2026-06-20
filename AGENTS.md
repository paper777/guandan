# Repository Guidelines

## Project Structure & Module Organization

Architecture notes live in `docs/guandan_server_tech_design.md`. Bot training notes live in
`docs/guandan_bot_training_plan.md`.

Source layout:

- `pyproject.toml`, `uv.lock`: root Python project and uv environment configuration.
- `client/`: shared CLI, Guandan table HTTP client, broker, and NPC protocol request types.
- `db/player/`: NPC player profiles, lineup selection, LLM config, and player statistics storage.
- `server/`: Python server package source.
- `server/domain/`: pure Guandan rules engine, state, commands, events, scoring, and tribute logic.
- `server/controllers/`: human, local bot, and external AI agent controller adapters.
- `server/api/`: FastAPI HTTP and WebSocket schemas/routes.
- `server/services/`: table actor, lobby, replay, and snapshot filtering.
- `server/persistence/`: SQLite models and repositories.
- `npc/`: external NPC policy implementations and policy server helpers.
- `training/`: in-process self-play environment, legal-action feature encoding, heuristic baseline, behavior cloning collection/training, and PPO training scaffolds.
- `tests/`: unit, property, and integration tests mirroring source modules.

Keep rule logic independent from web, SQLite, and wall-clock time.

## Build, Test, and Development Commands

Key commands:

- `python -m venv .venv && source .venv/bin/activate`: create and enter a local virtual environment.
- `pip install -e ".[dev]"`: install the server and NPC packages with development dependencies.
- `uv sync --dev`: create/update the uv-managed virtual environment from root `pyproject.toml` and `uv.lock`.
- `python3 -m unittest discover -s tests`: run the no-dependency server and NPC test suite.
- `uv run python -m unittest discover -s tests`: run server and NPC tests through the uv-managed environment.
- `pytest`: run the full test suite after installing `.[dev]`.
- `pytest tests/domain`: run rule-engine tests.
- `uv run guandan-server --reload`: run the local API server through the packaged entrypoint.
- `uv run uvicorn server.app.main:app --reload`: run the local API server directly with uvicorn.
- `python3 -m unittest discover -s tests/training`: run training pipeline tests.
- `uv run --extra train guandan-bc-collect data/bc/heuristic.jsonl --seed 1 --seed 2 --max-deals 1`: collect heuristic behavior-cloning samples.
- `uv run --extra train guandan-bc-train data/bc/heuristic.jsonl data/models/bc_ranker.pt --epochs 3 --device cuda`: train the behavior-cloning candidate ranker with CUDA.
- `uv run --extra train guandan-ppo-train data/models/ppo_actor_critic.pt --seed 1 --seed 2 --updates 10 --max-deals 1 --device cuda`: run self-play PPO training with CUDA.
- `nvidia-smi`: verify the local NVIDIA GPU and driver before CUDA training.

Document any command changes in this file when tooling is introduced.

## Coding Style & Naming Conventions

Use Python 3.12+ with 4-space indentation and type hints for public functions. Prefer dataclasses or Pydantic models for structured state. Use `snake_case` for modules, functions, variables, and fields; use `PascalCase` for classes and domain events.

Keep files focused. Rule parsing, comparison, reducer logic, controller adapters, and persistence should stay in separate modules.

## Testing Guidelines

Use `pytest` for all tests and Hypothesis for rule invariants. Name tests `test_*.py`, and mirror the source path where practical, for example `tests/domain/test_comparator.py`.

Prioritize tests for card conservation, legal hand parsing, bomb hierarchy, tribute flow, controller authorization, snapshot privacy, and replay determinism.

Training code should also test that generated legal actions are reducer-accepted, observation encoders do not expose opponent private hands, rollout rewards are attributed to the correct seats, and training CLIs can run without importing PyTorch unless the `train` extra is used.

## Commit & Pull Request Guidelines

There is no existing commit history yet. Use concise imperative commit messages, for example `Add hand comparator tests` or `Document controller protocol`.

Pull requests should include:

- A short summary of behavior or documentation changed.
- Linked issue or design note when relevant.
- Test results, or a clear note when tests were not run.
- API examples or screenshots only when user-facing behavior changes.

## Security & Configuration Tips

Do not expose another seat's private cards in logs, snapshots, bot prompts, or agent callbacks. Treat bots and external agents as untrusted command producers. SQLite should run with foreign keys enabled and WAL mode when persistence is implemented.

Training actor inputs must be limited to information equivalent to `SeatSnapshot`. It is acceptable for training-only critics, reward code, and diagnostics to inspect full in-process state, but that privileged state must not be written into BC samples, model prompts, NPC callbacks, or runtime bot decisions.
