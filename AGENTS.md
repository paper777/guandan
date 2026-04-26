# Repository Guidelines

## Project Structure & Module Organization

Architecture notes live in `docs/guandan_server_tech_design.md`.

Planned source layout:

- `server/`: Python server project root and package configuration.
- `server/guandan/domain/`: pure Guandan rules engine, state, commands, events, scoring, and tribute logic.
- `server/guandan/controllers/`: human, local bot, and external AI agent controller adapters.
- `server/guandan/api/`: FastAPI HTTP and WebSocket schemas/routes.
- `server/guandan/services/`: table actor, lobby, replay, and snapshot filtering.
- `server/guandan/persistence/`: SQLite models and repositories.
- `server/tests/`: unit, property, and integration tests mirroring source modules.

Keep rule logic independent from web, SQLite, and wall-clock time.

## Build, Test, and Development Commands

Key commands:

- `python -m venv .venv && source .venv/bin/activate`: create and enter a local virtual environment.
- `pip install -e "server[dev]"`: install the server package with development dependencies.
- `cd server && python3 -m unittest discover -s tests`: run the no-dependency test suite.
- `pytest`: run the full test suite after installing `server[dev]`.
- `pytest server/tests/domain`: run rule-engine tests.
- `cd server && uvicorn guandan.app.main:app --reload`: run the local API server.

Document any command changes in this file when tooling is introduced.

## Coding Style & Naming Conventions

Use Python 3.12+ with 4-space indentation and type hints for public functions. Prefer dataclasses or Pydantic models for structured state. Use `snake_case` for modules, functions, variables, and fields; use `PascalCase` for classes and domain events.

Keep files focused. Rule parsing, comparison, reducer logic, controller adapters, and persistence should stay in separate modules.

## Testing Guidelines

Use `pytest` for all tests and Hypothesis for rule invariants. Name tests `test_*.py`, and mirror the source path where practical, for example `server/tests/domain/test_comparator.py`.

Prioritize tests for card conservation, legal hand parsing, bomb hierarchy, tribute flow, controller authorization, snapshot privacy, and replay determinism.

## Commit & Pull Request Guidelines

There is no existing commit history yet. Use concise imperative commit messages, for example `Add hand comparator tests` or `Document controller protocol`.

Pull requests should include:

- A short summary of behavior or documentation changed.
- Linked issue or design note when relevant.
- Test results, or a clear note when tests were not run.
- API examples or screenshots only when user-facing behavior changes.

## Security & Configuration Tips

Do not expose another seat's private cards in logs, snapshots, bot prompts, or agent callbacks. Treat bots and external agents as untrusted command producers. SQLite should run with foreign keys enabled and WAL mode when persistence is implemented.
