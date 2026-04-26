# Godot Playable MVP UI Plan

## Summary

Build a desktop-first Godot 4.6 playable MVP under `ui/`. The client connects to the existing FastAPI Guandan server and supports create, join, fill remaining seats with default local bots, ready, start, play, pass, public table display, private hand display for the controlled seat, card selection, countdown display, and server rejection feedback.

The server remains authoritative. The Godot client renders snapshots and submits commands; it does not implement Guandan rule validation.

## Key Changes

- Add a Godot 4.6 project rooted at `ui/` using GDScript and built-in Godot HTTP networking.
- Use polling plus command responses for updates through the existing API:
  - `POST /tables`
  - `GET /tables`
  - `GET /tables/{table_id}`
  - `GET /tables/{table_id}/seats/{seat}/snapshot?controller_id=...`
  - `POST /tables/{table_id}/join-human`
  - `POST /tables/{table_id}/ready`
  - `POST /tables/{table_id}/start`
  - `POST /tables/{table_id}/play`
  - `POST /tables/{table_id}/pass`
- Store local session state in memory: `base_url`, `table_id`, `seat`, `player_id`, `controller_id`, selected card IDs, and latest snapshot.
- Render MVP cards from card IDs such as `D1-H-10` and `D2-BJ` as readable text/suit tiles; no card-art package is required.
- Add default local bots through the existing `POST /tables/{table_id}/join-local-bot` endpoint and mark each bot ready after it joins.

## UI Flow

- Connection/table screen:
  - Server URL input defaulting to `http://127.0.0.1:8000`.
  - Create table, refresh tables, and join table actions.
  - Seat picker for `E`, `S`, `W`, `N`.
  - Display name input.
  - Default-on option to fill the other three seats with local bots after the human joins.
- Lobby screen:
  - Four-seat layout showing occupied seats, names, player kinds, and controlled status.
  - Ready, start match, refresh, and fill-empty-seats-with-bots actions.
  - If public ready-state is not exposed by the server, rely on command response/rejection instead of inventing local truth.
- Game screen:
  - Table layout with the controlled player at the bottom.
  - Opponent seats show display name, hand count, current-turn marker, and finish-order marker.
  - Controlled hand shows selectable card tiles from private seat snapshot.
  - Actions: play selected cards, pass, refresh.
  - Disable play/pass unless `legal_action` permits acting; disable pass while leading.
  - Show countdown from `action_deadline_epoch_ms` when available.
  - Show recent accepted events and rejection messages.

## Implementation Notes

- Godot structure:
  - `ui/project.godot`
  - `ui/scenes/Main.tscn`
  - `ui/scenes/TableSelect.tscn`
  - `ui/scenes/Lobby.tscn`
  - `ui/scenes/GameTable.tscn`
  - `ui/scenes/CardTile.tscn`
  - `ui/scripts/ApiClient.gd`
  - `ui/scripts/SessionState.gd`
- `ApiClient.gd` wraps HTTP JSON requests and normalizes success/error payloads.
- `SessionState.gd` owns local identity and latest snapshots.
- Poll the private seat snapshot every 1 second while joined to a table, and refresh immediately after command responses.
- Do not extend WebSocket broadcasting, add login, persistence, replay browsing, card art, or advanced animations in this MVP.

## Test Plan

- Run existing server tests with `uv run python -m unittest discover -s server/tests`.
- Start the server with `uv run guandan-server --reload`.
- Open the Godot project from `ui/` and verify:
  - A client can create or join a table.
  - A human can join a chosen seat and receive `player_id` and `controller_id`.
  - Four seats can be filled, readied, and started.
  - The controlled seat sees its private hand.
  - Other seats show only hand counts.
  - Selected cards can be played when legal.
  - Pass works only when allowed.
  - Server rejections are visible.
  - Polling reflects turn changes, hand counts, finish order, and countdown fields.

## Assumptions

- Target platform is desktop first.
- Use Godot 4.6 with no external plugins.
- The project root is exactly `ui/`.
- The server remains the only authority for Guandan rules and hidden information.
