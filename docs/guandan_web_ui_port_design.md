# Guandan Web UI Port Design

## Goal

Port the useful parts of `resource/rlcard-showdown` into Guandan as a browser UI that runs from the
existing Python server. The first usable screen is a live table, not a marketing or leaderboard page.

The port keeps the RLCard Showdown table feel: a fixed tabletop, visible playing cards, current-turn
focus, quick player setup, and compact command controls. The runtime is Guandan-native: it talks only
to the existing table HTTP endpoints and honors the server snapshot privacy model.

## Source UI Inventory

`resource/rlcard-showdown` contains:

- React 16 app shell with navbar, leaderboard, replay routes, and a Doudizhu PvE demo.
- `DoudizhuGameBoard` tabletop layout with seat areas, card selection, pass/play controls, and timer
  imagery.
- CSS playing-card and table assets under `src/assets`.
- A separate Django leaderboard server and a DouZero demo API.

Only the tabletop interaction pattern and visual assets are portable. The leaderboard, replay pages,
Doudizhu move logic, socket.io client, Django server, and DouZero endpoints do not map to Guandan's
current server.

## Target Architecture

The Guandan UI is a no-build static SPA:

- `server/web_ui/index.html`: app shell.
- `server/web_ui/styles.css`: responsive tabletop and controls.
- `server/web_ui/app.js`: API client, table state, rendering, command dispatch, and bot automation.
- `server/web_ui/assets/showdown/`: copied RLCard Showdown table/timer assets.
- `server/web_ui/assets/cards/`: copied Guandan card PNGs.

The Python server exposes it at:

- `/`: redirect to `/ui/`.
- `/ui/`: SPA index.
- `/ui/*`: static UI assets.

The FastAPI factory registers the static UI. The minimal ASGI fallback also serves the same files so
`server.app.main:app` keeps a usable UI path if FastAPI is unavailable.

## API Mapping

The UI uses the existing endpoints without adding game commands:

- `POST /tables`: create a table.
- `GET /tables`: list tables.
- `GET /tables/{table_id}`: public snapshot for all observers.
- `POST /tables/{table_id}/join-human`: attach a browser-owned human controller.
- `POST /tables/{table_id}/join-local-bot`: attach browser-owned bot controllers for quick matches.
- `POST /tables/{table_id}/ready`: mark each joined seat ready.
- `POST /tables/{table_id}/start`: start the first deal or next deal.
- `GET /tables/{table_id}/seats/{seat}/snapshot?controller_id=...`: fetch private state only for
  locally owned controllers.
- `POST /tables/{table_id}/play`: play selected cards.
- `POST /tables/{table_id}/pass`: pass.
- `POST /tables/{table_id}/tribute`: submit tribute.
- `POST /tables/{table_id}/return-tribute`: return tribute.

The current WebSocket route is request/response oriented and does not broadcast table changes, so the
initial UI uses short HTTP polling. A later iteration can switch snapshot refresh to WebSocket if the
server adds broadcast events.

## Privacy Model

Public snapshots render:

- seated players,
- hand counts,
- phase,
- current turn,
- current level,
- finish order,
- current trick's public cards.

Private hand cards are fetched and rendered only for the selected locally owned human seat. Bot
controllers created by the browser may fetch their private snapshots for automation, but their cards
are not displayed.

Controller IDs stay in browser local storage, keyed by table ID and seat. Losing local storage means
the browser becomes an observer until it joins another seat or creates a new table.

## Table UX

The table rotates around the selected local human seat:

- selected seat at bottom,
- next seat in counter-clockwise Guandan turn order on the right,
- partner/opposite seat at top,
- previous seat on the left.

The table shows stable seat zones with count badges. Non-selected seats use card backs and counts.
The selected human seat uses the real hand and card selection.

Control panel responsibilities:

- create or select a table,
- quick-start one human plus three browser-driven bot seats,
- join empty seats as human or bot,
- ready joined seats,
- start deals when the table is ready,
- play/pass/tribute/return for the active human seat,
- show recent public events and rejection messages.

## Bot Automation

Browser-created bot seats use a deliberately simple policy:

- `lead`: play the first sorted card as a single,
- `play_or_pass`: pass,
- `tribute`: submit the first eligible tribute card,
- `return_tribute`: return the first eligible return card.

All bot decisions are validated by the authoritative reducer through existing HTTP commands. The UI
does not attempt to reimplement Guandan hand legality.

## Non-Goals

- No new game rules or reducer behavior.
- No leaderboard, tournament upload, replay viewer, or model probability panel.
- No Node, React, Material UI, Sass, or frontend build step.
- No exposure of opponent private cards in public snapshots or visible UI.

## Test Plan

- Unit test `/` redirects to `/ui/`.
- Unit test `/ui/` serves HTML.
- Unit test `/ui/app.js` serves JavaScript.
- Unit test an unknown `/ui/*` path returns 404.
- Run the existing unittest suite.
- Launch `uv run guandan-server` and verify the UI route loads locally.
