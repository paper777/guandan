# Client TUI Redesign

## Current Implementation Review

The existing `client/tui` implementation is a classic line-oriented CLI:

- `client/state_machine.py` owns the play loop and blocks on `read_command()`.
- `client/tui/render.py` renders Rich panels into plain text, then writes the text once per snapshot or command.
- `client/tui/commands.py` parses typed commands such as `play D-3 H-3`, `pass`, `tribute C3`, and `return C3`.

This design is reliable for tests and automation, but it is weak for live play:

- Mouse input is unavailable. The player must repeatedly type card labels or indices.
- The hand is text, not selectable state. It is easy to mistype duplicated physical cards.
- Player actions are printed as dense chronological lines. After several bot turns, it is hard to scan who acted, which cards were used, and which entries are only system transitions.
- Rendering and interaction are coupled to stdout. That prevents incremental refresh, timer updates, and mouse event handling.

The classic CLI remains useful and should stay available for scripts, snapshot printing, and tests.

## Design Goals

- Add a mouse-first TUI for human play using Textual and Rich.
- Keep the current CLI path stable for non-interactive use and existing tests.
- Reuse HTTP/session/bot-driving code instead of forking game rules.
- Make card selection explicit, visible, and deterministic.
- Split action history into structured fields: sequence, seat, action, cards, and detail.
- Preserve privacy boundaries: only the viewer's `SeatSnapshot` hand is rendered.

## Non-Goals

- This is not a rewrite of server rules, reducers, or controller authorization.
- This does not add WebSocket push. The Textual client still uses the existing HTTP client and refreshes after actions or timer expiry.
- This does not remove typed commands. `--ui classic` keeps the previous interface.

## Ten Improvement Rounds

1. Review the current TUI surface and identify the blocking input loop and dense event output as the main user pain.
2. Add a pure view-model layer in `client/tui/view_model.py` for cards, seats, tricks, action rows, and action availability.
3. Add tests for card selection state, tribute eligibility, structured action rows, seat markers, and active tricks.
4. Extend `drive_bot_turns()` with an optional response hook so the new UI can receive structured bot action events while the classic CLI output remains unchanged.
5. Add `client/tui/textual_app.py` with a Textual app, Rich-rendered board panels, clickable card buttons, explicit action buttons, and a `DataTable` action feed.
6. Add `--ui auto|classic|textual` to `guandan-cli`. `auto` selects Textual only for a real terminal; injected test input stays on the classic path.
7. Add Textual to project dependencies and update `uv.lock`.
8. Smoke-test the Textual app headlessly with the existing fake client to catch compose/query/click regressions.
9. Fix selected-card submission order so set-based UI state is submitted in visible hand order.
10. Document the architecture, controls, verification, and remaining backlog here.

## New Architecture

`client/tui/view_model.py`

- Converts public snapshots and seat snapshots into small immutable dataclasses.
- Builds `CardView` rows with `selected` and `eligible` state.
- Builds `ActionRow` records from server events.
- Computes action button availability from the legal prompt and selected cards.

`client/tui/textual_app.py`

- Owns the terminal UI and imports Textual directly.
- Uses stable card button slots instead of remounting widgets on every refresh.
- Runs blocking HTTP/session work in Textual workers.
- Uses the existing `prepare_default_table()`, `drive_bot_turns()`, and role observer hooks.
- Shows player actions in a table instead of one unstructured log stream.

`client/app.py`

- Adds the `--ui` option.
- Lazy-imports Textual mode only when selected.
- Keeps snapshot and classic play behavior intact.

## Controls

- Click a card to select or unselect it.
- `Play` submits selected cards for `lead` and `play_or_pass`.
- `Pass` submits a pass when the prompt allows it.
- `Tribute` and `Return` submit exactly one selected eligible card.
- `Clear` removes the current selection.
- `Refresh` reloads the table state.
- Keyboard bindings: `Enter` submits the primary selected action, `Space` passes, `c` clears, `r` refreshes, `q` quits.

## Width Budget

The Textual layout targets an 80-column terminal as the minimum practical viewport:

- The body is a full-width vertical stack, not two fixed-width side-by-side panes.
- The hand uses 8 card columns. A maximum 36-card hand fits in 5 rows.
- The action bar uses 6 columns, so `Play`, `Pass`, `Tribute`, `Return`, `Clear`, and `Refresh` fit on one row.
- The feed reserves 70 data columns across `Seq`, `Seat`, `Action`, `Cards`, and `Detail`, leaving room for table chrome.
- A mounted 80x40 geometry test checks that a full hand, every action button, and the feed stay inside the initial viewport.

## Action Feed

The action feed is a `DataTable` with columns:

- `Seq`: server event sequence.
- `Seat`: acting or relevant seat.
- `Action`: normalized event name such as `played single`, `passed`, `paid tribute`, or `deal ended`.
- `Cards`: readable card labels.
- `Detail`: last trick, next leader, position, or level movement.

This makes bot bursts easier to scan because each row has the same shape.

## Remaining Backlog

- Add a WebSocket-backed live refresh path once the client has a stable subscription API.
- Add richer legal-action hints, such as grouping selected cards by inferred hand type before submit.
- Add a compact mobile/narrow-terminal layout.
- Add persisted user preferences for sorting, color theme, and feed order.
- Add screenshot-based TUI regression tests if the project adopts a terminal rendering test harness.
