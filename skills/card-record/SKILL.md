---
name: card-record
description: Use when tracking visible Guandan cards with the repository's tools.card_recorder module, including match start, turn recording, unseen-card tracking, and match finish handling.
---

# Card Record

Use `tools.card_recorder.CardRecorder` when the task needs a durable or testable record of cards that have become public during a Guandan match.

Core workflow:

1. Import `CardRecorder` and `CARD_BY_ID`; import `Seat` for seat-safe calls.
2. Call `recorder.start_match(match_id)` before recording turns.
3. For each visible play, convert card IDs to `Card` objects with `CARD_BY_ID` and call `recorder.turn(seat, cards)`.
4. Read `recorder.current_match.seen_cards` and `recorder.current_match.unseen_cards` for card-record state.
5. Call `recorder.finish_match()` when the match ends.

Rules:

- Record only cards that are visible from public events or the acting player's submitted action.
- Never infer hidden cards from hand counts, model guesses, or private snapshots for another seat.
- Do not record passes as card turns.
- Let `CardRecorder` raise on duplicate, unknown, or already-seen cards; do not silently ignore those errors unless the user explicitly asks for best-effort behavior.
