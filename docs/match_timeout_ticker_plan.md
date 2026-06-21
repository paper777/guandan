# Match Timeout Ticker Design Plan

## Summary

Add a per-table match timeout ticker with a default 180-second action deadline. The ticker is owned by the table actor, not the domain reducer, so rule logic stays deterministic and independent from wall-clock time.

The ticker applies whenever a seat is expected to submit an action:

- leading or responding during `PLAYING`;
- submitting tribute during `TRIBUTE`;
- returning tribute during `TRIBUTE`;
- future action-prompt types added through the same actor path.

When the deadline expires, the actor applies a configured fallback command through the same reducer validation path used by humans, bots, and agents. The first implementation should default to `auto_pass`: pass when legal, otherwise play the smallest legal leading hand or smallest legal required tribute/return fallback.

## Goals

- Enforce a default 180-second action deadline per table match.
- Expose enough ticker state for clients to render a countdown.
- Keep timeout behavior auditable and replay-compatible.
- Preserve reducer purity by keeping all clock reads, sleeps, and cancellation in `TableActor`.
- Support humans, local bots, and external agents through one prompt/deadline mechanism.

## Non-Goals

- No chess-clock, bank-time, increment, pause, or tournament penalty system in the first version.
- No client-side authoritative timeout decisions. Clients may display a countdown, but the server decides expiry.
- No process-distributed timer coordination. Phase one remains single-process; persisted events let restart rebuild match state, but in-flight deadlines restart from fresh actor policy unless later persistence stores deadline anchors.

## Public Interface Changes

### Table Configuration

Add table-level timeout config:

```python
TableConfig:
  action_timeout_seconds: int = 180
  timeout_fallback: TimeoutFallback = "auto_pass"
```

Validation:

- `action_timeout_seconds` must be positive.
- Recommended first bound: `5 <= action_timeout_seconds <= 300`.
- Existing tables without explicit config use `180`.

HTTP table creation should accept optional config:

```json
{
  "action_timeout_seconds": 180,
  "timeout_fallback": "auto_pass"
}
```

For compatibility, `POST /tables` with an empty body continues to create a table with the default 180-second timeout.

### Snapshot Fields

Expose ticker state in public snapshots because current turn and deadline are public game information:

```python
PublicTableSnapshot:
  action_deadline_epoch_ms: int | None
  action_timeout_seconds: int
  acting_seat: Seat | None
```

`acting_seat` should match the required actor for the current prompt. In normal play this is the same as `current_turn`; during tribute it may identify the giver or receiver currently required to act.

Clients compute display countdown as:

```text
max(0, action_deadline_epoch_ms - client_now_epoch_ms)
```

The server should not send high-frequency countdown messages only for ticking. It broadcasts a fresh snapshot when the prompt changes, when a command is accepted, and when timeout fallback fires.

### Events

Add auditable events:

- `ActionPrompted`: emitted when a new seat action deadline starts.
- `ActionTimedOut`: emitted when the actor observes deadline expiry.
- `TimeoutFallbackApplied`: emitted when a fallback command is accepted.

Recommended payloads:

```json
{
  "type": "ActionPrompted",
  "payload": {
    "seat": "E",
    "kind": "lead",
    "deadline_epoch_ms": 1710000000000,
    "timeout_seconds": 180
  }
}
```

```json
{
  "type": "ActionTimedOut",
  "payload": {
    "seat": "E",
    "kind": "lead",
    "deadline_epoch_ms": 1710000000000
  }
}
```

```json
{
  "type": "TimeoutFallbackApplied",
  "payload": {
    "seat": "E",
    "kind": "lead",
    "fallback": "auto_pass",
    "command_type": "PlayCards",
    "event_seq_range": [44, 45]
  }
}
```

Implementation note: if event batching becomes awkward, `ActionTimedOut` and `TimeoutFallbackApplied` may be emitted by the actor as service/audit events around the reducer-generated fallback events. They still need stable sequence numbers if persisted in the match event log.

### Error Model

Add or standardize rejection codes:

- `ACTION_TIMEOUT`: command arrived after the actor had already advanced the prompt by timeout.
- `STALE_ACTION_PROMPT`: command references an old prompt ID or sequence.
- `INVALID_TIMEOUT_CONFIG`: table creation or update supplied invalid timeout settings.

Late commands should usually reject as `NOT_YOUR_TURN` after fallback advances the turn. `ACTION_TIMEOUT` is useful when the command includes a prompt ID that is known to have expired.

## Actor Design

### Prompt State

Add actor-owned runtime prompt state:

```python
ActivePrompt:
  prompt_id: str
  seat: Seat
  kind: "lead" | "play_or_pass" | "tribute" | "return_tribute"
  started_epoch_ms: int
  deadline_epoch_ms: int
  state_seq: int
  timeout_task: asyncio.Task | None
```

This state is not part of the domain reducer. The actor updates it after accepted commands by inspecting the new authoritative `MatchState`.

### Clock Injection

`TableActor` should accept injectable clock/scheduler dependencies for tests:

```python
Clock:
  now_epoch_ms() -> int

Sleeper:
  sleep(delay_seconds: float) -> Awaitable[None]
```

Production uses system time and `asyncio.sleep`. Tests use a fake clock or explicit short sleeps.

### Prompt Lifecycle

After every accepted state mutation:

1. Determine whether the new state requires an action prompt.
2. If no prompt is required, cancel any existing timeout task.
3. If the required prompt is unchanged, keep the existing deadline.
4. If the required prompt changed, cancel the old timeout task and start a new 180-second deadline.
5. Broadcast the fresh snapshot containing the new `action_deadline_epoch_ms`.

Prompt identity should include at least `(phase, acting_seat, event_seq, prompt_kind)` so duplicate broadcasts do not reset the timer.

### Timeout Execution

When the timeout task wakes:

1. Re-enter the table actor lock or queue a synthetic actor command so timeout handling is serialized with real commands.
2. Confirm the active prompt still matches the task's prompt ID.
3. Emit `ActionTimedOut`.
4. Build the fallback command.
5. Dispatch the fallback through the reducer.
6. Emit or persist `TimeoutFallbackApplied` if the fallback is accepted.
7. Recompute the next prompt and schedule its deadline.

If a human command arrives at the same time as the timeout, actor serialization decides the winner. The command that acquires the actor first is authoritative.

### Fallback Policy

Default fallback is `auto_pass`:

- For `play_or_pass`: submit `Pass`.
- For `lead`: submit `PlayCards` with the smallest legal single card in the hand.
- For `tribute`: submit the highest eligible tribute card required by the tribute rule.
- For `return_tribute`: submit the smallest allowed return card; when returning to a partner, choose the smallest card with rank `10` or lower.

The fallback must use the timed-out seat's currently attached controller ID when available. If no controller is attached, use an internal timeout controller identity with only the capability required to submit the fallback, and audit that the command came from timeout automation.

Rejected fallback commands are a service bug or incomplete fallback policy. They should emit a structured error log and leave the table in a paused/error state only if no safe fallback exists.

## API and Client Behavior

### HTTP

Update table creation schemas:

- `TableCreateRequest(action_timeout_seconds: int = 180, timeout_fallback: str = "auto_pass")`
- `TableCreateResponse(table_id: str, action_timeout_seconds: int, timeout_fallback: str)`

Existing clients that post `{}` or no body remain valid.

### WebSocket

Snapshots include deadline fields. Clients should render countdown locally and refresh from server snapshots after every event or reconnect.

Commands may optionally include `prompt_id` in the future. The first implementation can omit prompt IDs and rely on current reducer validation. Add prompt IDs only when stale-action UX becomes a real issue.

### CLI

The CLI should display countdown metadata when available:

```text
Turn: E
Timer: 32s remaining
```

The CLI remains a client. It should not auto-timeout human actions locally.

## Persistence and Replay

Persist timeout audit events in `match_events` once timeout automation exists. This makes replay explain why a pass or forced lead happened.

Replay should rebuild card/game state from reducer events as usual. Timeout audit events do not need to mutate domain state, but replay views should show them in the public event timeline.

Phase-one restart behavior:

- Rebuild match state from persisted events.
- Recompute the active prompt.
- Start a fresh 180-second deadline from actor startup time.

Future stricter behavior may persist `ActionPrompted.deadline_epoch_ms` and resume the remaining time after restart.

## Test Plan

### Unit Tests

- `TableConfig` defaults to 180 seconds.
- Invalid timeout config is rejected.
- Prompt detection returns the expected acting seat and kind for `lead`, `play_or_pass`, `tribute`, and `return_tribute`.
- `auto_pass` builds `Pass` for `play_or_pass`.
- `auto_pass` builds the smallest legal leading single for `lead`.
- Tribute and return fallback choose legal cards.

### Service Tests

- Starting a match schedules an `ActionPrompted` deadline for the first leader.
- Accepted human command before timeout cancels the old timer and schedules the next prompt.
- Timeout fires exactly once for a prompt.
- Late timeout task is ignored after the prompt changes.
- Simultaneous command and timeout are serialized; only one action mutates state.
- Timeout fallback events are persisted with the fallback command events.

### API Tests

- `POST /tables` with no body returns default 180-second timeout config.
- `POST /tables` accepts a custom timeout within bounds.
- Public and private snapshots include the deadline fields.
- WebSocket snapshot after reconnect includes current deadline fields.

### CLI Tests

- Snapshot formatting shows remaining time when deadline fields are present.
- CLI does not submit timeout fallback itself for a human prompt.

## Implementation Plan

1. Add table config models and default timeout settings.
2. Extend snapshots and schemas with public deadline fields.
3. Add actor prompt detection and runtime `ActivePrompt`.
4. Add scheduler/clock injection and timeout task lifecycle.
5. Implement `auto_pass` fallback command construction.
6. Persist and broadcast timeout audit events.
7. Update HTTP/WebSocket/CLI display surfaces.
8. Add the tests listed above, starting with actor service tests.

## Open Review Questions

- Should timeout config be immutable after table creation for v1?
- Should `ActionPrompted` be persisted immediately, or only broadcast until timeout automation is implemented?
- Should disconnected human seats use `auto_pass` after 180 seconds, or should a separate reconnect grace policy override table timeout?
- Do we want prompt IDs in client command payloads now, or defer until stale-action UX needs them?
