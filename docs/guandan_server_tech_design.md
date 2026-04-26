# Guandan Python Server Technical Design

## Status

Draft, based on the 2017 Huai'an Guandan competition rules page provided by the user: <http://cnmzppw.com/tv/20220301153936.html>.

The source page publishes 《掼蛋（国家）竞赛规则（2017版）》 / 《淮安掼蛋竞赛规则》. The first implementation will model the core online game rules and leave offline tournament administration, referee penalties, venue discipline, appeals, and table equipment rules out of runtime scope.

## Goals

- Build an authoritative Python server for four-player Guandan matches.
- Keep all rule decisions on the server: dealing, action validation, hand comparison, trick resolution, tribute, upgrades, scoring, and game end.
- Expose protocols suitable for human clients, built-in bots, and external AI agents.
- Make the rule engine deterministic, small, and heavily tested.
- Support later rule variants without rewriting table/session infrastructure.

## Non-Goals

- No real-money play, payment, or gambling workflow.
- No client UI in this phase.
- No tournament pairing, referee console, appeals, or manual penalty administration in the first version.
- No anti-cheat beyond server-authoritative hidden information, action validation, and audit logs.
- No AI bot strength optimization beyond a pluggable decision interface and simple baseline bots.

## Rule Summary From Source

Core rules to implement:

- A table has four players, two partnerships, with opposite seats on the same team.
- The game uses two standard decks, including jokers, for 108 cards total.
- Each player receives 27 cards.
- A match starts at level `2` and progresses through `3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, A+`.
- Each hand has a current level card. All cards of that rank are level cards; they rank above `A` and below small joker.
- The red-heart level cards are wild cards, also called `逢人配` / `红心参谋`. They may represent other non-joker cards in valid combinations.
- Play proceeds counter-clockwise.
- A leading player may lead any valid hand type. Other players may pass, beat with the same comparable type, or use an allowed bomb.
- A trick ends after three consecutive passes; the last player who played leads the next trick.
- If the first player out has an unbeatable final play, their partner borrows the lead (`借风`) if that partner is still active.
- A deal ends when three players have gone out, or immediately when one side takes first and second place (`双下`).
- Only the side containing the first player out upgrades.
- Upgrade count is based on the finishing position of the winner's partner:
  - Partner second: upgrade 3 levels.
  - Partner third: upgrade 2 levels.
  - Partner last: upgrade 1 level.
- Before later deals, the previous last-place player gives their highest eligible card to the previous winner; the receiver returns one card.
- On `双下`, both losing players tribute to the winning side.
- Tribute can be resisted (`抗贡`) when the losing side has the required big jokers, per the source rules.
- A player must report when their remaining hand reaches 10 cards or fewer. In the server this is a public automatic event, not a manual penalty flow.

Hand types to implement:

- Single card.
- Pair.
- Three consecutive pairs (`三连对` / `木板`), exactly three pairs.
- Three of a kind (`三同张`).
- Two consecutive triples (`三同连张` / `钢板`), exactly two triples.
- Full house style `三带对`: one triple plus one pair.
- Straight (`顺子`), exactly five consecutive ranks.
- Bomb: four or more cards of the same rank.
- Straight flush (`同花顺` / `火箭`), exactly five consecutive cards of the same suit.
- Four jokers (`四大天王` / `王炸`), the highest bomb.

Comparison rules to implement:

- Rank order for ordinary comparison is big joker, small joker, level rank, `A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2`.
- Same ordinary hand type compares by primary rank; `三带对` compares only the triple rank.
- Bombs can beat ordinary hand types.
- A larger bomb length beats any smaller bomb length.
- Same-length bombs compare by rank.
- Straight flush beats bombs with 5 or fewer cards.
- Bombs with 6 or more cards beat straight flushes.
- Four jokers beats every hand type.

The source contains tournament scoring and penalty rules. The server will preserve enough event/audit detail to add those later, but phase one will implement table-level gameplay scoring only.

## System Overview

The server is an authoritative state machine.

```text
Controllers                    Python Server
-----------                    -------------
HTTP table setup          ->   Lobby and controller registration
Human WebSocket actions   ->   Match actor / table state machine
Bot/agent command input   ->   Same match actor command queue
Seat snapshots            <-   Controller-filtered private views
Events                    <-   Public and private event stream
```

Recommended stack:

- Python 3.12+.
- FastAPI for HTTP and WebSocket endpoints.
- Pydantic v2 for request/response schemas.
- `asyncio` table actors: one serialized command queue per table.
- SQLite for users, tables, matches, replay metadata, events, and completed hand records.
- Pytest + Hypothesis for rule engine tests.

The rule engine must not depend on FastAPI, SQLite, or wall-clock time. It should be a pure Python package that accepts state plus command and returns new state plus events.

## Package Layout

```text
server/
  pyproject.toml
  guandan/
    api/
      http.py
      websocket.py
      schemas.py
    app/
      config.py
      main.py
    domain/
      cards.py
      hand_types.py
      hand_parser.py
      comparator.py
      state.py
      commands.py
      events.py
      reducer.py
      scoring.py
      tribute.py
    services/
      table_actor.py
      lobby.py
      replay_store.py
      snapshot_filter.py
    controllers/
      base.py
      human.py
      local_bot.py
      external_agent.py
      simple_policy.py
    persistence/
      models.py
      repositories.py
  tests/
    domain/
    api/
```

## Domain Model

### Card

Cards need stable physical identity because two decks contain duplicate rank/suit cards.

```python
CardId = str  # e.g. "D1-H-2", "D2-BJ"

Card:
  id: CardId
  deck: 1 | 2
  suit: "S" | "H" | "D" | "C" | None
  rank: "2" | "3" | ... | "A" | "SJ" | "BJ"
```

The server stores card IDs in player hands. Rule evaluation resolves IDs to card metadata through the current deck map.

### Seats and Teams

Seats are fixed as `E, S, W, N`. Counter-clockwise order is configured once, and opposite seats are partners:

- Team A: `E` and `W`.
- Team B: `S` and `N`.

The implementation should avoid hard-coding Chinese physical table assumptions into the engine. Seat order and partnership can be represented explicitly in `TableConfig`.

### Players and Controllers

A seat is occupied by a `PlayerRef`, but a seat is acted on by a `ControllerRef`. This distinction matters because the controller may be a human connection, an in-process bot, or an external AI agent.

```python
PlayerRef:
  id: str
  display_name: str
  kind: "human" | "bot" | "agent"

ControllerRef:
  id: str
  kind: "human_ws" | "local_bot" | "external_agent"
  seat: Seat
  player_id: str
  capabilities: set[ControllerCapability]
```

Controller capabilities:

- `PLAY`: may submit play/pass/tribute commands for its seat.
- `OBSERVE_PRIVATE`: may see the controlled seat's private hand.
- `OBSERVE_PUBLIC`: may see public table events.
- `AUTO_READY`: may mark ready without a UI prompt.
- `DEBUG_FULL_STATE`: development-only capability, never enabled in production matches.

Domain state should store only stable player identity and controller metadata needed for audit. Runtime controller instances live outside the reducer and communicate with the table actor through the same command queue as humans.

### Match State

```python
MatchState:
  table_id: str
  phase: MatchPhase
  level_by_team: dict[TeamId, Rank]
  current_level: Rank
  dealer_seed: str
  seats: dict[Seat, PlayerRef]
  controllers: dict[Seat, ControllerRef]
  deal: DealState | None
  scores: ScoreState
  event_seq: int
```

### Deal State

```python
DealState:
  hands: dict[Seat, tuple[CardId, ...]]
  active_seats: set[Seat]
  finish_order: list[Seat]
  leader: Seat
  turn: Seat
  current_trick: TrickState
  pass_count: int
  tribute: TributeState | None
  report_10_done: set[Seat]
```

### Trick State

```python
TrickState:
  lead_seat: Seat
  last_play: PlayedHand | None
  plays: list[TrickAction]
```

### Played Hand

```python
PlayedHand:
  cards: tuple[CardId, ...]
  type: HandType
  primary_rank: Rank
  length: int
  wild_assignments: tuple[WildAssignment, ...]
```

`wild_assignments` is required for red-heart level cards. Clients may submit an intended declaration, but the server must recompute and validate it.

## State Machine

### Phases

```text
WAITING_FOR_PLAYERS
READY_CHECK
DEALING
TRIBUTE
PLAYING
DEAL_COMPLETE
MATCH_COMPLETE
ABORTED
```

### Commands

Commands are the only way to mutate state:

- `JoinTable(player_id, controller_kind, requested_seat?)`
- `AttachController(player_id, seat, controller_ref)`
- `DetachController(controller_id, seat)`
- `LeaveTable(player_id)`
- `Ready(controller_id, seat)`
- `StartMatch(seed?)`
- `SubmitTribute(controller_id, seat, card_id)`
- `ReturnTribute(controller_id, seat, card_id)`
- `PlayCards(controller_id, seat, card_ids, declared_type?, wild_assignments?)`
- `Pass(controller_id, seat)`
- `RequestSnapshot(controller_id, seat?)`

Every command returns events or a structured rejection. Rejections are not state changes.

The reducer validates that the controller is currently attached to the submitted seat and has the required capability. This keeps rule logic identical for humans, bots, and agents.

### Events

Events are append-only and sequence-numbered:

- `TableCreated`
- `PlayerSeated`
- `ControllerAttached`
- `ControllerDetached`
- `MatchStarted`
- `DealStarted`
- `CardsDealt`
- `TributeRequired`
- `TributePaid`
- `TributeReturned`
- `TributeResisted`
- `CardsPlayed`
- `PlayerPassed`
- `TrickEnded`
- `PlayerFinished`
- `TenCardReport`
- `DealEnded`
- `LevelAdvanced`
- `MatchEnded`
- `CommandRejected`

For reconnects and replay, persist all public events plus private card-deal events. Client delivery must filter hidden card information.

## Controller Protocols

### Shared Controller Contract

All controller types use the same logical contract:

```python
class SeatController(Protocol):
    async def on_snapshot(self, snapshot: SeatSnapshot) -> None: ...
    async def on_event(self, event: VisibleEvent) -> None: ...
    async def request_action(self, prompt: ActionPrompt) -> ControllerCommand: ...
```

`ActionPrompt` is produced by the table actor when a seat must act. It contains:

- table ID, match ID, seat, current level, phase, sequence number;
- private hand for that seat only;
- legal action category: `play_or_pass`, `lead`, `tribute`, or `return_tribute`;
- current trick context and public finish order;
- optional server-computed legal action hints.

The domain reducer does not call controllers. The table actor owns prompting and timeout behavior.

### Human WebSocket Controller

Human clients connect through:

```text
GET /ws/tables/{table_id}
```

The connection receives seat-filtered snapshots and public events. It submits the same command schema as every other controller. A human can disconnect and later reattach to the same seat if authentication or guest token ownership matches.

### Local Bot Controller

Local bots run in the same Python process and do not use WebSocket. They subscribe to prompts from the table actor and enqueue commands directly.

Use cases:

- filling empty seats for development;
- deterministic scripted tests;
- simple baseline opponents;
- future stronger self-implemented policies.

Local bot policies should be deterministic when given a seed:

```python
BotPolicy:
  choose_action(snapshot: SeatSnapshot, prompt: ActionPrompt, rng: Random) -> ControllerCommand
```

The first policy can be intentionally simple: play the smallest legal hand when leading, beat with the smallest legal response when possible, otherwise pass.

### External AI Agent Controller

External AI agents are out-of-process controllers. They should use a separate agent protocol rather than pretending to be a browser client.

Recommended first protocol: outbound HTTP callback from the server to an agent endpoint.

```text
POST {agent_url}/guandan/action
```

Request:

```json
{
  "protocol_version": "1",
  "request_id": "uuid",
  "deadline_ms": 3000,
  "snapshot": {
    "table_id": "table_1",
    "seat": "E",
    "seq": 42,
    "phase": "PLAYING",
    "hand": ["D1-H-7", "D2-H-7"],
    "public_state": {}
  },
  "prompt": {
    "kind": "play_or_pass",
    "current_hand": {},
    "legal_action_hints": []
  }
}
```

Response:

```json
{
  "request_id": "uuid",
  "action": {
    "type": "play_cards",
    "card_ids": ["D1-H-7", "D2-H-7"],
    "declared_type": "pair",
    "wild_assignments": []
  }
}
```

The server must still validate the response as an ordinary controller command. Invalid, late, or unreachable agent responses become `CommandRejected` internally and should trigger the configured timeout fallback.

Agent protocol rules:

- Send only the controlled seat's private hand and public table information.
- Include `protocol_version` in every request.
- Include a deadline; late responses are ignored.
- Require a shared secret or signed request header per registered agent.
- Do not expose debug full state to external agents.
- Keep the action schema identical to WebSocket command payloads.

Timeout fallback is table-configurable:

- `auto_pass`: pass when legal, otherwise play the smallest legal leading hand.
- `simple_bot_takeover`: temporarily let a local bot act for the seat.
- `forfeit`: reserve for tournament mode, not phase one.

### Snapshot Models

Use separate models for public state and seat-private state:

```python
PublicTableSnapshot:
  table_id: str
  match_id: str
  phase: MatchPhase
  level_by_team: dict[TeamId, Rank]
  seats: dict[Seat, PublicPlayer]
  current_turn: Seat | None
  current_trick: PublicTrick
  finish_order: list[Seat]
  event_seq: int

SeatSnapshot:
  public: PublicTableSnapshot
  seat: Seat
  hand: tuple[CardId, ...]
  legal_action: ActionPrompt | None
```

Humans, local bots, and external agents all receive `SeatSnapshot`. Spectators receive only `PublicTableSnapshot`.

## Rule Engine Design

### Hand Parsing

`hand_parser.parse(cards, current_level, declaration=None) -> list[PlayedHand]`

The parser should return all legal interpretations for a selected card set. Some wild-card selections can be valid as more than one type, so the action is accepted only when either:

- exactly one interpretation exists, or
- the client declaration selects one legal interpretation.

Examples:

- Red-heart level card plus `8` can become a pair of `8`.
- Red-heart level card inside a straight must declare its represented rank and suit when relevant.
- Red-heart level cards cannot represent jokers.

The parser should be exhaustive but constrained by the maximum hand size. Avoid ad hoc string keys for combinations; return structured `PlayedHand` objects.

### Rank Ordering

Rank ordering is level-dependent. Implement a `RankContext`:

```python
RankContext:
  level: Rank
  compare_rank(a: Rank, b: Rank) -> int
  natural_sequence_index(rank: Rank) -> int
  is_level_card(card: Card) -> bool
  is_red_heart_level_card(card: Card) -> bool
```

Natural sequences differ from comparison rank. `A` and `2` have special sequence behavior for straights, connected pairs, and connected triples. Keep those rules in explicit sequence functions, not in the generic comparator.

### Hand Comparison

`comparator.can_beat(candidate, current, current_level) -> bool`

Algorithm:

1. If there is no current hand, candidate is legal.
2. If candidate is four jokers, it beats everything except no current hand comparison is needed.
3. If current is four jokers, nothing beats it.
4. If candidate and current are ordinary same-type hands, compare type, length if applicable, and primary rank.
5. If candidate is bomb-like and current is ordinary, apply bomb hierarchy.
6. If both are bomb-like:
   - four jokers wins;
   - 6+ card bomb beats straight flush;
   - straight flush beats bombs of length <= 5;
   - longer same-rank bombs beat shorter bombs;
   - same-length bombs compare rank;
   - same-type straight flush compares primary rank.

Bomb hierarchy should be encoded in one function and covered with table-driven tests.

### Turn Rules

On `PlayCards`:

1. Check phase is `PLAYING`.
2. Check the controller is attached to the current turn seat and has `PLAY`.
3. Check all card IDs are in the player's hand.
4. Parse and validate the selected cards.
5. Check candidate can beat current trick hand, unless leading.
6. Remove cards from hand.
7. Emit `CardsPlayed`.
8. If remaining cards are <= 10 and player has not reported before, emit `TenCardReport`.
9. If hand is empty, emit `PlayerFinished` and remove player from active seats.
10. Determine whether deal ends.
11. Otherwise determine next turn, skipping finished seats.

On `Pass`:

1. Reject if leading a new trick.
2. Increment pass count.
3. If all other active seats passed, end the trick and set next leader.
4. Otherwise advance turn.

### Borrowed Wind

When a player finishes and their last play is not beaten before the trick ends:

- If their partner is still active, the partner becomes next leader.
- If the partner is already finished or inactive, use normal next active seat fallback.

Model this as a `next_leader_after_trick()` helper so it is testable independently.

## Tribute Flow

At the start of a deal after the first:

1. Determine if normal tribute, double tribute, or tribute resistance applies from previous deal results and current hands.
2. If tribute is resisted, emit `TributeResisted` and set leader to previous winner.
3. Otherwise enter `TRIBUTE`.
4. Validate tribute cards:
   - normal tribute card must be the giver's highest eligible card, excluding red-heart level card as specified by the source rule.
   - double tribute maps higher tribute to previous winner and lower tribute to second finisher.
5. Validate return cards:
   - if returning to partner, returned rank must be `10` or lower.
   - if returning to opponent, any card is allowed.
6. Apply card movement atomically.
7. Set first leader according to the tribute rule.

Open detail for implementation: the source describes several double-tribute ordering cases. Encode these cases as unit tests before building the reducer branch.

## Scoring and Match End

Phase one should implement level progression:

```python
advance = {
  partner_finished_second: 3,
  partner_finished_third: 2,
  partner_finished_last: 1,
}
```

Rules:

- Only the first finisher's team advances.
- `A` must be played; a team cannot skip directly over `A`.
- A team wins the match only when it passes `A` under the source condition: first finisher's partner must not be last.

Persist both:

- compact level state for matchmaking and current table display;
- full deal result for replay and later tournament grade-point scoring.

Tournament score systems from the source, including field points and 26-point level-difference scoring, should be a later `scoring/tournament.py` module.

## API Design

### HTTP

```text
POST /tables
GET  /tables/{table_id}
POST /tables/{table_id}/join-human
POST /tables/{table_id}/join-local-bot
POST /tables/{table_id}/join-agent
POST /tables/{table_id}/leave
POST /tables/{table_id}/controllers/{controller_id}/detach
GET  /matches/{match_id}/replay
```

HTTP creates and discovers resources, registers non-human controllers, and manages controller attachment. Human gameplay uses WebSocket. Local bots and external AI agents use the controller protocol described above.

Example `join-agent` request:

```json
{
  "seat": "E",
  "display_name": "agent-alpha",
  "agent_url": "https://agent.example.com",
  "timeout_ms": 3000,
  "timeout_fallback": "simple_bot_takeover"
}
```

### WebSocket

```text
GET /ws/tables/{table_id}
```

Client to server messages:

```json
{
  "request_id": "uuid",
  "seat": "E",
  "type": "play_cards",
  "payload": {
    "card_ids": ["D1-H-7", "D2-H-7"],
    "declared_type": "pair",
    "wild_assignments": []
  }
}
```

Server to client messages:

```json
{
  "seq": 42,
  "type": "cards_played",
  "payload": {
    "seat": "E",
    "cards": ["D1-H-7", "D2-H-7"],
    "hand_type": "pair",
    "remaining_count": 18
  }
}
```

Snapshots are filtered per controller:

- Own hand includes exact card IDs.
- Other hands include counts only.
- Public trick, finish order, level, team scores, current turn, and tribute status are visible to everyone.
- Local bots and external agents receive the same private shape as a human controlling that seat.
- Spectators receive public snapshots only and cannot submit play commands.

## Persistence

Use one SQLite database file for the first implementation. SQLite is enough for a single-process server, local development, bots, replay testing, and early private deployment. Run the database in WAL mode so reads can continue while the actor appends events.

SQLite tables:

- `users`: account identity.
- `players`: stable player display identity for humans, bots, and agents.
- `controllers`: controller kind, seat attachment, capabilities, and agent callback config.
- `tables`: table configuration and lifecycle.
- `matches`: current and final match metadata.
- `match_events`: append-only event stream with sequence number, visibility, and payload.
- `deal_results`: compact indexed result records for stats.
- `idempotency_keys`: accepted command request IDs and their resulting event sequence range.

For active matches, keep state in memory inside the table actor. After every accepted command, append events to SQLite in a transaction before broadcasting. On process restart, rebuild state from events.

Recommended SQLite settings:

- `PRAGMA journal_mode = WAL`
- `PRAGMA foreign_keys = ON`
- `PRAGMA busy_timeout = 5000`

This design intentionally avoids Redis. Lobby state, presence, and active table lookup stay in process memory and are rebuilt from SQLite when the server starts. If the project later needs multiple server processes, that should be a separate scaling design rather than a hidden phase-one dependency.

## Concurrency Model

Use one actor per active table:

```text
Human WebSocket / local bot / external agent callback
  -> TableActor.queue
  -> reducer
  -> SQLite event store
  -> broadcaster / next controller prompt
```

Properties:

- Commands for a table are processed serially.
- No shared mutable match state outside the actor.
- Idempotency is handled with `(table_id, controller_id, request_id)`.
- Backpressure is per connection; slow clients receive snapshots after reconnect instead of blocking the actor.
- Local bot decisions must not run inside the reducer. If a bot policy is expensive, run it in a task with the same deadline behavior as an external agent.
- External AI agent callbacks are never allowed to block the table actor; the actor creates a prompt, waits up to the deadline, and applies the timeout fallback if needed.

## Validation and Error Model

Reject commands with stable machine-readable codes:

- `NOT_YOUR_TURN`
- `CONTROLLER_NOT_ATTACHED`
- `INSUFFICIENT_CONTROLLER_CAPABILITY`
- `INVALID_PHASE`
- `CARD_NOT_OWNED`
- `INVALID_HAND_TYPE`
- `AMBIGUOUS_WILD_CARD_DECLARATION`
- `DOES_NOT_BEAT_CURRENT_HAND`
- `CANNOT_PASS_WHEN_LEADING`
- `INVALID_TRIBUTE_CARD`
- `INVALID_RETURN_CARD`
- `AGENT_TIMEOUT`
- `AGENT_PROTOCOL_ERROR`

Rejections should include a short human-readable message and the latest authoritative sequence number.

## Testing Strategy

### Unit Tests

Focus on `domain/`:

- Deck generation has 108 unique physical cards.
- Dealing gives four 27-card hands.
- Level-card ordering for every level from `2` to `A`.
- Red-heart level card parsing for pair, triple, full house, straight, straight flush, connected pairs, connected triples, and bomb.
- Every legal hand type, including exact length constraints.
- Bomb hierarchy table.
- Same-type comparison.
- Trick end after three passes.
- Borrowed wind behavior.
- Finish order and `双下` early deal end.
- Normal tribute, double tribute, and tribute resistance.
- Upgrade rules, especially `A` and `A+`.
- Controller capability validation for each command type.

### Property Tests

Use Hypothesis for invariants:

- No card duplication or loss across deal, play, tribute, and return.
- A controller cannot play a card outside its attached seat's hand.
- Event replay always reconstructs the same state.
- Hidden snapshot never includes another seat's card IDs for human, local bot, or external agent controllers.
- Accepted play always reduces exactly the attached seat's hand by selected card count.
- A controller can mutate only its attached seat.

### Integration Tests

- Four websocket clients can complete a scripted deal.
- Mixed human, local bot, and external agent controllers can complete a scripted deal.
- Reconnect returns a correct filtered snapshot.
- Duplicate `request_id` does not apply an action twice.
- Agent timeout triggers the configured fallback.
- Invalid agent response is rejected and audited.
- Server restart can rebuild an active match from event log.

## Observability

Log every accepted command with:

- table ID,
- match ID,
- player ID,
- controller ID and controller kind,
- command type,
- prior sequence,
- resulting event sequence range,
- rejection code if rejected.

Metrics:

- active tables,
- websocket connections,
- local bot seats,
- external agent seats,
- command latency,
- controller decision latency by kind,
- reducer latency,
- rejected command count by code,
- SQLite event write latency,
- agent timeout count,
- reconnect count.

## Security and Fair Play

- Server shuffles and deals; clients never provide card order.
- Use cryptographically secure seed generation for production shuffles.
- Store shuffle seed only in private audit data until match completion.
- Filter hidden information at one service boundary.
- Validate every action against authoritative state.
- Rate-limit commands per connection, player, controller, and external agent endpoint.
- Sign or authenticate external agent callbacks.
- Treat local bots and external agents as untrusted command producers; they receive only seat-filtered snapshots.
- Audit controller attachment, detachment, timeout fallback, and takeover events.
- Keep replay/audit logs for dispute review.

## Implementation Plan

1. Create the pure domain model: cards, ranks, seats, teams, phases, commands, events, players, and controllers.
2. Implement deck generation, deterministic shuffle, deal, and public/seat snapshot filtering.
3. Implement hand parser without wild cards.
4. Add level-card rank context and red-heart wild-card parsing.
5. Implement hand comparison and bomb hierarchy.
6. Implement reducer for play/pass/trick/finish/deal-end flow.
7. Implement tribute and return flow.
8. Implement level progression and match completion.
9. Add controller protocol interfaces and a deterministic local bot policy.
10. Add FastAPI HTTP and human WebSocket shell.
11. Add table actor, in-memory repository, and scripted mixed-controller integration tests.
12. Add external AI agent callback protocol and timeout fallback.
13. Add SQLite event persistence and replay rebuild.
14. Add basic lobby and reconnect/reattach behavior.

## Open Questions

- Should the first product support only the 2017 Huai'an competition rules, or also common local variants?
- Should the first API support anonymous guest players, authenticated accounts, or both?
- Should table state be recoverable after every command from day one, or can persistence start after the pure engine is stable?
- Should 10-card reporting be purely automatic, or should clients display it as a visible player action?
- Should tournament scoring from the source be phase-two only, or required for the first release?
- Should external AI agents receive server-computed legal action hints, or only raw snapshots?
- Should a human be allowed to take over a bot or agent seat mid-match, and under what table settings?
- Should local bot policy be part of production deployment or only test/development support?
