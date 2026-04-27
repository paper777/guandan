# Dummy Bot Design

## Summary

Add a deterministic NPC foundation with a dummy bot policy. The dummy bot is intentionally weak and predictable: it keeps games moving for UI and integration testing, but it is not a strategic AI.

The server remains authoritative. The bot only proposes normal domain commands, and every command still passes through the same reducer validation as human and external-agent commands.

## Layout

- `client/`: shared CLI, Guandan table HTTP client, and NPC protocol request types.
- `npc/common/server.py`: generic HTTP policy server helper.
- `npc/dummy_bot`: dummy behavior only, plus a thin runner that hosts the policy through `npc/common.server`.
- `npc/broker`: owns all client/server interaction with the Guandan table API for out-of-process NPCs.
- `server/`: contains only the authoritative game server and no NPC actor runtime.

## Behavior

- When leading, play the lowest-ranked single card in the bot hand.
- When responding to an existing trick, pass.
- When paying tribute, submit the highest eligible non-wild tribute card.
- When returning tribute, submit the lowest legal return card, respecting the partner-return rank limit.
- If no legal fallback can be selected, do nothing and leave the server timeout fallback to handle the prompt.

## Integration

- Out-of-process policies must not call the Guandan table API directly; the broker joins seats, polls snapshots, and submits actions.
- `TableActor` does not run NPC policy. It only exposes snapshots, accepts commands, and applies timeout fallback.
- The broker serializes NPC decisions externally by polling snapshots and submitting normal table API commands.

## Non-Goals

- No hand-strength strategy, partnership inference, bomb conservation, or endgame search.
- No hidden information beyond the bot seat's private snapshot.
- No external service calls.
