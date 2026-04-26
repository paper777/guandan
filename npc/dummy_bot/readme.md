# Dummy Bot

The dummy bot is a minimal NPC policy used for local development and protocol testing.

Behavior:

- Pass whenever the prompt allows passing.
- If forced to lead, play the lowest-ranked single card in its private hand.
- During tribute prompts, choose the highest eligible tribute card or the lowest legal return card.
- Return a structured error when it cannot produce an action.

Run it as a small HTTP policy server:

```bash
python -m npc.dummy_bot.server --host 127.0.0.1 --port 9001
```

It accepts the common NPC action request shape and returns:

```json
{"type": "pass"}
```

or:

```json
{"type": "play_cards", "card_ids": ["D1-S-3"]}
```
