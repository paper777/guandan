# NPC Broker

The broker owns all client/server interaction with the Guandan table API for external NPC processes.

Responsibilities:

- Join NPC seats through the table API.
- Mark NPC seats ready.
- Poll private seat snapshots.
- Convert snapshots into policy action requests.
- Submit policy actions back to the table API.

Run three dummy NPCs for an existing table:

```bash
python -m npc.broker.broker --server-url http://127.0.0.1:8000 --table-id table-abc --seats S,W,N --start
```

Policies must not call the Guandan server directly. Add new bot behavior under its own `npc/<bot_type>/` package and let the broker drive it.
