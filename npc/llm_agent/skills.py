from __future__ import annotations

from client.api import JsonObject


CARD_RECORDER_SKILL: JsonObject = {
    "name": "card-recorder",
    "description": "Track visible Guandan cards using the same rules as tools.card_recorder.",
    "instructions": [
        "Use public CardsPlayed events and submitted actions as the only source of seen cards.",
        "Never infer hidden cards from hand counts, private snapshots for other seats, or guesses.",
        "Treat passes as no-card events.",
        "Use seen cards to reason about unseen cards and exhausted ranks, but only when the card IDs were actually visible.",
        "The matching repository utility is tools.card_recorder.CardRecorder.",
    ],
}


LLM_AGENT_SKILLS: tuple[JsonObject, ...] = (CARD_RECORDER_SKILL,)
