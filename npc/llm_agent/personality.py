from __future__ import annotations

from client.types import JsonObject


PERSONALITY_FACTORS: dict[str, JsonObject] = {
    "aggressive": {
        "type": "aggressive",
        "risk_tolerance": "high",
        "tempo_bias": "seize or keep initiative when a legal candidate creates a clear follow-up path",
        "bomb_usage": "willing to spend bombs for tempo, opponent denial, or partner delivery",
        "passing_bias": "low; pass mainly when no efficient beating candidate exists or partner is better positioned",
        "structure_bias": "may split weak structures when it materially reduces effective turns",
    },
    "balanced": {
        "type": "balanced",
        "risk_tolerance": "medium",
        "tempo_bias": "trade off personal turn reduction, partner tempo, and opponent blocking",
        "bomb_usage": "use bombs only when tempo or defense value is concrete",
        "passing_bias": "medium; pass when beating wastes control or hurts partner cooperation",
        "structure_bias": "preserve coherent structures unless the payoff is clear",
    },
    "defensive": {
        "type": "defensive",
        "risk_tolerance": "low",
        "tempo_bias": "protect control cards and prioritize blocking dangerous opponent endgames",
        "bomb_usage": "reserve bombs for defense, partner protection, or decisive late-trick control",
        "passing_bias": "high; prefer passing over spending important control without a concrete benefit",
        "structure_bias": "avoid splitting strong structures unless it prevents an opponent finish or sends partner out",
    },
}


def personality_context(personality: str | None) -> JsonObject:
    key = normalize_personality(personality)
    return dict(PERSONALITY_FACTORS[key])


def normalize_personality(personality: str | None) -> str:
    key = str(personality or "balanced").strip().lower().replace("_", "-")
    if key in PERSONALITY_FACTORS:
        return key
    return "balanced"
