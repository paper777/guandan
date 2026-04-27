from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from server.domain.seats import Seat


class PlayerKind(StrEnum):
    HUMAN = "human"
    BOT = "bot"
    AGENT = "agent"


class ControllerKind(StrEnum):
    HUMAN_WS = "human_ws"
    LOCAL_BOT = "local_bot"
    EXTERNAL_AGENT = "external_agent"


class ControllerCapability(StrEnum):
    PLAY = "PLAY"
    OBSERVE_PRIVATE = "OBSERVE_PRIVATE"
    OBSERVE_PUBLIC = "OBSERVE_PUBLIC"
    AUTO_READY = "AUTO_READY"
    DEBUG_FULL_STATE = "DEBUG_FULL_STATE"


@dataclass(frozen=True, slots=True)
class PlayerRef:
    id: str
    display_name: str
    kind: PlayerKind


@dataclass(frozen=True, slots=True)
class ControllerRef:
    id: str
    kind: ControllerKind
    seat: Seat
    player_id: str
    capabilities: frozenset[ControllerCapability] = field(default_factory=frozenset)

    def can(self, capability: ControllerCapability) -> bool:
        return capability in self.capabilities
