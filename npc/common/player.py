from __future__ import annotations

from abc import ABC, abstractmethod

from client.api import ActionRequest, JsonObject


class Player(ABC):
    """Abstract NPC player used by broker-compatible bots and agents."""

    @abstractmethod
    def choose_action(self, request: ActionRequest) -> JsonObject:
        """Return a command-like action for the broker or HTTP agent server."""

    def observe_action(self, observation: JsonObject) -> None:
        """Observe an action submitted by another broker-controlled player."""

