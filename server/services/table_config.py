from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


DEFAULT_ACTION_TIMEOUT_SECONDS = 45
MIN_ACTION_TIMEOUT_SECONDS = 5
MAX_ACTION_TIMEOUT_SECONDS = 300


class TimeoutFallback(StrEnum):
    AUTO_PASS = "auto_pass"


@dataclass(frozen=True, slots=True)
class TableConfig:
    action_timeout_seconds: int = DEFAULT_ACTION_TIMEOUT_SECONDS
    timeout_fallback: TimeoutFallback = TimeoutFallback.AUTO_PASS

    def __post_init__(self) -> None:
        if not MIN_ACTION_TIMEOUT_SECONDS <= self.action_timeout_seconds <= MAX_ACTION_TIMEOUT_SECONDS:
            raise ValueError(
                "action_timeout_seconds must be between "
                f"{MIN_ACTION_TIMEOUT_SECONDS} and {MAX_ACTION_TIMEOUT_SECONDS}"
            )
