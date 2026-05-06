from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


DEFAULT_ACTION_TIMEOUT_SECONDS = 45
MIN_ACTION_TIMEOUT_SECONDS = 5
MAX_ACTION_TIMEOUT_SECONDS = 300
DEFAULT_RANDOM_SEED_BYTES = 32
DEFAULT_RANDOM_SEED_PATH = Path("/dev/random")


class TimeoutFallback(StrEnum):
    AUTO_PASS = "auto_pass"


@dataclass(frozen=True, slots=True)
class TableConfig:
    action_timeout_seconds: int = DEFAULT_ACTION_TIMEOUT_SECONDS
    timeout_fallback: TimeoutFallback = TimeoutFallback.AUTO_PASS
    deal_seed: str | int | bytes | None = None
    random_seed_path: str | Path | None = DEFAULT_RANDOM_SEED_PATH
    random_seed_bytes: int = DEFAULT_RANDOM_SEED_BYTES

    def __post_init__(self) -> None:
        if not MIN_ACTION_TIMEOUT_SECONDS <= self.action_timeout_seconds <= MAX_ACTION_TIMEOUT_SECONDS:
            raise ValueError(
                "action_timeout_seconds must be between "
                f"{MIN_ACTION_TIMEOUT_SECONDS} and {MAX_ACTION_TIMEOUT_SECONDS}"
            )
        if self.random_seed_bytes <= 0:
            raise ValueError("random_seed_bytes must be positive")

    def seed_for_deal(self, deal_number: int) -> Any:
        if self.deal_seed is None:
            return _random_seed(self.random_seed_path, self.random_seed_bytes)
        if deal_number <= 1:
            return self.deal_seed
        if isinstance(self.deal_seed, bytes):
            return self.deal_seed + f":deal-{deal_number}".encode()
        return f"{self.deal_seed}:deal-{deal_number}"


def _random_seed(path: str | Path | None, size: int) -> bytes:
    if path is None:
        return os.urandom(size)
    try:
        with Path(path).open("rb") as random_device:
            seed = random_device.read(size)
    except OSError:
        return os.urandom(size)
    return seed if len(seed) == size else os.urandom(size)
