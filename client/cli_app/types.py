from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class CliResult:
    exit_code: int
    output: str


InputFn = Callable[[str], str | None]
OutputFn = Callable[[str], None]
