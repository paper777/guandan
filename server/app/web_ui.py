from __future__ import annotations

import mimetypes
from pathlib import Path


WEB_UI_ROUTE = "/ui"
WEB_UI_DIR = Path(__file__).resolve().parents[1] / "web_ui"
INDEX_FILE = WEB_UI_DIR / "index.html"


def web_ui_file_for_path(path: str) -> Path | None:
    if path in {"", "/", WEB_UI_ROUTE, f"{WEB_UI_ROUTE}/"}:
        return INDEX_FILE
    if not path.startswith(f"{WEB_UI_ROUTE}/"):
        return None
    relative = path.removeprefix(f"{WEB_UI_ROUTE}/")
    if not relative:
        return INDEX_FILE
    candidate = (WEB_UI_DIR / relative).resolve()
    try:
        candidate.relative_to(WEB_UI_DIR.resolve())
    except ValueError:
        return None
    if candidate.is_dir():
        candidate = candidate / "index.html"
    return candidate


def content_type_for_file(path: Path) -> str:
    if path.suffix == ".js":
        return "text/javascript; charset=utf-8"
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed is None:
        return "application/octet-stream"
    if guessed.startswith("text/") or guessed in {"application/json", "image/svg+xml"}:
        return f"{guessed}; charset=utf-8"
    return guessed
