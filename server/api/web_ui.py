from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from server.app.web_ui import WEB_UI_DIR


def register_web_ui(app: FastAPI) -> None:
    @app.get("/", include_in_schema=False)
    async def web_ui_root() -> RedirectResponse:
        return RedirectResponse(url="/ui/")

    if WEB_UI_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(WEB_UI_DIR), html=True), name="web-ui")
