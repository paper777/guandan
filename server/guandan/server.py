from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence


DEFAULT_APP = "guandan.app.main:app"


@dataclass(frozen=True, slots=True)
class ServerOptions:
    app: str
    host: str
    port: int
    reload: bool
    log_level: str


def parse_args(argv: Sequence[str] | None = None) -> ServerOptions:
    parser = argparse.ArgumentParser(prog="guandan-server", description="Run the Guandan API server.")
    parser.add_argument("--app", default=DEFAULT_APP, help="ASGI app import path.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--reload", action="store_true", help="Reload the server when source files change.")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        help="Uvicorn log level.",
    )
    args = parser.parse_args(argv)
    return ServerOptions(
        app=args.app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_args(argv)
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        if exc.name != "uvicorn":
            raise
        raise SystemExit("uvicorn is not installed; run `uv sync` in the server directory.") from exc

    uvicorn.run(
        options.app,
        host=options.host,
        port=options.port,
        reload=options.reload,
        log_level=options.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
