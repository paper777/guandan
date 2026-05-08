from __future__ import annotations

import argparse

from npc.common.server import run_policy_server
from npc.dummy_bot.player import DummyBotPlayer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal Guandan dummy bot HTTP agent.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    args = parser.parse_args()

    run_policy_server(DummyBotPlayer(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
