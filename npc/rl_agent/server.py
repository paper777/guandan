from __future__ import annotations

import argparse
from pathlib import Path

from npc.common.server import run_policy_server
from npc.rl_agent import RlAgentConfig, RlAgentPlayer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Guandan RL policy HTTP agent.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--model-path", default=None, help="PPO actor-critic or BC ranker checkpoint path.")
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda.")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser() if args.model_path else None
    run_policy_server(
        RlAgentPlayer(RlAgentConfig(model_path=model_path, device=args.device)),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
