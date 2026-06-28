# Guandan Server

Authoritative Python server foundation for four-player Guandan, with CLI clients, NPC policies, and an in-process training pipeline for learned bots.

## Run Locally

Start the API server:

```bash
uv run guandan-server
```

Start the CLI:

```bash
uv run guandan-cli
```

LLM player profiles and runtime data can be configured under `data/`.

## Tests

Run the rule engine tests:

```bash
python3 -m unittest discover -s tests/domain
```

Run the training pipeline tests:

```bash
python3 -m unittest discover -s tests/training
```

Run the full unittest suite with the repository root as top-level:

```bash
python3 -m unittest discover -s tests -t .
```

## Bot Training

Detailed training design and progress notes live in [docs/guandan_bot_training_plan.md](docs/guandan_bot_training_plan.md).

The current training pipeline includes:

- in-process self-play environment: `training/env.py`
- shared legal action generator: `server/domain/legal_actions.py`
- observation/action feature encoding: `training/encode.py`
- heuristic baseline and BC teacher: `training/heuristic.py`
- behavior-cloning sample collection: `training/collect.py`
- behavior-cloning trainer: `training/bc_train.py`
- self-play rollout and PPO scaffold: `training/rollout.py`, `training/ppo_train.py`
- checkpoint evaluation gate: `training/eval_gate.py`
- runtime RL NPC player: `npc/rl_agent/player.py`

Install/use the training extra through `uv run --extra train ...`. Verify GPU availability first:

```bash
nvidia-smi
uv run --extra train python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Collect behavior-cloning samples:

```bash
uv run --extra train guandan-bc-collect data/bc/heuristic.compact.jsonl.gz --seed-count 8 --max-deals 1 --workers 4 --compact
```

Build a tensor shard cache for repeated training:

```bash
uv run --extra train guandan-bc-cache data/bc/heuristic.compact.jsonl.gz data/bc/heuristic.bc-cache --shard-size 2048
```

Train the behavior-cloning candidate ranker on CUDA:

```bash
uv run --extra train guandan-bc-train data/bc/heuristic.compact.jsonl.gz data/models/bc_ranker.pt --epochs 3 --validation-fraction 0.1 --cache-dir data/bc/heuristic.bc-cache --batch-size 128 --device cuda
```

Run a PPO self-play continuation run on CUDA:

```bash
uv run --extra train guandan-ppo-train data/models/ppo_actor_critic.next.pt --init-model data/models/ppo_actor_critic.pt --seed-count 8 --updates 100 --epochs-per-update 3 --max-deals 4 --batch-size 256 --device cuda
```

Evaluate a checkpoint against fixed dummy, heuristic, and previous-model gates:

```bash
uv run --extra train guandan-eval-gate data/models/ppo_actor_critic.next.pt --previous-checkpoint data/models/ppo_actor_critic.pt --seed-count 4 --max-deals 1 --device cuda
```

Run the learned NPC policy server, with heuristic fallback when the checkpoint is unavailable:

```bash
uv run --extra train guandan-rl-agent-server --model-path data/models/ppo_actor_critic.pt --device cuda
```

Recent GPU smoke outputs:

- `data/bc/gpu_smoke.jsonl`: 20 deals, 1723 behavior-cloning samples.
- `data/models/gpu_smoke_bc_ranker.pt`: 3 BC epochs, loss `0.5954`, accuracy `0.815`.
- `data/models/gpu_smoke_ppo_actor_critic.pt`: 1 PPO update, 250 transitions, loss `0.1569`.

Training samples and runtime policy inputs must not include opponent private card identities. Keep learned-agent decisions constrained by the shared legal-action generator and reducer validation.
