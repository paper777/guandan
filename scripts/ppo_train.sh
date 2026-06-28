#!/bin/bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-data/models/ppo_actor_critic.pt}"
OUTPUT_MODEL="${OUTPUT_MODEL:-data/models/ppo_actor_critic.next.pt}"
SEED_COUNT="${SEED_COUNT:-4}"
UPDATES="${UPDATES:-100}"
EPOCHS_PER_UPDATE="${EPOCHS_PER_UPDATE:-3}"
MAX_DEALS="${MAX_DEALS:-32}"
# Keep PPO updates as true minibatches; full-rollout batches create one optimizer step per epoch.
BATCH_SIZE="${BATCH_SIZE:-1024}"
MAX_STEPS="${MAX_STEPS:-50000}"
DEVICE="${DEVICE:-cuda}"
OPPONENT_POOL="${OPPONENT_POOL:-self,heuristic,dummy,previous}"
# Opponent-pool training expands each seed into multiple rollout jobs; run several jobs concurrently by default.
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
REWARD_SHAPING_START="${REWARD_SHAPING_START:-0.02}"
REWARD_SHAPING_END="${REWARD_SHAPING_END:-0.0}"
EVAL_SEED_COUNT="${EVAL_SEED_COUNT:-4}"
EVAL_MAX_DEALS="${EVAL_MAX_DEALS:-1}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-20000}"

if [[ ! -f "${BASE_MODEL}" ]]; then
  echo "missing base PPO checkpoint: ${BASE_MODEL}" >&2
  exit 1
fi

uv run --extra train guandan-ppo-train "${OUTPUT_MODEL}" \
  --init-model "${BASE_MODEL}" \
  --seed-count "${SEED_COUNT}" \
  --updates "${UPDATES}" \
  --epochs-per-update "${EPOCHS_PER_UPDATE}" \
  --max-deals "${MAX_DEALS}" \
  --max-steps "${MAX_STEPS}" \
  --learning-rate 0.0001 \
  --gamma 0.995 \
  --gae-lambda 0.95 \
  --clip-epsilon 0.1 \
  --entropy-coef 0.003 \
  --value-coef 0.5 \
  --batch-size "${BATCH_SIZE}" \
  --max-grad-norm 0.5 \
  --target-kl 0.03 \
  --dropout 0.0 \
  --opponent-pool "${OPPONENT_POOL}" \
  --rollout-workers "${ROLLOUT_WORKERS}" \
  --reward-shaping-start "${REWARD_SHAPING_START}" \
  --reward-shaping-end "${REWARD_SHAPING_END}" \
  --device "${DEVICE}"

if [[ "${EVAL_SEED_COUNT}" != "0" ]]; then
  uv run --extra train guandan-eval-gate "${OUTPUT_MODEL}" \
    --previous-checkpoint "${BASE_MODEL}" \
    --seed-count "${EVAL_SEED_COUNT}" \
    --max-deals "${EVAL_MAX_DEALS}" \
    --max-steps "${EVAL_MAX_STEPS}" \
    --device "${DEVICE}"
fi
