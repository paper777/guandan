#!/bin/bash
set -euo pipefail

INIT_CHECKPOINT="${INIT_CHECKPOINT:-${BASE_MODEL:-data/models/bc_ranker.pt}}"
# First PPO bootstrap should use a BC ranker checkpoint. To continue PPO later, set INIT_CHECKPOINT to a PPO actor-critic checkpoint.
OUTPUT_MODEL="${OUTPUT_MODEL:-data/models/ppo_actor_critic.next.pt}"
SEED_COUNT="${SEED_COUNT:-10}"
UPDATES="${UPDATES:-10}"
MAX_DEALS="${MAX_DEALS:-24}"
EPOCHS_PER_UPDATE="${EPOCHS_PER_UPDATE:-3}"
# Keep PPO updates as true minibatches; full-rollout batches create one optimizer step per epoch.
BATCH_SIZE="${BATCH_SIZE:-1024}"
MAX_STEPS="${MAX_STEPS:-500000000}"
DEVICE="${DEVICE:-cuda}"
OPPONENT_POOL="${OPPONENT_POOL:-self,heuristic,previous}"
# Opponent-pool training expands each seed into multiple rollout jobs; run several jobs concurrently by default.
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-16}"
# Set ROLLOUT_PROCESSES=0 to fall back to threaded rollout.
ROLLOUT_PROCESSES="${ROLLOUT_PROCESSES:-16}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-16}"
INFERENCE_BATCH_WAIT_MS="${INFERENCE_BATCH_WAIT_MS:-1.0}"
# Set CANDIDATE_BUCKET_BATCHES=0 to restore fully shuffled PPO minibatches.
CANDIDATE_BUCKET_BATCHES="${CANDIDATE_BUCKET_BATCHES:-1}"
REWARD_SHAPING_START="${REWARD_SHAPING_START:-0.02}"
REWARD_SHAPING_END="${REWARD_SHAPING_END:-0.0}"
EVAL_SEED_COUNT="${EVAL_SEED_COUNT:-4}"
EVAL_MAX_DEALS="${EVAL_MAX_DEALS:-1}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-20000}"

if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "missing PPO init checkpoint: ${INIT_CHECKPOINT}" >&2
  exit 1
fi

CANDIDATE_BUCKET_FLAG=()
if [[ "${CANDIDATE_BUCKET_BATCHES}" == "0" ]]; then
  CANDIDATE_BUCKET_FLAG=(--no-candidate-bucket-batches)
fi

uv run --extra train guandan-ppo-train "${OUTPUT_MODEL}" \
  --init-policy "${INIT_CHECKPOINT}" \
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
  --rollout-processes "${ROLLOUT_PROCESSES}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE}" \
  --inference-batch-wait-ms "${INFERENCE_BATCH_WAIT_MS}" \
  "${CANDIDATE_BUCKET_FLAG[@]}" \
  --reward-shaping-start "${REWARD_SHAPING_START}" \
  --reward-shaping-end "${REWARD_SHAPING_END}" \
  --device "${DEVICE}"

if [[ "${EVAL_SEED_COUNT}" != "0" ]]; then
  uv run --extra train guandan-eval-gate "${OUTPUT_MODEL}" \
    --previous-checkpoint "${INIT_CHECKPOINT}" \
    --seed-count "${EVAL_SEED_COUNT}" \
    --max-deals "${EVAL_MAX_DEALS}" \
    --max-steps "${EVAL_MAX_STEPS}" \
    --device "${DEVICE}" | tee -i data/models/eval.log
fi
