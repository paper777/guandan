#!/bin/bash
set -euo pipefail

DATASET="${DATASET:-data/bc/heuristic_seed_1000.compact.jsonl.gz}"
CACHE_DIR="${CACHE_DIR:-data/bc/heuristic_seed_1000.bc-cache}"
OUTPUT_MODEL="${OUTPUT_MODEL:-data/models/bc_ranker.pt}"
EVAL_SEED_COUNT="${EVAL_SEED_COUNT:-10}"
EVAL_MAX_DEALS="${EVAL_MAX_DEALS:-32}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-20000}"
DEVICE="${DEVICE:-cuda}"

# uv run --extra train guandan-bc-train data/bc/heuristic_seed_100.compact.jsonl.gz data/models/bc_ranker.pt --epochs 10 --validation-fraction 0.1 --cache-dir data/bc/heuristic_seed_100.bc-cache --batch-size 128 --device cuda
uv run --extra train guandan-bc-train "${DATASET}" "${OUTPUT_MODEL}" --epochs 10 --validation-fraction 0.1 --cache-dir "${CACHE_DIR}" --batch-size 256 --device "${DEVICE}"

if [[ "${EVAL_SEED_COUNT}" != "0" ]]; then
  uv run --extra train guandan-eval-gate "${OUTPUT_MODEL}" \
    --seed-count "${EVAL_SEED_COUNT}" \
    --max-deals "${EVAL_MAX_DEALS}" \
    --max-steps "${EVAL_MAX_STEPS}" \
    --device "${DEVICE}"
fi
