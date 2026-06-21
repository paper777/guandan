#!/bin/bash
uv run --extra train guandan-ppo-train data/models/ppo_actor_critic.pt \
  --init-policy data/models/bc_ranker.pt \
  --seed-count 8 \
  --updates 100 \
  --epochs-per-update 3 \
  --max-deals 32 \
  --max-steps 200000 \
  --learning-rate 0.0001 \
  --gamma 0.995 \
  --gae-lambda 0.95 \
  --clip-epsilon 0.1 \
  --entropy-coef 0.003 \
  --value-coef 0.5 \
  --batch-size 33554432 \
  --max-grad-norm 0.5 \
  --target-kl 0.03 \
  --dropout 0.0 \
  --device cuda
