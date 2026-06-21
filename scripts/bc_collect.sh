#!/bin/bash
# uv run --extra train guandan-bc-collect data/bc/heuristic_seed_100.compact.jsonl.gz \
#     --max-deals 1000 --max-steps 1000000 --seed-count 100 --workers=16 --compact

uv run --extra train guandan-bc-collect data/bc/heuristic_seed_1000.compact.jsonl.gz \
    --max-deals 1000 --max-steps 1000000000 --seed-count 1000 --workers=16 --compact
