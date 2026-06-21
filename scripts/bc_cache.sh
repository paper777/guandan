#!/bin/bash
uv run --extra train guandan-bc-cache data/bc/heuristic_seed_1000.compact.jsonl.gz data/bc/heuristic_seed_1000.bc-cache --shard-size 2048
