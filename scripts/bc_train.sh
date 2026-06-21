
#!/bin/bash
# uv run --extra train guandan-bc-train data/bc/heuristic_seed_100.compact.jsonl.gz data/models/bc_ranker.pt --epochs 10 --validation-fraction 0.1 --cache-dir data/bc/heuristic_seed_100.bc-cache --batch-size 128 --device cuda
uv run --extra train guandan-bc-train data/bc/heuristic_seed_1000.compact.jsonl.gz data/models/bc_ranker.pt --epochs 10 --validation-fraction 0.1 --cache-dir data/bc/heuristic_seed_1000.bc-cache --batch-size 256 --device cuda
