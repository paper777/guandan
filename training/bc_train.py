from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from training.collect import BcSample, read_jsonl
from training.model import build_candidate_ranker, pair_feature_dim, require_torch


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    dataset_path: Path
    output_path: Path
    epochs: int = 3
    learning_rate: float = 1e-3
    hidden_dim: int = 256
    dropout: float = 0.1
    limit: int | None = None
    seed: int = 1
    device: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    samples: int
    epochs: int
    final_loss: float
    final_accuracy: float
    output_path: Path


def train_behavior_clone(config: TrainingConfig) -> TrainingSummary:
    samples = read_jsonl(config.dataset_path, limit=config.limit)
    if not samples:
        raise ValueError("dataset contains no behavior cloning samples")

    torch = require_torch()
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    observation_dim = len(samples[0].observation_values)
    action_dim = len(samples[0].action_names)
    _validate_dimensions(samples, observation_dim, action_dim)

    model = build_candidate_ranker(
        pair_feature_dim(observation_dim, action_dim),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    final_loss = 0.0
    final_accuracy = 0.0
    indices = list(range(len(samples)))
    for _epoch in range(config.epochs):
        rng.shuffle(indices)
        total_loss = 0.0
        correct = 0
        model.train()
        for index in indices:
            sample = samples[index]
            logits = _sample_logits(torch, model, sample, device)
            target = torch.tensor([sample.chosen_index], dtype=torch.long, device=device)
            loss = loss_fn(logits.unsqueeze(0), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            correct += int(torch.argmax(logits).item() == sample.chosen_index)
        final_loss = total_loss / len(samples)
        final_accuracy = correct / len(samples)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "observation_names": samples[0].observation_names,
            "action_names": samples[0].action_names,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "samples": len(samples),
            "epochs": config.epochs,
            "final_loss": final_loss,
            "final_accuracy": final_accuracy,
        },
        config.output_path,
    )
    return TrainingSummary(
        samples=len(samples),
        epochs=config.epochs,
        final_loss=final_loss,
        final_accuracy=final_accuracy,
        output_path=config.output_path,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = train_behavior_clone(
        TrainingConfig(
            dataset_path=Path(args.dataset),
            output_path=Path(args.output),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            limit=args.limit,
            seed=args.seed,
            device=args.device,
        )
    )
    print(
        f"trained {summary.samples} samples for {summary.epochs} epochs; "
        f"loss={summary.final_loss:.4f} accuracy={summary.final_accuracy:.3f}; "
        f"wrote {summary.output_path}",
        flush=True,
    )
    return 0


def _sample_logits(torch, model, sample: BcSample, device):
    observation = torch.tensor(sample.observation_values, dtype=torch.float32, device=device)
    actions = torch.tensor(sample.candidate_values, dtype=torch.float32, device=device)
    observations = observation.expand(actions.shape[0], -1)
    pair_features = torch.cat((observations, actions), dim=1)
    return model(pair_features)


def _validate_dimensions(samples: tuple[BcSample, ...], observation_dim: int, action_dim: int) -> None:
    for index, sample in enumerate(samples):
        if len(sample.observation_values) != observation_dim:
            raise ValueError(f"sample {index} has inconsistent observation dimension")
        if len(sample.action_names) != action_dim:
            raise ValueError(f"sample {index} has inconsistent action feature names")
        if not sample.candidate_values:
            raise ValueError(f"sample {index} has no candidates")
        if not 0 <= sample.chosen_index < len(sample.candidate_values):
            raise ValueError(f"sample {index} chosen_index is out of range")
        for candidate in sample.candidate_values:
            if len(candidate) != action_dim:
                raise ValueError(f"sample {index} has inconsistent action dimension")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Guandan behavior cloning candidate ranker.")
    parser.add_argument("dataset", help="Input JSONL dataset from training.collect.")
    parser.add_argument("output", help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
