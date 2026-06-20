from __future__ import annotations

from typing import Any


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional training dependency
        raise RuntimeError(
            "PyTorch is required for model training. Install a CUDA-enabled torch build before running BC training."
        ) from exc
    return torch


def build_candidate_ranker(input_dim: int, *, hidden_dim: int = 256, dropout: float = 0.1) -> Any:
    torch = require_torch()
    nn = torch.nn

    class CandidateRanker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, pair_features: Any) -> Any:
            return self.net(pair_features).squeeze(-1)

    return CandidateRanker()


def build_candidate_actor_critic(
    pair_input_dim: int,
    observation_dim: int,
    *,
    hidden_dim: int = 256,
    dropout: float = 0.1,
) -> Any:
    torch = require_torch()
    nn = torch.nn

    class CandidateActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_net = nn.Sequential(
                nn.Linear(pair_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.value_net = nn.Sequential(
                nn.Linear(observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def policy_logits(self, pair_features: Any) -> Any:
            return self.policy_net(pair_features).squeeze(-1)

        def value(self, observation: Any) -> Any:
            return self.value_net(observation).squeeze(-1)

    return CandidateActorCritic()


def pair_feature_dim(observation_dim: int, action_dim: int) -> int:
    return observation_dim + action_dim
