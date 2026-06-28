from __future__ import annotations

from typing import Any


DUAL_TOWER_ARCHITECTURE = "dual_tower_v1"
DEFAULT_MODEL_ARCHITECTURE = DUAL_TOWER_ARCHITECTURE


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional training dependency
        raise RuntimeError(
            "PyTorch is required for model training. Install a CUDA-enabled torch build before running BC training."
        ) from exc
    return torch


def build_candidate_ranker(
    input_dim: int,
    *,
    observation_dim: int | None = None,
    action_dim: int | None = None,
    hidden_dim: int = 256,
    dropout: float = 0.1,
) -> Any:
    torch = require_torch()
    nn = torch.nn
    observation_dim, action_dim = _require_split_dims(input_dim, observation_dim, action_dim)

    class CandidateRanker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_net = _build_dual_tower_policy_scorer(
                nn,
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

        def forward(self, pair_features: Any) -> Any:
            return self.policy_net(pair_features).squeeze(-1)

    return CandidateRanker()


def build_candidate_actor_critic(
    pair_input_dim: int,
    observation_dim: int,
    *,
    action_dim: int | None = None,
    value_input_dim: int | None = None,
    hidden_dim: int = 256,
    dropout: float = 0.1,
) -> Any:
    torch = require_torch()
    nn = torch.nn
    value_dim = value_input_dim or observation_dim
    observation_dim, action_dim = _require_split_dims(pair_input_dim, observation_dim, action_dim)

    class CandidateActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_net = _build_dual_tower_policy_scorer(
                nn,
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            self.value_net = nn.Sequential(
                nn.Linear(value_dim, hidden_dim),
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


def _require_split_dims(
    input_dim: int,
    observation_dim: int | None,
    action_dim: int | None,
) -> tuple[int, int]:
    if observation_dim is None or action_dim is None:
        raise ValueError("observation_dim and action_dim are required for dual-tower models")
    if observation_dim + action_dim != input_dim:
        raise ValueError(
            f"pair input dim mismatch: observation_dim + action_dim = {observation_dim + action_dim}, "
            f"input_dim = {input_dim}"
        )
    return observation_dim, action_dim


def _build_dual_tower_policy_scorer(nn, *, observation_dim: int, action_dim: int, hidden_dim: int, dropout: float):
    torch = __import__("torch")

    class DualTowerPolicyScorer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.observation_dim = observation_dim
            self.action_dim = action_dim
            self.state_encoder = nn.Sequential(
                nn.Linear(observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.action_encoder = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.scorer = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, pair_features: Any) -> Any:
            observations = pair_features[:, : self.observation_dim]
            actions = pair_features[:, self.observation_dim :]
            state = self.state_encoder(observations)
            action = self.action_encoder(actions)
            interaction = state * action
            distance = (state - action).abs()
            return self.scorer(torch.cat((state, action, interaction, distance), dim=1))

    return DualTowerPolicyScorer()
