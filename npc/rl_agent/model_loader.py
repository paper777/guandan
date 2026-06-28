from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from server.domain.legal_actions import ActionCandidate
from server.services.snapshots import SeatSnapshot
from training.encode import encode_action, encode_observation, validate_encoding_schema


DEFAULT_MODEL_PATH = Path("data/models/ppo_actor_critic.pt")


@dataclass(frozen=True, slots=True)
class RlAgentConfig:
    player_name: str | None = None
    seat: str | None = None
    model_path: Path | None = None
    device: str | None = None
    min_model_deadline_ms: int = 75
    provider_name: str = "rl-agent"

    @property
    def model_name(self) -> str:
        path = self.resolved_model_path()
        return path.name if path.exists() else "heuristic-fallback"

    def resolved_model_path(self) -> Path:
        if self.model_path is not None:
            return self.model_path
        env_path = os.environ.get("GUANDAN_RL_MODEL")
        if env_path:
            return Path(env_path).expanduser()
        return DEFAULT_MODEL_PATH


class RlModelLoader:
    def __init__(self, config: RlAgentConfig | None = None) -> None:
        self.config = config or RlAgentConfig()
        self._loaded: _LoadedModel | None = None
        self.unavailable_reason: str | None = None

    def choose_action(
        self,
        snapshot: SeatSnapshot,
        actions: tuple[ActionCandidate, ...],
    ) -> ActionCandidate | None:
        loaded = self._load()
        if loaded is None:
            return None
        if not actions:
            return None
        return loaded.choose_action(snapshot, actions)

    def _load(self) -> "_LoadedModel | None":
        if self._loaded is not None:
            return self._loaded
        path = self.config.resolved_model_path()
        if not path.exists():
            self.unavailable_reason = f"model checkpoint not found: {path}"
            return None
        try:
            self._loaded = _load_checkpoint(path, self.config.device)
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"
            return None
        return self._loaded


@dataclass(slots=True)
class _LoadedModel:
    torch: object
    model: object
    device: object
    observation_dim: int
    action_dim: int
    kind: str
    schema_version: str

    def choose_action(self, snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...]) -> ActionCandidate | None:
        observation = encode_observation(snapshot, schema_version=self.schema_version).values
        if len(observation) != self.observation_dim:
            raise ValueError(
                f"observation_dim mismatch: checkpoint={self.observation_dim} runtime={len(observation)}"
            )
        action_values = [encode_action(action, snapshot, schema_version=self.schema_version).values for action in actions]
        if any(len(values) != self.action_dim for values in action_values):
            raise ValueError(f"action_dim mismatch: checkpoint={self.action_dim}")
        torch = self.torch
        with torch.no_grad():
            observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
            action_tensor = torch.tensor(action_values, dtype=torch.float32, device=self.device)
            observations = observation_tensor.expand(action_tensor.shape[0], -1)
            pair_features = torch.cat((observations, action_tensor), dim=1)
            if self.kind == "ppo":
                logits = self.model.policy_logits(pair_features)
            else:
                logits = self.model(pair_features)
            index = int(torch.argmax(logits).item())
        return actions[index]


def _load_checkpoint(path: Path, device_name: str | None) -> _LoadedModel:
    from training.model import build_candidate_actor_critic, build_candidate_ranker, pair_feature_dim, require_torch

    torch = require_torch()
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    schema_version = validate_encoding_schema(checkpoint)
    observation_dim = int(checkpoint["observation_dim"])
    action_dim = int(checkpoint["action_dim"])
    hidden_dim = int(checkpoint.get("hidden_dim", 256))
    dropout = float(checkpoint.get("dropout", 0.0))
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint is missing model_state")

    if any(str(key).startswith("policy_net.") for key in model_state):
        model = build_candidate_actor_critic(
            pair_feature_dim(observation_dim, action_dim),
            observation_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(device)
        model.load_state_dict(model_state)
        kind = "ppo"
    elif any(str(key).startswith("net.") for key in model_state):
        model = build_candidate_ranker(
            pair_feature_dim(observation_dim, action_dim),
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(device)
        model.load_state_dict(model_state)
        kind = "bc"
    else:
        raise ValueError("checkpoint model_state is neither PPO actor-critic nor BC ranker")
    model.eval()
    return _LoadedModel(torch, model, device, observation_dim, action_dim, kind, schema_version)
