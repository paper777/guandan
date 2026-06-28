from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from server.domain.legal_actions import ActionCandidate
from server.domain.seats import SEATS, Seat, Team, team_for_seat
from server.services.snapshots import SeatSnapshot
from training.encode import (
    ENCODING_SCHEMA_VERSION,
    encode_action,
    encode_critic_observation,
    encode_observation,
    encoding_schema,
    validate_encoding_schema,
)
from training.heuristic import HeuristicPolicy
from training.model import (
    DEFAULT_MODEL_ARCHITECTURE,
    build_candidate_actor_critic,
    pair_feature_dim,
    require_torch,
)
from training.rollout import (
    RolloutDecision,
    RolloutEncodedFeatures,
    RolloutProfile,
    RolloutResult,
    RolloutTransition,
    collect_rollout,
)


@dataclass(frozen=True, slots=True)
class PpoConfig:
    output_path: Path
    init_policy_path: Path | None = None
    rollout_seeds: tuple[str, ...] = ("ppo-seed-0",)
    updates: int = 1
    epochs_per_update: int = 3
    max_deals_per_seed: int = 1
    max_steps_per_seed: int = 20_000
    learning_rate: float = 1e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.1
    entropy_coef: float = 0.003
    value_coef: float = 0.5
    batch_size: int = 256
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    normalize_advantages: bool = True
    hidden_dim: int = 256
    dropout: float = 0.0
    opponent_pool: tuple[str, ...] = ("self",)
    opponent_checkpoint_paths: tuple[Path, ...] = ()
    rollout_workers: int = 1
    rollout_processes: int = 0
    inference_batch_size: int = 1
    inference_batch_wait_ms: float = 1.0
    candidate_bucket_batches: bool = True
    reward_shaping_start: float = 0.02
    reward_shaping_end: float = 0.0
    seed: int = 1
    device: str | None = None
    log_updates: bool = True


@dataclass(frozen=True, slots=True)
class PpoSummary:
    updates: int
    transitions: int
    final_loss: float
    output_path: Path
    init_policy_path: Path | None = None
    init_checkpoint_kind: str | None = None


@dataclass(frozen=True, slots=True)
class RolloutMetrics:
    completed_deals: int
    steps: int
    stopped_reasons: dict[str, int]
    profile: RolloutProfile


@dataclass(frozen=True, slots=True)
class PolicyInferenceProfile:
    requests: int = 0
    batches: int = 0
    max_batch_size: int = 0
    inference_seconds: float = 0.0

    @property
    def average_batch_size(self) -> float:
        return self.requests / self.batches if self.batches else 0.0


@dataclass(frozen=True, slots=True)
class RolloutJob:
    policies: dict[Seat, object]
    seed: str
    max_deals: int
    max_steps: int
    record_seats: frozenset[Seat] | None
    reward_shaping_weight: float


@dataclass(frozen=True, slots=True)
class ProcessRolloutJob:
    seed: str
    max_deals: int
    max_steps: int
    record_seats: frozenset[Seat] | None
    reward_shaping_weight: float
    current_policy_seats: frozenset[Seat]
    opponent_name: str
    opponent_checkpoint_path: Path | None = None
    opponent_device_name: str = "cpu"
    schema_version: str = ENCODING_SCHEMA_VERSION
    centralized_critic: bool = True


def train_self_play_ppo(config: PpoConfig) -> PpoSummary:
    torch = require_torch()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    schema_version = _checkpoint_schema_version_for_training(torch, config.init_policy_path, device)
    observation_dim, action_dim, critic_observation_dim, critic_observation_names = _initial_dimensions(
        config.rollout_seeds[0],
        schema_version=schema_version,
    )
    model_architecture = DEFAULT_MODEL_ARCHITECTURE
    value_input_dim = critic_observation_dim
    model = build_candidate_actor_critic(
        pair_feature_dim(observation_dim, action_dim),
        observation_dim,
        action_dim=action_dim,
        value_input_dim=value_input_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    init_checkpoint_kind: str | None = None
    if config.init_policy_path is not None:
        init_checkpoint_kind = _initialize_model_from_checkpoint(
            torch,
            model,
            config.init_policy_path,
            observation_dim,
            action_dim,
            config.hidden_dim,
            device,
            schema_version,
            value_input_dim,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    final_loss = 0.0
    total_transitions = 0
    if config.log_updates:
        init_note = (
            f" init_checkpoint={config.init_policy_path} kind={init_checkpoint_kind}"
            if config.init_policy_path
            else ""
        )
        print(
            f"ppo config device={device} seeds={len(config.rollout_seeds)} updates={config.updates} "
            f"epochs_per_update={config.epochs_per_update} max_deals={config.max_deals_per_seed} "
            f"max_steps={config.max_steps_per_seed} batch_size={config.batch_size} lr={config.learning_rate:g} "
            f"gamma={config.gamma:g} gae_lambda={config.gae_lambda:g} clip={config.clip_epsilon:g} "
            f"entropy_coef={config.entropy_coef:g} target_kl={config.target_kl:g} "
            f"architecture={model_architecture} centralized_critic=True "
            f"opponents={','.join(config.opponent_pool)} rollout_workers={config.rollout_workers} "
            f"rollout_processes={config.rollout_processes} "
            f"inference_batch={config.inference_batch_size}@{config.inference_batch_wait_ms:g}ms "
            f"candidate_buckets={'on' if config.candidate_bucket_batches else 'off'} "
            f"reward_shaping={config.reward_shaping_start:g}->{config.reward_shaping_end:g}{init_note}",
            flush=True,
        )

    for update_index in range(config.updates):
        update_started_at = time.perf_counter()
        rollout_started_at = update_started_at
        shaping_weight = _linear_decay(
            config.reward_shaping_start,
            config.reward_shaping_end,
            update_index,
            max(config.updates, 1),
        )
        was_training = model.training
        model.eval()
        try:
            if config.rollout_processes > 0:
                process_jobs = _rollout_process_jobs_for_update(
                    config,
                    update_index=update_index,
                    device_name=str(device),
                    reward_shaping_weight=shaping_weight,
                    schema_version=schema_version,
                )
                rollouts, inference_profile = _collect_rollout_process_jobs(
                    process_jobs,
                    processes=config.rollout_processes,
                    torch=torch,
                    model=model,
                    device=device,
                    inference_batch_size=config.inference_batch_size,
                    inference_batch_wait_ms=config.inference_batch_wait_ms,
                )
            else:
                policy = TorchRolloutPolicy(
                    torch,
                    model,
                    device,
                    schema_version,
                    centralized_critic=True,
                    inference_batch_size=config.inference_batch_size,
                    inference_batch_wait_ms=config.inference_batch_wait_ms,
                )
                rollout_jobs = _rollout_jobs_for_update(
                    config,
                    policy,
                    update_index=update_index,
                    device_name=str(device),
                    reward_shaping_weight=shaping_weight,
                )
                try:
                    rollouts = _collect_rollout_jobs(rollout_jobs, workers=config.rollout_workers)
                finally:
                    policy.close()
                inference_profile = policy.inference_profile()
        finally:
            if was_training:
                model.train()
        for rollout in rollouts:
            if rollout.stopped_reason not in {"max_deals", "match_complete"}:
                raise RuntimeError(f"rollout stopped unexpectedly: {rollout.stopped_reason}")
        transitions: list[RolloutTransition] = [transition for rollout in rollouts for transition in rollout.transitions]
        rollout_elapsed = time.perf_counter() - rollout_started_at
        total_transitions += len(transitions)
        returns, advantages = _gae_returns_and_advantages(
            tuple(transitions),
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            normalize=config.normalize_advantages,
        )
        mean_return = sum(returns) / len(returns) if returns else 0.0
        if config.log_updates:
            rollout_metrics = _rollout_metrics(rollouts)
            print(
                f"ppo update={update_index + 1}/{config.updates} rollout "
                f"seeds={len(rollouts)} transitions={len(transitions)} "
                f"deals={rollout_metrics.completed_deals} steps={rollout_metrics.steps} "
                f"stops={_format_stop_counts(rollout_metrics.stopped_reasons)} "
                f"mean_return={mean_return:.4f} elapsed={rollout_elapsed:.2f}s "
                f"profile={_format_rollout_profile(rollout_metrics.profile)} "
                f"inference={_format_policy_inference_profile(inference_profile)}",
                flush=True,
            )
        mean_kl = 0.0
        train_elapsed = 0.0
        if transitions:
            model.train()
            indices = list(range(len(transitions)))
            train_started_at = time.perf_counter()
            for epoch_index in range(config.epochs_per_update):
                epoch_rng = random.Random(f"{config.seed}:{update_index}:{epoch_index}")
                if config.candidate_bucket_batches:
                    batch_iterator = _iter_candidate_bucket_batches(
                        transitions,
                        indices,
                        config.batch_size,
                        epoch_rng,
                    )
                else:
                    epoch_indices = list(indices)
                    epoch_rng.shuffle(epoch_indices)
                    batch_iterator = _iter_batches(epoch_indices, config.batch_size)
                total_loss = 0.0
                total_kl = 0.0
                batches = 0
                for batch_indices in batch_iterator:
                    batch_transitions = tuple(transitions[index] for index in batch_indices)
                    batch_returns = tuple(returns[index] for index in batch_indices)
                    batch_advantages = tuple(advantages[index] for index in batch_indices)
                    loss, approx_kl = _ppo_batch_loss(
                        torch,
                        model,
                        batch_transitions,
                        batch_returns,
                        batch_advantages,
                        config,
                        device,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    if config.max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    total_loss += float(loss.detach().cpu())
                    total_kl += approx_kl
                    batches += 1
                final_loss = total_loss / max(batches, 1)
                mean_kl = total_kl / max(batches, 1)
                early_stop = config.target_kl > 0.0 and mean_kl > config.target_kl
                if config.log_updates:
                    stop_note = " early_stop=target_kl" if early_stop else ""
                    print(
                        f"ppo update={update_index + 1}/{config.updates} "
                        f"epoch={epoch_index + 1}/{config.epochs_per_update} "
                        f"batches={batches} loss={final_loss:.4f} kl={mean_kl:.5f}{stop_note}",
                        flush=True,
                    )
                if early_stop:
                    break
            train_elapsed = time.perf_counter() - train_started_at
        if config.log_updates:
            update_elapsed = time.perf_counter() - update_started_at
            print(
                f"ppo update={update_index + 1}/{config.updates} done "
                f"transitions={len(transitions)} loss={final_loss:.4f} "
                f"mean_return={mean_return:.4f} kl={mean_kl:.5f} "
                f"total_transitions={total_transitions} train_elapsed={train_elapsed:.2f}s "
                f"elapsed={update_elapsed:.2f}s",
                flush=True,
            )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "critic_observation_dim": value_input_dim,
            "critic_observation_names": critic_observation_names,
            "encoding_schema": encoding_schema(schema_version),
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "model_architecture": model_architecture,
            "centralized_critic": True,
            "updates": config.updates,
            "transitions": total_transitions,
            "final_loss": final_loss,
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "clip_epsilon": config.clip_epsilon,
            "entropy_coef": config.entropy_coef,
            "value_coef": config.value_coef,
            "batch_size": config.batch_size,
            "max_grad_norm": config.max_grad_norm,
            "target_kl": config.target_kl,
            "normalize_advantages": config.normalize_advantages,
              "opponent_pool": config.opponent_pool,
              "opponent_checkpoint_paths": tuple(str(path) for path in config.opponent_checkpoint_paths),
              "rollout_workers": config.rollout_workers,
              "rollout_processes": config.rollout_processes,
              "inference_batch_size": config.inference_batch_size,
              "inference_batch_wait_ms": config.inference_batch_wait_ms,
              "candidate_bucket_batches": config.candidate_bucket_batches,
              "reward_shaping_start": config.reward_shaping_start,
              "reward_shaping_end": config.reward_shaping_end,
            "init_policy_path": str(config.init_policy_path) if config.init_policy_path else None,
            "init_checkpoint_kind": init_checkpoint_kind,
        },
        config.output_path,
    )
    return PpoSummary(
        config.updates,
        total_transitions,
        final_loss,
        config.output_path,
        config.init_policy_path,
        init_checkpoint_kind,
    )


class TorchRolloutPolicy:
    def __init__(
        self,
        torch,
        model,
        device,
        schema_version: str = ENCODING_SCHEMA_VERSION,
        *,
        centralized_critic: bool = True,
        inference_batch_size: int = 1,
        inference_batch_wait_ms: float = 1.0,
    ) -> None:
        self.torch = torch
        self.model = model
        self.device = device
        self.schema_version = schema_version
        self.centralized_critic = centralized_critic
        self._runner = (
            _BatchedPolicyInferenceRunner(
                torch,
                model,
                device,
                max_batch_size=inference_batch_size,
                batch_wait_seconds=max(inference_batch_wait_ms, 0.0) / 1000.0,
            )
            if inference_batch_size > 1
            else None
        )

    def choose_decision(self, snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...]) -> RolloutDecision:
        return self.choose_decision_with_state(snapshot, actions, None)

    def choose_decision_with_state(
        self,
        snapshot: SeatSnapshot,
        actions: tuple[ActionCandidate, ...],
        state,
    ) -> RolloutDecision:
        policy_input = _encode_policy_input(
            snapshot,
            actions,
            self.schema_version,
            centralized_critic=self.centralized_critic,
            state=state,
        )
        if self._runner is not None:
            output = self._runner.submit(policy_input)
        else:
            inference_mode = getattr(self.torch, "inference_mode", self.torch.no_grad)
            with inference_mode():
                output = _infer_policy_batch(self.torch, self.model, (policy_input,), self.device)[0]
        return RolloutDecision(
            actions[output.action_index],
            output.action_index,
            log_prob=output.log_prob,
            value=output.value,
            encoded_features=policy_input.encoded_features,
        )

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close()

    def inference_profile(self) -> PolicyInferenceProfile:
        if self._runner is None:
            return PolicyInferenceProfile()
        return self._runner.profile()


@dataclass(frozen=True, slots=True)
class _PolicyInferenceInput:
    observation_values: tuple[float, ...]
    action_values: tuple[tuple[float, ...], ...]
    value_input_values: tuple[float, ...]
    encoded_features: RolloutEncodedFeatures


@dataclass(frozen=True, slots=True)
class _PolicyInferenceOutput:
    action_index: int
    log_prob: float
    value: float


@dataclass(frozen=True, slots=True)
class _PolicyInferenceRequest:
    policy_input: _PolicyInferenceInput
    future: object


class _BatchedPolicyInferenceRunner:
    _STOP = object()

    def __init__(self, torch, model, device, *, max_batch_size: int, batch_wait_seconds: float) -> None:
        self.torch = torch
        self.model = model
        self.device = device
        self.max_batch_size = max(max_batch_size, 1)
        self.batch_wait_seconds = max(batch_wait_seconds, 0.0)
        self._queue: queue.Queue[object] = queue.Queue()
        self._closed = False
        self._profile_lock = threading.Lock()
        self._profile = PolicyInferenceProfile()
        self._thread = threading.Thread(target=self._run, name="guandan-ppo-inference", daemon=True)
        self._thread.start()

    def submit(self, policy_input: _PolicyInferenceInput) -> _PolicyInferenceOutput:
        if self._closed:
            raise RuntimeError("batched policy inference runner is closed")
        future = _SimpleFuture()
        self._queue.put(_PolicyInferenceRequest(policy_input, future))
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._STOP)
        self._thread.join()

    def profile(self) -> PolicyInferenceProfile:
        with self._profile_lock:
            return self._profile

    def _run(self) -> None:
        inference_mode = getattr(self.torch, "inference_mode", self.torch.no_grad)
        stop_after_batch = False
        while True:
            item = self._queue.get()
            if item is self._STOP:
                return
            batch = [item]
            deadline = time.perf_counter() + self.batch_wait_seconds
            while len(batch) < self.max_batch_size:
                timeout = deadline - time.perf_counter()
                if timeout <= 0.0:
                    break
                try:
                    item = self._queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if item is self._STOP:
                    stop_after_batch = True
                    break
                batch.append(item)
            self._process_batch(batch, inference_mode)
            if stop_after_batch:
                return

    def _process_batch(self, batch: list[object], inference_mode) -> None:
        requests = [item for item in batch if isinstance(item, _PolicyInferenceRequest)]
        if not requests:
            return
        started_at = time.perf_counter()
        try:
            with inference_mode():
                outputs = _infer_policy_batch(
                    self.torch,
                    self.model,
                    tuple(request.policy_input for request in requests),
                    self.device,
                )
        except Exception as exc:
            for request in requests:
                request.future.set_exception(exc)
            return
        elapsed = time.perf_counter() - started_at
        self._record_batch(len(requests), elapsed)
        for request, output in zip(requests, outputs):
            request.future.set_result(output)

    def _record_batch(self, batch_size: int, elapsed: float) -> None:
        with self._profile_lock:
            previous = self._profile
            self._profile = PolicyInferenceProfile(
                requests=previous.requests + batch_size,
                batches=previous.batches + 1,
                max_batch_size=max(previous.max_batch_size, batch_size),
                inference_seconds=previous.inference_seconds + elapsed,
            )


class _SimpleFuture:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._result: object | None = None
        self._exception: BaseException | None = None

    def set_result(self, result: object) -> None:
        self._result = result
        self._event.set()

    def set_exception(self, exception: BaseException) -> None:
        self._exception = exception
        self._event.set()

    def result(self) -> _PolicyInferenceOutput:
        self._event.wait()
        if self._exception is not None:
            raise self._exception
        if not isinstance(self._result, _PolicyInferenceOutput):
            raise RuntimeError("policy inference returned an invalid result")
        return self._result


@dataclass(frozen=True, slots=True)
class _RemoteInferenceRequest:
    response_id: int
    request_id: int
    policy_input: _PolicyInferenceInput


@dataclass(frozen=True, slots=True)
class _RemoteInferenceResponse:
    request_id: int
    output: _PolicyInferenceOutput | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _ProcessRolloutMessage:
    index: int
    result: RolloutResult | None = None
    error: str | None = None


class _MultiprocessPolicyInferenceServer:
    def __init__(
        self,
        *,
        torch,
        model,
        device,
        request_queue,
        response_queues: dict[int, object],
        max_batch_size: int,
        batch_wait_seconds: float,
    ) -> None:
        self.torch = torch
        self.model = model
        self.device = device
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.max_batch_size = max(max_batch_size, 1)
        self.batch_wait_seconds = max(batch_wait_seconds, 0.0)
        self._profile_lock = threading.Lock()
        self._profile = PolicyInferenceProfile()
        self._thread = threading.Thread(target=self._run, name="guandan-ppo-mp-inference", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.request_queue.put(None)
        self._thread.join()

    def profile(self) -> PolicyInferenceProfile:
        with self._profile_lock:
            return self._profile

    def _run(self) -> None:
        inference_mode = getattr(self.torch, "inference_mode", self.torch.no_grad)
        stop_after_batch = False
        while True:
            item = self.request_queue.get()
            if item is None:
                return
            batch = [item]
            deadline = time.perf_counter() + self.batch_wait_seconds
            while len(batch) < self.max_batch_size:
                timeout = deadline - time.perf_counter()
                if timeout <= 0.0:
                    break
                try:
                    item = self.request_queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if item is None:
                    stop_after_batch = True
                    break
                batch.append(item)
            self._process_batch(batch, inference_mode)
            if stop_after_batch:
                return

    def _process_batch(self, batch: list[object], inference_mode) -> None:
        requests = [item for item in batch if isinstance(item, _RemoteInferenceRequest)]
        if not requests:
            return
        started_at = time.perf_counter()
        try:
            with inference_mode():
                outputs = _infer_policy_batch(
                    self.torch,
                    self.model,
                    tuple(request.policy_input for request in requests),
                    self.device,
                )
        except Exception as exc:
            for request in requests:
                self._respond(request, _RemoteInferenceResponse(request.request_id, error=f"{type(exc).__name__}: {exc}"))
            return
        elapsed = time.perf_counter() - started_at
        self._record_batch(len(requests), elapsed)
        for request, output in zip(requests, outputs):
            self._respond(request, _RemoteInferenceResponse(request.request_id, output=output))

    def _respond(self, request: _RemoteInferenceRequest, response: _RemoteInferenceResponse) -> None:
        response_queue = self.response_queues.get(request.response_id)
        if response_queue is None:
            return
        response_queue.put(response)

    def _record_batch(self, batch_size: int, elapsed: float) -> None:
        with self._profile_lock:
            previous = self._profile
            self._profile = PolicyInferenceProfile(
                requests=previous.requests + batch_size,
                batches=previous.batches + 1,
                max_batch_size=max(previous.max_batch_size, batch_size),
                inference_seconds=previous.inference_seconds + elapsed,
            )


class _RemoteCurrentPolicy:
    def __init__(self, request_queue, response_queue, response_id: int, schema_version: str, *, centralized_critic: bool) -> None:
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.response_id = response_id
        self.schema_version = schema_version
        self.centralized_critic = centralized_critic
        self._next_request_id = 0

    def choose_decision(self, snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...]) -> RolloutDecision:
        return self.choose_decision_with_state(snapshot, actions, None)

    def choose_decision_with_state(self, snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...], state) -> RolloutDecision:
        policy_input = _encode_policy_input(
            snapshot,
            actions,
            self.schema_version,
            centralized_critic=self.centralized_critic,
            state=state,
        )
        request_id = self._next_request_id
        self._next_request_id += 1
        self.request_queue.put(_RemoteInferenceRequest(self.response_id, request_id, policy_input))
        response = self.response_queue.get()
        if not isinstance(response, _RemoteInferenceResponse):
            raise RuntimeError("remote policy inference returned an invalid response")
        if response.request_id != request_id:
            raise RuntimeError("remote policy inference response id mismatch")
        if response.error is not None:
            raise RuntimeError(response.error)
        if response.output is None:
            raise RuntimeError("remote policy inference returned no output")
        output = response.output
        return RolloutDecision(
            actions[output.action_index],
            output.action_index,
            log_prob=output.log_prob,
            value=output.value,
            encoded_features=policy_input.encoded_features,
        )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        rollout_seeds = _rollout_seeds_from_args(args.seed, args.seed_count)
    except ValueError as exc:
        parser.error(str(exc))
    summary = train_self_play_ppo(
        PpoConfig(
            output_path=Path(args.output),
            init_policy_path=Path(args.init_policy) if args.init_policy else None,
            rollout_seeds=rollout_seeds,
            updates=args.updates,
            epochs_per_update=args.epochs_per_update,
            max_deals_per_seed=args.max_deals,
            max_steps_per_seed=args.max_steps,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            batch_size=args.batch_size,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            normalize_advantages=not args.no_normalize_advantages,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
              opponent_pool=tuple(args.opponent_pool.split(",")) if args.opponent_pool else (),
              opponent_checkpoint_paths=tuple(Path(path) for path in (args.opponent_checkpoint or ())),
              rollout_workers=args.rollout_workers,
              rollout_processes=args.rollout_processes,
              inference_batch_size=args.inference_batch_size,
              inference_batch_wait_ms=args.inference_batch_wait_ms,
              candidate_bucket_batches=not args.no_candidate_bucket_batches,
              reward_shaping_start=args.reward_shaping_start,
              reward_shaping_end=args.reward_shaping_end,
              seed=args.torch_seed,
            device=args.device,
            log_updates=not args.quiet,
        )
    )
    init_note = (
        f" init_checkpoint={summary.init_policy_path} kind={summary.init_checkpoint_kind}"
        if summary.init_policy_path
        else ""
    )
    print(
        f"ppo updates={summary.updates} transitions={summary.transitions} "
        f"loss={summary.final_loss:.4f}{init_note}; wrote {summary.output_path}",
        flush=True,
    )
    return 0


def _initialize_model_from_checkpoint(
    torch,
    model,
    checkpoint_path: Path,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
    device,
    schema_version: str,
    value_input_dim: int,
) -> str:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    kind, model_state = _initial_model_state_from_checkpoint(
        checkpoint,
        observation_dim,
        action_dim,
        hidden_dim,
        schema_version=schema_version,
        value_input_dim=value_input_dim,
    )
    if kind == "ppo":
        model.load_state_dict(model_state)
    else:
        model.policy_net.load_state_dict(model_state)
    return kind


def _policy_state_from_bc_checkpoint(
    checkpoint: object,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
) -> dict[str, object]:
    kind, model_state = _initial_model_state_from_checkpoint(
        checkpoint,
        observation_dim,
        action_dim,
        hidden_dim,
        schema_version=ENCODING_SCHEMA_VERSION,
    )
    if kind != "bc":
        raise ValueError("checkpoint is not a BC ranker checkpoint")
    return model_state


def _initial_model_state_from_checkpoint(
    checkpoint: object,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
    *,
    schema_version: str = ENCODING_SCHEMA_VERSION,
    value_input_dim: int | None = None,
) -> tuple[str, dict[str, object]]:
    if not isinstance(checkpoint, dict):
        raise ValueError("initial checkpoint must be a dictionary")
    _validate_checkpoint_dim(checkpoint, "observation_dim", observation_dim)
    _validate_checkpoint_dim(checkpoint, "action_dim", action_dim)
    _validate_checkpoint_dim(checkpoint, "hidden_dim", hidden_dim)
    validate_encoding_schema(checkpoint, schema_version=schema_version)
    _validate_checkpoint_architecture(checkpoint)
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("initial checkpoint is missing model_state")
    if _model_state_is_ppo(model_state):
        if checkpoint.get("centralized_critic") is not True:
            raise ValueError("initial PPO checkpoint must use centralized_critic=True")
        if value_input_dim is None:
            raise ValueError("value_input_dim is required for PPO checkpoint initialization")
        _validate_checkpoint_dim(checkpoint, "critic_observation_dim", value_input_dim)
        return "ppo", model_state
    policy_state = _bc_ranker_state_to_policy_state(model_state)
    if policy_state:
        return "bc", policy_state
    raise ValueError("initial checkpoint model_state is neither PPO actor-critic nor BC ranker")


def _validate_checkpoint_dim(checkpoint: dict[str, object], key: str, expected: int) -> None:
    actual = checkpoint.get(key)
    if actual != expected:
        raise ValueError(f"initial checkpoint {key}={actual!r} does not match PPO {key}={expected!r}")


def _validate_checkpoint_architecture(checkpoint: dict[str, object]) -> None:
    actual = checkpoint.get("model_architecture")
    if actual != DEFAULT_MODEL_ARCHITECTURE:
        raise ValueError(
            f"initial checkpoint model_architecture={actual!r} "
            f"does not match PPO model_architecture={DEFAULT_MODEL_ARCHITECTURE!r}"
        )


def _bc_ranker_state_to_policy_state(model_state: dict[str, object]) -> dict[str, object]:
    return {
        str(key).removeprefix("policy_net."): value
        for key, value in model_state.items()
        if str(key).startswith("policy_net.") and not str(key).startswith("policy_net.value_net.")
    }


def _checkpoint_is_ppo(checkpoint: dict[str, object]) -> bool:
    model_state = checkpoint.get("model_state")
    return isinstance(model_state, dict) and _model_state_is_ppo(model_state)


def _model_state_is_ppo(model_state: dict[str, object]) -> bool:
    return any(str(key).startswith("policy_net.") for key in model_state) and any(
        str(key).startswith("value_net.") for key in model_state
    )


def _rollout_metrics(rollouts: list[RolloutResult]) -> RolloutMetrics:
    stopped_reasons: dict[str, int] = {}
    for rollout in rollouts:
        stopped_reasons[rollout.stopped_reason] = stopped_reasons.get(rollout.stopped_reason, 0) + 1
    return RolloutMetrics(
        completed_deals=sum(rollout.completed_deals for rollout in rollouts),
        steps=sum(rollout.steps for rollout in rollouts),
        stopped_reasons=stopped_reasons,
        profile=_sum_rollout_profiles(rollouts),
    )


def _format_stop_counts(stopped_reasons: dict[str, int]) -> str:
    if not stopped_reasons:
        return "none"
    return ",".join(f"{reason}:{count}" for reason, count in sorted(stopped_reasons.items()))


def _sum_rollout_profiles(rollouts: list[RolloutResult]) -> RolloutProfile:
    return RolloutProfile(
        decisions=sum(rollout.profile.decisions for rollout in rollouts),
        recorded_transitions=sum(rollout.profile.recorded_transitions for rollout in rollouts),
        candidate_count_total=sum(rollout.profile.candidate_count_total for rollout in rollouts),
        candidate_count_max=max((rollout.profile.candidate_count_max for rollout in rollouts), default=0),
        encoded_transition_reuses=sum(rollout.profile.encoded_transition_reuses for rollout in rollouts),
        encoded_transition_misses=sum(rollout.profile.encoded_transition_misses for rollout in rollouts),
        legal_action_seconds=sum(rollout.profile.legal_action_seconds for rollout in rollouts),
        policy_seconds=sum(rollout.profile.policy_seconds for rollout in rollouts),
        critic_encode_seconds=sum(rollout.profile.critic_encode_seconds for rollout in rollouts),
        transition_encode_seconds=sum(rollout.profile.transition_encode_seconds for rollout in rollouts),
        env_step_seconds=sum(rollout.profile.env_step_seconds for rollout in rollouts),
    )


def _format_rollout_profile(profile: RolloutProfile) -> str:
    if profile.decisions == 0:
        return "none"
    return (
        f"legal={profile.legal_action_seconds:.2f}s "
        f"policy={profile.policy_seconds:.2f}s "
        f"critic={profile.critic_encode_seconds:.2f}s "
        f"transition={profile.transition_encode_seconds:.2f}s "
        f"step={profile.env_step_seconds:.2f}s "
        f"candidates={profile.average_candidate_count:.1f}/{profile.candidate_count_max} "
        f"encoded_reuse={profile.encoded_transition_reuses}/{profile.recorded_transitions}"
    )


def _format_policy_inference_profile(profile: PolicyInferenceProfile) -> str:
    if profile.requests == 0:
        return "direct"
    return (
        f"requests={profile.requests} batches={profile.batches} "
        f"avg_batch={profile.average_batch_size:.1f} max_batch={profile.max_batch_size} "
        f"forward={profile.inference_seconds:.2f}s"
    )


def _rollout_jobs_for_update(
    config: PpoConfig,
    current_policy: TorchRolloutPolicy,
    *,
    update_index: int,
    device_name: str,
    reward_shaping_weight: float,
) -> list[RolloutJob]:
    opponents = _opponent_policies(config, device_name=device_name)
    jobs: list[RolloutJob] = []
    for seed in config.rollout_seeds:
        base_seed = f"{seed}:update:{update_index}"
        for opponent_name, opponent_policy in opponents:
            if opponent_name == "self":
                jobs.append(
                    RolloutJob(
                        policies={seat: current_policy for seat in SEATS},
                        seed=f"{base_seed}:self",
                        max_deals=config.max_deals_per_seed,
                        max_steps=config.max_steps_per_seed,
                        record_seats=None,
                        reward_shaping_weight=reward_shaping_weight,
                    )
                )
                continue
            for candidate_team in (Team.EAST_WEST, Team.SOUTH_NORTH):
                record_seats = frozenset(seat for seat in SEATS if team_for_seat(seat) == candidate_team)
                policies = {
                    seat: current_policy if seat in record_seats else opponent_policy
                    for seat in SEATS
                }
                jobs.append(
                    RolloutJob(
                        policies=policies,
                        seed=f"{base_seed}:{opponent_name}:{candidate_team.value}",
                        max_deals=config.max_deals_per_seed,
                        max_steps=config.max_steps_per_seed,
                        record_seats=record_seats,
                        reward_shaping_weight=reward_shaping_weight,
                    )
                )
    return jobs


def _rollout_process_jobs_for_update(
    config: PpoConfig,
    *,
    update_index: int,
    device_name: str,
    reward_shaping_weight: float,
    schema_version: str,
) -> list[ProcessRolloutJob]:
    opponents = _opponent_specs(config, device_name=device_name)
    jobs: list[ProcessRolloutJob] = []
    all_current_seats = frozenset(SEATS)
    for seed in config.rollout_seeds:
        base_seed = f"{seed}:update:{update_index}"
        for opponent_name, checkpoint_path, opponent_device_name in opponents:
            if opponent_name == "self":
                jobs.append(
                    ProcessRolloutJob(
                        seed=f"{base_seed}:self",
                        max_deals=config.max_deals_per_seed,
                        max_steps=config.max_steps_per_seed,
                        record_seats=None,
                        reward_shaping_weight=reward_shaping_weight,
                        current_policy_seats=all_current_seats,
                        opponent_name="self",
                        opponent_checkpoint_path=None,
                        opponent_device_name=opponent_device_name,
                        schema_version=schema_version,
                    )
                )
                continue
            for candidate_team in (Team.EAST_WEST, Team.SOUTH_NORTH):
                current_policy_seats = frozenset(seat for seat in SEATS if team_for_seat(seat) == candidate_team)
                jobs.append(
                    ProcessRolloutJob(
                        seed=f"{base_seed}:{opponent_name}:{candidate_team.value}",
                        max_deals=config.max_deals_per_seed,
                        max_steps=config.max_steps_per_seed,
                        record_seats=current_policy_seats,
                        reward_shaping_weight=reward_shaping_weight,
                        current_policy_seats=current_policy_seats,
                        opponent_name=opponent_name,
                        opponent_checkpoint_path=checkpoint_path,
                        opponent_device_name=opponent_device_name,
                        schema_version=schema_version,
                    )
                )
    return jobs


def _collect_rollout_jobs(jobs: list[RolloutJob], *, workers: int) -> list[RolloutResult]:
    if workers <= 1 or len(jobs) <= 1:
        return [_collect_rollout_job(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_collect_rollout_job, jobs))


def _collect_rollout_job(job: RolloutJob) -> RolloutResult:
    return collect_rollout(
        job.policies,
        seed=job.seed,
        max_deals=job.max_deals,
        max_steps=job.max_steps,
        record_seats=job.record_seats,
        reward_shaping_weight=job.reward_shaping_weight,
      )


def _collect_rollout_process_jobs(
    jobs: list[ProcessRolloutJob],
    *,
    processes: int,
    torch,
    model,
    device,
    inference_batch_size: int,
    inference_batch_wait_ms: float,
) -> tuple[list[RolloutResult], PolicyInferenceProfile]:
    if not jobs:
        return [], PolicyInferenceProfile()
    process_count = max(1, min(processes, len(jobs)))
    context = mp.get_context("spawn")
    request_queue = context.Queue()
    result_queue = context.Queue()
    response_queues = {index: context.Queue() for index in range(len(jobs))}
    server = _MultiprocessPolicyInferenceServer(
        torch=torch,
        model=model,
        device=device,
        request_queue=request_queue,
        response_queues=response_queues,
        max_batch_size=inference_batch_size,
        batch_wait_seconds=max(inference_batch_wait_ms, 0.0) / 1000.0,
    )
    active: dict[int, object] = {}
    results: list[RolloutResult | None] = [None for _ in jobs]
    next_index = 0
    try:
        next_index = _start_rollout_processes(
            context,
            jobs,
            request_queue,
            response_queues,
            result_queue,
            active,
            next_index,
            process_count,
        )
        completed = 0
        while completed < len(jobs):
            try:
                message = result_queue.get(timeout=0.1)
            except queue.Empty:
                _raise_for_dead_process(active)
                continue
            if not isinstance(message, _ProcessRolloutMessage):
                raise RuntimeError("rollout process returned an invalid message")
            process = active.pop(message.index, None)
            if process is not None:
                process.join()
            if message.error is not None:
                raise RuntimeError(message.error)
            if message.result is None:
                raise RuntimeError("rollout process returned no result")
            results[message.index] = message.result
            completed += 1
            next_index = _start_rollout_processes(
                context,
                jobs,
                request_queue,
                response_queues,
                result_queue,
                active,
                next_index,
                process_count,
            )
    except Exception:
        for process in active.values():
            if process.is_alive():
                process.terminate()
        for process in active.values():
            process.join()
        raise
    finally:
        server.close()
    return [result for result in results if result is not None], server.profile()


def _start_rollout_processes(
    context,
    jobs: list[ProcessRolloutJob],
    request_queue,
    response_queues: dict[int, object],
    result_queue,
    active: dict[int, object],
    next_index: int,
    process_count: int,
) -> int:
    while next_index < len(jobs) and len(active) < process_count:
        response_queue = response_queues[next_index]
        process = context.Process(
            target=_collect_rollout_process_entry,
            args=(next_index, jobs[next_index], request_queue, response_queue, result_queue),
        )
        process.start()
        active[next_index] = process
        next_index += 1
    return next_index


def _raise_for_dead_process(active: dict[int, object]) -> None:
    for index, process in tuple(active.items()):
        if not process.is_alive() and process.exitcode not in {None, 0}:
            process.join()
            active.pop(index, None)
            raise RuntimeError(f"rollout process {index} exited with code {process.exitcode}")


def _collect_rollout_process_entry(index: int, job: ProcessRolloutJob, request_queue, response_queue, result_queue) -> None:
    try:
        current_policy = _RemoteCurrentPolicy(
            request_queue,
            response_queue,
            index,
            job.schema_version,
            centralized_critic=job.centralized_critic,
        )
        opponent_policy = _process_opponent_policy(job)
        policies = {
            seat: current_policy if seat in job.current_policy_seats else opponent_policy
            for seat in SEATS
        }
        result = collect_rollout(
            policies,
            seed=job.seed,
            max_deals=job.max_deals,
            max_steps=job.max_steps,
            record_seats=job.record_seats,
            reward_shaping_weight=job.reward_shaping_weight,
        )
    except Exception as exc:
        result_queue.put(_ProcessRolloutMessage(index, error=f"{type(exc).__name__}: {exc}"))
        return
    result_queue.put(_ProcessRolloutMessage(index, result=result))


def _process_opponent_policy(job: ProcessRolloutJob):
    if job.opponent_name == "self":
        return _DummyPolicy()
    if job.opponent_name == "heuristic":
        return _ActionPolicyRolloutAdapter(HeuristicPolicy())
    if job.opponent_name == "dummy":
        return _ActionPolicyRolloutAdapter(_DummyPolicy())
    if job.opponent_checkpoint_path is not None:
        return _FrozenCheckpointRolloutPolicy(job.opponent_checkpoint_path, job.opponent_device_name)
    raise ValueError(f"unsupported process opponent: {job.opponent_name}")


def _opponent_policies(config: PpoConfig, *, device_name: str) -> list[tuple[str, object]]:
    opponents: list[tuple[str, object]] = []
    for token, checkpoint_path, opponent_device_name in _opponent_specs(
        config,
        device_name=device_name,
        checkpoint_device_name=device_name,
    ):
        if token == "self":
            opponents.append(("self", object()))
        elif token == "heuristic":
            opponents.append(("heuristic", _ActionPolicyRolloutAdapter(HeuristicPolicy())))
        elif token == "dummy":
            opponents.append(("dummy", _ActionPolicyRolloutAdapter(_DummyPolicy())))
        elif checkpoint_path is not None:
            opponents.append((token, _FrozenCheckpointRolloutPolicy(checkpoint_path, opponent_device_name)))
        else:
            raise ValueError(f"unsupported opponent pool entry: {token}")
    return opponents or [("self", object())]


def _opponent_specs(
    config: PpoConfig,
    *,
    device_name: str,
    checkpoint_device_name: str | None = None,
) -> list[tuple[str, Path | None, str]]:
    tokens = tuple(token.strip() for item in config.opponent_pool for token in item.split(",") if token.strip())
    if not tokens:
        tokens = ("self",)
    checkpoint_device = (
        checkpoint_device_name
        if checkpoint_device_name is not None
        else "cpu" if str(device_name).startswith("cuda") else device_name
    )
    opponents: list[tuple[str, Path | None, str]] = []
    for token in tokens:
        if token in {"self", "heuristic", "dummy"}:
            opponents.append((token, None, checkpoint_device))
        elif token == "previous" and config.init_policy_path is not None:
            opponents.append(("previous", config.init_policy_path, checkpoint_device))
        elif token == "previous":
            continue
        else:
            raise ValueError(f"unsupported opponent pool entry: {token}")
    for index, path in enumerate(config.opponent_checkpoint_paths):
        opponents.append((f"checkpoint{index}", path, checkpoint_device))
    return opponents


class _ActionPolicyRolloutAdapter:
    def __init__(self, policy) -> None:
        self.policy = policy

    def choose_decision(self, snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...]) -> RolloutDecision:
        action = self.policy.choose_action(snapshot, actions)
        return RolloutDecision(action=action, action_index=actions.index(action), log_prob=0.0, value=0.0)


class _FrozenCheckpointRolloutPolicy:
    def __init__(self, checkpoint_path: Path, device_name: str) -> None:
        from npc.rl_agent.model_loader import RlAgentConfig, RlModelLoader

        self.loader = RlModelLoader(RlAgentConfig(model_path=checkpoint_path, device=device_name))
        self.fallback = HeuristicPolicy()

    def choose_decision(self, snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...]) -> RolloutDecision:
        action = self.loader.choose_action(snapshot, actions) or self.fallback.choose_action(snapshot, actions)
        return RolloutDecision(action=action, action_index=actions.index(action), log_prob=0.0, value=0.0)


class _DummyPolicy:
    def choose_action(self, snapshot: SeatSnapshot, actions: tuple[ActionCandidate, ...]) -> ActionCandidate:
        if snapshot.legal_action == "play_or_pass":
            return next((action for action in actions if action.kind == "pass"), actions[0])
        plays = tuple(action for action in actions if action.kind == "play_cards")
        return min(
            plays or actions,
            key=lambda action: (
                action.length,
                action.primary_rank.value if action.primary_rank else "",
                action.card_ids,
            ),
        )


def _linear_decay(start: float, end: float, index: int, count: int) -> float:
    if count <= 1:
        return start
    fraction = min(max(index / (count - 1), 0.0), 1.0)
    return start + (end - start) * fraction


def _ppo_batch_loss(
    torch,
    model,
    transitions: tuple[RolloutTransition, ...],
    returns: tuple[float, ...],
    advantages: tuple[float, ...],
    config: PpoConfig,
    device,
):
    logits, values, mask = _transition_batch_logits_and_values(
        torch,
        model,
        transitions,
        device,
    )
    log_probs = torch.log_softmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)
    action_indices = torch.tensor(
        [transition.action_index for transition in transitions],
        dtype=torch.long,
        device=device,
    )
    new_log_probs = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
    old_log_probs = torch.tensor(
        [transition.old_log_prob for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    target_returns = torch.tensor(returns, dtype=torch.float32, device=device)
    advantage_values = torch.tensor(advantages, dtype=torch.float32, device=device)
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
    policy_loss = -torch.minimum(ratio * advantage_values, clipped * advantage_values).mean()
    value_loss = (values - target_returns).pow(2).mean()
    entropy = -(probs * log_probs.masked_fill(~mask, 0.0)).sum(dim=1).mean()
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
    approx_kl = ((ratio - 1.0) - log_ratio).mean().detach()
    return loss, float(approx_kl.cpu())


def _gae_returns_and_advantages(
    transitions: tuple[RolloutTransition, ...],
    *,
    gamma: float,
    gae_lambda: float,
    normalize: bool,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    returns = [0.0 for _ in transitions]
    advantages = [0.0 for _ in transitions]
    next_advantage_by_seat = {seat.value: 0.0 for seat in SEATS}
    next_value_by_seat = {seat.value: 0.0 for seat in SEATS}
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        if transition.done:
            next_advantage = 0.0
            next_value = 0.0
        else:
            next_advantage = next_advantage_by_seat[transition.seat]
            next_value = next_value_by_seat[transition.seat]
        delta = transition.reward + gamma * next_value - transition.value
        advantage = delta + gamma * gae_lambda * next_advantage
        advantages[index] = advantage
        returns[index] = advantage + transition.value
        next_advantage_by_seat[transition.seat] = advantage
        next_value_by_seat[transition.seat] = transition.value
    if normalize and len(advantages) > 1:
        mean = sum(advantages) / len(advantages)
        variance = sum((advantage - mean) ** 2 for advantage in advantages) / len(advantages)
        std = variance**0.5
        if std > 1e-8:
            advantages = [(advantage - mean) / std for advantage in advantages]
    return tuple(returns), tuple(advantages)


def _iter_batches(indices: list[int], batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _candidate_count_bucket(candidate_count: int) -> int:
    if candidate_count <= 1:
        return max(candidate_count, 0)
    return 1 << (candidate_count - 1).bit_length()


def _iter_candidate_bucket_batches(
    transitions: list[RolloutTransition],
    indices: list[int],
    batch_size: int,
    rng: random.Random,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    buckets: dict[int, list[int]] = {}
    for index in indices:
        bucket = _candidate_count_bucket(len(transitions[index].candidate_values))
        buckets.setdefault(bucket, []).append(index)
    bucket_keys = list(buckets)
    rng.shuffle(bucket_keys)
    batches: list[list[int]] = []
    for bucket_key in bucket_keys:
        bucket_indices = buckets[bucket_key]
        rng.shuffle(bucket_indices)
        batches.extend(_iter_batches(bucket_indices, batch_size))
    rng.shuffle(batches)
    yield from batches


def _checkpoint_schema_version_for_training(torch, checkpoint_path: Path | None, device) -> str:
    if checkpoint_path is None:
        return ENCODING_SCHEMA_VERSION
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("initial checkpoint must be a dictionary")
    return validate_encoding_schema(checkpoint)


def _initial_dimensions(seed: str, *, schema_version: str = ENCODING_SCHEMA_VERSION) -> tuple[int, int, int, tuple[str, ...]]:
    from training.env import GuandanTrainingEnv

    env = GuandanTrainingEnv()
    env.reset(seed=seed)
    actor = env.current_actor()
    if actor is None:
        raise RuntimeError("initial training environment has no actor")
    snapshot = env.observe(actor)
    actions = env.legal_actions(actor)
    if not actions:
        raise RuntimeError("initial training environment has no legal actions")
    observation_dim = len(encode_observation(snapshot, schema_version=schema_version).values)
    action_dim = len(encode_action(actions[0], snapshot, schema_version=schema_version).values)
    critic_observation = encode_critic_observation(env.state, actor, schema_version=schema_version)
    return observation_dim, action_dim, len(critic_observation.values), critic_observation.names


def _encode_policy_input(
    snapshot: SeatSnapshot,
    actions: tuple[ActionCandidate, ...],
    schema_version: str,
    *,
    centralized_critic: bool = True,
    state=None,
) -> _PolicyInferenceInput:
    if not actions:
        raise ValueError("policy inference requires at least one legal action")
    observation_vector = encode_observation(snapshot, schema_version=schema_version)
    action_vectors = tuple(encode_action(action, snapshot, schema_version=schema_version) for action in actions)
    action_values = tuple(vector.values for vector in action_vectors)
    if centralized_critic and state is not None:
        critic_observation = encode_critic_observation(state, snapshot.seat, schema_version=schema_version)
        value_input_values = critic_observation.values
        critic_observation_values = critic_observation.values
    else:
        value_input_values = observation_vector.values
        critic_observation_values = ()
    encoded_features = RolloutEncodedFeatures(
        observation_names=observation_vector.names,
        observation_values=observation_vector.values,
        action_names=action_vectors[0].names if action_vectors else (),
        candidate_values=action_values,
        critic_observation_values=critic_observation_values,
    )
    return _PolicyInferenceInput(
        observation_values=observation_vector.values,
        action_values=action_values,
        value_input_values=value_input_values,
        encoded_features=encoded_features,
    )


def _infer_policy_batch(torch, model, inputs: tuple[_PolicyInferenceInput, ...], device) -> tuple[_PolicyInferenceOutput, ...]:
    if not inputs:
        return ()
    logits, values = _policy_batch_logits_and_values(torch, model, inputs, device)
    distribution = torch.distributions.Categorical(logits=logits)
    indices = distribution.sample()
    log_probs = distribution.log_prob(indices)
    return tuple(
        _PolicyInferenceOutput(
            action_index=int(indices[row].item()),
            log_prob=float(log_probs[row].detach().cpu()),
            value=float(values[row].detach().cpu()),
        )
        for row in range(len(inputs))
    )


def _policy_batch_logits_and_values(torch, model, inputs: tuple[_PolicyInferenceInput, ...], device):
    if not inputs:
        raise ValueError("policy batch requires at least one inference input")
    observations = torch.tensor(
        [item.observation_values for item in inputs],
        dtype=torch.float32,
        device=device,
    )
    counts = torch.tensor([len(item.action_values) for item in inputs], dtype=torch.long, device=device)
    if bool((counts <= 0).any().item()):
        raise ValueError("policy inference input contains no legal actions")
    flat_actions = [action_values for item in inputs for action_values in item.action_values]
    action_features = torch.tensor(flat_actions, dtype=torch.float32, device=device)
    pair_observations = observations.repeat_interleave(counts, dim=0)
    flat_logits = model.policy_logits(torch.cat((pair_observations, action_features), dim=1))
    max_candidates = int(counts.max().item())
    logits = torch.full((len(inputs), max_candidates), -torch.inf, dtype=torch.float32, device=device)
    offset = 0
    for row, count_item in enumerate(counts.tolist()):
        logits[row, :count_item] = flat_logits[offset : offset + count_item]
        offset += count_item
    value_inputs = torch.tensor(
        [item.value_input_values for item in inputs],
        dtype=torch.float32,
        device=device,
    )
    values = model.value(value_inputs)
    return logits, values


def _transition_batch_logits_and_values(
    torch,
    model,
    transitions: tuple[RolloutTransition, ...],
    device,
):
    observations = torch.tensor(
        [transition.observation_values for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    counts = torch.tensor(
        [len(transition.candidate_values) for transition in transitions],
        dtype=torch.long,
        device=device,
    )
    flat_actions = [
        candidate_values
        for transition in transitions
        for candidate_values in transition.candidate_values
    ]
    action_features = torch.tensor(flat_actions, dtype=torch.float32, device=device)
    pair_observations = observations.repeat_interleave(counts, dim=0)
    flat_logits = model.policy_logits(torch.cat((pair_observations, action_features), dim=1))
    max_candidates = int(counts.max().item())
    logits = torch.full((len(transitions), max_candidates), -torch.inf, dtype=torch.float32, device=device)
    offset = 0
    for row, count_item in enumerate(counts.tolist()):
        logits[row, :count_item] = flat_logits[offset : offset + count_item]
        offset += count_item
    candidate_positions = torch.arange(max_candidates, device=device).unsqueeze(0)
    mask = candidate_positions < counts.unsqueeze(1)
    if any(not transition.critic_observation_values for transition in transitions):
        raise ValueError("centralized critic transitions are missing critic observation values")
    value_inputs = torch.tensor(
        [transition.critic_observation_values for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    return logits, model.value(value_inputs), mask


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Guandan candidate policy with self-play PPO.")
    parser.add_argument("output", help="Output checkpoint path.")
    parser.add_argument(
        "--init-policy",
        dest="init_policy",
        help="BC ranker or trained PPO actor-critic checkpoint used to initialize PPO training.",
    )
    parser.add_argument("--seed", action="append", help="Rollout seed. Can be repeated.")
    parser.add_argument("--seed-count", type=int, help="Generate rollout seeds ppo-seed-0..N-1.")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--epochs-per-update", type=int, default=3)
    parser.add_argument("--max-deals", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--no-normalize-advantages", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--opponent-pool", default="self")
    parser.add_argument("--opponent-checkpoint", action="append")
    parser.add_argument("--rollout-workers", type=int, default=1)
    parser.add_argument("--rollout-processes", type=int, default=0)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--inference-batch-wait-ms", type=float, default=1.0)
    parser.add_argument("--no-candidate-bucket-batches", action="store_true")
    parser.add_argument("--reward-shaping-start", type=float, default=0.02)
    parser.add_argument("--reward-shaping-end", type=float, default=0.0)
    parser.add_argument("--torch-seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true", help="Disable training progress output.")
    return parser


def _rollout_seeds_from_args(seed_args: list[str] | None, seed_count: int | None) -> tuple[str, ...]:
    if seed_args and seed_count is not None:
        raise ValueError("--seed-count cannot be combined with --seed")
    if seed_args:
        return tuple(seed_args)
    if seed_count is None:
        return ("ppo-seed-0",)
    if seed_count <= 0:
        raise ValueError("--seed-count must be positive")
    return tuple(f"ppo-seed-{index}" for index in range(seed_count))


if __name__ == "__main__":
    raise SystemExit(main())
