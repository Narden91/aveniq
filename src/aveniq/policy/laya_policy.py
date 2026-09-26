"""AVENIQ routing through a loaded Laya typed-decision model."""

from __future__ import annotations

import importlib.metadata
import os
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch

from .decision import PolicyDecision
from .engine import PolicyEngine
from .rule_policy import RulePolicy

PolicyMode = Literal["disabled", "shadow", "control"]
DEFAULT_CHECKPOINT = "convaiinnovations/laya-typed-decisions"
QUESTIONS = {
    "execution_class": {
        "type": "choice",
        "instructions": "Choose the least complex execution path that can satisfy this request.",
        "criteria": {
            "direct": "A greeting, acknowledgement, or simple fixed response needs no expert.",
            "single_expert": "One specialist can answer without coordinating other work.",
            "system2": "The request needs multiple dependent steps, tool use, or coordination.",
        },
    },
    "primary_expert": {
        "type": "choice",
        "instructions": "If one specialist handles this request, which specialist fits best?",
        "criteria": {
            "technical": "Programming, software, computing, or engineering.",
            "analytical": "Comparison, statistics, data analysis, or trade-offs.",
            "creative": "Stories, poems, marketing copy, or other creative writing.",
            "general": "General knowledge or explanation outside the other specialties.",
        },
    },
}
_MODEL_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _load_agent(checkpoint: str, device: Optional[str]):
    from laya import load

    if os.name == "nt" and not Path(checkpoint).exists():
        from huggingface_hub import snapshot_download

        local_dir = Path(
            os.environ.get(
                "AVENIQ_LAYA_CHECKPOINT_DIR",
                Path.home() / ".cache" / "aveniq" / checkpoint.replace("/", "--"),
            )
        )
        if (
            not (local_dir / "model.safetensors").exists()
            or not (local_dir / "rl_agent_config.json").exists()
        ):
            snapshot_download(
                checkpoint,
                local_dir=local_dir,
                allow_patterns=[
                    "rl_agent_config.json",
                    "model.safetensors",
                    "tokenizer/*",
                    "encoder/*",
                ],
            )
        checkpoint = str(local_dir)
    agent = load(checkpoint, device=device)
    parameter_device = next(agent.model.parameters()).device
    agent._aveniq_load_memory = (
        (
            torch.cuda.memory_allocated(parameter_device),
            torch.cuda.memory_reserved(parameter_device),
        )
        if parameter_device.type == "cuda"
        else (None, None)
    )
    return agent


class LayaInferenceEngine:
    """Run typed Laya decisions and measure the loaded model's forward call."""

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device_override: Optional[str] = None):
        self.checkpoint = checkpoint
        self.device_override = device_override

    @property
    def agent(self):
        return _load_agent(self.checkpoint, self.device_override)

    def model_metadata(self) -> Dict[str, Any]:
        agent = self.agent
        parameters = list(agent.model.parameters())
        if not parameters:
            raise RuntimeError("Laya checkpoint has no model parameters")
        devices = {parameter.device for parameter in parameters}
        if len(devices) != 1:
            raise RuntimeError(
                f"Laya reports {agent.device}, but parameters are on {sorted(map(str, devices))}"
            )
        parameter_device = next(iter(devices))
        if parameter_device.type != agent.device.type or (parameter_device.index or 0) != (
            agent.device.index or 0
        ):
            raise RuntimeError(
                f"Laya reports {agent.device}, but parameters are on {parameter_device}"
            )
        if (
            self.device_override
            and torch.device(self.device_override).type == "cuda"
            and agent.device.type != "cuda"
        ):
            raise RuntimeError("CUDA was requested, but Laya model parameters remain on CPU")
        checkpoint_path = None
        if Path(agent.model_id).exists():
            checkpoint_path = str(Path(agent.model_id).resolve())
        else:
            try:
                from huggingface_hub import try_to_load_from_cache

                weights = try_to_load_from_cache(self.checkpoint, "model.safetensors")
                if isinstance(weights, str):
                    checkpoint_path = str(Path(weights).parent)
            except (ImportError, ValueError):
                pass
        metadata_file = (
            Path(checkpoint_path)
            / ".cache"
            / "huggingface"
            / "download"
            / "model.safetensors.metadata"
            if checkpoint_path
            else None
        )
        checkpoint_revision = None
        checkpoint_etag = None
        if metadata_file and metadata_file.exists():
            lines = metadata_file.read_text(encoding="utf-8").splitlines()
            checkpoint_revision = lines[0] if lines else None
            checkpoint_etag = lines[1] if len(lines) > 1 else None
        cuda = parameter_device.type == "cuda"
        index = parameter_device.index or 0
        return {
            "laya_version": importlib.metadata.version("laya"),
            "checkpoint": self.checkpoint,
            "checkpoint_path": checkpoint_path,
            "checkpoint_revision": checkpoint_revision,
            "checkpoint_etag": checkpoint_etag,
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "device": str(parameter_device),
            "parameter_device": str(parameter_device),
            "dtype": str(agent.dtype),
            "parameter_dtype": str(parameters[0].dtype),
            "cuda_device_name": torch.cuda.get_device_name(index) if cuda else None,
            "cuda_compute_capability": (
                list(torch.cuda.get_device_capability(index)) if cuda else None
            ),
            "cuda_allocated_bytes_after_load": agent._aveniq_load_memory[0],
            "cuda_reserved_bytes_after_load": agent._aveniq_load_memory[1],
        }

    def predict(self, query: str) -> Dict[str, Any]:
        call_start = time.perf_counter()
        agent = self.agent
        model_meta = self.model_metadata()
        starts: list[float] = []
        forward_ms: list[float] = []
        cuda = agent.device.type == "cuda"

        def synchronize() -> None:
            if cuda:
                torch.cuda.synchronize(agent.device)

        def before_forward(_module, _inputs) -> None:
            synchronize()
            starts.append(time.perf_counter())

        def after_forward(_module, _inputs, _output) -> None:
            synchronize()
            forward_ms.append((time.perf_counter() - starts[-1]) * 1000)

        with _MODEL_LOCK:
            synchronize()
            pre_hook = agent.model.register_forward_pre_hook(before_forward)
            post_hook = agent.model.register_forward_hook(after_forward)
            try:
                result = agent.system_one(query, QUESTIONS)
            finally:
                pre_hook.remove()
                post_hook.remove()
            synchronize()
        if len(starts) != 1 or len(forward_ms) != 1:
            raise RuntimeError(f"Expected one Laya forward pass, observed {len(forward_ms)}")
        answers = result["answers"]
        execution = answers["execution_class"]
        expert = answers["primary_expert"]
        execution_class = execution["choice"]
        primary_expert = expert["choice"] if execution_class == "single_expert" else None
        confidence = (
            min(execution["answer_confidence"], expert["answer_confidence"])
            if primary_expert
            else execution["answer_confidence"]
        )
        return {
            **model_meta,
            "execution_class": execution_class,
            "primary_expert": primary_expert,
            "confidence": confidence,
            "probabilities": {
                "execution_class": execution["probabilities"],
                "primary_expert": expert["probabilities"],
            },
            "preprocessing_latency_ms": (starts[0] - call_start) * 1000,
            "forward_latency_ms": forward_ms[0],
            "total_policy_latency_ms": (time.perf_counter() - call_start) * 1000,
            "latency_ms": forward_ms[0],
            "backend": "laya.Agent.system_one",
            "token_usage": None,
        }


class LayaPolicy(PolicyEngine):
    """Use Laya in shadow or control mode; disabled mode uses RulePolicy only."""

    def __init__(
        self,
        mode: PolicyMode = "shadow",
        shadow_mode: Optional[bool] = None,
        confidence_threshold: float = 0.80,
        fallback_policy: Optional[PolicyEngine] = None,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: Optional[str] = None,
    ) -> None:
        self.mode = ("shadow" if shadow_mode else "control") if shadow_mode is not None else mode
        self.confidence_threshold = float(confidence_threshold)
        self.fallback_policy = fallback_policy or RulePolicy()
        self.checkpoint = checkpoint
        self._engine = LayaInferenceEngine(checkpoint, device)

    def predict_laya(self, query: str) -> Dict[str, Any]:
        return self._engine.predict(query)

    def evaluate(self, query: str, context: Optional[Dict[str, Any]] = None) -> PolicyDecision:
        if self.mode == "disabled":
            decision = self.fallback_policy.evaluate(query, context=context)
            record = {
                "checkpoint": None,
                "device": None,
                "latency_ms": None,
                "controlled_execution": False,
                "fallback_occurred": False,
                "token_usage": None,
            }
            decision.raw_predictions["laya_metadata"] = record
            decision.metadata["laya_metadata"] = record
            return decision

        try:
            prediction = self.predict_laya(query)
        except Exception as exc:
            decision = self.fallback_policy.evaluate(query, context=context)
            record = {
                "checkpoint": None,
                "device": None,
                "latency_ms": None,
                "controlled_execution": False,
                "fallback_occurred": True,
                "fallback_reason": f"inference_error: {exc}",
                "token_usage": None,
            }
            decision.raw_predictions["laya_metadata"] = record
            decision.metadata["laya_metadata"] = record
            return decision

        if self.mode == "shadow":
            decision = self.fallback_policy.evaluate(query, context=context)
            record = {
                **prediction,
                "controlled_execution": False,
                "fallback_occurred": False,
                "fallback_reason": None,
            }
            decision.raw_predictions.update(laya_shadow=record, laya_metadata=record, laya=record)
            decision.metadata.update(laya_shadow=record, laya_metadata=record)
            return decision

        confidence = float(prediction["confidence"])
        low_confidence = confidence < self.confidence_threshold
        execution_class = "system2" if low_confidence else prediction["execution_class"]
        record = {
            **prediction,
            "controlled_execution": True,
            "fallback_occurred": low_confidence,
            "fallback_reason": "low_confidence" if low_confidence else None,
        }
        return PolicyDecision(
            execution_class=execution_class,
            primary_expert=(
                prediction["primary_expert"] if execution_class == "single_expert" else None
            ),
            model_tier="strong" if execution_class == "system2" else "fast",
            needs_verification=execution_class == "system2",
            max_agent_calls=5 if execution_class == "system2" else 1,
            confidence=confidence,
            raw_predictions={"laya": record, "laya_control": record, "laya_metadata": record},
            metadata={"laya_metadata": record},
            reasoning=f"Laya control routing: {execution_class} (conf={confidence:.2f}).",
        )

    async def aevaluate(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> PolicyDecision:
        return self.evaluate(query, context=context)
