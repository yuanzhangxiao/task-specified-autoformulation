"""Fresh Phase-B D3 discovery with frozen native validation semantics."""

from __future__ import annotations

import fcntl
import importlib
import importlib.metadata
import json
import os
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.baselines.d3 import (
    D3_UPSTREAM_REVISION,
    run_d3_native_no_tools_development,
)
from autoformalism.baselines.d3_rollout import evaluate_validation
from autoformalism.baselines.models import BaselineConfig, BaselineDevelopmentResult
from autoformalism.data import BenchmarkRegistry
from autoformalism.llm.config import LLMConfig, LLMProvider, create_llm_client
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.baseline_validation import file_hash, load_public, read_json
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas import CandidateModel
from autoformalism.staged_topology import content_hash

PROTOCOL = "phase-b-d3-native-validation-1"
ADAPTATION = (
    "Runtime clarification for this D3-native baseline: each state_equations RHS "
    "is a discrete sample-to-sample increment f, evaluated as x_next = x + f. "
    "There is no dt multiplier. Native Adam fits all observed states using "
    "teacher-forced one-step training loss. Checkpoints and generations are "
    "selected on validation; recursive rollout is reported separately and is "
    "not fed back. During that rollout, target states start at observed initial "
    "values and then evolve without resets; supplied auxiliary paths and inputs "
    "remain available. This native discrete baseline does not certify the task's "
    "continuous-time or latent-mechanism requirements. Use RHS expressions only."
)


class Cell(BaseModel):
    """An exact Phase-B cell and finalized public prompt digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    benchmark_id: str = Field(pattern=r"^phase_b_[a-z0-9_]+$")
    tier: Literal["easy", "medium", "hard"]
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Campaign(BaseModel):
    """Bound the discovery budget before any paid provider calls."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: Literal["phase-b-d3-native-validation-1"] = PROTOCOL
    cells: tuple[Cell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = (0,)
    generations: int = Field(default=5, ge=1, le=20)
    patience: int = Field(default=5, ge=1, le=20)
    max_output_tokens: int = Field(default=8192, ge=128, le=32768)
    trajectory_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def unique_cells(self):
        """Reject duplicated tasks and invalid repetition identities."""
        cells = [(c.benchmark_id, c.tier) for c in self.cells]
        if len(set(cells)) != len(cells):
            raise ValueError("duplicate cells")
        if (
            not self.repetitions
            or len(set(self.repetitions)) != len(self.repetitions)
            or any(seed < 0 for seed in self.repetitions)
        ):
            raise ValueError("require unique nonnegative repetitions")
        if self.patience < self.generations:
            raise ValueError("patience must cover the frozen generation budget")
        return self


#: Mirrors LLMConfig.vllm_base_url; LLMConfig cannot be built without a model.
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000"


def _vllm_base_url() -> str:
    """The URL the client actually dials, which must be a real endpoint."""
    return os.environ.get("AF_VLLM_BASE_URL", DEFAULT_VLLM_BASE_URL)


def _endpoint_identity(provider: str) -> str:
    """Describe the endpoint for the sealed plan, without pinning its address.

    Separate from the URL the client dials. A served vLLM listens on a port
    chosen per job, and preparation runs before that server exists, so the
    literal URL differs between freezing a plan and resuming it. Identify the
    local case by kind; moving between a hosted API and a local server still
    changes the identity, which is what the resume guard is for.
    """
    if provider == "vllm":
        return "job-local vllm endpoint"
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


def environment_identity(provider: str = "openai") -> dict:
    """Require fitting dependencies before spending tokens; bind resume to code."""
    importlib.import_module("torch")
    return {
        "runtime_source_sha256": runtime_source_hash(),
        "provider": provider,
        "provider_endpoint_sha256": content_hash(
            {"base_url": _endpoint_identity(provider)}
        ),
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pydantic", "openai")
        },
    }


def _prompt_path(public_root: Path, benchmark: str, tier: str) -> Path:
    spec = BenchmarkRegistry().get(benchmark)
    path = public_root / spec.relative_root
    if spec.data_layout != "tidy_split_file":
        path /= spec.tier_directory_template.format(tier=tier)
    return path / "proposer_prompt.txt"


def prepare(
    config_path: Path,
    public_root: Path,
    root: Path,
    model: str,
    provider: str = "openai",
) -> dict:
    """Freeze exact public development inputs; never access historical/test data."""
    config = Campaign.model_validate(read_json(config_path))
    if provider not in {"openai", "vllm"}:
        raise ValueError(f"unsupported D3 provider: {provider!r}")
    if not model.strip() or ":" in model or model != model.strip():
        raise ValueError("supply an explicit model ID without a provider prefix")
    public_root = public_root.resolve()
    rows = []
    for cell in config.cells:
        _, context, identity = load_public(public_root, cell.benchmark_id, cell.tier)
        if identity["prompt"] != cell.public_prompt_sha256:
            raise ValueError(f"public prompt differs: {cell.benchmark_id}")
        for repetition in config.repetitions:
            rows.append(
                {
                    "index": len(rows),
                    "benchmark_id": cell.benchmark_id,
                    "tier": cell.tier,
                    "repetition": repetition,
                    "public_identity": identity,
                    "validation_context": context.model_dump(mode="json"),
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "config": config.model_dump(mode="json"),
                "provider": provider,
                "model": model,
                "max_attempts": 1,
                "public_root": str(public_root),
                "environment": environment_identity(provider),
                "upstream_revision": D3_UPSTREAM_REVISION,
                "adaptation": ADAPTATION,
                "maximum_logical_calls": len(rows) * config.generations,
                "rows": rows,
                "test_data_opened": False,
                "private_reference_opened": False,
                "selection": "native_one_step_validation",
            },
        )


class _NativeClient:
    """Keep native prompts while binding cached calls to exact cell/repetition."""

    def __init__(self, client, identity: dict):
        self.client, self.identity = client, identity

    def propose(self, *, system_prompt: str, user_prompt: str):
        return self.client.propose(
            system_prompt=system_prompt + "\n\n" + ADAPTATION,
            user_prompt=json.dumps(
                {
                    **json.loads(user_prompt),
                    "campaign_task": self.identity,
                },
                sort_keys=True,
            ),
        )


def accounting(path: Path) -> dict:
    """Count logged physical calls, including failures, without pricing tokens."""
    counts = Counter(
        physical_requests=0, observed_tokens=0, unknown_usage_requests=0, cache_hits=0
    )
    if path.exists():
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get("event") not in {"llm_response", "llm_failure"}:
                continue
            if event.get("cache_hit"):
                counts["cache_hits"] += 1
                continue
            # max_attempts=1: a failed outer attempt cannot also be included in
            # a later successful response's aggregated retry count.
            requests = int(event.get("provider_attempts", 1))
            counts["physical_requests"] += requests
            tokens = (event.get("usage") or {}).get("total_tokens")
            if tokens is None:
                counts["unknown_usage_requests"] += requests
            else:
                counts["observed_tokens"] += int(tokens)
    return dict(counts)


def run(root: Path, index: int, *, client=None) -> dict:
    """Resume one exact task; completed results cause no calls or refitting."""
    plan = sealed_read(root / "plan.json")
    provider = str(plan.get("provider", "openai"))
    if plan["protocol"] != PROTOCOL or plan["environment"] != environment_identity(
        provider
    ):
        raise ValueError(
            "protocol, code or dependencies changed since this plan was frozen. "
            "The identity covers every module in the package, so any commit "
            "invalidates it. Either run from the checkout that froze the plan, "
            f"or delete {root / 'plan.json'} and its results to re-freeze at "
            "the current code. Do not update a checkout mid-campaign."
        )
    if not 0 <= index < len(plan["rows"]):
        raise ValueError("task index out of range")
    row = plan["rows"][index]
    public_root = Path(plan["public_root"])
    data, context, identity = load_public(public_root, row["benchmark_id"], row["tier"])
    if (
        identity != row["public_identity"]
        or context.model_dump(mode="json") != row["validation_context"]
    ):
        raise ValueError("public development input drift")
    directory = root / "results" / str(index)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    with (directory / "task.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if result_path.exists():
            result = sealed_read(result_path)
            if result["plan_sha256"] != plan["artifact_sha256"]:
                raise ValueError("result belongs to a different plan")
            for name, digest in result["source_files"].items():
                if file_hash(directory / name) != digest:
                    raise ValueError("completed native source changed")
            return result
        config = Campaign.model_validate(plan["config"])
        native_path = directory / "native-selection.json"
        saved = sealed_read(native_path) if native_path.exists() else None
        if saved is not None and saved["plan_sha256"] != plan["artifact_sha256"]:
            raise ValueError("native selection belongs to a different plan")
        try:
            if saved is not None:
                selected = BaselineDevelopmentResult.model_validate(saved["selection"])
            else:
                actual_client = (
                    client
                    if client is not None
                    else create_llm_client(
                        LLMConfig(
                            provider=LLMProvider(provider),
                            model=plan["model"],
                            max_attempts=1,
                            max_output_tokens=config.max_output_tokens,
                            cache_directory=directory / "llm_cache",
                            log_path=directory / "llm_calls.jsonl",
                            proposal_target_channels=context.targets,
                            # The served port is job-local, so the client must
                            # use the same endpoint the plan identity records.
                            **(
                                {"vllm_base_url": _vllm_base_url()}
                                if provider == "vllm"
                                else {}
                            ),
                        )
                    )
                )
                selected = run_d3_native_no_tools_development(
                    BaselineConfig(
                        method="d3_native_no_tools",
                        seed=row["repetition"],
                        llm_model=f"{provider}:{plan['model']}",
                        d3_generations=config.generations,
                        d3_patience=config.patience,
                    ),
                    data,
                    context,
                    task_prompt=_prompt_path(
                        public_root, row["benchmark_id"], row["tier"]
                    ).read_text(),
                    work_directory=directory,
                    llm_client=_NativeClient(
                        actual_client,
                        {
                            "plan_sha256": plan["artifact_sha256"],
                            "task_index": index,
                            "repetition": row["repetition"],
                        },
                    ),
                )
                sealed_write(
                    native_path,
                    {
                        "plan_sha256": plan["artifact_sha256"],
                        "selection": selected.model_dump(mode="json"),
                    },
                )
            payload = selected.selection_payload
            evaluation = evaluate_validation(
                CandidateModel.model_validate(payload["candidate"]),
                payload["parameters"],
                data.train,
                data.validation,
                seconds=config.trajectory_seconds,
                saved_one_step_nmse=selected.validation_normalized_mse,
            )
            status = (
                evaluation["phase_b_rollout"]["status"]
                if evaluation["saved_one_step_matches"]
                else "native_check_failed"
            )
            outcome = {
                "status": status,
                "evaluation": evaluation,
                "selected_generation": payload["selected_generation"],
            }
        except (ArithmeticError, ValueError, RuntimeError, TypeError) as exc:
            outcome = {
                "status": "discovery_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        return sealed_write(
            result_path,
            {
                **row,
                **outcome,
                "protocol": PROTOCOL,
                "plan_sha256": plan["artifact_sha256"],
                "source_files": {
                    name: file_hash(directory / name)
                    for name in (
                        "native-selection.json",
                        "d3_checkpoint.json",
                        "llm_calls.jsonl",
                    )
                    if (directory / name).exists()
                },
                "accounting": accounting(directory / "llm_calls.jsonl"),
                "test_data_opened": False,
                "private_reference_opened": False,
                "selection_metric": "native_one_step_validation",
                "recursive_score_fed_to_proposer": False,
            },
        )


def report(root: Path) -> dict:
    """Keep missing/failed cells visible; report separate native/rollout scores."""
    plan = sealed_read(root / "plan.json")
    rows = []
    for task in plan["rows"]:
        path = root / "results" / str(task["index"]) / "result.json"
        row = (
            sealed_read(path)
            if path.exists()
            else {
                **task,
                "status": "pending",
                "accounting": accounting(path.parent / "llm_calls.jsonl"),
            }
        )
        if path.exists() and row["plan_sha256"] != plan["artifact_sha256"]:
            raise ValueError("result belongs to a different plan")
        if path.exists():
            for name, digest in row["source_files"].items():
                if file_hash(path.parent / name) != digest:
                    raise ValueError("completed native source changed")
        rows.append(row)
    counts = dict(Counter(row["status"] for row in rows))
    scores = {}
    for kind in ("native_one_step", "phase_b_rollout"):
        values = [
            r["evaluation"][kind]["normalized_mse"]
            for r in rows
            if r.get("evaluation", {}).get("saved_one_step_matches")
            and r["evaluation"][kind]["normalized_mse"] is not None
        ]
        scale = max(values) if values else 0.0
        median = float(np.median(np.asarray(values) / scale)) * scale if scale else 0.0
        scores[kind] = {
            "finite": len(values),
            "median_normalized_mse": median if values else None,
            "finite_only": True,
        }
    totals = Counter()
    for row in rows:
        totals.update(row.get("accounting", {}))
    value = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "status": "pending" if counts.get("pending") else "complete",
        "model": plan["model"],
        "expected": len(rows),
        "counts": counts,
        "metrics": scores,
        "accounting": dict(totals),
        "rows": rows,
        "test_data_opened": False,
        "private_reference_opened": False,
        "limitation": (
            "Fresh Phase-B D3-native-no-tools; validation-selected discrete models. "
            "Native one-step uses measured target histories; Phase-B rollout resets "
            "only supplied auxiliaries. No dt multiplier or ODE reinterpretation. "
            "Declared parameter bounds are audited, not imposed retroactively. "
            "No latent states or scientific/continuous-time certification. "
            "Validation is not an independent test estimate; failures remain visible."
        ),
    }
    value["artifact_sha256"] = content_hash(value)
    atomic_json(root / "summary.json", value)
    return value
