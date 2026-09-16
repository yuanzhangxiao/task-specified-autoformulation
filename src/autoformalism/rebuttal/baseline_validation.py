"""Frozen, validation-only replay of saved external models; no fitting or LLMs."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from autoformalism.baselines.models import BaselineDevelopmentResult
from autoformalism.baselines.raw_data_agent import raw_agent_validation_context
from autoformalism.config import DataConfig
from autoformalism.data import (
    BenchmarkLoader,
    BenchmarkRegistry,
    DatasetSplit,
    SplitName,
    TrainingScaler,
)
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.rebuttal.final_evaluation import FrozenEvaluationSubject
from autoformalism.rebuttal.final_evaluation_adapters import (
    SourceAdapterRequest,
    adapt_source,
    source_identity,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash

PROTOCOL = "saved-baseline-validation-1"
METHODS = ("sindy", "pysr", "d3", "raw_data_agent")


class RosterCell(BaseModel):
    """An exact public dataset identity; no cross-benchmark substitutions."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    benchmark_id: str = Field(min_length=1)
    tier: Literal["easy", "medium", "hard"]


class ValidationRoster(BaseModel):
    """Freeze expected identities, including unavailable historical D3 runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: Literal["saved-baseline-validation-1"]
    cells: tuple[RosterCell, ...]
    repetitions: tuple[int, ...] = (0, 1, 2)
    historical_d3_cells: tuple[RosterCell, ...] = ()
    historical_d3_repetitions: tuple[int, ...] = (0, 1, 2, 3, 4)
    historical_d3_source_precedence: dict[str, tuple[str, ...]] = Field(
        default_factory=dict
    )


class ReplaySettings(BaseModel):
    """One common solver and per-trajectory limit, frozen before replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    trajectory_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    relative_tolerance: float = Field(default=1e-6, gt=0, allow_inf_nan=False)
    absolute_tolerance: float = Field(default=1e-8, gt=0, allow_inf_nan=False)

    def fit_config(self) -> FitConfig:
        """Use the production simulation API without invoking a fitter."""
        return FitConfig(
            integration_backend="solve_ivp",
            integration_method="Radau",
            relative_tolerance=self.relative_tolerance,
            absolute_tolerance=self.absolute_tolerance,
            maximum_wall_time_seconds=self.trajectory_seconds,
            allow_derivative_regression=False,
        )


def read_json(path: Path) -> dict:
    """Read an artifact as data, never as executable instructions."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def file_hash(path: Path) -> str:
    """Hash bytes without interpreting model or score fields."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(roots: list[Path], output: Path) -> dict:
    """Find result/config files, excluding caches and all trajectory tables."""
    rows, errors, missing = [], [], []
    seen: set[Path] = set()
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            missing.append(str(root))
            continue
        for directory, children, names in os.walk(root):
            children[:] = sorted(
                name
                for name in children
                if name not in {"llm_cache", "cache", ".git", ".venv", "data_raw"}
                and not name.startswith(".")
            )
            for name in ("result.json", "run_config.json"):
                path = Path(directory) / name
                if name not in names or path in seen:
                    continue
                seen.add(path)
                try:
                    payload = read_json(path)
                    method = payload.get("method", "")
                    if not isinstance(method, str):
                        raise ValueError("artifact method must be text")
                    kind = "d3" if method.startswith("d3_") else method
                    if name == "run_config.json":
                        if "provider" not in payload or "repetition" not in payload:
                            continue
                        # This experiment is the GPT-5.6 Sol baseline only.
                        model = str(payload.get("model", "")).lower()
                        if not ("5.6" in model and "sol" in model):
                            continue
                        kind = "raw_data_agent"
                    if kind not in METHODS:
                        continue
                    rows.append(
                        {
                            "source_kind": kind,
                            "source_path": str(
                                path.parent if kind == "raw_data_agent" else path
                            ),
                            "benchmark_id": payload["benchmark_id"],
                            "tier": payload["tier"],
                            "repetition": int(
                                payload.get("repetition", payload.get("seed", 0))
                            ),
                            "identity_file_sha256": file_hash(path),
                        }
                    )
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    errors.append({"path": str(path), "error": str(exc)})
    return sealed_write(
        output,
        {
            "protocol": PROTOCOL,
            "stage": "inventory",
            "rows": sorted(rows, key=lambda row: row["source_path"]),
            "missing_roots": missing,
            "errors": errors,
            "test_data_opened": False,
            "live_llm_calls": 0,
        },
    )


def load_public(root: Path, benchmark: str, tier: str):
    """Load only train and validation, plus the public task prompt."""
    registry = BenchmarkRegistry()
    spec = registry.get(benchmark)
    development = BenchmarkLoader(registry).load_development(
        DataConfig(root=root, benchmark_id=benchmark, tier=tier)
    )
    prompt_root = root / spec.relative_root
    if spec.data_layout != "tidy_split_file":
        prompt_root /= spec.tier_directory_template.format(tier=tier)
    prompt_hash = file_hash(prompt_root / "proposer_prompt.txt")
    context = raw_agent_validation_context(development, spec)
    identity = {
        "train": development.train.fingerprint,
        "validation": development.validation.fingerprint,
        "prompt": prompt_hash,
    }
    return development, context, identity


def source_files(request: SourceAdapterRequest) -> dict[str, str]:
    """Seal all model inputs; never read provider caches or test tables."""
    path = request.source_path
    if request.source_kind == "raw_data_agent":
        files = [
            path / name
            for name in ("run_config.json", "candidate.json", "evaluation.json")
        ]
    else:
        files = [path]
        if request.source_kind == "d3":
            files.append(path.with_name("d3_checkpoint.json"))
    return {str(item): file_hash(item) for item in files}


def audit_source(request: SourceAdapterRequest, identity: dict) -> dict:
    """Reject incompatible explicit provenance; disclose absent legacy evidence."""
    path = request.source_path
    if request.source_kind == "raw_data_agent":
        config = read_json(path / "run_config.json")
        evaluation = read_json(path / "evaluation.json")
        if evaluation.get("schema_version") != "raw-data-agent-fitted-evaluation-1":
            raise ValueError("require the agent-fitted contract, not common refitting")
        if config.get("parameter_refit_applied") is not False:
            raise ValueError("raw agent must certify no parameter refit")
        hashes = config.get("public_input_hashes", {})
        if hashes.get("proposer_prompt.txt") != identity["prompt"]:
            raise ValueError("raw-agent public prompt differs or has no saved hash")
        if config.get("split_fingerprints") != {
            key: identity[key] for key in ("train", "validation")
        }:
            raise ValueError("raw-agent development data fingerprints differ")
        native = evaluation.get("validation_metrics", {}).get("normalized_mse")
        return {
            "source_data_provenance": "verified",
            "prior_test_evaluation": "not_recorded_in_source",
            "native_validation_nmse": native,
            "native_selection": "agent_fitted_development_selection",
        }
    payload = read_json(path)
    historical_refit_ignored = False
    if request.source_kind in {"sindy", "pysr"}:
        BaselineDevelopmentResult.model_validate(payload)
        native_selection = "one_step_validation_selection_train_fitted_equations"
    else:
        historical_refit_ignored = _check_d3_selection(path, payload)
        native_selection = payload.get("selected_hyperparameters", {}).get(
            "adaptation", "historical_d3"
        )
    return {
        "source_data_provenance": "not_recorded_in_result_verify_original_release",
        "historical_train_validation_refit_values_ignored": historical_refit_ignored,
        "parameter_source": (
            "selected_training_checkpoint"
            if request.source_kind == "d3"
            else "development_equations"
        ),
        "prior_test_evaluation": (
            "recorded_in_historical_result"
            if "test_normalized_mse" in payload
            else "not_recorded_in_source"
        ),
        "native_validation_nmse": payload.get("validation_normalized_mse"),
        "native_selection": native_selection,
    }


def _check_d3_selection(path: Path, payload: dict) -> bool:
    """Use training checkpoint values, including the pre-native D3 refit case."""
    hyperparameters = payload["selected_hyperparameters"]
    generation = int(hyperparameters["selected_generation"])
    records = [
        r
        for r in read_json(path.with_name("d3_checkpoint.json"))["records"]
        if r.get("generation") == generation
    ]
    if len(records) != 1:
        raise ValueError("D3 selected generation is missing or duplicated")
    record = records[0]
    equations = {
        e["state"]: e["rhs"] for e in record["candidate"].get("state_equations", [])
    }
    if payload["equations"] != equations:
        raise ValueError("D3 result and selected checkpoint equations differ")
    differing = (
        "selected_parameters" in hyperparameters
        and json.loads(hyperparameters["selected_parameters"]) != record["parameters"]
    )
    legacy_refit = (
        payload.get("method") == "d3_no_tools"
        and hyperparameters.get("adaptation") == "restricted_schema"
    )
    if differing and not legacy_refit:
        raise ValueError("D3 result and selected checkpoint parameters differ")
    return bool(differing and legacy_refit)


def _runtime() -> dict:
    return {
        "source_sha256": runtime_source_hash(),
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pydantic")
        },
    }


def prepare(
    inventory_path: Path,
    roster_path: Path,
    public_root: Path,
    legacy_root: Path,
    output: Path,
    settings: ReplaySettings,
) -> dict:
    """Freeze the expected roster without choosing ambiguous sources by score."""
    inputs = sealed_read(inventory_path)
    roster = ValidationRoster.model_validate(read_json(roster_path)).model_dump()
    if any(
        seed < 0
        for seed in (*roster["repetitions"], *roster["historical_d3_repetitions"])
    ):
        raise ValueError("repetitions must be nonnegative")
    rows = []
    cache = {}
    requests = [
        ("phase_b", kind, cell["benchmark_id"], cell["tier"], seed)
        for cell in roster["cells"]
        for seed in roster["repetitions"]
        for kind in METHODS
    ] + [
        ("historical_d3", "d3", cell["benchmark_id"], cell["tier"], seed)
        for cell in roster.get("historical_d3_cells", [])
        for seed in roster.get("historical_d3_repetitions", [])
    ]
    if len(requests) != len(set(requests)) or not requests:
        raise ValueError("roster must be nonempty and unique")
    for index, (cohort, kind, benchmark, tier, seed) in enumerate(requests):
        row = {
            "index": index,
            "cohort": cohort,
            "source_kind": kind,
            "benchmark_id": benchmark,
            "tier": tier,
            "repetition": seed,
        }
        matches = [
            item
            for item in inputs["rows"]
            if (
                item["source_kind"],
                item["benchmark_id"],
                item["tier"],
                item["repetition"],
            )
            == (kind, benchmark, tier, seed)
        ]
        row["discovered_sources"] = [item["source_path"] for item in matches]
        if cohort == "historical_d3":
            for directory in roster["historical_d3_source_precedence"].get(
                benchmark, ()
            ):
                preferred = [
                    item
                    for item in matches
                    if directory in Path(item["source_path"]).parts
                ]
                if preferred:
                    matches = preferred
                    row["historical_source_precedence"] = directory
                    break
        try:
            if not matches:
                raise FileNotFoundError(
                    "no saved source for this exact cell/repetition"
                )
            data_root = (public_root if cohort == "phase_b" else legacy_root).resolve()
            key = (str(data_root), benchmark, tier)
            if key not in cache:
                cache[key] = load_public(data_root, benchmark, tier)
            _, context, identity = cache[key]
            usable, rejected = {}, []
            for match in matches:
                request = SourceAdapterRequest(
                    request_id=f"validation_{index}",
                    source_kind=kind,
                    source_path=match["source_path"],
                    expected_benchmark_id=benchmark,
                    expected_tier=tier,
                    expected_repetition=seed,
                )
                try:
                    identity_file = (
                        request.source_path / "run_config.json"
                        if kind == "raw_data_agent"
                        else request.source_path
                    )
                    if file_hash(identity_file) != match["identity_file_sha256"]:
                        raise ValueError("source changed since inventory")
                    source_identity(request)
                    audit = audit_source(request, identity)
                    subject = adapt_source(request, context)
                    if subject.parameterization.status not in {
                        "available",
                        "not_required",
                    }:
                        raise ValueError("saved model lacks complete fitted parameters")
                    compile_candidate(subject.candidate, context)
                    files = source_files(request)
                    # Collapse only byte-identical copies of every required input.
                    signature = tuple(
                        sorted((Path(p).name, h) for p, h in files.items())
                    )
                    usable.setdefault(signature, (subject, files, audit))
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    rejected.append({"path": match["source_path"], "error": str(exc)})
            row["rejected_sources"] = rejected
            if not usable:
                reasons = "; ".join(item["error"] for item in rejected[:3])
                raise ValueError(f"no compatible saved source: {reasons}")
            if len(usable) != 1:
                raise ValueError(
                    f"require one compatible source; found {len(usable)}; "
                    "narrow inventory roots if multiple independent runs exist"
                )
            subject, files, audit = next(iter(usable.values()))
            row.update(
                status="ready",
                subject=subject.model_dump(mode="json"),
                source_files=files,
                data_root=str(data_root),
                data_identity=identity,
                audit=audit,
            )
        except Exception as exc:
            row.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return sealed_write(
        output / "plan.json",
        {
            "protocol": PROTOCOL,
            "runtime": _runtime(),
            "inventory_sha256": inputs["artifact_sha256"],
            "roster_sha256": file_hash(roster_path),
            "settings": settings.model_dump(),
            "rows": rows,
            "test_data_opened": False,
            "parameter_refit_applied": False,
            "live_llm_calls": 0,
        },
    )


def evaluate_validation(
    subject: FrozenEvaluationSubject,
    train: DatasetSplit,
    validation: DatasetSplit,
    settings: ReplaySettings,
) -> dict:
    """Open-loop ODE replay; initial observations only, frozen parameters, no resets."""
    if train.name is not SplitName.TRAIN or validation.name is not SplitName.VALIDATION:
        raise ValueError("validation replay requires TRAIN and VALIDATION, never TEST")
    if not train.trajectories or not validation.trajectories:
        raise ValueError("nonempty train and validation required")
    if subject.parameterization.status not in {"available", "not_required"}:
        raise ValueError("saved model lacks complete fitted parameters")
    compiled = compile_candidate(subject.candidate, subject.validation_context)
    scaling = TrainingScaler().fit(train).scales
    scales = {
        target: float(scaling[f"target:{target}"].standard_deviation)
        for target in subject.validation_context.targets
    }
    squared = {target: [] for target in scales}
    trajectories = []
    for trajectory in validation.trajectories:
        simulation = simulate_trajectory(
            compiled,
            trajectory,
            subject.parameterization.global_parameters,
            subject.parameterization.global_initial_conditions,
            settings.fit_config(),
            deadline=monotonic() + settings.trajectory_seconds,
            reset_observed_states=False,
        )
        row = {
            "trajectory_id": trajectory.trajectory_id,
            "success": simulation.success,
            "message": simulation.message,
        }
        if simulation.success:
            errors = {
                target: np.square(
                    (simulation.predictions[target] - trajectory.targets[target])
                    / scales[target]
                )
                for target in scales
            }
            if not all(np.isfinite(value).all() for value in errors.values()):
                row.update(success=False, message="nonfinite validation residuals")
            else:
                row["per_target_normalized_mse"] = {
                    target: float(np.mean(value)) for target, value in errors.items()
                }
                for target, value in errors.items():
                    squared[target].append(value)
        trajectories.append(row)
    complete = all(row["success"] for row in trajectories)
    per_target = (
        {
            target: float(np.mean(np.concatenate(values)))
            for target, values in squared.items()
        }
        if complete
        else {}
    )
    return {
        "status": "complete" if complete else "rollout_failed",
        "normalized_mse": float(np.mean(list(per_target.values())))
        if complete
        else None,
        "per_target_normalized_mse": per_target,
        "normalization_scales": scales,
        "trajectories": trajectories,
        "evaluation_protocol": "validation_open_rollout",
        "sample_policy": "all_samples_including_initial",
        "parameter_refit_applied": False,
        "validation_initials_fitted": False,
        "reset_observed_states": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "live_llm_calls": 0,
    }


def run(output: Path, shard: int = 0, shards: int = 1) -> dict:
    """Checkpoint each row; identical resume performs no additional simulation."""
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("invalid shard index/count")
    plan = sealed_read(output / "plan.json")
    if plan["runtime"] != _runtime():
        raise ValueError("evaluator source or library versions differ from frozen plan")
    settings = ReplaySettings.model_validate(plan["settings"])
    for row in plan["rows"]:
        if row["index"] % shards != shard:
            continue
        result_path = output / "results" / f"{row['index']:04d}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if result_path.exists():
                if sealed_read(result_path)["plan_sha256"] != plan["artifact_sha256"]:
                    raise ValueError("checkpoint belongs to a different plan")
                continue
            result = {"plan_sha256": plan["artifact_sha256"], "index": row["index"]}
            if row["status"] == "unavailable":
                result.update(status="source_unavailable", error=row["error"])
            else:
                try:
                    if any(
                        file_hash(Path(path)) != digest
                        for path, digest in row["source_files"].items()
                    ):
                        raise ValueError("source artifact changed after freezing")
                    development, context, identity = load_public(
                        Path(row["data_root"]), row["benchmark_id"], row["tier"]
                    )
                    subject = FrozenEvaluationSubject.model_validate(row["subject"])
                    if (
                        identity != row["data_identity"]
                        or context != subject.validation_context
                    ):
                        raise ValueError("public data/context changed after freezing")
                    result.update(
                        evaluate_validation(
                            subject, development.train, development.validation, settings
                        )
                    )
                except Exception as exc:
                    result.update(
                        status="evaluation_failed", error=f"{type(exc).__name__}: {exc}"
                    )
            sealed_write(result_path, result)
    return report(output)


def report(output: Path) -> dict:
    """Retain the denominator and report cohorts/cells without ranking."""
    plan = sealed_read(output / "plan.json")
    rows = []
    for row in plan["rows"]:
        path = output / "results" / f"{row['index']:04d}.json"
        result = sealed_read(path) if path.exists() else {"status": "pending"}
        if path.exists() and (
            result["plan_sha256"] != plan["artifact_sha256"]
            or result["index"] != row["index"]
        ):
            raise ValueError("result identity differs from plan")
        rows.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "index",
                        "cohort",
                        "source_kind",
                        "benchmark_id",
                        "tier",
                        "repetition",
                    )
                },
                "source_status": row["status"],
                "audit": row.get("audit"),
                "error": row.get("error"),
                **result,
            }
        )
    groups = []
    for key in sorted({(r["cohort"], r["source_kind"]) for r in rows}):
        selected = [r for r in rows if (r["cohort"], r["source_kind"]) == key]
        groups.append(
            {
                "cohort": key[0],
                "method": key[1],
                "expected": len(selected),
                "counts": dict(Counter(r["status"] for r in selected)),
            }
        )
    return {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "status": "partial"
        if any(r["status"] == "pending" for r in rows)
        else "complete",
        "groups": groups,
        "rows": rows,
        "test_data_opened": False,
        "private_reference_opened": False,
        "parameter_refit_applied": False,
        "live_llm_calls": 0,
        "limitation": "Validation-selected models, not independent test estimates. "
        "Historical D3 uses original cells and cannot populate Phase-B cells. "
        "Missing historical data/prompt provenance remains explicitly unverified. "
        "Native scores and common open rollouts have different semantics.",
    }
