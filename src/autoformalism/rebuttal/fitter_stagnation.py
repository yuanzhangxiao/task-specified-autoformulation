"""Frozen derivative-step and ODE-tolerance diagnosis, separate from model search."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.data import DatasetSplit, TrainingScaler
from autoformalism.expressions import compile_candidate
from autoformalism.fitting.stagnation import (
    RolloutOracle,
    inspect_steps,
    instrumented_fit,
)
from autoformalism.rebuttal.fitter_diagnostic import (
    PUBLIC_FILES,
    _finite_payload,
    _write_bytes,
    read_json,
    replay_parameters,
    runtime_identity,
    sha256,
    write_json,
)
from autoformalism.rebuttal.staged_fit_probe import StagedFitPlan, load_data
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.staged_topology import content_hash

TASKS = (
    {"name": "step_profile", "kind": "profile"},
    {"name": "current_default", "kind": "fit", "tight": False, "large_step": False},
    {"name": "current_large_step", "kind": "fit", "tight": False, "large_step": True},
    {"name": "tight_default", "kind": "fit", "tight": True, "large_step": False},
    {"name": "tight_large_step", "kind": "fit", "tight": True, "large_step": True},
)


class StagnationPlan(StrictSchema):
    """A matched numerical factorial on an immutable candidate and public snapshot."""

    protocol: Literal["fitter-stagnation-1"] = "fitter-stagnation-1"
    source_freeze_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_result_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    fit_seconds: FiniteFloat = Field(default=300, gt=0, le=600)
    profile_seconds: FiniteFloat = Field(default=600, gt=0, le=900)
    replay_seconds: FiniteFloat = Field(default=60, gt=0, le=180)
    grace_seconds: FiniteFloat = Field(default=60, gt=0, le=120)
    maximum_nfev: int = Field(default=50, ge=1, le=100)
    large_diff_step: FiniteFloat = Field(default=1e-4, gt=0, le=0.01)
    tight_rtol: FiniteFloat = Field(default=1e-9, gt=0, lt=1e-7)
    tight_atol: FiniteFloat = Field(default=1e-11, gt=0, lt=1e-9)
    profile_trajectories: int = Field(default=2, ge=1, le=4)
    profile_steps: tuple[FiniteFloat, ...] = Field(
        default=(1.4901161193847656e-8, 1e-6, 1e-4, 1e-2),
        min_length=2,
        max_length=8,
    )

    @model_validator(mode="after")
    def ordered_positive_steps(self) -> StagnationPlan:
        """Reject zero, duplicate and unordered perturbation scales."""
        if (
            self.profile_steps[0] <= 0
            or tuple(sorted(set(self.profile_steps))) != self.profile_steps
        ):
            raise ValueError("profile steps must be positive and strictly increasing")
        return self

    def worker_seconds(self, index: int) -> float:
        """Bound the entire task including independent replay and finalization."""
        return self.grace_seconds + (
            self.profile_seconds
            if index == 0
            else self.fit_seconds + self.replay_seconds
        )


def launch_identity() -> str:
    """Bind the diagnostic CLI, array worker and user-invoked submitter."""
    repository = Path(__file__).resolve().parents[3]
    return content_hash(
        {
            name: sha256(repository / name)
            for name in (
                "scripts/run_fitter_stagnation.py",
                "scripts/hpc/fitter_stagnation_delta.slurm",
                "scripts/hpc/submit_fitter_stagnation_delta.sh",
            )
        }
    )


def prepare_diagnosis(
    plan: StagnationPlan, source: Path, output: Path
) -> dict[str, Any]:
    """Copy verified old development inputs into a new, independently frozen run."""
    if source.resolve() == output.resolve():
        raise ValueError("diagnosis needs a new output directory")
    if sha256(source / "freeze.json") != plan.source_freeze_sha256:
        raise ValueError("source freeze digest differs")
    if sha256(source / "result.json") != plan.source_result_sha256:
        raise ValueError("source result digest differs")
    parent = read_json(source / "freeze.json")
    source_plan = StagedFitPlan.model_validate(parent["plan"])
    result = read_json(source / "result.json")
    if result.get("freeze_sha256") != plan.source_freeze_sha256:
        raise ValueError("source result identity differs")
    current = runtime_identity()
    if any(current[key] != parent["runtime"][key] for key in ("python", "packages")):
        raise ValueError("numerical dependency versions differ from the original run")
    assets = {}
    allowed = {"candidate.json", "source_function.json"} | {
        f"public/phase_b_v1/{source_plan.benchmark_id}/{name}" for name in PUBLIC_FILES
    }
    if set(parent["assets"]) != allowed:
        raise ValueError("source snapshot contains unexpected or missing assets")
    for relative, digest in parent["assets"].items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source.resolve()) or sha256(path) != digest:
            raise ValueError(f"source asset differs: {relative}")
        _write_bytes(output / relative, path.read_bytes(), immutable=True)
        if sha256(output / relative) != digest:
            raise ValueError("source changed while copying")
        assets[relative] = digest
    for name, origin in (
        ("source_freeze.json", "freeze.json"),
        ("source_result.json", "result.json"),
    ):
        _write_bytes(output / name, (source / origin).read_bytes(), immutable=True)
        assets[name] = sha256(output / name)
    if (
        assets["source_freeze.json"] != plan.source_freeze_sha256
        or assets["source_result.json"] != plan.source_result_sha256
    ):
        raise ValueError("source changed during preparation")
    dataset, context = load_data(output, source_plan)
    candidate = CandidateModel.model_validate(read_json(output / "candidate.json"))
    compile_candidate(candidate, context)
    if not candidate.parameters or any(
        p.scope.value != "global" for p in candidate.parameters
    ):
        raise ValueError("diagnosis requires nonempty global parameter layout")
    if any(i.fixed_value is None for i in candidate.initial_conditions):
        raise ValueError("diagnosis preserves fixed initial conditions")
    ids = sorted(t.trajectory_id for t in dataset.train.trajectories)[
        : plan.profile_trajectories
    ]
    if len(ids) != plan.profile_trajectories:
        raise ValueError("not enough training trajectories for the frozen profile")
    frozen = {
        "plan": plan.model_dump(mode="json"),
        "runtime": current,
        "launcher_sha256": launch_identity(),
        "assets": assets,
        "profile_trajectory_ids": ids,
        "tasks": list(TASKS),
        "test_data_opened": False,
        "private_reference_opened": False,
        "llm_calls": 0,
    }
    frozen["freeze_sha256"] = content_hash(frozen)
    write_json(output / "freeze.json", frozen, immutable=True)
    return frozen


def verify_diagnosis(output: Path) -> dict[str, Any]:
    """Validate exact source, launchers, runtime and inputs before cache replay."""
    frozen = read_json(output / "freeze.json")
    if (
        content_hash({k: v for k, v in frozen.items() if k != "freeze_sha256"})
        != frozen["freeze_sha256"]
    ):
        raise ValueError("diagnosis freeze digest differs")
    StagnationPlan.model_validate(frozen["plan"])
    if (
        frozen["runtime"] != runtime_identity()
        or frozen["launcher_sha256"] != launch_identity()
    ):
        raise ValueError("diagnosis runtime or launcher differs")
    for relative, digest in frozen["assets"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output.resolve()) or sha256(path) != digest:
            raise ValueError(f"diagnosis asset differs: {relative}")
    return frozen


def checkpoint(path: Path, identity: str) -> dict[str, Any] | None:
    """Read a completed phase only if its task identity matches."""
    if not path.exists():
        return None
    value = read_json(path)
    if value.get("identity") != identity:
        raise ValueError("checkpoint identity differs")
    return value


def execute_diagnosis(output: Path, index: int) -> dict[str, Any]:
    """Run one profile or one matched optimizer arm, preserving completed phases."""
    frozen = verify_diagnosis(output)
    if not 0 <= index < len(frozen["tasks"]):
        raise ValueError("unknown task index")
    task = frozen["tasks"][index]
    identity = content_hash([frozen["freeze_sha256"], task])
    root = output / "results" / task["name"]
    root.mkdir(parents=True, exist_ok=True)
    existing = checkpoint(root / "result.json", identity)
    if existing is not None:
        return existing
    plan = StagnationPlan.model_validate(frozen["plan"])
    parent = read_json(output / "source_freeze.json")
    parent_plan = StagedFitPlan.model_validate(parent["plan"])
    source_result = read_json(output / "source_result.json")
    dataset, context = load_data(output, parent_plan)
    candidate = CandidateModel.model_validate(read_json(output / "candidate.json"))
    model = compile_candidate(candidate, context)
    scaler = TrainingScaler().fit(dataset.train)
    scales = {
        name: scaler.scales[f"target:{name}"].standard_deviation
        for name in context.targets
    }
    config = parent_plan.fit_config.model_copy(
        update={
            "number_of_starts": 1,
            "maximum_function_evaluations": plan.maximum_nfev,
            "maximum_wall_time_seconds": plan.fit_seconds,
        }
    )
    tight = config.model_copy(
        update={
            "relative_tolerance": plan.tight_rtol,
            "absolute_tolerance": plan.tight_atol,
        }
    )
    start = source_result["default_replay"]["parameters"]
    attempt = 0
    while (root / f"attempt-{attempt}").exists():
        attempt += 1
    trace = root / f"attempt-{attempt}"
    trace.mkdir()
    if index == 0:
        ids = frozen["profile_trajectory_ids"]
        subset = DatasetSplit(
            dataset.train.name,
            tuple(
                next(t for t in dataset.train.trajectories if t.trajectory_id == name)
                for name in ids
            ),
            content_hash([dataset.train.fingerprint, ids]),
        )
        reports = []
        deadline = monotonic() + plan.profile_seconds
        for anchor_name, anchor in (
            ("all_ones", start),
            ("previous_fit", source_result["fit"]["global_parameters"]),
        ):
            for setting_name, settings in (("current", config), ("tight", tight)):
                cache = root / "profile" / f"{anchor_name}_{setting_name}"
                oracle = RolloutOracle(
                    model, subset, scales, settings, trace / cache.name, deadline
                )
                report = inspect_steps(
                    oracle,
                    anchor,
                    plan.profile_steps,
                    cache,
                    content_hash([identity, anchor_name, setting_name]),
                )
                reports.append(
                    {"anchor": anchor_name, "ode_tolerance": setting_name, **report}
                )
        tolerance_comparison = []
        for anchor_name in ("all_ones", "previous_fit"):
            arrays = []
            for setting in ("current", "tight"):
                with np.load(
                    root / "profile" / f"{anchor_name}_{setting}" / "base.npz",
                    allow_pickle=False,
                ) as data:
                    arrays.append(data["residual"])
            tolerance_comparison.append(
                {
                    "anchor": anchor_name,
                    "residual_difference_norm": float(
                        np.linalg.norm(arrays[0] - arrays[1])
                    ),
                }
            )
        result = {
            "identity": identity,
            "status": "complete",
            "task": task,
            "trajectory_ids": ids,
            "target_scales": scales,
            "reports": reports,
            "tolerance_comparison": tolerance_comparison,
        }
    else:
        settings = tight if task["tight"] else config
        fit = checkpoint(root / "fit.json", identity)
        if fit is None:
            oracle = RolloutOracle(
                model,
                dataset.train,
                scales,
                settings,
                trace,
                monotonic() + plan.fit_seconds,
            )
            fit = {
                "identity": identity,
                "settings": settings.model_dump(mode="json"),
                "diff_step": plan.large_diff_step if task["large_step"] else None,
                "fit": instrumented_fit(
                    oracle,
                    start,
                    diff_step=plan.large_diff_step if task["large_step"] else None,
                    max_nfev=plan.maximum_nfev,
                ),
            }
            write_json(root / "fit.json", fit)
        replay = checkpoint(root / "replay.json", identity)
        if replay is None:
            parameters = fit["fit"]["parameters"]
            replay = {
                "identity": identity,
                "replay": replay_parameters(
                    model, dataset, parameters, tight, plan.replay_seconds
                )
                if parameters is not None
                else None,
            }
            write_json(root / "replay.json", _finite_payload(replay))
        successful = replay["replay"] is not None and all(
            replay["replay"][split]["normalized_mse"] is not None
            for split in ("train", "validation")
        )
        result = {
            "identity": identity,
            "task": task,
            "status": "complete" if successful else "rollout_failed",
            **fit,
            **replay,
            "target_scales": scales,
        }
    result.update(test_data_opened=False, private_reference_opened=False, llm_calls=0)
    result = _finite_payload(result)
    write_json(root / "result.json", result)
    return result


def summarize_diagnosis(output: Path) -> dict[str, Any]:
    """Keep every frozen arm, including missing, failed and timed-out tasks."""
    frozen = verify_diagnosis(output)
    rows = []
    for task in frozen["tasks"]:
        identity = content_hash([frozen["freeze_sha256"], task])
        result = checkpoint(output / "results" / task["name"] / "result.json", identity)
        fit = (result or {}).get("fit", {})
        replay = (result or {}).get("replay") or {}
        rows.append(
            {
                "name": task["name"],
                "status": (result or {}).get("status", "missing"),
                "error": (result or {}).get("error"),
                "train_nmse": replay.get("train", {}).get("normalized_mse"),
                "validation_nmse": replay.get("validation", {}).get("normalized_mse"),
                **{
                    key: fit.get(key)
                    for key in (
                        "optimizer_success",
                        "message",
                        "selection",
                        "nfev",
                        "njev",
                        "actual_residual_calls",
                        "fit_seconds",
                        "optimality",
                        "parameters",
                    )
                },
            }
        )
    return {
        "freeze_sha256": frozen["freeze_sha256"],
        "rows": rows,
        "test_data_opened": False,
        "private_reference_opened": False,
        "llm_calls": 0,
        "model_selection_performed": False,
    }
