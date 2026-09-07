"""Frozen rollout-runtime and finite-difference campaign for user-run Delta jobs."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.data import DatasetSplit, TrainingScaler
from autoformalism.expressions import compile_candidate
from autoformalism.fitting.runtime_probe import (
    METHODS,
    derivative_cases,
    interpreter_profile,
    read_array,
    residual_distance,
    rollout_case,
)
from autoformalism.fitting.stagnation import RolloutOracle, instrumented_fit
from autoformalism.rebuttal.fitter_diagnostic import (
    _finite_payload,
    _write_bytes,
    read_json,
    replay_parameters,
    runtime_identity,
    sha256,
    write_json,
)
from autoformalism.rebuttal.fitter_stagnation import TASKS as PREVIOUS_TASKS
from autoformalism.rebuttal.fitter_stagnation import StagnationPlan, checkpoint
from autoformalism.rebuttal.staged_fit_probe import StagedFitPlan, load_data
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.staged_topology import content_hash

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
ANCHORS = ("all_ones", "intermediate", "late")
TASKS = tuple(
    [
        {"name": f"profile_{anchor}", "kind": "profile", "anchor": anchor}
        for anchor in ANCHORS
    ]
    + [
        {
            "name": f"fit_{method}_{policy}",
            "kind": "fit",
            "method": method,
            "policy": policy,
        }
        for method in METHODS
        for policy in ("relative", "scaled")
    ]
)


class RuntimePlan(StrictSchema):
    """Predeclared nine-task comparison, including guards and all resource limits."""

    protocol: Literal["fitter-runtime-1"] = "fitter-runtime-1"
    source_code_sha256: Digest
    source_plan_sha256: Digest
    anchor_hashes: dict[Literal["all_ones", "intermediate", "late"], Digest]
    profile_seconds: FiniteFloat = Field(default=600, gt=0, le=720)
    call_seconds: FiniteFloat = Field(default=60, gt=0, le=120)
    fit_seconds: FiniteFloat = Field(default=900, gt=0, le=1200)
    replay_seconds: FiniteFloat = Field(default=120, gt=0, le=180)
    grace_seconds: FiniteFloat = Field(default=60, gt=0, le=120)
    maximum_nfev: int = Field(default=100, ge=1, le=200)
    profile_trajectories: int = Field(default=2, ge=1, le=4)
    step: FiniteFloat = Field(default=1e-4, gt=0, le=0.01)
    scale_floor: FiniteFloat = Field(default=1.0, gt=0, le=100)
    accuracy_rms: FiniteFloat = Field(default=1e-5, gt=0, le=1e-3)
    accuracy_maximum: FiniteFloat = Field(default=1e-4, gt=0, le=1e-2)
    reference_rms: FiniteFloat = Field(default=1e-6, gt=0, le=1e-4)
    reference_maximum: FiniteFloat = Field(default=1e-5, gt=0, le=1e-3)

    @model_validator(mode="after")
    def complete_anchors(self) -> RuntimePlan:
        """Require every prespecified vector and a stricter reference check."""
        if set(self.anchor_hashes) != set(ANCHORS):
            raise ValueError("all three anchor hashes are required")
        if (
            self.reference_rms > self.accuracy_rms
            or self.reference_maximum > self.accuracy_maximum
        ):
            raise ValueError(
                "reference tolerance must be stricter than candidate tolerance"
            )
        return self

    def worker_seconds(self, index: int) -> float:
        """Bound one worker's numerical phases plus checkpoint finalization."""
        return self.grace_seconds + (
            self.profile_seconds
            if index < 3
            else self.fit_seconds + 2 * self.replay_seconds
        )


def launch_identity() -> str:
    """Bind the exact CLI and scheduler entry points to the numerical snapshot."""
    repository = Path(__file__).resolve().parents[3]
    return content_hash(
        {
            name: sha256(repository / name)
            for name in (
                "scripts/run_fitter_runtime.py",
                "scripts/hpc/fitter_runtime_delta.slurm",
                "scripts/hpc/submit_fitter_runtime_delta.sh",
            )
        }
    )


def prepare_runtime(plan: RuntimePlan, source: Path, output: Path) -> dict:
    """Verify the previous run and snapshot only pinned public inputs and anchors."""
    if source.resolve() == output.resolve():
        raise ValueError("runtime diagnosis requires a new output directory")
    parent = read_json(source / "freeze.json")
    if (
        content_hash({k: v for k, v in parent.items() if k != "freeze_sha256"})
        != parent["freeze_sha256"]
    ):
        raise ValueError("previous freeze digest differs")
    previous_plan = StagnationPlan.model_validate(parent["plan"])
    if content_hash(previous_plan.model_dump(mode="json")) != plan.source_plan_sha256:
        raise ValueError("previous plan differs")
    if parent["tasks"] != list(PREVIOUS_TASKS):
        raise ValueError("previous task matrix differs")
    current = runtime_identity()
    if parent["runtime"]["source_sha256"] != plan.source_code_sha256:
        raise ValueError("previous source code differs")
    if any(current[key] != parent["runtime"][key] for key in ("python", "packages")):
        raise ValueError("numerical dependencies differ from previous run")
    if sha256(source / "source_freeze.json") != previous_plan.source_freeze_sha256:
        raise ValueError("original candidate snapshot differs")
    if sha256(source / "source_result.json") != previous_plan.source_result_sha256:
        raise ValueError("original fit result differs")
    original = read_json(source / "source_freeze.json")
    allowed = set(original["assets"]) | {"source_freeze.json", "source_result.json"}
    if set(parent["assets"]) != allowed:
        raise ValueError("unexpected previous snapshot assets")
    if any(
        parent["assets"][name] != digest for name, digest in original["assets"].items()
    ):
        raise ValueError(
            "previous model or public inputs differ from original snapshot"
        )
    assets = {}
    for relative, digest in parent["assets"].items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source.resolve()) or sha256(path) != digest:
            raise ValueError(f"previous asset differs: {relative}")
        _write_bytes(output / relative, path.read_bytes(), immutable=True)
        if sha256(output / relative) != digest:
            raise ValueError("source changed during snapshot")
        assets[relative] = digest
    task = next(t for t in parent["tasks"] if t["name"] == "current_large_step")
    previous = source / "results/current_large_step"
    result = read_json(previous / "result.json")
    fit = read_json(previous / "fit.json")
    expected = content_hash([parent["freeze_sha256"], task])
    if (
        result.get("identity") != expected
        or fit.get("identity") != expected
        or result.get("status") != "complete"
        or result.get("fit") != fit.get("fit")
        or result.get("test_data_opened") is not False
        or result.get("private_reference_opened") is not False
    ):
        raise ValueError("previous fit provenance differs")
    iterations = fit["fit"]["iterations"]
    if len(iterations) < 2:
        raise ValueError("previous fit needs two completed parameter updates")
    anchors = {
        "all_ones": fit["fit"]["initial_parameters"],
        "intermediate": iterations[-2]["parameters"],
        "late": iterations[-1]["parameters"],
    }
    if {
        name: content_hash(values) for name, values in anchors.items()
    } != plan.anchor_hashes:
        raise ValueError("anchor vectors differ from the reviewed trace")
    for relative, value in (
        ("previous_freeze.json", parent),
        ("previous_fit.json", fit),
        ("previous_result.json", result),
    ):
        write_json(output / relative, value, immutable=True)
        assets[relative] = sha256(output / relative)
    original_plan = StagedFitPlan.model_validate(original["plan"])
    dataset, context = load_data(output, original_plan)
    candidate = CandidateModel.model_validate(read_json(output / "candidate.json"))
    model = compile_candidate(candidate, context)
    if not candidate.parameters or any(
        p.scope.value != "global" for p in candidate.parameters
    ):
        raise ValueError("runtime diagnosis requires global parameters")
    if any(i.fixed_value is None for i in candidate.initial_conditions):
        raise ValueError("runtime diagnosis preserves fixed initial values")
    for values in anchors.values():
        # Use exactly the production fitter's physical parameter bounds.
        oracle = RolloutOracle(
            model,
            dataset.train,
            dict.fromkeys(context.targets, 1.0),
            original_plan.fit_config,
            output / "preparation",
            None,
        )
        oracle.vector(values)
    ids = sorted(t.trajectory_id for t in dataset.train.trajectories)[
        : plan.profile_trajectories
    ]
    if len(ids) != plan.profile_trajectories:
        raise ValueError("insufficient training trajectories for the profile")
    frozen = {
        "plan": plan.model_dump(mode="json"),
        "runtime": current,
        "launcher_sha256": launch_identity(),
        "assets": assets,
        "anchors": anchors,
        "profile_trajectory_ids": ids,
        "tasks": list(TASKS),
        "test_data_opened": False,
        "private_reference_opened": False,
        "llm_calls": 0,
    }
    frozen["freeze_sha256"] = content_hash(frozen)
    write_json(output / "freeze.json", frozen, immutable=True)
    return frozen


def verify_runtime(output: Path) -> dict:
    frozen = read_json(output / "freeze.json")
    if (
        content_hash({k: v for k, v in frozen.items() if k != "freeze_sha256"})
        != frozen["freeze_sha256"]
    ):
        raise ValueError("runtime freeze digest differs")
    RuntimePlan.model_validate(frozen["plan"])
    if (
        frozen["runtime"] != runtime_identity()
        or frozen["launcher_sha256"] != launch_identity()
    ):
        raise ValueError("runtime source, dependencies or launcher differ")
    for relative, digest in frozen["assets"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output.resolve()) or sha256(path) != digest:
            raise ValueError(f"runtime asset differs: {relative}")
    return frozen


def load_problem(output: Path, frozen: dict) -> tuple:
    """Compile the frozen model and derive scales from training data only."""
    parent = StagedFitPlan.model_validate(
        read_json(output / "source_freeze.json")["plan"]
    )
    dataset, context = load_data(output, parent)
    model = compile_candidate(
        CandidateModel.model_validate(read_json(output / "candidate.json")), context
    )
    scaler = TrainingScaler().fit(dataset.train)
    scales = {
        name: scaler.scales[f"target:{name}"].standard_deviation
        for name in context.targets
    }
    return parent, dataset, model, scales


def profile_anchor(
    output: Path, root: Path, identity: str, task: dict, frozen: dict, plan: RuntimePlan
) -> dict:
    """Compare all predeclared methods at one exact reviewed parameter vector."""
    parent, dataset, model, scales = load_problem(output, frozen)
    ids = frozen["profile_trajectory_ids"]
    training = DatasetSplit(
        dataset.train.name,
        tuple(
            next(t for t in dataset.train.trajectories if t.trajectory_id == name)
            for name in ids
        ),
        content_hash([dataset.train.fingerprint, ids]),
    )
    parameters = frozen["anchors"][task["anchor"]]
    deadline = monotonic() + plan.profile_seconds
    cases = {}

    def evaluate(name, method, rtol, atol):
        settings = parent.fit_config.model_copy(
            update={
                "integration_method": method,
                "relative_tolerance": rtol,
                "absolute_tolerance": atol,
            }
        )
        record = rollout_case(
            root / "cases" / name,
            content_hash([identity, name]),
            model,
            training,
            parameters,
            scales,
            settings,
            deadline,
            plan.call_seconds,
        )
        cases[name] = record
        return record

    # Tight references are attempted first, then matched timed repeats.
    for method in ("Radau", "DOP853"):
        evaluate(f"{method}_reference", method, 1e-10, 1e-12)
        evaluate(f"{method}_tight_0", method, 1e-9, 1e-11)
    for method in METHODS:
        for name, rtol, atol in (("current", 1e-7, 1e-9), ("tight", 1e-9, 1e-11)):
            for repeat in range(2):
                evaluate(f"{method}_{name}_{repeat}", method, rtol, atol)

    def values(name):
        return read_array(
            root / "cases" / name / "residual.npz", cases[name]["array_sha256"]
        )

    reference = None
    reference_checks = []
    for method in ("Radau", "DOP853"):
        names = (f"{method}_reference", f"{method}_tight_0")
        if all(cases[n]["status"] == "complete" for n in names):
            difference = residual_distance(values(names[0]), values(names[1]))
            valid = (
                difference["rms"] <= plan.reference_rms
                and difference["maximum_absolute"] <= plan.reference_maximum
            )
            reference_checks.append({"method": method, "valid": valid, **difference})
            if valid and reference is None:
                reference = names[0]
    comparisons = []
    method_gates = {}
    for method in METHODS:
        for tolerance in ("current", "tight"):
            names = [f"{method}_{tolerance}_{r}" for r in range(2)]
            good = reference is not None and all(
                cases[n]["status"] == "complete" for n in names
            )
            row = {"method": method, "tolerance": tolerance, "complete": bool(good)}
            if good:
                differences = [
                    residual_distance(values(n), values(reference)) for n in names
                ]
                row.update(
                    rms=max(d["rms"] for d in differences),
                    maximum_absolute=max(d["maximum_absolute"] for d in differences),
                    repeat_difference=residual_distance(
                        values(names[0]), values(names[1])
                    ),
                    median_seconds=float(
                        np.median([cases[n]["seconds"] for n in names])
                    ),
                )
                good = (
                    row["rms"] <= plan.accuracy_rms
                    and row["maximum_absolute"] <= plan.accuracy_maximum
                )
            row["accuracy_pass"] = bool(good)
            comparisons.append(row)
            if tolerance == "current":
                method_gates[method] = bool(good)
    reference_method = reference.split("_")[0] if reference else "Radau"
    settings = parent.fit_config.model_copy(
        update={
            "integration_method": reference_method,
            "relative_tolerance": 1e-9,
            "absolute_tolerance": 1e-11,
        }
    )
    derivatives = derivative_cases(
        root / "derivatives",
        content_hash([identity, "derivatives", reference_method]),
        model,
        training,
        parameters,
        scales,
        settings,
        deadline,
        plan.call_seconds,
        step_factor=plan.step,
        scale_floor=plan.scale_floor,
    )
    interpreter = interpreter_profile(
        root,
        content_hash([identity, "interpreter"]),
        model,
        training.trajectories[0],
        parameters,
        parent.fit_config,
        deadline,
        plan.call_seconds,
    )
    return {
        "identity": identity,
        "status": "complete" if reference else "reference_failed",
        "task": task,
        "parameters": parameters,
        "trajectory_ids": ids,
        "target_scales": scales,
        "reference": reference,
        "reference_checks": reference_checks,
        "method_gates": method_gates,
        "comparisons": comparisons,
        "cases": cases,
        "derivatives": derivatives,
        "interpreter": interpreter,
    }


def fit_guard(output: Path, frozen: dict, method: str) -> dict:
    """Require pointwise accuracy at every frozen anchor before fitting a method."""
    checks = []
    for task in frozen["tasks"][:3]:
        identity = content_hash([frozen["freeze_sha256"], task])
        profile = checkpoint(
            output / "results" / task["name"] / "result.json", identity
        )
        checks.append(
            {
                "anchor": task["anchor"],
                "pass": bool(
                    profile
                    and profile.get("status") == "complete"
                    and profile["method_gates"].get(method)
                ),
            }
        )
    return {"pass": all(c["pass"] for c in checks), "checks": checks}


def execute_runtime(output: Path, index: int) -> dict:
    """Run or resume one bounded task without changing model structure or data."""
    frozen = verify_runtime(output)
    if not 0 <= index < len(TASKS):
        raise ValueError("unknown runtime task index")
    task = frozen["tasks"][index]
    identity = content_hash([frozen["freeze_sha256"], task])
    root = output / "results" / task["name"]
    existing = checkpoint(root / "result.json", identity)
    if existing is not None:
        return existing
    root.mkdir(parents=True, exist_ok=True)
    plan = RuntimePlan.model_validate(frozen["plan"])
    if task["kind"] == "profile":
        result = profile_anchor(output, root, identity, task, frozen, plan)
    else:
        guard = fit_guard(output, frozen, task["method"])
        if not guard["pass"]:
            result = {
                "identity": identity,
                "task": task,
                "status": "accuracy_guard_failed",
                "guard": guard,
            }
        else:
            parent, dataset, model, scales = load_problem(output, frozen)
            settings = parent.fit_config.model_copy(
                update={
                    "integration_method": task["method"],
                    "relative_tolerance": 1e-7,
                    "absolute_tolerance": 1e-9,
                    "finite_difference_policy": task["policy"],
                    "finite_difference_step": plan.step,
                    "finite_difference_scale_floor": plan.scale_floor,
                    "number_of_starts": 1,
                    "maximum_function_evaluations": plan.maximum_nfev,
                    "maximum_wall_time_seconds": plan.fit_seconds,
                }
            )
            fit = checkpoint(root / "fit.json", identity)
            if fit is None:
                attempt = 0
                while (root / f"attempt-{attempt}").exists():
                    attempt += 1
                oracle = RolloutOracle(
                    model,
                    dataset.train,
                    scales,
                    settings,
                    root / f"attempt-{attempt}",
                    monotonic() + plan.fit_seconds,
                )
                fit = {
                    "identity": identity,
                    "settings": settings.model_dump(mode="json"),
                    "fit": instrumented_fit(
                        oracle,
                        frozen["anchors"]["all_ones"],
                        diff_step=None,
                        max_nfev=plan.maximum_nfev,
                        settings=settings,
                    ),
                }
                write_json(root / "fit.json", fit)
            replays = {}
            parameters = fit["fit"]["parameters"]
            for method in ("DOP853", "Radau"):
                record = checkpoint(root / f"replay_{method}.json", identity)
                if record is None:
                    tight = settings.model_copy(
                        update={
                            "integration_method": method,
                            "relative_tolerance": 1e-9,
                            "absolute_tolerance": 1e-11,
                        }
                    )
                    record = {
                        "identity": identity,
                        "replay": replay_parameters(
                            model, dataset, parameters, tight, plan.replay_seconds
                        )
                        if parameters
                        else None,
                    }
                    write_json(root / f"replay_{method}.json", _finite_payload(record))
                replays[method] = record["replay"]
            finite = all(
                replays[m]
                and all(
                    replays[m][s]["normalized_mse"] is not None
                    for s in ("train", "validation")
                )
                for m in replays
            )
            agreement = (
                {
                    s: abs(
                        replays["DOP853"][s]["normalized_mse"]
                        - replays["Radau"][s]["normalized_mse"]
                    )
                    for s in ("train", "validation")
                }
                if finite
                else None
            )
            agrees = finite and all(
                value <= plan.accuracy_rms for value in agreement.values()
            )
            result = {
                "identity": identity,
                "task": task,
                "status": "complete" if agrees else "replay_unverified",
                "guard": guard,
                **fit,
                "replays": replays,
                "replay_score_difference": agreement,
            }
    result.update(test_data_opened=False, private_reference_opened=False, llm_calls=0)
    result = _finite_payload(result)
    write_json(root / "result.json", result)
    return result


def summarize_runtime(output: Path) -> dict:
    """Retain all task outcomes including failed guards, missing work and timeouts."""
    frozen = verify_runtime(output)
    rows = []
    for task in frozen["tasks"]:
        identity = content_hash([frozen["freeze_sha256"], task])
        record = checkpoint(output / "results" / task["name"] / "result.json", identity)
        rows.append(
            {
                "task": task,
                "result": record,
                "status": (record or {}).get("status", "missing"),
            }
        )
    return {
        "freeze_sha256": frozen["freeze_sha256"],
        "plan": frozen["plan"],
        "profile_trajectory_ids": frozen["profile_trajectory_ids"],
        "rows": rows,
        "model_selection_performed": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "llm_calls": 0,
    }
