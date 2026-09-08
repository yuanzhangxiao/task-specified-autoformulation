"""Matched, source-pinned signed-offset experiment with independent ODE replays."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.data import DatasetSplit, TrainingScaler
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import simulate_trajectory
from autoformalism.fitting.runtime_probe import (
    read_array,
    residual_distance,
    save_array,
)
from autoformalism.fitting.stagnation import RolloutOracle, instrumented_fit
from autoformalism.rebuttal.fitter_diagnostic import (
    PUBLIC_FILES,
    _finite_payload,
    _write_bytes,
    read_json,
    runtime_identity,
    sha256,
    write_json,
)
from autoformalism.rebuttal.fitter_runtime import TASKS as PARENT_TASKS
from autoformalism.rebuttal.fitter_runtime import RuntimePlan
from autoformalism.rebuttal.fitter_stagnation import checkpoint
from autoformalism.rebuttal.staged_fit_probe import StagedFitPlan, load_data
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.staged_topology import content_hash

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
VARIANTS = ("original", "signed_offset")
STARTS = ("all_ones", "perturbed")
OLD_ARMS = ("fit_Radau_relative", "fit_Radau_scaled")
CASES = {
    "Radau_tight": ("Radau", 1e-9, 1e-11),
    "BDF_tight": ("BDF", 1e-9, 1e-11),
    "Radau_refined": ("Radau", 1e-10, 1e-12),
}
TASKS = tuple(
    [{"name": "numerical_guard", "kind": "guard"}]
    + [
        {
            "name": f"fit_{variant}_{start}",
            "kind": "fit",
            "variant": variant,
            "start": start,
        }
        for start in STARTS
        for variant in VARIANTS
    ]
    + [
        {"name": f"verify_{arm}", "kind": "previous_replay", "arm": arm}
        for arm in OLD_ARMS
    ]
)


class OffsetPlan(StrictSchema):
    """One predeclared domain change; all other fit choices are matched."""

    protocol: Literal["fitter-signed-offset-1"] = "fitter-signed-offset-1"
    source_code_sha256: Digest
    source_plan_sha256: Digest
    source_candidate_sha256: Digest
    offset_parameter: Identifier = "c"
    zero_gain_parameters: tuple[Identifier, ...] = ("k", "k_u")
    start_seed: int = Field(default=20260909, ge=0)
    fit_seconds: FiniteFloat = Field(default=900, gt=0, le=900)
    replay_seconds: FiniteFloat = Field(default=120, gt=0, le=120)
    guard_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    grace_seconds: FiniteFloat = Field(default=60, gt=0, le=60)
    maximum_nfev: int = Field(default=150, ge=1, le=300)
    profile_trajectories: int = Field(default=2, ge=1, le=4)
    step: FiniteFloat = Field(default=1e-4, gt=0, le=0.01)
    accuracy_rms: FiniteFloat = Field(default=1e-6, gt=0, le=1e-5)
    accuracy_maximum: FiniteFloat = Field(default=1e-5, gt=0, le=1e-4)

    @model_validator(mode="after")
    def distinct_gains(self) -> OffsetPlan:
        """Require explicit separate gains for the analytic baseline control."""
        if (
            len(set(self.zero_gain_parameters)) != len(self.zero_gain_parameters)
            or self.offset_parameter in self.zero_gain_parameters
        ):
            raise ValueError("zero gains must be distinct from each other and offset")
        return self

    def worker_seconds(self, index: int) -> float:
        """Bound every numerical phase separately inside a 30-minute allocation."""
        return self.grace_seconds + (
            self.guard_seconds
            if index == 0
            else (self.fit_seconds if TASKS[index]["kind"] == "fit" else 0)
            + 6 * self.replay_seconds
        )


def launch_identity() -> str:
    """Pin scheduler and supervisor code in addition to the Python package."""
    repo = Path(__file__).resolve().parents[3]
    return content_hash(
        {
            name: sha256(repo / name)
            for name in (
                "scripts/run_fitter_offset.py",
                "scripts/hpc/fitter_offset_delta.slurm",
                "scripts/hpc/submit_fitter_offset_delta.sh",
            )
        }
    )


def signed_offset_candidate(candidate: CandidateModel, name: str) -> CandidateModel:
    """Change exactly one existing parameter's role/domain, preserving equations."""
    payload = candidate.model_dump(mode="json")
    matches = [p for p in payload["parameters"] if p["name"] == name]
    if len(matches) != 1:
        raise ValueError("offset parameter must exist exactly once")
    parameter = matches[0]
    if (
        parameter["role"] != "nonnegative_coefficient"
        or parameter["domain"] != "nonnegative"
        or parameter["bounds"] is not None
    ):
        raise ValueError("offset control requires an unbounded nonnegative coefficient")
    parameter["role"] = "offset"
    parameter.pop("domain")
    return CandidateModel.model_validate(payload)


def baseline_report(dataset) -> dict:
    """Compute constant baselines with means and normalization from training only."""
    channels = dataset.roles.targets
    if len(channels) != 1:
        raise ValueError("this paired diagnostic requires one target")
    channel = channels[0]
    values = np.concatenate([t.targets[channel] for t in dataset.train.trajectories])
    scale = TrainingScaler().fit(dataset.train).scales[f"target:{channel}"]
    mean = float(np.mean(values))
    std = float(scale.standard_deviation)
    report = {"channel": channel, "training_mean": mean, "training_scale": std}
    for label, split in (("train", dataset.train), ("validation", dataset.validation)):
        y = np.concatenate([t.targets[channel] for t in split.trajectories])
        report[label] = {
            "count": len(y),
            "zero_nmse": float(np.mean((y / std) ** 2)),
            "training_mean_nmse": float(np.mean(((y - mean) / std) ** 2)),
        }
    return report


def prepare_offset(plan: OffsetPlan, source: Path, output: Path) -> dict:
    """Snapshot the exact runtime-v2 model, public assets and saved Radau vectors."""
    if source.resolve() == output.resolve():
        raise ValueError("use a new output directory")
    parent = read_json(source / "freeze.json")
    if content_hash(
        {k: v for k, v in parent.items() if k != "freeze_sha256"}
    ) != parent.get("freeze_sha256"):
        raise ValueError("source freeze digest differs")
    runtime_plan = RuntimePlan.model_validate(parent["plan"])
    if (
        content_hash(parent["plan"]) != plan.source_plan_sha256
        or parent["runtime"]["source_sha256"] != plan.source_code_sha256
        or parent["tasks"] != list(PARENT_TASKS)
    ):
        raise ValueError("source runtime code, plan or tasks differ")
    current = runtime_identity()
    if any(current[k] != parent["runtime"][k] for k in ("python", "packages")):
        raise ValueError("numerical dependencies differ from runtime-v2")
    previous = read_json(source / "previous_freeze.json")
    if (
        sha256(source / "previous_freeze.json")
        != parent["assets"].get("previous_freeze.json")
        or content_hash(previous["plan"]) != runtime_plan.source_plan_sha256
        or sha256(source / "source_freeze.json")
        != previous["plan"]["source_freeze_sha256"]
    ):
        raise ValueError("original model/public snapshot ancestry differs")
    original = read_json(source / "source_freeze.json")
    original_plan = StagedFitPlan.model_validate(original["plan"])
    prefix = f"public/phase_b_v1/{original_plan.benchmark_id}"
    names = [
        "candidate.json",
        "source_freeze.json",
        *[f"{prefix}/{f}" for f in PUBLIC_FILES],
    ]
    assets = {}
    for name in names:
        path = (source / name).resolve()
        digest = sha256(path)
        if (
            not path.is_relative_to(source.resolve())
            or parent["assets"].get(name) != digest
            or (name != "source_freeze.json" and original["assets"].get(name) != digest)
        ):
            raise ValueError(f"source asset differs: {name}")
        _write_bytes(output / name, path.read_bytes(), immutable=True)
        if sha256(output / name) != digest:
            raise ValueError("source changed during snapshot")
        assets[name] = digest
    if assets["candidate.json"] != plan.source_candidate_sha256:
        raise ValueError("reviewed candidate differs")
    candidate = CandidateModel.model_validate(read_json(output / "candidate.json"))
    if candidate.constraints or any(
        i.fixed_value is None for i in candidate.initial_conditions
    ):
        raise ValueError(
            "paired control requires no extra constraints and fixed initials"
        )
    if any(p.scope.value != "global" for p in candidate.parameters):
        raise ValueError("paired control requires global parameters")
    changed = signed_offset_candidate(candidate, plan.offset_parameter)
    write_json(
        output / "signed_offset.json", changed.model_dump(mode="json"), immutable=True
    )
    assets["signed_offset.json"] = sha256(output / "signed_offset.json")
    dataset, context = load_data(output, original_plan)
    baselines = baseline_report(dataset)
    rng = np.random.default_rng(plan.start_seed)
    parameter_names = sorted(p.name for p in candidate.parameters)
    starts = {
        "all_ones": dict.fromkeys(parameter_names, 1.0),
        "perturbed": {
            name: float(np.exp(rng.uniform(-np.log(2), np.log(2))))
            for name in parameter_names
        },
    }
    if not set(plan.zero_gain_parameters) <= set(parameter_names):
        raise ValueError("analytic baseline gains are undeclared")
    baseline_vector = {
        **starts["all_ones"],
        **dict.fromkeys(plan.zero_gain_parameters, 0.0),
        plan.offset_parameter: baselines["training_mean"],
    }
    for label, variant in (("original", candidate), ("signed_offset", changed)):
        model = compile_candidate(variant, context)
        oracle = RolloutOracle(
            model,
            dataset.train,
            {context.targets[0]: baselines["training_scale"]},
            original_plan.fit_config,
            output / "preparation",
            None,
        )
        for values in starts.values():
            oracle.vector(values)
        if label == "signed_offset":
            oracle.vector(baseline_vector)
    prior_parameters = {}
    for arm in OLD_ARMS:
        task = next(t for t in parent["tasks"] if t["name"] == arm)
        fit = read_json(source / "results" / arm / "fit.json")
        result = read_json(source / "results" / arm / "result.json")
        identity = content_hash([parent["freeze_sha256"], task])
        if (
            fit.get("identity") != identity
            or result.get("identity") != identity
            or fit.get("fit") != result.get("fit")
            or result.get("test_data_opened") is not False
            or result.get("private_reference_opened") is not False
        ):
            raise ValueError(f"previous fit provenance differs: {arm}")
        prior_parameters[arm] = fit["fit"]["parameters"]
        if not prior_parameters[arm]:
            raise ValueError(f"previous fit has no parameters: {arm}")
        name = f"previous/{arm}.json"
        write_json(output / name, fit, immutable=True)
        assets[name] = sha256(output / name)
    write_json(output / "runtime_v2_freeze.json", parent, immutable=True)
    assets["runtime_v2_freeze.json"] = sha256(output / "runtime_v2_freeze.json")
    ids = sorted(t.trajectory_id for t in dataset.train.trajectories)[
        : plan.profile_trajectories
    ]
    if len(ids) != plan.profile_trajectories:
        raise ValueError("insufficient profile trajectories")
    frozen = {
        "plan": plan.model_dump(mode="json"),
        "runtime": current,
        "launcher_sha256": launch_identity(),
        "assets": assets,
        "starts": starts,
        "baseline_vector": baseline_vector,
        "baselines": baselines,
        "previous_parameters": prior_parameters,
        "profile_trajectory_ids": ids,
        "tasks": list(TASKS),
        "test_data_opened": False,
        "private_reference_opened": False,
        "llm_calls": 0,
    }
    frozen["freeze_sha256"] = content_hash(frozen)
    write_json(output / "freeze.json", frozen, immutable=True)
    return frozen


def verify_offset(output: Path) -> dict:
    """Fail closed on modified data, model, numerical environment or launchers."""
    frozen = read_json(output / "freeze.json")
    OffsetPlan.model_validate(frozen["plan"])
    if (
        content_hash({k: v for k, v in frozen.items() if k != "freeze_sha256"})
        != frozen["freeze_sha256"]
        or frozen["runtime"] != runtime_identity()
        or frozen["launcher_sha256"] != launch_identity()
        or frozen["tasks"] != list(TASKS)
    ):
        raise ValueError("offset freeze, runtime or launcher differs")
    for name, digest in frozen["assets"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or sha256(path) != digest:
            raise ValueError(f"offset asset differs: {name}")
    return frozen


def load_problem(output: Path, variant: str) -> tuple:
    """Load only development data and the explicitly selected frozen variant."""
    parent = StagedFitPlan.model_validate(
        read_json(output / "source_freeze.json")["plan"]
    )
    dataset, context = load_data(output, parent)
    filename = {"original": "candidate.json", "signed_offset": "signed_offset.json"}[
        variant
    ]
    model = compile_candidate(
        CandidateModel.model_validate(read_json(output / filename)), context
    )
    return parent, dataset, model


def _mean_square(values: np.ndarray) -> float:
    """Avoid overflowing a sum of finite squares, including on ARM longdouble."""
    maximum = float(np.max(np.abs(values)))
    if maximum == 0:
        return 0.0
    return float(np.mean((values / maximum) ** 2)) * maximum * maximum


def replay_split(
    root,
    identity,
    model,
    split,
    parameters,
    scale,
    settings,
    seconds,
    outer_deadline=None,
) -> dict:
    """Checkpoint each trajectory; score unclipped predictions with a split budget."""
    existing = checkpoint(root / "result.json", identity)
    if existing is not None:
        # A completed JSON does not authorize silently changed numerical arrays.
        for item in existing["trajectories"]:
            if item["status"] == "complete":
                read_array(root / item["array"], item["array_sha256"])
        return existing
    deadline = monotonic() + seconds
    if outer_deadline is not None:
        deadline = min(deadline, outer_deadline)
    channel = model.validated.context.targets[0]
    records, arrays = [], []
    for index, trajectory in enumerate(split.trajectories):
        key = content_hash([identity, trajectory.trajectory_id])
        path = root / f"trajectory-{index}.json"
        record = checkpoint(path, key)
        if record is None:
            started = monotonic()
            record = {"identity": key, "trajectory_id": trajectory.trajectory_id}
            try:
                if started >= deadline:
                    raise TimeoutError("replay split wall-clock limit reached")
                simulation = simulate_trajectory(
                    model,
                    trajectory,
                    parameters,
                    {},
                    settings,
                    deadline=deadline,
                    reset_observed_states=False,
                )
                if not simulation.success:
                    raise ValueError(simulation.message)
                yhat = simulation.predictions[channel]
                with np.errstate(over="ignore", invalid="ignore"):
                    values = np.column_stack(
                        (yhat, (yhat - trajectory.targets[channel]) / scale)
                    )
                    finite = np.isfinite(values).all() and np.isfinite(values**2).all()
                if not finite:
                    raise ValueError("nonfinite prediction, residual or squared metric")
                filename = f"trajectory-{index}.npz"
                record.update(
                    status="complete",
                    array=filename,
                    array_sha256=save_array(root / filename, values),
                )
            except (TimeoutError, ValueError, ArithmeticError, RuntimeError) as error:
                record.update(
                    status="timeout" if isinstance(error, TimeoutError) else "failed",
                    error=str(error),
                )
            record["seconds"] = monotonic() - started
            write_json(path, record)
        records.append(record)
        if record["status"] == "complete":
            arrays.append(read_array(root / record["array"], record["array_sha256"]))
    failures = [
        {"trajectory_id": r["trajectory_id"], "message": r["error"]}
        for r in records
        if r["status"] != "complete"
    ]
    result = {
        "identity": identity,
        "settings": settings.model_dump(mode="json"),
        "status": "complete" if not failures else "incomplete",
        "failures": failures,
        "trajectories": records,
        "normalized_mse": None,
    }
    if not failures:
        values = np.concatenate(arrays)
        prediction_mean = float(np.mean(values[:, 0]))
        result.update(
            normalized_mse=_mean_square(values[:, 1]),
            prediction_mean=prediction_mean,
            prediction_rms=float(np.sqrt(_mean_square(values[:, 0]))),
            prediction_std=float(np.sqrt(_mean_square(values[:, 0] - prediction_mean))),
            prediction_minimum=float(np.min(values[:, 0])),
            prediction_maximum=float(np.max(values[:, 0])),
        )
    write_json(root / "result.json", _finite_payload(result))
    return result


def residuals(root: Path, record: dict) -> np.ndarray:
    """Recover exact normalized residuals for pointwise solver comparison."""
    return np.concatenate(
        [
            read_array(root / t["array"], t["array_sha256"])[:, 1]
            for t in record["trajectories"]
        ]
    )


def verify_replays(
    root,
    identity,
    model,
    dataset,
    parameters,
    settings,
    plan,
    scale,
    *,
    training_only=False,
    outer_deadline=None,
) -> dict:
    """Compare independent methods and tolerance refinement on identical samples."""
    replays, checks = {}, []
    splits = {"train": dataset.train}
    if not training_only:
        splits["validation"] = dataset.validation
    cases = dict(CASES)
    if training_only:
        cases["Radau_fit"] = ("Radau", 1e-7, 1e-9)
    for label, split in splits.items():
        for name, (method, rtol, atol) in cases.items():
            config = settings.model_copy(
                update={
                    "integration_method": method,
                    "relative_tolerance": rtol,
                    "absolute_tolerance": atol,
                }
            )
            case = f"{name}/{label}"
            replays[case] = replay_split(
                root / case,
                content_hash([identity, case]),
                model,
                split,
                parameters,
                scale,
                config,
                plan.replay_seconds,
                outer_deadline,
            )
        comparisons = ["BDF_tight", "Radau_refined"]
        if training_only:
            comparisons.append("Radau_fit")
        for other in comparisons:
            names = (f"Radau_tight/{label}", f"{other}/{label}")
            complete = all(replays[n]["status"] == "complete" for n in names)
            difference = (
                residual_distance(*(residuals(root / n, replays[n]) for n in names))
                if complete
                else {}
            )
            checks.append(
                {
                    "split": label,
                    "cases": names,
                    **difference,
                    "pass": bool(
                        complete
                        and difference["rms"] <= plan.accuracy_rms
                        and difference["maximum_absolute"] <= plan.accuracy_maximum
                    ),
                }
            )
    return {
        "replays": replays,
        "checks": checks,
        "status": "complete" if all(c["pass"] for c in checks) else "replay_unverified",
    }


def numerical_guard(output, root, identity, frozen, plan) -> dict:
    """Verify both common starts and an exact signed baseline before fitting."""
    parent, dataset, model = load_problem(output, "original")
    ids = frozen["profile_trajectory_ids"]
    subset = DatasetSplit(
        dataset.train.name,
        tuple(t for t in dataset.train.trajectories if t.trajectory_id in ids),
        content_hash([dataset.train.fingerprint, ids]),
    )
    # DevelopmentDataset is frozen; use a tiny attribute view without adding test data.
    from types import SimpleNamespace

    samples = SimpleNamespace(train=subset)
    deadline = monotonic() + plan.guard_seconds
    cases = {}
    for name in (*STARTS, "signed_mean"):
        if name == "signed_mean":
            _, _, model = load_problem(output, "signed_offset")
        parameters = (
            frozen["baseline_vector"]
            if name == "signed_mean"
            else frozen["starts"][name]
        )
        record = verify_replays(
            root / name,
            content_hash([identity, name]),
            model,
            samples,
            parameters,
            parent.fit_config,
            plan,
            frozen["baselines"]["training_scale"],
            training_only=True,
            outer_deadline=deadline,
        )
        if name == "signed_mean":
            mean = frozen["baselines"]["training_mean"]
            record["analytic_baseline_pass"] = all(
                r["status"] == "complete"
                and abs(r["prediction_mean"] - mean) < 1e-8
                and r["prediction_std"] < 1e-8
                for r in record["replays"].values()
            )
        cases[name] = record
    good = (
        all(r["status"] == "complete" for r in cases.values())
        and cases["signed_mean"]["analytic_baseline_pass"]
    )
    return {"status": "complete" if good else "accuracy_guard_failed", "cases": cases}


def execute_offset(output: Path, index: int) -> dict:
    """Resume one predeclared task; optimization uses training data exclusively."""
    frozen = verify_offset(output)
    if not 0 <= index < len(TASKS):
        raise ValueError("unknown offset task index")
    plan = OffsetPlan.model_validate(frozen["plan"])
    task = TASKS[index]
    identity = content_hash([frozen["freeze_sha256"], task])
    root = output / "results" / task["name"]
    existing = checkpoint(root / "result.json", identity)
    if existing is not None:
        return existing
    if task["kind"] == "guard":
        result = numerical_guard(output, root, identity, frozen, plan)
    else:
        variant = task.get("variant", "original")
        parent, dataset, model = load_problem(output, variant)
        settings = parent.fit_config.model_copy(
            update={
                "integration_method": "Radau",
                "relative_tolerance": 1e-7,
                "absolute_tolerance": 1e-9,
                "finite_difference_policy": "scaled",
                "finite_difference_step": plan.step,
                "finite_difference_scale_floor": 1.0,
                "number_of_starts": 1,
                "maximum_function_evaluations": plan.maximum_nfev,
                "maximum_wall_time_seconds": plan.fit_seconds,
            }
        )
        if task["kind"] == "fit":
            guard = checkpoint(
                output / "results/numerical_guard/result.json",
                content_hash([frozen["freeze_sha256"], TASKS[0]]),
            )
            if not guard or guard["status"] != "complete":
                result = {
                    "status": "accuracy_guard_failed",
                    "error": "numerical guard incomplete or failed",
                }
                result.update(
                    identity=identity,
                    task=task,
                    test_data_opened=False,
                    private_reference_opened=False,
                    llm_calls=0,
                )
                write_json(root / "result.json", result)
                return result
            fit = checkpoint(root / "fit.json", identity)
            if fit is None:
                attempt = 0
                while (root / f"attempt-{attempt}").exists():
                    attempt += 1
                oracle = RolloutOracle(
                    model,
                    dataset.train,
                    {
                        frozen["baselines"]["channel"]: frozen["baselines"][
                            "training_scale"
                        ]
                    },
                    settings,
                    root / f"attempt-{attempt}",
                    monotonic() + plan.fit_seconds,
                )
                fit = {
                    "identity": identity,
                    "settings": settings.model_dump(mode="json"),
                    "fit": instrumented_fit(
                        oracle,
                        frozen["starts"][task["start"]],
                        diff_step=None,
                        max_nfev=plan.maximum_nfev,
                        settings=settings,
                    ),
                }
                write_json(root / "fit.json", fit)
            parameters = fit["fit"]["parameters"]
            result = {**fit}
        else:
            parameters = frozen["previous_parameters"][task["arm"]]
            result = {}
        result["parameters"] = parameters
        if parameters:
            result.update(
                verify_replays(
                    root / "replays",
                    identity,
                    model,
                    dataset,
                    parameters,
                    settings,
                    plan,
                    frozen["baselines"]["training_scale"],
                )
            )
        else:
            result.update(
                status="fit_failed", error="no finite parameter vector returned"
            )
    result.update(
        identity=identity,
        task=task,
        test_data_opened=False,
        private_reference_opened=False,
        llm_calls=0,
    )
    result = _finite_payload(result)
    write_json(root / "result.json", result)
    return result


def summarize_offset(output: Path) -> dict:
    """Retain missing tasks, solver failures and both starts without selection."""
    frozen = verify_offset(output)
    rows = []
    for task in TASKS:
        result = checkpoint(
            output / "results" / task["name"] / "result.json",
            content_hash([frozen["freeze_sha256"], task]),
        )
        rows.append(
            {
                "task": task,
                "status": (result or {}).get("status", "missing"),
                "result": result,
            }
        )
    return {
        "freeze_sha256": frozen["freeze_sha256"],
        "plan": frozen["plan"],
        "baselines": frozen["baselines"],
        "starts": frozen["starts"],
        "rows": rows,
        "model_selection_performed": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "llm_calls": 0,
    }
