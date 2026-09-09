"""Known-attainable active-dynamics controls for the unchanged rollout fitter."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from autoformalism.data import DatasetSplit, DevelopmentDataset, SplitName, Trajectory
from autoformalism.data.models import TierRoles
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig
from autoformalism.fitting.runtime_probe import (
    read_array,
    save_array,
)
from autoformalism.fitting.stagnation import (
    RolloutOracle,
    inspect_steps,
    instrumented_fit,
)
from autoformalism.rebuttal.fitter_diagnostic import (
    _finite_payload,
    read_json,
    runtime_identity,
    sha256,
    write_json,
)
from autoformalism.rebuttal.fitter_offset import baseline_report, verify_replays
from autoformalism.rebuttal.fitter_stagnation import checkpoint
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema
from autoformalism.staged_topology import content_hash

STARTS = ("all_ones", "broad", "near_truth_control")
CONTEXT = ValidationContext(targets=("v01",), external_inputs=("u01",))


class RecoveryTruth(StrictSchema):
    """An interior parameter vector with nonzero gains and explicit time units."""

    c: FiniteFloat
    k: FiniteFloat = Field(gt=0)
    k_p: FiniteFloat = Field(gt=0)
    k_u: FiniteFloat = Field(gt=0)
    tau: FiniteFloat = Field(gt=0)
    tau_p: FiniteFloat = Field(gt=0)
    tau_f: FiniteFloat = Field(gt=0)


class RecoveryCase(StrictSchema):
    """One prespecified synthetic system; fitting receives only its observations."""

    name: Identifier
    truth: RecoveryTruth


class RecoveryInput(StrictSchema):
    """A piecewise-linear excitation independent of any fitted result."""

    name: Identifier
    split: Literal["train", "validation"]
    times: tuple[FiniteFloat, ...] = Field(min_length=2)
    values: tuple[FiniteFloat, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def aligned(self) -> RecoveryInput:
        """Require ordered knots, matching values, and a zero initial time."""
        if (
            len(self.times) != len(self.values)
            or self.times[0] != 0
            or any(b <= a for a, b in pairwise(self.times))
        ):
            raise ValueError("input knots must be aligned and strictly increasing")
        return self


class RecoveryPlan(StrictSchema):
    """A small noiseless identifiability/recovery diagnostic, not a search run."""

    protocol: Literal["fitter-active-recovery-1"] = "fitter-active-recovery-1"
    cases: tuple[RecoveryCase, ...] = Field(min_length=1, max_length=2)
    inputs: tuple[RecoveryInput, ...] = Field(min_length=2, max_length=8)
    sample_step: FiniteFloat = Field(default=0.2, ge=1e-6)
    seed: int = Field(default=20260910, ge=0)
    fit_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    guard_seconds: FiniteFloat = Field(default=300, gt=0, le=300)
    replay_seconds: FiniteFloat = Field(default=60, gt=0, le=60)
    grace_seconds: FiniteFloat = Field(default=60, gt=0, le=60)
    maximum_nfev: int = Field(default=150, ge=1, le=150)
    step: FiniteFloat = Field(default=1e-4, gt=0, le=0.01)
    accuracy_rms: FiniteFloat = Field(default=1e-6, gt=0, le=1e-5)
    accuracy_maximum: FiniteFloat = Field(default=1e-5, gt=0, le=1e-4)
    truth_nmse_tolerance: FiniteFloat = Field(default=1e-10, gt=0, le=1e-8)
    recovery_nmse: FiniteFloat = Field(default=1e-4, gt=0, le=0.01)
    minimum_component_sd_fraction: FiniteFloat = Field(default=0.01, gt=0)

    @model_validator(mode="after")
    def unique_aligned_cases(self) -> RecoveryPlan:
        """Prevent split overlap and hidden interpolation changes at input corners."""
        if len({c.name for c in self.cases}) != len(self.cases):
            raise ValueError("duplicate case name")
        if len({i.name for i in self.inputs}) != len(self.inputs):
            raise ValueError("input identifiers must be unique across splits")
        if {i.split for i in self.inputs} != {"train", "validation"}:
            raise ValueError("both development splits are required")
        if len({i.times[-1] for i in self.inputs}) != 1:
            raise ValueError("all inputs must have the same observation horizon")
        for item in self.inputs:
            samples = np.asarray(item.times) / self.sample_step
            if not np.allclose(samples, np.rint(samples), rtol=0, atol=1e-9):
                raise ValueError("every input corner must lie on the sampling grid")
        if self.inputs[0].times[-1] / self.sample_step > 1000:
            raise ValueError("diagnostic is limited to 1001 samples per trajectory")
        return self

    def settings(self) -> FitConfig:
        """Use the corrected diagnostic policy without altering production defaults."""
        return FitConfig(
            integration_backend="solve_ivp",
            integration_method="Radau",
            allow_derivative_regression=False,
            relative_tolerance=1e-7,
            absolute_tolerance=1e-9,
            finite_difference_policy="scaled",
            finite_difference_step=self.step,
            finite_difference_scale_floor=1.0,
            maximum_function_evaluations=self.maximum_nfev,
            maximum_wall_time_seconds=self.fit_seconds,
        )

    def worker_seconds(self, kind: str) -> float:
        """Include separately budgeted full-split verification and cleanup time."""
        return self.grace_seconds + (
            self.guard_seconds
            if kind == "guard"
            else self.fit_seconds + 6 * self.replay_seconds
        )


def recovery_candidate() -> CandidateModel:
    """Encode the reviewed signed-offset equation family as a synthetic fixture."""
    return CandidateModel.model_validate(
        {
            "candidate_id": "active_recovery_fixture",
            "parent_candidate_id": None,
            "states": [{"name": n, "kind": "latent"} for n in ("m", "p", "f")],
            "state_equations": [
                {"state": "m", "rhs": "-m/tau + k_u*u01"},
                {"state": "p", "rhs": "-p/tau_p + m"},
                {"state": "f", "rhs": "k*v01**2/(1+v01**2) - f/tau_f"},
            ],
            "processes": [
                {
                    "name": "v01",
                    "expression": "k*f**2/(1+f**2) + m + k_p*p + k_u*u01 + c",
                }
            ],
            "observation_mappings": [{"channel": "v01", "expression": "v01"}],
            "parameters": [
                {"name": n, "role": role, "scope": "global"}
                for role, names in (
                    ("offset", ("c",)),
                    ("nonnegative_coefficient", ("k", "k_p", "k_u")),
                    ("time_constant", ("tau", "tau_p", "tau_f")),
                )
                for n in names
            ],
            "initial_conditions": [
                {"state": n, "fixed_value": 0, "scope": "global"}
                for n in ("m", "p", "f")
            ],
        }
    )


def _output(state: np.ndarray, u: np.ndarray | float, p: dict) -> np.ndarray:
    """Independent handwritten observation equation, in reference state order."""
    m, memory, feedback = state
    return (
        p["c"]
        + p["k"] * feedback**2 / (1 + feedback**2)
        + m
        + p["k_p"] * memory
        + p["k_u"] * u
    )


def reference_rollout(
    forcing: RecoveryInput,
    truth: dict,
    time: np.ndarray,
    method: str,
    deadline: float,
) -> np.ndarray:
    """Generate data independently of the restricted compiler and rollout wrapper.

    Integrate each declared forcing segment separately with a handwritten RHS.
    Return time, input, output, m, p, f; hidden states are for generator audit only.
    """
    states = np.empty((len(time), 3))
    states[0] = 0
    state = np.zeros(3)
    for left, right, u0, u1 in zip(
        forcing.times[:-1],
        forcing.times[1:],
        forcing.values[:-1],
        forcing.values[1:],
        strict=True,
    ):
        slope = (u1 - u0) / (right - left)

        def rhs(t, values, u0=u0, slope=slope, left=left):
            if monotonic() >= deadline:
                raise TimeoutError("reference generation deadline reached")
            u = u0 + slope * (t - left)
            m, memory, feedback = values
            output = _output(values, u, truth)
            return (
                -m / truth["tau"] + truth["k_u"] * u,
                -memory / truth["tau_p"] + m,
                truth["k"] * output**2 / (1 + output**2) - feedback / truth["tau_f"],
            )

        indices = np.flatnonzero((time > left + 1e-10) & (time <= right + 1e-10))
        evaluation_times = np.minimum(time[indices], right)
        solved = solve_ivp(
            rhs,
            (left, right),
            state,
            t_eval=evaluation_times,
            method=method,
            rtol=1e-11,
            atol=1e-13,
        )
        if not solved.success or not np.isfinite(solved.y).all():
            raise ValueError(f"reference integration failed: {solved.message}")
        states[indices] = solved.y.T
        state = solved.y[:, -1]
    u = np.interp(time, forcing.times, forcing.values)
    return np.column_stack((time, u, _output(states.T, u, truth), states))


def exact_memories(forcing: RecoveryInput, truth: dict, time: np.ndarray) -> np.ndarray:
    """Verify the linear memory subsystem by an augmented matrix exponential."""
    state = np.array([0.0, 0.0, forcing.values[0], 1.0])
    values = [state[:2].copy()]
    for left, right in pairwise(time):
        u0, u1 = np.interp([left, right], forcing.times, forcing.values)
        matrix = np.array(
            [
                [-1 / truth["tau"], 0, truth["k_u"], 0],
                [1, -1 / truth["tau_p"], 0, 0],
                [0, 0, 0, (u1 - u0) / (right - left)],
                [0, 0, 0, 0],
            ]
        )
        state[2] = u0
        state = expm(matrix * (right - left)) @ state
        values.append(state[:2].copy())
    return np.asarray(values)


def _launcher_hash() -> str:
    repo = Path(__file__).resolve().parents[3]
    return content_hash(
        {
            p: sha256(repo / p)
            for p in (
                "scripts/run_fitter_recovery.py",
                "scripts/hpc/fitter_recovery_delta.slurm",
                "scripts/hpc/submit_fitter_recovery_delta.sh",
            )
        }
    )


def prepare_recovery(plan: RecoveryPlan, output: Path) -> dict:
    """Freeze truth, excitation, starts and runtime before any numerical work."""
    rng = np.random.default_rng(plan.seed)
    names = sorted(RecoveryTruth.model_fields)
    broad = {n: float(np.exp(rng.uniform(np.log(0.25), np.log(4)))) for n in names}
    broad["c"] = float(rng.uniform(-2, 2))
    starts = {}
    for case in plan.cases:
        truth = case.truth.model_dump()
        starts[case.name] = {
            "all_ones": dict.fromkeys(names, 1.0),
            "broad": broad,
            "near_truth_control": {
                n: float(truth[n] * (0.85 if i % 2 else 1.15))
                if n != "c"
                else truth[n] + 0.2
                for i, n in enumerate(names)
            },
        }
    tasks = [
        {"name": f"guard_{c.name}", "kind": "guard", "case": c.name} for c in plan.cases
    ] + [
        {"name": f"fit_{c.name}_{s}", "kind": "fit", "case": c.name, "start": s}
        for c in plan.cases
        for s in STARTS
    ]
    candidate = recovery_candidate().model_dump(mode="json")
    write_json(output / "candidate.json", candidate, immutable=True)
    frozen = {
        "plan": plan.model_dump(mode="json"),
        "candidate_sha256": sha256(output / "candidate.json"),
        "runtime": runtime_identity(),
        "launcher_sha256": _launcher_hash(),
        "starts": starts,
        "tasks": tasks,
        "benchmark_data_opened": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "llm_calls": 0,
    }
    frozen["freeze_sha256"] = content_hash(frozen)
    write_json(output / "freeze.json", frozen, immutable=True)
    return frozen


def verify_recovery(output: Path) -> dict:
    """Keep an interrupted experiment tied to its exact generated-data recipe."""
    frozen = read_json(output / "freeze.json")
    if (
        frozen["freeze_sha256"]
        != content_hash({k: v for k, v in frozen.items() if k != "freeze_sha256"})
        or frozen["runtime"] != runtime_identity()
        or frozen["launcher_sha256"] != _launcher_hash()
        or frozen["candidate_sha256"] != sha256(output / "candidate.json")
    ):
        raise ValueError("recovery freeze, runtime, candidate or launcher differs")
    RecoveryPlan.model_validate(frozen["plan"])
    return frozen


def _reference_file(
    root: Path,
    identity: str,
    forcing: RecoveryInput,
    truth: dict,
    time: np.ndarray,
    method: str,
    deadline: float,
) -> tuple[np.ndarray, dict]:
    path = root / f"{forcing.name}_{method}.json"
    key = content_hash([identity, forcing.model_dump(mode="json"), method])
    record = checkpoint(path, key)
    if record is None:
        started = monotonic()
        values = reference_rollout(forcing, truth, time, method, deadline)
        filename = f"{forcing.name}_{method}.npz"
        record = {
            "identity": key,
            "array": filename,
            "array_sha256": save_array(root / filename, values),
            "seconds": monotonic() - started,
        }
        write_json(path, record)
    return read_array(root / record["array"], record["array_sha256"]), record


def _dataset(root: Path, records: list[dict], identity: str) -> DevelopmentDataset:
    """Only observations and supplied forcing reach the production residual."""
    splits = {}
    for label, name in (
        ("train", SplitName.TRAIN),
        ("validation", SplitName.VALIDATION),
    ):
        trajectories = []
        for record in records:
            if record["split"] != label:
                continue
            a = read_array(root / record["array"], record["array_sha256"])
            trajectories.append(
                Trajectory(
                    record["name"],
                    a[:, 0],
                    {"v01": a[:, 2]},
                    {},
                    {"u01": a[:, 1]},
                    {},
                    {},
                )
            )
        splits[label] = DatasetSplit(
            name, tuple(trajectories), content_hash([identity, label])
        )
    return DevelopmentDataset(
        "synthetic_active_recovery",
        "diagnostic",
        TierRoles(targets=("v01",)),
        splits["train"],
        splits["validation"],
    )


def _guard(output: Path, root: Path, frozen: dict, task: dict, identity: str) -> dict:
    plan = RecoveryPlan.model_validate(frozen["plan"])
    case = next(c for c in plan.cases if c.name == task["case"])
    truth = case.truth.model_dump()
    deadline = monotonic() + plan.guard_seconds
    count = round(plan.inputs[0].times[-1] / plan.sample_step) + 1
    time = np.arange(count) * plan.sample_step
    records, reference_checks, components = [], [], []
    reference_root = root / "reference"
    for forcing in plan.inputs:
        first, record = _reference_file(
            reference_root,
            identity,
            forcing,
            truth,
            time,
            "Radau",
            deadline,
        )
        second, _ = _reference_file(
            reference_root,
            identity,
            forcing,
            truth,
            time,
            "DOP853",
            deadline,
        )
        memories = exact_memories(forcing, truth, time)
        reference_checks.append(
            {
                "trajectory_id": forcing.name,
                "output_maximum_absolute": float(
                    np.max(np.abs(first[:, 2] - second[:, 2]))
                ),
                "exact_memory_maximum_absolute": float(
                    np.max(np.abs(first[:, 3:5] - memories))
                ),
            }
        )
        second_record = read_json(reference_root / f"{forcing.name}_DOP853.json")
        records.append(
            {
                **record,
                "name": forcing.name,
                "split": forcing.split,
                "comparison_reference": second_record,
            }
        )
        if forcing.split == "train":
            components.append(
                np.column_stack(
                    (
                        truth["k_u"] * first[:, 1],
                        first[:, 3],
                        truth["k_p"] * first[:, 4],
                        truth["k"] * first[:, 5] ** 2 / (1 + first[:, 5] ** 2),
                    )
                )
            )
    dataset = _dataset(reference_root, records, identity)
    baselines = baseline_report(dataset)
    scale = baselines["training_scale"]
    component_sd = np.std(np.concatenate(components), axis=0) / scale
    activity = dict(
        zip(
            ("direct_input", "m", "p_readout", "nonlinear_feedback"),
            map(float, component_sd),
            strict=True,
        )
    )
    reference_pass = all(
        c["output_maximum_absolute"] / scale <= 1e-7
        and c["exact_memory_maximum_absolute"] / scale <= 1e-7
        for c in reference_checks
    )
    activity_pass = (
        scale > 0.05 and min(activity.values()) >= plan.minimum_component_sd_fraction
    )
    model = compile_candidate(
        CandidateModel.model_validate(read_json(output / "candidate.json")), CONTEXT
    )
    verification = verify_replays(
        root / "truth_replays",
        identity,
        model,
        dataset,
        truth,
        plan.settings(),
        plan,
        scale,
        outer_deadline=deadline,
    )
    truth_pass = verification["status"] == "complete" and all(
        r.get("normalized_mse") is not None
        and r["normalized_mse"] <= plan.truth_nmse_tolerance
        for r in verification["replays"].values()
    )
    subset = DatasetSplit(
        dataset.train.name,
        dataset.train.trajectories[:1],
        content_hash([identity, "profile"]),
    )
    profiles = {}
    for label, anchor in (
        ("truth", truth),
        ("all_ones", frozen["starts"][case.name]["all_ones"]),
    ):
        directory = root / f"derivatives_{label}"
        oracle = RolloutOracle(
            model,
            subset,
            {"v01": scale},
            plan.settings(),
            directory / "calls",
            deadline,
        )
        try:
            profile = inspect_steps(
                oracle,
                anchor,
                (plan.step, plan.step / 10),
                directory / "points",
                content_hash([identity, label]),
            )
        except (TimeoutError, ValueError) as error:
            profile = {"status": "unavailable", "error": str(error)}
        profile["integration_failures"] = oracle.failures.count
        profiles[label] = profile
    good = reference_pass and activity_pass and truth_pass
    return {
        "status": "complete" if good else "truth_guard_failed",
        "records": records,
        "baselines": baselines,
        "reference_checks": reference_checks,
        "reference_pass": reference_pass,
        "component_sd_over_target_sd": activity,
        "activity_pass": activity_pass,
        "truth_pass": truth_pass,
        "truth_verification": verification,
        "derivative_profiles": profiles,
        "derivatives_are_diagnostic_not_an_identifiability_certificate": True,
    }


def _fit(output: Path, root: Path, frozen: dict, task: dict, identity: str) -> dict:
    plan = RecoveryPlan.model_validate(frozen["plan"])
    case = next(c for c in plan.cases if c.name == task["case"])
    guard_task = next(
        t for t in frozen["tasks"] if t["kind"] == "guard" and t["case"] == case.name
    )
    guard_root = output / "results" / guard_task["name"]
    guard_id = content_hash([frozen["freeze_sha256"], guard_task])
    guard = checkpoint(guard_root / "result.json", guard_id)
    if guard is None or guard["status"] != "complete":
        return {"status": "truth_guard_failed", "error": "case guard did not pass"}
    verify_result_arrays(output, guard)
    dataset = _dataset(guard_root / "reference", guard["records"], guard_id)
    scale = guard["baselines"]["training_scale"]
    model = compile_candidate(
        CandidateModel.model_validate(read_json(output / "candidate.json")), CONTEXT
    )
    settings = plan.settings()
    fitted = checkpoint(root / "fit.json", identity)
    if fitted is None:
        attempt = 0
        while (root / f"attempt-{attempt}").exists():
            attempt += 1
        oracle = RolloutOracle(
            model,
            dataset.train,
            {"v01": scale},
            settings,
            root / f"attempt-{attempt}",
            monotonic() + plan.fit_seconds,
        )
        fitted = {
            "identity": identity,
            "fit": instrumented_fit(
                oracle,
                frozen["starts"][case.name][task["start"]],
                diff_step=None,
                max_nfev=plan.maximum_nfev,
                settings=settings,
            ),
        }
        write_json(root / "fit.json", fitted)
    parameters = fitted["fit"]["parameters"]
    if parameters is None:
        return {**fitted, "status": "fit_failed", "error": "no finite fitted vector"}
    verification = verify_replays(
        root / "replays", identity, model, dataset, parameters, settings, plan, scale
    )
    scores = {
        label: verification["replays"][f"Radau_tight/{label}"].get("normalized_mse")
        for label in ("train", "validation")
    }
    ratios = {
        label: verification["replays"][f"Radau_tight/{label}"].get("prediction_std")
        for label in ("train", "validation")
    }
    ratios = {
        label: value / scale if value is not None else None
        for label, value in ratios.items()
    }
    recovered = verification["status"] == "complete" and all(
        v is not None and v <= plan.recovery_nmse for v in scores.values()
    )
    truth = case.truth.model_dump()
    return {
        **fitted,
        **verification,
        "trajectory_recovered": recovered,
        "training_only_optimization": True,
        "oracle_assisted_start": task["start"] == "near_truth_control",
        "scores": scores,
        "prediction_sd_over_training_target_sd": ratios,
        "response_collapsed": all(v is not None and v < 1e-3 for v in ratios.values()),
        "parameters": parameters,
        "truth_parameters": truth,
        "parameter_absolute_error": {
            n: abs(parameters[n] - v) for n, v in truth.items()
        },
        "parameter_recovery_is_not_a_pass_criterion": True,
    }


def verify_result_arrays(output: Path, result: dict) -> None:
    """Do not accept cached metrics after changing an array used by the task."""
    root = output / "results" / result["task"]["name"]
    if result["task"]["kind"] == "guard":
        for record in result.get("records", []):
            for item in (record, record["comparison_reference"]):
                read_array(root / "reference" / item["array"], item["array_sha256"])
        replays = result.get("truth_verification", {}).get("replays", {})
        replay_root = root / "truth_replays"
    else:
        guard_root = output / "results" / f"guard_{result['task']['case']}"
        guard = (
            read_json(guard_root / "result.json")
            if (guard_root / "result.json").exists()
            else None
        )
        if guard is not None:
            verify_result_arrays(output, guard)
        replays = result.get("replays", {})
        replay_root = root / "replays"
    for name, replay in replays.items():
        for record in replay["trajectories"]:
            if record["status"] == "complete":
                read_array(replay_root / name / record["array"], record["array_sha256"])


def execute_recovery(output: Path, index: int) -> dict:
    """Checkpoint one guard or fit and keep failed tasks terminal."""
    frozen = verify_recovery(output)
    if not 0 <= index < len(frozen["tasks"]):
        raise ValueError("unknown task index")
    task = frozen["tasks"][index]
    identity = content_hash([frozen["freeze_sha256"], task])
    root = output / "results" / task["name"]
    existing = checkpoint(root / "result.json", identity)
    if existing is not None:
        verify_result_arrays(output, existing)
        return existing
    started = monotonic()
    try:
        result = (_guard if task["kind"] == "guard" else _fit)(
            output, root, frozen, task, identity
        )
    except (TimeoutError, ValueError, ArithmeticError, RuntimeError) as error:
        result = {"status": "failed", "error": str(error)}
    result.update(
        identity=identity,
        task=task,
        task_seconds=monotonic() - started,
        test_data_opened=False,
        private_reference_opened=False,
        llm_calls=0,
    )
    result = _finite_payload(result)
    write_json(root / "result.json", result)
    return result


def summarize_recovery(output: Path) -> dict:
    """Report every arm, separating oracle-assisted starts and optimizer stops."""
    frozen = verify_recovery(output)
    rows = []
    for task in frozen["tasks"]:
        identity = content_hash([frozen["freeze_sha256"], task])
        record = checkpoint(output / "results" / task["name"] / "result.json", identity)
        if record is not None:
            verify_result_arrays(output, record)
        rows.append(
            {
                "task": task,
                "status": record["status"] if record else "missing",
                "result": record,
            }
        )
    result = {
        "freeze_sha256": frozen["freeze_sha256"],
        "rows": rows,
        "model_selection_performed": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "llm_calls": 0,
    }
    write_json(output / "summary.json", result)
    threshold = frozen["plan"]["recovery_nmse"]
    lines = [
        "# Active-dynamics fitter recovery",
        "",
        "Fixed noiseless synthetic equation family; "
        "train-only optimization; no test data.",
        "Near-truth starts use oracle knowledge and are reported separately.",
        f"Recovery requires NMSE <= {threshold:g} on both development splits "
        "and agreement across solvers and tighter tolerances.",
        "",
        "| Task | Status | Trajectory recovered | Optimizer success | Calls | "
        "Seconds | Train NMSE | Validation NMSE |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        record = row["result"] or {}
        fit = record.get("fit", {})
        scores = record.get("scores", {})
        lines.append(
            f"| {row['task']['name']} | {row['status']} | "
            f"{record.get('trajectory_recovered', '—')} | "
            f"{fit.get('optimizer_success', '—')} | "
            f"{fit.get('actual_residual_calls', '—')} | "
            f"{fit.get('fit_seconds', '—')} | {scores.get('train', '—')} | "
            f"{scores.get('validation', '—')} |"
        )
    for row in rows:
        r = row["result"] or {}
        lines += [
            "",
            "## " + row["task"]["name"],
            "",
            f"Status: {row['status']}; error: {r.get('error')}",
        ]
        if row["task"]["kind"] == "guard":
            lines += [
                f"Truth/reference/activity pass: {r.get('truth_pass')} / "
                f"{r.get('reference_pass')} / {r.get('activity_pass')}",
                f"Baselines: `{r.get('baselines')}`",
                f"Component SD / target SD: `{r.get('component_sd_over_target_sd')}`",
                f"Reference checks: `{r.get('reference_checks')}`",
            ]
            for name, replay in (
                r.get("truth_verification", {}).get("replays", {}).items()
            ):
                lines.append(
                    f"- Truth {name}: {replay['status']}; "
                    f"NMSE={replay.get('normalized_mse')}; "
                    f"failures={replay.get('failures')}"
                )
            for label, profile in r.get("derivative_profiles", {}).items():
                lines.append(
                    f"Derivative profile {label}; "
                    f"parameter order: `{profile.get('parameter_order')}`:"
                )
                if profile.get("error"):
                    lines.append(f"- unavailable: {profile['error']}")
                for step in profile.get("steps", []):
                    compact = {
                        key: step[key]
                        for key in (
                            "step_factor",
                            "forward_central_relative_difference",
                        )
                        if key in step
                    }
                    central = step.get("central", {})
                    compact.update(
                        central_condition=central.get("condition_number"),
                        central_column_norms=central.get("column_norms"),
                    )
                    lines.append(f"- `{compact}`")
                lines.append(f"Adjacent steps: `{profile.get('adjacent_steps')}`")
            continue
        lines += [
            f"Start: `{r.get('fit', {}).get('initial_parameters')}`",
            f"Stop: {r.get('fit', {}).get('message')}",
            f"Truth: `{r.get('truth_parameters')}`",
            f"Fitted: `{r.get('parameters')}`",
            "Prediction SD / training target SD: "
            f"`{r.get('prediction_sd_over_training_target_sd')}`",
            f"Numerical checks: `{r.get('checks')}`",
        ]
        for name, replay in r.get("replays", {}).items():
            lines.append(
                f"- {name}: {replay['status']}; "
                f"NMSE={replay.get('normalized_mse')}; "
                f"failures={replay.get('failures')}"
            )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return result
