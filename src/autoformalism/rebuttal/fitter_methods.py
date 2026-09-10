"""Frozen CPU comparison of coefficient matching, sensitivities and latent fits."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field, NonNegativeInt, model_validator
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from autoformalism.data import DatasetSplit, DevelopmentDataset, SplitName, Trajectory
from autoformalism.data.models import TierRoles
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import simulate_trajectory
from autoformalism.fitting.matching_probe import bounded_latent_start, matching_start
from autoformalism.fitting.runtime_probe import read_array, save_array
from autoformalism.fitting.sensitivity_probe import (
    SymbolicODE,
    SymbolicOracle,
    symbolic_rollout,
)
from autoformalism.fitting.stagnation import RolloutOracle, instrumented_fit
from autoformalism.rebuttal.fitter_diagnostic import (
    _finite_payload,
    read_json,
    runtime_identity,
    sha256,
    write_json,
)
from autoformalism.rebuttal.fitter_offset import baseline_report, verify_replays
from autoformalism.rebuttal.fitter_recovery import (
    CONTEXT,
    RecoveryPlan,
    exact_memories,
    recovery_candidate,
    reference_rollout,
)
from autoformalism.rebuttal.fitter_stagnation import checkpoint
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.staged_topology import content_hash

COMMON = ("production_fd", "compiled_fd", "forward_sensitivity")
MATCHING = ("derivative_init", "integral_init", "weak_init")
LATENT = ("shooting_init", "collocation_init")
PAIRED = ("forward_sensitivity", "collocation_init", "collocation_sensitivity")


class MethodsPlan(StrictSchema):
    """Prespecify data, noise, starts, eligibility, and total fitting budgets."""

    protocol: Literal["fitter-methods-1", "fitter-methods-2"] = "fitter-methods-1"
    reference: RecoveryPlan
    noise_fractions: tuple[FiniteFloat, ...] = (0.0, 0.03)
    seed: int = Field(default=20260909, ge=0)
    replicate_seeds: tuple[NonNegativeInt, ...] = Field(
        default=(20260909,), min_length=1, max_length=3
    )
    fit_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    initializer_seconds: FiniteFloat = Field(default=120, gt=0, le=120)
    replay_seconds: FiniteFloat = Field(default=30, gt=0, le=30)
    guard_seconds: FiniteFloat = Field(default=300, gt=0, le=600)
    grace_seconds: FiniteFloat = Field(default=60, gt=0, le=60)
    accuracy_rms: FiniteFloat = Field(default=1e-6, gt=0)
    accuracy_maximum: FiniteFloat = Field(default=1e-5, gt=0)

    @model_validator(mode="after")
    def valid_matrix(self):
        if (
            not 1 <= len(self.noise_fractions) <= 2
            or len(set(self.noise_fractions)) != len(self.noise_fractions)
            or any(n < 0 or n > 0.1 for n in self.noise_fractions)
        ):
            raise ValueError("provide one or two distinct noise fractions in [0,0.1]")
        if self.initializer_seconds >= self.fit_seconds:
            raise ValueError("initialization must fit within the total fit budget")
        if len(set(self.replicate_seeds)) != len(self.replicate_seeds):
            raise ValueError("replicate seeds must be distinct")
        if self.protocol == "fitter-methods-1" and (
            len(self.replicate_seeds) != 1 or self.guard_seconds > 300
        ):
            raise ValueError("replicates and larger guard budget require methods-v2")
        if self.reference.sample_step != 0.2:
            raise ValueError("methods-v1 uses a fixed 0.2 observation grid")
        if any(c.name not in {"moderate", "separated"} for c in self.reference.cases):
            raise ValueError("unknown hidden fixture")
        return self

    def settings(self):
        return self.reference.settings().model_copy(
            update={"maximum_wall_time_seconds": self.fit_seconds}
        )

    def worker_seconds(self, kind):
        if kind == "initializer":
            return self.grace_seconds + self.initializer_seconds
        return self.grace_seconds + (
            self.guard_seconds
            if kind == "guard"
            else self.fit_seconds + 6 * self.replay_seconds
        )


def affine_candidate() -> CandidateModel:
    """A fully observed nonlinear state equation, affine in all four coefficients."""
    return CandidateModel.model_validate(
        {
            "candidate_id": "affine_observed_control",
            "parent_candidate_id": None,
            "states": [{"name": "x", "kind": "latent"}],
            "state_equations": [{"state": "x", "rhs": "-a*x + b*tanh(x) + c*u01 + d"}],
            "observation_mappings": [{"channel": "v01", "expression": "x"}],
            "parameters": [
                {
                    "name": n,
                    "role": "offset" if n == "d" else "nonnegative_coefficient",
                    "scope": "global",
                }
                for n in ("a", "b", "c", "d")
            ],
            "initial_conditions": [
                {"state": "x", "scope": "global", "fixed_value": 0.0}
            ],
        }
    )


def launcher_identity() -> str:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "scripts/run_fitter_methods.py",
        "scripts/hpc/fitter_methods_delta.slurm",
        "scripts/hpc/submit_fitter_methods_delta.sh",
        "scripts/hpc/submit_fitter_methods_v2_delta.sh",
    )
    return content_hash({p: sha256(root / p) for p in paths})


def identity_runtime() -> dict:
    return {**runtime_identity(), "casadi": importlib.metadata.version("casadi")}


def prepare_methods(plan: MethodsPlan, output: Path) -> dict:
    """Freeze the plan, expressions and independent starts without fitting."""
    if importlib.metadata.version("casadi") != "3.7.2":
        raise ValueError("methods-v1 requires casadi==3.7.2")
    cases = [
        {
            "name": "affine_observed",
            "truth": {"a": 1.2, "b": 0.5, "c": 0.8, "d": -0.4},
            "observed_state": True,
        }
    ]
    if plan.protocol == "fitter-methods-2":
        cases = []
    cases += [
        {"name": c.name, "truth": c.truth.model_dump(), "observed_state": False}
        for c in plan.reference.cases
    ]
    starts, assets, paired_starts = {}, {}, {}
    for case in cases:
        name = case["name"]
        candidate = (
            affine_candidate() if case["observed_state"] else recovery_candidate()
        )
        path = output / f"candidate_{name}.json"
        write_json(path, candidate.model_dump(mode="json"), immutable=True)
        assets[path.name] = sha256(path)
        seeds = (
            plan.replicate_seeds
            if plan.protocol == "fitter-methods-2"
            else (plan.seed,)
        )
        paired_starts[name] = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            start = {
                n: float(np.exp(rng.uniform(np.log(0.25), np.log(4))))
                for n in sorted(case["truth"])
            }
            start["d" if case["observed_state"] else "c"] = float(rng.uniform(-2, 2))
            paired_starts[name].append(start)
        starts[name] = paired_starts[name][0]
    tasks = [
        {"name": f"guard_{c['name']}", "case": c["name"], "kind": "guard"}
        for c in cases
    ]
    for case in cases:
        for index, noise in enumerate(plan.noise_fractions):
            for method in COMMON + (MATCHING if case["observed_state"] else LATENT):
                tasks.append(
                    {
                        "name": f"fit_{case['name']}_noise{index}_{method}",
                        "kind": "fit",
                        "case": case["name"],
                        "noise_index": index,
                        "noise_fraction": noise,
                        "method": method,
                    }
                )
    if plan.protocol == "fitter-methods-2":
        tasks = [t for t in tasks if t["kind"] == "guard"]
        fits = []
        for case in cases:
            for noise_index, noise in enumerate(plan.noise_fractions):
                for replicate, seed in enumerate(plan.replicate_seeds):
                    pair = f"{case['name']}_noise{noise_index}_rep{replicate}"
                    shared = {
                        "case": case["name"],
                        "noise_index": noise_index,
                        "noise_fraction": noise,
                        "replicate": replicate,
                        "start_seed": seed,
                        "noise_seed": plan.seed + 104729 * replicate,
                        "pair": pair,
                        "initializer_task": f"init_{pair}",
                    }
                    tasks.append(
                        {**shared, "name": f"init_{pair}", "kind": "initializer"}
                    )
                    # Rotate scheduler order; every pair still contains every arm.
                    shift = (replicate + noise_index) % len(PAIRED)
                    for method in PAIRED[shift:] + PAIRED[:shift]:
                        fits.append(
                            {
                                **shared,
                                "name": f"fit_{pair}_{method}",
                                "kind": "fit",
                                "method": method,
                            }
                        )
        tasks.extend(fits)
    freeze = {
        "plan": plan.model_dump(mode="json"),
        "cases": cases,
        "starts": starts,
        "paired_starts": paired_starts,
        "tasks": tasks,
        "assets": assets,
        "runtime": identity_runtime(),
        "launcher": launcher_identity(),
        "test_data_opened": False,
        "benchmark_data_opened": False,
        "llm_calls": 0,
    }
    freeze["freeze_sha256"] = content_hash(freeze)
    write_json(output / "freeze.json", freeze, immutable=True)
    return freeze


def verify_methods(output: Path) -> dict:
    frozen = read_json(output / "freeze.json")
    if (
        frozen["freeze_sha256"]
        != content_hash({k: v for k, v in frozen.items() if k != "freeze_sha256"})
        or frozen["runtime"] != identity_runtime()
        or frozen["launcher"] != launcher_identity()
    ):
        raise ValueError("methods freeze/runtime/launcher differs")
    for name, digest in frozen["assets"].items():
        if sha256(output / name) != digest:
            raise ValueError("candidate asset differs")
    MethodsPlan.model_validate(frozen["plan"])
    return frozen


def _affine_reference(forcing, truth, time, method, deadline):
    values = np.empty(len(time))
    values[0] = 0.0
    state = [0.0]
    for i in range(len(forcing.times) - 1):
        a, b = forcing.times[i : i + 2]
        low, high = forcing.values[i : i + 2]

        def rhs(t, x, a=a, b=b, low=low, high=high):
            if monotonic() >= deadline:
                raise TimeoutError("affine reference deadline reached")
            u = low + (high - low) * (t - a) / (b - a)
            return (
                -truth["a"] * x + truth["b"] * np.tanh(x) + truth["c"] * u + truth["d"]
            )

        indices = np.flatnonzero((time > a + 1e-10) & (time <= b + 1e-10))
        solved = solve_ivp(
            rhs,
            (a, b),
            state,
            t_eval=np.minimum(time[indices], b),
            method=method,
            rtol=1e-11,
            atol=1e-13,
        )
        if not solved.success or not np.isfinite(solved.y).all():
            raise ValueError("affine reference failed")
        values[indices] = solved.y[0]
        state = solved.y[:, -1]
    return np.column_stack(
        (time, np.interp(time, forcing.times, forcing.values), values, values)
    )


def _data(
    output, case, records, root, noise_index, noise, scale, seed
) -> DevelopmentDataset:
    splits = {}
    for label, split_name in [
        ("train", SplitName.TRAIN),
        ("validation", SplitName.VALIDATION),
    ]:
        trajectories = []
        for record in records:
            if record["split"] != label:
                continue
            clean = read_array(root / record["array"], record["array_sha256"])
            salt = int(content_hash([seed, case, noise_index, record["name"]])[:16], 16)
            observed = clean[:, 2] + noise * scale * np.random.default_rng(
                salt
            ).standard_normal(len(clean))
            trajectories.append(
                Trajectory(
                    record["name"],
                    clean[:, 0],
                    {"v01": observed},
                    {},
                    {"u01": clean[:, 1]},
                    {},
                    {},
                )
            )
        splits[label] = DatasetSplit(
            split_name,
            tuple(trajectories),
            content_hash([case, label, noise_index, noise, seed]),
        )
    return DevelopmentDataset(
        "synthetic_fitter_methods",
        "diagnostic",
        TierRoles(targets=("v01",)),
        splits["train"],
        splits["validation"],
    )


def _load_case(output, frozen, name):
    case = next(c for c in frozen["cases"] if c["name"] == name)
    model = compile_candidate(
        CandidateModel.model_validate(read_json(output / f"candidate_{name}.json")),
        CONTEXT,
    )
    return case, model


def _guard(output, root, frozen, task, identity):
    plan = MethodsPlan.model_validate(frozen["plan"])
    case, model = _load_case(output, frozen, task["case"])
    system = SymbolicODE(model)
    deadline = monotonic() + plan.guard_seconds
    time = np.arange(round(plan.reference.inputs[0].times[-1] / 0.2) + 1) * 0.2
    reference = root / "reference"
    records = []
    errors = []
    for forcing in plan.reference.inputs:
        arrays = []
        meta = []
        for method in ["Radau", "DOP853"]:
            path = reference / f"{forcing.name}_{method}.json"
            key = content_hash([identity, forcing.name, method])
            record = checkpoint(path, key)
            if record is None:
                generator = (
                    _affine_reference if case["observed_state"] else reference_rollout
                )
                values = generator(forcing, case["truth"], time, method, deadline)
                filename = f"{forcing.name}_{method}.npz"
                record = {
                    "identity": key,
                    "array": filename,
                    "array_sha256": save_array(reference / filename, values),
                }
                write_json(path, record)
            arrays.append(
                read_array(reference / record["array"], record["array_sha256"])
            )
            meta.append(record)
        errors.append(float(np.max(abs(arrays[0][:, 2] - arrays[1][:, 2]))))
        if not case["observed_state"]:
            errors.append(
                float(
                    np.max(
                        abs(
                            arrays[0][:, 3:5]
                            - exact_memories(forcing, case["truth"], time)
                        )
                    )
                )
            )
        records.append(
            {
                **meta[0],
                "comparison": meta[1],
                "name": forcing.name,
                "split": forcing.split,
            }
        )
    dataset = _data(output, case["name"], records, reference, 0, 0, 1, plan.seed)
    baseline = baseline_report(dataset)
    scale = baseline["training_scale"]
    truth = np.array([case["truth"][n] for n in system.names])
    replays = verify_replays(
        root / "truth_replays",
        identity,
        model,
        dataset,
        case["truth"],
        plan.settings(),
        plan,
        scale,
        outer_deadline=deadline,
    )
    profiles = {}
    data = dataset.train.trajectories[0]
    anchors = [("truth", case["truth"]), ("broad", frozen["starts"][case["name"]])]
    if plan.protocol == "fitter-methods-2":
        anchors.extend(
            (f"broad_rep{i}", p)
            for i, p in enumerate(frozen["paired_starts"][case["name"]])
            if i > 0
        )
    for name, params in anchors:
        path = root / f"profile_{name}.json"
        key = content_hash([identity, name])
        report = checkpoint(path, key)
        if report is None:
            theta = np.array([params[n] for n in system.names])
            before = monotonic()
            value, jac, _, _ = symbolic_rollout(
                system, data, theta, plan.settings(), deadline, sensitivities=True
            )
            sensitivity_seconds = monotonic() - before
            production = simulate_trajectory(
                model,
                data,
                params,
                {},
                plan.settings(),
                deadline=deadline,
                reset_observed_states=False,
            )
            if not production.success:
                raise ValueError(production.message)
            differences = []
            derivative_seconds = []
            for step in [1e-4, 1e-5]:
                before = monotonic()
                columns = []
                for j in range(len(theta)):
                    h = step * max(1, abs(theta[j]))
                    plus, minus = theta.copy(), theta.copy()
                    plus[j] += h
                    minus[j] -= h
                    hi = symbolic_rollout(
                        system, data, plus, plan.settings(), deadline
                    )[0][:, 0]
                    lo = symbolic_rollout(
                        system, data, minus, plan.settings(), deadline
                    )[0][:, 0]
                    columns.append((hi - lo) / (2 * h))
                central = np.column_stack(columns)
                differences.append(
                    float(
                        np.linalg.norm(central - jac[:, 0, :])
                        / max(np.linalg.norm(central), 1e-15)
                    )
                )
                derivative_seconds.append(monotonic() - before)
            value_error = float(
                np.max(abs(value[:, 0] - production.predictions["v01"])) / scale
            )
            report = {
                "identity": key,
                "ad_central_relative_errors": differences,
                "production_maximum_normalized_difference": value_error,
                "sensitivity_seconds": sensitivity_seconds,
                "central_difference_seconds": derivative_seconds,
                "pass": bool(value_error <= 1e-5 and max(differences) <= 1e-3),
            }
            write_json(path, report)
        profiles[name] = report
    truth_ok = replays["status"] == "complete" and all(
        r["normalized_mse"] is not None and r["normalized_mse"] < 1e-10
        for r in replays["replays"].values()
    )
    good = (
        truth_ok
        and max(errors) / scale < 1e-7
        and all(r["pass"] for r in profiles.values())
    )
    return {
        "status": "complete" if good else "guard_failed",
        "records": records,
        "baseline": baseline,
        "reference_error_over_scale": max(errors) / scale,
        "truth_verification": replays,
        "profiles": profiles,
        "coefficient_audit": system.audit,
        "truth_parameters": dict(zip(system.names, truth, strict=True)),
    }


def _fit_inputs(output, frozen, task):
    """Load a verified synthetic case and expose only observations to fitting."""
    plan = MethodsPlan.model_validate(frozen["plan"])
    case, model = _load_case(output, frozen, task["case"])
    guard_task = next(
        t for t in frozen["tasks"] if t["name"] == f"guard_{case['name']}"
    )
    guard_root = output / "results" / guard_task["name"]
    guard = checkpoint(
        guard_root / "result.json", content_hash([frozen["freeze_sha256"], guard_task])
    )
    if guard is None or guard["status"] != "complete":
        return None
    verify_result_arrays(output, guard)
    clean_scale = guard["baseline"]["training_scale"]
    dataset = _data(
        output,
        case["name"],
        guard["records"],
        guard_root / "reference",
        task["noise_index"],
        task["noise_fraction"],
        clean_scale,
        task.get("noise_seed", plan.seed),
    )
    scale = baseline_report(dataset)["training_scale"]
    return plan, case, model, guard, guard_root, dataset, scale, clean_scale


def _start(frozen, task):
    """Select a prespecified start, independently of fit or reference quality."""
    if "replicate" in task:
        return frozen["paired_starts"][task["case"]][task["replicate"]]
    return frozen["starts"][task["case"]]


def _initializer(output, root, frozen, task, identity):
    """Compute one training-only collocation start shared by its two paired arms."""
    inputs = _fit_inputs(output, frozen, task)
    if inputs is None:
        return {"status": "guard_failed", "error": "case numerical guard did not pass"}
    plan, _, model, _, _, dataset, scale, _ = inputs
    started = monotonic()
    settings = plan.settings()
    system = SymbolicODE(model)
    layout = RolloutOracle(
        model,
        dataset.train,
        {"v01": scale},
        settings,
        root / "layout",
        started + plan.initializer_seconds,
    )
    initializer = bounded_latent_start(
        system,
        training=dataset.train,
        lower=layout.lower,
        upper=layout.upper,
        start=layout.vector(_start(frozen, task)),
        scale=scale,
        settings=settings,
        method="collocation_init",
        seconds=max(0, plan.initializer_seconds - (monotonic() - started)),
        directory=root / "initializer",
    )
    initializer.update(identity=identity, seconds=monotonic() - started)
    return {
        "status": "complete" if initializer["success"] else "initializer_failed",
        "initializer": initializer,
        "ordinary_start": _start(frozen, task),
    }


def _paired_initializer(output, frozen, task, plan):
    """Require a finished paired stage; failures fall back to the same start."""
    shared_task = next(
        t for t in frozen["tasks"] if t["name"] == task["initializer_task"]
    )
    key = content_hash([frozen["freeze_sha256"], shared_task])
    shared = checkpoint(output / "results" / shared_task["name"] / "result.json", key)
    if shared is None or shared["status"] == "guard_failed":
        raise ValueError(
            "shared initializer missing or blocked; run initializer stage first"
        )
    initializer = shared.get("initializer")
    if initializer is None:
        # Native crash/outer timeout consumes the entire initializer allocation.
        initializer = {
            "success": False,
            "parameters": None,
            "seconds": plan.initializer_seconds,
            "method": "collocation_init",
            "message": shared.get("error", shared["status"]),
        }
    return {
        **initializer,
        "shared_task": shared_task["name"],
        "shared_result_sha256": content_hash(shared),
    }


def _fit(output, root, frozen, task, identity):
    inputs = _fit_inputs(output, frozen, task)
    if inputs is None:
        return {"status": "guard_failed", "error": "case numerical guard did not pass"}
    plan, case, model, guard, guard_root, dataset, scale, clean_scale = inputs
    system = SymbolicODE(model)
    settings = plan.settings()
    method = task["method"]
    start = _start(frozen, task)
    shared_initializer = None
    if plan.protocol == "fitter-methods-2" and method != "forward_sensitivity":
        shared_initializer = _paired_initializer(output, frozen, task, plan)
    fitted = checkpoint(root / "fit.json", identity)
    if (
        fitted is not None
        and shared_initializer is not None
        and (
            fitted["initializer"]["shared_result_sha256"]
            != shared_initializer["shared_result_sha256"]
        )
    ):
        raise ValueError("shared initializer changed after fitting")
    if fitted is None:
        attempt = 0
        while (root / f"attempt-{attempt}").exists():
            attempt += 1
        attempt_root = root / f"attempt-{attempt}"
        attempt_root.mkdir(parents=True)
        started = monotonic()
        deadline = started + plan.fit_seconds
        layout = RolloutOracle(
            model,
            dataset.train,
            {"v01": scale},
            settings,
            attempt_root / "layout",
            deadline,
        )
        theta = layout.vector(start)
        init_path = root / "initializer.json"
        initializer = checkpoint(init_path, identity)
        if (
            initializer is not None
            and shared_initializer is not None
            and (
                initializer.get("shared_result_sha256")
                != shared_initializer["shared_result_sha256"]
            )
        ):
            raise ValueError("cached initializer differs from paired initializer")
        if initializer is None:
            if shared_initializer is not None:
                initializer = shared_initializer
            elif method in MATCHING:
                initializer = matching_start(
                    system,
                    dataset.train,
                    layout.lower,
                    layout.upper,
                    method,
                    weak_quadrature="trapezoid-v1"
                    if plan.protocol == "fitter-methods-1"
                    else "gauss5-linear-v2",
                )
            elif method in LATENT:
                initializer = bounded_latent_start(
                    system,
                    training=dataset.train,
                    lower=layout.lower,
                    upper=layout.upper,
                    start=theta,
                    scale=scale,
                    settings=settings,
                    method=method,
                    seconds=min(
                        plan.initializer_seconds, max(0, deadline - monotonic())
                    ),
                    directory=attempt_root / "initializer",
                )
            else:
                initializer = {
                    "success": False,
                    "parameters": None,
                    "seconds": 0,
                    "method": "none",
                }
            initializer["identity"] = identity
            write_json(init_path, _finite_payload(initializer))
        # Cached initialization is charged its original duration on a restarted fit.
        deadline = min(
            deadline, monotonic() + max(0, plan.fit_seconds - initializer["seconds"])
        )
        selected = initializer["parameters"] if initializer["success"] else start
        oracle = (
            SymbolicOracle(
                system,
                dataset.train,
                scale,
                settings,
                attempt_root / "calls",
                deadline,
                sensitivities=method
                in {"forward_sensitivity", "collocation_sensitivity"},
            )
            if method
            in {"compiled_fd", "forward_sensitivity", "collocation_sensitivity"}
            else RolloutOracle(
                model,
                dataset.train,
                {"v01": scale},
                settings,
                attempt_root / "calls",
                deadline,
            )
        )

        def optimizer(fun, x, **kwargs):
            if method in {"forward_sensitivity", "collocation_sensitivity"}:
                kwargs["jac"] = oracle.jacobian
            return least_squares(fun, x, **kwargs)

        report = instrumented_fit(
            oracle,
            selected,
            diff_step=None,
            max_nfev=settings.maximum_function_evaluations,
            settings=settings,
            optimizer=optimizer,
        )
        fitted = {
            "identity": identity,
            "fit": report,
            "initializer": initializer,
            "fit_budget_seconds": plan.fit_seconds,
            "total_fit_seconds": initializer["seconds"] + report["fit_seconds"],
            "symbolic_solver_counts": getattr(oracle, "solver_counts", None),
            "ordinary_start": start,
            "initializer_fallback": (
                method in MATCHING + LATENT or method == "collocation_sensitivity"
            )
            and not initializer["success"],
            "refinement_start": selected,
            "training_fingerprint": dataset.train.fingerprint,
        }
        write_json(root / "fit.json", _finite_payload(fitted))
    parameters = fitted["fit"]["parameters"]
    if parameters is None:
        return {**fitted, "status": "fit_failed"}
    verification = verify_replays(
        root / "replays", identity, model, dataset, parameters, settings, plan, scale
    )
    clean_scores = {}
    for label in ["train", "validation"]:
        replay = verification["replays"][f"Radau_tight/{label}"]
        if replay["status"] != "complete":
            clean_scores[label] = None
            continue
        residuals = []
        rows = [r for r in guard["records"] if r["split"] == label]
        for record, truth_record in zip(replay["trajectories"], rows, strict=True):
            predicted = read_array(
                root / "replays" / f"Radau_tight/{label}" / record["array"],
                record["array_sha256"],
            )[:, 0]
            clean = read_array(
                guard_root / "reference" / truth_record["array"],
                truth_record["array_sha256"],
            )[:, 2]
            residuals.extend((predicted - clean) / clean_scale)
        clean_scores[label] = float(np.mean(np.asarray(residuals) ** 2))
    from autoformalism.rebuttal.fitter_methods_report import parameter_equivalence

    return {
        **fitted,
        **verification,
        "clean_signal_nmse": clean_scores,
        "parameters": parameters,
        "parameter_absolute_error": {
            n: abs(parameters[n] - v) for n, v in case["truth"].items()
        },
        "parameter_equivalence": parameter_equivalence(case, parameters),
        "training_only_optimization": True,
        "hidden_labels_used": False,
        "noise_fraction": task["noise_fraction"],
        "data_training_scale": scale,
        "clean_scores_are_diagnostic_only": True,
    }


def verify_result_arrays(output, result):
    root = output / "results" / result["task"]["name"]
    initializer = result.get("initializer", {})
    if "shared_task" in initializer:
        shared = read_json(
            output / "results" / initializer["shared_task"] / "result.json"
        )
        if content_hash(shared) != initializer["shared_result_sha256"]:
            raise ValueError("shared initializer changed after fitting")
    if result["task"]["kind"] == "guard":
        for r in result.get("records", []):
            for item in [r, r["comparison"]]:
                read_array(root / "reference" / item["array"], item["array_sha256"])
        base = root / "truth_replays"
        replays = result.get("truth_verification", {}).get("replays", {})
    else:
        path = output / "results" / f"guard_{result['task']['case']}" / "result.json"
        if path.exists():
            verify_result_arrays(output, read_json(path))
        base = root / "replays"
        replays = result.get("replays", {})
    for name, replay in replays.items():
        for item in replay["trajectories"]:
            if item["status"] == "complete":
                read_array(base / name / item["array"], item["array_sha256"])


def execute_methods(output: Path, index: int) -> dict:
    frozen = verify_methods(output)
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
        handler = {"guard": _guard, "initializer": _initializer, "fit": _fit}[
            task["kind"]
        ]
        result = handler(output, root, frozen, task, identity)
    except (TimeoutError, ValueError, ArithmeticError, RuntimeError) as error:
        result = {"status": "failed", "error": str(error)[-2000:]}
    result = _finite_payload(
        {
            **result,
            "task": task,
            "identity": identity,
            "task_seconds": monotonic() - started,
            "test_data_opened": False,
            "benchmark_data_opened": False,
            "llm_calls": 0,
        }
    )
    write_json(root / "result.json", result)
    return result


def summarize_methods(output: Path) -> dict:
    frozen = verify_methods(output)
    rows = []
    for task in frozen["tasks"]:
        record = checkpoint(
            output / "results" / task["name"] / "result.json",
            content_hash([frozen["freeze_sha256"], task]),
        )
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
    }
    write_json(output / "summary.json", result)
    lines = [
        "# Fitter method comparison",
        "",
        "Same noisy observations, one frozen broad start, equal total fitting budgets.",
        "The scalar affine control exposes its state; latent cases expose v01 only.",
        "Clean-reference scores are diagnostic and never enter optimization. "
        "Completion means numerical verification, not recovery.",
        "",
        "| Case | Noise SD fraction | Method | Status | Init success | "
        "Optimizer success | Init + refinement s | Calls | Clean train NMSE | "
        "Clean validation NMSE |",
        "| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    if frozen["plan"]["protocol"] == "fitter-methods-2":
        lines[2:4] = [
            "Paired noisy observations and broad starts, equal total fitting budgets.",
            "Both latent cases expose v01 only; initializers share no hidden labels.",
        ]
    for row in rows:
        t = row["task"]
        r = row["result"] or {}
        f = r.get("fit", {})
        c = r.get("clean_signal_nmse", {})
        if t["kind"] != "fit":
            continue
        lines.append(
            f"| {t['case']} | {t['noise_fraction']} | {t['method']} | "
            f"{row['status']} | "
            f"{r.get('initializer', {}).get('success', '—')} | "
            f"{f.get('optimizer_success', '—')} | {r.get('total_fit_seconds', '—')} | "
            f"{f.get('actual_residual_calls', '—')} | {c.get('train', '—')} | "
            f"{c.get('validation', '—')} |"
        )
    for row in rows:
        r = row["result"] or {}
        lines += [
            "",
            f"## {row['task']['name']}",
            "",
            f"Status: {row['status']}; error: {r.get('error')}",
        ]
        if row["task"]["kind"] == "guard":
            lines += [
                f"Coefficient audit: `{r.get('coefficient_audit')}`",
                f"Derivative checks: `{r.get('profiles')}`",
                "Reference normalized maximum error: "
                f"{r.get('reference_error_over_scale')}",
            ]
            continue
        lines += [
            f"Initializer: `{r.get('initializer')}`",
            f"Ordinary start: `{r.get('ordinary_start')}`",
            f"Fitted: `{r.get('parameters')}`",
            f"Parameter absolute errors: `{r.get('parameter_absolute_error')}`",
            f"Known parameter equivalence: `{r.get('parameter_equivalence')}`",
            f"Stop: {r.get('fit', {}).get('message')}",
            f"Numerical checks: `{r.get('checks')}`",
        ]
        for name, replay in r.get("replays", {}).items():
            lines.append(
                f"- {name}: {replay['status']}; "
                f"observed-data NMSE={replay.get('normalized_mse')}; "
                f"failures={replay.get('failures')}"
            )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    if frozen["plan"]["protocol"] == "fitter-methods-2":
        from autoformalism.rebuttal.fitter_methods_report import write_paired_summary

        (output / "details.md").write_text("\n".join(lines) + "\n")
        result["paired_summary"] = write_paired_summary(output, frozen, rows)
        write_json(output / "summary.json", result)
    return result
