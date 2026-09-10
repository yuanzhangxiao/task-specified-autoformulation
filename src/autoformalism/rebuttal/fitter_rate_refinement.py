"""Paired refinement of immutable v4 starts; no collocation is rerun."""

from __future__ import annotations

import shutil
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field
from scipy.optimize import least_squares

from autoformalism.fitting.rate_refinement import RateCoordinates
from autoformalism.fitting.runtime_probe import read_array
from autoformalism.fitting.sensitivity_probe import (
    SymbolicODE,
    SymbolicOracle,
    symbolic_rollout,
)
from autoformalism.fitting.stagnation import RolloutOracle, instrumented_fit
from autoformalism.rebuttal.fitter_diagnostic import (
    _finite_payload,
    read_json,
    sha256,
    write_json,
)
from autoformalism.rebuttal.fitter_methods import (
    MethodsPlan,
    _fit_inputs,
    identity_runtime,
    verify_result_arrays,
)
from autoformalism.rebuttal.fitter_offset import CASES, verify_replays
from autoformalism.rebuttal.fitter_stagnation import checkpoint
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.staged_topology import content_hash

# Explicit import compatibility, independent of the NEW runtime identity.
SOURCE = {
    "source_sha256": "8a1f418a72785ff86e0cc80cc42f738416c58329bfc473cc7eef138e11733f38",
    "launcher": "5ed25ca878b5356d37d7b34ddace1374ca770d88a45c66ff981b3fd7fbced87c",
    "plan": "ce28363a1e8b7370e0b7c5ed30ae534d03d632cba9a1148efe0e991418e01ac8",
    "tasks": "d4cb9ed0abfa7627c4f3650bc1961632e9f2d93e48cf20c7c16983b4f5ac69d5",
}


class RatePlan(StrictSchema):
    """Freeze the source selection and verification budget before new fitting."""

    protocol: Literal["fitter-rate-refinement-1", "fitter-rate-stopping-1"] = (
        "fitter-rate-refinement-1"
    )
    source_tasks: tuple[str, ...] | None = None  # None requires all 39 v4 fits.
    replay_seconds: FiniteFloat = Field(default=90, gt=0, le=90)
    guard_seconds: FiniteFloat = Field(default=120, gt=0, le=180)
    accuracy_rms: FiniteFloat = Field(default=1e-6, gt=0)
    accuracy_maximum: FiniteFloat = Field(default=1e-5, gt=0)

    def arms(self) -> tuple[tuple[str, str, float | None], ...]:
        """Keep the stopping comparison distinct from the historical coordinate test."""
        if self.protocol == "fitter-rate-stopping-1":
            return (("rates_default", "rates", 1e-8), ("rates_no_ftol", "rates", None))
        return (("physical", "physical", 1e-8), ("rates", "rates", 1e-8))

    def worker_seconds(self, kind: str) -> float:
        return 60 + (
            self.guard_seconds + 2 * (600 + 6 * self.replay_seconds)
            if kind == "pair"
            else 6 * self.replay_seconds
        )


def launcher_identity() -> str:
    root = Path(__file__).resolve().parents[3]
    return content_hash(
        {
            n: sha256(root / n)
            for n in (
                "scripts/run_fitter_rate_refinement.py",
                "scripts/hpc/fitter_rate_refinement_delta.slurm",
                "scripts/hpc/submit_fitter_rate_refinement_delta.sh",
            )
        }
    )


def checked_file(root: Path, relative: str) -> Path:
    """Reject paths escaping an imported experiment, including symlink escapes."""
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError("source artifact path escapes experiment")
    return path


def import_source(
    source: Path, destination: Path, plan: RatePlan
) -> tuple[dict, dict, list]:
    """Copy only referenced JSON/arrays and verify the pinned historical protocol."""
    frozen = read_json(source / "freeze.json")
    if (
        frozen["freeze_sha256"]
        != content_hash({k: v for k, v in frozen.items() if k != "freeze_sha256"})
        or frozen["runtime"]["source_sha256"] != SOURCE["source_sha256"]
        or frozen["launcher"] != SOURCE["launcher"]
        or content_hash(frozen["plan"]) != SOURCE["plan"]
        or content_hash(frozen["tasks"]) != SOURCE["tasks"]
    ):
        raise ValueError("source is not the pinned v4 code, plan and task matrix")
    MethodsPlan.model_validate(frozen["plan"])
    tasks = [t for t in frozen["tasks"] if t["kind"] == "fit"]
    if plan.source_tasks is not None:
        selected = set(plan.source_tasks)
        if (
            not selected
            or len(selected) != len(plan.source_tasks)
            or selected - {t["name"] for t in tasks}
        ):
            raise ValueError("source selection must contain distinct v4 fit names")
        tasks = [t for t in tasks if t["name"] in selected]
    files = {"freeze.json"}
    for name, digest in frozen["assets"].items():
        if sha256(checked_file(source, name)) != digest:
            raise ValueError("source candidate hash differs")
        files.add(name)
    results = []
    for task in [t for t in frozen["tasks"] if t["kind"] == "guard"] + tasks:
        prefix = f"results/{task['name']}"
        result = checkpoint(
            checked_file(source, prefix + "/result.json"),
            content_hash([frozen["freeze_sha256"], task]),
        )
        if (
            result is None
            or result["status"] not in {"complete", "replay_unverified"}
            or result["task"] != task
        ):
            raise ValueError(f"source task missing or unusable: {task['name']}")
        verify_result_arrays(source, result)
        files.add(prefix + "/result.json")
        if task["kind"] == "guard":
            if result["status"] != "complete":
                raise ValueError("source guard did not pass")
            for record in result["records"]:
                for item in (record, record["comparison"]):
                    files.add(prefix + "/reference/" + item["array"])
            base = prefix + "/truth_replays/"
            replays = result["truth_verification"]["replays"]
        else:
            if not result.get("refinement_start") or not result.get("parameters"):
                raise ValueError("source lacks saved start or final parameters")
            results.append(result)
            init = result["initializer"]
            if "checkpoint_array" in init:
                files.add(prefix + "/block_initializer/" + init["checkpoint_array"])
            base, replays = prefix + "/replays/", result["replays"]
        for name, replay in replays.items():
            for record in replay["trajectories"]:
                if record["status"] == "complete":
                    files.add(base + name + "/" + record["array"])
    manifest = {}
    for name in sorted(files):
        src = checked_file(source, name)
        dst = checked_file(destination, name)
        digest = sha256(src)
        if dst.exists() and sha256(dst) != digest:
            raise ValueError("imported source changed during preparation")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(src, dst)
        if sha256(dst) != digest:
            raise ValueError("source copy hash differs")
        manifest[name] = digest
    return frozen, manifest, results


def prepare_rate(plan: RatePlan, source: Path, output: Path) -> dict:
    """Freeze a self-contained import; old experiment files are never mutated."""
    if (
        source.resolve() == output.resolve()
        or output.resolve().is_relative_to(source.resolve())
        or source.resolve().is_relative_to(output.resolve())
    ):
        raise ValueError("source and output directories must be separate")
    if (output / "freeze.json").exists():
        frozen = verify_rate(output)
        if frozen["plan"] != plan.model_dump(mode="json"):
            raise ValueError("rate plan differs on resume")
        return frozen
    parent, manifest, results = import_source(source, output / "source", plan)
    tasks = []
    for r in results:
        name = r["task"]["name"]
        tasks.append(
            {
                "name": "pair_" + name.removeprefix("fit_"),
                "kind": "pair",
                "source_task": name,
            }
        )
        if (
            r["status"] == "replay_unverified"
            and plan.protocol == "fitter-rate-refinement-1"
        ):
            tasks.append(
                {
                    "name": "retry_" + name.removeprefix("fit_"),
                    "kind": "retry",
                    "source_task": name,
                }
            )
    frozen = {
        "plan": plan.model_dump(mode="json"),
        "source_freeze": parent["freeze_sha256"],
        "source_manifest": manifest,
        "tasks": tasks,
        "runtime": identity_runtime(),
        "launcher": launcher_identity(),
        "source_directory": str(source.resolve()),
        "test_data_opened": False,
        "llm_calls": 0,
    }
    frozen["freeze_sha256"] = content_hash(frozen)
    write_json(output / "freeze.json", frozen, immutable=True)
    return frozen


def verify_rate(output: Path) -> dict:
    """Require unchanged new code and byte-identical imported evidence."""
    frozen = read_json(output / "freeze.json")
    if (
        frozen["freeze_sha256"]
        != content_hash({k: v for k, v in frozen.items() if k != "freeze_sha256"})
        or frozen["runtime"] != identity_runtime()
        or frozen["launcher"] != launcher_identity()
    ):
        raise ValueError("rate experiment freeze/runtime/launcher differs")
    RatePlan.model_validate(frozen["plan"])
    for name, digest in frozen["source_manifest"].items():
        if sha256(checked_file(output / "source", name)) != digest:
            raise ValueError("imported source artifact changed")
    return frozen


def start_guard(model, dataset, scale, settings, start, mapped, seconds) -> dict:
    """Compare exact saved points and audit direct rate derivatives on train only."""
    deadline = monotonic() + seconds
    system, old = SymbolicODE(mapped.model), SymbolicODE(model)
    beta = mapped.start(start)
    x = np.array([beta[n] for n in system.names])
    theta = np.array([start[n] for n in old.names])
    differences, derivatives, residuals = [], [], []
    for row in dataset.train.trajectories:
        y, jac, _, _ = symbolic_rollout(
            system, row, x, settings, deadline, sensitivities=True
        )
        physical = symbolic_rollout(old, row, theta, settings, deadline)[0]
        differences.extend(((y - physical) / scale).ravel())
        residuals.extend((y[:, 0] - row.targets["v01"]) / scale)
        if row is dataset.train.trajectories[0]:
            for name in mapped.reverse:
                i = system.names.index(name)
                step = 1e-4 * max(1, abs(x[i]))
                plus = x.copy()
                plus[i] += step
                fd = (
                    symbolic_rollout(system, row, plus, settings, deadline)[0] - y
                ) / step
                error = np.linalg.norm(fd - jac[:, :, i]) / max(
                    1, np.linalg.norm(jac[:, :, i])
                )
                derivatives.append(
                    {
                        "parameter": name,
                        "forward_step": step,
                        "relative_error": float(error),
                        "jacobian_norm": float(np.linalg.norm(jac[:, :, i])),
                    }
                )
    maximum = float(np.max(abs(np.array(differences))))
    return {
        "pass": maximum <= 1e-5
        and all(d["relative_error"] <= 0.01 for d in derivatives),
        "maximum_normalized_prediction_difference": maximum,
        "direct_rate_derivatives": derivatives,
        "training_only": True,
        "nominal_physical_start": start,
        "nominal_rate_start": beta,
        "nominal_training_cost": float(0.5 * np.dot(residuals, residuals)),
    }


def clean_scores(root, verification, guard, guard_root, scale) -> dict:
    """Compute clean-reference diagnostics only after fitting has ended."""
    scores = {}
    for label in ("train", "validation"):
        replay = verification["replays"][f"Radau_tight/{label}"]
        scores[label] = None
        if replay["status"] == "complete":
            residuals = []
            truth = [r for r in guard["records"] if r["split"] == label]
            for r, t in zip(replay["trajectories"], truth, strict=True):
                predicted = read_array(
                    root / f"Radau_tight/{label}" / r["array"], r["array_sha256"]
                )[:, 0]
                clean = read_array(
                    guard_root / "reference" / t["array"], t["array_sha256"]
                )[:, 2]
                residuals.extend((predicted - clean) / scale)
            scores[label] = float(np.mean(np.asarray(residuals) ** 2))
    return scores


def refine(
    root,
    identity,
    coordinates,
    model,
    dataset,
    scale,
    settings,
    start,
    mapped,
    seconds,
    *,
    ftol: float | None = 1e-8,
):
    """Checkpoint complete refinement before any validation or replay work."""
    fitted = checkpoint(root / "fit.json", identity)
    if fitted is not None:
        if fitted.get("stopping_policy", {}).get("ftol", 1e-8) != ftol:
            raise ValueError("refinement stopping policy differs on resume")
        return fitted
    # A killed native optimizer cannot be serialized. Do not silently reset its
    # fitting budget: completed fits replay; interrupted fitting requires a new run.
    if (root / "fit_started.json").exists():
        return {
            "identity": identity,
            "status": "interrupted_fit",
            "fit": {"parameters": None},
        }
    write_json(
        root / "fit_started.json",
        {"identity": identity, "budget_seconds": seconds},
        immutable=True,
    )
    layout = RolloutOracle(
        model, dataset.train, {"v01": scale}, settings, root / "layout", None
    )
    selected_model = mapped.model if coordinates == "rates" else model
    selected = mapped.start(start) if coordinates == "rates" else start
    deadline = monotonic() + seconds
    oracle = SymbolicOracle(
        SymbolicODE(selected_model),
        dataset.train,
        scale,
        settings,
        root / "calls",
        deadline,
        sensitivities=True,
    )
    if coordinates == "rates":
        oracle.lower, oracle.upper = mapped.bounds(layout.lower, layout.upper)

    def optimizer(fun, x, **kwargs):
        kwargs["jac"] = oracle.jacobian
        kwargs.update(ftol=ftol, xtol=1e-8, gtol=1e-8, x_scale=1.0)
        return least_squares(fun, x, **kwargs)

    try:
        report = instrumented_fit(
            oracle,
            selected,
            diff_step=None,
            max_nfev=settings.maximum_function_evaluations,
            settings=settings,
            optimizer=optimizer,
        )
    except (ValueError, RuntimeError, ArithmeticError) as error:
        # A physical-coordinate numerical failure must not suppress its rate arm.
        report = {
            "parameters": None,
            "optimizer_success": False,
            "message": str(error),
            "actual_residual_calls": oracle.calls,
            "fit_seconds": seconds - max(0, deadline - monotonic()),
        }
    call_path = root / "calls/000001.json"
    call = read_json(call_path) if call_path.exists() else {}
    fitted = {
        "identity": identity,
        "coordinates": coordinates,
        "stopping_policy": {"ftol": ftol, "xtol": 1e-8, "gtol": 1e-8, "x_scale": 1.0},
        "parameter_order": oracle.names,
        "raw_gradient_inf_norm": (
            float(np.max(np.abs(report["gradient"])))
            if report.get("gradient") is not None
            else None
        ),
        "fit": report,
        "refinement_budget_seconds": seconds,
        "nominal_start": selected,
        "first_evaluated_parameters": call.get("parameters"),
        "first_evaluation_cost": call.get("cost"),
        "training_only_optimization": True,
        "hidden_labels_used": False,
        "solver_counts": oracle.solver_counts,
    }
    if coordinates == "rates" and report["parameters"]:
        horizon = max(float(t.time[-1] - t.time[0]) for t in dataset.train.trajectories)
        fitted["physical_view"] = mapped.physical_view(report["parameters"], horizon)
    write_json(root / "fit.json", _finite_payload(fitted))
    return fitted


def seed_completed_replays(source_root, new_root, original, identity, settings) -> list:
    """Reuse successful trajectories; retry failures only under a fresh identity."""
    provenance = []
    for case, replay in original["replays"].items():
        method, rtol, atol = dict(CASES)[case.split("/")[0]]
        config = settings.model_copy(
            update={
                "integration_method": method,
                "relative_tolerance": rtol,
                "absolute_tolerance": atol,
            }
        )
        if replay["settings"] != config.model_dump(mode="json"):
            raise ValueError("retry settings differ from historical replay")
        case_id = content_hash([identity, case])
        for i, record in enumerate(replay["trajectories"]):
            if record["status"] != "complete":
                continue
            destination = new_root / case / record["array"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = source_root / case / record["array"]
            read_array(source, record["array_sha256"])
            if not destination.exists():
                shutil.copyfile(source, destination)
            read_array(destination, record["array_sha256"])
            entry = {
                **record,
                "identity": content_hash([case_id, record["trajectory_id"]]),
                "reused_from_source": True,
                "original_record_sha256": content_hash(record),
            }
            write_json(new_root / case / f"trajectory-{i}.json", entry, immutable=True)
            provenance.append(
                {
                    "case": case,
                    "trajectory_id": record["trajectory_id"],
                    "array_sha256": record["array_sha256"],
                }
            )
    return provenance


def execute_rate(output: Path, index: int) -> dict:
    """Run one saved-start pair or one frozen-parameter replay retry."""
    frozen = verify_rate(output)
    if not 0 <= index < len(frozen["tasks"]):
        raise ValueError("unknown rate task")
    task = frozen["tasks"][index]
    identity = content_hash([frozen["freeze_sha256"], task])
    root = output / "results" / task["name"]
    existing = checkpoint(root / "result.json", identity)
    if existing is not None:
        verify_rate_result(root, existing)
        return existing
    started = monotonic()
    plan = RatePlan.model_validate(frozen["plan"])
    parent = read_json(output / "source/freeze.json")
    original = read_json(
        output / "source/results" / task["source_task"] / "result.json"
    )
    inputs = _fit_inputs(output / "source", parent, original["task"])
    if inputs is None:
        raise ValueError("imported guard missing")
    old_plan, _, model, guard, guard_root, dataset, scale, clean_scale = inputs
    settings = old_plan.settings()
    try:
        if task["kind"] == "retry":
            reused = seed_completed_replays(
                output / "source/results" / task["source_task"] / "replays",
                root / "replays",
                original,
                identity,
                settings,
            )
            verification = verify_replays(
                root / "replays",
                identity,
                model,
                dataset,
                original["parameters"],
                settings,
                plan,
                scale,
            )
            result = {
                **verification,
                "parameters": original["parameters"],
                "original_status": original["status"],
                "reused_trajectories": reused,
                "fit_rerun": False,
                "replay_seconds_per_split": plan.replay_seconds,
                "clean_signal_nmse": clean_scores(
                    root / "replays", verification, guard, guard_root, clean_scale
                ),
            }
        else:
            mapped = RateCoordinates(model)
            check = checkpoint(root / "guard.json", identity)
            if check is None:
                check = {
                    **start_guard(
                        model,
                        dataset,
                        scale,
                        settings,
                        original["refinement_start"],
                        mapped,
                        plan.guard_seconds,
                    ),
                    "identity": identity,
                }
                write_json(root / "guard.json", check)
            arms = {}
            if check["pass"]:
                remaining = old_plan.fit_seconds - original["initializer"]["seconds"]
                if remaining <= 0:
                    raise ValueError(
                        "saved initialization consumed total fitting budget"
                    )
                for arm, coordinate, ftol in plan.arms():
                    arm_root = root / arm
                    arm_id = content_hash([identity, arm])
                    fitted = refine(
                        arm_root,
                        arm_id,
                        coordinate,
                        model,
                        dataset,
                        scale,
                        settings,
                        original["refinement_start"],
                        mapped,
                        remaining,
                        ftol=ftol,
                    )
                    params = fitted["fit"]["parameters"]
                    if params is None:
                        arms[arm] = {
                            **fitted,
                            "status": fitted.get("status", "fit_failed"),
                        }
                        continue
                    replay_model = mapped.model if coordinate == "rates" else model
                    verification = verify_replays(
                        arm_root / "replays",
                        arm_id,
                        replay_model,
                        dataset,
                        params,
                        settings,
                        plan,
                        scale,
                    )
                    arms[arm] = {
                        **fitted,
                        **verification,
                        "clean_signal_nmse": clean_scores(
                            arm_root / "replays",
                            verification,
                            guard,
                            guard_root,
                            clean_scale,
                        ),
                    }
            result = {
                "status": "complete"
                if check["pass"]
                and all(a["status"] == "complete" for a in arms.values())
                else "pair_unverified",
                "guard": check,
                "arms": arms,
                "original_initializer_seconds": original["initializer"]["seconds"],
                "collocation_rerun": False,
                "source_start_sha256": content_hash(original["refinement_start"]),
            }
    except (TimeoutError, ValueError, RuntimeError, ArithmeticError) as error:
        result = {"status": "failed", "error": str(error)[-2000:]}
    result.update(
        identity=identity,
        task=task,
        task_seconds=monotonic() - started,
        source_result_sha256=content_hash(original),
        test_data_opened=False,
        benchmark_data_opened=False,
        llm_calls=0,
    )
    write_json(root / "result.json", _finite_payload(result))
    return result


def verify_rate_result(root: Path, result: dict) -> None:
    """Verify saved replay arrays even when a task checkpoint already exists."""
    records = (
        [(root, result)]
        if result["task"]["kind"] == "retry"
        else [(root / n, a) for n, a in result.get("arms", {}).items()]
    )
    for directory, record in records:
        for case, replay in record.get("replays", {}).items():
            for row in replay["trajectories"]:
                if row["status"] == "complete":
                    read_array(
                        directory / "replays" / case / row["array"], row["array_sha256"]
                    )


def summarize_rate(output: Path) -> dict:
    """Separate numerical verification, optimizer stops and output recovery."""
    frozen = verify_rate(output)
    plan = RatePlan.model_validate(frozen["plan"])
    rows, results = [], []
    for index, task in enumerate(frozen["tasks"]):
        print(
            f"Checking {index + 1}/{len(frozen['tasks'])}: {task['name']}", flush=True
        )
        root = output / "results" / task["name"]
        result = checkpoint(
            root / "result.json", content_hash([frozen["freeze_sha256"], task])
        )
        if result is None:
            rows.extend(
                {
                    "source": task["source_task"],
                    "arm": arm,
                    "status": "missing",
                }
                for arm in (
                    [a[0] for a in plan.arms()] if task["kind"] == "pair" else ["retry"]
                )
            )
            continue
        verify_rate_result(root, result)
        results.append(result)
        entries = (
            result.get("arms", {}) if task["kind"] == "pair" else {"retry": result}
        )
        if not entries:
            entries = {arm: result for arm, _, _ in plan.arms()}
        for arm, item in entries.items():
            fit = item.get("fit", {})
            scores = item.get("clean_signal_nmse", {})
            rows.append(
                {
                    "source": task["source_task"],
                    "arm": arm,
                    "status": item.get("status"),
                    "optimizer_success": fit.get("optimizer_success"),
                    "message": fit.get("message"),
                    "calls": fit.get("actual_residual_calls"),
                    "seconds": fit.get("fit_seconds"),
                    "optimality": fit.get("optimality"),
                    "raw_gradient_inf_norm": item.get("raw_gradient_inf_norm"),
                    "parameter_order": item.get("parameter_order"),
                    "stopping_policy": item.get("stopping_policy"),
                    "clean_train_nmse": scores.get("train"),
                    "clean_validation_nmse": scores.get("validation"),
                    "recovered": item.get("status") == "complete"
                    and all(
                        scores.get(n) is not None and scores[n] <= 1e-4
                        for n in ("train", "validation")
                    ),
                    "first_evaluation_cost": item.get("first_evaluation_cost"),
                }
            )
    summary = {
        "identity": frozen["freeze_sha256"],
        "rows": rows,
        "results": results,
        "source_freeze": frozen["source_freeze"],
        "runtime": frozen["runtime"],
        "protocol": plan.protocol,
        "recovery_by_initializer": recovery_counts(rows),
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# Saved-start stopping comparison"
        if plan.protocol == "fitter-rate-stopping-1"
        else "# Saved-start rate refinement",
        "",
        (
            "Same saved v4 initializer points; both arms use direct rate "
            "sensitivities. "
            "Only ftol differs (1e-8 versus disabled). No collocation rerun."
        )
        if plan.protocol == "fitter-rate-stopping-1"
        else (
            "Same v4 initializer points; direct physical or rate sensitivities. "
            "No collocation rerun."
        ),
        (
            "Clean metrics are post-fit diagnostics. Native success is separate "
            "from verification and recovery."
        ),
        "",
        "Recovery counts (both clean NMSEs <= 1e-4; "
        "missing/failed fits count as unrecovered):",
        "",
        "| Initializer | Arm | Starts | Verified | Recovered | Total |",
        "| --- | --- | --- | ---: | ---: | ---: |",
        *[
            f"| {g['method']} | {g['arm']} | {g['starts']} | {g['verified']} | "
            f"{g['recovered']} | {g['total']} |"
            for g in summary["recovery_by_initializer"]
        ],
        "",
        (
            "| Source | Arm | Status | Recovered | Calls | Seconds | "
            "Clean train NMSE | Clean validation NMSE |"
        ),
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                str(r.get(k, "—"))
                for k in (
                    "source",
                    "arm",
                    "status",
                    "recovered",
                    "calls",
                    "seconds",
                    "clean_train_nmse",
                    "clean_validation_nmse",
                )
            )
            + " |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def recovery_counts(rows: list[dict]) -> list[dict]:
    """Report C/J/A recovery with missing pairs retained in the denominator."""
    groups: dict[tuple, dict] = {}
    methods = {
        "collocation_sensitivity": "C+S",
        "joint_collocation_sensitivity": "J+S",
        "alternating_collocation_sensitivity": "A+S",
    }
    for row in rows:
        method = next(
            (
                label
                for suffix, label in reversed(list(methods.items()))
                if row["source"].endswith("_" + suffix)
            ),
            "other",
        )
        key = (
            method,
            row["arm"],
            "stress" if "stress" in row["source"] else "ordinary",
        )
        group = groups.setdefault(
            key,
            {
                "method": key[0],
                "arm": key[1],
                "starts": key[2],
                "total": 0,
                "verified": 0,
                "recovered": 0,
            },
        )
        group["total"] += 1
        group["verified"] += row.get("status") == "complete"
        group["recovered"] += bool(row.get("recovered"))
    return list(groups.values())
