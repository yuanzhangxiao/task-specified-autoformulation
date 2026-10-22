"""Matched parent/child fits using the e7067f3 numerical routines.

All candidates receive the same fresh start policy and operation allowances.
Only training error selects a numerically verified endpoint; held-out prefixes
infer roots with shared parameters frozen. Historical campaigns are read-only.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_continuation as numerical
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_continuation as previous
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import construction_fit_reliability as reliability
from autoformalism.research import construction_paired_evaluation as evaluation
from autoformalism.research import construction_paired_inputs as source
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-parent-child-fit-1"
FITTER_BASE_COMMIT = "e7067f3"
STARTS = ("declared", "perturbed_declared")


class Policy(StrictSchema):
    """Identical finite budgets; no inherited fit or validation-dependent restart."""

    population: Literal["returned-24", "synthetic-smoke"] = "returned-24"
    search_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    diagnostic_seconds: FiniteFloat = Field(default=120, gt=0, le=120)
    verification_seconds: FiniteFloat = Field(default=180, gt=0, le=180)
    inference_seconds: FiniteFloat = Field(default=60, gt=0, le=60)
    forecast_seconds: FiniteFloat = Field(default=180, gt=0, le=180)
    maximum_calls: int = Field(default=1000, ge=1, le=1000)
    inference_calls: int = Field(default=100, ge=1, le=100)
    numerical: kernel.NumericalPolicy = Field(default_factory=kernel.NumericalPolicy)
    arm: Literal["affine_reduced"] = "affine_reduced"
    perturbation_seed: Literal[1729] = 1729


def _tasks(data: dict) -> list[dict]:
    """Keep every parent and accepted changed child in a stable roster."""
    return [
        {
            "index": i,
            "task_id": name,
            **{key: row[key] for key in ("role", "parent_task_id", "case_id")},
            "problem_sha256": public.content_sha256(row["problem"]),
        }
        for i, (name, row) in enumerate(sorted(data["records"].items()))
    ]


def prepare(root: Path, data: dict, policy: Policy) -> dict:
    """Seal data, starts and executable identity without numerical work."""
    data = source.validate_inputs(data, population=policy.population)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "fitter_base_commit": FITTER_BASE_COMMIT,
        "policy": policy.model_dump(mode="json"),
        "tasks": _tasks(data),
        "starts": list(STARTS),
        "start_policy": "declared_and_name_hashed_perturbation_no_historical_warmstart",
        "selection": "verified_training_error_only_before_validation",
        "validation": "shared_frozen_roots_from_prefix_score_strict_suffix",
        "normalization": "training_channel_std_floor_1e-12",
        "historical_budgets_renewed": False,
        "model_selection": False,
        "automatic_revision": False,
        "automatic_promotion": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return plan


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Require immutable scope; reports may inspect a historical runtime."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    policy = Policy.model_validate(plan["policy"])
    parsed = source.validate_inputs(data, population=policy.population)
    if (
        parsed != data
        or plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != _tasks(data)
        or plan["starts"] != list(STARTS)
        or plan["fitter_base_commit"] != FITTER_BASE_COMMIT
        or plan["llm_calls"] != 0
        or any(
            plan[k] is not False
            for k in (
                "historical_budgets_renewed",
                "model_selection",
                "automatic_revision",
                "automatic_promotion",
                "test_data_opened",
            )
        )
        or plan["start_policy"]
        != "declared_and_name_hashed_perturbation_no_historical_warmstart"
        or plan["selection"] != "verified_training_error_only_before_validation"
        or plan["validation"] != "shared_frozen_roots_from_prefix_score_strict_suffix"
        or plan["normalization"] != "training_channel_std_floor_1e-12"
    ):
        raise ValueError("paired fitting plan differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("paired fitting source/runtime differs")
    return plan, data


def starts(oracle) -> list[dict]:
    """Perturb by coordinate name, so shared names get the same random fraction."""
    original = oracle.vector(oracle.start)
    fractions = np.asarray(
        [
            int.from_bytes(hashlib.sha256(f"1729:{name}".encode()).digest()[:8], "big")
            / 2**64
            - 0.5
            for name in oracle.names
        ]
    )
    perturbed = np.clip(original + fractions * oracle.units, oracle.lower, oracle.upper)
    return [dict(oracle.start), oracle.parameters(perturbed)]


def _allowances(policy: Policy, data: dict) -> dict[str, float]:
    """Bound setup/checkpoint overhead separately from active numerical budgets."""
    allowed = {"fit_projection": 90.0}
    for name in STARTS:
        allowed.update(
            {
                f"{name}_qualification": policy.diagnostic_seconds + 90,
                f"{name}_initial_verification": policy.verification_seconds + 90,
                f"{name}_search": policy.search_seconds + 150,
                f"{name}_verification": policy.verification_seconds + 90,
                f"{name}_diagnosis": policy.diagnostic_seconds + 90,
            }
        )
    for index in range(len(data["validation"]["rows"])):
        allowed[f"val_{index}_inference"] = (
            policy.inference_seconds + policy.verification_seconds + 240
        )
        allowed[f"val_{index}_forecast"] = policy.forecast_seconds + 120
    return allowed


def _operations(folder: Path, identity: dict, allowed: dict) -> dict:
    """Authenticate all reservations, including interrupted and incomplete stages."""
    operations = {}
    for location in sorted((folder / "operations").glob("*")):
        if not location.is_dir():
            continue
        if location.is_symlink() or location.name not in allowed:
            raise ValueError("unknown paired fit operation")
        start = location / "started.json"
        if not start.exists():
            if any(
                (location / name).exists() for name in ("result.json", "progress.json")
            ):
                raise ValueError("operation artifact lacks reservation")
            continue
        bound = {
            "campaign": identity,
            "stage": location.name,
            "allowance_seconds": allowed[location.name],
        }
        if read_seal(start) != bound:
            raise ValueError("paired fit reservation differs")
        if (location / "result.json").exists():
            operations[location.name] = journal._read_operation(location, bound)
        else:
            progress = (
                public._read(location / "progress.json")
                if (location / "progress.json").exists()
                else {}
            )
            operations[location.name] = {
                "identity": bound,
                "status": "running_or_interrupted",
                "value": None,
                "observed_progress": progress,
                "wall_seconds": None,
                "cpu_seconds": None,
                "budget_charge_seconds": allowed[location.name],
                "accounting_complete": False,
                "budget_restarted": False,
            }
    return operations


def _selected(operations: dict) -> tuple[str | None, dict | None]:
    """A failed/unverified checkpoint cannot displace a verified endpoint."""
    origin, best = None, None
    for name in STARTS:
        for suffix in ("initial_verification", "verification"):
            key = f"{name}_{suffix}"
            point = reliability._verified(operations.get(key, {}))
            if point is not None and (
                best is None or reliability._improves(point, best)
            ):
                origin, best = key, point
    return origin, best


def _fit_packet(original, point: dict, training: PublicSplit) -> dict:
    """Project a verified original-coordinate endpoint into the frozen-prefix API."""
    parameters = point["parameters"]
    original.vector(parameters)
    return {
        "protocol": "prefix-training-fit-1",
        "problem_sha256": public.content_sha256(original.problem),
        "training_sha256": public.content_sha256(training),
        "fit_success": True,
        "parameters": parameters,
        "shared_parameters": {name: parameters[name] for name in original.shared},
        "training_roots": {
            row.trajectory_id: {
                name: parameters[f"trajectory:{row.trajectory_id}:{name}"]
                for name in original.roots
            }
            for row in original.rows
        },
        "training_scales": dict(original.scales),
        "normalized_mse": point["training_nmse"],
        "initialization_audit": original.audit,
        "selected_verification_sha256": public.content_sha256(point),
        "selection_data": "train_only",
        "optimization_convergence_implied": False,
    }


def _value(operation: dict | None, success: str) -> dict | None:
    """Only completed successful numerical receipts become reportable predictions."""
    operation = operation or {}
    value = operation.get("value") or {}
    return (
        value
        if operation.get("status") == "complete"
        and value.get("status", "complete") == "complete"
        and value.get(success)
        else None
    )


def _aggregate(scores: list[dict], expected: int) -> dict | None:
    """Require every validation row; never hide unavailable trajectories."""
    if len(scores) != expected:
        return None
    count = sum(s["sample_count"] for s in scores)
    per_target = {
        channel: sum(
            s["sample_count"] / count * s["per_target_normalized_mse"][channel]
            for s in scores
        )
        for channel in scores[0]["per_target_normalized_mse"]
    }
    return {
        "normalized_mse": sum(v / len(per_target) for v in per_target.values()),
        "per_target_normalized_mse": per_target,
        "sample_count": count,
    }


def _assemble(identity: dict, operations: dict, data: dict, *, terminal=False) -> dict:
    """Reconstruct endpoints and scores without numerical calls or model selection."""
    origin, best = _selected(operations)
    projection = _value(operations.get("fit_projection"), "fit_success")
    if projection and (
        best is None
        or projection["selected_verification_sha256"] != public.content_sha256(best)
        or projection["parameters"] != best["parameters"]
    ):
        raise ValueError("frozen fit differs from selected training endpoint")
    starts_summary = []
    for name in STARTS:
        search = (
            previous._search_checkpoint(operations[f"{name}_search"])
            if f"{name}_search" in operations
            else {}
        )
        verified = reliability._verified(operations.get(f"{name}_verification", {}))
        diagnosis = _value(operations.get(f"{name}_diagnosis"), "assessment")
        starts_summary.append(
            {
                "name": name,
                "qualification_passed": bool(
                    _value(operations.get(f"{name}_qualification"), "passed")
                ),
                "search_recorded": f"{name}_search" in operations,
                "stop_reason": search.get("stop_reason", "not_started"),
                "training_nmse": verified["training_nmse"] if verified else None,
                "optimization_assessment": diagnosis.get("assessment")
                if diagnosis
                else None,
            }
        )
    selected_assessment = None
    if (
        origin
        and origin.endswith("_verification")
        and not origin.endswith("initial_verification")
    ):
        name = origin.removesuffix("_verification")
        diag = _value(operations.get(f"{name}_diagnosis"), "assessment")
        if diag:
            selected_assessment = diag["assessment"]
    rows, scores = [], []
    for index, row in enumerate(PublicSplit.model_validate(data["validation"]).rows):
        inference = _value(operations.get(f"val_{index}_inference"), "fit_success")
        forecast = _value(operations.get(f"val_{index}_forecast"), "forecast_success")
        score, scoring_error = None, None
        if forecast:
            if not projection or not inference:
                raise ValueError("forecast lacks frozen fit or prefix receipt")
            if inference["fit_sha256"] != public.content_sha256(projection):
                raise ValueError("inference fit identity differs")
            if forecast["inference_sha256"] != public.content_sha256(inference):
                raise ValueError("forecast inference identity differs")
            with np.errstate(over="ignore", invalid="ignore"):
                score = kernel.score_forecast(
                    forecast,
                    row,
                    row.time[0] + data["prefix_policy"]["duration"],
                    projection["training_scales"],
                )
            if not np.isfinite(
                [score["normalized_mse"], *score["per_target_normalized_mse"].values()]
            ).all():
                scoring_error, score = "nonfinite validation score", None
            else:
                scores.append(score)
        rows.append(
            {
                "trajectory_id": row.trajectory_id,
                "inference_available": inference is not None,
                "forecast_available": forecast is not None,
                "score": score,
                "scoring_error": scoring_error,
            }
        )
    return {
        "protocol": PROTOCOL,
        "identity": identity,
        "status": "complete" if terminal else "incomplete",
        "selected_origin": origin,
        "numerically_verified": best is not None,
        "training_nmse": best["training_nmse"] if best else None,
        "training_per_target": best.get("per_target_normalized_mse") if best else None,
        "training_score_scope": (
            "full_training_reconstruction_with_fitted_training_roots"
        ),
        "retained": best,
        "optimization_assessment": selected_assessment,
        "optimization_status": (selected_assessment or {}).get(
            "classification", "optimization_unresolved"
        ),
        "starts": starts_summary,
        "validation_rows": rows,
        "validation_score": _aggregate(scores, len(rows)),
        "validation_available_rows": len(scores),
        "operations": operations,
        "cost": journal._cost(operations),
        "model_selection": False,
        "automatic_promotion": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }


def run_task(root: Path, index: int) -> dict:
    """Run two fixed searches and the held-out prefix evaluation, each once."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown paired fitting task")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    policy = Policy.model_validate(plan["policy"])
    allowed = _allowances(policy, data)
    folder = root / "results" / task["task_id"]
    problem = PrefixFitProblem.model_validate(
        data["records"][task["task_id"]]["problem"]
    )
    training = PublicSplit.model_validate(data["training"])

    def operate(name, work, *, build=True):
        def call(save):
            deadline = monotonic() + allowed[name]

            def check():
                if monotonic() >= deadline:
                    raise TimeoutError(
                        "paired fitting inclusive stage allowance exhausted"
                    )

            def checked_save(value):
                save(value)
                check()

            original, oracle = (
                numerical.make_oracle(problem, training, policy.numerical, policy.arm)
                if build
                else (None, None)
            )
            check()
            result = work(original, oracle, checked_save)
            check()
            return result

        return journal._operation(folder, identity, name, allowed[name], call)

    with public._lock(folder):
        terminal = folder / "result.json"
        if terminal.exists():
            saved = read_seal(terminal)
            rebuilt = _assemble(
                identity, _operations(folder, identity, allowed), data, terminal=True
            )
            if saved != rebuilt:
                raise ValueError("paired terminal differs from receipts")
            if read_seal(folder / "training_selection.json") != {
                "origin": rebuilt["selected_origin"],
                "endpoint": rebuilt["retained"],
            }:
                raise ValueError("training selection differs from verified history")
            return saved
        for index, name in enumerate(STARTS):
            qualified = operate(
                f"{name}_qualification",
                lambda o, a, save, i=index: numerical.qualification(
                    o,
                    a,
                    starts(o)[i],
                    policy.diagnostic_seconds,
                    save,
                    adequacy_policy=adequacy.Policy(precision_aware=True),
                ),
            )
            initial = operate(
                f"{name}_initial_verification",
                lambda o, a, save, i=index: numerical.verify_endpoint_ladder(
                    o,
                    a,
                    numerical.to_effective(o, a, starts(o)[i]),
                    policy.verification_seconds,
                    save,
                ),
            )
            if (
                not _value(qualified, "passed")
                or reliability._verified(initial) is None
            ):
                continue
            attempt = operate(
                f"{name}_search",
                lambda o, a, save, i=index: numerical.search(
                    a,
                    numerical.to_effective(o, a, starts(o)[i]),
                    policy.search_seconds,
                    policy.maximum_calls,
                    save,
                    reject_invalid_trials=True,
                ),
            )
            search = previous._search_checkpoint(attempt)
            if not search.get("best"):
                continue
            best = search["best"]["parameters"]
            checked = operate(
                f"{name}_verification",
                lambda o, a, save, b=best: numerical.verify_endpoint_ladder(
                    o, a, b, policy.verification_seconds, save
                ),
            )
            point = reliability._verified(checked)
            if point:
                operate(
                    f"{name}_diagnosis",
                    lambda o, a, save, b=best, p=point, s=search: numerical.diagnose(
                        a,
                        b,
                        p,
                        s,
                        policy.diagnostic_seconds,
                        save,
                        adequacy_policy=adequacy.Policy(precision_aware=True),
                    ),
                )
        origin, selected = _selected(_operations(folder, identity, allowed))
        seal(
            folder / "training_selection.json", {"origin": origin, "endpoint": selected}
        )
        if selected:
            projection = operate(
                "fit_projection", lambda o, a, save: _fit_packet(o, selected, training)
            )
            fit = _value(projection, "fit_success")
            if fit:
                prefix_policy = kernel.PrefixPolicy.model_validate(
                    data["prefix_policy"]
                )
                for index, row in enumerate(
                    PublicSplit.model_validate(data["validation"]).rows
                ):
                    prefix = kernel.make_prefix(row, prefix_policy)
                    inference = operate(
                        f"val_{index}_inference",
                        lambda o, a, save, p=prefix: evaluation.infer(
                            problem,
                            fit,
                            p,
                            policy.numerical,
                            policy.inference_seconds,
                            policy.verification_seconds,
                            policy.inference_calls,
                            save,
                        ),
                        build=False,
                    )
                    inferred = _value(inference, "fit_success")
                    if inferred:
                        schedule = kernel.make_forecast_schedule(row, prefix_policy)
                        operate(
                            f"val_{index}_forecast",
                            lambda o,
                            a,
                            save,
                            s=schedule,
                            i=inferred: evaluation.forecast(
                                problem,
                                fit,
                                i,
                                s,
                                policy.numerical,
                                policy.forecast_seconds,
                                save,
                            ),
                            build=False,
                        )
        result = public._jsonable(
            _assemble(
                identity, _operations(folder, identity, allowed), data, terminal=True
            )
        )
        seal(terminal, result)
        return result


def _spread(values: list[float]) -> dict:
    """Finite-score descriptive statistics; failures remain separately counted."""
    if not values:
        return {"count": 0, "median": None, "mad": None}
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("finite scores required for descriptive statistics")
    scale = max(float(np.max(np.abs(array))), 1.0)
    scaled = array / scale
    center = float(np.median(scaled))
    return {
        "count": len(values),
        "median": center * scale,
        "mad": float(np.median(np.abs(scaled - center))) * scale,
    }


def report(root: Path) -> dict:
    """Report all pairs and failures; never select the best child by validation."""
    plan, data = verify(root, runtime=False)
    policy = Policy.model_validate(plan["policy"])
    allowed = _allowances(policy, data)
    rows = []
    for task in plan["tasks"]:
        folder = root / "results" / task["task_id"]
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        terminal = (folder / "result.json").exists()
        operations = _operations(folder, identity, allowed)
        value = _assemble(identity, operations, data, terminal=terminal)
        if terminal and read_seal(folder / "result.json") != public._jsonable(value):
            raise ValueError("paired result differs from operation receipts")
        selection = folder / "training_selection.json"
        if terminal and not selection.exists():
            raise ValueError("terminal result lacks frozen training selection")
        if terminal and read_seal(selection) != {
            "origin": value["selected_origin"],
            "endpoint": value["retained"],
        }:
            raise ValueError("training selection differs from verified history")
        rows.append(
            task
            | {
                "status": "complete"
                if terminal
                else "partial"
                if operations
                else "pending",
                **{
                    key: value[key]
                    for key in (
                        "numerically_verified",
                        "training_nmse",
                        "training_per_target",
                        "optimization_status",
                        "optimization_assessment",
                        "starts",
                        "validation_score",
                        "validation_available_rows",
                        "cost",
                    )
                },
            }
        )
    by_id = {r["task_id"]: r for r in rows}
    pairs = []
    for child in rows:
        if child["role"] != "child":
            continue
        parent = by_id[child["parent_task_id"]]
        pair = {
            "parent": parent["task_id"],
            "child": child["task_id"],
            "parent_case": parent["case_id"],
            "child_case": child["case_id"],
        }
        for split in ("training", "validation"):
            p, c = (
                (
                    x["training_nmse"]
                    if split == "training"
                    else (x["validation_score"] or {}).get("normalized_mse")
                )
                for x in (parent, child)
            )
            pair[split] = {
                "parent_nmse": p,
                "child_nmse": c,
                "child_minus_parent": c - p
                if p is not None and c is not None
                else None,
            }
        pairs.append(pair)
    aggregates = {}
    for split in ("training", "validation"):
        deltas = [
            p[split]["child_minus_parent"]
            for p in pairs
            if p[split]["child_minus_parent"] is not None
        ]
        per_parent = []
        for parent in (r for r in rows if r["role"] == "parent"):
            values = [
                p[split]["child_minus_parent"]
                for p in pairs
                if p["parent"] == parent["task_id"]
                and p[split]["child_minus_parent"] is not None
            ]
            if values:
                per_parent.append(float(np.median(values)))
        score_summaries = {}
        for role in ("parent", "child"):
            cohort = [r for r in rows if r["role"] == role]
            values = [
                r["training_nmse"]
                if split == "training"
                else (r["validation_score"] or {}).get("normalized_mse")
                for r in cohort
            ]
            finite = [v for v in values if v is not None]
            score_summaries[role] = _spread(finite) | {
                "expected": len(cohort),
                "unavailable": len(cohort) - len(finite),
            }
        aggregates[split] = {
            "model_nmse": score_summaries,
            "paired_child_delta": _spread(deltas),
            "parent_median_delta": _spread(per_parent),
            "expected_pairs": len(pairs),
            "unavailable_pairs": len(pairs) - len(deltas),
            "improved_pairs": sum(x < 0 for x in deltas),
            "worsened_pairs": sum(x > 0 for x in deltas),
            "equal_pairs": sum(x == 0 for x in deltas),
            "interpretation": (
                "Descriptive finite pairs with explicit coverage; correlated children "
                "share parents. No best-child selection or causal attribution of "
                "optimizer vs construction error."
            ),
        }
    summary = {
        "protocol": PROTOCOL,
        "status": "complete"
        if all(r["status"] == "complete" for r in rows)
        else "incomplete",
        "expected": len(rows),
        "recorded": sum(r["status"] == "complete" for r in rows),
        "numerically_verified": sum(r["numerically_verified"] for r in rows),
        "validation_available": sum(r["validation_score"] is not None for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "rows": rows,
        "pairs": pairs,
        "aggregates": aggregates,
        "construction_episode_counts": dict(
            Counter(e["status"] for e in data["episodes"])
        ),
        "prefix_policy": data["prefix_policy"],
        "training_scope": "full reconstruction using fitted training roots",
        "validation_scope": "strictly after fixed prefix; shared parameters frozen",
        "scientific_compliance_assessed": False,
        "test_data_opened": False,
        "llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
    }
    (root / "report").mkdir(parents=True, exist_ok=True)
    public._write(root / "report/summary.json", summary)
    lines = [
        "# Parent-child fitting comparison",
        "",
        "Training reconstruction; validation forecasts strictly after the fixed "
        "prefix.",
        "All candidates and numerical failures retained. Median and unscaled MAD; "
        "no best-child selection.",
        "",
        f"Recorded {summary['recorded']}/{len(rows)}; "
        f"verified {summary['numerically_verified']}; "
        f"validation available {summary['validation_available']}.",
        "",
        "| Model | Role | Status | Train NMSE | Validation NMSE | Optimization |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        val = (row["validation_score"] or {}).get("normalized_mse")
        lines.append(
            f"| {row['task_id']} | {row['role']} | {row['status']} | "
            f"{row['training_nmse']} | {val} | {row['optimization_status']} |"
        )
    lines += [
        "",
        "| Pair | Train child - parent | Validation child - parent |",
        "| --- | ---: | ---: |",
    ]
    lines += [
        f"| {p['parent']} -> {p['child']} | "
        f"{p['training']['child_minus_parent']} | "
        f"{p['validation']['child_minus_parent']} |"
        for p in pairs
    ]
    (root / "report/SUMMARY.md").write_text("\n".join(lines) + "\n")
    return summary


def inspect(root: Path) -> dict:
    """Package receipts and inputs without fitting or automatic continuation."""
    summary = report(root)
    paths = [root / "inputs.json", root / "plan.json"]
    for folder in ("results", "report", "logs", "submissions"):
        paths.extend(p for p in (root / folder).rglob("*") if p.is_file())
    if (root / "launcher_execution.json").exists():
        paths.append(root / "launcher_execution.json")
    archive = root / "inspection.tar.gz"
    with (
        archive.open("wb") as stream,
        gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w") as tar,
    ):
        for path in sorted(paths):
            if path.is_symlink():
                raise ValueError("refusing symlink in paired inspection")
            info = tarfile.TarInfo(str(path.relative_to(root)))
            raw = path.read_bytes()
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    return {
        "archive": str(archive),
        "recorded": summary["recorded"],
        "expected": summary["expected"],
    }
