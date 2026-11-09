"""Sealed, resumable training campaign for the five Astra T2 breadth descendants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import t2_breadth_search as fitting
from autoformalism.fitting.t2_breadth_rollout import BreadthRollout
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import construction_remaining_portfolio as operations
from autoformalism.research import continuation_report_cost as cost
from autoformalism.research import insulin_revision_fitting as artifacts
from autoformalism.research import t2_breadth_fit_intake as intake
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "t2-breadth-training-fit-1"
STAGES = ("numerical", "physical")


class Policy(StrictSchema):
    """Each task gets these two equal reserved search stages plus qualification."""

    seconds_per_stage: int = Field(default=900, ge=1, le=900)
    calls_per_stage: int = Field(default=1000, ge=4, le=1000)
    verification_seconds: int = Field(default=240, ge=10, le=300)


def _plan(data, policy):
    return {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": list(intake.TASKS),
        "selection_split": "train",
        "training_sha256": intake.TRAINING,
        "construction_archive_sha256": intake.ARCHIVE,
        "prefix": data["prefix"],
        "llm_calls": 0,
        "validation_or_test_access": False,
    }


def prepare(root: Path, archive: Path, policy: Policy):
    data = intake.load_archive(archive)
    plan = _plan(data, policy)
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"tasks": plan["tasks"], "plan_sha256": public.content_sha256(plan)}


def verify(root: Path, *, runtime=True):
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    expected = _plan(data, Policy.model_validate(plan["policy"]))
    if not runtime:
        expected.update(source_sha256=plan["source_sha256"], runtime=plan["runtime"])
    if (
        plan != expected
        or data["archive_sha256"] != intake.ARCHIVE
        or public.content_sha256(data["training"]) != intake.TRAINING
    ):
        raise ValueError("T2 fitting plan, training or execution identity differs")
    return plan, data


def _allowances(policy):
    result = {
        "warm": policy.verification_seconds + 60,
        "seed_profile": 180,
        "nested": policy.verification_seconds + 60,
    }
    for stage in STAGES:
        result["search_" + stage] = policy.seconds_per_stage + 450 + 60
        for kind in ("numerical", "admissible"):
            result[f"verify_{stage}_{kind}"] = policy.verification_seconds + 60
    return result


def _value(op, *, partial=False):
    if op and op["status"] == "complete":
        return op["value"]
    return (op or {}).get("observed_progress", {}) if partial else {}


def _model(key, data):
    branch = "c15-4" if key == intake.CONTROL else key
    item = data["models"][branch]
    return BreadthRollout(
        branch,
        PrefixFitProblem.model_validate(item["problem"]),
        PublicSplit.model_validate(data["training"]),
        item["warm"],
    )


def _verified(model, p, data, seconds, expected=None):
    """Verify original-equation trajectories with both adaptive solvers."""
    from time import monotonic

    started = monotonic()
    record = fitting.diagnostics(model, p, data, seconds=seconds)
    D, _ = model.original(
        p, 1, method="DOP853", seconds=max(0.01, seconds - (monotonic() - started))
    )
    D_loss = model.nmse(D)
    # Compare full observed grids in original units, including low-insulin tails.
    certified = np.asarray(record.pop("_observed_predictions"))
    discrepancy = float(np.max(abs((D - certified) / model.scales)))
    passed = np.allclose(
        D, certified, rtol=5e-7, atol=1e-6 * model.scales
    ) and np.isclose(D_loss, record["training_nmse"], rtol=2e-7, atol=1e-9)
    if expected is not None:
        passed = passed and np.isclose(
            record["training_nmse"], expected["all_training_nmse"], rtol=2e-7, atol=1e-8
        )
        passed = passed and all(
            np.isclose(record["target_nmse"][n], v, rtol=2e-7, atol=1e-8)
            for n, v in expected["pool_per_target_nmse"].items()
        )
    if not passed:
        raise ValueError(
            f"original replay or warm scores differ: {discrepancy=}, {D_loss=}"
        )
    record.update(
        two_original_solvers_verified=True,
        verification_calls_attempted=2,
        verification_calls_started=2,
        verification_calls_completed=2,
        alternate_nmse=D_loss,
        maximum_scaled_solver_difference=discrepancy,
        fast_nmse=model.nmse(model.fast(p)[:, ::2]),
    )
    return record


def _records(ops):
    # Retain the earlier verified incumbent on ties. Disk receipts are read in
    # filename order, which differs from execution order after a restart.
    names = ["warm", "nested"]
    for stage in STAGES:
        names.extend([f"verify_{stage}_numerical", f"verify_{stage}_admissible"])
    records = {}
    for name in names:
        row = _value(ops.get(name))
        if row.get("two_original_solvers_verified"):
            records[name] = row
    return records


def _select(records, *, feasible):
    result = None
    origin = None
    for name, r in records.items():
        if feasible and not r["sampled_domain_passed"]:
            continue
        if result is None or r["training_nmse"] < result["training_nmse"] - max(
            1e-10, 1e-8 * result["training_nmse"]
        ):
            origin, result = name, r
    return {"origin": origin, "endpoint": result}


def _assemble(identity, data, ops):
    key = identity["task"]
    records = _records(ops)
    searches = {s: _value(ops.get("search_" + s), partial=True) for s in STAGES}
    finished = all(
        ops.get("search_" + s, {}).get("status") == "complete"
        and searches[s].get("status")
        in ("budget_exhausted", "solver_returned", "solver_stopped")
        and searches[s].get("search_calls_completed", 0) > 0
        for s in STAGES
    )
    verifications_ok = all(
        ops.get(f"verify_{s}_{k}", {}).get("status") == "complete"
        for s in STAGES
        for k in ("numerical", "admissible")
    )
    branch = "c15-4" if key == intake.CONTROL else key
    return {
        "identity": identity,
        "task": key,
        "branch": branch,
        "is_nested_control": key == intake.CONTROL,
        "status": "fit_complete"
        if finished and verifications_ok and "warm" in records
        else "incomplete",
        "warm": records.get("warm"),
        "best_numerical": _select(records, feasible=False),
        "best_sampled_admissible": _select(records, feasible=True),
        "parent_reference_nmse": data["fixed"][data["models"][branch]["origin"]][
            "all_training_nmse"
        ],
        "seed_profile": _value(ops.get("seed_profile")),
        "searches": searches,
        "cost": cost.canonical_cost(ops),
        "operation_receipts": {
            k: public.content_sha256(v) for k, v in sorted(ops.items())
        },
        "operation_errors": {
            k: v.get("error", v["status"])
            for k, v in ops.items()
            if v["status"] != "complete"
        },
        "training_only": True,
        "validation_or_test_access": False,
        "convergence_certified": False,
        "prefix_forecasts": "deferred",
    }


def run_task(root: Path, index: int):
    plan, data = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown T2 breadth fitting task")
    key = plan["tasks"][index]
    policy = Policy.model_validate(plan["policy"])
    bound = _allowances(policy)
    folder = root / "results" / key
    identity = {"plan_sha256": public.content_sha256(plan), "task": key}
    with public._lock(folder):
        ops = operations._operations(folder, identity, bound)
        end = folder / "result.json"
        if end.exists():
            saved = read_seal(end)
            cost.reconcile(saved, _assemble(identity, data, ops), ops)
            return saved

        def perform(name, work):
            ops[name] = journal._operation(folder, identity, name, bound[name], work)
            return _value(ops[name])

        model = _model(key, data)
        control = key == intake.CONTROL
        candidates = fitting.starts(model, control=control)

        def verify_point(point, expected=None):
            for r in _records(ops).values():
                if r["parameters"] == point:
                    return r | {
                        "verification_calls_attempted": 0,
                        "verification_calls_started": 0,
                        "verification_calls_completed": 0,
                    }
            if control and point[fitting.FLOOR] != 0.0:
                raise ValueError("zero-basal control changed its fixed amplitude")
            return _verified(model, point, data, policy.verification_seconds, expected)

        warm = perform(
            "warm",
            lambda _: verify_point(
                candidates["inherited"], data["fixed"]["c15" if control else key]
            ),
        )
        if warm.get("two_original_solvers_verified"):
            nested = (
                "nested_zero_basal"
                if key == "c15-4"
                else "nested_zero_memory"
                if key == "c15-3"
                else None
            )
            if nested:
                perform(
                    "nested",
                    lambda _: verify_point(candidates[nested], data["fixed"]["c15"]),
                )
            if model.affine:
                seed = min(
                    candidates.values(), key=lambda p: model.nmse(model.fast(p)[:, ::2])
                )
                profile = perform(
                    "seed_profile", lambda _: fitting.affine_root_seed(model, seed)
                )
                if profile.get("passed"):
                    candidates["affine_conditional_roots"] = profile["parameters"]
            for stage in STAGES:

                def run_search(save, stage=stage):
                    return fitting.search(
                        model,
                        candidates,
                        physical=stage == "physical",
                        control=control,
                        seconds=policy.seconds_per_stage,
                        calls=policy.calls_per_stage,
                        save=save,
                    )

                perform("search_" + stage, run_search)
                search_result = _value(ops["search_" + stage], partial=True)
                for label in ("numerical", "admissible"):
                    best = search_result.get("best", {}).get(label)
                    result = perform(
                        f"verify_{stage}_{label}",
                        lambda _, b=best: verify_point(b["parameters"])
                        if b
                        else {"status": "no_candidate_in_this_category"},
                    )
                    if result.get("two_original_solvers_verified"):
                        candidates[f"{stage}_{label}"] = result["parameters"]
        result = _assemble(identity, data, ops)
        seal(end, result)
        return result


def compact(row):
    def metric(name):
        return ((row[name] or {}).get("endpoint") or {}).get("training_nmse")

    return {
        "task": row["task"],
        "status": row["status"],
        "warm": (row.get("warm") or {}).get("training_nmse"),
        "best_nmse": metric("best_numerical"),
        "admissible_nmse": metric("best_sampled_admissible"),
        "errors": row["operation_errors"],
    }


def status(root: Path):
    plan = read_seal(root / "plan.json")
    rows = []
    for key in plan["tasks"]:
        if key not in intake.TASKS:
            raise ValueError("unexpected task path")
        terminal = root / "results" / key / "result.json"
        if terminal.is_file():
            r = read_seal(terminal)
            if r["identity"] != {
                "plan_sha256": public.content_sha256(plan),
                "task": key,
            }:
                raise ValueError("foreign T2 fitting terminal")
            rows.append(compact(r))
        else:
            rows.append({"task": key, "status": "pending"})
    return {
        "expected": len(rows),
        "recorded": sum(r["status"] != "pending" for r in rows),
        "rows": rows,
    }


def report(root: Path):
    plan, data = verify(root, runtime=False)
    rows = []
    checks = {}
    policy = Policy.model_validate(plan["policy"])
    for key in plan["tasks"]:
        folder = root / "results" / key
        identity = {"plan_sha256": public.content_sha256(plan), "task": key}
        ops = operations._operations(folder, identity, _allowances(policy))
        row = _assemble(identity, data, ops)
        if (folder / "result.json").exists():
            checks[key] = cost.reconcile(read_seal(folder / "result.json"), row, ops)
        rows.append(row)
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "expected": len(rows),
        "recorded": len(checks),
        "fitted": sum(
            r["status"] == "fit_complete" and r["task"] in checks for r in rows
        ),
        "rows": rows,
        "reconciled": checks,
        "training_only": True,
        "validation_or_test_access": False,
    }
    summary["status"] = "complete" if summary["fitted"] == len(rows) else "incomplete"
    with public._lock(root / "report"):
        public._write(root / "summary.json", summary)
        lines = [
            "# T2 breadth training fits",
            "",
            "| Model | Status | Inherited | Best verified | Sampled admissible |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for row in map(compact, rows):
            lines.append(
                f"| {row['task']} | {row['status']} | {row['warm']} | "
                f"{row['best_nmse']} | {row['admissible_nmse']} |"
            )
        (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def inspection(root: Path):
    return artifacts.inspection(root, reporter=report)
