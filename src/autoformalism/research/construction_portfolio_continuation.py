"""Saved-prerequisite qualification and certified affine incumbent continuation.

This opt-in campaign owns new finite allowances, preserves the old campaigns,
and never changes equations, parameter domains, prefix access or the objective.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_continuation as coordinates
from autoformalism.fitting import prefix_execution as execution
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_portfolio as numerical
from autoformalism.fitting import prefix_portfolio_forecast as affine
from autoformalism.fitting import prefix_portfolio_polish as polish
from autoformalism.fitting import prefix_portfolio_prerequisites as prerequisites
from autoformalism.fitting import prefix_verification as forecasting
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import construction_portfolio_fitting as first
from autoformalism.research import construction_portfolio_intake as intake
from autoformalism.research import construction_portfolio_recovery as previous
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem, PrefixPolicy
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-portfolio-continuation-1"
SOURCE_PLAN = "8f213fad2056c99cab1d00a971a24cc9fcd6e23453a9622d13e03b71089bd0bc"
SELECTED = (3, 10, 12, 14, 15)


class Policy(StrictSchema):
    """New fixed allowances; selection and numerical information stay training-only."""

    population: Literal["scaled-readout-5", "synthetic-smoke"] = "scaled-readout-5"
    prerequisite_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    coordinate_seconds: FiniteFloat = Field(default=240, gt=0, le=240)
    qualification_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    search_seconds: FiniteFloat = Field(default=1800, gt=0, le=1800)
    verification_seconds: FiniteFloat = Field(default=180, gt=0, le=180)
    diagnostic_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    maximum_calls: int = Field(default=500, gt=0, le=500)
    inference: kernel.NumericalPolicy = Field(
        default_factory=lambda: kernel.NumericalPolicy(
            seconds=45, maximum_calls=100, verification_seconds=60
        )
    )
    inference_execution: execution.SearchExecutionPolicy = Field(
        default_factory=lambda: execution.SearchExecutionPolicy(
            search_overall_seconds=75, verification_overall_seconds=90
        )
    )
    forecast_execution: forecasting.ExecutionPolicy = Field(
        default_factory=lambda: forecasting.ExecutionPolicy(
            numerical_seconds=90, overall_seconds=120
        )
    )


def _source_inputs(source: Path, policy: Policy) -> dict:
    """Authenticate every historical terminal before copying selected evidence."""
    plan, data = previous.verify(source, runtime=False)
    prior_policy = previous.Policy.model_validate(plan["policy"])
    selected = (
        SELECTED
        if policy.population == "scaled-readout-5"
        else range(len(plan["tasks"]))
    )
    records = []
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = source / "results" / task["task_id"]
        operations = previous._operations(
            folder,
            identity,
            previous._allowances(task, prior_policy, plan["training_rows"]),
        )
        terminal = read_seal(folder / "result.json")
        if terminal != previous._assemble(identity, data, operations):
            raise ValueError("source recovery terminal differs from receipts")
        if task["index"] not in selected:
            continue
        envelopes = {
            method: json.loads(
                (folder / f"operations/replay_{method}/result.json").read_text()
            )
            for method in ("Radau", "BDF")
            if task["kind"] == "nonlinear_qualification"
        }
        records.append(
            {
                "model_index": task["index"],
                "candidate_id": task["candidate_id"],
                "source_task": task,
                "source_terminal_sha256": public.content_sha256(terminal),
                "source_record": data["records"][task["index"]],
                "source_prefix_forecast": terminal["training_prefix_forecast"],
                "source_prefix_rows": terminal["training_prefix_rows"],
                "source_prefix_available_rows": terminal[
                    "training_prefix_available_rows"
                ],
                "source_assessment": terminal["assessment"],
                "replay_envelopes": envelopes,
            }
        )
    return {
        "protocol": PROTOCOL,
        "source_plan_sha256": public.content_sha256(plan),
        "portfolio": data["portfolio"],
        "original_policy": data["source_policy"],
        "records": records,
        "test_data_opened": False,
    }


def _tasks(data: dict, policy: Policy) -> list[dict]:
    """Bind the declared five-model follow-up to the complete original roster."""
    if data["protocol"] != PROTOCOL or data["test_data_opened"] is not False:
        raise ValueError("continuation input protocol differs")
    portfolio = data["portfolio"]
    production = policy.population == "scaled-readout-5"
    count = 23 if production else len(portfolio["models"])
    intake.validate_inputs(portfolio, expected_count=count)
    indices = list(SELECTED) if production else list(range(count))
    if production and data["source_plan_sha256"] != SOURCE_PLAN:
        raise ValueError("continuation source is not the reviewed recovery")
    if [r["model_index"] for r in data["records"]] != indices:
        raise ValueError("continuation requires the entire declared follow-up roster")
    tasks = []
    for index, record in enumerate(data["records"]):
        model = portfolio["models"][record["model_index"]]
        source = record["source_record"]
        task = record["source_task"]
        if not (
            model["candidate_id"]
            == record["candidate_id"]
            == source["candidate_id"]
            == task["candidate_id"]
        ):
            raise ValueError("continuation model identity differs")
        nonlinear = task["kind"] == "nonlinear_qualification"
        if task["kind"] not in {"nonlinear_qualification", "affine_prefix"}:
            raise ValueError("unknown source task kind")
        if type(task["coordinates"]) is not int or not 0 <= task["coordinates"] <= 128:
            raise ValueError("invalid coordinate count")
        if nonlinear:
            if set(record["replay_envelopes"]) != {"Radau", "BDF"}:
                raise ValueError("nonlinear continuation requires both source replays")
            for method, envelope in record["replay_envelopes"].items():
                value = envelope["value"]
                if public.content_sha256(value) != envelope["sha256"] or (
                    value["identity"]["campaign"]
                    != {"plan_sha256": data["source_plan_sha256"], "task": task}
                    or value["identity"]["stage"] != f"replay_{method}"
                ):
                    raise ValueError("inherited replay envelope differs")
        elif not first._verified(
            {"status": "complete", "value": source["endpoint"] or {}}
        ):
            raise ValueError("affine continuation requires a verified incumbent")
        tasks.append(
            {
                "index": index,
                "task_id": model["candidate_id"],
                "candidate_id": model["candidate_id"],
                "model_index": record["model_index"],
                "kind": "nonlinear_prerequisites"
                if nonlinear
                else "affine_continuation",
                "coordinates": task["coordinates"],
                "source_record_sha256": public.content_sha256(record),
                "problem_sha256": model["fitting_problem_sha256"],
            }
        )
    return tasks


def prepare(root: Path, source: Path, policy: Policy) -> dict:
    """Freeze a new campaign without numerical work or modifying source evidence."""
    if root.resolve() == source.resolve() or root.resolve().is_relative_to(
        source.resolve()
    ):
        raise ValueError("continuation root must be separate from the source")
    return prepare_inputs(root, _source_inputs(source, policy), policy)


def prepare_inputs(root: Path, data: dict, policy: Policy) -> dict:
    """Freeze validated evidence; synthetic tests use the same boundary."""
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": _tasks(data, policy),
        "training_rows": len(data["portfolio"]["training"]["rows"]),
        "test_data_opened": False,
        "live_llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
        "objective": "original_full_training_reconstruction",
        "prefix_scores_used_for_selection": False,
    }
    if reservation_ceiling(plan) > 13200:
        raise ValueError("continuation reservations exceed four-hour envelope")
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Require unchanged source, scientific inputs and finite allowance policy."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    policy = Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != _tasks(data, policy)
        or plan["training_rows"] != len(data["portfolio"]["training"]["rows"])
        or any(
            plan[k] is not False
            for k in (
                "test_data_opened",
                "automatic_revision",
                "automatic_promotion",
                "prefix_scores_used_for_selection",
            )
        )
        or plan["live_llm_calls"] != 0
        or plan["objective"] != "original_full_training_reconstruction"
        or reservation_ceiling(plan) > 13200
    ):
        raise ValueError("continuation identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("continuation source/runtime differs")
    return plan, data


def _allowances(task: dict, policy: Policy, rows: int) -> dict[str, float]:
    if task["kind"] == "nonlinear_prerequisites":
        return {
            "setup": 120,
            "baseline": 60,
            "jacobian": policy.prerequisite_seconds + 60,
            "qualification_summary": 60,
        } | {
            f"coordinate_{i:03d}": policy.coordinate_seconds + 60
            for i in range(task["coordinates"])
        }
    return {
        "setup": 120,
        "start_verification": policy.verification_seconds + 60,
        "qualification": policy.qualification_seconds + 60,
        "search": policy.search_seconds + 120,
        "endpoint_verification": policy.verification_seconds + 60,
        "diagnosis": policy.diagnostic_seconds + 60,
    } | {
        f"train_{i:02d}_{stage}": seconds
        for i in range(rows)
        for stage, seconds in (
            (
                "inference",
                policy.inference_execution.search_overall_seconds
                + policy.inference_execution.verification_overall_seconds,
            ),
            ("forecast", policy.forecast_execution.overall_seconds),
        )
    }


def reservation_ceiling(plan: dict) -> float:
    """Sum inclusive operation reservations, counting nested probes only once."""
    policy = Policy.model_validate(plan["policy"])
    return max(
        (
            sum(_allowances(t, policy, plan["training_rows"]).values())
            for t in plan["tasks"]
        ),
        default=0,
    )


def _probe_operations(folder: Path, identity: dict, index: int) -> dict:
    """Authenticate nested primitive receipts; their costs belong to the parent."""
    location = folder / "probes" / f"coordinate_{index:03d}"
    names = {}
    for path in (location / "operations").glob("*"):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", path.name):
            raise ValueError("unsafe probe stage")
        names[path.name] = 120
    return previous._operations(location, identity, names)


def _operations(folder: Path, identity: dict, allowances: dict) -> dict:
    operations = previous._operations(folder, identity, allowances)
    coordinates = {
        name for name in operations if re.fullmatch(r"coordinate_[0-9]{3}", name)
    }
    for path in (folder / "probes").glob("*"):
        if path.is_symlink() or not re.fullmatch(r"coordinate_[0-9]{3}", path.name):
            raise ValueError("invalid nested probe directory")
        if path.name not in allowances:
            raise ValueError("unknown coordinate probe directory")
        if path.name not in operations:
            raise ValueError("nested probes lack a parent reservation")
        coordinates.add(path.name)
    for name in sorted(coordinates):
        children = _probe_operations(folder, identity, int(name[-3:]))
        parent = first._value(operations.get(name))
        if parent and parent.get("primitive_receipts") != {
            key: public.content_sha256(value) for key, value in children.items()
        }:
            raise ValueError("nested probe receipts differ from coordinate")
    return operations


def _retained(record: dict, operations: dict) -> tuple[str, dict]:
    """An independently verified strict training improvement may replace source."""
    source = record["source_record"]["endpoint"]
    new = operations.get("endpoint_verification")
    qualified = first._value(operations.get("qualification")).get("passed")
    searched = first._value(operations.get("search"), partial=True).get("best")
    if (
        qualified
        and searched
        and first._verified(operations.get("start_verification"))
        and first._verified(new)
    ) and (source is None or new["value"]["training_nmse"] < source["training_nmse"]):
        return "continuation", new["value"]
    return "source", source


def _prefix_scores(data: dict, operations: dict) -> tuple[list, dict | None]:
    rows, scores = [], []
    portfolio = data["portfolio"]
    for i, row in enumerate(PublicSplit.model_validate(portfolio["training"]).rows):
        inference = first._value(operations.get(f"train_{i:02d}_inference"))
        forecast = first._value(operations.get(f"train_{i:02d}_forecast"))
        score = None
        if inference.get("fit_success") and forecast.get("forecast_success"):
            score = kernel.score_forecast(
                forecast,
                row,
                row.time[0] + portfolio["prefix"]["duration"],
                inference["training_scales"],
            )
            if score["available"]:
                scores.append(score)
        rows.append(
            {
                "trajectory_id": row.trajectory_id,
                "inference_success": bool(inference.get("fit_success")),
                "forecast_success": bool(forecast.get("forecast_success")),
                "score": score,
            }
        )
    if len(scores) != len(rows):
        return rows, None
    samples = sum(s["sample_count"] for s in scores)
    channels = scores[0]["per_target_normalized_mse"]
    per_target = {
        name: sum(
            s["per_target_normalized_mse"][name] * s["sample_count"] for s in scores
        )
        / samples
        for name in channels
    }
    return rows, {
        "normalized_mse": float(np.mean(list(per_target.values()))),
        "per_target_normalized_mse": per_target,
        "sample_count": samples,
    }


def _assemble(identity: dict, data: dict, operations: dict) -> dict:
    task = identity["task"]
    record = data["records"][task["index"]]
    origin, retained = _retained(record, operations)
    nonlinear = task["kind"] == "nonlinear_prerequisites"
    prefix_origin = "unavailable" if nonlinear else "source_recovery"
    rows, forecast = record["source_prefix_rows"], record["source_prefix_forecast"]
    if not nonlinear and origin == "continuation":
        prefix_origin = "new_prefix_inference"
        rows, forecast = _prefix_scores(data, operations)
    baseline = first._value(operations.get("baseline"))
    jacobian = first._value(operations.get("jacobian"))
    qualification = first._value(
        operations.get("qualification_summary" if nonlinear else "qualification")
    )
    search = first._value(operations.get("search"), partial=True)
    status = (
        "not_started"
        if not operations
        else "setup_unavailable"
        if not first._value(operations.get("setup"))
        else "prerequisite_unavailable"
        if nonlinear and not (baseline.get("passed") and jacobian.get("passed"))
        else "qualified_for_search"
        if nonlinear and qualification.get("search_qualified")
        else "derivatives_unqualified"
        if nonlinear
        else "precision_unqualified"
        if not qualification.get("passed")
        else "search_unavailable"
        if not search
        else "continued_verified"
        if origin == "continuation"
        else "continuation_no_improvement"
    )
    return {
        "status": "complete",
        "identity": identity,
        "candidate_id": task["candidate_id"],
        "kind": task["kind"],
        "continuation_status": status,
        "source_endpoint": record["source_record"]["endpoint"],
        "retained_origin": origin,
        "retained": retained,
        "training_reconstruction_nmse": (retained or {}).get("training_nmse"),
        "source_training_prefix_forecast": record["source_prefix_forecast"],
        "training_prefix_forecast": forecast if not nonlinear else None,
        "training_prefix_rows": rows if not nonlinear else [],
        "prefix_origin": prefix_origin,
        "qualification": qualification or None,
        "assessment": first._value(operations.get("diagnosis")).get("assessment"),
        "source_assessment": record["source_assessment"],
        "reductions": first._value(operations.get("setup")),
        "search_stop_reason": search.get("stop_reason"),
        "search_budget_exhausted": search.get("budget_exhausted"),
        "shared_parameter_search_attempted": "search" in operations,
        "operation_statuses": {n: v["status"] for n, v in operations.items()},
        "operation_receipts": {
            n: public.content_sha256(v) for n, v in operations.items()
        },
        "cost": journal._cost(operations),
        "test_data_opened": False,
        "live_llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
        "validation_status": "not_bundled_matching_frozen_split_required",
        "prefix_scores_used_for_selection": False,
    }


def run_task(root: Path, index: int) -> dict:
    """Run only unused operations; failed prerequisites never fan out into retries."""
    plan, data = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown continuation task index")
    task, record = plan["tasks"][index], data["records"][index]
    policy = Policy.model_validate(plan["policy"])
    portfolio = data["portfolio"]
    model = portfolio["models"][task["model_index"]]
    problem = PrefixFitProblem.model_validate(model["problem"])
    training = PublicSplit.model_validate(portfolio["training"])
    prefix_policy = PrefixPolicy.model_validate(portfolio["prefix"])
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = root / "results" / task["task_id"]
    allowances = _allowances(task, policy, len(training.rows))
    with public._lock(folder):
        operations = _operations(folder, identity, allowances)
        if (folder / "result.json").exists():
            result = _assemble(identity, data, operations)
            if result != read_seal(folder / "result.json"):
                raise ValueError("continuation terminal differs from evidence")
            return result
        compiled = None

        def oracles():
            nonlocal compiled
            if compiled is None:
                if task["kind"] == "nonlinear_prerequisites":
                    oracle = prerequisites.make_oracle(problem, training)
                    compiled = (oracle.original, oracle)
                else:
                    numerical_policy = first.Policy.model_validate(
                        data["original_policy"]
                    ).numerical
                    compiled = polish.make_oracle(
                        problem,
                        training,
                        numerical_policy,
                        record["source_record"]["endpoint"]["parameters"],
                    )
            return compiled

        def perform(name, function):
            operations[name] = journal._operation(
                folder, identity, name, allowances[name], function
            )
            return first._value(operations[name])

        def setup(save):
            original, oracle = oracles()
            if len(oracle.names) > task["coordinates"]:
                raise ValueError("continuation unexpectedly adds coordinates")
            value = {
                "original_coordinates": len(original.names),
                "optimizer_coordinates": len(oracle.names),
                "parameter_names": list(oracle.names),
                "audit": oracle.portfolio_audit,
            }
            save(value)
            return value

        if perform("setup", setup):
            original, oracle = oracles()
            if task["kind"] == "nonlinear_prerequisites":
                baseline = perform(
                    "baseline",
                    lambda save: prerequisites.import_baseline(
                        oracle,
                        record["replay_envelopes"]["Radau"],
                        record["replay_envelopes"]["BDF"],
                        30,
                        save,
                    ),
                )
                if baseline.get("passed"):
                    jacobian = perform(
                        "jacobian",
                        lambda save: prerequisites.prepare_jacobian(
                            oracle,
                            baseline,
                            policy.prerequisite_seconds,
                            save,
                        ),
                    )
                    if jacobian.get("passed"):
                        for j in range(task["coordinates"]):

                            def coordinate(save, j=j):
                                def probe_runner(key, vector, run):
                                    del vector
                                    if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", key):
                                        raise ValueError("unsafe primitive key")
                                    location = folder / "probes" / f"coordinate_{j:03d}"
                                    op = journal._operation(
                                        location, identity, key, 120, run
                                    )
                                    if op["status"] != "complete" or not op.get(
                                        "value"
                                    ):
                                        raise ValueError("probe unavailable")
                                    return op["value"]

                                value = prerequisites.qualify_coordinate(
                                    oracle,
                                    baseline,
                                    jacobian,
                                    j,
                                    policy.coordinate_seconds,
                                    save,
                                    probe_runner=probe_runner,
                                )
                                value["primitive_receipts"] = {
                                    n: public.content_sha256(v)
                                    for n, v in _probe_operations(
                                        folder, identity, j
                                    ).items()
                                }
                                save(value)
                                return value

                            perform(f"coordinate_{j:03d}", coordinate)
                        perform(
                            "qualification_summary",
                            lambda save: _saved(
                                prerequisites.aggregate_qualification(
                                    oracle,
                                    [
                                        first._value(
                                            operations.get(f"coordinate_{j:03d}")
                                        )
                                        for j in range(task["coordinates"])
                                        if first._value(
                                            operations.get(f"coordinate_{j:03d}")
                                        )
                                    ],
                                ),
                                save,
                            ),
                        )
            else:
                start = oracle.start
                perform(
                    "start_verification",
                    lambda save: numerical.verify_endpoint(
                        original,
                        oracle,
                        start,
                        policy.verification_seconds,
                        save,
                    ),
                )
                if first._verified(operations.get("start_verification")):
                    qualified = perform(
                        "qualification",
                        lambda save: polish.qualification(
                            original,
                            oracle,
                            start,
                            policy.qualification_seconds,
                            save,
                        ),
                    )
                    if qualified.get("passed"):
                        perform(
                            "search",
                            lambda save: polish.search(
                                oracle,
                                start,
                                policy.search_seconds,
                                policy.maximum_calls,
                                save,
                                qualification=qualified,
                            ),
                        )
                        search = first._value(operations.get("search"), partial=True)
                        if search.get("best"):
                            perform(
                                "endpoint_verification",
                                lambda save: numerical.verify_endpoint(
                                    original,
                                    oracle,
                                    search["best"]["parameters"],
                                    policy.verification_seconds,
                                    save,
                                ),
                            )
                origin, endpoint = _retained(record, operations)
                if endpoint:
                    effective = coordinates.to_effective(
                        original, oracle, endpoint["parameters"]
                    )
                    history = (
                        first._value(operations.get("search"), partial=True)
                        if origin == "continuation"
                        else {}
                    )
                    perform(
                        "diagnosis",
                        lambda save: numerical.diagnose(
                            original,
                            oracle,
                            effective,
                            endpoint,
                            history,
                            policy.diagnostic_seconds,
                            save,
                        ),
                    )
                    if origin == "continuation":
                        fit = first._fit_record(problem, training, original, endpoint)
                        for i, row in enumerate(training.rows):
                            prefix = kernel.make_prefix(row, prefix_policy)
                            inference = perform(
                                f"train_{i:02d}_inference",
                                lambda save, prefix=prefix: affine.infer_prefix(
                                    problem,
                                    fit,
                                    prefix,
                                    policy.inference,
                                    policy.inference_execution,
                                    save,
                                ),
                            )
                            if inference.get("fit_success"):
                                schedule = kernel.make_forecast_schedule(
                                    row, prefix_policy
                                )
                                perform(
                                    f"train_{i:02d}_forecast",
                                    lambda save,
                                    inference=inference,
                                    schedule=schedule: affine.forecast(
                                        problem,
                                        fit,
                                        inference,
                                        schedule,
                                        policy.inference,
                                        policy.forecast_execution,
                                        save,
                                    ),
                                )
        result = _assemble(identity, data, operations)
        seal(folder / "result.json", result)
        return result


def _saved(value, checkpoint):
    checkpoint(value)
    return value


def report(root: Path) -> dict:
    """Rebuild all results from receipts without invoking any numerical code."""
    plan, data = verify(root, runtime=False)
    policy = Policy.model_validate(plan["policy"])
    rows = []
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = root / "results" / task["task_id"]
        operations = _operations(
            folder, identity, _allowances(task, policy, plan["training_rows"])
        )
        value = _assemble(identity, data, operations)
        terminal = (folder / "result.json").exists()
        if terminal and read_seal(folder / "result.json") != value:
            raise ValueError("continuation report differs from terminal")
        rows.append(
            value
            | {
                "status": "complete"
                if terminal
                else "partial"
                if operations
                else "missing"
            }
        )
    recorded = sum(r["status"] == "complete" for r in rows)
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if recorded == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": recorded,
        "rows": rows,
        "status_counts": dict(Counter(r["continuation_status"] for r in rows)),
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitations": [
            "Shared search is limited to the two declared saved affine incumbents.",
            "Nonlinear qualification is not shared fitting or parameter confidence.",
            "Prefix scores use training-fitted shared coefficients; "
            "they are not independent validation.",
            "Nested probe costs are included in coordinate totals, not counted again.",
            "Interrupted allowances remain spent; source campaigns are never modified.",
        ],
    }
    with public._lock(root / "report"):
        public._write(root / "summary.json", summary)
        lines = [
            "# Construction portfolio continuation",
            "",
            f"{recorded}/{len(rows)} terminal tasks.",
            "",
            "| Candidate | Continuation status | Reconstruction NMSE | Prefix NMSE |",
            "| --- | --- | ---: | ---: |",
        ]
        lines += [
            f"| {r['candidate_id']} | {r['continuation_status']} | "
            f"{r['training_reconstruction_nmse']} | "
            f"{(r['training_prefix_forecast'] or {}).get('normalized_mse')} |"
            for r in rows
        ]
        (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def inspection(root: Path) -> Path:
    """Package provenance and bounded startup logs alongside all receipts."""
    report(root)
    paths = [
        root / n for n in ("inputs.json", "plan.json", "summary.json", "REPORT.md")
    ]
    paths += sorted((root / "results").rglob("*.json"))
    paths += sorted((root / "submissions").rglob("*.json")) + sorted(
        (root / "submissions").rglob("*.id")
    )
    paths += [
        p
        for p in (root / "launcher_execution.json", root / "submission_manifest.json")
        if p.exists()
    ]
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):

        def add(name, payload):
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(payload), 0o600
            archive.addfile(item, io.BytesIO(payload))

        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("inspection source escapes continuation root")
            add(str(path.relative_to(root)), path.read_bytes())
        logs = sorted(
            p for p in (root / "logs").glob("*") if p.suffix in {".out", ".err"}
        )
        manifest = {
            "maximum_files": 128,
            "maximum_bytes_per_file": 65536,
            "omitted_files": max(0, len(logs) - 128),
            "files": [],
        }
        for path in logs[-128:]:
            if (
                path.is_symlink()
                or not path.is_file()
                or not path.resolve().is_relative_to(root.resolve())
            ):
                raise ValueError("inspection log escapes continuation root")
            with path.open("rb") as stream:
                size = stream.seek(0, 2)
                stream.seek(max(0, size - 65536))
                payload = stream.read(65536)
            name = str(path.relative_to(root))
            add(name, payload)
            manifest["files"].append(
                {
                    "path": name,
                    "source_bytes_at_read": size,
                    "included_bytes": len(payload),
                    "truncated": size > len(payload),
                }
            )
        add("inspection_log_tails.json", json.dumps(manifest, sort_keys=True).encode())
    temporary = root / ".inspection.tar.gz.tmp"
    temporary.write_bytes(buffer.getvalue())
    destination = root / "inspection.tar.gz"
    temporary.replace(destination)
    return destination
