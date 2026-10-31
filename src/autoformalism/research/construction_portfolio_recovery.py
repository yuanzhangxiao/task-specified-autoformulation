"""Recover numerical qualification without replacing saved construction fits.

The source campaign is authenticated once and read only. New bounded operations
own new reservations; old spent budgets and terminal evidence are never reset.
"""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from collections import Counter
from copy import deepcopy
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
from autoformalism.fitting import prefix_portfolio_qualification as qualification
from autoformalism.fitting import prefix_verification as forecasting
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import construction_portfolio_fitting as previous
from autoformalism.research import construction_portfolio_intake as intake
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem, PrefixPolicy
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-portfolio-recovery-1"
SOURCE_PLAN = "8498d0d08631ed81543ef1f4a96eb3471b1017f295a06ccb72da7bbd02961725"


class Policy(StrictSchema):
    """Fixed recovery allowances; this campaign performs no shared search."""

    population: Literal["scaled-readout-23", "synthetic-smoke"] = "scaled-readout-23"
    replay_seconds: FiniteFloat = Field(default=180, gt=0, le=180)
    coordinate_seconds: FiniteFloat = Field(default=60, gt=0, le=60)
    diagnostic_seconds: FiniteFloat = Field(default=180, gt=0, le=180)
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


def _source_inputs(source: Path) -> dict:
    """Read and reconstruct source terminals without writing to the old root."""
    plan, portfolio = previous.verify(source, runtime=False)
    policy = previous.Policy.model_validate(plan["policy"])
    records = []
    for task, model in zip(plan["tasks"], portfolio["models"], strict=True):
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = source / "results" / task["task_id"]
        operations = previous._operations(
            folder, identity, policy, len(portfolio["training"]["rows"])
        )
        terminal = read_seal(folder / "result.json")
        if terminal != previous._assemble(identity, model, portfolio, operations):
            raise ValueError("source terminal differs from authenticated evidence")
        search = previous._value(operations.get("search"))
        compact = {
            key: deepcopy(search[key])
            for key in (
                "stop_reason",
                "budget_exhausted",
                "message",
                "status",
                "error",
                "best",
                "last_accepted_parameters",
            )
            if key in search
        }
        compact["accepted_steps"] = deepcopy(search.get("accepted_steps", [])[-3:])
        records.append(
            {
                "candidate_id": model["candidate_id"],
                "source_result_sha256": public.content_sha256(terminal),
                "source_search_sha256": public.content_sha256(operations.get("search")),
                "source_search_status": operations.get("search", {}).get("status"),
                "search": compact,
                "setup": terminal["setup"],
                "endpoint": terminal["retained"],
                "retained_origin": terminal["retained_origin"],
            }
        )
    return {
        "protocol": PROTOCOL,
        "source_plan_sha256": public.content_sha256(plan),
        "source_policy": plan["policy"],
        "portfolio": portfolio,
        "records": records,
        "test_data_opened": False,
    }


def _tasks(data: dict, policy: Policy) -> list[dict]:
    """Preserve the complete roster and the source-selected verified endpoints."""
    if data["protocol"] != PROTOCOL or data["test_data_opened"] is not False:
        raise ValueError("recovery source protocol differs")
    portfolio = data["portfolio"]
    count = 23 if policy.population == "scaled-readout-23" else len(portfolio["models"])
    intake.validate_inputs(portfolio, expected_count=count)
    if (
        policy.population == "scaled-readout-23"
        and data["source_plan_sha256"] != SOURCE_PLAN
    ):
        raise ValueError("recovery source is not the reviewed portfolio")
    if len(data["records"]) != count:
        raise ValueError("source records do not cover the complete roster")
    tasks = []
    for index, (model, record) in enumerate(
        zip(portfolio["models"], data["records"], strict=True)
    ):
        if model["candidate_id"] != record["candidate_id"]:
            raise ValueError("source model order differs")
        route = record["setup"]["routing"]["route"]
        coordinate_count = record["setup"]["optimizer_coordinates"]
        if type(coordinate_count) is not int or not 0 <= coordinate_count <= 128:
            raise ValueError("invalid recovery coordinate count")
        if route not in {"certified_affine", "original_ODE"}:
            raise ValueError("unrecognized source numerical route")
        if route == "certified_affine" and not previous._verified(
            {
                "status": "complete",
                "value": record["endpoint"] or {},
            }
        ):
            raise ValueError("affine recovery requires a saved verified endpoint")
        tasks.append(
            {
                "index": index,
                "task_id": model["candidate_id"],
                "candidate_id": model["candidate_id"],
                "kind": "affine_prefix"
                if route == "certified_affine"
                else "nonlinear_qualification",
                "coordinates": coordinate_count,
                "problem_sha256": model["fitting_problem_sha256"],
                "source_record_sha256": public.content_sha256(record),
            }
        )
    return tasks


def prepare(root: Path, source: Path, policy: Policy) -> dict:
    """Freeze a compact authenticated source snapshot, with no numerical work."""
    if root.resolve() == source.resolve() or root.resolve().is_relative_to(
        source.resolve()
    ):
        raise ValueError("recovery root must be separate from the source campaign")
    return prepare_inputs(root, _source_inputs(source), policy)


def prepare_inputs(root: Path, data: dict, policy: Policy) -> dict:
    """Freeze validated inputs; small synthetic campaigns share this boundary."""
    tasks = _tasks(data, policy)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": tasks,
        "training_rows": len(data["portfolio"]["training"]["rows"]),
        "shared_parameters_refitted": False,
        "validation_status": "not_bundled_matching_frozen_split_required",
        "test_data_opened": False,
        "live_llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
    }
    if reservation_ceiling(plan) > 6600:
        raise ValueError("recovery reservations exceed two-hour worker envelope")
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(tasks)}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Authenticate task boundaries; workers additionally require frozen code."""
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
                "shared_parameters_refitted",
                "test_data_opened",
                "automatic_revision",
                "automatic_promotion",
            )
        )
        or plan["live_llm_calls"] != 0
    ):
        raise ValueError("recovery plan identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("recovery source/runtime differs")
    return plan, data


def _allowances(task: dict, policy: Policy, rows: int) -> dict[str, float]:
    if task["kind"] == "affine_prefix":
        return {"setup": 120, "diagnosis": policy.diagnostic_seconds + 60} | {
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
    return {
        "setup": 120,
        "qualification_summary": 60,
        "replay_Radau": policy.replay_seconds + 60,
        "replay_BDF": policy.replay_seconds + 60,
    } | {
        f"coordinate_{i:03d}": policy.coordinate_seconds + 60
        for i in range(task["coordinates"])
    }


def reservation_ceiling(plan: dict) -> float:
    """Maximum per-model operation allowance, including checkpoint overhead."""
    policy = Policy.model_validate(plan["policy"])
    return max(
        (
            sum(_allowances(t, policy, plan["training_rows"]).values())
            for t in plan["tasks"]
        ),
        default=0,
    )


def _operations(folder: Path, identity: dict, allowances: dict) -> dict:
    """Read durable receipts without renewing interrupted operations."""
    operations = {}
    for directory in sorted((folder / "operations").glob("*")):
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or directory.name not in allowances
        ):
            raise ValueError("unknown recovery operation")
        started = directory / "started.json"
        if not started.exists():
            if any((directory / n).exists() for n in ("progress.json", "result.json")):
                raise ValueError("recovery operation lacks reservation")
            continue
        bound = {
            "campaign": identity,
            "stage": directory.name,
            "allowance_seconds": allowances[directory.name],
        }
        if read_seal(started) != bound:
            raise ValueError("recovery operation reservation differs")
        if (directory / "result.json").exists():
            operations[directory.name] = journal._read_operation(directory, bound)
        else:
            path = directory / "progress.json"
            progress = public._read(path) if path.exists() else {}
            operations[directory.name] = {
                "identity": bound,
                "status": "running_or_interrupted",
                "value": None,
                "observed_progress": progress,
                "progress_sha256": public.content_sha256(progress),
                "wall_seconds": None,
                "cpu_seconds": None,
                "budget_charge_seconds": allowances[directory.name],
                "accounting_complete": False,
                "budget_restarted": False,
            }
    return operations


def _replay_agreement(operations: dict) -> dict:
    """Require both independent state-only replays at the same supplied start."""
    left, right = [
        previous._value(operations.get(f"replay_{method}"))
        for method in ("Radau", "BDF")
    ]
    result = {
        "passed": False,
        "normalized_residual_discrepancy": None,
        "tolerance": 1e-7,
    }
    if not left.get("replay_completed") or not right.get("replay_completed"):
        return result
    for key in (
        "problem_sha256",
        "training_sha256",
        "parameters_sha256",
        "numerical_policy",
    ):
        if left.get(key) != right.get(key) or key not in left:
            raise ValueError("independent recovery replay identities differ")
    a, b = np.asarray(left["residual"], float), np.asarray(right["residual"], float)
    if (
        not a.size
        or a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("invalid independent recovery residuals")
    gap = float(np.max(np.abs(a - b)))
    return result | {"passed": gap <= 1e-7, "normalized_residual_discrepancy": gap}


def _assemble(identity: dict, data: dict, operations: dict) -> dict:
    """Report availability separately from optimization/scientific conclusions."""
    task = identity["task"]
    record = data["records"][task["index"]]
    portfolio = data["portfolio"]
    rows, scores = [], []
    for i, row in enumerate(PublicSplit.model_validate(portfolio["training"]).rows):
        inference = previous._value(operations.get(f"train_{i:02d}_inference"))
        forecast = previous._value(operations.get(f"train_{i:02d}_forecast"))
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
    aggregate = None
    if len(scores) == len(rows):
        total = sum(s["sample_count"] for s in scores)
        per_target = {
            name: sum(
                s["per_target_normalized_mse"][name] * s["sample_count"] for s in scores
            )
            / total
            for name in scores[0]["per_target_normalized_mse"]
        }
        aggregate = {
            "normalized_mse": float(np.mean(list(per_target.values()))),
            "per_target_normalized_mse": per_target,
            "sample_count": total,
        }
    checks = {
        name: previous._value(op)
        for name, op in operations.items()
        if name.startswith("coordinate_")
    }
    agreement = (
        _replay_agreement(operations)
        if task["kind"] == "nonlinear_qualification"
        else None
    )
    qualified = previous._value(operations.get("qualification_summary"))
    status = (
        "not_started"
        if not operations
        else "setup_unavailable"
        if not previous._value(operations.get("setup"))
        else "prefix_forecasts_complete"
        if task["kind"] == "affine_prefix" and aggregate is not None
        else "prefix_forecasts_incomplete"
        if task["kind"] == "affine_prefix"
        else "start_replay_unqualified"
        if not agreement["passed"]
        else "qualified_for_search"
        if qualified.get("search_qualified")
        else "derivatives_unqualified"
    )
    return {
        "status": "complete",
        "identity": identity,
        "candidate_id": task["candidate_id"],
        "kind": task["kind"],
        "recovery_status": status,
        "source_endpoint": record["endpoint"],
        "source_result_sha256": record["source_result_sha256"],
        "training_reconstruction_nmse": (record["endpoint"] or {}).get("training_nmse"),
        "training_prefix_available_rows": len(scores),
        "training_prefix_rows": rows,
        "training_prefix_forecast": aggregate,
        "assessment": previous._value(operations.get("diagnosis")).get("assessment"),
        "replay_agreement": agreement,
        "coordinate_checks": checks,
        "qualification": qualified or None,
        "operation_statuses": {name: op["status"] for name, op in operations.items()},
        "operation_receipts": {
            name: public.content_sha256(op) for name, op in operations.items()
        },
        "cost": journal._cost(operations),
        "shared_parameters_refitted": False,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
        "validation_status": "not_bundled_matching_frozen_split_required",
    }


def run_task(root: Path, index: int) -> dict:
    """Recover one original model under separately frozen finite allowances."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown recovery task index")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    policy = Policy.model_validate(plan["policy"])
    portfolio, record = data["portfolio"], data["records"][index]
    problem = PrefixFitProblem.model_validate(portfolio["models"][index]["problem"])
    training = PublicSplit.model_validate(portfolio["training"])
    prefix_policy = PrefixPolicy.model_validate(portfolio["prefix"])
    folder = root / "results" / task["task_id"]
    allowances = _allowances(task, policy, len(training.rows))
    with public._lock(folder):
        operations = _operations(folder, identity, allowances)
        if (folder / "result.json").exists():
            result = _assemble(identity, data, operations)
            if read_seal(folder / "result.json") != result:
                raise ValueError("recovery terminal differs from operation evidence")
            return result
        compiled = None

        def oracles():
            nonlocal compiled
            if compiled is None:
                if task["kind"] == "affine_prefix":
                    original_policy = previous.Policy.model_validate(
                        data["source_policy"]
                    )
                    compiled = numerical.make_oracle(
                        problem, training, original_policy.numerical
                    )
                else:
                    oracle = qualification.make_qualifier(problem, training)
                    compiled = (oracle.original, oracle)
                if len(compiled[1].names) != task["coordinates"]:
                    raise ValueError("recovery coordinate chart differs from source")
            return compiled

        def perform(name, work):
            operations[name] = journal._operation(
                folder, identity, name, allowances[name], work
            )
            return previous._value(operations[name])

        def setup(save):
            _original, reduced = oracles()
            expected = (
                "certified_affine"
                if task["kind"] == "affine_prefix"
                else "original_ODE"
            )
            if reduced.portfolio_audit["route"] != expected:
                raise ValueError("recovery propagation certificate differs")
            result = {
                "route": expected,
                "coordinates": len(reduced.names),
                "parameter_names": list(reduced.names),
            }
            save(result)
            return result

        if perform("setup", setup):
            original, oracle = oracles()
            if task["kind"] == "affine_prefix":
                endpoint = record["endpoint"]
                effective = coordinates.to_effective(
                    original, oracle, endpoint["parameters"]
                )
                search = (
                    record["search"]
                    if record["retained_origin"] == "endpoint_verification"
                    else {}
                )
                perform(
                    "diagnosis",
                    lambda save: numerical.diagnose(
                        original,
                        oracle,
                        effective,
                        endpoint,
                        search,
                        policy.diagnostic_seconds,
                        save,
                    ),
                )
                fit = previous._fit_record(problem, training, original, endpoint)
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
                        schedule = kernel.make_forecast_schedule(row, prefix_policy)
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
            else:
                for method in ("Radau", "BDF"):
                    perform(
                        f"replay_{method}",
                        lambda save, method=method: qualification.verify_start(
                            problem, training, method, policy.replay_seconds, save
                        ),
                    )
                if _replay_agreement(operations)["passed"]:
                    for j in range(task["coordinates"]):
                        perform(
                            f"coordinate_{j:03d}",
                            lambda save, j=j: qualification.qualify_coordinate(
                                oracle, j, policy.coordinate_seconds, save
                            ),
                        )

                    def aggregate(save):
                        checks = [
                            previous._value(operations.get(f"coordinate_{j:03d}"))
                            for j in range(task["coordinates"])
                        ]
                        result = qualification.aggregate_qualification(
                            oracle, [c for c in checks if c]
                        )
                        reference = np.asarray(
                            previous._value(operations["replay_Radau"])["residual"]
                        )
                        gaps = []
                        for check in checks:
                            if "base_residual" in check:
                                residual = np.asarray(check["base_residual"])
                                if residual.shape != reference.shape:
                                    raise ValueError(
                                        "coordinate/replay residual shapes differ"
                                    )
                                gaps.append(float(np.max(np.abs(reference - residual))))
                        gap = max(gaps, default=None)
                        result["lift_replay_discrepancy"] = gap
                        result["lift_replay_verified"] = (
                            gap is not None and np.isfinite(gap) and gap <= 1e-7
                        )
                        result["passed"] = bool(
                            result["passed"] and result["lift_replay_verified"]
                        )
                        result["search_qualified"] = bool(
                            result["search_qualified"]
                            and result["lift_replay_verified"]
                        )
                        save(result)
                        return result

                    perform("qualification_summary", aggregate)
        result = _assemble(identity, data, operations)
        seal(folder / "result.json", result)
        return result


def report(root: Path) -> dict:
    """Reconstruct reports without numerical work or implicit recovery retries."""
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
            raise ValueError("recovery terminal differs from operation evidence")
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
    completed = sum(row["status"] == "complete" for row in rows)
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if completed == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": completed,
        "rows": rows,
        "status_counts": dict(Counter(row["recovery_status"] for row in rows)),
        "shared_parameters_refitted": False,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitations": [
            "Shared parameters are unchanged; this is not another fitting search.",
            "Fresh-prefix training forecasts are not independent validation.",
            "Qualification is separate from stationarity, parameter confidence "
            "and science.",
            "Old and interrupted allowances are never renewed.",
            "Operation costs exclude queue, report, source authentication "
            "and resumed adapter construction.",
        ],
    }
    with public._lock(root / "report"):
        public._write(root / "summary.json", result)
        lines = [
            "# Construction portfolio numerical recovery",
            "",
            f"{completed}/{len(rows)} tasks recorded.",
            "",
            "| Candidate | Recovery status | Prefix rows | Forecast NMSE |",
            "| --- | --- | ---: | ---: |",
        ]
        lines += [
            f"| {r['candidate_id']} | {r['recovery_status']} | "
            f"{r['training_prefix_available_rows']} | "
            f"{(r['training_prefix_forecast'] or {}).get('normalized_mse')} |"
            for r in rows
        ]
        (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


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
                raise ValueError("inspection source escapes recovery root")
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
                raise ValueError("inspection log escapes recovery root")
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
