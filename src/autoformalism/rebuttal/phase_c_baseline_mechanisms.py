"""Public mechanism tests of the Phase C external baselines' sealed picks.

Each external baseline's development pick -- the model the frozen test
evaluation scores -- becomes the row the existing finite tests read, with the
same public rules, training operating points and settings as our own Phase C
construction baseline: ``mechanism_functional`` for the six named cells and
``basin_construction_assessment`` for the two detention cells. Nothing is
refit, reselected or run on held-out data, and no model is called.

One adapter per method:

* SINDy and PySR: the selected training equations, one per target, as an ODE
  (the train-only selection the frozen test scores, not the train+validation
  refit).
* LLM-ODE, under either prompt: the sealed system, as an ODE.
* D3: the native one-step map ``x[k+1] = x[k] + increment``. Supplied
  auxiliaries are clamped as in its native rollout, as in Phase B; the basin
  input conversions use the discrete identity ``area * coefficient = step``.
* LLM-SR: its picks are programs. A reviewed transcription into an ODE, with
  hidden states for the history a program reads (``llm_sr_transcriptions``),
  stands in for each pick, bound to its exact program text and coefficients.

Every model is read with the external-baseline grammar it was selected under.
A run that failed has no model: its predicates stay unresolved and it confirms
nothing, as an unavailable model of ours does. A run stopped by the service
is not a failure of the method; it must be rerun before it can be assessed.
"""

from __future__ import annotations

import hashlib
import json
import statistics as stats
from collections import Counter
from pathlib import Path

from autoformalism.baselines.d3_rollout import NativeMap
from autoformalism.baselines.models import BaselineDevelopmentResult
from autoformalism.expressions import ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal import llm_sr_transcriptions as transcriptions
from autoformalism.rebuttal import mechanism_functional as functional
from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate
from autoformalism.rebuttal.llm_sr_programs import observation_step
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import basin_construction_assessment as basin
from autoformalism.research import construction_baseline, phase_c_inputs
from autoformalism.schemas import (
    CandidateModel,
    InitialConditionSpec,
    ObservationMapping,
)
from autoformalism.schemas.public_fitting import PublicSplit
from autoformalism.staged_topology import content_hash

PROTOCOL = "phase-c-baseline-mechanisms-1"

#: Reported method -> the kind of run its sealed picks come from.
METHODS = {
    "sindy": "classical",
    "pysr": "classical",
    "d3": "d3",
    "llm_ode": "llm_ode",
    "llm_ode_published_prompt": "llm_ode",
    "llm_sr": "llm_sr",
}

#: Outcomes of the service, not the method. Such a task is rerun, never
#: reported as the method's failure, so it cannot be assessed yet.
INFRASTRUCTURE_STATUSES = frozenset({"endpoint_unavailable"})


def _file(path: Path) -> dict:
    """Where a source was read, and the digest of what was read."""
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _row(method: str, benchmark_id: str, tier: str, repetition: int) -> dict:
    return {
        "row_id": f"{method}:{benchmark_id}:{repetition}",
        "method": method,
        "benchmark_id": benchmark_id,
        "tier": tier,
        "repetition": repetition,
    }


def _failure(row: dict, sources: list[dict], reason: str, error=None) -> dict:
    return {
        **row,
        "status": "method_failure",
        "failure": reason,
        "error": error,
        "sources": sources,
    }


def _ready(row: dict, sources: list[dict], model: dict) -> dict:
    return {**row, "status": "ready", "sources": sources, "model": model}


def _model(
    candidate: CandidateModel,
    parameters: dict,
    context: ValidationContext,
    semantics: str,
    **extra,
) -> dict:
    """The row the existing tests read, in the external-baseline grammar."""
    return {
        "candidate": candidate.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
        "parameters": {name: float(value) for name, value in parameters.items()},
        "initials": {},
        "semantics": semantics,
        "grammar": "baseline",
        **extra,
    }


def _context(cell: dict) -> ValidationContext:
    return ValidationContext.model_validate(cell["context"])


# --- SINDy and PySR -----------------------------------------------------------


def classical_rows(root: Path, cells: dict) -> list[dict]:
    """SINDy and PySR: every frozen task, complete or not."""
    tasks = [
        json.loads(line)
        for line in (root / "frozen" / "task_plan.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows = []
    for task in tasks:
        method, name = task["method"], task["benchmark_id"]
        if method not in {"sindy", "pysr"} or name not in cells:
            raise ValueError(f"classical task {task['task_index']} is off the roster")
        row = _row(method, name, task["tier"], int(task["repetition"]))
        folder = (
            root / "runs" / method / f"{name}_{task['tier']}_seed{task['repetition']}"
        )
        result_path = folder / "result.json"
        if not result_path.is_file():
            status_path = folder / "run_status.json"
            status = json.loads(status_path.read_text())
            if status.get("status") in {"complete", "running", "pending"}:
                raise ValueError(f"{row['row_id']} has no result yet")
            rows.append(
                _failure(
                    row, [_file(status_path)], status["status"], status.get("error")
                )
            )
            continue
        result = BaselineDevelopmentResult.model_validate_json(result_path.read_text())
        if (result.method, result.benchmark_id, result.tier, result.seed) != (
            method,
            name,
            task["tier"],
            int(task["repetition"]),
        ):
            raise ValueError(f"{result_path} belongs to another task")
        if result.test_data_opened is not False:
            raise ValueError(f"{result_path} records opened test data")
        context = _context(cells[name])
        candidate = equation_candidate(method, result.equations, context)
        rows.append(
            _ready(
                row,
                [_file(result_path)],
                _model(candidate, {}, context, "continuous_time"),
            )
        )
    return rows


# --- the vendored campaigns: D3, LLM-ODE and LLM-SR ---------------------------


def d3_model(selection: BaselineDevelopmentResult, cell: dict) -> dict:
    """D3's native map: target increments, auxiliaries supplied as measured.

    Its saved auxiliary laws are never executed by its rollout, so, as in
    Phase B, they are kept aside and cannot supply a pathway. The sample
    interval is the map's time step.
    """
    context = _context(cell)
    payload = selection.selection_payload
    original = CandidateModel.model_validate(payload["candidate"])
    parameters = {name: float(value) for name, value in payload["parameters"].items()}
    NativeMap.build(
        original,
        parameters,
        (*context.targets, *context.auxiliaries),
        (*context.external_inputs, *context.fixed_covariates),
    )
    # The same effective model as Phase B's native rows (mechanism_audit_sources).
    effective = original.model_copy(
        update={
            "state_equations": tuple(
                e for e in original.state_equations if e.state in context.targets
            ),
            "observation_mappings": tuple(
                ObservationMapping(channel=t, expression=t) for t in context.targets
            ),
            "initial_conditions": tuple(
                InitialConditionSpec(state=t, scope="global", expression=t)
                for t in context.targets
            ),
        }
    )
    step = observation_step(
        public.unpack_split(PublicSplit.model_validate(cell["training"]))
    )
    return _model(
        effective,
        parameters,
        context,
        "native_increment",
        increment_step=step,
        native_auxiliary_equations_ignored=[
            e.model_dump(mode="json")
            for e in original.state_equations
            if e.state in context.auxiliaries
        ],
        native_boundary="observed_initial_targets_supplied_auxiliary_paths",
    )


def llm_ode_model(selection: BaselineDevelopmentResult, cell: dict) -> dict:
    """LLM-ODE's sealed system: constants are inlined, so no coefficients."""
    payload = selection.selection_payload
    candidate = CandidateModel.model_validate(payload["candidate"])
    return _model(
        candidate, payload.get("parameters", {}), _context(cell), "continuous_time"
    )


def llm_sr_model(
    selection: BaselineDevelopmentResult,
    cell: dict,
    index: int,
    reviewed: transcriptions.TranscriptionSet,
) -> dict:
    """LLM-SR's pick through its reviewed ODE transcription."""
    payload = selection.selection_payload
    if "programs" not in payload:
        # The reader turned every program into an equation of the present.
        return llm_ode_model(selection, cell)
    item = reviewed.for_task(index)
    if item is None:
        raise ValueError(f"LLM-SR task {index} has no reviewed transcription")
    if (item.benchmark_id, item.repetition) != (selection.benchmark_id, selection.seed):
        raise ValueError(f"LLM-SR task {index}'s transcription names another task")
    transcriptions.bind(item, payload["programs"])
    context = transcriptions.context_for(_context(cell))
    return _model(
        transcriptions.candidate(item, context),
        item.parameters,
        context,
        "continuous_time",
        transcription={
            "task": item.task,
            "rules": list(item.rules),
            "hidden_states": list(item.hidden_states),
            "status": reviewed.status,
        },
    )


def campaign_rows(
    method: str,
    run: Path,
    cells: dict,
    reviewed: transcriptions.TranscriptionSet | None = None,
) -> list[dict]:
    """One row per task of a sealed campaign; outcomes must be final."""
    plan = sealed_read(run / "plan.json")
    if method == "llm_sr":
        if reviewed is None:
            raise ValueError("LLM-SR needs its reviewed transcriptions")
        if reviewed.source_plan_sha256 != plan["artifact_sha256"]:
            raise ValueError("the transcriptions were made for another LLM-SR run")
    rows = []
    for source in plan["rows"]:
        index, name = int(source["index"]), source["benchmark_id"]
        if name not in cells:
            raise ValueError(f"{method} task {index} is off the roster")
        row = {**_row(method, name, source["tier"], int(source["repetition"])),
               "task": index}
        result_path = run / "results" / str(index) / "result.json"
        if not result_path.is_file():
            raise ValueError(f"{row['row_id']} has no outcome; finish it first")
        result = sealed_read(result_path)
        if (result["plan_sha256"], int(result["index"])) != (
            plan["artifact_sha256"],
            index,
        ):
            raise ValueError(f"{result_path} belongs to another plan or task")
        if result.get("test_data_opened") is not False:
            raise ValueError(f"{result_path} records opened test data")
        sources = [_file(run / "plan.json"), _file(result_path)]
        if result["status"] in INFRASTRUCTURE_STATUSES:
            raise ValueError(
                f"{row['row_id']} ended {result['status']}: an infrastructure "
                "failure is rerun, not assessed"
            )
        if result["status"] != "complete":
            rows.append(_failure(row, sources, result["status"], result.get("error")))
            continue
        selection_path = result_path.with_name("native-selection.json")
        sealed = sealed_read(selection_path)
        if sealed["plan_sha256"] != plan["artifact_sha256"]:
            raise ValueError(f"{selection_path} names another plan")
        sources.append(_file(selection_path))
        selection = BaselineDevelopmentResult.model_validate(sealed["selection"])
        if (selection.benchmark_id, selection.tier, selection.seed) != (
            name,
            source["tier"],
            int(source["repetition"]),
        ):
            raise ValueError(f"{selection_path} belongs to another task")
        programs = selection.selection_payload.get("programs") or {}
        if any(
            p.get("look_ahead", {}).get("reads_later_rows") for p in programs.values()
        ):
            # Decided 2026-10-15: a pick that reads later rows is LLM-SR's failure.
            rows.append(_failure(row, sources, "look_ahead"))
            continue
        cell = cells[name]
        if METHODS[method] == "d3":
            model = d3_model(selection, cell)
        elif METHODS[method] == "llm_ode":
            model = llm_ode_model(selection, cell)
        else:
            model = llm_sr_model(selection, cell, index, reviewed)
        rows.append(_ready(row, sources, model))
    return rows


# --- plan, run and report -----------------------------------------------------


def prepare(
    release: Path,
    construction_config: Path,
    sources: dict[str, Path],
    root: Path,
    llm_sr_transcriptions: Path | None = None,
) -> dict:
    """Freeze every pick of the named runs against the qualified public cells."""
    unknown = set(sources) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    if ("sindy" in sources or "pysr" in sources) and sources.get(
        "sindy"
    ) != sources.get("pysr"):
        raise ValueError("SINDy and PySR come from one classical run")
    config = construction_baseline.Config.model_validate_json(
        construction_config.read_text()
    ).model_dump(mode="json")
    receipt, cells = phase_c_inputs.qualified_public_cells(release, config)
    reviewed = (
        transcriptions.load(llm_sr_transcriptions) if llm_sr_transcriptions else None
    )
    rows: list[dict] = []
    if "sindy" in sources:
        rows.extend(classical_rows(sources["sindy"], cells))
    for method in ("d3", "llm_ode", "llm_ode_published_prompt", "llm_sr"):
        if method in sources:
            rows.extend(campaign_rows(method, sources[method], cells, reviewed))
    identities = [r["row_id"] for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("a method repeats a cell and repetition")
    for row in rows:
        if row["status"] == "ready":
            functional.Runtime(row["model"])  # compiles, or refuses, now
    with public._lock(root):
        return sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "settings": config["assessment"],
                "release_receipt_sha256": receipt,
                "construction_config": _file(construction_config),
                "llm_sr_transcriptions": (
                    {**_file(llm_sr_transcriptions), "status": reviewed.status}
                    if reviewed
                    else None
                ),
                "sources": {m: str(p.resolve()) for m, p in sorted(sources.items())},
                "cells": cells,
                "rows": rows,
                "test_data_opened": False,
                "private_reference_opened": False,
                "llm_calls": 0,
                "optimizer_calls": 0,
            },
        )


def assess(root: Path, index: int) -> dict:
    """Run the public tests on one frozen row; a saved result is reused."""
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] != PROTOCOL:
        raise ValueError("not a baseline mechanism plan")
    row = plan["rows"][index]
    directory = root / "results" / f"{index:03d}"
    identity = content_hash([plan["artifact_sha256"], index, row["row_id"]])
    with public._lock(directory / "assessment-lock"):
        path = directory / "assessment.json"
        if path.exists():
            saved = sealed_read(path)
            if saved["identity"] != identity:
                raise ValueError("assessment belongs to another plan or row")
            return saved
        cell = plan["cells"][row["benchmark_id"]]
        rules = cell["independent_rules"]
        if row["status"] == "ready":
            train = public.unpack_split(PublicSplit.model_validate(cell["training"]))
            settings = functional.Settings.model_validate(plan["settings"])
            findings = (
                basin.assess(row["model"], cell, train, settings, directory / "probes")
                if cell.get("assessment_policy") == basin.POLICY
                else functional.assess(
                    row["model"], rules, train, settings, directory / "probes"
                )
            )
            status = "assessed"
        else:
            status = "method_failure"
            findings = [
                {
                    "id": r["id"],
                    "public_requirement": r["public_requirement"],
                    "status": "unresolved",
                    "reason": f"no model: the run ended {row['failure']}",
                }
                for r in rules
            ]
        counts = Counter(f["status"] for f in findings)
        return sealed_write(
            path,
            {
                "identity": identity,
                "plan_sha256": plan["artifact_sha256"],
                "index": index,
                "row_id": row["row_id"],
                "status": status,
                "findings": findings,
                "counts": dict(counts),
                "requirements": len(rules),
                "confirmed": counts["pass"] / len(rules),
                "possible": (counts["pass"] + counts["unresolved"]) / len(rules),
                "all_confirmed": counts["pass"] == len(rules),
                "test_data_opened": False,
                "llm_calls": 0,
                "optimizer_calls": 0,
            },
        )


def _aggregate(rows: list[dict]) -> dict:
    """Case means over repetitions, then the macro mean and sample SD of cases."""
    if any(r["assessment_status"] == "pending" for r in rows):
        return {"status": "pending"}
    confirmed, possible = [], []
    for name in sorted({r["benchmark_id"] for r in rows}):
        case = [r for r in rows if r["benchmark_id"] == name]
        confirmed.append(stats.mean(r["confirmed"] for r in case))
        possible.append(stats.mean(r["possible"] for r in case))
    return {
        "status": "complete",
        "cases": len(confirmed),
        "runs": len(rows),
        "completed_runs": sum(r["status"] == "ready" for r in rows),
        "all_confirmed_runs": sum(bool(r["all_confirmed"]) for r in rows),
        "confirmed_mean": stats.mean(confirmed),
        "confirmed_sample_sd": stats.stdev(confirmed) if len(confirmed) > 1 else None,
        "possible_mean": stats.mean(possible),
    }


def report(root: Path) -> dict:
    """Every row, with per-method aggregates for all cells, named cells and basins."""
    plan = sealed_read(root / "plan.json")
    rows = []
    for index, row in enumerate(plan["rows"]):
        path = root / "results" / f"{index:03d}" / "assessment.json"
        result = sealed_read(path) if path.exists() else {}
        policy = plan["cells"][row["benchmark_id"]].get(
            "assessment_policy", "fitted-public-mechanism-tests-1"
        )
        rows.append(
            {
                key: row.get(key)
                for key in ("row_id", "method", "benchmark_id", "tier", "repetition")
            }
            | {
                "status": row["status"],
                "failure": row.get("failure"),
                "assessment_policy": policy,
                "assessment_status": result.get("status", "pending"),
                "confirmed": result.get("confirmed"),
                "possible": result.get("possible"),
                "all_confirmed": result.get("all_confirmed"),
                "counts": result.get("counts"),
                "findings": {
                    f["id"]: f["status"] for f in result.get("findings", [])
                },
            }
        )
    methods = {}
    for method in sorted({r["method"] for r in rows}):
        own = [r for r in rows if r["method"] == method]
        methods[method] = {
            "all_cells": _aggregate(own),
            **{
                policy: _aggregate([r for r in own if r["assessment_policy"] == policy])
                for policy in sorted({r["assessment_policy"] for r in own})
            },
        }
    value = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "rows": rows,
        "methods": methods,
        "llm_sr_transcriptions": plan["llm_sr_transcriptions"],
        "test_data_opened": False,
    }
    atomic_json(root / "summary.json", value)
    lines = [
        "# Phase C external baselines: public mechanism tests",
        "",
        "Fraction of public mechanism requirements confirmed per run, on the",
        "training operating points; unresolved requirements stay in the",
        "denominator and a failed run confirms none. No test data.",
        "",
        "| Method | Cells | Runs completed | All confirmed | Confirmed mean (SD) | "
        "Possible mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method, parts in methods.items():
        for scope, agg in parts.items():
            if agg["status"] != "complete":
                lines.append(f"| {method} | {scope} | pending | | | |")
                continue
            sd = agg["confirmed_sample_sd"]
            spread = "n/a" if sd is None else f"{sd:.3f}"
            lines.append(
                f"| {method} | {scope} | {agg['completed_runs']}/{agg['runs']} | "
                f"{agg['all_confirmed_runs']}/{agg['runs']} | "
                f"{agg['confirmed_mean']:.3f} ({spread}) | "
                f"{agg['possible_mean']:.3f} |"
            )
    lines += ["", "| Run | Status | Findings |", "|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['row_id']} | {r['assessment_status']}"
            f"{' (' + r['failure'] + ')' if r['failure'] else ''} | "
            + ", ".join(f"{k}: {v}" for k, v in r["findings"].items())
            + " |"
        )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return value
