"""Fit the admitted insulin revision from its authenticated nested endpoint.

The primary start has the historical search allowance. Two prespecified signed
starts are additional robustness experiments, never a best-of-three matched arm.
All fitting uses public training reconstruction, including its scoped roots.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import tarfile
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import construction_portfolio_fitting as first
from autoformalism.research import construction_portfolio_residual as retention
from autoformalism.research import construction_remaining_portfolio as operations
from autoformalism.research import continuation_report_cost as reporting
from autoformalism.research import meal_revision_metrics as metrics
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem, PrefixPolicy
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "insulin-revision-fitting-1"
INPUT_PROTOCOL = "insulin-revision-fitting-inputs-1"
SOURCE_PROBLEM = "76a4fadfbd2dd45a5c3ca3aad4b2f07e4d6c2f89f7564cd135ef4fc81a7c77af"
SOURCE_TRAINING = "169e2b72a1ba4145ac4ec5911875e9d7e064a525f7a1ace92cc2b151efe70795"
SOURCE_INPUTS = "7c6645cf688636d0ca0fba4985c09ad73e4560448d00254d29acc8dd1eb237ed"
SOURCE_MANIFEST = "d05bfb17edc7d462653412898d7eed728cf663053ac82f509f4ec53b69898e82"
DELAY = "k_insulin_delay"
# Max absolute inherited I/X diagonal, frozen from the pinned original endpoint.
# It is only a numerical start heuristic, never a domain/stability restriction.
FEEDBACK_UNIT = 0.021326192830597522
START_MULTIPLIER = 0.1
DOCUMENTS = (
    "admission",
    "baseline_endpoint",
    "fitting_contract",
    "nested_replay",
    "numerical",
    "parent_problem",
    "problem",
    "proposal",
    "provenance",
    "training",
    "warm_start_parameters",
)


class Policy(StrictSchema):
    """Matched primary search with separate qualification and replay reserves."""

    population: Literal["admitted-candidate-15", "synthetic-smoke"] = (
        "admitted-candidate-15"
    )
    search_seconds: FiniteFloat = Field(default=1800, gt=0, le=1800)
    maximum_calls: int = Field(default=2000, ge=1, le=2000)
    qualification_seconds: FiniteFloat = Field(default=300, gt=0, le=600)
    verification_seconds: FiniteFloat = Field(default=300, gt=0, le=300)
    checkpoint_interval_seconds: FiniteFloat = Field(default=30, gt=0, le=300)
    persistence_headroom_seconds: FiniteFloat = Field(default=900, ge=0, le=900)
    improvement_absolute_tolerance: FiniteFloat = Field(default=1e-10, ge=0)
    improvement_relative_tolerance: FiniteFloat = Field(default=1e-8, ge=0)


def load_intake(source: Path) -> dict:
    """Read only named, hash-authenticated portable documents; execute nothing."""
    manifest = json.loads((source / "manifest.json").read_text())
    expected = {name + ".json" for name in DOCUMENTS} | {"HANDOFF.md"}
    if (
        public.content_sha256(manifest) != SOURCE_MANIFEST
        or manifest.get("protocol") != "orion-insulin-revision-inputs-1"
        or set(manifest.get("files", {})) != expected
        or manifest.get("test_data_opened") is not False
        or manifest.get("new_model_fitted") is not False
    ):
        raise ValueError("reviewed insulin intake manifest differs")
    documents = {}
    for name in sorted(expected):
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("nonregular insulin intake member")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["files"][name]:
            raise ValueError(f"insulin intake file digest differs: {name}")
        if name.endswith(".json"):
            documents[name.removesuffix(".json")] = json.loads(raw)
    return {
        "protocol": INPUT_PROTOCOL,
        "manifest": manifest,
        "documents": documents,
        "test_data_opened": False,
        "validation_exported": False,
    }


def _validate(data: dict, policy: Policy) -> list[dict]:
    """Check model/data identity and preserve the complete nested original point."""
    if (
        set(data)
        != {
            "protocol",
            "manifest",
            "documents",
            "test_data_opened",
            "validation_exported",
        }
        or data["protocol"] != INPUT_PROTOCOL
        or set(data["documents"]) != set(DOCUMENTS)
        or data["test_data_opened"] is not False
        or data["validation_exported"] is not False
    ):
        raise ValueError("insulin intake schema or training boundary differs")
    production = policy.population == "admitted-candidate-15"
    if production and public.content_sha256(data) != SOURCE_INPUTS:
        raise ValueError("reviewed insulin source projection differs")
    d = data["documents"]
    problem = PrefixFitProblem.model_validate(d["problem"])
    parent = PrefixFitProblem.model_validate(d["parent_problem"])
    training = PublicSplit.model_validate(d["training"])
    kernel.NumericalPolicy.model_validate(d["numerical"])
    PrefixPolicy.model_validate(d["fitting_contract"]["prefix_policy"])
    model_hash, training_hash = (
        public.content_sha256(problem),
        public.content_sha256(training),
    )
    if (
        training.name != "train"
        or not training.rows
        or any(set(r.targets) != set(problem.context.targets) for r in training.rows)
        or d["fitting_contract"]["training_sha256"] != training_hash
        or d["admission"]["refit_handoff"]["problem_sha256"] != model_hash
        or d["nested_replay"]["problem_sha256"] != model_hash
        or d["nested_replay"]["training_sha256"] != training_hash
        or d["nested_replay"]["warm_start_sha256"]
        != public.content_sha256(d["warm_start_parameters"])
        or d["nested_replay"]["old_endpoint_sha256"]
        != public.content_sha256(d["baseline_endpoint"])
        or not operations._finite_endpoint(d["baseline_endpoint"])
    ):
        raise ValueError(
            "admitted model, warm endpoint, or public training identity differs"
        )
    warm = d["warm_start_parameters"]
    if (
        warm.get(DELAY) != 0.0
        or {k: v for k, v in warm.items() if k != DELAY}
        != d["baseline_endpoint"]["parameters"]
        or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in warm.values()
        )
        or parent.graph != problem.graph
    ):
        raise ValueError("nested endpoint or initialization graph differs")
    if production and (
        model_hash != SOURCE_PROBLEM
        or training_hash != SOURCE_TRAINING
        or len(warm) != 52
        or sum(n.startswith("trajectory:") for n in warm) != 32
    ):
        raise ValueError("reviewed model/training coordinate roster differs")
    return [
        {
            "index": i,
            "task_id": label,
            "start_label": label,
            "role": "primary_matched_budget" if i == 0 else "additional_robustness",
            "delay_start": value,
            "feedback_unit": FEEDBACK_UNIT,
            "start_multiplier": START_MULTIPLIER,
            "start_rule": "signed_fraction_of_pinned_inherited_I_X_diagonal_scale",
            "problem_sha256": model_hash,
            "warm_start_sha256": public.content_sha256(warm),
        }
        for i, (label, value) in enumerate(
            (
                ("nested-zero", 0.0),
                ("signed-negative", -START_MULTIPLIER * FEEDBACK_UNIT),
                ("signed-positive", START_MULTIPLIER * FEEDBACK_UNIT),
            )
        )
    ]


def prepare(root: Path, source: Path, policy: Policy) -> dict:
    """Freeze a fresh fitting campaign without numerical work on the login node."""
    if root.resolve().is_relative_to(
        source.resolve()
    ) or source.resolve().is_relative_to(root.resolve()):
        raise ValueError("campaign root must be separate from source intake")
    return prepare_inputs(root, load_intake(source), policy)


def prepare_inputs(root: Path, data: dict, policy: Policy) -> dict:
    """Seal exact inputs, implementation identity and independently labelled tasks."""
    plan = _plan(data, policy)
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def _plan(data: dict, policy: Policy) -> dict:
    return {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": _validate(data, policy),
        "test_data_opened": False,
        "live_llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
        "objective": "original_full_training_reconstruction",
        "selection_data": "public_training_only",
        "training_prefix_reestimated": False,
        "prefix_forecast_status": "deferred_separate_frozen_parameter_diagnostic",
        "matched_comparison": "primary_nested_zero_only",
        "robustness_budget": "two_additional_independently_budgeted_starts",
    }


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Rebuild all declared boundaries; optional offline reading never fits."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    expected = _plan(data, Policy.model_validate(plan["policy"]))
    if not runtime:
        expected.update(source_sha256=plan["source_sha256"], runtime=plan["runtime"])
    if plan != expected:
        raise ValueError("insulin fitting plan/source/runtime identity differs")
    return plan, data


def _allowances(policy: Policy) -> dict[str, float]:
    p = policy
    return {
        "setup": 180,
        "nesting": p.verification_seconds + 60,
        "qualification": p.qualification_seconds + 120,
        "profiled_start_verification": p.verification_seconds + 60,
        "search": p.search_seconds + p.persistence_headroom_seconds + 60,
        "endpoint_verification": p.verification_seconds + 60,
        "baseline_metrics": p.verification_seconds + 60,
        "metrics": p.verification_seconds + 60,
    }


def reservation_ceiling(policy: Policy) -> float:
    """Maximum per-worker elapsed reservation, including persistence and replays."""
    return sum(_allowances(policy).values())


def _source_endpoint(data: dict) -> dict:
    d = data["documents"]
    return d["baseline_endpoint"] | {
        "parameters": d["warm_start_parameters"],
        "model_sha256": public.content_sha256(d["problem"]),
        "source": "authenticated_original_equation_nested_replay",
        "nested_replay_sha256": public.content_sha256(d["nested_replay"]),
    }


def _retained(data: dict, ops: dict, policy: Policy) -> tuple[str, dict, dict]:
    """A failed search can never erase the authenticated nested baseline."""
    origin, incumbent, comparisons = (
        "source_nested_baseline",
        _source_endpoint(data),
        {},
    )
    for name in ("nesting", "profiled_start_verification", "endpoint_verification"):
        if not first._verified(ops.get(name)):
            continue
        candidate = first._value(ops[name])
        if not operations._finite_endpoint(candidate):
            continue
        decision = retention.compare_endpoints(incumbent, candidate, policy)
        comparisons[name] = decision
        if decision["classification"] == "meaningful_improvement":
            origin, incumbent = name, candidate
    return origin, incumbent, comparisons


def compare_refit_replay(
    data: dict, baseline: dict, revised: dict, endpoint: dict
) -> dict:
    """Compare independently replayed metrics under the portable admission hashes.

    This is the portable counterpart of the proposer script's helper: it uses
    the same summarization/comparison implementation but needs no provider files.
    """
    d = data["documents"]
    training_hash, model_hash = (
        public.content_sha256(d["training"]),
        public.content_sha256(d["problem"]),
    )
    for value in (baseline, revised):
        if (
            value.get("training_sha256") != training_hash
            or value.get("problem_sha256") != model_hash
        ):
            raise ValueError("comparison metrics belong to different training/model")
    if (
        d["admission"]["refit_handoff"]["problem_sha256"] != model_hash
        or endpoint.get("original_equations_verified") is not True
        or not math.isclose(
            revised["training_nmse"],
            endpoint["training_nmse"],
            rel_tol=1e-8,
            abs_tol=1e-10,
        )
        or not math.isclose(
            baseline["training_nmse"],
            d["baseline_endpoint"]["training_nmse"],
            rel_tol=1e-8,
            abs_tol=1e-10,
        )
    ):
        raise ValueError("comparison and admitted original replay differ")
    return {
        "status": "scored_training_replay",
        "refit_problem_sha256": model_hash,
        "training_sha256": training_hash,
        "fitted_parent_reference": d["provenance"]["historical_fit"],
        "metrics": revised,
        "comparison": metrics.compare(baseline, revised),
        "report": metrics.render_comparison(baseline, revised),
        "automatic_followup": False,
        "scope": "same_comparison_functions_with_authenticated_portable_admission",
    }


def _assemble(identity: dict, data: dict, ops: dict, policy: Policy) -> dict:
    origin, endpoint, comparisons = _retained(data, ops, policy)
    search = first._value(ops.get("search"), partial=True)
    qualified_stop = bool(
        ops.get("search", {}).get("status") == "complete"
        and search.get("search_calls_completed", 0) > 0
        and search.get("stop_reason") == "optimizer_stopped"
        and search.get("optimizer_success") is True
        and search.get("budget_exhausted") is False
        and (search.get("optimizer_terminal") or {}).get("feasible_evaluation_verified")
        is True
    )
    baseline, fitted = (
        first._value(ops.get("baseline_metrics")),
        first._value(ops.get("metrics")),
    )
    comparison = (
        compare_refit_replay(data, baseline, fitted, endpoint)
        if baseline and fitted
        else None
    )
    search_status = (
        "not_started"
        if not ops
        else "setup_unavailable"
        if not first._value(ops.get("setup")).get("passed")
        else "nesting_unqualified"
        if not first._verified(ops.get("nesting"))
        else "profile_unqualified"
        if not first._value(ops.get("qualification")).get("passed")
        else "profiled_start_unverified"
        if not first._verified(ops.get("profiled_start_verification"))
        else "search_not_started"
        if "search" not in ops
        else "search_interrupted_or_unavailable"
        if ops["search"]["status"] != "complete"
        else "search_budget_exhausted"
        if search.get("budget_exhausted")
        else "qualified_optimizer_stop"
        if qualified_stop
        else "optimizer_stop_unqualified"
    )
    task = identity["task"]
    return {
        "status": "complete",
        "identity": identity,
        "start_label": task["start_label"],
        "role": task["role"],
        "search_status": search_status,
        "retained": endpoint,
        "retained_origin": origin,
        "source_anchor_nmse": data["documents"]["baseline_endpoint"]["training_nmse"],
        "training_reconstruction_nmse": endpoint["training_nmse"],
        "per_target_normalized_mse": endpoint.get("per_target_normalized_mse"),
        "endpoint_comparisons": comparisons,
        "search": search,
        "qualified_search_stop": qualified_stop,
        "comparison": comparison,
        "metrics": fitted or None,
        "effective_kappa": (fitted.get("diagnostics") or {}).get("effective_kappa"),
        "baseline_metrics": baseline or None,
        "setup": first._value(ops.get("setup")),
        "qualification": first._value(ops.get("qualification")),
        "operation_statuses": {k: v["status"] for k, v in sorted(ops.items())},
        "operation_errors": {
            k: v["error"] for k, v in sorted(ops.items()) if v.get("error")
        },
        "operation_receipts": {
            k: public.content_sha256(v) for k, v in sorted(ops.items())
        },
        "cost": reporting.canonical_cost(ops),
        "cost_exclusions": [
            "Historical fit, source/runtime hashing, reporting, queue time, "
            "adapter rebuild on resume"
        ],
        "optimization_converged": False,
        "global_minimum_certified": False,
        "parameter_confidence_established": False,
        "scientific_correctness_certified": False,
        "prefix_forecast_status": "deferred_separate_frozen_parameter_diagnostic",
        "training_prefix_reestimated": False,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "automatic_revision": False,
    }


class _Run:
    """Reserve each numerical phase once and verify checkpoints before retention."""

    def __init__(self, root: Path, plan: dict, data: dict, index: int):
        self.task, self.data = plan["tasks"][index], data
        self.policy = Policy.model_validate(plan["policy"])
        self.identity = {"plan_sha256": public.content_sha256(plan), "task": self.task}
        self.folder = root / "results" / self.task["task_id"]
        self.allowances = _allowances(self.policy)
        self.ops = operations._operations(self.folder, self.identity, self.allowances)
        self.fit = None

    def perform(self, name, work):
        self.ops[name] = journal._operation(
            self.folder, self.identity, name, self.allowances[name], work
        )
        return first._value(self.ops[name])

    def setup(self, save):
        from autoformalism.fitting.insulin_revision_fit import InsulinRevisionFit

        d = self.data["documents"]
        self.fit = InsulinRevisionFit(
            PrefixFitProblem.model_validate(d["problem"]),
            PrefixFitProblem.model_validate(d["parent_problem"]),
            PublicSplit.model_validate(d["training"]),
            kernel.NumericalPolicy.model_validate(d["numerical"]),
            d["warm_start_parameters"],
            delay_start=self.task["delay_start"],
        )
        result = self.fit.setup_audit()
        if self.policy.population == "admitted-candidate-15" and not math.isclose(
            result["feedback_coordinate_unit"],
            self.task["feedback_unit"],
            rel_tol=1e-12,
            abs_tol=1e-14,
        ):
            raise ValueError("inherited feedback scale differs from frozen start rule")
        save(result)
        return result

    def execute(self):
        p, d = self.policy, self.data["documents"]
        setup = self.perform("setup", self.setup)
        if not setup.get("passed"):
            return
        if self.fit is None and public.content_sha256(
            self.setup(lambda _: None)
        ) != public.content_sha256(setup):
            raise ValueError("resumed revision fitting setup differs")
        fit = self.fit
        nesting = self.perform(
            "nesting",
            lambda save: fit.verify_start(
                d["baseline_endpoint"]["training_nmse"], p.verification_seconds
            ),
        )
        if not nesting.get("passed") or not first._verified(self.ops["nesting"]):
            return
        # These diagnostics describe the same model at the nested inherited point.
        self.perform(
            "baseline_metrics",
            lambda save: fit.metrics(nesting, p.verification_seconds),
        )
        qualified = self.perform(
            "qualification", lambda save: fit.qualify(p.qualification_seconds, save)
        )
        if qualified.get("passed"):
            self.perform(
                "profiled_start_verification",
                lambda save: fit.verify(
                    qualified["parameters"],
                    qualified["training_nmse"],
                    p.verification_seconds,
                    save,
                ),
            )
            if first._verified(self.ops["profiled_start_verification"]):
                self.perform("search", lambda save: fit.search(p, save))
                best = first._value(self.ops["search"], partial=True).get("best")
                if best and isinstance(best.get("parameters"), dict):
                    self.perform(
                        "endpoint_verification",
                        lambda save: fit.verify(
                            best["parameters"],
                            best["training_nmse"],
                            p.verification_seconds,
                            save,
                        ),
                    )
        _, endpoint, _ = _retained(self.data, self.ops, p)
        self.perform(
            "metrics", lambda save: fit.metrics(endpoint, p.verification_seconds)
        )


def run_task(root: Path, index: int) -> dict:
    """Run only unspent stages; a terminal resume is strictly read-only."""
    plan, data = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown insulin revision task index")
    folder = root / "results" / plan["tasks"][index]["task_id"]
    with public._lock(folder):
        run = _Run(root, plan, data, index)
        if (folder / "result.json").exists():
            saved = read_seal(folder / "result.json")
            reporting.reconcile(
                saved, _assemble(run.identity, data, run.ops, run.policy), run.ops
            )
            return saved
        run.execute()
        result = _assemble(run.identity, data, run.ops, run.policy)
        seal(folder / "result.json", result)
        return result


def report(root: Path) -> dict:
    """Read receipts only; separate the primary matched fit and extra starts."""
    plan, data = verify(root, runtime=False)
    policy = Policy.model_validate(plan["policy"])
    rows, reconciliations = [], {}
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = root / "results" / task["task_id"]
        ops = operations._operations(folder, identity, _allowances(policy))
        row = _assemble(identity, data, ops, policy)
        terminal = (folder / "result.json").is_file()
        if terminal:
            reconciliations[task["task_id"]] = reporting.reconcile(
                read_seal(folder / "result.json"), row, ops
            )
        rows.append(
            row
            | {"status": "complete" if terminal else "partial" if ops else "missing"}
        )
    recorded = sum(r["status"] == "complete" for r in rows)
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if recorded == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": recorded,
        "rows": rows,
        "search_status_counts": dict(Counter(r["search_status"] for r in rows)),
        "primary_task": plan["tasks"][0]["task_id"],
        "matched_comparison": "primary_nested_zero_only",
        "terminal_reconciliations": reconciliations,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitations": [
            "Primary fitting has 1800 seconds/2000 calls at default; robustness "
            "starts have separate additional allowances.",
            "Full training reconstruction estimates 32 trajectory roots; "
            "fresh prefix forecasting is deferred.",
            "Finite optimizer termination does not certify stationarity, global "
            "optimality, coefficient identifiability or physiological validity.",
            "The positive meal-rate chart searches k_abs>0; the exact-zero "
            "meal-decay branch is not searched in this milestone.",
            "The admitted real signed coupling is not restricted by the "
            "conditional stability threshold.",
            "Original-equation replay guards retained endpoints; an interrupted "
            "search never renews its numerical allowance.",
            "Stability and positivity diagnostics concern the fitted training "
            "endpoint, not all admissible interventions.",
        ],
    }
    with public._lock(root / "report"):
        public._write(root / "summary.json", summary)
        lines = [
            "# Admitted insulin revision fitting",
            "",
            f"{recorded}/{len(rows)} terminal attempts.",
            "",
            "| Start | Role | Search status | Retained training NMSE |",
            "| --- | --- | --- | ---: |",
        ]
        lines += [
            f"| {r['start_label']} | {r['role']} | {r['search_status']} | "
            f"{r['training_reconstruction_nmse']:.12g} |"
            for r in rows
        ]
        lines += ["", *summary["limitations"]]
        (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def inspection(root: Path) -> Path:
    """Package compact inputs, complete numerical receipts and bounded log tails."""
    report(root)
    paths = [
        root / n for n in ("plan.json", "inputs.json", "summary.json", "REPORT.md")
    ]
    paths += sorted((root / "results").rglob("*.json"))
    paths += sorted((root / "submissions").rglob("*.json"))
    paths += sorted((root / "submissions").rglob("*.id"))
    if (root / "launcher_execution.json").exists():
        paths.append(root / "launcher_execution.json")
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w") as archive,
    ):
        for path in sorted(set(paths)):
            if path.is_symlink() or not path.is_file():
                raise ValueError("nonregular inspection input")
            info = archive.gettarinfo(str(path), arcname=str(path.relative_to(root)))
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)
        logs = {}
        for path in sorted((root / "logs").glob("*")):
            if path.is_file() and not path.is_symlink():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 16384))
                    logs[path.name] = stream.read().decode(errors="replace")
        payload = json.dumps(logs, sort_keys=True).encode()
        info = tarfile.TarInfo("inspection_log_tails.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    result = root / "inspection.tar.gz"
    result.write_bytes(buffer.getvalue())
    return result
