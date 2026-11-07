"""Matched warm continuations with a separately labelled insulin domain control."""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import insulin_revision_fitting as old
from autoformalism.schemas.base import FiniteFloat

PROTOCOL = "insulin-positive-continuation-1"
INPUT_PROTOCOL = "insulin-positive-continuation-inputs-1"
SOURCE_INPUTS = "34e3ff4a6696775fc16adffa82d8b0bfb4db6ce7aa26af6c229b109547701558"
SOURCE_PLAN = "d187408a2a6b3453810da22b4b0a023b41fda3eb5636da9266f25567c430f851"
ARMS = ("unconstrained", "positive")


class Policy(old.Policy):
    """Equal additional search budgets; separate replay and diagnostic reserves."""

    diagnostic_seconds: FiniteFloat = Field(default=120, gt=0, le=300)


def _validate(data: dict, policy: Policy) -> list[dict]:
    if (
        set(data)
        != {
            "protocol",
            "source_inputs",
            "source_plan",
            "endpoints",
            "validation_or_test_access",
        }
        or data["protocol"] != INPUT_PROTOCOL
        or data["validation_or_test_access"] is not False
    ):
        raise ValueError("continuation intake schema or training boundary differs")
    if policy.population == "admitted-candidate-15" and (
        public.content_sha256(data) != SOURCE_INPUTS
        or public.content_sha256(data["source_plan"]) != SOURCE_PLAN
    ):
        raise ValueError("reviewed continuation source differs")
    source, plan = data["source_inputs"], data["source_plan"]
    old._validate(source, old.Policy(population=policy.population))
    if plan["inputs_sha256"] != public.content_sha256(source):
        raise ValueError("source plan and source input differ")
    expected = [t["task_id"] for t in plan["tasks"]]
    if [e["task_id"] for e in data["endpoints"]] != expected:
        raise ValueError("saved endpoint roster differs")
    tasks = []
    for item in data["endpoints"]:
        endpoint = item["endpoint"]
        if (
            not old.operations._finite_endpoint(endpoint)
            or endpoint.get("original_equations_verified") is not True
            or set(endpoint["parameters"])
            != set(source["documents"]["warm_start_parameters"])
            or any(not math.isfinite(v) for v in endpoint["parameters"].values())
        ):
            raise ValueError("unverified, incomplete or nonfinite source endpoint")
        for arm in ARMS:
            tasks.append(
                {
                    "index": len(tasks),
                    "task_id": item["task_id"] + "-" + arm,
                    "start_label": item["task_id"],
                    "arm": arm,
                    "role": "matched_additional_budget",
                    "source_endpoint_sha256": public.content_sha256(endpoint),
                    "source_result_sha256": item["source_result_sha256"],
                }
            )
    return tasks


def _plan(data: dict, policy: Policy) -> dict:
    return {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": _validate(data, policy),
        "selection_data": "original_training_samples_only",
        "objective": "same_normalized_least_squares_both_arms",
        "domain_control": (
            "positive arm: I>=0 at nodes and midpoints; finer replay gate"
        ),
        "source_reservations_renewed": False,
        "validation_or_test_access": False,
        "automatic_revision": False,
    }


def prepare(root: Path, source: Path, policy: Policy) -> dict:
    if root.resolve().is_relative_to(
        source.resolve()
    ) or source.resolve().is_relative_to(root.resolve()):
        raise ValueError("use a fresh root separate from the source intake")
    path = source / "continuation_inputs.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular continuation intake required")
    return prepare_inputs(root, read_seal(path), policy)


def prepare_inputs(root: Path, data: dict, policy: Policy) -> dict:
    plan = _plan(data, policy)
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    expected = _plan(data, Policy.model_validate(plan["policy"]))
    if not runtime:
        expected.update(source_sha256=plan["source_sha256"], runtime=plan["runtime"])
    if plan != expected:
        raise ValueError("continuation plan/source/runtime identity differs")
    return plan, data


def _allowances(p: Policy) -> dict[str, float]:
    return {
        "setup": 180,
        "source_verification": p.verification_seconds + 60,
        "fallback_verification": p.verification_seconds + 60,
        "qualification": p.qualification_seconds + 120,
        "profiled_start_verification": p.verification_seconds + 60,
        "search": p.search_seconds + p.persistence_headroom_seconds + 60,
        "endpoint_verification": p.verification_seconds + 60,
        "local_audit": p.diagnostic_seconds + 60,
        "probe_verification": p.verification_seconds + 60,
        "metrics": p.verification_seconds + 60,
    }


def reservation_ceiling(policy: Policy) -> float:
    return sum(_allowances(policy).values())


def _source(data: dict, task: dict) -> dict:
    return next(
        e["endpoint"] for e in data["endpoints"] if e["task_id"] == task["start_label"]
    )


def _retained(data: dict, task: dict, ops: dict, policy: Policy):
    """An infeasible source survives as a reference, never as a positive-arm fit."""
    origin, endpoint, comparisons = None, None, {}
    for name in (
        "source_verification",
        "fallback_verification",
        "profiled_start_verification",
        "endpoint_verification",
        "probe_verification",
    ):
        if not old.first._verified(ops.get(name)):
            continue
        candidate = old.first._value(ops[name])
        if (
            not old.operations._finite_endpoint(candidate)
            or candidate.get("eligible_for_arm") is not True
        ):
            continue
        if endpoint is None:
            origin, endpoint = name, candidate
        else:
            decision = old.retention.compare_endpoints(endpoint, candidate, policy)
            comparisons[name] = decision
            if decision["classification"] == "meaningful_improvement":
                origin, endpoint = name, candidate
    return origin, endpoint, comparisons


def _assemble(identity: dict, data: dict, ops: dict, policy: Policy) -> dict:
    task = identity["task"]
    origin, endpoint, comparisons = _retained(data, task, ops, policy)
    search = old.first._value(ops.get("search"), partial=True)
    audit = old.first._value(ops.get("local_audit"), partial=True)
    qualified_stop = bool(
        ops.get("search", {}).get("status") == "complete"
        and search.get("search_calls_completed", 0) > 0
        and search.get("stop_reason") == "optimizer_stopped"
        and search.get("optimizer_success") is True
        and search.get("budget_exhausted") is False
        and (search.get("optimizer_terminal") or {}).get("feasible_evaluation_verified")
        is True
    )
    metrics = old.first._value(ops.get("metrics"))
    source = _source(data, task)
    search_status = (
        "qualified_optimizer_stop"
        if qualified_stop
        else "search_budget_exhausted"
        if search.get("budget_exhausted")
        else "search_interrupted_or_unqualified"
        if "search" in ops
        else "not_started"
    )
    return {
        "status": "complete",
        "identity": identity,
        "start_label": task["start_label"],
        "arm": task["arm"],
        "role": task["role"],
        "retained": endpoint,
        "retained_origin": origin,
        "endpoint_comparisons": comparisons,
        "source_reference": source,
        "source_anchor_nmse": source["training_nmse"],
        "training_reconstruction_nmse": endpoint["training_nmse"] if endpoint else None,
        "per_target_normalized_mse": endpoint.get("per_target_normalized_mse")
        if endpoint
        else None,
        "domain_admissible": endpoint.get("domain_admissible") if endpoint else None,
        "source_domain_audit": old.first._value(ops.get("source_verification")).get(
            "concentration_audit"
        ),
        "concentration_audit": endpoint.get("concentration_audit")
        if endpoint
        else None,
        "metrics": metrics or None,
        "effective_kappa": metrics.get("diagnostics", {}).get("effective_kappa"),
        "setup": old.first._value(ops.get("setup")),
        "qualification": old.first._value(ops.get("qualification")),
        "search": search,
        "search_status": search_status,
        "qualified_search_stop": qualified_stop,
        "local_audit": audit,
        "local_audit_matches_retained": bool(
            endpoint
            and audit.get("audited_parameters_sha256")
            == public.content_sha256(endpoint["parameters"])
        ),
        "optimization_converged": False,
        "global_minimum_certified": False,
        "continuous_positivity_certified": False,
        "scientific_correctness_certified": False,
        "fit_adequacy": "descent_detected"
        if audit.get("resolved_descent")
        else "no_descent_in_sampled_directions"
        if audit.get("status") == "complete" and audit.get("all_probes_available")
        else "unresolved",
        "operation_statuses": {k: v["status"] for k, v in sorted(ops.items())},
        "operation_errors": {
            k: v["error"] for k, v in sorted(ops.items()) if v.get("error")
        },
        "operation_receipts": {
            k: public.content_sha256(v) for k, v in sorted(ops.items())
        },
        "cost": old.reporting.canonical_cost(ops),
        "phase_costs": {
            k: {
                field: v.get(field)
                for field in ("wall_seconds", "cpu_seconds", "budget_charge_seconds")
            }
            for k, v in sorted(ops.items())
        },
        "cost_scope": (
            "Phase wall/CPU includes all dense checks, QP solves and local probes. "
            "Nested numerical/call counters exclude setup, metric-only replay "
            "and local-probe suboperations; use phase totals for overhead."
        ),
        "validation_or_test_access": False,
        "live_llm_calls": 0,
        "automatic_revision": False,
        "prefix_forecast_status": "deferred",
    }


class _Run(old._Run):
    """Each continuation and control uses its own durable nonrenewable ledger."""

    def __init__(self, root, plan, data, index):
        self.task, self.data = plan["tasks"][index], data
        self.policy = Policy.model_validate(plan["policy"])
        self.identity = {"plan_sha256": public.content_sha256(plan), "task": self.task}
        self.folder = root / "results" / self.task["task_id"]
        self.allowances = _allowances(self.policy)
        self.ops = old.operations._operations(
            self.folder, self.identity, self.allowances
        )
        self.fit = None

    def setup(self, save):
        from autoformalism.fitting.insulin_positive_continuation import ContinuationFit

        d = self.data["source_inputs"]["documents"]
        self.fit = ContinuationFit(
            old.PrefixFitProblem.model_validate(d["problem"]),
            old.PrefixFitProblem.model_validate(d["parent_problem"]),
            old.PublicSplit.model_validate(d["training"]),
            old.kernel.NumericalPolicy.model_validate(d["numerical"]),
            d["warm_start_parameters"],
            endpoint=_source(self.data, self.task),
            enforce_positive=self.task["arm"] == "positive",
        )
        result = self.fit.setup_audit()
        save(result)
        return result

    def execute(self):
        p = self.policy
        setup = self.perform("setup", self.setup)
        if not setup.get("passed"):
            return
        if self.fit is None and self.setup(lambda _: None) != setup:
            raise ValueError("resumed continuation setup differs")
        fit = self.fit
        source = self.perform(
            "source_verification",
            lambda save: fit.verify_source(p.verification_seconds, save),
        )
        if not source.get("original_equations_verified"):
            return
        d = self.data["source_inputs"]["documents"]
        self.perform(
            "fallback_verification",
            lambda save: fit.verify(
                d["warm_start_parameters"],
                d["baseline_endpoint"]["training_nmse"],
                p.verification_seconds,
                save,
            ),
        )
        qualified = self.perform(
            "qualification", lambda save: fit.qualify(p.qualification_seconds, save)
        )
        if qualified.get("passed"):
            start = self.perform(
                "profiled_start_verification",
                lambda save: fit.verify(
                    qualified["parameters"],
                    qualified["training_nmse"],
                    p.verification_seconds,
                    save,
                ),
            )
            if start.get("eligible_for_arm") and start.get(
                "original_equations_verified"
            ):
                self.perform(
                    "search",
                    lambda save: fit.search(
                        p, save, profiled_start=qualified["parameters"]
                    ),
                )
                best = old.first._value(self.ops["search"], partial=True).get("best")
                if best:
                    self.perform(
                        "endpoint_verification",
                        lambda save: fit.verify(
                            best["parameters"],
                            best["training_nmse"],
                            p.verification_seconds,
                            save,
                        ),
                    )
        _, endpoint, _ = _retained(self.data, self.task, self.ops, p)
        if endpoint is not None:
            self.perform(
                "local_audit",
                lambda save: fit.local_audit(endpoint, p.diagnostic_seconds, save),
            )
            best = old.first._value(self.ops["local_audit"], partial=True).get("best")
            if best:
                self.perform(
                    "probe_verification",
                    lambda save: fit.verify(
                        best["parameters"],
                        best["training_nmse"],
                        p.verification_seconds,
                        save,
                    ),
                )
            _, endpoint, _ = _retained(self.data, self.task, self.ops, p)
            self.perform(
                "metrics", lambda save: fit.metrics(endpoint, p.verification_seconds)
            )


def run_task(root: Path, index: int) -> dict:
    plan, data = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown continuation task")
    folder = root / "results" / plan["tasks"][index]["task_id"]
    with public._lock(folder):
        run = _Run(root, plan, data, index)
        if (folder / "result.json").exists():
            saved = read_seal(folder / "result.json")
            old.reporting.reconcile(
                saved, _assemble(run.identity, data, run.ops, run.policy), run.ops
            )
            return saved
        run.execute()
        result = _assemble(run.identity, data, run.ops, run.policy)
        seal(folder / "result.json", result)
        return result


def report(root: Path) -> dict:
    plan, data = verify(root, runtime=False)
    policy, rows, receipts = Policy.model_validate(plan["policy"]), [], {}
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = root / "results" / task["task_id"]
        ops = old.operations._operations(folder, identity, _allowances(policy))
        row = _assemble(identity, data, ops, policy)
        terminal = (folder / "result.json").is_file()
        if terminal:
            receipts[task["task_id"]] = old.reporting.reconcile(
                read_seal(folder / "result.json"), row, ops
            )
        rows.append(
            row
            | {"status": "complete" if terminal else "partial" if ops else "missing"}
        )
    recorded = sum(row["status"] == "complete" for row in rows)
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if recorded == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": recorded,
        "rows": rows,
        "terminal_reconciliations": receipts,
        "comparison": (
            "paired within each saved warm endpoint, equal additional budgets"
        ),
        "validation_or_test_access": False,
        "automatic_revision": False,
        "limitations": [
            "Training reconstruction only; prefix prediction deferred.",
            "Concentration inequalities constrain training nodes and midpoints; "
            "a finer original replay gates retention.",
            "Sample positivity is not a continuous-time or unseen-input certificate.",
            "Only insulin is constrained; other output minima are reported.",
            "Finite local probes and optimizer success do not certify convergence "
            "or equation inadequacy.",
            "Source fits remain separately visible even if inadmissible for the "
            "positive arm.",
        ],
    }
    with public._lock(root / "report"):
        public._write(root / "summary.json", summary)
        lines = [
            "# Insulin continuation and concentration control",
            "",
            "| Source | Arm | Status | NMSE | Domain admissible |",
            "| --- | --- | --- | ---: | --- |",
        ]
        lines += [
            f"| {r['start_label']} | {r['arm']} | {r['search_status']} | "
            f"{r['training_reconstruction_nmse']} | {r['domain_admissible']} |"
            for r in rows
        ]
        (root / "REPORT.md").write_text(
            "\n".join([*lines, "", *summary["limitations"]]) + "\n"
        )
    return summary


def inspection(root: Path) -> Path:
    return old.inspection(root, reporter=report)
