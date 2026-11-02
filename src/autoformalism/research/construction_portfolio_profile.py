"""Separate, bounded coefficient-reduction and initial-root profiling campaign."""

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
from autoformalism.fitting import prefix_portfolio as numerical
from autoformalism.fitting import prefix_portfolio_followup as followup
from autoformalism.fitting import prefix_portfolio_nonlinear_search as precision
from autoformalism.fitting import prefix_portfolio_timescale as timescale
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import construction_portfolio_fitting as first
from autoformalism.research import construction_portfolio_search as previous
from autoformalism.research import continuation_report_cost as reporting
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-portfolio-profile-1"
SOURCE_PLAN = "6cddde26ba8ae9b91b5e086f3bde345b4e9e625fadeba397bbc4bd11394b4a9a"
SOURCE_RECORDS = (
    "9c9c9eccfde8e90fd3bbd1ec371f5eeaede484a333a19a49d8dad0cbb1c1706d",
    "3b835ce5727b33bf03a6fb912c779898a649b704029f221c70953960f6f85038",
    "9b5e225187cb66db0f404c7984db2dcdc32e10777144524a450201ea50e85ccc",
    "277e7f3b01126054fb1f4fca3954268cf646579c53bb0e1b3b8fe23c8aa7cbf6",
    "b9be22227d0b5f3136439ceb7b09617c297694d780d8ff615201380cc70888b3",
)
RECOVERY_INDICES = (10, 22)


class Policy(StrictSchema):
    """New finite work; historical numerical budgets never reset."""

    population: Literal["scaled-readout-5", "synthetic-smoke"] = "scaled-readout-5"
    prerequisite_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    coordinate_seconds: FiniteFloat = Field(default=240, gt=0, le=240)
    search_seconds: FiniteFloat = Field(default=1800, gt=0, le=1800)
    maximum_calls: int = Field(default=1000, gt=0, le=1000)
    verification_seconds: FiniteFloat = Field(default=300, gt=0, le=300)
    diagnostic_seconds: FiniteFloat = Field(default=600, gt=0, le=600)


def _source_inputs(source: Path) -> dict:
    """Authenticate all source terminals and nested probes before projection."""
    plan, data = previous.verify(source, runtime=False)
    policy = previous.Policy.model_validate(plan["policy"])
    records = []
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = source / "results" / task["task_id"]
        ops = previous._operations(folder, identity, previous._allowances(task, policy))
        terminal = read_seal(folder / "result.json")
        audit = reporting.reconcile(
            terminal, previous._assemble(identity, data, ops), ops
        )
        selected = ops if task["kind"] == "targeted_precision" else {}
        children = (
            {
                str(j): previous.previous._probe_operations(folder, identity, j)
                for j in RECOVERY_INDICES
            }
            if selected
            else {}
        )
        records.append(
            {
                "source_task": task,
                "terminal": terminal,
                "terminal_sha256": public.content_sha256(terminal),
                "operations": selected,
                "probes": children,
                "reconciliation": audit,
                "all_source_operations_authenticated": True,
                "accounting_addendum": _source_accounting(ops),
            }
        )
    return {
        "protocol": PROTOCOL,
        "source_plan": plan,
        "source_inputs": data,
        "records": records,
        "test_data_opened": False,
    }


def _source_accounting(operations: dict) -> dict:
    """Explicit inherited counter corrections without altering source costs."""
    rows = {}
    for name, op in operations.items():
        value = first._value(op, partial=True)
        if name.startswith("profile_") and name.endswith("_search"):
            rows[name] = {
                k: value.get("calls_" + k)
                for k in ("attempted", "started", "completed", "failed")
            }
        elif name == "chart_qualification":
            rows[name] = {k: value.get("calls_" + k) for k in ("started", "completed")}
        elif name.endswith("_diagnosis"):
            check = value.get("derivative_check") or {}
            rows[name] = {
                k: check.get("calls_" + k)
                for k in ("attempted", "started", "completed", "failed")
            }
    return {
        "scope": "Inherited profile/chart/derivative scopes, not new work; "
        "nested counts are not additional timing charges.",
        "operations": rows,
        "historical_terminal_modified": False,
    }


def _tasks(data: dict, policy: Policy) -> list[dict]:
    """Recompute the exact roster from authenticated source projections."""
    plan, old = data["source_plan"], data["source_inputs"]
    old_policy = previous.Policy.model_validate(plan["policy"])
    production = policy.population == "scaled-readout-5"
    if (
        data["protocol"] != PROTOCOL
        or data["test_data_opened"] is not False
        or plan["protocol"] != previous.PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(old)
        or plan["tasks"] != previous._tasks(old, old_policy)
        or len(data["records"]) != len(plan["tasks"])
    ):
        raise ValueError("source plan, data or roster differs")
    if production and public.content_sha256(plan) != SOURCE_PLAN:
        raise ValueError("source is not the reviewed search campaign")
    tasks = []
    for i, (task, record) in enumerate(
        zip(plan["tasks"], data["records"], strict=True)
    ):
        terminal = record["terminal"]
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        if (
            record["source_task"] != task
            or terminal["identity"] != identity
            or record["terminal_sha256"] != public.content_sha256(terminal)
            or record["all_source_operations_authenticated"] is not True
            or record["reconciliation"]["scientific_fields_exact"] is not True
        ):
            raise ValueError("source terminal authentication differs")
        if production and (
            len(SOURCE_RECORDS) != len(data["records"])
            or public.content_sha256(record) != SOURCE_RECORDS[i]
        ):
            raise ValueError("source projection differs from reviewed records")
        for stage, op in record["operations"].items():
            if public.content_sha256(op) != terminal["operation_receipts"][stage]:
                raise ValueError("source operation projection differs")
        for index, children in record["probes"].items():
            parent = first._value(
                record["operations"].get(f"coordinate_{int(index):03d}")
            )
            if parent and parent.get("primitive_receipts") != {
                k: public.content_sha256(v) for k, v in children.items()
            }:
                raise ValueError("source nested primitives differ")
        kind = (
            "precision_recovery"
            if task["kind"] == "targeted_precision"
            else "nonlinear_reduction"
            if task["kind"] == "nonlinear_search"
            else "root_profile"
        )
        seeds = (
            ("incumbent", "action_tau_10") if kind == "root_profile" else ("incumbent",)
        )
        for seed in seeds:
            if kind != "precision_recovery" and not terminal.get("retained"):
                raise ValueError("continuation requires verified source endpoint")
            if seed == "action_tau_10" and not terminal["profiles"][9][
                "verification"
            ].get("original_equations_verified"):
                raise ValueError("slower-action seed lacks original verification")
            tasks.append(
                {
                    "index": len(tasks),
                    "record_index": i,
                    "task_id": task["task_id"] + "-" + seed,
                    "candidate_id": task["candidate_id"],
                    "model_index": task["model_index"],
                    "kind": kind,
                    "seed": seed,
                    "problem_sha256": task["problem_sha256"],
                    "source_record_sha256": public.content_sha256(record),
                }
            )
    return tasks


def prepare(root: Path, source: Path, policy: Policy) -> dict:
    """Create a separate campaign; source receipts and completed work are read-only."""
    if root.resolve() == source.resolve() or root.resolve().is_relative_to(
        source.resolve()
    ):
        raise ValueError("profile root must be separate from source")
    return prepare_inputs(root, _source_inputs(source), policy)


def prepare_inputs(root: Path, data: dict, policy: Policy) -> dict:
    """Freeze the same experiment contract for production and synthetic smokes."""
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": _tasks(data, policy),
        "test_data_opened": False,
        "live_llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
        "training_prefix_reestimated": False,
        "objective": "original_full_training_reconstruction",
    }
    if reservation_ceiling(plan) > 13200:
        raise ValueError("profile reservation exceeds worker wall time")
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime=True) -> tuple[dict, dict]:
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    policy = Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != _tasks(data, policy)
        or any(
            plan[k] is not False
            for k in (
                "test_data_opened",
                "automatic_revision",
                "automatic_promotion",
                "training_prefix_reestimated",
            )
        )
        or plan["live_llm_calls"] != 0
        or plan["objective"] != "original_full_training_reconstruction"
        or reservation_ceiling(plan) > 13200
    ):
        raise ValueError("profile plan identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("profile code/runtime differs")
    return plan, data


def _allowances(task, p):
    base = {"setup": 180.0}
    if task["kind"] == "precision_recovery":
        return (
            base
            | {"import_prerequisites": 120, "precision_summary": 60}
            | {
                f"coordinate_{j:03d}": p.coordinate_seconds + 60
                for j in RECOVERY_INDICES
            }
        )
    common = base | {
        "start_verification": p.verification_seconds + 60,
        "qualification": p.prerequisite_seconds + 180,
        "search": p.search_seconds + 180,
        "endpoint_verification": p.verification_seconds + 60,
        "accepted_verification": p.verification_seconds + 60,
        "endpoint_alias": 60,
    }
    if task["kind"] == "nonlinear_reduction":
        return common | {
            "reduction_equivalence": p.prerequisite_seconds + 60,
            "retained_diagnosis": p.diagnostic_seconds + 60,
            "accepted_diagnosis": p.diagnostic_seconds + 60,
        }
    return common | {
        "meal_precision": p.diagnostic_seconds + 60,
        "profiled_start_verification": p.verification_seconds + 60,
    }


def reservation_ceiling(plan):
    p = Policy.model_validate(plan["policy"])
    return max((sum(_allowances(t, p).values()) for t in plan["tasks"]), default=0)


def _operations(folder, identity, allowances):
    return previous._operations(folder, identity, allowances)


def _selection(record, ops):
    selected = ("source", record["terminal"]["retained"])
    if (
        not first._value(ops.get("qualification")).get("passed")
        or not first._verified(ops.get("start_verification"))
        or (
            record["source_task"]["kind"] == "nonlinear_search"
            and not first._value(ops.get("reduction_equivalence")).get("passed")
        )
    ):
        return selected
    for stage in (
        "start_verification",
        "profiled_start_verification",
        "endpoint_verification",
    ):
        if first._verified(ops.get(stage)):
            value = first._value(ops[stage])
            if (
                selected[1] is None
                or value["training_nmse"] < selected[1]["training_nmse"]
            ):
                selected = (stage, value)
    return selected


def _assemble(identity, data, ops):
    task = identity["task"]
    record = data["records"][task["record_index"]]
    origin, retained = _selection(record, ops)
    search = first._value(ops.get("search"), partial=True)
    alias = first._value(ops.get("endpoint_alias"))
    status = (
        "not_started"
        if not ops
        else "setup_unavailable"
        if not first._value(ops.get("setup"))
        else "precision_recorded"
        if task["kind"] == "precision_recovery"
        else "improved_verified"
        if origin != "source"
        else "search_no_verified_improvement"
        if search
        else "qualification_unavailable"
    )
    return {
        "status": "complete",
        "identity": identity,
        "candidate_id": task["candidate_id"],
        "kind": task["kind"],
        "seed": task["seed"],
        "search_status": status,
        "retained_origin": origin,
        "retained": retained,
        "training_reconstruction_nmse": (retained or {}).get("training_nmse"),
        "source_endpoint": record["terminal"]["retained"],
        "qualification": first._value(ops.get("qualification")) or None,
        "precision": first._value(ops.get("precision_summary")) or None,
        "meal_precision": first._value(ops.get("meal_precision")) or None,
        "search_stop_reason": search.get("stop_reason"),
        "search_budget_exhausted": search.get("budget_exhausted"),
        "endpoint_alias": alias or None,
        "retained_assessment": first._value(ops.get("retained_diagnosis")) or None,
        "accepted_assessment": first._value(
            ops.get(
                "retained_diagnosis"
                if alias.get("same_parameters")
                else "accepted_diagnosis"
            )
        )
        or None,
        "training_prefix_reestimated": False,
        "source_training_prefix_forecast": record["terminal"].get(
            "source_training_prefix_forecast"
        ),
        "operation_statuses": {k: v["status"] for k, v in ops.items()},
        "operation_receipts": {k: public.content_sha256(v) for k, v in ops.items()},
        "cost": reporting.canonical_cost(ops),
        "evaluation_requests_by_operation": {
            k: first._value(v, partial=True).get("evaluation_requests_by_view")
            for k, v in ops.items()
        },
        "call_accounting": {
            "authoritative_scopes": "evaluation_requests_by_operation and "
            "profile_execution_by_operation; views/scopes may be nested",
            "legacy_cost_aliases_exhaustive": False,
            "missing_view_ledgers": [
                k
                for k, v in ops.items()
                if "evaluation_requests_by_view" not in first._value(v, partial=True)
            ],
        },
        "profile_execution_by_operation": {
            k: first._value(v, partial=True).get("profile_execution")
            for k, v in ops.items()
            if first._value(v, partial=True).get("profile_execution") is not None
        },
        "inherited_cost": record["terminal"]["cost"],
        "inherited_accounting_addendum": record["accounting_addendum"],
        "test_data_opened": False,
        "live_llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
        "global_minimum_certified": False,
        "parameter_confidence_percentage": None,
    }


class _Run:
    """A locked task, with deterministic compilation and nonrenewable stages."""

    def __init__(self, root, plan, data, index):
        self.plan, self.data = plan, data
        self.task = plan["tasks"][index]
        self.record = data["records"][self.task["record_index"]]
        self.policy = Policy.model_validate(plan["policy"])
        portfolio = data["source_inputs"]["source_inputs"]["portfolio"]
        model = portfolio["models"][self.task["model_index"]]
        self.problem = PrefixFitProblem.model_validate(model["problem"])
        self.training = PublicSplit.model_validate(portfolio["training"])
        self.identity = {"plan_sha256": public.content_sha256(plan), "task": self.task}
        self.folder = root / "results" / self.task["task_id"]
        self.allowances = _allowances(self.task, self.policy)
        self.ops = _operations(self.folder, self.identity, self.allowances)
        self.original = self.oracle = self.start = None

    def perform(self, stage, work):
        def run(save):
            def normalized(value):
                return followup.normalise_counts(value, search=stage == "search")

            result = followup.counted(
                work, lambda value: save(normalized(value)), self.original, self.oracle
            )
            return normalized(result)

        self.ops[stage] = journal._operation(
            self.folder, self.identity, stage, self.allowances[stage], run
        )
        return first._value(self.ops[stage])

    def setup(self, save):
        kind = self.task["kind"]
        if kind == "precision_recovery":
            self.oracle = precision.make_precision_oracle(self.problem, self.training)
            self.original = self.oracle.original
        elif kind == "nonlinear_reduction":
            from autoformalism.fitting import (
                prefix_portfolio_nonlinear_reduction as reduced,
            )

            self.original, self.oracle = reduced.make_oracle(
                self.problem,
                self.training,
                self.record["terminal"]["retained"]["parameters"],
            )
        else:
            settings = first.Policy.model_validate(
                self.data["source_inputs"]["source_inputs"]["original_policy"]
            ).numerical
            anchor = self.record["terminal"]["retained"]["parameters"]
            self.original, self.oracle = timescale.make_oracle(
                self.problem, self.training, settings, anchor
            )
        self.start = dict(self.oracle.start)
        if self.task["seed"] == "action_tau_10":
            saved = self.record["terminal"]["profiles"][9]["verification"]["parameters"]
            self.start = coordinates.to_effective(self.original, self.oracle, saved)
        return previous.previous._saved(
            {
                "names": list(self.oracle.names),
                "start": self.start,
                "audit": self.oracle.portfolio_audit,
                "optimizer_coordinates": len(self.oracle.names),
            },
            save,
        )

    def verify_endpoint(self, stage, parameters, expected=None):
        def work(save):
            result = numerical.verify_endpoint(
                self.original,
                self.oracle,
                parameters,
                self.policy.verification_seconds,
                save,
            )
            if (
                expected is not None
                and result.get("original_equations_verified")
                and not np.isclose(
                    result["training_nmse"], expected, rtol=1e-6, atol=1e-9
                )
            ):
                result.update(
                    original_equations_verified=False,
                    error="saved search loss differs from replay",
                )
            save(result)
            return result

        return self.perform(stage, work)

    def recover_precision(self):
        source = self.record["operations"]
        baseline, jacobian = (first._value(source[k]) for k in ("baseline", "jacobian"))

        def authenticate(save):
            precision._precision_prepared(self.oracle, baseline, jacobian)
            return previous.previous._saved(
                {
                    "passed": True,
                    "baseline_sha256": public.content_sha256(baseline),
                    "jacobian_sha256": public.content_sha256(jacobian),
                    "new_numerical_calls": 0,
                },
                save,
            )

        if not self.perform("import_prerequisites", authenticate).get("passed"):
            return
        for j in RECOVERY_INDICES:

            def work(save, j=j):
                reused = {}

                def probe(key, parameters, run):
                    if not re.fullmatch(r"probe_[0-9]{2}", key):
                        raise ValueError("unsafe precision primitive key")
                    inherited = self.record["probes"][str(j)]
                    value = followup.reuse_precision_probe(
                        self.oracle, inherited, key, parameters
                    )
                    if value is not None:
                        reused[key] = public.content_sha256(inherited[key])
                        return value
                    op = journal._operation(
                        self.folder / "probes" / f"coordinate_{j:03d}",
                        self.identity,
                        key,
                        120,
                        run,
                    )
                    if op["status"] != "complete" or not op.get("value"):
                        raise ValueError("new precision primitive unavailable")
                    return op["value"]

                result = precision.precision_coordinate(
                    self.oracle,
                    baseline,
                    jacobian,
                    j,
                    self.policy.coordinate_seconds,
                    save,
                    probe_runner=probe,
                )
                result["imported_probe_receipts"] = reused
                result["primitive_receipts"] = {
                    k: public.content_sha256(v)
                    for k, v in previous.previous._probe_operations(
                        self.folder, self.identity, j
                    ).items()
                }
                return previous.previous._saved(result, save)

            self.perform(f"coordinate_{j:03d}", work)
        receipts = [
            first._value(
                self.ops.get(f"coordinate_{j:03d}")
                if j in RECOVERY_INDICES
                else source.get(f"coordinate_{j:03d}")
            )
            for j in precision.PRECISION_INDICES
        ]
        self.perform(
            "precision_summary",
            lambda save: previous.previous._saved(
                precision.precision_summary(self.oracle, [r for r in receipts if r]),
                save,
            ),
        )

    def fit(self):
        p = self.policy
        if self.task["kind"] == "nonlinear_reduction":
            from autoformalism.fitting import (
                prefix_portfolio_nonlinear_reduction as reduced,
            )

            equivalence = self.perform(
                "reduction_equivalence",
                lambda save: reduced.chart_equivalence(
                    self.original, self.oracle, p.prerequisite_seconds, save
                ),
            )
            if not equivalence.get("passed"):
                return
        seed_record = (
            self.record["terminal"]["profiles"][9]["verification"]
            if self.task["seed"] == "action_tau_10"
            else self.record["terminal"]["retained"]
        )
        self.verify_endpoint(
            "start_verification", self.start, seed_record["training_nmse"]
        )
        if not first._verified(self.ops.get("start_verification")):
            return
        if self.task["kind"] == "root_profile":
            from autoformalism.fitting import prefix_portfolio_root_profile as roots

            qualified = self.perform(
                "qualification",
                lambda save: roots.root_profile_qualification(
                    self.oracle, self.start, p.prerequisite_seconds, save
                ),
            )
            if qualified.get("passed"):
                self.verify_endpoint(
                    "profiled_start_verification",
                    qualified["parameters"],
                    qualified.get("training_nmse"),
                )
            if qualified.get("passed") and first._verified(
                self.ops.get("profiled_start_verification")
            ):
                self.perform(
                    "search",
                    lambda save: roots.root_profile_search(
                        self.oracle, self.start, p.search_seconds, p.maximum_calls, save
                    ),
                )
        else:
            qualified = self.perform(
                "qualification",
                lambda save: numerical.qualification(
                    self.original, self.oracle, self.start, p.prerequisite_seconds, save
                ),
            )
            if qualified.get("passed"):
                self.perform(
                    "search",
                    lambda save: numerical.search(
                        self.oracle, self.start, p.search_seconds, p.maximum_calls, save
                    ),
                )
        search = first._value(self.ops.get("search"), partial=True)
        if not qualified.get("passed"):
            # An unavailable full sensitivity prerequisite must not be repeated
            # immediately as an endpoint diagnosis at the identical start.
            return
        if search.get("best"):
            self.verify_endpoint(
                "endpoint_verification",
                search["best"]["parameters"],
                search["best"].get("training_nmse"),
            )
        if self.task["kind"] == "root_profile":
            if self.task["seed"] == "incumbent":
                name = "k_A_decay" if self.task["model_index"] == 12 else "k_abs"
                self.perform(
                    "meal_precision",
                    lambda save: followup.meal_precision(
                        self.oracle, self.start, name, p.diagnostic_seconds, save
                    ),
                )
        else:
            self.diagnose(search)

    def diagnose(self, search):
        _, selected = _selection(self.record, self.ops)
        if not selected:
            return
        retained = coordinates.to_effective(
            self.original, self.oracle, selected["parameters"]
        )
        accepted = search.get("last_accepted_parameters")
        same = retained == accepted
        self.perform(
            "endpoint_alias",
            lambda save: previous.previous._saved(
                {
                    "same_parameters": same,
                    "retained_sha256": public.content_sha256(retained),
                    "accepted_sha256": public.content_sha256(accepted),
                    "portfolio_audit_sha256": public.content_sha256(
                        self.oracle.portfolio_audit
                    ),
                    "reuse_scope": "Identical model/data/chart/point/settings only; "
                    "history stays endpoint-bound.",
                },
                save,
            ),
        )
        for label, parameters in (("retained", retained), ("accepted", accepted)):
            if not parameters or (label == "accepted" and same):
                continue
            if label == "retained":
                verification = selected
            else:
                verification = self.verify_endpoint("accepted_verification", parameters)
            if verification.get("original_equations_verified"):
                self.perform(
                    label + "_diagnosis",
                    lambda save,
                    parameters=parameters,
                    verification=verification: numerical.diagnose(
                        self.original,
                        self.oracle,
                        parameters,
                        verification,
                        search,
                        self.policy.diagnostic_seconds,
                        save,
                    ),
                )


def run_task(root: Path, index: int) -> dict:
    """Resume unused reservations only; already started work never gets a new budget."""
    plan, data = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown profile task index")
    folder = root / "results" / plan["tasks"][index]["task_id"]
    with public._lock(folder):
        run = _Run(root, plan, data, index)
        if (folder / "result.json").exists():
            saved = read_seal(folder / "result.json")
            reporting.reconcile(saved, _assemble(run.identity, data, run.ops), run.ops)
            return saved
        if run.perform("setup", run.setup):
            if run.oracle is None:
                run.setup(lambda _: None)
            if run.task["kind"] == "precision_recovery":
                run.recover_precision()
            else:
                run.fit()
        result = _assemble(run.identity, data, run.ops)
        seal(folder / "result.json", result)
        return result


def report(root: Path) -> dict:
    """Reconstruct receipts without launching numerical verification."""
    plan, data = verify(root, runtime=False)
    p = Policy.model_validate(plan["policy"])
    rows, audits = [], {}
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = root / "results" / task["task_id"]
        ops = _operations(folder, identity, _allowances(task, p))
        value = _assemble(identity, data, ops)
        terminal = (folder / "result.json").exists()
        if terminal:
            audits[task["task_id"]] = reporting.reconcile(
                read_seal(folder / "result.json"), value, ops
            )
        rows.append(
            value
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
        "status_counts": dict(Counter(r["search_status"] for r in rows)),
        "terminal_reconciliations": audits,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitations": [
            "Training reconstruction only; new endpoints need causal-prefix forecasts.",
            "Bounded derivative-free profiles do not certify convergence "
            "or confidence.",
            "Timescale coordinates cover the negative-self-coefficient branch only.",
            "Inherited call corrections do not rewrite historical terminals.",
            "Evaluation requests by view may be nested; do not sum as physical solves.",
        ],
    }
    with public._lock(root / "report"):
        public._write(root / "summary.json", summary)
        lines = [
            "# Construction portfolio profiling",
            "",
            f"{recorded}/{len(rows)} terminal tasks.",
            "",
            "| Model | Seed | Status | Training NMSE |",
            "| --- | --- | --- | ---: |",
        ]
        lines += [
            f"| {r['candidate_id']} | {r['seed']} | {r['search_status']} | "
            f"{r['training_reconstruction_nmse']} |"
            for r in rows
        ]
        (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def inspection(root: Path) -> Path:
    """Package immutable provenance and receipts with bounded log tails."""
    report(root)
    paths = [
        root / n for n in ("plan.json", "inputs.json", "summary.json", "REPORT.md")
    ]
    paths += sorted((root / "results").rglob("*.json"))
    paths += sorted((root / "submissions").rglob("*.json"))
    paths += sorted((root / "submissions").rglob("*.id"))
    paths += [
        root / n
        for n in ("launcher_execution.json", "submission_manifest.json")
        if (root / n).exists()
    ]
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in sorted(set(paths)):
            if path.is_symlink() or not path.is_file():
                raise ValueError("nonregular inspection input")
            info = archive.gettarinfo(str(path), arcname=str(path.relative_to(root)))
            info.mtime = 0
            with path.open("rb") as handle:
                archive.addfile(info, handle)
        logs = {}
        for path in sorted((root / "logs").glob("*")):
            if path.is_file() and not path.is_symlink():
                with path.open("rb") as handle:
                    handle.seek(max(0, path.stat().st_size - 16384))
                    logs[path.name] = handle.read().decode(errors="replace")
        payload = json.dumps(logs, sort_keys=True).encode()
        info = tarfile.TarInfo("inspection_log_tails.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    output = root / "inspection.tar.gz"
    output.write_bytes(buffer.getvalue())
    return output
