"""Matched root-only and joint shared-forcing/initial-root fitting controls."""

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
from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_meal_continuation as base
from autoformalism.research import construction_meal_scale_recovery as previous
from autoformalism.research import construction_portfolio_fitting as first
from autoformalism.research import construction_portfolio_residual as retention
from autoformalism.research import continuation_report_cost as reporting
from autoformalism.schemas.base import FiniteFloat
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-meal-joint-profile-1"
SOURCE_PLAN = "5c0b0ffd13b1f3256234f5587e899f3eaadd0088d14cb23d2ba1c3bb6ec13955"
SOURCE_SEEDS = ("warm-flux", "fixed-0p001", "fixed-0p01", "fixed-0p1", "fixed-1")
SOURCE_RECORDS = (
    "8f552b3383bd06104458f7f255bdba769b75bc37bec171c9f6f0d44c002a082f",
    "b19c09fce0ad40cd522a4eff2b12ba829ee7d0579dcb01f3010c5f4cd70eefdd",
    "717579bd8745623a9d9d2403496ffda16f84c1efb00282f930c526ed20089b0c",
    "e9f5857d9ca7b6ee1a4225e8c896d148deac4bb52410e501af880b034592ee89",
    "c115c985fc472c4fc60fbf62cb25c1426a4dc6d9f945594534bb01fa0dc7e0a0",
)
ARMS = ("root_only", "joint")
SOURCE_TRAINING = previous.SOURCE_TRAINING
SOURCE_NUMERICAL = previous.SOURCE_NUMERICAL


class Policy(previous.Policy):
    """New reservations keep numerical limits separate from persistence headroom."""

    population: Literal["scaled-readout-meal-joint-1", "synthetic-smoke"] = (
        "scaled-readout-meal-joint-1"
    )
    persistence_headroom_seconds: FiniteFloat = Field(default=900, ge=0, le=900)
    endpoint_probe_seconds: FiniteFloat = Field(default=180, gt=0, le=180)

    @model_validator(mode="after")
    def bounded_overall_wall(self):
        """Reject an invalid combined search envelope before freezing a campaign."""
        if self.search_seconds + self.persistence_headroom_seconds > 3600:
            raise ValueError("combined search wall allowance exceeds 3600 seconds")
        return self


def _selected(plan):
    """Select saved positive controls in their frozen order, without ranking fits."""
    if plan["policy"]["population"] == "synthetic-smoke":
        return list(plan["tasks"])
    return [task for task in plan["tasks"] if task["seed"] in SOURCE_SEEDS]


def _source_inputs(source: Path) -> dict:
    """Authenticate checkpoint terminals and project their immediate evidence."""
    plan, data = previous.verify(source, runtime=False)
    policy = previous.Policy.model_validate(plan["policy"])
    records = []
    for task in _selected(plan):
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = source / "results" / task["task_id"]
        ops = previous._operations(folder, identity, previous._allowances(task, policy))
        terminal = read_seal(folder / "result.json")
        reconciliation = reporting.reconcile(
            terminal, previous._assemble(identity, data, ops, policy), ops
        )
        origin = data["records"][task["record_index"]]
        records.append(
            {
                "source_task": task,
                "problem": origin["problem"],
                "terminal": terminal,
                "operations": ops,
                "reconciliation": reconciliation,
                "terminal_sha256": public.content_sha256(terminal),
                "boundary_coordinate": task["boundary_coordinate"],
            }
        )
    return {
        "protocol": PROTOCOL,
        "source_plan": plan,
        "training": data["training"],
        "numerical": data["numerical"],
        "records": records,
        "test_data_opened": False,
    }


def _finite_endpoint(endpoint: dict) -> bool:
    """Require source replay evidence and a finite nonnegative training loss."""
    return bool(
        endpoint.get("original_equations_verified") is True
        and type(endpoint.get("training_nmse")) in (int, float)
        and np.isfinite(endpoint["training_nmse"])
        and endpoint["training_nmse"] >= 0
        and isinstance(endpoint.get("parameters"), dict)
        and isinstance(endpoint.get("effective_parameters"), dict)
    )


def _warm_endpoint(data: dict) -> dict:
    """The first frozen record supplies the independently retained warm incumbent."""
    return data["records"][0]["terminal"]["retained"]


def _tasks(data: dict, p: Policy) -> list[dict]:
    """Bind each recovery to its own valid trial and the shared warm incumbent."""
    source = data["source_plan"]
    production = p.population == "scaled-readout-meal-joint-1"
    if (
        data["protocol"] != PROTOCOL
        or data["test_data_opened"] is not False
        or source["protocol"] != previous.PROTOCOL
    ):
        raise ValueError("meal joint profile source protocol differs")
    if production and public.content_sha256(source) != SOURCE_PLAN:
        raise ValueError("source is not the reviewed meal scale recovery campaign")
    if production and (
        public.content_sha256(data["training"]) != SOURCE_TRAINING
        or public.content_sha256(data["numerical"]) != SOURCE_NUMERICAL
    ):
        raise ValueError("reviewed training data or numerical settings differ")
    expected = _selected(source)
    if len(expected) != len(data["records"]) or not expected:
        raise ValueError("source roster differs")
    if production and tuple(t["seed"] for t in expected) != SOURCE_SEEDS:
        raise ValueError("reviewed positive-control roster differs")
    if not production and source["policy"]["population"] != "synthetic-smoke":
        raise ValueError("synthetic policy cannot consume a production source")
    ids = [t["task_id"] for t in expected]
    if len(set(ids)) != len(ids) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in ids
    ):
        raise ValueError("unsafe or duplicate source task path")
    training = PublicSplit.model_validate(data["training"])
    if training.name != "train":
        raise ValueError("only public training data are permitted")
    kernel.NumericalPolicy.model_validate(data["numerical"])
    warm = _warm_endpoint(data)
    if not _finite_endpoint(warm):
        raise ValueError("source lacks a verified finite warm endpoint")
    tasks = []
    for index, (task, record) in enumerate(zip(expected, data["records"], strict=True)):
        terminal, ops = record["terminal"], record["operations"]
        identity = {"plan_sha256": public.content_sha256(source), "task": task}
        problem = PrefixFitProblem.model_validate(record["problem"])
        if (
            record["source_task"] != task
            or terminal["identity"] != identity
            or public.content_sha256(terminal) != record["terminal_sha256"]
            or record["reconciliation"]["scientific_fields_exact"] is not True
            or public.content_sha256(record["problem"]) != task["problem_sha256"]
            or terminal["operation_receipts"]
            != {key: public.content_sha256(value) for key, value in ops.items()}
        ):
            raise ValueError("source endpoint or receipt authentication differs")
        if production and public.content_sha256(record) != SOURCE_RECORDS[index]:
            raise ValueError("source projection differs from reviewed evidence")
        if record["problem"] != data["records"][0]["problem"]:
            raise ValueError("shared incumbent belongs to a different model")
        trial = terminal.get("trial_endpoint") or {}
        if not _finite_endpoint(trial) or not _finite_endpoint(
            terminal.get("retained") or {}
        ):
            raise ValueError("source lacks a verified finite trial endpoint")
        if production and tuple(problem.context.targets) != ("Gp", "I", "U"):
            raise ValueError("reviewed target contract differs")
        if task["kind"] != "root_profile":
            raise ValueError("unsupported recovery kind")
        boundary = record["boundary_coordinate"]
        if boundary != task["boundary_coordinate"] or not isinstance(boundary, str):
            raise ValueError("boundary coordinate differs")
        kernel.NumericalPolicy.model_validate(
            ops["setup"]["value"]["audit"]["numerical_policy"]
        )
        fixed = task["experiment"]["fixed"]
        if any(trial["effective_parameters"].get(k) != v for k, v in fixed.items()):
            raise ValueError("source trial changed its fixed rate")
        if production and (
            task["experiment"]["zero_representative"]
            or task["experiment"]["coordinate_chart"] != "meal_flux"
        ):
            raise ValueError("recovery requires positive meal coordinates")
        experiment = dict(task["experiment"]) | {
            "label": task["seed"],
            "meal_rate_start": None,
            "zero_representative": False,
            "start_rule": "saved verified trial; unchanged coefficients and initials",
        }
        for arm in ARMS:
            tasks.append(
                {
                    "index": len(tasks),
                    "record_index": index,
                    "task_id": task["task_id"] + "-" + arm,
                    "candidate_id": task["candidate_id"],
                    "model_index": task["model_index"],
                    "kind": task["kind"],
                    "arm": arm,
                    "seed": task["seed"],
                    "problem_sha256": task["problem_sha256"],
                    "source_record_sha256": public.content_sha256(record),
                    "source_trial_sha256": public.content_sha256(trial),
                    "warm_incumbent_sha256": public.content_sha256(warm),
                    "boundary_coordinate": boundary,
                    "experiment": experiment,
                }
            )
    return tasks


def prepare(root: Path, source: Path, policy: Policy) -> dict:
    """Freeze new work separately without modifying historical receipts."""
    if (
        root.resolve() == source.resolve()
        or root.resolve().is_relative_to(source.resolve())
        or source.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError("new root must be separate from source")
    return prepare_inputs(root, _source_inputs(source), policy)


def prepare_inputs(root: Path, data: dict, policy: Policy) -> dict:
    """Publish immutable source projection and numerical policy."""
    policy = Policy.model_validate(policy.model_dump(mode="json"))
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
        raise ValueError("reservations exceed worker envelope")
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime=True) -> tuple[dict, dict]:
    """Recompute roster, provenance and optionally exact execution runtime."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    p = Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != _tasks(data, p)
        or plan["objective"] != "original_full_training_reconstruction"
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
        or reservation_ceiling(plan) > 13200
    ):
        raise ValueError("meal joint profile campaign identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("meal joint profile campaign code/runtime differs")
    return plan, data


def _allowances(task, p):
    """Reserve explicit I/O headroom without expanding the numerical allowance."""
    result = previous._allowances(task, p)
    result["search"] = p.search_seconds + p.persistence_headroom_seconds + 60
    result["trial_start_verification"] = p.verification_seconds + 60
    result["endpoint_probe"] = p.endpoint_probe_seconds + 180
    return result


def reservation_ceiling(plan):
    p = Policy.model_validate(plan["policy"])
    return max((sum(_allowances(t, p).values()) for t in plan["tasks"]), default=0)


def _operations(folder, identity, allowances):
    return previous._operations(folder, identity, allowances)


def _retention_data(data: dict) -> dict:
    """Build a derived selection view; never alter authenticated source terminals."""
    warm = _warm_endpoint(data)
    records = [
        record | {"terminal": record["terminal"] | {"retained": warm}}
        for record in data["records"]
    ]
    return data | {"records": records}


def _assemble(identity, data, ops, policy):
    """Retain global and fixed-family incumbents independently of new failures."""
    result = base._assemble(identity, _retention_data(data), ops, policy)
    record = data["records"][identity["task"]["record_index"]]
    inherited_trial = record["terminal"]["trial_endpoint"]
    trial = result["trial_endpoint"]
    trial_comparison = (
        retention.compare_endpoints(inherited_trial, trial, policy) if trial else None
    )
    if (
        trial_comparison
        and trial_comparison["classification"] == "meaningful_improvement"
    ):
        trial_origin = "new_verified_trial"
    else:
        trial, trial_origin = inherited_trial, "source_verified_trial"
    comparison = retention.compare_endpoints(result["retained"], trial, policy)
    if comparison["classification"] == "meaningful_improvement":
        result |= {
            "retained": trial,
            "retained_origin": trial_origin,
            "training_reconstruction_nmse": trial["training_nmse"],
            "per_target_normalized_mse": trial.get("per_target_normalized_mse"),
        }
    probe = first._value(ops.get("endpoint_probe"), partial=True)
    return result | {
        "trial_endpoint": trial,
        "trial_training_nmse": trial["training_nmse"],
        "trial_origin": trial_origin,
        "source_trial_endpoint": inherited_trial,
        "source_record_endpoint": record["terminal"]["retained"],
        "retained_vs_selected_trial_comparison": comparison,
        "inherited_trial_vs_new_trial_comparison": trial_comparison,
        "source_terminal_sha256": record["terminal_sha256"],
        "shared_incumbent_source_task": data["records"][0]["source_task"]["task_id"],
        "shared_incumbent_inherited_cost": data["records"][0]["terminal"]["cost"],
        "inherited_cost_scope": (
            "Historical task and shared-source costs overlap; do not sum."
        ),
        "search_allowance": result["search_allowance"]
        | {
            "overall_wall_seconds": policy.search_seconds
            + policy.persistence_headroom_seconds,
            "persistence_headroom_seconds": policy.persistence_headroom_seconds,
        },
        "durable_improvements_preserved": True,
        "arm": identity["task"]["arm"],
        "profiled_start_endpoint": (
            first._value(ops["profiled_start_verification"])
            if first._verified(ops.get("profiled_start_verification"))
            else None
        ),
        "endpoint_probe": probe or None,
        "endpoint_probe_status": probe.get("status", "not_started"),
        "endpoint_probe_calls": {
            field: probe.get(field)
            for field in (
                "planned_calls",
                "calls_started",
                "calls_completed",
                "calls_failed",
            )
        },
        "endpoint_probe_allowance": {
            "numerical_seconds": policy.endpoint_probe_seconds,
            "overall_wall_seconds": policy.endpoint_probe_seconds + 120,
            "separate_from_search": True,
            "retention_changes_permitted": False,
        },
    }


class _Run(previous._Run):
    """Continue each rate's verified endpoint while retaining the shared best fit."""

    def __init__(self, root, plan, data, index):
        self.plan, self.data = plan, data
        self.task = plan["tasks"][index]
        self.record = data["records"][self.task["record_index"]]
        self.policy = Policy.model_validate(plan["policy"])
        self.problem = PrefixFitProblem.model_validate(self.record["problem"])
        self.training = PublicSplit.model_validate(data["training"])
        self.identity = {"plan_sha256": public.content_sha256(plan), "task": self.task}
        self.folder = root / "results" / self.task["task_id"]
        self.allowances = _allowances(self.task, self.policy)
        self.ops = _operations(self.folder, self.identity, self.allowances)
        self.original = self.oracle = self.start = None
        self.source_start = self.chart = None

    def setup(self, save):
        """Compile the common model and lift the trial without changing parameters."""
        from autoformalism.fitting import prefix_continuation as coordinates
        from autoformalism.fitting import prefix_meal_coordinates as meal
        from autoformalism.fitting import prefix_portfolio_timescale as timescale

        anchor = _warm_endpoint(self.data)["parameters"]
        settings = kernel.NumericalPolicy.model_validate(
            self.record["operations"]["setup"]["value"]["audit"]["numerical_policy"]
        )
        self.original, self.oracle = timescale.make_oracle(
            self.problem, self.training, settings, anchor
        )
        self.source_start = coordinates.to_effective(self.original, self.oracle, anchor)
        trial = self.record["terminal"]["trial_endpoint"]
        self.start = coordinates.to_effective(
            self.original, self.oracle, trial["parameters"]
        )
        experiment = self.task["experiment"]
        if any(self.start.get(k) != v for k, v in experiment["fixed"].items()):
            raise ValueError("lifted trial changed its held coordinate")
        self.oracle.vector(self.start)
        coordinate_start = None
        if self.task["candidate_id"] == "candidate-15":
            self.chart = meal.certify(self.oracle, self.start)
            coordinate_start = {
                "flux_gain": self.start[self.chart.rate]
                * self.start[self.chart.input_gain],
                "scaled_initials": self.chart.scaled_initials(self.start),
                "optimizer_units": dict(
                    zip(
                        self.chart.names,
                        self.chart.units(self.start).tolist(),
                        strict=True,
                    )
                ),
            }
        value = {
            "passed": True,
            "names": list(self.oracle.names),
            "start": self.start,
            "source_start": self.source_start,
            "audit": self.oracle.portfolio_audit,
            "optimizer_coordinates": len(self.oracle.names),
            "experiment": experiment,
            "coordinate_audit": self.chart.audit if self.chart else None,
            "coordinate_start": coordinate_start,
            "zero_representative_certificate": None,
            "source_trial_sha256": public.content_sha256(trial),
        }
        save(value)
        return value

    def profiler_factory(self):
        """Choose the declared algorithm; never fall back after certificate failure."""
        from autoformalism.fitting import prefix_meal_joint_profile as joint
        from autoformalism.fitting import prefix_portfolio_root_profile as roots

        return (
            joint.JointMealProfile if self.task["arm"] == "joint" else roots.RootProfile
        )

    def search(self, save):
        """Keep immediate durable improvements; add explicitly budgeted I/O time."""
        from autoformalism.fitting import prefix_portfolio_root_profile as roots
        from autoformalism.fitting.compact_search_checkpoint import (
            SearchCheckpointPolicy,
        )

        p = self.policy
        return roots.root_profile_search(
            self.oracle,
            self.start,
            p.search_seconds,
            p.maximum_calls,
            save,
            failure_diagnostics=True,
            profiler_factory=self.profiler_factory(),
            checkpoint_policy=SearchCheckpointPolicy(
                interval_seconds=p.checkpoint_interval_seconds
            ),
            fixed=self.task["experiment"]["fixed"],
            coordinate_chart=self.chart,
            overall_wall_seconds=p.search_seconds + p.persistence_headroom_seconds,
        )

    def fit(self):
        """Independently recheck global/trial points before any bounded search."""
        from autoformalism.fitting import prefix_portfolio_root_profile as roots

        p = self.policy
        self.verify_endpoint(
            "start_verification",
            self.source_start,
            _warm_endpoint(self.data)["training_nmse"],
        )
        if not first._verified(self.ops.get("start_verification")):
            return
        self.verify_endpoint(
            "trial_start_verification",
            self.start,
            self.record["terminal"]["trial_endpoint"]["training_nmse"],
        )
        if not first._verified(self.ops.get("trial_start_verification")):
            return
        qualified = self.perform(
            "qualification",
            lambda save: roots.root_profile_qualification(
                self.oracle,
                self.start,
                p.prerequisite_seconds,
                save,
                profiler_factory=self.profiler_factory(),
            ),
        )
        if not qualified.get("passed"):
            return
        self.verify_endpoint(
            "profiled_start_verification",
            qualified["parameters"],
            qualified.get("training_nmse"),
        )
        if not first._verified(self.ops.get("profiled_start_verification")):
            return
        self.perform("search", self.search)
        search = first._value(self.ops.get("search"), partial=True)
        if search.get("best"):
            self.verify_endpoint(
                "endpoint_verification",
                search["best"]["parameters"],
                search["best"].get("training_nmse"),
            )

        self.perform("endpoint_probe", self.probe_endpoint)

    def probe_endpoint(self, save):
        """Probe the best same-rate replayed point without modifying retention."""
        from autoformalism.fitting.profile_endpoint_probe import profile_endpoint_probe

        selected = first._value(self.ops["trial_start_verification"])
        for stage in ("profiled_start_verification", "endpoint_verification"):
            if not first._verified(self.ops.get(stage)):
                continue
            endpoint = first._value(self.ops[stage])
            if (
                retention.compare_endpoints(selected, endpoint, self.policy)[
                    "classification"
                ]
                == "meaningful_improvement"
            ):
                selected = endpoint
        subject = {
            "original_parameters_sha256": public.content_sha256(selected["parameters"]),
            "effective_parameters_sha256": public.content_sha256(
                selected["effective_parameters"]
            ),
            "training_nmse": selected["training_nmse"],
            "arm": self.task["arm"],
            "fixed_coordinates": self.task["experiment"]["fixed"],
            "retention_changes_permitted": False,
        }

        def checkpoint(value):
            save(value | {"subject": subject})

        result = profile_endpoint_probe(
            self.oracle,
            selected["effective_parameters"],
            self.policy.endpoint_probe_seconds,
            checkpoint,
            profiler_factory=self.profiler_factory(),
            fixed=self.task["experiment"]["fixed"],
            coordinate_chart=self.chart,
        )
        return result | {"subject": subject}


def run_task(root: Path, index: int) -> dict:
    """Resume unspent stages only; never repeat interrupted search allowances."""
    plan, data = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown meal joint profile task index")
    folder = root / "results" / plan["tasks"][index]["task_id"]
    p = Policy.model_validate(plan["policy"])
    with public._lock(folder):
        run = _Run(root, plan, data, index)
        if (folder / "result.json").exists():
            saved = read_seal(folder / "result.json")
            reporting.reconcile(
                saved, _assemble(run.identity, data, run.ops, p), run.ops
            )
            return saved
        if run.perform("setup", run.setup):
            if run.oracle is None:
                run.setup(lambda _: None)
            run.fit()
        result = _assemble(run.identity, data, run.ops, p)
        seal(folder / "result.json", result)
        return result


def _projection_pairs(rows: list[dict], policy: Policy) -> list[dict]:
    """Compare fixed-outer projections only when both independent replays exist."""
    result = []
    grouped = {}
    for row in rows:
        grouped.setdefault(row["seed"], {})[row["arm"]] = row
    for seed, arms in grouped.items():
        if (
            len(
                {
                    row["identity"]["task"]["source_trial_sha256"]
                    for row in arms.values()
                }
            )
            != 1
        ):
            raise ValueError("paired projections have different source starts")
        endpoints = {
            arm: row.get("profiled_start_endpoint") for arm, row in arms.items()
        }
        available = all(endpoints.get(arm) for arm in ARMS)
        comparison = (
            retention.compare_endpoints(
                endpoints["root_only"], endpoints["joint"], policy
            )
            if available
            else None
        )
        result.append(
            {
                "seed": seed,
                "status": "verified" if available else "unavailable",
                "scope": "matched frozen outer values; conditional projections only",
                "source_trial_sha256": {
                    arm: row["identity"]["task"]["source_trial_sha256"]
                    for arm, row in arms.items()
                },
                "training_nmse": {
                    arm: endpoint["training_nmse"] if endpoint else None
                    for arm, endpoint in endpoints.items()
                },
                "comparison": comparison,
                "joint_projection_nonworsening": (
                    comparison["classification"] != "worse" if comparison else None
                ),
                "outer_convergence_certified": False,
                "parameter_confidence_certified": False,
            }
        )
    return result


def report(root: Path) -> dict:
    """Reconstruct results without integrating, fitting, or replacing terminals."""
    plan, data = verify(root, runtime=False)
    p = Policy.model_validate(plan["policy"])
    rows, audits = [], {}
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = root / "results" / task["task_id"]
        ops = _operations(folder, identity, _allowances(task, p))
        row = _assemble(identity, data, ops, p)
        terminal = (folder / "result.json").exists()
        if terminal:
            audits[task["task_id"]] = reporting.reconcile(
                read_seal(folder / "result.json"), row, ops
            )
        rows.append(
            row
            | {"status": "complete" if terminal else "partial" if ops else "missing"}
        )
    recorded = sum(row["status"] == "complete" for row in rows)
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if recorded == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": recorded,
        "rows": rows,
        "paired_projection_comparisons": _projection_pairs(rows, p),
        "status_counts": dict(Counter(row["search_status"] for row in rows)),
        "terminal_reconciliations": audits,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitations": [
            "Training reconstruction only; no new causal-prefix forecasts.",
            "New warm continuations do not restore internal Powell directions.",
            "Residual-only qualification does not qualify derivatives or confidence.",
            "Conditional convex-block optimality is not outer convergence.",
            "Root-only and joint shared-gain/drift controls use matched search caps.",
            "The shared meal gain and drift are shared across all trajectories.",
            "Directional endpoint probes do not establish stationarity or confidence.",
            "Numerical ties retain the source incumbent.",
            "Fixed-coordinate trial losses are upper bounds, not certified minima.",
            "No parameter region is excluded by an unsuccessful nuisance solve.",
            "Meal-flux coordinates apply only on the strictly positive rate branch.",
            "Exact zero is not rerun in this recovery milestone.",
            "Each saved fixed-rate trial remains independently retained.",
            "Shared warm and task source costs overlap and must not be summed.",
            "Raw optimizer success is distinct from a paired finite terminal point.",
            "Routine progress counters can lag; operation reservations never renew.",
            "Overall-wall and numerical-active limits are separate.",
        ],
    }
    with public._lock(root / "report"):
        public._write(root / "summary.json", result)
        lines = [
            "# Construction meal joint profile",
            "",
            f"{recorded}/{len(rows)} terminal tasks.",
            "",
            "| Trial / arm | Outcome | Termination | Trial NMSE | Retained NMSE |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        lines += [
            f"| {row['candidate_id']} / {row['seed']} / {row['arm']} | "
            f"{row['search_status']} | "
            f"{row['search_termination']} | "
            f"{row['trial_training_nmse']} | "
            f"{row['training_reconstruction_nmse']} |"
            for row in rows
        ]
        (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def inspection(root: Path) -> Path:
    """Package compact provenance and immutable receipts for offline inspection."""
    report(root)
    paths = [
        root / n for n in ("plan.json", "inputs.json", "summary.json", "REPORT.md")
    ]
    paths += sorted((root / "results").rglob("*.json"))
    paths += sorted((root / "submissions").rglob("*.json"))
    paths += sorted((root / "submissions").rglob("*.id"))
    paths += (
        [root / "launcher_execution.json"]
        if (root / "launcher_execution.json").exists()
        else []
    )
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
    output = root / "inspection.tar.gz"
    output.write_bytes(buffer.getvalue())
    return output
