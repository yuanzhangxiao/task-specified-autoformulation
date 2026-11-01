"""Bounded searches and timescale diagnostics for five authenticated models.

Model 03 receives precision diagnostics only. Models 10/14 start from their
previously qualified BDF points. Models 12/15 keep their incumbents while their
stable-branch timescales are investigated. Every operation has one durable,
nonrenewable reservation; profile endpoints never select a production model.
"""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_continuation as coordinates
from autoformalism.fitting import prefix_portfolio as numerical
from autoformalism.fitting import prefix_portfolio_nonlinear_search as nonlinear
from autoformalism.fitting import prefix_portfolio_polish as polish
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import construction_portfolio_continuation as previous
from autoformalism.research import construction_portfolio_fitting as first
from autoformalism.research import continuation_report_cost as reporting
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-portfolio-search-1"
SOURCE_PLAN = "e92e3bf12f64431d6db74c812f849396747ed98bcb0a30ac02a60a47b3491ea1"
SELECTED = (3, 10, 12, 14, 15)
PRECISION_INDICES = (8, 10, 19, 22)
PROFILE_COUNT = 10
# Compact projections are pinned after authenticating the returned raw receipts.
# They contain imported numerical evidence, never a new fit or relaxed check.
SOURCE_EXPORT_SHA256S = (
    "1533b2c3d03791bcc68233c23997c7585303a3a49fa5584c4d33ed1d743bfaf6",
    "d05544cacad71ce9b9b28c5cf3eab3399695c32087e81a3981f1ba6212a826b0",
    "04d6e52b111a509c31c4dc7a35bdc73bb227390c338a3ac83e7e43b534fc3a91",
    "4f206a686d8db2cd6e90056384102dcb3ffa92ba82445fdfde4b685a58cf3192",
    "5281dc0f664a6d57ac3a7c95413f3fe2faad7853f12786cefd5a43402f74cd47",
)


class Policy(StrictSchema):
    """Finite additional work; inherited work and allowances remain separate."""

    population: Literal["scaled-readout-5", "synthetic-smoke"] = "scaled-readout-5"
    prerequisite_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    coordinate_seconds: FiniteFloat = Field(default=240, gt=0, le=240)
    search_seconds: FiniteFloat = Field(default=1800, gt=0, le=1800)
    maximum_calls: int = Field(default=500, gt=0, le=500)
    verification_seconds: FiniteFloat = Field(default=300, gt=0, le=300)
    diagnostic_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    profile_seconds: FiniteFloat = Field(default=120, gt=0, le=120)
    profile_calls: int = Field(default=120, gt=0, le=120)


def _project_record(
    task: dict, terminal: dict, ops: dict, reconciliation: dict
) -> dict:
    """Export only needed values; retain raw-operation hashes without probe copies."""
    selected = (
        {"search"}
        if task["kind"] == "affine_continuation"
        else (
            {"baseline", "jacobian", "qualification_summary"}
            | {f"coordinate_{j:03d}" for j in range(task["coordinates"])}
        )
    )
    if task["model_index"] == 3:
        selected = set()  # New precision evidence cannot reuse the older arrays.
    projected, provenance = {}, {}
    for name in sorted(selected & ops.keys()):
        op = ops[name]
        value = first._value(op, partial=True)
        omitted = (
            {"probe_receipts", "base_residual"}
            if name.startswith("coordinate_")
            else set()
        )
        if (
            "base_residual" in omitted
            and value.get("base_residual") != first._value(ops["baseline"])["residual"]
        ):
            raise ValueError("coordinate residual differs from shared baseline")
        projected[name] = {
            "identity": op["identity"],
            "status": op["status"],
            "value": {key: item for key, item in value.items() if key not in omitted},
        }
        provenance[name] = {
            "original_operation_sha256": public.content_sha256(op),
            "original_value_sha256": public.content_sha256(value),
            "projection_sha256": public.content_sha256(projected[name]),
            "omitted_value_fields": sorted(omitted & value.keys()),
        }
    return {
        "source_task": task,
        "source_terminal": terminal,
        "source_terminal_sha256": public.content_sha256(terminal),
        "operations": projected,
        "projection_provenance": provenance,
        "source_reconciliation": reconciliation,
        "raw_operations_and_nested_probes_authenticated": True,
    }


def _source_inputs(source: Path, policy: Policy) -> dict:
    """Authenticate all old terminals, including their nested primitive receipts."""
    plan, data = previous.verify(source, runtime=False)
    settings = previous.Policy.model_validate(plan["policy"])
    records = []
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = source / "results" / task["task_id"]
        ops = previous._operations(
            folder,
            identity,
            previous._allowances(task, settings, plan["training_rows"]),
        )
        terminal = read_seal(folder / "result.json")
        reconciliation = reporting.reconcile(
            terminal, previous._assemble(identity, data, ops), ops
        )
        records.append(_project_record(task, terminal, ops, reconciliation))
    return {
        "protocol": PROTOCOL,
        "source_plan": plan,
        "source_inputs": data,
        "records": records,
        "test_data_opened": False,
    }


def _tasks(data: dict, policy: Policy) -> list[dict]:
    """Recheck frozen source identities without invoking a numerical solver."""
    if data["protocol"] != PROTOCOL or data["test_data_opened"] is not False:
        raise ValueError("search input protocol differs")
    plan, prior = data["source_plan"], data["source_inputs"]
    prior_policy = previous.Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != previous.PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(prior)
        or plan["tasks"] != previous._tasks(prior, prior_policy)
        or len(data["records"]) != len(plan["tasks"])
    ):
        raise ValueError("source plan/inputs/roster differ")
    production = policy.population == "scaled-readout-5"
    if production and (
        public.content_sha256(plan) != SOURCE_PLAN
        or [t["model_index"] for t in plan["tasks"]] != list(SELECTED)
    ):
        raise ValueError("source is not the reviewed five-model continuation")
    tasks = []
    for index, (task, record) in enumerate(
        zip(plan["tasks"], data["records"], strict=True)
    ):
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        if record["source_task"] != task:
            raise ValueError("source task order differs")
        terminal = record["source_terminal"]
        if (
            record["source_terminal_sha256"] != public.content_sha256(terminal)
            or terminal["identity"] != identity
            or record["raw_operations_and_nested_probes_authenticated"] is not True
            or record["source_reconciliation"]["scientific_fields_exact"] is not True
        ):
            raise ValueError("source terminal or authentication differs")
        ops = record["operations"]
        if set(ops) != set(record["projection_provenance"]):
            raise ValueError("source projection roster differs")
        for stage, op in ops.items():
            if op["identity"] != {
                "campaign": identity,
                "stage": stage,
                "allowance_seconds": previous._allowances(
                    task, prior_policy, plan["training_rows"]
                )[stage],
            }:
                raise ValueError("source operation identity differs")
            projection = record["projection_provenance"][stage]
            if (
                projection["projection_sha256"] != public.content_sha256(op)
                or projection["original_operation_sha256"]
                != terminal["operation_receipts"][stage]
            ):
                raise ValueError("source numerical projection differs")
        if production and public.content_sha256(record) != SOURCE_EXPORT_SHA256S[index]:
            raise ValueError("source export differs from reviewed projection")
        kind = (
            "targeted_precision"
            if task["model_index"] == 3 and production
            else "nonlinear_search"
            if task["kind"] == "nonlinear_prerequisites"
            else "timescale_profiles"
        )
        if kind == "nonlinear_search" and not (terminal["qualification"] or {}).get(
            "search_qualified"
        ):
            raise ValueError("nonlinear search requires complete source qualification")
        if kind == "timescale_profiles" and not first._verified(
            {"status": "complete", "value": terminal["retained"] or {}}
        ):
            raise ValueError("timescale diagnostics require a verified source endpoint")
        tasks.append(
            {
                "index": index,
                "task_id": task["task_id"],
                "candidate_id": task["candidate_id"],
                "model_index": task["model_index"],
                "kind": kind,
                "coordinates": task["coordinates"],
                "source_record_sha256": public.content_sha256(record),
                "problem_sha256": task["problem_sha256"],
            }
        )
    return tasks


def prepare(root: Path, source: Path, policy: Policy) -> dict:
    """Create a separate snapshot; never mutate or re-run a source operation."""
    if root.resolve() == source.resolve() or root.resolve().is_relative_to(
        source.resolve()
    ):
        raise ValueError("search root must be separate from source")
    return prepare_inputs(root, _source_inputs(source, policy), policy)


def prepare_inputs(root: Path, data: dict, policy: Policy) -> dict:
    """Seal the same boundary for the real roster and numerical smoke fixtures."""
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
        "profile_endpoints_used_for_selection": False,
        "objective": "original_full_training_reconstruction",
        "training_prefix_reestimated": False,
    }
    if reservation_ceiling(plan) > 13200:
        raise ValueError("search reservations exceed four-hour envelope")
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Authenticate frozen source, objective, policy, and code/runtime identities."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    settings = Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != _tasks(data, settings)
        or any(
            plan[key] is not False
            for key in (
                "test_data_opened",
                "automatic_revision",
                "automatic_promotion",
                "profile_endpoints_used_for_selection",
                "training_prefix_reestimated",
            )
        )
        or plan["live_llm_calls"] != 0
        or plan["objective"] != "original_full_training_reconstruction"
        or reservation_ceiling(plan) > 13200
    ):
        raise ValueError("search plan identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("search source/runtime differs")
    return plan, data


def _allowances(task: dict, policy: Policy) -> dict[str, float]:
    base = {"setup": 120.0}
    if task["kind"] == "targeted_precision":
        return (
            base
            | {
                "replay_BDF": policy.prerequisite_seconds + 60,
                "replay_Radau": policy.prerequisite_seconds + 60,
                "baseline": 60,
                "jacobian": policy.prerequisite_seconds + 60,
                "precision_summary": 60,
            }
            | {
                f"coordinate_{j:03d}": policy.coordinate_seconds + 60
                for j in PRECISION_INDICES
            }
        )
    endpoints = {
        f"{name}_{stage}": seconds + 60
        for name in ("retained", "accepted")
        for stage, seconds in (
            ("verification", policy.verification_seconds),
            ("diagnosis", policy.diagnostic_seconds),
        )
    }
    if task["kind"] == "nonlinear_search":
        return (
            base
            | endpoints
            | {
                "qualification": 120,
                "start_verification": policy.verification_seconds + 60,
                "search": policy.search_seconds + 120,
                "endpoint_verification": policy.verification_seconds + 60,
            }
        )
    return (
        base
        | endpoints
        | {"chart_qualification": policy.prerequisite_seconds + 60}
        | {
            f"profile_{j:02d}_{stage}": seconds
            for j in range(PROFILE_COUNT)
            for stage, seconds in (
                ("search", policy.profile_seconds + 120),
                ("verification", policy.verification_seconds + 60),
            )
        }
    )


def reservation_ceiling(plan: dict) -> float:
    """Inclusive operation ceilings count nested coordinate probes only once."""
    policy = Policy.model_validate(plan["policy"])
    return max(
        (sum(_allowances(t, policy).values()) for t in plan["tasks"]), default=0.0
    )


def _operations(folder, identity, allowances):
    return previous._operations(folder, identity, allowances)


def _retained(record: dict, operations: dict) -> tuple[str, dict | None]:
    """Only an independently verified, qualified training improvement can win."""
    source = record["source_terminal"]["retained"]
    selected = ("source", source)
    start = operations.get("start_verification")
    if first._verified(start) and (
        source is None or start["value"]["training_nmse"] < source["training_nmse"]
    ):
        selected = ("verified_start", start["value"])
    new = operations.get("endpoint_verification")
    if (
        first._value(operations.get("qualification")).get("search_qualified")
        and first._verified(start)
        and first._value(operations.get("search"), partial=True).get("best")
        and first._verified(new)
        and (
            selected[1] is None
            or new["value"]["training_nmse"] < selected[1]["training_nmse"]
        )
    ):
        selected = ("bounded_search", new["value"])
    return selected


def _assemble(identity: dict, data: dict, operations: dict) -> dict:
    task = identity["task"]
    record = data["records"][task["index"]]
    origin, retained = _retained(record, operations)
    precision = first._value(operations.get("precision_summary"))
    search = first._value(operations.get("search"), partial=True)
    profiles = (
        [
            {
                "index": j,
                "search": first._value(
                    operations.get(f"profile_{j:02d}_search"), partial=True
                ),
                "verification": first._value(
                    operations.get(f"profile_{j:02d}_verification")
                ),
                "selected_as_main_endpoint": False,
            }
            for j in range(PROFILE_COUNT)
        ]
        if task["kind"] == "timescale_profiles"
        else []
    )
    status = (
        "not_started"
        if not operations
        else "setup_unavailable"
        if not first._value(operations.get("setup"))
        else "targeted_precision_recorded"
        if precision
        else "precision_unavailable"
        if task["kind"] == "targeted_precision"
        else "timescale_profiles_recorded"
        if any(p["verification"] for p in profiles)
        else "timescale_diagnostic_unavailable"
        if profiles
        else "improved_verified"
        if origin == "bounded_search"
        else "search_no_verified_improvement"
        if search
        else "search_unavailable"
    )
    return {
        "status": "complete",
        "identity": identity,
        "candidate_id": task["candidate_id"],
        "kind": task["kind"],
        "search_status": status,
        "retained_origin": origin,
        "retained": retained,
        "training_reconstruction_nmse": (retained or {}).get("training_nmse"),
        "source_endpoint": record["source_terminal"]["retained"],
        "source_training_prefix_forecast": record["source_terminal"][
            "training_prefix_forecast"
        ],
        "training_prefix_reestimated": False,
        "precision": precision or None,
        "qualification": first._value(operations.get("qualification")) or None,
        "chart_qualification": first._value(operations.get("chart_qualification"))
        or None,
        "retained_verification": first._value(operations.get("retained_verification"))
        or None,
        "retained_assessment": first._value(operations.get("retained_diagnosis"))
        or None,
        "accepted_verification": first._value(operations.get("accepted_verification"))
        or None,
        "accepted_assessment": first._value(operations.get("accepted_diagnosis"))
        or None,
        "profiles": profiles,
        "profiles_are_finite_budget_upper_envelopes": True,
        "profile_endpoints_used_for_selection": False,
        "search_stop_reason": search.get("stop_reason"),
        "search_budget_exhausted": search.get("budget_exhausted"),
        "operation_statuses": {k: v["status"] for k, v in operations.items()},
        "operation_receipts": {
            k: public.content_sha256(v) for k, v in operations.items()
        },
        "cost": reporting.canonical_cost(operations),
        "inherited_cost": record["source_terminal"]["cost"],
        "test_data_opened": False,
        "live_llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
        "global_minimum_certified": False,
        "parameter_confidence_percentage": None,
    }


class _Run:
    """One locked task's durable operations and lazily compiled numerical views."""

    def __init__(self, root, plan, data, index):
        self.plan, self.data = plan, data
        self.task, self.record = plan["tasks"][index], data["records"][index]
        self.policy = Policy.model_validate(plan["policy"])
        self.portfolio = data["source_inputs"]["portfolio"]
        model = self.portfolio["models"][self.task["model_index"]]
        self.problem = PrefixFitProblem.model_validate(model["problem"])
        self.training = PublicSplit.model_validate(self.portfolio["training"])
        self.identity = {"plan_sha256": public.content_sha256(plan), "task": self.task}
        self.folder = root / "results" / self.task["task_id"]
        self.allowances = _allowances(self.task, self.policy)
        self.ops = _operations(self.folder, self.identity, self.allowances)
        self.original = self.oracle = None

    def perform(self, stage, work):
        self.ops[stage] = journal._operation(
            self.folder, self.identity, stage, self.allowances[stage], work
        )
        return first._value(self.ops[stage])

    def setup(self, save):
        if self.task["kind"] == "targeted_precision":
            self.oracle = nonlinear.make_precision_oracle(self.problem, self.training)
            self.original = self.oracle.original
        elif self.task["kind"] == "nonlinear_search":
            self.oracle = nonlinear.make_oracle(self.problem, self.training)
            self.original = self.oracle.original
        else:
            settings = first.Policy.model_validate(
                self.data["source_inputs"]["original_policy"]
            ).numerical
            # The source anchor preserves the old chart/history identities exactly.
            anchor = self.data["source_inputs"]["records"][self.task["index"]][
                "source_record"
            ]["endpoint"]["parameters"]
            self.original, self.oracle = polish.make_oracle(
                self.problem, self.training, settings, anchor
            )
        return previous._saved(
            {
                "parameter_names": list(self.oracle.names),
                "optimizer_coordinates": len(self.oracle.names),
                "audit": self.oracle.portfolio_audit,
            },
            save,
        )

    def precision(self):
        p, oracle = self.policy, self.oracle
        replays = {
            method: self.perform(
                "replay_" + method,
                lambda save, method=method: nonlinear.precision_replay(
                    oracle, method, p.prerequisite_seconds, save
                ),
            )
            for method in ("BDF", "Radau")
        }
        if not all(r.get("passed") for r in replays.values()):
            return
        baseline = self.perform(
            "baseline",
            lambda save: previous._saved(
                nonlinear.precision_baseline(oracle, replays["BDF"], replays["Radau"]),
                save,
            ),
        )
        if not baseline.get("passed"):
            return
        jacobian = self.perform(
            "jacobian",
            lambda save: nonlinear.precision_jacobian(
                oracle, baseline, p.prerequisite_seconds, save
            ),
        )
        if not jacobian.get("passed"):
            return
        for j in PRECISION_INDICES:

            def work(save, j=j):
                def probe(key, vector, run):
                    del vector
                    import re

                    if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", key):
                        raise ValueError("unsafe primitive key")
                    op = journal._operation(
                        self.folder / "probes" / f"coordinate_{j:03d}",
                        self.identity,
                        key,
                        120,
                        run,
                    )
                    if op["status"] != "complete" or not op.get("value"):
                        raise ValueError("precision probe unavailable")
                    return op["value"]

                result = nonlinear.precision_coordinate(
                    oracle,
                    baseline,
                    jacobian,
                    j,
                    p.coordinate_seconds,
                    save,
                    probe_runner=probe,
                )
                result["primitive_receipts"] = {
                    key: public.content_sha256(value)
                    for key, value in previous._probe_operations(
                        self.folder, self.identity, j
                    ).items()
                }
                return previous._saved(result, save)

            self.perform(f"coordinate_{j:03d}", work)
        self.perform(
            "precision_summary",
            lambda save: previous._saved(
                nonlinear.precision_summary(
                    oracle,
                    [
                        first._value(self.ops.get(f"coordinate_{j:03d}"))
                        for j in PRECISION_INDICES
                        if first._value(self.ops.get(f"coordinate_{j:03d}"))
                    ],
                ),
                save,
            ),
        )

    def search(self):
        p, oracle = self.policy, self.oracle
        source = self.record["operations"]

        def authenticate():
            return nonlinear.authenticate_qualification(
                oracle,
                first._value(source["baseline"]),
                first._value(source["jacobian"]),
                [
                    first._value(source[f"coordinate_{j:03d}"])
                    | {"base_residual": first._value(source["baseline"])["residual"]}
                    for j in range(self.task["coordinates"])
                ],
                first._value(source["qualification_summary"]),
            )

        qualified = self.perform(
            "qualification", lambda save: previous._saved(authenticate(), save)
        )
        if not qualified.get("search_qualified"):
            return
        # A resumed journal does not reconstruct in-memory imported caches itself.
        if authenticate() != qualified:
            raise ValueError("imported qualification changed on resume")
        self.perform(
            "start_verification",
            lambda save: nonlinear.verify_endpoint(
                oracle,
                oracle.start,
                p.verification_seconds,
                save,
                expected_nmse=qualified["start_training_nmse"],
            ),
        )
        if first._verified(self.ops.get("start_verification")):
            self.perform(
                "search",
                lambda save: nonlinear.search(
                    oracle,
                    oracle.start,
                    p.search_seconds,
                    p.maximum_calls,
                    save,
                    qualification=qualified,
                ),
            )
            search = first._value(self.ops.get("search"), partial=True)
            if search.get("best"):
                self.perform(
                    "endpoint_verification",
                    lambda save: nonlinear.verify_endpoint(
                        oracle,
                        search["best"]["parameters"],
                        p.verification_seconds,
                        save,
                        expected_nmse=search["best"].get("training_nmse"),
                    ),
                )
        self.endpoints(first._value(self.ops.get("search"), partial=True))

    def endpoints(self, history):
        """Verify and diagnose best and last accepted points without mixing history."""
        p, oracle, original = self.policy, self.oracle, self.original
        _, retained = _retained(self.record, self.ops)
        retained_parameters = (
            coordinates.to_effective(original, oracle, retained["parameters"])
            if retained
            else None
        )
        for name, parameters in (
            ("retained", retained_parameters),
            ("accepted", history.get("last_accepted_parameters")),
        ):
            if not parameters:
                continue
            verification = self.perform(
                name + "_verification",
                lambda save, parameters=parameters: numerical.verify_endpoint(
                    original, oracle, parameters, p.verification_seconds, save
                ),
            )
            if first._verified(self.ops.get(name + "_verification")):
                self.perform(
                    name + "_diagnosis",
                    lambda save,
                    parameters=parameters,
                    verification=verification: numerical.diagnose(
                        original,
                        oracle,
                        parameters,
                        verification,
                        history,
                        p.diagnostic_seconds,
                        save,
                    ),
                )

    def timescales(self):
        from autoformalism.fitting import prefix_portfolio_timescale as timescale

        p = self.policy
        self.endpoints(
            first._value(self.record["operations"].get("search"), partial=True)
        )
        settings = first.Policy.model_validate(
            self.data["source_inputs"]["original_policy"]
        ).numerical
        compiled = None

        def chart_oracles():
            nonlocal compiled
            if compiled is None:
                compiled = timescale.make_oracle(
                    self.problem,
                    self.training,
                    settings,
                    self.record["source_terminal"]["retained"]["parameters"],
                )
            return compiled

        def qualify(save):
            original, chart = chart_oracles()
            audit = {
                "chart_audit": chart.portfolio_audit,
                "profile_grid": timescale.profile_grid(chart),
            }
            result = timescale.chart_equivalence(
                original,
                chart,
                p.prerequisite_seconds,
                lambda value: save(value | audit),
            )
            return result | audit

        qualified = self.perform(
            "chart_qualification",
            qualify,
        )
        if not qualified.get("passed"):
            return
        original, chart = chart_oracles()
        grid = timescale.profile_grid(chart)
        if len(grid) != PROFILE_COUNT:
            raise ValueError("timescale profile grid differs")
        for j, spec in enumerate(grid):
            stage = f"profile_{j:02d}"
            self.perform(
                stage + "_search",
                lambda save, spec=spec: timescale.profile_search(
                    chart, spec, p.profile_seconds, p.profile_calls, save
                ),
            )
            search = first._value(self.ops.get(stage + "_search"), partial=True)
            if search.get("best"):
                self.perform(
                    stage + "_verification",
                    lambda save, search=search: numerical.verify_endpoint(
                        original,
                        chart,
                        search["best"]["parameters"],
                        p.verification_seconds,
                        save,
                    ),
                )


def run_task(root: Path, index: int) -> dict:
    """Spend only unused operations and preserve every historical endpoint."""
    plan, data = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown search task index")
    folder = root / "results" / plan["tasks"][index]["task_id"]
    with public._lock(folder):
        run = _Run(root, plan, data, index)
        if (folder / "result.json").exists():
            saved = read_seal(folder / "result.json")
            reporting.reconcile(saved, _assemble(run.identity, data, run.ops), run.ops)
            return saved
        ready = run.perform("setup", run.setup)
        if ready:
            if run.oracle is None:
                # Rebuild only deterministic objects, never renew setup's allowance.
                run.setup(lambda value: None)
            if run.task["kind"] == "targeted_precision":
                run.precision()
            elif run.task["kind"] == "nonlinear_search":
                run.search()
            else:
                run.timescales()
        result = _assemble(run.identity, data, run.ops)
        seal(folder / "result.json", result)
        return result


def report(root: Path) -> dict:
    """Rebuild summaries solely from receipts; no refits or numerical diagnostics."""
    plan, data = verify(root, runtime=False)
    policy = Policy.model_validate(plan["policy"])
    rows, audits = [], {}
    for task in plan["tasks"]:
        folder = root / "results" / task["task_id"]
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        ops = _operations(folder, identity, _allowances(task, policy))
        result = _assemble(identity, data, ops)
        terminal = (folder / "result.json").exists()
        if terminal:
            audits[task["task_id"]] = reporting.reconcile(
                read_seal(folder / "result.json"), result, ops
            )
        rows.append(
            result
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
            "Tighter checks on four model03 directions are not "
            "full-model qualification.",
            "Search qualification is local; it does not establish convergence "
            "or a global minimum.",
            "Timescale profiles are finite-budget nuisance-refit upper envelopes, "
            "not confidence intervals.",
            "The stable positive-timescale chart is a conditional diagnostic; "
            "the full original family is unchanged.",
            "Previously reported prefix scores belong to the source endpoints; "
            "new endpoints need separate prefix evaluation.",
            "Best and accepted endpoints use only their own matching histories.",
            "Inherited costs are reported separately; "
            "interrupted new allowances remain spent.",
        ],
    }
    with public._lock(root / "report"):
        public._write(root / "summary.json", summary)
        lines = [
            "# Construction portfolio search",
            "",
            f"{recorded}/{len(rows)} terminal tasks.",
            "",
            "| Model | Status | Training reconstruction NMSE |",
            "| --- | --- | ---: |",
        ]
        lines += [
            f"| {r['candidate_id']} | {r['search_status']} | "
            f"{r['training_reconstruction_nmse']} |"
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
                raise ValueError("inspection source escapes search root")
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
                raise ValueError("inspection log escapes search root")
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
