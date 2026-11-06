"""Bounded fitting of the other 22 authenticated construction candidates.

This campaign keeps every model, including candidates whose declared starting
rollouts failed. It reuses independently verified endpoints and never renews a
started numerical reservation. Training reconstruction and forecasting remain
separate experiments.
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
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import construction_portfolio_fitting as first
from autoformalism.research import construction_portfolio_residual as retention
from autoformalism.research import continuation_report_cost as reporting
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem, PrefixPolicy
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-remaining-portfolio-1"
INPUT_PROTOCOL = "construction-remaining-portfolio-inputs-1"
AFFINE = (0, 5, 6, 7, 8, 9, 11, 12, 16, 18, 19, 20, 21, 22)
RECOVERY = (1, 2, 4, 13, 17)
ROSTER = tuple(i for i in range(23) if i != 15)
BASE_PLAN = "8498d0d08631ed81543ef1f4a96eb3471b1017f295a06ccb72da7bbd02961725"
BASE_INPUTS = "51fe5b40a4759860780b01b5b0a8d5b91e60816f1bc77617e73905fb370bdaea"
TARGETED_PLAN = "970eac91f92c978ec6250e4e010f6271257acc2957481bc5de1c6b1d3f3b7067"
CONDITIONAL_SUMMARY = "7689a05ebfedb1cb47d3a9c8934881604f348ba6275ded9047a5b2cd8ecc52d0"
SOURCE_TRAINING = retention.SOURCE_TRAINING
# The production projection is generated from the explicitly reviewed sources.
# Pinning its complete content also authenticates warm values and source receipts.
SOURCE_INPUTS = "f58984f7b01bcf3edf942cd6eb646ebcaf4a6c6276c2030bc0a489f20a8e5f52"
BASE_TERMINALS = (
    "5f83f090f3c32147dc0b5cb824ebff8cb85c7f72a4e84224dbca2c894c6250eb",
    "bc17dd0c501bc3ca10369810f62d140937385e550b6df5db4a4279f1dc50a534",
    "28333f437c202aaa92e49d21e7daf245efa4d708b700c1bbfc564d6cb13db8e4",
    "c03f4f59a4964f7d367f3b341a84bba30f7a903dff9e5f8f6081e24e03335b0a",
    "e7a15e2909f266d3346d5e0546c1e88e863105f3cb1dbcfebbd9b2349183ab19",
    "4fb649a6578157a857dd8230907d337e928a4ec7bb022531fa80f2e401d03f10",
    "991b1255bba7b51f3f5906f3bf35c4899331dd14fbbdac2e4f8ed533c3497c6b",
    "e07fa3ac26f4eb913712a3a6e0247b214ba0ead73ece64515de5b1e729b24516",
    "736e61f0894e276e5e5634804924a8cf7dd322bc75a8e7188f9b0fa4b722f7b8",
    "ade1930f50a76cdab783115889afa265cdfd4272a3f55321a8cd7d67594e1f8f",
    "c7d5d558364043cad0fae42f981ac8214bd2a6e63727d101816bbda5cc2a53b3",
    "eb708282c6a5e7884e469e2052a090a128f7dd3e5ccdb2ce26a084ed90fabcf3",
    "c363d29f2c740d795708376bef9434b4ebe3b97bfe2884342abbdc13f8749222",
    "ab1f19bb5b7db8345d2aea072fe549208c0dd18afea7e79bd58da9b825e74c84",
    "32e78a11899c3e692d15b8ecf4a2e3a4257930369da1ae3a688dc0dcb5c45921",
    "d190c96f51e3b95668c464218cb3601542510df90b084343149ff27d4774a7c6",
    "1bc872fe971300f09b2b5137b56e71c3b1969f011ff37156298902953ca4248e",
    "7a5f5a2e2200be530bd2eb4d1e70e363ef53bf7930a2563da25091a18698bbe2",
    "f7aa8aad782a9df15bb5d9824d0bb7ca019277a5c04b727780e691f55bc482e6",
    "e6956ca82a9c166b7bb1c3bfc38126755e4ac05690c4c8dbdcbe7e01bb6505d9",
    "f8e404a4dc5ae417c2d8f56dd18ba5e4ba1b1d1f572869848152cd9f2afd9870",
    "686f2973390ad4ba8e8d37109b57c4c915fb1ed0286ccbf0cb42c74d844fdaab",
    "d0e3b01b8c2b036dbb871e1eec3cc01b0163f28a999dd055eceaca87679ff73d",
)
TARGETED_TERMINALS = {
    "candidate-10-saturation-saved": (
        "999c1995c598f2275af7a9cdbd4d0b0b68b0987c505e693a59e35e3ef2906890"
    ),
    "candidate-12-free-control": (
        "125df40eaf196a06d978a1b0f531ac6c45c45088dbd87728d8afd059d7faea53"
    ),
    "candidate-12-decay-zero": (
        "edfa015b4bad0989bc729c511bdaf14c19bf63576fa4c2e11c4af7391bea09e3"
    ),
    "candidate-14-saturation-saved": (
        "ae2376ec3da53d47a23022b091549fb5c52c597ffe46025d48b44658a330fdf1"
    ),
}


class Policy(StrictSchema):
    """Explicit per-start search allowances and separately charged replays."""

    population: Literal["scaled-readout-remaining-22", "synthetic-smoke"] = (
        "scaled-readout-remaining-22"
    )
    qualification_seconds: FiniteFloat = Field(default=120, gt=0, le=180)
    search_seconds: FiniteFloat = Field(default=900, gt=0, le=900)
    recovery_search_seconds: FiniteFloat = Field(default=1800, gt=0, le=1800)
    maximum_calls: int = Field(default=1200, ge=1, le=2000)
    verification_seconds: FiniteFloat = Field(default=180, gt=0, le=300)
    feedback_seconds: FiniteFloat = Field(default=180, gt=0, le=180)
    persistence_headroom_seconds: FiniteFloat = Field(default=900, ge=0, le=900)
    nonlinear_rtol: FiniteFloat = Field(default=1e-11, gt=0, le=1e-11)
    nonlinear_atol: FiniteFloat = Field(default=1e-13, gt=0, le=1e-13)
    improvement_absolute_tolerance: FiniteFloat = Field(default=1e-10, ge=0)
    improvement_relative_tolerance: FiniteFloat = Field(default=1e-8, ge=0)


def _finite_endpoint(value: dict | None) -> bool:
    """Accept only finite independently verified original-equation endpoints."""
    return bool(
        isinstance(value, dict)
        and value.get("original_equations_verified") is True
        and value.get("status", "complete") == "complete"
        and type(value.get("training_nmse")) in (int, float)
        and np.isfinite(value["training_nmse"])
        and value["training_nmse"] >= 0
        and isinstance(value.get("parameters"), dict)
        and all(
            type(x) in (int, float) and np.isfinite(x)
            for x in value["parameters"].values()
        )
    )


def _pinned(path: Path, expected: str) -> dict:
    value = read_seal(path)
    if public.content_sha256(value) != expected:
        raise ValueError(f"reviewed source digest differs: {path}")
    return value


def make_intake(base: Path, targeted: Path, conditional: Path) -> dict:
    """Project exact reviewed sources into one portable training-only input."""
    base_plan, data = first.verify(base, runtime=False)
    if (
        public.content_sha256(base_plan) != BASE_PLAN
        or public.content_sha256(data) != BASE_INPUTS
    ):
        raise ValueError("reviewed original portfolio differs")
    targeted_plan = _pinned(targeted / "plan.json", TARGETED_PLAN)
    target_tasks = {t["task_id"]: t for t in targeted_plan["tasks"]}
    summary = _pinned(conditional / "verified-results.json", CONDITIONAL_SUMMARY)
    records = []
    for index in ROSTER:
        model = data["models"][index]
        cid = f"candidate-{index:02d}"
        if model["candidate_id"] != cid:
            raise ValueError("original model order differs")
        terminal = _pinned(
            base / "results" / cid / "result.json", BASE_TERMINALS[index]
        )
        task = base_plan["tasks"][index]
        if terminal["identity"] != {"plan_sha256": BASE_PLAN, "task": task}:
            raise ValueError("original terminal identity differs")
        provenance = {
            "base_plan_sha256": BASE_PLAN,
            "base_terminal_sha256": BASE_TERMINALS[index],
            "source_packet_sha256": model["source_packet_sha256"],
            "base_fitting_status": terminal["fitting_status"],
        }
        endpoints = {}
        if _finite_endpoint(terminal.get("retained")):
            endpoints["original_portfolio"] = terminal["retained"]
        anchor = terminal.get("retained")
        for label in TARGETED_TERMINALS:
            if not label.startswith(cid + "-"):
                continue
            target = _pinned(
                targeted / "results" / label / "result.json",
                TARGETED_TERMINALS[label],
            )
            if (
                target["identity"]
                != {
                    "plan_sha256": TARGETED_PLAN,
                    "task": target_tasks[label],
                }
                or target_tasks[label]["problem_sha256"]
                != model["fitting_problem_sha256"]
            ):
                raise ValueError("targeted source belongs to another model")
            endpoint = target["trial_endpoint"] if index == 12 else target["retained"]
            if not _finite_endpoint(endpoint):
                raise ValueError("targeted source has no verified endpoint")
            endpoints[label] = endpoint
            provenance[label] = TARGETED_TERMINALS[label]
            if not label.endswith("decay-zero"):
                anchor = endpoint
        if index in AFFINE:
            entry = summary["rows"][cid]
            receipt_folder = (
                conditional / cid / "operations" / entry["verification_stage"]
            )
            receipt = _pinned(
                receipt_folder / "result.json", entry["verification_receipt"]
            )
            journal._read_operation(receipt_folder, receipt["identity"])
            qualification_folder = (
                conditional / cid / "operations" / "conditional_roots"
            )
            qualified = _pinned(
                qualification_folder / "result.json", entry["qualification_receipt"]
            )
            journal._read_operation(qualification_folder, qualified["identity"])
            endpoint = receipt["value"]
            if (
                receipt["status"] != "complete"
                or not _finite_endpoint(endpoint)
                or qualified["status"] != "complete"
                or qualified["value"].get("passed") is not True
                or endpoint["training_nmse"] != entry["new"]
                or qualified["identity"]["campaign"]["terminal"] != entry["source"]
                or qualified["identity"]["campaign"]["model"]
                != model["fitting_problem_sha256"]
                or endpoint["parameters"] != qualified["value"]["original_parameters"]
            ):
                raise ValueError("conditional initial-state evidence differs")
            provenance["conditional_summary_sha256"] = CONDITIONAL_SUMMARY
            provenance["conditional_verification_receipt"] = entry[
                "verification_receipt"
            ]
            provenance["conditional_qualification_receipt"] = entry[
                "qualification_receipt"
            ]
            endpoints["conditional_roots"] = anchor = endpoint
        records.append(
            {
                "candidate_id": cid,
                "model_index": index,
                "family": model["family"],
                "problem": model["problem"],
                "problem_sha256": model["fitting_problem_sha256"],
                "route": "affine" if index in AFFINE else "nonlinear",
                "recovery": index in RECOVERY,
                "anchor": anchor,
                "incumbents": endpoints,
                "provenance": provenance,
                "inherited_cost": terminal["cost"],
            }
        )
    return {
        "protocol": INPUT_PROTOCOL,
        "training": data["training"],
        "prefix": data["prefix"],
        "numerical": first.Policy.model_validate(
            base_plan["policy"]
        ).numerical.model_dump(mode="json"),
        "records": records,
        "source": {
            "base_plan_sha256": BASE_PLAN,
            "base_inputs_sha256": BASE_INPUTS,
            "targeted_plan_sha256": TARGETED_PLAN,
            "conditional_summary_sha256": CONDITIONAL_SUMMARY,
        },
        "test_data_opened": False,
        "validation_exported": False,
    }


def _tasks(data: dict, policy: Policy) -> list[dict]:
    """Validate exact public boundaries, candidate identities and warm endpoints."""
    if (
        set(data)
        != {
            "protocol",
            "training",
            "prefix",
            "numerical",
            "records",
            "source",
            "test_data_opened",
            "validation_exported",
        }
        or data["protocol"] != INPUT_PROTOCOL
    ):
        raise ValueError("remaining portfolio input schema differs")
    if (
        data["test_data_opened"] is not False
        or data["validation_exported"] is not False
    ):
        raise ValueError("only public training inputs are permitted")
    production = policy.population == "scaled-readout-remaining-22"
    if production and public.content_sha256(data) != SOURCE_INPUTS:
        raise ValueError("reviewed remaining-22 projection differs")
    training = PublicSplit.model_validate(data["training"])
    if training.name != "train" or not training.rows:
        raise ValueError("nonempty public training split required")
    kernel.NumericalPolicy.model_validate(data["numerical"])
    PrefixPolicy.model_validate(data["prefix"])
    if production and (
        public.content_sha256(data["training"]) != SOURCE_TRAINING
        or tuple(r["model_index"] for r in data["records"]) != ROSTER
    ):
        raise ValueError("remaining-22 roster or training differs")
    tasks, names = [], set()
    for index, record in enumerate(data["records"]):
        cid = record["candidate_id"]
        if (
            not re.fullmatch(r"candidate-\d{2}", cid)
            or cid in names
            or cid == "candidate-15"
        ):
            raise ValueError("unsafe, duplicated or excluded candidate")
        names.add(cid)
        problem = PrefixFitProblem.model_validate(record["problem"])
        if public.content_sha256(problem) != record["problem_sha256"]:
            raise ValueError("model problem digest differs")
        if any(set(r.targets) != set(problem.context.targets) for r in training.rows):
            raise ValueError("model and measured target contracts differ")
        if (
            record["route"] not in {"affine", "nonlinear"}
            or type(record["recovery"]) is not bool
        ):
            raise ValueError("invalid numerical route")
        if record["recovery"] and record["route"] != "nonlinear":
            raise ValueError("affine recovery route is not scheduled")
        if record["anchor"] is not None and not _finite_endpoint(record["anchor"]):
            raise ValueError("unverified source anchor")
        if any(not _finite_endpoint(ep) for ep in record["incumbents"].values()):
            raise ValueError("unverified source incumbent")
        tasks.append(
            {
                "index": index,
                "task_id": cid,
                "candidate_id": cid,
                "model_index": record["model_index"],
                "problem_sha256": record["problem_sha256"],
                "record_sha256": public.content_sha256(record),
                "route": record["route"],
                "recovery": record["recovery"],
                "start_count": 5 if record["recovery"] else 2,
                "maximum_search_starts": 1 if record["recovery"] else 2,
                "search_phases": ["profile"]
                if record["route"] == "affine"
                else ["block_00", "block_01", "block_02", "release"],
            }
        )
    if not tasks:
        raise ValueError("empty remaining portfolio")
    return tasks


def prepare(root: Path, source: Path, policy: Policy) -> dict:
    """Freeze a compact sealed intake in a new independent campaign directory."""
    is_directory = source.is_dir()
    path = source / "inputs.json" if is_directory else source
    if path.resolve().is_relative_to(root.resolve()) or (
        is_directory and root.resolve().is_relative_to(source.resolve())
    ):
        raise ValueError("campaign root must be separate from source intake")
    return prepare_inputs(root, read_seal(path), policy)


def prepare_inputs(root: Path, data: dict, policy: Policy) -> dict:
    """Seal numerical budgets and source identities before any new fit."""
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
        "objective": "original_full_training_reconstruction",
        "training_prefix_reestimated": False,
        "prefix_forecast_status": "deferred_separate_frozen_parameter_diagnostic",
        "selection_data": "public_training_only",
    }
    if reservation_ceiling(plan) > 13200:
        raise ValueError("reservations exceed four-hour worker envelope")
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Authenticate the whole new campaign without accessing any remote source."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    policy = Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != _tasks(data, policy)
        or plan["objective"] != "original_full_training_reconstruction"
        or plan["selection_data"] != "public_training_only"
        or plan["prefix_forecast_status"]
        != "deferred_separate_frozen_parameter_diagnostic"
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
        raise ValueError("remaining portfolio plan identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("remaining portfolio source/runtime differs")
    return plan, data


def _allowances(task: dict, policy: Policy) -> dict[str, float]:
    """Reserve each phase once, including numerical and persistence overhead."""
    p = policy
    result = {
        "setup": 180,
        "source_verification": p.verification_seconds + 60,
        "feedback": p.feedback_seconds + 60,
    }
    for index in range(task["start_count"]):
        prefix = f"start_{index:02d}"
        result[prefix + "_qualification"] = p.qualification_seconds + 120
        result[prefix + "_verification"] = p.verification_seconds + 60
    for index in range(task["maximum_search_starts"]):
        prefix = f"fit_{index:02d}"
        seconds = p.recovery_search_seconds if task["recovery"] else p.search_seconds
        for phase in task["search_phases"]:
            search = seconds / len(task["search_phases"])
            overhead = (
                p.persistence_headroom_seconds + 60
                if task["route"] == "affine"
                else 120
            )
            result[prefix + "_" + phase] = search + overhead
            result[prefix + "_" + phase + "_verification"] = p.verification_seconds + 60
    return result


def reservation_ceiling(plan: dict) -> float:
    p = Policy.model_validate(plan["policy"])
    return max((sum(_allowances(t, p).values()) for t in plan["tasks"]), default=0)


def _operations(folder: Path, identity: dict, allowances: dict) -> dict:
    """Read immutable receipts and charge started operations without retrying."""
    result = {}
    for directory in sorted((folder / "operations").glob("*")):
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or directory.name not in allowances
        ):
            raise ValueError("unknown remaining portfolio operation")
        bound = {
            "campaign": identity,
            "stage": directory.name,
            "allowance_seconds": allowances[directory.name],
        }
        if not (directory / "started.json").exists():
            if any(
                (directory / name).exists() for name in ("result.json", "progress.json")
            ):
                raise ValueError("operation has no reservation")
            continue
        if read_seal(directory / "started.json") != bound:
            raise ValueError("operation reservation differs")
        if (directory / "result.json").exists():
            result[directory.name] = journal._read_operation(directory, bound)
        else:
            progress = (
                public._read(directory / "progress.json")
                if (directory / "progress.json").exists()
                else {}
            )
            result[directory.name] = {
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
    return result


def _retain(
    record: dict, operations: dict, policy: Policy
) -> tuple[str | None, dict | None, dict]:
    """Require original replay before any trial can displace a saved incumbent."""
    selected, origin, comparisons = None, None, {}
    candidates = [("source:" + name, ep) for name, ep in record["incumbents"].items()]
    candidates += [
        (name, first._value(op))
        for name, op in sorted(operations.items())
        if name.endswith("_verification") and first._verified(op)
    ]
    for name, endpoint in candidates:
        if not _finite_endpoint(endpoint):
            continue
        comparison = (
            retention.compare_endpoints(selected, endpoint, policy)
            if selected
            else None
        )
        comparisons[name] = comparison
        if selected is None or comparison["classification"] == "meaningful_improvement":
            selected, origin = endpoint, name
    return origin, selected, comparisons


def _assemble(identity: dict, record: dict, operations: dict, policy: Policy) -> dict:
    """Build honest model-level outcomes and separate the two kinds of stopping."""
    origin, retained, comparisons = _retain(record, operations, policy)
    searches = {}
    for name, op in sorted(operations.items()):
        if not name.startswith("fit_") or name.endswith("_verification"):
            continue
        value = first._value(op, partial=True)
        searches[name] = {
            "status": op["status"],
            "numerical_status": value.get("status"),
            "error": value.get("error"),
            "stop_reason": value.get("stop_reason"),
            "optimizer_success": value.get("optimizer_success"),
            "optimizer_raw_success": value.get("optimizer_raw_success"),
            "optimizer_terminal": value.get("optimizer_terminal"),
            "finite_terminal_qualified": (value.get("optimizer_terminal") or {}).get(
                "feasible_evaluation_verified"
            ),
            "optimizer_iterations": value.get("optimizer_iterations"),
            "initial_sweep_completed": value.get("initial_sweep_completed"),
            "budget_exhausted": value.get("budget_exhausted"),
            "calls_attempted": value.get("search_calls_attempted"),
            "calls_started": value.get("search_calls_started"),
            "calls_completed": value.get("search_calls_completed"),
            "calls_failed": value.get("search_calls_failed"),
            "best_training_nmse": (value.get("best") or {}).get("training_nmse"),
        }
    starts = {
        name: first._value(op, partial=True)
        for name, op in sorted(operations.items())
        if name.endswith("_qualification")
    }
    status = (
        "not_started"
        if not operations
        else "setup_unavailable"
        if not first._value(operations.get("setup"))
        else "bounded_search_completed"
        if searches and all(op["status"] == "complete" for op in searches.values())
        else "bounded_search_interrupted_or_unavailable"
        if searches
        else "no_qualified_start"
        if not retained
        else "verified_incumbent_preserved"
    )
    return {
        "status": "complete",
        "identity": identity,
        "candidate_id": record["candidate_id"],
        "route": record["route"],
        "recovery": record["recovery"],
        "family": record["family"],
        "search_status": status,
        "retained": retained,
        "retained_origin": origin,
        "training_reconstruction_nmse": retained.get("training_nmse")
        if retained
        else None,
        "per_target_normalized_mse": retained.get("per_target_normalized_mse")
        if retained
        else None,
        "source_anchor_nmse": (record["anchor"] or {}).get("training_nmse"),
        "endpoint_comparisons": comparisons,
        "starts": starts,
        "searches": searches,
        "setup": first._value(operations.get("setup")),
        "feedback": first._value(operations.get("feedback")),
        "source_provenance": record["provenance"],
        "inherited_cost": record["inherited_cost"],
        "inherited_cost_scope": (
            "Original portfolio only. Later source costs overlap and are not summed."
        ),
        "operation_statuses": {k: v["status"] for k, v in sorted(operations.items())},
        "operation_receipts": {
            k: public.content_sha256(v) for k, v in sorted(operations.items())
        },
        "cost": reporting.canonical_cost(operations),
        "cost_exclusions": [
            "Source/runtime hashing, reporting, scheduler wait, "
            "and adapter rebuild on resume."
        ],
        "training_prefix_reestimated": False,
        "prefix_forecast_status": "deferred_separate_frozen_parameter_diagnostic",
        "optimization_converged": False,
        "global_minimum_certified": False,
        "parameter_confidence_established": False,
        "equation_inadequacy_certified": False,
        "automatic_revision": False,
        "test_data_opened": False,
        "live_llm_calls": 0,
    }


class _Run:
    """Keep each start and search phase independently resumable and verified."""

    def __init__(self, root: Path, plan: dict, data: dict, index: int):
        self.task = plan["tasks"][index]
        self.record = data["records"][index]
        self.policy = Policy.model_validate(plan["policy"])
        self.problem = PrefixFitProblem.model_validate(self.record["problem"])
        self.training = PublicSplit.model_validate(data["training"])
        self.numerical = kernel.NumericalPolicy.model_validate(data["numerical"])
        if self.task["route"] == "nonlinear":
            self.numerical = self.numerical.model_copy(
                update={
                    "rtol": min(self.numerical.rtol, self.policy.nonlinear_rtol),
                    "atol": min(self.numerical.atol, self.policy.nonlinear_atol),
                }
            )
        self.identity = {"plan_sha256": public.content_sha256(plan), "task": self.task}
        self.folder = root / "results" / self.task["task_id"]
        self.allowances = _allowances(self.task, self.policy)
        self.ops = _operations(self.folder, self.identity, self.allowances)
        self.original = self.oracle = self.start = None

    def perform(self, name, work):
        self.ops[name] = journal._operation(
            self.folder, self.identity, name, self.allowances[name], work
        )
        return first._value(self.ops[name])

    def setup(self, save):
        """Freeze the coordinate chart, all seed vectors and block identities."""
        from autoformalism.fitting import remaining_portfolio_search as search

        anchor = self.record["anchor"]
        self.original, self.oracle, self.start = search.setup(
            self.problem,
            self.training,
            self.numerical,
            anchor["parameters"] if anchor else None,
            self.task["route"],
        )
        starts = search.deterministic_starts(
            self.oracle, self.start, self.task["route"], recovery=self.task["recovery"]
        )[: self.task["start_count"]]
        blocks = search.shared_blocks(self.oracle, size=3)
        if self.task["route"] == "nonlinear" and len(blocks) > 3:
            raise ValueError("shared coefficient roster exceeds three frozen blocks")
        result = {
            "passed": True,
            "starts": starts,
            "blocks": {f"block_{i:02d}": block for i, block in enumerate(blocks)},
            "block_policy": "all_shared_coordinate_blocks_then_all_coordinates",
            "names": list(self.oracle.names),
            "start": self.start,
            "original_coordinates": len(self.original.names),
            "optimizer_coordinates": len(self.oracle.names),
            "coordinate_audit": self.oracle.portfolio_audit,
            "numerical_policy": self.numerical.model_dump(mode="json"),
            "source_anchor_sha256": public.content_sha256(anchor),
            "training_sha256": public.content_sha256(self.training),
            "restarts_use_training_errors": False,
        }
        save(result)
        return result

    def verify_endpoint(self, name, parameters):
        from autoformalism.fitting import remaining_portfolio_search as search

        return self.perform(
            name,
            lambda save: search.verify(
                self.original,
                self.oracle,
                parameters,
                self.policy.verification_seconds,
                save,
            ),
        )

    def fit_start(self, index: int, endpoint: dict, setup: dict):
        """Retain the best verified phase before rotating or releasing variables."""
        from autoformalism.fitting import remaining_portfolio_search as search

        selected = endpoint
        task, p = self.task, self.policy
        seconds = p.recovery_search_seconds if task["recovery"] else p.search_seconds
        phases = task["search_phases"]
        for phase in phases:
            if phase.startswith("block_") and phase not in setup["blocks"]:
                continue
            name = f"fit_{index:02d}_{phase}"
            parameters = coordinates.to_effective(
                self.original, self.oracle, selected["parameters"]
            )
            self.perform(
                name,
                lambda save, parameters=parameters, phase=phase: search.search(
                    self.oracle,
                    parameters,
                    seconds / len(phases),
                    max(1, p.maximum_calls // len(phases)),
                    save,
                    task["route"],
                    block=setup["blocks"].get(phase),
                    persistence_headroom_seconds=p.persistence_headroom_seconds,
                ),
            )
            best = first._value(self.ops.get(name), partial=True).get("best")
            if best and isinstance(best.get("parameters"), dict):
                replay = self.verify_endpoint(
                    name + "_verification", best["parameters"]
                )
                if (
                    _finite_endpoint(replay)
                    and retention.compare_endpoints(selected, replay, p)[
                        "classification"
                    ]
                    == "meaningful_improvement"
                ):
                    selected = replay
        return selected

    def fit(self, setup: dict):
        """Try the prespecified starts; a failed start never terminates the roster."""
        from autoformalism.fitting import remaining_portfolio_search as search

        p = self.policy
        if self.record["anchor"]:
            self.verify_endpoint("source_verification", self.start)
        for index, start in enumerate(setup["starts"]):
            name = f"start_{index:02d}"
            qualified = self.perform(
                name + "_qualification",
                lambda save, start=start: search.qualify(
                    self.original,
                    self.oracle,
                    start["parameters"],
                    p.qualification_seconds,
                    save,
                    self.task["route"],
                ),
            )
            if not qualified.get("passed") or not isinstance(
                qualified.get("parameters"), dict
            ):
                continue
            endpoint = self.verify_endpoint(
                name + "_verification", qualified["parameters"]
            )
            if not _finite_endpoint(endpoint):
                continue
            self.fit_start(0 if self.task["recovery"] else index, endpoint, setup)
            if self.task["recovery"]:
                # Choose the first verified seed in fixed order, never the lowest
                # score among starts. Later seeds consume no reserve after success.
                break
        _, retained, _ = _retain(self.record, self.ops, p)
        if retained:
            from autoformalism.fitting.remaining_portfolio_feedback import (
                describe_endpoint,
            )

            self.perform(
                "feedback",
                lambda save: describe_endpoint(
                    self.original, retained, p.feedback_seconds, save
                ),
            )


def run_task(root: Path, index: int) -> dict:
    """Execute only unspent operations and preserve every verified incumbent."""
    plan, data = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown remaining portfolio task index")
    folder = root / "results" / plan["tasks"][index]["task_id"]
    with public._lock(folder):
        run = _Run(root, plan, data, index)
        if (folder / "result.json").exists():
            saved = read_seal(folder / "result.json")
            reporting.reconcile(
                saved, _assemble(run.identity, run.record, run.ops, run.policy), run.ops
            )
            return saved
        setup = run.perform("setup", run.setup)
        if setup:
            if run.oracle is None:
                rebuilt = run.setup(lambda _: None)
                if public.content_sha256(rebuilt) != public.content_sha256(setup):
                    raise ValueError("resumed numerical setup differs")
            run.fit(setup)
        result = _assemble(run.identity, run.record, run.ops, run.policy)
        seal(folder / "result.json", result)
        return result


def report(root: Path) -> dict:
    """Reconstruct all 22 outcomes from receipts without integrating or fitting."""
    plan, data = verify(root, runtime=False)
    p = Policy.model_validate(plan["policy"])
    rows, reconciliations = [], {}
    for task, record in zip(plan["tasks"], data["records"], strict=True):
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = root / "results" / task["task_id"]
        operations = _operations(folder, identity, _allowances(task, p))
        row = _assemble(identity, record, operations, p)
        terminal = (folder / "result.json").exists()
        if terminal:
            reconciliations[task["task_id"]] = reporting.reconcile(
                read_seal(folder / "result.json"), row, operations
            )
        rows.append(
            row
            | {
                "status": "complete"
                if terminal
                else "partial"
                if operations
                else "missing"
            }
        )
    recorded = sum(row["status"] == "complete" for row in rows)
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if recorded == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": recorded,
        "rows": rows,
        "status_counts": dict(Counter(row["search_status"] for row in rows)),
        "retained_models": sum(row["retained"] is not None for row in rows),
        "terminal_reconciliations": reconciliations,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "optimization_converged": False,
        "global_minimum_certified": False,
        "limitations": [
            "Bounded fitting attempts do not certify full stationarity or "
            "global optimality.",
            "A finite failed seed set does not establish equation-family "
            "infeasibility.",
            "Residual-only qualification does not qualify parameter "
            "sensitivities or confidence.",
            "Conditional root optimality does not establish "
            "shared-coefficient optimality.",
            "Nonlinear blocks hold other parameters only temporarily; the "
            "final phase releases all.",
            "Training reconstruction uses full-trajectory fitted roots; fresh "
            "prefix forecasts are deferred.",
            "No validation or test data are loaded or reported.",
            "Descriptive trajectory mismatches do not uniquely identify a "
            "scientific repair.",
            "Started or interrupted numerical allowances are never "
            "automatically renewed.",
            "Numerical ties retain the earlier verified incumbent.",
        ],
    }
    with public._lock(root / "report"):
        public._write(root / "summary.json", summary)
        lines = [
            "# Remaining construction portfolio",
            "",
            f"{recorded}/{len(rows)} terminal tasks.",
            "",
            "| Candidate | Route | Search status | Source NMSE | Retained NMSE |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        lines += [
            f"| {r['candidate_id']} | {r['route']} | {r['search_status']} | "
            f"{r['source_anchor_nmse']} | {r['training_reconstruction_nmse']} |"
            for r in rows
        ]
        lines += [
            "",
            "Training reconstruction only. Fresh observation-prefix forecasts "
            "remain pending.",
        ]
        (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def inspection(root: Path) -> Path:
    """Package compact inputs, numerical receipts and bounded scheduler logs."""
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
    output = root / "inspection.tar.gz"
    output.write_bytes(buffer.getvalue())
    return output
