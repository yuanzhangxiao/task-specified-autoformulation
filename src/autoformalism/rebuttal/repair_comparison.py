"""Frozen matched repair comparison with optional pre-fitting scientific feedback."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from time import monotonic
from typing import Literal

from pydantic import Field

from autoformalism.expressions import ModelValidationError, compile_candidate
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    fit_collocation_forward_sensitivity,
)
from autoformalism.fitting.initialization import (
    LatentInitializationPlan,
    apply_initialization_plan,
)
from autoformalism.fitting.sensitivity_contract import SensitivityContractError
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedTopologyClient,
    atomic_json,
)
from autoformalism.rebuttal.repair_evidence import (
    Finding,
    decision_report,
    domain_findings,
    memory_path_findings,
    model_hash,
    numerical_findings,
)
from autoformalism.rebuttal.repair_scientific_judge import (
    judge_protocol,
    perform_review,
    review_cost,
    review_request,
)
from autoformalism.rebuttal.repair_transactions import (
    default_initialization,
    request_repair,
)
from autoformalism.rebuttal.staged_multiround_feedback_campaign import (
    _public_fit_context,
    _verified_plan,
)
from autoformalism.rebuttal.staged_multiround_feedback_campaign import (
    freeze_campaign as freeze_inputs,
)
from autoformalism.rebuttal.staged_prefit_fitting_campaign import load_public_data
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash

ARMS = ("redesigned_runtime", "redesigned_prefit_judge")


def selected_arms(arm: str | None) -> tuple[str, ...]:
    """Validate execution selection without changing the frozen comparison plan."""
    if arm is None:
        return ARMS
    if arm not in ARMS:
        raise ValueError(f"unknown comparison arm: {arm}")
    return (arm,)


class RepairComparisonConfig(StrictSchema):
    """One frozen fitter and one edit budget shared by the two arms."""

    protocol: Literal["repair-feedback-comparison-1"] = "repair-feedback-comparison-1"
    rounds: int = Field(default=4, ge=1, le=8)
    max_consecutive_no_change: int = Field(default=2, ge=2, le=4)
    fit: CollocationSensitivityConfig = CollocationSensitivityConfig(
        initializer_seconds=120,
        refinement_seconds=180,
        maximum_function_evaluations=240,
        collocation_node_start="rollout_or_observed",
        node_warmup_seconds=5,
        least_squares_ftol=None,
        collocation_diagnostics=True,
        recovery_policy="feasible",
        recovery_max_starts=10,
        recovery_probe_seconds=10,
    )
    initialization_policy: Literal["shared_training_fitted_values_or_causal_maps"] = (
        "shared_training_fitted_values_or_causal_maps"
    )
    judge_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    memory_targets: tuple[str, ...] = ("v01",)
    nonlinear_targets: tuple[str, ...] = ("v01",)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def launcher_hash() -> str:
    """Bind the separate comparison entry points, not just package code."""
    repo = Path(__file__).resolve().parents[3]
    return content_hash(
        {
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                repo / "scripts/repair_feedback_comparison.py",
                repo / "scripts/smoke_repair_feedback_comparison.py",
                repo / "scripts/hpc/run_repair_comparison_aces.sh",
                repo / "scripts/hpc/submit_repair_comparison_aces.sh",
            )
        }
    )


class RepairBudgetExceeded(ValueError):
    """A terminal arm outcome, not a scientific model failure."""


class BudgetedRepairClient(StagedTopologyClient):
    """Keep physical-request and reservation budgets across process/server swaps."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.records = [read(p) for p in sorted(self.directory.glob("*.json"))]
        if any(r["request"]["namespace"] != self.namespace for r in self.records):
            raise ValueError("provider cache namespace differs")

    def call(self, **kwargs):
        try:
            record = super().call(**kwargs)
        except ValueError as exc:
            if str(exc) in {
                "total provider request budget exhausted",
                "total token budget cannot accommodate the next request",
            }:
                raise RepairBudgetExceeded(str(exc)) from exc
            raise
        self.records = list({r["request_hash"]: r for r in self.records}.values())
        return record


def freeze(source: Path, output: Path, judge_revision: str) -> dict:
    """Copy eligible public source once; freeze both arms before any requests."""
    config = RepairComparisonConfig(judge_revision=judge_revision)
    repo = Path(__file__).resolve().parents[3]
    inputs = freeze_inputs(
        repo / "configs/staged_multiround_feedback_v6.json", source, output / "inputs"
    )
    tasks = []
    for task in inputs["tasks"]:
        # Counterbalance arm order; model seed and all nonjudge settings are equal.
        for arm in ARMS if task["seed"] % 2 == 0 else reversed(ARMS):
            tasks.append(
                {
                    **task,
                    "source_task_id": task["task_id"],
                    "arm": arm,
                    "task_id": f"{task['task_id']}_{arm}",
                }
            )
    plan = {
        "schema_version": config.protocol,
        "config": config.model_dump(mode="json"),
        "input_plan_sha256": inputs["plan_sha256"],
        "runtime_source_sha256": runtime_source_hash(),
        "judge_protocol": judge_protocol(),
        "tasks": tasks,
        "fitter_origin": "549e03945a90e817bd377b49bad76c68f85e7672",
        "public_obligation_quote": inputs["config"]["public_obligation_quote"],
        "public_obligation_prompt_sha256": inputs["config"][
            "public_obligation_prompt_sha256"
        ],
        "proposer_settings": inputs["config"]["model_settings"],
        "serving_image_sha256": inputs["config"]["serving_image_sha256"],
        "test_data_opened": False,
        "private_reference_opened": False,
        "final_evaluation_judge_changed": False,
        "automatic_winner_defined": False,
    }
    # Include scripts too: edits to transport cannot silently resume a frozen plan.
    plan["launcher_sha256"] = launcher_hash()
    plan["plan_sha256"] = content_hash(plan)
    path = output / "plan.json"
    if path.exists() and read(path) != plan:
        raise ValueError("existing comparison plan differs; use a fresh output root")
    atomic_json(path, plan)
    return plan


def verify(root: Path) -> dict:
    plan = read(root / "plan.json")
    if (
        content_hash({k: v for k, v in plan.items() if k != "plan_sha256"})
        != plan["plan_sha256"]
    ):
        raise ValueError("comparison digest differs")
    if plan["runtime_source_sha256"] != runtime_source_hash():
        raise ValueError("runtime differs from frozen comparison")
    if plan["launcher_sha256"] != launcher_hash():
        raise ValueError("launcher differs from frozen comparison")
    if _verified_plan(root / "inputs")["plan_sha256"] != plan["input_plan_sha256"]:
        raise ValueError("public input plan differs")
    if plan["judge_protocol"] != judge_protocol():
        raise ValueError("scientific judge protocol differs")
    return plan


class JudgePending(DeferredCall):
    """Checkpoint the scientific review request before switching serving roles."""


def _assess(
    root: Path,
    task: dict,
    directory: Path,
    parent: CandidateModel,
    candidate: CandidateModel,
    initial: LatentInitializationPlan,
    dataset,
    context,
    prompt: str,
    config: RepairComparisonConfig,
    reference_initial: LatentInitializationPlan | None = None,
) -> dict:
    """Separate runtime, science and numerical results; never gate on judge score."""
    identity = content_hash(
        [
            model_hash(candidate),
            initial.model_dump(mode="json"),
            config.model_dump(mode="json"),
            task["arm"],
            model_hash(parent),
            reference_initial.model_dump(mode="json") if reference_initial else None,
        ]
    )
    saved = directory / "assessment.json"
    if saved.exists():
        result = read(saved)
        if result["identity"] != identity:
            raise ValueError("assessment provenance differs")
        return result
    runtime = domain_findings(candidate, context) + memory_path_findings(
        candidate, context, config.memory_targets
    )
    model = None
    try:
        model = compile_candidate(candidate, context)
        lowered, _, _ = apply_initialization_plan(model, initial)
    except (ValueError, ModelValidationError) as exc:
        runtime.append(
            Finding(
                source="runtime",
                stage="prefit",
                candidate_sha256=model_hash(candidate),
                category="contract",
                code="COMPILER_UNAVAILABLE",
                certainty="observed",
                blocking=True,
                observation=str(exc),
                recheck="compiler compatibility, not scientific correctness",
            )
        )
    science, review = [], None
    if (
        task["arm"] == "redesigned_prefit_judge"
        and model is not None
        and not any(f.blocking for f in runtime)
    ):
        # Previous model is the comparative reference; initial review uses a self-pair.
        parent_model = compile_candidate(parent, context)
        parent_initial = reference_initial or default_initialization(parent, context)
        prior, _, _ = apply_initialization_plan(parent_model, parent_initial)
        request = review_request(
            prior.validated.candidate,
            lowered.validated.candidate,
            lowered.validated.context,
            prompt,
            task["seed"],
            config.judge_revision,
        )
        review_key = content_hash(request)
        review_dir = root / "reviews" / review_key
        atomic_json(directory / "review_ref.json", {"request_sha256": review_key})
        atomic_json(review_dir / "request.json", request)
        if not (review_dir / "review.json").exists():
            raise JudgePending("pre-fit scientific review queued")
        review = read(review_dir / "review.json")
        if review["request_sha256"] != review_key:
            raise ValueError("scientific response provenance differs")
        science = [
            Finding.model_validate(f).model_copy(
                update={"candidate_sha256": model_hash(candidate)}
            )
            for f in review["findings"]
        ]
    fit_path = directory / "fit.json"
    if fit_path.exists():
        fit = read(fit_path)
    elif any(f.blocking for f in runtime):
        fit = {"status": "not_run", "failure_class": "runtime_contract"}
        atomic_json(fit_path, fit)
    else:
        marker = directory / "fit_started.json"
        if marker.exists():
            # Do not reset a consumed native optimizer budget after a killed job.
            fit = {
                "status": "interrupted",
                "failure_class": "infrastructure",
                "error": "fit interrupted without terminal result; no automatic refit",
            }
        else:
            atomic_json(marker, {"identity": identity})
            try:
                fit = fit_collocation_forward_sensitivity(
                    model,
                    dataset.train,
                    dataset.validation,
                    config.fit,
                    directory / "numerical",
                    initialization_plan=initial,
                )
            except Exception as exc:
                fit = {
                    "status": "unavailable",
                    "failure_class": "fitter_capability"
                    if isinstance(exc, SensitivityContractError)
                    else "fit_runtime",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        atomic_json(fit_path, fit)
    result = {
        "identity": identity,
        "candidate_sha256": model_hash(candidate),
        "runtime": [f.model_dump(mode="json") for f in runtime],
        "science": [f.model_dump(mode="json") for f in science],
        "scientific_review": review,
        "fit": fit,
        "numerical": [
            f.model_dump(mode="json") for f in numerical_findings(candidate, fit)
        ]
        if fit.get("status") != "not_run"
        else [],
    }
    result = json.loads(json.dumps(result))
    atomic_json(saved, result)
    return result


def run_task(root: Path, task: dict, client: StagedTopologyClient) -> dict:
    """Resume an independent arm without letting failed actions erase evidence."""
    plan = verify(root)
    config = RepairComparisonConfig.model_validate(plan["config"])
    output = root / "results" / task["task_id"]
    output.mkdir(parents=True, exist_ok=True)
    inputs = root / "inputs"
    dataset, context = load_public_data(
        inputs / "frozen/public", task["benchmark_id"], task["tier"]
    )
    prompt = (
        inputs
        / "frozen/public/phase_b_v1"
        / task["benchmark_id"]
        / "proposer_prompt.txt"
    ).read_text()
    prompt = re.split(r"(?m)^F\.\s+Required response\s*$", prompt, maxsplit=1)[
        0
    ].rstrip()
    parent = CandidateModel.model_validate_json(
        (inputs / task["candidate_path"]).read_text()
    )
    initial = default_initialization(parent, context)
    state_path = output / "state.json"
    state = (
        read(state_path)
        if state_path.exists()
        else {
            "rounds": [],
            "history": [],
            "best": None,
            "candidate": parent.model_dump(mode="json"),
            "initialization": initial.model_dump(mode="json"),
            "last_numerical": [],
            "status": "running",
        }
    )
    if state["status"] != "running":
        return state
    if "assessment" not in state:
        assessment = _assess(
            root,
            task,
            output / "baseline",
            parent,
            parent,
            initial,
            dataset,
            context,
            prompt,
            config,
        )
        state.update(assessment=assessment, last_numerical=assessment["numerical"])
        if assessment["fit"].get("status") == "complete":
            state["best"] = {
                "round": 0,
                "candidate": state["candidate"],
                "initialization": state["initialization"],
                "fit": assessment["fit"],
            }
        atomic_json(state_path, state)
    if state["assessment"]["fit"].get("failure_class") in {
        "infrastructure",
        "fit_runtime",
        "fitter_capability",
    }:
        state["status"] = "blocked_fitter"
        atomic_json(state_path, state)
        return state
    for index in range(len(state["rounds"]) + 1, config.rounds + 1):
        parent = CandidateModel.model_validate(state["candidate"])
        initial = LatentInitializationPlan.model_validate(state["initialization"])
        assessment = state["assessment"]
        report = decision_report(
            parent,
            [Finding.model_validate(f) for f in assessment["runtime"]],
            [Finding.model_validate(f) for f in assessment["science"]],
            [Finding.model_validate(f) for f in state["last_numerical"]],
            state["history"],
        )
        report["public_training_facts"] = _public_fit_context(dataset)
        directory = output / f"round_{index:03d}"
        atomic_json(directory / "report.json", report)
        candidate, new_initial, transaction = request_repair(
            parent,
            initial,
            context,
            report,
            prompt,
            client,
            directory,
            index,
            nonlinear_targets=config.nonlinear_targets,
            memory_targets=config.memory_targets,
        )
        outcome = transaction["status"]
        new_assessment = None
        if outcome == "committed":
            new_assessment = _assess(
                root,
                task,
                directory,
                parent,
                candidate,
                new_initial,
                dataset,
                context,
                prompt,
                config,
                reference_initial=initial,
            )
            state.update(
                candidate=candidate.model_dump(mode="json"),
                initialization=new_initial.model_dump(mode="json"),
                assessment=new_assessment,
            )
            if new_assessment["numerical"]:
                state["last_numerical"] = new_assessment["numerical"]
            fit = new_assessment["fit"]
            if fit.get("failure_class") in {
                "infrastructure",
                "fit_runtime",
                "fitter_capability",
            }:
                state["status"] = "blocked_fitter"
            if fit.get("status") == "complete":
                score = float(
                    (fit.get("validation") or {}).get("normalized_mse", math.inf)
                )
                best_score = float(
                    ((state["best"] or {}).get("fit", {}).get("validation") or {}).get(
                        "normalized_mse", math.inf
                    )
                )
                if math.isfinite(score) and (
                    state["best"] is None or score < best_score
                ):
                    state["best"] = {
                        "round": index,
                        "candidate": state["candidate"],
                        "initialization": state["initialization"],
                        "fit": fit,
                    }
        record = {
            "round_index": index,
            "objective_category": report["objective_category"],
            "outcome": outcome,
            "candidate_sha256": model_hash(candidate),
            "transaction": transaction,
            "assessment": new_assessment,
            "last_numerical_preserved": outcome != "committed",
        }
        state["rounds"].append(record)
        state["history"].append(
            {
                "round": index,
                "outcome": outcome,
                "hypothesis": transaction.get("hypothesis"),
                "scope": (transaction.get("audit") or {}).get("scope"),
                "diagnostics": [
                    d for a in transaction["attempts"] for d in a["diagnostics"]
                ],
            }
        )
        if len(state["history"]) >= config.max_consecutive_no_change and all(
            r["outcome"] == "no_change"
            for r in state["history"][-config.max_consecutive_no_change :]
        ):
            state["status"] = "stopped_no_progress"
        atomic_json(directory / "round.json", record)
        atomic_json(state_path, state)
        if state["status"] != "running":
            break
    if state["status"] == "running":
        state["status"] = "complete"
    atomic_json(state_path, state)
    return state


def run_pass(
    root: Path,
    role: Literal["proposer", "judge"],
    base_url: str,
    wall_seconds: float = 14400,
    *,
    arm: str | None = None,
) -> dict:
    """Run selected arms independently; serialize only workers sharing an arm."""
    arms = selected_arms(arm)
    if role not in {"proposer", "judge"}:
        raise ValueError(f"unknown serving role: {role}")
    if role == "judge" and arms == ("redesigned_runtime",):
        raise ValueError("the runtime-only arm never calls a scientific judge")
    plan = verify(root)
    deadline = monotonic() + wall_seconds
    with ExitStack() as stack:
        # All-arm manual runs acquire the same locks in fixed order, so they
        # cannot overlap a split job. ExitStack releases earlier locks on error.
        for selected in arms:
            lock = stack.enter_context((root / f"worker-{selected}.lock").open("w"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if role == "judge":
            for request_path in sorted((root / "reviews").glob("*/request.json")):
                if monotonic() > deadline - 300:
                    break
                perform_review(read(request_path), request_path.parent, base_url)
        else:
            from autoformalism.llm.staged_topology import StagedModelSettings

            for task in plan["tasks"]:
                if task["arm"] not in arms:
                    continue
                if monotonic() >= deadline - 600:
                    break
                client = BudgetedRepairClient(
                    settings=StagedModelSettings.model_validate(
                        plan["proposer_settings"]
                    ),
                    base_url=base_url,
                    directory=root / "results" / task["task_id"] / "calls",
                    namespace=content_hash([plan["plan_sha256"], task]),
                    seed=task["seed"],
                    can_start=lambda: monotonic() < deadline - 600,
                )
                try:
                    run_task(root, task, client)
                except DeferredCall:
                    continue
                except RepairBudgetExceeded as exc:
                    path = root / "results" / task["task_id"] / "state.json"
                    state = read(path) if path.exists() else {}
                    state.update(status="budget_exhausted", error=str(exc))
                    atomic_json(path, state)
    # Two arm jobs may finish simultaneously. Recompute the combined snapshot
    # under its own lock instead of overwriting it with an earlier arm snapshot.
    with (root / "summary.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        atomic_json(root / "results/summary.json", summarize(root))
        summary = summarize(root, arm=arm)
        if arm is not None:
            atomic_json(root / "results" / f"summary-{arm}.json", summary)
    return summary


def summarize(root: Path, *, arm: str | None = None) -> dict:
    """Separate arm costs, hard failures, abstention, fit success and best fit."""
    arms = selected_arms(arm)
    plan = read(root / "plan.json")
    rows = []
    for task in plan["tasks"]:
        if task["arm"] not in arms:
            continue
        directory = root / "results" / task["task_id"]
        state = (
            read(directory / "state.json")
            if (directory / "state.json").exists()
            else {}
        )
        calls = [read(p) for p in (directory / "calls").glob("*.json")]
        assessments = [read(p) for p in directory.glob("*/assessment.json")]
        review_ids = {
            read(p)["request_sha256"] for p in directory.glob("*/review_ref.json")
        }
        costs = [review_cost(root / "reviews" / identity) for identity in review_ids]
        fits = [a["fit"] for a in assessments]
        diagnostics = Counter(
            d["code"]
            for r in state.get("rounds", [])
            for a in r["transaction"]["attempts"]
            for d in a["diagnostics"]
        )
        rows.append(
            {
                "task_id": task["task_id"],
                "arm": task["arm"],
                "seed": task["seed"],
                "status": state.get("status", "pending"),
                "error": state.get("error"),
                "recorded_rounds": len(state.get("rounds", [])),
                "outcomes": dict(
                    Counter(r["outcome"] for r in state.get("rounds", []))
                ),
                "diagnostic_counts": dict(diagnostics),
                "best_round": (state.get("best") or {}).get("round"),
                "best_validation_nmse": (
                    ((state.get("best") or {}).get("fit") or {}).get("validation") or {}
                ).get("normalized_mse"),
                "physical_proposer_requests": len(calls),
                "observed_proposer_tokens": sum(
                    r.get("observed_total_tokens") or 0 for r in calls
                ),
                "proposer_usage_missing_requests": sum(
                    r.get("observed_total_tokens") is None for r in calls
                ),
                "provider_seconds": sum(r.get("latency_seconds", 0) for r in calls),
                "scientific_review_count": len(review_ids),
                "physical_judge_requests": sum(c["physical_requests"] for c in costs),
                "observed_judge_tokens": sum(c["observed_total_tokens"] for c in costs),
                "judge_usage_missing_events": sum(
                    c["usage_missing_events"] for c in costs
                ),
                "assessed_candidates_including_baseline": len(assessments),
                "fit_status_counts": dict(
                    Counter(f.get("status", "missing") for f in fits)
                ),
                "finite_candidates_including_baseline": sum(
                    f.get("status") == "complete" for f in fits
                ),
                "baseline": compact_fit(
                    (read(directory / "baseline/assessment.json").get("fit") or {})
                    if (directory / "baseline/assessment.json").exists()
                    else {}
                ),
                "revision_diagnostics": [
                    dict(round=r["round_index"], attempt=a["attempt"], **d)
                    for r in state.get("rounds", [])
                    for a in r["transaction"]["attempts"]
                    for d in a["diagnostics"]
                ],
                "rounds": [compact_round(r) for r in state.get("rounds", [])],
            }
        )
    review_enabled = "redesigned_prefit_judge" in arms
    reviews = list((root / "reviews").glob("*/review.json")) if review_enabled else []
    pending = (
        sum(
            not (p.parent / "review.json").exists()
            for p in (root / "reviews").glob("*/request.json")
        )
        if review_enabled
        else 0
    )
    return {
        "schema_version": "repair-feedback-comparison-summary-1",
        "plan_sha256": plan["plan_sha256"],
        "selected_arm": arm,
        "status": "complete"
        if all(r["status"] not in {"pending", "running"} for r in rows)
        else "incomplete",
        "planned_tasks": len(rows),
        "terminal_tasks": sum(r["status"] not in {"pending", "running"} for r in rows),
        "scientific_reviews": len(reviews),
        "pending_scientific_reviews": pending,
        "by_arm": [
            {
                "arm": arm,
                "tasks": len(selected),
                "tasks_with_finite_candidate": sum(
                    r["best_validation_nmse"] is not None for r in selected
                ),
                "recorded_rounds": sum(r["recorded_rounds"] for r in selected),
                "physical_proposer_requests": sum(
                    r["physical_proposer_requests"] for r in selected
                ),
                "physical_judge_requests": sum(
                    r["physical_judge_requests"] for r in selected
                ),
                "observed_proposer_tokens": sum(
                    r["observed_proposer_tokens"] for r in selected
                ),
                "observed_judge_tokens": sum(
                    r["observed_judge_tokens"] for r in selected
                ),
            }
            for arm in arms
            for selected in [[r for r in rows if r["arm"] == arm]]
        ],
        "rows": rows,
        "automatic_winner_defined": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "final_evaluation_judge_changed": False,
    }


def compact_round(record: dict) -> dict:
    """A short result row with no ambiguity between a no-op and an attempted fit."""
    fit = (record.get("assessment") or {}).get("fit") or {}
    return {
        "round": record["round_index"],
        "outcome": record["outcome"],
        "category": record["objective_category"],
        "scope": (record["transaction"].get("audit") or {}).get("scope"),
        **compact_fit(fit),
    }


def compact_fit(fit: dict) -> dict:
    """Expose solver feasibility and messages separately from native termination."""
    initializer, refinement = fit.get("initializer") or {}, fit.get("refinement") or {}
    return {
        "fit_status": fit.get("status", "not_run"),
        "failure_class": fit.get("failure_class"),
        "error": fit.get("error"),
        "training_nmse": (fit.get("training") or {}).get("normalized_mse"),
        "validation_nmse": (fit.get("validation") or {}).get("normalized_mse"),
        "initializer_success": initializer.get("success"),
        "initializer_message": initializer.get("message"),
        "native_optimizer_success": refinement.get("optimizer_native_success"),
        "verified_optimizer_success": refinement.get("optimizer_success"),
        "finite_residual_evaluations": refinement.get("valid_residual_evaluations"),
        "optimizer_message": refinement.get("message"),
        "training_failures": (fit.get("training") or {}).get("failed_trajectories", []),
        "validation_failures": (fit.get("validation") or {}).get(
            "failed_trajectories", []
        ),
    }
