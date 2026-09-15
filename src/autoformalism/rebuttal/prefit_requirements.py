"""Frozen saved-model requirement feedback pilot; no fitting or data loading."""

from __future__ import annotations

import fcntl
import hashlib
import json
import signal
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.expressions import ModelValidationError
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    atomic_json,
    visible_response,
)
from autoformalism.rebuttal import prefit_construction_campaign as construction
from autoformalism.rebuttal.prefit_construction_audit import (
    _inputs,
    read_audit,
    reconstruct,
)
from autoformalism.rebuttal.prefit_feedback import (
    EpisodeClient,
    _check_events,
    _records,
    _save_state,
    runtime_identity,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas.base import StrictSchema
from autoformalism.search.repair_provenance import (
    STAGE_AWARE,
    attempt_feedback,
    with_repair_provenance,
)
from autoformalism.search.requirement_feedback import (
    SIGN_SYSTEM_PROMPT,
    STRICT_REPAIR,
    SYSTEM_PROMPT,
    TOPOLOGY_SIGN_REPAIR,
    FunctionRepair,
    RepairPolicy,
    RequirementBinding,
    apply_requirement_repair,
    diagnose_requirements,
    model_review_facts,
    repair_failure_feedback,
)
from autoformalism.search.requirement_feedback import repair_payload as payload_for
from autoformalism.staged_topology import content_hash

PROTOCOL = "prefit-requirement-feedback-1"
ARMS = ("local_only", "requirement_feedback")
LIMITATION = (
    "Necessary nonlinear syntax on a public input/target feedback path, not scientific "
    "recovery or fit quality. Other scientific concerns are advisory. One RHS may "
    "change; topology revisions are reported for a later stage. Controls measure "
    "runtime preservation; repair seeds repeat source models. This is not a training-"
    "evidence or old-versus-new full-controller comparison."
)


class RequirementConfig(StrictSchema):
    """Explicitly reviewed requirement bindings and a bounded matched experiment."""

    protocol: Literal["prefit-requirement-feedback-1"] = PROTOCOL
    platform: Literal["aces-h100x1"] = "aces-h100x1"
    serving_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_settings: StagedModelSettings
    bindings: tuple[RequirementBinding, ...] = Field(min_length=1, max_length=32)
    seeds: tuple[int, ...] = (0, 1, 2)
    expected_source_models: int = Field(default=12, ge=1, le=128)
    expected_source_plan_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    served_context_tokens: int = Field(default=32768, ge=8192)
    wall_seconds: int = Field(default=3600, ge=600)
    shutdown_margin_seconds: int = Field(default=300, ge=60)
    repair_policy: RepairPolicy = STRICT_REPAIR
    feedback_policy: Literal["legacy", "stage-aware-1"] = "legacy"
    replay_source_plan_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def bounded(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("repair seeds must be nonempty and unique")
        if (
            self.model_settings.maximum_requests
            != self.model_settings.attempts_per_step
        ):
            raise ValueError("one shared physical request budget per repair episode")
        if (
            not self.model_settings.timeout_seconds
            < self.shutdown_margin_seconds
            < self.wall_seconds
        ):
            raise ValueError("provider timeout must fit inside shutdown margin")
        if len({(b.cell, b.requirement_id) for b in self.bindings}) != len(
            self.bindings
        ):
            raise ValueError("duplicate public requirement binding")
        return self


def launcher_hash() -> str:
    """Pin the CLI and scheduler/server entry points alongside runtime code."""
    repo = Path(__file__).resolve().parents[3]
    names = (
        "scripts/prefit_requirement_campaign.py",
        "scripts/hpc/run_staged_topology_server.sh",
        "scripts/hpc/submit_prefit_requirements_aces.sh",
        "scripts/hpc/run_prefit_requirements_aces.sh",
    )
    return content_hash(
        {n: hashlib.sha256((repo / n).read_bytes()).hexdigest() for n in names}
    )


def freeze(source: Path, config_path: Path, root: Path) -> dict:
    """Reconstruct sealed source models and copy only model/public-brief evidence."""
    if root.resolve().is_relative_to(
        source.resolve()
    ) or source.resolve().is_relative_to(root.resolve()):
        raise ValueError(
            "new campaign root must be separate from the historical source"
        )
    config = RequirementConfig.model_validate_json(config_path.read_text())
    plan = sealed_read(source / "plan.json")
    if plan["protocol"] != construction.CONSTRUCTION_ONLY_PROTOCOL:
        raise ValueError("source must be a construction-only audit campaign")
    if (
        config.expected_source_plan_sha256
        and plan["artifact_sha256"] != config.expected_source_plan_sha256
    ):
        raise ValueError("source plan differs from the pinned experiment")
    if len(plan["tasks"]) != config.expected_source_models:
        raise ValueError("source task count differs")
    if len({t["task_id"] for t in plan["tasks"]}) != len(plan["tasks"]):
        raise ValueError("duplicate source task identity")
    if {b.cell for b in config.bindings} - set(plan["cells"]):
        raise ValueError("requirement binding names an absent source cell")
    for name in (
        "model",
        "model_revision",
        "reasoning_effort",
        "temperature",
        "max_output_tokens",
    ):
        if (
            config.model_settings.model_dump()[name]
            != plan["config"]["model_settings"][name]
        ):
            raise ValueError(f"proposer inference differs from source: {name}")
    cases = []
    for task in plan["tasks"]:
        directory = source / "results" / task["task_id"]
        saved, binding, _ = _inputs(source, plan, task)
        audit = read_audit(directory, binding)
        if saved is None or audit is None or audit["status"] != "passed":
            raise ValueError(
                f"source model lacks a passing sealed audit: {task['task_id']}"
            )
        cell = plan["cells"][task["cell"]]
        rebuilt = reconstruct(cell, saved)
        if not rebuilt["certificate"]["passed"]:
            raise ValueError("reconstructed source audit failed")
        for key in (
            "certificate",
            "function_slots",
            "handoff",
            "scientific_review_facts",
        ):
            if rebuilt[key] != audit[key]:
                raise ValueError("source audit differs from current reconstruction")
        bundle = {
            "source_task": task,
            "source_audit_sha256": audit["artifact_sha256"],
            "source_construction_sha256": saved["artifact_sha256"],
            "brief": cell["brief"],
            "context": cell["context"],
            "topology": {
                k: saved["topology"][k] for k in ("inventory", "equations", "topology")
            },
            "candidate": rebuilt["handoff"]["candidate"],
            "initialization": rebuilt["handoff"]["initialization"],
            "slots": rebuilt["function_slots"],
            "scientific_review_facts": rebuilt["scientific_review_facts"],
        }
        bindings = [
            b.model_dump(mode="json") for b in config.bindings if b.cell == task["cell"]
        ]
        baseline = diagnose_requirements(bundle, bindings)
        cases.append(
            {
                "case_id": content_hash(bundle),
                "bundle": bundle,
                "bindings": bindings,
                "baseline": baseline,
                "review_facts": model_review_facts(bundle),
                "cohort": "repair" if baseline["requirement_gap"] else "control",
            }
        )
    tasks = []
    for index, case in enumerate(cases):
        for seed in config.seeds:
            for arm in ARMS if (index + seed) % 2 == 0 else reversed(ARMS):
                tasks.append(
                    {
                        "task_id": f"{case['case_id']}_s{seed}_{arm}",
                        "case_id": case["case_id"],
                        "seed": seed,
                        "arm": arm,
                        "cohort": case["cohort"],
                    }
                )
    return sealed_write(
        root / "plan.json",
        {
            "protocol": PROTOCOL,
            "config": config.model_dump(mode="json"),
            "source_root": str(source.resolve()),
            "source_plan_sha256": plan["artifact_sha256"],
            "cases": cases,
            "tasks": tasks,
            "runtime_source_sha256": runtime_source_hash(),
            "runtime_identity": runtime_identity(),
            "launcher_sha256": launcher_hash(),
            "parameter_fitting_performed": False,
            "scientific_judge_called": False,
            "trajectory_data_opened": False,
            "test_data_opened": False,
            "private_reference_opened": False,
        },
    )


def verify(root: Path) -> dict:
    """Resume from immutable copied models without reopening the historical run."""
    plan = sealed_read(root / "plan.json")
    config = RequirementConfig.model_validate(plan["config"])
    if plan["protocol"] != PROTOCOL or (
        plan["runtime_source_sha256"] != runtime_source_hash()
        or plan["runtime_identity"] != runtime_identity()
        or plan["launcher_sha256"] != launcher_hash()
    ):
        raise ValueError("frozen requirement campaign runtime differs")
    if len(plan["cases"]) != config.expected_source_models:
        raise ValueError("frozen source denominator differs")
    for case in plan["cases"]:
        if content_hash(case["bundle"]) != case["case_id"]:
            raise ValueError("frozen source model changed")
    return plan


def _state(root: Path, plan: dict, task: dict) -> tuple[dict, list[dict]]:
    directory = root / "results" / task["task_id"]
    identity = content_hash([plan["artifact_sha256"], task])
    path = directory / "state.json"
    state = (
        sealed_read(path)
        if path.exists()
        else {
            "identity": identity,
            "status": "running",
            "attempts": [],
            "final": None,
            "stop_reason": None,
        }
    )
    state.pop("artifact_sha256", None)
    if state["identity"] != identity:
        raise ValueError("requirement episode identity differs")
    records = _records(directory / "calls", identity)
    _check_events(state, records)
    return state, records


def run_episode(root: Path, plan: dict, task: dict, client: EpisodeClient) -> dict:
    """Commit a successful one-slot revision; every rejected proposal rolls back."""
    if task not in plan["tasks"]:
        raise ValueError("task does not belong to frozen campaign")
    config = RequirementConfig.model_validate(plan["config"])
    directory = root / "results" / task["task_id"]
    identity = content_hash([plan["artifact_sha256"], task])
    if (
        client.namespace != identity
        or client.settings != config.model_settings
        or client.seed != task["seed"]
        or client.directory.resolve() != (directory / "calls").resolve()
    ):
        raise ValueError("client differs from frozen requirement task")
    case = next(c for c in plan["cases"] if c["case_id"] == task["case_id"])
    state, records = _state(root, plan, task)
    bundle, baseline = case["bundle"], case["baseline"]
    if (task["arm"] == "local_only" or baseline["route"] != "function_repair") and (
        state["attempts"] or records
    ):
        raise ValueError("no-call route contains provider work")
    if state["status"] == "complete":
        return state
    path = directory / "state.json"
    if task["arm"] == "local_only" or baseline["route"] != "function_repair":
        state.update(
            status="complete",
            stop_reason=(
                "local_only_accepted"
                if task["arm"] == "local_only"
                else "preserved_requirement_control"
                if baseline["route"] == "preserve"
                else "topology_revision_required"
            ),
            final={
                "candidate": bundle["candidate"],
                "initialization": bundle["initialization"],
                "candidate_sha256": content_hash(bundle["candidate"]),
                "diagnosis": baseline,
                "selected_interaction": None,
                "protected_slots_preserved": len(bundle["slots"]),
            },
        )
        _save_state(path, state)
        return state
    _save_state(path, state)
    for attempt in range(
        len(state["attempts"]), config.model_settings.attempts_per_step
    ):
        retry = state["attempts"][-1]["feedback"] if state["attempts"] else None
        try:
            record = client.call(
                system=SIGN_SYSTEM_PROMPT
                if config.repair_policy == TOPOLOGY_SIGN_REPAIR
                else SYSTEM_PROMPT,
                user="Whole-model requirement repair\n"
                + json.dumps(
                    payload_for(
                        bundle, baseline, retry, repair_policy=config.repair_policy
                    )
                ),
                response_model=FunctionRepair,
                step="requirement_function_repair",
                attempt=attempt,
            )
        except ValueError as exc:
            if str(exc) not in {
                "total provider request budget exhausted",
                "total token budget cannot accommodate the next request",
            }:
                raise
            state["stop_reason"] = "budget_exhausted"
            break
        reply = None
        try:
            reply = visible_response(record)
            result = apply_requirement_repair(
                bundle, case["bindings"], reply, repair_policy=config.repair_policy
            )
            if config.feedback_policy == STAGE_AWARE:
                result = with_repair_provenance(bundle, result)
        except (ValueError, TypeError, KeyError, ModelValidationError) as exc:
            feedback = (
                attempt_feedback(bundle, record, reply, exc)
                if config.feedback_policy == STAGE_AWARE
                else repair_failure_feedback(bundle, reply, exc)
                if (config.repair_policy == TOPOLOGY_SIGN_REPAIR)
                else {
                    "error": str(exc)[:6000],
                    "rejected_reply": reply,
                    "transaction": "rolled_back",
                }
            )
            result = None
        else:
            feedback = {"error": None, "transaction": "committed"}
        state["attempts"].append(
            {
                "attempt": attempt,
                "request_hash": record["request_hash"],
                "record_sha256": content_hash(record),
                "accepted": result is not None,
                "response": reply,
                "feedback": feedback,
            }
        )
        if result is not None:
            state.update(
                status="complete",
                stop_reason="requirement_syntax_repaired",
                final=result,
            )
        _save_state(path, state)
        if state["status"] == "complete":
            return state
    state.update(
        status="complete",
        stop_reason=state["stop_reason"] or "attempts_exhausted",
        final={
            "candidate": bundle["candidate"],
            "initialization": bundle["initialization"],
            "candidate_sha256": content_hash(bundle["candidate"]),
            "diagnosis": baseline,
            "selected_interaction": None,
            "protected_slots_preserved": len(bundle["slots"]),
        },
    )
    _save_state(path, state)
    return state


def summarize(root: Path) -> dict:
    """Report model-level gaps, preservation, failures and every charged call."""
    plan = verify(root)
    rows = []
    cases = {c["case_id"]: c for c in plan["cases"]}
    for task in plan["tasks"]:
        state, records = _state(root, plan, task)
        final = state["final"]
        case = cases[task["case_id"]]
        rows.append(
            {
                **task,
                "source_task": case["bundle"]["source_task"],
                "status": state["status"],
                "stop_reason": state["stop_reason"],
                "baseline_gap": case["baseline"]["requirement_gap"],
                "final_gap": final["diagnosis"]["requirement_gap"] if final else None,
                "changed": bool(
                    final
                    and final["candidate_sha256"]
                    != content_hash(case["bundle"]["candidate"])
                ),
                "first_attempt_repaired": bool(
                    state["attempts"] and state["attempts"][0]["accepted"]
                ),
                "outer_sign_normalizations": len(
                    final.get("outer_sign_normalizations", [])
                )
                if final
                else 0,
                "protected_slots_preserved": final["protected_slots_preserved"]
                if final
                else 0,
                "physical_requests": len(records),
                "observed_tokens": sum(
                    r.get("observed_total_tokens") or 0 for r in records
                ),
                "budget_charge": sum(r.get("budget_charge", 0) for r in records),
                "unknown_usage_requests": sum(
                    r.get("observed_total_tokens") is None for r in records
                ),
                "provider_seconds": sum(r.get("latency_seconds", 0) for r in records),
            }
        )
    arms = {}
    for arm in ARMS:
        groups = {}
        for cohort in ("repair", "control"):
            selected = [r for r in rows if r["arm"] == arm and r["cohort"] == cohort]
            groups[cohort] = {
                "expected": len(selected),
                "terminal": sum(r["status"] == "complete" for r in selected),
                "no_bound_requirement_gap_final": sum(
                    r["final_gap"] is False for r in selected
                ),
                "requirement_gap_remaining": sum(
                    r["final_gap"] is True for r in selected
                ),
                "changed_models": sum(r["changed"] for r in selected),
                "first_attempt_repaired": sum(
                    r["first_attempt_repaired"] for r in selected
                ),
                "topology_revision_required": sum(
                    r["stop_reason"] == "topology_revision_required" for r in selected
                ),
                **{
                    k: sum(r[k] for r in selected)
                    for k in (
                        "protected_slots_preserved",
                        "outer_sign_normalizations",
                        "physical_requests",
                        "observed_tokens",
                        "budget_charge",
                        "unknown_usage_requests",
                        "provider_seconds",
                    )
                },
            }
        arms[arm] = groups
    pairs = defaultdict(dict)
    for row in rows:
        if row["cohort"] == "repair":
            pairs[(row["case_id"], row["seed"])][row["arm"]] = row
    paired = Counter()
    for pair in pairs.values():
        if any(pair[a]["status"] != "complete" for a in ARMS):
            paired["pending"] += 1
        else:
            paired[
                "feedback_resolved_local_gap"
                if pair["local_only"]["final_gap"]
                and not pair["requirement_feedback"]["final_gap"]
                else "gap_unresolved"
            ] += 1
    report = {
        "protocol": PROTOCOL,
        "repair_policy": plan["config"].get("repair_policy", STRICT_REPAIR),
        "plan_sha256": plan["artifact_sha256"],
        "status": "complete"
        if all(r["status"] == "complete" for r in rows)
        else "partial",
        "source_models": len(cases),
        "models_with_bound_requirements": sum(
            bool(c["bindings"]) for c in cases.values()
        ),
        "models_without_bound_requirements": sum(
            not c["bindings"] for c in cases.values()
        ),
        "distinct_source_gaps": sum(c["cohort"] == "repair" for c in cases.values()),
        "arms": arms,
        "paired_repairs": dict(paired),
        "rows": rows,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "trajectory_data_opened": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "limitation": LIMITATION,
    }
    atomic_json(root / "summary.json", report)
    return report


def replay_saved_repairs(root: Path, previous_root: Path) -> dict:
    """Compare interpretation of exact cached replies, without new provider work.

    This is response admissibility, not a counterfactual fresh-run success rate.
    The old plan and each request ledger are verified; model cases must agree
    exactly with the new reconstructed source. Output belongs only to the new run.
    """
    plan = verify(root)
    config = RequirementConfig.model_validate(plan["config"])
    previous = sealed_read(previous_root / "plan.json")
    if root.resolve().is_relative_to(
        previous_root.resolve()
    ) or previous_root.resolve().is_relative_to(root.resolve()):
        raise ValueError("replay output must differ from historical campaign")
    if (
        not config.replay_source_plan_sha256
        or previous["artifact_sha256"] != config.replay_source_plan_sha256
        or previous["protocol"] != PROTOCOL
    ):
        raise ValueError("prior requirement campaign differs from pinned replay source")
    if config.repair_policy != TOPOLOGY_SIGN_REPAIR:
        raise ValueError("saved-response replay requires topology-owned sign policy")
    if previous["config"].get("repair_policy", STRICT_REPAIR) != STRICT_REPAIR:
        raise ValueError("replay baseline must use the strict policy")
    cases = {c["case_id"]: c for c in plan["cases"]}
    if (
        previous["cases"] != plan["cases"]
        or previous["tasks"] != plan["tasks"]
        or previous["source_plan_sha256"] != plan["source_plan_sha256"]
        or previous["config"]["model_settings"] != plan["config"]["model_settings"]
    ):
        raise ValueError("replay cases, tasks or proposer settings differ")
    rows = []
    for task in previous["tasks"]:
        state, records = _state(previous_root, previous, task)
        if state["status"] != "complete":
            raise ValueError("prior requirement episode is incomplete")
        if task["arm"] != "requirement_feedback" or task["cohort"] != "repair":
            if records or state["attempts"]:
                raise ValueError("prior no-call route contains provider work")
            continue
        case = cases[task["case_id"]]
        for attempt in state["attempts"]:
            record = next(
                r for r in records if r["request_hash"] == attempt["request_hash"]
            )
            try:
                raw = visible_response(record)
            except (ValueError, TypeError, KeyError):
                raw = None
            if raw != attempt["response"]:
                raise ValueError("saved reply differs from cached provider response")
            outcomes = {}
            for policy in (STRICT_REPAIR, TOPOLOGY_SIGN_REPAIR):
                try:
                    result = apply_requirement_repair(
                        case["bundle"], case["bindings"], raw, repair_policy=policy
                    )
                except (ValueError, TypeError, KeyError, ModelValidationError) as exc:
                    outcomes[policy] = {"accepted": False, "error": str(exc)[:6000]}
                else:
                    outcomes[policy] = {
                        "accepted": True,
                        "error": None,
                        "selected_function": result["selected_function"],
                        "outer_sign_normalizations": result.get(
                            "outer_sign_normalizations", []
                        ),
                    }
                    if (
                        policy == STRICT_REPAIR
                        and result["candidate"] != state["final"]["candidate"]
                    ):
                        raise ValueError(
                            "strict replay differs from saved accepted candidate"
                        )
            if outcomes[STRICT_REPAIR]["accepted"] != attempt["accepted"]:
                raise ValueError("strict replay differs from saved acceptance")
            rows.append(
                {
                    "task": task,
                    "attempt": attempt["attempt"],
                    "request_hash": attempt["request_hash"],
                    "response": raw,
                    "outcomes": outcomes,
                }
            )
    old = sum(r["outcomes"][STRICT_REPAIR]["accepted"] for r in rows)
    new = sum(r["outcomes"][TOPOLOGY_SIGN_REPAIR]["accepted"] for r in rows)
    regressions = sum(
        r["outcomes"][STRICT_REPAIR]["accepted"]
        and not r["outcomes"][TOPOLOGY_SIGN_REPAIR]["accepted"]
        for r in rows
    )
    return sealed_write(
        root / "replay.json",
        {
            "protocol": "prefit-sign-response-replay-1",
            "new_plan_sha256": plan["artifact_sha256"],
            "source_plan_sha256": previous["artifact_sha256"],
            "saved_responses": len(rows),
            "strict_accepted": old,
            "normalized_accepted": new,
            "newly_admissible": new - old + regressions,
            "acceptance_regressions": regressions,
            "live_llm_calls": 0,
            "parameter_fitting_performed": False,
            "rows": rows,
            "limitation": "Exact saved-response admissibility only; no counterfactual "
            "fresh-run success rate, scientific certification or fitting result.",
        },
    )


def run(root: Path, base_url: str, *, wall_seconds: float | None = None) -> dict:
    """Drain on signal/deadline and serialize each episode independently."""
    plan = verify(root)
    config = RequirementConfig.model_validate(plan["config"])
    stopped = False
    deadline = time.monotonic() + (
        wall_seconds if wall_seconds is not None else config.wall_seconds
    )

    def stop(*_):
        nonlocal stopped
        stopped = True

    prior = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for task in plan["tasks"]:
            directory = root / "results" / task["task_id"]
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "worker.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                client = EpisodeClient(
                    settings=config.model_settings,
                    base_url=base_url,
                    directory=directory / "calls",
                    namespace=content_hash([plan["artifact_sha256"], task]),
                    seed=task["seed"],
                    can_start=lambda: not stopped
                    and time.monotonic() + config.shutdown_margin_seconds < deadline,
                )
                try:
                    run_episode(root, plan, task, client)
                except DeferredCall:
                    return summarize(root)
    finally:
        for sig, handler in prior.items():
            signal.signal(sig, handler)
    return summarize(root)
