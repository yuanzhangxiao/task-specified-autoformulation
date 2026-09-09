"""Frozen public-only integration of topology and polarity before functions."""

from __future__ import annotations

import fcntl
import hashlib
import json
import signal
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    StagedTopologyClient,
    atomic_json,
)
from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec
from autoformalism.rebuttal.staged_topology_campaign import (
    public_validation_context,
    runtime_source_hash,
)
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.staged_topology import ModelingLimits, PublicScientificBrief
from autoformalism.search.staged_topology_runner import run_staged_topology
from autoformalism.staged_topology import build_scientific_brief, content_hash
from autoformalism.targets import PublicTargetContract


class PrefunctionIntegrationGates(StrictSchema):
    """Prospective exact gates for the six-task construction integration."""

    minimum_topology_completion: float = Field(ge=0, le=1)
    minimum_target_coverage: float = Field(ge=0, le=1)
    minimum_source_coverage: float = Field(ge=0, le=1)
    minimum_mechanism_coverage: float = Field(ge=0, le=1)
    minimum_polarity_consistency: float = Field(ge=0, le=1)


class PrefunctionIntegrationConfig(StrictSchema):
    """Two-public-benchmark, three-seed, pre-function construction campaign."""

    protocol: Literal["scientific-staged-prefunction-integration-1"]
    purpose: str = Field(min_length=1)
    platform: Literal["aces-h100x1"]
    serving_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_settings: StagedModelSettings
    served_context_tokens: int = Field(ge=16384)
    public_cells: tuple[str, ...] = Field(min_length=2, max_length=2)
    seeds: tuple[int, ...] = Field(min_length=3, max_length=3)
    limits: ModelingLimits
    gates: PrefunctionIntegrationGates
    wall_seconds: int = Field(ge=60)
    shutdown_margin_seconds: int = Field(ge=30)

    @model_validator(mode="after")
    def bounded_unique_tasks(self) -> PrefunctionIntegrationConfig:
        """Fix the intended matrix and require a feasible drain margin."""
        if len(set(self.public_cells)) != 2:
            raise ValueError("exactly two distinct public cells are required")
        if len(set(self.seeds)) != 3 or any(seed < 0 for seed in self.seeds):
            raise ValueError("exactly three distinct nonnegative seeds are required")
        if (
            not self.model_settings.timeout_seconds
            < self.shutdown_margin_seconds
            < self.wall_seconds
        ):
            raise ValueError("require timeout < shutdown margin < worker wall time")
        return self


class HybridPrefunctionGates(StrictSchema):
    """Readiness gates for hybrid variable construction before functions."""

    minimum_topology_completion: float = Field(ge=0, le=1)
    minimum_conditional_target_coverage: float = Field(ge=0, le=1)
    minimum_conditional_source_coverage: float = Field(ge=0, le=1)
    minimum_conditional_topology_obligation_coverage: float = Field(ge=0, le=1)


class HybridPrefunctionConfig(StrictSchema):
    """Real-public hybrid variable and memory-path construction campaign."""

    protocol: Literal["scientific-staged-prefunction-hybrid-2"]
    purpose: str = Field(min_length=1)
    platform: Literal["aces-h100x1"]
    serving_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_settings: StagedModelSettings
    served_context_tokens: int = Field(ge=16384)
    public_cells: tuple[str, ...] = Field(min_length=2, max_length=2)
    seeds: tuple[int, ...] = Field(min_length=3, max_length=3)
    limits: ModelingLimits
    gates: HybridPrefunctionGates
    wall_seconds: int = Field(ge=60)
    shutdown_margin_seconds: int = Field(ge=30)

    @model_validator(mode="after")
    def bounded_unique_tasks(self) -> HybridPrefunctionConfig:
        """Fix the intended matrix and require a feasible drain margin."""
        if len(set(self.public_cells)) != 2:
            raise ValueError("exactly two distinct public cells are required")
        if len(set(self.seeds)) != 3 or any(seed < 0 for seed in self.seeds):
            raise ValueError("exactly three distinct nonnegative seeds are required")
        if (
            not self.model_settings.timeout_seconds
            < self.shutdown_margin_seconds
            < self.wall_seconds
        ):
            raise ValueError("require timeout < shutdown margin < worker wall time")
        return self


PrefunctionConfig = PrefunctionIntegrationConfig | HybridPrefunctionConfig


def _config_from_mapping(value: dict[str, Any]) -> PrefunctionConfig:
    """Load the exact protocol-specific config without schema widening."""
    if value.get("protocol") == "scientific-staged-prefunction-integration-1":
        return PrefunctionIntegrationConfig.model_validate(value)
    if value.get("protocol") == "scientific-staged-prefunction-hybrid-2":
        return HybridPrefunctionConfig.model_validate(value)
    raise ValueError("unsupported prefunction protocol")


def prefunction_launcher_hash() -> str:
    """Bind the CLI and every scheduler launcher used by the frozen plan."""
    repository = Path(__file__).resolve().parents[3]
    paths = (
        "scripts/staged_prefunction_campaign.py",
        "scripts/hpc/run_staged_topology_server.sh",
        "scripts/hpc/staged_prefunction_aces.slurm",
        "scripts/hpc/submit_staged_prefunction_aces.sh",
    )
    return content_hash(
        {
            path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
            for path in paths
        }
    )


def hybrid_prefunction_launcher_hash() -> str:
    """Bind the hybrid CLI and scheduler launchers used by the v2 plan."""
    repository = Path(__file__).resolve().parents[3]
    paths = (
        "scripts/staged_prefunction_campaign.py",
        "scripts/hpc/run_staged_topology_server.sh",
        "scripts/hpc/staged_prefunction_hybrid_aces.slurm",
        "scripts/hpc/submit_staged_prefunction_hybrid_aces.sh",
    )
    return content_hash(
        {
            path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
            for path in paths
        }
    )


def _level_zero_contract(
    mechanism: MechanismEvaluationSpec,
) -> list[dict[str, object]]:
    """Expose reviewed public signs without manufacturing hidden information."""
    return [
        {
            "requirement_id": item.id,
            "targets": list(item.required_targets),
            "drivers": list(item.required_drivers),
            "public_pathway_sign": item.required_sign,
            "direct_exact_source_outer_weight_sign": (
                item.required_sign
                if item.required_sign in {"positive", "negative"}
                else "unrestricted"
            ),
            "indirect_path_sign_inference_performed": False,
            "provenance": "reviewed_public_mechanism_spec",
        }
        for item in mechanism.required_mechanisms
    ]


def freeze_prefunction_campaign(
    config_path: Path, public_root: Path, repository: Path, output: Path
) -> dict[str, Any]:
    """Freeze the exact public briefs, reviewed contracts, settings, and order."""
    config = _config_from_mapping(json.loads(config_path.read_text()))
    tasks: list[dict[str, Any]] = []
    for cell in config.public_cells:
        prompt_path = public_root / "phase_b_v1" / cell / "proposer_prompt.txt"
        target_path = (
            repository / "configs/target_eval/phase_b_v2/specs" / f"{cell}.json"
        )
        mechanism_path = (
            repository / "configs/mechanism_eval/phase_b_v1/specs" / f"{cell}.json"
        )
        prompt = prompt_path.read_text()
        target = PublicTargetContract.model_validate_json(target_path.read_text())
        mechanism = MechanismEvaluationSpec.model_validate_json(
            mechanism_path.read_text()
        )
        context = public_validation_context(cell)
        brief = build_scientific_brief(
            prompt, context, target, mechanism, limits=config.limits
        )
        source = {
            "benchmark_id": cell,
            "public_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "target_contract_sha256": hashlib.sha256(
                target_path.read_bytes()
            ).hexdigest(),
            "mechanism_contract_sha256": hashlib.sha256(
                mechanism_path.read_bytes()
            ).hexdigest(),
            "brief": brief.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "level_zero_polarity_contract": _level_zero_contract(mechanism),
        }
        for seed in config.seeds:
            tasks.append(
                {
                    "task_id": f"{cell}_seed{seed}",
                    "benchmark_id": cell,
                    "seed": seed,
                    "source": source,
                }
            )
    plan = {
        "config": config.model_dump(mode="json"),
        "runtime_source_sha256": runtime_source_hash(),
        "launcher_sha256": (
            hybrid_prefunction_launcher_hash()
            if isinstance(config, HybridPrefunctionConfig)
            else prefunction_launcher_hash()
        ),
        "tasks": tasks,
        "parameter_fitting_performed": False,
        "function_generation_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_winner_defined": False,
    }
    plan["plan_sha256"] = content_hash(plan)
    if output.exists():
        if json.loads(output.read_text()) != plan:
            raise ValueError("existing frozen prefunction campaign differs")
    else:
        atomic_json(output, plan)
    return plan


def _task_result(
    task: dict[str, Any],
    config: PrefunctionConfig,
    base_url: str,
    root: Path,
    identity: str,
    can_start: Any,
) -> dict[str, Any]:
    """Run one public construction using the real staged topology controller."""
    client = StagedTopologyClient(
        settings=config.model_settings,
        base_url=base_url,
        directory=root / "calls",
        namespace=identity,
        seed=task["seed"],
        can_start=can_start,
    )
    source = task["source"]
    return run_staged_topology(
        PublicScientificBrief.model_validate(source["brief"]),
        ValidationContext.model_validate(source["context"]),
        client,
        root,
        audit_public_polarity_policy=True,
        hybrid_variable_construction=isinstance(config, HybridPrefunctionConfig),
        proposer_owns_unfixed_signs=isinstance(config, HybridPrefunctionConfig),
    )


def run_prefunction_campaign(
    plan_path: Path,
    output: Path,
    base_url: str,
    *,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Drain, checkpoint, and deterministically resume the six frozen tasks."""
    plan = json.loads(plan_path.read_text())
    expected = content_hash(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if expected != plan.get("plan_sha256"):
        raise ValueError("frozen prefunction plan digest mismatch")
    if runtime_source_hash() != plan["runtime_source_sha256"]:
        raise ValueError("runtime source differs from frozen prefunction campaign")
    config = _config_from_mapping(plan["config"])
    launcher_hash = (
        hybrid_prefunction_launcher_hash()
        if isinstance(config, HybridPrefunctionConfig)
        else prefunction_launcher_hash()
    )
    if launcher_hash != plan["launcher_sha256"]:
        raise ValueError("launcher differs from frozen prefunction campaign")
    deadline = time.monotonic() + (wall_seconds or config.wall_seconds)
    stop = False

    def drain(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    previous = {
        sig: signal.signal(sig, drain) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    records: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    try:
        for task in plan["tasks"]:
            root = output / task["task_id"]
            root.mkdir(parents=True, exist_ok=True)
            with (root / "worker.lock").open("w") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                identity = content_hash([plan["plan_sha256"], task])
                terminal = root / "terminal.json"
                if terminal.exists():
                    record = json.loads(terminal.read_text())
                    if record["identity"] != identity:
                        raise ValueError("terminal result belongs to another task")
                else:
                    try:
                        result = _task_result(
                            task,
                            config,
                            base_url,
                            root,
                            identity,
                            lambda: not stop
                            and time.monotonic()
                            < deadline - config.shutdown_margin_seconds,
                        )
                    except DeferredCall:
                        break
                    record = {
                        "identity": identity,
                        "task_id": task["task_id"],
                        "benchmark_id": task["benchmark_id"],
                        "seed": task["seed"],
                        "result": result,
                    }
                    atomic_json(terminal, record)
                records.append(record)
                atomic_json(
                    output / "summary.json", summarize_prefunction(plan, records)
                )
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    summary = summarize_prefunction(plan, records)
    atomic_json(output / "summary.json", summary)
    return summary


def _rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def summarize_prefunction(
    plan: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Report topology, source, mechanism, and polarity endpoints separately."""
    if plan["config"]["protocol"] == "scientific-staged-prefunction-hybrid-2":
        return summarize_hybrid_prefunction(plan, records)
    indexed = {item["task_id"]: item for item in records}
    rows: list[dict[str, Any]] = []
    for task in plan["tasks"]:
        record = indexed.get(task["task_id"])
        result = record["result"] if record else None
        audits = result.get("polarity_policy_audits", []) if result else []
        rows.append(
            {
                "task_id": task["task_id"],
                "benchmark_id": task["benchmark_id"],
                "seed": task["seed"],
                "result_present": result is not None,
                "status": result.get("status") if result else None,
                "complete_topology": bool(result and result["complete_topology"]),
                "target_coverage": bool(
                    result and result["public_target_coverage_passed"]
                ),
                "source_coverage": bool(
                    result and result["public_source_coverage_passed"]
                ),
                "mechanism_coverage": bool(
                    result and result["public_mechanism_coverage_passed"]
                ),
                "polarity_consistency": bool(
                    result and result["public_polarity_consistency_passed"]
                ),
                "term_count": sum(int(item["term_count"]) for item in audits),
                "fixed_evidence_term_count": sum(
                    int(item["fixed_evidence_term_count"]) for item in audits
                ),
                "unrestricted_term_count": sum(
                    int(item["unrestricted_term_count"]) for item in audits
                ),
                "polarity_error_count": sum(
                    int(item["term_count"]) - int(item["correct_term_count"])
                    for item in audits
                ),
                "physical_requests": result.get("physical_requests", 0)
                if result
                else 0,
                "observed_total_tokens": result.get("observed_total_tokens", 0)
                if result
                else 0,
                "provider_seconds": result.get("provider_seconds", 0.0)
                if result
                else 0.0,
                "error": result.get("error") if result else None,
            }
        )
    planned = len(rows)
    metrics = {
        "topology_completion_rate": _rate(
            sum(item["complete_topology"] for item in rows), planned
        ),
        "target_coverage_rate": _rate(
            sum(item["target_coverage"] for item in rows), planned
        ),
        "source_coverage_rate": _rate(
            sum(item["source_coverage"] for item in rows), planned
        ),
        "mechanism_coverage_rate": _rate(
            sum(item["mechanism_coverage"] for item in rows), planned
        ),
        "polarity_consistency_rate": _rate(
            sum(item["polarity_consistency"] for item in rows), planned
        ),
    }
    gates = PrefunctionIntegrationGates.model_validate(plan["config"]["gates"])
    checks = {
        "minimum_topology_completion": metrics["topology_completion_rate"] is not None
        and metrics["topology_completion_rate"] >= gates.minimum_topology_completion,
        "minimum_target_coverage": metrics["target_coverage_rate"] is not None
        and metrics["target_coverage_rate"] >= gates.minimum_target_coverage,
        "minimum_source_coverage": metrics["source_coverage_rate"] is not None
        and metrics["source_coverage_rate"] >= gates.minimum_source_coverage,
        "minimum_mechanism_coverage": metrics["mechanism_coverage_rate"] is not None
        and metrics["mechanism_coverage_rate"] >= gates.minimum_mechanism_coverage,
        "minimum_polarity_consistency": metrics["polarity_consistency_rate"] is not None
        and metrics["polarity_consistency_rate"] >= gates.minimum_polarity_consistency,
    }
    complete = sum(item["result_present"] for item in rows) == planned
    return {
        "schema_version": "scientific-staged-prefunction-integration-summary-1",
        "status": "complete" if complete else "incomplete",
        "overall_result": "pass" if complete and all(checks.values()) else "fail",
        "plan_sha256": plan["plan_sha256"],
        "planned_tasks": planned,
        "terminal_results": sum(item["result_present"] for item in rows),
        **metrics,
        "checks": checks,
        "total_terms": sum(item["term_count"] for item in rows),
        "fixed_evidence_terms": sum(item["fixed_evidence_term_count"] for item in rows),
        "unrestricted_terms": sum(item["unrestricted_term_count"] for item in rows),
        "polarity_errors": sum(item["polarity_error_count"] for item in rows),
        "physical_requests": sum(item["physical_requests"] for item in rows),
        "observed_total_tokens": sum(item["observed_total_tokens"] for item in rows),
        "provider_seconds": sum(item["provider_seconds"] for item in rows),
        "rows": rows,
        "function_generation_performed": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_winner_defined": False,
    }


def summarize_hybrid_prefunction(
    plan: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Separate completion from conditional scientific-topology endpoints."""
    indexed = {item["task_id"]: item for item in records}
    rows: list[dict[str, Any]] = []
    for task in plan["tasks"]:
        record = indexed.get(task["task_id"])
        result = record["result"] if record else None
        complete = bool(result and result["complete_topology"])
        audits = result.get("polarity_policy_audits", []) if result else []
        audit_rows = [row for audit in audits for row in audit.get("rows", [])]
        events = result.get("events", []) if result else []
        variable_events = [
            event
            for event in events
            if str(event.get("step", "")).startswith("variables_")
        ]
        equation_events = [
            event
            for event in events
            if str(event.get("step", "")).startswith("equation_")
        ]
        memory_failures = [
            event
            for event in equation_events
            if "dynamic-memory" in str(event.get("error", ""))
        ]
        repaired_memory_steps = {
            str(event["step"])
            for event in memory_failures
            if any(
                later.get("step") == event.get("step") and later.get("accepted")
                for later in equation_events
            )
        }
        sign_counts = {
            sign: sum(
                row.get("observed_outer_weight_sign") == sign for row in audit_rows
            )
            for sign in ("positive", "negative", "unrestricted")
        }
        rows.append(
            {
                "task_id": task["task_id"],
                "benchmark_id": task["benchmark_id"],
                "seed": task["seed"],
                "result_present": result is not None,
                "status": result.get("status") if result else None,
                "complete_topology": complete,
                "target_coverage_conditional": (
                    bool(result["public_target_coverage_passed"]) if complete else None
                ),
                "source_coverage_conditional": (
                    bool(result["public_source_coverage_passed"]) if complete else None
                ),
                "topology_obligation_coverage_conditional": (
                    bool(result["public_mechanism_coverage_passed"])
                    if complete
                    else None
                ),
                "explicit_sign_compliance_conditional": (
                    bool(result["public_polarity_consistency_passed"])
                    if complete
                    else None
                ),
                "explicit_sign_evidence_term_count": sum(
                    bool(row.get("public_fixed_evidence")) for row in audit_rows
                ),
                "explicit_sign_error_count": sum(
                    bool(row.get("public_fixed_evidence"))
                    and not bool(row.get("correct"))
                    for row in audit_rows
                ),
                "proposer_owned_sign_counts": sign_counts,
                "resolved_variable_agenda_skip_count": sum(
                    bool(event.get("skipped_as_resolved")) for event in variable_events
                ),
                "partial_variable_acceptance_count": sum(
                    bool(event.get("partial_acceptance")) for event in variable_events
                ),
                "accepted_variable_entry_count": sum(
                    len(event.get("accepted_variable_names", []))
                    for event in variable_events
                ),
                "rejected_variable_entry_count": sum(
                    len(event.get("rejected_variables", []))
                    for event in variable_events
                ),
                "memory_equation_repair_activation_count": len(memory_failures),
                "memory_equation_repair_recovered_step_count": len(
                    repaired_memory_steps
                ),
                "physical_requests": result.get("physical_requests", 0)
                if result
                else 0,
                "observed_total_tokens": result.get("observed_total_tokens", 0)
                if result
                else 0,
                "provider_seconds": result.get("provider_seconds", 0.0)
                if result
                else 0.0,
                "error": result.get("error") if result else None,
            }
        )
    planned = len(rows)
    completed = [row for row in rows if row["complete_topology"]]

    def conditional_rate(field: str) -> float | None:
        return _rate(sum(bool(row[field]) for row in completed), len(completed))

    metrics = {
        "topology_completion_rate": _rate(len(completed), planned),
        "conditional_target_coverage_rate": conditional_rate(
            "target_coverage_conditional"
        ),
        "conditional_source_coverage_rate": conditional_rate(
            "source_coverage_conditional"
        ),
        "conditional_topology_obligation_coverage_rate": conditional_rate(
            "topology_obligation_coverage_conditional"
        ),
    }
    gates = HybridPrefunctionGates.model_validate(plan["config"]["gates"])
    checks = {
        "minimum_topology_completion": metrics["topology_completion_rate"] is not None
        and metrics["topology_completion_rate"] >= gates.minimum_topology_completion,
        "minimum_conditional_target_coverage": metrics[
            "conditional_target_coverage_rate"
        ]
        is not None
        and metrics["conditional_target_coverage_rate"]
        >= gates.minimum_conditional_target_coverage,
        "minimum_conditional_source_coverage": metrics[
            "conditional_source_coverage_rate"
        ]
        is not None
        and metrics["conditional_source_coverage_rate"]
        >= gates.minimum_conditional_source_coverage,
        "minimum_conditional_topology_obligation_coverage": metrics[
            "conditional_topology_obligation_coverage_rate"
        ]
        is not None
        and metrics["conditional_topology_obligation_coverage_rate"]
        >= gates.minimum_conditional_topology_obligation_coverage,
    }
    terminal = sum(bool(row["result_present"]) for row in rows)
    return {
        "schema_version": "scientific-staged-prefunction-hybrid-summary-2",
        "status": "complete" if terminal == planned else "incomplete",
        "overall_result": (
            "pass" if terminal == planned and all(checks.values()) else "fail"
        ),
        "plan_sha256": plan["plan_sha256"],
        "planned_tasks": planned,
        "terminal_results": terminal,
        **metrics,
        "checks": checks,
        "explicit_sign_evidence_term_count": sum(
            row["explicit_sign_evidence_term_count"] for row in rows
        ),
        "explicit_sign_error_count": sum(
            row["explicit_sign_error_count"] for row in rows
        ),
        "proposer_owned_sign_counts": {
            sign: sum(row["proposer_owned_sign_counts"][sign] for row in rows)
            for sign in ("positive", "negative", "unrestricted")
        },
        "resolved_variable_agenda_skip_count": sum(
            row["resolved_variable_agenda_skip_count"] for row in rows
        ),
        "partial_variable_acceptance_count": sum(
            row["partial_variable_acceptance_count"] for row in rows
        ),
        "accepted_variable_entry_count": sum(
            row["accepted_variable_entry_count"] for row in rows
        ),
        "rejected_variable_entry_count": sum(
            row["rejected_variable_entry_count"] for row in rows
        ),
        "memory_equation_repair_activation_count": sum(
            row["memory_equation_repair_activation_count"] for row in rows
        ),
        "memory_equation_repair_recovered_step_count": sum(
            row["memory_equation_repair_recovered_step_count"] for row in rows
        ),
        "physical_requests": sum(row["physical_requests"] for row in rows),
        "observed_total_tokens": sum(row["observed_total_tokens"] for row in rows),
        "provider_seconds": sum(row["provider_seconds"] for row in rows),
        "rows": rows,
        "function_generation_performed": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_winner_defined": False,
    }
