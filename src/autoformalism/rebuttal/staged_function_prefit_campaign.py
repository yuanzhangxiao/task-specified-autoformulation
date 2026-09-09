"""Freeze and run functions on exact passed public pre-function topologies."""

from __future__ import annotations

import ast
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
from autoformalism.rebuttal.staged_prefunction_campaign import (
    HybridPrefunctionConfig,
)
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.candidate import CandidateModel, StateKind
from autoformalism.schemas.construction import FunctionalDraft
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    ModelingLimits,
    PublicScientificBrief,
)
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.staged_functions import has_nonlinear_source_dependence
from autoformalism.staged_topology import content_hash


class FunctionPrefitGates(StrictSchema):
    """Prospective readiness gates for the combined pre-fit handoff."""

    minimum_complete_model_rate: float = Field(ge=0, le=1)
    minimum_deterministic_prefit_pass_rate: float = Field(ge=0, le=1)
    minimum_nonlinear_obligation_pass_rate: float = Field(ge=0, le=1)
    minimum_latent_initializer_coverage_rate: float = Field(ge=0, le=1)


class FunctionPrefitConfig(StrictSchema):
    """One function realization for every passed topology in the 2x3 matrix."""

    protocol: Literal["scientific-staged-function-prefit-handoff-1"]
    purpose: str = Field(min_length=1)
    platform: Literal["aces-h100x1"]
    serving_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_settings: StagedModelSettings
    served_context_tokens: int = Field(ge=16384)
    public_cells: tuple[str, ...] = Field(min_length=2, max_length=2)
    seeds: tuple[int, ...] = Field(min_length=3, max_length=3)
    limits: ModelingLimits
    gates: FunctionPrefitGates
    generation_granularity: Literal["equation_batch_atomic_repair"]
    function_repair_policy: Literal["certified_outer_gain"]
    wall_seconds: int = Field(ge=60)
    shutdown_margin_seconds: int = Field(ge=30)

    @model_validator(mode="after")
    def fixed_public_matrix(self) -> FunctionPrefitConfig:
        """Reject accidental changes to the intended public 2x3 handoff."""
        if len(set(self.public_cells)) != 2:
            raise ValueError("exactly two distinct public cells are required")
        if len(set(self.seeds)) != 3 or any(seed < 0 for seed in self.seeds):
            raise ValueError("exactly three distinct nonnegative seeds are required")
        if not (
            self.model_settings.timeout_seconds
            < self.shutdown_margin_seconds
            < self.wall_seconds
        ):
            raise ValueError("require timeout < shutdown margin < worker wall time")
        return self


def function_prefit_launcher_hash() -> str:
    """Bind the campaign CLI and ACES launch path to the frozen plan."""
    repository = Path(__file__).resolve().parents[3]
    paths = (
        "scripts/staged_function_prefit_campaign.py",
        "scripts/hpc/run_staged_topology_server.sh",
        "scripts/hpc/staged_function_prefit_aces.slurm",
        "scripts/hpc/submit_staged_function_prefit_aces.sh",
    )
    return content_hash(
        {
            path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
            for path in paths
        }
    )


def freeze_function_prefit_campaign(
    config_path: Path,
    source_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Pin every passed topology artifact and its public Level-0 contract."""
    config = FunctionPrefitConfig.model_validate_json(config_path.read_text())
    source_plan_path = source_root / "plan.json"
    source_summary_path = source_root / "results" / "summary.json"
    source_plan = json.loads(source_plan_path.read_text())
    source_summary = json.loads(source_summary_path.read_text())
    source_digest = content_hash(
        {key: value for key, value in source_plan.items() if key != "plan_sha256"}
    )
    if source_digest != source_plan.get("plan_sha256"):
        raise ValueError("source prefunction plan digest mismatch")
    source_config = HybridPrefunctionConfig.model_validate(source_plan["config"])
    _require_matched_source(config, source_config)
    expected_source_checks = {
        "minimum_topology_completion",
        "minimum_conditional_target_coverage",
        "minimum_conditional_source_coverage",
        "minimum_conditional_topology_obligation_coverage",
    }
    source_checks = source_summary.get("checks", {})
    if (
        source_summary.get("plan_sha256") != source_digest
        or source_summary.get("status") != "complete"
        or source_summary.get("overall_result") != "pass"
        or source_summary.get("terminal_results") != 6
        or set(source_checks) != expected_source_checks
        or not all(source_checks.values())
    ):
        raise ValueError("source prefunction campaign did not pass its frozen gates")

    tasks = []
    for original in source_plan["tasks"]:
        source_task_id = original["task_id"]
        source_result_path = source_root / "results" / source_task_id / "result.json"
        source_terminal_path = (
            source_root / "results" / source_task_id / "terminal.json"
        )
        source_result = json.loads(source_result_path.read_text())
        source_terminal = json.loads(source_terminal_path.read_text())
        expected_identity = content_hash([source_digest, original])
        if (
            source_terminal.get("identity") != expected_identity
            or source_terminal.get("result") != source_result
        ):
            raise ValueError(f"source terminal mismatch: {source_task_id}")
        if (
            source_result.get("status") != "complete"
            or not source_result.get("complete_topology")
            or not source_result.get("public_structure_checks_passed")
            or not source_result.get("public_target_coverage_passed")
            or not source_result.get("public_source_coverage_passed")
            or not source_result.get("public_mechanism_coverage_passed")
            or source_result.get("test_data_opened") is not False
            or source_result.get("private_reference_opened") is not False
            or source_result.get("function_generation_performed") is not False
            or source_result.get("parameter_fitting_performed") is not False
        ):
            raise ValueError(
                "source topology is not a passed public handoff: "
                f"{source_task_id}"
            )
        public = original["source"]
        tasks.append(
            {
                "task_id": f"{source_task_id}_functions",
                "source_task_id": source_task_id,
                "benchmark_id": original["benchmark_id"],
                "seed": original["seed"],
                "brief": public["brief"],
                "context": public["context"],
                "source": source_result,
                "source_result_sha256": content_hash(source_result),
                "generation_granularity": config.generation_granularity,
                "function_repair_policy": config.function_repair_policy,
            }
        )
    expected_order = [
        (cell, seed) for cell in config.public_cells for seed in config.seeds
    ]
    observed_order = [(item["benchmark_id"], item["seed"]) for item in tasks]
    if observed_order != expected_order:
        raise ValueError("source task matrix or order differs from the frozen config")

    plan = {
        "schema_version": "scientific-staged-function-prefit-handoff-plan-1",
        "config": config.model_dump(mode="json"),
        "source_plan_sha256": source_digest,
        "source_summary_sha256": content_hash(source_summary),
        "source_artifact_ledger_sha256": content_hash(
            [
                {
                    "task_id": task["source_task_id"],
                    "result_sha256": task["source_result_sha256"],
                }
                for task in tasks
            ]
        ),
        "runtime_source_sha256": runtime_source_hash(),
        "launcher_sha256": function_prefit_launcher_hash(),
        "tasks": tasks,
        "source_topologies_regenerated": False,
        "function_generation_performed": True,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_winner_defined": False,
    }
    plan["plan_sha256"] = content_hash(plan)
    if output.exists():
        if json.loads(output.read_text()) != plan:
            raise ValueError("existing frozen function prefit campaign differs")
    else:
        atomic_json(output, plan)
    return plan


def _require_matched_source(
    config: FunctionPrefitConfig,
    source: HybridPrefunctionConfig,
) -> None:
    """Match scientific and serving settings while allowing larger call ledgers."""
    comparisons = (
        (config.platform, source.platform, "platform"),
        (config.serving_image_sha256, source.serving_image_sha256, "image"),
        (config.served_context_tokens, source.served_context_tokens, "context"),
        (config.public_cells, source.public_cells, "public cells"),
        (config.seeds, source.seeds, "seeds"),
        (config.limits, source.limits, "modeling limits"),
    )
    for actual, expected, label in comparisons:
        if actual != expected:
            raise ValueError(f"function prefit campaign {label} differ from source")
    inference_fields = (
        "model",
        "model_revision",
        "reasoning_effort",
        "temperature",
        "max_output_tokens",
        "timeout_seconds",
        "attempts_per_step",
    )
    for field in inference_fields:
        actual = getattr(config.model_settings, field)
        expected = getattr(source.model_settings, field)
        if actual != expected:
            raise ValueError(f"function prefit model setting differs: {field}")


def deterministic_prefit_audit(
    brief: PublicScientificBrief,
    source: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Certify only mechanically decidable completeness and syntax facts."""
    candidate = CandidateModel.model_validate(result["candidate"])
    draft = FunctionalDraft.model_validate(result["draft"])
    equations = tuple(
        EquationDefinition.model_validate(item) for item in source["equations"]
    )
    expected_terms = [
        (equation.name, term)
        for equation in equations
        for term in equation.terms
    ]
    accepted = result["accepted_functions"]
    sign_and_source_match = len(accepted) == len(expected_terms)
    nonlinear_required = 0
    nonlinear_passed = 0
    if sign_and_source_match:
        for accepted_item, (lhs, term) in zip(accepted, expected_terms, strict=True):
            selected = accepted_item["selected_term"]
            expected_sign = (
                term.outer_weight_sign.value
                if hasattr(term, "outer_weight_sign")
                else ("positive" if term.outer_sign == "add" else "negative")
            )
            if (
                selected["lhs"] != lhs
                or tuple(selected["sources"]) != term.sources
                or selected["outer_weight_sign"] != expected_sign
            ):
                sign_and_source_match = False
                break
            obligation = selected["functional_obligation"]
            if obligation["requires_nonlinear_source_dependence"]:
                nonlinear_required += 1
                tree = ast.parse(accepted_item["expression"], mode="eval")
                nonlinear_passed += has_nonlinear_source_dependence(
                    tree, set(term.sources)
                )
    targets = {
        item.name for item in brief.public_variables if item.data_role == "target"
    }
    mapped = [item.channel for item in candidate.observation_mappings]
    latent = {item.name for item in candidate.states if item.kind is StateKind.LATENT}
    initialized = {item.state for item in candidate.initial_conditions}
    checks = {
        "source_topology_passed": bool(
            source.get("complete_topology")
            and source.get("public_structure_checks_passed")
        ),
        "source_topology_digest_preserved": (
            result.get("source_topology_result_sha256") == content_hash(source)
        ),
        "complete_compiled_model": bool(result.get("complete_model")),
        "exact_target_mapping_coverage": (
            len(mapped) == len(set(mapped)) and set(mapped) == targets
        ),
        "exact_interaction_function_coverage": (
            len(draft.interaction_functions) == len(expected_terms)
            and len(accepted) == len(expected_terms)
        ),
        "frozen_term_sources_and_signs_preserved": sign_and_source_match,
        "latent_initializer_coverage": initialized & latent == latent,
        "nonlinear_obligations_satisfied": nonlinear_passed == nonlinear_required,
        "candidate_schema_and_expression_contracts_validated": True,
    }
    return {
        "schema_version": "deterministic-prefit-certificate-1",
        "passed": all(checks.values()),
        "checks": checks,
        "target_channels": sorted(targets),
        "mapped_target_channels": mapped,
        "interaction_count": len(expected_terms),
        "nonlinear_obligation_count": nonlinear_required,
        "nonlinear_obligation_pass_count": nonlinear_passed,
        "latent_state_count": len(latent),
        "initialized_latent_state_count": len(initialized & latent),
        "parameter_count": len(candidate.parameters),
        "scientific_scope_note": (
            "This certificate does not judge scientific adequacy, units, "
            "parsimony, or empirical fit."
        ),
        "parameter_fitting_performed": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }


def run_function_prefit_campaign(
    plan_path: Path,
    output: Path,
    base_url: str,
    *,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Generate functions for the frozen matrix with resume and drain."""
    plan = json.loads(plan_path.read_text())
    if content_hash(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ) != plan.get("plan_sha256"):
        raise ValueError("frozen function prefit plan digest mismatch")
    if runtime_source_hash() != plan["runtime_source_sha256"]:
        raise ValueError("runtime source differs from frozen function prefit campaign")
    if function_prefit_launcher_hash() != plan["launcher_sha256"]:
        raise ValueError("launcher differs from frozen function prefit campaign")
    config = FunctionPrefitConfig.model_validate(plan["config"])
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
                        raise ValueError(
                            "terminal result belongs to another frozen task"
                        )
                else:
                    client = StagedTopologyClient(
                        settings=config.model_settings,
                        base_url=base_url,
                        directory=root / "calls",
                        namespace=identity,
                        seed=task["seed"],
                        can_start=lambda: not stop
                        and time.monotonic()
                        < deadline - config.shutdown_margin_seconds,
                    )
                    try:
                        result = run_staged_functions(
                            PublicScientificBrief.model_validate(task["brief"]),
                            ValidationContext.model_validate(task["context"]),
                            task["source"],
                            client,
                            root,
                            generation_granularity=task["generation_granularity"],
                            function_repair_policy=task["function_repair_policy"],
                        )
                    except DeferredCall:
                        break
                    if result.get("complete_model"):
                        result["deterministic_prefit_certificate"] = (
                            deterministic_prefit_audit(
                                PublicScientificBrief.model_validate(task["brief"]),
                                task["source"],
                                result,
                            )
                        )
                        atomic_json(root / "result.json", result)
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
                    output / "summary.json",
                    summarize_function_prefit(plan, records),
                )
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    summary = summarize_function_prefit(plan, records)
    atomic_json(output / "summary.json", summary)
    return summary


def summarize_function_prefit(
    plan: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Keep model, repair, initializer, and deterministic endpoints separate."""
    indexed = {record["task_id"]: record for record in records}
    rows = [_summary_row(task, indexed.get(task["task_id"])) for task in plan["tasks"]]
    complete = [row for row in rows if row["complete_model"]]
    audits = [audit for row in rows for audit in row["batch_term_audits"]]
    repaired = [audit for audit in audits if audit["atomic_repair_attempted"]]
    certificates = [
        row["deterministic_prefit_certificate"]
        for row in complete
        if row["deterministic_prefit_certificate"] is not None
    ]
    nonlinear_total = sum(item["nonlinear_obligation_count"] for item in certificates)
    nonlinear_passed = sum(
        item["nonlinear_obligation_pass_count"] for item in certificates
    )
    latent_total = sum(item["latent_state_count"] for item in certificates)
    latent_initialized = sum(
        item["initialized_latent_state_count"] for item in certificates
    )
    metrics = {
        "complete_model_rate": _ratio(len(complete), len(rows)),
        "deterministic_prefit_pass_rate": _ratio(
            sum(item["passed"] for item in certificates), len(rows)
        ),
        "nonlinear_obligation_pass_rate": _ratio(
            nonlinear_passed, nonlinear_total
        ) if nonlinear_total else 1.0,
        "latent_initializer_coverage_rate": _ratio(
            latent_initialized, latent_total
        ) if latent_total else 1.0,
    }
    gates = FunctionPrefitGates.model_validate(plan["config"]["gates"])
    checks = {
        "minimum_complete_model_rate": _meets(
            metrics["complete_model_rate"], gates.minimum_complete_model_rate
        ),
        "minimum_deterministic_prefit_pass_rate": _meets(
            metrics["deterministic_prefit_pass_rate"],
            gates.minimum_deterministic_prefit_pass_rate,
        ),
        "minimum_nonlinear_obligation_pass_rate": _meets(
            metrics["nonlinear_obligation_pass_rate"],
            gates.minimum_nonlinear_obligation_pass_rate,
        ),
        "minimum_latent_initializer_coverage_rate": _meets(
            metrics["latent_initializer_coverage_rate"],
            gates.minimum_latent_initializer_coverage_rate,
        ),
    }
    terminal = sum(row["result_present"] for row in rows)
    return {
        "schema_version": "scientific-staged-function-prefit-handoff-summary-1",
        "status": "complete" if terminal == len(rows) else "incomplete",
        "overall_result": (
            "pass" if terminal == len(rows) and all(checks.values()) else "fail"
        ),
        "plan_sha256": plan["plan_sha256"],
        "source_plan_sha256": plan["source_plan_sha256"],
        "source_artifact_ledger_sha256": plan["source_artifact_ledger_sha256"],
        "planned_tasks": len(rows),
        "terminal_results": terminal,
        **metrics,
        "checks": checks,
        "batch_term_count": len(audits),
        "batch_term_acceptance_rate": _ratio(
            sum(audit["batch_accepted"] for audit in audits), len(audits)
        ),
        "atomic_repair_activation_count": len(repaired),
        "atomic_repair_success_rate": _ratio(
            sum(audit["atomic_repair_succeeded"] for audit in repaired),
            len(repaired),
        ),
        "physical_requests": sum(row["physical_requests"] for row in rows),
        "observed_total_tokens": sum(row["observed_total_tokens"] for row in rows),
        "provider_seconds": sum(row["provider_seconds"] for row in rows),
        "rows": rows,
        "source_topologies_regenerated": False,
        "function_generation_performed": True,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_winner_defined": False,
    }


def _summary_row(task: dict[str, Any], record: dict[str, Any] | None) -> dict[str, Any]:
    result = record["result"] if record is not None else None
    return {
        "task_id": task["task_id"],
        "source_task_id": task["source_task_id"],
        "benchmark_id": task["benchmark_id"],
        "seed": task["seed"],
        "source_result_sha256": task["source_result_sha256"],
        "result_present": record is not None,
        "status": result.get("status") if result else None,
        "complete_model": bool(result and result.get("complete_model")),
        "error": result.get("error") if result else None,
        "batch_term_audits": result.get("batch_term_audits", []) if result else [],
        "deterministic_prefit_certificate": (
            result.get("deterministic_prefit_certificate") if result else None
        ),
        "physical_requests": result.get("physical_requests", 0) if result else 0,
        "observed_total_tokens": (
            result.get("observed_total_tokens", 0) if result else 0
        ),
        "provider_seconds": result.get("provider_seconds", 0.0) if result else 0.0,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _meets(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold
