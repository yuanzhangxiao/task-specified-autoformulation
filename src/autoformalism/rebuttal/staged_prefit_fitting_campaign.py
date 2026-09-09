"""Freeze and fit the exact six-candidate staged pre-fit handoff."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import statistics
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import monotonic, process_time
from typing import Any, Literal

from pydantic import Field, model_validator

from autoformalism.baselines.raw_data_agent import (
    fit_result_payload,
    raw_agent_validation_context,
)
from autoformalism.config import DataConfig
from autoformalism.data import (
    BenchmarkLoader,
    BenchmarkRegistry,
    DerivativeProvenance,
    DevelopmentDataset,
)
from autoformalism.expressions import (
    ModelValidationError,
    compile_candidate,
    validate_profiled_latent_basis_parameterization,
)
from autoformalism.fitting import FitConfig, fit_candidate
from autoformalism.rebuttal.fitter_diagnostic import PUBLIC_FILES
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.search.identity import candidate_identity
from autoformalism.staged_topology import content_hash


class PrefitFittingConfig(StrictSchema):
    """Frozen source identity and one deterministic fitting policy."""

    protocol: Literal["scientific-staged-prefit-fitting-handoff-1"]
    purpose: str = Field(min_length=1)
    source_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_summary_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_cells: tuple[str, ...] = Field(min_length=2, max_length=2)
    seeds: tuple[int, ...] = Field(min_length=3, max_length=3)
    public_asset_sha256: dict[str, dict[str, str]]
    route_policy: Literal["exact_profiled_else_bounded_rollout"]
    profiled_fit_config: FitConfig
    rollout_fit_config: FitConfig

    @model_validator(mode="after")
    def fixed_policy(self) -> PrefitFittingConfig:
        """Reject matrix expansion or a route that can use estimated derivatives."""
        if len(set(self.public_cells)) != 2:
            raise ValueError("exactly two distinct public cells are required")
        if len(set(self.seeds)) != 3 or any(seed < 0 for seed in self.seeds):
            raise ValueError("exactly three distinct nonnegative seeds are required")
        if set(self.public_asset_sha256) != set(self.public_cells):
            raise ValueError("public asset cells differ from the fitting matrix")
        for benchmark_id, assets in self.public_asset_sha256.items():
            if set(assets) != set(PUBLIC_FILES) or any(
                re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for digest in assets.values()
            ):
                raise ValueError(f"invalid public asset ledger: {benchmark_id}")
        profiled = self.profiled_fit_config
        rollout = self.rollout_fit_config
        if (
            profiled.parameter_fit_strategy != "profiled_latent_basis_linear_ridge"
            or not profiled.allow_derivative_regression
        ):
            raise ValueError(
                "profiled route requires exact-derivative variable projection"
            )
        if (
            rollout.parameter_fit_strategy != "bounded_nonlinear"
            or rollout.allow_derivative_regression
        ):
            raise ValueError("rollout route must be derivative-free bounded nonlinear")
        for settings in (profiled, rollout):
            if settings.maximum_wall_time_seconds is None:
                raise ValueError("every fit route requires a wall-clock limit")
        return self


def launcher_hash() -> str:
    """Bind all commands that can create or consume frozen artifacts."""
    repository = Path(__file__).resolve().parents[3]
    paths = (
        "scripts/staged_prefit_fitting_campaign.py",
        "scripts/hpc/staged_prefit_fitting_prepare_cpu.slurm",
        "scripts/hpc/staged_prefit_fitting_worker_cpu.slurm",
        "scripts/hpc/staged_prefit_fitting_summary_cpu.slurm",
        "scripts/hpc/submit_staged_prefit_fitting_aces.sh",
    )
    return content_hash(
        {
            path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
            for path in paths
        }
    )


def load_public_data(
    public_root: Path, benchmark_id: str, tier: str
) -> tuple[DevelopmentDataset, Any]:
    """Open only public train/validation tables and build their runtime context."""
    registry = BenchmarkRegistry()
    spec = registry.get(benchmark_id)
    if spec.data_layout != "tidy_split_file" or spec.one_step_target_history:
        raise ValueError("fitting handoff requires Phase-B free-rollout public data")
    dataset = BenchmarkLoader(registry).load_development(
        DataConfig(benchmark_id=benchmark_id, tier=tier, root=public_root)
    )
    return dataset, raw_agent_validation_context(dataset, spec)


def fitting_route(
    candidate: CandidateModel, dataset: DevelopmentDataset, context: Any
) -> dict[str, Any]:
    """Choose the backend from structure and train-data provenance before fitting."""
    model = compile_candidate(candidate, context)
    diagnostics: list[dict[str, str]] = []
    try:
        report = validate_profiled_latent_basis_parameterization(model.validated)
    except ModelValidationError as exc:
        structural = False
        diagnostics = [asdict(item) for item in exc.diagnostics]
        affine: list[str] = []
        outer: list[str] = []
        reciprocal: list[dict[str, Any]] = []
    else:
        structural = True
        affine = list(report.affine_parameter_names)
        outer = list(report.outer_parameter_names)
        reciprocal = [asdict(item) for item in report.reciprocal_transformations]
    direct = {
        state: channel
        for state, channel in model.direct_state_observation_channels.items()
        if channel in context.targets
    }
    provenance = sorted(
        {
            trajectory.derivative_provenance.value
            for trajectory in dataset.train.trajectories
        }
    )
    exact = bool(direct) and all(
        trajectory.derivative_provenance is DerivativeProvenance.EXACT
        and set(direct.values()) <= set(trajectory.derivatives)
        for trajectory in dataset.train.trajectories
    )
    selected = (
        "profiled_latent_basis_linear_ridge"
        if structural and exact
        else "bounded_nonlinear"
    )
    if not structural:
        reason = "profiled structural certificate unavailable"
    elif not exact:
        reason = "public training derivatives lack exact provenance"
    else:
        reason = "profiled structure and exact public derivative contract certified"
    return {
        "schema_version": "staged-prefit-fitting-route-1",
        "route_policy": "exact_profiled_else_bounded_rollout",
        "selected_backend": selected,
        "selection_reason": reason,
        "profiled_structural_compatible": structural,
        "profiled_structural_diagnostics": diagnostics,
        "profiled_affine_parameters": affine,
        "profiled_outer_parameters": outer,
        "certified_reciprocal_transformations": reciprocal,
        "direct_target_state_channels": direct,
        "training_derivative_provenance": provenance,
        "exact_training_derivative_contract_available": exact,
        "validation_used_for_routing": False,
    }


def candidate_audit(
    candidate: CandidateModel, source: dict[str, Any]
) -> dict[str, Any]:
    """Expose compact mechanical facts without assigning a scientific score."""
    identity = candidate_identity(candidate).model_dump(mode="json")
    roles = Counter(item.role.value for item in candidate.parameters)
    domains = Counter(item.domain.value for item in candidate.parameters)
    latent = [item.name for item in candidate.states if item.kind.value == "latent"]
    initials = {
        item.state: (
            {"kind": "fixed_value", "value": item.fixed_value}
            if item.fixed_value is not None
            else {"kind": "expression", "expression": item.expression}
        )
        for item in candidate.initial_conditions
        if item.state in latent
    }
    equations = [
        {"lhs": f"d({item.state})/dt", "rhs": item.rhs}
        for item in candidate.state_equations
    ] + [{"lhs": item.name, "rhs": item.expression} for item in candidate.processes]
    audits = source.get("batch_term_audits", [])
    return {
        "schema_version": "staged-prefit-candidate-audit-1",
        "candidate_identity": identity,
        "equations": equations,
        "observation_mappings": [
            item.model_dump(mode="json") for item in candidate.observation_mappings
        ],
        "state_count": len(candidate.states),
        "latent_state_names": latent,
        "algebraic_process_count": len(candidate.processes),
        "parameter_count": len(candidate.parameters),
        "parameter_role_counts": dict(sorted(roles.items())),
        "parameter_domain_counts": dict(sorted(domains.items())),
        "parameters": [item.model_dump(mode="json") for item in candidate.parameters],
        "latent_initializers": initials,
        "atomic_repair_activation_count": sum(
            bool(item.get("atomic_repair_attempted")) for item in audits
        ),
        "deterministic_prefit_certificate": source.get(
            "deterministic_prefit_certificate"
        ),
        "scientific_score_assigned": False,
    }


def freeze_campaign(
    config_path: Path,
    source_root: Path,
    public_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Freeze exact candidate results, public development files, audits and routes."""
    config = PrefitFittingConfig.model_validate_json(config_path.read_text())
    source_plan_path = source_root / "plan.json"
    source_summary_path = source_root / "results" / "summary.json"
    source_plan = _read_object(source_plan_path)
    source_summary = _read_object(source_summary_path)
    source_digest = content_hash(
        {key: value for key, value in source_plan.items() if key != "plan_sha256"}
    )
    if (
        source_digest != source_plan.get("plan_sha256")
        or source_digest != config.source_plan_sha256
    ):
        raise ValueError("source function plan digest differs")
    if _sha256(source_summary_path) != config.source_summary_file_sha256:
        raise ValueError("source function summary file differs")
    if (
        source_summary.get("plan_sha256") != source_digest
        or source_summary.get("status") != "complete"
        or source_summary.get("overall_result") != "pass"
        or source_summary.get("terminal_results") != 6
        or source_summary.get("complete_model_rate") != 1.0
        or source_summary.get("deterministic_prefit_pass_rate") != 1.0
        or source_summary.get("test_data_opened") is not False
        or source_summary.get("private_reference_opened") is not False
    ):
        raise ValueError("source function campaign did not pass its frozen gates")

    frozen = output_root / "frozen"
    public_assets: dict[str, str] = {}
    datasets: dict[str, tuple[DevelopmentDataset, Any]] = {}
    for benchmark_id in config.public_cells:
        source_public = public_root / "phase_b_v1" / benchmark_id
        for name in PUBLIC_FILES:
            expected_hash = config.public_asset_sha256[benchmark_id][name]
            if _sha256(source_public / name) != expected_hash:
                raise ValueError(f"public input differs: {benchmark_id}/{name}")
            destination = frozen / "public" / "phase_b_v1" / benchmark_id / name
            _copy_once(source_public / name, destination)
            public_assets[str(destination.relative_to(output_root))] = _sha256(
                destination
            )
        tier = benchmark_id.rsplit("_", 1)[-1]
        datasets[benchmark_id] = load_public_data(frozen / "public", benchmark_id, tier)

    expected = [(cell, seed) for cell in config.public_cells for seed in config.seeds]
    tasks: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    observed: list[tuple[str, int]] = []
    for index, source_task in enumerate(source_plan.get("tasks", [])):
        benchmark_id = str(source_task["benchmark_id"])
        seed = int(source_task["seed"])
        observed.append((benchmark_id, seed))
        source_task_id = str(source_task["task_id"])
        result_path = source_root / "results" / source_task_id / "result.json"
        terminal_path = source_root / "results" / source_task_id / "terminal.json"
        result = _read_object(result_path)
        terminal = _read_object(terminal_path)
        if (
            terminal.get("identity") != content_hash([source_digest, source_task])
            or terminal.get("result") != result
        ):
            raise ValueError(f"source result identity differs: {source_task_id}")
        certificate = result.get("deterministic_prefit_certificate", {})
        if (
            result.get("status") != "complete"
            or result.get("complete_model") is not True
            or certificate.get("passed") is not True
            or result.get("parameter_fitting_performed") is not False
            or result.get("test_data_opened") is not False
            or result.get("private_reference_opened") is not False
        ):
            raise ValueError(
                f"source candidate is not a passed pre-fit artifact: {source_task_id}"
            )
        candidate = CandidateModel.model_validate(result["candidate"])
        prompt_path = (
            frozen / "public" / "phase_b_v1" / benchmark_id / "proposer_prompt.txt"
        )
        prompt_parts = re.split(
            r"(?m)^F\.\s+Required response\s*$",
            prompt_path.read_text(),
            maxsplit=1,
        )
        if (
            len(prompt_parts) != 2
            or prompt_parts[0].rstrip() != source_task["brief"]["scientific_context"]
        ):
            raise ValueError(f"public prompt differs from source brief: {benchmark_id}")
        dataset, context = datasets[benchmark_id]
        route = fitting_route(candidate, dataset, context)
        audit = candidate_audit(candidate, result)
        task_id = f"{source_task_id}_fit"
        candidate_path = frozen / "candidates" / f"candidate_{index:03d}.json"
        source_path = frozen / "sources" / f"source_{index:03d}.json"
        _write_once_json(candidate_path, candidate.model_dump(mode="json"))
        _write_once_json(source_path, result)
        row = {
            "task_index": index,
            "task_id": task_id,
            "source_task_id": source_task_id,
            "benchmark_id": benchmark_id,
            "tier": benchmark_id.rsplit("_", 1)[-1],
            "seed": seed,
            "candidate_path": str(candidate_path.relative_to(output_root)),
            "candidate_file_sha256": _sha256(candidate_path),
            "source_path": str(source_path.relative_to(output_root)),
            "source_file_sha256": _sha256(source_path),
            "source_result_sha256": content_hash(result),
            "candidate_audit": audit,
            "route": route,
        }
        tasks.append(row)
        audits.append(
            {
                key: row[key]
                for key in (
                    "task_index",
                    "task_id",
                    "benchmark_id",
                    "seed",
                    "candidate_audit",
                    "route",
                )
            }
        )
    if observed != expected or len(tasks) != 6:
        raise ValueError("source candidate matrix or ordering differs")

    _write_once_json(frozen / "source_plan.json", source_plan)
    _write_once_json(frozen / "source_summary.json", source_summary)
    _write_once_json(frozen / "candidate_audit.json", {"rows": audits})
    plan = {
        "schema_version": "scientific-staged-prefit-fitting-handoff-plan-1",
        "config": config.model_dump(mode="json"),
        "source_plan_sha256": source_digest,
        "source_summary_file_sha256": _sha256(source_summary_path),
        "source_candidate_ledger_sha256": content_hash(
            [
                {
                    "task_id": task["task_id"],
                    "candidate_file_sha256": task["candidate_file_sha256"],
                    "source_file_sha256": task["source_file_sha256"],
                }
                for task in tasks
            ]
        ),
        "public_asset_ledger": public_assets,
        "public_asset_ledger_sha256": content_hash(public_assets),
        "runtime_source_sha256": runtime_source_hash(),
        "launcher_sha256": launcher_hash(),
        "tasks": tasks,
        "candidate_regeneration_performed": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_winner_defined": False,
    }
    plan["plan_sha256"] = content_hash(plan)
    _write_once_json(output_root / "plan.json", plan)
    freeze = {
        "schema_version": "scientific-staged-prefit-fitting-freeze-1",
        "status": "frozen_before_fitting",
        "plan_sha256": plan["plan_sha256"],
        "source_candidate_ledger_sha256": plan["source_candidate_ledger_sha256"],
        "public_asset_ledger_sha256": plan["public_asset_ledger_sha256"],
        "task_count": len(tasks),
        "candidate_regeneration_performed": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    _write_once_json(frozen / "freeze_manifest.json", freeze)
    _write_once(
        frozen / "candidate_audit.md",
        _audit_markdown(audits).encode(),
    )
    return freeze


def run_task(output_root: Path, task_index: int) -> dict[str, Any]:
    """Fit one exact frozen candidate and checkpoint every terminal outcome."""
    plan = _verified_plan(output_root)
    tasks = plan["tasks"]
    if not 0 <= task_index < len(tasks):
        raise ValueError(f"task index out of range: {task_index}")
    task = tasks[task_index]
    destination = output_root / "tasks" / f"task_{task_index:03d}.json"
    if destination.exists():
        existing = _read_object(destination)
        if existing.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("fit checkpoint belongs to another frozen plan")
        return existing
    candidate_path = output_root / task["candidate_path"]
    if _sha256(candidate_path) != task["candidate_file_sha256"]:
        raise ValueError("frozen candidate differs")
    dataset, context = load_public_data(
        output_root / "frozen" / "public", task["benchmark_id"], task["tier"]
    )
    candidate = CandidateModel.model_validate_json(candidate_path.read_text())
    observed_route = fitting_route(candidate, dataset, context)
    if observed_route != task["route"]:
        raise ValueError("fitting route differs from frozen pre-fit decision")
    config = PrefitFittingConfig.model_validate(plan["config"])
    settings = (
        config.profiled_fit_config
        if task["route"]["selected_backend"] == "profiled_latent_basis_linear_ridge"
        else config.rollout_fit_config
    ).model_copy(update={"random_seed": task["seed"]})
    started = monotonic()
    cpu_started = process_time()
    try:
        model = compile_candidate(candidate, context)
        fit = fit_candidate(model, dataset.train, dataset.validation, settings)
    except Exception as exc:  # frozen proposer output remains untrusted input
        fit_payload: dict[str, Any] = {
            "fit_call_completed": False,
            "fit_success": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:8000],
            "fit": None,
        }
    else:
        fit_payload = {
            "fit_call_completed": True,
            "fit_success": bool(fit.success),
            "error_type": None,
            "error": fit.message,
            "fit": fit_result_payload(fit),
        }
    result = {
        "schema_version": "scientific-staged-prefit-fitting-task-1",
        "status": "complete",
        "plan_sha256": plan["plan_sha256"],
        "task_index": task_index,
        "task_id": task["task_id"],
        "source_task_id": task["source_task_id"],
        "benchmark_id": task["benchmark_id"],
        "tier": task["tier"],
        "seed": task["seed"],
        "candidate_file_sha256": task["candidate_file_sha256"],
        "candidate_identity": task["candidate_audit"]["candidate_identity"],
        "candidate_audit": task["candidate_audit"],
        "route": task["route"],
        "fit_config": settings.model_dump(mode="json"),
        "fit_wall_seconds": monotonic() - started,
        "fit_process_cpu_seconds": process_time() - cpu_started,
        **fit_payload,
        "candidate_regeneration_performed": False,
        "scientific_judge_called": False,
        "validation_used_for_parameter_fitting": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    _write_once_json(destination, result)
    return result


def summarize_campaign(output_root: Path) -> dict[str, Any]:
    """Summarize structural, fit, validation, and resource endpoints separately."""
    plan = _verified_plan(output_root)
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for task in plan["tasks"]:
        index = int(task["task_index"])
        path = output_root / "tasks" / f"task_{index:03d}.json"
        if not path.exists():
            rows.append(
                {
                    "task_index": index,
                    "task_id": task["task_id"],
                    "result_present": False,
                }
            )
            continue
        result = _read_object(path)
        if result.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError(f"fit task belongs to another plan: {path}")
        fit = result.get("fit") or {}
        diagnostics = fit.get("diagnostics", [])
        row = {
            "task_index": index,
            "task_id": task["task_id"],
            "benchmark_id": task["benchmark_id"],
            "seed": task["seed"],
            "result_present": True,
            "deterministic_prefit_pass": bool(
                task["candidate_audit"]["deterministic_prefit_certificate"]["passed"]
            ),
            "selected_backend": task["route"]["selected_backend"],
            "profiled_structural_compatible": task["route"][
                "profiled_structural_compatible"
            ],
            "exact_training_derivative_contract_available": task["route"][
                "exact_training_derivative_contract_available"
            ],
            "fit_call_completed": result["fit_call_completed"],
            "fit_success": result["fit_success"],
            "training_normalized_mse": fit.get("training_normalized_mse"),
            "validation_normalized_mse": fit.get("validation_normalized_mse"),
            "validation_per_target_normalized_mse": fit.get(
                "validation_per_target_normalized_mse"
            ),
            "function_evaluations": sum(
                int(item.get("function_evaluations", 0)) for item in diagnostics
            ),
            "integration_failures": sum(
                int(item.get("integration_failures", 0)) for item in diagnostics
            ),
            "fit_wall_seconds": result["fit_wall_seconds"],
            "fit_process_cpu_seconds": result["fit_process_cpu_seconds"],
            "parameter_count": task["candidate_audit"]["parameter_count"],
            "atomic_repair_activation_count": task["candidate_audit"][
                "atomic_repair_activation_count"
            ],
            "error_type": result["error_type"],
            "error": result["error"],
        }
        rows.append(row)
        artifacts.append(
            {
                "path": str(path.relative_to(output_root)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    terminal = sum(bool(row["result_present"]) for row in rows)
    successful = [row for row in rows if row.get("fit_success")]
    routes = Counter(
        row.get("selected_backend") for row in rows if row["result_present"]
    )
    summary = {
        "schema_version": "scientific-staged-prefit-fitting-summary-1",
        "status": "complete" if terminal == len(rows) else "incomplete",
        "plan_sha256": plan["plan_sha256"],
        "planned_tasks": len(rows),
        "terminal_results": terminal,
        "deterministic_prefit_pass_rate": _rate(
            sum(bool(row.get("deterministic_prefit_pass")) for row in rows), len(rows)
        ),
        "profiled_structural_compatibility_rate": _rate(
            sum(bool(row.get("profiled_structural_compatible")) for row in rows),
            len(rows),
        ),
        "exact_training_derivative_contract_rate": _rate(
            sum(
                bool(row.get("exact_training_derivative_contract_available"))
                for row in rows
            ),
            len(rows),
        ),
        "selected_backend_counts": dict(
            sorted((str(key), value) for key, value in routes.items())
        ),
        "fit_success_rate": _rate(len(successful), len(rows)),
        "median_training_normalized_mse_conditional_on_success": _median(
            [row["training_normalized_mse"] for row in successful]
        ),
        "median_validation_normalized_mse_conditional_on_success": _median(
            [row["validation_normalized_mse"] for row in successful]
        ),
        "total_function_evaluations": sum(
            int(row.get("function_evaluations", 0)) for row in rows
        ),
        "total_integration_failures": sum(
            int(row.get("integration_failures", 0)) for row in rows
        ),
        "total_fit_wall_seconds": sum(
            float(row.get("fit_wall_seconds", 0.0)) for row in rows
        ),
        "total_fit_process_cpu_seconds": sum(
            float(row.get("fit_process_cpu_seconds", 0.0)) for row in rows
        ),
        "rows": rows,
        "candidate_regeneration_performed": False,
        "parameter_fitting_performed": True,
        "scientific_judge_called": False,
        "validation_used_for_parameter_fitting": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_winner_defined": False,
    }
    summary_root = output_root / "summary"
    _write_once(
        summary_root / "task_artifact_ledger.jsonl",
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in artifacts).encode(),
    )
    summary["task_artifact_ledger_sha256"] = _sha256(
        summary_root / "task_artifact_ledger.jsonl"
    )
    _write_once_json(summary_root / "summary.json", summary)
    _write_once(summary_root / "summary.md", _summary_markdown(summary).encode())
    return summary


def _verified_plan(output_root: Path) -> dict[str, Any]:
    plan = _read_object(output_root / "plan.json")
    if content_hash(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ) != plan.get("plan_sha256"):
        raise ValueError("frozen fitting plan digest differs")
    if plan.get("runtime_source_sha256") != runtime_source_hash():
        raise ValueError("runtime source differs from frozen fitting plan")
    if plan.get("launcher_sha256") != launcher_hash():
        raise ValueError("launcher differs from frozen fitting plan")
    for relative, expected in plan["public_asset_ledger"].items():
        if _sha256(output_root / relative) != expected:
            raise ValueError(f"frozen public asset differs: {relative}")
    for task in plan["tasks"]:
        for path_key, digest_key in (
            ("candidate_path", "candidate_file_sha256"),
            ("source_path", "source_file_sha256"),
        ):
            relative = str(task[path_key])
            if _sha256(output_root / relative) != task[digest_key]:
                raise ValueError(f"frozen candidate source differs: {relative}")
    return plan


def _audit_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Frozen staged pre-fit candidate audit",
        "",
        "No scientific score is assigned. Equations, parameters, repairs, and "
        "fitting-route certificates are reported as separate mechanical facts.",
        "",
    ]
    for row in rows:
        audit = row["candidate_audit"]
        route = row["route"]
        lines.extend(
            [
                f"## {row['task_id']}",
                "",
                f"- Parameters: {audit['parameter_count']}",
                f"- Latent states: {', '.join(audit['latent_state_names']) or 'none'}",
                f"- Atomic repairs: {audit['atomic_repair_activation_count']}",
                "- Profiled structural compatibility: "
                f"{str(route['profiled_structural_compatible']).lower()}",
                f"- Selected backend: `{route['selected_backend']}`",
                f"- Route reason: {route['selection_reason']}",
                "",
                "```text",
            ]
        )
        lines.extend(f"{item['lhs']} = {item['rhs']}" for item in audit["equations"])
        lines.extend(["```", ""])
    return "\n".join(lines)


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Frozen staged pre-fit candidate fitting",
        "",
        "The six exact passed candidates were fit once under a route frozen "
        "before fitting. Validation was evaluated after train-only parameter "
        "fitting. No candidate was regenerated and no judge, test data, or "
        "private reference was used.",
        "",
        "| Benchmark | Seed | Route | Fit | Train NMSE | Validation NMSE "
        "| Wall seconds |",
        "|:---|---:|:---|:---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        if not row["result_present"]:
            lines.append(
                f"| missing | {row['task_index']} | N/A | no | N/A | N/A | N/A |"
            )
            continue
        fit_status = "yes" if row["fit_success"] else "no"
        training = _number(row["training_normalized_mse"])
        validation = _number(row["validation_normalized_mse"])
        lines.append(
            f"| {row['benchmark_id']} | {row['seed']} | "
            f"{row['selected_backend']} | {fit_status} | {training} | "
            f"{validation} | {row['fit_wall_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Fit success: {summary['fit_success_rate']:.3f}",
            "Median validation NMSE conditional on success: "
            f"{_number(summary['median_validation_normalized_mse_conditional_on_success'])}",
            "",
            "Structural certificates, empirical fit, and resources remain "
            "separate endpoints. No automatic winner is defined.",
            "",
        ]
    )
    return "\n".join(lines)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _median(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(statistics.median(finite))


def _number(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.6g}"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once_json(path: Path, value: Any) -> None:
    _write_once(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"frozen/checkpoint artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    temporary.replace(path)


def _copy_once(source: Path, destination: Path) -> None:
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise ValueError(f"frozen public copy differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
