"""Matched initial construction with/without descriptive public training evidence."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import signal
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.baselines.raw_data_agent import fit_result_payload
from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.expressions import ValidationContext
from autoformalism.fitting import FitConfig, fit_candidate
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    atomic_json,
)
from autoformalism.rebuttal.fitter_diagnostic import (
    PUBLIC_FILES,
    _finite_payload,
    runtime_identity,
)
from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec
from autoformalism.rebuttal.repair_comparison import BudgetedRepairClient
from autoformalism.rebuttal.staged_topology_campaign import (
    public_validation_context,
    runtime_source_hash,
)
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.staged_topology import ModelingLimits, PublicScientificBrief
from autoformalism.search.causal_initialization import compile_initialization_result
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.search.staged_topology_runner import run_staged_topology
from autoformalism.search.training_evidence import (
    EvidenceSettings,
    TrainingEvidence,
    build_training_evidence,
    validate_training_evidence,
)
from autoformalism.staged_topology import build_scientific_brief, content_hash
from autoformalism.targets import PublicTargetContract

ARMS = ("brief_only", "training_evidence")


class ConstructionCampaignConfig(StrictSchema):
    """Freeze all scientific choices and shared numerical/provider budgets."""

    protocol: Literal["prefit-matched-construction-1"] = "prefit-matched-construction-1"
    platform: Literal["aces-h100x1"] = "aces-h100x1"
    serving_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_settings: StagedModelSettings
    served_context_tokens: int = Field(default=32768, ge=16384)
    public_cells: tuple[str, ...] = Field(min_length=1, max_length=2)
    seeds: tuple[int, ...] = (0, 1, 2)
    limits: ModelingLimits = ModelingLimits()
    evidence: EvidenceSettings = EvidenceSettings()
    initialization_policy: Literal["causal_training"] = "causal_training"
    generation_granularity: Literal["equation_batch_atomic_repair"] = (
        "equation_batch_atomic_repair"
    )
    function_repair_policy: Literal["certified_outer_gain"] = "certified_outer_gain"
    scientific_judge: Literal["off"] = "off"
    fit: FitConfig = FitConfig(
        number_of_starts=1,
        allow_derivative_regression=False,
        maximum_function_evaluations=240,
        maximum_wall_time_seconds=300,
        integration_method="Radau",
    )
    wall_seconds: int = Field(default=10800, ge=60)
    shutdown_margin_seconds: int = Field(default=300, ge=30)

    @model_validator(mode="after")
    def matched_bounded_policy(self):
        """Reject unsupported history, duplicate tasks and unbounded numerical work."""
        if len(set(self.public_cells)) != len(self.public_cells) or len(
            set(self.seeds)
        ) != len(self.seeds):
            raise ValueError("duplicate construction tasks")
        if not self.seeds or any(s < 0 for s in self.seeds):
            raise ValueError("seeds must be distinct and nonnegative")
        registry = BenchmarkRegistry()
        for cell in self.public_cells:
            spec = registry.get(cell)
            if spec.data_layout != "tidy_split_file" or spec.one_step_target_history:
                raise ValueError(
                    "construction pilot requires public free-rollout cells"
                )
        if (
            self.fit.maximum_wall_time_seconds is None
            or self.fit.allow_derivative_regression
            or self.fit.parameter_fit_strategy != "bounded_nonlinear"
            or self.fit.nonlinear_initializer != "none"
        ):
            raise ValueError(
                "pilot pins bounded general rollout without derivative regression"
            )
        if (
            not self.model_settings.timeout_seconds
            < self.shutdown_margin_seconds
            < self.wall_seconds
        ):
            raise ValueError(
                "require provider timeout < shutdown margin < worker deadline"
            )
        return self


def read(path: Path) -> dict:
    """Read one immutable JSON object."""
    return json.loads(path.read_text())


def launcher_hash() -> str:
    """Pin the CLI and shared serving launcher along with package source."""
    repo = Path(__file__).resolve().parents[3]
    names = (
        "scripts/prefit_construction_campaign.py",
        "scripts/hpc/run_staged_topology_server.sh",
        "scripts/hpc/run_prefit_construction_aces.sh",
        "scripts/hpc/submit_prefit_construction_aces.sh",
    )
    return content_hash(
        {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in names}
    )


def load_development(public_root: Path, cell: str):
    """Open only train/validation; construction context is registry-only."""
    return BenchmarkLoader().load_development(
        DataConfig(benchmark_id=cell, tier=cell.rsplit("_", 1)[-1], root=public_root)
    )


def _sealed_write(path: Path, payload: dict) -> None:
    """Bind even terminal failures to exact content and reject accidental rewrites."""
    result = {**payload, "artifact_sha256": content_hash(payload)}
    if path.exists() and read(path) != result:
        raise ValueError(f"frozen artifact differs: {path}")
    atomic_json(path, result)


def _sealed_read(path: Path) -> dict:
    result = read(path)
    if result.get("artifact_sha256") != content_hash(
        {k: v for k, v in result.items() if k != "artifact_sha256"}
    ):
        raise ValueError(f"artifact digest differs: {path}")
    return result


def freeze(config_path: Path, public_root: Path, output: Path) -> dict:
    """Copy four public assets per cell and freeze both arms before provider calls."""
    config = ConstructionCampaignConfig.model_validate_json(config_path.read_text())
    repo = Path(__file__).resolve().parents[3]
    cells = {}
    tasks = []
    for cell_index, cell in enumerate(config.public_cells):
        source = public_root / "phase_b_v1" / cell
        hashes = {}
        for name in PUBLIC_FILES:
            payload = (source / name).read_bytes()
            destination = output / "public/phase_b_v1" / cell / name
            if destination.exists() and destination.read_bytes() != payload:
                raise ValueError("frozen public asset differs; use a new output root")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        context = public_validation_context(cell)
        prompt = (source / "proposer_prompt.txt").read_text()
        target = PublicTargetContract.model_validate_json(
            (repo / "configs/target_eval/phase_b_v2/specs" / f"{cell}.json").read_text()
        )
        mechanism = MechanismEvaluationSpec.model_validate_json(
            (
                repo / "configs/mechanism_eval/phase_b_v1/specs" / f"{cell}.json"
            ).read_text()
        )
        brief = build_scientific_brief(
            prompt, context, target, mechanism, limits=config.limits
        )
        dataset = load_development(output / "public", cell)
        packet = build_training_evidence(dataset.train, context, config.evidence)
        cells[cell] = {
            "assets": hashes,
            "context": context.model_dump(mode="json"),
            "brief": brief.model_dump(mode="json"),
            "evidence": packet.model_dump(mode="json"),
            "target_contract": target.model_dump(mode="json"),
            "mechanism_spec": mechanism.model_dump(mode="json"),
        }
        for seed in config.seeds:
            order = ARMS if (cell_index + seed) % 2 == 0 else reversed(ARMS)
            for arm in order:
                tasks.append(
                    {
                        "task_id": f"{cell}_seed{seed}_{arm}",
                        "cell": cell,
                        "seed": seed,
                        "arm": arm,
                    }
                )
    plan = {
        "protocol": config.protocol,
        "config": config.model_dump(mode="json"),
        "cells": cells,
        "tasks": tasks,
        "runtime_source_sha256": runtime_source_hash(),
        "numerical_runtime": runtime_identity(),
        "launcher_sha256": launcher_hash(),
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_winner_defined": False,
        "repair_rounds": 0,
    }
    _sealed_write(output / "plan.json", plan)
    return _sealed_read(output / "plan.json")


def verify(root: Path) -> dict:
    """Reject changed source, dependencies, public assets or evidence on resume."""
    plan = _verify_frozen_inputs(root)
    if (
        plan["runtime_source_sha256"] != runtime_source_hash()
        or plan["numerical_runtime"] != runtime_identity()
    ):
        raise ValueError("runtime differs from frozen construction campaign")
    if plan["launcher_sha256"] != launcher_hash():
        raise ValueError("launcher differs from frozen construction campaign")
    return plan


def _verify_frozen_inputs(root: Path) -> dict:
    """Verify sealed inputs without authorizing execution by a different runtime."""
    plan = _sealed_read(root / "plan.json")
    config = ConstructionCampaignConfig.model_validate(plan["config"])
    if plan["protocol"] != config.protocol:
        raise ValueError("unsupported frozen construction protocol")
    for cell, payload in plan["cells"].items():
        if set(payload["assets"]) != set(PUBLIC_FILES):
            raise ValueError("frozen public asset ledger is incomplete")
        for name, digest in payload["assets"].items():
            if (
                name not in PUBLIC_FILES
                or hashlib.sha256(
                    (root / "public/phase_b_v1" / cell / name).read_bytes()
                ).hexdigest()
                != digest
            ):
                raise ValueError("frozen public asset differs")
        validate_training_evidence(
            TrainingEvidence.model_validate(payload["evidence"]),
            ValidationContext.model_validate(payload["context"]),
        )
    return plan


def _identity(plan: dict, task: dict) -> str:
    return content_hash([plan["artifact_sha256"], task])


def _terminal(path: Path, identity: str) -> dict | None:
    if not path.exists():
        return None
    result = _sealed_read(path)
    if result["identity"] != identity:
        raise ValueError("task result belongs to another frozen construction")
    return result


def _cost(records: list[dict]) -> dict:
    records = list({r["request_hash"]: r for r in records}.values())
    return {
        "physical_requests": len(records),
        "budget_charge": sum(r.get("budget_charge", 0) for r in records),
        "observed_tokens": sum(r.get("observed_total_tokens") or 0 for r in records),
        "requests_with_unknown_usage": sum(
            r.get("observed_total_tokens") is None for r in records
        ),
        "provider_seconds": sum(r.get("latency_seconds", 0) for r in records),
        "statuses": dict(Counter(r["status"] for r in records)),
    }


def construct_task(
    root: Path, plan: dict, task: dict, client: BudgetedRepairClient
) -> dict:
    """Run the same staged controller in both arms with one shared call budget."""
    directory = root / "results" / task["task_id"]
    identity = _identity(plan, task)
    path = directory / "construction.json"
    _cache_records(directory / "calls", identity)
    prior = _terminal(path, identity)
    if prior is not None:
        return prior
    config = ConstructionCampaignConfig.model_validate(plan["config"])
    if (
        client.settings != config.model_settings
        or client.seed != task["seed"]
        or client.namespace != identity
        or client.directory.resolve() != (directory / "calls").resolve()
    ):
        raise ValueError("client differs from frozen construction settings")
    cell = plan["cells"][task["cell"]]
    brief = PublicScientificBrief.model_validate(cell["brief"])
    context = ValidationContext.model_validate(cell["context"])
    packet = (
        TrainingEvidence.model_validate(cell["evidence"])
        if task["arm"] == "training_evidence"
        else None
    )
    topology = run_staged_topology(
        brief,
        context,
        client,
        directory / "topology",
        hybrid_variable_construction=True,
        audit_public_polarity_policy=True,
        proposer_owns_unfixed_signs=True,
        training_evidence=packet,
    )
    functions = None
    if topology["complete_topology"] and topology["public_structure_checks_passed"]:
        functions = run_staged_functions(
            brief,
            context,
            topology,
            client,
            directory / "functions",
            generation_granularity=config.generation_granularity,
            function_repair_policy=config.function_repair_policy,
            initialization_policy=config.initialization_policy,
            training_evidence=packet,
        )
    complete = bool(functions and functions["complete_model"])
    record = {
        "identity": identity,
        "status": "complete" if complete else "construction_failed",
        "topology": topology,
        "functions": functions,
        "cost": _cost(client.records),
        "request_hashes": sorted(r["request_hash"] for r in client.records),
        "evidence_sha256": packet.packet_sha256 if packet else None,
        "validation_observations_sent": False,
        "test_data_opened": False,
    }
    _sealed_write(path, record)
    return _sealed_read(path)


def fit_task(root: Path, plan: dict, task: dict) -> dict | None:
    """Fit the canonical initializer once; interrupted optimizer budgets stay spent."""
    directory = root / "results" / task["task_id"]
    identity = _identity(plan, task)
    path = directory / "fit.json"
    prior = _terminal(path, identity)
    if prior is not None:
        return prior
    construction = _terminal(directory / "construction.json", identity)
    if construction is None:
        return None
    record = {
        "identity": identity,
        "status": "not_run",
        "reason": "construction_failed",
        "fit": None,
    }
    if construction["status"] == "complete":
        marker = directory / "fit_started.json"
        if marker.exists():
            if _sealed_read(marker)["identity"] != identity:
                raise ValueError("fit marker identity differs")
            record.update(
                status="interrupted", reason="consumed fit budget; no automatic refit"
            )
        else:
            config = ConstructionCampaignConfig.model_validate(plan["config"])
            artifact = construction["functions"]["initialization"]
            context = ValidationContext.model_validate(
                plan["cells"][task["cell"]]["context"]
            )
            model = compile_initialization_result(
                CandidateModel.model_validate(artifact["base_candidate"]),
                context,
                artifact,
            )
            dataset = load_development(root / "public", task["cell"])
            settings = config.fit.model_copy(
                update={"random_seed": config.fit.random_seed + task["seed"]}
            )
            _sealed_write(marker, {"identity": identity})
            started = monotonic()
            try:
                fitted = fit_candidate(
                    model,
                    dataset.train,
                    dataset.validation,
                    settings,
                    initial_global_parameters=artifact["guesses"],
                )
                payload = _finite_payload(fit_result_payload(fitted))
                training_initials = fitted.training_trajectory_initial_conditions
                validation_initials = fitted.validation_trajectory_initial_conditions
                record.update(
                    status="evaluated",
                    reason=None,
                    fit=payload,
                    validation_initials_fitted=False,
                    training_initial_overrides={
                        k: dict(v) for k, v in training_initials.items()
                    },
                    validation_initial_overrides={
                        k: dict(v) for k, v in validation_initials.items()
                    },
                )
            except Exception as exc:
                record.update(
                    status="fit_failed", reason=f"{type(exc).__name__}: {exc}"
                )
            record["fit_seconds"] = monotonic() - started
    _sealed_write(path, record)
    return _sealed_read(path)


def _cache_records(directory: Path, namespace: str) -> list[dict]:
    """Account for partial work without issuing a request or forgetting reservations."""
    records = []
    for path in sorted(directory.glob("*.json")):
        record = read(path)
        if (
            record["request_hash"] != path.stem
            or content_hash(record["request"]) != path.stem
            or record["request"]["namespace"] != namespace
        ):
            raise ValueError("provider request provenance differs")
        records.append(record)
    expected = set()

    def collect(value):
        if isinstance(value, dict):
            if "request_hash" in value:
                expected.add(value["request_hash"])
            expected.update(value.get("request_hashes", []))
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    for name in (
        "construction.json",
        "topology/progress.json",
        "functions/progress.json",
        "functions/initialization/state.json",
    ):
        path = directory.parent / name
        if path.exists():
            collect(read(path))
    if not expected <= {r["request_hash"] for r in records}:
        raise ValueError("provider cache is missing a recorded request")
    return records


def _validated_client(config, task, directory, namespace, base_url, can_start):
    _cache_records(directory, namespace)
    client = BudgetedRepairClient(
        settings=config.model_settings,
        seed=task["seed"],
        directory=directory,
        namespace=namespace,
        base_url=base_url,
        can_start=can_start,
    )
    for record in client.records:
        key = record["request_hash"]
        if (
            content_hash(record["request"]) != key
            or not (directory / f"{key}.json").is_file()
        ):
            raise ValueError("provider request provenance differs")
    return client


def run(
    root: Path,
    stage: Literal["construct", "fit"],
    base_url: str = "http://unused",
    *,
    arm: str | None = None,
    wall_seconds: float | None = None,
) -> dict:
    """Drain independent tasks with per-task locks and allocation-aware deferral."""
    if stage not in {"construct", "fit"} or arm not in {None, *ARMS}:
        raise ValueError("unknown stage or arm")
    plan = verify(root)
    config = ConstructionCampaignConfig.model_validate(plan["config"])
    deadline = monotonic() + (
        wall_seconds if wall_seconds is not None else config.wall_seconds
    )
    stopped = False

    def drain(signum, frame):
        nonlocal stopped
        stopped = True

    handlers = {s: signal.signal(s, drain) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for task in plan["tasks"]:
            if arm is not None and task["arm"] != arm:
                continue
            if stopped or monotonic() >= deadline - config.shutdown_margin_seconds:
                break
            directory = root / "results" / task["task_id"]
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "worker.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if stage == "fit":
                    if (
                        monotonic() + config.fit.maximum_wall_time_seconds
                        >= deadline - config.shutdown_margin_seconds
                    ):
                        break
                    fit_task(root, plan, task)
                else:
                    client = _validated_client(
                        config,
                        task,
                        directory / "calls",
                        _identity(plan, task),
                        base_url,
                        lambda: not stopped
                        and monotonic() < deadline - config.shutdown_margin_seconds,
                    )
                    try:
                        construct_task(root, plan, task, client)
                    except DeferredCall:
                        break
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return summarize(root, plan=plan)


def summarize(root: Path, *, plan: dict | None = None) -> dict:
    """Retain failures and paired denominators; never pick a scientific winner."""
    plan = plan or verify(root)
    result = _build_summary(root, plan)
    atomic_json(root / "summary.json", result)
    return result


def report(root: Path) -> dict:
    """Read an older frozen run without writing artifacts or enabling its resume."""
    plan = _verify_frozen_inputs(root)
    result = _build_summary(root, plan)
    keys = ("runtime_source_sha256", "numerical_runtime", "launcher_sha256")
    frozen = {key: plan[key] for key in keys}
    reporter = {
        "runtime_source_sha256": runtime_source_hash(),
        "numerical_runtime": runtime_identity(),
        "launcher_sha256": launcher_hash(),
    }
    result["reporting"] = {
        "mode": "read_only",
        "frozen_execution": frozen,
        "reporter": reporter,
        "execution_runtime_matches": frozen == reporter,
        "execution_authorized": False,
    }
    return result


def _build_summary(root: Path, plan: dict) -> dict:
    """Collect verified terminal records and cached costs without mutating the run."""
    rows = []
    for task in plan["tasks"]:
        directory = root / "results" / task["task_id"]
        identity = _identity(plan, task)
        construction = _terminal(directory / "construction.json", identity)
        fitted = _terminal(directory / "fit.json", identity)
        numerical = (fitted or {}).get("fit") or {}
        selected_diagnostic = next(
            (
                d
                for d in numerical.get("diagnostics", [])
                if d.get("start_index") == numerical.get("best_start_index")
            ),
            {},
        )
        functions = (construction or {}).get("functions") or {}
        initialization = functions.get("initialization") or {}
        candidate = functions.get("candidate") or {}
        topology = (construction or {}).get("topology") or {}
        cost = _cost(_cache_records(directory / "calls", identity))
        score = numerical.get("validation_normalized_mse")
        if score is not None and not math.isfinite(score):
            score = None
        valid = score is not None and not numerical.get(
            "validation_failed_trajectories", ["unavailable"]
        )
        rows.append(
            {
                **task,
                "construction_status": (construction or {}).get("status", "pending"),
                "construction_error": functions.get("error") or topology.get("error"),
                "fit_status": (fitted or {}).get("status", "pending"),
                "fit_reason": (fitted or {}).get("reason"),
                "fit_success": numerical.get("success"),
                "optimizer_success": selected_diagnostic.get("success"),
                "optimizer_status": selected_diagnostic.get("status"),
                "optimizer_message": selected_diagnostic.get("message"),
                "first_training_nmse": numerical.get("training_normalized_mse"),
                "training_failed_trajectories": numerical.get(
                    "training_failed_trajectories"
                ),
                "validation_failed_trajectories": numerical.get(
                    "validation_failed_trajectories"
                ),
                "first_validation_nmse": score if valid else None,
                "best_validation_nmse": score if valid else None,
                "best_is_first_only": True,
                "fit_seconds": (fitted or {}).get("fit_seconds"),
                "state_count": len(candidate.get("states", [])) if candidate else None,
                "equation_term_count": sum(
                    len(e["terms"]) for e in topology.get("equations", [])
                )
                if topology
                else None,
                "candidate_sha256": content_hash(candidate) if candidate else None,
                "topology_sha256": content_hash(topology.get("topology"))
                if topology.get("topology")
                else None,
                "initialization_modes": {
                    k: v["initial"]["mode"]
                    for k, v in (initialization.get("plan") or {})
                    .get("rules", {})
                    .items()
                },
                "cost": cost,
            }
        )
    arms = {
        arm: {
            "expected": sum(r["arm"] == arm for r in rows),
            "constructed": sum(
                r["arm"] == arm and r["construction_status"] == "complete" for r in rows
            ),
            "finite_validation": sum(
                r["arm"] == arm and r["first_validation_nmse"] is not None for r in rows
            ),
            "physical_requests": sum(
                r["cost"]["physical_requests"] for r in rows if r["arm"] == arm
            ),
            "observed_tokens": sum(
                r["cost"]["observed_tokens"] for r in rows if r["arm"] == arm
            ),
            "budget_charge": sum(
                r["cost"]["budget_charge"] for r in rows if r["arm"] == arm
            ),
            "provider_seconds": sum(
                r["cost"]["provider_seconds"] for r in rows if r["arm"] == arm
            ),
            "fit_seconds": sum(r["fit_seconds"] or 0 for r in rows if r["arm"] == arm),
        }
        for arm in ARMS
    }
    pairs = []
    for cell, seed in sorted({(r["cell"], r["seed"]) for r in rows}):
        paired = {r["arm"]: r for r in rows if r["cell"] == cell and r["seed"] == seed}
        a, b = (paired[arm] for arm in ARMS)
        scores = (a["first_validation_nmse"], b["first_validation_nmse"])
        pairs.append(
            {
                "cell": cell,
                "seed": seed,
                "both_finite_validation": all(s is not None for s in scores),
                "validation_nmse_evidence_minus_brief": scores[1] - scores[0]
                if all(s is not None for s in scores)
                else None,
                "topology_differs": a["topology_sha256"] != b["topology_sha256"]
                if a["topology_sha256"] and b["topology_sha256"]
                else None,
                "initialization_modes_differ": a["initialization_modes"]
                != b["initialization_modes"]
                if a["candidate_sha256"] and b["candidate_sha256"]
                else None,
            }
        )
    result = {
        "protocol": plan["protocol"],
        "plan_sha256": plan["artifact_sha256"],
        "status": "complete"
        if all(r["fit_status"] != "pending" for r in rows)
        else "partial",
        "construction_complete": all(
            r["construction_status"] != "pending" for r in rows
        ),
        "arms": arms,
        "rows": rows,
        "pairs": pairs,
        "automatic_winner_defined": False,
        "repair_rounds": 0,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    return result
