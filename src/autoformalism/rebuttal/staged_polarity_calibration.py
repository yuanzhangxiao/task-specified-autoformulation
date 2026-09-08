"""Cached topology-stage calibration of explicit interaction polarities."""

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
    visible_response,
)
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    ModelingLimits,
    PublicScientificBrief,
    PublicVariable,
    ScientificRequirement,
    ScientificVariable,
    equation_reply_model,
)
from autoformalism.search.staged_topology_prompts import (
    render_equation_topology_system_prompt,
    render_equation_topology_user_prompt,
)
from autoformalism.staged_topology import (
    content_hash,
    lower_topology,
    validate_equation,
)


class PolarityCalibrationGates(StrictSchema):
    """Predeclared exact gates for an intentionally explicit diagnostic."""

    minimum_response_success: float = Field(ge=0, le=1)
    minimum_exact_source_coverage: float = Field(ge=0, le=1)
    minimum_polarity_accuracy: float = Field(ge=0, le=1)
    minimum_fixed_polarity_accuracy: float = Field(ge=0, le=1)
    minimum_unrestricted_accuracy: float = Field(ge=0, le=1)


class PolarityCalibrationConfig(StrictSchema):
    """One warm-server, no-fit topology-polarity calibration."""

    protocol: Literal["scientific-staged-polarity-calibration-1"]
    purpose: str = Field(min_length=1)
    platform: Literal["aces-h100x1"]
    serving_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_settings: StagedModelSettings
    served_context_tokens: int = Field(ge=16384)
    seeds: tuple[int, ...] = Field(min_length=1, max_length=8)
    gates: PolarityCalibrationGates
    wall_seconds: int = Field(ge=60)
    shutdown_margin_seconds: int = Field(ge=30)

    @model_validator(mode="after")
    def bounded_unique_tasks(self) -> PolarityCalibrationConfig:
        """Reject duplicate seeds and an impossible shutdown schedule."""
        if len(self.seeds) != len(set(self.seeds)) or any(
            seed < 0 for seed in self.seeds
        ):
            raise ValueError("seeds must be unique and nonnegative")
        if (
            not self.model_settings.timeout_seconds
            < self.shutdown_margin_seconds
            < self.wall_seconds
        ):
            raise ValueError("require timeout < shutdown margin < worker wall time")
        return self


def polarity_calibration_launcher_hash() -> str:
    """Bind every executable launcher used by the frozen ACES campaign."""
    repository = Path(__file__).resolve().parents[3]
    paths = (
        "scripts/staged_polarity_calibration.py",
        "scripts/hpc/run_staged_topology_server.sh",
        "scripts/hpc/staged_polarity_calibration_aces.slurm",
        "scripts/hpc/submit_staged_polarity_calibration_aces.sh",
    )
    return content_hash(
        {
            path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
            for path in paths
        }
    )


def polarity_calibration_source() -> dict[str, Any]:
    """Return explicit public evidence and separately held evaluator labels."""
    brief = PublicScientificBrief(
        scientific_context=(
            "Construct target x as a continuous-time response. External input u "
            "has an explicitly activating contribution to x. State x also has a "
            "stabilizing self-relaxation contribution. Supplied context q contributes "
            "to x, but the public information does not specify whether that effect "
            "is activating or inhibiting. Include an unknown constant baseline whose "
            "sign is likewise unspecified. These are four distinct contributions."
        ),
        public_variables=(
            PublicVariable(name="x", data_role="target"),
            PublicVariable(name="u", data_role="external_input"),
            PublicVariable(name="q", data_role="auxiliary"),
        ),
        requirements=(
            ScientificRequirement(
                id="activating_input_drive",
                public_requirement=(
                    "one explicitly activating contribution from u to x"
                ),
                targets=("x",),
                drivers=("u",),
            ),
            ScientificRequirement(
                id="stabilizing_self_relaxation",
                public_requirement=(
                    "one stabilizing self-relaxation contribution from x to x"
                ),
                targets=("x",),
                drivers=("x",),
            ),
            ScientificRequirement(
                id="direction_unspecified_context",
                public_requirement=(
                    "one contribution from q to x whose direction is unspecified"
                ),
                targets=("x",),
                drivers=("q",),
            ),
            ScientificRequirement(
                id="signed_baseline",
                public_requirement=(
                    "one source-free constant baseline contribution with "
                    "unspecified sign"
                ),
                targets=("x",),
                drivers=(),
            ),
        ),
        limits=ModelingLimits(
            generated_variables=1, terms_per_equation=4, total_terms=4
        ),
    )
    context = ValidationContext(
        targets=("x",), auxiliaries=("q",), external_inputs=("u",)
    )
    inventory = (
        ScientificVariable(
            name="x", definition="differential", scientific_role="target response"
        ),
        ScientificVariable(
            name="u", definition="supplied", scientific_role="external input"
        ),
        ScientificVariable(
            name="q", definition="supplied", scientific_role="supplied context"
        ),
    )
    obligations = (
        {"sources": ["u"], "expected_outer_weight_sign": "positive"},
        {"sources": ["x"], "expected_outer_weight_sign": "negative"},
        {"sources": ["q"], "expected_outer_weight_sign": "unrestricted"},
        {"sources": [], "expected_outer_weight_sign": "unrestricted"},
    )
    return {
        "brief": brief.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
        "inventory": [item.model_dump(mode="json") for item in inventory],
        "selected_lhs": {"name": "x", "definition": "differential"},
        "relevant_agenda": {
            "purpose": "define x using the four displayed public obligations",
            "requirements": [item.id for item in brief.requirements],
            "targets": ["x"],
        },
        "expected_obligations": list(obligations),
    }


def freeze_polarity_calibration(config_path: Path, output: Path) -> dict[str, Any]:
    """Freeze public evidence, held evaluator labels, settings, and seeds."""
    config = PolarityCalibrationConfig.model_validate_json(config_path.read_text())
    source = polarity_calibration_source()
    tasks = [
        {"task_id": f"polarity_seed{seed}", "seed": seed, "source": source}
        for seed in config.seeds
    ]
    plan = {
        "config": config.model_dump(mode="json"),
        "runtime_source_sha256": runtime_source_hash(),
        "launcher_sha256": polarity_calibration_launcher_hash(),
        "tasks": tasks,
        "expected_labels_disclosed_to_provider": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_winner_defined": False,
    }
    plan["plan_sha256"] = content_hash(plan)
    if output.exists():
        if json.loads(output.read_text()) != plan:
            raise ValueError("existing frozen polarity calibration differs")
    else:
        atomic_json(output, plan)
    return plan


def _provider_user_payload(source: dict[str, Any], diagnostic: str | None) -> str:
    """Render only public evidence; evaluator labels remain outside the request."""
    inventory = source["inventory"]
    allowed = [item["name"] for item in inventory if item["definition"] != "unused"]
    return render_equation_topology_user_prompt(
        public_brief_json=json.dumps(source["brief"], sort_keys=True),
        agenda_json=json.dumps(source["relevant_agenda"], sort_keys=True),
        inventory_json=json.dumps(inventory, sort_keys=True),
        selected_lhs_json=json.dumps(source["selected_lhs"], sort_keys=True),
        equation_sketch_json="[]",
        allowed_sources_json=json.dumps(allowed),
        diagnostics_json=diagnostic,
    )


def _usage(client: StagedTopologyClient) -> dict[str, Any]:
    """Summarize provider resources without treating bytes as observed tokens."""
    measured = [
        item["observed_total_tokens"]
        for item in client.records
        if isinstance(item.get("observed_total_tokens"), int)
    ]
    return {
        "physical_requests": len(client.records),
        "observed_total_tokens": sum(measured),
        "unmeasured_requests": len(client.records) - len(measured),
        "provider_seconds": sum(
            float(item.get("latency_seconds", 0.0)) for item in client.records
        ),
    }


def _source_key(sources: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(sources))


def audit_polarity_reply(
    terms: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score exact source-set coverage and outer polarity without role matching."""
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for term in terms:
        grouped.setdefault(_source_key(term["sources"]), []).append(term)
    expected_by_key = {_source_key(item["sources"]): item for item in expected}
    rows = []
    for key, obligation in expected_by_key.items():
        matches = grouped.get(key, [])
        observed = (
            matches[0]["outer_weight_sign"] if len(matches) == 1 else None
        )
        anticipated = obligation["expected_outer_weight_sign"]
        rows.append(
            {
                "sources": list(key),
                "expected_outer_weight_sign": anticipated,
                "observed_outer_weight_sign": observed,
                "match_count": len(matches),
                "correct": len(matches) == 1 and observed == anticipated,
            }
        )
    extra = sorted(
        [list(key) for key in grouped if key not in expected_by_key], key=str
    )
    duplicate = sorted(
        [list(key) for key, values in grouped.items() if len(values) > 1], key=str
    )
    fixed = [
        row for row in rows if row["expected_outer_weight_sign"] != "unrestricted"
    ]
    unrestricted = [
        row for row in rows if row["expected_outer_weight_sign"] == "unrestricted"
    ]
    exact_coverage = not extra and not duplicate and all(
        row["match_count"] == 1 for row in rows
    )
    return {
        "obligation_count": len(rows),
        "correct_polarity_count": sum(bool(row["correct"]) for row in rows),
        "fixed_obligation_count": len(fixed),
        "correct_fixed_polarity_count": sum(bool(row["correct"]) for row in fixed),
        "unrestricted_obligation_count": len(unrestricted),
        "correct_unrestricted_count": sum(
            bool(row["correct"]) for row in unrestricted
        ),
        "exact_source_coverage": exact_coverage,
        "extra_source_sets": extra,
        "duplicate_source_sets": duplicate,
        "rows": rows,
        "calibration_pass": exact_coverage and all(row["correct"] for row in rows),
    }


def _run_task(
    task: dict[str, Any],
    config: PolarityCalibrationConfig,
    client: StagedTopologyClient,
    output: Path,
) -> dict[str, Any]:
    """Request one topology equation with bounded contract-only retries."""
    source = task["source"]
    brief = PublicScientificBrief.model_validate(source["brief"])
    context = ValidationContext.model_validate(source["context"])
    inventory = tuple(
        ScientificVariable.model_validate(item) for item in source["inventory"]
    )
    allowed = tuple(item.name for item in inventory if item.definition != "unused")
    response_model = equation_reply_model(
        allowed, maximum_terms=brief.limits.terms_per_equation
    )
    events: list[dict[str, Any]] = []
    diagnostic: str | None = None
    for attempt in range(config.model_settings.attempts_per_step):
        rejected: object = None
        record = client.call(
            system=render_equation_topology_system_prompt(),
            user=_provider_user_payload(source, diagnostic),
            response_model=response_model,
            step="equation_x",
            attempt=attempt,
        )
        try:
            rejected = visible_response(record)
            reply = response_model.model_validate(rejected)
            if reply.inventory_revision is not None:
                raise ValueError("calibration inventory is sufficient; define x")
            definition = EquationDefinition(
                name="x", definition="differential", terms=reply.terms
            )
            validate_equation(inventory, (), definition, brief.limits)
            topology, _ = lower_topology(brief, inventory, (definition,), context)
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)[:6000]
            diagnostic = json.dumps(
                {"rejected_response": rejected, "error": error}, sort_keys=True
            )
            events.append({"attempt": attempt, "accepted": False, "error": error})
            atomic_json(output / "progress.json", {"events": events})
            continue
        serialized_terms = [item.model_dump(mode="json") for item in reply.terms]
        audit = audit_polarity_reply(
            serialized_terms, source["expected_obligations"]
        )
        events.append({"attempt": attempt, "accepted": True, "error": None})
        return {
            "status": "complete",
            "response_success": True,
            "accepted_terms": serialized_terms,
            "topology": topology.model_dump(mode="json"),
            "audit": audit,
            "events": events,
            "error": None,
            **_usage(client),
        }
    return {
        "status": "failed",
        "response_success": False,
        "accepted_terms": [],
        "topology": None,
        "audit": None,
        "events": events,
        "error": "bounded topology-response repair exhausted",
        **_usage(client),
    }


def run_polarity_calibration(
    plan_path: Path,
    output: Path,
    base_url: str,
    *,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Run and checkpoint all frozen seeds on one warm local provider."""
    plan = json.loads(plan_path.read_text())
    expected = content_hash(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if expected != plan.get("plan_sha256"):
        raise ValueError("frozen polarity calibration digest mismatch")
    if runtime_source_hash() != plan["runtime_source_sha256"]:
        raise ValueError("runtime source differs from frozen polarity calibration")
    if polarity_calibration_launcher_hash() != plan["launcher_sha256"]:
        raise ValueError("launcher differs from frozen polarity calibration")
    config = PolarityCalibrationConfig.model_validate(plan["config"])
    deadline = time.monotonic() + (wall_seconds or config.wall_seconds)
    stop = False

    def drain(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    previous = {
        sig: signal.signal(sig, drain) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    records: list[dict[str, Any]] = []
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
                        result = _run_task(task, config, client, root)
                    except DeferredCall:
                        break
                    atomic_json(root / "result.json", result)
                    record = {
                        "identity": identity,
                        "task_id": task["task_id"],
                        "seed": task["seed"],
                        "result": result,
                    }
                    atomic_json(terminal, record)
                records.append(record)
                atomic_json(output / "summary.json", summarize(plan, records))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    summary = summarize(plan, records)
    atomic_json(output / "summary.json", summary)
    return summary


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(plan: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate exact calibration gates without selecting a scientific model."""
    indexed = {item["task_id"]: item for item in records}
    rows = []
    for task in plan["tasks"]:
        record = indexed.get(task["task_id"])
        result = record["result"] if record else None
        audit = result.get("audit") if result else None
        rows.append(
            {
                "task_id": task["task_id"],
                "seed": task["seed"],
                "result_present": result is not None,
                "response_success": result.get("response_success", False)
                if result
                else False,
                "exact_source_coverage": audit.get("exact_source_coverage")
                if audit
                else None,
                "calibration_pass": audit.get("calibration_pass")
                if audit
                else None,
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
    terminal = [item for item in rows if item["result_present"]]
    response_count = sum(bool(item["response_success"]) for item in terminal)
    source_count = sum(bool(item["exact_source_coverage"]) for item in terminal)
    audits = [
        record["result"]["audit"]
        for record in records
        if record["result"].get("audit") is not None
    ]
    obligation_count = sum(item["obligation_count"] for item in audits)
    correct_count = sum(item["correct_polarity_count"] for item in audits)
    fixed_count = sum(item["fixed_obligation_count"] for item in audits)
    correct_fixed = sum(item["correct_fixed_polarity_count"] for item in audits)
    unrestricted_count = sum(
        item["unrestricted_obligation_count"] for item in audits
    )
    correct_unrestricted = sum(item["correct_unrestricted_count"] for item in audits)
    metrics = {
        "response_success": _rate(response_count, len(terminal)),
        "exact_source_coverage": _rate(source_count, len(terminal)),
        "polarity_accuracy": _rate(correct_count, obligation_count),
        "fixed_polarity_accuracy": _rate(correct_fixed, fixed_count),
        "unrestricted_accuracy": _rate(correct_unrestricted, unrestricted_count),
    }
    gates = PolarityCalibrationGates.model_validate(plan["config"]["gates"])
    checks = {
        "minimum_response_success": metrics["response_success"] is not None
        and metrics["response_success"] >= gates.minimum_response_success,
        "minimum_exact_source_coverage": metrics["exact_source_coverage"] is not None
        and metrics["exact_source_coverage"] >= gates.minimum_exact_source_coverage,
        "minimum_polarity_accuracy": metrics["polarity_accuracy"] is not None
        and metrics["polarity_accuracy"] >= gates.minimum_polarity_accuracy,
        "minimum_fixed_polarity_accuracy": metrics["fixed_polarity_accuracy"]
        is not None
        and metrics["fixed_polarity_accuracy"]
        >= gates.minimum_fixed_polarity_accuracy,
        "minimum_unrestricted_accuracy": metrics["unrestricted_accuracy"] is not None
        and metrics["unrestricted_accuracy"] >= gates.minimum_unrestricted_accuracy,
    }
    complete = len(terminal) == len(rows)
    return {
        "schema_version": "scientific-staged-polarity-calibration-summary-1",
        "status": "complete" if complete else "incomplete",
        "overall_result": "pass" if complete and all(checks.values()) else "fail",
        "plan_sha256": plan["plan_sha256"],
        "planned_tasks": len(rows),
        "terminal_results": len(terminal),
        **metrics,
        "checks": checks,
        "physical_requests": sum(item["physical_requests"] for item in terminal),
        "observed_total_tokens": sum(
            item["observed_total_tokens"] for item in terminal
        ),
        "provider_seconds": sum(item["provider_seconds"] for item in terminal),
        "rows": rows,
        "expected_labels_disclosed_to_provider": False,
        "automatic_winner_defined": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
