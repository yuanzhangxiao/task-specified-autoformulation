"""Freeze, run and report a Phase-B LLM-SR campaign.

Mirrors the LLM-ODE campaign: the plan is sealed before any provider call, a
completed task resumes without spending one, and a completed search seals the
model beside its result so the frozen evaluator can adapt it.

LLM-SR learns one function per run, so a cell with several targets runs their
pipeline once per target, which is what their specification format requires.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from autoformalism.baselines.models import BaselineDevelopmentResult
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.baseline_validation import load_public
from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate
from autoformalism.rebuttal.llm_ode_campaign import (
    cell_arrays,
    observed_channels,
    public_prompt_text,
    public_task_specification,
)
from autoformalism.rebuttal.llm_sr_driver import build_searcher  # noqa: F401
from autoformalism.rebuttal.prefit_replay import (
    content_hash,
    sealed_read,
    sealed_write,
)
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.rebuttal.vendored_campaign import VendoredCampaignPlan

PROTOCOL = "phase-b-llm-sr-campaign-1"


class SearcherFactory(Protocol):
    """Runs upstream's search for one cell, injected so tests need no provider."""

    def __call__(self, **kwargs) -> dict: ...


def environment_identity() -> dict:
    """Bind resume to the code and endpoint kind, never to a per-job port."""
    return {
        "runtime_source_sha256": runtime_source_hash(),
        "provider": "vllm",
        "endpoint_kind": "job-local vllm endpoint",
    }


def prepare(config_path: Path, public_root: Path, root: Path) -> dict:
    """Freeze the campaign before any provider call."""
    plan = VendoredCampaignPlan.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )
    if plan.method != "llm_sr":
        raise ValueError(f"expected an llm_sr plan, not {plan.method!r}")
    public = public_root.expanduser().resolve()
    rows = []
    for cell in plan.cells:
        development, context, identity = load_public(
            public, cell.benchmark_id, cell.tier
        )
        channels = observed_channels(development.train)
        specification = (
            public_task_specification(
                public_prompt_text(public, cell.benchmark_id, cell.tier)
            )
            if plan.prompt_policy.supplies_public_task_specification
            else ""
        )
        for repetition in plan.repetitions:
            rows.append(
                {
                    "index": len(rows),
                    "benchmark_id": cell.benchmark_id,
                    "tier": cell.tier,
                    "repetition": repetition,
                    "public_identity": identity,
                    "channels": list(channels),
                    "searched_targets": list(context.targets),
                    "prompt": specification,
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    return sealed_write(
        root / "plan.json",
        {
            "protocol": PROTOCOL,
            "plan": plan.model_dump(mode="json"),
            "public_root": str(public),
            "environment": environment_identity(),
            "reporting_qualifications": list(plan.reporting_qualifications()),
            # One request per samples_per_prompt samples, per searched target.
            "maximum_logical_samples": sum(
                plan.budget.declared * len(row["searched_targets"]) for row in rows
            ),
            "rows": rows,
        },
    )


def run(root: Path, index: int, *, search: SearcherFactory | None = None) -> dict:
    """Resume one task; a completed result causes no call and no refitting."""
    sealed = sealed_read(root / "plan.json")
    if (
        sealed["protocol"] != PROTOCOL
        or sealed["environment"] != environment_identity()
    ):
        raise ValueError(
            "protocol or code changed since this plan was frozen. Run from the "
            f"checkout that froze it, or delete {root / 'plan.json'} and its "
            "results to re-freeze at the current code."
        )
    if not 0 <= index < len(sealed["rows"]):
        raise ValueError("task index out of range")
    row = sealed["rows"][index]
    directory = root / "results" / str(index)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        return sealed_read(result_path)

    development, context, identity = load_public(
        Path(sealed["public_root"]), row["benchmark_id"], row["tier"]
    )
    if identity != row["public_identity"]:
        raise ValueError("public development input drift")
    if search is None:  # pragma: no cover - requires the vendored checkout
        raise ValueError(
            "supply a searcher factory bound to the pinned LLM-SR checkout"
        )
    from autoformalism.rebuttal.llm_ode_driver import development_rollout_error

    train = cell_arrays(development.train)
    outcome = search(
        channels=tuple(row["channels"]),
        targets=tuple(context.targets),
        values=train.states,
        derivatives=train.derivatives,
        description=row.get("prompt", ""),
        directory=directory,
        development=(development.train, development.validation),
        context=context,
        score_rollout=development_rollout_error,
    )

    if outcome["status"] == "complete":
        equations = outcome["equations"]
        selection = BaselineDevelopmentResult(
            method="llm_sr",
            benchmark_id=row["benchmark_id"],
            tier=row["tier"],
            seed=row["repetition"],
            equations=equations,
            selected_hyperparameters={
                "llm_samples": int(sealed["plan"]["budget"]["declared"]),
                "num_islands": int(sealed["plan"].get("islands", 10)),
                "selection": "development_rollout_error",
            },
            selection_payload={
                "candidate": equation_candidate(
                    "llm_sr", equations, context
                ).model_dump(mode="json"),
                # Coefficients are refitted into the expressions, so none remain.
                "parameters": {},
            },
            training_normalized_mse=float(outcome["training_rollout_error"]),
            validation_normalized_mse=float(outcome["development_rollout_error"]),
            elapsed_wall_seconds=outcome.get("accounting", {}).get("search_seconds"),
        )
        sealed_write(
            directory / "native-selection.json",
            {
                "plan_sha256": sealed["artifact_sha256"],
                "selection": selection.model_dump(mode="json"),
            },
        )
    return sealed_write(
        result_path,
        {
            **{
                key: row[key]
                for key in ("index", "benchmark_id", "tier", "repetition")
            },
            "protocol": PROTOCOL,
            "plan_sha256": sealed["artifact_sha256"],
            "status": outcome["status"],
            "error": outcome.get("error"),
            "equations": outcome.get("equations"),
            "development_rollout_error": outcome.get("development_rollout_error"),
            "accounting": outcome.get("accounting", {}),
            "test_data_opened": False,
            "private_reference_opened": False,
            "selection_metric": "development_rollout_error",
        },
    )


def report(root: Path) -> dict:
    """Coverage before scores, and the same persisted summary D3 writes."""
    sealed = sealed_read(root / "plan.json")
    rows = []
    for task in sealed["rows"]:
        path = root / "results" / str(task["index"]) / "result.json"
        rows.append(
            sealed_read(path) if path.exists() else {**task, "status": "pending"}
        )
    counts: dict[str, int] = {}
    for item in rows:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    expected = len(sealed["rows"])
    complete = counts.get("complete", 0)
    value = {
        "protocol": PROTOCOL,
        "status": "complete" if complete == expected else "pending",
        "expected": expected,
        "terminal_success": complete,
        "not_started": counts.get("pending", 0),
        "terminal_scientific_failure": sum(
            count
            for status, count in counts.items()
            if status in {"inexpressible", "rollout_failed", "no_candidates"}
        ),
        "infrastructure_failure": counts.get("endpoint_unavailable", 0),
        "frozen_models": sum(
            1
            for task in sealed["rows"]
            if (
                root / "results" / str(task["index"]) / "native-selection.json"
            ).is_file()
        ),
        "counts": counts,
        "reporting_qualifications": sealed["reporting_qualifications"],
        "rows": rows,
    }
    value["artifact_sha256"] = content_hash(value)
    atomic_json(root / "summary.json", value)
    return value
