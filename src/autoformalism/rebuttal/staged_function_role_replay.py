"""Offline replay of certified role repairs over frozen hybrid artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas.staged_functions import InteractionFunctionReply
from autoformalism.staged_functions import repair_certified_outer_gain_role

_DIAGNOSTIC_CODE = re.compile(r"(?:^|;\s*)([A-Z][A-Z0-9_]+):")


def replay_hybrid_outer_gain_repairs(
    source_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Project the v2 deterministic repair over v1 responses without LLM calls."""
    results_root = source_root / "results"
    summary_path = results_root / "summary.json"
    if not summary_path.is_file():
        summary_path = source_root / "summary.json"
        results_root = source_root
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing source summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete":
        raise ValueError("source hybrid campaign is not complete")
    for key in (
        "parameter_fitting_performed",
        "scientific_judge_called",
        "test_data_opened",
        "private_reference_opened",
    ):
        if summary.get(key) is not False:
            raise ValueError(f"source public-only boundary failed: {key}")

    rows: list[dict[str, Any]] = []
    role_only_failures = 0
    certified_role_repairs = 0
    projected_llm_repairs = 0
    saved_provider_attempts = 0
    terminal_paths = sorted(results_root.glob("*/terminal.json"))
    if len(terminal_paths) != summary.get("terminal_results"):
        raise ValueError("source terminal count differs from frozen summary")
    for terminal_path in terminal_paths:
        terminal = json.loads(terminal_path.read_text())
        result = terminal["result"]
        audits = result.get("batch_term_audits", [])
        accepted = result.get("accepted_functions", [])
        if len(audits) != len(accepted):
            raise ValueError(f"audit/function count mismatch: {terminal_path}")
        events = result.get("events", [])
        for audit, accepted_function in zip(audits, accepted, strict=True):
            error = audit.get("batch_error")
            if not error:
                continue
            codes = tuple(_DIAGNOSTIC_CODE.findall(error))
            role_only = codes == ("SIGNED_WEIGHT_WITH_TOPOLOGY_POLARITY",)
            role_only_failures += int(role_only)
            source_names = set(accepted_function["selected_term"]["sources"])
            provider_reply = InteractionFunctionReply.model_validate(
                audit["batch_function"]
            )
            repaired, repairs = repair_certified_outer_gain_role(
                provider_reply,
                source_names,
            )
            certified = role_only and bool(repairs)
            certified_role_repairs += int(certified)
            needs_llm_repair = not certified
            projected_llm_repairs += int(needs_llm_repair)
            atomic_step = f"atomic_repair_{audit['interaction_id']}"
            attempts = sum(event.get("step") == atomic_step for event in events)
            if certified:
                saved_provider_attempts += attempts
            rows.append(
                {
                    "task_id": terminal.get("task_id"),
                    "seed": terminal.get("seed"),
                    "interaction_id": audit["interaction_id"],
                    "lhs": audit["lhs"],
                    "diagnostic_codes": list(codes),
                    "role_only_failure": role_only,
                    "certified_deterministic_repair": certified,
                    "projected_llm_atomic_repair": needs_llm_repair,
                    "source_expression": provider_reply.expression,
                    "repaired_parameters": [
                        item.model_dump(mode="json") for item in repaired.parameters
                    ],
                    "repair_records": [
                        item.model_dump(mode="json") for item in repairs
                    ],
                    "source_atomic_provider_attempts": attempts,
                }
            )

    source_atomic_repairs = summary.get("atomic_repair_activation_count")
    if source_atomic_repairs != len(rows):
        raise ValueError("source atomic-repair count differs from audited failures")
    report = {
        "schema_version": "staged-function-outer-gain-replay-1",
        "status": (
            "pass"
            if role_only_failures == certified_role_repairs and role_only_failures > 0
            else "fail"
        ),
        "source_root": str(source_root),
        "source_summary_file_sha256": hashlib.sha256(
            summary_path.read_bytes()
        ).hexdigest(),
        "runtime_source_sha256": runtime_source_hash(),
        "new_llm_calls_made": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "source_atomic_repair_activation_count": source_atomic_repairs,
        "source_role_only_failure_count": role_only_failures,
        "certified_deterministic_role_repair_count": certified_role_repairs,
        "projected_llm_atomic_repair_count": projected_llm_repairs,
        "source_physical_requests": summary["physical_requests"],
        "projected_physical_requests": (
            summary["physical_requests"] - saved_provider_attempts
        ),
        "projected_saved_provider_attempts": saved_provider_attempts,
        "rows": rows,
    }
    atomic_json(output, report)
    return report
