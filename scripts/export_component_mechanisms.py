#!/usr/bin/env python3
"""Freeze retained component-campaign models for the common mechanism assessment."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.mechanism_audit_sources import BUNDLE_PROTOCOL
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.review_deadline_pipeline import certificates
from autoformalism.schemas.public_fitting import PublicFitRequest
from autoformalism.staged_topology import content_hash

if __package__:
    from .summarize_functional_mechanisms import aggregate, markdown
else:
    from summarize_functional_mechanisms import aggregate, markdown


def treatment(task: dict) -> str:
    """Do not pool prompt variants or component ablations."""
    return (
        f"autoformalism:{task['arm']}:c{int(task['critic'])}"
        f"v{int(task['scientific_verifier'])}s{int(task['shared_processes'])}"
    )


def retained(source: Path, plan: dict, task: dict, round_index: int | None):
    """Use a declared round or the latest recorded incumbent, never rank models."""
    latest = None
    last_record = None
    observed = []
    limit = plan["config"]["rounds"] - 1 if round_index is None else round_index
    for index in range(limit + 1):
        path = source / "results" / task["task_id"] / f"round_{index:02d}/result.json"
        if not path.exists():
            continue
        value = sealed_read(path)
        if value["task"] != task or value["round"] != index:
            raise ValueError(f"task/round identity differs: {path}")
        if value.get("test_data_opened") is not False:
            raise ValueError("require development-only saved rounds")
        observed.append(index)
        last_record = value
        if (round_index is None and value.get("selected")) or index == round_index:
            latest = value
    return (latest or last_record) if round_index is None else latest, observed


def fitted_row(source: Path, plan: dict, task: dict, result: dict | None) -> dict:
    """Preserve exact lowered equations, initial maps, parameters and provenance."""
    cell = plan["cells"][task["cell"]]
    row = {
        "method": treatment(task),
        "benchmark_id": task["cell"],
        "tier": cell["target_contract"]["tier"],
        "repetition": task["seed"],
        "task": task,
        "semantics": "continuous_time",
        "status": "unavailable",
        "source_plan_sha256": plan["artifact_sha256"],
        "public_prompt_sha256": cell["assets"]["proposer_prompt.txt"],
        "source_round": result["round"] if result else None,
        "source_result_sha256": result["artifact_sha256"] if result else None,
        "source_terminal_status": None,
        "error": "no retained fitted model at the declared snapshot",
        "legacy_certificate": None,
        "legacy_graph_requirement_ids": [
            r["id"] for r in cell["mechanism_spec"]["required_mechanisms"]
        ],
    }
    selected = (result or {}).get("selected")
    if not selected:
        if (
            result
            and result["round"] == plan["config"]["rounds"] - 1
            and result["status"] in {"fit_failed", "construction_failed"}
        ):
            row["source_terminal_status"] = result["status"]
        return row
    origin = selected["origin_round"]
    if selected["origin_task"] != task["task_id"] or not 0 <= origin <= result["round"]:
        raise ValueError("retained model lineage differs")
    request = PublicFitRequest.model_validate(selected["request"])
    fit = selected["fit"]
    lowered, _, _ = public._lower(request)
    candidate = lowered.validated.candidate.model_dump(mode="json")
    context = lowered.validated.context.model_dump(mode="json")
    if (
        public.content_sha256(request.model_dump(mode="json")) != fit["request_sha256"]
        or public.content_sha256(candidate) != fit["lowered_candidate_sha256"]
        or selected["bundle"]["candidate"] != candidate
        or public.content_sha256(cell["training"]) != fit["training_content_sha256"]
        or public.content_sha256(cell["validation"]) != fit["validation_content_sha256"]
    ):
        raise ValueError("retained request/model/development identity differs")
    receipt_path = (
        source / "results" / task["task_id"] / f"round_{origin:02d}/fit/result.json"
    )
    receipt = json.loads(receipt_path.read_text())
    if (
        public.content_sha256(receipt) != selected["fit_result_sha256"]
        or receipt["result"] != fit
        or receipt["sha256"] != public.content_sha256(fit)
    ):
        raise ValueError("retained fit receipt differs")
    parameters = fit.get("parameters")
    if fit["status"] != "complete" or not isinstance(parameters, dict):
        raise ValueError("retained model has no complete fit")
    if set(parameters) != set(lowered.parameter_names) or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in parameters.values()
    ):
        raise ValueError("incomplete or nonfinite fitted parameters")
    # Post-hoc reporting is identical in verifier-on and verifier-off treatments.
    certificate = certificates(
        {"candidate": candidate, "initialization": {"context": context}},
        cell,
        {**task, "scientific_verifier": True},
    )
    row.update(
        status="ready",
        error=None,
        candidate=candidate,
        context=context,
        parameters=parameters,
        initials={},
        legacy_certificate=certificate,
        historical_certificate=selected.get("certificate"),
        selected_origin_round=origin,
        validation_nmse=fit["validation"].get("normalized_mse"),
        source_id=str(receipt_path),
        source_fit_sha256=selected["fit_result_sha256"],
    )
    return row


def legacy_scores(rows: list[dict]) -> dict:
    """Report graph obligations, target predicates and runtime validity separately."""
    metrics = {k: [] for k in ("graph_requirements", "target_predicates", "runtime")}
    statuses = {"satisfied": "pass", "failed": "fail", "ambiguous": "unresolved"}
    for row in rows:
        cert = row["legacy_certificate"]
        for name, output in metrics.items():
            if cert is None:
                count = (
                    len(row["legacy_graph_requirement_ids"])
                    if name == "graph_requirements"
                    else 1
                )
                checks = [{"status": "unresolved"} for _ in range(count)]
            elif name == "runtime":
                checks = [{"status": "pass" if cert["runtime_valid"] else "fail"}]
            else:
                entries = (
                    cert["mechanisms"]["mechanism_results"]
                    if name == "graph_requirements"
                    else cert["targets"]["predicates"]
                )
                checks = [{"status": statuses[e["status"]]} for e in entries]
            output.append(
                {
                    **{k: row[k] for k in ("method", "benchmark_id", "repetition")},
                    "mechanisms": checks,
                }
            )
    return {k: aggregate(v) for k, v in metrics.items()}


def export(source: Path, output: Path, round_index: int | None = None) -> dict:
    """Read saved development artifacts only; never finalize or advance a campaign."""
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("write the snapshot outside the source campaign")
    # A completed export freezes selection even if more campaign visits finish.
    if (output / "models.json").exists():
        saved = sealed_read(output / "models.json")
        if (
            saved["source_root"] != str(source.resolve())
            or saved["snapshot_round"] != round_index
        ):
            raise ValueError("existing snapshot has different inputs")
        plan = sealed_read(source / "plan.json")
        if saved["source_plan_sha256"] != plan["artifact_sha256"]:
            raise ValueError("source plan changed")
        write_legacy(saved, output)
        return saved
    plan = sealed_read(source / "plan.json")
    if plan["protocol"] != "final-component-campaign-1":
        raise ValueError("require final-component-campaign-1")
    if round_index is not None and not 0 <= round_index < plan["config"]["rounds"]:
        raise ValueError("round outside planned campaign")
    rows = []
    for task in plan["tasks"]:
        result, observed = retained(source, plan, task, round_index)
        row = fitted_row(source, plan, task, result)
        row["recorded_rounds"] = observed
        row["last_recorded_round"] = max(observed) if observed else None
        rows.append(row)
    if len({(r["method"], r["benchmark_id"], r["repetition"]) for r in rows}) != len(
        rows
    ):
        raise ValueError("duplicate planned lineage")
    output.mkdir(parents=True, exist_ok=True)
    bundle = sealed_write(
        output / "models.json",
        {
            "protocol": BUNDLE_PROTOCOL,
            "rows": rows,
            "source_root": str(source.resolve()),
            "source_plan_sha256": plan["artifact_sha256"],
            "snapshot_round": round_index,
            "selection": "latest_recorded_incumbent"
            if round_index is None
            else "fixed_round",
            "endpoint_scope": (
                "search incumbent before final pruning; unequal progress is retained"
            ),
            "exporter_sha256": content_hash(Path(__file__).read_text()),
            "test_data_opened": False,
            "llm_calls": 0,
            "optimizer_calls": 0,
            "solver_rollouts": 0,
            "model_changes": 0,
        },
    )
    write_legacy(bundle, output)
    return bundle


def write_legacy(bundle: dict, output: Path) -> None:
    """Regenerate reports from the frozen bundle after interrupted exports."""
    rows = bundle["rows"]
    scores = legacy_scores(rows)
    sealed_write(
        output / "legacy_summary.json",
        {
            "source_bundle_sha256": bundle["artifact_sha256"],
            "metrics": scores,
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "interpretation": (
                "post-hoc graph/target/runtime checks, not functional compliance"
            ),
        },
    )
    (output / "LEGACY.md").write_text(
        "\n\n".join(f"## {k}\n\n{markdown(v)}" for k, v in scores.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--latest-retained", action="store_true")
    choice.add_argument("--round", type=int)
    args = parser.parse_args()
    value = export(args.source, args.output, args.round)
    print(
        json.dumps(
            {
                "rows": len(value["rows"]),
                "selection": value["selection"],
                "status_counts": dict(Counter(r["status"] for r in value["rows"])),
                "bundle": str(args.output / "models.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
