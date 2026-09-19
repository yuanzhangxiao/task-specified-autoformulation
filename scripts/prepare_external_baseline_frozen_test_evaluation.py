#!/usr/bin/env python3
"""Freeze external-baseline sources, verify them, and seal the execution record."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from autoformalism.rebuttal.external_baseline_freeze import (
    ExternalBaselineSource,
    load_external_baseline_plan,
    load_reused_freeze,
    public_development_identity,
    reconcile,
    require_executable_freeze,
    resolve_external_baseline_sources,
    unavailable_reused_rows,
    verify_recorded_hashes,
    verify_request_roster_association,
)
from autoformalism.rebuttal.final_evaluation_adapters import SourceAdapterRequest
from autoformalism.rebuttal.final_evaluation_pilot import validate_hidden_audit

ROSTER_NAME = "external_baseline_roster.jsonl"
REQUESTS_NAME = "source_adapter_requests.jsonl"
WITHHELD_NAME = "withheld_source_outcomes.jsonl"
MANIFEST_NAME = "external_baseline_freeze.json"
EXECUTION_NAME = "execution_record.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--symbolic-development-freeze", type=Path)
    parser.add_argument("--d3-campaign-root", type=Path)
    parser.add_argument("--hidden-audit", type=Path)
    parser.add_argument("--raw-agent-freeze-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--authorize-execution", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--seal-execution-record", action="store_true")
    parser.add_argument("--verify-execution-record", action="store_true")
    parser.add_argument("--chain-inputs-digest", type=str)
    parser.add_argument("--public-identity", type=str)
    parser.add_argument("--adapted-root", type=Path)
    parser.add_argument("--public-data-root", type=Path)
    parser.add_argument("--evaluator-commit", type=str)
    parser.add_argument("--maximum-trajectory-wall-time-seconds", type=float)
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    if args.verify:
        _verify_existing(output_root)
        return
    if args.seal_execution_record:
        _seal(parser, args, output_root)
        return
    if args.verify_execution_record:
        _verify_execution_record(parser, args, output_root)
        return
    _prepare(parser, args, output_root)


def _prepare(parser, args, output_root: Path) -> None:
    """Resolve every planned identity and write the frozen roster."""
    plan = load_external_baseline_plan(args.config)
    if args.authorize_execution:
        require_executable_freeze(plan)
    for name in ("symbolic_development_freeze", "d3_campaign_root", "hidden_audit"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required to prepare")

    audit_sha256 = validate_hidden_audit(args.hidden_audit, plan.hidden_contract_audit)
    roots = {
        "sindy": args.symbolic_development_freeze,
        "pysr": args.symbolic_development_freeze,
        "d3_native_no_tools": args.d3_campaign_root,
    }
    requests, sources = resolve_external_baseline_sources(plan, roots=roots)

    reused_manifest: dict[str, object] | None = None
    for method in plan.methods:
        if method.source_layout != "reused_external_freeze":
            continue
        if args.raw_agent_freeze_manifest is None:
            if args.authorize_execution:
                parser.error(
                    f"--raw-agent-freeze-manifest is required to execute "
                    f"{method.method_id}"
                )
            sources = (
                *sources,
                *unavailable_reused_rows(
                    plan,
                    method,
                    "upstream freeze manifest not supplied to this inventory",
                ),
            )
            continue
        imported, reused_sources = load_reused_freeze(
            plan, method, args.raw_agent_freeze_manifest
        )
        requests = (*requests, *imported)
        sources = (*sources, *reused_sources)
        reused_manifest = {
            "method_id": method.method_id,
            "path": str(args.raw_agent_freeze_manifest.expanduser().resolve()),
            "sha256": _sha256(args.raw_agent_freeze_manifest),
            "imported_request_count": len(imported),
        }

    counts = reconcile(plan, sources)
    verify_request_roster_association(sources, requests)

    output_root.mkdir(parents=True, exist_ok=True)
    requests_path = output_root / REQUESTS_NAME
    _write_or_validate(
        requests_path, "".join(item.model_dump_json() + "\n" for item in requests)
    )
    roster_path = output_root / ROSTER_NAME
    _write_or_validate(
        roster_path, "".join(item.model_dump_json() + "\n" for item in sources)
    )
    withheld = tuple(
        outcome for item in sources if (outcome := item.outcome()) is not None
    )
    withheld_path = output_root / WITHHELD_NAME
    _write_or_validate(
        withheld_path, "".join(item.model_dump_json() + "\n" for item in withheld)
    )

    manifest = {
        "schema_version": "phase-b-external-baseline-freeze-2",
        "status": plan.status,
        "execution_authorized": bool(args.authorize_execution),
        "plan_path": str(args.config.expanduser().resolve()),
        "plan_sha256": _sha256(args.config),
        "hidden_contract_audit_sha256": audit_sha256,
        "cell_count": len(plan.cells),
        "repetition_count": len(plan.repetitions),
        "reporting_roster": plan.reporting_roster,
        "method_ids": [item.method_id for item in plan.methods],
        "reused_freeze": reused_manifest,
        "withheld_outcome_count": len(withheld),
        "source_adapter_requests_sha256": _sha256(requests_path),
        "external_baseline_roster_sha256": _sha256(roster_path),
        "withheld_source_outcomes_sha256": _sha256(withheld_path),
        "train_plus_validation_refit_permitted": False,
        "parameter_refit_applied": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "weighted_overall_score_defined": False,
        **counts,
    }
    manifest_path = output_root / MANIFEST_NAME
    _write_or_validate(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    _write_or_validate(
        manifest_path.with_name(f"{manifest_path.name}.sha256"),
        f"{_sha256(manifest_path)}  {manifest_path.name}\n",
    )
    print(
        f"roster={counts['roster_row_count']}/{counts['expected_source_count']} "
        f"available={counts['available_source_count']} "
        f"missing={counts['missing_source_count']} "
        f"evaluator_unsupported={counts['evaluator_unsupported_count']} "
        f"requests={counts['adapter_request_count']} "
        f"execution_authorized={manifest['execution_authorized']}"
    )


def _verify_existing(output_root: Path) -> None:
    """Re-check digests, request/roster association, and source hashes."""
    manifest = json.loads((output_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    roster_path = output_root / ROSTER_NAME
    requests_path = output_root / REQUESTS_NAME
    if _sha256(roster_path) != manifest.get("external_baseline_roster_sha256"):
        raise ValueError("frozen roster digest differs from its manifest")
    if _sha256(requests_path) != manifest.get("source_adapter_requests_sha256"):
        raise ValueError("frozen request digest differs from its manifest")
    withheld_path = output_root / WITHHELD_NAME
    if _sha256(withheld_path) != manifest.get("withheld_source_outcomes_sha256"):
        raise ValueError("frozen withheld-outcome digest differs from its manifest")
    sources = _read_roster(roster_path)
    requests = tuple(
        SourceAdapterRequest.model_validate_json(line)
        for line in requests_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    verify_request_roster_association(sources, requests)
    verify_recorded_hashes(sources)
    print(
        f"verified {len(sources)} roster rows, {len(requests)} requests, "
        "and every recorded source hash"
    )


def _seal(parser, args, output_root: Path) -> None:
    """Bind adaptation outputs and evaluation settings before test entry."""
    for name in (
        "adapted_root",
        "public_data_root",
        "evaluator_commit",
        "chain_inputs_digest",
        "maximum_trajectory_wall_time_seconds",
    ):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required to seal")
    manifest_path = output_root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("execution_authorized") is not True:
        raise ValueError("cannot seal an execution record for an unauthorized freeze")
    adapted = args.adapted_root.expanduser().resolve()
    subjects = adapted / "frozen_evaluation_subjects.jsonl"
    outcomes = adapted / "source_adapter_outcomes.jsonl"
    for path in (subjects, outcomes):
        if not path.is_file():
            raise ValueError(f"adaptation output is missing: {path}")
    plan = load_external_baseline_plan(args.config)
    record = {
        "schema_version": "phase-b-external-baseline-execution-record-2",
        "freeze_manifest_sha256": _sha256(manifest_path),
        "source_adapter_requests_sha256": manifest["source_adapter_requests_sha256"],
        "external_baseline_roster_sha256": manifest[
            "external_baseline_roster_sha256"
        ],
        "withheld_source_outcomes_sha256": manifest[
            "withheld_source_outcomes_sha256"
        ],
        "frozen_evaluation_subjects_sha256": _sha256(subjects),
        "source_adapter_outcomes_sha256": _sha256(outcomes),
        "public_data_root": str(args.public_data_root.expanduser().resolve()),
        "public_data_identity_sha256": public_development_identity(
            args.public_data_root, plan
        ),
        "evaluator_code_commit": args.evaluator_commit,
        "chain_inputs_digest": args.chain_inputs_digest,
        "maximum_trajectory_wall_time_seconds": (
            args.maximum_trajectory_wall_time_seconds
        ),
        "test_data_opened": False,
    }
    record_path = output_root / EXECUTION_NAME
    _write_or_validate(record_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    _write_or_validate(
        record_path.with_name(f"{record_path.name}.sha256"),
        f"{_sha256(record_path)}  {record_path.name}\n",
    )
    print(f"sealed execution record at {record_path}")


def _verify_execution_record(parser, args, output_root: Path) -> None:
    """Re-check the sealed contract before the evaluator opens test data.

    Everything is compared in Python against resolved paths. A companion
    digest checked with `sha256sum -c` from another working directory resolves
    the recorded basename against that directory instead, which silently fails.
    """
    for name in ("adapted_root", "evaluator_commit", "chain_inputs_digest"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required to verify")
    if args.public_data_root is None and args.public_identity is None:
        parser.error("--public-data-root or --public-identity is required to verify")

    record_path = output_root / EXECUTION_NAME
    record = json.loads(record_path.read_text(encoding="utf-8"))
    companion = record_path.with_name(f"{record_path.name}.sha256").read_text(
        encoding="utf-8"
    ).split()
    if companion[:1] != [_sha256(record_path)]:
        raise ValueError("execution record differs from its companion digest")

    manifest_path = output_root / MANIFEST_NAME
    checks: list[tuple[str, object, object]] = [
        ("freeze manifest", _sha256(manifest_path), record["freeze_manifest_sha256"]),
        (
            "adapted subjects",
            _sha256(args.adapted_root / "frozen_evaluation_subjects.jsonl"),
            record["frozen_evaluation_subjects_sha256"],
        ),
        (
            "adapted outcomes",
            _sha256(args.adapted_root / "source_adapter_outcomes.jsonl"),
            record["source_adapter_outcomes_sha256"],
        ),
        (
            "frozen roster",
            _sha256(output_root / ROSTER_NAME),
            record["external_baseline_roster_sha256"],
        ),
        (
            "frozen requests",
            _sha256(output_root / REQUESTS_NAME),
            record["source_adapter_requests_sha256"],
        ),
        ("evaluator commit", args.evaluator_commit, record["evaluator_code_commit"]),
        (
            "chain inputs digest",
            args.chain_inputs_digest,
            record["chain_inputs_digest"],
        ),
        (
            "trajectory wall-time limit",
            args.maximum_trajectory_wall_time_seconds,
            record["maximum_trajectory_wall_time_seconds"],
        ),
    ]
    if args.public_data_root is not None:
        checks.append(
            (
                "public data root",
                str(args.public_data_root.expanduser().resolve()),
                record["public_data_root"],
            )
        )
        identity = (
            args.public_identity
            if args.public_identity is not None
            else public_development_identity(
                args.public_data_root, load_external_baseline_plan(args.config)
            )
        )
    else:
        identity = args.public_identity
    checks.append(
        ("public data identity", identity, record["public_data_identity_sha256"])
    )

    differing = [name for name, actual, sealed in checks if actual != sealed]
    if differing:
        raise ValueError(
            "execution differs from the sealed record: " + ", ".join(differing)
        )
    if record.get("test_data_opened") is not False:
        raise ValueError("sealed record already reports opened test data")
    print(f"execution record verified: {len(checks)} bound settings match")


def _read_roster(path: Path) -> tuple[ExternalBaselineSource, ...]:
    return tuple(
        ExternalBaselineSource.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _write_or_validate(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"existing frozen artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
