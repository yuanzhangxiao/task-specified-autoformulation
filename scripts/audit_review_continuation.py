#!/usr/bin/env python3
"""Replay saved v2 content under v3 checks; no fitting, provider calls or test data."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from autoformalism.expressions import ModelValidationError
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.search import review_revision_v3 as edits


def convert(raw: dict, target: str) -> dict:
    """Rename interface fields only; never guess the meaning of a wrong mapping."""
    allowed = {
        "hypothesis",
        "evidence_ids",
        "equations",
        "remove",
        "mappings",
        "initializers",
    }
    if set(raw) - allowed:
        raise ValueError("saved response has unrecognized content fields")
    mappings = raw.get("mappings", [])
    if any(m["channel"] != target for m in mappings):
        raise ValueError(
            "saved mapping names an internal/nonpublic channel; "
            "requires a new proposer reply"
        )
    expressions = {m["expression"] for m in mappings}
    if len(expressions) > 1:
        raise ValueError("saved output mappings conflict")
    return {
        "hypothesis": raw["hypothesis"],
        "evidence_refs": raw.get("evidence_ids", []),
        "equations": raw.get("equations", []),
        "remove_variables": raw.get("remove", []),
        "initializers": raw.get("initializers", []),
        "output_expression": next(iter(expressions), None),
    }


def audit(root: Path) -> dict:
    """Record all saved attempts, including the first newly exposed blocking error."""
    plan = io.verify(root)
    if plan["protocol"] != io.CONTINUATION_PROTOCOL:
        raise ValueError("requires a continuation import")
    source = Path(plan["continuation"]["source_root"])
    original = io.verify(source, execution=False)
    if original["artifact_sha256"] != plan["continuation"]["source_plan_sha256"]:
        raise ValueError("source plan changed")
    records = []
    for task in original["tasks"]:
        for index in range(1, plan["continuation"]["source_round"] + 1):
            path = io.round_path(source, task, index) / "proposal.json"
            if not path.exists():
                continue
            proposal = sealed_read(path)
            parent = io.read_round(source, task, index - 1)
            if parent is None or proposal["parent_sha256"] != parent["artifact_sha256"]:
                raise ValueError("saved proposal parent differs")
            for i, attempt in enumerate(proposal.get("attempts", [])):
                record = {
                    "task": task["task_id"],
                    "round": index,
                    "attempt": i,
                    "proposal_sha256": proposal["artifact_sha256"],
                    "source_accepted": attempt["accepted"],
                }
                try:
                    selected = parent["selected"]
                    bundle, packet = selected["bundle"], selected["packet"]
                    raw = attempt.get("raw")
                    if raw is None:
                        raise ValueError(
                            "no parsed model content; delivery/schema failure retained"
                        )
                    result = edits.apply_edits(
                        bundle, packet, convert(raw, bundle["context"]["targets"][0])
                    )
                    if result["bundle"] is not None:
                        certificate = pipeline.certificates(
                            result["bundle"], original["cells"][task["cell"]], task
                        )
                        if not certificate["eligible_for_development_selection"]:
                            raise ValueError(
                                "public requirement or ablation check still fails"
                            )
                    record.update(
                        status="content_valid",
                        outcome=result["outcome"],
                        citation_audit=result["provenance"]["citation_audit"],
                    )
                except (ValueError, KeyError, TypeError, ModelValidationError) as error:
                    record.update(status="still_blocked", error=str(error)[:1500])
                records.append(record)
    return sealed_write(
        root / "source_response_audit.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "records": records,
            "status_counts": dict(Counter(r["status"] for r in records)),
            "fitting_calls": 0,
            "llm_calls": 0,
            "test_data_opened": False,
            "interpretation": "Content checks only; no prediction-quality claim",
        },
    )


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
