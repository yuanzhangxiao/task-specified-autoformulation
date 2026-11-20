#!/usr/bin/env python3
"""Enumerate bounded topology edits for a public training-pair contradiction.

Pass the audit input {"draft": ..., "evidence": ...} and list every source
permitted for this search. The output preserves the input topology
digest and records each graph edit, its status and outstanding conflicts.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.schemas.base import StrictSchema
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_compatibility import TrainingCompatibilityEvidence
from autoformalism.search.topology_repair_generator import generate_repairs
from autoformalism.staged_topology import content_hash


class RepairInput(StrictSchema):
    """Input shared with the offline topology diagnostic CLI."""

    draft: Draft
    evidence: TrainingCompatibilityEvidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--left", required=True, help="First training trajectory ID")
    parser.add_argument("--right", required=True, help="Second training trajectory ID")
    parser.add_argument("--time", required=True, type=float, help="Matched sample time")
    parser.add_argument("--allowed-source", required=True, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Choose an output different from the source evidence.")
    source = RepairInput.model_validate_json(args.input.read_bytes())
    edits = generate_repairs(
        source.draft,
        source.evidence,
        args.target,
        allowed_sources=tuple(args.allowed_source),
        pair=(args.left, args.right),
        time=args.time,
    )
    counts = dict(Counter(edit.status for edit in edits))
    report = sealed_write(
        args.output,
        {
            "protocol": "offline-bounded-topology-repairs-1",
            "draft_sha256": content_hash(source.draft.model_dump(mode="json")),
            "evidence_sha256": source.evidence.sha256,
            "target": args.target,
            "witness_pair": [args.left, args.right],
            "witness_time": args.time,
            "allowed_sources": sorted(set(args.allowed_source)),
            "assessments": [e.model_dump(mode="json") for e in edits],
            "status_counts": counts,
            "optimizer_calls": 0,
            "llm_calls": 0,
            "test_data_opened": False,
        },
    )
    print(
        json.dumps(
            {
                "report": str(args.output),
                "sha256": report["artifact_sha256"],
                "status_counts": counts,
            }
        )
    )


if __name__ == "__main__":
    main()
