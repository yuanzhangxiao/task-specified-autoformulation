#!/usr/bin/env python3
"""Embed warm-start parameters in a frozen candidate pool for remote replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from autoformalism.rebuttal.artifacts import (
    CandidateArtifact,
    candidate_warm_start_parameters,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    records = tuple(
        CandidateArtifact.model_validate_json(line)
        for line in args.source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    portable = []
    missing = []
    for record in records:
        parameters = candidate_warm_start_parameters(record)
        if record.candidate.parameters and not parameters:
            missing.append(record.artifact_id)
        portable.append(
            record.model_copy(update={"fitted_global_parameters": parameters})
        )
    serialized = "".join(item.model_dump_json() + "\n" for item in portable)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(args.output)

    report = {
        "schema_version": "portable_candidate_pool_v1",
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "candidate_count": len(portable),
        "embedded_warm_start_count": sum(
            bool(item.fitted_global_parameters) for item in portable
        ),
        "parameterized_without_warm_start": missing,
    }
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if missing:
        raise SystemExit("parameterized candidates are missing portable warm starts")


if __name__ == "__main__":
    main()
