#!/usr/bin/env python3
"""Collect every evaluated model and its metrics into one transferable bundle.

The results of a comparison live in several roots written on different dates:
a frozen roster, per-subject adapter outcomes, the adapted models themselves,
and the final evaluation records. Reading them means knowing which file joins
to which on what key. This writes one directory that stands on its own.

It reads only what an evaluation already produced. Nothing is recomputed, no
model is run, and no data of any kind is opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

#: Endpoint fields flattened into the per-subject table.
METRIC_FIELDS = (
    "target_nmse",
    "target_status",
    "runtime_valid",
    "mechanism_compliance",
    "mechanism_coverage",
    "mechanism_status",
    "complexity_states",
    "complexity_latent_states",
    "complexity_parameters",
    "complexity_terms",
)


def sha256(path: Path) -> str:
    """Identify each source file so a transferred bundle can be checked."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    """Every object in a JSON Lines file, skipping blank lines."""
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def equations_of(subject: dict) -> dict[str, str]:
    """The model itself: one right-hand side per state."""
    candidate = subject.get("candidate") or {}
    return {
        str(item["state"]): str(item["rhs"])
        for item in candidate.get("state_equations") or []
    }


def metrics_of(record: dict | None) -> dict[str, Any]:
    """Flatten the separate endpoints, keeping each one distinguishable."""
    if record is None:
        return dict.fromkeys(METRIC_FIELDS)
    mechanism = record.get("public_mechanism") or {}
    evaluation = mechanism.get("evaluation") or {}
    target = record.get("target_prediction") or {}
    complexity = record.get("complexity") or {}
    return {
        "target_nmse": target.get("normalized_mse"),
        "target_status": target.get("status"),
        "runtime_valid": (record.get("runtime") or {}).get("valid"),
        "mechanism_compliance": evaluation.get("mechanism_compliance"),
        "mechanism_coverage": evaluation.get("mechanism_coverage"),
        "mechanism_status": mechanism.get("status"),
        "complexity_states": complexity.get("state_count"),
        "complexity_latent_states": complexity.get("latent_state_count"),
        "complexity_parameters": complexity.get("parameter_count"),
        "complexity_terms": complexity.get("additive_term_count"),
    }


def collect(label: str, root: Path) -> tuple[list[dict], dict]:
    """One evaluation root, joined on the keys the evaluation itself uses."""
    frozen = root / "frozen"
    adapted = root / "adapted"
    final = root / "final-evaluation"
    roster = read_jsonl(frozen / "external_baseline_roster.jsonl")
    outcomes = read_jsonl(root / "combined_source_outcomes.jsonl") or read_jsonl(
        adapted / "source_adapter_outcomes.jsonl"
    )
    subjects = read_jsonl(adapted / "frozen_evaluation_subjects.jsonl")
    records = read_jsonl(final / "final_evaluation_records.jsonl")

    by_request = {str(item["request_id"]): item for item in outcomes}
    by_subject_id = {str(item["subject_id"]): item for item in subjects}
    by_record = {str(item["subject_id"]): item for item in records}

    rows = []
    for planned in roster:
        request_id = str(planned["request_id"])
        outcome = by_request.get(request_id) or {}
        subject_id = outcome.get("subject_id")
        subject = by_subject_id.get(str(subject_id)) if subject_id else None
        record = by_record.get(str(subject_id)) if subject_id else None
        equations = equations_of(subject) if subject else {}
        rows.append(
            {
                "evaluation": label,
                "request_id": request_id,
                "method": planned.get("method_id"),
                "benchmark_id": planned.get("benchmark_id"),
                "tier": planned.get("tier"),
                "repetition": planned.get("repetition"),
                "artifact_status": planned.get("artifact_status"),
                "adapter_status": outcome.get("status"),
                "terminal_status": planned.get("terminal_status") or "",
                "reason": planned.get("reason") or outcome.get("error") or "",
                "equation_count": len(equations),
                "equations": equations,
                "source_path": planned.get("source_path"),
                "subject_id": subject_id,
                **metrics_of(record),
            }
        )
    provenance = {
        "label": label,
        "root": str(root),
        "files": {
            str(path.relative_to(root)): sha256(path)
            for path in (
                frozen / "external_baseline_roster.jsonl",
                frozen / "external_baseline_freeze.json",
                frozen / "execution_record.json",
                root / "combined_source_outcomes.jsonl",
                adapted / "frozen_evaluation_subjects.jsonl",
                final / "final_evaluation_records.jsonl",
            )
            if path.is_file()
        },
        "planned": len(roster),
        "with_model": sum(1 for row in rows if row["equation_count"]),
    }
    return rows, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        action="append",
        required=True,
        metavar="LABEL=ROOT",
        help="an evaluation root and the label to record it under",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    provenance: list[dict] = []
    for item in args.evaluation:
        label, _, root = item.partition("=")
        if not root:
            parser.error(f"expected LABEL=ROOT, got {item!r}")
        collected, source = collect(label, Path(root).expanduser().resolve())
        rows.extend(collected)
        provenance.append(source)

    seen: dict[str, str] = {}
    for row in rows:
        method, label = str(row["method"]), str(row["evaluation"])
        if seen.setdefault(method, label) != label:
            raise ValueError(
                f"method {method!r} appears under two evaluations "
                f"({seen[method]!r} and {label!r}); the same model would be "
                "counted twice under different conditions"
            )

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Models, one object per planned identity, with its equations intact.
    (out / "models.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    # The same rows flattened, for anything that reads a table.
    flat = [
        {key: value for key, value in row.items() if key != "equations"}
        | {"equations": json.dumps(row["equations"], sort_keys=True)}
        for row in rows
    ]
    with (out / "models_and_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]) if flat else ["request_id"])
        writer.writeheader()
        writer.writerows(flat)

    manifest = {
        "schema_version": "phase-b-results-package-1",
        "test_data_opened": False,
        "evaluations": provenance,
        "planned_total": len(rows),
        "methods": sorted({str(row["method"]) for row in rows}),
        "with_model": sum(1 for row in rows if row["equation_count"]),
        "with_score": sum(1 for row in rows if row["target_nmse"] is not None),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
