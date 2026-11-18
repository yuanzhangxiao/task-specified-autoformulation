"""Report inclusive retained equation revisions and overlapping latent additions.

Uses authenticated, frozen audit results only. No trajectories, fitting, model
selection, or LLM calls are needed. Joint state/equation edits enter the primary
counts; fixed-roster counts remain a labeled comparison subset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts.collect_iclr_latent_histories import read_sealed
from scripts.iclr_latent_revision_audit import write_table

FIELDS = (
    "equation_revision_count",
    "nonlinear_form_revision_count",
    "fixed_roster_equation_revision_count",
    "fixed_roster_nonlinear_form_revision_count",
    "latent_addition_count",
    "numerically_cited_latent_addition_count",
    "both_latent_and_equation_revision",
    "both_latent_and_nonlinear_form_revision",
    "same_visit_latent_and_nonlinear_form_revision",
)


def keyed(rows: list[dict]) -> dict:
    """Reject duplicated run/snapshot identities instead of silently merging."""
    result = {}
    for row in rows:
        key = row["run_id"], row["snapshot"]
        if key in result:
            raise ValueError("duplicate run/snapshot identity")
        result[key] = row
    return result


def overlap_rows(functions: dict, latents: dict) -> list[dict]:
    """Join the same selected endpoints and allow both model- and visit-overlap."""
    function_rows, latent_rows = keyed(functions["runs"]), keyed(latents["runs"])
    births = defaultdict(set)
    for state in latents["states"]:
        if state["retained_pure_augmentation"]:
            births[(state["run_id"], state["snapshot"])].add(state["birth"]["round"])
    forms = defaultdict(set)
    for event in functions["events"]:
        if event["nonlinear_shape_revision_retained"]:
            forms[(event["run_id"], event["snapshot"])].add(event["round"])
    rows = []
    for key, source in function_rows.items():
        row = dict(source)
        row["assessed"] = bool(
            row["available"]
            and not row.get("audit_error")
            and not row.get("history_audit_errors")
        )
        if row["assessed"]:
            other = latent_rows.get(key)
            if not other or not other["available"]:
                raise ValueError("matching latent endpoint unavailable")
            if other["model_id"] != row["model_id"]:
                raise ValueError("selected endpoint mismatch between audits")
            row.update(
                equation_revision_count=row[
                    "retained_equation_revision_any_roster_count"
                ],
                nonlinear_form_revision_count=row[
                    "retained_shape_revision_any_roster_count"
                ],
                fixed_roster_equation_revision_count=row[
                    "retained_equation_revision_count"
                ],
                fixed_roster_nonlinear_form_revision_count=row[
                    "retained_shape_revision_count"
                ],
                latent_addition_count=other["retained_pure_augmentation_count"],
                numerically_cited_latent_addition_count=other[
                    "numerically_cited_addition_count"
                ],
            )
            row["both_latent_and_equation_revision"] = int(
                row["latent_addition_count"] > 0 and row["equation_revision_count"] > 0
            )
            row["both_latent_and_nonlinear_form_revision"] = int(
                row["latent_addition_count"] > 0
                and row["nonlinear_form_revision_count"] > 0
            )
            row["same_visit_latent_and_nonlinear_form_revision"] = int(
                bool(births[key] & forms[key])
            )
        rows.append(row)
    return rows


def summary(rows: list[dict]) -> dict:
    """Keep unavailable or unassessed runs out of the measured distributions."""
    ready = [r for r in rows if r["assessed"]]
    result = {
        "assigned": len(rows),
        "available": sum(r["available"] for r in rows),
        "assessed": len(ready),
    }
    for field in FIELDS:
        counts = Counter(r[field] for r in ready)
        present = sum(v for k, v in counts.items() if k > 0)
        result[field] = {
            "histogram": dict(sorted(counts.items())),
            "models_with_any": present,
            "percent_of_assessed": 100 * present / len(ready) if ready else None,
            "percent_of_assignments": 100 * present / len(rows) if rows else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("function_audit", type=Path)
    parser.add_argument("latent_audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    functions, latents = (
        read_sealed(args.function_audit),
        read_sealed(args.latent_audit),
    )
    rows = overlap_rows(functions, latents)
    tables = {}
    for fields in [
        ("snapshot", "run_class", "proposer"),
        ("snapshot", "run_class", "proposer", "arm"),
        ("snapshot", "cohort", "benchmark", "arm"),
    ]:
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[f] for f in fields)].append(row)
        tables["/".join(fields)] = [
            {**dict(zip(fields, key, strict=True)), **summary(group)}
            for key, group in sorted(groups.items())
        ]
    args.output.mkdir(parents=True, exist_ok=True)
    sealed_write(
        args.output / "overlap.json",
        json.loads(
            json.dumps(
                {
                    "protocol": "inclusive-equation-latent-overlap-1",
                    "function_audit_sha256": functions["artifact_sha256"],
                    "latent_audit_sha256": latents["artifact_sha256"],
                    "definitions": {
                        "equation_revision": (
                            "Retained expression/dependency revision, "
                            "allowing joint state edits."
                        ),
                        "nonlinear_form_revision": (
                            "Retained nonlinear-class change in an existing "
                            "equation/readout, allowing joint state edits; "
                            "not necessarily a globally new operator."
                        ),
                        "latent_addition": (
                            "Retained active strict augmentation from the latent "
                            "audit; dimension increases without simultaneous "
                            "state removal."
                        ),
                        "both": (
                            "The same endpoint retains both categories; "
                            "the revision visits may differ."
                        ),
                        "same_visit": (
                            "A retained latent addition and nonlinear-form "
                            "revision occurred in the same proposal visit."
                        ),
                    },
                    "runs": rows,
                    "tables": tables,
                }
            )
        ),
    )
    write_table(args.output / "runs.csv", rows)
    print(f"Reported {len(rows) // 2} assignments with overlapping categories")


if __name__ == "__main__":
    main()
