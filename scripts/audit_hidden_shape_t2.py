"""Audit variation at matched starts using the twelve sealed T2 fitting packets."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import hidden_start_contrasts as starts
from scripts.construction_functions import file_hash
from scripts.matched_equation_experiment import DEFAULT, _registered

FROZEN = "9610204f6aba2a826c373a32cd83f56021ece9302f043b69bfa18f86230384e8"
OUTPUT = DEFAULT.parent / "partial-shape-20261126/t2"


def _summarize(equations: list[dict]) -> dict:
    return {
        "equation_fold_cases": len(equations),
        "cases_with_at_least_one_equal_hidden_pair": sum(
            e["equal_hidden_initial_pairs"] > 0 for e in equations
        ),
        "cases_with_isolated_variation": sum(
            any(s["isolated_pairs"] for s in e["sources"]) for e in equations
        ),
        "cases_with_three_or_more_levels": sum(
            any(s["maximum_levels"] >= 3 for s in e["sources"]) for e in equations
        ),
        "cases_with_repeated_level_conflict": sum(
            any(s["status"] == "inconsistent_repeated_level" for s in e["sources"])
            for e in equations
        ),
        "cases_with_certified_affine_exclusion": sum(
            any(s["status"] == "affine_excluded_on_slice" for s in e["sources"])
            for e in equations
        ),
        "cases_with_isolated_jumps": sum(
            any(j["isolated_interior_jumps"] for j in e["jump_candidates"])
            for e in equations
        ),
        "cases_with_one_hidden_filter_graph": sum(
            bool(e["filter_candidates"]) for e in equations
        ),
    }


def run(source: Path, out: Path) -> dict:
    """Verify the parent plan and packet at each checkpoint, including on resume."""
    parent = _registered(source / "plan.json", FROZEN)
    with _lock(out):
        identity = sealed_write(
            out / "plan.json",
            {
                "protocol": "hidden-shape-t2-fold-audit-1",
                "source_plan": FROZEN,
                "script": file_hash(Path(__file__)),
                "diagnostic": file_hash(Path(starts.__file__)),
                "algebraic_initial_error": 0.0,
                "derivative_error_bound": None,
                "scope": "Fit-fold samples only; no forecasts or held data opened.",
            },
        )["artifact_sha256"]
        equations = []
        receipts = []
        for panel in parent["panels"]:
            name = panel["panel_id"]
            packet = _registered(source / "packets" / f"{name}.json", panel["packet"])
            if packet["panel_id"] != name or set(panel["held_ids"]).intersection(
                row["trajectory_id"] for row in packet["train_rows"]
            ):
                raise ValueError("Packet identity or split differs")
            checkpoint = out / "panels" / f"{name}.json"
            expected = {"panel": name, "packet": panel["packet"], "plan": identity}
            if checkpoint.exists():
                result = sealed_read(checkpoint)
                if any(result[k] != v for k, v in expected.items()):
                    raise ValueError("Panel checkpoint differs")
            else:
                graph = packet["graph"]
                algebraic = {
                    k: 0.0
                    for k, v in graph["variable_types"].items()
                    if v == "algebraic"
                }
                report = starts.matched_starts(
                    graph,
                    packet["train_rows"],
                    interpolation=packet["interpolation"],
                    algebraic_error=algebraic,
                )
                result = sealed_write(checkpoint, {**expected, **report})
            receipts.append(result["artifact_sha256"])
            equations.extend({**e, "panel": name} for e in result["equations"])
        hidden = [e for e in equations if e["hidden_sources"]]
        return sealed_write(
            out / "summary.json",
            {
                "protocol": "hidden-shape-t2-fold-audit-1",
                "plan": identity,
                "panels": receipts,
                "distinct_graphs": len({p["graph_id"] for p in parent["panels"]}),
                "all_measured": _summarize(equations),
                "hidden_sourced": _summarize(hidden),
                "cases": [
                    {
                        "panel": e["panel"],
                        "target": e["target"],
                        "type": e["type"],
                        "hidden_sources": e["hidden_sources"],
                        "equal_hidden_pairs": e["equal_hidden_initial_pairs"],
                        "all_pairs": e["all_pairs"],
                        "maximum_levels": max(
                            (s["maximum_levels"] for s in e["sources"]), default=0
                        ),
                        "sources_with_isolated_variation": [
                            s["source"] for s in e["sources"] if s["isolated_pairs"]
                        ],
                    }
                    for e in hidden
                ],
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(run(args.source, args.output)["hidden_sourced"])
