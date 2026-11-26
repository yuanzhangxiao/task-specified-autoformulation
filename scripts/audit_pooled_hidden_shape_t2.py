"""Exploratory T2 matched-start coverage after pooling complementary training folds."""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import hidden_start_contrasts as starts
from scripts import audit_hidden_shape_t2 as folds
from scripts.construction_functions import file_hash
from scripts.matched_equation_experiment import DEFAULT, _registered

OUTPUT = folds.OUTPUT.parent / "t2-pooled"


def _pool(source: Path, panels: list[dict]) -> tuple[dict, list[dict]]:
    """Require the two sealed construction folds to be complementary and compatible."""
    if [p["fold"] for p in panels] != [0, 1]:
        raise ValueError("Expected two complementary training folds")
    packets = [
        _registered(source / "packets" / f"{p['panel_id']}.json", p["packet"])
        for p in panels
    ]
    if any(
        p["panel_id"] != cfg["panel_id"] for p, cfg in zip(packets, panels, strict=True)
    ):
        raise ValueError("Frozen packet identity differs")
    for key in ("graph", "interpolation"):
        if packets[0][key] != packets[1][key]:
            raise ValueError(f"Fold-specific {key} differs")
    rows = []
    ids = []
    for packet, other in zip(packets, reversed(panels), strict=True):
        part = packet["train_rows"]
        ordered = sorted(row["trajectory_id"] for row in part)
        if ordered != sorted(other["held_ids"]) or len(set(ordered)) != len(part):
            raise ValueError("Training folds are not complementary")
        if not all(i.startswith("train_") for i in ordered):
            raise ValueError("Only training events can enter the pooled study")
        ids.extend(ordered)
        rows.extend(part)
    if len(set(ids)) != len(rows):
        raise ValueError("Training events overlap")
    return packets[0], sorted(rows, key=lambda row: row["trajectory_id"])


def _counts(equations: list[dict]) -> dict:
    counts = folds._summarize(equations)
    counts["equations"] = counts.pop("equation_fold_cases")
    return counts


def _rate_sensitivity(equations: list[dict]) -> dict | None:
    """Give one ordered source slice and its largest three-level secant departure."""
    best = None
    maximum = Fraction(0)
    for equation in sorted(equations, key=lambda e: (e["target"], e["graph"])):
        if not equation["hidden_sources"] or equation["type"] != "differential":
            continue
        for source in sorted(equation["sources"], key=lambda s: s["source"]):
            for triple in combinations(source["largest_slice"], 3):
                xa, xb, xc = [Fraction(s["level"]) for s in triple]
                scale = max(abs(xa), abs(xb), abs(xc), 1)
                if min(xb - xa, xc - xb) <= Fraction(1, 10**9) * scale:
                    continue
                weight = (xb - xa) / (xc - xa)
                for obs in product(*(s["observations"] for s in triple)):
                    left, middle, right = [Fraction(r["estimate"]) for r in obs]
                    affine = (1 - weight) * left + weight * right
                    departure = abs(middle - affine)
                    # A middle-rate interval expands by epsilon, as does the
                    # interpolated endpoint interval (nonnegative weights sum to 1).
                    if departure > maximum:
                        maximum = departure
                        best = {
                            "graph": equation["graph"],
                            "target": equation["target"],
                            "source": source["source"],
                            "hidden_sources": equation["hidden_sources"],
                            "events": [
                                {
                                    "id": r["id"],
                                    "source_level": s["level"],
                                    "forward_secant": r["estimate"],
                                    "first_interval_duration": r["dt"],
                                }
                                for s, r in zip(triple, obs, strict=True)
                            ],
                            "interpolation_weight": float(weight),
                            "affine_middle_from_endpoint_secants": float(affine),
                            "absolute_middle_departure": float(departure),
                            "required_strict_uniform_rate_error": float(departure / 2),
                            "given_rate_error": None,
                            "status": "sensitivity_only",
                            "selection": (
                                "First eligible target, graph and source by name. "
                                "Largest absolute departure among event triples in "
                                "its stored matched slice."
                            ),
                        }
            if best is not None:
                return best
    return None


def run(source: Path, out: Path) -> dict:
    """Report design coverage on all training events."""
    parent = _registered(source / "plan.json", folds.FROZEN)
    grouped = defaultdict(list)
    for row in parent["panels"]:
        grouped[row["graph_id"]].append(row)
    with _lock(out):
        plan = sealed_write(
            out / "plan.json",
            {
                "protocol": "exploratory-pooled-hidden-shape-t2-1",
                "parent": folds.FROZEN,
                "script": file_hash(Path(__file__)),
                "diagnostic": file_hash(Path(starts.__file__)),
                "summarizer": file_hash(Path(folds.__file__)),
                "scope": "Exploratory coverage using all training events.",
                "algebraic_initial_error": 0.0,
                "right_rate_bound": None,
            },
        )["artifact_sha256"]
        all_equations = []
        receipts = []
        event_counts = []
        for graph_id, panels in grouped.items():
            ordered = sorted(panels, key=lambda p: p["fold"])
            packet, rows = _pool(source, ordered)
            name = packet["panel_id"].rsplit("-fold", 1)[0]
            reference = {
                "plan": plan,
                "graph_id": graph_id,
                "graph": name,
                "source_packets": [p["packet"] for p in ordered],
                "training_ids": [r["trajectory_id"] for r in rows],
            }
            event_counts.append(len(rows))
            checkpoint = out / "graphs" / f"{name}.json"
            if checkpoint.exists():
                result = sealed_read(checkpoint)
                if any(result[k] != v for k, v in reference.items()):
                    raise ValueError("Pooled graph checkpoint differs")
            else:
                graph = packet["graph"]
                algebraic = {
                    s: 0.0
                    for s, kind in graph["variable_types"].items()
                    if kind == "algebraic"
                }
                contrasts = starts.matched_starts(
                    graph,
                    rows,
                    interpolation=packet["interpolation"],
                    algebraic_error=algebraic,
                )
                result = sealed_write(checkpoint, {**reference, **contrasts})
            receipts.append(result["artifact_sha256"])
            all_equations.extend({**e, "graph": name} for e in result["equations"])
        hidden = [e for e in all_equations if e["hidden_sources"]]
        return sealed_write(
            out / "summary.json",
            {
                "plan": plan,
                "graphs": receipts,
                "events_per_graph": event_counts,
                "all_measured": _counts(all_equations),
                "hidden_sourced": _counts(hidden),
                "rate_bound_sensitivity": _rate_sensitivity(hidden),
                "three_level_contrasts": [
                    {
                        "graph": e["graph"],
                        "target": e["target"],
                        "source": s["source"],
                        "response": e["response"],
                        "hidden_sources": e["hidden_sources"],
                        "levels": [v["level"] for v in s["largest_slice"]],
                        "status": s["status"],
                    }
                    for e in all_equations
                    for s in e["sources"]
                    if s["maximum_levels"] >= 3
                ],
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(run(args.source, args.output)["hidden_sourced"])
