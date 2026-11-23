"""Replay the November 22 matched 20B graph replies under the measured-variable gate."""

from __future__ import annotations

import argparse
from collections import Counter
from html import escape
from pathlib import Path

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import preproposal_graph_repair as repair
from autoformalism.research import preproposal_topology as topology
from autoformalism.research.preproposal_cases import T2_CASE
from autoformalism.research.preproposal_selection import SelectionIntake
from scripts.construction_functions import file_hash

SAVED_PLAN = "67c92fdb2455cf6843d4cb6e08b5064e8170da291766d93299ddd8dc03b065a3"
DEFAULT_SOURCE = Path("output/preproposal-matched-results-20261122/returned")
DEFAULT_DEST = Path("output/complete-graphs-review-20261122")


def _receipt(source, path, plan_digest):
    """Verify the saved assessment and the unedited provider response it references."""
    slot = path.stem
    saved = sealed_read(path)
    directory = source / "steps" / slot
    reply = sealed_read(directory / "reply.json")
    physical = directory / "calls" / (reply["request_hash"] + ".json")
    if (saved["slot"], saved["plan_sha256"], saved["reply_sha256"]) != (
        slot,
        plan_digest,
        reply["artifact_sha256"],
    ) or file_hash(physical) != reply["physical_receipt_sha256"]:
        raise ValueError(f"Archived request and assessment differ at {slot}")
    return saved["assessment"], reply


def _focus(graph, assess, returned):
    """Add the insulin edge, then test each repair for U on that same child."""
    insulin = repair.Edit("add_source", "I", "Gp")
    configurations = {
        "Original graph": (),
        "Add Gp to dI/dt": (insulin,),
        "Also add Gt to U": (insulin, repair.Edit("add_source", "U", "Gt")),
        "Alternatively use Uid(0) = g(Gt(0))": (
            insulin,
            repair.Edit("add_public_initial", "Uid", "Gt"),
        ),
    }
    rows = []
    for label, edits in configurations.items():
        child = repair._apply(graph, edits)
        outcome = assess(None, child)
        rows.append(
            {
                "label": label,
                "graph": child,
                "assessment": outcome,
                "returned_by_repair": outcome.get("structure_sha256") in returned,
            }
        )
    return rows


def replay(source: Path) -> dict:
    """Replay all accepted and rejected graphs on their original training input."""
    plan = sealed_read(source / "plan.json")
    intake = sealed_read(source / "intake.json")
    if (
        plan["artifact_sha256"] != SAVED_PLAN
        or plan["intake_sha256"] != intake["artifact_sha256"]
    ):
        raise ValueError(
            "Expected the frozen matched-search worker and its training intake"
        )
    cases = SelectionIntake.model_validate(intake["selection"]).cases
    rows, focused = [], []
    old_children, new_children = set(), set()
    for case in cases:
        frozen = intake["frozen_cases"][case.case_id] | {
            "check_all_measured": True,
            "allow_topology_latents": True,
        }
        assess = topology.make_topology_assessor(case, frozen)
        for path in sorted((source / "results").glob(case.case_id + "--seed-*")):
            slot = path.stem
            if slot.endswith("--select"):
                continue
            old, reply = _receipt(source, path, SAVED_PLAN)
            if reply["status"] != "visible":
                raise ValueError(f"Expected a visible saved graph at {slot}")
            graph = reply["value"]
            row = {
                "slot": slot,
                "reply_sha256": reply["artifact_sha256"],
                "case": case.case_id,
                "old_status": old["status"],
            }
            if slot.endswith("--graph-first"):
                row["arm"] = "graph_first"
                checked = repair.assess_and_repair(case, graph, frozen)
                row["assessment"] = {k: v for k, v in checked.items() if k != "repairs"}
                children = checked["repairs"]
                new_ids = {v["structure_sha256"] for v in children}
                row["return_count"] = len(children)
                row["new_children"] = children
                new_children.update(new_ids)
                row["old_children"] = [
                    {
                        "saved_structure_sha256": child["structure_sha256"],
                        "edits": child["edits"],
                        "assessment": assess(None, child["graph"]),
                    }
                    for child in old["repairs"]
                ]
                old_children.update(c["structure_sha256"] for c in old["repairs"])
                if slot == T2_CASE + "--seed-100--graph-first":
                    focused = _focus(graph, assess, new_ids)
            else:
                row["arm"] = "selected_source"
                choice = {
                    "candidate_id": old["candidate_id"],
                    "rationale": "Archived choice.",
                }
                row["assessment"] = assess(choice, graph)
            rows.append(row)
    inputs = Counter(row["arm"] for row in rows)
    if inputs != {"selected_source": 25, "graph_first": 10} or len(focused) != 4:
        raise ValueError(f"Incomplete archived graph reply set: {inputs}")
    later_gaps = []
    for row in rows:
        for child in row.get("old_children", ()):
            checks = child["assessment"].get("training_checks", {})
            contradicted = {
                k for k, v in checks.items() if v["status"] == "contradicted"
            }
            if contradicted == {"U"} and checks["U"]["initial_conflicting_pairs"] == 0:
                later_gaps.append(checks["U"]["max_observed_group_gap"])
    return {
        "source_plan_sha256": SAVED_PLAN,
        "source_intake_sha256": intake["artifact_sha256"],
        "provider_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
        "status_counts": {
            arm: dict(
                Counter(r["assessment"]["status"] for r in rows if r["arm"] == arm)
            )
            for arm in inputs
        },
        "old_distinct_children": len(old_children),
        "new_distinct_children": len(new_children),
        "saved_children_failing_only_later_U": len(later_gaps),
        "largest_later_U_gap_in_these_children": max(later_gaps, default=None),
        "admitted_saved_children": sum(
            c["assessment"]["status"] == repair.PASS
            for r in rows
            for c in r.get("old_children", ())
        ),
        "rows": rows,
        "focus": focused,
    }


def render(report: dict) -> str:
    """Summarize replay coverage and show each registered equation edit and outcome."""

    def table(labels, rows):
        header = "".join(f"<th>{escape(k)}</th>" for k in labels)
        lines = ["".join(f"<td>{escape(str(v))}</td>" for v in row) for row in rows]
        return (
            "<table><tr>"
            + header
            + "</tr>"
            + "".join("<tr>" + v + "</tr>" for v in lines)
            + "</table>"
        )

    rows = report["rows"]
    parents = [r for r in rows if r["arm"] == "graph_first"]
    largest_gap = report["largest_later_U_gap_in_these_children"]
    focus = []
    for item in report["focus"]:
        a = item["assessment"]
        checks = a.get("training_checks", {})
        focus.append(
            (
                item["label"],
                a["status"],
                checks.get("U", {}).get("initial_conflicting_pairs", "—"),
                ", ".join(n for n, c in checks.items() if c["status"] == "contradicted")
                or "None",
                item["returned_by_repair"],
            )
        )
    return "\n".join(
        [
            "<!doctype html><html lang=en><meta charset=utf-8>"
            "<title>Saved graph replay</title>",
            "<style>body{max-width:1100px;margin:3em auto;padding:0 1em;"
            "font:17px/1.45 system-ui;color:#193347}th,td{padding:.45em .7em;"
            "border-bottom:1px solid #cad4d5;text-align:left}"
            "table{border-collapse:collapse}h1,h2{color:#14333e}</style>",
            "<h1>Saved graph replay</h1>",
            "<p>These 35 graphs were written by GPT-OSS-20B for the November "
            "22 source-selection and graph-first experiment. Each reply is "
            "replayed verbatim. "
            "The updated runtime permits fully defined new latent variables "
            "and tests each generated variable with training measurements. "
            "The original inventory and source permissions are "
            "preserved.</p>",
            table(
                ("Graph replies", "Examined", "Status counts"),
                [
                    (arm, sum(v.values()), ", ".join(f"{k} {n}" for k, n in v.items()))
                    for arm, v in report["status_counts"].items()
                ],
            ),
            "<h2>Graph-first search</h2>",
            f"<p>{report['admitted_saved_children']} of the "
            f"{report['old_distinct_children']} previously returned structures pass "
            "the updated checks. Repeating the deterministic search produces "
            f"{report['new_distinct_children']} distinct passing structures. "
            "Each repair can add up to two edges or initial-value arguments.</p>",
            f"<p>{report['saved_children_failing_only_later_U']} archived children "
            "have consistent initial values and pass every test except later U. "
            "Their largest discrepancy among equal arguments is "
            f"{format(largest_gap, '.6g') if largest_gap is not None else 'N/A'} "
            "mg kg⁻¹ min⁻¹. The registered tolerance for U was zero.</p>",
            table(
                (
                    "Case and seed",
                    "Original reply under new checks",
                    "Checked repairs",
                    "Passing repairs",
                    "Budget exhausted",
                ),
                [
                    (
                        r["slot"].removesuffix("--graph-first"),
                        r["assessment"]["status"],
                        r["assessment"]["checked"],
                        r["return_count"],
                        r["assessment"]["budget_exhausted"],
                    )
                    for r in parents
                ],
            ),
            "<h2>Two repairs for the same algebraic output</h2>",
            "<p>The original exact-screen graph for seed 100 declares U = "
            "H(Uii, I, Uid), with Uid(0) shared across trajectories. Rows "
            "three and four each modify the graph in row two. The "
            "initial-time count uses the 120 pairs of training trajectories. "
            "Whole-graph assessment includes later observations.</p>",
            table(
                (
                    "Edit",
                    "Status",
                    "Conflicting pairs for U(0)",
                    "Variables with a detected conflict",
                    "Returned as minimal repair",
                ),
                focus,
            ),
            "<p>All conclusions concern the available training data. "
            "Function construction and numerical fitting were outside this "
            "replay.</p></html>",
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    data = replay(args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "replay.json", data)
    (args.out / "REPLAY.html").write_text(render(data), encoding="utf-8")
    print({k: v for k, v in data.items() if k not in ("rows", "focus")})
