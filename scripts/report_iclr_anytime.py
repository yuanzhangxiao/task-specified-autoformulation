"""Join authenticated checkpoint costs, check endpoint parity, and plot budgets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import iclr_anytime_response as study
from scripts import iclr_scaling_response as metrics

ORDER = ("1", "5", "10", "15", "pruned")
FIELDS = ("test_nmse", "response_rse", "directional_accuracy", "shape_error")


def cost_index(files: list[Path]) -> dict:
    """Check each recorded checkpoint against its original scientific history."""
    index = {}
    for path, cohort in zip(files, ("anchors_20b", "stress_20b"), strict=True):
        export = sealed_read(path)
        source = study.history(study.PROJECT / study.HISTORIES / f"{cohort}.json.gz")
        if export["plan_sha256"] != source["plan_sha256"]:
            raise ValueError("cost and model exports belong to different campaigns")
        runs = {r["task"]["task_id"]: r for r in source["runs"]}
        for row in export["rows"]:
            if row["checkpoint"] != "pruned":
                visit = next(
                    v
                    for v in runs[row["task_id"]]["visits"]
                    if v["round"] == int(row["checkpoint"]) - 1
                )
                if row["result_sha256"] != visit["result_sha256"]:
                    raise ValueError("cost and selected model receipts differ")
            key = (cohort, row["task_id"], row["checkpoint"])
            if key in index:
                raise ValueError("duplicate cost receipt")
            index[key] = row
    return index


def parity(rows: list[dict]) -> dict:
    """Final forecasts must reproduce the already reported physical-panel scores."""
    old_cstr = sealed_read(
        study.PROJECT / "output/iclr-specification-panel-20261116-v3/summary.json"
    )["rows"]
    old_glucose = sealed_read(study.PROJECT / study.GLUCOSE_PANEL / "summary.json")[
        "rows"
    ]
    differences = []
    for row in rows:
        if row["checkpoint"] != "pruned" or row["condition"] not in (
            "sensor_00",
            "acquisition_00",
        ):
            continue
        old = (
            next(
                r
                for r in old_cstr
                if r["method"] == "full" and r["repetition"] == row["seed"]
            )
            if row["condition"] == "sensor_00"
            else next(
                r
                for r in old_glucose
                if r["condition"] == "acquisition_00" and r["seed"] == row["seed"]
            )
        )
        if row["status"] != "complete":
            raise ValueError("a formerly scored anchor endpoint became unavailable")
        differences.extend(abs(row[field] - old[field]) for field in FIELDS)
    if len(differences) != 24 or max(differences) > 1e-8:
        raise ValueError("final anchor scores do not reproduce the existing reports")
    return {
        "endpoints": 6,
        "metrics_per_endpoint": 4,
        "maximum_absolute_difference": max(differences),
    }


def report(root: Path, cost_files: list[Path]) -> dict:
    """Report medians and full ranges, with costs covering failed attempts too."""
    plan = sealed_read(root / "plan.json")
    summary = json.loads((root / "summary.json").read_text())
    if summary["plan"] != plan["artifact_sha256"]:
        raise ValueError("summary and model freeze differ")
    if any(r["status"] == "pending" for r in summary["rows"]):
        raise ValueError("finish every planned checkpoint before final reporting")
    costs = cost_index(cost_files)
    rows = []
    for r in summary["rows"]:
        c = costs[(r["cohort"], r["source_id"], r["checkpoint"])]
        rows.append(
            r
            | {
                "total_minutes": c["total_seconds"] / 60
                if not c["unknown_visit_costs"] and not c.get("unknown_pruning_cost")
                else None,
                "fit_minutes": c["fit_seconds"] / 60
                if not c["unknown_visit_costs"]
                else None,
                "proposer_tokens": c["proposer_tokens"]
                if not c["unknown_token_calls"]
                else None,
                "proposer_calls": c["proposer_calls"],
            }
        )
    groups = []
    for group in summary["groups"]:
        members = [
            r
            for r in rows
            if r["condition"] == group["condition"]
            and r["checkpoint"] == group["checkpoint"]
        ]
        group = dict(group)
        group["all_attempt_costs"] = {
            field: metrics.interval([r[field] for r in members])
            if all(r[field] is not None for r in members)
            else None
            for field in (
                "total_minutes",
                "fit_minutes",
                "proposer_tokens",
                "proposer_calls",
            )
        }
        groups.append(group)
    mechanisms = []
    for row in plan["rows"]:
        if row["family"] != "cstr":
            continue
        value = sealed_read(root / "subjects" / f"{row['key']}.json")
        if value["plan"] != plan["artifact_sha256"]:
            raise ValueError("mechanism result belongs to another plan")
        for quantity in ("feed_transport", "reaction_heat", "jacket_heat"):
            result = (
                study.pooled(value["cases"], {quantity}, 24)
                if value["status"] == "evaluated"
                else {"status": value["status"]}
            )
            mechanisms.append(
                {
                    "condition": row["condition"],
                    "seed": row["seed"],
                    "checkpoint": row["checkpoint"],
                    "quantity": quantity,
                    **result,
                }
            )
    value = {
        "protocol": "iclr-anytime-cost-response-report-1",
        "plan": plan["artifact_sha256"],
        "cost_exports": {str(p): metrics.digest(p) for p in cost_files},
        "rows": rows,
        "groups": groups,
        "mechanism_rows": mechanisms,
        "anchor_parity": parity(rows),
        "cost_scope": (
            "All attempted seeds, including no retained model; "
            "recorded search/pruning only"
        ),
        "response_scope": (
            "All planned panel cases within each scored run; "
            "unavailable runs stay in denominators"
        ),
        "test_metrics_used_for_selection": False,
    }
    atomic_json(root / "COST_RESPONSE.json", value)
    with (root / "COST_RESPONSE.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "Full performance and cost by fixed search budget.",
        "Scores: median [worst, best] over complete seeds; costs: all attempted seeds.",
        "Recorded time includes proposal, critic, fitting and final pruning; "
        "excludes queues, server startup and this evaluation.",
        "No checkpoint was selected by these test scores.",
        "",
        "| Condition | Visits/pruning | Scored | Test NMSE | Response RSE | "
        "Direction | Shape | Total min | Proposer tokens |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for group in groups:

        def fmt(value, precision=2):
            return (
                "—"
                if value is None
                else f"{value['median']:.{precision}f} "
                f"[{value['worst']:.{precision}f}, {value['best']:.{precision}f}]"
            )

        scores = " | ".join(fmt(group["statistics"][field]) for field in FIELDS)
        cost = group["all_attempt_costs"]
        lines.append(
            f"| {group['condition']} | {group['checkpoint']} | "
            f"{group['complete']}/{group['planned']} | {scores} | "
            f"{fmt(cost['total_minutes'], 0)} | "
            f"{fmt(cost['proposer_tokens'], 0)} |"
        )
    (root / "COST_RESPONSE.txt").write_text("\n".join(lines) + "\n")
    return value


def plots(root: Path, value: dict) -> None:
    """Export conventional scientific figures with every seed visible."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for column, (condition, title) in enumerate(
        (
            ("sensor_00", "CSTR-hard · 24 interventions"),
            ("acquisition_00", "T2-hard · six-case panel"),
        )
    ):
        for seed in (0, 1, 2):
            rows = [
                next(
                    r
                    for r in value["rows"]
                    if r["condition"] == condition
                    and r["seed"] == seed
                    and r["checkpoint"] == k
                )
                for k in ORDER
            ]
            for axis, metric, ylabel in (
                (axes[0, column], "test_nmse", "Held-out NMSE"),
                (axes[1, column], "response_rse", "Response RSE"),
            ):
                axis.plot(
                    [r["total_minutes"] for r in rows],
                    [r.get(metric) for r in rows],
                    "o-",
                    alpha=0.65,
                    label=f"Seed {seed}",
                )
                axis.set(xlabel="Recorded search + pruning (minutes)", ylabel=ylabel)
                axis.grid(alpha=0.2)
        axes[0, column].set_title(title)
        axes[0, column].legend(frameon=False)
    fig.suptitle("Recorded incumbents at visits 1, 5, 10, 15 and after pruning")
    fig.savefig(root / "search_budget.png", dpi=180)
    fig.savefig(root / "search_budget.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--anchor-costs", type=Path, required=True)
    parser.add_argument("--stress-costs", type=Path, required=True)
    args = parser.parse_args()
    value = report(args.root, [args.anchor_costs, args.stress_costs])
    plots(args.root, value)
    print(
        json.dumps(
            {"rows": len(value["rows"]), "anchor_parity": value["anchor_parity"]}
        )
    )


if __name__ == "__main__":
    main()
