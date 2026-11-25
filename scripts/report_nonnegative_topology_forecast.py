"""Compare matched topology forecasts before and after nonnegative training fits."""

from __future__ import annotations

import argparse
from collections import Counter
from html import escape
from os.path import relpath
from pathlib import Path
from statistics import median

import numpy as np

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import trajectory_function_fitting as fitting
from autoformalism.staged_topology import content_hash
from scripts import topology_forecast_fit as adapter
from scripts.report_revision_round import CSS, _table, number
from scripts.report_topology_forecast_match import (
    ARMS,
    LABEL,
    _equations,
    _key,
    _max_insulin_error,
    _minima,
    _nmse_by_target,
    _value,
)
from scripts.topology_forecast_eligible import qualifies

BASELINE = (
    Path(__file__).resolve().parents[1] / "output/matched-topology-forecast-20261124"
)
DEFAULT = BASELINE.with_name("matched-nonnegative-forecast-20261124")


def verify_matched_plans(old: dict, new: dict) -> None:
    """Require the exact graph cohort, formulas, initial seeds and fitting limits."""
    for name in ("cohort", "validation_release"):
        if old.get(name) != new.get(name):
            raise ValueError(f"Different {name} between fits")
    for key in ("starts", "max_evaluations", "wall_seconds"):
        if old["fit_budget"].get(key) != new["fit_budget"].get(key):
            raise ValueError(f"Unequal fitting budgets for {key}")
    fields = ("graph_index", "rank", "fit_key", "seed", "formula", "prefit")
    if len(old["slots"]) != len(new["slots"]) or any(
        a.get(k) != b.get(k)
        for a, b in zip(old["slots"], new["slots"], strict=True)
        for k in fields
    ):
        raise ValueError("Formula slots differ between fits")


def _load(root: Path, plan: dict) -> dict:
    """Load verified training freezes before reading their development evaluations."""
    frozen = sealed_read(root / "frozen.json")
    rule = sealed_read(root / "eligibility-rule.json")
    admissible = sealed_read(root / "eligibility-freeze.json")
    plan_hash = plan["artifact_sha256"]
    if (
        any(r["plan"] != plan_hash for r in (frozen, rule, admissible))
        or rule["artifact_sha256"] != admissible["rule_sha256"]
        or frozen["fit_receipts"] != admissible["fit_receipts"]
    ):
        raise ValueError(
            "Training selections or eligibility rule refer to different fits"
        )
    receipts = {}
    all_graphs = {_key(row): row["graph"] for row in plan["cohort"]["rows"]}
    for slot in plan["slots"]:
        key = slot["fit_key"]
        receipt = sealed_read(root / "fits" / f"{key}.json")
        if (
            key in receipts
            or receipt["plan"] != plan_hash
            or receipt["slot_sha256"] != content_hash(slot)
            or receipt["formula"] != slot["formula"]
            or receipt["artifact_sha256"] != frozen["fit_receipts"][key]
        ):
            raise ValueError("Fit receipt disagrees with its frozen slot")
        receipts[key] = receipt
    if set(receipts) != set(frozen["fit_receipts"]):
        raise ValueError("Fit slots and freeze differ")
    dev = sealed_read(root / "development.json")
    approved = sealed_read(root / "eligibility-development.json")
    if dev["freeze"] != frozen["artifact_sha256"] or (
        approved["freeze"] != admissible["artifact_sha256"]
    ):
        raise ValueError("Forecasts refer to another training freeze")
    primary = _match_selection(frozen["selections"], dev["rows"])
    if set(primary) != set(all_graphs) or len(primary) != len(plan["cohort"]["rows"]):
        raise ValueError("Scored graph list differs from the cohort")
    chosen = [g["winner"] for g in admissible["groups"] if g["winner"] is not None]
    eligible = _match_selection(chosen, approved["rows"])
    return {
        "frozen": frozen,
        "rule": rule["rule"],
        "graphs": all_graphs,
        "groups": {(g["inventory"], g["arm"]): g for g in admissible["groups"]},
        "receipts": receipts,
        "primary": primary,
        "eligible": eligible,
        "development_sha256": dev["artifact_sha256"],
        "eligible_development_sha256": approved["artifact_sha256"],
    }


def _match_selection(selected: list[dict], scored: list[dict]) -> dict:
    by_key = {_key(r): r for r in scored}
    if len(by_key) != len(scored) or len(scored) != len(selected):
        raise ValueError("Training and development selection sizes differ")
    for row in selected:
        evaluated = by_key.get(_key(row))
        if evaluated is None or any(evaluated.get(k) != v for k, v in row.items()):
            raise ValueError("Development model differs from its training selection")
    return by_key


def _median(values) -> float | None:
    scores = [v for v in values if v is not None]
    return median(scores) if scores else None


def _winner(data: dict, group: dict, inventory: str, arm: str, eligible: bool):
    if eligible:
        return (
            data["eligible"][_key(group["winner"])]
            if group["winner"] is not None
            else None
        )
    key = data["frozen"]["winners"][inventory][arm]
    return data["primary"][inventory, arm, key] if key is not None else None


def _peaks(curve: dict) -> dict:
    """Report the earliest time attaining each sampled insulin peak."""
    time = curve["time"]
    result = {"event": curve["trajectory_id"]}
    for side in ("observed", "predicted"):
        values = curve[side]["I"]
        if not time or len(values) != len(time):
            raise ValueError("Insulin peak requires aligned observations and times")
        i = max(range(len(time)), key=values.__getitem__)
        result[side] = {"value": values[i], "time": time[i]}
    return result


def _model(row: dict | None) -> dict | None:
    if row is None:
        return None
    score = row["score"] or {}
    valid = _value(row) is not None
    return {
        "fit_key": row["fit_key"],
        "graph_id": row["graph_id"],
        "train_nmse": row["train_nmse"],
        "development_nmse": _value(row),
        "development_status": score.get("status", "unavailable"),
        "negative_samples": score.get("negative_samples", {}) if valid else None,
        "minimum_prediction": score.get("minimum_prediction", {}) if valid else None,
        "per_target_nmse": score.get("per_target_nmse", {}) if valid else {},
        "insulin_peaks": [_peaks(c) for c in score.get("curves", [])] if valid else [],
    }


def _counts(plan: dict, data: dict, inventory: str, arm: str):
    statuses = Counter()
    evaluations = starts = feasible = 0
    for slot in plan["slots"]:
        row = plan["cohort"]["rows"][slot["graph_index"]]
        if (row["inventory"], row["arm"]) != (inventory, arm):
            continue
        result = data["receipts"][slot["fit_key"]]["result"]
        statuses[result["status"]] += 1
        evaluations += result.get("evaluations", 0)
        starts += result.get("starts_attempted", 0)
        feasible += qualifies(slot["prefit"], result)
    return dict(statuses), evaluations, starts, feasible


def compare(old_plan: dict, old: dict, new_plan: dict, new: dict) -> list[dict]:
    """Summarize matched scores and keep every failed graph in the denominator."""
    verify_matched_plans(old_plan, new_plan)
    if old["rule"] != new["rule"] or old["groups"].keys() != new["groups"].keys():
        raise ValueError("Eligibility criteria or inventory groups differ")
    cells = []
    for inv, arm in sorted(old["groups"]):
        keys = sorted(k for k in old["primary"] if k[:2] == (inv, arm))
        common = [
            k
            for k in keys
            if all(_value(d["primary"][k]) is not None for d in (old, new))
        ]
        cell = {"inventory": inv, "arm": arm, "common_scored_graphs": len(common)}
        for label, plan, d in (
            ("original", old_plan, old),
            ("constrained", new_plan, new),
        ):
            group = d["groups"][inv, arm]
            rows = [d["primary"][k] for k in keys]
            scored = [r for r in rows if _value(r) is not None]
            qualified_rows = [
                r
                for r in rows
                if r.get("fit_key") is not None
                and qualifies(r["prefit"], d["receipts"][r["fit_key"]]["result"])
            ]
            statuses, calls, starts, eligible_slots = _counts(plan, d, inv, arm)
            if (
                len(keys) != group["advanced_graphs"]
                or eligible_slots != group["eligible_slots"]
            ):
                raise ValueError("Registered qualification counts disagree with fits")
            cell[label] = {
                "graphs": len(keys),
                "slots": group["formula_slots"],
                "eligible_graphs": group["eligible_graphs"],
                "eligible_slots": eligible_slots,
                "numerically_scored": len(scored),
                "negative_dev_graph_finalists": sum(
                    any(r["score"]["negative_samples"].values()) for r in scored
                ),
                "median_scored": _median(_value(r) for r in scored),
                "median_common": _median(_value(d["primary"][k]) for k in common),
                "qualified_finalists_scored": sum(
                    _value(r) is not None for r in qualified_rows
                ),
                "median_qualified_finalists": _median(
                    _value(r) for r in qualified_rows
                ),
                "nonnegative_development_qualified_finalists": sum(
                    _value(r) is not None
                    and not any(r["score"]["negative_samples"].values())
                    for r in qualified_rows
                ),
                "statuses": statuses,
                "evaluations": calls,
                "starts": starts,
                "best_train": _model(_winner(d, group, inv, arm, False)),
                "best_eligible_train": _model(_winner(d, group, inv, arm, True)),
            }
        cells.append(cell)
    return cells


def _replay_minima(row: dict, formula: dict, fit: dict) -> dict:
    """Recompute original vector minima at training samples on both step sizes."""
    case = adapter._training(row["case"], row["dataset"])
    events = fitting._rows(case["evidence"]["trajectories"], "train")
    targets = sorted(events[0].targets)
    minima = {}
    for steps in (2, 4):
        model = fitting._Rollout(row["graph"], formula, targets, steps)
        params = [fit["parameters"][k] for k in model.parameters]
        predicted = model.run([model.record(e) for e in events], params)
        mins = np.min(np.concatenate([y for y, _ in predicted], axis=1), axis=1)
        if not np.isfinite(mins).all():
            raise ValueError("Nonfinite original control replay")
        minima[str(steps)] = dict(zip(targets, map(float, mins), strict=True))
    if any(
        not np.isclose(v, fit["minimum_prediction"][k], rtol=1e-7, atol=1e-7)
        for k, v in minima["4"].items()
    ):
        raise ValueError("Original training minima failed to reproduce")
    return minima


def optimizer_control(plan: dict, original: dict, constrained: dict) -> list[dict]:
    """Compare cold refits with previously found, fixed feasible coefficients."""
    control = []
    for s in plan["slots"]:
        key = s["fit_key"]
        before = original["receipts"][key]["result"]
        # Isolate vector feasibility irrespective of any prefit equation check.
        if not qualifies({"status": "eligible"}, before):
            continue
        after = constrained["receipts"][key]["result"]
        row = plan["cohort"]["rows"][s["graph_index"]]
        mins = _replay_minima(row, s["formula"], before)
        feasible = all(min(bounds.values()) >= -1e-10 for bounds in mins.values())
        old_loss = before["training_nmse"]
        tol = max(1e-8, 1e-6 * abs(old_loss))
        new_loss = after.get("training_nmse")
        reached = (
            (qualifies({"status": "eligible"}, after) and new_loss <= old_loss + tol)
            if feasible
            else None
        )
        control.append(
            {
                "fit_key": key,
                "inventory": row["inventory"],
                "arm": row["arm"],
                "original_train_nmse": old_loss,
                "constrained_train_nmse": new_loss,
                "original_training_minima_by_substeps": mins,
                "both_grids_nonnegative": feasible,
                "reached_original": reached,
            }
        )
    return control


def _peak_figure(
    root: Path, inventory: str, models: list[tuple[str, str, dict]]
) -> str:
    """Plot all insulin events with old and new eligible training winners."""
    if not models:
        return ""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    curves = [
        {c["trajectory_id"]: c for c in row["score"]["curves"]} for _, _, row in models
    ]
    events = sorted(curves[0])
    if not events:
        return ""
    if any(set(c) != set(events) for c in curves):
        raise ValueError("Development schedules differ")
    fig = Figure(figsize=(10, 2.7 * ((len(events) + 1) // 2) + 0.9))
    canvas = FigureCanvasAgg(fig)
    axes = list(fig.subplots((len(events) + 1) // 2, 2, squeeze=False).flat)
    for event, ax in zip(events, axes, strict=False):
        ref = curves[0][event]
        ax.plot(
            ref["time"], ref["observed"]["I"], color="#20252a", lw=1.9, label="Observed"
        )
        for (method, arm, _), series in zip(models, curves, strict=True):
            curve = series[event]
            if (
                curve["time"] != ref["time"]
                or curve["observed"]["I"] != ref["observed"]["I"]
            ):
                raise ValueError("Recorded insulin trajectories differ")
            ax.plot(
                curve["time"],
                curve["predicted"]["I"],
                color="#00796b" if arm == "algorithm" else "#ad4e15",
                ls="--" if method == "original" else "-",
                lw=1.7,
                label=f"{LABEL[arm]} · {method}",
            )
        ax.set(title=event, xlabel="Minutes", ylabel="Insulin (pmol/L)")
        ax.axhline(0, color="0.65", lw=0.7)
        ax.grid(alpha=0.18)
    for ax in axes[len(events) :]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncols=min(3, len(models) + 1),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    filename = f"insulin-{inventory}.png"
    canvas.print_png(root / filename)
    return (
        f'<figure><img width="100%" alt="Insulin forecasts for {escape(inventory)}" '
        f'src="{escape(filename, quote=True)}"></figure>'
    )


def render(old_root: Path, root: Path) -> Path:
    """Verify the two runs and render their paired comparison after scoring."""
    plans = [sealed_read(r / "plan.json") for r in (old_root, root)]
    verify_matched_plans(*plans)
    old, new = (_load(r, p) for r, p in zip((old_root, root), plans, strict=True))
    cells = compare(plans[0], old, plans[1], new)
    controls = optimizer_control(plans[0], old, new)
    sealed_write(
        root / "comparison-summary.json",
        {
            "original_plan": plans[0]["artifact_sha256"],
            "constrained_plan": plans[1]["artifact_sha256"],
            "original_development": old["development_sha256"],
            "constrained_development": new["development_sha256"],
            "original_eligible_development": old["eligible_development_sha256"],
            "constrained_eligible_development": new["eligible_development_sha256"],
            "groups": cells,
            "optimizer_controls": controls,
        },
    )
    qualifications, portfolios, numerics, details, fit_costs = [], [], [], [], []
    for cell in cells:
        inv, arm = cell["inventory"], cell["arm"]
        for method, d in (("original", old), ("constrained", new)):
            m = cell[method]
            prefix = [inv, LABEL[arm], method]
            best = m["best_eligible_train"]
            row = _winner(d, d["groups"][inv, arm], inv, arm, True)
            qualifications.append(
                [
                    *prefix,
                    f"{m['eligible_graphs']}/{m['graphs']}",
                    f"{m['eligible_slots']}/{m['slots']}",
                    number(best["development_nmse"]) if best else "—",
                    _minima(row["score"]) if row else "—",
                ]
            )
            portfolios.append(
                [
                    *prefix,
                    best["fit_key"] if best else "—",
                    number(best["train_nmse"]) if best else "—",
                    _nmse_by_target(row) if row else "—",
                    _max_insulin_error(row) if row else "—",
                ]
            )
            cost = ", ".join(f"{k} ({v})" for k, v in m["statuses"].items())
            fit_costs.append([*prefix, cost, m["starts"], m["evaluations"]])
            med_eligible = (
                f"{number(m['median_qualified_finalists'])} "
                f"({m['qualified_finalists_scored']}/{m['eligible_graphs']})"
            )
            numerical = m["best_train"]
            numerics.append(
                [
                    *prefix,
                    number(numerical["development_nmse"]) if numerical else "—",
                    f"{number(m['median_scored'])} "
                    f"({m['numerically_scored']}/{m['graphs']})",
                    f"{number(m['median_common'])} "
                    f"({cell['common_scored_graphs']}/{m['graphs']})",
                    med_eligible,
                    f"{m['nonnegative_development_qualified_finalists']}/"
                    f"{m['qualified_finalists_scored']}",
                    f"{m['negative_dev_graph_finalists']}/{m['numerically_scored']}",
                ]
            )
    for inv in sorted({c["inventory"] for c in cells}):
        curves, peak_rows, equation_rows = [], [], []
        for method, data in (("original", old), ("constrained", new)):
            for arm in ARMS:
                winner = _winner(data, data["groups"][inv, arm], inv, arm, True)
                if not winner or _value(winner) is None:
                    continue
                curves.append((method, arm, winner))
                for curve in winner["score"]["curves"]:
                    peak = _peaks(curve)
                    meas, pred = peak["observed"], peak["predicted"]
                    peak_rows.append(
                        [
                            LABEL[arm],
                            method,
                            peak["event"],
                            number(meas["value"]),
                            number(meas["time"]),
                            number(pred["value"]),
                            number(pred["time"]),
                        ]
                    )
                eq = _equations(
                    winner,
                    data["graphs"][_key(winner)],
                    data["receipts"][winner["fit_key"]],
                )
                if method == "original":
                    target = relpath(old_root / "fits", root)
                    eq = eq.replace(
                        'href="fits/', f'href="{escape(target, quote=True)}/'
                    )
                equation_rows.append(f"<p>{escape(method.capitalize())}</p>" + eq)
        details.extend(
            [
                f"<h3>{escape(inv)}</h3>",
                _peak_figure(root, inv, curves),
                _table(
                    [
                        "Generator",
                        "Fit",
                        "Event",
                        "Observed peak (pmol/L)",
                        "Minute",
                        "Predicted peak (pmol/L)",
                        "Minute",
                    ],
                    peak_rows,
                ),
                *equation_rows,
            ]
        )
    overview = []
    for inv in sorted({cell["inventory"] for cell in cells}):
        values = {cell["arm"]: cell for cell in cells if cell["inventory"] == inv}
        row = [inv]
        for arm in ARMS:
            old_cell, new_cell = values[arm]["original"], values[arm]["constrained"]
            row.append(
                f"{old_cell['eligible_graphs']} → {new_cell['eligible_graphs']}"
                f" of {new_cell['graphs']}"
            )
        for arm in ARMS:
            winner = values[arm]["constrained"]["best_eligible_train"]
            row.append(number(winner["development_nmse"]) if winner else "—")
        overview.append(row)
    totals = {}
    for arm in ARMS:
        arm_cells = [c for c in cells if c["arm"] == arm]
        totals[arm] = [
            sum(c[k]["eligible_graphs"] for c in arm_cells)
            for k in ("original", "constrained")
        ]
    original_algo, new_algo = totals["algorithm"]
    original_repair, new_repair = totals["20b_repair"]
    checked_controls = sum(c["both_grids_nonnegative"] for c in controls)
    reached = sum(c["reached_original"] is True for c in controls)
    html = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Nonnegative fitting on matched graphs</title>",
        f"<style>{CSS}</style>",
        "<main><h1>Nonnegative fitting on matched graphs</h1>",
        f"<p><strong>Algorithm-generated graphs yielding a qualifying fit increased "
        f"from {original_algo} to {new_algo}. The count for 20B plus repair changed "
        f"from {original_repair} to {new_repair}.</strong></p>",
        "<p>The same variable inventories, graph structures, equation forms and "
        "parameter initialization seeds enter both fits. An eligible graph has at "
        "least one fitted formula with no demonstrated prefit contradiction, a "
        "verified training forecast, and zero negative samples across the observed "
        "glucose, insulin and utilization targets on training.</p>",
        _table(
            [
                "Inventory",
                "Algorithm eligible graphs, original → constrained",
                "20B + repair eligible graphs, original → constrained",
                "Algorithm eligible winner NMSE on development, constrained",
                "20B + repair eligible winner NMSE on development, constrained",
            ],
            overview,
        ),
        "<h2>Eligible graphs and forecasts</h2><p>Each inventory and generator "
        "advances eight graphs with two formulas per graph. Each forecast in this "
        "table uses the eligible formula with lowest training NMSE in that group. "
        "A missing winner indicates an empty training-eligible pool. NMSE averages "
        "squared forecast errors divided by training variance for each target. "
        "The development evaluation uses four events.</p>",
        _table(
            [
                "Inventory",
                "Generator",
                "Fit",
                "Eligible graphs",
                "Eligible formulas",
                "Development NMSE",
                "Negative minima on development",
            ],
            qualifications,
        ),
        "<h2>Forecast error across graph finalists</h2><p>Each graph finalist is "
        "the numerically verified formula with lowest training error. The first "
        "column of scores uses the best training finalist in the inventory and arm. "
        "Median NMSE includes the number of scored graphs over the number advanced. "
        "The common subset uses only graphs scored under both fitting methods. "
        "The qualifying-finalist median includes qualifying training finalists "
        "with verified development forecasts over all training-eligible graphs. "
        "A graph can have a different formula as its qualifying choice. "
        "The last column counts forecasts with a negative development sample.</p>",
        _table(
            [
                "Inventory",
                "Generator",
                "Fit",
                "Best training finalist NMSE on development",
                "Median NMSE (scored/advanced)",
                "Common-graph median (common/advanced)",
                "Qualifying-finalist median (scored/eligible)",
                "Qualifying finalists nonnegative on development",
                "All scored finalists negative on development",
            ],
            numerics,
        ),
        "<h2>Insulin peaks and fit details</h2>",
        _table(
            [
                "Inventory",
                "Generator",
                "Fit",
                "Eligible winner",
                "Training NMSE",
                "Development NMSE by target",
                "Maximum |insulin error| (pmol/L)",
            ],
            portfolios,
        ),
        "<p>Plots include each scored eligible winner on every development event. "
        "A peak is the earliest sampled time attaining the maximum within one event. "
        "Times are measured in minutes. All four development events have zero "
        "administered insulin.</p>",
        *details,
        "<h2>Optimizer control on previously feasible coefficients</h2>",
        f"<p><strong>{reached} of {checked_controls} cold refits reached "
        "their earlier, independently verified feasible training NMSE.</strong></p>",
        "<p>Each row represents the same equation fitted from its original starts. "
        "We independently re-evaluated the original fitted coefficients at all "
        "training samples with two and four integration substeps. For rows passing "
        "both grids these coefficients exhibit a feasible point at the original "
        "loss. A cold refit reaches that loss if its verified training NMSE is at "
        "most the original value plus max(1e-8, 1e-6 * original). The control uses "
        "archived fits for diagnosis after selection.</p>",
        _table(
            [
                "Inventory",
                "Generator",
                "Slot",
                "Both grids nonnegative on replay",
                "Original training NMSE",
                "Constrained training NMSE",
                "Reached original training NMSE",
            ],
            [
                [
                    c["inventory"],
                    LABEL[c["arm"]],
                    c["fit_key"],
                    "yes" if c["both_grids_nonnegative"] else "no",
                    number(c["original_train_nmse"]),
                    number(c["constrained_train_nmse"]),
                    {True: "yes", False: "no", None: "—"}[c["reached_original"]],
                ]
                for c in controls
            ],
        ),
        "<h2>Computational work</h2>",
        "<p>The original least-squares fit and the constrained SLSQP fit receive "
        "identical starts and limits. In the constrained fit each two-substep or "
        "four-substep full training evaluation consumes one call. The solver "
        "proposes trial coefficients, enforces inequalities on sampled predictions "
        "and retains coefficients only after checking both integration grids. "
        "This establishes nonnegativity at the recorded training times.</p>",
        _table(
            [
                "Inventory",
                "Generator",
                "Fit",
                "Slot outcomes",
                "Starts attempted",
                "Reported evaluations",
            ],
            fit_costs,
        ),
        '<p><a href="comparison-summary.json">Paired results and frozen identities'
        "</a></p></main></html>",
    ]
    path = root / "REPORT.html"
    path.write_text("\n".join(html), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=BASELINE)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(render(args.original.resolve(), args.root.resolve()))
