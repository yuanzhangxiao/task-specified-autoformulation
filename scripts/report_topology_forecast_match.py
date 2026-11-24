"""Render the frozen, inventory-matched topology-to-forecast comparison."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from html import escape
from itertools import product
from math import inf, isfinite
from pathlib import Path
from statistics import median

from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.staged_topology import content_hash
from scripts.report_revision_round import CSS, _table, number

ARMS = ("algorithm", "20b_repair")
LABEL = {"algorithm": "Algorithm", "20b_repair": "20B + repair"}
UNITS = {"Gp": "mg/kg", "I": "pmol/L", "U": "mg/kg/min", "h_down": "m"}
KEYS = ("inventory", "arm", "graph_id")


def _key(row: dict) -> tuple[str, str, str]:
    return tuple(row[k] for k in KEYS)


def _value(row: dict) -> float | None:
    """Count only finite, resolution-verified full-event development forecasts."""
    score = row.get("score") or {}
    val = score.get("pooled_nmse")
    if score.get("status") == "scored" and val is not None and isfinite(val):
        return val
    return None


def _worst_rank(row: dict) -> float:
    value = _value(row)
    return inf if value is None else value


def cross_pairs(left: list[dict], right: list[dict]) -> tuple[int, int, int]:
    """Wins, ties and losses when comparing all cross-arm pairs in one inventory."""
    wins = ties = losses = 0
    for a, b in product(left, right):
        x, y = _worst_rank(a), _worst_rank(b)
        wins += x < y
        ties += x == y
        losses += x > y
    return wins, ties, losses


def _minima(score: dict | None) -> str:
    if not score or score.get("status") != "scored":
        return "—"
    bad = {
        k: score["minimum_prediction"][k]
        for k, n in score.get("negative_samples", {}).items()
        if n and k in score.get("minimum_prediction", {})
    }
    return ", ".join(
        f"{k} {number(v)} {UNITS.get(k, '')}".strip() for k, v in bad.items()
    ) or ("none" if "negative_samples" in score else "unavailable")


def _nmse_by_target(row: dict) -> str:
    score = row.get("score") or {}
    return (
        ", ".join(
            f"{k} {number(v)}" for k, v in score.get("per_target_nmse", {}).items()
        )
        or "—"
    )


def _max_insulin_error(row: dict) -> str:
    """Maximum error across every sampled time in every scored development event."""
    if _value(row) is None:
        return "—"
    differences = (
        (abs(observed - predicted), event["trajectory_id"], t, observed, predicted)
        for event in row["score"].get("curves", [])
        if "I" in event["observed"]
        for t, observed, predicted in zip(
            event["time"], event["observed"]["I"], event["predicted"]["I"], strict=True
        )
    )
    worst = max(differences, default=None)
    if worst is None:
        return "—"
    error, event_id, time, observed, predicted = worst
    return (
        f"{number(error)} ({event_id}, minute {number(time)}, "
        f"observed {number(observed)}, predicted {number(predicted)})"
    )


def _winner_id(value) -> str | None:
    return value.get("graph_id") if isinstance(value, dict) else value


def _equations(row: dict, graph: dict, receipt: dict) -> str:
    """Display the selected equation, initial values and fitted shared coefficients."""
    key = row.get("fit_key")
    if not key:
        return ""
    formula = receipt.get("formula") or {}
    fit = receipt.get("result") or {}
    lines = []
    for name, rhs in formula.get("equations", {}).items():
        state = graph["variable_types"][name] == "differential"
        lines.append(f"{'d(' + name + ')/dt' if state else name} = {rhs}")
    for name, rule in graph.get("initializers", {}).items():
        init = formula.get("initializers", {}).get(name)
        if init is None and rule.get("mode") == "observed_initial":
            init = f"observed {rule['channel']}(0)"
        lines.append(f"{name}(0) = {init if init is not None else json.dumps(rule)}")
    params = fit.get("parameters") or {}
    coefficients = ", ".join(f"{k}={number(v)}" for k, v in sorted(params.items()))
    return (
        f"<details><summary>{escape(LABEL[row['arm']])} · "
        f"{escape(row['graph_id'][:12])} · formula {row['rank']}</summary>"
        f"<pre>{escape(chr(10).join(lines))}</pre>"
        f"<p>Fitted coefficients · {escape(coefficients or 'none')}</p>"
        f'<a href="fits/{escape(str(key), quote=True)}.json">Fit receipt</a></details>'
    )


def _example_plot(root: Path, panel: str, rows: list[dict]) -> str:
    """Plot both training-selected winners on the same first development event."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    use = [r for r in rows if _value(r) is not None and r["score"].get("curves")]
    if not use:
        return ""
    events = [{e["trajectory_id"]: e for e in r["score"]["curves"]} for r in use]
    common = set.intersection(*(set(xs) for xs in events))
    if not common:
        return ""
    event = min(common)
    ref = events[0][event]
    names = sorted(ref["observed"])
    fig = Figure(figsize=(8.0, 2.25 * len(names)), layout="constrained")
    canvas = FigureCanvasAgg(fig)
    axes = fig.subplots(len(names), 1, squeeze=False, sharex=True)
    colors = {"algorithm": "#168274", "20b_repair": "#b9511e"}
    for name, (ax,) in zip(names, axes, strict=True):
        ax.plot(
            ref["time"],
            ref["observed"][name],
            color="#253440",
            ls="--",
            lw=2,
            label="Observed",
        )
        for r, lookup in zip(use, events, strict=True):
            e = lookup[event]
            if e["time"] != ref["time"] or e["observed"][name] != ref["observed"][name]:
                raise ValueError(
                    "Forecasts disagree on public event times or observations"
                )
            ax.plot(
                e["time"],
                e["predicted"][name],
                lw=1.6,
                color=colors[r["arm"]],
                label=LABEL[r["arm"]],
            )
        ax.set_ylabel(f"{name} ({UNITS[name]})" if name in UNITS else name)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=9, ncols=len(use) + 1)
    axes[-1, 0].set_xlabel("Time (minutes)" if "I" in names else "Time")
    filename = f"portfolio-forecast-{panel}.png"
    canvas.print_png(root / filename)
    return (
        f'<figure><img src="{filename}" alt="Observed and predicted trajectories '
        f'for {escape(event, quote=True)}" style="width:100%"><figcaption>'
        f"{escape(event)} · First development event by ID.</figcaption></figure>"
    )


def render(root: Path) -> Path:
    """Render after verifying that each score belongs to a frozen graph and formula."""
    plan = sealed_read(root / "plan.json")
    frozen = sealed_read(root / "frozen.json")
    dev = sealed_read(root / "development.json")
    if (
        frozen["plan"] != plan["artifact_sha256"]
        or dev["freeze"] != frozen["artifact_sha256"]
    ):
        raise ValueError("Development and training freeze do not match")
    cohort_rows = plan["cohort"]["rows"]
    graphs = {_key(r): r["graph"] for r in cohort_rows}
    selected = {_key(r): r for r in frozen["selections"]}
    scored = {_key(r): r for r in dev["rows"]}
    if (
        len(graphs) != len(cohort_rows)
        or len(selected) != len(frozen["selections"])
        or len(scored) != len(dev["rows"])
        or graphs.keys() != selected.keys()
        or selected.keys() != scored.keys()
    ):
        raise ValueError("Cohort slots and development rows differ")
    groups = defaultdict(lambda: defaultdict(list))
    # Validate all attempts so the costs include failed and unselected formulas.
    receipts = {}
    attempts = defaultdict(list)
    for slot in plan["slots"]:
        key = slot["fit_key"]
        record = sealed_read(root / "fits" / f"{key}.json")
        if (
            record["plan"] != plan["artifact_sha256"]
            or record["slot_sha256"] != content_hash(slot)
            or record["artifact_sha256"] != frozen["fit_receipts"][key]
        ):
            raise ValueError("Fit receipt differs from frozen plan or result")
        graph = cohort_rows[slot["graph_index"]]
        attempts[graph["inventory"], graph["arm"]].append((slot, record["result"]))
        receipts[key] = record
    for key, r in scored.items():
        if any(
            r.get(k) != selected[key].get(k) for k in ("rank", "fit_key", "train_nmse")
        ):
            raise ValueError("Development graph finalist differs from training freeze")
        groups[r["inventory"]][r["arm"]].append(r)
    winners = frozen["winners"]
    early = frozen.get("nonnegative_winners", {})
    early_rows = {_key(r): r for r in dev.get("nonnegative_rows", [])}
    rule = sealed_read(root / "eligibility-rule.json")
    admissible = sealed_read(root / "eligibility-freeze.json")
    admitted_dev = sealed_read(root / "eligibility-development.json")
    if (
        admissible["plan"] != plan["artifact_sha256"]
        or rule["plan"] != plan["artifact_sha256"]
        or admissible["rule_sha256"] != rule["artifact_sha256"]
        or admissible["fit_receipts"] != frozen["fit_receipts"]
        or admitted_dev["freeze"] != admissible["artifact_sha256"]
    ):
        raise ValueError("Eligibility cohort or training selection differs")
    include = {(g["inventory"], g["arm"]): g for g in admissible["groups"]}
    admitted_rows = {_key(r): r for r in admitted_dev["rows"]}
    expected = {_key(g["winner"]) for g in include.values() if g["winner"]}
    if set(admitted_rows) != expected or len(admitted_dev["rows"]) != len(expected):
        raise ValueError("Eligibility forecast list differs from sealed winners")

    graph_rows, win_rows, summaries, chances, details, costs = [], [], [], [], [], []
    qualified_rows = []
    checks = Counter()
    for panel, (inventory, arms) in enumerate(sorted(groups.items())):
        a, b = [arms[k] for k in ARMS]
        if len(a) != len(b):
            raise ValueError("Generator arms differ in graph count for one inventory")
        wins, ties, losses = cross_pairs(a, b)
        chances.append([inventory, wins, ties, losses, wins + ties + losses])
        plot_rows, equation_rows, qualified_plots, qualified_equations = [], [], [], []
        for arm in ARMS:
            rows = arms[arm]
            jobs = attempts[inventory, arm]
            group = include[inventory, arm]
            if group["advanced_graphs"] != len(rows) or group["formula_slots"] != len(
                jobs
            ):
                raise ValueError("Eligibility summary disagrees on denominator")
            qualification = [
                inventory,
                LABEL[arm],
                f"{group['eligible_graphs']}/{len(rows)}",
                f"{group['eligible_slots']}/{len(jobs)}",
            ]
            if group["winner"]:
                chosen = group["winner"]
                key = _key(chosen)
                r = admitted_rows[key]
                if any(r[k] != v for k, v in chosen.items()):
                    raise ValueError("Scored eligible formula differs from frozen one")
                qualified_rows.append(
                    [
                        *qualification,
                        r["graph_id"][:12],
                        number(r["train_nmse"]),
                        number(_value(r)),
                        _nmse_by_target(r),
                        _minima(r.get("score")),
                        _max_insulin_error(r),
                    ]
                )
                qualified_plots.append(r)
                qualified_equations.append(
                    _equations(r, graphs[key], receipts[r["fit_key"]])
                )
            else:
                qualified_rows.append(qualification + ["—"] * 6)
            admitted_fits = {
                slot["graph_index"]
                for slot, result in jobs
                if (slot.get("prefit") or {}).get("status") == "eligible"
                and result["status"] == "fitted"
            }
            costs.append(
                [
                    inventory,
                    LABEL[arm],
                    len(jobs),
                    len(admitted_fits),
                    dict(
                        Counter(
                            (s.get("prefit") or {}).get("status", "unavailable")
                            for s, _ in jobs
                        )
                    ),
                    dict(Counter(result["status"] for _, result in jobs)),
                    sum(result.get("starts_attempted", 0) for _, result in jobs),
                    sum(result.get("evaluations", 0) for _, result in jobs),
                    sum(result.get("failed_rollouts", 0) for _, result in jobs),
                    round(sum(result.get("seconds", 0) for _, result in jobs)),
                ]
            )
            finite = [r for r in rows if _value(r) is not None]
            values = [_value(r) for r in finite]
            all_values = [_worst_rank(r) for r in rows]
            complete = sum(r.get("rank") is not None for r in rows)
            negatives = sum(
                _minima(r["score"]) not in ("none", "unavailable") for r in finite
            )
            med = median(all_values)
            summaries.append(
                [
                    inventory,
                    LABEL[arm],
                    str(len(rows)),
                    str(complete),
                    str(len(finite)),
                    number(median(values)) if values else "—",
                    number(med) if isfinite(med) else "failed",
                    str(negatives),
                ]
            )
            for r in rows:
                status = (r.get("score") or {}).get("status", r["fit_status"])
                checks[status] += 1
                graph_rows.append(
                    [
                        inventory,
                        LABEL[arm],
                        r["graph_id"][:12],
                        "—" if r["rank"] is None else str(r["rank"]),
                        str(status),
                        (r.get("prefit") or {}).get("status", "—"),
                        number(r["train_nmse"]),
                        number(_value(r)),
                        _nmse_by_target(r),
                        _minima(r.get("score")),
                    ]
                )
            for rule, key, lookup in (
                ("Lowest training error", winners[inventory][arm], scored),
                (
                    "Nonnegative training forecasts",
                    early.get(inventory, {}).get(arm),
                    early_rows,
                ),
            ):
                gid = _winner_id(key)
                if not gid:
                    win_rows.append([inventory, LABEL[arm], rule] + ["—"] * 7)
                    continue
                row = lookup[inventory, arm, gid]
                if isinstance(key, dict) and row["fit_key"] != key["fit_key"]:
                    raise ValueError("Portfolio choice differs from the frozen formula")
                win_rows.append(
                    [
                        inventory,
                        LABEL[arm],
                        rule,
                        gid[:12],
                        (row.get("prefit") or {}).get("status", "—"),
                        "yes" if row["physical_train"] else "no",
                        number(row["train_nmse"]),
                        number(_value(row)),
                        _nmse_by_target(row),
                        _minima(row.get("score")),
                    ]
                )
                if rule == "Lowest training error":
                    plot_rows.append(row)
                    equation_rows.append(
                        _equations(
                            row, graphs[inventory, arm, gid], receipts[row["fit_key"]]
                        )
                    )
        details.append(
            f"<h3>{escape(inventory)}</h3>"
            + "<h4>Lowest training error</h4>"
            + _example_plot(root, str(panel), plot_rows)
            + "".join(equation_rows)
            + ("<h4>Qualified on training</h4>" if qualified_plots else "")
            + _example_plot(root, f"qualified-{panel}", qualified_plots)
            + "".join(qualified_equations)
        )

    pool_rows = []
    for inventory, counts in sorted(plan["cohort"]["inventory_counts"].items()):
        for arm in ARMS:
            slots = counts["arms"][arm]
            pool_rows.append(
                [
                    inventory,
                    LABEL[arm],
                    slots["admitted_occurrences"],
                    slots["distinct_admitted"],
                    slots["matching_inventory"],
                    counts["selected_per_arm"],
                    len(counts["selected_in_both_arms"]),
                ]
            )
    structure_rows = []
    for arm in ARMS:
        sources = [
            r["graph"]["sources"]
            for r in cohort_rows
            if r["arm"] == arm
            and {"Gp", "I"} <= r["graph"].get("sources", {}).keys()
            and all(
                r["graph"]["variable_types"][v] == "differential" for v in ("Gp", "I")
            )
        ]
        if sources:
            edges = [sum(len(s) for s in graph.values()) for graph in sources]
            proof_counts = []
            for proof in ("pointwise_checks", "ordered_checks"):
                counts = Counter()
                for slot in plan["slots"]:
                    graph_row = cohort_rows[slot["graph_index"]]
                    if (
                        graph_row["arm"] != arm
                        or "I" not in graph_row["graph"]["sources"]
                    ):
                        continue
                    checks_for_target = (slot.get("prefit") or {}).get(proof, {})
                    counts[
                        (checks_for_target.get("I") or {}).get("status", "unavailable")
                    ] += 1
                proof_counts.append(dict(counts))
            structure_rows.append(
                [
                    LABEL[arm],
                    len(sources),
                    sum("insulin_pmol_per_kg_min" in graph["I"] for graph in sources),
                    sum({"EGP", "E", "U"} <= set(graph["Gp"]) for graph in sources),
                    f"{median(edges):g} ({min(edges)} to {max(edges)})",
                    *proof_counts,
                ]
            )
    structure = ""
    if structure_rows:
        structure = (
            "<h2>How did the selected T2 graphs differ?</h2>"
            "<p>A direct source appears as an argument in the specified equation. "
            "The edge count sums all declared arguments across the model's equations. "
            "Multiple repaired graphs can descend from one proposed graph. The "
            "last two columns count formulas tested for a necessary insulin error "
            "greater than 1 pmol/L on training. An unverified certificate leaves "
            "the question open.</p><p>Only train_015 of 16 training events included "
            "an insulin dose, and all four development events had zero administered "
            "insulin, so these forecasts cannot rank models by dose response.</p>"
            + _table(
                [
                    "Generator",
                    "Graphs",
                    "Insulin infusion in dI/dt",
                    "EGP, E and U jointly in dGp/dt",
                    "Median RHS edges (range)",
                    "Insulin direction check",
                    "Insulin ordering check",
                ],
                structure_rows,
            )
        )
    budget = escape(json.dumps(plan["fit_budget"], sort_keys=True))
    total = len(cohort_rows)
    yields = []
    for arm in ARMS:
        arm_groups = [g for g in include.values() if g["arm"] == arm]
        count = sum(g["eligible_graphs"] for g in arm_groups)
        advanced = sum(g["advanced_graphs"] for g in arm_groups)
        winning = sum(g["winner"] is not None for g in arm_groups)
        yields.append(
            f"{LABEL[arm]} qualified {count} of {advanced} graphs, "
            f"with a training-qualified choice in {winning} of {len(arm_groups)} "
            "matched inventories."
        )
    body = [
        "<h1>From topology to development forecasts</h1>",
        f"<p>{total} graphs advanced across {len(groups)} matched inventories. "
        "One arm generated graphs algorithmically. The other began with GPT-OSS-20B "
        "graphs and applied the deterministic repair algorithm. Both received equal "
        "graph slots within each inventory, the same function constructor and the "
        "same per-formula fitting budget. "
        f"Budget · <code>{budget}</code>.</p>",
        "<p>The primary analysis scores every executable formula selected on "
        "training, including formulas with a demonstrated pre-fit contradiction. "
        "Forecast error, pre-fit eligibility and sampled nonnegativity address "
        "distinct questions.</p>",
        "<h2>Which models passed all training admission checks?</h2>"
        + f"<p><strong>{escape(' '.join(yields))}</strong></p>"
        + "<p>Count a graph if at least one formula had no contradiction proved "
        "by the pre-fit checks, produced a "
        "verified training rollout, and kept every measured target nonnegative "
        "at sampled training times (tolerance 10<sup>-10</sup>). For each arm "
        "and inventory, select the qualified formula with lowest training NMSE. "
        "An empty cell means no model qualified in that portfolio. An eligible "
        "insulin formula can have an unresolved direction or ordering check. "
        "The maximum insulin error uses every sampled development time and "
        "event. It measures the largest observed miss directly in pmol/L.</p>",
        _table(
            [
                "Inventory",
                "Generator",
                "Qualified graphs",
                "Qualified formulas",
                "Selected graph",
                "Training NMSE",
                "Development NMSE",
                "By target",
                "Negative minimum on development",
                "Max. absolute insulin error (pmol/L)",
            ],
            qualified_rows,
        ),
        "<h2>All advanced graphs</h2><p>Each graph contributes its training-selected "
        "formula. The scored median uses verified development forecasts. The all-slot "
        "median ranks a missing or failed forecast after any finite score. Negative "
        "counts are graphs that predicted at least one negative measured target at a "
        "sampled development time. A finite trajectory can fail this check.</p>",
        _table(
            [
                "Inventory",
                "Generator",
                "Advanced",
                "Training finalist",
                "Scored",
                "Scored median NMSE",
                "All-slot median",
                "Negative on development",
            ],
            summaries,
        ),
        "<h2>Training-selected portfolio models</h2><p>Each arm keeps the candidate "
        "with the lowest training error for that inventory. A second selection "
        "requires all measured targets to stay at or above the negative-sample "
        "threshold (-10<sup>-10</sup>) at sampled training times. Development "
        "NMSE uses the target scales fitted on training. Training and development "
        "trajectories are distinct. Inventories for a task share events.</p>",
        _table(
            [
                "Inventory",
                "Generator",
                "Training selection",
                "Graph",
                "Pre-fit check",
                "Nonnegative on training",
                "Training NMSE",
                "Development NMSE",
                "By target",
                "Negative minimum",
            ],
            win_rows,
        ),
        "<h2>Forecast examples</h2>",
        *details,
        "<h2>How much do individual draws vary?</h2><p>Within each inventory, compare "
        "every advanced algorithm graph with every advanced 20B-plus-repair graph. "
        "Counts give the exact distribution for a uniform draw from each portfolio. "
        "A failed forecast loses to any finite result, and two failures tie. Pair "
        "counts share graphs and events. They describe this cohort.</p>",
        _table(
            [
                "Inventory",
                "Algorithm lower NMSE",
                "Tie",
                "20B + repair lower NMSE",
                "Graph pairs",
            ],
            chances,
        ),
        structure,
        "<h2>Individual graphs</h2>"
        + _table(
            [
                "Inventory",
                "Generator",
                "Graph",
                "Formula rank",
                "Outcome",
                "Pre-fit check",
                "Training NMSE",
                "Development NMSE",
                "By target",
                "Negative minimum",
            ],
            graph_rows,
        ),
        "<h2>Function construction and fitting effort</h2><p>Each slot counts "
        "once, including any unsuccessful fits. Objective evaluations include "
        "failed training rollouts. Seconds are summed over individual fits. "
        "The pre-fit checks screen for specific contradictions against training "
        "measurements. A score alone can conceal a contradiction.</p>",
        _table(
            [
                "Inventory",
                "Generator",
                "Formula slots",
                "Graphs with eligible fit",
                "Pre-fit checks",
                "Fit outcomes",
                "Starts",
                "Evaluations",
                "Failed rollouts",
                "Fit seconds",
            ],
            costs,
        ),
        "<h2>Which inventories supplied graphs?</h2><p>Within each generator we "
        "deduplicated admitted graphs and kept only exact variable-type inventories. "
        "We sorted each pool by a fixed hash of inventory and graph identity and took "
        "the same number from each arm, limited by the smaller pool and the "
        f"registered cap of {plan['cohort']['cap_per_inventory_arm']}. Inventories "
        "with zero matched slots show where a forecast comparison is unavailable.</p>",
        _table(
            [
                "Inventory",
                "Generator",
                "Admitted occurrences",
                "Unique graphs",
                "Exact inventory",
                "Advanced",
                "Selected by both",
            ],
            pool_rows,
        ),
        "<p>Development outcomes · "
        + escape(json.dumps(checks, sort_keys=True))
        + ".</p>",
        '<p><a href="plan.json">Frozen cohort and budget</a> · '
        '<a href="frozen.json">Frozen training decisions</a> · '
        '<a href="eligibility-freeze.json">Qualified training decisions</a> · '
        '<a href="development.json">Development scores and full curves</a></p>',
    ]
    protocol = "../../docs/PHASE_C_MATCHED_TOPOLOGY_FORECAST.md"
    if (root / protocol).is_file():
        body.append(
            f'<p><a href="{protocol}">Generation costs, protocol and '
            "reproduction commands</a></p>"
        )
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Matched topology forecasts</title><style>{CSS}</style><main>"
        + "".join(body)
        + "</main></html>"
    )
    dest = root / "REPORT.html"
    dest.write_text(html)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(render(parser.parse_args().root))
