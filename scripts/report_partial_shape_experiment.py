"""Render the partial-observation experiments from their sealed local results."""

# ruff: noqa: E501, RUF001 -- HTML prose and typeset mathematical equations.

from __future__ import annotations

import argparse
import base64
from collections import Counter
from html import escape
from io import BytesIO
from itertools import accumulate
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure

from autoformalism.rebuttal.prefit_replay import sealed_read

DEFAULT = Path(__file__).resolve().parents[1] / "output/partial-shape-20261126"
FAMILIES = {
    "affine": "affine",
    "offset_plus_square": "offset + square",
    "unresolved": "unresolved",
}


def fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.5g}"


def table(headers, body) -> str:
    """Escape all fields and allow wide tables to scroll within narrow viewports."""
    head = "".join(f'<th scope="col">{escape(str(c))}</th>' for c in headers)
    rows = "".join(
        "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>"
        for row in body
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def load_summary(root: Path, name: str) -> dict:
    value = sealed_read(root / name / "summary.json")
    if value["plan"] != sealed_read(root / name / "plan.json")["artifact_sha256"]:
        raise ValueError(f"Wrong plan for {name}")
    return value


def cycle_plot(exact: dict, interaction: dict) -> str:
    """Show how an additive forcing must return to its original value around a cycle."""
    traces = [r["witnesses"] for r in (exact, interaction)]
    transitions = [r["levels"] for r in traces[0]]
    if transitions != [r["levels"] for r in traces[1]] or not transitions:
        raise ValueError("Controls do not share a cycle")
    levels = [transitions[0][0], *(b for _, b in transitions)]
    if levels[0] != levels[-1] or levels[:-1] != [a for a, _ in transitions]:
        raise ValueError("Expected a closed sequence of input levels")
    with mpl.rc_context({"svg.hashsalt": "partial-shape-cycle", "font.size": 11}):
        fig = Figure(figsize=(7.0, 2.9), layout="constrained")
        ax = fig.subplots()
        ax.axhline(0, color="#707984", linewidth=0.8)
        for w, label, color, marker in zip(
            traces,
            ["Additive hidden effect", "Multiplicative hidden effect"],
            ["#136350", "#8d3f24"],
            ["o", "s"],
            strict=True,
        ):
            ax.plot(
                range(len(levels)),
                list(accumulate((r["contrast"] for r in w), initial=0)),
                marker=marker,
                color=color,
                label=label,
                linewidth=2.2,
            )
        ax.set(
            xticks=list(range(len(levels))),
            xticklabels=[fmt(z) for z in levels],
            xlabel="Input level reached",
            ylabel="Cumulative rate change",
        )
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="lower left")
        buf = BytesIO()
        fig.savefig(buf, format="svg", metadata={"Date": None})
    image = base64.b64encode(buf.getvalue()).decode("ascii")
    end = [sum(r["contrast"] for r in w) for w in traces]
    return f'<figure><img src="data:image/svg+xml;base64,{image}" alt="Cumulative contrast at the return to the initial level is additive {fmt(end[0])}, multiplicative {fmt(end[1])}."><figcaption>Four events complete one input cycle. Summing their measured changes in rate cancels any single-valued additive response around the cycle.</figcaption></figure>'


def jumps_section(root: Path) -> str:
    summary = load_summary(root, "jumps")
    cases = {}
    for name, meta in summary["cases"].items():
        rec = sealed_read(root / "jumps" / "cases" / f"{name}.json")
        if (rec["name"], rec["plan"], rec["artifact_sha256"]) != (
            name,
            summary["plan"],
            meta["receipt"],
        ):
            raise ValueError("Jump receipt differs from sealed summary")
        cases[name] = rec["result"]
    exact, bounded, interaction = (
        cases[n] for n in ("exact", "bounded", "interaction")
    )
    values = exact["components"][0]["levels"]
    bounds = bounded["components"][0]["levels"]
    rows = []
    for e, b in zip(values, bounds, strict=True):
        if e["level"] != b["level"]:
            raise ValueError("Level grids differ")
        rows.append(
            [
                fmt(e["level"]),
                e["exact_relative_value"],
                f"[{fmt(b['lower'])}, {fmt(b['upper'])}]",
            ]
        )
    degree = exact["anchored_polynomial_interpolant"]
    components = cases["disconnected"]["components"]
    unavailable = [
        name.replace("_", " ")
        for name, r in cases.items()
        if r["status"] == "abstained"
    ]
    return f"""<section><h2>Changes at an isolated input switch</h2>
    <p>Suppose ẏ = A + g(u), with A continuous when u switches from a to b. Subtracting the left rate from the right rate gives g(b) − g(a), even when A includes unknown states prepared differently in each event. The experiment uses sampled y alone to estimate the two rates.</p>
    {cycle_plot(exact, interaction)}
    {table(["Input level", "g(u) − g(0), exact control", "Interval with observation error and hidden drift"], rows)}
    <p>The exact values determine an interpolating polynomial of degree {degree["degree"]} among polynomials of degree at most {degree["unique_within_degree_at_most"]}, up to the common offset. Functions outside that class can match the same four points. The bounded-data control also {escape(bounded["affine"]["status"])} affine forcing. The multiplicative control violates the additive constraints by at least {fmt(interaction["minimum_cycle_violation"])} around the displayed cycle.</p>
    <p>Removing the jumps linking the two groups leaves {len(components)} separate offsets. The screen abstained for {escape(", ".join(unavailable))}.</p>
    <a href="jumps/summary.json">Jump results</a></section>"""


def decision(counts: dict) -> str:
    return (
        ", ".join(f"{n} {FAMILIES.get(k, k)}" for k, n in sorted(counts.items())) or "—"
    )


def pathway_section(root: Path) -> str:
    summary = load_summary(root, "pathway")
    plan = sealed_read(root / "pathway/plan.json")
    rows = []
    for row in summary["rows"]:
        counts = row["decisions"]
        scores = row["mean_evaluation_rmse"]
        rows.append(
            [
                FAMILIES[row["truth"]],
                2 if row["narrow"] else 4,
                fmt(row["noise"]),
                f"{row['estimated']}/{row['runs']}",
                decision(counts["same_levels"]),
                decision(counts["intermediate_levels"]),
                fmt(scores.get("affine")),
                fmt(scores.get("offset_plus_square")),
            ]
        )
    eq = summary["equivalence"]["cases"]
    example_path = "pathway/trials/offset_plus_square-0.0-0-0.json"
    example = sealed_read(root / example_path)
    if example["plan"] != summary["plan"]:
        raise ValueError("Pathway example belongs to another plan")
    screened = example["fit"]["family_screen"]
    excluded = ", ".join(k for k, r in screened.items() if r["status"] == "excluded")
    estimated = ", ".join(k for k, r in screened.items() if r["status"] == "estimated")
    narrow = [r for r in summary["rows"] if r["narrow"]]
    tied = sum(r["decisions"]["same_levels"].get("unresolved", 0) for r in narrow)
    selected = sum(
        r["decisions"]["intermediate_levels"].get(r["truth"], 0) for r in narrow
    )
    trials = sum(r["runs"] for r in narrow)
    return f"""<section><h2>A hidden state between the input and output</h2>
    <p>For ż = −bz + g(u) and ẏ = −ay + kz, eliminating z gives y″ + (a+b)y′ + aby = kg(u). Integrating this identity over windows estimates the combined forcing kg(u) and the two decay rates up to permutation. Each event starts from a different hidden initial value.</p>
    <p>In the clean square-law control at four input levels (seed 0), the integral constraints excluded {escape(excluded)}. The {escape(estimated)} families admitted estimates. A cubic family includes quadratics. <a href="{example_path}">Per-family fit and bounds</a></p>
    <p>We compare the equally sized hypotheses q(u) = c₀ + c₁u and q(u) = c₀ + c₁u². They coincide at u = 0 and 1 for shared coefficients. Whole development events at intermediate inputs can distinguish them. After using observations through time {fmt(plan["prefix_end"])} to initialize the forecast, a candidate must gain more than {fmt(plan["margin"])} RMSE on a majority of events and lose by that margin on none. The last two columns show mean RMSE on fresh synthetic evaluation events after that prefix.</p>
    <p>With two training input levels the choice remained unresolved on {tied} of {trials} runs when the development events repeated those levels. Intermediate development levels selected the generating law on {selected} of {trials}.</p>
    {table(["Generating law", "Training levels", "Noise bound", "Estimated / trials", "Preferred at training levels", "Preferred at intermediate levels", "Affine RMSE", "Offset + square RMSE"], rows)}
    <p>A forecast preference ranks the two specified families. Exact hidden-coordinate ambiguities persist even with unlimited events of this kind. For ẏ = z² and ż = −bz, the transformation w = z² gives ẏ = w and ẇ = −2bw, with w(0) = z(0)² for every positive hidden initial state.</p>
    {table(["Alternative decay for w", "Equivalent by substitution", "Largest observed discrepancy"], [[v["other_decay"], "Yes" if v["proved_equivalent"] else "No", fmt(v["largest_observed_difference"])] for v in eq])}
    <a href="pathway/summary.json">Pathway results</a></section>"""


def repeated_level(root: Path, pooled: dict, value: dict, units: str) -> str:
    """Find the largest within-level contrast on the same matched initial slice."""
    graph = sealed_read(root / "t2-pooled/graphs" / (value["graph"] + ".json"))
    if (
        graph["plan"] != pooled["plan"]
        or graph["artifact_sha256"] not in pooled["graphs"]
    ):
        raise ValueError("Pooled graph differs from sealed summary")
    eq = next(r for r in graph["equations"] if r["target"] == value["target"])
    source = next(s for s in eq["sources"] if s["source"] == value["source"])
    contrasts = []
    for level in source["largest_slice"]:
        obs = sorted(level["observations"], key=lambda r: r["estimate"])
        if len(obs) >= 2:
            contrasts.append(
                (
                    obs[-1]["estimate"] - obs[0]["estimate"],
                    level["level"],
                    obs[0],
                    obs[-1],
                )
            )
    if not contrasts:
        return ""
    spread, x, low, high = max(contrasts, key=lambda item: item[0])
    if spread <= 0:
        return ""
    return f"""<p>At the same {escape(value["source"])} level ({x:.9g}), all other initial arguments also agree. The observed rates are {high["estimate"]:+.7f} for {escape(high["id"])} and {low["estimate"]:+.7f}{units} for {escape(low["id"])}. If each rate error were strictly below half this spread (approximately {spread / 2:.6f}{units}), a single-valued instantaneous equation using those arguments could not explain both initial rates. This would also exclude nonlinear functions on that graph. This bound has yet to be established.</p>"""


def rate_sensitivity(root: Path, pooled: dict) -> str:
    """Render one observed three-level comparison and its conditional rejection bound."""
    value = pooled.get("rate_bound_sensitivity")
    if value is None:
        return ""
    source = escape(value["source"])
    target = escape(value["target"])
    rows = [
        [
            e["id"],
            fmt(e["source_level"]),
            fmt(e["first_interval_duration"]),
            fmt(e["forward_secant"]),
        ]
        for e in value["events"]
    ]
    units = " pmol/L/min" if value["target"] == "I" else " in observed rate units"
    return f"""<h3>How accurate would an initial rate measurement need to be?</h3>
    <p>In {escape(value["graph"])}, three training events have the same values of the other initial arguments for {target}′ while {source} varies. These are one-minute changes divided by elapsed time. The table uses the triple with the largest departure from a straight line within the selected group.</p>
    {table(["Training event", f"Initial {source}", "Elapsed minutes", f"Mean {target} rate"], rows)}
    <p>A straight line through the two endpoint rates gives {fmt(value["affine_middle_from_endpoint_secants"])}{units} at the middle level. The observed middle rate is {fmt(value["events"][1]["forward_secant"])}{units}, a difference of {fmt(value["absolute_middle_departure"])}. If the three estimated rates were each within strictly less than {fmt(value["required_strict_uniform_rate_error"])}{units} of the true initial rate, the two error intervals could not bridge this difference. The data contract supplies no such bound, so an affine effect on this slice remains possible.</p>{repeated_level(root, pooled, value, units)}"""


def t2_section(root: Path) -> str:
    s = load_summary(root, "t2")
    keys = {
        "equation_fold_cases": "Measured target equations across folds",
        "cases_with_at_least_one_equal_hidden_pair": "At least one pair with matching hidden initial values",
        "cases_with_isolated_variation": "One source varies with the others fixed",
        "cases_with_three_or_more_levels": "Three or more isolated source levels",
        "cases_with_repeated_level_conflict": "Inconsistent response at a repeated level",
        "cases_with_certified_affine_exclusion": "Affine forcing excluded on an observed slice",
        "cases_with_isolated_jumps": "An isolated interior input switch",
        "cases_with_one_hidden_filter_graph": "A matching single-hidden-state pathway",
    }
    rows = [
        [label, s["all_measured"][key], s["hidden_sourced"][key]]
        for key, label in keys.items()
    ]
    names = Counter(
        source
        for row in s["cases"]
        for source in row["sources_with_isolated_variation"]
    )
    sources = ", ".join(f"{name} ({n})" for name, n in sorted(names.items())) or "none"
    pooled_text = ""
    if (root / "t2-pooled/summary.json").exists():
        pooled = load_summary(root, "t2-pooled")
        p = pooled["hidden_sourced"]
        contrasts = sorted(
            (r["graph"], r["target"], r["source"])
            for r in pooled["three_level_contrasts"]
            if r["hidden_sources"]
        )
        pooled_text = f"""<h3>Pooling the public training events</h3>
        <p>In a separate exploratory analysis using all {min(pooled["events_per_graph"])} training events per graph and counting each equation only once per graph, {p["cases_with_three_or_more_levels"]} of {p["equations"]} hidden-source equations supplied at least three isolated levels. Their measured responses were one-minute slopes. The available uncertainty information yielded {p["cases_with_certified_affine_exclusion"]} qualified exclusions of affine forcing.</p>
        {table(["Graph", "Measured differential target", "Varied source"], contrasts)}
        <a href="t2-pooled/summary.json">Pooled training evidence</a>{rate_sensitivity(root, pooled)}"""
    return f"""<section><h2>Where the frozen T2-easy graphs supplied evidence</h2>
    <p>The audit used the fitting events from {len(s["panels"])} panels on {s["distinct_graphs"]} existing graphs. An equation in the same graph is counted once for each fitting fold here. The last column restricts the count to equations with hidden sources. Matching hidden initial values follows from each graph’s declared initializer.</p>
    {table(["Available evidence", "All measured equations", "With hidden sources"], rows)}
    <p>Observed sources with isolated variation among hidden-source cases, with the number of cases in parentheses, were {escape(sources)}. A pair at two input levels gives one difference. Curvature requires more levels with the other arguments controlled, or stronger assumptions. Forecast improvement on T2 was outside this audit.</p>
    <a href="t2/summary.json">T2 panel counts and cases</a>{pooled_text}</section>"""


def render(root: Path) -> Path:
    body = t2_section(root) + jumps_section(root) + pathway_section(root)
    path = root / "REPORT.html"
    path.write_text(
        f"""<!doctype html><html lang="en"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1"><title>Learning equation shapes with hidden states</title>
    <style>body{{margin:2rem auto 5rem;padding:0 1rem;max-width:68rem;color:#152c35;font:1.02rem/1.5 system-ui,sans-serif}}h1,h2{{line-height:1.15}}h2{{margin-top:2rem}}p{{max-width:83ch}}a{{color:#145b82}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.92rem}}td,th{{padding:.45rem .6rem;text-align:left;border-bottom:1px solid #c8d6dc}}th{{background:#edf4f4}}img{{width:100%;max-width:700px}}figure{{margin:1rem 0}}figcaption{{color:#40545b;max-width:75ch}}section{{margin-top:1.5rem}}</style>
    <main><h1>Learning equation shapes with hidden states</h1><p>These experiments ask which properties of an equation can be determined from measured trajectories when some states are hidden, and when a forecast can distinguish alternatives.</p>{body}</main></html>""",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    print(render(parser.parse_args().root))
