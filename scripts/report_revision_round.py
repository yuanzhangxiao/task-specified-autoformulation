"""Render the frozen one-round construction experiment from its sealed receipts."""

from __future__ import annotations

import argparse
import io
import json
import math
from html import escape
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts.revision_round import metric

CASES = {
    "basin": (
        "Coupled basin",
        "Fit the upstream and downstream discharge powers. "
        "Rank by mean depth NMSE across six omitted training events. "
        "Every power pair uses the same two gain starts on each fitting split.",
    ),
    "reactor": (
        "CSTR-easy",
        "Change concentration power and temperature dependence in "
        "the reaction heat law. Rank by full-event training temperature NMSE. "
        "Each candidate uses two starts and short/full-event optimization.",
    ),
    "insulin": (
        "T2-easy",
        "Change the initial contribution to disposal and its "
        "half-life. Rank by pooled training NMSE. Refit the disposal intercept, "
        "tissue slope and any added amplitude, with the remaining coefficients "
        "fixed. Require the training initial-order margins and physical checks.",
    ),
}
POOLS = {
    "algorithm": "Frozen shortlist · 3",
    "algorithm_screen": "Training screen · 3",
    "20b_first_three": "20B · seed 100 · 3",
    "20b_seed101_three": "20B · seed 101 · 3",
    "20b_seed102_three": "20B · seed 102 · 3",
    "20b_all_nine": "20B · all seeds · 9",
}
CSS = """
body{margin:auto;max-width:1120px;padding:30px 5vw 70px;color:#172d34;
font:16px/1.5 system-ui,sans-serif;background:#fafaf8}h1,h2,h3{line-height:1.17}
h1{font-size:2.55rem;max-width:890px}h2{font-size:1.75rem;border-top:2px solid
#246a6b;padding-top:18px;margin-top:38px} h3{font-size:1.1rem;margin:24px 0 8px}
p{max-width:96ch}nav a,a{color:#145e70}nav a{margin-right:18px}
.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;
font-variant-numeric:tabular-nums;font-size:.90rem}
th,td{padding:7px 9px;text-align:left;
border-bottom:1px solid #dae3e3;vertical-align:top}th{background:#e8f0ef}td:first-child
{font-weight:600}code{white-space:pre-wrap;overflow-wrap:anywhere;color:#294952}
figure{margin:12px 0 16px;max-width:760px}figure svg{height:auto;width:100%}
small,figcaption{color:#475c62}details{margin:10px 0}summary{cursor:pointer}
"""


def number(value: float | None) -> str:
    """Keep small NMSEs visible while distinguishing missing scores."""
    return f"{value:.5g}" if value is not None and math.isfinite(value) else "—"


def _table(headings: list[str], rows: list[list[str]]) -> str:
    """Escape all evidence and provider text before using it in the document."""
    header = "".join(f"<th>{escape(s)}</th>" for s in headings)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(s))}</td>" for s in row) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="scroll"><table><thead><tr>{header}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _plot(points: list[tuple[str, float, float]]) -> str:
    """Pair training and development scores on a common log axis within a case."""
    import matplotlib as mpl
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    usable = [r for r in points if all(math.isfinite(x) and x > 0 for x in r[1:])]
    if not usable:
        return ""
    with mpl.rc_context({"svg.hashsalt": "one-round-revision", "svg.fonttype": "none"}):
        fig = Figure(figsize=(7.2, 1.2 + 0.27 * len(usable)), layout="constrained")
        canvas = FigureCanvasSVG(fig)
        ax = fig.subplots()
        for i, (_, tr, dev) in enumerate(usable):
            ax.plot([tr, dev], [i, i], color="#9ca8aa", zorder=0)
        ax.scatter(
            [p[1] for p in usable],
            range(len(usable)),
            color="#187878",
            marker="o",
            label="Training",
        )
        ax.scatter(
            [p[2] for p in usable],
            range(len(usable)),
            color="#b44921",
            marker="D",
            label="Development",
        )
        ax.set_yticks(range(len(usable)), [p[0] for p in usable])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("Pooled NMSE · logarithmic scale · lower is better")
        ax.grid(axis="x", alpha=0.18)
        fig.legend(loc="outside upper right", ncols=2, frameon=False)
        stream = io.StringIO()
        canvas.print_svg(stream, metadata={"Date": None})
    svg = stream.getvalue()
    return (
        '<figure aria-label="Training and development scores">'
        + svg[svg.index("<svg") :]
        + "</figure>"
    )


def _costs(receipts: dict) -> list[str]:
    values = list(receipts.values())
    known = [
        r["usage"]["total_tokens"]
        for r in values
        if isinstance(r.get("usage"), dict)
        and r["usage"].get("total_tokens") is not None
    ]
    return [
        str(len(values)),
        str(sum(r["status"] == "responded" for r in values)),
        str(sum(known)),
        str(len(values) - len(known)),
        number(sum(r.get("seconds", 0) for r in values)),
    ]


def _effort(fits: dict, case: str) -> str:
    if case == "insulin":
        values = [r.get("fit", {}).get("forward_evaluations") for r in fits.values()]
        unit = "forward evaluations"
    else:
        field = "evaluations" if case == "basin" else "optimizer_evaluations"
        values = [r.get(field) for r in fits.values()]
        unit = "residual evaluations" if case == "basin" else "optimizer evaluations"
    known = [v for v in values if v is not None]
    return f"{sum(known):,} {unit} recorded for {len(known)}/{len(values)} fits"


def _case(
    root: Path, case: str, training: dict, choices: dict, scores: dict, seeds: list[int]
) -> str:
    title, scope = CASES[case]
    parent = training["parent"]
    keys = list(
        dict.fromkeys(
            [
                parent,
                *training["slots"]["algorithm"],
                *training["slots"].get("algorithm_screen", []),
                *training["slots"]["20b_all_nine"],
            ]
        )
    )
    ids = {
        k: f"M{i}" if i else "Parent"
        for i, k in enumerate(k for k in keys if k is not None)
    }
    pick_rows, pool_rows = [], []
    for arm, label in POOLS.items():
        if arm not in choices:
            continue
        decision, slots = choices[arm], training["slots"][arm]
        valid = [k for k in slots if k is not None]
        pool_rows.append(
            [
                label,
                str(len(slots)),
                str(len(slots) - len(valid)),
                str(len(valid) - len(set(valid))),
                str(valid.count(parent)),
                str(decision["eligible_slots"]),
                str(decision["distinct_eligible_edits"]),
            ]
        )

        def show(k):
            if k is None:
                return "—"
            return f"{ids[k]} ({number(scores.get(k, {}).get('nmse'))})"

        pick_rows.append(
            [label, show(decision["algorithm"])]
            + [show(decision["20b"].get(str(s))) for s in seeds]
            + [
                ", ".join(
                    ids.get(decision.get("uniform", {}).get(str(s)), "—") for s in seeds
                )
            ]
        )
    rows, points = [], []
    for key, name in ids.items():
        fit = training["fits"][key]
        if key in training["eligible"]:
            edit = training["eligible"][key]
        else:
            edit = sealed_read(root / "fits" / case / f"{key}.json")["edit"]
        result = scores.get(key, {})
        dev = result.get("nmse")
        train = fit.get("train_nmse")
        if train is not None and dev is not None:
            points.append((name, train, dev))
        target = result.get("per_target_nmse", {})
        if case == "reactor" and dev is not None:
            target = {"T": dev}
        checks = []
        if not fit["eligible"]:
            checks.append(fit.get("reason", "Ineligible"))
        if result.get("negative_samples"):
            checks.append(
                "negative samples " + str(sum(result["negative_samples"].values()))
            )
        if case == "insulin" and result:
            checks.append(
                f"initial reversals {result.get('reversed_initial_pairs')}/"
                f"{result.get('comparable_initial_pairs')}"
            )
            if not result.get("half_gap_passed"):
                checks.append("development half-gap fails")
            if not result.get("physical_checks_passed"):
                checks.append("development domain fails")
        rows.append(
            [
                name,
                json.dumps(edit, sort_keys=True),
                number(train),
                number(metric(fit)) if fit["eligible"] else "—",
                number(dev),
                ", ".join(f"{k} {number(v)}" for k, v in target.items()) or "—",
                " · ".join(checks) or ("Eligible" if fit["eligible"] else "—"),
            ]
        )
    parent_dev = scores.get(parent, {}).get("nmse")
    # Both quantities must be better/worse on the exact same two fitted models.
    comparisons = [
        n
        for k, n in ids.items()
        if k in scores
        and k != parent
        and parent_dev is not None
        and scores[k].get("nmse") is not None
        and training["fits"][k].get("train_nmse") is not None
        and training["fits"][parent].get("train_nmse") is not None
        and training["fits"][k]["train_nmse"] < training["fits"][parent]["train_nmse"]
        and scores[k]["nmse"] > parent_dev
    ]
    tradeoff = (
        (
            " Lower training NMSE with higher development NMSE than the parent in "
            + ", ".join(comparisons)
            + "."
        )
        if comparisons
        else ""
    )
    screen = training.get("screen", {})
    counts = {k: v for k, v in screen.items() if type(v) in (float, int)}
    counts.update(screen.get("cost", {}))
    screen_info = (
        (
            f"<p>Training screen counts &nbsp; "
            f"{escape(json.dumps(counts, sort_keys=True))}.</p>"
            f"<p>{escape(screen.get('criterion', ''))}</p>"
        )
        if counts
        else ""
    )
    return (
        f'<section id="{case}"><h2>{escape(title)}</h2><p>{escape(scope)}</p>'
        + "<h3>Candidate yield</h3>"
        + _table(
            [
                "Generator and slots",
                "Slots",
                "Missing",
                "Repeated slots",
                "Parent returns",
                "Eligible slots",
                "Unique eligible children",
            ],
            pool_rows,
        )
        + "<h3>Selections and development NMSE</h3>"
        + _table(
            [
                "Generator and slots",
                "Numerical pick",
                *[f"20B rank {s}" for s in seeds],
                "Uniform picks (seed order)",
            ],
            pick_rows,
        )
        + screen_info
        + _plot(points)
        + "<h3>Every distinct edit</h3>"
        + _table(
            [
                "ID",
                "Edit",
                "Training NMSE",
                "Selection score",
                "Development NMSE",
                "By target",
                "Checks",
            ],
            rows,
        )
        + f"<p>{escape(_effort(training['fits'], case) + '.' + tradeoff)}</p></section>"
    )


def render(root: Path) -> Path:
    """Validate the plan-to-development chain and render all frozen choices."""
    plan, training, frozen, dev = (
        sealed_read(root / f"{name}.json")
        for name in ("plan", "training", "frozen", "development")
    )
    if (training["plan"], frozen["training"], dev["freeze"]) != (
        plan["artifact_sha256"],
        training["artifact_sha256"],
        frozen["artifact_sha256"],
    ):
        raise ValueError("Revision report receipts disagree")
    sections = [
        _case(
            root,
            case,
            training["cases"][case],
            frozen["choices"][case],
            dev["cases"][case],
            plan["seeds"],
        )
        for case in CASES
    ]
    nav = " ".join(f'<a href="#{k}">{v[0]}</a>' for k, v in CASES.items())
    costs = [
        ["Generate", *_costs(training["generation_cost"])],
        ["Rank", *_costs(frozen["rank_cost"])],
    ]
    valid_ranks = sum(
        r.get("permutation_valid", False) for r in frozen["rank_cost"].values()
    )
    summary_rows = []
    for case, (name, _) in CASES.items():
        scores = dev["cases"][case]
        parent = training["cases"][case]["parent"]
        row = [name, number(scores.get(parent, {}).get("nmse"))]
        for arm in POOLS:
            key = frozen["choices"][case].get(arm, {}).get("algorithm")
            row.append(number(scores.get(key, {}).get("nmse")))
        summary_rows.append(row)
    summary = _table(["Case", "Refit parent", *POOLS.values()], summary_rows)
    cost_table = _table(
        ["Stage", "Receipts", "Replies", "Recorded tokens", "Unknown usage", "Seconds"],
        costs,
    )
    text = f"""<!DOCTYPE html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>One round of scientific model revision</title><style>{CSS}</style><main>
<h1>One round of scientific model revision</h1>
<p>Three saved parent equations were refitted alongside edits proposed from training
evidence. Each proposer seed has three slots. The numerical selector ranks eligible
fits by the training metric. GPT-OSS-20B sees the eligible equations and training
evidence with shuffled candidate labels. The prompt withholds generator identity
and fit scores. Each pool uses the order of its own candidates in the complete
ranking. Each selector can keep the refitted parent.</p>
<h3>Development NMSE of the numerical selector's pick</h3>
{summary}
<nav>{nav}</nav>
<p>The frozen shortlist uses three prespecified families for T2 and CSTR.
The training screen ranks edits using measured trajectories, then fits its three
nominees. The two algorithm pools coincide for the basin. The nine-slot 20B pool
contains the three separate seeds. Compare generators at three slots and examine
the nine-slot pool separately for the benefit of extra calls.</p>
<p>Repeats count within a pool. Eligible slots include repeats and returns of the
parent. The unique-child count excludes the parent. Each distinct model is fitted
once per case. Numbers in parentheses are pooled development NMSE. The report shows
an em dash for unavailable picks or scores. Development evaluation covers the frozen
selections and the parent. Two edits can yield equivalent forecasts.</p>
{"".join(sections)}
<h2>Compute and interpretation</h2>
{cost_table}
<p>{valid_ranks} of {len(frozen["rank_cost"])} rank replies gave a full permutation.
Provider seconds sum individual requests. Each generator uses the same fitting
procedure within a case. Evaluation units differ across cases.</p>
<p>These three development cases have been examined previously. This study tests one
revision from each supplied parent in a bounded edit grammar. The ranking seeds
measure variation on the same events. A fresh benchmark and a frozen policy are
needed to test transfer. Scientific ranking expresses a preference. Predictive
accuracy and mechanism compliance need independent evidence.</p>
<p>The separate topology replay produced 40 eligible graphs in 52 full checks across
five saved inventories on two benchmark cells. An algorithm repaired 8 of 15
20B-proposed parents, yielding 113 distinct graphs under a larger search budget.
<a href="topology-audit.md">Read the variable-to-topology audit</a>.</p>
<p><a href="plan.json">Plan</a> · <a href="screens.json">Training screens</a>
· <a href="training.json">Training receipts</a>
· <a href="frozen.json">Frozen choices</a>
· <a href="development.json">Development scores</a></p>
</main></html>"""
    dest = root / "REPORT.html"
    dest.write_text(text, encoding="utf-8")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(render(parser.parse_args().root))
