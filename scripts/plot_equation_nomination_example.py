"""Illustrate training-only equation nomination using saved free forecasts."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import numpy as np

from autoformalism.rebuttal.prefit_replay import sealed_read


def nominated_target(packet: dict, slot: int | None) -> str | None:
    """Find a fully observed rate whose first nonlinear choice survived fitting."""
    if slot is None:
        return None
    families = packet["shape"]["candidate_families"][slot]
    for name, row in sorted(packet["shape"]["evidence"]["equations"].items()):
        options = row.get("shortlist", [])
        if (
            row["status"] == "all_arguments_observed"
            and row["response"] == "interval_average_rate"
            and options
            and options[0]["shape"] != "affine"
            and options[0]["shape"] == families[name]
        ):
            return name
    return None


def example(root: Path) -> tuple:
    """Select in registered order, without reading any omitted observations."""
    plan = sealed_read(root / "plan.json")
    for panel in plan["panels"]:
        panel_id = panel["panel_id"]
        packet = sealed_read(root / "packets" / f"{panel_id}.json")
        if packet["artifact_sha256"] != panel["packet"]:
            raise ValueError("Packet differs from registered experiment")
        selected = sealed_read(root / "selections" / panel_id / "shape.json")
        target = nominated_target(packet, selected["selected_slot"])
        if target is not None:
            return panel, packet, selected, target
    raise ValueError("No fitted nonlinear nomination with fully observed arguments")


def render(root: Path) -> Path:
    """Draw the two sealed forecasts for the first omitted event of that panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel, packet, selected, target = example(root)
    panel_id = panel["panel_id"]
    evidence = packet["shape"]["evidence"]["equations"][target]
    options = evidence["shortlist"]
    if (target, options[0]["shape"]) != ("Gp", "hinge U"):
        raise ValueError("This T2 figure uses public glucose/disposal units and labels")
    event = min(panel["held_ids"])
    arms = ("original", "shape")
    scores, curves = {}, {}
    for arm in arms:
        s = sealed_read(root / "scores" / panel_id / f"{arm}.json")
        sel = sealed_read(root / "selections" / panel_id / f"{arm}.json")
        if (s["selection"], s["held_sha256"], s["status"]) != (
            sel["artifact_sha256"],
            panel["held_sha256"],
            "scored",
        ):
            raise ValueError("Held score does not match the training selection")
        scores[arm] = s
        curves[arm] = next(
            x for x in s["grid"]["curves"] if x["trajectory_id"] == event
        )
    first, second = (curves[a] for a in arms)
    time = first["time"]
    y = first["observed"][target]
    if second["time"] != time or second["observed"][target] != y:
        raise ValueError("Different observed trajectories")
    threshold = np.median(
        [v for row in packet["train_rows"] for v in row["targets"]["U"]]
    )
    colors = {"original": "#9d4f27", "shape": "#00796b"}
    with plt.rc_context(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    ):
        fig, axes = plt.subplots(
            1, 2, figsize=(10.8, 3.8), width_ratios=[1, 2], layout="constrained"
        )
        ax, forecast = axes
        labels = ["Hinge in disposal", "Affine", "Hinge in meal rate"]
        if [x["shape"] for x in options] != [
            "hinge U",
            "affine",
            "hinge meal_rate_g_per_min",
        ]:
            raise ValueError("Unexpected shortlist for illustrated graph")
        vals = [x["rmse"] for x in options]
        bars = ax.barh(
            labels, vals, color=[colors["shape"], "#AEB6BB", "#AEB6BB"], height=0.48
        )
        ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.23)
        ax.set_xlabel("Rate RMSE (mg/kg/min)")
        ax.set_title("Glucose equation shortlist", loc="left", weight="bold")
        forecast.plot(time, y, color="#24282c", lw=2.3, label="Observed")
        for arm, label, style in (
            ("original", "Original constructor", "--"),
            ("shape", "Trajectory-informed", "-"),
        ):
            forecast.plot(
                time,
                curves[arm]["predicted"][target],
                style,
                color=colors[arm],
                lw=1.9,
                label=label,
            )
        forecast.set(
            xlabel="Time (min)",
            ylabel="Plasma glucose mass (mg/kg)",
            xlim=(time[0], time[-1]),
        )
        forecast.set_title(
            f"Free forecast on omitted event {event}", loc="left", weight="bold"
        )
        forecast.legend(loc="best", frameon=False)
        fig.savefig(root / "NOMINATION-EXAMPLE.png", dpi=160, facecolor="white")
        plt.close(fig)

    def metrics(arm):
        s = scores[arm]
        row = next(x for x in s["events"] if x["trajectory_id"] == event)
        return (
            row["targets"][target]["nmse"],
            s["targets"][target]["nmse"],
            s["pooled_nmse"],
        )

    table_rows = "".join(
        "<tr><th>"
        + label
        + "</th>"
        + "".join(f"<td>{v:.4f}</td>" for v in metrics(arm))
        + "</tr>"
        for arm, label in (
            ("original", "Original constructor"),
            ("shape", "Trajectory-informed"),
        )
    )
    fitted = sealed_read(
        root / "fits" / panel_id / "shape" / f"{selected['selected_slot']}.json"
    )
    if fitted["artifact_sha256"] != selected["receipts"][selected["selected_slot"]]:
        raise ValueError("Unmatched parameter vector")
    formula = packet["shape"]["candidates"][selected["selected_slot"]]["equations"][
        target
    ]
    page = root / "NOMINATION-EXAMPLE.html"
    page.write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Choosing a glucose equation</title>

<style>
body{{font:17px/1.55 system-ui,sans-serif;color:#202629;
max-width:980px;margin:3rem auto;padding:0 1.2rem}}
img{{max-width:100%}}
h1,h2{{line-height:1.2}}
th,td{{padding:.45rem .8rem;
border-bottom:1px solid #ddd;text-align:right}}
th:first-child{{text-align:left}}
code{{overflow-wrap:anywhere}}
.eq{{font:1.25rem/1.5 Georgia,serif}}
</style>
<main><h1>How the constructor chose a glucose equation</h1>
<p>The fixed T2-easy graph gives the plasma glucose derivative three arguments:
total glucose disposal
<i>U</i>, insulin-independent disposal <i>U<sub>ii</sub></i>, and meal rate <i>r</i>.
The constructor compared {evidence["compared"]} forms on the
{len(packet["train_rows"])} fitting events.
It measured changes in glucose over 1, 4 and 16 time steps and compared those
changes per minute
with the average rate predicted by each equation. Each whole event was omitted in one of
two inner folds.</p>

<p class="eq">d<i>G<sub>p</sub></i>/d<i>t</i> = a + b<i>U</i> + c<i>U<sub>ii</sub></i>
+ d<i>r</i> + e max(<i>U</i> &minus; {threshold:.3f}, 0)</p>
<p>The last term lets the slope change above the median disposal of
{threshold:.3f} mg/kg/min.
This is the selected library form in physical units, with coefficients re-expressed
for the displayed units.
A whole-model fit subsequently adjusted its coefficients and those in the other
equations together.</p>
<img
 src="NOMINATION-EXAMPLE.png"
 alt="Shortlisted glucose rate errors and free forecasts of plasma glucose for the
first held event">
<h2>Does the full model forecast better?</h2>
<table>
<thead><tr><th>Constructor</th><th>Event glucose NMSE</th>
<th>All omitted events, glucose NMSE</th>

<th>All omitted events and targets,
NMSE</th></tr></thead><tbody>{table_rows}</tbody></table>
<p>On these eight omitted training events the trajectory-informed model improves the
glucose forecast,
while its average error over glucose, insulin and disposal increases. The forecasts
integrate each complete system,
so the difference cannot be assigned to the glucose term alone. This experiment does
not identify the true glucose law.</p>
<p><small>The example uses the first graph and fold in registered order where all
arguments were measured and the
training-selected model retained the locally preferred nonlinear form for a
differential equation
({escape(panel_id)}). The plotted event is the first omitted identifier. The local
score is
rate RMSE; the table uses complete-model rollout NMSE normalized by fitting-event
scales.</small></p>
<details>
<summary>Exact constructed right-hand side</summary>
<code>{escape(formula)}</code></details></main></html>""")
    return page


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(render(parser.parse_args().root))
