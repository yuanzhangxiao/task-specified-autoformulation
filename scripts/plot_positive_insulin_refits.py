"""Plot the four matched development events from the frozen joint-fit report."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read


def plot(report_file: Path, image: Path) -> None:
    """Compare the two insulin laws at the recorded observation times in each slot."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    payload = sealed_read(report_file)
    report = payload["report"]
    if report["split"] != "val" or not payload["freeze"]:
        raise ValueError("Expected a frozen development-validation report")
    families = ("restoring", "positive_EGP")
    event_ids = [f"val_{n:03}" for n in range(4)]
    fig = Figure(figsize=(12.5, 5.3), facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 4, sharex=True, sharey="col")
    maxima = [0.0] * len(event_ids)
    minima = [0.0] * len(event_ids)
    traces = []
    for row, slot in enumerate(("002", "015")):
        stages = [report["models"][f"{slot}-{f}"]["joint"] for f in families]
        for j, stage in enumerate(stages):
            cases = {e["event"]: e for e in stage["events"]}
            if sorted(cases) != event_ids or len(cases) != len(stage["events"]):
                raise ValueError(f"Expected four unique development events for {slot}")
            for i, name in enumerate(event_ids):
                event = cases[name]
                if event["status"] != "scored":
                    raise ValueError(
                        f"Forecast unresolved for {slot}/{families[j]}/{name}"
                    )
                time = event["time"]
                observed, predicted = event["observed"]["I"], event["predicted"]["I"]
                ax = axes[row, i]
                if row == 0 and j == 0:
                    traces.append((time, observed))
                    ax.set_title(name, fontsize=11, loc="left")
                if (time, observed) != traces[i]:
                    raise ValueError(
                        f"Time or observations differ across models for {name}"
                    )
                if j == 0:
                    ax.plot(time, observed, color="#232525", label="Observed", lw=1.8)
                style = (
                    {"color": "#B85421", "ls": "--", "label": "Restoring"}
                    if j == 0
                    else {"color": "#1269AA", "ls": "-", "label": "Positive production"}
                )
                ax.plot(time, predicted, lw=1.6, **style)
                maxima[i] = max(maxima[i], max(observed), max(predicted))
                minima[i] = min(minima[i], min(observed), min(predicted))
        axes[row, 0].set_ylabel(f"Slot {slot}\nInsulin (pmol/L)", fontsize=10)
    for col, upper in enumerate(maxima):
        for ax in axes[:, col]:
            ax.set_ylim(minima[col] * 1.07, max(1.0, upper * 1.07))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.grid(axis="y", alpha=0.17)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.076, bottom=0.11, top=0.80, right=0.99, hspace=0.18)
    fig.suptitle("Insulin trajectories after joint refitting", fontsize=15, y=0.99)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        ncol=3,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
        fontsize=10,
    )
    fig.supxlabel("Time (min)", y=0.02, fontsize=11)
    fig.savefig(image, dpi=170)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("report", type=Path)
    cli.add_argument("png", type=Path)
    args = cli.parse_args()
    plot(args.report, args.png)
