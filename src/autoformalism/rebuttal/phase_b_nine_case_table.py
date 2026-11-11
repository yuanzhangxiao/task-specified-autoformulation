"""Per-case and aggregate validation NMSE of the nine-case baselines.

Each task's validation free-rollout NMSE; a task that ended in the method's
own failure (no usable model, a rollout that did not finish, a fit stopped at
its wall-clock limit) counts as +inf. A case's score is the median over its
repetitions, with the unscaled median absolute deviation; a method's
aggregate is the median over cases, with the same deviation, and the share of
tasks with a finite score. A pending task leaves its method incomplete rather
than failed: outages are rerun, not scored.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from autoformalism.rebuttal import phase_b_literature as literature
from autoformalism.rebuttal import phase_b_nine_case as nine
from autoformalism.rebuttal import phase_b_nine_case_d3 as d3_family
from autoformalism.rebuttal import phase_b_nine_case_spec_sindy as spec_sindy

SHORT = {
    "phase_b_dalla_man_t1_canonical_named_easy": "T1-easy canonical named",
    "phase_b_anonymous_system_t1_canonical_obfuscated_easy": (
        "T1-easy canonical obfuscated"
    ),
    "phase_b_dalla_man_t1_perturbed_named_easy": "T1-easy perturbed named",
    "phase_b_anonymous_system_t1_perturbed_obfuscated_easy": (
        "T1-easy perturbed obfuscated"
    ),
    "phase_b_dalla_man_t1_canonical_named_hard": "T1-hard canonical named",
    "phase_b_dalla_man_t2_canonical_named_easy": "T2-easy canonical named",
    "phase_b_dalla_man_t2_canonical_named_hard": "T2-hard canonical named",
    "phase_b_cstr_controlled_reactor_mechanism_canonical_named_easy": (
        "CSTR easy named"
    ),
    "phase_b_alien_device_unknown_device_mechanism_canonical_functional_easy": (
        "Alien device easy functional"
    ),
}


def cases() -> tuple[str, ...]:
    """The nine cases, in the inputs record's order."""
    return tuple(nine.load_inputs().cells)


def task_score(row: Mapping[str, Any]) -> float | None:
    """One task's validation NMSE; +inf for the method's failure; None if pending."""
    status = row["status"]
    if status == "pending":
        return None
    protocol = row.get("protocol")
    if protocol == d3_family.PROTOCOL:
        if status != "complete":
            return math.inf
        rollout = row["evaluation"]["phase_b_rollout"]
        value = rollout.get("normalized_mse")
        return float(value) if rollout["status"] == "complete" else math.inf
    if protocol == spec_sindy.PROTOCOL:
        if status != "complete":
            return math.inf
        return float(row["validation_normalized_mse"])
    raise ValueError(f"unknown task protocol {protocol!r}")


def median_and_deviation(values: Sequence[float]) -> tuple[float, float]:
    """The median and unscaled median absolute deviation; +inf counts."""
    array = np.asarray(values, dtype=float)
    center = float(np.median(array))
    if not math.isfinite(center):
        return center, math.nan
    return center, float(np.median(np.abs(array - center)))


def method_scores(summary: Mapping[str, Any]) -> dict[str, list[float | None]]:
    """Each case's task scores, in repetition order."""
    scores: dict[str, list[tuple[int, float | None]]] = {}
    for row in summary["rows"]:
        scores.setdefault(row["benchmark_id"], []).append(
            (row["repetition"], task_score(row))
        )
    return {
        case: [value for _, value in sorted(values, key=lambda item: item[0])]
        for case, values in scores.items()
    }


def summarize_method(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Per-case medians and the aggregate over cases for one method's run."""
    scores = method_scores(summary)
    per_case = {}
    for case, values in scores.items():
        finished = [value for value in values if value is not None]
        center, deviation = (
            median_and_deviation(finished) if finished else (None, None)
        )
        per_case[case] = {
            "scores": values,
            "median": center,
            "deviation": deviation,
            "complete": len(finished) == len(values),
        }
    medians = [item["median"] for item in per_case.values()]
    complete = all(item["complete"] for item in per_case.values())
    tasks = [value for values in scores.values() for value in values]
    finite = [value for value in tasks if value is not None and math.isfinite(value)]
    aggregate = (
        median_and_deviation(medians) if complete and medians else (None, None)
    )
    return {
        "per_case": per_case,
        "median": aggregate[0],
        "deviation": aggregate[1],
        "cases": len(per_case),
        "complete": complete,
        "finite_share": len(finite) / len(tasks) if tasks else None,
    }


def summarize_literature(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each literature model's selected validation NMSE, per case."""
    models: dict[str, dict[str, Any]] = {}
    for row in summary["rows"]:
        column = "bergman" if row["model"] == "bergman" else "literature"
        value = row["validation_nmse"]
        models.setdefault(column, {})[row["benchmark_id"]] = {
            "median": math.inf if value is None and row["complete"] else value,
            "unfitted": row.get("unfitted_validation_nmse"),
            "model": row["model"],
            "label": row["label"],
            "complete": row["complete"],
        }
    return models


def table(
    runs: Mapping[str, Mapping[str, Any]],
    literature_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The comparison: methods' per-case medians and their aggregates."""
    methods = {name: summarize_method(summary) for name, summary in runs.items()}
    literature_models = (
        summarize_literature(literature_summary) if literature_summary else {}
    )
    for name, per_case in literature_models.items():
        medians = [item["median"] for item in per_case.values()]
        complete = all(item["complete"] for item in per_case.values())
        center, deviation = (
            median_and_deviation(medians) if complete else (None, None)
        )
        methods[name] = {
            "per_case": per_case,
            "median": center,
            "deviation": deviation,
            "cases": len(per_case),
            "complete": complete,
            "finite_share": None,
        }
    return {
        "metric": "validation free-rollout NMSE; failures as +inf",
        "aggregation": "median over repetitions per case; median and unscaled "
        "median absolute deviation over cases",
        "cases": list(cases()),
        "methods": methods,
        "not_applicable": literature.NOT_APPLICABLE,
        "test_data_opened": False,
    }


def _cell(value: float | None, deviation: float | None = None) -> str:
    if value is None:
        return "pending"
    if math.isinf(value):
        return "failed"
    text = f"{value:.3g}" if value < 1e4 else f"{value:.2e}"
    if deviation is not None and not math.isnan(deviation):
        spread = f"{deviation:.3g}" if deviation < 1e4 else f"{deviation:.2e}"
        text += f" [{spread}]"
    return text


def markdown(result: Mapping[str, Any]) -> str:
    """The comparison as a Markdown table."""
    names = list(result["methods"])
    lines = [
        "| Case | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for case in result["cases"]:
        cells = []
        for name in names:
            item = result["methods"][name]["per_case"].get(case)
            if item is None:
                cells.append("n/a")
            else:
                cells.append(_cell(item["median"], item.get("deviation")))
        lines.append(f"| {SHORT.get(case, case)} | " + " | ".join(cells) + " |")
    aggregates = []
    for name in names:
        method = result["methods"][name]
        cell = _cell(method["median"], method["deviation"])
        aggregates.append(f"**{cell}** ({method['cases']} cases)")
    lines.append("| **Median over cases [MAD]** | " + " | ".join(aggregates) + " |")
    shares = []
    for name in names:
        share = result["methods"][name]["finite_share"]
        shares.append("-" if share is None else f"{share:.0%}")
    lines.append("| Tasks with a finite score | " + " | ".join(shares) + " |")
    return "\n".join(lines) + "\n"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
