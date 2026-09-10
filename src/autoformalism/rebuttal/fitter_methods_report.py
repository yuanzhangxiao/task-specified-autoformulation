"""Post-fit diagnostics; no functions here supply an optimizer or select a model."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from autoformalism.rebuttal.fitter_diagnostic import write_json


def parameter_equivalence(case: dict, parameters: dict) -> dict:
    """Distance to the known admissible memory-pole swap, not an identifiability test.

    For the fixed hidden fixture q=m+k_p*p has transfer function
    k_u*(s+1/tau_p+k_p)/((s+1/tau)*(s+1/tau_p)). Swapping the two
    poles preserves q and the nonlinear feedback, if the new gain is admissible.
    This diagnostic is called only after fitting and independent replay.
    """
    truth = case["truth"]
    aliases = {"generating": dict(truth)}
    if not case["observed_state"]:
        swapped = {
            **truth,
            "tau": truth["tau_p"],
            "tau_p": truth["tau"],
            "k_p": truth["k_p"] + 1 / truth["tau_p"] - 1 / truth["tau"],
        }
        if swapped["k_p"] >= 0:
            aliases["memory_pole_swap"] = swapped
    distances = {
        name: max(
            abs(parameters[n] - value) / max(1.0, abs(value))
            for n, value in alias.items()
        )
        for name, alias in aliases.items()
    }
    nearest = min(distances, key=distances.get)
    return {
        "diagnostic_only": True,
        "known_admissible_aliases": aliases,
        "nearest_alias": nearest,
        "scaled_linf_distance": distances[nearest],
        "scaled_linf_distances": distances,
        "nearest_absolute_errors": {
            n: abs(parameters[n] - value) for n, value in aliases[nearest].items()
        },
        "complete_equivalence_class_claimed": False,
        "used_for_fitting_or_recovery_gate": False,
    }


def _number(value) -> str:
    return "—" if value is None else f"{value:.6g}"


def write_paired_summary(output: Path, frozen: dict, rows: list[dict]) -> dict:
    """Retain failed outcomes and separate verification from output recovery."""
    names = {
        "forward_sensitivity": "S",
        "collocation_init": "C+FD",
        "collocation_sensitivity": "C+S",
    }
    threshold = frozen["plan"]["reference"]["recovery_nmse"]
    fits, groups, pairs = [], {}, {}
    for row in rows:
        t, r = row["task"], row["result"] or {}
        if t["kind"] != "fit":
            continue
        scores = r.get("clean_signal_nmse", {})
        verified = row["status"] == "complete"
        recovered = verified and all(
            scores.get(s) is not None and scores[s] <= threshold
            for s in ("train", "validation")
        )
        item = {
            "case": t["case"],
            "noise": t["noise_fraction"],
            "replicate": t["replicate"],
            "method": t["method"],
            "status": row["status"],
            "verified": verified,
            "output_recovered": recovered,
            "fallback": r.get("initializer_fallback"),
            "seconds": r.get("total_fit_seconds"),
            "calls": r.get("fit", {}).get("actual_residual_calls"),
            "train_nmse": scores.get("train"),
            "validation_nmse": scores.get("validation"),
            "alias": r.get("parameter_equivalence", {}).get("nearest_alias"),
            "alias_distance": r.get("parameter_equivalence", {}).get(
                "scaled_linf_distance"
            ),
        }
        fits.append(item)
        groups.setdefault((t["case"], t["noise_fraction"], t["method"]), []).append(
            item
        )
        pairs.setdefault(t["pair"], {})[t["method"]] = r
    aggregates = []
    for (case, noise, method), items in groups.items():
        times = [
            r["seconds"] for r in items if r["verified"] and r["seconds"] is not None
        ]
        scores = [r["validation_nmse"] for r in items if r["verified"]]
        aggregates.append(
            {
                "case": case,
                "noise": noise,
                "method": method,
                "planned": len(items),
                "verified": sum(r["verified"] for r in items),
                "output_recovered": sum(r["output_recovered"] for r in items),
                "fallbacks": sum(r["fallback"] is True for r in items),
                "median_seconds_verified": float(np.median(times)) if times else None,
                "worst_validation_nmse_verified": max(scores) if scores else None,
            }
        )
    pairing = []
    for pair, methods in pairs.items():
        fd, sensitivity = (
            methods.get("collocation_init", {}),
            methods.get("collocation_sensitivity", {}),
        )
        a, b = fd.get("initializer", {}), sensitivity.get("initializer", {})
        ready = bool(a and b)
        same = ready and (
            a.get("shared_result_sha256") is not None
            and a["shared_result_sha256"] == b.get("shared_result_sha256")
            and fd.get("refinement_start") == sensitivity.get("refinement_start")
            and fd.get("training_fingerprint")
            == sensitivity.get("training_fingerprint")
        )
        pairing.append(
            {
                "pair": pair,
                "both_fit_records_present": ready,
                "same_data_and_initializer": same if ready else None,
            }
        )
    report = {
        "aggregates": aggregates,
        "fits": fits,
        "pairing_checks": pairing,
        "diagnostic_recovery_threshold": threshold,
        "stage_statuses": {
            kind: dict(Counter(r["status"] for r in rows if r["task"]["kind"] == kind))
            for kind in ("guard", "initializer", "fit")
        },
    }
    write_json(output / "compact.json", report)
    passed = sum(p["same_data_and_initializer"] is True for p in pairing)
    lines = [
        "# Paired collocation and sensitivity comparison",
        "",
        "S = ordinary start + sensitivity; C+FD = collocation + production "
        "finite differences; "
        "C+S = the same collocation start + sensitivity.",
        f"Output recovery: verified clean-reference train AND validation NMSE "
        f"<= {threshold:g}. "
        "Reference scores and alias distances are post-fit diagnostics only.",
        "Seconds include the original initializer charge for each logical fit. "
        "Physical initializer work is shared. Missing/failed runs stay in "
        "denominators.",
        "",
        f"Stage outcomes: `{report['stage_statuses']}`",
        f"Shared-start checks: {passed} "
        f"passed / {len(pairing)} planned pairs; "
        f"{sum(p['same_data_and_initializer'] is False for p in pairing)} mismatches.",
        "",
        "| Case | Noise | Route | Verified | Recovered | Fallbacks | Median "
        "s* | Worst val NMSE* |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for a in aggregates:
        lines.append(
            f"| {a['case']} | {a['noise']} | {names[a['method']]} | "
            f"{a['verified']}/{a['planned']} | "
            f"{a['output_recovered']}/{a['planned']} | "
            f"{a['fallbacks']} | {_number(a['median_seconds_verified'])} | "
            f"{_number(a['worst_validation_nmse_verified'])} |"
        )
    lines += [
        "",
        "*Aggregates use numerically verified runs; inspect every outcome below. "
        "Three replicates are a robustness check, not a population "
        "success-rate estimate.",
        "",
        "| Case | Noise | Rep | Route | Status | Fallback | s | Calls | Train "
        "NMSE | Val NMSE | Alias distance |",
        "| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in fits:
        lines.append(
            f"| {r['case']} | {r['noise']} | {r['replicate']} | {names[r['method']]} | "
            f"{r['status']} | {r['fallback']} | {_number(r['seconds'])} | "
            f"{r['calls']} | "
            f"{_number(r['train_nmse'])} | {_number(r['validation_nmse'])} | "
            f"{_number(r['alias_distance'])} |"
        )
    lines += [
        "",
        "Alias distance is scaled distance to the generating vector or its admissible "
        "memory-pole swap; it is not a recovery gate or a claim of unique parameters. "
        "See compact.json for aliases per fit, details.md for diagnostics, "
        "and summary.json for complete records.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return report
