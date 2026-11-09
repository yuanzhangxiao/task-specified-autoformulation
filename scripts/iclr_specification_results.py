"""Report specification treatments and test an observable implication on training."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts.iclr_anchor_fidelity import VERSION as FIDELITY_VERSION
from scripts.iclr_anchor_fidelity import verify_controlled_bundle
from scripts.iclr_submission_evidence import table

VARIANTS = ("full", "underspecified", "incorrect", "overrestrictive", "no_spec")


def contradiction_pairs(rows: list[dict], excluded: str | None) -> list[dict]:
    """Enumerate identical permitted histories and observed violations of equality."""
    witnesses = []
    for a, b in combinations(rows, 2):
        if (
            a["time"] != b["time"]
            or a["fixed_covariates"] != b["fixed_covariates"]
            or a["auxiliaries"] != b["auxiliaries"]
        ):
            continue
        y, z = (np.asarray(r["targets"]["T"], dtype=float) for r in (a, b))
        if (
            len(y) < 2
            or y.shape != z.shape
            or not all(np.isfinite(v).all() for v in (y, z))
        ):
            raise ValueError("invalid observed temperature pair")
        u, v = a["external_inputs"], b["external_inputs"]
        if y[0] != z[0] or set(u) != set(v) or any(u[k][0] != v[k][0] for k in u):
            continue
        if any(u[k] != v[k] for k in u if k != excluded):
            continue
        delta = z - y
        i = int(np.argmax(abs(delta)))
        # Roundoff allowance for this noiseless export.
        tol = 1e-8 * max(1.0, float(np.max(abs(y))), float(np.max(abs(z))))
        witnesses.append(
            {
                "left": a["trajectory_id"],
                "right": b["trajectory_id"],
                "peak_time_min": a["time"][i],
                "abs_temperature_gap_K": abs(float(delta[i])),
                "comparison_tolerance_K": tol,
                "violates_invariance": bool(abs(delta[i]) > tol),
                "pair_mean_mse_lower_bound_K2": float(np.mean(delta**2) / 4),
            }
        )
    return witnesses


def development(public_root: Path) -> dict:
    """This equality test applies only to the clean, shared-preparation reactor."""
    directory = public_root / "sensor_00"
    files = {}
    for name in ("specification", "train"):
        body = json.loads((directory / (name + ".json")).read_text())
        if public.content_sha256(body["value"]) != body["sha256"]:
            raise ValueError("modified public input")
        files[name] = body
    spec = files["specification"]["value"]
    if (
        spec["targets"] != ["T"]
        or spec["auxiliaries"]
        or set(spec["external_inputs"]) != {"Tf", "Cf", "Tjf"}
        or spec.get("noise_sd_fraction") != 0
        or "same reproducible preparation" not in spec["preparation_contract"]
        or files["train"]["value"]["name"] != "train"
    ):
        raise ValueError(
            "invariance witness requires clean public reactor training data"
        )
    rows = files["train"]["value"]["rows"]
    return {
        "train_sha256": files["train"]["sha256"],
        "public_spec_sha256": files["specification"]["sha256"],
        "excluded_driver": "Tjf",
        "reference_assumption": "common hidden preparation; deterministic trajectories",
        "restricted": contradiction_pairs(rows, "Tjf"),
        "all_input_control": contradiction_pairs(rows, None),
        "criterion": (
            "An excluded input cannot alter the outcome when remaining "
            "inputs and permitted initial information agree."
        ),
        "interpretation": (
            "A discrepancy is a witness under the noiseless contract. "
            "Absence of such a pair is inconclusive."
        ),
    }


def report(
    search: Path, public_root: Path, output: Path, fidelity: Path | None
) -> dict:
    """Join planned arms, any frozen development results, and post-freeze responses."""
    plan = sealed_read(search / "plan.json")
    variants = sealed_read(search / "spec_variants.json")
    if variants["artifact_sha256"] != plan["variants_sha256"]:
        raise ValueError("specification variant manifest differs")
    witness = development(public_root)
    bundles = []
    frozen = search / "frozen_models.json"
    selected = {}
    if frozen.exists():
        bundle = sealed_read(frozen)
        if bundle["protocol"] != "iclr-controlled-search-models-1":
            raise ValueError("expected internal controlled-search export")
        verify_controlled_bundle(frozen, bundle)
        selected = {r["source_id"]: r for r in bundle["rows"]}
        bundles.append(bundle["artifact_sha256"])
    scores = {}
    fidelity_sha = None
    if fidelity:
        cohort = sealed_read(fidelity / "frozen_selection_manifest.json")
        assessment = sealed_read(fidelity / "summary.json")
        scored = sealed_read(fidelity / "scored_rows.json")
        plan_id = cohort["artifact_sha256"]
        roster_hashes = {m["source_bundle_sha256"] for m in cohort["roster"]}
        rows = scored["rows"]
        if (
            not bundles
            or not set(bundles) <= roster_hashes
            or any(
                x["protocol"] != FIDELITY_VERSION for x in (cohort, assessment, scored)
            )
            or cohort["mode"] != "confirmation"
            or scored["mode"] != "confirmation"
            or assessment["source_mode"] != "confirmation"
            or assessment["plan_sha256"] != plan_id
            or scored["plan_sha256"] != plan_id
            or set(scored["source_bundle_hashes"]) != roster_hashes
            or assessment["scored_rows_sha256"] != scored["artifact_sha256"]
            or assessment["metrics_sha256"] != scored["metrics_sha256"]
            or public.content_sha256(rows) != scored["metrics_sha256"]
            or assessment["row_count"] != len(rows)
        ):
            raise ValueError("unbound numerical results or missing confirmatory cohort")
        fidelity_sha = assessment["artifact_sha256"]
        for score in rows:
            index = score["subject"]
            if type(index) is not int or not 0 <= index < len(cohort["roster"]):
                raise ValueError("score refers to an unknown subject")
            model = cohort["roster"][index]
            if any(
                score[k] != model[k]
                for k in (
                    "source_id",
                    "source_bundle_sha256",
                    "method",
                    "repetition",
                    "benchmark_id",
                )
            ) or score["source_benchmark_id"] != model.get(
                "source_benchmark_id", model["benchmark_id"]
            ):
                raise ValueError(
                    "score subject identity differs from the frozen roster"
                )
            if model["source_bundle_sha256"] not in bundles:
                continue
            source = selected.get(model["source_id"])
            if (
                source is None
                or any(
                    source[k] != model[k] for k in ("method", "repetition", "status")
                )
                or source["benchmark_id"] != score["source_benchmark_id"]
            ):
                raise ValueError("fidelity subject differs from the search export")
            scores.setdefault(model["source_id"], []).append(score)
    # Scientific task mutations are only declared for the two hard anchors.
    rows = []
    effects = []
    for block in plan["blocks"]:
        full = next(
            (
                selected.get(t["task_id"])
                for t in block["tasks"]
                if t["treatment"] == "full"
            ),
            None,
        )
        for task in block["tasks"]:
            arm = task["treatment"]
            cell = task["cell"]
            if arm not in VARIANTS:
                continue
            source = task["task_id"]
            model = selected.get(source, {})
            v, base = model.get("validation_nmse"), (full or {}).get("validation_nmse")
            counterexample = (
                cell == "sensor_00"
                and arm == "incorrect"
                and any(r["violates_invariance"] for r in witness["restricted"])
            )
            status = model.get("status", "pending")
            rows.append(
                {
                    "cell": cell,
                    "seed": task["seed"],
                    "arm": arm,
                    "source_id": source,
                    "search_status": status,
                    "validation_nmse": v,
                    "full_validation_nmse": base,
                    "paired_validation_delta": v - base
                    if v is not None and base is not None
                    else None,
                    "empirical_status": "training_invariance_counterexample"
                    if counterexample
                    else "unresolved",
                    "logical_inconsistency": "not_established",
                    "critic_diagnosis": "see search_visits.csv on returned campaign",
                    "independent_outcomes": len(scores.get(source, [])),
                }
            )
            effects.extend(
                {"cell_alias": cell, "arm": arm, "seed": task["seed"], **s}
                for s in scores.get(source, [])
            )
    report = sealed_write(
        output / "report.json",
        {
            "protocol": "iclr-specification-sensitivity-1",
            "search_plan": plan["artifact_sha256"],
            "variant_manifest": variants["artifact_sha256"],
            "frozen_bundles": bundles,
            "confirmatory_assessment": fidelity_sha,
            "outcomes": rows,
            "training_witness": witness,
            "training_diagnostic_supplied_to_proposer": False,
            "scope": (
                "The equality witness uses public training data. "
                "Search failure cannot establish inconsistent physics."
            ),
        },
    )
    table(output / "spec_sensitivity.csv", rows)
    effects_path = output / "independent_spec_effects.csv"
    if effects:
        table(effects_path, effects)
    else:
        effects_path.write_text(
            "cell_alias,arm,seed,source_id,case,quantity,readout,effect_rse,status\n"
        )
    table(output / "reactor_training_pairs.csv", witness["restricted"])
    largest = max(
        (r["abs_temperature_gap_K"] for r in witness["restricted"]), default=0.0
    )
    (output / "failure_cases.md").write_text(
        "# Specification sensitivity\n\n"
        f"The reactor's jacket-independent premise implies equal output on "
        f"{len(witness['restricted'])} matched training pairs. Their largest "
        f"temperature difference is {largest:.3f} K. The CSV contains every pair "
        "and its error bound. Insulin and glucose trajectories alone leave disposal "
        "versus suppressed production unresolved. These diagnostics are computed "
        "outside the search.\n"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--search", required=True, type=Path)
    p.add_argument("--public", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--fidelity", type=Path)
    a = p.parse_args()
    report(a.search, a.public, a.output, a.fidelity)
