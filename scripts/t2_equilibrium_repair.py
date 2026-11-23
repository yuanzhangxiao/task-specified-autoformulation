"""Test a fasting equilibrium and first-order insulin recovery on certified parents."""

from __future__ import annotations

import argparse
import json
from math import isfinite, sqrt
from pathlib import Path

import numpy as np

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.affine_direction_certificate import certify
from autoformalism.research.preproposal_cases import T2
from autoformalism.research.trajectory_function_fitting import (
    _check_grid,
    _rows,
    score_formula,
)
from autoformalism.research.trajectory_function_revision import assess_fixed
from autoformalism.search.topology_compatibility import TrainingCompatibilityEvidence
from autoformalism.staged_topology import content_hash
from scripts import trajectory_boundary_trial as source
from scripts import trajectory_functions as trial
from scripts.construction_functions import file_hash

DRUG = "insulin_pmol_per_kg_min"
MEAL = "meal_rate_g_per_min"
ARMS = ("bounded", "equilibrium", "restoring")
RHO = "recovery_rho"
# The two affine-in-Gt expressions identified by the direction certificate.
# Preserve the saturating transformation of the administered input in slot 015.
GTERM = "((Gt-(139.399493762))/23.3690639615)"
UTERM = f"tanh((({DRUG}-(0.00109011627907))/0.019502624033))"
LAWS = {
    2: ("afcoef5", "afcoef6", None, f"afcoef5+afcoef6*{GTERM}"),
    15: ("afcoef6", "afcoef7", "afcoef8", f"afcoef6+afcoef7*{GTERM}+afcoef8*{UTERM}"),
}


def candidates(
    slot: int,
    original: dict,
    bounded: dict,
    params: dict,
    scale: float,
    reference: dict,
) -> dict:
    """Keep warm coefficients, eliminate the free intercept and restore to I_fast."""
    intercept, slope_g, slope_u, signature = LAWS[slot]
    if (
        original["equations"]["I"] != signature
        or source.boundary(original, "I", scale) != bounded
        or params.keys() != bounded["parameters"].keys()
        or RHO in params
    ):
        raise ValueError("The saved equation differs from the qualified source law")
    i_fast, g_fast = float(reference["I"]), float(reference["Gt"])
    u_fast = float(reference[DRUG]) if slope_u else 0.0
    if not all(isfinite(v) for v in (i_fast, g_fast, u_fast)) or i_fast <= 0:
        raise ValueError("Require finite fasting references and positive fasting I")
    warm = bounded | {"parameters": params}
    equilibrium = f"{slope_g}*((Gt-({g_fast!r}))/23.3690639615)"
    if slope_u:
        at_ref = f"tanh((({u_fast!r}-(0.00109011627907))/0.019502624033))"
        equilibrium += f"+{slope_u}*({UTERM}-{at_ref})"
    remaining = {n: v for n, v in params.items() if n != intercept}
    models = {"bounded": warm}
    for arm in ARMS[1:]:
        f = equilibrium
        starts = remaining
        if arm == "restoring":
            f += f"+{RHO}**2*(({i_fast!r})-I)"
            starts = remaining | {RHO: sqrt(1 / 60)}
        plain = {
            **warm,
            "equations": warm["equations"] | {"I": f},
            "parameters": starts,
        }
        models[arm] = source.boundary(plain, "I", scale)
    return models


def _code() -> dict:
    from scripts import audit_affine_direction

    return {
        "driver": file_hash(Path(__file__)),
        "direction_audit": file_hash(Path(audit_affine_direction.__file__)),
        "source_trial": file_hash(Path(source.__file__)),
        "runtime": source._code(),
    }


def prepare(campaign: Path, boundary: Path, root: Path) -> tuple[dict, list]:
    """Bind true training certificates to the exact fitted bounded parent vectors."""
    from scripts.audit_affine_direction import audit

    if any(root.resolve().is_relative_to(p.resolve()) for p in (campaign, boundary)):
        raise ValueError("Use a result directory outside previous experiments")
    _, intake, _, _ = source._sources(campaign)
    bp = sealed_read(boundary / "plan.json")
    ranking = sealed_read(boundary / "ranking.json")
    rows = intake["train"][T2]
    scale = source.training_scale(rows, "I")
    if (
        bp["source_plan"] != source.ARCHIVED_PLAN
        or bp["source_finalists"] != source.FINALISTS
        or ranking["plan"] != bp["artifact_sha256"]
        or bp["scale"] != scale
    ):
        raise ValueError("Bounded source fit differs from the registered T2 cohort")
    certificate = audit(campaign, boundary, tolerance=1.0)
    witnesses = {
        r["slot"]: r
        for r in certificate["rows"]
        if r["status"] == "conflict" and r["arm"] == "bounded"
    }
    if set(witnesses) != set(LAWS):
        raise ValueError("Expected exactly two direction-certified bounded arm models")
    models = []
    shared = {}
    for slot in sorted(LAWS):
        m = next(
            i
            for i, record in enumerate(bp["models"])
            if record["source_slot"] == slot and record["arm"] == "bounded"
        )
        model = bp["models"][m]
        parent = next(
            r for r in bp["models"] if r["source_slot"] == slot and r["arm"] == "parent"
        )
        fit = sealed_read(boundary / "fits" / f"{m:02}" / "result.json")
        if (
            ranking["fit_receipts"][m] != fit["artifact_sha256"]
            or fit["plan"] != ranking["plan"]
            or fit["index"] != m
            or fit["selected_option"] is None
            or model["parent_fit"] != parent["parent_fit"]
        ):
            raise ValueError(
                "Bounded parameter checkpoint differs from its training selection"
            )
        retained = fit["options"][fit["selected_option"]]
        if (
            retained["status"] != "fitted"
            or retained["graph_sha256"] != content_hash(model["graph"])
            or retained["formula_sha256"] != content_hash(model["formula"])
            or retained["negative_samples"]["I"] != 0
        ):
            raise ValueError(
                "Expected a verified bounded parent with observed nonnegative insulin"
            )
        witness = witnesses[slot]
        ref = witness["reference"]
        choices = {
            "event": ref["event"],
            "I": ref["initial_target"],
            **ref["forcing_values"],
        }
        if any(k in shared and shared[k] != v for k, v in choices.items()):
            raise ValueError(
                "The parents do not have a common certified fasting reference"
            )
        shared.update(choices)
        formulas = candidates(
            slot,
            parent["formula"],
            model["formula"],
            retained["parameters"],
            scale,
            choices,
        )
        g, arm, _ = trial.slots(intake)[slot]
        if arm != "algorithm":
            raise ValueError("Expected an original algorithm candidate")
        frozen = trial.training_case(intake, intake["graphs"][g])
        evidence = TrainingCompatibilityEvidence.model_validate(frozen["evidence"])
        for arm in ARMS:
            checked = assess_fixed(model["graph"], frozen, formulas[arm])
            if checked["status"] != "eligible":
                raise ValueError(f"Slot {slot} {arm} admission: {checked['reason']}")
            direction = certify(
                model["graph"], checked["formula"], "I", evidence, tolerance=1.0
            )
            models.append(
                {
                    "source_slot": slot,
                    "arm": arm,
                    "formula": checked["formula"],
                    "graph": model["graph"],
                    "source_fit": fit["artifact_sha256"],
                    "certificate": content_hash(witness),
                    "direction_check": direction,
                }
            )
    return sealed_write(
        root / "plan.json",
        {
            "protocol": "t2-equilibrium-repair-1",
            "source_ranking": ranking["artifact_sha256"],
            "source_plan": source.ARCHIVED_PLAN,
            "budget": source.BUDGET,
            "scale": scale,
            "fasting_reference": shared,
            "certificates": {str(s): w for s, w in witnesses.items()},
            "code": _code(),
            "models": models,
            "arms": list(ARMS),
            "restoring": (
                "Squared clearance gain, initially 1/60 per minute. The equilibrium "
                "arm independently fits the zero-gain submodel."
            ),
            "training_selection": (
                "Retain the best verified vector per arm. For the nested choice "
                "retain the lower training NMSE of equilibrium and restoration."
            ),
            "controls": (
                "Evaluate all predeclared controls, including those with a remaining "
                "direction conflict. A common equilibrium applies at the common "
                "training forcing only."
            ),
            "training_intake": trial.TRAIN,
            "validation_intake": trial.VALIDATION,
            "test_data_opened": False,
        },
    ), rows


def train(campaign: Path, boundary: Path, root: Path) -> dict:
    """Fit the matched candidates and freeze their order for each parent."""
    with _lock(root):
        plan, rows = prepare(campaign, boundary, root)
        fits = [source._fit(root, plan, i, rows) for i in range(len(plan["models"]))]
        groups = {}
        for i, (m, r) in enumerate(zip(plan["models"], fits, strict=True)):
            group = groups.setdefault(str(m["source_slot"]), [])
            if (j := r["selected_option"]) is not None:
                group.append((r["options"][j]["training_nmse"], i))
        order = {k: [i for _, i in sorted(v)] for k, v in groups.items()}
        nested = {
            k: next((i for i in o if plan["models"][i]["arm"] != "bounded"), None)
            for k, o in order.items()
        }
        return sealed_write(
            root / "ranking.json",
            {
                "plan": plan["artifact_sha256"],
                "fit_receipts": [r["artifact_sha256"] for r in fits],
                "training_order": order,
                "nested_choice": nested,
                "validation_opened": False,
            },
        )


def behavior(curves: list, rows: list, reference_event: str) -> dict:
    """Summarize I peaks, final-hour errors and drift on unforced trajectories."""
    by_id = {r["trajectory_id"]: r for r in rows}
    records = []
    for curve in curves:
        row = by_id[curve["trajectory_id"]]
        t = np.asarray(curve["time"])
        y = np.asarray(curve["observed"]["I"])
        p = np.asarray(curve["predicted"]["I"])
        tail = t >= t[-1] - 60
        fasting = all(np.all(np.asarray(row["inputs"][n]) == 0) for n in (MEAL, DRUG))
        record = {
            "event": row["trajectory_id"],
            "fasting": bool(fasting),
            "final_error_pmol_L": float(p[-1] - y[-1]),
            "last_hour_mae_pmol_L": float(np.mean(np.abs(p[tail] - y[tail]))),
            "peak_time_error_min": float(t[np.argmax(p)] - t[np.argmax(y)])
            if np.ptp(y) > 1.0
            else None,
            "peak_height_error_pmol_L": float(p.max() - y.max())
            if np.ptp(y) > 1.0
            else None,
            "max_drift_from_initial_pmol_L": float(np.max(abs(p - y[0])))
            if fasting
            else None,
            "observed_max_drift_pmol_L": float(np.max(abs(y - y[0])))
            if fasting
            else None,
        }
        records.append(record)
    peaks = [
        abs(r["peak_time_error_min"])
        for r in records
        if r["peak_time_error_min"] is not None
    ]
    drifts = [r["max_drift_from_initial_pmol_L"] for r in records if r["fasting"]]
    return {
        "events": records,
        "reference_event_prediction_drift_pmol_L": next(
            (
                r["max_drift_from_initial_pmol_L"]
                for r in records
                if r["event"] == reference_event
            ),
            None,
        ),
        "median_abs_peak_time_error_min": float(np.median(peaks)) if peaks else None,
        "median_last_hour_mae_pmol_L": float(
            np.median([r["last_hour_mae_pmol_L"] for r in records])
        ),
        "maximum_unforced_drift_pmol_L": max(drifts, default=None),
    }


def evaluate(
    root: Path, *, campaign: Path | None = None, validation: Path | None = None
) -> dict:
    if (campaign is None) == (validation is None):
        raise ValueError(
            "Supply an original campaign for training or validation.json for validation"
        )
    split = "train" if campaign else "val"
    with _lock(root):
        plan = sealed_read(root / "plan.json")
        rank = sealed_read(root / "ranking.json")
        if rank["plan"] != plan["artifact_sha256"] or plan["code"] != _code():
            raise ValueError("Training freeze differs")
        fits = []
        for i, digest in enumerate(rank["fit_receipts"]):
            fit = sealed_read(root / "fits" / f"{i:02}" / "result.json")
            if (
                fit["artifact_sha256"] != digest
                or fit["plan"] != rank["plan"]
                or fit["index"] != i
            ):
                raise ValueError("Fit differs from the training freeze")
            fits.append(fit)
        if len(fits) != len(plan["models"]):
            raise ValueError("Incomplete fitting cohort")
        if campaign:
            _, intake, _, _ = source._sources(campaign)
            rows = intake["train"][T2]
        else:
            if validation.name != "validation.json":
                raise ValueError("Expected registered validation.json")
            held = sealed_read(validation)
            if (
                held["artifact_sha256"] != plan["validation_intake"]
                or held["training_intake_sha256"] != plan["training_intake"]
            ):
                raise ValueError("Unexpected validation cohort")
            rows = held["rows"][T2]
        records = []
        for i, (m, r) in enumerate(zip(plan["models"], fits, strict=True)):
            path = root / split / f"{i:02}.json"
            identity = {"ranking": rank["artifact_sha256"], "index": i, "split": split}
            if path.exists():
                receipt = sealed_read(path)
                if any(receipt[k] != v for k, v in identity.items()):
                    raise ValueError("Rollout checkpoint differs from training freeze")
            else:
                j = r["selected_option"]
                result = (
                    r["options"][j] if j is not None else {"status": "no_verified_fit"}
                )
                if j is None:
                    score = result
                elif campaign:
                    parsed = _rows(rows, "train")
                    names = sorted(parsed[0].targets)
                    scales = np.array([result["scales"][n] for n in names])
                    score = _check_grid(
                        m["graph"],
                        m["formula"],
                        parsed,
                        result["parameters"],
                        names,
                        scales,
                        curves=True,
                    )
                else:
                    score = score_formula(m["graph"], m["formula"], result, rows)
                receipt = sealed_write(
                    path,
                    identity
                    | {
                        "source_slot": m["source_slot"],
                        "arm": m["arm"],
                        "score": score,
                        "insulin_behavior": behavior(
                            score["curves"], rows, plan["fasting_reference"]["event"]
                        )
                        if score["status"] == "scored"
                        else None,
                        "restoring_gain_per_min": result["parameters"][RHO] ** 2
                        if j is not None and m["arm"] == "restoring"
                        else None,
                    },
                )
            metrics = receipt["score"]
            records.append(
                {
                    "slot": m["source_slot"],
                    "arm": m["arm"],
                    "evaluations": r["evaluations"],
                    "direction_check": m["direction_check"]["status"],
                    **source._metrics(metrics, "pooled_nmse"),
                    "behavior": {
                        k: v
                        for k, v in (receipt["insulin_behavior"] or {}).items()
                        if k != "events"
                    },
                    "restoring_gain_per_min": receipt["restoring_gain_per_min"],
                    "receipt": receipt["artifact_sha256"],
                }
            )
        return sealed_write(
            root / f"{split}-summary.json",
            {
                "split": split,
                "ranking": rank["artifact_sha256"],
                "rows": records,
                "nested_choice": rank["nested_choice"],
                "test_data_opened": False,
                "llm_calls": 0,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="action", required=True)
    for action in ("train", "validate"):
        sub = command.add_parser(action)
        sub.add_argument("--root", required=True, type=Path)
        for flag in ("campaign", "boundary") if action == "train" else ("validation",):
            sub.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    if args.action == "train":
        train(args.campaign, args.boundary, args.root)
        result = evaluate(args.root, campaign=args.campaign)
    else:
        result = evaluate(args.root, validation=args.validation)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
