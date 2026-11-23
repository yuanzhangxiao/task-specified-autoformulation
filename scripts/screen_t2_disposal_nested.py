"""Test incremental tissue-glucose terms in the intact slot-015 disposal balance."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import screen_t2_disposal_revisions as first

OUT = Path("output/t2-disposal-nested-20261123-v1")
ORIGINAL_FREEZE = "92440ffe919680c049a91e867d5d99b204d6fb94e5220f2e9ab3db8270ca3791"
ORIGINAL_CODE = "d61f6adc61d61cbccd677f67611421c063bceb94519dd2e608574dd0c1b645d7"


def menu(parent: dict) -> list[first.Family]:
    """Retain the full parent when the single new gain is zero."""
    controls = first.menu(parent)[:2]
    rhs = parent["formula"]["equations"]["U"]
    linear = "((Gt-100)/100)"
    saturating = "(max(0,Gt)/(100+max(0,Gt))-0.5)"
    action = "(I_act-afcoef15)"  # Fixed offset in the existing action initializer.
    alternatives = {
        "add_tissue": linear,
        "add_tissue_action": action + "*" + linear,
        "add_saturating_tissue_action": action + "*" + saturating,
    }
    return controls + [
        first.Family(
            tag,
            rhs + "+disp_tissue*" + term,
            ("afcoef11", "afcoef13", "disp_tissue"),
            (),
            ("afcoef13",),
        )
        for tag, term in alternatives.items()
    ]


def minimum_slope(name: str, params: dict, tissue: np.ndarray) -> float:
    """Analytic minimum of dU/dI_act on the supplied tissue-glucose range."""
    if not len(tissue) or not np.all(np.isfinite(tissue)):
        raise ValueError("A finite tissue glucose series is required")
    endpoints = np.array([min(tissue), max(tissue)])
    if name == "add_tissue_action":
        h = (endpoints - 100) / 100
    elif name == "add_saturating_tissue_action":
        g = np.maximum(0.0, endpoints)
        h = g / (100 + g) - 0.5
    elif name in {"incumbent", "refit_existing", "add_tissue"}:
        h = np.zeros(2)
    else:
        raise ValueError("Unknown nested family")
    return float(min(params["afcoef13"] + params.get("disp_tissue", 0) * h))


def admissibility(name: str, candidate: dict, rows: list[dict]) -> dict:
    """Check the fitted insulin-action slope even when the modifying gain is signed."""
    p = candidate["model"]["formula"]["parameters"]
    gt = {r["trajectory_id"]: np.asarray(r["auxiliaries"]["Gt"]) for r in rows}
    all_gt = np.concatenate(list(gt.values()))
    folds = {}
    for event, tissue in gt.items():
        fit = p | candidate["fold_coefficients"][event]
        folds[event] = minimum_slope(name, fit, tissue)
    return {
        "full_min_slope": minimum_slope(name, p, all_gt),
        "held_min_slope": min(folds.values()),
        "held_event_min_slopes": folds,
    }


def select(candidates: dict) -> tuple[str, dict]:
    """Compare structural extensions to both the exact and matched-fit controls."""
    reasons = {}
    for name, candidate in candidates.items():
        if name in {"incumbent", "refit_existing"}:
            continue
        failures = []
        for split, slope in (
            ("training", "full_min_slope"),
            ("held_training", "held_min_slope"),
        ):
            evaluated = candidate[split]
            if candidate["action"][slope] <= first.EPS:
                failures.append(split + "_nonpositive_action_slope")
            if evaluated["min_U_minus_Uii"] < -first.EPS or (
                min(evaluated["minima"].values()) < -first.EPS
            ):
                failures.append(split + "_negative_target_or_dependent_disposal")
            for control_name in ("incumbent", "refit_existing"):
                control = candidates[control_name][split]
                if (
                    evaluated["per_target_nmse"]["U"]
                    >= control["per_target_nmse"]["U"] - first.EPS
                ):
                    failures.append(split + "_U_vs_" + control_name)
                if evaluated["pooled_nmse"] >= control["pooled_nmse"] - first.EPS:
                    failures.append(split + "_pooled_vs_" + control_name)
        reasons[name] = failures
    passing = (n for n, errors in reasons.items() if not errors)
    winner = min(
        passing,
        key=lambda n: candidates[n]["held_training"]["pooled_nmse"],
        default="incumbent",
    )
    return winner, reasons


def run(models: Path, training: Path, previous: Path, root: Path) -> dict:
    """Freeze the nested menu and checkpoint training fits with fixed upstream laws."""
    prior = sealed_read(previous)
    archive = sealed_read(models)
    parent = archive["models"]["015-positive_EGP"]["joint"]
    if (
        prior["artifact_sha256"] != ORIGINAL_FREEZE
        or first.file_hash(Path(first.__file__)) != ORIGINAL_CODE
        or archive["artifact_sha256"] != first.MODEL_ARCHIVE
        or prior["parent_model_sha256"] != content_hash(parent)
    ):
        raise ValueError("The prior screen or fitted slot 015 differs")
    rows = first.intake.load_training(training)
    families = menu(parent)
    with _lock(root):
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": "t2-015-nested-coupled-disposal-1",
                "source": first.file_hash(Path(__file__)),
                "source_first_screen": ORIGINAL_CODE,
                "source_first_training_results": ORIGINAL_FREEZE,
                "rollout_kernel": first.file_hash(Path(first.kernel.__file__)),
                "parent_model_sha256": prior["parent_model_sha256"],
                "training_sha256": prior["training_sha256"],
                "training_events": [row["trajectory_id"] for row in rows],
                "families": [
                    {
                        k: list(v) if isinstance(v, tuple) else v
                        for k, v in asdict(f).items()
                    }
                    for f in families
                ],
                "fitting": (
                    "Use four-substep coupled RK4 with frozen upstream laws and Gp "
                    "slope. Fit the intercept, action gain and at most one added term "
                    "using bounded-variable least squares. The added gain is signed. "
                    "Tissue scale is fixed at 100 mg/kg. The action reference is the "
                    "parent initializer intercept. Uii gain stays frozen."
                ),
                "selection": (
                    "Improve both full and conditional held-event U and pooled NMSE "
                    "vs frozen and matched comparators. Require dU/dI_act >1e-8 on "
                    "full and held-event tissue ranges, nonnegative sampled targets "
                    "and U-Uii. Select by held-event pooled NMSE."
                ),
                "validation_opened": False,
                "test_opened": False,
                "llm_calls": 0,
            },
        )
        candidates = {}
        for f in families:
            path = root / "families" / (f.name + ".json")
            if not path.exists():
                if f.name in ("incumbent", "refit_existing"):
                    value = prior["candidates"][f.name]
                else:
                    value = first.assess(parent, f, rows)
                value = value | {"action": admissibility(f.name, value, rows)}
                sealed_write(path, {"plan": plan["artifact_sha256"], "result": value})
            receipt = sealed_read(path)
            if receipt["plan"] != plan["artifact_sha256"]:
                raise ValueError("Nested fit checkpoint belongs to another plan")
            candidates[f.name] = receipt["result"]
        chosen, reasons = select(candidates)
        return sealed_write(
            root / "training-freeze.json",
            {
                "plan": plan["artifact_sha256"],
                "parent_model_sha256": content_hash(parent),
                "training_sha256": prior["training_sha256"],
                "incumbent": "incumbent",
                "control": "refit_existing",
                "selected": chosen,
                "candidates": candidates,
                "exclusions": reasons,
                "validation_opened": False,
                "test_opened": False,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=first.MODELS)
    parser.add_argument("--training", type=Path, default=first.TRAIN)
    parser.add_argument(
        "--previous", type=Path, default=first.OUT / "training-freeze.json"
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    freeze = run(args.models, args.training, args.previous, args.out)
    print(f"Nested training screen retained {freeze['selected']}")


if __name__ == "__main__":
    main()
