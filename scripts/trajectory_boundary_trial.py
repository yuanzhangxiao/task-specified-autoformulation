"""Test a concentration boundary on saved, training-selected T2 equations."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research.preproposal_cases import T2
from autoformalism.research.trajectory_function_fitting import (
    fit_formula,
    score_formula,
)
from autoformalism.research.trajectory_function_revision import (
    assess_fixed,
    assess_revision,
)
from autoformalism.staged_topology import content_hash
from scripts import trajectory_functions as trial
from scripts.construction_functions import file_hash
from scripts.replay_trajectory_function_revisions import (
    ARCHIVED_GENERATION,
    ARCHIVED_PLAN,
)

FINALISTS = "f16cc5e777a10192919995f66789d82f28b83d706710e8390443b5e3063eb94c"
REASON = (
    "The training fit predicts negative concentrations from positive observed "
    "initials. Taper only outward flux as concentration approaches zero."
)
ARMS = ("parent", "bounded")
BUDGET = {"max_evaluations": 1000, "starts": 2, "wall_seconds": 180.0, "seed": 100}


def _code() -> dict:
    return {
        "script": file_hash(Path(__file__)),
        "runtime": runtime_source_hash(),
        "campaign": file_hash(Path(trial.__file__)),
    }


def training_scale(rows: list, target: str) -> float:
    """Calculate the pooled fifth percentile of strictly positive training values."""
    if not rows or any(not r["trajectory_id"].startswith("train_") for r in rows):
        raise ValueError("Require training trajectories")
    values = [np.asarray(r["targets"][target], dtype=float) for r in rows]
    if any(
        x.ndim != 1 or not len(x) or x[0] <= 0 or not np.isfinite(x).all()
        for x in values
    ):
        raise ValueError("Require finite observations and positive observed initials")
    joined = np.concatenate(values)
    return float(np.percentile(joined[joined > 0], 5))


def boundary(formula: dict, target: str, scale: float) -> dict:
    """Preserve positive flux and taper negative flux from full rate to zero."""
    if not isfinite(scale) or scale <= 0:
        raise ValueError("Boundary scale must be finite and positive")
    f = formula["equations"][target]
    bounded = f"max(({f}),0)+min(({f}),0)*min(1,max(0,{target}/({scale!r})))"
    return {**formula, "equations": {**formula["equations"], target: bounded}}


def _sources(campaign: Path) -> tuple:
    plan = sealed_read(campaign / "plan.json")
    intake = sealed_read(campaign / "intake.json")
    generation = trial._frozen_candidates(campaign, plan)
    frozen = sealed_read(campaign / "finalists.json")
    if (
        plan["artifact_sha256"] != ARCHIVED_PLAN
        or intake["artifact_sha256"] != plan["intake_sha256"]
        or intake["source"] != trial.TRAIN
        or generation["artifact_sha256"] != ARCHIVED_GENERATION
        or frozen["artifact_sha256"] != FINALISTS
        or frozen["plan"] != ARCHIVED_PLAN
    ):
        raise ValueError("Expected the original job 2404388 training receipts")
    return plan, intake, generation, frozen


def _best_fits(
    campaign: Path, plan: dict, intake: dict, generation: dict, frozen: dict
) -> dict:
    """Select the lowest verified training NMSE per original T2 algorithm graph."""
    groups = {}
    for i, (g, arm, _) in enumerate(trial.slots(intake)):
        row = intake["graphs"][g]
        if arm != "algorithm" or row["dataset"] != T2:
            continue
        candidate = trial._read_candidate(campaign, plan, i)
        fit = sealed_read(campaign / "fits" / f"{i:03}" / "result.json")
        if (
            candidate["arm"] != arm
            or candidate["graph"] != row["structure_sha256"]
            or candidate["artifact_sha256"] != generation["candidates"][i]
            or fit["artifact_sha256"] != frozen["fit_receipts"][i]
            or fit["index"] != i
            or fit["generation"] != generation["artifact_sha256"]
            or fit["candidate"] != candidate["artifact_sha256"]
        ):
            raise ValueError("Original model or fit identity changed")
        group = groups.setdefault(g, [])
        result = fit["result"]
        if result["status"] == "fitted" and isfinite(result["training_nmse"]):
            if result["formula_sha256"] != content_hash(candidate["formula"]) or result[
                "graph_sha256"
            ] != content_hash(row["graph"]):
                raise ValueError("Fit equations differ from the saved model")
            group.append((result["training_nmse"], i, candidate, fit))
    return {
        g: min(group, key=lambda item: item[:2]) for g, group in groups.items() if group
    }


def prepare(campaign: Path, root: Path) -> tuple[dict, list]:
    """Bind matched models, original fit vectors and the training-only scale."""
    if root.resolve().is_relative_to(campaign.resolve()):
        raise ValueError("Use a separate root outside the original campaign")
    plan, intake, generation, frozen = _sources(campaign)
    rows = intake["train"][T2]
    scale = training_scale(rows, "I")
    models, screening = [], []
    for g, (_, i, candidate, fit) in _best_fits(
        campaign, plan, intake, generation, frozen
    ).items():
        row = intake["graphs"][g]
        graph = row["graph"]
        original = fit["result"]
        active = (
            graph["variable_types"].get("I") == "differential"
            and graph["initializers"].get("I")
            == {"mode": "observed_initial", "channel": "I"}
            and original["negative_samples"].get("I", 0) > 0
        )
        screening.append(
            {
                "slot": i,
                "selected": active,
                "negative_samples": original["negative_samples"],
            }
        )
        if not active:
            continue
        case = trial.training_case(intake, row)
        parent = {**candidate["formula"], "parameters": original["parameters"]}
        for arm in ARMS:
            changed = parent if arm == "parent" else boundary(parent, "I", scale)
            if arm == "bounded" and "I" not in graph["sources"]["I"]:
                audit = assess_revision(graph, case, changed, REASON)
            else:
                audit = assess_fixed(graph, case, changed)
            if audit["status"] != "eligible":
                raise ValueError(
                    f"Slot {i}, {arm}: {audit.get('reason', audit['status'])}"
                )
            models.append(
                {
                    "source_slot": i,
                    "arm": arm,
                    "parent_candidate": candidate["artifact_sha256"],
                    "parent_fit": fit["artifact_sha256"],
                    "graph": audit["graph"],
                    "formula": audit["formula"],
                    "source_patch": audit.get("patch", {}),
                    "source_training_nmse": original["training_nmse"],
                    "source_negative_samples": original["negative_samples"],
                }
            )
    return sealed_write(
        root / "plan.json",
        {
            "protocol": "trajectory-boundary-trial-1",
            "code": _code(),
            "source_plan": ARCHIVED_PLAN,
            "source_finalists": FINALISTS,
            "validation_intake": trial.VALIDATION,
            "training_intake": trial.TRAIN,
            "screening": screening,
            "models": models,
            "budget": BUDGET,
            "target": "I",
            "scale": scale,
            "scale_unit": "pmol/L",
            "scale_rule": "Positive pooled training I, fifth percentile",
            "selection": (
                "Within each arm, retain the lowest verified pooled training NMSE "
                "of the warm initial and refit. Retain the warm initial on ties."
            ),
            "boundary_claim": (
                "At I=0 the continuous RHS is max(F,0). For nonnegative initials, "
                "well-defined forcing and locally Lipschitz state dynamics, I "
                "remains nonnegative while solutions exist. Check numerical "
                "predictions separately."
            ),
            "test_data_opened": False,
        },
    ), rows


def _fit(root: Path, plan: dict, i: int, rows: list) -> dict:
    folder = root / "fits" / f"{i:02}"
    identity = {"plan": plan["artifact_sha256"], "index": i}
    path = folder / "result.json"
    reservation = folder / "reservation.json"
    warm_path = folder / "warm.json"
    if path.exists():
        receipt = sealed_read(path)
        if any(receipt[k] != v for k, v in identity.items()):
            raise ValueError("Boundary fit belongs to a different plan or slot")
        return receipt
    model = plan["models"][i]
    options = []
    if reservation.exists():
        sealed_write(reservation, identity)
        if warm_path.exists():
            cached = sealed_read(warm_path)
            if any(cached[k] != v for k, v in identity.items()):
                raise ValueError("Warm initial belongs to a different plan or slot")
            options.append(cached["result"])
        outcome = "interrupted"
    else:
        sealed_write(reservation, identity)
        budget = plan["budget"]
        started = monotonic()
        warm = fit_formula(
            model["graph"],
            model["formula"],
            rows,
            **(budget | {"max_evaluations": 1, "starts": 1}),
        )
        sealed_write(warm_path, identity | {"result": warm})
        options.append(warm)
        seconds = budget["wall_seconds"] - (monotonic() - started)
        if seconds > 0:
            refined = fit_formula(
                model["graph"],
                model["formula"],
                rows,
                **(
                    budget
                    | {
                        "max_evaluations": budget["max_evaluations"] - 1,
                        "wall_seconds": seconds,
                    }
                ),
            )
            options.append(refined)
        outcome = "assessed"
    verified = [
        j
        for j, r in enumerate(options)
        if r["status"] == "fitted" and isfinite(r["training_nmse"])
    ]
    selected = min(
        verified, key=lambda j: (options[j]["training_nmse"], j), default=None
    )
    return sealed_write(
        path,
        identity
        | {
            "status": outcome,
            "options": options,
            "selected_option": selected,
            "evaluations": sum(r["evaluations"] for r in options),
            "evaluation_count_complete": outcome == "assessed",
        },
    )


def train(campaign: Path, root: Path) -> dict:
    """Freeze both arms and the per-parent training comparison before validation."""
    with _lock(root):
        plan, rows = prepare(campaign, root)
        fits = [_fit(root, plan, i, rows) for i in range(len(plan["models"]))]
        order = {}
        for i, (model, fit) in enumerate(zip(plan["models"], fits, strict=True)):
            group = order.setdefault(str(model["source_slot"]), [])
            j = fit["selected_option"]
            if j is not None:
                group.append((fit["options"][j]["training_nmse"], i))
        return sealed_write(
            root / "ranking.json",
            {
                "plan": plan["artifact_sha256"],
                "fit_receipts": [f["artifact_sha256"] for f in fits],
                "training_order": {
                    k: [i for _, i in sorted(v)] for k, v in order.items()
                },
                "validation_opened": False,
            },
        )


def _metrics(result: dict, key: str) -> dict:
    return {
        name: result.get(name)
        for name in (
            "status",
            key,
            "per_target_nmse",
            "negative_samples",
            "minimum_prediction",
            "maximum_scaled_grid_difference",
        )
    }


def validate(root: Path, validation: Path) -> dict:
    """Score the predeclared six arms at their retained training vectors."""
    with _lock(root):
        plan = sealed_read(root / "plan.json")
        ranking = sealed_read(root / "ranking.json")
        if ranking["plan"] != plan["artifact_sha256"] or _code() != plan["code"]:
            raise ValueError("Boundary trial plan or training freeze differs")
        fits = []
        for i, digest in enumerate(ranking["fit_receipts"]):
            fit = sealed_read(root / "fits" / f"{i:02}" / "result.json")
            if (
                fit["artifact_sha256"] != digest
                or fit["index"] != i
                or fit["plan"] != ranking["plan"]
            ):
                raise ValueError(
                    "Boundary fit does not match frozen training selection"
                )
            fits.append(fit)
        if len(fits) != len(plan["models"]) or validation.name != "validation.json":
            raise ValueError(
                "Expected all frozen models and registered validation.json"
            )
        source = sealed_read(validation)
        if (
            source["artifact_sha256"] != plan["validation_intake"]
            or source["training_intake_sha256"] != plan["training_intake"]
        ):
            raise ValueError("Unexpected validation cohort")
        summary = []
        for i, (model, fit) in enumerate(zip(plan["models"], fits, strict=True)):
            j = fit["selected_option"]
            retained = (
                fit["options"][j] if j is not None else {"status": "no_verified_fit"}
            )
            path = root / "validation" / f"{i:02}.json"
            identity = {"ranking": ranking["artifact_sha256"], "index": i}
            if path.exists():
                record = sealed_read(path)
                if any(record[k] != v for k, v in identity.items()):
                    raise ValueError("Validation receipt differs from training freeze")
            else:
                score = (
                    score_formula(
                        model["graph"], model["formula"], retained, source["rows"][T2]
                    )
                    if j is not None
                    else retained
                )
                record = sealed_write(path, identity | {"score": score})
            summary.append(
                {
                    "slot": model["source_slot"],
                    "arm": model["arm"],
                    "index": i,
                    "evaluations": fit["evaluations"],
                    "evaluation_count_complete": fit["evaluation_count_complete"],
                    "retained": {0: "warm_initial", 1: "refit"}.get(j),
                    "incumbent": _metrics(fit["options"][0], "training_nmse")
                    if fit["options"]
                    else None,
                    "training": _metrics(retained, "training_nmse"),
                    "validation": _metrics(record["score"], "pooled_nmse"),
                    "receipt": record["artifact_sha256"],
                }
            )
        return sealed_write(
            root / "summary.json",
            {
                "ranking": ranking["artifact_sha256"],
                "scale": plan["scale"],
                "rows": summary,
                "llm_calls": 0,
                "test_data_opened": False,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name, arg in (("train", "campaign"), ("validate", "validation")):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    result = (
        train(args.campaign, args.root)
        if args.action == "train"
        else validate(args.root, args.validation)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
