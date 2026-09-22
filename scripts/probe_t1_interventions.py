#!/usr/bin/env python3
"""Freeze and evaluate an exploratory T1 intervention suite on saved models."""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import dalla_man as reference
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.candidate import CandidateModel
from autoformalism.schemas.public_fitting import PublicSplit
from scripts.replay_t1_curves import replay_one

CELL = "phase_b_anonymous_system_t1_canonical_obfuscated_easy"
PRIMARY = ("obfuscated_raw_data_agent_rep0", "obfuscated_raw_data_agent_rep1")
EXPECTED_MODELS = (*PRIMARY, "cell01_seed0_no_spec")


def source_identity() -> dict:
    """Bind checkpoints to the runtime and both analysis entry points."""
    return {
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "scripts": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("probe_t1_interventions.py", "replay_t1_curves.py")
        },
    }


def cases() -> list[dict]:
    """Predeclare both perturbation directions and a meal-spacing probe."""
    rows = []
    for label, meals in (("fasting", []), ("meal", [[60.0, 60.0]])):
        for multiplier in (1.0, 0.8, 1.2):
            rows.append(
                {
                    "id": f"{label}_gt{round(multiplier * 100)}",
                    "meals": meals,
                    "initial_gt_multiplier": multiplier,
                    "paired_control": f"{label}_gt100" if multiplier != 1 else None,
                }
            )
    rows.append(
        {
            "id": "split_meal_gt100",
            "meals": [[60.0, 30.0], [90.0, 30.0]],
            "initial_gt_multiplier": 1.0,
            "paired_control": "meal_gt100",
        }
    )
    return rows


def freeze(source: Path, output: Path) -> dict:
    """Choose models by recorded training accuracy before generating new outcomes."""
    models = sealed_read(source / "models.json")
    scores = sealed_read(source / "summary.json")
    eligible = [
        r
        for r in scores["models"]
        if r["cell"] == CELL and r["train"]["complete"] and r["train"]["nmse"] < 0.05
    ]
    if {r["model_id"] for r in eligible} != set(EXPECTED_MODELS):
        raise ValueError("unexpected training-eligible model inventory")
    jobs = [
        next(j for j in models["models"] if j["model_id"] == name)
        for name in EXPECTED_MODELS
    ]
    return sealed_write(
        output / "plan.json",
        {
            "protocol": "t1-exploratory-intervention-1",
            "identity": source_identity(),
            "source_models_sha256": models["artifact_sha256"],
            "source_scores_sha256": scores["artifact_sha256"],
            "models": jobs,
            "training_scores": eligible,
            "primary_pair": list(PRIMARY),
            "cases": cases(),
            "training_scale": eligible[0]["training_scale"],
            "duration": 300.0,
            "dt": 1.0,
            "reference_parameters": asdict(reference.DallaManParameters()),
            "reference_solvers": ["Radau", "DOP853"],
            "reference_rtol": 1e-10,
            "reference_atol": 1e-12,
            "reference_max_step": 0.5,
            "agreement_tolerance": 2e-5,
            "initial_condition_policy": (
                "Change initial Gt only; keep Gp and every other physical state fixed. "
                "This changes initial total glucose mass, unlike the registered "
                "mass-conserving Gp/Gt redistribution. It is a new synthetic "
                "initial-condition experiment, not a change to released benchmarks."
            ),
            "design_rationale": (
                "The primary pair has training NMSE 0.0368/0.0412. "
                "One model uses a large contemporaneous tissue-glucose coefficient; "
                "the other includes three meal-memory states. Independent Gt "
                "initialization challenges the learned auxiliary relationship. "
                "Meal splitting checks a new input schedule without changing "
                "initial mass. Neither intervention is guaranteed to favor memory."
            ),
            "evidence_scope": (
                "Exploratory, mechanism-directed case study chosen after inspecting "
                "training fits and equations. No outcome-based case selection. "
                "Not an untouched benchmark test or a causal ablation of memory. "
                "All eligible models and all declared cases are reported."
            ),
            "parameter_refit_applied": False,
            "test_data_opened": False,
            "private_reference_used_for_diagnostic": True,
            "live_llm_calls": 0,
        },
    )


def generate_reference(case: dict, plan: dict, method: str) -> dict:
    """Recompute all public channels from one internally consistent trajectory."""
    parameters = reference.DallaManParameters(**plan["reference_parameters"])
    basal = reference.compute_dalla_man_basal(parameters)
    initial = basal.initial_state.copy()
    initial[reference.STATE_INDEX["Gt"]] *= case["initial_gt_multiplier"]

    def solve(*args, **kwargs):
        kwargs.update(
            method=method,
            rtol=plan["reference_rtol"],
            atol=plan["reference_atol"],
            max_step=plan["reference_max_step"],
        )
        return solve_ivp(*args, **kwargs)

    # Local solver substitution only; the trusted physical equations stay intact.
    with patch.object(reference, "solve_ivp", solve):
        result = reference.simulate_dalla_man(
            meals=tuple(tuple(x) for x in case["meals"]),
            duration=plan["duration"],
            dt=plan["dt"],
            variant="original",
            parameters=parameters,
            initial_state=tuple(initial),
            basal_reference=basal,
        )
    return {
        "trajectory_id": case["id"],
        "time": result.time.tolist(),
        "targets": {"v01": result.states[:, reference.STATE_INDEX["Gp"]].tolist()},
        "auxiliaries": {
            "v02": result.derived["EGP"].tolist(),
            "v03": result.derived["Uii"].tolist(),
            "v04": result.derived["E"].tolist(),
            "v05": result.states[:, reference.STATE_INDEX["Gt"]].tolist(),
        },
        "external_inputs": {"u01": result.meal_event_g.tolist()},
    }


def effect_error(record: dict, control: dict) -> dict:
    """Score paired response differences separately from ordinary trajectory NMSE."""
    if not record["success"] or not control["success"]:
        return {"status": "rollout_failed", "relative_squared_error": None}
    observed = np.asarray(record["observed"]) - control["observed"]
    predicted = np.asarray(record["predicted"]) - control["predicted"]
    signal = float(np.dot(observed, observed))
    informative = signal > 1e-8
    return {
        "status": "complete" if informative else "negligible_reference_effect",
        "relative_squared_error": float(np.sum((predicted - observed) ** 2) / signal)
        if informative
        else None,
        "reference_peak_abs_effect": float(np.max(np.abs(observed))),
        "model_peak_abs_effect": float(np.max(np.abs(predicted))),
        "reference_delta": observed.tolist(),
        "predicted_delta": predicted.tolist(),
    }


def run(output: Path) -> dict:
    """Evaluate the frozen suite, retaining every case and every fitted model."""
    with public._lock(output):
        plan = sealed_read(output / "plan.json")
        if plan["identity"] != source_identity():
            raise ValueError("diagnostic source or runtime changed after freeze")
        refs = {}
        for case in plan["cases"]:
            path = output / "references" / f"{case['id']}.json"
            if path.exists():
                refs[case["id"]] = sealed_read(path)
                continue
            first, second = [
                generate_reference(case, plan, solver)
                for solver in plan["reference_solvers"]
            ]
            discrepancies = {
                channel: float(
                    np.max(np.abs(np.asarray(values) - second[role][channel]))
                )
                for role in ("targets", "auxiliaries")
                for channel, values in first[role].items()
            }
            if max(discrepancies.values()) > plan["agreement_tolerance"]:
                raise ValueError(f"reference solvers disagree: {discrepancies}")
            refs[case["id"]] = sealed_write(
                path,
                {
                    "row": first,
                    "solver_max_differences": discrepancies,
                },
            )
        compiled = {
            job["model_id"]: compile_candidate(
                CandidateModel.model_validate(job["candidate"]),
                ValidationContext.model_validate(job["context"]),
            )
            for job in plan["models"]
        }
        records = {}
        for case in plan["cases"]:
            row = refs[case["id"]]["row"]
            trajectory = public.unpack_split(
                PublicSplit.model_validate(
                    {
                        "name": "val",
                        "fingerprint": public.content_sha256(row),
                        "rows": [row],
                    }
                )
            ).trajectories[0]
            for job in plan["models"]:
                record = replay_one(
                    compiled[job["model_id"]],
                    trajectory,
                    job["parameters"],
                    job["initials"],
                    "v01",
                    plan["training_scale"],
                    output / "replays" / job["model_id"] / f"{case['id']}.json",
                )
                records[(job["model_id"], case["id"])] = record
        rows, curves = [], []
        for case in plan["cases"]:
            for job in plan["models"]:
                record = records[(job["model_id"], case["id"])]
                effect = (
                    effect_error(
                        record, records[(job["model_id"], case["paired_control"])]
                    )
                    if case["paired_control"]
                    else None
                )
                rows.append(
                    {
                        "model_id": job["model_id"],
                        "case_id": case["id"],
                        "success": record["success"],
                        "nmse": record["nmse"],
                        "effect": effect,
                    }
                )
                for i, time in enumerate(record["time"]):
                    curves.append(
                        {
                            "model_id": job["model_id"],
                            "case_id": case["id"],
                            "time_min": time,
                            "reference_Gp": record["observed"][i],
                            "predicted_Gp": record["predicted"][i]
                            if record["success"]
                            else None,
                        }
                    )
        with (output / "curves.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
            writer.writeheader()
            writer.writerows(curves)
        return sealed_write(
            output / "summary.json",
            {
                "plan_sha256": plan["artifact_sha256"],
                "rows": rows,
                "reference_checks": {
                    k: v["solver_max_differences"] for k, v in refs.items()
                },
                "status": "complete",
                "evidence_scope": plan["evidence_scope"],
                "test_data_opened": False,
                "parameter_refit_applied": False,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.source is None:
            parser.error("freeze requires --source")
        freeze(args.source, args.output)
    else:
        run(args.output)
