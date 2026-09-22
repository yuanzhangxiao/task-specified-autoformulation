#!/usr/bin/env python3
"""Inspect all saved T1 brief-only endpoints and replay frozen canonical probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.candidate import CandidateModel
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit
from scripts.probe_t1_full_interventions import adapt_channels, copy_sealed
from scripts.probe_t1_interventions import effect_error
from scripts.replay_t1_curves import (
    CELLS,
    PLAN_HASH,
    SETTINGS,
    archive_inputs,
    replay_model,
    replay_one,
    write_csv,
)

TASKS = tuple(
    f"cell{cell:02d}_seed{seed}_brief_only" for cell in range(4) for seed in (0, 1)
)


def source_identity() -> dict:
    """Bind this analysis and its reusable evaluator to the saved checkpoints."""
    return {
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "settings": SETTINGS.model_dump(mode="json"),
        "scripts": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "replay_t1_brief_only.py",
                "replay_t1_curves.py",
                "probe_t1_full_interventions.py",
                "probe_t1_interventions.py",
            )
        },
    }


def selected_job(result: dict, task: dict) -> dict | None:
    """Use the retained fit, rejecting a changed task, lowering or parameter set."""
    if result["task"] != task or result["round"] != 3:
        raise ValueError("checkpoint task or round differs")
    selected = result["selected"]
    if selected is None:
        return None
    model, _, _ = public._lower(PublicFitRequest.model_validate(selected["request"]))
    candidate = model.validated.candidate.model_dump(mode="json")
    fit = selected["fit"]
    if public.content_sha256(candidate) != fit["lowered_candidate_sha256"]:
        raise ValueError("lowering changed the fitted model")
    if set(model.parameter_names) != set(fit["parameters"]):
        raise ValueError("saved parameter set differs from compiled model")
    return {
        "model_id": task["task_id"],
        "cell": task["cell"],
        "method": task["arm"],
        "seed": task["seed"],
        "candidate": candidate,
        "context": model.validated.context.model_dump(mode="json"),
        "parameters": fit["parameters"],
        "initials": {},
        "saved_training_nmse": fit["training"]["normalized_mse"],
        "saved_validation_nmse": fit["validation"]["normalized_mse"],
        "source_result_sha256": result["artifact_sha256"],
        "origin_round": selected["origin_round"],
        "certificate": selected["certificate"],
    }


def freeze(archive: Path, probe: Path, output: Path) -> dict:
    """Freeze all eight supplied lineages before any new outcome is calculated."""
    archive_inputs(
        archive,
        output / "inputs",
        (
            "plan.json",
            *(f"results/{task}/round_03/result.json" for task in TASKS),
        ),
    )
    source = sealed_read(output / "inputs/plan.json")
    if source["artifact_sha256"] != PLAN_HASH:
        raise ValueError("unexpected historical plan")
    models, unavailable = [], []
    for task_id in TASKS:
        task = next(t for t in source["tasks"] if t["task_id"] == task_id)
        result = sealed_read(
            output / "inputs/results" / task_id / "round_03/result.json"
        )
        job = selected_job(result, task)
        if job is None:
            unavailable.append({"model_id": task_id, "reason": "no retained endpoint"})
        else:
            models.append(job)
    prior = sealed_read(probe / "plan.json")
    if prior["protocol"] != "t1-exploratory-intervention-1":
        raise ValueError("expected the original canonical diagnostic suite")
    imports = {}
    for case in prior["cases"]:
        name = f"references/{case['id']}.json"
        imports[name] = copy_sealed(probe / name, output / name)
    plan = sealed_write(
        output / "plan.json",
        {
            "protocol": "t1-brief-only-inspection-1",
            "identity": source_identity(),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "source_plan_sha256": PLAN_HASH,
            "probe_plan_sha256": prior["artifact_sha256"],
            "imported_artifacts": imports,
            "models": models,
            "unavailable": unavailable,
            "cases": prior["cases"],
            "training_scale": prior["training_scale"],
            "intervention_model_ids": [
                j["model_id"] for j in models if j["cell"] in CELLS
            ],
            "primary_model_id": "cell01_seed1_brief_only",
            "scope": (
                "All available supplied brief-only endpoints replay their own "
                "train/validation data. Canonical models additionally receive the "
                "same seven frozen synthetic cases; perturbed models are never "
                "scored against canonical references. "
                "Primary model was identified from existing scores before this replay. "
                "No refitting, original test access, new references or proposer calls."
            ),
        },
    )
    lines = ["# Exact retained brief-only equations and fitted parameters", ""]
    for job in models:
        lines += [
            f"## {job['model_id']}",
            "",
            "```json",
            json.dumps(
                {k: job[k] for k in ("candidate", "parameters", "origin_round")},
                indent=2,
            ),
            "```",
            "",
        ]
    (output / "equations.md").write_text("\n".join(lines))
    return plan


def canonical_case_row(row: dict, job: dict) -> dict:
    """Never substitute the canonical physical reference for a perturbed cell."""
    if job["cell"] not in CELLS:
        raise ValueError("canonical probes require a canonical model")
    return adapt_channels(row, job["cell"])


def export_curves(output: Path, plan: dict) -> None:
    """Export every replay and its per-trajectory score for external plotting."""
    curves, scores = [], []
    for job in plan["models"]:
        for split in ("train", "validation", "diagnostic"):
            folder = output / "replays" / job["model_id"] / split
            for path in sorted(folder.glob("*.json")):
                if path.name.endswith(".started.json"):
                    continue
                r = sealed_read(path)
                common = {
                    "model_id": job["model_id"],
                    "cell": job["cell"],
                    "split": split,
                    "trajectory_id": r["trajectory_id"],
                }
                scores.append({**common, "success": r["success"], "nmse": r["nmse"]})
                for i, time in enumerate(r["time"]):
                    curves.append(
                        {
                            **common,
                            "time_min": time,
                            "reference_Gp": r["observed"][i],
                            "predicted_Gp": r["predicted"][i] if r["success"] else None,
                        }
                    )
    write_csv(output / "curves.csv", curves)
    write_csv(output / "trajectory_scores.csv", scores)


def run(output: Path, workers: int = 1) -> dict:
    """Run unchanged saved candidates and preserve all failures and all outcomes."""
    with public._lock(output):
        plan = sealed_read(output / "plan.json")
        if plan["identity"] != source_identity():
            raise ValueError("analysis source or runtime changed")
        source = sealed_read(output / "inputs/plan.json")
        if source["artifact_sha256"] != plan["source_plan_sha256"]:
            raise ValueError("source data changed")
        for path, digest in plan["imported_artifacts"].items():
            if sealed_read(output / path)["artifact_sha256"] != digest:
                raise ValueError("reference changed")
        args = [(j, source["cells"][j["cell"]], output) for j in plan["models"]]
        if workers == 1:
            models = [replay_model(arg) for arg in args]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                models = list(executor.map(replay_model, args))
        diagnostics = []
        for job in plan["models"]:
            if job["model_id"] not in plan["intervention_model_ids"]:
                continue
            model = compile_candidate(
                CandidateModel.model_validate(job["candidate"]),
                ValidationContext.model_validate(job["context"]),
            )
            records = {}
            for case in plan["cases"]:
                row = canonical_case_row(
                    sealed_read(output / "references" / f"{case['id']}.json")["row"],
                    job,
                )
                trajectory = public.unpack_split(
                    PublicSplit.model_validate(
                        {
                            "name": "val",
                            "fingerprint": public.content_sha256(row),
                            "rows": [row],
                        }
                    )
                ).trajectories[0]
                records[case["id"]] = replay_one(
                    model,
                    trajectory,
                    job["parameters"],
                    job["initials"],
                    job["context"]["targets"][0],
                    plan["training_scale"],
                    output
                    / "replays"
                    / job["model_id"]
                    / "diagnostic"
                    / f"{case['id']}.json",
                )
            for case in plan["cases"]:
                r = records[case["id"]]
                diagnostics.append(
                    {
                        "model_id": job["model_id"],
                        "case_id": case["id"],
                        "success": r["success"],
                        "nmse": r["nmse"],
                        "effect": effect_error(r, records[case["paired_control"]])
                        if case["paired_control"]
                        else None,
                    }
                )
        export_curves(output, plan)
        return sealed_write(
            output / "summary.json",
            {
                "plan_sha256": plan["artifact_sha256"],
                "status": "complete",
                "models": models,
                "diagnostics": diagnostics,
                "unavailable": plan["unavailable"],
                "scope": plan["scope"],
                "parameter_fitting_performed": False,
                "test_data_opened": False,
                "live_llm_calls": 0,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 9))
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.archive is None or args.probe is None:
            parser.error("freeze requires --archive and --probe")
        freeze(args.archive, args.probe, args.output)
    else:
        run(args.output, args.workers)
