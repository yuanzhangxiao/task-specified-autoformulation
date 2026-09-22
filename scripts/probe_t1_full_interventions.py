#!/usr/bin/env python3
"""Extend the saved T1 diagnostic with every available Full endpoint, without refit."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.candidate import CandidateModel
from autoformalism.schemas.public_fitting import PublicSplit
from scripts.probe_t1_interventions import effect_error
from scripts.replay_t1_curves import CELLS, replay_one

FULL_IDS = tuple(
    f"cell{cell:02d}_seed{seed}_full" for cell in (0, 1) for seed in (0, 1)
)
CHANNELS = {
    "targets": {"v01": "Gp"},
    "auxiliaries": {"v02": "EGP", "v03": "Uii", "v04": "E", "v05": "Gt"},
    "external_inputs": {"u01": "meal_event_g"},
}


def source_identity() -> dict:
    """Bind this extension to its evaluator, runtime and analysis dependencies."""
    return {
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "scripts": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "probe_t1_full_interventions.py",
                "probe_t1_interventions.py",
                "replay_t1_curves.py",
            )
        },
    }


def adapt_channels(row: dict, cell: str) -> dict:
    """Rename public channels only; preserve every time and numerical observation."""
    if cell not in CELLS:
        raise ValueError("unsupported T1 cell")
    if any(set(row[role]) != set(names) for role, names in CHANNELS.items()):
        raise ValueError("unexpected diagnostic channels")
    return {
        **row,
        **{
            role: {
                (names[key] if cell == CELLS[0] else key): list(values)
                for key, values in row[role].items()
            }
            for role, names in CHANNELS.items()
        },
    }


def copy_sealed(source: Path, destination: Path) -> str:
    """Copy an authenticated artifact while preserving its original digest."""
    value = sealed_read(source)
    digest = value.pop("artifact_sha256")
    assert sealed_write(destination, value)["artifact_sha256"] == digest
    return digest


def freeze(source: Path, probe: Path, output: Path) -> dict:
    """Import exact reference curves and prior comparators; add all four Full models."""
    models = sealed_read(source / "models.json")
    scores = sealed_read(source / "summary.json")
    prior = sealed_read(probe / "plan.json")
    prior_summary = sealed_read(probe / "summary.json")
    if (
        prior["source_models_sha256"] != models["artifact_sha256"]
        or prior["source_scores_sha256"] != scores["artifact_sha256"]
        or prior_summary["plan_sha256"] != prior["artifact_sha256"]
    ):
        raise ValueError("source model/score/diagnostic identity differs")
    full = [j for j in models["models"] if j["model_id"].endswith("_full")]
    if sorted(j["model_id"] for j in full) != sorted(FULL_IDS):
        raise ValueError("expected all four saved Full endpoints")
    imports = {}
    for case in prior["cases"]:
        relative = f"references/{case['id']}.json"
        imports[relative] = copy_sealed(probe / relative, output / relative)
        for job in prior["models"]:
            relative = f"replays/{job['model_id']}/{case['id']}.json"
            imports[relative] = copy_sealed(probe / relative, output / relative)
    return sealed_write(
        output / "plan.json",
        {
            "protocol": "t1-full-intervention-extension-1",
            "identity": source_identity(),
            "source_models_sha256": models["artifact_sha256"],
            "prior_plan_sha256": prior["artifact_sha256"],
            "prior_summary_sha256": prior_summary["artifact_sha256"],
            "imported_artifacts": imports,
            "models": [*full, *prior["models"]],
            "full_model_ids": list(FULL_IDS),
            "cases": prior["cases"],
            "training_scale": prior["training_scale"],
            "original_scores": [
                row
                for row in scores["models"]
                if row["model_id"] in {j["model_id"] for j in [*full, *prior["models"]]}
            ],
            "evidence_scope": (
                "Exploratory extension using all four available Full endpoints. "
                "Same seven synthetic diagnostic cases and saved reference arrays. "
                "Named channels are exact aliases. No new model selection, fitting, "
                "test data or proposer calls. These comparisons do not isolate memory."
            ),
        },
    )


def run(output: Path) -> dict:
    """Replay Full with frozen parameters, keeping imported comparisons unchanged."""
    with public._lock(output):
        plan = sealed_read(output / "plan.json")
        if plan["identity"] != source_identity():
            raise ValueError("diagnostic source or runtime changed after freeze")
        for relative, digest in plan["imported_artifacts"].items():
            if sealed_read(output / relative)["artifact_sha256"] != digest:
                raise ValueError(f"imported artifact changed: {relative}")
        records = {}
        for job in plan["models"]:
            model_id = job["model_id"]
            compiled = compile_candidate(
                CandidateModel.model_validate(job["candidate"]),
                ValidationContext.model_validate(job["context"]),
            )
            if set(compiled.parameter_names) != set(job["parameters"]):
                raise ValueError("fitted parameters differ from compiled model")
            for case in plan["cases"]:
                row = adapt_channels(
                    sealed_read(output / "references" / f"{case['id']}.json")["row"],
                    job["cell"],
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
                records[model_id, case["id"]] = replay_one(
                    compiled,
                    trajectory,
                    job["parameters"],
                    job["initials"],
                    job["context"]["targets"][0],
                    plan["training_scale"],
                    output / "replays" / model_id / f"{case['id']}.json",
                )
        rows, curves = [], []
        for job in plan["models"]:
            for case in plan["cases"]:
                record = records[job["model_id"], case["id"]]
                rows.append(
                    {
                        "model_id": job["model_id"],
                        "case_id": case["id"],
                        "success": record["success"],
                        "nmse": record["nmse"],
                        "effect": effect_error(
                            record, records[job["model_id"], case["paired_control"]]
                        )
                        if case["paired_control"]
                        else None,
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
                "status": "complete",
                "evidence_scope": plan["evidence_scope"],
                "test_data_opened": False,
                "parameter_refit_applied": False,
                "live_llm_calls": 0,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.source is None or args.probe is None:
            parser.error("freeze requires --source and --probe")
        freeze(args.source, args.probe, args.output)
    else:
        run(args.output)
