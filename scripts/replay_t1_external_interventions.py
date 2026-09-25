#!/usr/bin/env python3
"""Replay frozen external T1 endpoints on existing diagnostic interventions.

No fitting, proposal calls, test trajectories, or new reference simulation.
Continuous and native discrete models retain their respective runtime semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from autoformalism.baselines.d3_rollout import NativeMap, finite_mean, predict
from autoformalism.data import TrainingScaler
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.final_evaluation import FrozenEvaluationSubject
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicSplit
from scripts.probe_t1_interventions import effect_error
from scripts.replay_t1_curves import SETTINGS, pooled_error, replay_one, write_csv

CELLS = (
    "phase_b_dalla_man_t1_canonical_named_easy",
    "phase_b_anonymous_system_t1_canonical_obfuscated_easy",
    "phase_b_dalla_man_t1_perturbed_named_easy",
)
MEMBERS = {
    "external-baseline-evaluation-v1/adapted/frozen_evaluation_subjects.jsonl": (
        "4400ef56dd841a7aaf52737c2c5db182e82ba4cb500517dce08d3084fd757a46"
    ),
    "external-baseline-d3-test-v2/adapted/frozen_evaluation_subjects.jsonl": (
        "b3fd15b1b59b7b7317355efd8a210903f00cdb097b6d3ceaaf2944de7c27a421"
    ),
}
METHODS = {
    "sindy": "sindy",
    "pysr": "pysr",
    "raw_data_agent:openai:gpt-5.6-sol": "sol",
    "d3_native_no_tools": "d3",
}
CHANNELS = {
    "targets": {"v01": "Gp"},
    "auxiliaries": {"v02": "EGP", "v03": "Uii", "v04": "E", "v05": "Gt"},
    "external_inputs": {"u01": "meal_event_g"},
}


def sha(path: Path) -> str:
    """Hash a local immutable input or implementation file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize_subject(row: dict) -> FrozenEvaluationSubject:
    """Drop existing endpoint scores before validation or model selection."""
    payload = {
        key: row[key]
        for key in (
            "schema_version",
            "subject_id",
            "method",
            "benchmark_id",
            "tier",
            "repetition",
            "selection_frozen",
            "source_provenance",
            "candidate",
            "parameterization",
            "validation_context",
        )
    }
    payload.update(
        execution_semantics=row.get(
            "execution_semantics", "continuous_ode_free_rollout"
        ),
        private_metrics_opened_after_freeze=False,
        target_prediction={"status": "missing"},
    )
    subject = FrozenEvaluationSubject.model_validate(payload)
    if (
        public.content_sha256(subject.candidate)
        != subject.source_provenance.candidate_sha256
    ):
        raise ValueError("frozen candidate digest differs")
    if subject.parameterization.status not in {"available", "not_required"}:
        raise ValueError("incomplete frozen parameterization")
    if subject.validation_context.lagged_targets:
        raise ValueError("future target forcing is not permitted")
    return subject


def translate_reference(row: dict, cell: str, context: dict) -> dict:
    """Map the same physical reference into the original public channel names."""
    result = {key: row[key] for key in ("trajectory_id", "time")}
    for role, mapping in CHANNELS.items():
        renamed = {
            (name if "obfuscated" in cell else mapping[name]): value
            for name, value in row[role].items()
        }
        if set(context[role]) != set(renamed):
            raise ValueError(f"reference and original public {role} differ")
        result[role] = renamed
    return result


def freeze(archive: Path, public_plan: Path, reference_root: Path, root: Path) -> dict:
    """Select every repetition in three fixed cells, independently of scores."""
    subjects = []
    coverage = Counter()
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        if {m.name for m in members} != set(MEMBERS) or len(members) != len(MEMBERS):
            raise ValueError("unexpected archive inventory")
        for member in members:
            if not member.isfile() or member.size > 32_000_000:
                raise ValueError("expected bounded regular files")
            source = stream.extractfile(member)
            assert source is not None
            payload = source.read()
            if hashlib.sha256(payload).hexdigest() != MEMBERS[member.name]:
                raise ValueError(
                    "archive member differs from supplied package manifest"
                )
            for line in payload.splitlines():
                row = json.loads(line)
                coverage[row["method"]] += 1
                if row["benchmark_id"] in CELLS:
                    subjects.append(sanitize_subject(row).model_dump(mode="json"))
    identities = [(r["benchmark_id"], r["method"], r["repetition"]) for r in subjects]
    expected = {
        (cell, method, rep) for cell in CELLS for method in METHODS for rep in range(3)
    }
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("require all 36 exact cell/method/repetition endpoints")
    cells = sealed_read(public_plan)["cells"]
    prior = sealed_read(reference_root / "plan.json")
    references = {
        variant: {
            case["id"]: sealed_read(
                reference_root / "references" / variant / (case["id"] + ".json")
            )
            for case in prior["cases"]
        }
        for variant in ("original", "perturbed_b1")
    }
    for refset in references.values():
        for record in refset.values():
            if (
                max(record["solver_max_differences"].values())
                > prior["settings"]["agreement_tolerance"]
            ):
                raise ValueError("saved reference solvers disagree")
    return sealed_write(
        root / "plan.json",
        {
            "protocol": "t1-external-frozen-interventions-1",
            "archive_sha256": sha(archive),
            "archive_members": MEMBERS,
            "coverage": dict(coverage),
            "public_plan_sha256": sha(public_plan),
            "source_sha256": public._source_identity(),
            "runtime": public._runtime(),
            "script_sha256": sha(Path(__file__)),
            "settings": SETTINGS.model_dump(mode="json"),
            "cells": {name: cells[name] for name in CELLS},
            "subjects": sorted(
                subjects,
                key=lambda r: (r["benchmark_id"], r["method"], r["repetition"]),
            ),
            "cases": prior["cases"],
            "references": references,
            "selection": "all four methods and three repetitions in each fixed cell",
            "parameter_fitting_performed": False,
            "live_llm_calls": 0,
            "test_trajectories_opened": False,
            "existing_test_metrics_used": False,
            "reference_use": (
                "Reuse existing exploratory diagnostic references; no new simulation."
            ),
            "limitations": [
                "Conditional T1-easy prediction with supplied auxiliaries.",
                (
                    "Exploratory interventions already inspected during demonstration "
                    "development, not confirmatory tests."
                ),
                (
                    "Original training-file hashes are absent from adapted subjects; "
                    "data bind by benchmark ID and public channel contract."
                ),
            ],
        },
    )


def discrete_record(model, trajectory, target, scale, path):
    """Retain native one-step increments and supplied-auxiliary policy exactly."""
    if path.exists():
        return sealed_read(path)
    observed = trajectory.targets[target]
    record = {
        "trajectory_id": trajectory.trajectory_id,
        "time": trajectory.time.tolist(),
        "observed": observed.tolist(),
        "predicted": None,
        "nmse": None,
    }
    try:
        prediction = predict(model, trajectory, teacher_forced=False)[
            :, model.states.index(target)
        ]
        with np.errstate(over="raise", invalid="raise"):
            error = np.square((prediction - observed) / scale)
        record.update(
            success=True,
            message="native x_next = x + f; no dt multiplier",
            predicted=prediction.tolist(),
            nmse=finite_mean(error),
        )
    except (ArithmeticError, ValueError, TypeError, TimeoutError) as exc:
        record.update(success=False, message=f"{type(exc).__name__}: {exc}")
    return sealed_write(path, record)


def run_subject(arguments: tuple) -> dict:
    """Own all saved records for one frozen baseline endpoint."""
    raw, plan, root = arguments
    subject = FrozenEvaluationSubject.model_validate(raw)
    model_id = (
        f"cell{CELLS.index(subject.benchmark_id)}_"
        f"{METHODS[subject.method]}_rep{subject.repetition}"
    )
    directory = root / "replays" / model_id
    if (directory / "summary.json").exists():
        return sealed_read(directory / "summary.json")
    cell = plan["cells"][subject.benchmark_id]
    context = subject.validation_context
    for role in ("targets", "auxiliaries", "external_inputs", "fixed_covariates"):
        if set(getattr(context, role)) != set(cell["context"][role]):
            raise ValueError(f"baseline and public data {role} differ")
    train = public.unpack_split(PublicSplit.model_validate(cell["training"]))
    validation = public.unpack_split(PublicSplit.model_validate(cell["validation"]))
    (target,) = context.targets
    scale = TrainingScaler().fit(train).scales[f"target:{target}"].standard_deviation
    discrete = subject.execution_semantics == "discrete_increment_recursive_rollout"
    if discrete:
        first = train.trajectories[0]
        model = NativeMap.build(
            subject.candidate,
            subject.parameterization.global_parameters,
            (*first.targets, *first.auxiliaries),
            (*first.external_inputs, *first.fixed_covariates),
        )
        # The native learned increment is tied to the original sample grid.
        if any(
            not np.allclose(np.diff(t.time), 1.0)
            for t in (*train.trajectories, *validation.trajectories)
        ):
            raise ValueError(
                "D3 reference grid must match the training one-minute grid"
            )
    else:
        model = compile_candidate(subject.candidate, context)

    def replay(trajectory, path):
        if discrete:
            return discrete_record(model, trajectory, target, scale, path)
        return replay_one(
            model,
            trajectory,
            subject.parameterization.global_parameters,
            subject.parameterization.global_initial_conditions,
            target,
            scale,
            path,
        )

    result = {
        "model_id": model_id,
        "subject_id": subject.subject_id,
        "cell": subject.benchmark_id,
        "method": METHODS[subject.method],
        "repetition": subject.repetition,
        "execution_semantics": subject.execution_semantics,
        "training_scale": scale,
        "candidate_sha256": subject.source_provenance.candidate_sha256,
        "bounds_violations": list(model.bounds_violations) if discrete else [],
    }
    # Complete all declared intervention cases before development-score reporting.
    records = {}
    variant = "perturbed_b1" if "perturbed" in subject.benchmark_id else "original"
    for case in plan["cases"]:
        row = translate_reference(
            plan["references"][variant][case["id"]]["row"],
            subject.benchmark_id,
            context.model_dump(mode="json"),
        )
        split = PublicSplit.model_validate(
            {"name": "val", "fingerprint": "diagnostic", "rows": [row]}
        )
        (trajectory,) = public.unpack_split(split).trajectories
        if discrete and not np.allclose(np.diff(trajectory.time), 1.0):
            raise ValueError("D3 intervention sampling differs")
        records[case["id"]] = replay(
            trajectory, directory / "probes" / (case["id"] + ".json")
        )
    result["probes"] = {}
    for case in plan["cases"]:
        record = records[case["id"]]
        control = records.get(case["paired_control"])
        effect = effect_error(record, control) if control is not None else None
        result["probes"][case["id"]] = {
            "success": record["success"],
            "nmse": record["nmse"],
            "message": record["message"],
            "effect_relative_squared_error": effect["relative_squared_error"]
            if effect
            else None,
        }
    for name, split in (("train", train), ("validation", validation)):
        rows = [
            replay(t, directory / name / f"{index:03d}.json")
            for index, t in enumerate(split.trajectories)
        ]
        result[name] = {
            "nmse": pooled_error(rows),
            "complete": all(r["success"] for r in rows),
            "successful": sum(r["success"] for r in rows),
            "expected": len(rows),
        }
    result = sealed_write(directory / "summary.json", result)
    print(
        json.dumps({k: result[k] for k in ("model_id", "train", "validation")}),
        flush=True,
    )
    return result


def run(args) -> dict:
    """Freeze, checkpoint, and export all endpoints with failure visibility."""
    with public._lock(args.root):
        plan = freeze(args.archive, args.public_plan, args.reference_root, args.root)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(
                pool.map(run_subject, [(s, plan, args.root) for s in plan["subjects"]])
            )
        curves = []
        for row in rows:
            for case in plan["cases"]:
                record = sealed_read(
                    args.root
                    / "replays"
                    / row["model_id"]
                    / "probes"
                    / (case["id"] + ".json")
                )
                for i, time in enumerate(record["time"]):
                    curves.append(
                        {
                            "model_id": row["model_id"],
                            "cell": row["cell"],
                            "method": row["method"],
                            "repetition": row["repetition"],
                            "case": case["id"],
                            "time": time,
                            "reference": record["observed"][i],
                            "prediction": record["predicted"][i]
                            if record["success"]
                            else None,
                            "success": record["success"],
                        }
                    )
        write_csv(args.root / "curves.csv", curves)
        return sealed_write(
            args.root / "summary.json",
            {
                "protocol": plan["protocol"],
                "plan_sha256": plan["artifact_sha256"],
                "status": "complete",
                "expected": len(plan["subjects"]),
                "rows": rows,
                "parameter_fitting_performed": False,
                "live_llm_calls": 0,
                "test_trajectories_opened": False,
                "existing_test_metrics_used": False,
                "limitations": plan["limitations"],
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--public-plan", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    result = run(parser.parse_args())
    print(
        json.dumps({key: result[key] for key in ("status", "expected", "plan_sha256")})
    )
