"""Isolated fixed-skeleton attainable-data and reference-model comparisons."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.data import TrainingScaler
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    fit_collocation_forward_sensitivity,
)
from autoformalism.fitting.initialization import LatentInitializationPlan
from autoformalism.rebuttal.attainability_controls import (
    choose_truth,
    fixed_state_fit,
    generated_problem,
    ordinary_start,
    simulate_split,
    system_for,
)
from autoformalism.rebuttal.attainability_reference import (
    native_replay_audit,
    reference_problem,
    shared_boundary_lower_bound,
)
from autoformalism.rebuttal.fitter_diagnostic import read_json, sha256, write_json
from autoformalism.rebuttal.fitter_methods import identity_runtime
from autoformalism.rebuttal.piecewise_campaign import replay, safe_path, unpack_split
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash


class AttainabilityPlan(StrictSchema):
    """Budgets frozen before results; ordinary starts are independent of truth."""

    protocol: Literal["fitter-attainability-1"] = "fitter-attainability-1"
    generation_seconds: float = Field(default=600, gt=0, le=600)
    fit: CollocationSensitivityConfig = CollocationSensitivityConfig(
        initializer_seconds=300,
        refinement_seconds=600,
        maximum_function_evaluations=480,
        collocation_maximum_iterations=1000,
        collocation_node_start="rollout_or_observed",
        node_warmup_seconds=5,
        least_squares_ftol=None,
        collocation_diagnostics=True,
        recovery_policy="feasible",
        recovery_handoff="best_valid",
        sensitivity_invalid_trials="reject",
        collocation_assembly="mapped",
    )


def code_identity():
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "src/autoformalism").rglob("*.py"))
    paths += [
        root / "scripts/run_attainability_campaign.py",
        root / "scripts/hpc/attainability_delta.slurm",
        root / "scripts/hpc/submit_attainability_delta.sh",
    ]
    return content_hash({str(p.relative_to(root)): sha256(p) for p in paths})


def _identity(value):
    if value.get("identity") != content_hash(
        {k: v for k, v in value.items() if k != "identity"}
    ):
        raise ValueError("artifact identity differs")


def prepare(plan, output, source, reference=None):
    """Copy frozen public inputs; no simulation or optimization on login nodes."""
    if source.resolve().is_relative_to(
        output.resolve()
    ) or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("source/output must be separate")
    original = read_json(source / "freeze.json")
    _identity(original)
    if (
        original.get("plan", {}).get("protocol") != "fitter-feasibility-comparison-1"
        or original.get("test_data_opened") is not False
    ):
        raise ValueError("expected public-only feasibility source")
    cases, tasks, assets = [], [], {}

    def put(relative, value):
        path = output / relative
        write_json(path, value, immutable=True)
        assets[relative] = sha256(path)

    def add_task(i, dataset, arm):
        tasks.append(
            {"index": len(tasks), "case_index": i, "dataset": dataset, "arm": arm}
        )

    def add(label, problem, oracle=False):
        i = len(cases)
        put(f"problems/{i:03d}.json", problem)
        cases.append({"index": i, "label": label, "reference_skeleton": oracle})
        for arm in (
            "fixed_full",
            "fixed_mesh",
            "blind0_full",
            "blind0_mesh",
            "blind1_mesh",
            "near_mesh",
        ):
            add_task(i, "synthetic", arm)
        for arm in (
            ("blind0_mesh", "blind1_mesh", "near_mesh") if oracle else ("blind0_mesh",)
        ):
            add_task(i, "actual", arm)
        if oracle:
            for arm in ("blind0_mesh", "blind1_mesh"):
                add_task(i, "shared_initials", arm)

    first = None
    for case in original["cases"]:
        if case["synthetic"]:
            continue
        relative = f"problems/{case['index']:03d}.json"
        path = safe_path(source, relative)
        if sha256(path) != original["assets"][relative]:
            raise ValueError("source public problem hash differs")
        problem = read_json(path)
        for split in ("train", "val"):
            unpack_split(problem["splits"][split])
        first = first or problem
        add(case["label"], problem)
    if first is None:
        raise ValueError("source contains no public candidates")
    if reference:
        bundle = read_json(reference)
        _identity(bundle)
        if (
            bundle.get("protocol") != "isolated-fitter-reference-1"
            or bundle.get("test_data_opened") is not False
            or bundle.get("proposer_access") is not False
        ):
            raise ValueError("expected isolated reference export")
        put("reference_input.json", bundle)
        add("reference_skeleton", reference_problem(first, bundle), True)
    frozen = {
        "plan": plan.model_dump(mode="json"),
        "cases": cases,
        "tasks": tasks,
        "assets": assets,
        "source_identity": original["identity"],
        "code": code_identity(),
        "runtime": identity_runtime(),
        "test_data_opened": False,
        "private_reference_opened": bool(reference),
        "proposer_access": False,
        "llm_calls": 0,
    }
    frozen["identity"] = content_hash(frozen)
    write_json(output / "freeze.json", frozen, immutable=True)
    return frozen


def verify(output):
    frozen = read_json(output / "freeze.json")
    _identity(frozen)
    if frozen["code"] != code_identity() or frozen["runtime"] != identity_runtime():
        raise ValueError("code/runtime differs; use the pinned checkout")
    for relative, expected in frozen["assets"].items():
        if sha256(safe_path(output, relative)) != expected:
            raise ValueError("frozen input differs: " + relative)
    return frozen


def _checkpoint(path, identity):
    saved = read_json(path)
    if saved.get("identity") != identity:
        raise ValueError("checkpoint identity differs")
    return saved


def generate(output, index):
    """One supervised generation job per model, before the fitting array."""
    frozen = verify(output)
    case = frozen["cases"][index]
    identity = content_hash([frozen["identity"], "generate", index])
    root = output / f"generated/{index:03d}"
    path = root / "result.json"
    if path.exists():
        return _checkpoint(path, identity)
    record = {"identity": identity, "case": case, "status": "generation_failed"}
    if (root / "started.json").exists():
        _checkpoint(root / "started.json", identity)
        record.update(
            status="interrupted", error="Generation interrupted; no fresh budget"
        )
        write_json(path, record, immutable=True)
        return record
    write_json(root / "started.json", {"identity": identity}, immutable=True)
    problem = read_json(output / f"problems/{index:03d}.json")
    plan = AttainabilityPlan.model_validate(frozen["plan"])
    started = monotonic()

    def stage(name):
        write_json(
            root / "stage.json",
            {"identity": identity, "stage": name, "seconds": monotonic() - started},
        )

    try:
        if case["reference_skeleton"]:
            stage("native_reference_audit")
            bundle = read_json(output / "reference_input.json")
            truth = bundle["truth"]
            native = native_replay_audit(problem, bundle["spec"])
            record["native_generator_audit"] = native
            record["shared_initialization_limit"] = shared_boundary_lower_bound(problem)
            write_json(
                root / "native_audit.json",
                {
                    "native_generator_audit": native,
                    "shared_initialization_limit": record[
                        "shared_initialization_limit"
                    ],
                },
            )
            if (
                max(
                    r["maximum_absolute_difference"]
                    for rows in native.values()
                    for r in rows
                )
                > 1e-5
            ):
                raise ValueError("original generator does not reproduce public data")
            stage("reference_training_simulation")
            y, states = simulate_split(problem, truth, "train")
        else:
            stage("candidate_training_truth_design")
            truth, y, states, attempts = choose_truth(
                problem, root, plan.generation_seconds
            )
            record["truth_design"] = attempts
        # Training alone selected truth; validation cannot trigger a new choice.
        write_json(
            root / "selected_truth.json", {"parameters": truth, "training_only": True}
        )
        stage("validation_simulation")
        val, _ = simulate_split(problem, truth, "val")
        clean = {"train": y, "val": val}
        synthetic = generated_problem(problem, clean)
        lowered, _ = system_for(synthetic)
        scale = (
            TrainingScaler()
            .fit(unpack_split(synthetic["splits"]["train"]))
            .scales["target:v01"]
            .standard_deviation
        )
        if not np.isfinite(scale) or scale < 1e-3:
            raise ValueError("generated control has insufficient output variation")
        stage("independent_generating_replays")
        checks = {
            k: replay(
                lowered.model,
                unpack_split(synthetic["splits"][k]),
                truth,
                scale,
                clean[k],
            )
            for k in ("train", "val")
        }
        record["generating_replays"] = checks
        if not all(c["pass"] for c in checks.values()):
            raise ValueError("generating point is not independently replay verified")
        if case["reference_skeleton"]:
            real_scale = (
                TrainingScaler()
                .fit(unpack_split(problem["splits"]["train"]))
                .scales["target:v01"]
                .standard_deviation
            )
            record["sampled_input_truth_scores"] = {
                k: float(
                    np.mean(
                        np.concatenate(
                            [
                                (
                                    (
                                        np.asarray(clean[k][r["trajectory_id"]])
                                        - r["targets"]["v01"]
                                    )
                                    / real_scale
                                )
                                ** 2
                                for r in problem["splits"][k]["rows"]
                            ]
                        )
                    )
                )
                for k in ("train", "val")
            }
        record["activity"] = {
            "output_sd": scale,
            "state_ranges": np.ptp(
                np.concatenate(list(states.values())), axis=0
            ).tolist(),
        }
        files = {"synthetic.json": synthetic, "truth.json": truth, "clean.json": clean}
        record["assets"] = {}
        for name, value in files.items():
            write_json(root / name, value, immutable=True)
            record["assets"][name] = sha256(root / name)
        record["status"] = "complete"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["seconds"] = monotonic() - started
    stage(record["status"])
    write_json(path, record, immutable=True)
    return record


def execute(output, index):
    """Fit one frozen problem, with independent checkpointed production replays."""
    frozen = verify(output)
    task = frozen["tasks"][index]
    case = frozen["cases"][task["case_index"]]
    identity = content_hash([frozen["identity"], task])
    directory = output / f"results/task_{index:03d}"
    result_file, fit_file = directory / "result.json", directory / "fit.json"
    if result_file.exists():
        return _checkpoint(result_file, identity)
    record = {
        "identity": identity,
        "task": task,
        "case": case,
        "test_data_opened": False,
        "proposer_access": False,
        "hidden_trajectory_labels_used": False,
        "oracle_initials": task["arm"].startswith("fixed")
        or (case["reference_skeleton"] and task["dataset"] != "shared_initials"),
        "oracle_start": task["arm"].startswith(("near", "fixed")),
    }
    if not fit_file.exists() and (directory / "fit_started.json").exists():
        _checkpoint(directory / "fit_started.json", identity)
        record.update(
            status="interrupted", error="Native fit interrupted; no fresh budget"
        )
        write_json(result_file, record, immutable=True)
        return record
    root = output / f"generated/{case['index']:03d}"
    generator_id = content_hash([frozen["identity"], "generate", case["index"]])
    generated = (
        _checkpoint(root / "result.json", generator_id)
        if (root / "result.json").exists()
        else {}
    )
    needs_generated = (
        case["reference_skeleton"]
        or task["dataset"] == "synthetic"
        or record["oracle_start"]
    )
    if needs_generated and generated.get("status") != "complete":
        record.update(status="generation_unavailable", error=generated.get("error"))
        write_json(result_file, record, immutable=True)
        return record
    for name, digest in generated.get("assets", {}).items():
        if sha256(safe_path(root, name)) != digest:
            raise ValueError("generated artifact changed")
    problem = read_json(
        root / "synthetic.json"
        if task["dataset"] == "synthetic"
        else output / f"problems/{case['index']:03d}.json"
    )
    if task["dataset"] == "shared_initials":
        problem = deepcopy(problem)
        problem["initialization_plan"] = {
            "rules": {
                s["name"]: {"initial": {"mode": "value", "guess": 0.0}}
                for s in problem["candidate"]["states"]
                if s["kind"] == "latent"
            }
        }
    plan = AttainabilityPlan.model_validate(frozen["plan"])
    target = None if task["arm"].endswith("full") else 24000
    config = plan.fit.model_copy(update={"collocation_target_variables": target})
    system, _ = system_for(problem)
    if system.has_piecewise:
        config = config.model_copy(update={"recovery_policy": "branch_aware"})
    start = ordinary_start(problem, int(task["arm"].startswith("blind1")))
    if record["oracle_start"]:
        truth = read_json(root / "truth.json")
        rng = np.random.default_rng(20260914)
        start = {
            n: float(v * np.exp(rng.normal(0, 0.2))) if v else 0.1
            for n, v in sorted(truth.items())
        }
    started = monotonic()
    try:
        if fit_file.exists():
            fit = _checkpoint(fit_file, identity)["fit"]
        else:
            write_json(
                directory / "fit_started.json", {"identity": identity}, immutable=True
            )
            if task["arm"].startswith("fixed"):
                fit = fixed_state_fit(
                    problem,
                    truth,
                    target,
                    directory / "fixed",
                    config.initializer_seconds,
                )
            else:
                model = compile_candidate(
                    CandidateModel.model_validate(problem["candidate"]),
                    ValidationContext.model_validate(problem["context"]),
                )
                fit = fit_collocation_forward_sensitivity(
                    model,
                    unpack_split(problem["splits"]["train"]),
                    unpack_split(problem["splits"]["val"]),
                    config,
                    directory,
                    initial_parameters=start,
                    initialization_plan=LatentInitializationPlan.model_validate(
                        problem["initialization_plan"]
                    ),
                )
            write_json(fit_file, {"identity": identity, "fit": fit}, immutable=True)
        record["fit"] = fit
        if task["arm"].startswith("fixed"):
            record["status"] = "complete" if fit["success"] else "node_solve_failed"
        else:
            record["status"] = fit["status"]
            if fit.get("parameters"):
                train = unpack_split(problem["splits"]["train"])
                scale = (
                    TrainingScaler().fit(train).scales["target:v01"].standard_deviation
                )
                clean = (
                    read_json(root / "clean.json")
                    if task["dataset"] == "synthetic"
                    else None
                )
                checks = {}
                for key in ("train", "val"):
                    path = directory / f"replay_{key}.json"
                    if not path.exists():
                        check = replay(
                            system.model,
                            unpack_split(problem["splits"][key]),
                            fit["parameters"],
                            scale,
                            clean[key] if clean else None,
                        )
                        write_json(
                            path, {"identity": identity, "check": check}, immutable=True
                        )
                    checks[key] = _checkpoint(path, identity)["check"]
                record["replays"] = checks
                record["verified"] = all(c["pass"] for c in checks.values())
                record["status"] = (
                    "complete" if record["verified"] else "replay_unverified"
                )
                if clean:
                    record["recovered"] = record["verified"] and all(
                        c["scores"]["Radau"]["clean_nmse"] <= 1e-4
                        for c in checks.values()
                    )
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    record["seconds"] = monotonic() - started
    write_json(result_file, record, immutable=True)
    return record
