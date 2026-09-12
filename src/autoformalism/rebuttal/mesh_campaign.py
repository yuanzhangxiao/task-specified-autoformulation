"""Paired fitter recovery and controlled model-feature diagnostics."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.data import DatasetSplit, TrainingScaler, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    fit_collocation_forward_sensitivity,
)
from autoformalism.fitting.initialization import (
    LatentInitializationPlan,
    apply_initialization_plan,
)
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.rebuttal.feasibility_campaign import coupled_problem
from autoformalism.rebuttal.fitter_diagnostic import read_json, sha256, write_json
from autoformalism.rebuttal.fitter_methods import identity_runtime
from autoformalism.rebuttal.initialization_campaign import synthetic_problem
from autoformalism.rebuttal.piecewise_campaign import pack_split, replay, unpack_split
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash


class MeshExperiment(StrictSchema):
    """Ablate handoffs, assembly and mesh; report larger budgets separately."""

    protocol: Literal["fitter-mesh-comparison-1"] = "fitter-mesh-comparison-1"
    fit: CollocationSensitivityConfig = CollocationSensitivityConfig(
        initializer_seconds=120,
        refinement_seconds=180,
        maximum_function_evaluations=240,
        collocation_node_start="rollout_or_observed",
        node_warmup_seconds=5,
        least_squares_ftol=None,
        collocation_diagnostics=True,
        recovery_max_starts=10,
        recovery_probe_seconds=10,
    )


def arm_config(
    plan: MeshExperiment, arm: str, piecewise: bool
) -> CollocationSensitivityConfig:
    """Change only the declared factors; retain one shared refinement budget."""
    if arm not in {
        "previous",
        "handoff",
        "mapped",
        "mesh12000",
        "mesh24000",
        "more_time",
    }:
        raise ValueError("unknown comparison arm")
    updates = {
        "recovery_policy": "branch_aware" if piecewise else "feasible",
        "recovery_handoff": "screened" if arm == "previous" else "best_valid",
        "sensitivity_invalid_trials": "abort" if arm == "previous" else "reject",
        "collocation_assembly": "unrolled"
        if arm in {"previous", "handoff"}
        else "mapped",
        "collocation_target_variables": int(arm[4:])
        if arm.startswith("mesh")
        else None,
        "collocation_mesh_substeps": 1,
        "collocation_maximum_iterations": 1000 if arm == "more_time" else 150,
    }
    if arm == "more_time":
        updates["initializer_seconds"] = 300
    return CollocationSensitivityConfig.model_validate(
        {**plan.fit.model_dump(), **updates}
    )


def matched_problem(variant: str, template: dict) -> tuple[dict, dict]:
    """Match public time grids and sample counts, using independent stable dynamics.

    The constant-input control deliberately does not claim to match public forcing
    complexity. Its purpose is to expose setup/size effects with attainable data.
    """
    problem, _ = coupled_problem(variant)
    splits, reference = {}, {}
    for name in ("train", "val"):
        source = unpack_split(template["splits"][name])
        rows, clean = [], {}
        for index, row in enumerate(source.trajectories):
            time = row.time.copy()
            amplitude = (0.0, 0.5, 1.0)[index % 3]

            def rhs(t, x, amplitude=amplitude):
                m, c, f, v = x
                return [
                    0.3 * amplitude - m + 0.03 * (c + f),
                    0.03 * m + 0.03 * f - c,
                    0.03 * m * m + 0.03 * c - f + 0.03 * v,
                    0.2 * amplitude + 0.03 * m + 0.03 * c + 0.03 * f * f - v,
                ]

            solution = solve_ivp(
                rhs,
                (time[0], time[-1]),
                [0.2, 0.1, 0.3, 1.0],
                t_eval=time,
                method="DOP853",
                rtol=1e-11,
                atol=1e-13,
            )
            if not solution.success or not np.isfinite(solution.y).all():
                raise ValueError("independent matched reference failed")
            key = f"{name}_{index}"
            y = solution.y[3]
            rows.append(
                Trajectory(
                    key,
                    time,
                    {"v01": y},
                    {},
                    {"u01": np.full_like(time, amplitude)},
                    {},
                    {},
                )
            )
            clean[key] = y.tolist()
        splits[name] = pack_split(
            DatasetSplit(
                source.name,
                tuple(rows),
                content_hash(
                    ["matched-coupled", name, [r.time.tolist() for r in rows]]
                ),
            )
        )
        reference[name] = clean
    problem["splits"] = splits
    problem["feature_control"]["public_time_grids_and_counts"] = True
    problem["feature_control"]["public_forcing_matched"] = False
    return problem, reference


def prepare(plan: MeshExperiment, output: Path, source: Path | None = None) -> dict:
    """Freeze controls and all public-only cases, without using prior fit results."""
    if source and (
        source.resolve().is_relative_to(output.resolve())
        or output.resolve().is_relative_to(source.resolve())
    ):
        raise ValueError("source and output must be separate")
    cases, tasks, assets = [], [], {}

    def add(label, problem, reference=None, group="control", large=False):
        index = len(cases)
        for folder, value in (("problems", problem), ("references", reference)):
            if value is not None:
                path = output / f"{folder}/{index:03d}.json"
                write_json(path, value, immutable=True)
                assets[str(path.relative_to(output))] = sha256(path)
        model = compile_candidate(
            CandidateModel.model_validate(problem["candidate"]),
            ValidationContext.model_validate(problem["context"]),
        )
        model, _, _ = apply_initialization_plan(
            model,
            LatentInitializationPlan.model_validate(problem["initialization_plan"]),
        )
        piecewise = SymbolicODE(model, allow_piecewise=True).has_piecewise
        cases.append(
            {
                "index": index,
                "label": label,
                "synthetic": reference is not None,
                "group": group,
                "piecewise": piecewise,
                "public_size": large,
            }
        )
        arms = ["previous", "handoff", "mapped"]
        if large:
            arms += ["mesh12000", "mesh24000"]
        if group == "public" or label == "matched/quadratic":
            arms += ["more_time"]
        for arm in arms:
            tasks.append({"index": len(tasks), "case_index": index, "arm": arm})

    for variant in ("quadratic", "scaled"):
        problem, reference = coupled_problem(variant)
        add(f"coupled/{variant}", problem, reference)
    for noise in (0.0, 0.03):
        problem, reference = synthetic_problem("piecewise", noise, 0)
        add(f"piecewise/noise{noise}/start0", problem, reference)
    source_identity = None
    if source:
        frozen = read_json(source / "freeze.json")
        if (
            frozen.get("plan", {}).get("protocol") != "fitter-feasibility-comparison-1"
            or frozen.get("test_data_opened") is not False
            or frozen.get("identity")
            != content_hash({k: v for k, v in frozen.items() if k != "identity"})
        ):
            raise ValueError("expected verified public-only feasibility freeze")
        source_identity = frozen["identity"]
        public = []
        for case in frozen["cases"]:
            if case["synthetic"]:
                continue
            relative = f"problems/{case['index']:03d}.json"
            if sha256(source / relative) != frozen["assets"][relative]:
                raise ValueError("public problem hash differs")
            problem = read_json(source / relative)
            public.append(problem)
            add(case["label"], problem, group="public", large=True)
        if not public:
            raise ValueError("source has no public candidates")
        for variant in ("quadratic", "scaled"):
            problem, reference = matched_problem(variant, public[0])
            add(f"matched/{variant}", problem, reference, group="matched", large=True)
    frozen = {
        "plan": plan.model_dump(mode="json"),
        "cases": cases,
        "tasks": tasks,
        "assets": assets,
        "code": code_identity(),
        "runtime": identity_runtime(),
        "source_identity": source_identity,
        "test_data_opened": False,
        "llm_calls": 0,
    }
    frozen["identity"] = content_hash(frozen)
    write_json(output / "freeze.json", frozen, immutable=True)
    return frozen


def code_identity() -> str:
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "src/autoformalism").rglob("*.py"))
    paths += [
        root / "scripts/run_mesh_campaign.py",
        root / "scripts/hpc/mesh_delta.slurm",
        root / "scripts/hpc/submit_mesh_delta.sh",
    ]
    return content_hash({str(p.relative_to(root)): sha256(p) for p in paths})


def verify(output: Path) -> dict:
    frozen = read_json(output / "freeze.json")
    if frozen["identity"] != content_hash(
        {k: v for k, v in frozen.items() if k != "identity"}
    ):
        raise ValueError("freeze identity differs")
    if frozen["code"] != code_identity() or frozen["runtime"] != identity_runtime():
        raise ValueError("code/runtime differs; use pinned checkout")
    for relative, expected in frozen["assets"].items():
        if sha256(output / relative) != expected:
            raise ValueError(f"frozen input differs: {relative}")
    return frozen


def execute(output: Path, index: int) -> dict:
    """One fit per task; preserve completed stages and interrupted budgets."""
    frozen = verify(output)
    task = frozen["tasks"][index]
    case = frozen["cases"][task["case_index"]]
    directory = output / f"results/task_{index:03d}"
    identity = content_hash([frozen["identity"], task])

    def load(path):
        item = read_json(path)
        if item["identity"] != identity:
            raise ValueError("checkpoint identity differs")
        return item

    result_file, fit_file = directory / "result.json", directory / "fit.json"
    if result_file.exists():
        return load(result_file)
    record = {
        "identity": identity,
        "task": task,
        "case": case,
        "test_data_opened": False,
    }
    started = monotonic()
    if not fit_file.exists() and (directory / "fit_started.json").exists():
        load(directory / "fit_started.json")
        record.update(
            status="interrupted",
            error="Native fit interrupted; no fresh budget granted",
        )
        write_json(result_file, record, immutable=True)
        return record
    problem = read_json(output / f"problems/{case['index']:03d}.json")
    model = compile_candidate(
        CandidateModel.model_validate(problem["candidate"]),
        ValidationContext.model_validate(problem["context"]),
    )
    initial_plan = LatentInitializationPlan.model_validate(
        problem["initialization_plan"]
    )
    lowered, _, _ = apply_initialization_plan(model, initial_plan)
    train, val = [unpack_split(problem["splits"][k]) for k in ("train", "val")]
    config = arm_config(
        MeshExperiment.model_validate(frozen["plan"]), task["arm"], case["piecewise"]
    )
    try:
        if fit_file.exists():
            fit = load(fit_file)["fit"]
        else:
            write_json(
                directory / "fit_started.json", {"identity": identity}, immutable=True
            )
            fit = fit_collocation_forward_sensitivity(
                model,
                train,
                val,
                config,
                directory,
                initial_parameters=problem["start"],
                initialization_plan=initial_plan,
            )
            write_json(fit_file, {"identity": identity, "fit": fit}, immutable=True)
        record.update(status=fit["status"], fit=fit)
        if fit.get("parameters"):
            reference = (
                read_json(output / f"references/{case['index']:03d}.json")
                if case["synthetic"]
                else None
            )
            scale = TrainingScaler().fit(train).scales["target:v01"].standard_deviation
            checks = {}
            for split in (train, val):
                path = directory / f"replay_{split.name.value}.json"
                if not path.exists():
                    check = replay(
                        lowered,
                        split,
                        fit["parameters"],
                        scale,
                        reference[split.name.value] if reference else None,
                    )
                    write_json(
                        path, {"identity": identity, "check": check}, immutable=True
                    )
                checks[split.name.value] = load(path)["check"]
            record["replays"] = checks
            record["verified"] = all(c["pass"] for c in checks.values())
            record["status"] = "complete" if record["verified"] else "replay_unverified"
            if reference:
                record["recovered"] = record["verified"] and all(
                    c["scores"]["Radau"]["clean_nmse"] <= 1e-4 for c in checks.values()
                )
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    record["seconds"] = monotonic() - started
    write_json(result_file, record, immutable=True)
    return read_json(result_file)
