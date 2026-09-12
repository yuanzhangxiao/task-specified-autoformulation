"""Paired fitter recovery and controlled model-feature diagnostics."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.data import DatasetSplit, SplitName, TrainingScaler, Trajectory
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
from autoformalism.rebuttal.fitter_diagnostic import read_json, sha256, write_json
from autoformalism.rebuttal.fitter_methods import identity_runtime
from autoformalism.rebuttal.initialization_campaign import synthetic_problem
from autoformalism.rebuttal.piecewise_campaign import pack_split, replay, unpack_split
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash


class FeasibilityExperiment(StrictSchema):
    """Equal total phase budgets; add branch comparison only to piecewise cases."""

    protocol: Literal["fitter-feasibility-comparison-1"] = (
        "fitter-feasibility-comparison-1"
    )
    fit: CollocationSensitivityConfig = CollocationSensitivityConfig(
        initializer_seconds=120,
        refinement_seconds=180,
        maximum_function_evaluations=240,
        collocation_node_start="rollout_or_observed",
        node_warmup_seconds=5,
        least_squares_ftol=None,
        collocation_diagnostics=True,
    )


def coupled_problem(variant: str) -> tuple[dict, dict]:
    """Known-attainable four-state family with 15 weights; no latent labels exported.

    The scale variant changes only the latent coordinate f -> 100*f. The safe-start
    variant changes only the ordinary parameter guess. All references come from an
    independent hard-coded unscaled RHS, not the fitted expression compiler.
    """
    if variant not in {"linear", "quadratic", "safe_start", "scaled", "long_horizon"}:
        raise ValueError("unknown coupled control")
    linear = variant == "linear"
    multiplier = 100 if variant == "scaled" else 1
    f = "f" if multiplier == 1 else "(f/100)"
    mp, fp = "m" if linear else "m**2", f if linear else f"({f})**2"
    equations = {
        "m": f"u_m*u01-d_m*m+cf_m*(c+{f})",
        "c": f"m_c*m+f_c*{f}-d_c*c",
        "f": f"{multiplier}*(m_f*{mp}+c_f*c-d_f*{f}+v_f*v01)",
        "v01": f"u_v*u01+m_v*m+c_v*c+f_v*{fp}-d_v*v01",
    }
    names = (
        "u_m",
        "d_m",
        "cf_m",
        "m_c",
        "f_c",
        "d_c",
        "m_f",
        "c_f",
        "d_f",
        "v_f",
        "u_v",
        "m_v",
        "c_v",
        "f_v",
        "d_v",
    )
    candidate = {
        "candidate_id": "feature_" + variant,
        "parent_candidate_id": None,
        "states": [
            {"name": n, "kind": "observed" if n == "v01" else "latent"}
            for n in equations
        ],
        "state_equations": [{"state": n, "rhs": rhs} for n, rhs in equations.items()],
        "observation_mappings": [{"channel": "v01", "expression": "v01"}],
        "parameters": [
            {"name": n, "role": "nonnegative_coefficient", "scope": "global"}
            for n in names
        ],
        "initial_conditions": [
            {
                "state": n,
                "scope": "global",
                **({"expression": "v01"} if n == "v01" else {"fixed_value": 0.0}),
            }
            for n in equations
        ],
    }
    splits, reference = {}, {}
    horizon = 60 if variant == "long_horizon" else 20
    for split, amplitudes in (
        (SplitName.TRAIN, (0.0, 0.5, 1.0)),
        (SplitName.VALIDATION, (0.0, 0.8)),
    ):
        rows, clean = [], {}
        for index, amplitude in enumerate(amplitudes):
            time = np.linspace(
                0,
                horizon * (1.2 if split is SplitName.VALIDATION else 1),
                101 if horizon == 20 else 301,
            )

            def rhs(t, x, amplitude=amplitude):
                m, c, f, v = x
                return [
                    0.3 * amplitude - m + 0.03 * (c + f),
                    0.03 * m + 0.03 * f - c,
                    0.03 * (m if linear else m * m) + 0.03 * c - f + 0.03 * v,
                    0.2 * amplitude
                    + 0.03 * m
                    + 0.03 * c
                    + 0.03 * (f if linear else f * f)
                    - v,
                ]

            solution = solve_ivp(
                rhs,
                (0, time[-1]),
                [0.2, 0.1, 0.3, 1.0],
                t_eval=time,
                method="DOP853",
                rtol=1e-11,
                atol=1e-13,
            )
            if not solution.success:
                raise ValueError("independent coupled reference failed")
            key = f"{split.value}_{index}"
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
        splits[split.value] = pack_split(
            DatasetSplit(
                split,
                tuple(rows),
                content_hash(["coupled", linear, horizon, split.value]),
            )
        )
        reference[split.value] = clean
    start = dict.fromkeys(names, 0.1)
    if variant == "safe_start":
        start = {n: 0.5 if n.startswith("d_") else 0.02 for n in names}
    return {
        "candidate": candidate,
        "context": ValidationContext(
            targets=("v01",), external_inputs=("u01",)
        ).model_dump(mode="json"),
        "splits": splits,
        "start": start,
        "initialization_plan": {
            "rules": {
                n: {"initial": {"mode": "value", "guess": 0.0}} for n in ("m", "c", "f")
            }
        },
        "feature_control": {
            "variant": variant,
            "latent_scale": multiplier,
            "reference_parameters_used_for_fitting": False,
        },
    }, reference


def code_identity() -> str:
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "src/autoformalism").rglob("*.py"))
    paths += [
        root / "scripts/run_feasibility_campaign.py",
        root / "scripts/hpc/feasibility_delta.slurm",
        root / "scripts/hpc/submit_feasibility_delta.sh",
    ]
    return content_hash({str(p.relative_to(root)): sha256(p) for p in paths})


def prepare(
    plan: FeasibilityExperiment, output: Path, source: Path | None = None
) -> dict:
    """Freeze all controls and every public case from physical-initialization v1."""
    if source and (
        source.resolve().is_relative_to(output.resolve())
        or output.resolve().is_relative_to(source.resolve())
    ):
        raise ValueError("source and output must be separate")
    cases, tasks, assets = [], [], {}

    def add(label, problem, reference=None, group="control"):
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
            }
        )
        for arm in (
            ["legacy", "feasible", "branch_aware"]
            if piecewise
            else ["legacy", "feasible"]
        ):
            tasks.append({"index": len(tasks), "case_index": index, "arm": arm})

    for kind, noises, starts in (
        ("shared", (0.0, 0.03), (0,)),
        ("causal_map", (0.0,), (0,)),
        ("known_zero", (0.0,), (0,)),
        ("piecewise", (0.0, 0.03), (0, 1)),
    ):
        for noise in noises:
            for start in starts:
                problem, reference = synthetic_problem(kind, noise, start)
                add(f"{kind}/noise{noise}/start{start}", problem, reference)
    for variant in ("linear", "quadratic", "safe_start", "scaled", "long_horizon"):
        problem, reference = coupled_problem(variant)
        add(f"coupled/{variant}", problem, reference, "model_feature")
    source_identity = None
    if source:
        frozen = read_json(source / "freeze.json")
        if (
            frozen.get("plan", {}).get("protocol")
            != "physical-initialization-comparison-1"
            or frozen.get("test_data_opened") is not False
            or frozen.get("identity")
            != content_hash({k: v for k, v in frozen.items() if k != "identity"})
        ):
            raise ValueError(
                "expected verified public-only physical-initialization freeze"
            )
        source_identity = frozen["identity"]
        for case in frozen["cases"]:
            if case["synthetic"]:
                continue
            relative = f"problems/{case['index']:03d}.json"
            if sha256(source / relative) != frozen["assets"][relative]:
                raise ValueError("public problem hash differs")
            add(case["label"], read_json(source / relative), group="public")
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
    config = FeasibilityExperiment.model_validate(frozen["plan"]).fit.model_copy(
        update={"recovery_policy": task["arm"]}
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
