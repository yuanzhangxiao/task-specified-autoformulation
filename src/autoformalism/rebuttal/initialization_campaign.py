"""Paired physical-initialization comparison with frozen equations and data."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field
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
from autoformalism.rebuttal.fitter_diagnostic import read_json, sha256, write_json
from autoformalism.rebuttal.fitter_methods import identity_runtime
from autoformalism.rebuttal.piecewise_campaign import pack_split, replay, unpack_split
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash


class InitializationExperiment(StrictSchema):
    """Fixed fitting budgets; paired arms differ only in latent initialization."""

    protocol: Literal["physical-initialization-comparison-1"] = (
        "physical-initialization-comparison-1"
    )
    fit: CollocationSensitivityConfig = CollocationSensitivityConfig(
        initializer_seconds=120,
        refinement_seconds=180,
        maximum_function_evaluations=240,
        collocation_node_start="rollout_or_observed",
        node_warmup_seconds=5,
        least_squares_ftol=None,
    )
    starts: int = Field(default=3, ge=1, le=3)
    noise: tuple[float, ...] = (0.0, 0.03)


def synthetic_problem(kind: str, noise: float, start: int) -> tuple[dict, dict]:
    """Independent reference integration; no latent labels enter fitting assets."""
    if kind not in {"shared", "causal_map", "known_zero", "piecewise"}:
        raise ValueError("unknown initialization control")
    if noise not in (0.0, 0.03) or start not in range(3):
        raise ValueError("unknown noise or start")
    candidate = {
        "candidate_id": f"initial_{kind}",
        "parent_candidate_id": None,
        "states": [{"name": "m", "kind": "latent"}, {"name": "y", "kind": "latent"}],
        "state_equations": [
            {"state": "m", "rhs": "-rate*m+gain*u01"},
            {
                "state": "y",
                "rhs": "max(m-1,0)-0.7*y" if kind == "piecewise" else "m-0.7*y",
            },
        ],
        "observation_mappings": [{"channel": "v01", "expression": "y"}],
        "parameters": [
            {"name": "rate", "scope": "global", "role": "rate"},
            {"name": "gain", "scope": "global", "role": "coefficient"},
        ],
        "initial_conditions": [
            {"state": "m", "scope": "global", "fixed_value": 0.0},
            {"state": "y", "scope": "global", "expression": "v01"},
        ],
    }
    rule = {"mode": "value", "guess": (0.0, 0.5, 5.0)[start]}
    if kind == "causal_map":
        rule = {
            "mode": "map",
            "expression": "a+b*v01",
            "parameters": [
                {"name": "a", "guess": (0.0, 0.5, 3.0)[start]},
                {"name": "b", "guess": (0.0, 0.5, 3.0)[start]},
            ],
        }
    splits, reference = {}, {}
    for name, amplitudes, end in (
        (SplitName.TRAIN, (0.0, 0.5, 1.0), 5.0),
        (SplitName.VALIDATION, (0.0, 0.8), 6.0),
    ):
        rows, clean = [], {}
        for index, amplitude in enumerate(amplitudes):
            y0 = (
                (index * 0.6 if name is SplitName.TRAIN else 1.5 + index * 0.3)
                if kind == "causal_map"
                else 0.0
            )
            m0 = (
                1 + 2 * y0
                if kind == "causal_map"
                else 0.0
                if kind == "known_zero"
                else 3.0
                if kind == "piecewise"
                else 2.0
            )
            time = np.linspace(0, end, 51)

            def rhs(t, state, amplitude=amplitude):
                m, y = state
                return [
                    -0.3 * m + 0.8 * amplitude,
                    (max(m - 1, 0) if kind == "piecewise" else m) - 0.7 * y,
                ]

            solved = solve_ivp(
                rhs,
                (0, end),
                [m0, y0],
                t_eval=time,
                method="DOP853",
                rtol=1e-11,
                atol=1e-13,
            )
            if not solved.success:
                raise ValueError("synthetic reference integration failed")
            y = solved.y[1]
            rng = np.random.default_rng(
                90321 + index + (100 if name is SplitName.VALIDATION else 0)
            )
            observed = y + noise * max(float(np.std(y)), 1e-12) * rng.normal(
                size=len(y)
            )
            # The controlled preparation observes the initial boundary exactly.
            observed[0] = y0
            key = f"{name.value}_{index}"
            rows.append(
                Trajectory(
                    key,
                    time,
                    {"v01": observed},
                    {},
                    {"u01": np.full_like(time, amplitude)},
                    {},
                    {},
                )
            )
            clean[key] = y.tolist()
        splits[name.value] = pack_split(
            DatasetSplit(name, tuple(rows), content_hash([kind, noise, name.value]))
        )
        reference[name.value] = clean
    return {
        "candidate": candidate,
        "context": ValidationContext(
            targets=("v01",), external_inputs=("u01",)
        ).model_dump(mode="json"),
        "splits": splits,
        "start": {"rate": (0.1, 1.0, 2.0)[start], "gain": (0.2, 1.5, -0.3)[start]},
        "initialization_plan": {"rules": {"m": {"initial": rule}}},
    }, reference


def code_identity() -> str:
    """Bind actual source and launcher contents, including uncommitted smoke code."""
    root = Path(__file__).resolve().parents[3]
    paths = [
        *sorted((root / "src").rglob("*.py")),
        root / "scripts/run_initialization_campaign.py",
        root / "scripts/hpc/initialization_delta.slurm",
        root / "scripts/hpc/submit_initialization_delta.sh",
    ]
    return content_hash({str(p.relative_to(root)): sha256(p) for p in paths})


def prepare(
    plan: InitializationExperiment, output: Path, source: Path | None = None
) -> dict:
    """Freeze controls and import the already frozen public piecewise problems."""
    if source is not None and (
        source.resolve().is_relative_to(output.resolve())
        or output.resolve().is_relative_to(source.resolve())
    ):
        raise ValueError("source and output must be separate")
    cases, assets = [], {}

    def add(label, problem, reference=None):
        index = len(cases)
        for folder, payload in (("problems", problem), ("references", reference)):
            if payload is not None:
                path = output / f"{folder}/{index:03d}.json"
                write_json(path, payload, immutable=True)
                assets[str(path.relative_to(output))] = sha256(path)
        cases.append(
            {"index": index, "label": label, "synthetic": reference is not None}
        )

    for kind in ("shared", "causal_map", "known_zero", "piecewise"):
        for noise in plan.noise:
            for start in range(plan.starts):
                problem, reference = synthetic_problem(kind, noise, start)
                add(f"{kind}/noise{noise}/start{start}", problem, reference)
    source_identity = None
    if source:
        frozen = read_json(source / "freeze.json")
        if (
            frozen.get("plan", {}).get("protocol") != "piecewise-fitting-comparison-1"
            or frozen.get("test_data_opened") is not False
        ):
            raise ValueError(
                "source must be the frozen public-only piecewise comparison"
            )
        if frozen["identity"] != content_hash(
            {k: v for k, v in frozen.items() if k != "identity"}
        ):
            raise ValueError("source manifest identity differs")
        source_identity = frozen["identity"]
        from autoformalism.schemas import CandidateModel

        for case in frozen["cases"]:
            if case["synthetic"]:
                continue
            relative = f"problems/{case['index']:03d}.json"
            if sha256(source / relative) != frozen["assets"][relative]:
                raise ValueError("public problem hash differs")
            problem = read_json(source / relative)
            model = compile_candidate(
                CandidateModel.model_validate(problem["candidate"]),
                ValidationContext.model_validate(problem["context"]),
            )
            rules = {}
            for item in model.validated.candidate.initial_conditions:
                if item.state in model.direct_state_observation_channels:
                    continue
                if item.fixed_value is not None:
                    rules[item.state] = {
                        "initial": {"mode": "value", "guess": item.fixed_value}
                    }
                # Keep existing causal maps unchanged; do not invent new science.
            problem["initialization_plan"] = {"rules": rules}
            add(case["label"], problem)
    tasks = [
        {"index": 2 * c["index"] + i, "case_index": c["index"], "arm": arm}
        for c in cases
        for i, arm in enumerate(("fixed", "fitted"))
    ]
    frozen = {
        "plan": plan.model_dump(mode="json"),
        "cases": cases,
        "tasks": tasks,
        "assets": assets,
        "runtime": identity_runtime(),
        "code": code_identity(),
        "source_identity": source_identity,
        "test_data_opened": False,
        "llm_calls": 0,
    }
    frozen["identity"] = content_hash(frozen)
    write_json(output / "freeze.json", frozen, immutable=True)
    return frozen


def verify(output: Path) -> dict:
    """Verify frozen inputs before fitting; final summary is file-only."""
    frozen = read_json(output / "freeze.json")
    if frozen["identity"] != content_hash(
        {k: v for k, v in frozen.items() if k != "identity"}
    ):
        raise ValueError("freeze identity differs")
    if frozen["code"] != code_identity() or frozen["runtime"] != identity_runtime():
        raise ValueError("source or runtime differs; use the pinned checkout")
    for relative, expected in frozen["assets"].items():
        if sha256(output / relative) != expected:
            raise ValueError(f"frozen input differs: {relative}")
    return frozen


def execute(output: Path, index: int) -> dict:
    """One arm per worker; preserve fit and replay checkpoints across restarts."""
    from autoformalism.schemas import CandidateModel

    frozen = verify(output)
    task = frozen["tasks"][index]
    case = frozen["cases"][task["case_index"]]
    directory = output / f"results/task_{index:03d}"
    identity = content_hash([frozen["identity"], task])
    result_file = directory / "result.json"
    if result_file.exists():
        result = read_json(result_file)
        if result["identity"] != identity:
            raise ValueError("saved result identity differs")
        return result
    problem = read_json(output / f"problems/{case['index']:03d}.json")
    model = compile_candidate(
        CandidateModel.model_validate(problem["candidate"]),
        ValidationContext.model_validate(problem["context"]),
    )
    train, val = (unpack_split(problem["splits"][name]) for name in ("train", "val"))
    plan = InitializationExperiment.model_validate(frozen["plan"])
    # Both arms enforce observed identity boundaries. Only the latent rule differs.
    initial_plan = LatentInitializationPlan.model_validate(
        problem["initialization_plan"] if task["arm"] == "fitted" else {}
    )
    fitted_model, _, _ = apply_initialization_plan(model, initial_plan)
    started = monotonic()
    record = {
        "identity": identity,
        "task": task,
        "case": case,
        "test_data_opened": False,
        "llm_calls": 0,
    }
    fit_file = directory / "fit.json"
    # A killed non-checkpointable native optimization is never silently restarted.
    if not fit_file.exists() and (directory / "fit_started.json").exists():
        record.update(
            status="interrupted",
            error="Retained interrupted attempt; no fresh fitting budget granted",
        )
        write_json(result_file, record)
        return record
    try:
        if fit_file.exists():
            checkpoint = read_json(fit_file)
            if checkpoint["identity"] != identity:
                raise ValueError("saved fit identity differs")
            fit = checkpoint["fit"]
        else:
            write_json(
                directory / "fit_started.json", {"identity": identity}, immutable=True
            )
            fit = fit_collocation_forward_sensitivity(
                model,
                train,
                val,
                plan.fit,
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
                        fitted_model,
                        split,
                        fit["parameters"],
                        scale,
                        reference[split.name.value] if reference else None,
                    )
                    write_json(
                        path, {"identity": identity, "check": check}, immutable=True
                    )
                checkpoint = read_json(path)
                if checkpoint["identity"] != identity:
                    raise ValueError("saved replay identity differs")
                checks[split.name.value] = checkpoint["check"]
            record["replays"] = checks
            record["verified"] = all(c["pass"] for c in checks.values())
            record["status"] = "complete" if record["verified"] else "replay_unverified"
            if reference:
                record["recovered"] = record["verified"] and all(
                    c.get("scores", {}).get("Radau", {}).get("clean_nmse", float("inf"))
                    <= 1e-4
                    for c in checks.values()
                )
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    record["seconds"] = monotonic() - started
    write_json(result_file, record, immutable=True)
    return read_json(result_file)
