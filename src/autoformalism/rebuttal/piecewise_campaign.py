"""Frozen CPU comparison of exact piecewise collocation and rollout fitting."""

from __future__ import annotations

import shutil
from itertools import pairwise
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import model_validator
from scipy.integrate import solve_ivp

from autoformalism.data import DatasetSplit, SplitName, TrainingScaler, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    _role_start,
    fit_collocation_forward_sensitivity,
)
from autoformalism.rebuttal.fitter_diagnostic import (
    _finite_payload,
    read_json,
    sha256,
    write_json,
)
from autoformalism.rebuttal.fitter_methods import identity_runtime
from autoformalism.rebuttal.staged_prefit_fitting_campaign import load_public_data
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash

CASES = ("ode_threshold", "observation_threshold", "fitted_threshold", "repeated_clamp")
METHODS = ("branch_sensitivity", "directional_poll")


class PiecewisePlan(StrictSchema):
    """One paired job per case/noise/start; no numerical choices use validation."""

    protocol: Literal["piecewise-fitting-comparison-1"] = (
        "piecewise-fitting-comparison-1"
    )
    fit: CollocationSensitivityConfig = CollocationSensitivityConfig(
        initializer_seconds=45,
        refinement_seconds=120,
        maximum_function_evaluations=240,
        collocation_node_start="rollout_or_observed",
    )

    @property
    def worker_seconds(self) -> float:
        """Include final production replays and independent cross-checks."""
        per_arm = self.fit.initializer_seconds + 3 * self.fit.refinement_seconds + 240
        return 4 * per_arm + 120

    @model_validator(mode="after")
    def scheduler_budget(self):
        if self.worker_seconds > 2940:
            raise ValueError("paired worker must fit the 50-minute scheduler limit")
        return self


def safe_path(root: Path, relative: str) -> Path:
    """Reject absolute paths and traversal in external bundle manifests."""
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError("artifact path escapes experiment")
    return path


def verify_bundle(root: Path) -> dict:
    """Accept only the previously audited public candidate export format."""
    bundle = read_json(root / "bundle.json")
    if (
        bundle.get("protocol") != "collocation-node-cases-1"
        or bundle.get("identity")
        != content_hash({k: v for k, v in bundle.items() if k != "identity"})
        or bundle.get("test_data_opened") is not False
        or bundle.get("private_reference_opened") is not False
        or bundle.get("llm_calls") != 0
    ):
        raise ValueError("source bundle identity or split policy differs")
    for relative, expected in bundle["files"].items():
        parts = Path(relative).parts
        allowed = (
            len(parts) == 2
            and parts[0] in {"candidates", "provenance"}
            and parts[-1].endswith(".json")
        ) or (
            len(parts) == 4
            and parts[:2] == ("public", "phase_b_v1")
            and parts[-1]
            in {"manifest.json", "proposer_prompt.txt", "train.csv", "validation.csv"}
        )
        if not allowed or sha256(safe_path(root, relative)) != expected:
            raise ValueError("unsupported or changed source bundle asset")
    for case in bundle["cases"]:
        required = [case["candidate"]] + [
            f"public/phase_b_v1/{case['benchmark_id']}/{name}"
            for name in (
                "manifest.json",
                "proposer_prompt.txt",
                "train.csv",
                "validation.csv",
            )
        ]
        if any(name not in bundle["files"] for name in required):
            raise ValueError("case is missing public provenance")
    return bundle


def pack_split(split: DatasetSplit) -> dict:
    """Serialize only public observations, supplied forcing and fixed covariates."""
    return {
        "name": split.name.value,
        "fingerprint": split.fingerprint,
        "rows": [
            {
                "trajectory_id": row.trajectory_id,
                "time": row.time.tolist(),
                "targets": {k: v.tolist() for k, v in row.targets.items()},
                "auxiliaries": {k: v.tolist() for k, v in row.auxiliaries.items()},
                "external_inputs": {
                    k: v.tolist() for k, v in row.external_inputs.items()
                },
                "fixed_covariates": dict(row.fixed_covariates),
            }
            for row in split.trajectories
        ],
    }


def unpack_split(payload: dict) -> DatasetSplit:
    """Restore a train/validation split; no test capability exists here."""
    name = SplitName(payload["name"])
    if name not in {SplitName.TRAIN, SplitName.VALIDATION}:
        raise ValueError("piecewise campaign may not open test")
    rows = []
    for row in payload["rows"]:
        fields = {
            key: {k: np.asarray(v) for k, v in row[key].items()}
            for key in ("targets", "auxiliaries", "external_inputs")
        }
        rows.append(
            Trajectory(
                row["trajectory_id"],
                np.asarray(row["time"]),
                **fields,
                fixed_covariates=row["fixed_covariates"],
                derivatives={},
            )
        )
    return DatasetSplit(name, tuple(rows), payload["fingerprint"])


def synthetic_problem(case: str, noise: float, start_index: int) -> tuple[dict, dict]:
    """Generate independent DOP853 references, never by the fitted symbolic graph."""
    if case not in CASES or start_index not in range(3) or noise not in (0.0, 0.03):
        raise ValueError("unknown synthetic control")
    parameter = "b" if case == "fitted_threshold" else "a"
    rhs = {
        "ode_threshold": "a*u01 - 0.4*max(x-10,0)",
        "observation_threshold": "a*u01",
        "fitted_threshold": "u01",
        "repeated_clamp": "a*u01 - 0.3*(x-10)",
    }[case]
    observation = {
        "ode_threshold": "x",
        "observation_threshold": "max(x-10,0)",
        "fitted_threshold": "max(x-b,0)",
        "repeated_clamp": "min(max(x-10,-0.5),0.7)+0.1*abs(x-10)",
    }[case]
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": case,
            "parent_candidate_id": None,
            "states": [{"name": "x", "kind": "latent"}],
            "state_equations": [{"state": "x", "rhs": rhs}],
            "observation_mappings": [{"channel": "v01", "expression": observation}],
            "parameters": [
                {
                    "name": parameter,
                    "scope": "global",
                    "role": "offset" if parameter == "b" else "rate",
                }
            ],
            "initial_conditions": [{"state": "x", "scope": "global", "fixed_value": 9}],
        }
    )
    context = ValidationContext(targets=("v01",), external_inputs=("u01",))
    reference, splits = {}, {}
    for split, amplitudes, end in (
        (SplitName.TRAIN, (1.0, 1.4), 4.0),
        (SplitName.VALIDATION, (0.8,), 5.0),
    ):
        rows, clean = [], {}
        for index, amplitude in enumerate(amplitudes):
            time = np.linspace(0, end, 41)
            u = (
                2 * amplitude * np.cos(2 * np.pi * time / 2.5)
                if case == "repeated_clamp"
                else np.full_like(time, amplitude)
            )

            def generate(t, x, time=time, u=u):
                forcing = float(np.interp(t, time, u))
                if case == "ode_threshold":
                    return [forcing - 0.4 * max(x[0] - 10, 0)]
                if case == "repeated_clamp":
                    return [forcing - 0.3 * (x[0] - 10)]
                return [forcing]

            # Respect every supplied-input knot; one adaptive solve can miss a
            # small interpolation kink even with tight tolerances.
            state = np.array([9.0])
            samples = [9.0]
            for left, right in pairwise(time):
                result = solve_ivp(
                    generate,
                    (left, right),
                    state,
                    t_eval=[right],
                    method="DOP853",
                    rtol=1e-11,
                    atol=1e-13,
                )
                if not result.success:
                    raise ValueError("independent reference integration failed")
                state = result.y[:, -1]
                samples.append(float(state[0]))
            x = np.asarray(samples)
            y = (
                x
                if case == "ode_threshold"
                else np.clip(x - 10, -0.5, 0.7) + 0.1 * abs(x - 10)
                if case == "repeated_clamp"
                else np.maximum(x - 10, 0)
            )
            row_id = f"{split.value}_{index}"
            rng = np.random.default_rng(
                91273 + index + (100 if split is SplitName.VALIDATION else 0)
            )
            observed = y + noise * max(float(np.std(y)), 1e-12) * rng.normal(
                size=len(y)
            )
            rows.append(
                Trajectory(row_id, time, {"v01": observed}, {}, {"u01": u}, {}, {})
            )
            clean[row_id] = y.tolist()
        data = DatasetSplit(
            split, tuple(rows), content_hash([case, noise, split.value])
        )
        splits[split.value] = pack_split(data)
        reference[split.value] = clean
    start = (
        (8.0, 10.0, 16.0)[start_index]
        if parameter == "b"
        else (0.05, 0.25, 2.0)[start_index]
    )
    return {
        "candidate": candidate.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
        "splits": splits,
        "start": {parameter: start},
    }, reference


def launcher_identity() -> str:
    root = Path(__file__).resolve().parents[3]
    files = (
        "scripts/run_piecewise_campaign.py",
        "scripts/hpc/piecewise_delta.slurm",
        "scripts/hpc/submit_piecewise_delta.sh",
    )
    return content_hash({name: sha256(root / name) for name in files})


def prepare(plan: PiecewisePlan, output: Path, source: Path | None = None) -> dict:
    """Freeze all controls and optional public candidates without selecting by fit."""
    if (output / "freeze.json").exists():
        frozen = verify(output)
        if frozen["plan"] != plan.model_dump(mode="json"):
            raise ValueError("plan differs on resume")
        expected_source = verify_bundle(source)["identity"] if source else None
        if expected_source != frozen["source_identity"]:
            raise ValueError("source differs on resume")
        return frozen
    cases, assets = [], {}

    def add_case(label, problem, reference=None, **metadata):
        index = len(cases)
        path = output / f"problems/{index:03d}.json"
        write_json(path, problem, immutable=True)
        assets[str(path.relative_to(output))] = sha256(path)
        if reference is not None:
            path = output / f"references/{index:03d}.json"
            write_json(path, reference, immutable=True)
            assets[str(path.relative_to(output))] = sha256(path)
        cases.append(
            {
                "index": index,
                "label": label,
                "synthetic": reference is not None,
                **metadata,
            }
        )

    for case in CASES:
        for noise in (0.0, 0.03):
            for start in range(3):
                problem, reference = synthetic_problem(case, noise, start)
                add_case(
                    f"{case}/noise{noise}/start{start}",
                    problem,
                    reference,
                    noise=noise,
                    start_index=start,
                )
    source_identity = None
    if source:
        if source.resolve().is_relative_to(
            output.resolve()
        ) or output.resolve().is_relative_to(source.resolve()):
            raise ValueError("source and output must be separate")
        bundle = verify_bundle(source)
        source_identity = bundle["identity"]
        for relative in ("bundle.json", *bundle["files"]):
            destination = safe_path(output / "source", relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256(destination) != sha256(
                safe_path(source, relative)
            ):
                raise ValueError("partial source snapshot differs")
            shutil.copyfile(safe_path(source, relative), destination)
            assets[str(destination.relative_to(output))] = sha256(destination)
        for case in bundle["cases"]:
            dataset, context = load_public_data(
                output / "source/public", case["benchmark_id"], case["tier"]
            )
            candidate = CandidateModel.model_validate(
                read_json(safe_path(source, case["candidate"]))
            )
            add_case(
                case["name"],
                {
                    "candidate": candidate.model_dump(mode="json"),
                    "context": context.model_dump(mode="json"),
                    "splits": {
                        "train": pack_split(dataset.train),
                        "val": pack_split(dataset.validation),
                    },
                    "start": _role_start(candidate, dataset.train),
                },
            )
    frozen = {
        "plan": plan.model_dump(mode="json"),
        "cases": cases,
        "assets": assets,
        "runtime": identity_runtime(),
        "launcher": launcher_identity(),
        "source_identity": source_identity,
        "test_data_opened": False,
        "llm_calls": 0,
    }
    frozen["identity"] = content_hash(frozen)
    write_json(output / "freeze.json", frozen, immutable=True)
    return frozen


def verify(output: Path) -> dict:
    """Refuse changed source, configuration, runtime or frozen inputs."""
    frozen = read_json(output / "freeze.json")
    if (
        frozen["identity"]
        != content_hash({k: v for k, v in frozen.items() if k != "identity"})
        or frozen["runtime"] != identity_runtime()
        or frozen["launcher"] != launcher_identity()
    ):
        raise ValueError("piecewise freeze or runtime differs")
    PiecewisePlan.model_validate(frozen["plan"])
    for name, expected in frozen["assets"].items():
        if sha256(safe_path(output, name)) != expected:
            raise ValueError(f"frozen input differs: {name}")
    return frozen


def replay(model, split, parameters, scale, reference=None) -> dict:
    """Independent BDF and tighter Radau checks; clean labels are scoring-only."""
    scores, predictions = {}, {}
    settings = FitConfig(
        integration_method="BDF",
        relative_tolerance=1e-9,
        absolute_tolerance=1e-11,
        maximum_wall_time_seconds=60,
    )
    for method in ("BDF", "Radau"):
        residuals, values, clean_errors = [], {}, []
        deadline = monotonic() + 60
        for row in split.trajectories:
            result = simulate_trajectory(
                model,
                row,
                parameters,
                {},
                settings.model_copy(update={"integration_method": method}),
                reset_observed_states=False,
                deadline=deadline,
            )
            if not result.success:
                return {
                    "pass": False,
                    "error": f"{method}/{row.trajectory_id}: {result.message}",
                }
            y = result.predictions["v01"]
            values[row.trajectory_id] = y
            residuals.extend(((y - row.targets["v01"]) / scale) ** 2)
            if reference is not None:
                clean_errors.extend(
                    ((y - np.asarray(reference[row.trajectory_id])) / scale) ** 2
                )
        predictions[method] = values
        scores[method] = {
            "nmse": float(np.mean(residuals)),
            "clean_nmse": float(np.mean(clean_errors)) if clean_errors else None,
        }
    difference = max(
        float(np.max(abs(predictions["BDF"][key] - values) / scale))
        for key, values in predictions["Radau"].items()
    )
    return {
        "pass": difference < 1e-4,
        "maximum_scaled_prediction_difference": difference,
        "scores": scores,
    }


def execute(output: Path, index: int) -> dict:
    """Run four paired arms with shared saved initializer per mesh."""
    frozen = verify(output)
    if not 0 <= index < len(frozen["cases"]):
        raise ValueError("unknown case index")
    case = frozen["cases"][index]
    root = output / f"results/case_{index:03d}"
    problem = read_json(output / f"problems/{index:03d}.json")
    candidate = CandidateModel.model_validate(problem["candidate"])
    context = ValidationContext.model_validate(problem["context"])
    model = compile_candidate(candidate, context)
    train, val = (unpack_split(problem["splits"][name]) for name in ("train", "val"))
    scale = TrainingScaler().fit(train).scales["target:v01"].standard_deviation
    plan = PiecewisePlan.model_validate(frozen["plan"])
    records = []
    for mesh in (1, 2):
        for method in METHODS:
            path = root / f"mesh{mesh}_{method}"
            result_file = path / "result.json"
            identity = content_hash([frozen["identity"], index, mesh, method])
            if result_file.exists():
                saved = read_json(result_file)
                if saved["identity"] != identity:
                    raise ValueError("result identity differs")
                records.append(saved)
                continue
            settings = plan.fit.model_copy(
                update={
                    "collocation_mesh_substeps": mesh,
                    "piecewise_refinement": method,
                    "piecewise_policy": "allow",
                }
            )
            record = {
                "identity": identity,
                "case": case,
                "method": method,
                "mesh": mesh,
                "test_data_opened": False,
                "llm_calls": 0,
            }
            started = monotonic()
            try:
                # Both arms use the same recorded collocation solution. The adapter
                # binds each arm directory separately before reading its initializer.
                shared = root / f"mesh{mesh}_initializer.json"
                if shared.exists() and not (path / "initializer.json").exists():
                    write_json(path / "initializer.json", read_json(shared))
                fit_file = path / "fit.json"
                if fit_file.exists():
                    fit = read_json(fit_file)
                elif (
                    method == "branch_sensitivity"
                    and (path / "fit_started.json").exists()
                ):
                    raise ValueError("interrupted branch fit; no fresh budget granted")
                else:
                    write_json(
                        path / "fit_started.json",
                        {"identity": identity},
                        immutable=True,
                    )
                    fit = fit_collocation_forward_sensitivity(
                        model,
                        train,
                        val,
                        settings,
                        path,
                        initial_parameters=problem["start"],
                    )
                    write_json(fit_file, fit)
                if (path / "initializer.json").exists() and not shared.exists():
                    write_json(
                        shared, read_json(path / "initializer.json"), immutable=True
                    )
                record.update(fit=fit, status=fit["status"])
                if fit.get("parameters"):
                    reference = (
                        read_json(output / f"references/{index:03d}.json")
                        if case["synthetic"]
                        else None
                    )
                    record["replays"] = {
                        split.name.value: replay(
                            model,
                            split,
                            fit["parameters"],
                            scale,
                            reference[split.name.value] if reference else None,
                        )
                        for split in (train, val)
                    }
                    if not all(row["pass"] for row in record["replays"].values()):
                        record["status"] = "replay_unverified"
            except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
                record.update(status="failed", error=str(error)[-2500:])
            record["seconds"] = monotonic() - started
            write_json(result_file, _finite_payload(record))
            records.append(record)
    return {"status": "complete", "records": records}


def summarize(output: Path) -> dict:
    """Read checkpointed results only: no ODE solves on the summary job."""
    frozen = verify(output)
    rows = []
    for case in frozen["cases"]:
        for mesh in (1, 2):
            for method in METHODS:
                path = (
                    output
                    / f"results/case_{case['index']:03d}"
                    / f"mesh{mesh}_{method}/result.json"
                )
                record = read_json(path) if path.exists() else {}
                supervisor_path = (
                    output / f"results/case_{case['index']:03d}/supervisor.json"
                )
                supervisor = (
                    read_json(supervisor_path) if supervisor_path.exists() else {}
                )
                if record and record["identity"] != content_hash(
                    [frozen["identity"], case["index"], mesh, method]
                ):
                    raise ValueError("result identity differs")
                fit = record.get("fit", {})
                replays = record.get("replays", {})
                rows.append(
                    {
                        "case": case["label"],
                        "mesh": mesh,
                        "method": method,
                        "status": record.get(
                            "status", supervisor.get("status", "missing")
                        ),
                        "train_nmse": fit.get("training", {}).get("normalized_mse")
                        if fit.get("training")
                        else None,
                        "validation_nmse": fit.get("validation", {}).get(
                            "normalized_mse"
                        )
                        if fit.get("validation")
                        else None,
                        "clean_validation_nmse": replays.get("val", {})
                        .get("scores", {})
                        .get("Radau", {})
                        .get("clean_nmse"),
                        "calls": fit.get("refinement", {}).get("actual_residual_calls"),
                        "fit_seconds": fit.get("refinement", {}).get("fit_seconds"),
                        "initializer_success": fit.get("initializer", {}).get(
                            "success"
                        ),
                        "error": record.get("error"),
                        "path": str(path.relative_to(output)),
                    }
                )
    report = {
        "identity": frozen["identity"],
        "rows": rows,
        "test_data_opened": False,
        "llm_calls": 0,
    }
    write_json(output / "summary.json", report)
    lines = [
        "# Exact piecewise fitting comparison",
        "",
        "C+S = shared collocation + branch-sensitivity heuristic; "
        "C+P = shared collocation + derivative-free polling.",
        "Same training observations and refinement ceilings; "
        "mesh changes add latent nodes, not observations.",
        "Completion means independent numerical replay, "
        "not optimizer convergence or scientific recovery.",
        "",
        "| Case | Mesh | Method | Status | Train NMSE | Validation NMSE | "
        "Clean validation NMSE | Calls |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            row[k]
            for k in (
                "case",
                "mesh",
                "method",
                "status",
                "train_nmse",
                "validation_nmse",
                "clean_validation_nmse",
                "calls",
            )
        ]
        lines.append(
            "| "
            + " | ".join(
                "—" if v is None else f"{v:.6g}" if isinstance(v, float) else str(v)
                for v in values
            )
            + " |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return report
