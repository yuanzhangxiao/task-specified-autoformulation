"""Measure the meal-driven glucose flux in frozen historical T1-easy equations."""

from __future__ import annotations

import argparse
import csv
import hashlib
from functools import partial
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.data import Trajectory
from autoformalism.expressions import (
    CompiledModel,
    ValidationContext,
    compile_candidate,
)
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import simulation
from autoformalism.fitting.models import FitConfig
from autoformalism.rebuttal import dalla_man
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.reference_integration import ReferenceSolver
from autoformalism.schemas import CandidateModel

SOURCE = Path("artifacts/t1-curves-round12-2026-09-22/models.json")
SOURCE_SEAL = "f87e9429fe8ac08d7e0a5ed34419b658b7de569bed6c9af927b8b67680bb9d65"
CASES = {
    "fasting": (),
    "meal_45": ((37, 45),),
    "meal_75": ((83, 75),),
    "split_75": ((43, 30), (128, 45)),
}
CHANNELS = {"Gp": "v01", "EGP": "v02", "Uii": "v03", "E": "v04", "Gt": "v05"}
END = 360


def flux_scores(time: np.ndarray, predicted: np.ndarray, truth: np.ndarray) -> dict:
    """Compare signed flux waveforms, integrated appearance and peak timing."""
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("expected an increasing time grid")
    if any(
        x.shape != time.shape or not np.isfinite(x).all()
        for x in (time, predicted, truth)
    ):
        raise ValueError("expected aligned finite fluxes")
    energy = float(truth @ truth)
    area = float(np.trapezoid(truth, time))
    if energy <= 0 or area <= 0:
        raise ValueError("reference meal response must have positive energy and area")
    error = predicted - truth
    peak = float(predicted.max())
    return {
        "response_rse": float((error @ error) / energy),
        "area_ratio": float(np.trapezoid(predicted, time) / area),
        "peak_error_mg_per_kg_min": peak - float(truth.max()),
        "peak_time_error_min": (
            float(time[predicted.argmax()] - time[truth.argmax()])
            if peak > 1e-8
            else None
        ),
    }


def reference(meals: tuple, method: str) -> dict:
    """Read private flux and the public basal preparation from the simulator."""
    run = dalla_man.simulate_dalla_man(
        meals=meals,
        duration=END,
        dt=1,
        variant="original",
        solver=ReferenceSolver(method=method),
    )
    basal = {name: float(run.derived[name][0]) for name in ("EGP", "E", "Uii")}
    for name in ("Gp", "Gt"):
        basal[name] = float(run.states[0, dalla_man.STATE_INDEX[name]])
    return {"ra": run.derived["Ra"].tolist(), "basal": basal}


def clamped_states(
    model: CompiledModel,
    trajectory: Trajectory,
    parameters: dict,
    initials: dict,
    method: str,
) -> np.ndarray:
    """Hold plasma glucose fixed while evolving every other fitted state."""
    context = model.validated.context
    (target,) = trajectory.targets
    direct = model.direct_state_observation_channels
    if set(direct.values()) != {target} or len(direct) != 1 or context.lagged_targets:
        raise ValueError("probe requires a unique direct target and no target forcing")
    index = model.state_names.index(next(iter(direct)))
    forcing = simulation.trajectory_forcing(model, trajectory)
    initial = simulation.trajectory_initial_state(
        model, trajectory, initials, parameters=parameters
    )
    deadline = monotonic() + 45
    rhs = partial(model.rhs, parameters=parameters, forcing=forcing)

    def clamped(t: float, state: np.ndarray) -> np.ndarray:
        if monotonic() > deadline:
            raise TimeoutError("clamped rollout exceeded 45 seconds")
        out = rhs(t, state)
        out[index] = 0.0
        return out

    cfg = FitConfig(
        integration_method=method, relative_tolerance=1e-10, absolute_tolerance=1e-12
    )
    states = simulation._adaptive_free_intervals(
        clamped,
        model,
        trajectory,
        initial,
        cfg,
        deadline,
        None,
    )
    for state in states.T:
        model.validate_state_constraints(state)
    return states


def glucose_rate(
    model: CompiledModel, trajectory: Trajectory, parameters: dict, states: np.ndarray
) -> np.ndarray:
    """Read the original glucose equation at supplied state/forcing combinations."""
    index = model.state_names.index(next(iter(model.direct_state_observation_channels)))
    forcing = simulation.trajectory_forcing(model, trajectory)
    flux = []
    for t, state in zip(trajectory.time, states.T, strict=True):
        flux.append(model.rhs(t, state, parameters, forcing)[index])
    return np.array(flux)


def public_clamp(cell: str, basal: dict, meals: tuple) -> Trajectory:
    """Construct synthetic clamps from basal public channels and the meal command."""
    time = np.arange(END + 1, dtype=float)
    named = "canonical_named_easy" in cell
    aliases = {k: k if named else v for k, v in CHANNELS.items()}
    pulse = np.zeros_like(time)
    for minute, grams in meals:
        pulse[minute] += grams
    return Trajectory(
        trajectory_id="fixed_basal",
        time=time,
        targets={aliases["Gp"]: np.full_like(time, basal["Gp"])},
        auxiliaries={
            aliases[k]: np.full_like(time, basal[k]) for k in basal if k != "Gp"
        },
        external_inputs={"meal_event_g" if named else "u01": pulse},
        fixed_covariates={},
        derivatives={},
    )


def model_record(job: dict, ref: dict) -> dict:
    """Score every planned schedule, retaining solver failures and disagreements."""
    candidate = CandidateModel.model_validate(job["candidate"])
    model = compile_candidate(
        candidate, ValidationContext.model_validate(job["context"])
    )
    rows = {}
    controls = {}
    for case, meals in CASES.items():
        tr = public_clamp(job["cell"], ref["fasting"]["basal"], meals)
        try:
            states = {
                method: clamped_states(
                    model, tr, job["parameters"], job["initials"], method
                )
                for method in ("Radau", "BDF")
            }
            rates = {
                m: glucose_rate(model, tr, job["parameters"], s)
                for m, s in states.items()
            }
            if case == "fasting":
                controls = {m: (states[m], rates[m]) for m in states}
                rows[case] = {"status": "control"}
                continue
            if not controls:
                raise ValueError("fasting control unavailable")
            primary, check = (rates[k] - controls[k][1] for k in ("Radau", "BDF"))
            # Reapply the same meal forcing at each solver's fasting internal state.
            direct, direct_check = (
                glucose_rate(model, tr, job["parameters"], s) - rate
                for s, rate in (controls["Radau"], controls["BDF"])
            )
            truth = np.subtract(ref[case]["ra"], ref["fasting"]["ra"])
            if not (
                np.allclose(primary, check, rtol=1e-4, atol=1e-6)
                and np.allclose(direct, direct_check, rtol=1e-4, atol=1e-6)
            ):
                raise ValueError("independent response solvers disagree")
            rows[case] = {
                "status": "ok",
                **flux_scores(tr.time, primary, truth),
                "fasting_memory_rse": flux_scores(tr.time, direct, truth)[
                    "response_rse"
                ],
                "delta_flux": primary.tolist(),
            }
        except (ArithmeticError, RuntimeError, TimeoutError, ValueError) as exc:
            rows[case] = {"status": "unresolved", "reason": str(exc)}
    return {
        "cell": job["cell"],
        "method": job["method"],
        "model_id": job["model_id"],
        "repetition": job["seed"],
        "model_sha256": public.content_sha256(candidate),
        "cases": rows,
    }


def run(source: Path, root: Path) -> None:
    """Bind the cohort, schedules and runtime before computing reusable checkpoints."""
    bundle = sealed_read(source)
    if bundle["artifact_sha256"] != SOURCE_SEAL:
        raise ValueError("expected the frozen canonical round-12 model archive")
    plan = sealed_write(
        root / "plan.json",
        {
            "protocol": "t1-meal-appearance-clamp-1",
            "source_seal": SOURCE_SEAL,
            "horizon_min": END,
            "sample_interval_min": 1,
            "meal_schedules": {k: list(map(list, v)) for k, v in CASES.items()},
            "solver_source_sha256": {
                name: hashlib.sha256(
                    (Path(dalla_man.__file__).parents[1] / name).read_bytes()
                ).hexdigest()
                for name in (
                    "rebuttal/dalla_man.py",
                    "input_schedule.py",
                    "reference_integration.py",
                    "fitting/simulation.py",
                    "expressions/compiler.py",
                    "expressions/validation.py",
                    "expressions/parser.py",
                )
            },
            "runtime": public._runtime(),
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "models": [j["model_id"] for j in bundle["models"]],
            "unavailable_in_source": bundle["unavailable"],
        },
    )
    key = plan["artifact_sha256"]
    path = root / "reference.json"
    if not path.exists():
        refs = {}
        for case, meals in CASES.items():
            first, other = (reference(meals, m) for m in ("Radau", "DOP853"))
            np.testing.assert_allclose(first["ra"], other["ra"], rtol=1e-6, atol=1e-8)
            refs[case] = first
        sealed_write(path, {"plan_sha256": key, "cases": refs})
    ref = sealed_read(path)
    if ref["plan_sha256"] != key:
        raise ValueError("reference belongs to a different plan")
    scores = []
    for job in bundle["models"]:
        path = root / "models" / (job["model_id"] + ".json")
        if not path.exists():
            result = model_record(job, ref["cases"])
            sealed_write(path, {**result, "plan_sha256": key})
        record = sealed_read(path)
        if record["plan_sha256"] != key or record["model_id"] != job["model_id"]:
            raise ValueError("model checkpoint identity differs")
        for case, row in record["cases"].items():
            if case == "fasting":
                continue
            scores.append(
                {
                    **{
                        k: record[k]
                        for k in ("model_id", "repetition", "method", "cell")
                    },
                    "case": case,
                    **{k: v for k, v in row.items() if k != "delta_flux"},
                }
            )
    with (root / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for row in scores for k in row))
        )
        writer.writeheader()
        writer.writerows(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=SOURCE)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with public._lock(args.output):
        run(args.models, args.output)
