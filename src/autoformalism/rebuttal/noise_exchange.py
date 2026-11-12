"""Package frozen external-baseline picks for the T2 noise handoff, and forecast them.

The handoff (``noise_handoff.exchange``, supplied with the public release) calls
an adapter's ``fit`` and ``predict``. Here ``fit`` never searches: each method's
own runner searched and sealed every acquisition and seed under its frozen plan,
and ``fit`` reads that seal for one acquisition and seed, checks that it was
developed on exactly that acquisition's public files, writes one self-contained
model file, and reports the validation free-rollout NMSE computed with the
reader that forecasts. A task the service stopped is not the method's failure:
``fit`` raises, and the task resumes once the run has its outcome.

A forecast reads only what a blind query supplies: the clock, the two targets'
initial readings, the auxiliary's path and the measured inputs.

* Continuous models (SINDy, PySR, specification-informed SINDy, LLM-SR's
  expression form, LLM-ODE): Full's backend with the external-baseline grammar,
  Radau or BDF at rtol 1e-8 and atol 1e-10, piecewise-linear forcing restarted
  at each knot, the state carried across, 120 s a trajectory.
* The latent neural ODE and latent-state SINDy: their own replays with the
  same solvers and limits.
* D3 and HDTwinGen: their native one-step maps, x[k+1] = x[k] + f, on the
  one-minute grid, hybrid networks included; the supplied auxiliary is read as
  measured, as in their own recursive rollout. An increment is never scaled by
  the step or read as a rate.

The method code is pinned: ``fit`` and ``predict`` refuse to run unless this
checkout is clean and at the configuration's ``implementation_revision``.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.baselines import latent_ode, latent_sindy
from autoformalism.baselines.d3_rollout import NativeMap
from autoformalism.baselines.d3_rollout import predict as native_rollout
from autoformalism.baselines.hybrid_networks import NetworkSpec
from autoformalism.baselines.models import BaselineDevelopmentResult
from autoformalism.data import Trajectory
from autoformalism.expressions import (
    ValidationContext,
    baseline_validator,
    compile_candidate,
)
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.rebuttal import phase_c_baseline_mechanisms as mechanisms
from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate
from autoformalism.rebuttal.phase_c_baselines import (
    NOISE_CELLS,
    PhaseCBaselineCell,
    load_cell,
)
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.schemas import CandidateModel

FORMAT = "t2-noise-baseline-model-1"
RTOL, ATOL, SECONDS = 1e-8, 1e-10, 120.0
REPO = Path(__file__).resolve().parents[3]
#: Outcomes of the service, not the method: such a task is rerun.
SERVICE_STATUSES = frozenset({"endpoint_unavailable", "infrastructure_failure"})
NMSE_DEFINITION = (
    "validation free-rollout mean squared error of each target divided by that "
    "target's variance over the acquisition's pooled training samples, averaged "
    "over both targets and the four validation trajectories; each rollout starts "
    "from the trajectory's initial readings, as a blind query does"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def pinned(config: Mapping) -> str:
    """Refuse to run method code other than the configuration's revision."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    head = git("rev-parse", "HEAD")
    if head != config["implementation_revision"]:
        raise ValueError(
            f"checkout is at {head}, not the pinned {config['implementation_revision']}"
        )
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("the pinned checkout has local changes")
    return head


def blind_trajectory(inputs: Mapping) -> Trajectory:
    """The predictor-visible trajectory of a blind query: only initial readings."""
    time = np.asarray(inputs["time"], dtype=float)
    return Trajectory(
        trajectory_id="blind",
        time=time,
        targets={
            k: np.full_like(time, float(v)) for k, v in inputs["targets0"].items()
        },
        auxiliaries={
            k: np.asarray(v, dtype=float) for k, v in inputs["auxiliaries"].items()
        },
        external_inputs={
            k: np.asarray(v, dtype=float) for k, v in inputs["external_inputs"].items()
        },
        fixed_covariates={},
        derivatives={},
    )


def _query(trajectory: Trajectory) -> dict:
    """What a blind query would supply for a development trajectory."""
    return {
        "time": trajectory.time,
        "targets0": {name: float(v[0]) for name, v in trajectory.targets.items()},
        "auxiliaries": trajectory.auxiliaries,
        "external_inputs": trajectory.external_inputs,
    }


# --- forecasting --------------------------------------------------------------


def forecast(
    model: Mapping, trajectory: Trajectory, solver: str
) -> dict[str, np.ndarray]:
    """Both targets over the trajectory's clock, from its initial readings only."""
    context = ValidationContext.model_validate(model["context"])
    if set(trajectory.targets) != set(context.targets) or set(
        trajectory.external_inputs
    ) != set(context.external_inputs):
        raise ValueError("the query's channels differ from the model's")
    kind = model["kind"]
    if kind == "native_map":
        if solver != "native":
            raise ValueError("a native map takes its own one-step updates")
        steps = np.diff(np.asarray(trajectory.time, dtype=float))
        if np.max(np.abs(steps - model["step"])) > 1e-9 * max(1.0, model["step"]):
            raise ValueError("the query's grid is not the map's sample interval")
        native = NativeMap.build(
            CandidateModel.model_validate(model["candidate"]),
            model["parameters"],
            (*context.targets, *context.auxiliaries),
            (*context.external_inputs, *context.fixed_covariates),
            networks=tuple(
                NetworkSpec.model_validate(item) for item in model["networks"]
            ),
            network_weights=model["network_weights"],
        )
        states = native_rollout(
            native, trajectory, teacher_forced=False, seconds=SECONDS
        )
        return {t: states[:, native.states.index(t)] for t in context.targets}
    if solver not in ("Radau", "BDF"):
        raise ValueError("a continuous model is integrated by Radau or BDF")
    if kind in ("latent_ode", "latent_sindy"):
        replay = (
            latent_ode.LatentODE if kind == "latent_ode" else latent_sindy.LatentSINDy
        )
        ode = replay(model)
        supplied = {**trajectory.external_inputs, **trajectory.auxiliaries}
        inputs = np.stack(
            [np.asarray(supplied[n], dtype=float) for n in ode.inputs], -1
        )
        y, _ = ode.simulate(
            np.asarray(trajectory.time, dtype=float),
            np.array([trajectory.targets[t][0] for t in ode.targets], dtype=float),
            inputs,
            method=solver,
            rtol=RTOL,
            atol=ATOL,
            seconds=SECONDS,
        )
        return {t: y[:, i] for i, t in enumerate(ode.targets)}
    if kind != "continuous":
        raise ValueError(f"unknown model kind {kind!r}")
    compiled = compile_candidate(
        CandidateModel.model_validate(model["candidate"]),
        context,
        validator=baseline_validator(),
    )
    sim = simulate_trajectory(
        compiled,
        trajectory,
        model["parameters"],
        model["initials"],
        FitConfig(
            integration_method=solver,
            relative_tolerance=RTOL,
            absolute_tolerance=ATOL,
            allow_derivative_regression=False,
        ),
        reset_observed_states=False,
        deadline=monotonic() + SECONDS,
    )
    if not sim.success:
        raise ValueError(sim.message)
    return {t: np.asarray(sim.predictions[t], dtype=float) for t in context.targets}


def validation_nmse(model: Mapping, cell: PhaseCBaselineCell, solver: str) -> float:
    """The handoff's validation metric, read through ``forecast``."""
    targets = cell.context.targets
    pooled = {
        t: np.concatenate([tr.targets[t] for tr in cell.dataset.train.trajectories])
        for t in targets
    }
    variance = {t: float(np.var(values)) for t, values in pooled.items()}
    errors = []
    for trajectory in cell.dataset.validation.trajectories:
        curves = forecast(model, blind_trajectory(_query(trajectory)), solver)
        for t in targets:
            residual = np.asarray(curves[t]) - np.asarray(trajectory.targets[t])
            errors.append(float(np.mean(residual**2)) / variance[t])
    value = float(np.mean(errors))
    if not math.isfinite(value):
        raise ValueError("validation error is not finite")
    return value


# --- reading each method's seal -------------------------------------------------


def _recorded(result: Mapping) -> tuple[str, float] | tuple[None, None]:
    """The validation score the method's own run recorded, where it has one."""
    rollout = (result.get("evaluation") or {}).get("phase_b_rollout") or {}
    for key, value in (
        ("validation_normalized_mse", result.get("validation_normalized_mse")),
        ("development_rollout_error", result.get("development_rollout_error")),
        ("evaluation.phase_b_rollout.normalized_mse", rollout.get("normalized_mse")),
    ):
        if isinstance(value, int | float) and not isinstance(value, bool):
            return key, float(value)
    return None, None


def _failure(reason: str, sources: list, **extra) -> tuple[None, dict]:
    return None, {"status": "no_model", "reason": reason, "sources": sources, **extra}


def _plan_row(root: Path, cell: PhaseCBaselineCell, seed: int) -> tuple[dict, dict]:
    """The frozen plan and its one row for this acquisition and seed."""
    plan = sealed_read(root / "plan.json")
    name = cell.dataset.benchmark_id
    rows = [
        r
        for r in plan["rows"]
        if r["benchmark_id"] == name and int(r["repetition"]) == seed
    ]
    if len(rows) != 1:
        raise ValueError(
            f"the plan at {root} has no single task for {name} seed {seed}"
        )
    if rows[0]["public_identity"] != cell.identity:
        raise ValueError(f"the run at {root} was developed on other public files")
    return plan, rows[0]


def _sealed_result(root: Path, plan: dict, row: dict) -> tuple[Path, dict]:
    path = root / "results" / str(int(row["index"])) / "result.json"
    if not path.is_file():
        raise ValueError(f"{path} has no outcome yet; finish the run first")
    result = sealed_read(path)
    if result["plan_sha256"] != plan["artifact_sha256"]:
        raise ValueError(f"{path} belongs to another plan")
    if result.get("test_data_opened") is not False:
        raise ValueError(f"{path} records opened test data")
    if result["status"] in SERVICE_STATUSES:
        raise ValueError(
            f"{path} ended {result['status']}: rerun it, it is not a failure"
        )
    return path, result


def _continuous(
    candidate: CandidateModel, cell: PhaseCBaselineCell, parameters=None
) -> dict:
    return {
        "kind": "continuous",
        "candidate": candidate.model_dump(mode="json"),
        "context": cell.context.model_dump(mode="json"),
        "parameters": {k: float(v) for k, v in (parameters or {}).items()},
        "initials": {},
    }


def read_classical(native: Mapping, cell: PhaseCBaselineCell, seed: int):
    """SINDy or PySR: the selected training equations, one per target, as an ODE."""
    method, root = native["method"], Path(native["root"])
    name = cell.dataset.benchmark_id
    folder = root / "runs" / method / f"{name}_{cell.dataset.tier}_seed{seed}"
    identity_path = folder / "phase_c_identity.json"
    if json.loads(identity_path.read_text()) != cell.identity:
        raise ValueError(f"{folder} was developed on other public files")
    sources = [_source(root / "frozen" / "plan.json"), _source(identity_path)]
    result_path = folder / "result.json"
    if not result_path.is_file():
        status_path = folder / "run_status.json"
        status = json.loads(status_path.read_text())
        if status.get("status") in {"complete", "running", "pending"}:
            raise ValueError(f"{folder} has no result yet")
        return _failure(
            f"{method} ended {status['status']}: {status.get('error')}",
            [*sources, _source(status_path)],
            native_status=status["status"],
        )
    result = BaselineDevelopmentResult.model_validate_json(result_path.read_text())
    if (result.method, result.benchmark_id, result.tier, result.seed) != (
        method,
        name,
        cell.dataset.tier,
        seed,
    ) or result.test_data_opened is not False:
        raise ValueError(f"{result_path} belongs to another task or opened test data")
    model = _continuous(
        equation_candidate(method, result.equations, cell.context), cell
    )
    return model, {
        "status": "complete",
        "sources": [*sources, _source(result_path)],
        "recorded_validation": [
            "validation_normalized_mse",
            result.validation_normalized_mse,
        ],
        "selected_hyperparameters": result.selected_hyperparameters,
    }


def read_vendored(native: Mapping, cell: PhaseCBaselineCell, seed: int):
    """LLM-ODE under either prompt, or LLM-SR's expression form: the sealed system."""
    method, root = native["method"], Path(native["root"])
    plan, row = _plan_row(root, cell, seed)
    path, result = _sealed_result(root, plan, row)
    sources = [_source(root / "plan.json"), _source(path)]
    accounting = result.get("accounting")
    if result["status"] != "complete":
        return _failure(
            f"{method} ended {result['status']}: {result.get('error')}",
            sources,
            native_status=result["status"],
            accounting=accounting,
        )
    selection_path = path.with_name("native-selection.json")
    sealed = sealed_read(selection_path)
    if sealed["plan_sha256"] != plan["artifact_sha256"]:
        raise ValueError(f"{selection_path} names another plan")
    selection = BaselineDevelopmentResult.model_validate(sealed["selection"])
    if (selection.benchmark_id, selection.seed) != (cell.dataset.benchmark_id, seed):
        raise ValueError(f"{selection_path} belongs to another task")
    info = {"context": cell.context.model_dump(mode="json")}
    if method in ("llm_ode", "llm_ode_published_prompt"):
        sealed_model = mechanisms.llm_ode_model(selection, info)
    elif method == "llm_sr_expression":
        sealed_model = mechanisms.llm_sr_expression_model(selection, info)
    else:
        raise ValueError(f"no reader for the vendored method {method!r}")
    model = _continuous(
        CandidateModel.model_validate(sealed_model["candidate"]),
        cell,
        sealed_model["parameters"],
    )
    key, value = _recorded(result)
    return model, {
        "status": "complete",
        "sources": [*sources, _source(selection_path)],
        "recorded_validation": [key, value],
        "accounting": accounting,
    }


def read_d3_family(native: Mapping, cell: PhaseCBaselineCell, seed: int):
    """D3 or HDTwinGen: the native one-step map, with any hybrid networks."""
    method, root = native["method"], Path(native["root"])
    plan, row = _plan_row(root, cell, seed)
    path, result = _sealed_result(root, plan, row)
    sources = [_source(root / "plan.json"), _source(path)]
    accounting = result.get("accounting")
    if result["status"] != "complete":
        return _failure(
            f"{method} ended {result['status']}: {result.get('error')}",
            sources,
            native_status=result["status"],
            accounting=accounting,
        )
    selection_path = path.with_name("native-selection.json")
    sealed = sealed_read(selection_path)
    if sealed["plan_sha256"] != plan["artifact_sha256"]:
        raise ValueError(f"{selection_path} names another plan")
    payload = BaselineDevelopmentResult.model_validate(
        sealed["selection"]
    ).selection_payload
    times = cell.dataset.train.trajectories[0].time
    model = {
        "kind": "native_map",
        "candidate": payload["candidate"],
        "context": cell.context.model_dump(mode="json"),
        "parameters": {k: float(v) for k, v in payload["parameters"].items()},
        "networks": list(payload.get("networks", ())),
        "network_weights": payload.get("network_weights"),
        "step": float(np.diff(np.asarray(times, dtype=float))[0]),
    }
    key, value = _recorded(result)
    return model, {
        "status": "complete",
        "sources": [*sources, _source(selection_path)],
        "recorded_validation": [key, value],
        "selected_generation": result.get("selected_generation"),
        "selection_metric": result.get("selection_metric"),
        "accounting": accounting,
    }


def read_spec_sindy(native: Mapping, cell: PhaseCBaselineCell, seed: int):
    """Specification-informed SINDy: its equations, LLM features written inline."""
    root = Path(native["root"])
    plan, row = _plan_row(root, cell, seed)
    path, result = _sealed_result(root, plan, row)
    sources = [_source(root / "plan.json"), _source(path)]
    if result["status"] != "complete":
        return _failure(
            f"spec-informed SINDy ended {result['status']}: {result.get('error')}",
            sources,
            native_status=result["status"],
            accounting=result.get("accounting"),
        )
    candidate = equation_candidate(
        "llm_feature_sindy", result["equations"], cell.context
    )
    return _continuous(candidate, cell), {
        "status": "complete",
        "sources": sources,
        "recorded_validation": [
            "validation_normalized_mse",
            result["validation_normalized_mse"],
        ],
        "designed_features": result.get("designed_features"),
        "accounting": result.get("accounting"),
    }


def _read_latent(native: Mapping, cell: PhaseCBaselineCell, seed: int, kind: str):
    """A latent-state model the plan's validation replay chose for this seed."""
    root = Path(native["root"])
    plan = sealed_read(root / "plan.json")
    name = cell.dataset.benchmark_id
    if plan["cells"][name]["identity"] != cell.identity:
        raise ValueError(f"the {kind} plan at {root} read other public files")
    selection = sealed_read(root / "selection.json")
    if selection["plan_sha256"] != plan["artifact_sha256"]:
        raise ValueError(f"the {kind} selection belongs to another plan")
    rows = [
        r
        for r in selection["rows"]
        if r["benchmark_id"] == name and r["repetition"] == seed
    ]
    if len(rows) != 1:
        raise ValueError(
            f"the {kind} selection has no single row for {name} seed {seed}"
        )
    row = rows[0]
    sources = [_source(root / "plan.json"), _source(root / "selection.json")]
    if row["status"] != "complete":
        return _failure(f"{kind} ended {row['status']}", sources)
    model = {**row["model"], "kind": kind}
    record = {
        "status": "complete",
        "sources": sources,
        "recorded_validation": ["selection.validation_nmse", row["validation_nmse"]],
        "selected_latent_dimension": row["selected_latent_dimension"],
    }
    for key in ("selected_threshold", "kept_terms", "equations"):
        if key in row:
            record[key] = row[key]
    return model, record


def read_latent_ode(native: Mapping, cell: PhaseCBaselineCell, seed: int):
    """The latent neural ODE: the latent dimension its validation replay chose."""
    return _read_latent(native, cell, seed, "latent_ode")


def read_latent_sindy(native: Mapping, cell: PhaseCBaselineCell, seed: int):
    """Latent-state SINDy: the latent dimension and threshold validation chose."""
    return _read_latent(native, cell, seed, "latent_sindy")


READERS: dict[str, Callable] = {
    "classical": read_classical,
    "vendored": read_vendored,
    "d3_family": read_d3_family,
    "spec_sindy": read_spec_sindy,
    "latent_ode": read_latent_ode,
    "latent_sindy": read_latent_sindy,
}


# --- the adapter's two functions -------------------------------------------------


def _write_model(path: Path, record: dict) -> None:
    """One ZIP with the model record, written byte-for-byte reproducibly."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo("model.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, json.dumps(record, sort_keys=True, allow_nan=False))
    path.write_bytes(buffer.getvalue())


def read_model(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        record = json.loads(archive.read("model.json"))
    if record.get("format") != FORMAT:
        raise ValueError(f"{path} is not a T2 noise baseline model")
    return record


def fit(task: Mapping, cell_dir: Path, task_dir: Path, config: Mapping) -> dict:
    """Package one acquisition and seed's sealed pick; never search."""
    revision = pinned(config)
    name, seed = task["cell"], int(task["seed"])
    if name not in NOISE_CELLS or cell_dir.name != name:
        raise ValueError("the handoff passed a cell outside the noise study")
    files = {f: _sha256(cell_dir / f) for f in task["public_files"]}
    if files != task["public_files"]:
        raise ValueError("the acquisition's public files differ from the task")
    cell = load_cell(cell_dir.parent.parent, name)
    native = config["native"]
    model, record = READERS[native["kind"]](native, cell, seed)
    receipt = {
        "cell": name,
        "seed": seed,
        "public_files": task["public_files"],
        "method": config["method"],
        "configuration": dict(config),
        "implementation_revision": revision,
        "public_identity": cell.identity,
        "native": record,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    task_dir.mkdir(parents=True, exist_ok=True)
    if model is None:
        receipt["status"] = "no_model"
        (task_dir / "fit_receipt.json").write_text(json.dumps(receipt, indent=1))
        return {"status": "no_model", "reason": record["reason"]}
    solver = "native" if model["kind"] == "native_map" else "Radau"
    started = monotonic()
    try:
        value = validation_nmse(model, cell, solver)
    except (ValueError, ArithmeticError, TimeoutError) as exc:
        reason = f"validation free rollout failed ({solver}): {exc}"[:500]
        receipt.update(status="no_model", reason=reason)
        (task_dir / "fit_receipt.json").write_text(json.dumps(receipt, indent=1))
        return {"status": "no_model", "reason": reason}
    _write_model(
        task_dir / "model.bin",
        {
            "format": FORMAT,
            "method": config["method"],
            "cell": name,
            "seed": seed,
            "public_identity": cell.identity,
            "model": model,
        },
    )
    key, recorded = record.get("recorded_validation", [None, None])
    receipt.update(
        status="ready",
        validation_nmse=value,
        validation_nmse_definition=NMSE_DEFINITION,
        validation_solver=solver,
        validation_seconds=monotonic() - started,
        rehearsal={
            "native_key": key,
            "native_recorded": recorded,
            "relative_difference": (
                abs(value - recorded) / abs(recorded)
                if isinstance(recorded, int | float) and recorded
                else None
            ),
        },
    )
    (task_dir / "fit_receipt.json").write_text(json.dumps(receipt, indent=1))
    return {"status": "ready", "validation_nmse": value}


def predict(model_path: Path, inputs: Mapping, solver: str, config: Mapping) -> dict:
    """Both targets for one blind query, in physical units, from the frozen file."""
    pinned(config)
    record = read_model(model_path)
    if record["method"] != config["method"]:
        raise ValueError("the model file belongs to another method")
    curves = forecast(record["model"], blind_trajectory(inputs), solver)
    return {t: [float(v) for v in curves[t]] for t in ("Gp", "I")}
