"""Authenticate T2 disposal starts and check coupled rollouts on dense meshes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.optimize import minimize_scalar

from autoformalism.expressions.compiler import SAFE_DIVISION_EPSILON
from autoformalism.research.trajectory_function_fitting import _Rollout, _rows
from autoformalism.staged_topology import content_hash
from scripts import positive_insulin_refit_inputs as intake
from scripts.positive_insulin_refit_inputs import load_training as load_training

MODELS = (
    "015-parent",
    "015-add-tissue-start",
    "015-add-tissue-feasible",
    "015-delayed-meal-start",
)
MANIFEST = "1bdad67dd738fff3f8aaf25eb5e307beca9d1dc671f2a19b3a1fc165674ae8e9"
INSULIN = "e715a0da48dc9a594d3489a3cd299fb505a6656f2eea1cd34e9cb16c022cd1f1"
DOMAIN_TOL = 1e-8
ACTIVE_TOL = 1e-6
SUBDIVISIONS = 16
GRID_TOL = 1e-5
TARGETS = intake.TARGETS
DELAYED_GP = "meal_balance_offset+meal_balance_egp*EGP-U+meal_balance_gain*D_meal"
DELAYED_RATE = "(meal_rate_g_per_min-D_meal)/meal_balance_tau"


def _members(archive: Path, names: tuple[str, ...]) -> dict:
    """Read only requested JSON members from the registered exchange."""
    with ZipFile(archive) as z:
        manifest = z.read("manifest.json")
        if hashlib.sha256(manifest).hexdigest() != MANIFEST:
            raise ValueError("Wrong T2 disposal handoff manifest")
        hashes = json.loads(manifest)["sha256"]
        if sorted(z.namelist()) != sorted([*hashes, "manifest.json"]):
            raise ValueError("Missing, duplicate or unexpected handoff member")
        values = {}
        for name in names:
            raw = z.read(name)
            if hashlib.sha256(raw).hexdigest() != hashes[name]:
                raise ValueError(f"Corrupt T2 disposal handoff member: {name}")
            values[name] = json.loads(raw)
        return values


def load_models(archive: Path) -> dict:
    """Authenticate all four complete graphs against their training sources."""
    files = tuple(f"models/{tag}.json" for tag in MODELS)
    sources = (
        "feasible-refit/training-freeze.json",
        "nested-qualification/training-freeze.json",
        "diagnosis/filtered-full-model.json",
    )
    data = _members(archive, files + sources)
    feasible, nested, delayed = (data[key] for key in sources)
    if feasible["validation_opened_at_freeze"] or nested["validation_opened_at_freeze"]:
        raise ValueError("Development access in training provenance")
    expected = (
        (feasible, feasible["models"]["parent"]),
        (nested, nested["models"]["selected"]),
        (feasible, feasible["models"]["selected"]),
        (delayed, delayed["model"]),
    )
    models = {}
    for tag, name, (src, model) in zip(MODELS, files, expected, strict=True):
        rec = data[name]
        if not (
            rec["name"] == tag
            and rec["model"] == model
            and rec["source_sha256"] == src["artifact_sha256"]
            and rec["model_sha256"] == content_hash(model)
        ):
            raise ValueError(f"Model and training source differ: {tag}")
        if reasons := parameter_violations(model):
            raise ValueError(
                f"Ineligible starting parameter declaration: {tag}, {reasons}"
            )
        models[tag] = model
    return models


def load_validation(path: Path) -> list[dict]:
    """Load the registered development cohort after the caller freezes training."""
    plan = {
        "training_intake": intake.original.campaign.TRAIN,
        "validation_intake": intake.original.campaign.VALIDATION,
    }
    return [
        r.model_dump(mode="json") for r in intake.original.read_validation(path, plan)
    ]


def parameter_violations(model: dict) -> list[str]:
    """Check the frozen insulin law and the signed, literal glucose balance."""
    f, g = model["formula"], model["graph"]
    p = f["parameters"]
    reasons = []
    insulin = [
        f["equations"]["I"],
        g["initializers"]["I"],
        model["knots"],
        model["scalar_model"],
        {k: v for k, v in p.items() if k.startswith("pos_")},
    ]
    if content_hash(insulin) != INSULIN:
        reasons.append("insulin law, initializer, knots or coefficients changed")
    if not all(np.isfinite(list(p.values()))):
        reasons.append("nonfinite parameter vector")
    if min(p["afcoef10"], p["afcoef13"]) <= 0:
        reasons.append("nonpositive insulin-action or action-disposal gain")
    if "D_meal" in f["equations"]:
        if (
            f["equations"]["Gp"] != DELAYED_GP
            or f["equations"]["D_meal"] != DELAYED_RATE
            or f["initializers"].get("D_meal") != "0"
        ):
            reasons.append(
                "changed meal delivery equation, literal sink or zero initial"
            )
        if min(p["meal_balance_egp"], p["meal_balance_gain"]) <= 0:
            reasons.append("nonpositive EGP or delivery gain")
        if p["meal_balance_tau"] <= SAFE_DIVISION_EPSILON:
            reasons.append("meal delivery timescale at or below the division guard")
    return reasons


def _fine_arrays(engine, model, event, base, parts):
    """Independently propagate the complete graph at intermediate mesh times."""
    times = np.r_[
        np.linspace(base[:-1], base[1:], parts, endpoint=False).T.ravel(), base[-1]
    ]
    params = [model["formula"]["parameters"][k] for k in engine.parameters]
    initial = [event.initial[k] for k in engine.readings]
    u = np.array(
        [np.interp(times, event.time, event.supplied[k]) for k in engine.forcings]
    )
    args = (initial, u, np.diff(times)[None, :])
    pred, states = engine.run([(engine._flow(len(times)), args)], params)[0]
    values = dict(zip(TARGETS, pred, strict=True))
    if "D_meal" in engine.states:
        values["D_meal"] = states[engine.states.index("D_meal")]
    uii = np.interp(times, event.time, event.supplied["Uii"])
    values["U_minus_Uii"] = values["U"] - uii
    return times, values, states, u, uii


def _refined_minimum(times, gap, states, forcing, uii, engine, params):
    """Check smooth bracketed minima of dependent disposal with a local RK4 replay."""
    i = int(np.argmin(gap))
    lowest, when = float(gap[i]), float(times[i])
    minima = np.where((gap[1:-1] < gap[:-2]) & (gap[1:-1] <= gap[2:]))[0] + 1
    for mid in minima:
        # Split at the shared node; each half has affine supplied channels.
        for left in (mid - 1, mid):
            dt = times[left + 1] - times[left]
            if dt <= 1e-12:
                continue

            def evaluate(t, left=left, dt=dt):
                a = t / dt
                u0 = forcing[:, left]
                u = u0 + a * (forcing[:, left + 1] - u0)
                x = engine.step(states[:, left], params, u0, u, t)
                obs = engine.observe(x, params, u)
                return float(obs[TARGETS.index("U")]) - (
                    uii[left] + a * (uii[left + 1] - uii[left])
                )

            extremum = minimize_scalar(
                evaluate, method="bounded", bounds=(0, dt), options={"xatol": 1e-9}
            )
            if not extremum.success or not np.isfinite(extremum.fun):
                raise ArithmeticError("Unresolved local disposal minimum")
            if extremum.fun < lowest:
                lowest, when = float(extremum.fun), float(times[left] + extremum.x)
    return lowest, when


def _check_event(engine, model, row, event_score, split):
    event = intake._event(row, split)
    base = intake.original.mesh(event, model["knots"])
    times, values, states, u, uii = _fine_arrays(
        engine, model, event, base, SUBDIVISIONS
    )
    previous = _fine_arrays(engine, model, event, base, SUBDIVISIONS // 2)[1]
    common = np.searchsorted(times, event.time)
    grid_delta = max(
        float(np.max(abs(values[k][::2] - previous[k])) / model["scales"][k])
        for k in TARGETS
    )
    source_delta = max(
        float(
            np.max(abs(values[k][common] - event_score["predicted"][k]))
            / model["scales"][k]
        )
        for k in TARGETS
    )
    minimum = {k: float(v.min()) for k, v in values.items()}
    when = {k: float(times[np.argmin(v)]) for k, v in values.items()}
    minimum["U_minus_Uii"], when["U_minus_Uii"] = _refined_minimum(
        times,
        values["U_minus_Uii"],
        states,
        u,
        uii,
        engine,
        [model["formula"]["parameters"][n] for n in engine.parameters],
    )
    reasons = [f"negative {k}" for k, v in minimum.items() if v < -DOMAIN_TOL]
    if max(grid_delta, source_delta) > GRID_TOL:
        reasons.append("independent dense-grid replay discrepancy")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "nodes": len(times),
        "minimum_values": minimum,
        "minimum_times": when,
        "minimum_U_minus_Uii": minimum["U_minus_Uii"],
        "active_counts": {
            k: int(np.count_nonzero(v <= ACTIVE_TOL)) for k, v in values.items()
        },
        "max_scaled_dense_grid_change": grid_delta,
        "max_scaled_source_difference": source_delta,
    }


def replay(model: dict, rows: list[dict]) -> dict:
    """Score the source rollouts and independently check the frozen physical domains."""
    result = intake.replay(model, rows)
    split = rows[0]["trajectory_id"].split("_", 1)[0]
    reasons = parameter_violations(model)
    engine = _Rollout(model["graph"], model["formula"], TARGETS, 2)
    checked = []
    for row, e in zip(_rows(rows, split), result["events"], strict=True):
        if e["status"] != "scored":
            reasons.append(f"{e['event']} {e['status']}")
            continue
        try:
            d = _check_event(engine, model, row, e, split)
        except ArithmeticError as exc:
            d = {"eligible": False, "reasons": [str(exc)]}
        e["domain"] = d
        reasons.extend(f"{e['event']} {message}" for message in d["reasons"])
        if "minimum_values" in d:
            checked.append((e["event"], d))
    values, locations, counts = {}, {}, {}
    if len(checked) == len(rows):
        for key in checked[0][1]["minimum_values"]:
            label, d = min(checked, key=lambda x: x[1]["minimum_values"][key])
            values[key] = d["minimum_values"][key]
            locations[key] = {"event": label, "time": d["minimum_times"][key]}
            counts[key] = sum(d["active_counts"][key] for _, d in checked)
    p = model["formula"]["parameters"]
    result["domain"] = {
        "eligible": not reasons,
        "reasons": reasons,
        "minimum_values": values,
        "minimum_locations": locations,
        "minimum_U_minus_Uii": values.get("U_minus_Uii"),
        "active_counts": counts,
        "action_gains": {
            "action_from_I": p["afcoef10"] / 111.507257324,
            "disposal_from_action": p["afcoef13"],
        },
        "feasibility_tolerance": DOMAIN_TOL,
        "near_active_threshold": ACTIVE_TOL,
        "numerical_scope": "16 subdivisions per knot interval checked against 8. "
        "Local RK4 search around bracketed disposal minima. Finite-mesh evidence.",
    }
    return result


def verify_start(archive: Path, tag: str, model: dict, result: dict) -> dict:
    """Reproduce each reference training score from the authenticated source records."""
    if tag not in MODELS or model != load_models(archive)[tag]:
        raise ValueError("Mismatched starting vector")
    if (
        result["model_sha256"]
        != content_hash(intake.with_parameters(model, model["formula"]["parameters"]))
        or tuple(e["event"] for e in result["events"]) != intake.TRAIN_IDS
        or result["summary"]["failed_events"]
    ):
        raise ValueError("Partial or misidentified training replay")
    if tag in ("015-add-tissue-start", "015-delayed-meal-start"):
        name = "diagnosis/filtered-full-training.json"
        receipt = _members(archive, (name,))[name]
        delayed = tag == "015-delayed-meal-start"
        ref = receipt["training" if delayed else "parent_training"]
        key = "candidate_model" if delayed else "parent_model"
        if not receipt["training_complete"] or receipt["source"][key] != content_hash(
            model
        ):
            raise ValueError(
                "Delayed meal start/parent differs from training diagnosis"
            )
    else:
        name = "feasible-refit/train-report.json"
        receipt = _members(archive, (name,))[name]
        saved = receipt["models"]["parent" if tag == "015-parent" else "selected"]
        if saved["model_sha256"] != content_hash(model) or receipt["split"] != "train":
            raise ValueError("Registered training replay refers to different model")
        ref = saved["summary"]
    target = result["summary"]["available_per_target_nmse"]
    expected = ref["available_per_target_nmse"]
    if not all(
        np.isclose(target[k], expected[k], atol=1e-9, rtol=1e-9) for k in TARGETS
    ):
        raise ValueError("Starting target NMSEs differ from training receipt")
    return {
        "source": name,
        "source_digest": receipt["artifact_sha256"],
        "expected": expected,
        "actual": target,
        "pooled_nmse": result["summary"]["pooled_nmse"],
    }
