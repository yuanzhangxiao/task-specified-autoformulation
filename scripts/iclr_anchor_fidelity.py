"""Score frozen models on physiological and reactor interventions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.benchmarks.phase_b_generation import (
    PhaseBProtocol,
    phase_b_protocols,
    simulate_phase_b,
)
from autoformalism.benchmarks.phase_c_protocols import phase_c_protocols
from autoformalism.benchmarks.phase_c_release import SPEC_PATHS, load_public_cell
from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.expressions.diagnostics import (
    ModelValidationError,
    RuntimeExpressionError,
)
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting import (
    FitConfig,
    simulate_trajectory,
)
from autoformalism.fitting import (
    public_fitting as public,
)
from autoformalism.fitting.simulation import trajectory_forcing
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.reference_integration import ReferenceSolver
from autoformalism.schemas import CandidateModel
from autoformalism.search.identity import candidate_identity
from scripts import evaluate_t2_latent_dynamics as probe
from scripts import iclr_reference_fits as fits
from scripts.analyze_t2_hard_selection import match_pair

VERSION = "iclr-independent-anchor-assays-1"
LEGACY = probe.CELL
CELLS = (LEGACY, *fits.CELLS)
UNITS = {
    "Gp": "mg/kg",
    "I": "pmol/L",
    "T": "K",
    "C": "relative concentration",
    "Tj": "K",
    "Ra": "mg/kg/min",
    "Uid": "mg/kg/min",
    "EGP": "mg/kg/min",
    "X": "simulator remote action",
    "reaction_heat": "K/normalized time unit",
    "jacket_heat": "K/normalized time unit",
}


def base_cell(cell):
    return fits.T2 if cell == LEGACY else cell


def cases(cell: str, mode: str) -> dict:
    """Freeze input contrasts, controls and windows before simulation."""
    if cell not in CELLS or mode not in ("development", "confirmation"):
        raise ValueError("unsupported anchor or study mode")
    is_cstr = cell == fits.CSTR
    roster = (
        phase_c_protocols("cstr")
        if is_cstr
        else phase_b_protocols(
            "dalla_man",
            task="T2" if cell in (LEGACY, fits.T2) else "T1",
            input_contract="legacy-events-1"
            if cell == LEGACY
            else "continuous-rates-1",
        )
    )
    by_id = {r.protocol_id: r for r in roster if r.split == "train"}
    calib = (
        ["train_zero", "train_cf_plus_step", "train_tjf_pulse", "train_tf_minus_step"]
        if is_cstr
        else ["train_basal", "train_meal_30g_t0", "train_meal_90g_t0"]
        + (["train_insulin_pulse"] if cell in (LEGACY, fits.T2) else [])
    )
    table = {
        k: {"protocol": by_id[k].model_dump(mode="json"), "role": "train_calibration"}
        for k in calib
    }
    base = by_id[calib[0]]

    def add(name, spec, control=None, window=None):
        config = base.model_dump(mode="json") | {
            "protocol_id": "test_extra_" + name,
            "split": "test",
            "specification": spec,
        }
        table[name] = {
            "protocol": config,
            "role": "contrast" if control else "control",
            "control": control,
            "window": window,
        }

    # Disjoint fixed schedules for the retrospective assay and future frozen searches.
    if is_cstr:
        shift = 0 if mode == "development" else 1.4
        add("basal", {"kind": "constant", "value": [0, 0, 0]})
        for label, index, amp, on, duration in (
            ("feed_low", 0, -0.65, 4.3, 3.4),
            ("feed_high", 0, 0.88, 12.2, 5.1),
            ("jacket_cool", 2, -0.83, 7.3, 3.6),
            ("jacket_warm", 2, 0.68, 15.1, 5.4),
        ):
            on = round(on + shift, 1)
            amp *= 1 if mode == "development" else 0.93
            v = [0.0, 0.0, 0.0]
            v[index] = amp
            add(
                label,
                {
                    "kind": "step",
                    "start": on,
                    "end": round(on + duration, 1),
                    "value": v,
                },
                "basal",
                [on + 1, on + duration + 1],
            )
    else:
        shift = 0 if mode == "development" else 8
        meal = 48 if mode == "development" else 53
        add("fasting", {"meals": []})
        add(
            "fed", {"meals": [[33 + shift, meal]]}, "fasting", [48 + shift, 110 + shift]
        )
        add(
            "split",
            {"meals": [[33 + shift, meal / 2], [101 + shift, meal / 2]]},
            "fed",
            [45 + shift, 100 + shift],
        )
        if cell in (LEGACY, fits.T2):
            for label, base_name, amps, on, duration in (
                ("short_low", "fasting", 0.27, 74, 17),
                ("short_high", "fasting", 0.64, 74, 17),
                ("late_long", "fasting", 0.42, 147, 29),
                ("fed_long", "fed", 0.42, 84, 29),
            ):
                on += shift
                amps *= 1 if mode == "development" else 0.91
                add(
                    label,
                    {
                        "meals": table[base_name]["protocol"]["specification"]["meals"],
                        "insulin": [[on, on + duration, amps]],
                    },
                    base_name,
                    [on + 10, on + 55],
                )
    return table


def training_sd(cell, public_root, legacy_root):
    if cell == LEGACY:
        data = np.genfromtxt(
            legacy_root / cell / "train.csv", delimiter=",", names=True
        )
        return {n: float(np.std(data[n])) for n in fits.TARGETS[fits.T2]}
    _, training, _ = load_public_cell(public_root / cell)
    return {
        n: float(np.std(np.concatenate([tr.targets[n] for tr in training.rows])))
        for n in fits.TARGETS[cell]
    }


ALIASES = {"acquisition_00": fits.T2, "sensor_00": fits.CSTR}


def canonicalize(row: dict, public_root: Path, stress_public_root: Path | None):
    """Verify the public source and both clean splits for each stress alias."""
    source_id = row["benchmark_id"]
    if source_id not in ALIASES and source_id not in fits.CELLS:
        return row
    cell = ALIASES.get(source_id, source_id)
    source_root = stress_public_root if source_id in ALIASES else public_root
    if source_root is None:
        raise ValueError("stress aliases require --stress-public-root")
    directory = source_root / source_id
    if (
        hashlib.sha256((directory / "specification.json").read_bytes()).hexdigest()
        != row["public_spec_sha256"]
    ):
        raise ValueError("frozen model public specification hash differs")
    hashes = row.get("public_file_sha256", {})
    if not {"train.json", "val.json", "specification.json"} <= hashes.keys():
        raise ValueError("missing public search file identities")
    for filename in ("train.json", "val.json", "specification.json"):
        if (
            hashlib.sha256((directory / filename).read_bytes()).hexdigest()
            != hashes[filename]
        ):
            raise ValueError("changed public anchor data")
    if source_id in ALIASES:
        a, at, av = load_public_cell(directory)
        b, bt, bv = load_public_cell(public_root / cell)
        if (
            a["public_prompt"] != b["public_prompt"]
            or at.rows != bt.rows
            or av.rows != bv.rows
        ):
            raise ValueError("stress anchor differs from the clean canonical release")
    return {**row, "source_benchmark_id": source_id, "benchmark_id": cell}


def verify_controlled_bundle(
    source: Path, bundle: dict, *, allow_development_pilot: bool = False
) -> None:
    """Cross-check every planned task against the search and final freeze receipts."""
    source_plan = sealed_read(source.parent / "plan.json")
    if source_plan.get("development_pilot") and not allow_development_pilot:
        raise ValueError("development pilot is excluded from scientific evaluation")
    receipt = sealed_read(source.parent / "evaluation_freeze.json")
    rows = bundle["rows"]
    expected = [
        task["task_id"] for block in source_plan["blocks"] for task in block["tasks"]
    ]
    actual = [r["source_id"] for r in rows]
    if (
        bundle["plan_sha256"] != source_plan["artifact_sha256"]
        or receipt["plan_sha256"] != source_plan["artifact_sha256"]
        or bundle["selection_freeze_sha256"] != receipt["artifact_sha256"]
        or public.content_sha256(rows) != receipt["roster_sha256"]
        or sorted(expected) != sorted(actual)
        or len(set(actual)) != len(actual)
        or any(r["status"] == "pending" for r in rows)
    ):
        raise ValueError("incomplete or changed frozen search roster")


def prepare(
    sources,
    root,
    mode,
    public_root,
    legacy_root,
    data_root,
    readouts=None,
    stress_public_root=None,
):
    """Freeze subjects, pair calipers and reviewed direct readouts."""
    cohorts = []
    omitted = []
    for source in sources:
        bundle = sealed_read(source)
        protocol = bundle["protocol"]
        if protocol not in (
            "iclr-controlled-search-models-1",
            "public-mechanism-model-bundle-1",
            fits.VERSION,
        ):
            raise ValueError("unknown subject bundle format")
        if mode == "confirmation" and protocol == "public-mechanism-model-bundle-1":
            raise ValueError("archived selection may only use the development schedule")
        if protocol == "iclr-controlled-search-models-1":
            verify_controlled_bundle(source, bundle)
        for r in bundle["rows"]:
            if protocol == "iclr-controlled-search-models-1":
                r = canonicalize(r, public_root, stress_public_root)
            cell = r["benchmark_id"]
            if cell not in CELLS or (
                protocol == "public-mechanism-model-bundle-1" and "task" not in r
            ):
                omitted.append(
                    {
                        "benchmark_id": cell,
                        "method": r["method"],
                        "repetition": r["repetition"],
                    }
                )
                continue
            if (
                protocol
                in (
                    "iclr-controlled-search-models-1",
                    "public-mechanism-model-bundle-1",
                )
                and r.get("semantics", "continuous_time") != "continuous_time"
            ):
                raise ValueError("unsupported search semantics")
            cohorts.append(
                {
                    **r,
                    "source_bundle_sha256": bundle["artifact_sha256"],
                    "source_protocol": protocol,
                }
            )
    if mode == "confirmation" and not any(
        r["source_protocol"] == "iclr-controlled-search-models-1" for r in cohorts
    ):
        raise ValueError("confirmation requires a frozen internal search cohort")
    ids = [(r["benchmark_id"], r["method"], r["repetition"]) for r in cohorts]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate internal subjects")
    annotations = {} if readouts is None else json.loads(readouts.read_text())
    unknown = set(annotations)
    summaries = []
    for i, r in enumerate(cohorts):
        c = base_cell(r["benchmark_id"])
        digest = public.content_sha256(r["candidate"]) if r.get("candidate") else None
        mappings = annotations.get(digest, {})
        unknown.discard(digest)
        for name, v in mappings.items():
            if (
                name not in fits.HIDDEN[c]
                or v["unit"] != UNITS[name]
                or not v["evidence"].strip()
            ):
                raise ValueError("require a unit-verified, documented physical mapping")
            RestrictedParser().parse(v["expression"], location="direct readout")
        r["direct_readouts"] = mappings
        if r["source_protocol"] == fits.VERSION:
            continue
        item = {
            "source_id": str(i),
            "status": r["status"],
            "method": r["method"],
            "seed": r["repetition"],
            "cell": r["benchmark_id"],
            "group": r.get("task", {}).get("arm", "controlled"),
        }
        if r["status"] == "ready":
            m = CandidateModel.model_validate(r["candidate"])
            item.update(
                states=len(m.states),
                parameters=len(m.parameters),
                validation_nmse=r.get("validation_nmse"),
                functional_sha256=candidate_identity(m).functional_sha256,
            )
        summaries.append(item)
    if unknown:
        raise ValueError("readout references unknown candidate hashes")
    # Wrong-specification arms cannot answer the same-task matched-mechanism question.
    pairs = [
        match_pair(a, b)
        for a, b in itertools.combinations(summaries, 2)
        if (a["cell"], a["seed"], a["group"]) == (b["cell"], b["seed"], b["group"])
        and all(
            x["method"] in ("full", "fixed_gamma", "no_latent")
            or x["method"].startswith("autoformalism:")
            for x in (a, b)
        )
    ]
    cells = sorted({r["benchmark_id"] for r in cohorts})
    plan = {
        "protocol": VERSION,
        "mode": mode,
        "source": source_hash(),
        "runtime": public._runtime(),
        "roster": cohorts,
        "skipped_out_of_scope": omitted,
        "pairs": pairs,
        "training_sd": {
            cell: training_sd(cell, public_root, legacy_root) for cell in cells
        },
        "schedules": {cell: cases(cell, mode) for cell in cells},
        "cstr_spec_sha256": hashlib.sha256(
            (data_root / SPEC_PATHS["cstr"]).read_bytes()
        ).hexdigest()
        if fits.CSTR in cells
        else None,
    }
    return sealed_write(root / "frozen_selection_manifest.json", plan)


def source_hash():
    """Pin generator, integration, calibration and observation implementations."""
    from autoformalism import input_schedule, reference_integration
    from autoformalism.benchmarks import phase_b_generation, phase_c_protocols
    from autoformalism.expressions import compiler
    from autoformalism.fitting import simulation
    from autoformalism.rebuttal import dalla_man

    modules = [
        fits,
        probe,
        input_schedule,
        reference_integration,
        phase_b_generation,
        phase_c_protocols,
        compiler,
        simulation,
        dalla_man,
    ]
    paths = [Path(__file__), *(Path(m.__file__) for m in modules)]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def reference(cell, item, data_root, method):
    p = PhaseBProtocol.model_validate(item["protocol"])
    traj = simulate_phase_b(
        p,
        dynamics="perturbed" if cell == fits.T1P else "canonical",
        data_root=data_root,
        solver=ReferenceSolver(method=method),
    )
    fields = {n: traj.states[:, i] for i, n in enumerate(traj.state_names)}
    fields.update(traj.derived)
    fields.update({n: traj.inputs[:, i] for i, n in enumerate(traj.input_names)})
    if cell == fits.CSTR:
        params = json.loads((data_root / SPEC_PATHS["cstr"]).read_text())["parameters"]
        fields["reaction_heat"] = (
            params["source_gain"]
            * params["k0"]
            * np.exp(-params["E_over_R"] / np.maximum(fields["T"], 250))
            * np.maximum(fields["C"], 0)
        )
        fields["jacket_heat"] = params["exchange_rate"] * (fields["Tj"] - fields["T"])
    c = base_cell(cell)
    channels = [*fits.TARGETS[c], *fits.HIDDEN[c], *fits.CHANNELS[c]]
    if cell == LEGACY:
        channels = [
            "meal_event_g" if k == "meal_rate_g_per_min" else k for k in channels
        ]
    return {"time": traj.time.tolist(), **{n: fields[n].tolist() for n in channels}}


def trajectory(cell, record):
    c = base_cell(cell)
    targets = fits.TARGETS[c]
    inputs = (
        ("Cf", "Tf", "Tjf")
        if c == fits.CSTR
        else (
            ("meal_event_g" if cell == LEGACY else "meal_rate_g_per_min",)
            + (("insulin_pmol_per_kg_min",) if c == fits.T2 else ())
        )
    )
    aux = set(fits.CHANNELS[c]) - set(inputs) - {"meal_rate_g_per_min"}
    return Trajectory(
        trajectory_id="public_forcing",
        time=np.asarray(record["time"]),
        targets={n: np.full(len(record["time"]), record[n][0]) for n in targets},
        external_inputs={n: np.asarray(record[n]) for n in inputs},
        auxiliaries={n: np.asarray(record[n]) for n in aux},
        fixed_covariates={},
        derivatives={},
    )


def predict(row, tr, solver):
    c = base_cell(row["benchmark_id"])
    if row["source_protocol"] == fits.VERSION:
        f = fits.family(
            c, len(tr.time) - 1, 4 if solver == "Radau" else 8, jacobian=False
        )
        pred = np.asarray(
            f(
                [tr.targets[n][0] for n in fits.TARGETS[c]],
                fits.public_inputs(c, tr),
                row["fit"]["theta"],
            )
        )
        names = fits.TARGETS[c] + ["direct." + n for n in fits.HIDDEN[c]]
        return {n: a.tolist() for n, a in zip(names, pred, strict=True)}
    context = ValidationContext.model_validate(row["context"])
    if (
        set(context.targets) != set(tr.targets)
        or set(context.auxiliaries) != set(tr.auxiliaries)
        or set(context.external_inputs) != set(tr.external_inputs)
        or context.lagged_targets
        or context.fixed_covariates
    ):
        raise ValueError("model violates public input or initialization contract")
    candidate = CandidateModel.model_validate(row["candidate"])
    model = compile_candidate(candidate, context)
    sim = simulate_trajectory(
        model,
        tr,
        row["parameters"],
        row["initials"],
        FitConfig(
            integration_method=solver,
            relative_tolerance=1e-8,
            absolute_tolerance=1e-10,
            allow_derivative_regression=False,
        ),
        reset_observed_states=False,
        deadline=monotonic() + 45,
    )
    if not sim.success:
        raise ValueError(sim.message)
    result = {n: y.tolist() for n, y in sim.predictions.items()}
    observed = (
        set(model.direct_state_observation_channels)
        | set(context.targets)
        | set(context.auxiliaries)
    )
    for i, name in enumerate(model.state_names):
        if candidate.states[i].kind.value == "latent" and name not in observed:
            result["latent." + name] = sim.states[i].tolist()
    for name, mapping in row["direct_readouts"].items():
        expr = RestrictedParser().parse(
            mapping["expression"], location="reviewed readout"
        )
        force = trajectory_forcing(model, tr)
        result["direct." + name] = [
            model.evaluate_expression(expr, t, state, row["parameters"], force)
            for t, state in zip(tr.time, sim.states.T, strict=True)
        ]
    return result


def scores(time, reference, control, pred, pred_control, window, scale):
    data = probe.metrics(time, reference, pred, control, pred_control)
    r = np.asarray(reference) - control
    y = np.asarray(pred) - pred_control
    t = np.asarray(time)
    idx = (t >= window[0]) & (t <= window[1])
    a, b = float(np.trapezoid(r[idx], t[idx])), float(np.trapezoid(y[idx], t[idx]))
    floor = 1e-8 * max(1.0, max(abs(np.asarray(reference)))) * (window[1] - window[0])
    return data | {
        "absolute_nmse": data["rmse"] ** 2 / scale**2 if scale > 1e-12 else None,
        "reference_effect_auc": float(np.trapezoid(r, t)),
        "predicted_effect_auc": float(np.trapezoid(y, t)),
        "reference_window_auc": a,
        "predicted_window_auc": b,
        "window_direction_correct": bool(abs(b) > floor and np.sign(a) == np.sign(b))
        if abs(a) > floor
        else None,
    }


def evaluate_one(args):
    root, i, row, schedules, refs, plan_hash, scales = args
    c = base_cell(row["benchmark_id"])
    folder = root / "subjects" / f"{i:04d}"
    curves = {}
    cross = {}
    failures = {}
    for k, r in refs.items():
        if row["status"] != "ready":
            continue
        path = folder / f"{k}.json"
        if not path.exists():
            reservation = folder / (k + ".started.json")
            if reservation.exists():
                value = {"error": "interrupted_rollout"}
            else:
                sealed_write(reservation, {"plan": plan_hash, "case": k, "subject": i})
                try:
                    tr = trajectory(row["benchmark_id"], r)
                    a, b = (predict(row, tr, solver) for solver in ("Radau", "BDF"))
                    gap = probe.solver_agreement(a, b, 1e-4)
                    value = {"curves": a, "verification_curves": b, "solver_gap": gap}
                except (
                    ValueError,
                    RuntimeError,
                    ArithmeticError,
                    ModelValidationError,
                    RuntimeExpressionError,
                ) as e:
                    value = {"error": str(e)}
            sealed_write(path, {"plan": plan_hash, **value})
        record = sealed_read(path)
        if record["plan"] != plan_hash:
            raise ValueError("stale subject checkpoint")
        if "error" in record:
            failures[k] = record["error"]
        else:
            curves[k], cross[k] = record["curves"], record["verification_curves"]
    calib = [k for k, v in schedules.items() if v["role"] == "train_calibration"]
    decoded = {}
    hidden_scale = {}
    result = []
    for h in fits.HIDDEN[c]:
        truth = np.concatenate([refs[k][h] for k in calib])
        hidden_scale[h] = float(np.std(truth))
        if all(k in curves for k in calib):
            variables = {
                n: np.concatenate([curves[k][n] for k in calib])
                for n in curves[calib[0]]
                if n.startswith("latent.")
            }
            decoded[h] = probe.calibrate(variables, truth)
    for k, item in schedules.items():
        if item["role"] != "contrast":
            continue
        base = item["control"]
        for kind, n in [("target", n) for n in fits.TARGETS[c]] + [
            (kind, h) for h in fits.HIDDEN[c] for kind in ("direct", "oracle_affine")
        ]:
            status = row["status"]
            key = n if kind == "target" else "direct." + n
            if row["status"] == "ready":
                if k not in curves or base not in curves:
                    status = "rollout_unavailable"
                elif kind == "oracle_affine" and "name" not in decoded.get(n, {}):
                    status = (
                        "no_correspondence"
                        if n in decoded
                        else "calibration_unavailable"
                    )
                elif kind != "oracle_affine" and key not in curves[k]:
                    status = "physical_correspondence_unavailable"
                else:
                    status = "complete"
            metrics = {"status": status}
            if status == "complete":

                def read(values, case, *, kind=kind, n=n, key=key):
                    return (
                        probe.decode(decoded[n], values[case])
                        if kind == "oracle_affine"
                        else np.asarray(values[case][key])
                    )

                try:
                    a, b = read(curves, k), read(curves, base)
                    alt = read(cross, k) - read(cross, base)
                    amp = max(
                        1e-8,
                        np.max(np.abs(np.subtract(refs[k][n], refs[base][n]))),
                        np.max(abs(a - b)),
                    )
                    if np.max(np.abs(alt - (a - b))) / amp > 5e-3:
                        metrics = {"status": "effect_solver_disagreement"}
                    else:
                        metrics = scores(
                            refs[k]["time"],
                            refs[k][n],
                            refs[base][n],
                            a,
                            b,
                            item["window"],
                            scales[n] if kind == "target" else hidden_scale[n],
                        )
                except (ValueError, ArithmeticError) as exc:
                    metrics = {"status": "metric_unavailable", "reason": str(exc)}
            if "peak_time_error_min" in metrics:
                metrics["peak_time_error"] = metrics.pop("peak_time_error_min")
            result.append(
                {
                    "subject": i,
                    "time_unit": "normalized reactor time" if c == fits.CSTR else "min",
                    "source_id": row["source_id"],
                    "source_benchmark_id": row.get(
                        "source_benchmark_id", row["benchmark_id"]
                    ),
                    "source_bundle_sha256": row["source_bundle_sha256"],
                    "benchmark_id": row["benchmark_id"],
                    "method": row["method"],
                    "repetition": row["repetition"],
                    "case": k,
                    "control": base,
                    "window_start": item["window"][0],
                    "window_end": item["window"][1],
                    "quantity": n,
                    "unit": UNITS[n],
                    "readout": kind,
                    **metrics,
                }
            )
    return {
        "scores": result,
        "training_probes": decoded,
        "reference_units": UNITS,
        "failures": failures,
    }


def csv_table(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(dict.fromkeys(k for r in rows for k in r))
            if rows
            else ["status"],
        )
        w.writeheader()
        w.writerows(rows)


def run(root, data_root, workers):
    with public._lock(root):
        plan = sealed_read(root / "frozen_selection_manifest.json")
        if (
            plan["protocol"] != VERSION
            or plan["source"] != source_hash()
            or plan["runtime"] != public._runtime()
        ):
            raise ValueError("evaluator identity changed since freeze")
        if (
            plan["cstr_spec_sha256"]
            and plan["cstr_spec_sha256"]
            != hashlib.sha256((data_root / SPEC_PATHS["cstr"]).read_bytes()).hexdigest()
        ):
            raise ValueError("CSTR generator changed")
        refs = {}
        for cell, s in plan["schedules"].items():
            refs[cell] = {}
            for k, case in s.items():
                path = root / "references" / cell / (k + ".json")
                if not path.exists():
                    a, b = (
                        reference(cell, case, data_root, m) for m in ("DOP853", "Radau")
                    )
                    gap = probe.solver_agreement(a, b, 2e-6)
                    sealed_write(
                        path,
                        {
                            "plan": plan["artifact_sha256"],
                            "reference": a,
                            "solver_gap": gap,
                        },
                    )
                rec = sealed_read(path)
                if rec["plan"] != plan["artifact_sha256"]:
                    raise ValueError("stale reference checkpoint")
                refs[cell][k] = rec["reference"]
        args = [
            (
                root,
                i,
                r,
                plan["schedules"][r["benchmark_id"]],
                refs[r["benchmark_id"]],
                plan["artifact_sha256"],
                plan["training_sd"][r["benchmark_id"]],
            )
            for i, r in enumerate(plan["roster"])
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(evaluate_one, args))
        rows = [r for x in results for r in x["scores"]]
        inputs = sorted(
            {
                r["source_bundle_sha256"]
                for r in plan["roster"]
                if r["source_protocol"] == "iclr-controlled-search-models-1"
            }
        )
        score_record = sealed_write(
            root / "scored_rows.json",
            {
                "protocol": VERSION,
                "source_bundle_sha256": inputs[0] if len(inputs) == 1 else None,
                "source_bundle_hashes": sorted(
                    {r["source_bundle_sha256"] for r in plan["roster"]}
                ),
                "mode": plan["mode"],
                "metrics_sha256": public.content_sha256(rows),
                "plan_sha256": plan["artifact_sha256"],
                "rows": rows,
            },
        )
        summary = sealed_write(
            root / "summary.json",
            {
                "protocol": VERSION,
                "plan": plan["artifact_sha256"],
                "subjects": results,
                "plan_sha256": plan["artifact_sha256"],
                "scored_rows_sha256": score_record["artifact_sha256"],
                "metrics_sha256": score_record["metrics_sha256"],
                "row_count": len(rows),
                "rows": rows,
                "source_mode": plan["mode"],
                "external_baselines_evaluated": False,
            },
        )
        csv_table(root / "intervention_effects.csv", rows)
        csv_table(
            root / "hidden_process_metrics.csv",
            [r for r in rows if r["readout"] != "target"],
        )
        lookup = {
            (str(r["subject"]), r["case"], r["quantity"], r["readout"]): r for r in rows
        }
        pairs = []
        for p in plan["pairs"]:
            if not p["eligible"]:
                pairs.append(p)
                continue
            for (i, k, n, a), left in lookup.items():
                if i != p["left"]:
                    continue
                right = lookup[p["right"], k, n, a]
                pairs.append(
                    {
                        **p,
                        "case": k,
                        "quantity": n,
                        "readout": a,
                        "left_rse": left.get("effect_rse"),
                        "right_rse": right.get("effect_rse"),
                        "left_status": left["status"],
                        "right_status": right["status"],
                    }
                )
        csv_table(root / "matched_fit_pairs.csv", pairs)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--models", type=Path, action="append")
    parser.add_argument("--readouts", type=Path)
    parser.add_argument("--stress-public-root", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--public-root",
        type=Path,
        default=Path("artifacts/phase-c-development-final-v2/public"),
    )
    parser.add_argument("--legacy-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("data_raw"))
    parser.add_argument(
        "--mode", choices=("development", "confirmation"), default="development"
    )
    parser.add_argument("--workers", type=int, default=2)
    a = parser.parse_args()
    if a.stage == "prepare":
        if not a.models:
            parser.error("prepare requires --models")
        print(
            prepare(
                a.models,
                a.root,
                a.mode,
                a.public_root,
                a.legacy_root,
                a.data_root,
                a.readouts,
                a.stress_public_root,
            )["artifact_sha256"]
        )
    else:
        print(run(a.root, a.data_root, a.workers)["artifact_sha256"])
