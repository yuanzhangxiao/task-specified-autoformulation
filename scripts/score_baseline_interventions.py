"""Score external baselines' sealed picks on the intervention study's schedules.

The independent anchor assay (``scripts/iclr_anchor_fidelity.py``) scores frozen
models on held-out input interventions, each against its own matched control:
the response's direction, its peak time and its relative squared error. This
script puts the external baselines' sealed picks through that assay's own code:
its confirmation schedules, its reference responses (DOP853, checked against
Radau), its public input trajectories, its per-subject checkpoints and its
metrics (``evaluate_one``). Only the prediction differs: a baseline's model is
read with the external-baseline grammar it was selected under, and D3's model
is its native one-step map on the sample grid.

A baseline has no latent coordinate, so only its target rows can be complete;
its hidden-process rows are recorded unavailable, as for any such model.

Stages, in order:

* ``freeze``: every pick of the named sealed runs, failures included, with the
  schedules and training scales. Opens public data only. A frozen suite
  (``configs/*_probes_*.json``) may replace the study's schedules, and the
  study's own frozen models may join the scored subjects.
* ``rehearse``: the same predictor on the public validation trajectories, beside
  each run's recorded validation error.
* ``score``: reference responses and paired metrics. Optionally rescores named
  models of the study itself, to show the shared code reproduces its numbers.
* ``report``: one row per method, cell, case and seed, and a summary over seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from contextlib import contextmanager
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.baselines import latent_ode
from autoformalism.baselines.d3_rollout import predict as native_rollout
from autoformalism.benchmarks.phase_c_protocols import phase_c_protocols
from autoformalism.benchmarks.phase_c_release import SPEC_PATHS, load_public_cell
from autoformalism.expressions import (
    ValidationContext,
    baseline_validator,
    compile_candidate,
)
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import llm_sr_transcriptions as transcriptions
from autoformalism.rebuttal import phase_c_baseline_mechanisms as mechanisms
from autoformalism.rebuttal.mechanism_probes import _native
from autoformalism.rebuttal.phase_c_baselines import tier_of, verify_release
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import CandidateModel
from scripts import evaluate_t2_latent_dynamics as probe
from scripts import iclr_anchor_fidelity as anchor

PROTOCOL = "baseline-intervention-scores-1"
MODE = "confirmation"
#: The study's prediction settings (``anchor.predict``), kept for every model.
RTOL, ATOL, SECONDS = 1e-8, 1e-10, 45.0
#: The public fields that define a cell's context, as the adapters read them.
CONTEXT_FIELDS = ("targets", "auxiliaries", "external_inputs", "fixed_covariates")
#: The latent neural ODE (``baselines.latent_ode``), which runs outside the
#: vendored campaigns and is sealed by ``scripts/run_latent_ode_baseline.py``.
LATENT_ODE = "latent_ode"
METHODS = (*mechanisms.METHODS, LATENT_ODE)
#: The study's stress aliases, for rescoring its own models.
ALIASES = anchor.ALIASES


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluator_identity() -> dict:
    """The study code whose schedules, references and metrics are reused."""
    files = (Path(anchor.__file__), Path(probe.__file__), Path(__file__))
    return {path.name: _digest(path) for path in files}


def _run_cells(method: str, root: Path) -> set[str]:
    if method == LATENT_ODE:
        return {r["benchmark_id"] for r in sealed_read(root / "selection.json")["rows"]}
    if method in ("sindy", "pysr"):
        lines = (root / "frozen" / "task_plan.jsonl").read_text().splitlines()
        return {json.loads(line)["benchmark_id"] for line in lines if line.strip()}
    return {row["benchmark_id"] for row in sealed_read(root / "plan.json")["rows"]}


def public_cells(release: Path, names) -> dict:
    """Each cell's public context and training split, as the adapters read them."""
    cells = {}
    for name in sorted(names):
        spec, train, _ = load_public_cell(release / "public" / name)
        context = ValidationContext.model_validate({k: spec[k] for k in CONTEXT_FIELDS})
        cells[name] = {
            "context": context.model_dump(mode="json"),
            "training": train.model_dump(mode="json"),
        }
    return cells


def suite_schedules(cell: str, suite: dict) -> dict:
    """A frozen suite's schedules, built as the study builds its own.

    Calibration schedules are the named training protocols; the control and
    every probe copy the base training protocol with only their input
    specification changed, as ``anchor.cases`` does.
    """
    if suite.get("status") != "frozen_before_references" or suite["cell"] != cell:
        raise ValueError("the suite is not frozen for this cell")
    if cell != anchor.fits.CSTR:
        raise ValueError("only the reactor has a probe suite")
    by_id = {r.protocol_id: r for r in phase_c_protocols("cstr") if r.split == "train"}
    table = {
        name: {
            "protocol": by_id[name].model_dump(mode="json"),
            "role": "train_calibration",
        }
        for name in suite["calibration"]
    }
    base = by_id[suite["base_protocol"]].model_dump(mode="json")

    def protocol(name, specification):
        return base | {
            "protocol_id": "probe_" + name,
            "split": "test",
            "specification": specification,
        }

    control = suite["control"]
    table[control["name"]] = {
        "protocol": protocol(control["name"], control["specification"]),
        "role": "control",
        "control": None,
        "window": None,
    }
    for probe_case in suite["probes"]:
        name = probe_case["name"]
        if name in table:
            raise ValueError(f"probe {name} repeats a schedule name")
        table[name] = {
            "protocol": protocol(name, probe_case["specification"]),
            "role": "contrast",
            "control": control["name"],
            "window": list(probe_case["window"]),
            "group": probe_case["group"],
        }
    return table


def study_rows(path: Path, cell: str) -> list[dict]:
    """Every frozen model of the study on a cell, as its evaluator reads them."""
    bundle = sealed_read(path)
    rows = []
    for r in bundle["rows"]:
        if ALIASES.get(r["benchmark_id"], r["benchmark_id"]) != cell:
            continue
        rows.append(
            {
                **r,
                "benchmark_id": cell,
                "source_benchmark_id": r["benchmark_id"],
                "source_bundle_sha256": bundle["artifact_sha256"],
                "source_protocol": bundle["protocol"],
                "direct_readouts": {},
            }
        )
    if not rows:
        raise ValueError("the study bundle has no model on this cell")
    return rows


def subject(row: dict, bundle: str) -> dict:
    """One sealed pick, in the shape the study's ``evaluate_one`` reads."""
    return {
        "source_id": row["row_id"],
        "source_protocol": PROTOCOL,
        "source_bundle_sha256": bundle,
        "benchmark_id": row["benchmark_id"],
        "method": row["method"],
        "repetition": row["repetition"],
        "status": row["status"],
        "failure": row.get("failure"),
        "model": row.get("model"),
        "sources": row["sources"],
        "recorded": row.get("recorded"),
        "direct_readouts": {},
    }


def latent_ode_rows(root: Path) -> list[dict]:
    """The latent ODE's selected model per cell and seed, failures included."""
    plan = sealed_read(root / "plan.json")
    selection = sealed_read(root / "selection.json")
    if (
        plan["protocol"] != latent_ode.PROTOCOL
        or selection["plan_sha256"] != plan["artifact_sha256"]
        or selection.get("test_data_opened") is not False
    ):
        raise ValueError("the latent ODE selection belongs to another plan")
    sources = [
        mechanisms._file(root / "plan.json"),
        mechanisms._file(root / "selection.json"),
    ]
    rows = []
    for r in selection["rows"]:
        name = r["benchmark_id"]
        row = mechanisms._row(LATENT_ODE, name, tier_of(name), int(r["repetition"]))
        if r["status"] != "complete":
            rows.append(mechanisms._failure(row, sources, r["status"]))
            continue
        rows.append(
            {
                **mechanisms._ready(row, sources, r["model"]),
                "recorded": ["selection.validation_nmse", r["validation_nmse"]],
            }
        )
    return rows


def predict(row: dict, tr, solver: str) -> dict:
    """A baseline's free rollout of the study's public input trajectory."""
    model = row["model"]
    context = ValidationContext.model_validate(model["context"])
    if (
        set(context.targets) != set(tr.targets)
        or set(context.auxiliaries) != set(tr.auxiliaries)
        or set(context.external_inputs) != set(tr.external_inputs)
        or context.lagged_targets
        or context.fixed_covariates
    ):
        raise ValueError("model violates public input or initialization contract")
    if model["semantics"] == "latent_neural_ode":
        ode = latent_ode.LatentODE(model)
        supplied = {**tr.external_inputs, **tr.auxiliaries}
        inputs = np.stack(
            [np.asarray(supplied[n], dtype=float) for n in ode.inputs], -1
        )
        y, h = ode.simulate(
            np.asarray(tr.time, dtype=float),
            np.array([tr.targets[t][0] for t in ode.targets], dtype=float),
            inputs,
            method=solver,
            rtol=RTOL,
            atol=ATOL,
            seconds=SECONDS,
        )
        # Latent coordinates go to the study's train-calibrated probe, as ours do.
        return {t: y[:, i].tolist() for i, t in enumerate(ode.targets)} | {
            f"latent.h{j}": h[:, j].tolist() for j in range(h.shape[1])
        }
    if model["semantics"] == "native_increment":
        step = np.diff(np.asarray(tr.time, dtype=float))
        if np.max(np.abs(step - model["increment_step"])) > 1e-9 * max(
            1.0, model["increment_step"]
        ):
            raise ValueError("the schedule's grid is not the map's sample interval")
        native = _native(model)
        states = native_rollout(native, tr, teacher_forced=False, seconds=SECONDS)
        return {t: states[:, native.states.index(t)].tolist() for t in context.targets}
    if model["semantics"] != "continuous_time" or model.get("grammar") != "baseline":
        raise ValueError("unsupported baseline semantics or grammar")
    compiled = compile_candidate(
        CandidateModel.model_validate(model["candidate"]),
        context,
        validator=baseline_validator(),
    )
    sim = simulate_trajectory(
        compiled,
        tr,
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
    return {n: y.tolist() for n, y in sim.predictions.items()}


@contextmanager
def baseline_predictions():
    """Route the study's ``evaluate_one`` to ``predict`` for baseline subjects."""
    original = anchor.predict

    def route(row, tr, solver):
        if row.get("source_protocol") == PROTOCOL:
            return predict(row, tr, solver)
        return original(row, tr, solver)

    anchor.predict = route
    try:
        yield
    finally:
        anchor.predict = original


def freeze(
    root: Path,
    release: Path,
    sources: dict[str, Path],
    cells: list[str],
    llm_sr_transcriptions: Path | None = None,
    suite: Path | None = None,
    study_models: Path | None = None,
) -> dict:
    """Seal every pick of the named runs on the named cells, failures included."""
    if (root / "plan.json").exists():
        raise ValueError("this root is already frozen")
    unknown = set(sources) - set(METHODS)
    if unknown or not sources or not set(cells) <= set(anchor.fits.CELLS):
        raise ValueError("unknown methods or cells outside the study")
    frozen_suite = json.loads(suite.read_text()) if suite else None
    if frozen_suite and len(cells) != 1:
        raise ValueError("a probe suite is frozen for one cell")
    names = set().union(*(_run_cells(m, p) for m, p in sources.items()))
    receipt = verify_release(release, sorted(names))
    public_by_name = public_cells(release, names)
    reviewed = (
        transcriptions.load(llm_sr_transcriptions) if llm_sr_transcriptions else None
    )
    rows = []
    for method, path in sorted(sources.items()):
        if method == LATENT_ODE:
            rows += latent_ode_rows(path)
        elif method in ("sindy", "pysr"):
            found = mechanisms.classical_rows(path, public_by_name)
            rows += [r for r in found if r["method"] == method]
        else:
            rows += mechanisms.campaign_rows(method, path, public_by_name, reviewed)
    rows = [r for r in rows if r["benchmark_id"] in cells]
    bundle = public.content_sha256(rows)
    subjects = [subject(r, bundle) for r in rows]
    for s in subjects:
        if s["status"] == "ready" and s["model"]["semantics"] == "latent_neural_ode":
            latent_ode.LatentODE(s["model"])
        elif s["status"] == "ready" and s["model"]["semantics"] == "native_increment":
            _native(s["model"])
        elif s["status"] == "ready":
            compile_candidate(
                CandidateModel.model_validate(s["model"]["candidate"]),
                ValidationContext.model_validate(s["model"]["context"]),
                validator=baseline_validator(),
            )
    root.mkdir(parents=True, exist_ok=True)
    return sealed_write(
        root / "plan.json",
        {
            "protocol": PROTOCOL,
            "mode": MODE,
            "evaluator": evaluator_identity(),
            "release_receipt_sha256": receipt,
            "sources": {
                m: {"path": str(p.resolve()), "cells": sorted(_run_cells(m, p))}
                for m, p in sorted(sources.items())
            },
            "llm_sr_transcriptions": (
                {
                    "path": str(llm_sr_transcriptions.resolve()),
                    "status": reviewed.status,
                }
                if reviewed
                else None
            ),
            "cells": sorted(cells),
            "suite": (
                {
                    "path": str(suite.resolve()),
                    "sha256": _digest(suite),
                    "name": frozen_suite["suite"],
                }
                if frozen_suite
                else None
            ),
            "schedules": {
                c: suite_schedules(c, frozen_suite)
                if frozen_suite
                else anchor.cases(c, MODE)
                for c in sorted(cells)
            },
            "study_rows": (
                [r for c in sorted(cells) for r in study_rows(study_models, c)]
                if study_models
                else []
            ),
            "study_models": (
                {"path": str(study_models.resolve()), "sha256": _digest(study_models)}
                if study_models
                else None
            ),
            "training_sd": {
                c: anchor.training_sd(c, release / "public", Path(".")) for c in cells
            },
            "rows": subjects,
            "test_data_opened": False,
            "private_reference_opened": False,
        },
    )


def _recorded_validation(row: dict):
    """The free-rollout validation error a run sealed for its pick.

    SINDy and PySR record ``validation_normalized_mse``; LLM-ODE and LLM-SR
    ``development_rollout_error``; D3 its recursive ``phase_b_rollout`` beside
    the one-step loss it selects on; the latent ODE the error it was selected on.
    """
    if row.get("recorded"):
        key, value = row["recorded"]
        return key, float(value)
    for source in row["sources"]:
        path = Path(source["path"])
        if path.name != "result.json":
            continue
        value = json.loads(path.read_text())
        rollout = value.get("evaluation", {}).get("phase_b_rollout", {})
        for key, found in (
            ("validation_normalized_mse", value.get("validation_normalized_mse")),
            ("development_rollout_error", value.get("development_rollout_error")),
            ("evaluation.phase_b_rollout", rollout.get("normalized_mse")),
        ):
            if isinstance(found, int | float):
                return key, float(found)
    return None, None


def rehearse(root: Path, release: Path) -> dict:
    """Free rollouts of the public validation trajectories with ``predict``."""
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] != PROTOCOL:
        raise ValueError("not a baseline intervention plan")
    results = []
    for row in plan["rows"]:
        if row["status"] != "ready":
            continue
        cell = release / "public" / row["benchmark_id"]
        _, train, validation = load_public_cell(cell)
        train_rows = public.unpack_split(train).trajectories
        targets = row["model"]["context"]["targets"]
        scale = {
            t: float(np.var(np.concatenate([tr.targets[t] for tr in train_rows])))
            for t in targets
        }
        errors = {t: [] for t in targets}
        status = "complete"
        for tr in public.unpack_split(validation).trajectories:
            try:
                curves = predict(row, tr, "Radau")
            except (ValueError, ArithmeticError, TimeoutError) as exc:
                status = f"failed: {exc}"
                break
            for t in targets:
                errors[t].append((np.asarray(curves[t]) - tr.targets[t]) ** 2)
        per_target = [np.mean(np.concatenate(errors[t])) / scale[t] for t in targets]
        nmse = float(np.mean(per_target)) if status == "complete" else float("inf")
        key, recorded = _recorded_validation(row)
        results.append(
            {
                "source_id": row["source_id"],
                "status": status,
                "validation_nmse_training_variance": nmse,
                "recorded_key": key,
                "recorded": recorded,
                "relative_difference": abs(nmse - recorded) / abs(recorded)
                if recorded and np.isfinite(nmse)
                else None,
            }
        )
    return sealed_write(
        root / "rehearsal.json",
        {"plan_sha256": plan["artifact_sha256"], "rows": results},
    )


def _parity_rows(paths: list[Path], keep: set[str]) -> list[dict]:
    """The study's own frozen models, as its evaluator reads them."""
    rows = []
    for path in paths:
        bundle = sealed_read(path)
        for r in bundle["rows"]:
            if r["source_id"] not in keep:
                continue
            rows.append(
                {
                    **r,
                    "benchmark_id": ALIASES.get(r["benchmark_id"], r["benchmark_id"]),
                    "source_benchmark_id": r["benchmark_id"],
                    "source_bundle_sha256": bundle["artifact_sha256"],
                    "source_protocol": bundle["protocol"],
                    "direct_readouts": {},
                    "parity_check": True,
                }
            )
    if {r["source_id"] for r in rows} != keep:
        raise ValueError("a parity model is missing from the named bundles")
    return rows


def score(
    root: Path,
    data_root: Path,
    parity_models: list[Path] | None = None,
    parity_ids: set[str] | None = None,
) -> dict:
    """Reference responses, then the study's paired metrics for every subject."""
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] != PROTOCOL or plan["evaluator"] != evaluator_identity():
        raise ValueError("the plan or the study's evaluator changed since freezing")
    refs = {}
    for cell, schedule in plan["schedules"].items():
        refs[cell] = {}
        for case, item in schedule.items():
            path = root / "references" / cell / f"{case}.json"
            if not path.exists():
                a, b = (
                    anchor.reference(cell, item, data_root, method)
                    for method in ("DOP853", "Radau")
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
            record = sealed_read(path)
            if record["plan"] != plan["artifact_sha256"]:
                raise ValueError("stale reference checkpoint")
            refs[cell][case] = record["reference"]
    # The study's own models are scored under a root of their own, so a subject
    # index always names the same model as its checkpoints.
    groups = [(root, plan["rows"], False)]
    if plan.get("study_rows"):
        groups.append((root / "study", plan["study_rows"], False))
    if parity_models:
        ids = sorted(parity_ids or ())
        folder = root / "parity" / public.content_sha256(ids)[:16]
        groups.append((folder, _parity_rows(parity_models, set(ids)), True))
    spec = data_root / SPEC_PATHS["cstr"]
    written = []
    with baseline_predictions():
        for folder, subjects, parity in groups:
            scored, failures = [], {}
            for i, row in enumerate(subjects):
                cell = row["benchmark_id"]
                result = anchor.evaluate_one(
                    (
                        folder,
                        i,
                        row,
                        plan["schedules"][cell],
                        refs[cell],
                        plan["artifact_sha256"],
                        plan["training_sd"][cell],
                    )
                )
                scored += [{**r, "parity_check": parity} for r in result["scores"]]
                if result["failures"]:
                    failures[row["source_id"]] = result["failures"]
            written.append(
                sealed_write(
                    folder / "scores.json",
                    {
                        "protocol": PROTOCOL,
                        "plan_sha256": plan["artifact_sha256"],
                        "parity_check": parity,
                        "rows": scored,
                        "failures": failures,
                        "cstr_spec_sha256": _digest(spec)
                        if anchor.fits.CSTR in plan["cells"]
                        else None,
                        "test_data_opened": False,
                        "private_reference_opened": True,
                    },
                )
            )
    return written[0] if len(written) == 1 else {"scores": written}


def report(root: Path) -> dict:
    """Target rows per seed, and per method, cell and case over seeds."""
    files = [
        root / "scores.json",
        root / "study" / "scores.json",
        *sorted((root / "parity").glob("*/scores.json")),
    ]
    rows = [
        r
        for path in files
        if path.exists()
        for r in sealed_read(path)["rows"]
        if r["readout"] == "target"
    ]
    keys = (
        "method",
        "repetition",
        "benchmark_id",
        "case",
        "control",
        "quantity",
        "status",
        "direction_correct",
        "window_direction_correct",
        "peak_time_error",
        "effect_rse",
        "absolute_nmse",
        "reference_signed_peak",
        "predicted_signed_peak",
        "parity_check",
    )
    with (root / "target_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: [str(r[k]) for k in keys[:6]]))
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = tuple(
            r[k] for k in ("benchmark_id", "method", "case", "quantity", "parity_check")
        )
        groups.setdefault(key, []).append(r)
    summary = []
    for (cell, method, case, quantity, parity), items in sorted(groups.items()):
        done = [r for r in items if r["status"] == "complete"]
        summary.append(
            {
                "benchmark_id": cell,
                "method": method,
                "parity_check": parity,
                "case": case,
                "quantity": quantity,
                "seeds": len(items),
                "complete": len(done),
                "direction_correct": sum(bool(r["direction_correct"]) for r in done),
                "window_direction_correct": sum(
                    bool(r["window_direction_correct"]) for r in done
                ),
                "median_effect_rse": statistics.median(r["effect_rse"] for r in done)
                if done
                else None,
                "median_abs_peak_time_error": statistics.median(
                    abs(r["peak_time_error"])
                    for r in done
                    if r["peak_time_error"] is not None
                )
                if any(r["peak_time_error"] is not None for r in done)
                else None,
                "statuses": sorted({r["status"] for r in items}),
            }
        )
    return sealed_write(root / "report.json", {"rows": summary})


def _sources(values: list[str]) -> dict[str, Path]:
    pairs = [value.split("=", 1) for value in values]
    if any(len(p) != 2 for p in pairs) or len({p[0] for p in pairs}) != len(pairs):
        raise ValueError("give each source once, as METHOD=RUN_ROOT")
    return {method: Path(path) for method, path in pairs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "rehearse", "score", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--release", type=Path, default=Path("artifacts/phase-c-development-final-v2")
    )
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--cell", action="append", default=[])
    parser.add_argument("--llm-sr-transcriptions", type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--study-models", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data_raw"))
    parser.add_argument("--parity-models", type=Path, action="append")
    parser.add_argument("--parity-id", action="append", default=[])
    a = parser.parse_args()
    if a.stage == "freeze":
        out = freeze(
            a.root,
            a.release,
            _sources(a.source),
            a.cell,
            a.llm_sr_transcriptions,
            a.suite,
            a.study_models,
        )
        print(out["artifact_sha256"], len(out["rows"]), "subjects")
    elif a.stage == "rehearse":
        print(json.dumps(rehearse(a.root, a.release)["rows"], indent=1))
    elif a.stage == "score":
        out = score(a.root, a.data_root, a.parity_models, set(a.parity_id))
        for part in out.get("scores", [out]):
            print(part["artifact_sha256"], len(part["rows"]), "rows")
    else:
        print(json.dumps(report(a.root)["rows"], indent=1))
