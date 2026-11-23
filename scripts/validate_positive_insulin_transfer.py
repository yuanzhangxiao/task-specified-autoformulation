"""Freeze positive insulin models and replay development validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.preproposal_cases import T2
from autoformalism.research.trajectory_function_fitting import _Rollout, _rows
from autoformalism.research.trajectory_function_revision import (
    _check_graph,
    assess_fixed,
)
from autoformalism.staged_topology import content_hash
from scripts import compare_insulin_order_repairs as scalar
from scripts import trajectory_functions as campaign
from scripts import transfer_positive_insulin_law as portable
from scripts.construction_functions import file_hash

RANK = "0b313a5995875387b2b72268c1cdd8781d07ea6cc5a89c0586661a035cd958dd"
IDS = tuple(f"val_{n:03}" for n in range(4))
MODELS = (
    ("002-restoring", 2, 2),
    ("002-positive_EGP", 2, 2),
    ("015-restoring", 15, 5),
    ("015-positive_EGP", 15, 5),
)
TARGETS = portable.TARGETS


def paths(evidence):
    return evidence / "aces-equilibrium/results", evidence / "positive-rates-final"


def prepare(evidence, root):
    """Authenticate full training fits. This stage accepts no validation data path."""
    cold, pos = paths(evidence)
    raw = scalar.source(evidence / "paired")
    intake = sealed_read(evidence / "paired/generation/intake.json")
    rank = sealed_read(cold / "ranking.json")
    coldplan = sealed_read(cold / "plan.json")
    pp = sealed_read(pos / "plan.json")
    summary = sealed_read(pos / "summary.json")
    saved = sealed_read(pos / "saved-replay.json")
    full = sealed_read(pos / "full-EGP.json")
    train = [raw[n] for n in sorted(raw)]
    selected = full["arms"]["nonnegative"]
    if (
        rank["artifact_sha256"] != RANK
        or rank["plan"] != coldplan["artifact_sha256"]
        or rank["validation_opened"] is not False
        or coldplan["training_intake"] != campaign.TRAIN
        or coldplan["validation_intake"] != campaign.VALIDATION
        or summary["artifact_sha256"] != portable.SCALAR
        or pp["artifact_sha256"] != summary["plan"]
        or pp["protocol"] != "positive-insulin-production-2"
        or pp["train"] != scalar.TRAIN
        or full["artifact_sha256"] != summary["checkpoints"]["full"]["EGP"]
        or full["plan"] != pp["artifact_sha256"]
        or (full["split"], full["source"]) != ("full", "EGP")
        or saved["artifact_sha256"] != portable.SCALAR_REPLAY
        or saved["summary"] != portable.SCALAR
        or [r["event"] for r in selected["scores"]] != sorted(raw)
        or not all(r["verified"] for r in selected["scores"])
        or not any(
            x["split"] == "full"
            and x["weights"] == "nonnegative"
            and x["sources"] == "EGP"
            and x["scores_sha256"] == content_hash(selected["scores"])
            for x in saved["verified"]
        )
        or portable.production.calibrate(train, portable.production.CHANNEL_ORDER)
        != full["knots"]
    ):
        raise ValueError(
            "Full training coefficients, basis, cohort or provenance differs"
        )
    models = {}
    for tag, slot, index in MODELS:
        m = coldplan["models"][index]
        r = sealed_read(cold / "fits" / f"{index:02}" / "result.json")
        if (
            r["artifact_sha256"] != rank["fit_receipts"][index]
            or r["plan"] != rank["plan"]
            or r["index"] != index
            or r["selected_option"] is None
            or (m["source_slot"], m["arm"]) != (slot, "restoring")
        ):
            raise ValueError("Wrong full-training restoring parent")
        opt = r["options"][r["selected_option"]]
        if (
            opt["status"] != "fitted"
            or opt["graph_sha256"] != content_hash(m["graph"])
            or opt["formula_sha256"] != content_hash(m["formula"])
            or set(opt["parameters"]) != set(m["formula"]["parameters"])
        ):
            raise ValueError("Unqualified or substituted full-training parent vector")
        parent = {
            "graph": m["graph"],
            "formula": m["formula"] | {"parameters": opt["parameters"]},
        }
        model = (
            parent
            if tag.endswith("restoring")
            else portable.export(parent, full["knots"], selected["selected"])
        )
        case = portable.transfer.fold_case(intake, slot, train)
        gates = _check_graph(
            model["graph"], case, "Verify the complete exported model on training data."
        )
        form = assess_fixed(model["graph"], case, model["formula"])
        if gates["status"] != "topology_screen_passed" or form["status"] != "eligible":
            raise ValueError(
                f"Training gate failed for {tag}: {gates['status']}, {form['status']}"
            )
        models[tag] = {
            **model,
            "scales": opt["scales"],
            "parent_slot": slot,
            "parent_fit_sha256": r["artifact_sha256"],
            "source_index": index,
            "training_topology": gates["status"],
            "training_formula": form["status"],
            "balance": form["balance_check"],
            "expected_parent_score": f"val/{index:02}.json",
            "expected_parent_score_identity": {
                "ranking": RANK,
                "index": index,
                "split": "val",
                "arm": "restoring",
                "source_slot": slot,
            },
        }
    return sealed_write(
        root / "plan.json",
        {
            "protocol": "t2-positive-transfer-development-validation-1",
            "models": models,
            "script": file_hash(Path(__file__)),
            "export_code": file_hash(Path(portable.__file__)),
            "selected_positive_training_fit": full["artifact_sha256"],
            "parent_ranking": RANK,
            "training_sha256": scalar.TRAIN,
            "training_intake": campaign.TRAIN,
            "validation_intake": campaign.VALIDATION,
            "dataset": T2,
            "expected_validation_ids": list(IDS),
            "validation_opened_at_freeze": False,
            "test_opened": False,
            "fit_calls": 0,
            "llm_calls": 0,
            "policy": (
                "Score all four fixed models on the four reused development- "
                "validation events. Use only supplied forcing histories, public "
                "initial readings and fitted initializers. Reproduce restoring "
                "forecasts at four original-grid RK4 substeps with the original "
                "two/four scaled gate of 0.01. Compare to the archived "
                "development-validation curves and scores at atol=rtol=1e-8. "
                "Positive models use basis-knot-crossing RK4 grids of 2,4,8,16,32 "
                "substeps; accept the first two levels with successive scaled "
                "change <=1e-5 and error <=1e-5 +1e-7*abs(independent I), and min "
                "I >= -1e-10. Independent scalar DOP853 integrates each piecewise "
                "linear production interval using the raw supplied histories and "
                "fixed weights only. Return unresolved numerical failures. No "
                "subsequent model changes or selection."
            ),
        },
    )


@dataclass
class Inputs:
    """Validation forcing histories and observed initials available to the forecast."""

    name: str
    time: np.ndarray
    supplied: dict
    initial: dict


def forecast_inputs(row):
    if row.trajectory_id not in IDS or not (set(row.targets) == set(TARGETS)):
        raise ValueError("Wrong validation event ID or public targets")
    t = np.asarray(row.time)
    supplied = {
        **row.inputs,
        **row.auxiliaries,
        **{k: np.full(len(t), v) for k, v in row.covariates.items()},
    }
    initial = {k: v[0] for k, v in (supplied | row.targets).items()}
    if set(supplied) & set(TARGETS) or not np.all(np.diff(t) > 0):
        raise ValueError("Incorrect validation roles or time knots")
    return Inputs(row.trajectory_id, t, supplied, initial)


def mesh(event, knots):
    """Find input-hat crossings, preserving every observed time knot."""
    t = event.time
    times = [t]
    for name, values in knots.items():
        x = np.asarray(event.supplied[name])
        left, right = x[:-1], x[1:]
        for v in values:
            sel = ((left < v) & (v < right)) | ((right < v) & (v < left))
            times.append(
                t[:-1][sel]
                + np.diff(t)[sel] * (v - left[sel]) / (right[sel] - left[sel])
            )
    return np.unique(np.concatenate(times))


def scalar_solve(event, knots, selected):
    """Solve the scalar ODE independently on each affine-forcing interval."""
    times = mesh(event, knots)
    w = portable.production.checked_weights(knots, selected)
    q = np.zeros(len(times))
    start = 0
    for name in portable.production.channel_order(knots):
        k = knots[name]
        x = np.interp(times, event.time, event.supplied[name])
        q += np.interp(x, k, w[start : start + len(k)])
        start += len(k)
    y = [event.initial["I"]]
    for dt, lo, hi in zip(np.diff(times), q[:-1], q[1:], strict=True):

        def rhs(t, state, dt=dt, lo=lo, hi=hi):
            return lo + (hi - lo) * (t / dt) - state / selected["tau"]

        sol = solve_ivp(
            rhs, (0, float(dt)), [y[-1]], method="DOP853", rtol=1e-11, atol=1e-11
        )
        if not sol.success or not np.isfinite(sol.y[0, -1]):
            raise ArithmeticError("Independent scalar integration failed")
        y.append(sol.y[0, -1])
    return np.asarray(y)[np.searchsorted(times, event.time)]


def predict(model, event, times, substeps):
    engine = _Rollout(model["graph"], model["formula"], TARGETS, substeps)
    pars = [model["formula"]["parameters"][n] for n in engine.parameters]
    u = np.asarray(
        [np.interp(times, event.time, event.supplied[n]) for n in engine.forcings]
    )
    z = [event.initial[n] for n in engine.readings]
    y = engine.run(
        [(engine._flow(len(times)), (z, u, np.diff(times).reshape(1, -1)))], pars
    )[0][0]
    return y[:, np.searchsorted(times, event.time)]


def forecast(model, event):
    positive = "scalar_model" in model
    times = mesh(event, model.get("knots", {}))
    reference = (
        scalar_solve(event, model["knots"], model["scalar_model"]) if positive else None
    )
    log = []
    previous = None
    for st in portable.LEVELS if positive else (2, 4):
        p = predict(model, event, times, st)
        grid = (
            float(
                np.max(
                    abs(previous - p)
                    / np.array([model["scales"][n] for n in TARGETS])[:, None]
                )
            )
            if previous is not None
            else None
        )
        diff = None if reference is None else float(max(abs(reference - p[1])))
        log.append(
            {
                "substeps": st,
                "max_scaled_grid_difference": grid,
                "max_scalar_difference": diff,
            }
        )
        physical = not positive or (
            np.allclose(p[1], reference, atol=portable.ATOL, rtol=portable.RTOL)
            and min(p[1]) >= -1e-10
        )
        if (
            grid is not None
            and grid <= (portable.SCALED_GRID if positive else 0.01)
            and physical
        ):
            return p, log
        previous = p
    return None, log


def read_validation(path, plan):
    if path.name != "validation.json":
        raise ValueError("Expected the registered validation intake")
    source = sealed_read(path)
    if (
        source["artifact_sha256"] != plan["validation_intake"]
        or source["training_intake_sha256"] != plan["training_intake"]
    ):
        raise ValueError("Validation file differs from the frozen intake")
    rows = _rows(source["rows"][T2], "val")
    if sorted(r.trajectory_id for r in rows) != list(IDS):
        raise ValueError("Development-validation IDs differ")
    return sorted(rows, key=lambda r: r.trajectory_id)


def original_record(cold, model):
    recap = sealed_read(cold / "val-summary.json")
    old = sealed_read(cold / model["expected_parent_score"])
    link = next(
        r
        for r in recap["rows"]
        if (r["slot"], r["arm"]) == (model["parent_slot"], "restoring")
    )
    if (
        recap["ranking"] != RANK
        or recap["split"] != "val"
        or link["receipt"] != old["artifact_sha256"]
        or any(old[k] != v for k, v in model["expected_parent_score_identity"].items())
        or old["score"]["status"] != "scored"
    ):
        raise ValueError("Archived restoring comparison is unavailable")
    return old


def summary(records):
    good = [r for r in records if r["status"] == "scored"]
    totals = portable.transfer.coupled.aggregate(records, len(IDS))
    for key, fn in (("negative_samples", sum), ("minima", min), ("max_abs_error", max)):
        totals[key] = {k: fn(r[key][k] for r in good) for k in TARGETS} if good else {}
    return totals


def validate(root, cold, path):
    plan = sealed_read(root / "plan.json")
    if (
        plan["script"] != file_hash(Path(__file__))
        or plan["export_code"] != file_hash(Path(portable.__file__))
        or plan["expected_validation_ids"] != list(IDS)
        or plan["validation_opened_at_freeze"]
    ):
        raise ValueError("Frozen evaluation contract differs")
    final = root / "summary.json"
    if final.exists():
        out = sealed_read(final)
        if out["plan"] != plan["artifact_sha256"]:
            raise ValueError("Existing validation summary differs from plan")
        for tag, sha in out["records"].items():
            if sealed_read(root / (tag + ".json"))["artifact_sha256"] != sha:
                raise ValueError("Saved forecasts differ")
        return out
    rows = read_validation(path, plan)
    reports = {}
    aggregates = {}
    for tag, _, _ in MODELS:
        model = plan["models"][tag]
        save = root / (tag + ".json")
        if save.exists():
            rec = sealed_read(save)
            if rec["plan"] != plan["artifact_sha256"] or rec["model"] != content_hash(
                model
            ):
                raise ValueError("Forecast checkpoint differs from frozen model")
        else:
            old = original_record(cold, model) if tag.endswith("restoring") else None
            curves = (
                {r["trajectory_id"]: r for r in old["score"]["curves"]} if old else {}
            )
            events = []
            for row in rows:
                name = row.trajectory_id
                try:
                    p, log = forecast(model, forecast_inputs(row))
                except ArithmeticError as exc:
                    events.append(
                        {
                            "event": name,
                            "status": "integration_failed",
                            "reason": str(exc),
                        }
                    )
                    continue
                if p is None:
                    events.append(
                        {"event": name, "status": "grid_unresolved", "grid": log}
                    )
                    continue
                raw = row.model_dump(mode="json")
                e = portable._metrics(raw, p, model["scales"])
                if old:
                    arch = curves[name]
                    y = np.asarray([arch["predicted"][n] for n in TARGETS])
                    e["archived_forecast_difference"] = float(np.max(abs(y - p)))
                    if (
                        arch["observed"] != raw["targets"]
                        or arch["time"] != raw["time"]
                        or not np.allclose(y, p, atol=1e-8, rtol=1e-8)
                    ):
                        e["status"] = "archive_difference"
                events.append(
                    e
                    | {
                        "time": raw["time"],
                        "observed": raw["targets"],
                        "predicted": {
                            k: v.tolist() for k, v in zip(TARGETS, p, strict=True)
                        },
                        "grid": log,
                    }
                )
            agg = summary(events)
            if (
                old
                and agg["pooled_nmse"] is not None
                and any(
                    not np.isclose(a, b, atol=1e-8, rtol=1e-8)
                    for a, b in [
                        (agg["pooled_nmse"], old["score"]["pooled_nmse"]),
                        *[
                            (
                                agg["available_per_target_nmse"][n],
                                old["score"]["per_target_nmse"][n],
                            )
                            for n in TARGETS
                        ],
                    ]
                )
            ):
                for ev in events:
                    ev["status"] = "archive_scores_differ"
                agg = summary(events)
            rec = sealed_write(
                save,
                {
                    "plan": plan["artifact_sha256"],
                    "model": content_hash(model),
                    "events": events,
                    "summary": agg,
                    "archived_parent_receipt": old["artifact_sha256"] if old else None,
                },
            )
        reports[tag] = rec
        aggregates[tag] = rec["summary"]
    comparisons = {}
    for slot in (2, 15):
        parent = reports[f"{slot:03}-restoring"]["events"]
        child = reports[f"{slot:03}-positive_EGP"]["events"]
        comparisons[str(slot)] = portable.transfer.paired_results(parent, child)
    return sealed_write(
        final,
        {
            "plan": plan["artifact_sha256"],
            "records": {k: v["artifact_sha256"] for k, v in reports.items()},
            "groups": aggregates,
            "paired": comparisons,
            "optimizer_calls": 0,
            "llm_calls": 0,
            "validation_events_reused": list(IDS),
            "test_opened": False,
        },
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cmd = cli.add_subparsers(dest="action", required=True)
    prep = cmd.add_parser("prepare")
    prep.add_argument("--evidence", type=Path, required=True)
    val = cmd.add_parser("validate")
    val.add_argument("--archive", type=Path, required=True)
    val.add_argument("--validation", type=Path, required=True)
    for c in (prep, val):
        c.add_argument("--root", type=Path, required=True)
    a = cli.parse_args()
    with _lock(a.root):
        if a.action == "prepare":
            plan = prepare(a.evidence, a.root)
            result = {"plan": plan["artifact_sha256"], "models": list(plan["models"])}
        else:
            result = validate(a.root, a.archive, a.validation)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
