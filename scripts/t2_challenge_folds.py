"""Refit two fixed T2 insulin families in three whole-training-event holdouts."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import casadi as ca
import numpy as np

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.casadi_initializer import _translate
from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.preproposal_cases import T2
from autoformalism.research.trajectory_function_families import _coefficients
from autoformalism.research.trajectory_function_fitting import (
    _check_grid,
    _rows,
    validate_formula,
)
from autoformalism.staged_topology import content_hash
from scripts import t2_equilibrium_repair as family
from scripts import trajectory_boundary_trial as original
from scripts import trajectory_functions as campaign_api
from scripts.construction_functions import file_hash

ANCHOR = "train_000"
ARMS = ("bounded", "restoring")
SLOTS = (2, 15)
GRAPHS = {
    2: "83634b70e6f844143a4ff30de803c4c58f1ddac331fc6236a7752a4dab405f19",
    15: "707cbf085bb61925c52c327c77723532445a911c35831e808d2e2f50daf7da70",
}


def folds(rows):
    """Hold out three consecutive ID blocks. Reserve the prespecified reference ID."""
    ids = sorted(r["trajectory_id"] for r in rows)
    if len(ids) != len(set(ids)) or any(not n.startswith("train_") for n in ids):
        raise ValueError("Require unique training event IDs")
    challenge = [n for n in ids if n != ANCHOR]
    if len(challenge) < 3:
        raise ValueError("Need three nonempty challenge blocks")
    return [
        {"held": list(block), "fit": [n for n in ids if n not in block]}
        for block in np.array_split(np.asarray(challenge, dtype=object), 3)
    ]


def fasting_reference(rows):
    """Obtain the one declared fasting reference from the fitting events only."""
    anchors = [r for r in rows if r["trajectory_id"] == ANCHOR]
    if len(anchors) != 1:
        return None
    r = anchors[0]
    insulin, gt = np.array(r["targets"]["I"]), np.array(r["auxiliaries"]["Gt"])
    if not (
        insulin[0] > 0
        and np.all(insulin == insulin[0])
        and np.all(gt == gt[0])
        and all(
            np.all(np.asarray(r["inputs"][n]) == 0) for n in (family.DRUG, family.MEAL)
        )
    ):
        return None
    return {
        "event": ANCHOR,
        "I": float(insulin[0]),
        "Gt": float(gt[0]),
        family.DRUG: 0.0,
    }


def cold_parameters(graph, formula, rows):
    """Initialize a fixed affine-in-coefficients formula on the fitting fold."""
    # Only parameter names are used from the archived formula. Its numerical values
    # include estimates from other folds and cannot be optimization starts here.
    params = dict.fromkeys(formula["parameters"], 0.0)
    parser = RestrictedParser()
    for node, text in formula["equations"].items():
        ex = parser.parse(text, location=node)
        names = sorted(
            ex.symbols & params.keys(), key=lambda n: int(n.removeprefix("afcoef"))
        )
        drivers = sorted(ex.symbols - params.keys())
        differential = graph["variable_types"][node] == "differential"
        guess = 1 / 300 if differential else 0.1
        if node not in rows[0]["targets"]:
            params.update(dict.fromkeys(names[1:], guess))
            continue
        x, p = ca.SX.sym("drivers", len(drivers)), ca.SX.sym("coeff", len(names))
        env = dict(zip(drivers, ca.vertsplit(x), strict=True))
        env.update(zip(names, ca.vertsplit(p), strict=True))
        field = _translate(ca, ex, env)
        # CasADi rejects an unbound coefficient if the feature matrix depends on it.
        features = ca.Function("known_columns", [x], [ca.jacobian(field, p).T])
        matrices, outcome = [], []
        for row in rows:
            known = row["inputs"] | row["auxiliaries"] | row["targets"]
            t = np.asarray(row["time"])
            xx = np.asarray([known.get(d, np.zeros(len(t))) for d in drivers])
            phi = np.asarray(features(xx)).T
            y = np.asarray(row["targets"][node])
            if differential:
                phi = (phi[:-1] + phi[1:]) / 2
                y = np.diff(y) / np.diff(t)
            matrices.append(phi)
            outcome.append(y)
        matrix = np.vstack(matrices)
        coeffs = _coefficients(matrix, np.concatenate(outcome))
        # Give an initially invisible latent contribution a small nonzero coefficient
        # so later trajectory fitting can update both sides of a latent pathway.
        coeffs[np.all(matrix == 0, axis=0)] = guess
        params.update(zip(names, coeffs.tolist(), strict=True))
    return params


def _sources(campaign):
    plan, intake, generation, _ = original._sources(campaign)
    models = {}
    for slot in SLOTS:
        g, arm, _ = campaign_api.slots(intake)[slot]
        row = intake["graphs"][g]
        candidate = campaign_api._read_candidate(campaign, plan, slot)
        if (
            row["dataset"] != T2
            or arm != "algorithm"
            or candidate["artifact_sha256"] != generation["candidates"][slot]
            or candidate["graph"] != row["structure_sha256"]
        ):
            raise ValueError("Unexpected source candidate")
        graph = row["graph"]
        updated = graph["sources"] | {"I": sorted(set(graph["sources"]["I"]) | {"I"})}
        bounded_graph = graph | {"sources": updated}
        if content_hash(bounded_graph) != GRAPHS[slot]:
            raise ValueError("Bounded graph differs from the fixed reference family")
        models[slot] = (
            bounded_graph,
            candidate["formula"],
            candidate["artifact_sha256"],
        )
    return intake["train"][T2], models


def prepare(campaign: Path, root: Path):
    if root.resolve().is_relative_to(campaign.resolve()):
        raise ValueError("Use a result directory outside the original study")
    rows, sources = _sources(campaign)
    splits = folds(rows)
    design = {
        "protocol": "t2-training-challenge-folds-1",
        "source_plan": original.ARCHIVED_PLAN,
        "source_generation": original.ARCHIVED_GENERATION,
        "source_candidates": {str(k): v[2] for k, v in sources.items()},
        "code": {
            "trial": file_hash(Path(__file__)),
            "family": file_hash(Path(family.__file__)),
            "source": original._code(),
        },
        "budget": original.BUDGET,
        "folds": splits,
        "arms": list(ARMS),
        "scope": (
            "Retrospective transfer of fixed graphs, normalized function templates "
            "and a hypothesized restoring edit selected using the original full "
            "training cohort. All adjustable coefficients, taper scales, fasting "
            "references and NMSE scales are rebuilt on each fitting subset. "
            "No complete-pipeline or independent benchmark claim."
        ),
        "scoring": (
            "Each challenge event is predicted once per arm using its observed "
            "initial values, allowed forcing and fold-fitted coefficients. "
            "Pool squared, fold-normalized errors over observed samples, then "
            "weight all three targets equally. Count failures before aggregation."
        ),
        "optimizer_initialization": (
            "Use only equation terms and coefficient names from "
            "source templates. Fit observed rates/values on fold-local measured "
            "arguments using weak ridge; set unavailable latent drivers to zero "
            "for this initialization only, give invisible coefficients small "
            "nonzero starts, latent initializers zero and recovery_rho=sqrt(1/60). "
            "Each arm uses the same fixed two-start optimizer and seed."
        ),
        "validation_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }
    by_id = {r["trajectory_id"]: r for r in rows}
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as lock:
        # Array tasks can arrive simultaneously. Share the preparation, keep the
        # per-fit lock nonblocking for accidental duplicate workers on one index.
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "plan.json"
        if path.exists():
            record = sealed_read(path)
            if any(record[k] != v for k, v in design.items()):
                raise ValueError("Training replication design changed")
            return record, by_id
        models = []
        for fold, split in enumerate(splits):
            fit_rows = [by_id[n] for n in split["fit"]]
            ref = fasting_reference(fit_rows)
            for slot in SLOTS:
                graph, template, _ = sources[slot]
                choices = {}
                if ref:
                    scale = original.training_scale(fit_rows, "I")
                    p0 = cold_parameters(graph, template, fit_rows)
                    base = template | {"parameters": p0}
                    bounded = original.boundary(base, "I", scale)
                    choices = family.candidates(slot, base, bounded, p0, scale, ref)
                for arm in ARMS:
                    tag = {"fold": fold, "slot": slot, "arm": arm}
                    if not ref:
                        models.append(
                            tag | {"status": "no_qualified_fasting_reference"}
                        )
                        continue
                    models.append(
                        tag
                        | {
                            "status": "prepared",
                            "scale": scale,
                            "reference": ref,
                            "graph": graph,
                            "formula": validate_formula(graph, choices[arm]),
                        }
                    )
        return sealed_write(path, {**design, "models": models}), by_id


def event_metrics(curve, scales):
    """Return sufficient statistics for sample-weighted errors in every target."""
    n = len(curve["time"])
    targets = sorted(scales)
    sse = {}
    for key in targets:
        errors = np.asarray(curve["predicted"][key]) - np.asarray(
            curve["observed"][key]
        )
        sse[key] = float(errors @ errors)
    norm = {k: sse[k] / scales[k] ** 2 for k in targets}
    y = np.asarray(curve["observed"]["I"])
    pred = np.asarray(curve["predicted"]["I"])
    return {
        "event": curve["trajectory_id"],
        "status": "scored",
        "samples": n,
        "sse": sse,
        "normalized_sse": norm,
        "per_target_nmse": {k: s / n for k, s in norm.items()},
        "pooled_nmse": float(np.mean(list(norm.values())) / n),
        "minimum_I": float(np.min(pred)),
        "observed_minimum_I": float(np.min(y)),
        "negative_I_samples": int(np.count_nonzero(pred < -1e-10)),
        "near_zero_I_with_observed_above_one": int(
            np.count_nonzero((pred < 1) & (y >= 1))
        ),
    }


def _score(model, result, held):
    names = sorted(result["scales"])
    scale = np.array([result["scales"][n] for n in names])

    def check(group):
        return _check_grid(
            model["graph"],
            model["formula"],
            _rows(group, "train"),
            result["parameters"],
            names,
            scale,
            curves=True,
        )

    pooled = check(held)
    entries = []
    if pooled["status"] == "scored":
        curves = pooled["curves"]
    else:
        curves = []
        for row in held:
            score = check([row])
            if score["status"] == "scored":
                curves.extend(score["curves"])
            else:
                entries.append(
                    {
                        "event": row["trajectory_id"],
                        "status": score["status"],
                        "reason": score.get("reason"),
                    }
                )
    behavior = (
        {r["event"]: r for r in family.behavior(curves, held, ANCHOR)["events"]}
        if curves
        else {}
    )
    entries.extend(
        event_metrics(c, result["scales"])
        | {"insulin_behavior": behavior[c["trajectory_id"]]}
        for c in curves
    )
    return {
        "grid_status": pooled["status"],
        "max_scaled_grid_difference": pooled.get("maximum_scaled_grid_difference"),
        "events": sorted(entries, key=lambda r: r["event"]),
    }


def run_one(root, plan, rows, index):
    if not 0 <= index < len(plan["models"]):
        raise ValueError("Invalid case index")
    m = plan["models"][index]
    split = plan["folds"][m["fold"]]
    identity = {
        "plan": plan["artifact_sha256"],
        "index": index,
        "slot": m["slot"],
        "fold": m["fold"],
        "arm": m["arm"],
    }
    with _lock(root / "checks" / f"{index:02}"):
        path = root / "checks" / f"{index:02}" / "outcome.json"
        if path.exists():
            record = sealed_read(path)
            if any(record[k] != v for k, v in identity.items()):
                raise ValueError("Outcome differs from fold or arm")
            return record
        if set(split["fit"]) & set(split["held"]):
            raise ValueError("Challenge fold overlaps fitting events")
        result = {"status": m["status"]}
        if m["status"] == "prepared":
            fit = original._fit(root, plan, index, [rows[n] for n in split["fit"]])
            result = {
                "fit": fit["artifact_sha256"],
                "status": fit["status"],
                "evaluations": fit["evaluations"],
                "selected_option": fit["selected_option"],
                "attempts": [
                    {
                        "stage": j,
                        "status": v["status"],
                        "reasons": v["stop_reasons"],
                        "starts_attempted": v["starts_attempted"],
                        "seconds": v["seconds"],
                        "failed_rollouts": v["failed_rollouts"],
                    }
                    for j, v in enumerate(fit["options"])
                ],
            }
            if fit["selected_option"] is not None:
                chosen = fit["options"][fit["selected_option"]]
                result.update(_score(m, chosen, [rows[n] for n in split["held"]]))
                result.update(
                    status="fitted",
                    train_nmse=chosen["training_nmse"],
                    parameters=chosen["parameters"],
                    scales=chosen["scales"],
                    train_per_target_nmse=chosen["per_target_nmse"],
                    restoring_rate_per_min=(
                        chosen["parameters"][family.RHO] ** 2
                        if m["arm"] == "restoring"
                        else None
                    ),
                )
            else:
                result["status"] = "no_verified_fit"
        result.setdefault(
            "events", [{"event": k, "status": result["status"]} for k in split["held"]]
        )
        return sealed_write(path, identity | result)


def aggregate(events, expected):
    """Keep incomplete groups explicit while reporting scores on available events."""
    if len(events) != expected or len({e["event"] for e in events}) != expected:
        raise ValueError("Duplicate or missing challenge event")
    good = [r for r in events if r["status"] == "scored"]
    counts = {
        "events": expected,
        "scored_events": len(good),
        "failed_events": expected - len(good),
    }
    if not good:
        return counts | {"pooled_nmse": None}
    n = sum(r["samples"] for r in good)
    names = sorted(good[0]["sse"])
    nmse = {k: sum(r["normalized_sse"][k] for r in good) / n for k in names}
    per = {k: float(np.sqrt(sum(r["sse"][k] for r in good) / n)) for k in names}
    mean = float(np.mean(list(nmse.values())))
    timing = [
        abs(t)
        for r in good
        if (t := r.get("insulin_behavior", {}).get("peak_time_error_min")) is not None
    ]
    return counts | {
        "samples": n,
        "available_per_target_nmse": nmse,
        "available_rmse": per,
        "available_pooled_nmse": mean,
        "pooled_nmse": mean if len(good) == expected else None,
        "near_zero_I_with_observed_above_one": sum(
            r["near_zero_I_with_observed_above_one"] for r in good
        ),
        "negative_I_samples": sum(r["negative_I_samples"] for r in good),
        "events_with_insulin_peak": len(timing),
        "median_abs_peak_time_error_min": float(np.median(timing)) if timing else None,
    }


def report(root):
    plan = sealed_read(root / "plan.json")
    records = []
    for i, m in enumerate(plan["models"]):
        path = root / "checks" / f"{i:02}" / "outcome.json"
        if not path.exists():
            return {
                "status": "incomplete",
                "completed": sum(
                    (root / "checks" / f"{j:02}" / "outcome.json").exists()
                    for j in range(len(plan["models"]))
                ),
            }
        r = sealed_read(path)
        if (
            r["plan"] != plan["artifact_sha256"]
            or r["index"] != i
            or any(r[k] != m[k] for k in ("fold", "slot", "arm"))
        ):
            raise ValueError("Outcome identity changed")
        if (
            "fit" in r
            and sealed_read(root / "fits" / f"{i:02}" / "result.json")[
                "artifact_sha256"
            ]
            != r["fit"]
        ):
            raise ValueError("Fit changed after challenge scoring")
        records.append(r)
    groups = {}
    for slot in SLOTS:
        group = {}
        mapped = {}
        for arm in ARMS:
            matching = [r for r in records if (r["slot"], r["arm"]) == (slot, arm)]
            ev = [e for r in matching for e in r["events"]]
            group[arm] = aggregate(ev, sum(len(s["held"]) for s in plan["folds"]))
            mapped[arm] = {e["event"]: e for e in ev if e["status"] == "scored"}
        both = mapped["bounded"].keys() & mapped["restoring"].keys()
        group["paired"] = {
            "events": len(both),
            "restoring_lower_pooled": sum(
                mapped["restoring"][i]["pooled_nmse"]
                < mapped["bounded"][i]["pooled_nmse"]
                for i in both
            ),
            "restoring_lower_insulin": sum(
                mapped["restoring"][i]["sse"]["I"] < mapped["bounded"][i]["sse"]["I"]
                for i in both
            ),
        }
        groups[str(slot)] = group
    return sealed_write(
        root / "summary.json",
        {
            "status": "complete",
            "plan": plan["artifact_sha256"],
            "checks": [r["artifact_sha256"] for r in records],
            "groups": groups,
            "evaluations": sum(r.get("evaluations", 0) for r in records),
            "validation_opened": False,
            "test_data_opened": False,
            "llm_calls": 0,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="action", required=True)
    run = cmd.add_parser("run")
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--index", type=int)
    for sub in (run, cmd.add_parser("report")):
        sub.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        plan, rows = prepare(args.campaign, args.root)
        if args.index is None:
            for i in range(len(plan["models"])):
                run_one(args.root, plan, rows, i)
        else:
            run_one(args.root, plan, rows, args.index)
    print(json.dumps(report(args.root), indent=2))


if __name__ == "__main__":
    main()
