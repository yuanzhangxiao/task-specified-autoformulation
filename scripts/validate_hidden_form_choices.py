"""Evaluate frozen shape choices on the registered validation set."""

from __future__ import annotations

import argparse
from inspect import getsource
from pathlib import Path

import numpy as np

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import forecast_disagreement as policy
from autoformalism.research import trajectory_function_fitting as rollout
from autoformalism.staged_topology import content_hash
from scripts import report_hidden_form_forecasts as reports
from scripts import run_hidden_form_forecasts as fits
from scripts.audit_hidden_shape_t2 import FROZEN
from scripts.construction_functions import file_hash
from scripts.matched_equation_experiment import _registered
from scripts.topology_forecast_fit import INPUTS, TRAIN, VALIDATION

LOCKED_CHOICES = "ba0dda85d48364cc79e8fae0577015152107c77b661aa56c51a9d26eeff006a2"
LOCKED_FITS = "3f08fd281e122a1d8765739dc7a25be19dce7c9f8bad72485d243dfb63202d40"


def prepare(root: Path, source: Path) -> dict:
    """Freeze choices, fits and scoring design before opening validation."""
    fp = _registered(root / "plan.json", LOCKED_FITS)
    choices = _registered(root / "choices.json", LOCKED_CHOICES)
    if (
        _registered(root / "selection_plan.json", choices["policy"])["fit_plan"]
        != LOCKED_FITS
    ):
        raise ValueError("Different selection fit plan")
    original = _registered(source / "plan.json", FROZEN)
    if (fp["source_plan"], original["registered_training"]) != (FROZEN, TRAIN):
        raise ValueError("Different fitting release")
    endpoints = {}
    decisions = []
    for panel in fp["panels"]:
        name = panel["panel_id"]
        endpoints[name] = {}
        for family in fits.FAMILIES:
            folder = root / "fits" / name / family
            best = sealed_read(folder / "best.json")
            if (best["plan"], best["panel_id"], best["family"]) != (
                LOCKED_FITS,
                name,
                family,
            ):
                raise ValueError("Fit endpoint differs")
            endpoints[name][family] = best["artifact_sha256"]
        for i, rot in enumerate(panel["choice_rotations"]):
            receipt = _registered(
                reports._choice_path(root, name, i), choices["receipts"][len(decisions)]
            )
            if (
                receipt["policy"],
                receipt["panel"],
                receipt["rotation"],
                set(receipt["events"]),
            ) != (choices["policy"], name, i, set(rot["selection_ids"])):
                raise ValueError("Decision differs")
            decisions.append(
                {
                    "panel": name,
                    "rotation": i,
                    "decision": receipt["artifact_sha256"],
                    "eligible": receipt["eligible"],
                    "votes": receipt["votes"],
                }
            )
    if len(decisions) != len(choices["receipts"]):
        raise ValueError("Wrong rotation count")
    with _lock(root / "official-validation" / "prepare-lock"):
        return sealed_write(
            root / "official-validation" / "plan.json",
            {
                "protocol": "frozen-forecast-choice-external-validation-1",
                "fit_plan": LOCKED_FITS,
                "choice_manifest": LOCKED_CHOICES,
                "train_intake": TRAIN,
                "validation_intake": VALIDATION,
                "dataset": original["dataset"],
                "endpoints": endpoints,
                "decisions": decisions,
                "bounds": fp["nonnegative_targets"],
                "metric": (
                    "Post-initial equal-target NMSE, fitting scales, "
                    "equal weights across events and rotations."
                ),
                "eligibility": (
                    "Grid agreement and nonnegativity on each event; "
                    "all events required for each rotation."
                ),
                "event_level_secondary": "Common eligible event-rotations.",
                "code": {
                    p.name: file_hash(p)
                    for p in (
                        Path(__file__),
                        Path(policy.__file__),
                        Path(rollout.__file__),
                    )
                },
                "aggregation": content_hash(getsource(reports.compare)),
            },
        )


def _score(graph, formula, fit, row, bounds):
    """Score one validation event, enforcing positivity and grid agreement."""
    got = rollout.score_formula(graph, formula, fit, [row])
    result = {
        "status": got["status"],
        "nmse": None,
        "negative_samples": got.get("negative_samples", {}),
        "grid_difference": got.get("maximum_scaled_grid_difference"),
    }
    if got["status"] != "scored":
        return result
    curve = got["curves"][0]
    if curve["trajectory_id"] != row["trajectory_id"]:
        raise ValueError("Different forecast event")
    per_target = {
        name: float(
            np.mean(
                (
                    (np.array(values[1:]) - curve["observed"][name][1:])
                    / fit["scales"][name]
                )
                ** 2
            )
        )
        for name, values in curve["predicted"].items()
    }
    score = float(np.mean(list(per_target.values())))
    ok = all(got["negative_samples"][k] == 0 for k in bounds)
    return {
        **result,
        "status": "scored" if ok else "negative_forecast",
        "nmse": score if ok else None,
        "diagnostic_nmse": score,
        "per_target_nmse": per_target,
        "minimum_prediction": got["minimum_prediction"],
    }


def _assess(root, source, plan, panel, family, events):
    name = panel["panel_id"]
    training = root / "fits" / name / family
    identity = plan["endpoints"][name][family]
    best = _registered(training / "best.json", identity)
    start = best["best_start"]
    packet = _registered(
        source / "packets" / f"{panel['packet_id']}.json", panel["packet"]
    )
    if start is None:
        fit = None
    else:
        item = _registered(training / f"{start}.json", best["starts"][start])
        graph = packet["graph"]
        form = panel["forms"][family]
        form = {
            **form,
            "parameters": fits.starting_vectors(graph, form, plan["bounds"])[start],
        }
        if (item["plan"], item["start"], item["formula"]) != (
            LOCKED_FITS,
            start,
            content_hash(form),
        ):
            raise ValueError("Formula or training start differs")
        fit = item["result"]
    results = {}
    for row in events:
        eid = row["trajectory_id"]
        path = (
            root / "official-validation" / "forecasts" / name / family / f"{eid}.json"
        )
        marker = {
            "plan": plan["artifact_sha256"],
            "event": eid,
            "panel": name,
            "family": family,
            "fit": identity,
        }
        if path.exists():
            data = sealed_read(path)
            if any(data[k] != v for k, v in marker.items()):
                raise ValueError("Forecast checkpoint differs")
        else:
            out = (
                _score(graph, form, fit, row, plan["bounds"])
                if fit is not None
                else {"status": "no_fit", "nmse": None, "negative_samples": {}}
            )
            data = sealed_write(path, {**marker, **out})
        results[eid] = data
    return results


def evaluate(root: Path, source: Path) -> dict:
    """Score frozen fits and choices on the authenticated validation intake."""
    out = root / "official-validation"
    if not (out / "plan.json").exists():
        raise ValueError("Prepare the scoring protocol before evaluating validation")
    plan = prepare(root, source)
    fp = _registered(root / "plan.json", LOCKED_FITS)
    with _lock(out / "evaluation-lock"):
        validation = _registered(INPUTS / "validation.json", plan["validation_intake"])
        if (validation["training_intake_sha256"], validation["split"]) != (
            plan["train_intake"],
            "val",
        ):
            raise ValueError("Mismatched validation release")
        records = validation["rows"][plan["dataset"]]
        ids = [r.trajectory_id for r in rollout._rows(records, "val")]
        rows = []
        paired_events = []
        failures = []
        for p in fp["panels"]:
            name = p["panel_id"]
            events = {
                f: _assess(root, source, plan, p, f, records) for f in fits.FAMILIES
            }
            losses = {
                eid: {f: events[f][eid]["nmse"] for f in fits.FAMILIES} for eid in ids
            }
            for values in events.values():
                failures.extend(
                    {
                        k: r[k]
                        for k in (
                            "panel",
                            "family",
                            "event",
                            "status",
                            "negative_samples",
                        )
                    }
                    for r in values.values()
                    if r["status"] != "scored"
                )
            for dec in [d for d in plan["decisions"] if d["panel"] == name]:
                row = {
                    **dec,
                    "graph": p["graph_name"],
                    "target": p["target"],
                    "source": p["source"],
                }
                rows.append(
                    {
                        **row,
                        "scores": {
                            k: policy.score_votes(v, losses)
                            for k, v in dec["votes"].items()
                        },
                    }
                )
                for eid in ids:
                    paired_events.append(
                        {
                            **row,
                            "event": eid,
                            "scores": {
                                k: policy.score_votes(v, {eid: losses[eid]})
                                for k, v in dec["votes"].items()
                            },
                        }
                    )
        return sealed_write(
            out / "summary.json",
            {
                "plan": plan["artifact_sha256"],
                "choice_manifest": plan["choice_manifest"],
                "events": ids,
                "rows": rows,
                "failures": failures,
                "summary": reports.compare(rows),
                "competitive_only": reports.compare(
                    [r for r in rows if len(r["eligible"]) == 2]
                ),
                "event_level_secondary": reports.compare(paired_events),
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    parser.add_argument("--root", type=Path, default=fits.DEFAULT)
    parser.add_argument("--source", type=Path, default=fits.SOURCE)
    args = parser.parse_args()
    receipt = (prepare if args.stage == "prepare" else evaluate)(args.root, args.source)
    print(receipt["artifact_sha256"])
