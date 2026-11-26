"""Fit matched one-source affine and square forms on frozen T2 training folds."""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import trajectory_function_fitting as rollout
from autoformalism.research import trajectory_nonnegative_fitting as bounded
from autoformalism.staged_topology import content_hash
from scripts.audit_hidden_shape_t2 import FROZEN
from scripts.construction_functions import file_hash
from scripts.matched_equation_experiment import DEFAULT as SOURCE
from scripts.matched_equation_experiment import _registered

DEFAULT = SOURCE.parent / "hidden-form-forecast-20261126"
CONTRASTS = (
    ("graph00", "U", "EGP"),
    ("graph04", "I", "EGP"),
    ("graph05", "I", "EGP"),
    ("graph05", "U", "I"),
)
FAMILIES = ("affine", "square")
STARTS = ("fold_local", "constant_observed")
BUDGET = {"max_evaluations": 500, "wall_seconds": 90.0, "seed": 100, "starts": 1}


def rotations(fit_ids: list[str], other_ids: list[str]) -> list[dict]:
    """Make two choices using disjoint event sets and balance held exposure."""
    if (
        len(fit_ids) != 8
        or len(other_ids) != 8
        or len(set(fit_ids + other_ids)) != 16
        or any(not i.startswith("train_") for i in fit_ids + other_ids)
    ):
        raise ValueError("Require sixteen distinct training events")
    a, b = sorted(other_ids)[::2], sorted(other_ids)[1::2]
    return [{"selection_ids": a, "held_ids": b}, {"selection_ids": b, "held_ids": a}]


def competing_forms(graph: dict, baseline: dict, target: str, source: str) -> dict:
    """Replace one normalized raw x basis by its tangent-matched raw x² basis."""
    form = rollout.validate_formula(graph, baseline)
    if source not in graph["sources"][target]:
        raise ValueError("Unregistered edge")
    # Match the exact normalized basis emitted by the trusted candidate constructor.
    # The parser below independently validates the updated full equation.
    number = r"-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?"
    pattern = rf"\(\({re.escape(source)}-\(({number})\)\)/({number})\)"
    expr = form["equations"][target]
    terms = list(re.finditer(pattern, expr))
    if len(terms) != 1:
        raise ValueError("Expected a single normalized affine source in the equation")
    token = terms[0]
    center, scale = map(float, token.groups())
    if center <= 0 or scale <= 0:
        raise ValueError("Tangent matching requires a positive source mean and scale")
    square = deepcopy(form)
    # Algebraically equal to (x²-c²)/(2cs). Local value/slope match at c.
    replacement = f"({token.group()}*(({source}+({center:.12g}))/{2 * center:.12g}))"
    square["equations"][target] = (
        expr[: token.start()] + replacement + expr[token.end() :]
    )
    return {"affine": form, "square": rollout.validate_formula(graph, square)}


def starting_vectors(graph: dict, form: dict, targets: list[str]) -> dict:
    """Supply two starts, including a feasible common constant observed trajectory."""
    parser = RestrictedParser()
    observed_params = set()
    for name in targets:
        if graph["initializers"].get(name, {}).get("mode") != "observed_initial":
            raise ValueError("The constant-trajectory start requires observed initials")
        parsed = parser.parse(form["equations"][name], location=f"equation.{name}")
        observed_params.update(parsed.symbols & form["parameters"].keys())
    constant = {
        k: 0.0 if k in observed_params else v for k, v in form["parameters"].items()
    }
    return {"fold_local": form["parameters"], "constant_observed": constant}


def prepare(root: Path, source: Path) -> dict:
    """Register hypotheses and whole-event partitions before observing forecasts."""
    parent = _registered(source / "plan.json", FROZEN)
    index = {p["panel_id"]: p for p in parent["panels"]}
    panels = []
    for graph_name, target, src in CONTRASTS:
        for fold in (0, 1):
            base = f"{graph_name}-fold{fold}"
            cfg = index[base]
            other = index[f"{graph_name}-fold{1 - fold}"]
            packet = _registered(source / "packets" / f"{base}.json", cfg["packet"])
            fit_ids = [r["trajectory_id"] for r in packet["train_rows"]]
            if set(packet["shape"]["candidate_families"][0].values()) != {"affine"}:
                raise ValueError(
                    "Expected an affine graph in the pinned fitting packet"
                )
            forms = competing_forms(
                packet["graph"], packet["shape"]["candidates"][0], target, src
            )
            panels.append(
                {
                    "panel_id": f"{graph_name}-{target}-{src}-fold{fold}",
                    "graph_name": graph_name,
                    "target": target,
                    "source": src,
                    "packet_id": base,
                    "packet": cfg["packet"],
                    "other_packet_id": other["panel_id"],
                    "other_packet": other["packet"],
                    "opposite_sha256": cfg["held_sha256"],
                    "fit_ids": fit_ids,
                    "choice_rotations": rotations(fit_ids, cfg["held_ids"]),
                    "forms": forms,
                }
            )
    with _lock(root):
        return sealed_write(
            root / "plan.json",
            {
                "protocol": "hidden-form-whole-event-forecast-1",
                "source_plan": FROZEN,
                "families": list(FAMILIES),
                "starts": list(STARTS),
                "per_start_budget": BUDGET,
                "nonnegative_targets": parent["nonnegative_targets"],
                "panels": panels,
                "scope": "Exploratory edges previously screened on all training events",
                "code": {
                    Path(p).name: file_hash(Path(p))
                    for p in (__file__, rollout.__file__, bounded.__file__)
                },
                "versions": {x: version(x) for x in ("casadi", "numpy", "scipy")},
            },
        )


def _train_start(
    folder: Path,
    plan: str,
    name: str,
    graph: dict,
    form: dict,
    rows: list,
    targets: list,
) -> dict:
    identity = {"plan": plan, "start": name, "formula": content_hash(form)}
    path = folder / f"{name}.json"
    if path.exists():
        receipt = sealed_read(path)
        if any(receipt[k] != v for k, v in identity.items()):
            raise ValueError("Different fit design at existing checkpoint")
        return receipt
    reservation = folder / f"{name}.start.json"
    interrupted = reservation.exists()
    sealed_write(reservation, identity)
    if interrupted:
        result = {"status": "interrupted", "evaluations": None}
    else:
        result = bounded.fit_nonnegative(graph, form, rows, targets=targets, **BUDGET)
        if (
            result["budget"] != BUDGET
            or result["evaluations"] > BUDGET["max_evaluations"]
        ):
            raise ValueError("Unexpected fit budget")
    return sealed_write(path, {**identity, "result": result})


def _forecast(
    graph: dict, form: dict, fit: dict, row: dict, targets: list[str]
) -> dict:
    """Generate endogenous states from permitted initials and known forcings."""
    scales = fit["scales"]
    ordered = sorted(scales)
    verified = rollout._check_grid(
        graph,
        form,
        rollout._rows([row], "train"),
        fit["parameters"],
        ordered,
        np.array([scales[k] for k in ordered]),
        curves=True,
    )
    summary = {
        "status": verified["status"],
        "scales": scales,
        "nmse": None,
        "maximum_scaled_grid_difference": verified.get(
            "maximum_scaled_grid_difference"
        ),
    }
    if verified["status"] != "scored":
        return summary
    curve = verified["curves"][0]
    negatives = verified["negative_samples"]
    by_target = {
        k: float(
            np.mean(
                (
                    (np.asarray(curve["predicted"][k][1:]) - curve["observed"][k][1:])
                    / scales[k]
                )
                ** 2
            )
        )
        for k in ordered
    }
    diagnostic = float(np.mean(list(by_target.values())))
    valid = all(negatives[k] == 0 for k in targets)
    return {
        **summary,
        **curve,
        "negative_samples": negatives,
        "status": "scored" if valid else "negative_forecast",
        "per_target_nmse": by_target,
        "diagnostic_nmse": diagnostic,
        "nmse": diagnostic if valid else None,
    }


def _run(job) -> dict:
    root, source, plan_id, panel, family, targets = job
    pid = panel["panel_id"]
    folder = root / "fits" / pid / family
    marker = {"plan": plan_id, "panel_id": pid, "family": family}

    def packet(key):
        return _registered(
            source / "packets" / f"{panel[key + '_id']}.json", panel[key]
        )

    with _lock(folder):
        data = packet("packet")
        graph, train = data["graph"], data["train_rows"]
        if [r["trajectory_id"] for r in train] != panel["fit_ids"]:
            raise ValueError("Unexpected fit event set")
        form = panel["forms"][family]
        receipts = {}
        for start, values in starting_vectors(graph, form, targets).items():
            f = {**form, "parameters": values}
            receipts[start] = _train_start(
                folder, plan_id, start, graph, f, train, targets
            )
        fitted = [
            (r["result"]["training_nmse"], i, k)
            for i, (k, r) in enumerate(receipts.items())
            if r["result"]["status"] == "fitted"
        ]
        best = min(fitted)[2] if fitted else None
        summary = sealed_write(
            folder / "best.json",
            {
                **marker,
                "best_start": best,
                "starts": {key: r["artifact_sha256"] for key, r in receipts.items()},
                "training_nmse": min(fitted)[0] if fitted else None,
            },
        )
        # A training-only endpoint is frozen before reading any prospective forecasts.
        other = packet("other_packet")
        held = other["train_rows"]
        if content_hash(held) != panel["opposite_sha256"] or graph != other["graph"]:
            raise ValueError("Frozen complementary training data differ")
        for row in held:
            eid = row["trajectory_id"]
            identity = {**marker, "event_id": eid, "fit": summary["artifact_sha256"]}
            path = root / "curves" / pid / family / f"{eid}.json"
            if path.exists():
                prior = sealed_read(path)
                if any(prior[k] != v for k, v in identity.items()):
                    raise ValueError("Different event identity")
                continue
            out = (
                _forecast(graph, form, receipts[best]["result"], row, targets)
                if best is not None
                else {"status": "no_fit", "nmse": None}
            )
            sealed_write(path, {**identity, **out})
        return {
            "panel": pid,
            "family": family,
            "start": best,
            "training_nmse": summary["training_nmse"],
        }


def run(root: Path, source: Path, workers: int) -> None:
    plan = prepare(root, source)
    jobs = [
        (root, source, plan["artifact_sha256"], p, fam, plan["nonnegative_targets"])
        for p in plan["panels"]
        for fam in FAMILIES
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for receipt in pool.map(_run, jobs):
            print(receipt, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.root, args.source, args.workers)
