#!/usr/bin/env python3
"""Fit the three decoded T2 insulin-recovery equations and their parent."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.t2_breadth_rollout import BreadthRollout
from autoformalism.research import t2_breadth_fitting as fit
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit
from autoformalism.staged_topology import content_hash
from scripts import run_t2_breadth_multistart as prior

PACKET_SHA = "1f8a0bb908f48940af457e29f8e30e4356418a81481a3704ac945d27b612e250"
PARENT = "c15-2"
MODELS = (
    PARENT,
    "c15-2-seed109-initial",
    "c15-2-seed109-correction3",
    "c15-2-seed116-correction3",
)
STARTS = ("supplied", "activated")


def initialization_policy(graph: dict) -> dict:
    """The complete saved roots override constructor guesses for trajectory slots."""
    policy = deepcopy(graph)
    for rule in policy["rules"].values():
        initial = rule["initial"]
        if initial["mode"] == "value" and initial["scope"] == "trajectory":
            initial.pop("guess", None)
    return policy


def load_packet(path: Path) -> dict:
    """Authenticate the supplied ZIP and join the original train/model identities."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != PACKET_SHA:
        raise ValueError("different 20B fitting handoff")
    with ZipFile(path) as archive:

        def record(name):
            payload = json.loads(archive.read(name))
            digest = payload.pop("artifact_sha256")
            if content_hash(payload) != digest:
                raise ValueError(f"constructor seal differs for {name}")
            return payload | {"artifact_sha256": digest}

        manifest = record("manifest.json")
        source = record(manifest["parent_file"])
        training = source["training"]
        policy = source["fitting_contract"]
        parent = source["parent"]
        ref = parent["parent_reference"]
        point = parent["parameters_with_training_roots"]
        if (
            training["name"] != "train"
            or len(training["rows"]) != 16
            or public.content_sha256(training) != policy["training_sha256"]
            or public.content_sha256(point) != ref["fitted_parameter_sha256"]
            or public.content_sha256(parent["problem"]) != ref["fitted_problem_sha256"]
        ):
            raise ValueError("parent or public training identity differs")
        models = {
            PARENT: {
                "problem": parent["problem"],
                "warm": point,
                "smoke": manifest["parent_fitted_point_replay"],
            }
        }
        for item in manifest["candidates"]:
            d = record(item["file"])
            name = d["branch"]
            if (
                name in models
                or name != item["name"]
                or d["artifact_sha256"] != item["artifact_sha256"]
                or public.content_sha256(d["problem"]) != item["model_sha256"]
                or d["parent_reference"] != ref
                or d["training_sha256"] != policy["training_sha256"]
                or d["source_plan_sha256"] != manifest["source_plan_sha256"]
                or initialization_policy(d["problem"]["graph"])
                != initialization_policy(parent["problem"]["graph"])
                or d["fixed_start_parameters_with_training_roots"]
                != point | d["new_parameter_fixed_start"]
            ):
                raise ValueError(f"revision identity or initialization differs: {name}")
            models[name] = {
                "problem": d["problem"],
                "warm": d["fixed_start_parameters_with_training_roots"],
                "smoke": d["fixed_start_smoke"],
            }
    if set(models) != set(MODELS):
        raise ValueError("unexpected decoded revision roster")
    return {"models": models, "training": training, "policy": policy}


def model_for(packet: dict, name: str) -> BreadthRollout:
    item = packet["models"][name]
    model = BreadthRollout(
        name,
        PrefixFitProblem.model_validate(item["problem"]),
        PublicSplit.model_validate(packet["training"]),
        item["warm"],
    )
    if sum(n.startswith("trajectory:") for n in model.names) != 32:
        raise ValueError("expected both hidden initials free on all 16 trajectories")
    return model


def start_vector(model: BreadthRollout, kind: str) -> dict:
    """Give every alternate start the same parent perturbation and release new gains."""
    p = model.warm.copy()
    if kind == "activated":
        p["k_prod"] *= 0.8
        if "k_I_feedback" in p:
            p.update(k_X_decay_glucose=1e-5, k_I_feedback=1e-3)
        if "Gp_thr_action" in p:
            glucose = model.observed[:, :, model.channels.index("Gp")]
            p.update(
                af_process_gain_6_0=0.0004,
                k_glucose_action_thr=1 / float(np.std(glucose)),
                Gp_thr_action=float(np.median(glucose)),
            )
    elif kind != "supplied":
        raise ValueError("unknown start")
    model.reduce(p)
    return p


def verify_point(model: BreadthRollout, p: dict) -> dict:
    """Score a full free rollout and independently replay the measured grid."""
    dense, _ = model.original(p, 8, method="Radau")
    other, _ = model.original(p, 1, method="DOP853")
    measured = dense[:, ::8]
    if not np.allclose(measured, other, rtol=5e-7, atol=1e-6 * model.scales):
        raise ValueError("adaptive original-equation replays disagree")
    for channel in ("Gp", "I"):
        ci = model.channels.index(channel)
        np.testing.assert_allclose(
            measured[:, 0, ci], model.observed[:, 0, ci], atol=1e-9, rtol=0
        )
    minima = dense.min(axis=(0, 1))
    insulin = model.channels.index("I")
    large = next(
        i for i, row in enumerate(model.rows) if row.trajectory_id == "train_005"
    )
    a, b = (model.time.tolist().index(t) for t in (240, 300))
    delta = float(measured[large, b, insulin] - measured[large, a, insulin])
    return {
        "parameters": p,
        "parameter_sha256": public.content_sha256(p),
        "training_nmse": model.nmse(measured),
        "sampled_domain_passed": bool(
            np.all(minima >= -fit.fitting.DOMAIN_TOL * model.scales)
        ),
        "minima": dict(zip(model.channels, minima.tolist(), strict=True)),
        "late_insulin_change": delta,
        "max_scaled_solver_difference": float(
            np.max(abs(measured - other) / model.scales)
        ),
    }


def check_start(replay: Mapping, smoke: Mapping) -> None:
    """Check the two supplied reference scores before optimizing that model."""
    if not np.allclose(
        [replay["training_nmse"], replay["late_insulin_change"]],
        [smoke["pooled_training_nmse"], smoke["I_300_minus_240_pmol_L"]],
        rtol=2e-7,
        atol=1e-7,
    ):
        raise ValueError("fixed-start handoff replay differs")


def run_one(path: Path, root: Path, name: str, kind: str, plan: dict) -> dict:
    """Journal search blocks and retain independently replayed checkpoints."""
    packet = load_packet(path)
    model = model_for(packet, name)
    start = start_vector(model, kind)
    key = f"{name}/{kind}"
    folder = root / key
    identity = {"plan_sha256": public.content_sha256(plan), "task": key}
    if public.content_sha256(start) != plan["starts"][key]:
        raise ValueError("fit start differs from plan")
    with public._lock(folder):
        verified, known, history, errors = {}, {}, [], {}

        def operation(label, allowance, action):
            receipt = fit.journal._operation(folder, identity, label, allowance, action)
            if receipt["status"] != "complete":
                errors[label] = receipt.get("error", receipt["status"])
            return receipt.get("value") or receipt.get("observed_progress") or {}

        def replay(label, point):
            digest = public.content_sha256(point)
            if digest in known:
                return
            row = operation(label, 600, lambda _: verify_point(model, point))
            if row.get("parameter_sha256") == digest:
                verified[label] = row
                known[digest] = label

        def best(physical):
            return fit._select(verified, feasible=physical)

        replay("start", start)
        if "start" not in verified:
            raise ValueError(f"{key} start unavailable: {errors}")
        if kind == "supplied":
            check_start(verified["start"], packet["models"][name]["smoke"])
        for block, physical, cap in prior.BLOCKS:
            seeds = {"start": start}
            for feasible in (False, True):
                choice = best(feasible)
                if choice["endpoint"]:
                    seeds[choice["origin"]] = choice["endpoint"]["parameters"]

            def search(save, seeds=seeds, physical=physical, cap=cap):
                return fit.fitting.search(
                    model,
                    seeds,
                    physical=physical,
                    control=False,
                    seconds=prior.SEARCH_SECONDS,
                    calls=cap,
                    save=save,
                )

            result = operation(f"search_{block}", prior.SEARCH_SECONDS + 510, search)
            for category, entry in result.get("best", {}).items():
                replay(f"{block}_{category}", entry["parameters"])
            history.append(
                {
                    "block": block,
                    "cap": cap,
                    "calls": result.get("search_calls_started"),
                    "valid": result.get("search_calls_completed"),
                    "stop": result.get("status"),
                    "solver": result.get("solver_status"),
                    "retained_nmse": (best(True)["endpoint"] or {}).get(
                        "training_nmse"
                    ),
                }
            )
        complete = not errors and all(
            h["valid"]
            and h["stop"] in ("budget_exhausted", "solver_returned", "solver_stopped")
            for h in history
        )
        row = {
            "identity": identity,
            "status": "complete" if complete else "incomplete",
            "start": verified["start"],
            "best": best(True),
            "numerical": best(False),
            "errors": errors,
            "blocks": history,
            "reduced_dimension": len(model.names),
        }
        fit.seal(folder / "fit.json", row)
        return row


def run(path: Path, root: Path, workers: int) -> dict:
    packet = load_packet(path)
    tasks = [(name, kind) for kind in STARTS for name in MODELS]
    models = {name: model_for(packet, name) for name in MODELS}
    plan = {
        "protocol": "t2-20b-replayed-revision-refits-1",
        "zip_sha256": PACKET_SHA,
        "training_sha256": packet["policy"]["training_sha256"],
        "prefix_policy": packet["policy"]["prefix_policy"],
        "models": {
            n: public.content_sha256(v["problem"]) for n, v in packet["models"].items()
        },
        "starts": {
            f"{n}/{k}": public.content_sha256(start_vector(models[n], k))
            for n, k in tasks
        },
        "blocks": [list(b) for b in prior.BLOCKS],
        "seconds_per_block": prior.SEARCH_SECONDS,
        "selection": "lowest verified training NMSE with all 3 targets in domain",
        "held_out_access": False,
        "runtime": public._runtime(),
        "source_sha256": public._source_identity(),
        "scripts": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(prior.__file__))
        },
    }
    with public._lock(root):
        fit.seal(root / "plan.json", plan)
    print("Verified handoff and froze eight independent fits.", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(run_one, path, root, n, k, plan) for n, k in tasks]
        results = [job.result() for job in jobs]
    summary = {
        "status": "complete"
        if all(r["status"] == "complete" for r in results)
        else "incomplete",
        "plan_sha256": public.content_sha256(plan),
        "fits": {r["identity"]["task"]: public.content_sha256(r) for r in results},
    }
    with public._lock(root):
        fit.seal(root / "fits.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.packet, args.root, args.workers), indent=2))
