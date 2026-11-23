"""Evaluate fixed steps from each disposal incumbent to its two fitted endpoints."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import fit_positive_insulin_handoff as orion
from scripts import positive_insulin_refit_inputs as intake
from scripts import qualify_t2_disposal_revision as qualify
from scripts import refit_t2_disposal_revision as matched
from scripts.construction_functions import file_hash

OUT = Path("output/t2-disposal-feasible-20261123-v1")
FRACTIONS = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 1.0)
TOL = 1e-10


def interpolate(start: dict, end: dict, fraction: float) -> dict:
    """Interpolate named parameters, preserving both endpoints exactly."""
    if start.keys() != end.keys() or not 0 <= fraction <= 1:
        raise ValueError("Interpolation requires aligned vectors and a unit fraction")
    if fraction == 0:
        return dict(start)
    if fraction == 1:
        return dict(end)
    return {k: p + fraction * (end[k] - p) for k, p in start.items()}


def winner(trials: list[dict]) -> dict:
    """Select the best feasible training score, keeping the first on ties."""
    best = trials[0]
    if best["fraction"] != 0 or not best["domain"]["eligible"]:
        raise ValueError("The first trial must be an admissible incumbent")
    for item in trials[1:]:
        if (
            item["domain"]["eligible"]
            and item["training_nmse"] < best["training_nmse"] - TOL
        ):
            best = item
    return best


def scan(
    arm: str,
    model: dict,
    attempts: dict,
    rows: list,
    root: Path,
    source: Path,
    source_plan: str,
    plan: str,
) -> dict:
    """Evaluate one prespecified two-ray grid, reusing the matched endpoint replays."""
    trials = []
    p = attempts["incumbent"]["parameters"]
    cache = orion.cache_replay
    base = intake.with_parameters(model, p)
    ref = cache(base, rows, root=source, identity=source_plan, split="train")
    for label, frac in [
        ("incumbent", 0.0),
        *((n, s) for n in orion.STARTS for s in FRACTIONS),
    ]:
        params = interpolate(p, attempts[label]["parameters"], frac)
        fitted = intake.with_parameters(model, params)
        use_source = frac in (0, 1)
        result = cache(
            fitted,
            rows,
            root=source if use_source else root,
            identity=source_plan if use_source else plan,
            split="train",
        )
        score = result["summary"]["pooled_nmse"]
        checks = matched.domain(result, rows, params)
        if checks["eligible"]:
            qualify.paired(ref["events"], result["events"])
        if use_source and (
            not np.isclose(
                score, attempts[label]["training_nmse"], atol=1e-12, rtol=1e-12
            )
            or checks != attempts[label]["domain"]
        ):
            raise ValueError("Cached fit endpoint differs from matched receipt")
        trials.append(
            {
                "start": label,
                "fraction": frac,
                "parameters": params,
                "training_nmse": score,
                "domain": checks,
            }
        )
    return {"arm": arm, "trials": trials, "winner": winner(trials)}


def run(source: Path, root: Path, jobs: int, train_only: bool) -> dict:
    """Freeze the design and train-only selection before evaluating either endpoint."""
    if jobs not in {1, 2}:
        raise ValueError("Choose one or two workers")
    with _lock(root):
        src = sealed_read(source / "training-freeze.json")
        previous = sealed_read(source / "plan.json")
        if (
            src["plan"] != previous["artifact_sha256"]
            or previous["source"] != matched.FREEZE
            or src["validation_opened_at_freeze"] is not False
        ):
            raise ValueError("Invalid source campaign")
        for arm in matched.ARMS:
            if src["fits"][arm]["winner"] != "incumbent" or set(
                src["fits"][arm]["attempts"]
            ) != {"incumbent", *orion.STARTS}:
                raise ValueError("Expected two rejected or tied endpoints per arm")
            if content_hash(src["models"][arm]) != previous["model_sha256"][arm]:
                raise ValueError("Source vector differs from registered incumbent")
        paths = [
            Path(__file__),
            Path(matched.__file__),
            Path(qualify.__file__),
            Path(orion.__file__),
            Path(intake.__file__),
            Path(intake.original.__file__),
        ]
        plan = sealed_write(
            root / "plan.json",
            {
                "source_training_freeze": src["artifact_sha256"],
                "source_plan": src["plan"],
                "code": {p.name: file_hash(p) for p in paths},
                "fractions": [0.0, *FRACTIONS],
                "starts": list(orion.STARTS),
                "domain": previous["domain"],
                "selection_tolerance": TOL,
                "training": previous["training"],
                "development": previous["validation"],
            },
        )
        identity = plan["artifact_sha256"]
        rows = intake.load_training(orion.BASE / "train.json")
        args = [
            (
                a,
                src["models"][a],
                src["fits"][a]["attempts"],
                rows,
                root,
                source,
                src["plan"],
                identity,
            )
            for a in matched.ARMS
        ]
        if jobs == 1:
            scans = [scan(*x) for x in args]
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                future = [pool.submit(scan, *x) for x in args]
                scans = [f.result() for f in future]
        best = {
            s["arm"]: intake.with_parameters(
                src["models"][s["arm"]], s["winner"]["parameters"]
            )
            for s in scans
        }
        freeze = sealed_write(
            root / "training-freeze.json",
            {
                "plan": identity,
                "models": best,
                "scans": {s["arm"]: s for s in scans},
                "validation_opened_at_freeze": False,
            },
        )
        scores = {}
        for split in ("train",) if train_only else ("train", "val"):
            if split == "val":
                if (
                    sealed_read(root / "train-report.json")["freeze"]
                    != freeze["artifact_sha256"]
                ):
                    raise ValueError("Training replay required before development")
                v = intake.original.read_validation(
                    orion.BASE / "validation.json",
                    {
                        "training_intake": plan["training"],
                        "validation_intake": plan["development"],
                    },
                )
                rows = [x.model_dump(mode="json") for x in v]
            models = {
                "parent": src["models"]["parent"],
                "retained_parent": best["parent"],
                "selected": best["selected"],
            }
            report = qualify.evaluate(
                models,
                rows,
                root,
                identity,
                split,
                sealed_read(orion.OUT / f"{split}-report.json"),
            )
            raw = report["models"]
            report["models"] = {
                k: raw[n]
                for k, n in [("parent", "retained_parent"), ("selected", "selected")]
            }
            report["paired"] = {
                "selected": qualify.paired(
                    raw["retained_parent"]["events"], raw["selected"]["events"]
                )
            }
            record = sealed_write(
                root / f"{split}-report.json",
                report | {"freeze": freeze["artifact_sha256"]},
            )
            scores[split] = {
                k: v["summary"]["pooled_nmse"] for k, v in record["models"].items()
            }
        return scores


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, default=matched.OUT)
    cli.add_argument("--root", type=Path, default=OUT)
    cli.add_argument("--jobs", type=int, default=2)
    cli.add_argument("--train-only", action="store_true")
    args = cli.parse_args()
    print(json.dumps(run(args.source, args.root, args.jobs, args.train_only), indent=2))


if __name__ == "__main__":
    main()
