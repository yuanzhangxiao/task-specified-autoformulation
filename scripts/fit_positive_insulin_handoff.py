"""Matched T2 refits from the four authenticated Orion starting vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import positive_insulin_refit_inputs as intake
from scripts import positive_insulin_refit_kernel as kernel
from scripts import positive_insulin_refit_report as reporting
from scripts.construction_functions import file_hash

PROTOCOL = "t2-orion-positive-insulin-matched-1"
STAGES = ("fixed_insulin", "joint")
STARTS = ("warm", "displaced")
BASE = Path("output/trajectory-functions-20261123/inputs")
ZIP = Path("output/basin-autopilot-20261123/positive-validation/orion-fit-starts.zip")
OUT = Path("output/t2-orion-positive-refits-20261123-v1")


def design(archive: Path, training: Path, budget: int) -> dict:
    """Identify the fitting code, training data and predetermined optimizer design."""
    if budget < 1:
        raise ValueError("Each optimization requires at least one evaluation")
    sources = [Path(__file__), Path(kernel.__file__), Path(intake.__file__)]
    return {
        "protocol": PROTOCOL,
        "archive": file_hash(archive),
        "training": file_hash(training),
        "code": {p.name: file_hash(p) for p in sources},
        "stages": list(STAGES),
        "starts": list(STARTS),
        "max_nfev_per_start": budget,
        "selection": "lowest qualified original-equation training pooled NMSE, "
        "retaining the incumbent on a tie",
        "displacement": "active named coefficients multiplied by 0.8 or 1.2 "
        "from the low bit of SHA-256(name); exact zeros retained",
    }


def optimizer_start(
    model: dict, incumbent: dict, stage: str, start: str
) -> dict[str, float]:
    """Move the same named coordinates by the same factor in a matched pair."""
    if start == "warm":
        return dict(incumbent)
    if start != "displaced" or stage not in STAGES:
        raise ValueError("Unknown optimizer stage or start")
    insulin_names = (
        RestrictedParser()
        .parse(model["formula"]["equations"]["I"], location="insulin law")
        .symbols
    )
    out = dict(incumbent)
    for name, value in incumbent.items():
        if stage == "joint" or name not in insulin_names:
            parity = hashlib.sha256(name.encode()).digest()[0] & 1
            out[name] = value * (1.2 if parity else 0.8)
    return out


def cache_replay(
    model: dict, rows: list[dict], *, root: Path, identity: str, split: str
) -> dict:
    """Replay original equations once per complete model/vector and input split."""
    key = content_hash({"model": model, "split": split, "plan": identity})
    path = root / "replays" / split / f"{key}.json"
    if not path.exists():
        results = intake.replay(model, rows)
        sealed_write(path, {"key": key, "results": results})
    record = sealed_read(path)
    if record["key"] != key:
        raise ValueError("Replay differs from the registered model or split")
    return record["results"]


def qualified_score(replay: dict) -> float:
    """Only a complete set of qualified event rollouts can enter selection."""
    summary = replay["summary"]
    value = summary.get("pooled_nmse")
    if summary["failed_events"] or value is None or not math.isfinite(value):
        raise ArithmeticError("Incomplete or nonfinite original-equation replay")
    return float(value)


def checked_record(path: Path, identity: str) -> dict:
    """Read a completed stage tied to its exact inputs."""
    record = sealed_read(path)
    if record["identity"] != identity:
        raise ValueError(f"Checkpoint belongs to a different fit: {path}")
    return record["result"]


def fit_attempt(problem, stage, initial, budget, model, rows, replay) -> dict:
    """A failed numerical attempt leaves the previous qualified model eligible."""
    try:
        fitted = problem.fit_stage(stage, initial, max_nfev=budget)
    except ArithmeticError as exc:
        return {
            "parameters": None,
            "training_nmse": None,
            "optimizer_training_nmse": None,
            "diagnostics": {"status": "numerically_unavailable", "reason": str(exc)},
        }
    stage_model = intake.with_parameters(model, fitted["parameters"])
    verified = replay(stage_model, rows)
    try:
        score = qualified_score(verified)
    except ArithmeticError:
        score = None
    return {
        **fitted,
        "optimizer_training_nmse": fitted["training_nmse"],
        "training_nmse": score,
    }


def train_one(
    tag: str,
    *,
    archive: Path,
    training: Path,
    root: Path,
    plan: dict,
    starts_only: bool,
) -> dict:
    """Reproduce the warm start and fit both stages under a per-model lock."""
    model = intake.load_models(archive)[tag]
    rows = intake.load_training(training)
    identity = content_hash([plan["artifact_sha256"], tag])
    replay = partial(
        cache_replay, root=root, identity=plan["artifact_sha256"], split="train"
    )
    folder = root / "fits" / tag
    with _lock(folder):
        start_path = folder / "start.json"
        if not start_path.exists():
            result = replay(model, rows)
            score = qualified_score(result)
            diagnostics = intake.verify_start(tag, model, result)
            sealed_write(
                start_path,
                {
                    "identity": identity,
                    "result": {
                        "parameters": model["formula"]["parameters"],
                        "training_nmse": score,
                        "diagnostics": {"source_verification": diagnostics},
                    },
                },
            )
        stages = {"start": checked_record(start_path, identity)}
        if starts_only:
            return stages
        problem = kernel.FitProblem(model, rows)
        incumbent = stages["start"]
        for stage in STAGES:
            best = incumbent
            selected = "incumbent"
            attempts = {}
            for label in STARTS:
                path = folder / f"{stage}-{label}.json"
                initial = optimizer_start(model, incumbent["parameters"], stage, label)
                attempt_id = content_hash([identity, stage, label, initial])
                if not path.exists():
                    candidate = fit_attempt(
                        problem,
                        stage,
                        initial,
                        plan["max_nfev_per_start"],
                        model,
                        rows,
                        replay,
                    )
                    sealed_write(path, {"identity": attempt_id, "result": candidate})
                candidate = checked_record(path, attempt_id)
                value = candidate["training_nmse"]
                attempts[label] = {
                    "training_nmse": value,
                    "optimizer_training_nmse": candidate["optimizer_training_nmse"],
                    "solver": candidate["diagnostics"],
                }
                if value is not None and value < best["training_nmse"]:
                    best, selected = candidate, label
            incumbent = {
                "parameters": best["parameters"],
                "training_nmse": best["training_nmse"],
                "diagnostics": {"selected": selected, "attempts": attempts},
            }
            stages[stage] = incumbent
            print(
                f"{tag} {stage} {selected} {incumbent['training_nmse']:.9g}", flush=True
            )
        return stages


def collect(fn, jobs: int) -> dict:
    """Collect every prespecified model before opening the next phase."""
    if jobs == 1:
        return {tag: fn(tag) for tag in intake.TAGS}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return dict(zip(intake.TAGS, pool.map(fn, intake.TAGS), strict=True))


def run(
    archive: Path,
    training: Path,
    validation: Path,
    root: Path,
    budget: int,
    jobs: int,
    train_only: bool = False,
) -> dict:
    """Finish all source checks and training selections before development replay."""
    if jobs < 1:
        raise ValueError("jobs must be positive")
    with _lock(root):
        plan = sealed_write(root / "plan.json", design(archive, training, budget))
        identity = plan["artifact_sha256"]
        worker = partial(
            train_one, archive=archive, training=training, root=root, plan=plan
        )
        collect(partial(worker, starts_only=True), jobs)
        stages = collect(partial(worker, starts_only=False), jobs)
        seal_path = root / "training-freeze.json"
        sealed_write(
            seal_path, {"protocol": PROTOCOL, "identity": identity, "models": stages}
        )
        seal = sealed_read(seal_path)
        models = intake.load_models(archive)
        exports = {
            tag: {
                stage: intake.with_parameters(
                    model, seal["models"][tag][stage]["parameters"]
                )
                for stage in STAGES
            }
            for tag, model in models.items()
        }
        model_path = root / "fitted-models.json"
        sealed_write(model_path, {"freeze": seal["artifact_sha256"], "models": exports})
        splits = ["train"] if train_only else ["train", "val"]
        paths = {}
        for split in splits:
            path = root / f"{split}-report.json"
            if split == "train":
                rows = intake.load_training(training)
            else:
                joint = {tag: exported["joint"] for tag, exported in exports.items()}
                rows = intake.load_validation(validation, joint)
            replay = partial(cache_replay, root=root, identity=identity, split=split)
            results = reporting.report(seal, models, rows, replay, split)
            sealed_write(
                path,
                {
                    "freeze": seal["artifact_sha256"],
                    "report_code_sha256": file_hash(Path(reporting.__file__)),
                    "report": results,
                },
            )
            paths[split] = str(path)
        return {
            "training_freeze": str(seal_path),
            "models": str(model_path),
            "reports": paths,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ZIP)
    parser.add_argument("--train", type=Path, default=BASE / "train.json")
    parser.add_argument("--validation", type=Path, default=BASE / "validation.json")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.archive,
                args.train,
                args.validation,
                args.output,
                args.max_nfev,
                args.jobs,
                args.train_only,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
