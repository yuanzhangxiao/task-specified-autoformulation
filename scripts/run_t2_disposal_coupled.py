"""Prospective matched fits of Astra's delayed glucose balance and immediate parent."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from importlib.metadata import version
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import fit_positive_insulin_handoff as orion
from scripts import fit_t2_disposal_constrained as optimizer
from scripts import positive_insulin_refit_inputs as insulin
from scripts import qualify_t2_disposal_revision as comparisons
from scripts import refit_t2_disposal_revision as prior
from scripts import report_t2_disposal_coupled as reporting
from scripts import t2_disposal_coupled_replay as data
from scripts.construction_functions import file_hash
from scripts.continue_t2_disposal_refits import interpolate

ARCHIVE = Path("output/t2-disposal-results-20261123/orion-disposal-handoff.zip")
OUT = Path("output/t2-orion-disposal-coupled-20261123-v1")
INPUTS = orion.BASE
SOURCES = {
    "orion_parent": "015-parent",
    "immediate_start": "015-add-tissue-start",
    "feasible_reference": "015-add-tissue-feasible",
    "delayed_start": "015-delayed-meal-start",
}
ARMS = ("immediate", "delayed")
BACKOFF = (1.0, 0.999, 0.99, 0.9, 0.5)
PROTOCOL = "t2-orion-disposal-coupled-1"


def plan(archive: Path, training: Path, budget: int) -> dict:
    """Fix the comparison, numerical budget and training-only retention rule."""
    if budget < 1:
        raise ValueError("Positive per-start evaluation budget required")
    code = [
        Path(__file__),
        Path(optimizer.__file__),
        Path(data.__file__),
        Path(orion.kernel.__file__),
        Path(insulin.__file__),
        Path(insulin.original.__file__),
    ]
    return {
        "protocol": PROTOCOL,
        "archive": file_hash(archive),
        "training": file_hash(training),
        "code": {p.name: file_hash(p) for p in code},
        "dependencies": {k: version(k) for k in ("casadi", "numpy", "scipy")},
        "arms": list(ARMS),
        "sources": SOURCES,
        "starts": list(orion.STARTS),
        "stage": "insulin and upstream production knots fixed",
        "budget_per_arm_per_start": budget,
        "fractions_if_endpoint_unqualified": list(BACKOFF),
        "selection": "Per start, try the endpoint, then back off only if needed "
        "toward the supplied model until full training replay qualifies. "
        "Use qualified pooled training NMSE to compare both attempts to the incumbent "
        "with a 1e-10 tie tolerance.",
    }


def cached_replay(
    root: Path, identity: str, split: str, model: dict, rows: list
) -> dict:
    """Cache by frozen inputs, complete equations, parameter vector and split."""
    key = content_hash([identity, split, model])
    path = root / "replays" / split / f"{key}.json"
    if not path.exists():
        sealed_write(path, {"key": key, "result": data.replay(model, rows)})
    result = sealed_read(path)
    if result["key"] != key:
        raise ValueError("Replay cache differs from its original inputs")
    return result["result"]


def training_record(result: dict, params: dict) -> dict:
    """Keep scores and admissibility distinct, including for unsuccessful fits."""
    return {
        "parameters": params,
        "training_nmse": result["summary"]["pooled_nmse"],
        "domain": result["domain"],
    }


def checked_attempt(model, source, rows, replay, fitted) -> dict:
    """Independently verify an optimizer endpoint and any needed feasibility backoff."""
    attempts = []
    p0 = model["formula"]["parameters"]
    for fraction in BACKOFF:
        p = interpolate(p0, fitted["parameters"], fraction)
        candidate = insulin.with_parameters(model, p)
        if candidate["scalar_model"] != model["scalar_model"]:
            raise ValueError("A fitted vector changed the frozen insulin law")
        result = replay(candidate, rows)
        record = training_record(result, p)
        attempts.append(
            {
                "fraction": fraction,
                "domain": record["domain"],
                "training_nmse": record["training_nmse"],
            }
        )
        if record["domain"]["eligible"]:
            orion.qualified_score(result)
            comparisons.paired(source["events"], result["events"])
            break
    return record | {
        "fraction": fraction,
        "checks": attempts,
        "diagnostics": fitted["diagnostics"],
        "optimizer_training_nmse": fitted["training_nmse"],
    }


def fit_arm(arm: str, *, models: dict, rows: list, root: Path, design: dict) -> tuple:
    """Resume two matched starts and select solely by qualified training rollouts."""
    identity = design["artifact_sha256"]
    model = models[f"{arm}_start"]
    replay = partial(cached_replay, root, identity, "train")
    source = replay(model, rows)
    attempts = {"incumbent": training_record(source, model["formula"]["parameters"])}
    for start in orion.STARTS:
        key = content_hash([identity, arm, start, model])
        path = root / "attempts" / arm / f"{start}.json"
        if not path.exists():
            try:
                fitted = optimizer.fit(
                    model, rows, start, budget=design["budget_per_arm_per_start"]
                )
            except ArithmeticError as exc:
                result = {
                    "parameters": None,
                    "training_nmse": None,
                    "domain": {"eligible": False, "reasons": [str(exc)]},
                    "diagnostics": {"status": "numerically_unavailable"},
                }
            else:
                result = checked_attempt(model, source, rows, replay, fitted)
            sealed_write(path, {"identity": key, "result": result})
        attempts[start] = orion.checked_record(path, key)
        print(arm, start, attempts[start]["training_nmse"], flush=True)
    best = prior.choose(attempts)
    return arm, {"winner": best, "attempts": attempts}


def publish_scores(
    root: Path,
    frozen: dict,
    models: dict,
    rows: list,
    split: str,
) -> dict:
    """Score the same frozen fitted models and the supplied references on a split."""
    scored = {
        label: cached_replay(root, frozen["plan"], split, model, rows)
        for label, model in models.items()
    }
    report = reporting.summarize(scored, split)
    return sealed_write(
        root / f"{split}-report.json",
        {
            "freeze": frozen["artifact_sha256"],
            "report": report,
            "report_code": file_hash(Path(reporting.__file__)),
            "domains": {label: r["domain"] for label, r in scored.items()},
        },
    )


def run(archive: Path, root: Path, *, train_only: bool, jobs: int, budget: int) -> dict:
    """Authenticate all sources, run the paired fit and freeze before development."""
    if jobs not in (1, 2):
        raise ValueError("Use one or two CPU workers")
    with _lock(root):
        packed = data.load_models(archive)
        models = {label: packed[tag] for label, tag in SOURCES.items()}
        training = INPUTS / "train.json"
        rows = data.load_training(training)
        design = sealed_write(root / "plan.json", plan(archive, training, budget))
        identity = design["artifact_sha256"]
        checks = {}
        for label, tag in SOURCES.items():
            result = cached_replay(root, identity, "train", models[label], rows)
            checks[label] = data.verify_start(archive, tag, models[label], result)
            if (
                label in ("delayed_start", "immediate_start")
                and not result["domain"]["eligible"]
            ):
                raise ValueError(
                    f"Matched starting vector failed training domain: {label}"
                )
        verified = sealed_write(
            root / "source-checks.json", {"plan": identity, "checks": checks}
        )
        worker = partial(fit_arm, models=models, rows=rows, root=root, design=design)
        if jobs == 1:
            fits = dict(worker(a) for a in ARMS)
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                fits = dict(pool.map(worker, ARMS))
        for arm, fit in fits.items():
            selected = fit["attempts"][fit["winner"]]
            models[f"{arm}_refit"] = insulin.with_parameters(
                models[f"{arm}_start"], selected["parameters"]
            )
        frozen = sealed_write(
            root / "training-freeze.json",
            {
                "plan": identity,
                "sources": verified["artifact_sha256"],
                "fits": fits,
                "models": models,
                "validation_opened_at_freeze": False,
            },
        )
        publish_scores(root, frozen, models, rows, "train")
        if not train_only:
            frozen = sealed_read(root / "training-freeze.json")
            if (
                sealed_read(root / "train-report.json")["freeze"]
                != frozen["artifact_sha256"]
            ):
                raise ValueError(
                    "A verified training report is required before development"
                )
            development = data.load_validation(INPUTS / "validation.json")
            publish_scores(root, frozen, frozen["models"], development, "val")
        return {
            "output": str(root),
            "training_freeze": frozen["artifact_sha256"],
            "selection": {k: v["winner"] for k, v in fits.items()},
            "splits": ["train"] if train_only else ["train", "val"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--root", type=Path, default=OUT)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.archive,
                args.root,
                train_only=args.train_only,
                jobs=args.jobs,
                budget=args.budget,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
