"""Matched downstream refits of the frozen slot-015 disposal revision and parent."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import fit_positive_insulin_handoff as orion
from scripts import positive_insulin_refit_inputs as intake
from scripts import qualify_t2_disposal_revision as qualify
from scripts.construction_functions import file_hash

SOURCE = Path("output/t2-disposal-nested-qualification-20261123-v1")
FREEZE = "d015d075923383016ff3ec716435d22fc255ee3ac639804ffb261170feb3f183"
OUT = Path("output/t2-disposal-matched-20261123-v1")
ARMS = ("parent", "selected")
STAGE = "fixed_insulin"
BUDGET = 80


def domain(replay: dict, rows: list, params: dict) -> dict:
    """Test the two directed action gains and the sampled concentration domains."""
    failures = []
    score = replay["summary"]["pooled_nmse"]
    if score is None or replay["summary"]["failed_events"] or not np.isfinite(score):
        return {"eligible": False, "reasons": ["incomplete training rollout"]}
    if not (params["afcoef10"] > 0 and params["afcoef13"] > 0):
        failures.append("nonpositive insulin-to-action or action-to-disposal gain")
    if [e["event"] for e in replay["events"]] != [r["trajectory_id"] for r in rows]:
        raise ValueError("Training event IDs differ")
    minimum = min(
        qualify.disposal_contribution(e, row)["min_U_minus_Uii"]
        for e, row in zip(replay["events"], rows, strict=True)
    )
    if minimum < -1e-8:
        failures.append("negative insulin-dependent disposal")
    if any(any(e["negative_samples"].values()) for e in replay["events"]):
        failures.append("negative predicted target")
    return {
        "eligible": not failures,
        "reasons": failures,
        "minimum_U_minus_Uii": minimum,
        "d_U_d_action": params["afcoef13"],
        "d_action_rate_d_I": params["afcoef10"] / 111.507257324,
    }


def choose(attempts: dict) -> str:
    """Prefer the incumbent on numerical ties in qualified training NMSE."""
    if not attempts["incumbent"]["domain"]["eligible"]:
        raise ValueError("Ineligible source incumbent")
    best = "incumbent"
    for name in orion.STARTS:
        a = attempts[name]
        if (
            a["domain"]["eligible"]
            and a["training_nmse"] < attempts[best]["training_nmse"] - 1e-10
        ):
            best = name
    return best


def fit_arm(
    arm: str, *, model: dict, archived: dict, rows: list, root: Path, identity: str
) -> tuple[str, dict]:
    """Qualify the incumbent and both endpoint searches against all training rows."""
    problem = orion.kernel.FitProblem(model, rows)
    p = model["formula"]["parameters"]
    replay = partial(orion.cache_replay, root=root, identity=identity, split="train")
    source = replay(model, rows)
    if not np.isclose(
        orion.qualified_score(source),
        archived["summary"]["pooled_nmse"],
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError("Source NMSE differs from the nested qualification")
    for fresh, old in zip(source["events"], archived["events"], strict=True):
        if fresh["event"] != old["event"] or not all(
            np.allclose(
                fresh["predicted"][k], old["predicted"][k], atol=1e-8, rtol=1e-8
            )
            for k in qualify.TARGETS
        ):
            raise ValueError("Source training predictions differ")
    attempts = {
        "incumbent": {
            "training_nmse": orion.qualified_score(source),
            "parameters": p,
            "domain": domain(source, rows, p),
        }
    }
    for name in orion.STARTS:
        start = orion.optimizer_start(model, p, STAGE, name)
        path = root / "attempts" / arm / f"{name}.json"
        key = content_hash([identity, arm, name, start])
        if not path.exists():
            result = orion.fit_attempt(
                problem, STAGE, start, BUDGET, model, rows, replay
            )
            param = result["parameters"]
            assessment = {
                "eligible": False,
                "reasons": ["unavailable optimizer endpoint"],
            }
            if param is not None:
                if any(param[k] != p[k] for k in problem.insulin_parameters):
                    raise ArithmeticError("Optimizer moved a fixed insulin coefficient")
                final = replay(intake.with_parameters(model, param), rows)
                assessment = domain(final, rows, param)
                if assessment["eligible"]:
                    qualify.paired(source["events"], final["events"])
            sealed_write(
                path, {"identity": key, "result": result | {"domain": assessment}}
            )
        attempts[name] = orion.checked_record(path, key)
    return arm, {"winner": choose(attempts), "attempts": attempts}


def run(source: Path, root: Path, train_only: bool, jobs: int) -> dict:
    """Train and freeze both endpoints before development replay."""
    if jobs not in {1, 2}:
        raise ValueError("Choose one or two fitting workers")
    with _lock(root):
        handoff = sealed_read(source / "training-freeze.json")
        screen = sealed_read(source / "train-report.json")
        if (
            handoff["artifact_sha256"] != FREEZE
            or handoff["training_report"] != screen["artifact_sha256"]
            or handoff["validation_opened_at_freeze"] is not False
            or handoff["candidate_names"]["selected"] != "add_tissue"
        ):
            raise ValueError("Unauthenticated disposal handoff")
        models = {k: handoff["models"][k] for k in ARMS}
        qualify.check_patch(models["parent"], models["selected"])
        files = [
            Path(__file__),
            Path(qualify.__file__),
            Path(orion.__file__),
            Path(orion.kernel.__file__),
            Path(intake.__file__),
            Path(intake.original.__file__),
        ]
        plan = sealed_write(
            root / "plan.json",
            {
                "source": FREEZE,
                "training": intake.original.campaign.TRAIN,
                "validation": intake.original.campaign.VALIDATION,
                "starts": list(orion.STARTS),
                "model_sha256": {k: content_hash(m) for k, m in models.items()},
                "stage": STAGE,
                "max_nfev": BUDGET,
                "selection": (
                    "Lowest training NMSE among qualified endpoints; "
                    "incumbent retained within 1e-10."
                ),
                "domain": (
                    "Positive action-path gains; sampled concentrations >= -1e-10 "
                    "and U-Uii >= -1e-8."
                ),
                "code": {f.name: file_hash(f) for f in files},
            },
        )
        identity = plan["artifact_sha256"]
        training = intake.load_training(orion.BASE / "train.json")
        common = {"rows": training, "root": root, "identity": identity}
        tasks = [(arm, models[arm], screen["models"][arm]) for arm in ARMS]
        if jobs == 1:
            fits = dict(fit_arm(a, model=m, archived=s, **common) for a, m, s in tasks)
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = [
                    pool.submit(fit_arm, a, model=m, archived=s, **common)
                    for a, m, s in tasks
                ]
                fits = dict(f.result() for f in futures)
        final = {
            k: intake.with_parameters(
                models[k], a["attempts"][a["winner"]]["parameters"]
            )
            for k, a in fits.items()
        }
        # Keep the Orion parent as the independent replay reference.
        compare = {
            "parent": models["parent"],
            "child_start": models["selected"],
            "parent_refit": final["parent"],
            "child_refit": final["selected"],
        }
        freeze = sealed_write(
            root / "training-freeze.json",
            {
                "plan": identity,
                "fits": fits,
                "models": final,
                "validation_opened_at_freeze": False,
            },
        )
        scores = {}
        for split in ("train",) if train_only else ("train", "val"):
            if split == "train":
                rows = training
            else:
                check = sealed_read(root / "train-report.json")
                if check["freeze"] != freeze["artifact_sha256"]:
                    raise ValueError(
                        "Development requires both qualified training endpoints"
                    )
                params = {
                    "training_intake": plan["training"],
                    "validation_intake": plan["validation"],
                }
                rows = [
                    r.model_dump(mode="json")
                    for r in intake.original.read_validation(
                        orion.BASE / "validation.json", params
                    )
                ]
            reference = sealed_read(orion.OUT / f"{split}-report.json")
            report = qualify.evaluate(compare, rows, root, identity, split, reference)
            scores_by_arm = report["models"]
            report["models"] = {
                "parent_start": scores_by_arm["parent"],
                "child_start": scores_by_arm["child_start"],
                "parent": scores_by_arm["parent_refit"],
                "selected": scores_by_arm["child_refit"],
            }
            pairs = report["models"]
            report["paired"] = {
                key: qualify.paired(pairs[a]["events"], pairs[b]["events"])
                for key, a, b in [
                    ("selected", "parent", "selected"),
                    ("parent_refit", "parent_start", "parent"),
                    ("child_refit", "child_start", "selected"),
                ]
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--root", type=Path, default=OUT)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.root, args.train_only, args.jobs), indent=2))


if __name__ == "__main__":
    main()
