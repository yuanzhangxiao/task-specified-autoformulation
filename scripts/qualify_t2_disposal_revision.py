"""Replay the training-selected disposal revision in the complete T2-easy model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import fit_positive_insulin_handoff as orion
from scripts import positive_insulin_refit_inputs as intake
from scripts.construction_functions import file_hash

OUT = Path("output/t2-disposal-qualification-20261123-v1")
SELECTION = Path("output/t2-disposal-screen-20261123-v1/training-freeze.json")
TAG = "015-positive_EGP"
PARENT = "0a350191b9b26e941be3654f8a8bb3c0af0ac34848866ccc36a926e25448d022"
TARGETS = intake.TARGETS


def check_patch(parent: dict, child: dict) -> None:
    """Only the disposal equation, its sources and its own coefficients can vary."""
    g, f = parent["graph"], parent["formula"]
    new_g, new_f = child["graph"], child["formula"]
    desired_g = g | {"sources": g["sources"] | {"U": new_g["sources"]["U"]}}
    desired_f = f | {
        "equations": f["equations"] | {"U": new_f["equations"]["U"]},
        "parameters": new_f["parameters"],
    }
    allowed = {"Gp", "Gt", "Uii", "I_act"}
    if (
        new_g != desired_g
        or new_f != desired_f
        or not set(new_g["sources"]["U"]) <= allowed
        or any(child[k] != parent[k] for k in ["knots", "scalar_model", "scales"])
    ):
        raise ValueError("Changed a protected equation, graph component or insulin law")
    parser = RestrictedParser()
    protected = [v for k, v in f["equations"].items() if k != "U"]
    for expr in [*protected, *f["initializers"].values()]:
        for symbol in parser.parse(expr, location="frozen equation").symbols:
            if (
                symbol in f["parameters"]
                and new_f["parameters"].get(symbol) != f["parameters"][symbol]
            ):
                raise ValueError(f"Shared or protected parameter changed: {symbol}")


def check_screen(result: dict, expected: dict) -> None:
    """Check per-event disposal residuals against the independent training screen."""
    by_event = {e["event"]: e["sse"]["U"] for e in result["events"]}
    if set(by_event) != set(expected) or not all(
        np.isclose(by_event[k], expected[k], rtol=1e-5, atol=1e-7) for k in by_event
    ):
        raise ValueError("Free-rollout U errors differ from the screening predictions")


def disposal_contribution(event: dict, row: dict) -> dict:
    """Measure the inferred insulin-dependent disposal U minus the supplied Uii."""
    total = np.asarray(event["predicted"]["U"])
    independent = np.asarray(row["auxiliaries"]["Uii"])
    if total.shape != independent.shape or not np.isfinite(independent).all():
        raise ValueError("Unmatched supplied disposal series")
    delta = total - independent
    return {
        "min_U_minus_Uii": float(delta.min()),
        "negative_U_minus_Uii_samples": int(np.sum(delta < -1e-8)),
    }


def paired(before: list[dict], after: list[dict]) -> dict:
    """Ignore solver roundoff when counting improvements in a frozen target."""
    if [e["event"] for e in before] != [e["event"] for e in after] or not before:
        raise ValueError("Comparison requires matched, complete events")
    metrics = (*TARGETS, "pooled")
    counts = {k: {"improved": 0, "tied": 0, "worse": 0} for k in metrics}
    max_diff = 0.0
    for base, candidate in zip(before, after, strict=True):
        if any(e["status"] != "scored" for e in [base, candidate]):
            raise ArithmeticError("Unqualified rollout")
        for k in metrics:
            a, b = [
                e["pooled_nmse"] if k == "pooled" else e["per_target_nmse"][k]
                for e in [base, candidate]
            ]
            tie = np.isclose(a, b, atol=1e-8, rtol=1e-7)
            counts[k]["tied" if tie else "improved" if b < a else "worse"] += 1
        p, q = [np.asarray(e["predicted"]["I"]) for e in (base, candidate)]
        if not np.allclose(p, q, atol=2e-5, rtol=2e-7):
            raise ArithmeticError("Frozen insulin trajectories changed")
        max_diff = max(max_diff, float(np.max(np.abs(p - q))))
    return {"events": len(before), "counts": counts, "max_insulin_difference": max_diff}


def evaluate(
    models: dict, rows: list, root: Path, plan: str, split: str, archive: dict
) -> dict:
    """Independently integrate, verify the parent, and quantify coupled effects."""
    results = {}
    for name, model in models.items():
        result = orion.cache_replay(model, rows, root=root, identity=plan, split=split)
        orion.qualified_score(result)
        events = result["events"]
        if [e["event"] for e in events] != [r["trajectory_id"] for r in rows]:
            raise ValueError("Mismatched trajectory cohort")
        events = [
            e | disposal_contribution(e, row)
            for e, row in zip(events, rows, strict=True)
        ]
        summary = result["summary"] | {
            "negative_samples": {
                k: sum(e["negative_samples"][k] for e in events) for k in TARGETS
            },
            "minima": {k: min(e["minima"][k] for e in events) for k in TARGETS},
            "min_U_minus_Uii": min(e["min_U_minus_Uii"] for e in events),
            "negative_U_minus_Uii_samples": sum(
                e["negative_U_minus_Uii_samples"] for e in events
            ),
        }
        results[name] = result | {"summary": summary, "events": events}
    base = results["parent"]
    original = archive["report"]["models"][TAG]["joint"]
    if [e["event"] for e in base["events"]] != [e["event"] for e in original["events"]]:
        raise ValueError("Archived parent event IDs differ")
    for a, b in zip(base["events"], original["events"], strict=True):
        if not all(
            np.allclose(a["predicted"][k], b["predicted"][k], rtol=1e-8, atol=1e-8)
            for k in TARGETS
        ) or not np.isclose(a["pooled_nmse"], b["pooled_nmse"], rtol=1e-8, atol=1e-8):
            raise ValueError("Parent trajectories or scores differ from Orion")
    return {
        "split": split,
        "models": results,
        "paired": {
            name: paired(base["events"], row["events"])
            for name, row in results.items()
            if name != "parent"
        },
        "scales": models["parent"]["scales"],
        "source_report_sha256": archive["artifact_sha256"],
    }


def run(
    selection: Path,
    source: Path,
    data: Path,
    root: Path,
    *,
    train_only: bool = False,
) -> dict:
    """Replay and freeze training before opening the development split."""
    with _lock(root):
        screen = sealed_read(selection)
        fitted = sealed_read(source / "fitted-models.json")
        parent = fitted["models"][TAG]["joint"]
        if (
            content_hash(parent) != PARENT
            or screen["parent_model_sha256"] != PARENT
            or screen["training_sha256"] != intake.original.campaign.TRAIN
            or screen["validation_opened"] is not False
            or screen["test_opened"] is not False
        ):
            raise ValueError("Different frozen parent or training intake")
        names = {"selected": screen["selected"]}
        if screen.get("control") not in {None, screen["selected"]}:
            names["control"] = screen["control"]
        models = {"parent": parent}
        for key, name in names.items():
            model = screen["candidates"][name]["model"]
            check_patch(parent, model)
            models[key] = model
        plan = sealed_write(
            root / "plan.json",
            {
                "selection": screen["artifact_sha256"],
                "model_export": fitted["artifact_sha256"],
                "model_sha256": {k: content_hash(v) for k, v in models.items()},
                "candidate_names": names,
                "training": screen["training_sha256"],
                "validation": intake.original.campaign.VALIDATION,
                "code": {
                    p.name: file_hash(p)
                    for p in [
                        Path(__file__),
                        Path(intake.__file__),
                        Path(intake.original.__file__),
                    ]
                },
            },
        )["artifact_sha256"]
        splits = ["train"] if train_only else ["train", "val"]
        scores = {}
        training_sha = None
        for split in splits:
            if split == "train":
                rows = intake.load_training(data / "train.json")
            else:
                freeze = sealed_read(root / "training-freeze.json")
                if freeze["plan"] != plan or freeze["training_report"] != training_sha:
                    raise ValueError("Training qualification differs from freeze")
                val_rows = intake.original.read_validation(
                    data / "validation.json",
                    {
                        "training_intake": screen["training_sha256"],
                        "validation_intake": intake.original.campaign.VALIDATION,
                    },
                )
                rows = [r.model_dump(mode="json") for r in val_rows]
            archived = sealed_read(source / f"{split}-report.json")
            if archived["freeze"] != fitted["freeze"]:
                raise ValueError(
                    "Reference report differs from the fitted-model freeze"
                )
            report = evaluate(models, rows, root, plan, split, archived)
            if split == "train":
                for key, name in names.items():
                    check_screen(
                        report["models"][key],
                        screen["candidates"][name]["training"]["event_U_sse"],
                    )
                checked = report["models"]["selected"]["summary"]
                if checked["negative_U_minus_Uii_samples"] or any(
                    checked["negative_samples"].values()
                ):
                    raise ValueError(
                        "Training rollout violates the concentration or disposal domain"
                    )
            record = sealed_write(
                root / f"{split}-report.json", report | {"plan": plan}
            )
            if split == "train":
                training_sha = record["artifact_sha256"]
                sealed_write(
                    root / "training-freeze.json",
                    {
                        "plan": plan,
                        "training_report": training_sha,
                        "models": models,
                        "candidate_names": names,
                        "validation_opened_at_freeze": False,
                    },
                )
            scores[split] = {
                k: v["summary"]["pooled_nmse"] for k, v in record["models"].items()
            }
        return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--source", type=Path, default=orion.OUT)
    parser.add_argument("--data", type=Path, default=orion.BASE)
    parser.add_argument("--root", type=Path, default=OUT)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.selection,
                args.source,
                args.data,
                args.root,
                train_only=args.train_only,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
