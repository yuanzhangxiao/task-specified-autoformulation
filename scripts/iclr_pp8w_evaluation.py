"""Score the complete frozen Delta follow-up on the existing held-out panels."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import iclr_anytime_response as scores
from scripts import iclr_cstr_observation_panel as cstr
from scripts import iclr_scaling_response as metrics
from scripts import iclr_stress_evaluation as stress


def freeze(models: Path, design: Path, panels: Path, root: Path) -> dict:
    """Require all 18 frozen endpoints before opening any held-out references."""
    bundle, manifest = sealed_read(models), sealed_read(design / "manifest.json")
    expected = {(r["benchmark_id"], r["repetition"]) for r in manifest["runs"]}
    actual = {(r["benchmark_id"], r["repetition"]) for r in bundle["rows"]}
    if (
        bundle["protocol"] != "iclr-controlled-search-models-1"
        or not bundle.get("selection_freeze_sha256")
        or len(bundle["rows"]) != 18
        or len(expected) != 18
        or actual != expected
        or any(
            r["method"] != "full" or r["status"] not in {"ready", "no_retained_model"}
            for r in bundle["rows"]
        )
    ):
        raise ValueError("complete frozen eighteen-model Full roster required")
    cases = {c["cell"]: c for c in manifest["cells"]}
    rows = []
    for source in bundle["rows"]:
        cell = cases[source["benchmark_id"]]
        if source["public_file_sha256"] != cell["public_files"]:
            raise ValueError("frozen model training/validation inputs differ")
        assays = ["matched_acquisition"]
        if cell["cell"] == "cstr_noise_10pct":
            assays.append("clean_initial_diagnostic")
        for assay in assays:
            acquired = dict(cell)
            if assay == "clean_initial_diagnostic":
                acquired["noise"] = 0
            endpoint = None
            if source["status"] == "ready":
                endpoint = {
                    k: source[k]
                    for k in ("candidate", "context", "parameters", "initials")
                } | {
                    "benchmark_id": cell["source"],
                    "source_protocol": bundle["protocol"],
                    "direct_readouts": {},
                    "status": "ready",
                }
            key = f"{cell['cell']}_s{source['repetition']}_{assay}"
            rows.append(
                {
                    "key": key,
                    "condition": cell["cell"],
                    "seed": source["repetition"],
                    "assay": assay,
                    "family": cell["family"],
                    "acquisition": acquired,
                    "endpoint": endpoint,
                    "model_id": content_hash([endpoint, acquired, assay]),
                }
            )
    panel = sealed_read(panels)
    return sealed_write(
        root / "plan.json",
        {
            "protocol": "iclr-pp8w-heldout-1",
            "models": bundle["artifact_sha256"],
            "design": manifest["artifact_sha256"],
            "panels": panel["artifact_sha256"],
            "rows": rows,
            "references": {k: panel["references"][k] for k in ("cstr", "t2")},
            "training_sd": panel["training_sd"],
            "cstr_schedule": panel["cstr_schedule"],
            "source": cstr.source_identity(),
            "implementation": metrics.digest(Path(__file__)),
            "metric_implementation": metrics.digest(Path(scores.__file__)),
            "initialization": (
                "Primary: existing stress acquisition, with noisy initial targets; "
                "same noise draw for a paired contrast/control. Secondary noisy-CSTR "
                "diagnostic: same fitted model with clean initial observations."
            ),
            "test_used_for_selection": False,
        },
    )


def cstr_input(context: dict, record: dict, cell: dict, case: str):
    """Expose only acquired initial targets and declared inputs/auxiliaries."""
    public = {
        "trajectory_id": case,
        "time": [float(t) for t in record["time"]],
        **{
            role: {name: record[name] for name in context[role]}
            for role in ("targets", "auxiliaries", "external_inputs")
        },
    }
    return stress.public_input(public, cell)


def cstr_predictions(root: Path, plan: dict, row: dict) -> dict:
    """Reuse declared-mechanism scoring, preserving paired acquisition and gates."""
    anchor, declared, _ = cstr.native()
    endpoint = row["endpoint"]
    view = declared.view_of(endpoint)
    attribution = cstr.attribute_observed(view)
    results = []
    for case, schedule in plan["cstr_schedule"].items():
        if schedule["role"] != "contrast":
            continue
        references, predictions = {}, {}
        for side, name in (("perturbed", case), ("control", schedule["control"])):
            reference = scores.read_reference(plan["references"]["cstr"][name])[
                "reference"
            ]
            references[side] = reference
            path = root / "predictions" / row["key"] / f"{case}_{side}.json"
            if not path.exists():
                try:
                    trajectory = cstr_input(
                        endpoint["context"], reference, row["acquisition"], case
                    )
                    curves = declared.curves(view, attribution, trajectory)
                    alternate = anchor.predict(endpoint, trajectory, "BDF")
                    anchor.probe.solver_agreement(
                        {"T": curves["T"]}, {"T": alternate["T"]}, 1e-4
                    )
                    result = {
                        "status": "complete",
                        "curves": curves,
                        "alternate_temperature": alternate["T"],
                    }
                except (
                    ValueError,
                    ArithmeticError,
                    RuntimeError,
                    TimeoutError,
                ) as error:
                    result = {
                        "status": "rollout_unavailable",
                        "reason": str(error)[:500],
                    }
                sealed_write(path, {"plan": plan["artifact_sha256"], **result})
            predictions[side] = sealed_read(path)
            if predictions[side]["plan"] != plan["artifact_sha256"]:
                raise ValueError("forecast cache identity differs")
        p, q = predictions["perturbed"], predictions["control"]
        r, c = references["perturbed"], references["control"]
        gate = cstr.target_score(
            anchor, p, q, r, c, schedule, plan["training_sd"]["cstr"]["T"]
        )
        if gate["status"] != "complete":
            results.append({"case": case, "status": gate["status"]})
            continue
        truth, base = (
            {"T": r["T"], **declared._truth(r)},
            {"T": c["T"], **declared._truth(c)},
        )
        for quantity in ("T", *declared.MECHANISMS):
            result = scores.continuous_score(
                r["time"],
                truth[quantity],
                base[quantity],
                p["curves"][quantity],
                q["curves"][quantity],
                plan["training_sd"]["cstr"]["T"] if quantity == "T" else None,
                integral=False,
            )
            results.append(
                {"case": case, "quantity": quantity, "status": "complete", **result}
            )
    return {"cases": results, "attribution": attribution, "planned_cases": 24}


def one_subject(args):
    root, row = args
    plan = sealed_read(root / "plan.json")
    path = root / "subjects" / f"{row['key']}.json"
    if not path.exists():
        if row["endpoint"] is None:
            result = {"status": "model_unavailable", "cases": []}
        else:
            evaluator = (
                cstr_predictions if row["family"] == "cstr" else scores.t2_predictions
            )
            result = {"status": "evaluated", **evaluator(root, plan, row)}
        sealed_write(path, {"plan": plan["artifact_sha256"], **result})
    result = sealed_read(path)
    if result["plan"] != plan["artifact_sha256"]:
        raise ValueError("subject identity differs")
    return {"key": row["key"], "status": result["status"]}


def report(root: Path) -> dict:
    plan = sealed_read(root / "plan.json")
    rows, groups = [], []
    for row in plan["rows"]:
        path = root / "subjects" / f"{row['key']}.json"
        value = sealed_read(path) if path.exists() else {"status": "pending"}
        if path.exists() and value["plan"] != plan["artifact_sha256"]:
            raise ValueError("subject belongs to another plan")
        target = {"T"} if row["family"] == "cstr" else {"Gp", "I"}
        pooled = (
            scores.pooled(value["cases"], target, 24 if row["family"] == "cstr" else 6)
            if value["status"] == "evaluated"
            else {"status": value["status"]}
        )
        rows.append(
            {k: row[k] for k in ("condition", "assay", "seed", "family")} | pooled
        )
    for condition, assay in sorted({(r["condition"], r["assay"]) for r in rows}):
        group = [r for r in rows if (r["condition"], r["assay"]) == (condition, assay)]
        complete = [r for r in group if r["status"] == "complete"]
        groups.append(
            {
                "condition": condition,
                "assay": assay,
                "planned": len(group),
                "complete": len(complete),
                "statistics": {
                    field: metrics.interval(
                        [r[field] for r in complete],
                        higher_is_better=field == "directional_accuracy",
                    )
                    if complete
                    else None
                    for field in (
                        "test_nmse",
                        "response_rse",
                        "directional_accuracy",
                        "shape_error",
                    )
                },
            }
        )
    from autoformalism.llm.staged_topology import atomic_json

    value = {"plan": plan["artifact_sha256"], "rows": rows, "groups": groups}
    atomic_json(root / "summary.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--panels", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "freeze":
        if not all((args.models, args.design, args.panels)):
            parser.error("freeze requires --models, --design and --panels")
        result = freeze(args.models, args.design, args.panels, args.root)
        print(json.dumps({"endpoints": len(result["rows"])}))
    elif args.stage == "run":
        plan = sealed_read(args.root / "plan.json")
        if (
            plan["source"] != cstr.source_identity()
            or plan["implementation"] != metrics.digest(Path(__file__))
            or plan["metric_implementation"] != metrics.digest(Path(scores.__file__))
        ):
            raise ValueError("evaluation implementation changed")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(
                one_subject, [(args.root, row) for row in plan["rows"]]
            ):
                print(json.dumps(result), flush=True)
        report(args.root)
    else:
        print(json.dumps(report(args.root)["groups"], indent=2))


if __name__ == "__main__":
    main()
