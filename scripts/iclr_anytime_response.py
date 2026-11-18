"""Score validation-selected search checkpoints on the frozen intervention panels."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.staged_topology import content_hash
from scripts import iclr_cstr_observation_panel as cstr
from scripts import iclr_glucose_stress_panel as glucose
from scripts import iclr_scaling_response as metrics
from scripts import iclr_stress_data as acquisition
from scripts import iclr_stress_evaluation as stress

PROTOCOL = "iclr-anytime-responses-1"
PROJECT = Path(__file__).resolve().parents[1]
HISTORIES = Path("output/iclr-all-run-latent-audit-20261117/package/sources")
DESIGN = Path("output/iclr-stress-final-20261115/design")
CSTR_PANEL = Path("output/iclr-cstr-observation-panel-20261115")
GLUCOSE_PANEL = Path("output/iclr-glucose-six-case-20261116")
CHECKPOINTS = (0, 4, 9, 14)


def history(path: Path) -> dict:
    """Authenticate the immutable compressed scientific-history export."""
    value = json.loads(gzip.decompress(path.read_bytes()))
    if (
        content_hash({k: v for k, v in value.items() if k != "artifact_sha256"})
        != value["artifact_sha256"]
    ):
        raise ValueError("scientific history checksum differs")
    return value


def checkpoint_rows(source: dict, conditions: dict, cohort: str) -> list[dict]:
    """Use the recorded incumbent at each fixed visit, never the best test score."""
    rows = []
    for run in source["runs"]:
        task = run["task"]
        if task.get("treatment") != "full" or task["cell"] not in conditions:
            continue
        cell = conditions[task["cell"]]
        if cell.get("study") == "corrected_easy":
            continue
        if cell["family"] not in ("cstr", "t2", "alien"):
            continue
        visits = {v["round"]: v for v in run["visits"]}
        if set(visits) != set(range(15)):
            raise ValueError(
                "anytime cohort requires the complete fifteen-visit history"
            )
        checkpoints = [(str(i + 1), visits[i]["selected_model"]) for i in CHECKPOINTS]
        checkpoints.append(("pruned", run["final_model"]))
        for checkpoint, model_id in checkpoints:
            model = source["models"].get(model_id)
            endpoint = (
                None
                if model is None
                else {
                    "candidate": model["candidate"],
                    "context": model["context"],
                    "parameters": model["parameters"],
                    "initials": {},
                    "status": "ready",
                    "benchmark_id": cell["source"]
                    if cell["family"] != "alien"
                    else task["cell"],
                    "source_protocol": "iclr-controlled-search-models-1",
                    "direct_readouts": {},
                }
            )
            rows.append(
                {
                    "cohort": cohort,
                    "source_history": source["artifact_sha256"],
                    "source_id": task["task_id"],
                    "condition": task["cell"],
                    "family": cell["family"],
                    "seed": task["seed"],
                    "checkpoint": checkpoint,
                    "visits": 15 if checkpoint == "pruned" else int(checkpoint),
                    "key": f"{cohort}_{task['task_id']}_{checkpoint}",
                    "model_id": model_id,
                    "status": "ready" if endpoint else "model_unavailable",
                    "endpoint": endpoint,
                    "acquisition": cell,
                    "validation_nmse": model["validation"]["normalized_mse"]
                    if model
                    else None,
                    "state_count": len(model["candidate"]["states"]) if model else None,
                    "latent_state_count": sum(
                        s["kind"] == "latent" for s in model["candidate"]["states"]
                    )
                    if model
                    else None,
                    "parameter_count": len(model["parameters"]) if model else None,
                }
            )
    return rows


def freeze(root: Path, *, include_stress: bool = False) -> dict:
    """Pin schedules, models and all fixed checkpoint choices before new rollouts."""
    manifest = sealed_read(PROJECT / DESIGN / "manifest.json")
    conditions = {c["cell"]: c for c in manifest["cells"]}
    sources = ["anchors_20b"] + (["stress_20b"] if include_stress else [])
    rows = []
    source_files = {}
    for name in sources:
        path = PROJECT / HISTORIES / f"{name}.json.gz"
        source_files[str(path)] = metrics.digest(path)
        rows.extend(checkpoint_rows(history(path), conditions, name))
    cp, gp = [
        sealed_read(PROJECT / p / "plan.json") for p in (CSTR_PANEL, GLUCOSE_PANEL)
    ]
    for p in (CSTR_PANEL, GLUCOSE_PANEL):
        source_files[str(PROJECT / p / "plan.json")] = metrics.digest(
            PROJECT / p / "plan.json"
        )
    references = {"cstr": {}, "t2": {}, "alien": {}}
    for case, spec in cp["references"].items():
        # Copies in the existing observation package have the original seal.
        path = PROJECT / CSTR_PANEL / "references" / f"{case}.json"
        if metrics.digest(path) != spec["sha256"]:
            raise ValueError("CSTR panel reference changed")
        references["cstr"][case] = {"path": str(path), "sha256": spec["sha256"]}
    for case in gp["panels"]["T2"]:
        references["t2"][case] = {}
        for side in ("perturbed", "control"):
            path = PROJECT / GLUCOSE_PANEL / "references" / f"T2_{case}_{side}.json"
            references["t2"][case][side] = {
                "path": str(path),
                "sha256": metrics.digest(path),
            }
    if include_stress:
        for r in rows:
            if r["family"] == "alien" and r["condition"] not in references["alien"]:
                path = (
                    PROJECT
                    / "output/iclr-stress-final-20261115/evaluation/references"
                    / f"{r['condition']}.json"
                )
                if not path.is_file():
                    raise ValueError(f"missing frozen scaling references: {path}")
                references["alien"][r["condition"]] = {
                    "path": str(path),
                    "sha256": metrics.digest(path),
                }
    anchor, declared, _ = cstr.native()
    return metrics.write(
        root / "plan.json",
        {
            "protocol": PROTOCOL,
            "rows": rows,
            "sources": source_files,
            "references": references,
            "cstr_schedule": cp["schedules"],
            "training_sd": {"cstr": cp["training_sd"], "t2": gp["training_sd"]["T2"]},
            "implementation": metrics.digest(Path(__file__)),
            "numerical_implementations": {
                m.__name__: metrics.digest(Path(m.__file__))
                for m in (anchor, declared, stress, metrics, glucose)
            },
            "runtime": public._runtime(),
            "checkpoint_policy": (
                "Recorded validation-selected incumbent after visits 1/5/10/15; "
                "final pruning separately"
            ),
            "selection_changed": False,
            "test_metrics_used_for_selection": False,
            "cost_scope": (
                "Revision budget available; cumulative time/tokens require "
                "original per-visit receipts"
            ),
        },
    )


def read_reference(spec: dict) -> dict:
    """Verify the existing reference bytes before opening their physical values."""
    path = Path(spec["path"])
    if metrics.digest(path) != spec["sha256"]:
        raise ValueError("saved reference changed")
    value = json.loads(path.read_text())
    return read_seal(path) if set(value) == {"sha256", "value"} else sealed_read(path)


def continuous_score(
    time, truth, control, predicted, baseline, scale=None, *, integral=True
) -> dict:
    """Use integral RSE and the existing continuous direction/shape definitions."""
    a, b, p, q, t = [np.asarray(x) for x in (truth, control, predicted, baseline, time)]
    values = metrics.response_metrics(a, b, p, q)
    effect, estimated = a - b, p - q
    return {
        "test_nmse": float(np.mean((p - a) ** 2) / scale**2) if scale else None,
        "response_rse": float(
            np.trapezoid((effect - estimated) ** 2, t) / np.trapezoid(effect**2, t)
            if integral
            else np.mean((effect - estimated) ** 2) / np.mean(effect**2)
        )
        if values["status"] == "complete"
        else None,
        "directional_accuracy": values.get("sign_agreement"),
        "shape_error": values.get("shape_rse"),
        "effect_status": values["status"],
    }


def cstr_predictions(root: Path, plan: dict, row: dict) -> dict:
    """Integrate declared terms and independently check reactor temperature."""
    anchor, declared, _ = cstr.native()
    endpoint = row["endpoint"]
    view = declared.view_of(endpoint)
    assignment = cstr.attribute_observed(view)
    output = {}
    for case, spec in plan["references"]["cstr"].items():
        cache = root / "predictions" / row["model_id"] / f"cstr_{case}.json"
        ref = read_reference(spec)["reference"]
        if not cache.exists():
            try:
                tr = cstr.observation_trajectory(endpoint["context"], ref)
                curves = declared.curves(view, assignment, tr)
                alternate = anchor.predict(endpoint, tr, "BDF")
                anchor.probe.solver_agreement(
                    {"T": curves["T"]}, {"T": alternate["T"]}, 1e-4
                )
                value = {
                    "status": "complete",
                    "curves": curves,
                    "alternate_temperature": alternate["T"],
                }
            except (ValueError, ArithmeticError, RuntimeError, TimeoutError) as error:
                value = {"status": "rollout_unavailable", "reason": str(error)[:500]}
            metrics.write(
                cache,
                {
                    "plan": plan["artifact_sha256"],
                    "reference": spec["sha256"],
                    "model": row["model_id"],
                    **value,
                },
            )
        value = sealed_read(cache)
        if (
            value["plan"] != plan["artifact_sha256"]
            or value["reference"] != spec["sha256"]
        ):
            raise ValueError("stale CSTR anytime prediction")
        output[case] = (ref, value)
    results = []
    for case, schedule in plan["cstr_schedule"].items():
        if schedule["role"] != "contrast":
            continue
        ref, predicted = output[case]
        control, baseline = output[schedule["control"]]
        if predicted["status"] != "complete" or baseline["status"] != "complete":
            results.append({"case": case, "status": "rollout_unavailable"})
            continue
        target = cstr.target_score(
            anchor,
            predicted,
            baseline,
            ref,
            control,
            schedule,
            plan["training_sd"]["cstr"]["T"],
        )
        if target["status"] != "complete":
            results.append({"case": case, "status": target["status"]})
            continue
        truths, controls = declared._truth(ref), declared._truth(control)
        for quantity in ("T", *declared.MECHANISMS):
            a, b = (
                (ref["T"], control["T"])
                if quantity == "T"
                else (truths[quantity], controls[quantity])
            )
            # A missing declared mechanism has zero modeled response and remains scored.
            result = continuous_score(
                ref["time"],
                a,
                b,
                predicted["curves"][quantity],
                baseline["curves"][quantity],
                plan["training_sd"]["cstr"]["T"] if quantity == "T" else None,
                integral=False,
            )
            results.append(
                {
                    "case": case,
                    "quantity": quantity,
                    "status": "complete",
                    "represented": quantity == "T"
                    or assignment["represented"][quantity],
                    **result,
                }
            )
    return {"cases": results, "planned_cases": 24, "assignment": assignment}


def t2_predictions(root: Path, plan: dict, row: dict) -> dict:
    """Score exactly the existing six glucose tests with acquisition preserved."""
    results = []
    for case, specs in plan["references"]["t2"].items():
        refs = {side: read_reference(spec)["reference"] for side, spec in specs.items()}
        pair = glucose.public_pair(row, case, refs["perturbed"], refs["control"])
        cache = (
            root / "predictions" / row["model_id"] / f"{row['condition']}_{case}.json"
        )
        if not cache.exists():
            try:
                a, b = stress.predict(row["endpoint"], row["acquisition"], pair)
                value = {
                    "status": "complete",
                    "predicted": {k: v.tolist() for k, v in a.items()},
                    "control": {k: v.tolist() for k, v in b.items()},
                }
            except (ValueError, ArithmeticError, RuntimeError, TimeoutError) as error:
                value = {"status": "rollout_unavailable", "reason": str(error)[:500]}
            metrics.write(
                cache,
                {"plan": plan["artifact_sha256"], "pair": content_hash(pair), **value},
            )
        value = sealed_read(cache)
        if value["plan"] != plan["artifact_sha256"] or value["pair"] != content_hash(
            pair
        ):
            raise ValueError("stale T2 anytime prediction")
        for target in row["endpoint"]["context"]["targets"]:
            result = glucose.score(
                value,
                refs["perturbed"],
                refs["control"],
                target,
                plan["training_sd"]["t2"][target],
            )
            results.append({"case": case, "quantity": target, **result})
    return {"cases": results, "planned_cases": 6}


def scaling_predictions(root: Path, plan: dict, row: dict) -> dict:
    """Reuse the three saved scaling schedules and their matched controls."""
    cell = row["condition"]
    refs = read_reference(plan["references"]["alien"][cell])["records"]
    training = read_seal(PROJECT / DESIGN / "public" / cell / "train.json")["rows"]
    scales = acquisition.train_scales(training)
    rows = []
    for pair in refs:
        path = root / "predictions" / row["model_id"] / f"{cell}_{pair['case']}.json"
        if not path.exists():
            try:
                a, b = stress.predict(row["endpoint"], row["acquisition"], pair)
                value = {
                    "status": "complete",
                    "predicted": {k: v.tolist() for k, v in a.items()},
                    "control": {k: v.tolist() for k, v in b.items()},
                }
            except (ValueError, ArithmeticError, RuntimeError, TimeoutError) as error:
                value = {"status": "rollout_unavailable", "reason": str(error)[:500]}
            metrics.write(
                path,
                {"plan": plan["artifact_sha256"], "pair": content_hash(pair), **value},
            )
        value = sealed_read(path)
        if value["plan"] != plan["artifact_sha256"] or value["pair"] != content_hash(
            pair
        ):
            raise ValueError("stale scaling prediction")
        for target in row["endpoint"]["context"]["targets"]:
            score = (
                continuous_score(
                    pair["perturbed"]["time"],
                    pair["perturbed"]["targets"][target],
                    pair["control"]["targets"][target],
                    value["predicted"][target],
                    value["control"][target],
                    scales[target],
                )
                if value["status"] == "complete"
                else {}
            )
            rows.append(
                {
                    "case": pair["case"],
                    "quantity": target,
                    "status": value["status"],
                    **score,
                }
            )
    return {"cases": rows, "planned_cases": 3}


def one_subject(job: tuple) -> dict:
    """Checkpoint one predetermined model endpoint without refitting."""
    root, row = job
    plan = sealed_read(root / "plan.json")
    path = root / "subjects" / f"{row['key']}.json"
    if path.exists():
        result = sealed_read(path)
        if result["plan"] != plan["artifact_sha256"]:
            raise ValueError("stale anytime subject")
        return result
    if row["endpoint"] is None:
        value = {"status": "model_unavailable", "cases": []}
    elif row["family"] == "cstr":
        value = {"status": "evaluated", **cstr_predictions(root, plan, row)}
    elif row["family"] == "t2":
        value = {"status": "evaluated", **t2_predictions(root, plan, row)}
    elif row["family"] == "alien":
        value = {"status": "evaluated", **scaling_predictions(root, plan, row)}
    else:
        raise ValueError("unsupported anytime family")
    return metrics.write(
        path, {"plan": plan["artifact_sha256"], "key": row["key"], **value}
    )


def pooled(cases: list[dict], quantities: set[str], planned: int) -> dict:
    """Keep all planned tests; median RSE, mean accuracy and shape within a run."""
    rows = [c for c in cases if c.get("quantity") in quantities]
    if len(rows) != planned * len(quantities) or any(
        r["status"] != "complete" for r in rows
    ):
        return {"status": "incomplete_panel"}
    result = {"status": "complete"}
    for field in ("test_nmse", "response_rse", "directional_accuracy", "shape_error"):
        values = [r[field] for r in rows if r.get(field) is not None]
        result[field] = (
            (
                statistics.median(values)
                if field in {"response_rse", "test_nmse"}
                else statistics.mean(values)
            )
            if values
            else None
        )
    return result


def report(root: Path) -> dict:
    """Summarize every planned seed, preserving unavailable model denominators."""
    plan = sealed_read(root / "plan.json")
    rows = []
    for row in plan["rows"]:
        path = root / "subjects" / f"{row['key']}.json"
        value = (
            sealed_read(path) if path.exists() else {"status": "pending", "cases": []}
        )
        targets = {"cstr": {"T"}, "t2": {"Gp", "I"}, "alien": {"v01"}}[row["family"]]
        scores = (
            pooled(
                value["cases"],
                targets,
                {"cstr": 24, "t2": 6, "alien": 3}[row["family"]],
            )
            if value["status"] == "evaluated"
            else {"status": value["status"]}
        )
        if row["family"] == "alien" and scores["status"] == "complete":
            scores["test_nmse"] = statistics.mean(
                c["test_nmse"] for c in value["cases"]
            )
        rows.append(
            {k: v for k, v in row.items() if k not in ("endpoint", "acquisition")}
            | scores
        )
    groups = []
    order = {"1": 0, "5": 1, "10": 2, "15": 3, "pruned": 4}
    for condition, checkpoint in sorted(
        {(r["condition"], r["checkpoint"]) for r in rows},
        key=lambda pair: (pair[0], order[pair[1]]),
    ):
        group = [
            r
            for r in rows
            if r["condition"] == condition and r["checkpoint"] == checkpoint
        ]
        complete = [r for r in group if r["status"] == "complete"]
        groups.append(
            {
                "condition": condition,
                "checkpoint": checkpoint,
                "planned": len(group),
                "complete": len(complete),
                "statistics": {
                    field: metrics.interval(
                        [r[field] for r in complete if r.get(field) is not None],
                        higher_is_better=field == "directional_accuracy",
                    )
                    if complete
                    else None
                    for field in (
                        "test_nmse",
                        "response_rse",
                        "directional_accuracy",
                        "shape_error",
                        "state_count",
                        "parameter_count",
                    )
                },
            }
        )
    result = {
        "protocol": PROTOCOL,
        "plan": plan["artifact_sha256"],
        "rows": rows,
        "groups": groups,
        "selection_changed": False,
    }
    # Reports can be refreshed as independent subjects complete; sealed inputs cannot.
    from autoformalism.llm.staged_topology import atomic_json

    atomic_json(root / "summary.json", result)
    lines = [
        "Full search checkpoints: median [worst, best] over search seeds.",
        "Within each run: median NMSE and response RSE; "
        "mean directional accuracy and shape error.",
        "No test-based checkpoint selection. "
        "Final pruning includes its additional refit.",
        "",
        "| Condition | Visits / pruning | Complete | Test NMSE | "
        "Response RSE | Direction | Shape |",
        "|---|---|---|---|---|---|---|",
    ]

    def formatted(value):
        return (
            "—"
            if value is None
            else f"{value['median']:.2f} [{value['worst']:.2f}, {value['best']:.2f}]"
        )

    for group in groups:
        values = " | ".join(
            formatted(group["statistics"][k])
            for k in (
                "test_nmse",
                "response_rse",
                "directional_accuracy",
                "shape_error",
            )
        )
        lines.append(
            f"| {group['condition']} | {group['checkpoint']} | "
            f"{group['complete']}/{group['planned']} | {values} |"
        )
    (root / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    """Run the frozen post-hoc audit using existing numerical references only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--include-stress", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.stage == "freeze":
        plan = freeze(args.root, include_stress=args.include_stress)
        print(
            json.dumps({"plan": plan["artifact_sha256"], "subjects": len(plan["rows"])})
        )
    elif args.stage == "run":
        plan = sealed_read(args.root / "plan.json")
        if plan["implementation"] != metrics.digest(Path(__file__)):
            raise ValueError("anytime implementation changed after freeze")
        # A model reused by several visits shares cached rollouts. Run those visits
        # sequentially within a seed to avoid concurrent writes to identical caches.
        for checkpoint in ("1", "5", "10", "15", "pruned"):
            jobs = [
                (args.root, r) for r in plan["rows"] if r["checkpoint"] == checkpoint
            ]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for result in pool.map(one_subject, jobs):
                    print(
                        json.dumps({"key": result["key"], "status": result["status"]}),
                        flush=True,
                    )
            report(args.root)
    else:
        print(json.dumps({"rows": len(report(args.root)["rows"])}))


if __name__ == "__main__":
    main()
