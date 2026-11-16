"""Unify frozen historical and acquisition Full models on six-case test panels.

T1 and T2 use their existing Phase C test schedules, with shared targets across
easy/hard tiers and clean-training normalization. Historical equations keep
their meal-event encoding. Existing qualified forecasts are authenticated and
rescored; only the missing T1 forecasts require new numerical integration.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from autoformalism.benchmarks.phase_b_generation import (
    PhaseBProtocol,
    simulate_phase_b,
)
from autoformalism.benchmarks.phase_c_protocols import phase_c_protocols
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.reference_integration import ReferenceSolver
from autoformalism.staged_topology import content_hash
from scripts import iclr_historical_observation as historical
from scripts import iclr_scaling_response as metrics
from scripts import iclr_stress_evaluation as stress

PROTOCOL = "iclr-glucose-six-case-panel-1"
PROJECT = Path(__file__).resolve().parents[1]
ROSTER = Path("output/iclr-panel-alignment-20261116/pending_glucose_roster.json")
TRANSFER = Path("output/iclr-historical-transfer-20261116")
QUALITY = Path("output/iclr-data-quality-panel-20261116")
TARGETS = {"T1": ("Gp",), "T2": ("Gp", "I")}
LABELS = {
    "acquisition_00": "Current T2-hard: 0% noise, 1 min",
    "acquisition_01": "Current T2-hard: 1% noise, 1 min",
    "acquisition_02": "Current T2-hard: 5% noise, 1 min",
    "acquisition_03": "Current T2-hard: 10% noise, 1 min",
    "acquisition_04": "Current T2-hard: 0% noise, 2 min",
    "acquisition_05": "Current T2-hard: 0% noise, 4 min",
    **{
        c: "Historical "
        + c.replace("phase_b_dalla_man_", "").replace("_canonical_named_", "-").upper()
        for c in historical.CELLS
    },
}


def schedules(task: str) -> dict:
    """Keep exactly six existing tests and their original matched controls."""
    if task not in TARGETS:
        raise ValueError("unsupported glucose task")
    result = {}
    for p in phase_c_protocols("dalla_man", task=task):
        if p.split != "test":
            continue
        quiet = {"meals": []}
        if "initial_shift" in p.specification:
            quiet["initial_shift"] = p.specification["initial_shift"]
        result[p.protocol_id] = {
            "perturbed": p.model_dump(mode="json"),
            "control": p.model_copy(update={"specification": quiet}).model_dump(
                mode="json"
            ),
        }
    if len(result) != 6:
        raise ValueError("six-case protocol changed")
    return result


def event_input(time: list, meals: list) -> list:
    """Encode declared meal amounts at on-grid onsets for historical equations."""
    t = np.asarray(time, dtype=float)
    values = np.zeros_like(t)
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("invalid meal event grid")
    for onset, grams in meals:
        indices = np.flatnonzero(np.isclose(t, onset, rtol=0, atol=1e-10))
        if len(indices) != 1 or not np.isfinite(grams) or grams <= 0:
            raise ValueError("unrepresentable meal event")
        values[indices[0]] += grams
    return values.tolist()


def reference(protocol: dict) -> dict:
    """Generate continuous-rate truth, checking both states and extra auxiliaries."""
    p = PhaseBProtocol.model_validate(protocol)
    if p.input_contract != "continuous-rates-1" or p.split != "test":
        raise ValueError("expected a continuous-rate held-out protocol")
    a, b = [
        simulate_phase_b(p, solver=ReferenceSolver(method=m))
        for m in ("DOP853", "Radau")
    ]
    if not np.array_equal(a.time, b.time) or not np.allclose(
        a.states, b.states, rtol=2e-6, atol=2e-6
    ):
        raise ValueError("reference state solvers disagree")
    fields = {n: a.states[:, i] for i, n in enumerate(a.state_names)}
    fields.update(a.derived)
    fields.update({n: a.inputs[:, i] for i, n in enumerate(a.input_names)})
    for n in ("I", "U", "EGP", "E", "Uii"):
        if not np.allclose(a.derived[n], b.derived[n], rtol=2e-6, atol=2e-6):
            raise ValueError("reference auxiliary solvers disagree")
    names = {"Gp", "I", "U", "EGP", "Uii", "E", "Gt", *p.input_names}
    return {
        "time": a.time.tolist(),
        "fields": {n: fields[n].tolist() for n in sorted(names)}
        | {"meal_event_g": event_input(a.time.tolist(), p.specification["meals"])},
    }


def same_preparation(a: dict, b: dict) -> bool:
    """Allow sharing physical truth when only the declared input subset differs."""
    return {k: v for k, v in a.items() if k != "input_names"} == {
        k: v for k, v in b.items() if k != "input_names"
    } and set(a["input_names"]) <= set(b["input_names"])


def sources_match(project: Path, sources: dict) -> None:
    """Reject changed source bytes before importing any frozen artifact."""
    for name, expected in sources.items():
        if metrics.digest(project / name) != expected:
            raise ValueError(f"source changed: {name}")


def freeze(root: Path, project: Path) -> dict:
    """Pin the already enumerated 26 models, six-case protocols and normalizers."""
    roster = sealed_read(project / ROSTER)
    sources_match(project, roster["sources"])
    transfer = sealed_read(project / TRANSFER / "plan.json")
    cohort_sources = {
        d["artifact_sha256"]: d
        for p in roster["sources"]
        if p.endswith("/frozen_models.json")
        or p == "output/iclr-historical-observation-20261116/plan.json"
        for d in [sealed_read(project / p)]
    }
    rows = []
    for r in roster["models"]:
        current = "acquisition" in r
        task = "T2" if current else historical.task_of(r["condition"])
        if r["endpoint"] not in cohort_sources[r["source_bundle"]]["rows"]:
            raise ValueError("model is absent from its frozen source cohort")
        if current and any(
            r["acquisition"]["scales"][c] != transfer["training_sd"]["T2"][c]
            for c in TARGETS["T2"]
        ):
            raise ValueError("current T2 normalization changed")
        rows.append(
            {
                "key": f"{r['condition']}_seed{r['seed']}",
                "condition": r["condition"],
                "label": LABELS[r["condition"]],
                "task": task,
                "seed": r["seed"],
                "origin": "current" if current else "historical",
                "source_id": r["source_id"],
                "source_bundle": r["source_bundle"],
                "status": r["source_model_status"],
                "endpoint": r["endpoint"],
                "acquisition": r.get("acquisition", {}),
            }
        )
    expected = {(f"acquisition_{i:02d}", s) for i in range(6) for s in range(3)} | {
        (c, s) for c in historical.CELLS for s in (0, 1)
    }
    if len(rows) != 26 or {(r["condition"], r["seed"]) for r in rows} != expected:
        raise ValueError("planned model denominator changed")
    sources = roster["sources"] | {
        str(p): metrics.digest(project / p)
        for p in (ROSTER, TRANSFER / "plan.json", TRANSFER / "summary.json")
    }
    panels = {task: schedules(task) for task in TARGETS}
    for r in rows:
        if r["origin"] == "current":
            p = QUALITY / "t2" / r["condition"] / "plan.json"
            sources[str(p)] = metrics.digest(project / p)
    return metrics.write(
        root / "plan.json",
        {
            "protocol": PROTOCOL,
            "rows": rows,
            "panels": panels,
            "training_sd": transfer["training_sd"],
            "sources": sources,
            "implementation": metrics.digest(Path(__file__)),
            "source": historical.anchor.source_hash(),
            "metrics": metrics.digest(Path(metrics.__file__)),
            "runtime": public._runtime(),
            "selection_changed": False,
            "test_metrics_used_for_selection": False,
            "scope": "All 26 planned models in the agreed glucose comparison",
        },
    )


def references(root: Path, plan: dict, project: Path) -> dict:
    """Reuse exact saved T2 truths, extending the T1 panel without changing them."""
    transfer = sealed_read(project / TRANSFER / "plan.json")
    result = {}
    for task, panel in plan["panels"].items():
        for case, sides in panel.items():
            for side, protocol in sides.items():
                key = f"{task}_{case}_{side}"
                path = root / "references" / f"{key}.json"
                binding = {"plan": plan["artifact_sha256"], "protocol": protocol}
                if not path.exists():
                    old_key = f"stress_{case}_{side}"
                    source = transfer["conditions"][old_key]
                    inherited = same_preparation(protocol, source["protocol"])
                    source_receipt = None
                    if inherited:
                        p = project / TRANSFER / "references" / f"{old_key}.json"
                        record = sealed_read(p)
                        if (
                            record["plan"] != transfer["artifact_sha256"]
                            or record["condition"] != old_key
                        ):
                            raise ValueError("unbound saved physical reference")
                        value = record["reference"]
                        source_receipt = record
                        for n, v in source["saved"].items():
                            got = value["time"] if n == "time" else value["fields"][n]
                            if got != v:
                                raise ValueError("saved reference values changed")
                    else:
                        value = reference(protocol)
                    metrics.write(
                        path,
                        binding
                        | {"reference": value, "source_receipt": source_receipt},
                    )
                record = sealed_read(path)
                if any(record[k] != v for k, v in binding.items()):
                    raise ValueError("stale six-case reference")
                result[key] = record
    return result


def public_pair(row: dict, case: str, a: dict, b: dict) -> dict:
    """Project physical trajectories onto each model's original channel contract."""
    context = row["endpoint"]["context"]
    result = {"case": case}
    for side, ref in (("perturbed", a), ("control", b)):
        result[side] = {
            "trajectory_id": case,
            "time": ref["time"],
            **{
                role: {n: ref["fields"][n] for n in context[role]}
                for role in ("targets", "auxiliaries", "external_inputs")
            },
        }
    return result


def import_forecast(project: Path, row: dict, case: str, pair: dict) -> dict | None:
    """Authenticate previous paired solver checks; never substitute metric rows."""
    if row["origin"] == "current":
        folder = project / QUALITY / "t2" / row["condition"]
        old_plan = sealed_read(folder / "plan.json")
        if (
            old_plan["cell"] != row["acquisition"]
            or old_plan["models"] != row["source_bundle"]
        ):
            raise ValueError("current forecast belongs to another frozen cohort")
        p = folder / "curves" / row["source_id"] / f"{case}.json"
        value = sealed_read(p)
        if (
            value["plan_sha256"] != old_plan["artifact_sha256"]
            or value["source_id"] != row["source_id"]
            or value["reference_sha256"] != public.content_sha256(pair)
        ):
            raise ValueError("current forecast/reference binding differs")
        return {
            "status": "complete",
            "predicted": value["predicted"],
            "control": value["control"],
            "qualification": "native paired Radau/BDF check",
            "imported_receipts": [value],
        }
    if row["task"] == "T1":
        return None
    old_root = project / TRANSFER
    old_plan = sealed_read(old_root / "plan.json")
    if row["endpoint"] not in old_plan["rows"]:
        raise ValueError("historical model changed")
    refs, forecasts = [], []
    for side in ("perturbed", "control"):
        ref = sealed_read(old_root / "references" / f"stress_{case}_{side}.json")
        p = old_root / "predictions" / row["source_id"]
        pred = sealed_read(p / f"{ref['artifact_sha256']}.json")
        if (
            pred["plan"] != old_plan["artifact_sha256"]
            or pred["model"] != row["endpoint"]["model_id"]
            or pred["reference"] != ref["artifact_sha256"]
        ):
            raise ValueError("historical forecast/reference binding differs")
        for role in ("targets", "auxiliaries", "external_inputs"):
            for n, v in pair[side][role].items():
                if v != ref["reference"]["fields"][n]:
                    raise ValueError("historical forecast used different inputs")
        if pair[side]["time"] != ref["reference"]["time"]:
            raise ValueError("historical forecast time grid differs")
        refs.append(ref["reference"])
        forecasts.append(pred)
    for c in TARGETS["T2"]:
        checked = historical.score_pair(
            *forecasts, *refs, c, old_plan["training_sd"]["T2"][c], "original_heldout"
        )
        if checked["status"] != "complete":
            return {"status": checked["status"], "imported_receipts": forecasts}
    return {
        "status": "complete",
        "predicted": forecasts[0]["predicted"],
        "control": forecasts[1]["predicted"],
        "qualification": "historical Radau/BDF check, paired effect rechecked",
        "imported_receipts": forecasts,
    }


def forecast(job: tuple) -> dict:
    """Checkpoint one immutable model/contrast, including explicit failures."""
    root, project, plan_hash, row, case, a, b = job
    pair = public_pair(row, case, a["reference"], b["reference"])
    binding = {
        "plan": plan_hash,
        "model": content_hash(row["endpoint"]),
        "key": row["key"],
        "case": case,
        "references": [a["artifact_sha256"], b["artifact_sha256"]],
        "public_pair": content_hash(pair),
    }
    path = root / "predictions" / row["key"] / f"{case}.json"
    if not path.exists():
        value = import_forecast(project, row, case, pair)
        if value is None:
            try:
                p, q = stress.predict(row["endpoint"], row["acquisition"], pair)
                value = {
                    "status": "complete",
                    "predicted": {k: v.tolist() for k, v in p.items()},
                    "control": {k: v.tolist() for k, v in q.items()},
                    "qualification": "new paired Radau/BDF check",
                }
            except (ValueError, ArithmeticError, RuntimeError, TimeoutError) as exc:
                value = {"status": "rollout_unavailable", "reason": str(exc)}
        metrics.write(path, binding | value)
    value = sealed_read(path)
    if any(value[k] != v for k, v in binding.items()):
        raise ValueError("stale six-case forecast")
    return value


def score(value: dict, a: dict, b: dict, target: str, scale: float) -> dict:
    """Use the native six-case RSE integral and common response diagnostics."""
    if value["status"] != "complete":
        return {"status": value["status"]}
    if not np.isfinite(scale) or scale <= 0 or a["time"] != b["time"]:
        raise ValueError("invalid normalization or paired time grid")
    y, c = [np.asarray(r["fields"][target]) for r in (a, b)]
    p, q = [np.asarray(value[k][target]) for k in ("predicted", "control")]
    d = metrics.response_metrics(y, c, p, q)
    effect, predicted, t = y - c, p - q, np.asarray(a["time"])
    return {
        "status": "complete",
        "effect_status": d["status"],
        "test_nmse": float(np.mean((p - y) ** 2) / scale**2),
        "response_rse": float(
            np.trapezoid((predicted - effect) ** 2, t) / np.trapezoid(effect**2, t)
        )
        if d["status"] == "complete"
        else None,
        "directional_accuracy": d.get("sign_agreement"),
        "shape_error": d.get("shape_rse"),
        "shape_gain": d.get("best_scale"),
    }


def parity(project: Path, records: list[dict]) -> int:
    """Require every previously scored T2 case/channel to reproduce its value."""
    quality = sealed_read(project / QUALITY / "summary.json")
    transfer = sealed_read(project / TRANSFER / "summary.json")
    expected = {
        (r["cell"], r["seed"], r["case"], r["target"]): r
        for r in quality["case_rows"]
        if r["family"] == "T2"
    }
    expected.update(
        {
            (r["cell"], r["seed"], r["case"], r["target"]): r
            for r in transfer["case_rows"]
            if r["panel"] == "phase_c_t2_stress"
        }
    )
    seen = set()
    for r in (v for v in records if v["task"] == "T2"):
        key = r["condition"], r["seed"], r["case"], r["target"]
        old = expected[key]
        for metric in historical.SCORES:
            a, b = r.get(metric), old.get(metric)
            if (a is None) != (b is None) or (
                a is not None and not np.isclose(a, b, rtol=1e-8, atol=1e-10)
            ):
                raise ValueError(f"previous six-case metric differs: {key}, {metric}")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("incomplete previous six-case coverage")
    return len(seen)


def run(root: Path, project: Path, workers: int) -> dict:
    """Evaluate all retained models and retain missing seeds in every denominator."""
    plan = sealed_read(root / "plan.json")
    sources_match(project, plan["sources"])
    if (
        plan["implementation"] != metrics.digest(Path(__file__))
        or plan["source"] != historical.anchor.source_hash()
        or plan["metrics"] != metrics.digest(Path(metrics.__file__))
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("six-case evaluator or runtime changed")
    refs = references(root, plan, project)
    jobs = []
    for row in plan["rows"]:
        if row["status"] != "ready":
            continue
        for case in plan["panels"][row["task"]]:
            pair = [refs[f"{row['task']}_{case}_{s}"] for s in ("perturbed", "control")]
            jobs.append((root, project, plan["artifact_sha256"], row, case, *pair))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, value in enumerate(pool.map(forecast, jobs), 1):
            print(
                f"{i}/{len(jobs)} {value['key']} {value['case']}: {value['status']}",
                flush=True,
            )
    records, rows, per_target = [], [], []
    for endpoint in plan["rows"]:
        row = {
            k: v for k, v in endpoint.items() if k not in ("endpoint", "acquisition")
        }
        row["model_identity"] = content_hash(endpoint["endpoint"])
        if row["status"] != "ready":
            rows.append(row)
            continue
        measured = []
        for case in plan["panels"][row["task"]]:
            value = sealed_read(root / "predictions" / row["key"] / f"{case}.json")
            pair = [
                refs[f"{row['task']}_{case}_{s}"]["reference"]
                for s in ("perturbed", "control")
            ]
            for c in TARGETS[row["task"]]:
                result = score(value, *pair, c, plan["training_sd"][row["task"]][c])
                measured.append(row | {"case": case, "target": c, **result})
        records.extend(measured)
        row.update(historical.pool(measured))
        row["case_target_records"] = len(measured)
        row["active_responses"] = sum(
            r.get("effect_status") == "complete" for r in measured
        )
        rows.append(row)
        for c in TARGETS[row["task"]]:
            per_target.append(
                {"key": row["key"], "target": c}
                | historical.pool([r for r in measured if r["target"] == c])
            )
    groups = []
    for condition, label in LABELS.items():
        cohort = [r for r in rows if r["condition"] == condition]
        complete = [r for r in cohort if r["status"] == "complete"]
        groups.append(
            {
                "condition": condition,
                "label": label,
                "planned": len(cohort),
                "scored": len(complete),
                "statistics": {
                    k: metrics.interval(
                        [r[k] for r in complete],
                        higher_is_better=k == "directional_accuracy",
                    )
                    if complete
                    else None
                    for k in historical.SCORES
                },
            }
        )
    result = metrics.write(
        root / "summary.json",
        {
            "protocol": PROTOCOL,
            "plan": plan["artifact_sha256"],
            "rows": rows,
            "case_rows": records,
            "per_target": per_target,
            "groups": groups,
            "reproduced_previous_t2_case_target_records": parity(project, records),
            "definition": {
                "panels": (
                    "Six existing Phase C tests per task; "
                    "T1 and T2 have task-specific sixth tests."
                ),
                "targets": TARGETS,
                "within_seed": (
                    "Median NMSE/RSE; mean continuous direction accuracy "
                    "and whole-response shape error."
                ),
                "across_seeds": (
                    "Median [worst, best]; no pooling across historical/current "
                    "cohorts or easy/hard tiers."
                ),
                "zero_effect": (
                    "Initial-shift test and its matched control coincide: "
                    "include its NMSE, leave response metrics undefined."
                ),
                "normalization": (
                    "Fixed clean Phase C training variance for each task/target."
                ),
                "historical_inputs": (
                    "Declared meal event amounts retained; "
                    "physical truth uses continuous ingestion."
                ),
            },
        },
    )
    lines = [
        "Full models on the six-case glucose-insulin stress panels",
        "",
        "| Cohort / condition | Scored/planned | Test NMSE | Response RSE | "
        "Directional accuracy | Shape error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g in groups:
        cells = [g["label"], f"{g['scored']}/{g['planned']}"]
        for k in historical.SCORES:
            v = g["statistics"][k]
            cells.append(
                "unavailable"
                if v is None
                else f"{v['median']:.2f} [{v['worst']:.2f}, {v['best']:.2f}]"
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.extend(f"{k}: {v}" for k, v in result["definition"].items())
    (root / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    """Run independently of notebooks; immutable checkpoints support resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--project", default=PROJECT, type=Path)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    root, project = args.root.resolve(), args.project.resolve()
    freeze(root, project)
    run(root, project, args.workers)


if __name__ == "__main__":
    main()
