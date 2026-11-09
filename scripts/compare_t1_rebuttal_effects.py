"""Compare frozen T1 models on matched physical interventions."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.public_fitting import PublicSplit
from scripts.probe_t1_full_interventions import adapt_channels
from scripts.probe_t1_interventions import cases, effect_error
from scripts.replay_t1_curves import CELLS, TASKS, replay_one

MODELS = "t1-curves-round12-2026-09-22"
PREVIOUS = "t1-intervention-full-2026-09-22"
EXTERNAL = "t1-external-intervention-replay-2026-09-25-v1"
PERTURBED = "t1-perturbed-meal-comparison-2026-09-25"
METHODS = ("sindy", "pysr", "d3", "sol")
PROBES = tuple(c["id"] for c in cases())


def contrasts() -> list[tuple[str, str, str]]:
    """Pair every original contrast and meal vs fasting at each preparation."""
    original = [
        (c["id"], c["id"], c["paired_control"]) for c in cases() if c["paired_control"]
    ]
    return original + [
        (f"meal_vs_fasting_gt{level}", f"meal_gt{level}", f"fasting_gt{level}")
        for level in (80, 100, 120)
    ]


def peak(time: np.ndarray, curve: np.ndarray) -> tuple[float, float | None]:
    """Signed value and first time at maximum absolute effect; zero has no timing."""
    index = int(np.argmax(np.abs(curve)))
    value = float(curve[index])
    return value, float(time[index]) if value else None


def effect_metrics(probe: dict, control: dict) -> dict:
    """Measure total paired Gp response over a common grid including sign and time."""
    if not np.array_equal(probe["time"], control["time"]):
        raise ValueError("intervention and control grids differ")
    time = np.asarray(probe["time"], dtype=float)
    if time.ndim != 1 or time.size < 2 or not np.all(np.diff(time) > 0):
        raise ValueError("expected a finite increasing time grid")
    for record in (probe, control):
        for key in ("observed", "predicted"):
            if key == "predicted" and not record["success"]:
                continue
            curve = np.asarray(record[key], dtype=float)
            if curve.shape != time.shape or not np.isfinite(curve).all():
                raise ValueError("invalid rollout curve")
    result = effect_error(probe, control)
    metrics = {
        "status": result["status"],
        "effect_rse": result["relative_squared_error"],
        "reference_signed_peak": None,
        "predicted_signed_peak": None,
        "reference_peak_min": None,
        "predicted_peak_min": None,
        "correct_peak_direction": None,
        "absolute_peak_magnitude_error": None,
        "absolute_peak_timing_error_min": None,
    }
    if result["status"] != "complete":
        return metrics
    ref, ref_time = peak(time, np.asarray(result["reference_delta"]))
    pred, pred_time = peak(time, np.asarray(result["predicted_delta"]))
    metrics.update(
        reference_signed_peak=ref,
        predicted_signed_peak=pred,
        reference_peak_min=ref_time,
        predicted_peak_min=pred_time,
        correct_peak_direction=bool(np.sign(ref) == np.sign(pred)),
        absolute_peak_magnitude_error=abs(abs(pred) - abs(ref)),
        absolute_peak_timing_error_min=(
            abs(pred_time - ref_time)
            if ref_time is not None and pred_time is not None
            else None
        ),
    )
    return metrics


def csv_report(path: Path, rows: list[dict]) -> None:
    """Publish a deterministic CSV atomically, retaining unavailable columns."""
    fields = dict.fromkeys(k for row in rows for k in row)
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def check_record(record: dict, reference: dict) -> None:
    """Reject misplaced or different public reference trajectories before pairing."""
    if (
        record["trajectory_id"] != reference["trajectory_id"]
        or record["time"] != reference["time"]
        or record["observed"] != reference["targets"]["v01"]
    ):
        raise ValueError("rollout does not match diagnostic reference")


def complete_replay(job: dict, refs: dict, cache: Path, scale: float) -> dict:
    """Replay an available ablation which was missing from the historical probes."""
    model = compile_candidate(
        CandidateModel.model_validate(job["candidate"]),
        ValidationContext.model_validate(job["context"]),
    )
    if set(model.parameter_names) != set(job["parameters"]):
        raise ValueError("frozen parameters do not match candidate")
    records = {}
    cache.mkdir(parents=True, exist_ok=True)
    for probe, ref in refs.items():
        row = adapt_channels(ref, job["cell"])
        split = PublicSplit.model_validate(
            {"name": "val", "fingerprint": public.content_sha256(row), "rows": [row]}
        )
        (trajectory,) = public.unpack_split(split).trajectories
        destination = cache / f"{probe}.json"
        marker = destination.with_suffix(".started.json")
        if marker.exists() and not destination.exists():
            # Pure CPU replay is safe to repeat. Preserve the interrupted attempt.
            sealed_read(marker)
            suffix = len(list(cache.glob(f"{probe}.interrupted.*.json"))) + 1
            marker.rename(cache / f"{probe}.interrupted.{suffix}.json")
        records[probe] = replay_one(
            model,
            trajectory,
            job["parameters"],
            job["initials"],
            job["context"]["targets"][0],
            scale,
            destination,
        )
    return records


def summarize_model(identity: dict, records: dict, refs: dict) -> list[dict]:
    """Return all planned contrasts, including an explicit row for a missing model."""
    for probe, record in records.items():
        check_record(record, refs[probe])
    rows = []
    for label, case, control in contrasts():
        row = {**identity, "contrast": label, "probe": case, "control": control}
        if records:
            measured = records[case]
            row.update(
                probe_absolute_nmse=measured["nmse"],
                control_absolute_nmse=records[control]["nmse"],
                **effect_metrics(measured, records[control]),
            )
        else:
            row.update(status="model_unavailable")
        rows.append(row)
    return rows


def report(rows: list[dict], endpoints: list[dict]) -> str:
    """Show both exploratory meal contrasts, including every planned repetition."""
    scores = {(r["model_id"], r["contrast"]): r for r in rows}

    def fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    anchor = next(e["model_id"] for e in endpoints if e["status"] == "available")
    meal = scores[anchor, "meal_vs_fasting_gt100"]
    split = scores[anchor, "split_meal_gt100"]
    lines = [
        "# Response to meal timing and dose on canonical T1",
        "",
        "Each frozen model received the same physical meal and tissue preparations. "
        "Responses subtract each model's own control. The single-meal contrast "
        "compares 60 g at minute 60 with fasting. The spacing contrast changes that "
        "meal to 30 g at minutes 60 and 90. All supplied physiological auxiliaries "
        "were regenerated and were available throughout the 300-minute rollouts. "
        "Glucose was predicted from its initial value.",
        "",
        f"The meal reference peaked at {meal['reference_signed_peak']:+.2f} mg/kg "
        f"at minute {meal['reference_peak_min']:.0f}. After splitting the meal, "
        "the reference difference from the single meal reached "
        f"{split['reference_signed_peak']:+.2f} mg/kg at minute "
        f"{split['reference_peak_min']:.0f}. Each signed peak below is the excursion "
        "of greatest absolute size. "
        "The response relative squared error (RSE) uses the entire difference curve. "
        "Perfect agreement scores 0 and predicting no response scores 1.",
        "",
        "| Cell | Method | Seed/rep | 60 g vs fasting RSE | Peak (mg/kg) | Peak min | "
        "Split vs single RSE | Peak (mg/kg) | Peak min |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for endpoint in endpoints:
        name = endpoint["model_id"]
        info = [
            "named" if endpoint["cell"] == CELLS[0] else "obfuscated",
            endpoint["method"],
            str(endpoint["seed"]),
        ]
        for contrast in ("meal_vs_fasting_gt100", "split_meal_gt100"):
            r = scores[name, contrast]
            info.extend(
                fmt(r.get(key))
                for key in ("effect_rse", "predicted_signed_peak", "predicted_peak_min")
            )
        lines.append("| " + " | ".join(info) + " |")
    lines.extend(
        [
            "",
            "The local Full, no-latent and prediction-only runs used round-12 retained "
            "endpoints from the six-cell development campaign. No-latent restricts "
            "differential states to observed targets. Prediction-only (no_spec) "
            "removes scientific context and requirements. One named no-latent seed "
            "produced no retained model. D3 uses its native sample-increment update. "
            "Methods have distinct search budgets. The named and obfuscated versions "
            "share the same canonical physiology and input protocols.",
            "",
            "All eight contrasts per endpoint, including ±20% initial tissue shifts "
            "under fasting and after a meal, are in paired_responses.csv. Endpoint "
            "fit quality and latent counts are in endpoints.csv. The study is an "
            "exploratory secondary analysis of previously inspected schedules. "
            "An accurate response can arise through supplied auxiliary channels.",
            "",
            "The separate perturbed-obfuscated sensitivity analysis is in "
            "perturbed_meal_solver_verified.csv. It includes both Full seeds and "
            "all three Sol repetitions. Flags give the original independent-solver "
            "agreement for each endpoint and its matched control.",
            "",
        ]
    )
    return "\n".join(lines)


def local_rows(
    artifacts: Path, root: Path, refs: dict, frozen: dict, original: dict
) -> tuple[list[dict], list[dict], list[str]]:
    """Collect available runs and explicitly retain every assigned missing seed."""
    local_jobs = {j["model_id"]: j for j in frozen["models"] if j["model_id"] in TASKS}
    unavailable = {j["model_id"] for j in frozen["unavailable"] if "model_id" in j}
    expected_missing = set(TASKS) - set(local_jobs)
    if expected_missing != unavailable or set(local_jobs) & unavailable:
        raise ValueError("unaccounted planned ablation")
    source_models = {j["model_id"]: j for j in original["models"]}
    repeated = []
    endpoints, rows = [], []
    for key in TASKS:
        job = local_jobs.get(key)
        cell_tag, seed_tag, method = key.split("_", 2)
        entry = {
            "model_id": key,
            "cell": CELLS[int(cell_tag[-2:])],
            "method": method,
            "seed": int(seed_tag.removeprefix("seed")),
        }
        records = {}
        endpoint = dict(entry, status="model_unavailable")
        if job:
            if (job["cell"], job["method"], job["seed"]) != (
                entry["cell"],
                method,
                entry["seed"],
            ):
                raise ValueError("misidentified local run")
            if key in source_models:
                if source_models[key] != job:
                    raise ValueError("replayed source candidate differs")
                records = {
                    p: sealed_read(artifacts / PREVIOUS / "replays" / key / f"{p}.json")
                    for p in PROBES
                }
            else:
                records = complete_replay(
                    job, refs, root / key, original["training_scale"]
                )
                repeated.append(key)
            endpoint.update(
                status="available",
                train_nmse=job["saved_training_nmse"],
                validation_nmse=job["saved_validation_nmse"],
                latent_states=sum(
                    s["kind"] == "latent" for s in job["candidate"]["states"]
                ),
                source_receipt=job["source_result_sha256"],
                candidate_sha256=public.content_sha256(job["candidate"]),
                model_source="round12_retained",
            )
        endpoints.append(endpoint)
        rows.extend(summarize_model(entry, records, refs))
    return endpoints, rows, repeated


def external_rows(
    artifacts: Path, refs: dict, external: dict, portfolio: dict
) -> tuple[list[dict], list[dict]]:
    """Join all three external repetitions to their native saved executions."""
    rows, endpoints = [], []
    subjects = {s["subject_id"]: s for s in portfolio["subjects"]}
    external_keys = set()
    for result in external["rows"]:
        if result["cell"] not in CELLS:
            continue
        method, rep = result["method"], result["repetition"]
        key = f"cell{CELLS.index(result['cell'])}_{method}_rep{rep}"
        if key != result["model_id"] or key in external_keys:
            raise ValueError("duplicate or misplaced baseline")
        external_keys.add(key)
        subject = subjects[result["subject_id"]]
        if (
            not subject["selection_frozen"]
            or result["candidate_sha256"] != public.content_sha256(subject["candidate"])
            or result["cell"] != subject["benchmark_id"]
            or result["repetition"] != subject["repetition"]
            or result["execution_semantics"] != subject["execution_semantics"]
        ):
            raise ValueError("baseline candidate identity differs")
        directory = artifacts / EXTERNAL / "replays" / key / "probes"
        records = {p: sealed_read(directory / f"{p}.json") for p in PROBES}
        for probe, record in records.items():
            saved = result["probes"][probe]
            if (record["success"], record["nmse"]) != (saved["success"], saved["nmse"]):
                raise ValueError("baseline probe differs from its frozen summary")
        entry = {
            "cell": result["cell"],
            "model_id": key,
            "method": method,
            "seed": rep,
        }
        endpoints.append(
            dict(
                entry,
                status="available",
                train_nmse=result["train"]["nmse"],
                validation_nmse=result["validation"]["nmse"],
                latent_states=sum(
                    s["kind"] == "latent" for s in subject["candidate"]["states"]
                ),
                candidate_sha256=result["candidate_sha256"],
                source_receipt=result["subject_id"],
                model_source=result["execution_semantics"],
            )
        )
        rows.extend(summarize_model(entry, records, refs))
    expected = {
        f"cell{i}_{m}_rep{r}" for i in range(2) for m in METHODS for r in range(3)
    }
    if external_keys != expected:
        raise ValueError("incomplete external baseline roster")

    return endpoints, rows


def run(artifacts: Path, output: Path, cache: Path) -> dict:
    """Complete ablation coverage and export all fixed canonical T1 endpoints."""
    frozen = sealed_read(artifacts / MODELS / "models.json")
    original = sealed_read(artifacts / PREVIOUS / "plan.json")
    external = sealed_read(artifacts / EXTERNAL / "summary.json")
    portfolio = sealed_read(artifacts / EXTERNAL / "plan.json")
    if (
        original["source_models_sha256"] != frozen["artifact_sha256"]
        or external["plan_sha256"] != portfolio["artifact_sha256"]
        or original["cases"] != cases()
        or portfolio["cases"] != cases()
    ):
        raise ValueError("incompatible archived models, probes, or external results")
    sources = {
        f"{MODELS}/models.json": frozen["artifact_sha256"],
        f"{PREVIOUS}/plan.json": original["artifact_sha256"],
        f"{EXTERNAL}/plan.json": portfolio["artifact_sha256"],
        f"{EXTERNAL}/summary.json": external["artifact_sha256"],
    }
    refs = {}
    for probe in PROBES:
        ref = sealed_read(artifacts / PREVIOUS / "references" / f"{probe}.json")
        if (
            original["imported_artifacts"][f"references/{probe}.json"]
            != ref["artifact_sha256"]
            or portfolio["references"]["original"][probe]["row"] != ref["row"]
        ):
            raise ValueError("physical reference identity differs between archives")
        refs[probe] = ref["row"]

    identity = {
        "models": sources,
        "runtime": public._runtime(),
        "runtime_source_sha256": public._source_identity(),
        "replay_source_sha256": hashlib.sha256(
            Path(replay_one.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "adapter_source_sha256": hashlib.sha256(
            Path(adapt_channels.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    # Bind replay checkpoints to the archived models, references and numerical code.
    root = cache / public.content_sha256(identity)
    with public._lock(root):
        endpoints, rows, repeated = local_rows(artifacts, root, refs, frozen, original)
    baseline_endpoints, baseline_rows = external_rows(
        artifacts, refs, external, portfolio
    )
    endpoints.extend(baseline_endpoints)
    rows.extend(baseline_rows)

    perturbed = sealed_read(artifacts / PERTURBED / "solver-audit.json")
    perturbed_plan = sealed_read(artifacts / PERTURBED / "plan.json")
    if perturbed["plan_sha256"] != perturbed_plan["artifact_sha256"]:
        raise ValueError("perturbed sensitivity archive does not match its design")
    sources = dict(sources)
    sources[f"{PERTURBED}/solver-audit.json"] = perturbed["artifact_sha256"]
    sources[f"{PERTURBED}/plan.json"] = perturbed_plan["artifact_sha256"]
    summary = {
        "protocol": "rebuttal-t1-paired-responses-1",
        "planned_endpoints": len(endpoints),
        "available_endpoints": sum(e["status"] == "available" for e in endpoints),
        "paired_rows": len(rows),
        "scored_rows": sum(r["status"] == "complete" for r in rows),
        "newly_replayed_ablations": repeated,
        "source_artifact_seals": sources,
        "replay_identity": identity,
        "cache_identity": root.name,
        "units": {
            "signed_peak": "mg/kg",
            "timing": "min",
            "effect_rse": "dimensionless",
        },
        "scope": (
            "Retrospective canonical T1-easy. Both naming conditions share "
            "one simulator and seven schedules. Physiological auxiliaries are "
            "regenerated per condition. No benchmark test, fitting or provider call. "
            "Effects include all supplied paths, including tissue glucose. "
            "Each model subtracts its own paired control. A zero response scores 1. "
            "One no-latent seed had no selected endpoint. "
            "Perturbed-obfuscated sensitivity uses a separate schedule grid and its "
            "archived independent-solver qualification. Blank scores are unavailable."
        ),
    }
    with public._lock(output):
        sealed_write(output / "manifest.json", summary)
        csv_report(output / "endpoints.csv", endpoints)
        csv_report(output / "paired_responses.csv", rows)
        csv_report(
            output / "perturbed_meal_solver_verified.csv",
            perturbed["verified_metric_rows"],
        )
        (output / "README.md").write_text(report(rows, endpoints))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.artifacts, args.output, args.cache))
