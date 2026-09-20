"""Sealed development release and bounded frozen-fitter attainability audit.

Reference equations/labels live only under diagnostic/. No proposer, critic or
held-out test is run. Repeat commands retain consumed numerical attempts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.benchmarks.detention import (
    GEOMETRY,
    RATING_HEADS,
    RATING_VALUES,
    TRUTH,
    DetentionConfig,
    generate_case,
    reference_request,
)
from autoformalism.expressions import PiecewiseLinearForcing
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

REPO = Path(__file__).resolve().parents[3]


def _seal(path: Path, value: dict) -> None:
    """Publish a content-addressed artifact once; reject changed resume content."""
    payload = {"value": value, "sha256": public.content_sha256(value)}
    if path.exists():
        if public._read(path) != payload:
            raise ValueError(f"sealed artifact differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        public._write(path, payload)


def _read_seal(path: Path) -> dict:
    payload = public._read(path)
    if public.content_sha256(payload["value"]) != payload["sha256"]:
        raise ValueError(f"sealed artifact digest differs: {path}")
    return payload["value"]


def equation_audit(request: PublicFitRequest) -> dict:
    """Restricted compiled-equation probes; witnesses, not a general symbolic proof."""
    model, _, _ = public._lower(request)
    covariates = {**GEOMETRY, "initial_up": 0.2, "inflow_up": 37.0, "inflow_down": 19.0}
    forcing = PiecewiseLinearForcing(
        [0, 1],
        {key: [v, v] for key, v in covariates.items()},
        allowed_channels=request.context.forcing_channels,
    )
    parameters = {n: TRUTH[n] for n in model.parameter_names}
    errors, reference_errors = [], []
    for up in (0.1, 0.35, 0.4, 0.85, 1.35, 2.1):
        for down in (0.05, 0.12, 0.22, 0.62, 1.1):
            x = {"h_up": up, "h_down": down}
            values = model.rhs(
                0, [x[n] for n in model.state_names], parameters, forcing
            )
            rhs = dict(zip(model.state_names, values, strict=True))
            qout = TRUTH["k_outlet"] * np.interp(
                max(down - GEOMETRY["crest_down"], 0), RATING_HEADS, RATING_VALUES
            )
            total = GEOMETRY["area_down"] * rhs["h_down"] + qout - 19
            expected_down = (19 - qout) / GEOMETRY["area_down"]
            if "h_up" in rhs:
                total += GEOMETRY["area_up"] * rhs["h_up"] - 37
                q = TRUTH["k_transfer"] * np.interp(
                    max(up - GEOMETRY["crest_up"], 0), RATING_HEADS, RATING_VALUES
                )
                reference_errors.append(
                    abs(rhs["h_up"] - (37 - q) / GEOMETRY["area_up"])
                )
                expected_down += q / GEOMETRY["area_down"]
            reference_errors.append(abs(rhs["h_down"] - expected_down))
            errors.append(abs(total))
    return {
        "maximum_balance_probe_error_m3_per_minute": float(max(errors)),
        "maximum_reference_rhs_error": float(max(reference_errors)),
        "pass": max(errors) < 1e-8 and max(reference_errors) < 1e-10,
        "interpretation": "physical-depth equation probes; not universal certification",
    }


def verify(root: Path) -> dict:
    """Check source and all immutable release files; never regenerate labels."""
    plan = _read_seal(root / "plan.json")
    if plan["source_sha256"] != public._source_identity():
        raise ValueError("source changed; use the pinned source")
    for relative, expected in plan["files"].items():
        path = root / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"release file changed: {relative}")
    return plan


def prepare(root: Path, config_path: Path) -> dict:
    """Create isolated public train/val files and evaluator-only diagnostics."""
    config = DetentionConfig.model_validate_json(config_path.read_text())
    with public._lock(root):
        if (root / "plan.json").exists():
            plan = verify(root)
            if plan["config"] != config.model_dump(mode="json"):
                raise ValueError("configuration changed")
            return plan
        tasks, artifacts, audits = [], [], {}
        for case in ("coupled", "independent"):
            cases, private = generate_case(config, case)
            prompt = (REPO / "configs/detention_prompts" / f"{case}.md").read_text()
            for noise_index, fractions in enumerate(config.noise_fractions):
                directory = root / "public" / case / f"noise{noise_index}"
                for split in ("train", "val"):
                    path = directory / f"{split}.json"
                    _seal(path, cases[f"noise{noise_index}"][split])
                    artifacts.append(path)
                path = directory / "specification.json"
                _seal(
                    path,
                    {
                        "protocol": config.protocol,
                        "case": case,
                        "public_prompt": prompt,
                        "noise_sd_fraction": fractions,
                        "initial_observation_noise": 0,
                        "time_unit": "minute",
                        "test_released": False,
                        "targets": ["h_down"],
                        "auxiliaries": [],
                        "external_inputs": ["inflow_up", "inflow_down"],
                        "fixed_covariates": [*GEOMETRY, "initial_up"],
                    },
                )
                artifacts.append(path)
                for start_index, start in enumerate(config.starts):
                    task = {
                        "index": len(tasks),
                        "case": case,
                        "noise_index": noise_index,
                        "start_index": start_index,
                        "name": f"{case}_noise{noise_index}_start{start_index}",
                    }
                    request = reference_request(case, start)
                    path = root / "diagnostic" / f"{task['name']}.request.json"
                    _seal(path, request.model_dump(mode="json"))
                    artifacts.append(path)
                    tasks.append(task)
            path = root / "diagnostic" / f"{case}.reference.json"
            _seal(path, private)
            artifacts.append(path)
            request = reference_request(case, config.starts[0])
            inline = reference_request(case, config.starts[0], inline=True)
            audits[case] = {
                "named": equation_audit(request),
                "inline": equation_audit(inline),
                "max_reference_solver_difference_m": max(
                    r["solver_depth_difference_m"] for r in private["rows"].values()
                ),
                "max_relative_reference_balance_error": max(
                    r["maximum_relative_balance_error"]
                    for r in private["rows"].values()
                ),
            }
            parameters = {name: TRUTH[name] for name in request.parameter_guesses}
            audits[case]["production_truth_replay"] = replay(
                root, {"case": case, "noise_index": 0}, request, parameters
            )
        gates = all(
            v["named"]["pass"]
            and v["inline"]["pass"]
            and v["production_truth_replay"]["replay_agreement"]
            and all(
                s["clean_nmse"] < 1e-8
                for s in v["production_truth_replay"]["scores"].values()
            )
            for v in audits.values()
        )
        head = np.linspace(0, 4, 4001)
        table = np.interp(head, RATING_HEADS, RATING_VALUES)
        plan = {
            "protocol": config.protocol,
            "config": config.model_dump(mode="json"),
            "source_sha256": public._source_identity(),
            "runtime": public._runtime(),
            "tasks": tasks,
            "audits": audits,
            "generation_gate_passed": gates,
            "rating_table_max_absolute_weir_difference": float(
                np.max(abs(table - head**1.5))
            ),
            "rating_table_max_relative_weir_difference_above_5cm": float(
                np.max(abs(table[head >= 0.05] / head[head >= 0.05] ** 1.5 - 1))
            ),
            "test_generated": False,
            "llm_calls": 0,
            "files": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in artifacts
            },
        }
        _seal(root / "plan.json", plan)
        return plan


def replay(root: Path, task: dict, request: PublicFitRequest, parameters: dict) -> dict:
    """Independent BDF/Radau consistency and post-fit clean engineering diagnostics."""
    model, _, _ = public._lower(request)
    reference = _read_seal(root / "diagnostic" / f"{task['case']}.reference.json")
    path = root / "public" / task["case"] / f"noise{task['noise_index']}"
    train = PublicSplit.model_validate(_read_seal(path / "train.json"))
    scale = float(np.std([r.targets["h_down"] for r in train.rows]))
    clean_scale = reference["clean_training_sd_m"]
    scores, errors = {}, []
    replay_started = monotonic()
    for split in ("train", "val"):
        rows = public.unpack_split(
            PublicSplit.model_validate(_read_seal(path / f"{split}.json"))
        )
        observed, clean, peaks, times, warning, differences = [], [], [], [], [], []
        for row in rows.trajectories:
            trajectories = []
            for method in ("Radau", "BDF"):
                sim = simulate_trajectory(
                    model,
                    row,
                    parameters,
                    {},
                    FitConfig(
                        integration_backend="solve_ivp",
                        integration_method=method,
                        relative_tolerance=1e-9,
                        absolute_tolerance=1e-11,
                    ),
                    deadline=replay_started + 240,
                    reset_observed_states=False,
                )
                if not sim.success:
                    errors.append(
                        f"{split}/{row.trajectory_id}/{method}: {sim.message}"
                    )
                    break
                trajectories.append(sim.predictions["h_down"])
            if len(trajectories) != 2:
                continue
            a, b = trajectories
            target = np.asarray(reference["rows"][row.trajectory_id]["states"][1])
            observed.extend(((a - row.targets["h_down"]) / scale) ** 2)
            clean.extend(((a - target) / clean_scale) ** 2)
            differences.append(float(np.max(abs(a - b)) / scale))
            peaks.append(float(abs(a.max() - target.max())))
            times.append(
                float(abs(row.time[np.argmax(a)] - row.time[np.argmax(target)]))
            )
            warning.append(
                float(
                    np.mean(
                        (a >= GEOMETRY["warning_depth"])
                        != (target >= GEOMETRY["warning_depth"])
                    )
                )
            )
        complete = len(peaks) == len(rows.trajectories)
        scores[split] = {
            "available": complete,
            "observed_nmse": float(np.mean(observed)) if complete else None,
            "clean_nmse": float(np.mean(clean)) if complete else None,
            "mean_peak_depth_error_m": float(np.mean(peaks)) if complete else None,
            "mean_peak_time_error_minutes": float(np.mean(times)) if complete else None,
            "warning_classification_disagreement": float(np.mean(warning))
            if complete
            else None,
            "maximum_scaled_solver_difference": max(differences, default=None),
        }
    agreed = not errors and all(
        s["available"] and s["maximum_scaled_solver_difference"] < 1e-5
        for s in scores.values()
    )
    return {
        "scores": scores,
        "errors": errors,
        "replay_agreement": agreed,
        "strict_output_recovery": agreed
        and all(s["clean_nmse"] <= 1e-4 for s in scores.values()),
        "practical_output_recovery": agreed
        and all(s["clean_nmse"] <= 0.01 for s in scores.values()),
        "post_fit_only": True,
        "seconds": monotonic() - replay_started,
    }


def run_task(root: Path, index: int) -> dict:
    """One exact skeleton/start/noise task, with no automatic consumed-budget retry."""
    plan = verify(root)
    if not plan["generation_gate_passed"]:
        raise ValueError("generation gate failed")
    if index not in range(len(plan["tasks"])):
        raise ValueError("invalid task index")
    task = plan["tasks"][index]
    directory = root / "results" / task["name"]
    request = PublicFitRequest.model_validate(
        _read_seal(root / "diagnostic" / f"{task['name']}.request.json")
    )
    inputs = root / "public" / task["case"] / f"noise{task['noise_index']}"
    with public._lock(directory):
        train = PublicSplit.model_validate(_read_seal(inputs / "train.json"))
        val = PublicSplit.model_validate(_read_seal(inputs / "val.json"))
        public.prepare_fit(request, train, val, directory / "fit")
        result = public.execute_fit(directory / "fit")
        if (directory / "assessment.json").exists():
            saved = _read_seal(directory / "assessment.json")
            if saved["fit_result_sha256"] != public.content_sha256(result):
                raise ValueError("assessment fit identity changed")
            return saved
        assessment = {
            "task": task,
            "fit": result.model_dump(mode="json"),
            "fit_result_sha256": public.content_sha256(result),
            "diagnostic": None,
            "test_data_opened": False,
        }
        if result.status == "complete":
            marker = directory / "replay_started.json"
            if marker.exists():
                assessment["diagnostic"] = {
                    "status": "interrupted",
                    "replay_agreement": False,
                }
            else:
                _seal(marker, {"fit_result_sha256": public.content_sha256(result)})
                try:
                    assessment["diagnostic"] = replay(
                        root, task, request, dict(result.parameters)
                    )
                except (ValueError, RuntimeError, TimeoutError) as error:
                    assessment["diagnostic"] = {
                        "status": "replay_failed",
                        "error": str(error),
                        "replay_agreement": False,
                    }
        _seal(directory / "assessment.json", assessment)
        return assessment


def report(root: Path) -> dict:
    """Show missing/failed arms explicitly and never promote a partial benchmark."""
    plan = verify(root)
    rows = []
    for task in plan["tasks"]:
        path = root / "results" / task["name"] / "assessment.json"
        if path.exists():
            value = _read_seal(path)
            rows.append(
                {
                    "task": task,
                    "status": value["fit"]["status"],
                    "training": value["fit"]["training"],
                    "validation": value["fit"]["validation"],
                    "diagnostic": value["diagnostic"],
                }
            )
        else:
            rows.append({"task": task, "status": "missing"})
    counts = {
        status: sum(r["status"] == status for r in rows)
        for status in sorted({r["status"] for r in rows})
    }
    value = {
        "protocol": plan["protocol"],
        "plan_sha256": public.content_sha256(plan),
        "generation_gate_passed": plan["generation_gate_passed"],
        "status_counts": counts,
        "rows": rows,
        "test_data_opened": False,
        "automatic_promotion": False,
        "automatic_followup": False,
    }
    lines = [
        "# Detention-basin development audit",
        "",
        f"Status counts: {counts}",
        "",
        "Known reference skeleton and rating shape; "
        "two coefficients in the coupled case.",
        "Frozen fitter, two broad starts, no discovery or test evaluation.",
        "Independent basins are a negative control, not another benchmark family.",
        "",
        "| Case | Noise | Start | Status | Train NMSE | Validation NMSE "
        "| Replay | Strict clean recovery |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        task, diagnostic = row["task"], row.get("diagnostic") or {}
        fields = [
            task["case"],
            task["noise_index"],
            task["start_index"],
            row["status"],
            row.get("training", {}).get("normalized_mse"),
            row.get("validation", {}).get("normalized_mse"),
            diagnostic.get("replay_agreement"),
            diagnostic.get("strict_output_recovery"),
        ]
        lines.append("| " + " | ".join(str(f) for f in fields) + " |")
    lines += [
        "",
        "Replay means numerical consistency, not scientific correctness.",
        "Strict recovery requires both clean train/val NMSE <= 1e-4 "
        "and replay agreement.",
        "Review generation checks, all starts, physical assumptions "
        "and failures before promotion.",
    ]
    with public._lock(root / "report"):
        public._write(root / "summary.json", value)
        temporary = root / "SUMMARY.md.tmp"
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(root / "SUMMARY.md")
    return value
