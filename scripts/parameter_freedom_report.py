"""Standard-library-only reporting of saved freedom comparisons and progress."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def read(path: Path, default=None):
    """Read a completed atomic artifact, or leave a missing result explicit."""
    return json.loads(path.read_text()) if path.exists() else default


def write(path: Path, value) -> None:
    """Replace one report atomically, without touching experiment checkpoints."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def progress(root: Path, observations: int, initializer_seconds: float) -> list[dict]:
    """Show evaluation cost over elapsed time; optimizer calls are not iterations."""
    rows = []
    screen = read(root / "recovery/feasibility_screen.json", [])
    clock = 0.0
    for entry in screen:
        clock += entry.get("seconds") or 0.0
        cost = entry.get("cost")
        rows.append(
            {
                "stage": "screen",
                "source": entry["source"],
                "stage_evaluation": len(rows) + 1,
                "cumulative_evaluation_seconds": clock,
                "train_nmse": 2 * cost / observations if cost is not None else None,
                "status": entry.get("evaluation_status"),
            }
        )
    calls = 0
    for directory in sorted((root / "recovery").glob("augmented_*")):
        stage_clock = 0.0
        for path in sorted(directory.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json")):
            item = read(path)
            stage_clock += item.get("seconds") or 0.0
            calls += 1
            cost = item.get("cost")
            rows.append(
                {
                    "stage": directory.name,
                    "sensitivity_evaluation": calls,
                    "cumulative_stage_evaluation_seconds": stage_clock,
                    "train_nmse": 2 * cost / observations if cost is not None else None,
                    "status": item.get("status"),
                }
            )
    # Callback times include optimizer algebra and overhead, unlike summed solve
    # times above. Their clock origin is stored in the stage report.
    refinement = read(root / "recovery/refinement_result.json", {})
    for stage in refinement.get("stages", []):
        offset = stage.get("started_at_refinement_seconds")
        for item in (stage.get("result") or {}).get("iterations", []):
            rows.append(
                {
                    "stage": "optimizer_iteration",
                    "source": stage.get("source"),
                    "iteration": item["iteration"],
                    "nfev": item["nfev"],
                    "refinement_elapsed_seconds": offset + item["seconds"]
                    if offset is not None
                    else None,
                    "fit_elapsed_seconds": initializer_seconds
                    + offset
                    + item["seconds"]
                    if offset is not None
                    else None,
                    "train_nmse": 2 * item["cost"] / observations,
                }
            )
    return rows


def report(output: Path) -> dict:
    """Summarize all six planned arms, including blocked or interrupted work."""
    frozen = read(output / "freeze.json")
    gate = read(output / "gate/result.json", {})
    if not gate and (output / "gate/supervisor.json").exists():
        supervisor = read(output / "gate/supervisor.json")
        if supervisor.get("exit_code"):
            gate = {"pass": False, "status": "worker_failed", **supervisor}
    rows, details = [], []
    for task in frozen["tasks"]:
        root = output / f"results/task_{task['index']:03d}"
        record = read(root / "result.json", {})
        saved = read(root / "fit.json", {})
        fit = saved.get("fit", record.get("fit", {}))
        init, ref = fit.get("initializer", {}), fit.get("refinement", {})
        p = read(output / f"problems/{task['index']:03d}.json")
        count = sum(len(r["time"]) for r in p["splits"]["train"]["rows"])
        stages = ref.get("stages", [])
        primary = (init.get("portfolio") or [init])[0]
        phase = primary.get("phase_timing") or read(
            root / "collocation/phase_timing.json", {}
        )
        worker = primary.get("worker_timing") or read(
            root / "collocation/worker_timing.json", {}
        )
        c_progress = primary.get("progress") or read(
            root / "collocation/progress.json", {}
        )
        history = c_progress.get("iterations", [])
        sensitivity = [s for s in stages if s["mode"] == "sensitivity"]
        selected = [
            s
            for s in sensitivity
            if (s.get("result") or {}).get("parameters") == fit.get("parameters")
            and fit.get("parameters")
        ]
        selected_result = selected[-1].get("result", {}) if selected else {}
        replays = record.get("replays", {})
        supervisor = read(root / "supervisor.json", {})
        status = record.get("status") or (
            "blocked_by_gate"
            if gate and not gate.get("pass")
            else "worker_timeout"
            if supervisor.get("exit_code") == 124
            else "worker_failed"
            if supervisor.get("exit_code")
            else "verification_pending"
            if saved
            else "started_without_result"
            if (root / "fit_started.json").exists()
            else "missing"
        )
        row = {
            **task,
            "status": status,
            "C_accepted": init.get("success"),
            "C_seconds": init.get("portfolio_total_seconds", init.get("seconds")),
            "C_iterations_recorded": len(history),
            "screen_seconds": ref.get("screening_seconds"),
            "refinement_seconds": ref.get("fit_seconds"),
            "sensitivity_calls": sum(
                (s.get("result") or {}).get(
                    "actual_residual_calls", s.get("actual_residual_calls", 0)
                )
                for s in sensitivity
            ),
            "optimizer_iterations": sum(
                len(s.get("result", {}).get("iterations", [])) for s in sensitivity
            ),
            "selected_optimizer_native_success": selected_result.get(
                "optimizer_native_success"
            ),
            "selected_optimizer_stop": selected_result.get("message"),
            "verified": record.get("verified"),
            "strict_recovery": record.get("recovered"),
            **{
                f"{k}_nmse": (
                    (replays.get(k) or {}).get("scores", {}).get("Radau") or {}
                ).get("clean_nmse")
                for k in ("train", "val")
            },
        }
        rows.append(row)
        curve = progress(root, count, row["C_seconds"] or 0.0)
        if root.exists():
            write(root / "progress_summary.json", curve)
        details.append(
            {
                **row,
                "initializer_message": init.get("message"),
                "worker_timing": worker,
                "phase_timing": phase,
                "time_to_first_C_iteration_seconds": history[0]["seconds"]
                if history
                else None,
                "C_last_iteration": history[-1] if history else None,
                "stages": [
                    {k: v for k, v in s.items() if k != "result"}
                    | {
                        "result": {
                            k: s.get("result", {}).get(k)
                            for k in (
                                "message",
                                "optimizer_native_success",
                                "optimality",
                                "actual_residual_calls",
                                "fit_seconds",
                                "residual_seconds",
                            )
                        },
                    }
                    for s in stages
                ],
                "error": record.get("error"),
            }
        )
    summary = {
        "identity": frozen["identity"],
        "gate": gate,
        "counts": dict(Counter(r["status"] for r in rows)),
        "rows": rows,
    }
    write(output / "summary.json", summary)
    write(
        output / "diagnostics.json",
        {"identity": frozen["identity"], "records": details},
    )
    lines = [
        "# Parameter freedom and optimization budget comparison",
        "",
        f"Preflight passed: {gate.get('pass')}. Status counts: {summary['counts']}.",
        "",
        "Six synthetic reference controls, one CPU per task. Fixed-basis and "
        "matched free-shape arms start from identical physical models; both "
        "receive reference shape values. Original controls reuse the v3 starts. "
        "All near starts (including original shapes) and all per-trajectory "
        "hidden initials are reference-assisted.",
        "",
        "No test data or proposer/judge access. Strict recovery requires verified "
        "train and validation NMSE <=1e-4; other errors remain quantitative results. "
        "Native success is distinct from replay verification. All observations "
        "and input interpolation are unchanged.",
        "",
        "| Start | Arm | Parameters | Status | C accepted | C s | Screen s | "
        "Refine s | Sensitivity calls | Optimizer iterations | "
        "Native success of selected stage | Verified | Train NMSE | Validation NMSE |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                str(r[k])
                for k in (
                    "start",
                    "arm",
                    "free_parameters",
                    "status",
                    "C_accepted",
                    "C_seconds",
                    "screen_seconds",
                    "refinement_seconds",
                    "sensitivity_calls",
                    "optimizer_iterations",
                    "selected_optimizer_native_success",
                    "verified",
                    "train_nmse",
                    "val_nmse",
                )
            )
            + " |"
        )
    lines += [
        "",
        "Phase timing, stopping messages and iteration evidence: diagnostics.json. "
        "Training progress: results/task_*/progress_summary.json. Summed evaluation "
        "time excludes optimizer overhead; callback elapsed time includes it.",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines))
    return summary
