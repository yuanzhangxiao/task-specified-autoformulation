"""Standard-library-only report: fit quality and solver agreement are separate."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def read(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def report(output: Path) -> dict:
    frozen = read(output / "freeze.json")
    gate = read(output / "gate/result.json", {})
    rows, details = [], []
    for task in frozen["tasks"]:
        root = output / f"results/task_{task['index']:03d}"
        record = read(root / "result.json", {})
        fit = read(root / "fit.json", {}).get("fit", record.get("fit", {}))
        supervisor = read(root / "supervisor.json", {})
        status = record.get("status") or (
            "blocked_by_gate"
            if gate and not gate.get("pass")
            else "worker_timeout"
            if supervisor.get("exit_code") == 124
            else "worker_failed"
            if supervisor.get("exit_code")
            else "replay_pending"
            if fit
            else "missing"
        )
        coupled = read(root / "coupled/result.json", {}).get("fit", {})
        initializer = coupled.get("initializer") or read(
            root / "coupled/fit/initializer.json", {}
        )
        stages = fit.get("stages", [])
        row = {
            "index": task["index"],
            "family": task["family"],
            "strategy": task["strategy"],
            "parameters": task["parameters"],
            "states": task["states"],
            "status": status,
            "C_accepted": initializer.get("success"),
            "C_message": initializer.get("message"),
            "C_seconds": initializer.get("seconds"),
            "replay_agreement": record.get("numerical_replay_pass"),
            **record.get("fit_bands", {}),
            **{
                f"{key}_nmse": record.get("replays", {})
                .get(key, {})
                .get("scores", {})
                .get("Radau", {})
                .get("clean_nmse")
                for key in ("train", "val")
            },
            "stage_calls": sum(s.get("actual_residual_calls", 0) for s in stages),
            "stage_call_accounting_complete": all(
                s["status"] != "interrupted" for s in stages
            ),
            "selection_calls": fit.get("selection", {}).get("actual_residual_calls"),
            "interrupted_stages": [
                s["name"] for s in stages if s["status"] == "interrupted"
            ],
            "supervisor_exit_code": supervisor.get("exit_code"),
            "oracle_initials": task["oracle_initials"],
            "oracle_shapes": task["oracle_shapes"],
        }
        rows.append(row)
        details.append(
            {
                **row,
                "stages": stages,
                "selection": fit.get("selection"),
                "replays": record.get("replays"),
                "selected_parameters": fit.get("parameters"),
                "phase_timing": read(
                    root / "coupled/fit/collocation/phase_timing.json"
                ),
                "collocation_progress": read(
                    root / "coupled/fit/collocation/progress.json"
                ),
                "error": record.get("error"),
            }
        )
    counts = dict(Counter(r["status"] for r in rows))
    result = {
        "identity": frozen["identity"],
        "counts": counts,
        "rows": rows,
        "gate": gate,
    }
    write(output / "summary.json", result)
    write(
        output / "diagnostics.json", {"identity": frozen["identity"], "rows": details}
    )
    lines = [
        "# Final fitter alternatives",
        "",
        f"Planned arms: {len(rows)}. Status counts: {counts}. "
        f"Gate passed: {gate.get('pass')}.",
        "",
        "All reference starts are ordinary. Fixed-shape controls retain oracle "
        "shapes and known hidden initials.",
        "The smaller model uses only observed initial output and training-fitted "
        "causal initial maps; it has a different skeleton and boundary contract.",
        "Replay agreement means BDF/Radau numerical consistency, not accurate "
        "prediction. All NMSEs compare predictions with observations.",
        "Strict/good/practical require both train and validation NMSE <= "
        "1e-4 / 0.01 / 0.1 and passing replay. These bands are not scientific "
        "certification.",
        "No automatic expansion or new runs follow this comparison. Retain "
        "failures and discuss freezing the fitter.",
        "",
        "| Family | Route | Status | C accepted | Replay agreement | Train NMSE "
        "| Validation NMSE | Strict | Good | Practical |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                str(r.get(k))
                for k in (
                    "family",
                    "strategy",
                    "status",
                    "C_accepted",
                    "replay_agreement",
                    "train_nmse",
                    "val_nmse",
                    "strict",
                    "good",
                    "practical",
                )
            )
            + " |"
        )
    lines += [
        "",
        "Smaller-model shared/causal initialization limitation (training only):",
        "",
        "```json",
        json.dumps(frozen["smaller_model_initialization_limit"], indent=2),
        "```",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines))
    return result
