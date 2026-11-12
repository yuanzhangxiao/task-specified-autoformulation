"""Score frozen external forecasts with Full's intervention and error metrics."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

from .exchange import (
    TARGETS,
    checked_curve,
    fingerprint,
    publish,
    read,
    validate_freeze,
)


def evaluate(key_path: Path, freeze_path: Path, forecasts: Path) -> dict:
    # The private kit supplies this module directly from Full's pinned scorer.
    from . import official

    key, frozen = read(key_path), read(freeze_path)
    validate_freeze(frozen)
    plan = frozen["plan"]
    if (
        key["format"] != "t2-noise-answer-key-1"
        or plan["release_sha256"] != key["release_sha256"]
        or key["full_evaluator_sha256"] != official.SOURCE_SHA256
    ):
        raise ValueError("wrong input release or Full evaluator")
    rows = []
    hashes = {}
    for terminal in frozen["rows"]:
        task = terminal["task"]
        cell = key["cells"][task["cell"]]
        if task["public_files"] != cell["public_files"]:
            raise ValueError("frozen model was fitted on different public files")
        row = {
            "method": plan["config"]["method"],
            "cell": task["cell"],
            "seed": task["seed"],
            "noise_percent": cell["noise_percent"],
            "semantics": plan["config"]["semantics"],
            "search_status": terminal["status"],
            "validation_nmse": terminal["validation_nmse"],
            "model_sha256": terminal["model_sha256"],
            "reason": terminal["reason"],
        }
        if terminal["status"] == "no_model":
            rows.append(
                {
                    **row,
                    "evaluation_status": "model_unavailable",
                    "test_nmse": None,
                    "common_scale_test_nmse": None,
                    "response_rse": None,
                    "trajectories": [],
                }
            )
            continue
        bound = {
            "freeze_sha256": fingerprint(frozen),
            "queries_sha256": key["queries_sha256"],
            "key": task["key"],
            "model_sha256": terminal["model_sha256"],
        }
        predictions = {}
        for ref in key["references"]:
            tag = ref["blind_id"]
            path = forecasts / task["key"] / (tag + ".json")
            saved = read(path)
            if saved["id"] != tag or saved["binding"] != bound:
                raise ValueError(
                    "prediction is bound to a different freeze, model or test input"
                )
            if saved["status"] not in ("complete", "unresolved"):
                raise ValueError("invalid prediction status")
            hashes[f"{task['key']}/{tag}"] = fingerprint(saved)
            predictions[tag] = saved
        result = official.score(
            predictions,
            cell,
            key["references"],
            cell["condition_sd"],
            lambda pred, spec, pair, mode=row["semantics"]: forecast_pair(
                official, mode, pred, pair
            ),
        )
        result.pop(
            "evaluation_seconds"
        )  # A replay of identical curves has an identical report.
        rows.append({**row, **result})
    return {
        "format": "t2-noise-scores-1",
        "freeze_sha256": fingerprint(frozen),
        "answer_key_sha256": fingerprint(key),
        "forecast_hashes": hashes,
        "rows": rows,
    }


def forecast_pair(official, semantics: str, predictions: dict, pair: dict):
    """Check numerical reproducibility using Full's gate before computing metrics."""
    record = predictions[pair["blind_id"]]
    if record["status"] == "unresolved":
        raise official.SolverFailure(record["reason"])
    methods = ["Radau", "BDF"] if semantics == "continuous_ode" else ["native"]
    sets = []
    for part in ("perturbed", "control"):
        block = record["curves"][part]
        if set(block) != set(methods):
            raise ValueError("wrong solvers or native update for the frozen method")
        values = []
        for m in methods:
            curve = checked_curve(block[m], len(pair[part]["time"]))
            values.append({c: np.asarray(curve[c], dtype=float) for c in TARGETS})
        sets.append(values)
    primary, control = sets
    if semantics == "continuous_ode":
        return official.checked(pair, primary, control)
    return primary[0], control[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--key", type=Path, default=Path("private/answer-key.json"))
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--forecasts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = evaluate(a.key, a.freeze, a.forecasts)
    publish(a.out / "scores.json", result)
    stream = io.StringIO()
    rows = [
        {k: v for k, v in row.items() if k != "trajectories"} for row in result["rows"]
    ]
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    # This convenience view can always be regenerated from the immutable JSON.
    (a.out / "scores.csv").write_bytes(stream.getvalue().encode())
    print(
        json.dumps(
            {
                "planned": len(rows),
                "complete": sum(r["evaluation_status"] == "complete" for r in rows),
                "scores": str(a.out / "scores.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
