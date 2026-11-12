"""Constant-initial-value transport smoke example. Exclude from scientific results."""

import json
from pathlib import Path


def fit(task: dict, cell_dir: Path, task_dir: Path, config: dict) -> dict:
    if config["method"] != "smoke_native":
        raise ValueError("this adapter is only a handoff smoke test")
    train, val = [
        json.loads((cell_dir / f).read_bytes())["value"]["rows"]
        for f in ("train.json", "val.json")
    ]
    scores = []
    for channel in ("Gp", "I"):
        values = [y for r in train for y in r["targets"][channel]]
        mean = sum(values) / len(values)
        variance = sum((y - mean) ** 2 for y in values) / len(values)
        for row in val:
            y = row["targets"][channel]
            scores.append(sum((x - y[0]) ** 2 for x in y) / len(y) / variance)
    (task_dir / "model.bin").write_bytes(b"initial-value persistence, v1\n")
    receipt = {
        "cell": task["cell"],
        "seed": task["seed"],
        "public_files": task["public_files"],
        "selection": "fixed smoke example",
    }
    (task_dir / "fit_receipt.json").write_text(json.dumps(receipt))
    return {"status": "ready", "validation_nmse": sum(scores) / len(scores)}


def predict(model_path: Path, inputs: dict, solver: str, config: dict) -> dict:
    if (
        solver != "native"
        or model_path.read_bytes() != b"initial-value persistence, v1\n"
    ):
        raise ValueError("wrong smoke model or native update")
    return {c: [v] * len(inputs["time"]) for c, v in inputs["targets0"].items()}
