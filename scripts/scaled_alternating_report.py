"""Read-only, stdlib-only report; safe to run without numerical packages."""

import hashlib
import json
from collections import Counter
from pathlib import Path


def report(output: Path):
    frozen = json.loads((output / "freeze.json").read_text())
    rows = []
    for index, arm in enumerate(frozen["tasks"]):
        path = output / f"results/task_{index:03d}/result.json"
        if path.exists():
            row = json.loads(path.read_text())
            payload = {"freeze": frozen["identity"], "arm": arm}
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest()
            if row["identity"] != digest:
                raise ValueError("result identity differs")
        else:
            row = {"arm": arm, "status": "missing"}
        rows.append(row)
    gate_path = output / "gate/result.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    counts = dict(Counter(r["status"] for r in rows))
    lines = [
        "# Scaled joint and alternating collocation comparison",
        "",
        f"Planned arms: 2. Status counts: {counts}. Gate passed: {gate.get('pass')}.",
        "",
        (
            "J+S = joint penalized collocation then sensitivity refinement. A+S "
            "= bounded linear weights alternating with nodes/shapes, then the "
            "same refinement."
        ),
        (
            "Same ordinary start, frozen nodes/units, all observations, "
            "physical boundaries and total budgets. Soft defects differ from "
            "earlier exact-C constraints."
        ),
        (
            "NMSE scores only v01. Replay agreement checks numerical "
            "consistency, not prediction accuracy. No automatic follow-up."
        ),
        "",
        (
            "| Arm | Status | Initializer native success | Surrogate objective "
            "| Selected | Train NMSE | Validation NMSE | Replay | Practical |"
        ),
        "| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for r in rows:
        c = (r.get("initializer") or {}).get("initializer") or {}
        rp = r.get("replay", {})
        selected = r.get("selected", {})
        passed = all(rp.get(s, {}).get("pass", False) for s in ("train", "val"))
        values = [
            r["arm"],
            r["status"],
            c.get("native_success"),
            c.get("objective"),
            selected.get("source"),
            rp.get("train", {}).get("nmse"),
            rp.get("val", {}).get("nmse"),
            passed,
            r.get("practical"),
        ]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    result = {"counts": counts, "gate": gate, "rows": rows}
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("summary.json", json.dumps(result, indent=2) + "\n"),
        ("SUMMARY.md", "\n".join(lines) + "\n"),
    ):
        path = output / name
        temp = path.with_suffix(".tmp")
        temp.write_text(value)
        temp.replace(path)
    return result
