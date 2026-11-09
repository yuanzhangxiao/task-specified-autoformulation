"""Render fitted seed-zero equations from the frozen paired anchor run."""

from __future__ import annotations

import argparse
from math import isfinite
from pathlib import Path

from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.rebuttal.iclr_search_controls import ARMS
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.schemas import CandidateModel
from scripts.iclr_anchor_fidelity import verify_controlled_bundle

ANCHORS = {"acquisition_00": "T2-hard", "sensor_00": "CSTR-hard"}


def model_lines(row: dict) -> dict[str, str]:
    """Read equations, processes and causal initializers from the selected model."""
    candidate = CandidateModel.model_validate(row["candidate"])
    if content_sha256(candidate) != row["model_sha256"]:
        raise ValueError("candidate differs from selected model digest")
    fitted = row["parameters"]
    if set(fitted) != {p.name for p in candidate.parameters} or any(
        type(v) not in (int, float) or not isfinite(v) for v in fitted.values()
    ):
        raise ValueError("selected coefficients do not match candidate parameters")
    lines = {
        f"state {s.name}": f"{s.kind.value}, {s.unit}, {s.description}"
        for s in candidate.states
    }
    lines.update({f"d({e.state})/dt": e.rhs for e in candidate.state_equations})
    lines.update({f"process {p.name}": p.expression for p in candidate.processes})
    lines.update(
        {f"observe {m.channel}": m.expression for m in candidate.observation_mappings}
    )
    for ic in candidate.initial_conditions:
        rule = ic.expression
        if rule is None:
            if ic.fixed_value is not None:
                rule = str(ic.fixed_value)
            elif ic.initialization_range is not None:
                bounds = ic.initialization_range
                rule = f"range [{bounds.lower}, {bounds.upper}]"
            else:
                rule = "unspecified"
        lines[f"initial {ic.state}"] = f"{rule} ({ic.scope.value})"
    lines.update(
        {f"coefficient {p.name}": repr(fitted[p.name]) for p in candidate.parameters}
    )
    return lines


def render(root: Path) -> str:
    """Authenticate the roster and compare every arm on both hard anchors."""
    source = root / "frozen_models.json"
    bundle = sealed_read(source)
    if bundle["protocol"] != "iclr-controlled-search-models-1":
        raise ValueError("expected frozen controlled-search model export")
    verify_controlled_bundle(source, bundle)
    plan = sealed_read(root / "plan.json")
    tasks = {t["task_id"]: t for b in plan["blocks"] for t in b["tasks"]}
    selected = {}
    for row in bundle["rows"]:
        key = (row["benchmark_id"], row["repetition"], row["method"])
        task = tasks[row["source_id"]]
        if key != (task["cell"], task["seed"], task["treatment"]) or key in selected:
            raise ValueError("frozen row differs from planned lineage")
        selected[key] = row
    expected = {(c, s, a) for c in ANCHORS for s in (0, 1, 2) for a in ARMS}
    if set(selected) != expected:
        raise ValueError("expected complete two-anchor, three-seed, eight-arm campaign")
    out = [
        "# Paired anchor equations for seed 0",
        "",
        "Each comparison uses the Full arm from the same cell. Equation differences "
        "use the model's own variable names. Coefficients give fitted values.",
        "",
        f"Frozen roster `{bundle['artifact_sha256']}`  ",
        f"Plan `{bundle['plan_sha256']}`  ",
        f"Selection freeze `{bundle['selection_freeze_sha256']}`",
    ]
    for cell, label in ANCHORS.items():
        out.extend(["", f"## {label} (`{cell}`)"])
        full = selected[cell, 0, "full"]
        baseline = model_lines(full) if full["status"] == "ready" else None
        for arm in ARMS:
            row = selected[cell, 0, arm]
            out.extend(
                [
                    "",
                    f"### {arm}",
                    "",
                    f"Task `{row['source_id']}`  ",
                    f"Endpoint `{row['source_sha256']}`",
                    "",
                ]
            )
            if row["status"] == "no_retained_model":
                out.append(
                    "No retained model. Final visit status "
                    f"`{row['last_visit_status']}`."
                )
                continue
            if row["status"] != "ready":
                raise ValueError("unknown terminal status")
            out.extend(
                [
                    f"Model `{row['model_sha256']}`  ",
                    f"Validation NMSE `{row['validation_nmse']}`",
                    "",
                ]
            )
            current = model_lines(row)
            if arm == "full" or baseline is None:
                if arm != "full":
                    out.extend(
                        [
                            "Full produced no retained model.",
                            "",
                        ]
                    )
                diff = [f"{k} = {v}" for k, v in current.items()]
            else:
                diff = []
                for k in dict.fromkeys((*baseline, *current)):
                    if baseline.get(k) != current.get(k):
                        if k in baseline:
                            diff.append(f"- {k} = {baseline[k]}")
                        if k in current:
                            diff.append(f"+ {k} = {current[k]}")
                if diff:
                    out.extend(["Changes from Full (`-` removed, `+` added)", ""])
            if diff:
                # Indent every line to keep model-generated text inside Markdown code.
                for line in diff:
                    out.extend(f"    {part}" for part in line.splitlines())
            else:
                out.append(
                    "All displayed model terms and coefficients match Full."
                )
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = render(args.search)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result, encoding="utf-8")
