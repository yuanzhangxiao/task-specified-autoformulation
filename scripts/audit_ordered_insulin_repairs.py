"""Replay the 18 matched repairs using only their frozen training inputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import count
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.ordered_driver_certificate import certify
from autoformalism.research.trajectory_function_revision import permitted_sources
from autoformalism.search.topology_compatibility import TrainingCompatibilityEvidence
from scripts import pointwise_repair_trial as panel
from scripts import trajectory_functions as trial

GENERATION = "cf4767812bd09006295142790aec22b874d7241ffe42d456309c3df8e1f2372a"


def single_source_checks(
    graph: dict,
    formula: dict,
    target: str,
    case: dict,
    evidence: TrainingCompatibilityEvidence,
    tolerance: float,
) -> dict:
    """Try adding one linear term for each permitted, independently supplied history."""
    ctx = evidence.context
    supplied = set(ctx.external_inputs) | set(ctx.auxiliaries)
    allowed = supplied & set(permitted_sources(graph, case)) & set(case["histories"])
    choices = sorted(
        allowed - graph["variable_types"].keys() - set(graph["sources"][target])
    )
    used = (
        supplied
        | set(ctx.fixed_covariates)
        | set(permitted_sources(graph, case))
        | formula["parameters"].keys()
    )
    gain = next(f"order_gain{i}" for i in count() if f"order_gain{i}" not in used)
    results = {}
    for source in choices:
        sources = {
            **graph["sources"],
            target: sorted([*graph["sources"][target], source]),
        }
        rhs = f"({formula['equations'][target]})+{gain}*{source}"
        changed = {
            **formula,
            "equations": {**formula["equations"], target: rhs},
            "parameters": {**formula["parameters"], gain: 1.0},
        }
        results[source] = certify(
            {**graph, "sources": sources},
            changed,
            target,
            evidence,
            tolerance=tolerance,
        )
    return results


def audit(root: Path, tolerance: float = 1.0) -> dict:
    """Check the three originally admitted repairs and preserve earlier decisions."""
    plan = sealed_read(root / "plan.json")
    intake = sealed_read(root / "intake.json")
    freeze = trial._frozen_candidates(root, plan)
    if (
        freeze["artifact_sha256"] != GENERATION
        or plan["protocol"] != panel.PROTOCOL
        or plan["intake_sha256"] != intake["artifact_sha256"]
        or intake["artifact_sha256"] != panel.INTAKE
    ):
        raise ValueError("Expected the original sealed 18-repair training panel")

    rows = []
    for index, (slot, seed, condition) in enumerate(panel.slots()):
        candidate = trial._read_candidate(root, plan, index)
        receipt = candidate["artifact_sha256"]
        if receipt != freeze["candidates"][index] or (
            candidate["parent_slot"],
            candidate["seed"],
            candidate["condition"],
        ) != (slot, seed, condition):
            raise ValueError("Candidate receipt differs from the registered repair")
        row = {
            "index": index,
            "parent_slot": slot,
            "seed": seed,
            "condition": condition,
            "receipt": receipt,
            "original_status": candidate["status"],
            "audit_status": "previously_ineligible",
        }
        if candidate["status"] == "eligible":
            parent = intake["parents"][str(slot)]
            context = panel.training_case(intake, parent)
            evidence = TrainingCompatibilityEvidence.model_validate(context["evidence"])
            proof = certify(
                candidate["graph_dict"],
                candidate["formula"],
                panel.TARGET,
                evidence,
                tolerance=tolerance,
            )
            row.update(audit_status=proof["status"], proof=proof)
            if proof["status"] == "conflict":
                row["with_one_extra_supplied_source"] = single_source_checks(
                    candidate["graph_dict"],
                    candidate["formula"],
                    panel.TARGET,
                    context,
                    evidence,
                    tolerance,
                )
        rows.append(row)
    return {
        "protocol": "ordered-driver-repair-replay-1",
        "source_generation": freeze["artifact_sha256"],
        "training_intake": intake["artifact_sha256"],
        "target": panel.TARGET,
        "tolerance": tolerance,
        "counts": dict(Counter(r["audit_status"] for r in rows)),
        "rows": rows,
        "optimizer_calls": 0,
        "llm_calls": 0,
        "validation_opened": False,
        "test_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.generation.resolve()):
        parser.error("Use an output path outside the frozen generation")
    result = sealed_write(args.output, audit(args.generation, args.tolerance))
    print(json.dumps({"counts": result["counts"], "sha256": result["artifact_sha256"]}))


if __name__ == "__main__":
    main()
