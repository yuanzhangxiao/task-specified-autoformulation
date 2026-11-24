"""Replay the matched insulin repairs and build a shortlist through live admission."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.search.topology_compatibility import TrainingCompatibilityEvidence
from autoformalism.staged_topology import content_hash
from scripts import audit_ordered_insulin_repairs as archived
from scripts import pointwise_repair_trial as panel


def summarize(result: dict) -> dict:
    """Keep eligibility distinct from a proof or measurement of accuracy."""
    target = panel.TARGET
    if result["status"] in {"eligible", "pointwise_conflict"} and (
        target not in result.get("pointwise_checks", {})
        or target not in result.get("ordered_checks", {})
    ):
        raise ValueError("Missing pointwise decision on the candidate")
    return {
        "status": result["status"],
        "interval_check": result.get("pointwise_checks", {}).get(target),
        "order_check": result.get("ordered_checks", {}).get(target),
    }


def screen_extensions(
    intake: dict, starting_models: list[tuple[int, dict, dict, dict]]
) -> dict:
    """Assemble complete target edits and keep only models admitted by the runtime."""
    edits, menu = [], []
    target = panel.TARGET
    for slot, origin, base_graph, base_formula in starting_models:
        parent = intake["parents"][str(slot)]
        case = panel.training_case(intake, parent)
        evidence = TrainingCompatibilityEvidence.model_validate(case["evidence"])
        for source, graph, formula in archived.linear_source_extensions(
            base_graph, base_formula, target, case, evidence
        ):
            rhs = formula["equations"][target]
            new_params = {
                name: v
                for name, v in formula["parameters"].items()
                if name not in parent["formula"]["parameters"]
            }
            patch = {
                "rhs": rhs,
                "new_parameters": [
                    {"name": name, "initial": val}
                    for name, val in sorted(new_params.items())
                ],
            }
            reviewed = panel.assess(intake, slot, patch)
            decision = {
                **origin,
                "source_added": source,
                **summarize(reviewed),
            }
            if reviewed["status"] == "eligible":
                if reviewed["graph_dict"] != graph or reviewed["formula"] != formula:
                    raise ValueError("Admitted model differs from the generated edit")
                model = {"graph": graph, "formula": formula}
                model_id = content_hash(model)
                decision["model_id"] = model_id
                menu.append(
                    {
                        "model_id": model_id,
                        **origin,
                        "added_source": source,
                        "target_rhs": rhs,
                        "model": model,
                    }
                )
            edits.append(decision)
    return {
        "extensions": edits,
        "shortlist": menu,
        "extension_counts": dict(Counter(r["status"] for r in edits)),
    }


def screen_parents(intake: dict) -> dict:
    """Build options directly from the three frozen parents without reading replies."""
    if intake["artifact_sha256"] != panel.INTAKE:
        raise ValueError("Expected the registered training parents and evidence")
    bases = [
        (int(slot), {"parent_slot": int(slot)}, parent["graph"], parent["formula"])
        for slot, parent in sorted(
            intake["parents"].items(), key=lambda row: int(row[0])
        )
    ]
    return {
        "protocol": "parent-repair-options-1",
        "training_intake": panel.INTAKE,
        "target": panel.TARGET,
        "tolerance": panel.TOLERANCE,
        **screen_extensions(intake, bases),
        "scope": "Single additional linear supplied source per parent. "
        "Accuracy and physiological validity require separate evaluation.",
        "llm_calls": 0,
        "optimizer_calls": 0,
        "validation_opened": False,
        "test_opened": False,
    }


def screen(root: Path) -> dict:
    """Give the recorded edits and generated alternatives the same admission path."""
    intake, records = archived.frozen_panel(root)
    rows, bases = [], []
    target = panel.TARGET
    for record in records:
        index, slot = record["index"], record["parent_slot"]
        reply = sealed_read(root / "steps" / f"slot-{index:03}" / "reply.json")
        if reply["artifact_sha256"] != record["reply_sha256"]:
            raise ValueError("Reply differs from the authenticated candidate")
        replay = panel.assess(intake, slot, reply["value"])
        row = {
            "index": index,
            "parent_slot": slot,
            "receipt": record["artifact_sha256"],
            "original_status": record["status"],
            **summarize(replay),
        }
        rows.append(row)
        if (
            record["status"] != "eligible"
            or (row["order_check"] or {}).get("status") != "conflict"
        ):
            continue
        bases.append(
            (
                slot,
                {"parent_slot": slot, "from_reply": index},
                record["graph_dict"],
                record["formula"],
            )
        )
    return {
        "protocol": "ordered-repair-options-1",
        "original_generation": archived.GENERATION,
        "training_intake": intake["artifact_sha256"],
        "target": target,
        "tolerance": panel.TOLERANCE,
        "replies": rows,
        **screen_extensions(intake, bases),
        "reply_counts": dict(Counter(r["status"] for r in rows)),
        "scope": "One added linear supplied source per originally eligible reply "
        "rejected by the ordering proof. Shortlisted models have no contradiction "
        "found by the applicable checks. Their accuracy is unmeasured.",
        "optimizer_calls": 0,
        "llm_calls": 0,
        "validation_opened": False,
        "test_opened": False,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    source = cli.add_mutually_exclusive_group(required=True)
    source.add_argument("--generation", type=Path, help="Replay 18 original replies")
    source.add_argument(
        "--intake", type=Path, help="Generate directly from the registered parents"
    )
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    frozen_root = args.generation if args.generation else args.intake.parent
    if args.output.resolve().is_relative_to(frozen_root.resolve()):
        cli.error("Write the new screen outside the original generation directory")
    data = (
        screen(args.generation)
        if args.generation
        else screen_parents(sealed_read(args.intake))
    )
    result = sealed_write(args.output, data)
    print(
        json.dumps(
            {k: result[k] for k in ("reply_counts", "extension_counts") if k in result}
        )
    )
