"""Compare one 20B insulin-rate repair with and without its measured conflict."""

from __future__ import annotations

import argparse
import ast
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Annotated

from pydantic import Field

from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research.affine_direction_certificate import certify
from autoformalism.research.construction_revision_edits import _canonical_rhs
from autoformalism.research.trajectory_function_revision import (
    _check_graph,
    _feedback,
    assess_fixed,
    assess_revision,
    permitted_sources,
)
from autoformalism.schemas.base import StrictSchema
from autoformalism.search.topology_compatibility import TrainingCompatibilityEvidence
from autoformalism.staged_topology import content_hash
from scripts import t2_breadth as pilot
from scripts import trajectory_functions as trial
from scripts.construction_functions import file_hash
from scripts.submit_trajectory_functions import ENTRYPOINTS
from scripts.trajectory_function_prompts import Parameter

PROTOCOL = "pointwise-repair-witness-1"
SOURCE = "4b6df6d0fefa5fb0fabe0c445c2100a319bbd0d987aeb9d02b2c7bdf84090dd3"
INTAKE = "22213dcee97375a06dfb628a7c4d6a28416bd378545b9acca808136cc7c71c74"
PARENTS = (2, 14, 15)
SEEDS = (211, 212, 213)
CONDITIONS = ("control", "witness")
TARGET = "I"
DATASET = "phase_c_dalla_man_t2_canonical_named_easy_rates_v1"
TOLERANCE = 1.0
FILES = (
    *ENTRYPOINTS,
    "scripts/pointwise_repair_trial.py",
    "scripts/hpc/start_pointwise_repair.sh",
)
SETTINGS = trial.SETTINGS.model_copy(update={"max_output_tokens": 3072})


class Repair(StrictSchema):
    """Replace only the plasma insulin derivative and declare new coefficients."""

    rhs: Annotated[str, Field(min_length=1, max_length=4096)]
    new_parameters: Annotated[list[Parameter], Field(max_length=24)]


SYSTEM = """Revise only the RHS of the named differential state. Its parent
failed the marked training accuracy requirement for every choice of shared
parameters. Propose a scientifically justified rate law and choose its sources
from the permitted list. The runtime merges your edit into the parent, infers
the new dependencies and preserves all other equations and initializers.
Return one scalar rhs and new_parameters. You can reuse parent coefficients by
name. Declare each genuinely new shared coefficient once with a finite initial
guess. Use numbers, variable and coefficient names, +, -, *, /, integer powers
** from -16 to 16, and abs, exp, log, sqrt, min, max, sin, tanh, sigmoid or
softplus. The independent variable is time in minutes. Use generated variables
at their current model values. A change of parameter names or starting values
alone cannot resolve the conflict."""


def slots() -> list[tuple[int, int, str]]:
    """Pair conditions at each parent and seed, alternating which runs first."""
    return [
        (parent, seed, condition)
        for j, seed in enumerate(SEEDS)
        for k, parent in enumerate(PARENTS)
        for condition in (CONDITIONS if (j + k) % 2 == 0 else CONDITIONS[::-1])
    ]


def training_case(data: dict, parent: dict) -> dict:
    case = data["cases"][parent["case_id"]]
    return {**case, "evidence": {**case["evidence"], "trajectories": data["train"]}}


def read_source(source: Path) -> dict:
    """Verify the registered training-only admission and its three exact witnesses."""
    summary = sealed_read(source / "admission.json")
    plan = sealed_read(source / "plan.json")
    generation = trial._frozen_candidates(source, plan)
    data = sealed_read(source / "intake.json")
    if (
        summary["artifact_sha256"] != SOURCE
        or summary["newly_blocked_fit_indices"] != list(PARENTS)
        or summary["plan"] != plan["artifact_sha256"]
        or summary["generation"] != generation["artifact_sha256"]
        or plan["intake_sha256"] != data["artifact_sha256"]
        or data["pointwise_tolerances"] != {DATASET: {TARGET: TOLERANCE}}
    ):
        raise ValueError("Expected the original sealed three-conflict training cohort")
    parents = {}
    cases = {}
    for index in PARENTS:
        saved = trial._read_candidate(source, plan, index)
        g, _, _ = trial.slots(data)[index]
        row = data["graphs"][g]
        if (
            saved["artifact_sha256"] != generation["candidates"][index]
            or row["dataset"] != DATASET
            or saved["status"] != "pointwise_conflict"
            or saved["archived_status"] != "eligible"
        ):
            raise ValueError("Wrong blocked candidate or graph")
        graph = trial.fitting_graph(saved, row)
        context = trial.training_case(data, row)
        evidence = TrainingCompatibilityEvidence.model_validate(context["evidence"])
        proof = certify(graph, saved["formula"], TARGET, evidence, tolerance=TOLERANCE)
        if proof["status"] != "conflict" or proof != saved["pointwise_checks"][TARGET]:
            raise ValueError("Training direction witness did not reproduce")
        cases[row["case"]] = data["cases"][row["case"]]
        parents[str(index)] = {
            "case_id": row["case"],
            "graph": graph,
            "formula": saved["formula"],
            "archive_candidate_sha256": saved["archived_candidate_sha256"],
            "replayed_candidate_sha256": saved["artifact_sha256"],
            "dataset": DATASET,
            "witness": proof,
        }
    return {
        "source": SOURCE,
        "parents": parents,
        "cases": cases,
        "train": data["train"][DATASET],
    }


def request(data: dict, parent_index: int, condition: str) -> dict:
    """Share all public request fields except the measured witness."""
    if condition not in CONDITIONS:
        raise ValueError("Unknown evidence condition")
    parent = data["parents"][str(parent_index)]
    case, graph, form = (
        data["cases"][parent["case_id"]],
        parent["graph"],
        parent["formula"],
    )
    packet = {
        "scientific_task": case["public_brief"]["scientific_context"],
        "public_requirements": case["public_brief"]["requirements"],
        "edit_target": TARGET,
        "training_requirement": {
            "absolute_error_pmol_per_l": TOLERANCE,
            "requirement": "Match measured plasma insulin at every training sample.",
            "parent_outcome": "Certified conflict for all shared parameters.",
        },
        "variable_types": graph["variable_types"],
        "initialization_rules": graph["initializers"],
        "parent_equations": form["equations"],
        "parent_initializers": form["initializers"],
        "parent_coefficient_names": sorted(form["parameters"]),
        "current_sources": graph["sources"],
        "permitted_sources": sorted(
            set(permitted_sources(graph, case)) | set(graph["sources"][TARGET])
        ),
        "measured_witness": None,
    }
    if condition == "witness":
        w = parent["witness"]
        packet["measured_witness"] = {
            "reference": {
                k: w["reference"][k]
                for k in (
                    "event",
                    "time",
                    "forcing_values",
                    "initial_target",
                    "observed_final",
                    "derivative_bound",
                )
            },
            "driver": w["driver"],
            "driver_side": w["side"],
            **{
                k: {
                    n: w[k][n]
                    for n in (
                        "event",
                        "time",
                        "driver_range",
                        "observed",
                        "required_change",
                        "maximum_permitted_change",
                    )
                }
                for k in ("rise", "fall")
            },
            "interpretation": (
                "The rise rules out a nonpositive slope times the selected side. "
                "The fall rules out a nonnegative slope times that side."
            ),
            "assumptions": [
                "Shared parameters, observed initial insulin, complete exact "
                "piecewise-linear inputs, no resets and no unlisted forcing.",
                *w["reference"]["assumptions"],
            ],
        }
    return packet


def _plan(data: dict) -> dict:
    if data["artifact_sha256"] != INTAKE:
        raise ValueError("The selected training parents or evidence changed")
    schema = Repair.model_json_schema()
    rows = [
        {
            "parent_slot": i,
            "seed": seed,
            "condition": arm,
            "request_sha256": content_hash([SYSTEM, request(data, i, arm), schema]),
        }
        for i, seed, arm in slots()
    ]
    return {
        "protocol": PROTOCOL,
        "source_admission": SOURCE,
        "intake_sha256": data["artifact_sha256"],
        "slots": rows,
        "maximum_generation_requests": len(rows),
        "maximum_fit_slots": len(rows),
        "maximum_optimizer_calls": 0,
        "correction_rounds": 0,
        "settings": SETTINGS.model_dump(mode="json"),
        "runtime_source": runtime_source_hash(),
        "entrypoints": {p: file_hash(pilot.REPO / p) for p in FILES},
        "validation_data_opened": False,
        "test_data_opened": False,
    }


def prepare(root: Path, source: Path) -> dict:
    """Freeze all 18 prompts and parent inputs before generation."""
    value = read_source(source)
    with _lock(root / "prepare-lock"):
        data = sealed_write(root / "intake.json", value)
        return sealed_write(root / "plan.json", _plan(data))


def load(root: Path) -> tuple[dict, dict]:
    data, plan = sealed_read(root / "intake.json"), sealed_read(root / "plan.json")
    if data["source"] != SOURCE or content_hash(_plan(data)) != plan["artifact_sha256"]:
        raise ValueError("Repair campaign training inputs, prompts or code differ")
    return plan, data


def _key(text, private):
    """Compare arithmetic trees up to bijective private-coefficient renaming."""
    parser = RestrictedParser()
    names = {}

    def visit(node):
        if isinstance(node, ast.Name):
            if node.id in private:
                return ("private", names.setdefault(node.id, len(names)))
            return ("name", node.id)
        if isinstance(node, ast.Constant):
            return ("constant", node.value)
        if isinstance(node, ast.UnaryOp):
            item = visit(node.operand)
            return item if isinstance(node.op, ast.UAdd) else ("minus", item)
        if isinstance(node, ast.Call):
            return ("call", node.func.id, tuple(visit(arg) for arg in node.args))
        a, b = visit(node.left), visit(node.right)
        op = type(node.op).__name__
        if op in ("Add", "Mult"):
            a, b = sorted((a, b))
        zero, one = ("constant", 0), ("constant", 1)
        if op == "Add" and zero in (a, b):
            return b if a == zero else a
        if op == "Sub" and b == zero:
            return a
        if op == "Mult" and one in (a, b):
            return b if a == one else a
        if op == "Mult" and a == zero and b[0] in ("name", "private", "constant"):
            return zero
        if op == "Pow" and b == one:
            return a
        return (op, a, b)

    return visit(parser.parse(text, location=TARGET).tree.body)


def assess(data: dict, index: int, raw: dict) -> dict:
    """Merge a target-only change, rerun the complete graph and pointwise checks."""
    patch = Repair.model_validate(raw)
    parent = data["parents"][str(index)]
    form, graph, case = parent["formula"], parent["graph"], training_case(data, parent)
    old = form["equations"][TARGET]
    rhs = _canonical_rhs(
        patch.rhs, TARGET, "dynamic", case["evidence"]["context"]["time_symbol"]
    )
    ledger = []
    if rhs != patch.rhs:
        ledger.append(
            {"kind": "exact_rhs_label_or_whitespace", "before": patch.rhs, "after": rhs}
        )
    parser = RestrictedParser()
    parsed = parser.parse(rhs, location=TARGET)
    symbols = parsed.symbols
    if ast.dump(parsed.tree) != ast.dump(ast.parse(rhs, mode="eval")):
        normalized = ast.unparse(parsed.tree)
        ledger.append(
            {"kind": "current_time_symbol_notation", "before": rhs, "after": normalized}
        )
        rhs = normalized
    additions = {p.name: p.initial for p in patch.new_parameters}
    existing = form["parameters"]
    if (
        len(additions) != len(patch.new_parameters)
        or additions.keys() & existing.keys()
    ):
        raise ValueError(
            "Declare each new coefficient once. Reuse existing coefficients by name."
        )
    if extra := additions.keys() - symbols:
        raise ValueError(
            f"Declared new coefficients absent from the edited RHS: {sorted(extra)}"
        )
    equations = {**form["equations"], TARGET: rhs}
    other = set()
    other_laws = {k: v for k, v in form["equations"].items() if k != TARGET}
    for key, text in (*other_laws.items(), *form["initializers"].items()):
        other.update(parser.parse(text, location=key).symbols)
    parameters = {
        k: v for k, v in (existing | additions).items() if k in symbols or k in other
    }
    proposed = {
        "equations": equations,
        "initializers": form["initializers"],
        "parameters": parameters,
    }
    private_old = existing.keys() - other
    private_new = parameters.keys() - other
    details = {
        "authored_rhs": patch.rhs,
        "effective_rhs": rhs,
        "normalizations": ledger,
        "removed_unused_parent_parameters": sorted(existing.keys() - parameters.keys()),
        "new_parameters": additions,
    }
    outcome = assess_revision(
        graph,
        case,
        proposed,
        "Revise the insulin rate to address the proven training conflict.",
    )
    if outcome["status"] == "no_topology_change":
        outcome = assess_fixed(graph, case, proposed)
        if outcome["status"] == "eligible":
            topology = _check_graph(
                graph, case, "Verify the retained topology against the public task."
            )
            if topology["status"] != "topology_screen_passed":
                return {
                    **details,
                    "status": topology["status"],
                    "reason": _feedback(topology),
                    "graph_review": topology,
                }
            outcome["assessment"] = topology
    if outcome["status"] != "eligible":
        return {**details, **outcome}
    old_key, new_key = _key(old, private_old), _key(rhs, private_new)
    details["parent_rhs"] = old
    details["normalized_target_rate_sha256"] = content_hash(new_key)
    if old_key == new_key:
        return {
            **details,
            "status": "no_scientific_change",
            "reason": "Identical rate up to elementary identities and local "
            "coefficient renaming.",
        }
    child = outcome.pop("graph")
    details["source_change"] = {
        "before": graph["sources"][TARGET],
        "after": child["sources"][TARGET],
    }
    result = trial.pointwise_admission(
        {
            "profile": trial.TARGET_RHS,
            "cases": data["cases"],
            "train": {DATASET: data["train"]},
            "pointwise_tolerances": {DATASET: {TARGET: TOLERANCE}},
        },
        {
            "dataset": DATASET,
            "case": parent["case_id"],
            "graph": child,
            "structure_sha256": content_hash(child),
        },
        {
            "graph": content_hash(child),
            "formula": outcome["formula"],
            "status": "eligible",
        },
    )
    return {
        **details,
        **outcome,
        "graph_dict": child,
        "status": result["status"],
        "pointwise_checks": result["pointwise_checks"],
        "ordered_checks": result["ordered_checks"],
        **(
            {"repair_feedback": result["repair_feedback"]}
            if "repair_feedback" in result
            else {}
        ),
        "pointwise_outcome": "conflict_proved"
        if result["status"] == "pointwise_conflict"
        else "no_conflict_proved",
        "trajectory_accuracy": "unmeasured",
    }


def run(
    root: Path,
    endpoint: str,
    *,
    runtime_record=None,
    transport=None,
    token_transport=None,
) -> dict:
    from scripts.submit_t2_breadth import bind

    plan, data = load(root)
    if transport is None:
        if runtime_record is None:
            raise ValueError("Live generation requires the sealed 20B server record")
        bind(root, Path(runtime_record), endpoint)
    ids = []
    with _lock(root / "generation-lock"):
        for j, (i, seed, arm) in enumerate(slots()):
            slot = f"slot-{j:03}"
            reply = pilot._ask(
                root,
                plan,
                slot,
                seed,
                SYSTEM,
                request(data, i, arm),
                Repair,
                endpoint,
                transport,
                token_transport,
                {"physical_requests": 0, "slots": {}},
                len(slots()),
                sampling_step=f"insulin-repair-{i}",
            )
            record = {
                "index": j,
                "plan": plan["artifact_sha256"],
                "parent_slot": i,
                "seed": seed,
                "condition": arm,
                "reply_sha256": reply["artifact_sha256"],
                "status": reply["status"],
            }
            if reply["status"] == "visible":
                record["attempted_patch"] = reply["value"]
                try:
                    record.update(assess(data, i, reply["value"]))
                except (ValueError, ModelValidationError, RecursionError) as error:
                    record.update(status="invalid_patch", reason=str(error))
            elif reply.get("error"):
                record["reason"] = reply["error"]
            row = sealed_write(root / "candidates" / f"{j:03}.json", record)
            ids.append(row["artifact_sha256"])
        sealed_write(
            root / "generation.json",
            {"plan": plan["artifact_sha256"], "candidates": ids},
        )
    return status(root)


def verified_records(root: Path) -> list[dict]:
    """Verify identities and physical requests for all 18 terminal records."""
    plan, _ = load(root)
    freeze = trial._frozen_candidates(root, plan)
    records = []
    for j, (i, seed, arm) in enumerate(slots()):
        row = trial._read_candidate(root, plan, j)
        if row["artifact_sha256"] != freeze["candidates"][j] or (
            row["parent_slot"],
            row["seed"],
            row["condition"],
        ) != (i, seed, arm):
            raise ValueError("Candidate differs from its matched slot")
        records.append(row)
    return records


def status(root: Path) -> dict:
    plan, _ = load(root)
    done = (root / "generation.json").is_file()
    observed = (
        verified_records(root)
        if done
        else [
            trial._read_candidate(root, plan, j)
            for j in range(len(slots()))
            if (root / "candidates" / f"{j:03}.json").is_file()
        ]
    )
    physical = [
        json.loads(p.read_text()) for p in (root / "steps").glob("*/calls/*.json")
    ]
    return {
        "status": "complete" if done else "incomplete",
        "recorded": len(observed),
        "expected": len(slots()),
        "condition_statuses": {
            arm: dict(Counter(r["status"] for r in observed if r["condition"] == arm))
            for arm in CONDITIONS
        },
        "physical_requests": len(physical),
        "observed_tokens": sum(r.get("observed_total_tokens") or 0 for r in physical),
        "requests_with_unknown_usage": sum(
            r.get("observed_total_tokens") is None for r in physical
        ),
        "optimizer_calls": 0,
        "validation_data_opened": False,
        "test_data_opened": False,
    }


def inspect(root: Path) -> dict:
    result = status(root)
    if result["status"] != "complete":
        return result
    output = root / "inspection.tar.gz"
    with _lock(root / "inspect-lock"):
        temporary = output.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz") as archive:
            for file in [
                *(root / n for n in ("plan.json", "intake.json", "generation.json")),
                *sorted(root.glob("candidates/*.json")),
                *sorted(root.glob("steps/*/reply.json")),
                *sorted(root.glob("steps/*/calls/*.json")),
            ]:
                archive.add(file, arcname=file.relative_to(root), recursive=False)
        temporary.replace(output)
    return {**result, "archive": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status", "inspect"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--runtime-record", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        result = prepare(args.root, args.source)
    elif args.action == "run":
        if args.endpoint is None or args.runtime_record is None:
            parser.error("run requires --endpoint and --runtime-record")
        result = run(args.root, args.endpoint, runtime_record=args.runtime_record)
    else:
        result = inspect(args.root) if args.action == "inspect" else status(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
