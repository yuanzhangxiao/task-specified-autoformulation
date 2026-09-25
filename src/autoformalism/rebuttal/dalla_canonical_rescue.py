"""Reference-informed, sign-only paired rescue of explicit canonical starts.

This development diagnostic can correct an already constrained assembly sign.
It never infers a direction from fitted values or claims autonomous discovery.
"""

from __future__ import annotations

import ast
import copy
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.fitting.collocation_sensitivity import _role_start
from autoformalism.rebuttal import dalla_rescue as rescue
from autoformalism.rebuttal import dalla_sign_diagnostic as diagnostic
from autoformalism.rebuttal.dalla_sign_repair import sign_integrity
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.candidate import ParameterRole
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit
from autoformalism.schemas.staged import InteractionPolarity
from autoformalism.search import sign_review
from autoformalism.sign_contract import (
    analyze_outer_weight,
    strip_outer_negative_factors,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "dalla-canonical-sign-rescue-1"
SOURCE_PROTOCOL = "dalla-canonical-sign-sources-1"
REPO = rescue.REPO
LABEL = "assistant_specified_reference_informed_diagnostic"
LIMITATION = (
    "Post-hoc, reference-informed direction hypotheses on preselected historical "
    "models, not autonomous discovery. Anonymous channel interpretation is supplied "
    "by the analyst. No reference coefficients or nonlinear laws are inserted. "
    "Training fits parameters and ranks deletions; validation accepts pruning. "
    "Initializers and inner laws are unchanged. No intervention/test scores enter "
    "this run. Endpoint intervention performance remains unestablished."
)


class Edit(StrictSchema):
    """An explicit, identity-bound whole-term direction, not a semantic verdict."""

    state: str
    parameter: str
    original_expression: str
    sign: Literal["positive", "negative"]
    rationale: str = Field(min_length=12)


class ModelDecision(StrictSchema):
    task_id: str = Field(pattern=r"^[a-z0-9_]+$")
    source_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assumptions: tuple[str, ...] = Field(min_length=1)
    edits: tuple[Edit, ...] = Field(min_length=1, max_length=64)


class Decisions(StrictSchema):
    protocol: Literal["dalla-canonical-sign-rescue-1"] = PROTOCOL
    decision_source: Literal["assistant_specified_reference_informed_diagnostic"] = (
        LABEL
    )
    models: tuple[ModelDecision, ...] = Field(min_length=1, max_length=6)


def launcher_hash() -> str:
    return content_hash(
        {
            "diagnostic": diagnostic.launcher_hash(),
            "builder": diagnostic._digest(
                REPO / "scripts/build_dalla_canonical_sources.py"
            ),
            "launcher": diagnostic._digest(
                REPO / "scripts/hpc/launch_dalla_canonical_rescue.sh"
            ),
            "smoke": diagnostic._digest(
                REPO / "scripts/smoke_dalla_canonical_rescue.py"
            ),
        }
    )


def apply(request: PublicFitRequest, edits: tuple[Edit, ...]) -> dict:
    """Change only isolated numerator-gain assembly signs and magnitude domains."""
    changed = copy.deepcopy(request.model_dump(mode="json"))
    candidate = changed["base_candidate"]
    specs = {p["name"]: p for p in candidate["parameters"]}
    roles = {p.name: p.role for p in request.base_candidate.parameters}
    expressions = [e["rhs"] for e in candidate["state_equations"]]
    expressions += [p["expression"] for p in candidate["processes"]]
    expressions += [p["expression"] for p in candidate["observation_mappings"]]
    uses = Counter(
        n.id
        for expression in expressions
        for n in ast.walk(sign_review._parse(expression))
        if isinstance(n, ast.Name)
    )
    protected = str(request.initialization_plan.model_dump(mode="json"))
    protected += str(candidate["initial_conditions"]) + str(candidate["constraints"])
    pending = {e.parameter: e for e in edits}
    if len(pending) != len(edits):
        raise ValueError("duplicate gain decision")
    records, reset = [], []
    for i, equation in enumerate(candidate["state_equations"]):
        nodes = sign_review._terms(sign_review._parse(equation["rhs"]).body)
        touched = False
        for j, node in enumerate(nodes):
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            chosen = names & pending.keys()
            if not chosen:
                continue
            if len(chosen) != 1:
                raise ValueError("ambiguous term decision")
            name = chosen.pop()
            edit = pending.pop(name)
            spec = specs.get(name)
            if (
                spec is None
                or uses[name] != 1
                or spec["role"] not in {"coefficient", "nonnegative_coefficient"}
                or spec["domain"] not in {"real", "nonnegative"}
                or spec["bounds"] is not None
                or spec["initialization_range"] is not None
                or re.search(rf"\b{re.escape(name)}\b", protected)
            ):
                raise ValueError("gain is shared, bounded or protected")
            if equation["state"] != edit.state or ast.dump(node) != ast.dump(
                sign_review._parse(edit.original_expression).body
            ):
                raise ValueError("decision does not match exact original term")
            normalized, count = strip_outer_negative_factors(ast.Expression(body=node))
            # Identify even an already constrained gain, without relaxing the model.
            analysis = analyze_outer_weight(
                normalized,
                {**roles, name: ParameterRole.COEFFICIENT},
                InteractionPolarity.POSITIVE,
            )
            if analysis.diagnostic_code or analysis.identified_parameter != name:
                raise ValueError("requires one identified outer numerator gain")
            old_sign = "negative" if count % 2 else "positive"
            records.append(
                {
                    "state": edit.state,
                    "parameter": name,
                    "slot_id": f"state_{i}_term_{j}",
                    "original_expression": edit.original_expression,
                    "original_outer_sign": old_sign,
                    "original_domain": spec["domain"],
                    "normalized_expression": ast.unparse(normalized),
                    "decision": {
                        "outer_weight_sign": edit.sign,
                        "rationale": edit.rationale,
                    },
                }
            )
            if old_sign != edit.sign or spec["role"] != "nonnegative_coefficient":
                reset.append(name)
                changed["parameter_guesses"].pop(name, None)
                value = normalized.body
                nodes[j] = (
                    ast.UnaryOp(op=ast.USub(), operand=value)
                    if edit.sign == "negative"
                    else value
                )
                spec.update(role="nonnegative_coefficient", domain="nonnegative")
                touched = True
        if touched:
            joined = nodes[0]
            for node in nodes[1:]:
                joined = ast.BinOp(left=joined, op=ast.Add(), right=node)
            equation["rhs"] = ast.unparse(ast.fix_missing_locations(joined))
    if pending or not reset:
        raise ValueError("unknown gain decision or no actual sign/domain correction")
    provenance = {
        "policy": PROTOCOL,
        "decision_source": LABEL,
        "parent_request_sha256": content_hash(request.model_dump(mode="json")),
        "fixed_slots": records,
        "reset_gains": reset,
        "initialization_plan_preserved": True,
        "scientific_correctness_certified": False,
    }
    changed["source"] = {
        "stage": "controller_revision",
        "task_id": request.source.task_id,
        "artifact_sha256": content_hash(provenance),
    }
    return {
        "request": PublicFitRequest.model_validate(changed).model_dump(mode="json"),
        "provenance": provenance,
    }


def derive(source: Path, decisions_path: Path) -> dict:
    """Build matched repaired/control starts without fitting or reference access."""
    source_packet = sealed_read(source)
    decisions = Decisions.model_validate(public._read(decisions_path))
    if (
        source_packet["protocol"] != SOURCE_PROTOCOL
        or source_packet.get("test_data_opened") is not False
        or source_packet.get("intervention_data_used") is not False
    ):
        raise ValueError("requires public training/validation source packet")
    sources = {r["task"]["task_id"]: r for r in source_packet["rows"]}
    if len(sources) != len(source_packet["rows"]) or len(
        {d.task_id for d in decisions.models}
    ) != len(decisions.models):
        raise ValueError("duplicate source or decision task")
    if set(sources) != {d.task_id for d in decisions.models}:
        raise ValueError("decisions require exact source task coverage")
    rows, patches = [], {}
    for decision in decisions.models:
        row = sources[decision.task_id]
        if content_hash(row) != decision.source_row_sha256:
            raise ValueError("source model identity differs")
        parent = rescue._request(row["request"])
        patch = apply(parent, decision.edits)
        child = PublicFitRequest.model_validate(patch["request"])
        cell = source_packet["cells"][row["task"]["cell"]]
        training = PublicSplit.model_validate(cell["training"])
        seed = sibling_fit.compatible_seed(parent, child, row["parameters"], training)
        model, _, _ = public._lower(child)
        starts = _role_start(model.validated.candidate, public.unpack_split(training))
        parameters = dict(seed["parameters"])
        reset = patch["provenance"]["reset_gains"]
        parameters.update({n: starts[n] for n in reset})
        # Same-domain sign reversals must also reset the old, often zero, magnitude.
        sibling_fit.compatible_seed(child, child, parameters, training)
        seed = {
            "compatible_seed": seed,
            "reset_for_sign_or_domain_change": {n: starts[n] for n in reset},
            "parameters": parameters,
            "parameter_sha256": content_hash(parameters),
        }
        patches[decision.task_id] = patch
        for arm, req, vector in (
            ("repaired", child, parameters),
            ("unchanged_control", parent, row["parameters"]),
        ):
            rows.append(
                {
                    "task": {
                        **row["task"],
                        "task_id": f"{decision.task_id}_{arm}",
                        "diagnostic": {
                            "protocol": PROTOCOL,
                            "decision_source": LABEL,
                            "source_task_id": decision.task_id,
                            "comparison_arm": arm,
                            "decision_sha256": content_hash(
                                decision.model_dump(mode="json")
                            ),
                        },
                    },
                    "request": req.model_dump(mode="json"),
                    "parameters": vector,
                    "provenance": {
                        "source": row,
                        "decision": decision.model_dump(mode="json"),
                    },
                    "warm_start_audit": seed if arm == "repaired" else None,
                }
            )
    return {
        "decisions": decisions.model_dump(mode="json"),
        "patches": patches,
        "inputs": {
            "protocol": rescue.INPUT_PROTOCOL,
            "cells": source_packet["cells"],
            "rows": rows,
            "test_data_opened": False,
        },
    }


def freeze(source: Path, decisions: Path, root: Path, *, site: str = "aces") -> dict:
    source, decisions, root = source.resolve(), decisions.resolve(), root.resolve()
    if site not in {"aces", "delta"} or any(
        p.is_relative_to(root) for p in (source, decisions)
    ):
        raise ValueError("unsupported site or source inside output root")
    rescue.pruning.history.require_open(root)
    derived = derive(source, decisions)
    with public._lock(root):
        inputs = sealed_write(root / "paired-inputs.json", derived["inputs"])
        fit_plan = rescue.freeze(root / "paired-inputs.json", root / "fitting")
        return sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "site": site,
                "source": str(source),
                "source_file_sha256": diagnostic._digest(source),
                "decisions_path": str(decisions),
                "decisions_file_sha256": diagnostic._digest(decisions),
                **{k: v for k, v in derived.items() if k != "inputs"},
                "input_sha256": inputs["artifact_sha256"],
                "fitting_plan_sha256": fit_plan["artifact_sha256"],
                "source_sha256": public._source_identity(),
                "runtime": public._runtime(),
                "launcher_sha256": launcher_hash(),
                "task_count": len(inputs["rows"]),
                "limitation": LIMITATION,
            },
        )


def verify(root: Path) -> dict:
    rescue.pruning.history.require_open(root)
    plan = sealed_read(root / "plan.json")
    if (
        plan["protocol"] != PROTOCOL
        or plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
        or plan["launcher_sha256"] != launcher_hash()
        or diagnostic._digest(Path(plan["source"])) != plan["source_file_sha256"]
        or diagnostic._digest(Path(plan["decisions_path"]))
        != plan["decisions_file_sha256"]
    ):
        raise ValueError("canonical rescue identity differs")
    derived = derive(Path(plan["source"]), Path(plan["decisions_path"]))
    inputs = sealed_read(root / "paired-inputs.json")
    if (
        any(plan[k] != v for k, v in derived.items() if k != "inputs")
        or inputs["artifact_sha256"] != plan["input_sha256"]
        or {k: v for k, v in inputs.items() if k != "artifact_sha256"}
        != derived["inputs"]
        or len(inputs["rows"]) != plan["task_count"]
        or rescue.verify(root / "fitting")["artifact_sha256"]
        != plan["fitting_plan_sha256"]
    ):
        raise ValueError("canonical paired inputs differ")
    return plan


def _audit(plan: dict, result: dict) -> dict | None:
    task = result["task"]["diagnostic"]
    if task["comparison_arm"] != "repaired" or not result["selected_fit"]:
        return None
    return sign_integrity(
        {"patch": plan["patches"][task["source_task_id"]]},
        result["selected_request"],
        result["selected_fit"],
    )


def run_one(root: Path, index: int) -> dict:
    plan = verify(root)
    result = rescue.run_one(root / "fitting", index)
    return {
        "status": result["status"],
        "task": result["task"],
        "sign_integrity": _audit(plan, result),
    }


def report(root: Path) -> dict:
    with public._lock(root / "reporting"):
        if not (root / "plan.json").exists():
            result = {"protocol": PROTOCOL, "status": "not_prepared", "rows": []}
            public._write(root / "summary.json", result)
            return result
        plan = verify(root)
        summary = rescue.report(root / "fitting")
        models = public._read(root / "fitting/models.json")
        audits = {
            m["task"]["task_id"]: _audit(plan, m["result"]) for m in models["models"]
        }
        for row in summary["rows"]:
            row["sign_integrity"] = audits.get(row["task"]["task_id"])
        labels = {
            "protocol": PROTOCOL,
            "plan_sha256": plan["artifact_sha256"],
            "decision_source": LABEL,
            "scientific_correctness_certified": False,
            "decisions": plan["decisions"],
            "limitation": LIMITATION,
            "sign_integrity": audits,
            "live_llm_calls": 0,
        }
        public._write(root / "models.json", {**models, **labels})
        summary.update(labels)
        public._write(root / "summary.json", summary)
        return summary
