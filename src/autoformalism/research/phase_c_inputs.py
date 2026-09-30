"""Explicit public-only adapter for the first corrected construction baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.benchmarks.phase_c_release import load_public_cell
from autoformalism.expressions import ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.mechanism_audit import rubric
from autoformalism.rebuttal.mechanism_functional import Rule
from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec
from autoformalism.research import basin_construction_assessment as basin
from autoformalism.search.training_evidence import build_training_evidence
from autoformalism.staged_topology import build_scientific_brief
from autoformalism.targets import PublicTargetContract

REPO = Path(__file__).resolve().parents[3]
LEGACY = (
    *(
        f"phase_b_dalla_man_{task}_canonical_named_{tier}"
        for task in ("t1", "t2")
        for tier in ("easy", "hard")
    ),
    "phase_b_cstr_controlled_reactor_mechanism_canonical_named_easy",
    "phase_b_alien_device_unknown_device_mechanism_canonical_functional_easy",
)
ROSTER = {
    name.replace("phase_b_", "phase_c_", 1)
    + ("_rates_v1" if "dalla_man" in name else "_reset_v1"): name
    for name in LEGACY
}
ROSTER.update(dict.fromkeys(basin.BASINS))


def rename_channels(value):
    """Rebind one explicitly reviewed channel rename, not arbitrary text semantics."""
    if isinstance(value, str):
        return "meal_rate_g_per_min" if value == "meal_event_g" else value
    if isinstance(value, list):
        return [rename_channels(v) for v in value]
    if isinstance(value, dict):
        return {k: rename_channels(v) for k, v in value.items()}
    return value


def cell(directory: Path, config: dict) -> dict:
    """Keep original admission rules separate from independent fitted tests."""
    spec, train, val = load_public_cell(directory)
    name = spec["benchmark_id"]
    if name not in ROSTER or spec["protocol"] != "phase-c-development-2":
        raise ValueError("cell is outside the reviewed Phase C baseline")
    prompt = (directory / "proposer_prompt.txt").read_text()
    if prompt != spec["public_prompt"] + "\n" or spec["test_released"]:
        raise ValueError("public prompt or development-only boundary differs")
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    context = ValidationContext.model_validate(
        {
            k: spec[k]
            for k in ("targets", "auxiliaries", "external_inputs", "fixed_covariates")
        }
    )
    old = ROSTER[name]

    def contract(folder):
        value = rename_channels(json.loads((REPO / folder / f"{old}.json").read_text()))
        return {**value, "benchmark_id": name, "public_prompt_sha256": digest}

    if name in basin.BASINS:
        target, mechanism, rules = basin.contracts(name, prompt, digest)
    else:
        target = PublicTargetContract.model_validate(
            contract("configs/target_eval/phase_b_v2/specs")
        )
        mechanism = MechanismEvaluationSpec.model_validate(
            contract("configs/mechanism_eval/phase_b_v1/specs")
        )
        rules = rename_channels(
            json.loads((REPO / "configs/mechanism_functional_v1.json").read_text())[
                "cells"
            ][old]["mechanisms"]
        )
        quotes = {
            r.text.rstrip(".") for r in rubric(prompt) if r.category == "task_mechanism"
        }
        if {r["public_requirement"].rstrip(".") for r in rules} != quotes:
            raise ValueError(
                "independent tests do not cover the exact public mechanism bullets"
            )
        if {
            r.public_requirement.rstrip(".") for r in mechanism.required_mechanisms
        } != quotes:
            raise ValueError(
                "admission requirements differ from the public mechanism bullets"
            )
    names = {*context.targets, *context.auxiliaries, *context.external_inputs}
    for raw in rules:
        if raw["kind"].startswith("basin_"):
            continue
        r = Rule.model_validate(raw)
        if r.target not in context.targets or any(
            getattr(r, k) not in names
            for k in ("driver", "feed", "jacket", "reactant")
            if getattr(r, k)
        ):
            raise ValueError(
                "independent mechanism rule references a nonpublic channel"
            )
    from autoformalism.schemas.staged_topology import ModelingLimits
    from autoformalism.search.training_evidence import EvidenceSettings

    brief = (
        basin.brief(prompt, context, mechanism, config["limits"])
        if name in basin.BASINS
        else build_scientific_brief(
            prompt,
            context,
            target,
            mechanism,
            limits=ModelingLimits.model_validate(config["limits"]),
        )
    )
    return {
        "public_specification": spec,
        "brief": brief.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
        "target_contract": target.model_dump(mode="json"),
        "mechanism_spec": mechanism.model_dump(mode="json"),
        "independent_rules": rules,
        "assessment_policy": basin.POLICY
        if name in basin.BASINS
        else "fitted-public-mechanism-tests-1",
        "training": train.model_dump(mode="json"),
        "validation": val.model_dump(mode="json"),
        "evidence": build_training_evidence(
            public.unpack_split(train),
            context,
            EvidenceSettings.model_validate(config["evidence"]),
        ).model_dump(mode="json"),
    }


def qualified_public_cells(release: Path, config: dict) -> tuple[str, dict]:
    """Check publication receipt and selected public hashes; never open diagnostics."""
    summary = read_seal(release / "summary.json")
    if (
        summary["protocol"] != "phase-c-development-2"
        or not summary["whole_phase_c_roster_ready"]
        or summary["ready_cells"] != 28
        or summary["test_generated"]
    ):
        raise ValueError("requires the qualified 28-cell development release")
    cells = {}
    for name in ROSTER:
        directory = release / "public" / name
        for filename in (
            "specification.json",
            "proposer_prompt.txt",
            "train.json",
            "val.json",
        ):
            key = f"public/{name}/{filename}"
            if (
                hashlib.sha256((directory / filename).read_bytes()).hexdigest()
                != summary["files"][key]
            ):
                raise ValueError(f"published public file changed: {key}")
        cells[name] = cell(directory, config)
    return hashlib.sha256((release / "summary.json").read_bytes()).hexdigest(), cells
