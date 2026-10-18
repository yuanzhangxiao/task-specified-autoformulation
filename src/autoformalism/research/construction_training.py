"""Train-only observations and qualified, non-gating construction questions.

Complete arrays and conditioning remain in a sealed sidecar. The prompt receives
a bounded text projection and conditional questions, never benchmark-specific
dependencies, hidden reference equations, validation scores or test observations.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_compatibility import (
    EvidenceContract,
    TrainingCompatibilityEvidence,
    build_evidence,
    diagnose,
    validate_evidence,
)
from autoformalism.search.training_evidence import (
    TrainingEvidence,
    build_training_evidence,
    validate_training_evidence,
)
from autoformalism.staged_topology import content_hash

BUNDLE_PROTOCOL = "construction-training-evidence-bundle-1"
FEEDBACK_PROTOCOL = "construction-training-advice-1"
HISTORICAL_QUESTION_IDENTITY = "full-draft-1"
SEMANTIC_QUESTION_IDENTITY = "diagnostic-semantics-2"
# The initializer recheck path also uses scoped topology under this opt-in policy.
MATHEMATICAL_QUESTION_IDENTITY = "diagnostic-semantics-3"


def load_evidence(raw: dict) -> TrainingCompatibilityEvidence:
    """Validate full train evidence, including its digest and public channels."""
    evidence = TrainingCompatibilityEvidence.model_validate(raw)
    validate_evidence(evidence)
    return evidence


def _number(value: float) -> str:
    """Six significant digits are display rounding, never equality tolerance."""
    return f"{value:.6g}"


def _same_input_contrast(evidence, target: str) -> str | None:
    """Largest observed contrast with equal target initial and recorded inputs.

    Other observed or hidden initial states need not match. This is deliberately
    an observation rather than the closed-subsystem theorem used after topology.
    """
    best = None
    for left, right in combinations(evidence.trajectories, 2):
        if (
            left.time != right.time
            or left.inputs != right.inputs
            or left.covariates != right.covariates
            or left.targets[target][0] != right.targets[target][0]
        ):
            continue
        for i, (a, b) in enumerate(
            zip(left.targets[target], right.targets[target], strict=True)
        ):
            gap = abs(a - b)
            if gap > 0 and (best is None or gap > best[0]):
                best = (gap, left, right, i, a, b)
    if best is None:
        return None
    _, left, right, i, a, b = best
    return (
        f"{target}: {left.trajectory_id} vs {right.trajectory_id} have identical "
        f"recorded input samples/covariates and {target}(t0)="
        f"{_number(left.targets[target][0])}, but at t={_number(left.time[i])} "
        f"{target}={_number(a)} vs {_number(b)}. Other initial states and "
        "dependencies are not matched by this observation."
    )


def _constant_input_change(evidence, target: str) -> str | None:
    """Show largest observed range under constant sampled inputs, not a cause."""
    best = None
    for item in evidence.trajectories:
        if any(len(set(values)) != 1 for values in item.inputs.values()):
            continue
        values = item.targets[target]
        low = min(range(len(values)), key=values.__getitem__)
        high = max(range(len(values)), key=values.__getitem__)
        span = values[high] - values[low]
        if span > 0 and (best is None or span > best[0]):
            best = span, item, low, high
    if best is None:
        return None
    _, item, low, high = best
    inputs = ", ".join(f"{n}={_number(v[0])}" for n, v in sorted(item.inputs.items()))
    values = item.targets[target]
    indices = sorted({0, low, high, len(values) - 1})
    observations = ", ".join(
        f"t={_number(item.time[i])}: {_number(values[i])}" for i in indices
    )
    return (
        f"{item.trajectory_id}: constant sampled inputs [{inputs}], yet {target} "
        f"changes ({observations}). This does not identify a missing dependence."
    )


def initial_highlights(
    evidence: TrainingCompatibilityEvidence, *, maximum_characters: int = 3500
) -> str:
    """Project complete observations to deterministic, bounded, factual prose.

    Contrasts take priority over per-trajectory extrema. Complete lines are kept;
    omitted counts are explicit. No peak pairing or causal interpretation occurs.
    """
    validate_evidence(evidence)
    if maximum_characters < 700:
        raise ValueError("training highlight budget must be at least 700 characters")
    header = (
        f"Training highlights: {len(evidence.trajectories)} complete trajectories; "
        "sampled observations, values rounded for display. Equalities are tested "
        "on original numbers. These facts do not assign causes or required edges."
    )
    facts = []
    for target in evidence.context.targets:
        if line := _same_input_contrast(evidence, target):
            facts.append(line)
    for target in evidence.context.targets:
        if line := _constant_input_change(evidence, target):
            facts.append(line)
    for name in evidence.context.external_inputs:
        constant = sum(
            len(set(item.inputs[name])) == 1 for item in evidence.trajectories
        )
        zero = sum(
            all(value == 0 for value in item.inputs[name])
            for item in evidence.trajectories
        )
        facts.append(
            f"Input {name}: {constant}/{len(evidence.trajectories)} sampled traces "
            f"constant, {zero} all zero; interpolation="
            f"{evidence.contract.interpolation.get(name, 'unknown')}."
        )
    # Round-robin targets before trajectories avoids filling the budget with one
    # target only. All extrema are sampled, including endpoint/plateau maxima.
    for item in evidence.trajectories:
        for target in evidence.context.targets:
            values = item.targets[target]
            low = min(range(len(values)), key=values.__getitem__)
            high = max(range(len(values)), key=values.__getitem__)
            maximum_count = sum(value == values[high] for value in values)
            facts.append(
                f"{item.trajectory_id} {target}: initial={_number(values[0])}, "
                f"final={_number(values[-1])}; min={_number(values[low])} at "
                f"t={_number(item.time[low])}; max={_number(values[high])} at "
                f"t={_number(item.time[high])} (first of {maximum_count} tied samples)."
            )
    suffix = (
        " Full arrays remain in the evidence artifact. Missing a displayed fact "
        "does not establish its absence; peak timing alone is not a causal delay."
    )
    lines = [header]
    kept = 0
    for fact in facts:
        footer = (
            f"Displayed facts={kept + 1}/{len(facts)}; omitted={len(facts) - kept - 1}."
        )
        candidate = "\n".join([*lines, fact, footer + suffix])
        if len(candidate) > maximum_characters:
            continue
        lines.append(fact)
        kept += 1
    lines.append(
        f"Displayed facts={kept}/{len(facts)}; omitted={len(facts) - kept}." + suffix
    )
    return "\n".join(lines)


def _diagnostic_topology(draft: Draft, scope: tuple[str, ...]) -> dict:
    """Project a diagnostic's ancestors without explanatory prose or edit records.

    Keep kinds, signs, conversion text and contribution multiplicity even where
    the present theorem only needs dependence. Reordering declarations is not a
    scientific edit. A process's consumers outside this diagnostic's subsystem
    do not affect its question; its complete declaration remains in provenance.
    """
    names = set(scope)

    def ordered(values: list[dict]) -> list[dict]:
        return sorted(values, key=lambda item: json.dumps(item, sort_keys=True))

    return {
        "variables": ordered(
            [
                {"name": item.name, "definition": item.definition}
                for item in draft.variables
                if item.name in names
            ]
        ),
        "equations": ordered(
            [
                {
                    "name": item.name,
                    "terms": ordered(
                        [
                            {
                                "sources": sorted(term.sources),
                                "outer_weight_sign": term.outer_weight_sign.value,
                            }
                            for term in item.terms
                        ]
                    ),
                }
                for item in draft.equations
                if item.name in names
            ]
        ),
        "processes": ordered(
            [
                {
                    "name": item.name,
                    "depends_on": sorted(item.depends_on),
                    "kind": item.kind,
                    "uses": ordered(
                        [
                            use.model_dump(mode="json")
                            for use in item.uses
                            if use.target in names
                        ]
                    ),
                }
                for item in draft.processes
                if item.name in names
            ]
        ),
    }


def compatibility_feedback(
    draft: Draft,
    evidence: TrainingCompatibilityEvidence,
    *,
    identity_policy: str = HISTORICAL_QUESTION_IDENTITY,
) -> dict:
    """Give candidate-specific questions with witnesses and explicit hypotheses.

    Retaining a deliberate approximation is an allowed scientific decision. A
    structural acceptance gate must never consume the conditional-conflict count.
    The historical identity remains the default for exact replay. Opt-in semantic
    identities preserve an answer across prose or independent-subsystem edits,
    while retaining the full draft hash separately for artifact provenance.
    """
    if identity_policy not in {
        HISTORICAL_QUESTION_IDENTITY,
        SEMANTIC_QUESTION_IDENTITY,
        MATHEMATICAL_QUESTION_IDENTITY,
    }:
        raise ValueError(
            f"unsupported compatibility identity policy: {identity_policy}"
        )
    report = diagnose(draft, evidence)
    questions = []
    for row in report.diagnostics:
        if row.status != "conditional_conflict":
            continue
        value = {
            "target": row.target,
            "property": row.property,
            "scope": list(row.scope),
            "verified": list(row.verified),
            "assumptions": list(row.assumptions),
            "witnesses": [w.model_dump(mode="json") for w in row.witnesses],
            "conditioning": _conditioning_witness(draft, evidence, row),
            "question": row.question,
            "interpretation": (
                "Conditional conflict with exact recorded reproduction, not a "
                "fit-quality gate. Choose a scientific revision, identify an "
                "inapplicable hypothesis, or retain an intended approximation."
            ),
        }
        if identity_policy == HISTORICAL_QUESTION_IDENTITY:
            identity = {
                "evidence_sha256": report.evidence_sha256,
                "topology_sha256": report.topology_sha256,
                "diagnostic": row.model_dump(mode="json"),
            }
        else:
            semantic_hash = content_hash(_diagnostic_topology(draft, row.scope))
            value["semantic_topology_sha256"] = semantic_hash
            identity = {
                "policy": identity_policy,
                "evidence_sha256": report.evidence_sha256,
                "semantic_topology_sha256": semantic_hash,
                "diagnostic": row.model_dump(mode="json"),
            }
        value["question_id"] = content_hash(identity)[:20]
        questions.append(value)
    feedback = {
        "protocol": FEEDBACK_PROTOCOL,
        "evidence_sha256": report.evidence_sha256,
        "topology_sha256": report.topology_sha256,
        "acceptance_gate": False,
        "advisory_questions": questions,
        "counts": dict(Counter(row.status for row in report.diagnostics)),
        "full_report": report.model_dump(mode="json"),
    }
    if identity_policy != HISTORICAL_QUESTION_IDENTITY:
        feedback["question_identity_policy"] = identity_policy
    return feedback


def _recorded_value(item, name: str, index: int, time_symbol: str) -> float:
    """Read one permitted recorded value; never substitute missing channels."""
    if name == time_symbol:
        return item.time[index]
    if name in item.covariates:
        return item.covariates[name]
    for group in (item.targets, item.auxiliaries, item.inputs):
        if name in group:
            return group[name][index]
    raise ValueError(f"qualified conditioning channel unavailable: {name}")


def _conditioning_witness(draft, evidence, row) -> dict:
    """Display the actual matching information behind an already qualified test."""
    generated = {v.name for v in draft.variables} | {p.name for p in draft.processes}
    supplied = sorted(set(row.scope) - generated)
    trajectories = {v.trajectory_id: v for v in evidence.trajectories}
    ids = sorted({name for witness in row.witnesses for name in witness.trajectories})
    clock = evidence.context.time_symbol
    if row.property == "algebraic_map":
        return {
            "matched_complete_supplied_arguments": [
                {
                    "trajectory": name,
                    "sample": index,
                    "values": {
                        key: _recorded_value(trajectories[name], key, index, clock)
                        for key in supplied
                    },
                }
                for witness in row.witnesses
                for name, index in zip(
                    witness.trajectories, witness.sample_indices, strict=True
                )
            ]
        }
    if row.property == "scalar_autonomous":
        return {
            "constant_external_arguments": {
                name: {
                    key: _recorded_value(trajectories[name], key, 0, clock)
                    for key in supplied
                }
                for name in ids
            },
            "comparison": "Constancy checked using all defining knots.",
        }
    states = sorted(
        v.name
        for v in draft.variables
        if v.definition == "differential" and v.name in row.scope
    )
    initial = {}
    for name in ids:
        initial[name] = {}
        for state in states:
            rule = evidence.contract.initial_rules[state]
            arguments = (state,) if rule.mode == "observed_initial" else rule.arguments
            initial[name][state] = {
                "rule": rule.mode,
                "recorded_arguments": {
                    key: _recorded_value(trajectories[name], key, 0, clock)
                    for key in arguments
                },
            }
    return {
        "matched_initial_information": initial,
        "matched_complete_forcing_channels": supplied,
        "interpolation": {
            name: evidence.contract.interpolation[name]
            for name in supplied
            if name in evidence.contract.interpolation
        },
        "comparison": (
            "All defining forcing knots/covariates and time origins were compared, "
            "not only displayed samples. Full values are in the sealed evidence."
        ),
    }


def _public_value(path: Path) -> tuple[dict, str]:
    """Read one explicit public artifact and verify the exporter content seal."""
    packed = json.loads(path.read_text())
    if not isinstance(packed, dict) or set(packed) != {"value", "sha256"}:
        raise ValueError(f"public artifact wrapper differs: {path}")
    if content_sha256(packed["value"]) != packed["sha256"]:
        raise ValueError(f"public artifact digest differs: {path}")
    return packed["value"], packed["sha256"]


def _dataset(train: dict) -> DatasetSplit:
    """Reject other splits before reading any rows or arrays."""
    if train.get("name") != "train":
        raise ValueError("construction evidence requires public train only")
    rows = tuple(
        Trajectory(
            trajectory_id=row["trajectory_id"],
            time=np.asarray(row["time"], dtype=float),
            targets={k: np.asarray(v, float) for k, v in row["targets"].items()},
            auxiliaries={
                k: np.asarray(v, float) for k, v in row["auxiliaries"].items()
            },
            external_inputs={
                k: np.asarray(v, float) for k, v in row["external_inputs"].items()
            },
            fixed_covariates=row["fixed_covariates"],
            derivatives={},
        )
        for row in train["rows"]
    )
    return DatasetSplit(SplitName.TRAIN, rows, train.get("fingerprint", ""))


def _qualified_contract(
    runtime: EvidenceContract,
    spec: dict,
    context: ValidationContext,
    train_digest: str,
    spec_digest: str,
) -> EvidenceContract:
    """Combine explicit runtime hypotheses with public metadata conservatively.

    Runtime callers must deliberately supply compiler/model-class assertions;
    the exporter never infers uniqueness, open-loop semantics or initialization
    from numerical agreement. Noise metadata supplies no finite bound unless the
    public release explicitly declares noiseless observations.
    """
    channels = (
        context.targets
        + context.auxiliaries
        + context.external_inputs
        + context.fixed_covariates
    )
    noiseless = (
        spec.get("noise_sd_fraction") == 0
        and spec.get("initial_observation_noise") == 0
    )
    continuous = spec.get("input_contract") == "continuous-rates-1"
    return EvidenceContract(
        **(
            runtime.model_dump(mode="json")
            | {
                "provenance": (
                    f"Public train={train_digest}; specification={spec_digest}. "
                    "Explicit runtime/candidate-class hypotheses: "
                    f"{runtime.provenance} "
                    "Noiseless bounds require both public noise fields equal zero; "
                    "missing metadata remains unknown. "
                    "No claim of universal compatibility."
                ),
                "absolute_error_bounds": dict.fromkeys(
                    channels, 0 if noiseless else None
                ),
                "interpolation": {
                    name: runtime.interpolation.get(name, "unknown")
                    if continuous
                    else "unknown"
                    for name in context.external_inputs + context.auxiliaries
                },
                "initial_rules": {
                    name: rule.model_dump(mode="json")
                    for name, rule in runtime.initial_rules.items()
                    if name in context.targets + context.auxiliaries
                },
                "reset_impulse_free": runtime.reset_impulse_free
                if continuous
                else None,
            }
        )
    )


def export_bundle(
    source_plan: Path,
    public_root: Path,
    output: Path,
    runtime_contract: EvidenceContract,
) -> dict:
    """Seal complete public training for exactly the frozen source-plan roster.

    Only ``<public_root>/<case>/train.json`` and ``specification.json`` are read.
    The historical bounded evidence's source digest must match; auxiliary values
    and covariates omitted from that historical digest are newly sealed in full.
    Existing output is immutable and an identical rerun is deterministic.
    """
    plan = sealed_read(source_plan)
    cells, source_files = {}, {}
    for name, cell in sorted(plan["cells"].items()):
        if re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
            raise ValueError("unsafe public benchmark directory name")
        context = ValidationContext.model_validate(cell["context"])
        previous = TrainingEvidence.model_validate(cell["evidence"])
        validate_training_evidence(previous, context)
        train, train_digest = _public_value(public_root / name / "train.json")
        spec, spec_digest = _public_value(public_root / name / "specification.json")
        if spec.get("benchmark_id") != name:
            raise ValueError(f"public specification benchmark differs: {name}")
        dataset = _dataset(train)
        rebuilt = build_training_evidence(dataset, context, previous.settings)
        if rebuilt.source_sha256 != previous.source_sha256:
            raise ValueError(f"historical training source digest differs: {name}")
        evidence = build_evidence(
            dataset,
            context,
            _qualified_contract(
                runtime_contract, spec, context, train_digest, spec_digest
            ),
        )
        cells[name] = evidence.model_dump(mode="json")
        source_files[name] = {
            "public_train_sha256": train_digest,
            "public_specification_sha256": spec_digest,
            "historical_source_sha256": previous.source_sha256,
            "historical_packet_sha256": previous.packet_sha256,
            "source_digest_matches": True,
            "complete_channels_newly_sealed": list(
                context.auxiliaries + context.fixed_covariates
            ),
        }
    return sealed_write(
        output,
        {
            "protocol": BUNDLE_PROTOCOL,
            "source_plan_sha256": plan["artifact_sha256"],
            "cells": cells,
            "source_files": source_files,
            "llm_calls": 0,
            "optimizer_calls": 0,
            "test_data_opened": False,
        },
    )


def load_bundle(path: Path, source_plan: dict | None = None) -> dict:
    """Verify all full evidence and optional source-plan identity before a run."""
    bundle = sealed_read(path)
    if (
        bundle.get("protocol") != BUNDLE_PROTOCOL
        or bundle.get("test_data_opened") is not False
    ):
        raise ValueError("construction training bundle protocol differs")
    if source_plan is not None:
        if bundle["source_plan_sha256"] != source_plan["artifact_sha256"]:
            raise ValueError("construction training source plan differs")
        if set(bundle["cells"]) != set(source_plan["cells"]):
            raise ValueError("construction training case roster differs")
    for name, raw in bundle["cells"].items():
        evidence = load_evidence(raw)
        if source_plan is not None:
            cell = source_plan["cells"][name]
            if evidence.context != ValidationContext.model_validate(cell["context"]):
                raise ValueError("construction training context differs")
            if (
                bundle["source_files"][name]["historical_source_sha256"]
                != cell["evidence"]["source_sha256"]
            ):
                raise ValueError("construction training historical source differs")
    return bundle
