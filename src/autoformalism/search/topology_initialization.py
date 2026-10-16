"""Opt-in initializer dependency handoffs and conditional family diagnostics.

Declarations describe dependencies, never executable initializers. The family
comparison conditions on every runtime-permitted initial quantity without choosing
a formula or assigning a scientific initialization policy to an unresolved state.
Historical evidence and topology diagnostics remain unchanged.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_compatibility import (
    Diagnostic,
    InitialRule,
    TrainingCompatibilityEvidence,
    Witness,
    _closure,
    _common_missing,
    _forcing_equal,
    _graph,
    _initial_key,
    _pair,
    diagnose,
    validate_evidence,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "topology-initializer-dependencies-1"


class InitialInformationContract(StrictSchema):
    """Runtime-owned bounds on available initial information, never proposer prose.

    ``complete`` asserts that permitted_symbols exhausts the information available
    to an initializer. Unknown or broader history cannot support the family test.
    """

    protocol: Literal["recorded-t0-initial-information-1"] = (
        "recorded-t0-initial-information-1"
    )
    provenance: str = Field(min_length=1, max_length=16000)
    scope: Literal["recorded_t0", "unknown", "history_or_unrecorded"] = "unknown"
    shared_deterministic: bool | None = None
    complete: bool | None = None
    permitted_symbols: tuple[Identifier, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def unique_symbols(self):
        """Do not silently normalize an ambiguous runtime contract."""
        if len(set(self.permitted_symbols)) != len(self.permitted_symbols):
            raise ValueError("permitted initial symbols must be distinct")
        return self


class InitializerDependencyRecord(StrictSchema):
    """One state-local declaration using the existing initial-rule vocabulary."""

    state: Identifier
    rule: InitialRule

    @model_validator(mode="after")
    def valid_arguments(self):
        """Reject unused and duplicate arguments instead of weakening a claim."""
        arguments = self.rule.arguments
        if len(arguments) != len(set(arguments)):
            raise ValueError("initializer arguments must be distinct")
        if self.rule.mode == "common_map":
            if not arguments:
                raise ValueError("common_map requires explicit initial arguments")
        elif arguments:
            raise ValueError("only common_map accepts initial arguments")
        return self


class InitializerDependencyHandoff(StrictSchema):
    """Complete durable dependency records; missing states remain unresolved."""

    protocol: Literal["topology-initializer-dependencies-1"] = PROTOCOL
    records: tuple[InitializerDependencyRecord, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def unique_states(self):
        """Require one unambiguous record per selected state."""
        names = [record.state for record in self.records]
        if len(names) != len(set(names)):
            raise ValueError("initializer records must name distinct states")
        return self


class ClosedWitnessRecheck(StrictSchema):
    """Recomputed prefix witnesses plus explicit reasons for non-preservation."""

    protocol: Literal["closed-witness-prefix-recheck-1"] = (
        "closed-witness-prefix-recheck-1"
    )
    evidence_sha256: str
    topology_sha256: str
    diagnostics: tuple[Diagnostic, ...] = ()
    checks: tuple[dict, ...] = ()


def _saved_pair(question: dict, witness: object, evidence) -> tuple:
    """Resolve exact saved coordinates; observations are always reread from evidence."""
    target = question.get("target")
    if target not in evidence.context.targets:
        raise ValueError("saved witness target is not a public target")
    saved = Witness.model_validate(witness)
    if any(
        len(value) != 2
        for value in (
            saved.trajectories,
            saved.sample_indices,
            saved.times,
            saved.observations,
        )
    ):
        raise ValueError("closed witness requires exactly two aligned samples")
    if saved.trajectories[0] == saved.trajectories[1]:
        raise ValueError("closed witness requires distinct trajectories")
    trajectories = {row.trajectory_id: row for row in evidence.trajectories}
    rows = []
    for name, index, time in zip(
        saved.trajectories, saved.sample_indices, saved.times, strict=True
    ):
        if name not in trajectories:
            raise ValueError("saved witness trajectory is unavailable")
        row = trajectories[name]
        if index < 0 or index >= len(row.time) or row.time[index] != time:
            raise ValueError("saved witness sample/time coordinates differ")
        rows.append(row)
    if saved.times[0] != saved.times[1]:
        raise ValueError("closed witness requires the same observation time")
    return target, saved, rows[0], rows[1]


def _prefix_conditioning(target, saved, left, right, graph, kinds, errors, evidence):
    """Reuse every existing closed-subsystem assumption through the witness time."""
    scope = _closure(target, graph)
    states = {name for name in scope if kinds.get(name) == "differential"}
    missing = [*_common_missing(evidence), *errors]
    if target not in graph or not states:
        missing.append("no generated differential ancestor subsystem")
    if evidence.contract.reset_impulse_free is not True:
        missing.append("reset/impulse-free open-loop interval is not established")
    if evidence.contract.uniqueness_assumed is not True:
        missing.append("unique-solution candidate-class hypothesis is unavailable")
    if left.time[0] != right.time[0]:
        missing.append("different time origins; no identical-start comparison")
    if missing:
        return scope, False, missing
    equal = True
    for state in sorted(states):
        a, am = _initial_key(left, state, evidence)
        b, bm = _initial_key(right, state, evidence)
        missing.extend(am + bm)
        equal &= a == b
    forcing, fm = _forcing_equal(
        left, right, scope - set(graph), evidence, saved.times[0]
    )
    return scope, bool(equal and forcing), sorted(set(missing + fm))


def recheck_closed_witnesses(
    draft: Draft,
    evidence: TrainingCompatibilityEvidence,
    previous_questions: list[dict] | tuple[dict, ...],
) -> ClosedWitnessRecheck:
    """Preserve old matched-pair conflicts using only causal forcing prefixes.

    A forcing difference after an old conflicting observation cannot explain that
    observation. Missing assumptions and invalid coordinates never create a pass.
    Call with separately qualified derived evidence when initializer information
    has been declared or bounded by the explicit recorded-t0 family contract.
    """
    validate_evidence(evidence)
    graph, kinds, graph_errors = _graph(draft, evidence.context)
    checks, supported, scopes = [], {}, {}
    for question in previous_questions:
        if not isinstance(question, dict):
            checks.append({"status": "invalid_saved_witness", "reason": "not a record"})
            continue
        if question.get("property") != "closed_subsystem":
            continue
        witnesses = question.get("witnesses")
        if not isinstance(witnesses, (list, tuple)) or not witnesses:
            checks.append(
                {"status": "invalid_saved_witness", "reason": "missing paired samples"}
            )
            continue
        for raw in witnesses:
            check = {
                "target": question.get("target"),
                "previous_question_id": question.get("question_id"),
            }
            try:
                target, saved, left, right = _saved_pair(question, raw, evidence)
            except (ValueError, TypeError, KeyError) as exc:
                checks.append(
                    {**check, "status": "invalid_saved_witness", "reason": str(exc)}
                )
                continue
            check.update(
                trajectories=list(saved.trajectories),
                sample_indices=list(saved.sample_indices),
                conditioning_end_time=saved.times[0],
            )
            scope, equal, missing = _prefix_conditioning(
                target, saved, left, right, graph, kinds, graph_errors, evidence
            )
            if missing:
                checks.append({**check, "status": "unavailable", "reasons": missing})
                continue
            if not equal:
                checks.append(
                    {
                        **check,
                        "status": "conditioning_distinction",
                        "reason": "Initial information or forcing through this "
                        "observation differs; equality does not apply to this pair.",
                    }
                )
                continue
            pair = _pair(
                left,
                saved.sample_indices[0],
                right,
                saved.sample_indices[1],
                target,
                evidence,
            )
            if pair.observations[0] == pair.observations[1]:
                status = "no_observed_difference"
            elif pair.error_intervals_disjoint is not True:
                status = "uncertainty_limited"
            else:
                status = "preserved"
                supported.setdefault(target, {})[
                    (pair.trajectories, pair.sample_indices)
                ] = pair
                scopes[target] = scope
            checks.append(
                {
                    **check,
                    "status": status,
                    "recomputed_witness": pair.model_dump(mode="json"),
                }
            )
    diagnostics = []
    for target, pairs in sorted(supported.items()):
        diagnostics.append(
            Diagnostic(
                property="closed_subsystem",
                target=target,
                status="conditional_conflict",
                scope=tuple(sorted(scopes[target])),
                verified=(
                    "Saved sample coordinates were checked against sealed evidence; "
                    "observations were recomputed, not copied from the prior question.",
                    "All ancestor initializers and complete forcing/covariates were "
                    "compared only through each saved conflicting observation. "
                    "Later forcing differences cannot explain an earlier difference.",
                ),
                assumptions=(
                    "Shared deterministic functions and parameters have unique "
                    "solutions on the common existence interval.",
                ),
                comparisons=len(pairs),
                witnesses=tuple(pairs.values()),
                question="The saved pair still has identical permitted initial "
                "information and forcing through the conflicting observation. "
                "Identify a justified missing distinction or support a specific "
                "assumption challenge; no new dependence is prescribed.",
            )
        )
    return ClosedWitnessRecheck(
        evidence_sha256=evidence.sha256,
        topology_sha256=content_hash(draft.model_dump(mode="json")),
        diagnostics=tuple(diagnostics),
        checks=tuple(checks),
    )


def _validated(
    draft: Draft,
    evidence: TrainingCompatibilityEvidence,
    contract: InitialInformationContract,
    handoff: InitializerDependencyHandoff | None,
) -> tuple[InitialInformationContract, InitializerDependencyHandoff]:
    """Check seals, runtime ownership and stale records without changing inputs."""
    validate_evidence(evidence)
    contract = InitialInformationContract.model_validate(
        contract.model_dump(mode="json")
    )
    handoff = InitializerDependencyHandoff.model_validate(
        (handoff or InitializerDependencyHandoff()).model_dump(mode="json")
    )
    context = evidence.context
    public = set(context.targets + context.auxiliaries + context.external_inputs)
    public.update(context.fixed_covariates)
    public.add(context.time_symbol)
    permitted = set(contract.permitted_symbols)
    if invalid := permitted - public:
        raise ValueError(f"unavailable initial information: {sorted(invalid)}")
    states = {v.name for v in draft.variables if v.definition == "differential"}
    for record in handoff.records:
        if record.state not in states:
            raise ValueError(
                f"stale initializer record: {record.state} is not a differential state"
            )
        frozen = evidence.contract.initial_rules.get(record.state)
        if frozen is not None and frozen.mode != "unknown":
            if record.rule != frozen:
                raise ValueError(f"runtime initialization is fixed for {record.state}")
        elif record.rule.mode == "observed_initial":
            raise ValueError("observed_initial requires a frozen runtime rule")
        if invalid := set(record.rule.arguments) - permitted:
            raise ValueError(f"unavailable initializer arguments: {sorted(invalid)}")
    return contract, handoff


def _unavailable(contract: InitialInformationContract, *, family: bool) -> list[str]:
    """Describe unsupported assumptions explicitly, including unknown booleans."""
    reasons = []
    if contract.scope != "recorded_t0":
        reasons.append("recorded-t0-only initial information is not established")
    if contract.shared_deterministic is not True:
        reasons.append("shared deterministic initialization is not established")
    if family and contract.complete is not True:
        reasons.append("the permitted initial information is not established complete")
    return reasons


def _rules(
    draft: Draft,
    evidence: TrainingCompatibilityEvidence,
    contract: InitialInformationContract,
    handoff: InitializerDependencyHandoff,
) -> tuple[dict[str, InitialRule], list[str]]:
    """Keep known runtime rules and add only supported explicit declarations."""
    rules = dict(evidence.contract.initial_rules)
    if not _unavailable(contract, family=False):
        rules.update({record.state: record.rule for record in handoff.records})
    unresolved = sorted(
        v.name
        for v in draft.variables
        if v.definition == "differential"
        and rules.get(v.name, InitialRule(mode="unknown")).mode == "unknown"
    )
    return rules, unresolved


def derived_evidence(
    draft: Draft,
    evidence: TrainingCompatibilityEvidence,
    contract: InitialInformationContract,
    handoff: InitializerDependencyHandoff | None = None,
    *,
    family: bool = False,
) -> TrainingCompatibilityEvidence:
    """Return separately sealed conditioning for existing diagnostics/feedback.

    Family rules are mathematical upper-family conditioning, not retained choices.
    When the runtime permits no initial arguments, that family is a shared constant.
    The original seal remains in provenance and must identify historical witnesses.
    """
    contract, handoff = _validated(draft, evidence, contract, handoff)
    if family and (reasons := _unavailable(contract, family=True)):
        raise ValueError("family qualification unavailable: " + "; ".join(reasons))
    rules, unresolved = _rules(draft, evidence, contract, handoff)
    if family:
        arguments = tuple(sorted(contract.permitted_symbols))
        for state in unresolved:
            rules[state] = InitialRule(
                mode="common_map" if arguments else "shared_parameter",
                arguments=arguments,
            )
    raw = evidence.model_dump(mode="json", exclude={"sha256"})
    raw["contract"]["initial_rules"] = {
        name: rule.model_dump(mode="json") for name, rule in sorted(rules.items())
    }
    contract_sha = content_hash(contract.model_dump(mode="json"))
    conditioning = "maximal permitted initial family" if family else "declared only"
    raw["contract"]["provenance"] += (
        f" {PROTOCOL}; original evidence={evidence.sha256}; "
        f"initial information contract={contract_sha}; "
        f"declarations={content_hash(handoff.model_dump(mode='json'))}; "
        f"conditioning={conditioning}. "
        f"Runtime initial-information provenance: {contract.provenance}. "
        "No executable initializer or fitted parameter is established."
    )
    result = TrainingCompatibilityEvidence.model_validate(
        {**raw, "sha256": content_hash(raw)}
    )
    validate_evidence(result)
    return result


def request(
    draft: Draft,
    evidence: TrainingCompatibilityEvidence,
    contract: InitialInformationContract,
    handoff: InitializerDependencyHandoff | None = None,
) -> dict:
    """Build a readable offline request without provider calls or draft edits."""
    contract, handoff = _validated(draft, evidence, contract, handoff)
    _, unresolved = _rules(draft, evidence, contract, handoff)
    differential = {v.name for v in draft.variables if v.definition == "differential"}
    return {
        "protocol": PROTOCOL,
        "original_evidence_sha256": evidence.sha256,
        "runtime_initial_information": contract.model_dump(mode="json"),
        "fixed_runtime_rules": {
            name: rule.model_dump(mode="json")
            for name, rule in sorted(evidence.contract.initial_rules.items())
            if name in differential and rule.mode != "unknown"
        },
        "retained_handoff": handoff.model_dump(mode="json"),
        "unresolved_states": unresolved,
        "instruction": (
            "Describe initialization dependencies for differential states only. "
            "shared_parameter means one unknown initial value shared across "
            "trajectories; "
            "common_map lists the initial public quantities used by one shared "
            "deterministic map. A channel means its recorded value at t0, never its "
            "later path. unknown leaves initialization unresolved. Known runtime "
            "rules cannot be changed. Do not use trajectory IDs, future information, "
            "hidden channels or generated-only states. Return a complete handoff, "
            "retaining intended existing records. No formula is selected or evaluated "
            "here; dependency acceptance does not establish scientific correctness. "
            "The runtime contract is not part of the proposer response."
        ),
        "response_template": InitializerDependencyHandoff().model_dump(mode="json"),
        "response_schema": InitializerDependencyHandoff.model_json_schema(),
    }


def inspect(
    draft: Draft,
    evidence: TrainingCompatibilityEvidence,
    contract: InitialInformationContract,
    handoff: InitializerDependencyHandoff | None = None,
    *,
    qualify_family: bool = True,
) -> dict:
    """Report declared and family-qualified diagnostics with separate source seals."""
    contract, handoff = _validated(draft, evidence, contract, handoff)
    _, unresolved = _rules(draft, evidence, contract, handoff)
    declared = derived_evidence(draft, evidence, contract, handoff)
    unavailable = _unavailable(contract, family=True)
    family = (
        derived_evidence(draft, evidence, contract, handoff, family=True)
        if qualify_family and not unavailable
        else None
    )
    return {
        "protocol": PROTOCOL,
        "original_evidence_sha256": evidence.sha256,
        "initial_information_contract": contract.model_dump(mode="json"),
        "handoff": handoff.model_dump(mode="json"),
        "unresolved_states": unresolved,
        "declaration_unavailable": _unavailable(contract, family=False),
        "declared_evidence_sha256": declared.sha256,
        "declared_report": diagnose(draft, declared).model_dump(mode="json"),
        "family_evidence_sha256": family.sha256 if family else None,
        "family_qualified_report": (
            diagnose(draft, family).model_dump(mode="json") if family else None
        ),
        "family_qualification": {
            "status": "assessed" if family else "deferred" if qualify_family else "off",
            "unavailable": unavailable if qualify_family else [],
            "conditioned_unresolved_states": unresolved if family else [],
            "maximal_initial_arguments": sorted(contract.permitted_symbols),
            "interpretation": (
                "Under the explicit complete recorded-t0/shared deterministic "
                "contract, "
                "equal values of ALL permitted initial quantities force equal "
                "initializations for any map in this family. Different arguments "
                "do not establish different initial values. No particular map or "
                "shared initial value has been chosen for an unresolved state. "
                "Existing exact-conditioning/noise, full-forcing, shared-parameter "
                "and uniqueness requirements remain in force. Deferred comparisons "
                "are not scientific passes."
            ),
        },
        "executable_initializers_selected": False,
        "acceptance_gate": False,
        "llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }
