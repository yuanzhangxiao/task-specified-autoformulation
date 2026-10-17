"""Opt-in initializer dependency handoffs and conditional family diagnostics.

Declarations describe dependencies, never executable initializers. The family
comparison conditions on every runtime-permitted initial quantity without choosing
a formula or assigning a scientific initialization policy to an unresolved state.
Historical evidence and topology diagnostics remain unchanged.
"""

from __future__ import annotations

from itertools import combinations
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
    _value,
    diagnose,
    validate_evidence,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "topology-initializer-dependencies-1"
INITIAL_TIME_POLICY = "initial-algebraic-dependencies-1"


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


class MeasuredInitialBinding(StrictSchema):
    """Reviewed identity between a declared state and a public initial reading.

    This runtime-owned record is not inferred from names or proposer prose.
    Different coordinates/units need an explicit transformation and are outside
    this narrow identity-binding contract.
    """

    state: Identifier
    source: Identifier
    relationship: Literal["same_quantity_same_units"] = "same_quantity_same_units"
    public_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_excerpt: str = Field(min_length=1, max_length=8000)
    variable_declaration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: str = Field(min_length=1, max_length=4000)


class MeasuredInitialInformationContract(InitialInformationContract):
    """Opt-in public identity bindings; v1 serialization remains unchanged."""

    protocol: Literal["recorded-t0-initial-information-2"] = (
        "recorded-t0-initial-information-2"
    )
    measured_initial_bindings: tuple[MeasuredInitialBinding, ...] = Field(
        default=(), max_length=256
    )

    @model_validator(mode="after")
    def distinct_bindings(self):
        """Reject competing declarations instead of choosing one silently."""
        states = [binding.state for binding in self.measured_initial_bindings]
        if len(states) != len(set(states)):
            raise ValueError("measured initial bindings must name distinct states")
        return self


def validate_information_contract(value) -> InitialInformationContract:
    """Revalidate a versioned runtime contract without changing historical data."""
    raw = value.model_dump(mode="json") if isinstance(value, StrictSchema) else value
    schema = (
        MeasuredInitialInformationContract
        if isinstance(raw, dict)
        and raw.get("protocol") == "recorded-t0-initial-information-2"
        else InitialInformationContract
    )
    return schema.model_validate(raw)


def measured_bindings(contract) -> tuple[MeasuredInitialBinding, ...]:
    """Return only explicitly attested identity bindings, never inferred matches."""
    return getattr(
        validate_information_contract(contract), "measured_initial_bindings", ()
    )


def validate_public_bindings(draft: Draft, scientific_context: str, contract) -> None:
    """Fail closed when a reviewed public excerpt or state declaration changes."""
    variables = {variable.name: variable for variable in draft.variables}
    for binding in measured_bindings(contract):
        if (
            binding.public_context_sha256 != content_hash(scientific_context)
            or binding.public_excerpt not in scientific_context
        ):
            raise ValueError(
                "MEASURED_INITIAL_PUBLIC_SOURCE: reviewed public text differs"
            )
        variable = variables.get(binding.state)
        if (
            variable is None
            or variable.definition != "differential"
            or content_hash(variable.model_dump(mode="json"))
            != binding.variable_declaration_sha256
        ):
            raise ValueError(
                f"MEASURED_INITIAL_STATE_CHANGED: {binding.state} no longer matches "
                "its reviewed differential declaration; explicit contract review is "
                "required before changing that coordinate or interpretation"
            )


def fixed_rules(draft, evidence, contract) -> dict[str, InitialRule]:
    """Keep recorded runtime initials plus explicit measured identity dependencies."""
    states = {variable.name: variable for variable in draft.variables}
    rules = dict(evidence.contract.initial_rules)
    for binding in measured_bindings(contract):
        variable = states.get(binding.state)
        if (
            variable is None
            or variable.definition != "differential"
            or content_hash(variable.model_dump(mode="json"))
            != binding.variable_declaration_sha256
        ):
            raise ValueError(
                f"MEASURED_INITIAL_STATE_CHANGED: {binding.state} no longer matches "
                "its reviewed differential declaration; revise the runtime binding "
                "explicitly rather than silently dropping it"
            )
        if binding.source not in contract.permitted_symbols:
            raise ValueError(
                f"MEASURED_INITIAL_SOURCE_UNAVAILABLE: {binding.source} is not "
                "permitted recorded initial information"
            )
        rule = InitialRule(mode="common_map", arguments=(binding.source,))
        previous_rule = rules.get(binding.state)
        if (
            previous_rule is not None
            and previous_rule.mode != "unknown"
            and previous_rule != rule
        ):
            raise ValueError(
                f"MEASURED_INITIAL_FIXED_CONFLICT: {binding.state} already has "
                "a different runtime initial rule"
            )
        rules[binding.state] = rule
    return rules


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


def _initial_scope(target: str, graph: dict, kinds: dict) -> set[str]:
    """Expand readout dependencies at t0 without traversing differential RHSs."""
    scope, pending = set(), [target]
    while pending:
        name = pending.pop()
        if name in scope:
            continue
        scope.add(name)
        if kinds.get(name) != "differential":
            pending.extend(graph.get(name, ()))
    return scope


def _initial_conditioning(target, left, right, graph, kinds, errors, evidence):
    """Compare state initializer rules and exact supplied values at the origin.

    The differential equation determines changes after initialization, not the
    initial value. An RHS ancestor can therefore distinguish later trajectories
    without distinguishing a readout at t0. Initializer arguments are followed
    through their declared public-information rules by ``_initial_key``.
    """
    scope = _initial_scope(target, graph, kinds)
    states = {name for name in scope if kinds.get(name) == "differential"}
    missing = [*_common_missing(evidence), *errors]
    if target not in graph or not states:
        missing.append("no generated differential ancestor subsystem")
    # Keep the existing candidate-class qualification; this narrow policy does
    # not broaden the supported class of resets, implicit equations or DAEs.
    if evidence.contract.reset_impulse_free is not True:
        missing.append("reset/impulse-free open-loop interval is not established")
    if evidence.contract.uniqueness_assumed is not True:
        missing.append("unique-solution candidate-class hypothesis is unavailable")
    if missing:
        return scope, False, missing
    equal = True
    for state in sorted(states):
        a, am = _initial_key(left, state, evidence)
        b, bm = _initial_key(right, state, evidence)
        equal &= a == b
        missing.extend(am + bm)
    for name in sorted(scope - set(graph)):
        if (
            name != evidence.context.time_symbol
            and evidence.contract.absolute_error_bounds.get(name) != 0
        ):
            missing.append(f"exact initial supplied argument {name} is not established")
        try:
            equal &= _value(left, name, 0, evidence.context.time_symbol) == _value(
                right, name, 0, evidence.context.time_symbol
            )
        except KeyError:
            missing.append(f"initial supplied argument {name} is unavailable")
    return scope, bool(equal), sorted(set(missing))


def initial_time_questions(
    draft: Draft, evidence: TrainingCompatibilityEvidence
) -> list[dict]:
    """Locate initial readout contrasts without choosing any scientific repair.

    These are candidate pairs, not established conflicts. Pass them to
    ``recheck_closed_witnesses`` with ``INITIAL_TIME_POLICY`` to qualify their
    conditioning and uncertainty. Only sealed training samples at a common
    initial time are considered; no ODE integration or initializer is evaluated.
    """
    validate_evidence(evidence)
    graph, kinds, _ = _graph(draft, evidence.context)
    questions = []
    for target in evidence.context.targets:
        if kinds.get(target) != "algebraic" or not any(
            kinds.get(name) == "differential"
            for name in _initial_scope(target, graph, kinds)
        ):
            continue
        witnesses = [
            _pair(left, 0, right, 0, target, evidence).model_dump(mode="json")
            for left, right in combinations(evidence.trajectories, 2)
            if left.time[0] == right.time[0]
            and left.targets[target][0] != right.targets[target][0]
        ]
        if witnesses:
            questions.append(
                {
                    "question_id": content_hash(
                        {
                            "policy": INITIAL_TIME_POLICY,
                            "evidence_sha256": evidence.sha256,
                            "topology_sha256": content_hash(
                                draft.model_dump(mode="json")
                            ),
                            "target": target,
                        }
                    )[:20],
                    "property": "closed_subsystem",
                    "target": target,
                    "witnesses": witnesses,
                }
            )
    return questions


def recheck_closed_witnesses(
    draft: Draft,
    evidence: TrainingCompatibilityEvidence,
    previous_questions: list[dict] | tuple[dict, ...],
    *,
    initial_time_policy: str | None = None,
) -> ClosedWitnessRecheck:
    """Preserve old matched-pair conflicts using only causal forcing prefixes.

    A forcing difference after an old conflicting observation cannot explain that
    observation. Missing assumptions and invalid coordinates never create a pass.
    Call with separately qualified derived evidence when initializer information
    has been declared or bounded by the explicit recorded-t0 family contract.
    The opt-in initial-time policy stops dependency expansion at differential
    states at the common initial time. The default preserves historical replay.
    """
    if initial_time_policy not in {None, INITIAL_TIME_POLICY}:
        raise ValueError("unknown initial-time conditioning policy")
    validate_evidence(evidence)
    graph, kinds, graph_errors = _graph(draft, evidence.context)
    checks, supported, scopes, initial_targets = [], {}, {}, set()
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
            at_initial_time = (
                initial_time_policy == INITIAL_TIME_POLICY
                and saved.times[0] == left.time[0] == right.time[0]
            )
            if at_initial_time:
                scope, equal, missing = _initial_conditioning(
                    target, left, right, graph, kinds, graph_errors, evidence
                )
                check.update(
                    conditioning_policy=INITIAL_TIME_POLICY,
                    conditioning_scope=sorted(scope),
                    initialized_states=sorted(
                        name for name in scope if kinds.get(name) == "differential"
                    ),
                    instantaneous_supplied_sources=sorted(scope - set(graph)),
                )
            else:
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
                if at_initial_time:
                    initial_targets.add(target)
                scopes[target] = scopes.get(target, set()) | scope
            checks.append(
                {
                    **check,
                    "status": status,
                    "recomputed_witness": pair.model_dump(mode="json"),
                }
            )
    diagnostics = []
    for target, pairs in sorted(supported.items()):
        initial_only = target in initial_targets and all(
            pair.sample_indices == (0, 0) for pair in pairs.values()
        )
        diagnostics.append(
            Diagnostic(
                property="closed_subsystem",
                target=target,
                status="conditional_conflict",
                scope=tuple(sorted(scopes[target])),
                verified=(
                    "Saved sample coordinates were checked against sealed evidence; "
                    "observations were recomputed, not copied from the prior question.",
                    (
                        "Initializers and supplied values relevant to each readout "
                        "were compared at the initial time; later comparisons used "
                        "the complete ancestor subsystem and forcing prefix. "
                        "Later differences cannot explain an earlier observation."
                        if target in initial_targets
                        else "All ancestor initializers and complete "
                        "forcing/covariates were compared only through each saved "
                        "conflicting observation. Later forcing differences cannot "
                        "explain an earlier difference."
                    ),
                )
                + (
                    (
                        "At the common initial time, algebraic/process dependencies "
                        "were expanded only to differential states and exact supplied "
                        "readings. Those states' declared initial rules were compared; "
                        "their ODE RHS ancestors cannot change the initial value.",
                    )
                    if target in initial_targets
                    else ()
                ),
                assumptions=(
                    "Shared deterministic functions and parameters have unique "
                    "solutions on the common existence interval.",
                ),
                comparisons=len(pairs),
                witnesses=tuple(pairs.values()),
                question=(
                    "At the common initial time this readout receives identical "
                    "state initializations and supplied values, yet its recorded "
                    "outputs differ. An ODE RHS dependency can change later "
                    "values but cannot change an already fixed initial value. "
                    "Revise a justified readout or initializer dependence, or "
                    "support a specific assumption challenge; no edge is prescribed."
                    if initial_only
                    else "The saved pair still has identical permitted initial "
                    "information and forcing through the conflicting observation. "
                    "Identify a justified missing distinction or support a specific "
                    "assumption challenge; no new dependence is prescribed."
                ),
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
    contract = validate_information_contract(contract)
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
    frozen_rules = fixed_rules(draft, evidence, contract)
    for record in handoff.records:
        if record.state not in states:
            raise ValueError(
                f"stale initializer record: {record.state} is not a differential state"
            )
        frozen = frozen_rules.get(record.state)
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
    rules = fixed_rules(draft, evidence, contract)
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
    result = {
        "protocol": PROTOCOL,
        "original_evidence_sha256": evidence.sha256,
        "runtime_initial_information": contract.model_dump(mode="json"),
        "fixed_runtime_rules": {
            name: rule.model_dump(mode="json")
            for name, rule in sorted(fixed_rules(draft, evidence, contract).items())
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
    if isinstance(contract, MeasuredInitialInformationContract):
        result["instruction"] = result["instruction"].replace(
            "No formula is selected or evaluated here;",
            "No formula is selected or evaluated for editable initializers here; "
            "displayed runtime identity formulas are already fixed;",
        )
        result["editable_states"] = sorted(
            differential - set(result["fixed_runtime_rules"])
        )
        result["measured_initial_values"] = [
            {
                "state": binding.state,
                "source": binding.source,
                "initial_equation": f"{binding.state}(t0) = {binding.source}",
                "interpretation": "Direct measured initial value in the same "
                "quantity and units; identity is fixed, not a learned map.",
            }
            for binding in contract.measured_initial_bindings
        ]
        result["instruction"] += (
            " Return records only for editable_states. Fixed runtime initials "
            "remain present without repeating them. Measured identity bindings "
            "cannot become a fitted shared value or another learned map."
        )
    return result


def rejected_handoff_feedback(draft, evidence, contract, attempted) -> dict:
    """Expose valid intended sibling edits without committing a rejected reply.

    Each returned suggestion was checked in isolation against the retained
    topology. The next full handoff is still validated atomically; this receipt
    does not establish combined completeness or scientific correctness.
    """
    contract = validate_information_contract(contract)
    attempted = InitializerDependencyHandoff.model_validate(
        attempted.model_dump(mode="json")
    )
    frozen = {
        name: rule
        for name, rule in fixed_rules(draft, evidence, contract).items()
        if rule.mode != "unknown"
    }
    valid, invalid, fixed = [], [], []
    for record in attempted.records:
        try:
            _validated(
                draft,
                evidence,
                contract,
                InitializerDependencyHandoff(records=(record,)),
            )
        except ValueError as exc:
            invalid.append({"state": record.state, "error": str(exc)})
        else:
            if record.state in frozen:
                fixed.append(record.state)
            else:
                valid.append(record.model_dump(mode="json"))
    return {
        "transaction_committed": False,
        "valid_intended_editable_records": valid,
        "invalid_records": invalid,
        "unnecessary_fixed_records": fixed,
        "instruction": "No part of the rejected reply was committed. Preserve "
        "these valid intended editable records if still scientifically intended, "
        "correct the invalid records, omit fixed runtime records, and resubmit "
        "the complete intended editable handoff. Do not replace valid scientific "
        "choices merely because an unrelated record failed. These isolated checks "
        "use the retained topology. A newly intended state absent from that "
        "topology must be resubmitted together with its initializer in a joint "
        "topology repair; absence from the retained topology does not by itself "
        "make that intended initializer scientifically invalid.",
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
