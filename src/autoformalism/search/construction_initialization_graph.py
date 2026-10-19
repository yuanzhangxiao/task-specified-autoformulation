"""Opt-in construction declarations for acyclic initial-state dependencies.

Topology chooses only modes, scopes and dependencies. The runtime owns observed
identities, justified known values and the permitted information boundary; a later
formula stage owns expressions. No declaration here evaluates data or fits a value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import Field, model_validator

from autoformalism.expressions import ValidationContext
from autoformalism.fitting.initialization import KnownInitialValue
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixPolicy
from autoformalism.search.construction_ledger import Draft
from autoformalism.staged_topology import content_hash

PROTOCOL = "construction-initializer-dependencies-1"
CONTRACT_PROTOCOL = "construction-initializer-graph-contract-1"
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class GraphValueDependency(StrictSchema):
    """One independent unknown, with explicit training or trajectory scope."""

    mode: Literal["value"] = "value"
    scope: Literal["shared_training", "trajectory"]


class GraphMapDependency(StrictSchema):
    """A dependent initial value; the two namespaces never imply each other."""

    mode: Literal["map"] = "map"
    state_initials: tuple[Identifier, ...] = Field(default=(), max_length=64)
    public_initials: tuple[Identifier, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def explicit_dependencies(self) -> GraphMapDependency:
        """Reject redundant or empty maps before analytic formula construction."""
        for field in ("state_initials", "public_initials"):
            values = getattr(self, field)
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {field} dependency")
        if not self.state_initials and not self.public_initials:
            raise ValueError("map requires at least one explicit dependency")
        return self


class GraphKnownDependency(StrictSchema):
    """Select an already established runtime value without supplying a number."""

    mode: Literal["known"] = "known"


GraphDependencyRule = Annotated[
    GraphValueDependency | GraphMapDependency | GraphKnownDependency,
    Field(discriminator="mode"),
]


class GraphDependencyRecord(StrictSchema):
    """One latent state's scientific choice with its retained explanation."""

    state: Identifier
    rule: GraphDependencyRule
    justification: str = Field(min_length=1, max_length=2000)


class GraphDependencyHandoff(StrictSchema):
    """A complete topology choice for latent differential state initials."""

    protocol: Literal["construction-initializer-dependencies-1"] = PROTOCOL
    records: tuple[GraphDependencyRecord, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def distinct_states(self) -> GraphDependencyHandoff:
        """Do not resolve competing declarations by their arrival order."""
        names = [record.state for record in self.records]
        if len(names) != len(set(names)):
            raise ValueError("duplicate initializer state record")
        return self


class GraphConstructionContract(StrictSchema):
    """Runtime information and identities bound to reviewed source declarations.

    Known-value justifications survive export. State declaration hashes include
    scientific meaning when built from a Draft. A contract built with explicit
    names instead attests those coordinates; it cannot claim a reviewed Draft.
    The context identity ignores only the compiler's fitted-initialization flag.
    """

    protocol: Literal["construction-initializer-graph-contract-1"] = CONTRACT_PROTOCOL
    provenance: str = Field(min_length=1, max_length=8000)
    permitted_public_initials: tuple[Identifier, ...] = Field(max_length=256)
    observed_initial_channels: Mapping[Identifier, Identifier] = Field(
        default_factory=dict, max_length=64
    )
    known_initials: Mapping[Identifier, KnownInitialValue] = Field(
        default_factory=dict, max_length=64
    )
    trajectory_initialization: Literal["forbidden", "observation_prefix"] = "forbidden"
    prefix_policy: PrefixPolicy | None = None
    context_sha256: Digest
    public_context_sha256: Digest
    state_identity_kind: Literal["draft_declaration", "state_name"]
    state_declaration_sha256: Mapping[Identifier, Digest] = Field(max_length=64)

    @model_validator(mode="after")
    def coherent_runtime_bindings(self) -> GraphConstructionContract:
        """Reject conflicting fixed roles and unavailable identity sources."""
        if len(self.permitted_public_initials) != len(
            set(self.permitted_public_initials)
        ):
            raise ValueError("duplicate permitted public initial")
        states = set(self.state_declaration_sha256)
        fixed = set(self.observed_initial_channels) | set(self.known_initials)
        if extra := fixed - states:
            raise ValueError(f"runtime initial for unavailable states: {sorted(extra)}")
        if (
            overlap := self.observed_initial_channels.keys()
            & self.known_initials.keys()
        ):
            raise ValueError(f"observed/known initial conflict: {sorted(overlap)}")
        if extra := set(self.observed_initial_channels.values()) - set(
            self.permitted_public_initials
        ):
            raise ValueError(
                f"observed initial channel is not permitted: {sorted(extra)}"
            )
        if (
            self.prefix_policy is not None
            and self.trajectory_initialization != "observation_prefix"
        ):
            raise ValueError("prefix policy requires observation_prefix authorization")
        return self


def _context_digest(context: ValidationContext) -> str:
    """Bind scientific channel availability independently of compiler lowering."""
    context = ValidationContext.model_validate(context.model_dump(mode="json"))
    return content_hash(
        context.model_dump(mode="json", exclude={"fitted_initialization"})
    )


def _state_identities(
    draft: Draft | None, state_names: Sequence[str] | None
) -> tuple[str, dict[str, str]]:
    """Record either reviewed full declarations or an explicit coordinate list."""
    if (draft is None) == (state_names is None):
        raise ValueError(
            "supply exactly one of draft or state_names when building contract"
        )
    if draft is not None:
        draft = Draft.model_validate(draft.model_dump(mode="json"))
        return "draft_declaration", {
            variable.name: content_hash(variable.model_dump(mode="json"))
            for variable in draft.variables
            if variable.definition == "differential"
        }
    names = tuple(state_names or ())
    if len(names) != len(set(names)):
        raise ValueError("duplicate differential state name")
    return "state_name", {name: content_hash({"state": name}) for name in names}


def build_contract(
    draft: Draft | None = None,
    context: ValidationContext | None = None,
    *,
    state_names: Sequence[str] | None = None,
    scientific_context: str = "",
    permitted_public_initials: Sequence[str],
    observed_initial_channels: Mapping[str, str] | None = None,
    known_initials: Mapping[str, KnownInitialValue | dict] | None = None,
    trajectory_initialization: Literal["forbidden", "observation_prefix"] = "forbidden",
    prefix_policy: PrefixPolicy | None = None,
    provenance: str,
) -> GraphConstructionContract:
    """Freeze a runtime boundary without inferring observation identities.

    Only the caller can attest identities and known values. A matching state and
    channel name alone does not establish an observed identity.
    """
    if context is None:
        raise ValueError("runtime ValidationContext is required")
    identity_kind, identities = _state_identities(draft, state_names)
    contract = GraphConstructionContract.model_validate(
        {
            "provenance": provenance,
            "permitted_public_initials": tuple(permitted_public_initials),
            "observed_initial_channels": dict(observed_initial_channels or {}),
            "known_initials": dict(known_initials or {}),
            "trajectory_initialization": trajectory_initialization,
            "prefix_policy": prefix_policy,
            "context_sha256": _context_digest(context),
            "public_context_sha256": content_hash(scientific_context),
            "state_identity_kind": identity_kind,
            "state_declaration_sha256": identities,
        }
    )
    return validate_contract(
        contract,
        draft=draft,
        state_names=state_names,
        context=context,
        scientific_context=scientific_context,
    )


def _resolved_names(
    contract: GraphConstructionContract, state_aliases: Mapping[str, str] | None
) -> dict[str, str]:
    """Translate only state identities, rejecting collisions with retained names."""
    aliases = dict(state_aliases or {})
    source = set(contract.state_declaration_sha256)
    if extra := aliases.keys() - source:
        raise ValueError(f"unknown initializer state alias sources: {sorted(extra)}")
    # Reuse the schema's identifier grammar rather than accepting arbitrary text.
    for name in aliases.values():
        GraphDependencyRecord(
            state=name, rule=GraphKnownDependency(), justification="alias validation"
        )
    resolved = {state: aliases.get(state, state) for state in source}
    if len(set(resolved.values())) != len(resolved) or any(
        alias in source and state != alias for state, alias in aliases.items()
    ):
        raise ValueError("ambiguous or colliding initializer state aliases")
    return resolved


def validate_contract(
    contract: GraphConstructionContract,
    *,
    draft: Draft | None = None,
    state_names: Sequence[str] | None = None,
    context: ValidationContext | None = None,
    scientific_context: str | None = None,
    state_aliases: Mapping[str, str] | None = None,
) -> GraphConstructionContract:
    """Revalidate schema and supplied source identities before using a contract.

    ``draft`` retains original scientific names; ``state_names`` can instead name
    compiled coordinates, translated only through explicit one-to-one aliases.
    """
    contract = GraphConstructionContract.model_validate(
        contract.model_dump(mode="json")
    )
    resolved = _resolved_names(contract, state_aliases)
    if draft is not None:
        _, declarations = _state_identities(draft, None)
        if contract.state_identity_kind == "state_name":
            declarations = {
                name: content_hash({"state": name}) for name in declarations
            }
        if declarations != dict(contract.state_declaration_sha256):
            raise ValueError(
                "stale initializer state declaration or differential inventory"
            )
    if state_names is not None:
        if len(state_names) != len(set(state_names)):
            raise ValueError("duplicate differential state name")
        if set(state_names) != set(resolved.values()):
            raise ValueError("stale initializer state inventory or compiled aliases")
    if context is not None:
        if _context_digest(context) != contract.context_sha256:
            raise ValueError("stale initializer public channel context")
        public = (
            set(context.targets) | set(context.forcing_channels) | {context.time_symbol}
        )
        if extra := set(contract.permitted_public_initials) - public:
            raise ValueError(f"unknown public initial channels: {sorted(extra)}")
        measured = set(context.targets) | set(context.auxiliaries)
        if extra := set(contract.observed_initial_channels.values()) - measured:
            raise ValueError(
                f"identity-observed source is not a measured channel: {sorted(extra)}"
            )
        if contract.prefix_policy is not None and (
            extra := set(contract.prefix_policy.future_inputs)
            - set(context.external_inputs)
        ):
            raise ValueError(
                f"prefix future inputs are not external inputs: {sorted(extra)}"
            )
    if (
        scientific_context is not None
        and content_hash(scientific_context) != contract.public_context_sha256
    ):
        raise ValueError("stale initializer scientific context")
    return contract


def validate_handoff(
    handoff: GraphDependencyHandoff,
    contract: GraphConstructionContract,
    *,
    draft: Draft | None = None,
    state_names: Sequence[str] | None = None,
    context: ValidationContext | None = None,
    scientific_context: str | None = None,
    state_aliases: Mapping[str, str] | None = None,
) -> GraphDependencyHandoff:
    """Check a complete acyclic declaration and optionally translate state names."""
    contract = validate_contract(
        contract,
        draft=draft,
        state_names=state_names,
        context=context,
        scientific_context=scientific_context,
        state_aliases=state_aliases,
    )
    handoff = GraphDependencyHandoff.model_validate(handoff.model_dump(mode="json"))
    states = set(contract.state_declaration_sha256)
    observed = set(contract.observed_initial_channels)
    records = {record.state: record.rule for record in handoff.records}
    if redefined := records.keys() & observed:
        raise ValueError(
            f"cannot redefine observed state initials: {sorted(redefined)}"
        )
    if extra := records.keys() - states:
        raise ValueError(
            f"initializer record is not a differential state: {sorted(extra)}"
        )
    if missing := states - observed - records.keys():
        raise ValueError(f"missing latent initializer records: {sorted(missing)}")
    for state, rule in records.items():
        if isinstance(rule, GraphKnownDependency):
            if state not in contract.known_initials:
                raise ValueError(
                    f"known initial is not established by runtime: {state}"
                )
        elif state in contract.known_initials:
            raise ValueError(f"cannot replace runtime known initial: {state}")
        if isinstance(rule, GraphValueDependency):
            if (
                rule.scope == "trajectory"
                and contract.trajectory_initialization != "observation_prefix"
            ):
                raise ValueError(
                    "trajectory root requires observation_prefix authorization"
                )
        elif isinstance(rule, GraphMapDependency):
            if extra := set(rule.state_initials) - states:
                raise ValueError(
                    f"map state initials are not differential states at {state}: "
                    f"{sorted(extra)}"
                )
            if extra := set(rule.public_initials) - set(
                contract.permitted_public_initials
            ):
                raise ValueError(
                    f"unavailable public initial at {state}: {sorted(extra)}"
                )
    _dependency_order(handoff)
    aliases = _resolved_names(contract, state_aliases)
    translated = []
    for record in handoff.records:
        payload = record.model_dump(mode="json")
        payload["state"] = aliases[record.state]
        if isinstance(record.rule, GraphMapDependency):
            payload["rule"]["state_initials"] = [
                aliases[name] for name in record.rule.state_initials
            ]
        translated.append(payload)
    return GraphDependencyHandoff.model_validate({"records": translated})


def _dependency_order(handoff: GraphDependencyHandoff) -> tuple[str, ...]:
    """Return deterministic latent order and reject direct or indirect cycles."""
    records = {record.state: record.rule for record in handoff.records}
    visiting: set[str] = set()
    visited: set[str] = set()
    ordered: list[str] = []

    def visit(state: str) -> None:
        if state in visiting:
            raise ValueError(f"cyclic initializer state dependency at {state}")
        if state in visited or state not in records:
            return
        visiting.add(state)
        rule = records[state]
        if isinstance(rule, GraphMapDependency):
            for dependency in sorted(rule.state_initials):
                visit(dependency)
        visiting.remove(state)
        visited.add(state)
        ordered.append(state)

    for state in sorted(records):
        visit(state)
    return tuple(ordered)


def initializer_aliases(rule: GraphDependencyRule) -> dict[str, dict[str, str]]:
    """Generate deterministic disjoint aliases for the later formula namespace."""
    if not isinstance(rule, GraphMapDependency):
        return {"state_initials": {}, "public_initials": {}}
    return {
        "state_initials": {
            f"state_{index}": name
            for index, name in enumerate(sorted(rule.state_initials), 1)
        },
        "public_initials": {
            f"public_{index}": name
            for index, name in enumerate(sorted(rule.public_initials), 1)
        },
    }


def dependency_summary(
    handoff: GraphDependencyHandoff, contract: GraphConstructionContract
) -> dict[str, dict[str, list[str]]]:
    """Expand transitive dependency scopes without selecting or evaluating maps.

    A trajectory root anywhere upstream prevents a deterministic t0 equality
    claim. Observed dependencies resolve to their exact runtime-bound channel.
    """
    handoff = validate_handoff(handoff, contract)
    result: dict[str, dict[str, list[str]]] = {}
    fields = (
        "state_initials",
        "public_initials",
        "shared_training_roots",
        "trajectory_roots",
        "known_initials",
    )
    for state, channel in contract.observed_initial_channels.items():
        result[state] = {field: [] for field in fields}
        result[state]["public_initials"] = [channel]
    records = {record.state: record.rule for record in handoff.records}
    for state in _dependency_order(handoff):
        rule = records[state]
        sources: dict[str, set[str]] = {field: set() for field in fields}
        if isinstance(rule, GraphValueDependency):
            sources[f"{rule.scope}_roots"].add(state)
        elif isinstance(rule, GraphKnownDependency):
            sources["known_initials"].add(state)
        else:
            sources["public_initials"].update(rule.public_initials)
            sources["state_initials"].update(rule.state_initials)
            for dependency in rule.state_initials:
                for field in fields:
                    sources[field].update(result[dependency][field])
        result[state] = {field: sorted(values) for field, values in sources.items()}
    return {state: result[state] for state in sorted(result)}
