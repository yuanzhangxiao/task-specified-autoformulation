"""Judge-independent mechanism coverage and structural graph metrics."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.expressions import RestrictedParser
from autoformalism.schemas import CandidateModel


class MechanismRequirement(BaseModel):
    """Public task predicate for one required generated mechanism."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    public_requirement: str | None = None
    tag_aliases: tuple[str, ...] = ()
    required_drivers: tuple[str, ...] = ()
    required_targets: tuple[str, ...] = ()
    must_be_generated: bool = True
    requires_dynamic_memory: bool = False
    required_sign: Literal["positive", "negative", "unspecified"] = "unspecified"
    forbid_current_future_target_input: bool = True


class MechanismEvaluationSpec(BaseModel):
    """Public benchmark/tier mechanism-evaluation configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["mechanism-evaluation-spec-1"] = (
        "mechanism-evaluation-spec-1"
    )
    source: Literal["legacy", "public_prompt"] = "legacy"
    benchmark_id: str
    tier: str
    public_prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    required_mechanisms: tuple[MechanismRequirement, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def public_specs_are_prompt_committed(self) -> MechanismEvaluationSpec:
        """Require provenance for prospective public-prompt specifications."""
        identifiers = [item.id for item in self.required_mechanisms]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("mechanism requirement identifiers must be unique")
        if self.source == "public_prompt":
            if self.public_prompt_sha256 is None:
                raise ValueError("public-prompt specification requires prompt SHA-256")
            if any(not item.public_requirement for item in self.required_mechanisms):
                raise ValueError(
                    "public-prompt requirements require their public source text"
                )
        return self


class PredicateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanism_id: str
    predicate: str
    status: Literal["satisfied", "failed", "ambiguous"]
    evidence: str


class MechanismComplianceResult(BaseModel):
    """Conjunctive certification for one public mechanism requirement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanism_id: str
    status: Literal["satisfied", "failed", "ambiguous"]
    applicable_predicates: int = Field(ge=1)
    satisfied_predicates: int = Field(ge=0)
    predicates: tuple[PredicateResult, ...] = Field(min_length=1)


class MechanismEvaluation(BaseModel):
    """Independent coverage and structural-validity result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanism_coverage: float
    mechanism_compliance: float
    mechanism_compliance_complete: bool
    structural_validity: float
    covered_mechanisms: tuple[str, ...]
    compliant_mechanisms: tuple[str, ...]
    mechanism_results: tuple[MechanismComplianceResult, ...]
    predicates: tuple[PredicateResult, ...]
    manual_review_required: bool


def evaluate_mechanisms(
    candidate: CandidateModel, spec: MechanismEvaluationSpec
) -> MechanismEvaluation:
    """Evaluate explicit tagged claims against the expression dependency graph."""
    graph = _dependency_graph(candidate)
    reverse = _reverse_graph(graph)
    latent_state_names = {
        item.name for item in candidate.states if item.kind.value == "latent"
    }
    tags: dict[str, set[str]] = defaultdict(set)
    for state in candidate.states:
        for tag in state.mechanisms:
            tags[_normalize_tag(tag)].add(state.name)
    for process in candidate.processes:
        for tag in process.mechanisms:
            tags[_normalize_tag(tag)].add(process.name)

    predicates: list[PredicateResult] = []
    covered: list[str] = []
    mechanism_results: list[MechanismComplianceResult] = []
    for requirement in spec.required_mechanisms:
        legacy_predicates: list[PredicateResult] = []
        accepted_tags = tuple(
            _normalize_tag(tag) for tag in (requirement.id, *requirement.tag_aliases)
        )
        claims = set().union(*(tags.get(tag, set()) for tag in accepted_tags))
        target_nodes = {f"target:{name}" for name in requirement.required_targets}
        connected_claims = {
            claim
            for claim in claims
            if not target_nodes or _reaches_any(graph, claim, target_nodes)
        }
        coverage_ok = bool(connected_claims)
        if coverage_ok:
            covered.append(requirement.id)
        legacy_predicates.append(
            PredicateResult(
                mechanism_id=requirement.id,
                predicate="generated_connected_coverage",
                status="satisfied" if coverage_ok else "failed",
                evidence=(
                    f"connected tagged components: {sorted(connected_claims)}"
                    if coverage_ok
                    else (
                        "tagged components without required target path: "
                        f"{sorted(claims)}"
                    )
                ),
            )
        )
        for driver in requirement.required_drivers:
            driven = any(
                driver in _ancestors(reverse, claim) for claim in connected_claims
            )
            legacy_predicates.append(
                PredicateResult(
                    mechanism_id=requirement.id,
                    predicate=f"driver:{driver}",
                    status="satisfied" if driven else "failed",
                    evidence=f"driver ancestry checked for {sorted(connected_claims)}",
                )
            )
        if requirement.requires_dynamic_memory:
            dynamic_components = {
                claim
                for claim in connected_claims
                if claim in latent_state_names
                or bool(_ancestors(reverse, claim) & latent_state_names)
            }
            dynamic = bool(dynamic_components)
            legacy_predicates.append(
                PredicateResult(
                    mechanism_id=requirement.id,
                    predicate="dynamic_memory",
                    status="satisfied" if dynamic else "failed",
                    evidence=(
                        "tagged components with latent-state memory: "
                        f"{sorted(dynamic_components)}"
                    ),
                )
            )
        if requirement.required_sign != "unspecified":
            legacy_predicates.append(
                PredicateResult(
                    mechanism_id=requirement.id,
                    predicate=f"regulatory_sign:{requirement.required_sign}",
                    status="ambiguous",
                    evidence="nonlinear multi-path sign requires blinded manual review",
                )
            )
        predicates.extend(legacy_predicates)
        compliance_predicates = _compliance_predicates(
            requirement,
            claims=claims,
            graph=graph,
            reverse=reverse,
            latent_state_names=latent_state_names,
        )
        statuses = {item.status for item in compliance_predicates}
        status: Literal["satisfied", "failed", "ambiguous"]
        if "failed" in statuses:
            status = "failed"
        elif "ambiguous" in statuses:
            status = "ambiguous"
        else:
            status = "satisfied"
        mechanism_results.append(
            MechanismComplianceResult(
                mechanism_id=requirement.id,
                status=status,
                applicable_predicates=len(compliance_predicates),
                satisfied_predicates=sum(
                    item.status == "satisfied" for item in compliance_predicates
                ),
                predicates=compliance_predicates,
            )
        )
    scored = [item for item in predicates if item.status != "ambiguous"]
    validity = (
        sum(item.status == "satisfied" for item in scored) / len(scored)
        if scored
        else 0.0
    )
    compliant = tuple(
        item.mechanism_id for item in mechanism_results if item.status == "satisfied"
    )
    return MechanismEvaluation(
        mechanism_coverage=len(covered) / len(spec.required_mechanisms),
        mechanism_compliance=len(compliant) / len(spec.required_mechanisms),
        mechanism_compliance_complete=all(
            item.status != "ambiguous" for item in mechanism_results
        ),
        structural_validity=validity,
        covered_mechanisms=tuple(covered),
        compliant_mechanisms=compliant,
        mechanism_results=tuple(mechanism_results),
        predicates=tuple(predicates),
        manual_review_required=any(item.status == "ambiguous" for item in predicates),
    )


def mechanism_claim_components(
    candidate: CandidateModel,
    spec: MechanismEvaluationSpec,
) -> dict[str, tuple[str, ...]]:
    """Return tag-matched candidate components for each public requirement."""
    tagged: dict[str, set[str]] = defaultdict(set)
    for component in (*candidate.states, *candidate.processes):
        for tag in component.mechanisms:
            tagged[_normalize_tag(tag)].add(component.name)
    return {
        requirement.id: tuple(
            sorted(
                set().union(
                    *(
                        tagged.get(_normalize_tag(tag), set())
                        for tag in (requirement.id, *requirement.tag_aliases)
                    )
                )
            )
        )
        for requirement in spec.required_mechanisms
    }


def _compliance_predicates(
    requirement: MechanismRequirement,
    *,
    claims: set[str],
    graph: dict[str, set[str]],
    reverse: dict[str, set[str]],
    latent_state_names: set[str],
) -> tuple[PredicateResult, ...]:
    """Build the prospective D/R/P/M/S conjunction for one requirement."""
    results: list[PredicateResult] = []
    if requirement.must_be_generated:
        results.append(
            PredicateResult(
                mechanism_id=requirement.id,
                predicate="declared_component",
                status="satisfied" if claims else "failed",
                evidence=f"tag-matched declared components: {sorted(claims)}",
            )
        )
    connected_claims: set[str] = set()
    for target in requirement.required_targets:
        target_node = f"target:{target}"
        reaching = {
            claim for claim in claims if _reaches_any(graph, claim, {target_node})
        }
        connected_claims.update(reaching)
        results.append(
            PredicateResult(
                mechanism_id=requirement.id,
                predicate=f"target_path:{target}",
                status="satisfied" if reaching else "failed",
                evidence=(
                    f"declared components reaching target {target}: {sorted(reaching)}"
                ),
            )
        )
    if not requirement.required_targets:
        connected_claims = set(claims)
    for driver in requirement.required_drivers:
        driven = {
            claim
            for claim in connected_claims
            if driver == claim or driver in _ancestors(reverse, claim)
        }
        results.append(
            PredicateResult(
                mechanism_id=requirement.id,
                predicate=f"driver:{driver}",
                status="satisfied" if driven else "failed",
                evidence=(
                    f"target-connected components with driver {driver}: "
                    f"{sorted(driven)}"
                ),
            )
        )
    if requirement.requires_dynamic_memory:
        dynamic = {
            claim
            for claim in connected_claims
            if claim in latent_state_names
            or bool(_ancestors(reverse, claim) & latent_state_names)
        }
        results.append(
            PredicateResult(
                mechanism_id=requirement.id,
                predicate="dynamic_memory",
                status="satisfied" if dynamic else "failed",
                evidence=(
                    "target-connected components with latent-state memory: "
                    f"{sorted(dynamic)}"
                ),
            )
        )
    if requirement.required_sign != "unspecified":
        results.append(
            PredicateResult(
                mechanism_id=requirement.id,
                predicate=f"regulatory_sign:{requirement.required_sign}",
                status="ambiguous",
                evidence=(
                    "nonlinear multi-path sign is not deterministically "
                    "certified by this evaluator"
                ),
            )
        )
    if not results:
        raise ValueError(
            f"mechanism requirement {requirement.id!r} has no applicable predicate"
        )
    return tuple(results)


def _dependency_graph(candidate: CandidateModel) -> dict[str, set[str]]:
    parser = RestrictedParser()
    graph: dict[str, set[str]] = defaultdict(set)
    for process in candidate.processes:
        parsed = parser.parse(process.expression, location=f"process:{process.name}")
        for symbol in parsed.symbols:
            graph[symbol].add(process.name)
    for equation in candidate.state_equations:
        parsed = parser.parse(equation.rhs, location=f"equation:{equation.state}")
        for symbol in parsed.symbols:
            graph[symbol].add(equation.state)
    for mapping in candidate.observation_mappings:
        parsed = parser.parse(
            mapping.expression, location=f"observation:{mapping.channel}"
        )
        for symbol in parsed.symbols:
            graph[symbol].add(f"target:{mapping.channel}")
    return {key: set(values) for key, values in graph.items()}


def _reverse_graph(graph: dict[str, set[str]]) -> dict[str, set[str]]:
    reverse: dict[str, set[str]] = defaultdict(set)
    for source, destinations in graph.items():
        for destination in destinations:
            reverse[destination].add(source)
    return {key: set(values) for key, values in reverse.items()}


def _reaches_any(graph: dict[str, set[str]], source: str, targets: set[str]) -> bool:
    pending = deque([source])
    visited = {source}
    while pending:
        current = pending.popleft()
        if current in targets:
            return True
        for item in graph.get(current, set()) - visited:
            visited.add(item)
            pending.append(item)
    return False


def _ancestors(reverse: dict[str, set[str]], node: str) -> set[str]:
    pending = deque([node])
    visited: set[str] = set()
    while pending:
        current = pending.popleft()
        for item in reverse.get(current, set()) - visited:
            visited.add(item)
            pending.append(item)
    return visited


def _normalize_tag(tag: str) -> str:
    """Normalize superficial case/separator variation without semantic matching."""
    return re.sub(r"[^a-z0-9]+", "_", tag.casefold()).strip("_")
