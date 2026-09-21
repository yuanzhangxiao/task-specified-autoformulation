"""Opt-in, transactional dependency corrections during function construction.

Public fixed covariates can be added mechanically. Every other source change
requires an explicit proposer decision. Neither prose nor a process name grants
permission to change signs, consumers, units, boundaries, or available channels.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.schemas.staged_functions import InteractionFunctionReply
from autoformalism.schemas.staged_topology import EquationDefinition, ScientificVariable
from autoformalism.search import shared_process_contract as shared
from autoformalism.search import signed_processes as signed
from autoformalism.staged_topology import (
    _ancestors,
    content_hash,
    lower_topology,
    public_structure_checks,
)

POLICY = "local-function-dependencies-1"


def required_checks(brief, equations) -> list[dict]:
    """Retain legacy predicates and enforce the typed dynamic-memory flag too."""
    checks = list(public_structure_checks(brief, equations))
    graph = {e.name: {n for t in e.terms for n in t.sources} for e in equations}
    dynamic = {e.name for e in equations if e.definition == "differential"}
    for requirement in brief.requirements:
        if not requirement.requires_dynamic_memory or requirement.positive_requirements:
            continue  # Legacy checks already cover the latter case.
        for target in requirement.targets:
            ancestors = _ancestors(target, graph)
            for driver in requirement.drivers:
                checks.append(
                    {
                        "requirement": requirement.id,
                        "kind": "memory_path",
                        "target": target,
                        "driver": driver,
                        "passed": any(
                            n in ancestors and driver in _ancestors(n, graph)
                            for n in dynamic - {target, driver}
                        ),
                    }
                )
    return checks


class DependencyFunctionReply(InteractionFunctionReply):
    """The expression determines names; no redundant source list is requested."""

    revise_dependencies: bool = False


def plain(reply: InteractionFunctionReply) -> InteractionFunctionReply:
    """Keep the legacy expression/parameter schema at the compilation boundary."""
    return InteractionFunctionReply.model_validate(
        reply.model_dump(include={"expression", "parameters"})
    )


def slot(source: dict, identifier: str) -> tuple[int, int]:
    """Resolve only an existing runtime slot, never an arbitrary provider path."""
    match = re.fullmatch(r"term_(\d+)_(\d+)", identifier)
    if match is None:
        raise ValueError("unknown interaction identifier")
    i, j = map(int, match.groups())
    try:
        source["equations"][i]["terms"][j]
    except (IndexError, KeyError) as exc:
        raise ValueError("unknown interaction identifier") from exc
    return i, j


def fixed_sources(brief, context, source: dict) -> set[str]:
    """Only selected, supplied, public fixed quantities are implicit additions."""
    return (
        set(context.fixed_covariates)
        & {v.name for v in brief.public_variables if v.data_role == "covariate"}
        & {v["name"] for v in source["inventory"] if v["definition"] == "supplied"}
    )


def prepare(
    brief,
    context,
    source: dict,
    identifier: str,
    reply,
    *,
    preserve_process_paths=False,
):
    """Return a proposed source transaction; caller commits only after binding.

    Exact shared consumers remain immutable. A law dependency edit updates both the
    equation and its shared declaration, so every consumer still refers to one law.
    """
    i, j = slot(source, identifier)
    previous = source["equations"][i]["terms"][j]["sources"]
    expected = set(previous)
    parsed = RestrictedParser().parse(reply.expression, location="dependency repair")
    actual = set(parsed.symbols) - {p.name for p in reply.parameters}
    if actual == expected:
        return source, None
    explicit = bool(getattr(reply, "revise_dependencies", False))
    allowed = {v["name"] for v in source["inventory"] if v["definition"] != "unused"}
    if actual - allowed:
        raise ValueError(f"unavailable dependency names: {sorted(actual - allowed)}")
    if not explicit and (
        expected - actual or actual - expected - fixed_sources(brief, context, source)
    ):
        raise ValueError(
            f"dependency revision required: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}. Correct the expression, or set "
            "revise_dependencies=true to propose its actual dependencies. "
            "Required public paths and shared consumer identities still apply."
        )
    updated = deepcopy(source)
    # Preserve order of retained declarations; new names have deterministic order.
    names = [n for n in previous if n in actual] + sorted(actual - expected)
    updated["equations"][i]["terms"][j]["sources"] = names
    contract = updated.get("shared_process_contract")
    lhs = updated["equations"][i]["name"]
    for binding in (contract or {}).get("bindings", []):
        if binding["proposal"]["name"] == lhs:
            binding["proposal"]["depends_on"] = names
            if "signed_declaration" in binding:
                binding["signed_declaration"]["depends_on"] = names
                # Rebuild from the typed declaration; signs/conversions are untouched.
                rebuilt = signed.binding_for(
                    signed.SignedProcess.model_validate(binding["signed_declaration"])
                )
                binding.clear()
                binding.update(rebuilt)
    inventory = tuple(
        ScientificVariable.model_validate(v) for v in updated["inventory"]
    )
    equations = tuple(
        EquationDefinition.model_validate(e) for e in updated["equations"]
    )
    shared.validate_contract(contract, equations, inventory)
    if preserve_process_paths:
        from autoformalism.search.process_assembly_contract import preserve_paths

        preserve_paths(brief, source, updated)
    topology, _ = lower_topology(brief, inventory, equations, context)
    checks = required_checks(brief, equations)
    failed = [c for c in checks if not c["passed"]]
    if failed:
        raise ValueError(
            "dependency revision breaks public pathways: " + json.dumps(failed)
        )
    updated["topology"] = topology.model_dump(mode="json")
    updated["public_structure_checks"] = list(checks)
    updated["public_structure_checks_passed"] = True
    event = {
        "interaction_id": identifier,
        "kind": "proposer_revision" if explicit else "public_fixed_covariates",
        "before_sources": list(previous),
        "after_sources": names,
        "reply": plain(reply).model_dump(mode="json"),
        "revise_dependencies": explicit,
        "before_sha256": content_hash(source),
        "after_sha256": content_hash(updated),
    }
    return updated, event


def replay(brief, context, original: dict, result: dict) -> tuple[dict, dict]:
    """Independently reproduce committed edits and reject ledger tampering."""
    from autoformalism.search import process_assembly_contract as assembly

    assembly_policy = result.get("assembly_policy", "legacy")
    assembly.validate_policy(assembly_policy)
    if result.get("dependency_policy", "strict") == "strict":
        if result.get("dependency_revisions") or result.get("effective_source"):
            raise ValueError("dependency edits require an explicit policy")
        return original, {}
    if result["dependency_policy"] != POLICY:
        raise ValueError("unknown dependency policy")
    if result["source_topology_result_sha256"] != content_hash(original):
        raise ValueError("dependency source identity differs")
    current, before = original, {}
    indices = []
    for event in result["dependency_revisions"]:
        identifier = event["interaction_id"]
        indices.append(slot(current, identifier))
        if identifier in before or indices != sorted(indices):
            raise ValueError("dependency revisions must follow unique slot order")
        before[identifier] = current
        reply = DependencyFunctionReply(
            **event["reply"], revise_dependencies=event["revise_dependencies"]
        )
        updated, expected = prepare(
            brief,
            context,
            current,
            identifier,
            reply,
            preserve_process_paths=assembly_policy == assembly.POLICY,
        )
        if expected != event:
            raise ValueError("dependency revision ledger differs")
        current = updated
    if current != result["effective_source"]:
        raise ValueError("effective dependency source differs from replay")
    if any(
        not c["passed"]
        for c in required_checks(
            brief,
            tuple(EquationDefinition.model_validate(e) for e in current["equations"]),
        )
    ):
        raise ValueError("effective dependency source fails public pathways")
    return current, before


def prompt(system: str, *, atomic: bool) -> str:
    """Replace contradictory frozen-source instructions only for the opt-in arm."""
    # New instructions stand alone; legacy prompts remain byte-for-byte unchanged.
    prefix = system.split("Your response fills only", 1)[-1] if atomic else None
    if atomic:
        # Keep the grammar/sign/role rules, omit the obsolete no-topology-edit tail.
        rules = (
            "Your response fills only" + prefix.split("Do not emit an assignment", 1)[0]
        )
    else:
        rules = system.split("Each slot contains a runtime-owned", 1)[-1]
        rules = (
            "Each slot contains a runtime-owned"
            + rules.split("Do not emit assignments", 1)[0]
        )
    return (
        "Assign scalar interaction functions for the runtime-selected slots. "
        "Return only the requested JSON schema. A batch must match the displayed "
        "slot order and length. The displayed source list is the current decision. "
        "Use its variables plus any needed permitted_fixed_covariates. Do not add "
        "dummy factors such as area/area merely to mention a source. "
        + (
            "For this local repair, set revise_dependencies=true only if you intend "
            "to correct the selected term's source list to the variables actually "
            "used in your expression. Otherwise set it false. You may use only "
            "displayed available_dependencies. The runtime recomputes the graph "
            "and rejects lost required paths, algebraic cycles or altered shared "
            "consumer identities. No signs, consumers or other functions can change. "
            if atomic
            else "If another dependency change is needed, the runtime will request a "
            "local repair that permits an explicit dependency revision. "
        )
        + "A process has ONE defining function; consumer slots reuse it. Inspect "
        "process_consumer_conversions: these conversions are already assigned at "
        "the uses, outside this law. Do not duplicate them in the law. Units and "
        "scientific adequacy are your responsibility; prose is not a certificate.\n"
        + rules
    )


def request_context(text: str, brief, context, source: dict) -> str:
    """Expose source choices and exact assembly conversions, without hidden data."""
    lead, raw = text.split("\n", 1)
    value = json.loads(raw)
    value["dependency_policy"] = POLICY
    value["permitted_fixed_covariates"] = sorted(fixed_sources(brief, context, source))
    value["available_dependencies"] = [
        v["name"] for v in source["inventory"] if v["definition"] != "unused"
    ]
    value["frozen_equation_sketch"] = source["equations"]
    value["process_consumer_conversions"] = {
        b["proposal"]["name"]: b["signed_declaration"]["uses"]
        for b in (source.get("shared_process_contract") or {}).get("bindings", [])
        if "signed_declaration" in b
    }
    return lead + "\n" + json.dumps(value, sort_keys=True, separators=(",", ":"))


def connectivity(candidate: dict | None) -> dict:
    """Report disconnected modeled processes without guessing physical meaning."""
    if candidate is None:
        return {"available": False}
    from autoformalism.schemas import CandidateModel
    from autoformalism.targets import _dependency_graph, _reaches

    model = CandidateModel.model_validate(candidate)
    graph = _dependency_graph(model)
    rows = [
        {
            "process": p.name,
            "targets": [
                m.channel
                for m in model.observation_mappings
                if _reaches(graph, p.name, "target:" + m.channel)
            ],
        }
        for p in model.processes
    ]
    return {
        "available": True,
        "processes": rows,
        "without_target_path": [r["process"] for r in rows if not r["targets"]],
        "scope": (
            "Syntactic reachability only; no physical outlet or balance certification."
        ),
        "automatic_equation_repair": False,
    }
