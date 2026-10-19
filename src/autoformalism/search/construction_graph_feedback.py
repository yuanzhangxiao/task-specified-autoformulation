"""Opt-in deterministic acceptance and evidence cards for initializer repair.

All current conditional conflicts block this policy. Unknown or deferred evidence
does not become a passed scientific check. Historical diagnostic policies retain
their original advisory behavior; this module only gates the new repair path.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

from autoformalism.search.construction_graph_evidence import (
    inspect_graph_initialization,
)
from autoformalism.search.construction_initialization_graph import (
    GraphConstructionContract,
    GraphDependencyHandoff,
)
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.construction_readable import model_view
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    _closure,
    _forcing_equal,
    _graph,
    _value,
)
from autoformalism.search.topology_initialization import _initial_scope
from autoformalism.staged_topology import content_hash

PROTOCOL = "initialization-model-data-acceptance-1"
_SURFACES = ("retained_witness_report", "initial_time_report", "declared_report")
_UNRESOLVED = {
    "deferred",
    "unavailable",
    "invalid_saved_witness",
    "uncertainty_limited",
    "retained_unresolved",
}


def _coordinates(witness: dict, target: str, evidence: TrainingCompatibilityEvidence):
    """Read actual samples by coordinates; never display historical scalar copies."""
    ids, indices = witness["trajectories"], witness["sample_indices"]
    if len(ids) == 1:
        ids = list(ids) * len(indices)
    if not indices or len(ids) != len(indices):
        raise ValueError("conflict witness coordinates are incomplete")
    trajectories = {item.trajectory_id: item for item in evidence.trajectories}
    points = []
    for name, index in zip(ids, indices, strict=True):
        item = trajectories[name]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("conflict witness sample index is invalid")
        points.append(
            {
                "trajectory": name,
                "sample_index": index,
                "time": item.time[index],
                "observed_target": item.targets[target][index],
            }
        )
    key = content_hash(sorted((p["trajectory"], p["sample_index"]) for p in points))
    return key, points


def _initializer_relations(
    states: set[str],
    contract: GraphConstructionContract,
    handoff: GraphDependencyHandoff,
) -> list[str]:
    """Show current dependency equations, including transitive state initials."""
    records = {record.state: record.rule for record in handoff.records}
    scope, pending = set(states), list(states)
    while pending:
        rule = records.get(pending.pop())
        for parent in getattr(rule, "state_initials", ()):
            if parent not in scope:
                scope.add(parent)
                pending.append(parent)
    equations = []
    for state in sorted(scope):
        if state in contract.observed_initial_channels:
            rhs = f"recorded {contract.observed_initial_channels[state]}(t0)"
        elif state in contract.known_initials:
            rhs = "the fixed runtime-known value"
        else:
            rule = records[state]
            if rule.mode == "value":
                rhs = (
                    "one shared training constant"
                    if rule.scope == "shared_training"
                    else "one trajectory-specific root inferred from the allowed prefix"
                )
            elif rule.mode == "map":
                args = [f"{name}(t0)" for name in rule.state_initials]
                args += [f"recorded {name}(t0)" for name in rule.public_initials]
                rhs = f"g_{state}({', '.join(args)}; shared parameters)"
            else:
                rhs = "the fixed runtime-known value"
        equations.append(f"{state}(t0) = {rhs}")
    return equations


def _witness_card(points, target, property_, draft, evidence, contract, handoff):
    """Expose the complete permitted t0 table and qualified forcing comparison."""
    if property_ not in {"closed_subsystem", "algebraic_map", "scalar_autonomous"}:
        raise ValueError("unsupported current conflict property")
    trajectories = {item.trajectory_id: item for item in evidence.trajectories}
    ids = list(dict.fromkeys(point["trajectory"] for point in points))
    selected = [trajectories[name] for name in ids]
    graph, kinds, _ = _graph(draft, evidence.context)
    initial_only = (
        all(point["sample_index"] == 0 for point in points)
        and property_ == "closed_subsystem"
    )
    scope = (
        _initial_scope(target, graph, kinds)
        if initial_only
        else _closure(target, graph)
    )
    states = {name for name in scope if kinds.get(name) == "differential"}
    public_values = []
    for channel in contract.permitted_public_initials:
        values = [
            _value(item, channel, 0, evidence.context.time_symbol) for item in selected
        ]
        public_values.append(
            {"channel": channel, "values": values, "equal": len(set(values)) == 1}
        )
    view = model_view(draft)
    relation = [
        row["topology"]
        for row in view["balances"]
        if row["variable"] in scope
        and not (initial_only and row["type"] == "differential")
    ]
    relation += [row["topology"] for row in view["processes"] if row["name"] in scope]
    result = {
        "observations": points,
        "evidence_sha256": evidence.sha256,
        "target_absolute_error_bound": evidence.contract.absolute_error_bounds.get(
            target
        ),
        "model_relation": relation,
        "initializer_relations": _initializer_relations(states, contract, handoff),
        "paired_permitted_initial_measurements": {
            "trajectories": ids,
            "rows": public_values,
            "source": "Re-read from sealed training arrays at each trajectory's t0.",
            "interpretation": "All permitted initial channels are shown. Equality of a "
            "map's complete chosen arguments forces equal map outputs; a differing "
            "unused channel is a possible distinction, not a prescribed dependency.",
        },
    }
    if initial_only:
        result["conflict"] = (
            f"The current initializer/readout dependencies force the same {target}(t0) "
            "for these matched initial arguments, but the recorded values differ. "
            "Later ODE evolution cannot change an already fixed initial value."
        )
        result["logical_assumptions"] = [
            "Initial maps and readouts are shared deterministic functions.",
            "Conditioning values are exact under the recorded evidence contract.",
        ]
    elif property_ == "closed_subsystem":
        end = min(point["time"] for point in points)
        equal, missing = _forcing_equal(
            selected[0], selected[-1], scope - set(graph), evidence, end
        )
        if missing or not equal:
            raise ValueError("current conflict forcing no longer matches its evidence")
        result["forcing_prefix"] = {
            "start": selected[0].time[0],
            "end": end,
            "equal": True,
            "channels": sorted(scope - set(graph) - {evidence.context.time_symbol}),
            "verification": "Compared complete defining interpolation knots through "
            "this observation; later input differences cannot explain it.",
        }
        result["conflict"] = (
            "Equal initials and equal forcing through the displayed time imply "
            f"equal {target} "
            "under this closed subsystem, but the recorded values differ. Other states "
            "cannot distinguish it without a declared dependence."
        )
        result["logical_assumptions"] = [
            "Shared deterministic dynamics have unique solutions on this interval, "
            "without resets or impulses.",
            "The subsystem contains all declared ancestors; conditioning values "
            "follow the recorded evidence contract.",
        ]
    elif property_ == "algebraic_map":
        result["matched_algebraic_arguments"] = [
            {
                "channel": name,
                "values": [
                    _value(
                        trajectories[point["trajectory"]],
                        name,
                        point["sample_index"],
                        evidence.context.time_symbol,
                    )
                    for point in points
                ],
            }
            for name in sorted(scope - set(graph))
        ]
        result["conflict"] = (
            "The complete chosen algebraic arguments match, while recorded "
            f"{target} differs. One shared deterministic algebraic function cannot "
            "reproduce both values exactly."
        )
        result["logical_assumptions"] = [
            "The algebraic functions and parameters are shared and deterministic."
        ]
    else:
        result["conflict"] = (
            f"Recorded {target} reverses direction with constant external arguments; "
            "a directly observed closed scalar autonomous ODE with unique solutions "
            "and no resets cannot reproduce this reversal exactly."
        )
        result["logical_assumptions"] = [
            "The scalar autonomous ODE has unique solutions and no resets or impulses."
        ]
    result["logical_assumptions"].append(
        "This tests exact consistency with recorded values under the displayed "
        "measurement-error bound; it is not a numerical fit-quality score."
    )
    return result


def acceptance_feedback(
    draft: Draft,
    evidence: TrainingCompatibilityEvidence,
    contract: GraphConstructionContract,
    handoff: GraphDependencyHandoff,
    previous_questions: list[dict] | tuple[dict, ...] = (),
) -> dict:
    """Recompute a candidate and gate conflicts with concise repair evidence.

    A representative pair is displayed per conflicting property/target; all unique
    witness coordinates and diagnostic sources remain in the machine-readable gate.
    Acceptance means no currently demonstrated contradiction under this policy,
    not that deferred witnesses were resolved or scientific correctness established.
    """
    assessment = inspect_graph_initialization(
        draft, evidence, contract, handoff, previous_questions
    )
    groups, deferred = {}, []
    for surface in _SURFACES:
        report = assessment[surface]
        for diagnostic in report["diagnostics"]:
            if diagnostic["status"] != "conditional_conflict":
                if diagnostic["status"] == "deferred":
                    deferred.append(
                        {
                            "source": surface,
                            "kind": "diagnostic",
                            **deepcopy(diagnostic),
                        }
                    )
                continue
            property_, target = diagnostic["property"], diagnostic["target"]
            group = groups.setdefault(
                (property_, target), {"rows": [], "witnesses": {}}
            )
            group["rows"].append((surface, diagnostic))
            for witness in diagnostic["witnesses"]:
                key, points = _coordinates(witness, target, evidence)
                group["witnesses"].setdefault(key, points)
        for check in report.get("checks", ()):
            if check["status"] in _UNRESOLVED:
                deferred.append(
                    {"source": surface, "kind": "saved_witness", **deepcopy(check)}
                )
    cards, witnesses = [], []
    for (property_, target), group in sorted(groups.items()):
        if not group["witnesses"]:
            raise ValueError("conditional conflict has no current witness")
        # Prefer the retained historical counterexample so feedback stays stable;
        # every surface still contributes to gating and the complete audit index.
        representative = next(iter(group["witnesses"].values()))
        sources = list(dict.fromkeys(surface for surface, _ in group["rows"]))
        card = {
            "property": property_,
            "target": target,
            "conflict_id": content_hash(
                [PROTOCOL, property_, target, sorted(group["witnesses"])]
            )[:20],
            "status": "requires_repair",
            "sources": sources,
            "unique_witness_count": len(group["witnesses"]),
            **_witness_card(
                representative, target, property_, draft, evidence, contract, handoff
            ),
        }
        cards.append(card)
        witnesses.extend(
            {"property": property_, "target": target, "points": points}
            for points in group["witnesses"].values()
        )
    blocked = bool(cards)
    return {
        "protocol": PROTOCOL,
        "assessment": assessment,
        "blocked": blocked,
        "eligible": not blocked,
        "status": "model_data_conflict"
        if blocked
        else ("eligible_with_deferred_evidence" if deferred else "eligible"),
        "conflict_count": len(cards),
        "witness_count": len(witnesses),
        "conflicts": cards,
        "all_conflict_witnesses": witnesses,
        "deferred_count": len(deferred),
        "deferred_checks": deferred,
        "deferred_status_counts": dict(
            sorted(Counter(row["status"] for row in deferred).items())
        ),
        "repair_options": [
            "Revise the complete latent initializer graph using scientifically "
            "justified permitted initial distinctions. Matching complete map "
            "arguments still imply matching outputs.",
            *(
                [
                    "If hidden initial conditions vary independently, an authorized "
                    "trajectory root may represent that preparation. Its equality "
                    "evidence remains deferred until prefix inference and "
                    "identifiability are assessed; this is not a demonstrated repair."
                ]
                if contract.trajectory_initialization == "observation_prefix"
                else []
            ),
            "If no scientifically justified initializer can resolve the contradiction "
            "with the frozen dynamics/readout, report needs_topology_revision. Do "
            "not alter observed identities or use forbidden information. Acknowledging "
            "the conflict alone does not satisfy this check.",
        ],
        "scientific_repair_established": False,
        "interpretation": "Current model-data contradictions block acceptance under "
        "their displayed assumptions. No conflict found is not scientific correctness; "
        "deferred or unavailable evidence remains unresolved. This gate does not fit "
        "functions or infer trajectory roots.",
        "llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }
