"""Scope-aware presentation of existing certificates and structural edit failures.

This module does not run a scientific check, relax an acceptance condition, or
infer a replacement model. Witness selection uses only already-saved evidence.
"""

from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from itertools import combinations
from math import isfinite
from typing import Any

from autoformalism.expressions import ModelValidationError
from autoformalism.research.construction_revision_edits import (
    _bounded_ast,
    _canonical_rhs,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "construction-structural-feedback-1"
_ACTIONS = {
    "edits": 32,
    "new_parameters": 64,
    "new_states": 8,
    "new_processes": 16,
    "remove_states": 8,
    "remove_processes": 16,
    "initialization_edits": 16,
}
_REPAIR_SCOPE = (
    "Return a complete patch relative to current_model: whole RHS replacements, "
    "new latent states with typed initializers, new algebraic processes, new "
    "parameters, and edits to eligible latent initializers are supported. "
    "Topology is derived from the resulting expressions. Preserve required public "
    "pathways, recorded time-zero identities, target mappings and data access. "
    "A rejected patch changes nothing; include every intended correction again. "
    "For a change outside this contract, return decision=requires_wider_revision "
    "without edits and explain it. That outcome does not accept the model or "
    "clear its conflicts."
)


def _strength(witness: dict) -> float:
    """Rank same-property/target witnesses, never compare unrelated target units."""
    for key in (
        "contradiction_margin_lower_bound",
        "maximum_initial_absolute_error_lower_bound",
        "observation_gap",
        "weighted_observation_defect",
        "determinant",
    ):
        value = witness.get(key)
        if isinstance(value, (int, float)) and isfinite(value):
            return abs(value)
    return 0.0


def _selected(conflicts: list[dict]) -> list[dict]:
    """Keep the strongest available witness for every property and affected target."""
    selected: dict[tuple[str, str], dict] = {}
    for original in conflicts:
        row = deepcopy(original)
        witnesses = row.get("witnesses", [])
        if witnesses:
            row["witnesses"] = [max(witnesses, key=_strength)]
        key = (row.get("property", ""), row.get("target", ""))
        strength = _strength(row.get("witness", {}))
        if row.get("witnesses"):
            strength = _strength(row["witnesses"][0])
        current = selected.get(key)
        old = _strength(current.get("witness", {})) if current is not None else -1.0
        if current and current.get("witnesses"):
            old = _strength(current["witnesses"][0])
        if current is None or strength > old:
            selected[key] = row
    return [selected[key] for key in sorted(selected)]


def _identity_guidance(property_name: str) -> str:
    """State the failed necessary condition without prescribing scientific laws."""
    if property_name == "affine_initial_superposition":
        return (
            "The displayed shared affine subsystem preserves this trajectory "
            "combination. Changing only its shared linear coefficients or adding "
            "a common basal term leaves the identity intact under the stated "
            "conditioning. Revise the relevant dependence/function family or "
            "eligible initialization rule and explain which premise changes."
        )
    if "rank_one" in property_name:
        return (
            "The closed linear subsystem has only one independently varying "
            "initial direction, so all trajectory differences have one time shape. "
            "Changing that root's values alone is insufficient. A justified new "
            "latent state/initializer, coupling or function family is permitted; "
            "its effect must reach this target. Explain which premise changes."
        )
    if property_name == "closed_subsystem_equality":
        return (
            "The complete relevant initial information and forcing histories "
            "match, so this deterministic closed subsystem gives identical target "
            "trajectories. A shared basal source may change both trajectories but "
            "does not distinguish them. Revise a justified dependence or eligible "
            "initializer; state which equality premise no longer holds."
        )
    return (
        "Revise a scientifically justified dependence, function or eligible "
        "initializer so the displayed necessary condition no longer conflicts "
        "with this target's observations. A new explanation alone is not an edit."
    )


def _refine_affine_witnesses(view: dict, parent: dict) -> None:
    """Strengthen presentation of the same certified identity, never acceptance.

    The existing report already qualified the subsystem, initializer assumptions,
    trajectory triple and affine identity. Keep those unchanged and inspect only
    additional common times in authenticated training data. This is not an
    independent revalidation of the saved theorem or an NMSE lower bound.
    """
    from autoformalism.research.construction_affine_family_consistency import (
        MAX_SAMPLES,
        _bounded,
        _common_rows,
        _display,
        _guard,
        _Unsupported,
    )
    from autoformalism.search.topology_compatibility import (
        TrainingCompatibilityEvidence,
        validate_evidence,
    )

    raw = parent.get("_consistency_evidence")
    if raw is None:
        return
    training = TrainingCompatibilityEvidence.model_validate(raw)
    validate_evidence(training)
    if training.sha256 != view["evidence_sha256"]:
        raise ValueError("witness refinement evidence identity differs")
    trajectories = {row.trajectory_id: row for row in training.trajectories}
    for check in view["conflicts"]:
        if check.get("status") != "conditional_conflict" or not check.get("witnesses"):
            continue
        original = deepcopy(check["witnesses"][0])
        try:
            rows = [trajectories[name] for name in original["trajectories"]]
            if len(rows) != 3 or any(len(row.time) > MAX_SAMPLES for row in rows):
                raise _Unsupported("witness sample/trajectory roster exceeds budget")
            if training.contract.absolute_error_bounds.get(check["target"]) != 0:
                raise _Unsupported("saved affine witness requires exact target values")
            if not _common_rows(rows, set(check["complete_common_forcing"]), training):
                raise _Unsupported("saved affine forcing premise does not match")
            common = sorted(
                set(rows[0].time).intersection(*(row.time for row in rows[1:]))
            )
            if (
                check["property"] == "linear_initial_difference_rank_one"
                and len(common) > 128
            ):
                # Bounded pair search; preserve original witness times and sample
                # the remaining full interval evenly. Superposition is linear in
                # the sample count and scans every common recorded time instead.
                common = sorted(
                    set(original["times"])
                    | {common[(len(common) - 1) * i // 127] for i in range(128)}
                )
            indices = [{time: i for i, time in enumerate(row.time)} for row in rows]
            values = {
                time: [
                    Fraction(row.targets[check["target"]][index[time]])
                    for row, index in zip(rows, indices, strict=True)
                ]
                for time in common
            }
            best, best_strength, comparisons = original, (Fraction(0), Fraction(0)), 0
            if check["property"] == "affine_initial_superposition":
                weights = [Fraction(value) for value in original["weights_exact"]]
                if len(weights) != 3 or sum(weights) != 0:
                    raise _Unsupported("saved superposition weight identity differs")
                for time, observed in values.items():
                    defect = _bounded(
                        sum(
                            (a * b for a, b in zip(weights, observed, strict=True)),
                            Fraction(0),
                        )
                    )
                    guard = _bounded(_guard(observed) * sum(map(abs, weights)))
                    margin = _bounded(abs(defect) - guard)
                    comparisons += 1
                    strength = (margin / guard, margin) if guard else (margin, margin)
                    if margin > 0 and strength > best_strength:
                        best_strength = strength
                        best = original | {
                            "time": time,
                            "sample_indices": [index[time] for index in indices],
                            "observations": [_display(value) for value in observed],
                            "weighted_observation_defect_exact": str(defect),
                            "weighted_observation_defect": _display(defect),
                            "target_numerical_guard": _display(guard),
                        }
            elif check["property"] == "linear_initial_difference_rank_one":
                if len(original["possibly_different_initial_states"]) != 1:
                    raise _Unsupported("saved initial-direction identity differs")
                for first, second in combinations(common, 2):
                    observations = [values[first], values[second]]
                    matrix = [
                        [line[1] - line[0], line[2] - line[0]] for line in observations
                    ]
                    a, b = matrix[0]
                    c, d = matrix[1]
                    defect = _bounded(a * d - b * c)
                    guard = _guard([value for line in observations for value in line])
                    determinant_guard = _bounded(
                        2 * guard * sum(map(abs, (a, b, c, d))) + 8 * guard * guard
                    )
                    margin = _bounded(abs(defect) - determinant_guard)
                    comparisons += 1
                    strength = (
                        (margin / determinant_guard, margin)
                        if determinant_guard
                        else (margin, margin)
                    )
                    if margin > 0 and strength > best_strength:
                        best_strength = strength
                        best = original | {
                            "times": [first, second],
                            "observations": [
                                [_display(value) for value in line]
                                for line in observations
                            ],
                            "difference_matrix_exact": [
                                [str(value) for value in line] for line in matrix
                            ],
                            "difference_matrix": [
                                [_display(value) for value in line] for line in matrix
                            ],
                            "determinant_exact": str(defect),
                            "determinant": _display(defect),
                            "determinant_numerical_guard": _display(determinant_guard),
                        }
            else:
                continue
            check["original_witness"] = original
            check["witnesses"] = [best]
            check["witness_refinement"] = {
                "qualification_same_family": True,
                "selection": (
                    "maximum positive margin divided by the unchanged numerical "
                    "guard; absolute margin breaks ties"
                ),
                "common_times_examined": len(common),
                "comparisons": comparisons,
                "trajectory_triple_unchanged": True,
                "initial_identity_unchanged": True,
                "acceptance_changed": False,
                "limitation": (
                    "Uses the existing qualified certificate and its assumptions; "
                    "not a new model-family proof or approximate-NMSE bound."
                ),
            }
        except (_Unsupported, KeyError, ValueError) as error:
            check["witness_refinement"] = {
                "qualification_same_family": False,
                "original_witness_retained": True,
                "acceptance_changed": False,
                "unavailable": str(error),
            }


def consistency_view(report: dict, parent: dict | None = None) -> dict:
    """Compact authenticated results without changing statuses or proof thresholds."""
    if report.get("feedback_protocol") == PROTOCOL:
        # Rejected-trial diagnostics may already contain this presentation. Never
        # reinterpret its compact `witness` as the raw checker `witnesses` list.
        return deepcopy(report)
    result = {
        key: deepcopy(report[key])
        for key in (
            "status",
            "child_problem_sha256",
            "evidence_sha256",
            "assumptions",
            "target_comparison_policy",
            "equality_status",
        )
        if key in report
    }
    result["feedback_protocol"] = PROTOCOL
    result["conflicts"] = _selected(report.get("conflicts", []))
    for conflict in result["conflicts"]:
        conflict["repair_feedback"] = _identity_guidance(conflict.get("property", ""))
    result["other_checks"] = [
        {
            key: deepcopy(row[key])
            for key in ("property", "target", "status", "unavailable")
            if key in row
        }
        for row in report.get("checks", [])
        if row["status"] != "conditional_conflict"
    ]
    if "function_consistency" in report:
        from autoformalism.research.construction_directional_feedback import (
            function_view,
        )

        source = deepcopy(report["function_consistency"])
        source["conflicts"] = _selected(source["conflicts"])
        view = function_view(source)
        view["full_conflict_count"] = len(report["function_consistency"]["conflicts"])
        for conflict in view["conflicts"]:
            conflict["repair_feedback"] = (
                "The bound belongs to the displayed model, not to the data. "
                "Change the relevant equations, dependencies, new-state structure "
                "or eligible initializer so its bound admits the witnessed change. "
                "Changing other targets alone cannot clear this target's conflict."
            )
            zero = sorted(
                name
                for name, bounds in conflict["complete_forcing_ranges"].items()
                if bounds == [0, 0]
            )
            if zero:
                conflict["zero_forcing"] = {
                    "channels": zero,
                    "scope": "authenticated complete forcing interval of this witness",
                    "explanation": (
                        "These supplied channels are identically zero on the proved "
                        "interval. Multiplying them by another finite factor adds "
                        "no direct forcing there. Their earlier effects may persist "
                        "in memory states; this does not assert those states are zero."
                    ),
                }
        result["function_consistency"] = view
    if "initial_family_consistency" in report:
        from autoformalism.research.construction_initial_family_consistency import (
            prompt_view,
        )

        source = deepcopy(report["initial_family_consistency"])
        source["conflicts"] = _selected(source["conflicts"])
        view = prompt_view(source)
        for conflict in view["conflicts"]:
            conflict["repair_direction"] = (
                "No choice of these shared initializer/readout coefficients can "
                "reproduce the initial observations within the declared errors. "
                "Revise an eligible latent initializer or an editable algebraic "
                "definition feeding the fixed target mapping. Changing only the "
                "ODE after time zero cannot resolve this certificate. Recorded "
                "public initials and target mappings remain fixed."
            )
        result["initial_family_consistency"] = view
    if "affine_family_consistency" in report:
        from autoformalism.research.construction_affine_family_consistency import (
            prompt_view,
        )

        view = deepcopy(prompt_view(report["affine_family_consistency"]))
        view["conflicts"] = _selected(view["conflicts"])
        if parent is not None:
            _refine_affine_witnesses(view, parent)
        for conflict in view["conflicts"]:
            conflict["repair_feedback"] = _identity_guidance(conflict["property"])
        result["affine_family_consistency"] = view
    result["repair_scope"] = _REPAIR_SCOPE
    result["interpretation"] = (
        "Qualified conflicts block export. Deferred checks do not certify repair "
        "or scientific correctness. Witnesses are selected separately for every "
        "property and target. Any additional times use the same saved qualified "
        "affine identity and unchanged guard, with the original witness retained. "
        "This presentation does not strengthen any acceptance threshold."
    )
    if parent is not None:
        result["editable_components"] = _editable(parent)
    return result


def _editable(parent: dict) -> dict:
    """Use the implementation's initializer eligibility, not inferred semantics."""
    from autoformalism.research.construction_structural_revision import (
        editable_initial_states,
    )
    from autoformalism.schemas.prefix_initialization import PrefixFitProblem

    problem = PrefixFitProblem.model_validate(parent.get("problem", parent))
    return {
        "dynamic": [row.state for row in problem.base_candidate.state_equations],
        "algebraic": [row.name for row in problem.base_candidate.processes],
        "initialization_edits": editable_initial_states(
            problem, parent.get("scientific_aliases")
        ),
    }


def _duplicates(reply: Any) -> list[dict]:
    """Describe duplicate operations without guessing which declaration to adopt."""
    if not isinstance(reply, dict):
        return []
    found = []
    for field, key in (
        ("edits", "component"),
        ("new_states", "name"),
        ("new_processes", "name"),
        ("new_parameters", "name"),
        ("initialization_edits", "state"),
    ):
        values = reply.get(field, [])
        if not isinstance(values, list):
            continue
        counts = Counter(
            item[key]
            for item in values[:256]
            if isinstance(item, dict) and isinstance(item.get(key), str)
        )
        found += [
            {"field": field, "name": name, "declarations": count}
            for name, count in sorted(counts.items())
            if count > 1
        ]
    for added_field, repeated_field, key in (
        ("new_states", "edits", "component"),
        ("new_processes", "edits", "component"),
        ("new_states", "initialization_edits", "state"),
    ):
        added = reply.get(added_field, [])
        repeated = reply.get(repeated_field, [])
        if not isinstance(added, list) or not isinstance(repeated, list):
            continue
        names = {
            item["name"]
            for item in added[:256]
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        for item in repeated[:256]:
            if (
                isinstance(item, dict)
                and isinstance(item.get(key), str)
                and item[key] in names
            ):
                found.append(
                    {
                        "field": repeated_field,
                        "name": item[key],
                        "also_declared_in": added_field,
                    }
                )
    return found


def feedback_guidance(
    code: str, details: dict, parent: dict, reply: Any = None
) -> dict:
    """Attach actionable interface advice without adopting an invalid proposal."""
    result = {"code": code, "details": deepcopy(details), "repair_scope": _REPAIR_SCOPE}
    result["editable_components"] = _editable(parent)
    duplicates = _duplicates(reply)
    if duplicates:
        result["duplicate_declarations"] = duplicates
        result["duplicate_feedback"] = (
            "Provide exactly one complete declaration per component in its action "
            "list. Identical redundant declarations normalize within the original "
            "list limits; different RHSs, kinds or initial rules remain conflicts. "
            "A new state includes its RHS and initial rule in new_states; a new "
            "process includes its expression in new_processes. Do not repeat those "
            "in edits or initialization_edits. Consolidate your intended version; "
            "do not delete unrelated required pathways."
        )
    if code == "DECLARATION_CONFLICT":
        result["specific_feedback"] = (
            "The listed declarations describe different actions. Choose one "
            "complete RHS, kind and initializer for each name; the runtime cannot "
            "infer which version you intended. Different scopes, parameter roles "
            "and alias bindings are not formatting differences. Put new states "
            "and their initial rules together in new_states, new algebraic laws "
            "in new_processes, and existing definitions in edits."
        )
    elif code in {"COMPONENT_IDENTITY", "DECLARATION_IDENTITY", "PARAMETER_IDENTITY"}:
        result["specific_feedback"] = (
            "Use edits for existing dynamic/algebraic definitions, new_states or "
            "new_processes for new names, and initialization_edits only for "
            "displayed existing latent initial rules. A new state's initial rule "
            "belongs in new_states[].initial. initialization_graph and decision "
            "are control fields, not equation names. A parameter belongs in "
            "new_parameters and an expression, never in edits as a component. "
            "Initializer-map coefficients belong in that map's parameters list, "
            "not new_parameters unless independently used in dynamics. New names "
            "must not shadow states, processes, parameters or public channels; "
            "init_ and af_internal_ names are reserved. A dynamic self-dependence "
            "is allowed. Changing dZ/dt to u integrates u; it does not make Z "
            "algebraic. Changing an existing component's kind requires a wider "
            "revision; use that explicit decision if necessary."
        )
    elif code in {
        "INITIALIZATION_IDENTITY",
        "INITIALIZATION_SCOPE",
        "FROZEN_CONSTRAINT",
        "REMOVAL_SCOPE",
    }:
        result["specific_feedback"] = (
            "Use the displayed eligible initializer targets and the typed initial "
            "choice in the schema. initialization_edits replaces an existing "
            "eligible latent rule; new_states[].initial supplies a new state's "
            "complete rule. Do not list a new state in initialization_edits. "
            "For maps, every expression alias must be declared in state_initials "
            "or public_initials, with coefficients in the map's parameters list. "
            "Public recorded time-zero values, known initials "
            "and protected mappings are not editable. A trajectory root is "
            "permitted only when the experiment authorizes its observation-window "
            "policy; the proposer cannot choose that window or access later data."
        )
    elif code == "UNKNOWN_EVIDENCE":
        result["specific_feedback"] = (
            "Use the displayed allowed_evidence_ids exactly. A witness property "
            "label is not a separate evidence ID unless explicitly listed or "
            "resolved by the authenticated request. Do not invent an ID or cite "
            "held-out data. This citation correction does not repair the model."
        )
    elif code == "UNUSED_NEW_PARAMETER":
        result["specific_feedback"] = (
            "new_parameters declares shared dynamics coefficients, which must "
            "occur in a resulting model expression. A coefficient used only in "
            "an initializer belongs in that map's parameters list. Do not add "
            "an artificial model term merely to make a declaration used."
        )
    elif code == "REVISION_VALIDATION":
        result["specific_feedback"] = (
            "Correct the expression or initialization graph named in the "
            "diagnostic. In an initializer map, every initial-value symbol must "
            "have an explicit state_initials or public_initials alias; naming a "
            "model state in the expression alone does not bind its initial value. "
            "Declare map coefficients locally in parameters, remove unused aliases, "
            "and keep state-initial dependencies acyclic."
        )
    elif code in {
        "REPEATED_EDIT",
        "REPEATED_PATCH",
        "UNCHANGED_EDIT",
        "REPEATED_EQUATION_EDIT",
    }:
        result["specific_feedback"] = (
            "This scientific patch repeats a previous attempt; changing the "
            "explanation or sources metadata does not change the model. Address "
            "the displayed target-specific witness with a different coherent "
            "patch and say which premise changes. A repeated patch is not a "
            "successful repair. If the intended correction is outside this "
            "contract, use requires_wider_revision without edits."
        )
    elif code == "CONFLICTING_OPERATIONS":
        result["specific_feedback"] = (
            "Do not both edit and remove the same component. Return one coherent "
            "set of operations, including all references and initializers affected "
            "by a removal. No partial changes have been applied."
        )
    else:
        result["specific_feedback"] = (
            "Correct the listed locations in one complete patch. Whole RHS edits "
            "replace that definition, so retain all intended unaffected terms. "
            "Check each affected target's witness, not only the first one."
        )
    if isinstance(details.get("assessment"), dict):
        result["details"]["assessment"] = consistency_view(
            details["assessment"], parent
        )
    return result


def _expression(text: str, name: str, kind: str) -> str:
    source = _canonical_rhs(text, name, kind, "t")
    _bounded_ast(source, "structural repeat bookkeeping")
    return ast.dump(ast.parse(source, mode="eval"), include_attributes=False)


def _initial(value: dict, state: str) -> dict:
    result = deepcopy(value)
    if result.get("mode") == "map":
        result["expression"] = _expression(result["expression"], state, "algebraic")
        for field in ("parameters", "state_initials", "public_initials"):
            result[field] = sorted(result.get(field, []), key=content_hash)
    return result


def patch_signature(reply: Any) -> str | None:
    """Hash bounded scientific actions, ignoring prose, order and sources metadata.

    This is syntactic bookkeeping only, not algebraic equivalence or validation.
    Initializer scope, mappings and roles remain part of the signature.
    """
    if not isinstance(reply, dict):
        return None
    try:
        from autoformalism.research.construction_structural_revision import (
            StructuralRevisionReply,
            normalize_structural_reply,
        )

        # Validate first so unknown fields or invalid deletion/initializer shapes
        # cannot be mistaken for a previously attempted legal scientific patch.
        for field, limit in _ACTIONS.items():
            value = reply.get(field, [])
            if not isinstance(value, list) or len(value) > limit:
                return None
        normalized_reply, _ = normalize_structural_reply(reply)
        value = StructuralRevisionReply.model_validate(normalized_reply).model_dump(
            mode="json"
        )
        normalized = {field: value[field] for field in _ACTIONS}
        for field, name, expression, kind in (
            ("edits", "component", "expression", None),
            ("new_states", "name", "rhs", "dynamic"),
            ("new_processes", "name", "expression", "algebraic"),
        ):
            for row in normalized[field]:
                row.pop("sources", None)
                row.pop("description", None)
                row[expression] = _expression(
                    row[expression], row[name], kind or row["kind"]
                )
                if "initial" in row:
                    row["initial"] = _initial(row["initial"], row[name])
        for row in normalized["initialization_edits"]:
            row["initial"] = _initial(row["initial"], row["state"])
        return content_hash(
            {
                "decision": value["decision"],
                **{
                    field: sorted(rows, key=content_hash)
                    for field, rows in normalized.items()
                },
            }
        )
    except (ValueError, TypeError, KeyError, RecursionError, ModelValidationError):
        return None
