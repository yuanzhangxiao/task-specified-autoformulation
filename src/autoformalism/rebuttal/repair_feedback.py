"""Read-only explanations of initialization and actual repair effects."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING

from autoformalism.expressions import (
    RestrictedParser,
    ValidationContext,
    compile_candidate,
)
from autoformalism.fitting.initialization import (
    LatentInitializationPlan,
    apply_initialization_plan,
)
from autoformalism.schemas import CandidateModel

if TYPE_CHECKING:
    from autoformalism.rebuttal.repair_transactions import RepairAction

PROTOCOL = "repair-feedback-evidence-3"


def initialization_facts(
    candidate: CandidateModel,
    plan: LatentInitializationPlan,
    context: ValidationContext,
    fit: dict,
) -> dict:
    """Distinguish boundary policy, fitted values and collocation-stage failure.

    Lower the same initialization plan as the fitter, without reading any data or
    running optimization. Missing parameter evidence stays unknown, never zero.
    """
    _, guesses, lowering = apply_initialization_plan(
        compile_candidate(candidate, context), plan
    )
    parameters = fit.get("parameters") or {}
    bindings = []
    for state, binding in sorted(lowering["bindings"].items()):
        names = binding["parameters"]
        values = {
            name: parameters[name]
            for name in names
            if isinstance(parameters.get(name), (int, float))
            and not isinstance(parameters[name], bool)
            and math.isfinite(parameters[name])
        }
        bindings.append(
            {
                "state": state,
                "policy_present": True,
                "mode": binding["mode"],
                "training_fitted_policy": binding["mode"] != "known",
                "optimizer_guesses_not_fixed_values": {
                    name: guesses[name] for name in names
                },
                "selected_fit_parameter_values": values,
                "selected_values_available": len(values) == len(names)
                if names
                else None,
                "already_shared_training_value": binding["mode"] == "value",
            }
        )
    init = fit.get("initializer") or {}
    return {
        "schema_version": PROTOCOL,
        "latent_boundaries": bindings,
        "observed_initial_channels": lowering["observed_initial_channels"],
        "collocation_optimizer_initialization": {
            "success": init.get("success"),
            "message": init.get("message"),
            "failure_does_not_imply_missing_latent_initializers": True,
        },
        "selected_fit_status": fit.get("status", "not_run"),
        "validation_initial_parameters_fitted": False,
        "interpretation": (
            "A zero guess is not a fixed zero initial state. Existing mode=value "
            "rules are already shared training-fitted values. "
            "Repeating causal_map=null "
            "does not add this capability or retry a timed-out optimizer. A numerical "
            "stage timeout is not evidence that a boundary policy is absent."
        ),
    }


def _syntax(expression: str | None) -> str | None:
    if expression is None:
        return None
    return ast.dump(
        RestrictedParser().parse(expression, location="repair_effect").tree,
        include_attributes=False,
    )


def _equations(candidate: CandidateModel) -> dict[str, list[str]]:
    return {e.state: ["dynamic", e.rhs] for e in candidate.state_equations} | {
        p.name: ["algebraic", p.expression] for p in candidate.processes
    }


def action_effects(
    parent: CandidateModel,
    revised: CandidateModel,
    plan: LatentInitializationPlan,
    new_plan: LatentInitializationPlan,
    action: RepairAction,
) -> list[dict]:
    """Report actual before/after changes, not a proposer's claimed explanation.

    Equation equality is restricted-AST equality, not general algebraic equivalence.
    The existing whole-candidate identity check still owns commit/no-change.
    """
    effects = []
    before, after = _equations(parent), _equations(revised)
    for edit in action.equations:
        old, new = before.get(edit.component), after.get(edit.component)
        same = (
            old is not None
            and new is not None
            and (old[0], _syntax(old[1])) == (new[0], _syntax(new[1]))
        )
        effects.append(
            {
                "kind": "equation",
                "target": edit.component,
                "status": "unchanged" if same else "changed",
                "reason": "same_kind_and_expression_ast"
                if same
                else "equation_changed",
                "before": old,
                "after": new,
            }
        )
    old_maps = {m.channel: m.expression for m in parent.observation_mappings}
    new_maps = {m.channel: m.expression for m in revised.observation_mappings}
    for edit in action.mappings:
        old, new = old_maps.get(edit.channel), new_maps.get(edit.channel)
        same = _syntax(old) == _syntax(new)
        effects.append(
            {
                "kind": "mapping",
                "target": edit.channel,
                "status": "unchanged" if same else "changed",
                "reason": "same_mapping_ast" if same else "mapping_changed",
                "before": old,
                "after": new,
            }
        )
    for edit in action.initializers:
        old, new = plan.rules.get(edit.state), new_plan.rules.get(edit.state)
        same = old == new
        effects.append(
            {
                "kind": "initializer",
                "target": edit.state,
                "status": "unchanged" if same else "changed",
                "reason": "shared_training_value_already_enabled"
                if same and old is not None and old.initial.mode == "value"
                else "same_initializer_policy"
                if same
                else "initializer_policy_changed",
                "before": old.model_dump(mode="json") if old else None,
                "after": new.model_dump(mode="json") if new else None,
            }
        )
    for name in action.remove:
        effects.append(
            {
                "kind": "remove",
                "target": name,
                "status": "changed",
                "reason": "variable_removed",
                "before": before.get(name),
                "after": None,
            }
        )
    return effects
