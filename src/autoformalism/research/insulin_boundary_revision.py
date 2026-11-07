"""Candidate-15 revision feedback and a bounded concentration-boundary check."""

from __future__ import annotations

import ast
import math

from autoformalism.expressions.compiler import _evaluate
from autoformalism.expressions.diagnostics import RuntimeExpressionError
from autoformalism.expressions.parser import ParsedExpression
from autoformalism.rebuttal import basin_algebra as algebra
from autoformalism.research.construction_revision_edits import RevisionValidationError
from autoformalism.schemas.prefix_initialization import PrefixFitProblem

PROTOCOL = "candidate-15-insulin-boundary-revision-1"
INTAKE_PROTOCOL = "candidate-15-reviewed-boundary-intake-1"
INTAKE_SHA256 = "4daf9b76e848cc9a8d42b3fdcc4f30b1a823c0f0a8eabfb72e802d7c54012afa"
MEAL = "meal_rate_g_per_min"
INFUSION = "insulin_pmol_per_kg_min"

INSTRUCTION = """Revise the fitted insulin balance so that insulin can stay
nonnegative during depletion. I is an absolute concentration. At I=0 the
assembled dI/dt must point inward or be zero under admissible conditions.
Use the training_evidence below. At the reported boundary the fitted action
term keeps removing insulin even when I has reached zero. Write the complete
boundary rate implied by your change and state the signs/domains needed for it.
Express the repair in the dynamics. Clipping predictions or changing the loss
leaves the balance unchanged. The one-minute extension is a model diagnostic.

Also consider the ten-minute meal: how can the model produce the measured
small early rise and the later peak? Account for the direct ingestion term in
the assembled I equation. You may replace I's entire RHS, or insulin_production
if changing that process suffices. Choose the sources, timing mechanism and
function form. Preserve the required meal-to-glucose and delayed
insulin-to-disposal paths. The negative feedback into insulin production was
a fitted model choice. You may revise that connection. Explain your predicted
ingestion-window change and peak timing and height.

You may add one connected differential state with a causal initializer, one
connected algebraic process, and up to eight shared dynamics parameters.
Use the displayed public channels and generated states. State domain assumptions
explicitly, including assumptions on any new state. Review each source and
sink at I=0 after substituting your proposed processes."""


def _insulin_tree(problem: PrefixFitProblem) -> ast.AST:
    candidate = problem.base_candidate
    rhs = next(row.rhs for row in candidate.state_equations if row.state == "I")
    return algebra.expanded(rhs, {p.name: p.expression for p in candidate.processes})


def _polynomial(tree: ast.AST) -> algebra.Polynomial | None:
    """Use only total polynomial operations for exact identity/sign conclusions."""
    if any(isinstance(node, (ast.Call, ast.Div, ast.Pow)) for node in ast.walk(tree)):
        return None
    try:
        return algebra.formal(tree)
    except ValueError:
        return None


def _at_zero(poly: algebra.Polynomial, names: set[str]) -> algebra.Polynomial:
    return {p: c for p, c in poly.items() if not any(n in names for n, _ in p)}


def _at_witness(tree: ast.AST, problem: PrefixFitProblem, witness: dict) -> dict:
    """Evaluate only referenced names, at the saved states and explicit guesses."""
    values = {
        **witness["state_at_boundary"],
        **witness["inputs"],
        problem.context.time_symbol: witness["time_min"],
        **problem.parameter_guesses,
    }
    functions = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - functions
    missing = sorted(names - values.keys())
    if missing:
        return {"status": "deferred", "unavailable_symbols": missing}
    parsed = ParsedExpression(
        "expanded insulin balance", ast.Expression(tree), frozenset(names)
    )
    try:
        rate = _evaluate(parsed, values)
    except RuntimeExpressionError:
        return {"status": "deferred", "reason": "undefined at the supplied point"}
    return {"status": "evaluated", "rate_pmol_L_min": rate}


def check(parent: PrefixFitProblem, child: PrefixFitProblem, witness: dict) -> dict:
    """Keep exact structural failures distinct from unfitted point diagnostics."""
    if witness["state_at_boundary"]["I"] != 0 or witness["inputs"] != {
        MEAL: 0,
        INFUSION: 0,
    }:
        raise ValueError("expected a zero-insulin, zero-input witness")
    before = _insulin_tree(parent)
    old_rate = _at_witness(before, parent, witness)["rate_pmol_L_min"]
    if old_rate >= 0 or not math.isclose(
        old_rate, witness["reference_rate_pmol_L_min"], rel_tol=1e-9, abs_tol=1e-10
    ):
        raise ValueError("reference insulin equation does not reproduce the witness")
    report = {
        "reference_rate_pmol_L_min": old_rate,
        "trajectory": witness["trajectory"],
        "time_min": witness["time_min"],
        "state_at_boundary": witness["state_at_boundary"],
        "inputs": witness["inputs"],
        "candidate_at_inherited_values_and_new_parameter_guesses": {
            "status": "deferred"
        },
        "boundary_sign": "deferred",
        "sign_assumptions": "I=0, Gp>=0, nonnegative meal/insulin input rates, "
        "declared parameter domains. Other states and forcing may have either sign.",
        "continuous_trajectory_positivity_verified": False,
        "refit_required": True,
    }
    try:
        after = _insulin_tree(child)
    except (ValueError, RecursionError):
        return report | {"unavailable": "bounded process expansion exceeded"}
    report["candidate_at_inherited_values_and_new_parameter_guesses"] = _at_witness(
        after, child, witness
    )
    new_poly = _polynomial(after)
    if new_poly is None:
        return report
    boundary = _at_zero(new_poly, {"I"})
    depleted = _at_zero(boundary, {MEAL, INFUSION})
    old_poly = _polynomial(before)
    if old_poly is not None and depleted == _at_zero(old_poly, {"I", MEAL, INFUSION}):
        report["boundary_sign"] = "outward_witness_retained"
        raise RevisionValidationError(
            "UNCHANGED_OUTWARD_INSULIN_BOUNDARY",
            "The complete zero-input rate at I=0 is algebraically unchanged. "
            "The saved fitted boundary therefore still has the reported outward rate.",
            boundary_check=report,
        )
    positive = {
        p.name for p in child.base_candidate.parameters if p.domain.value == "positive"
    }
    nonnegative = positive | {
        p.name
        for p in child.base_candidate.parameters
        if p.domain.value == "nonnegative"
    }
    state = witness["state_at_boundary"]
    known_positive = positive | {name for name, value in state.items() if value > 0}
    known_nonnegative = nonnegative | {
        name for name, value in state.items() if value >= 0
    }
    at_witness = _at_zero(
        depleted, {name for name, value in state.items() if value == 0}
    )
    if all(
        c < 0 and _unsigned(p, known_nonnegative) for p, c in at_witness.items()
    ) and any(all(name in known_positive for name, _ in p) for p in at_witness):
        report["boundary_sign"] = "strictly_outward_at_displayed_state"
        raise RevisionValidationError(
            "NEGATIVE_INSULIN_BOUNDARY",
            "The rate at the displayed depleted state is strictly negative "
            "throughout the declared parameter domains.",
            boundary_check=report,
        )
    nonnegative.update({"Gp", MEAL, INFUSION})
    if all(
        coefficient >= 0 and _unsigned(powers, nonnegative)
        for powers, coefficient in boundary.items()
    ):
        report["boundary_sign"] = "nonnegative_under_stated_assumptions"
    return report


def _unsigned(powers: algebra.Monomial, nonnegative: set[str]) -> bool:
    return all(power % 2 == 0 or name in nonnegative for name, power in powers)
