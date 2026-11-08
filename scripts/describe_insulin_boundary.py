"""Generate proposer feedback from a saved T2 insulin-boundary run, offline."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

from autoformalism.expressions.diagnostics import (
    ModelValidationError,
    RuntimeExpressionError,
)
from autoformalism.rebuttal import basin_algebra as algebra
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import insulin_boundary_revision as boundary
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from scripts import propose_meal_insulin_revision as revision


def _meal_coefficient(problem: PrefixFitProblem) -> float | None:
    """Return a state-independent direct coefficient at the supplied parameters."""
    try:
        terms = algebra.formal(boundary._insulin_tree(problem))
        coefficient = algebra.coefficient(terms, boundary.MEAL)
        shared = {
            p.name for p in problem.base_candidate.parameters if p.scope == "global"
        }
        if coefficient is None or not algebra.symbols(coefficient) <= (
            shared & problem.parameter_guesses.keys()
        ):
            return None
        value = algebra.number(coefficient, problem.parameter_guesses)
    except (
        ModelValidationError,
        RuntimeExpressionError,
        ValueError,
        ArithmeticError,
        RecursionError,
    ):
        return None
    return value if math.isfinite(value) else None


def _clock_feedback(problem: PrefixFitProblem, added_states: set[str]) -> list[str]:
    """Check added state rates after inlining their algebraic processes."""
    model = problem.base_candidate
    processes = {p.name: p.expression for p in model.processes}
    time = problem.context.time_symbol
    messages = []
    for eq in model.state_equations:
        if eq.state not in added_states:
            continue
        try:
            rate = algebra.expanded(eq.rhs, processes)
        except (ValueError, ModelValidationError, RecursionError):
            continue
        functions = {n.func.id for n in ast.walk(rate) if isinstance(n, ast.Call)}
        names = {n.id for n in ast.walk(rate) if isinstance(n, ast.Name)} - functions
        shared = {p.name for p in model.parameters if p.scope == "global"}
        if names - shared == {time}:
            messages.append(
                f"d({eq.state})/dt uses only absolute time and shared parameters. "
                "Where defined, this rate is identical at a given time on shifted-meal "
                "and no-meal schedules. Different initial values can offset the state. "
                "The change in this state equals the integral of its rate. "
                "If this component should respond to meal timing or dose, identify "
                "which supplied input or downstream state drives its rate."
            )
        denominators = {
            n.right.id
            for n in ast.walk(rate)
            if isinstance(n, ast.BinOp)
            and isinstance(n.op, ast.Div)
            and isinstance(n.right, ast.Name)
        }
        for p in model.parameters:
            if (
                p.name in denominators
                and p.domain != "positive"
                and (p.bounds is None or p.bounds.lower <= 0 <= p.bounds.upper)
            ):
                messages.append(
                    f"The RHS for {eq.state} divides by {p.name}, whose declared "
                    "role and bounds allow zero. Require a nonzero divisor "
                    "or revise the RHS."
                )
    return messages


def _missing_state_term(
    child: PrefixFitProblem, witness: dict, symbol: str
) -> str | None:
    """Isolate one added state if it appears as a single direct additive term."""
    tree = boundary._insulin_tree(child)
    pending = [(1, tree)]
    signs = []
    while pending:
        sign, node = pending.pop()
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            pending.append((sign, node.left))
            pending.append(
                (sign if isinstance(node.op, ast.Add) else -sign, node.right)
            )
        elif isinstance(node, ast.Name) and node.id == symbol:
            signs.append(sign)
        elif any(isinstance(n, ast.Name) and n.id == symbol for n in ast.walk(node)):
            return None
    if len(signs) != 1:
        return None
    zero = {**witness, "state_at_boundary": {**witness["state_at_boundary"], symbol: 0}}
    value = boundary._at_witness(tree, child, zero)
    if value["status"] != "evaluated":
        return None
    offset = value["rate_pmol_L_min"]
    term = symbol if signs[0] == 1 else "-" + symbol
    if offset:
        term += f" {offset:+.8g}"
    return f"At the saved point and parameter values, dI/dt = {term}."


def feedback(inputs: dict, reply: dict | None = None) -> dict:
    """Describe the fitted witness and optionally assess one explicit reply."""
    if (
        inputs.get("revision_protocol") != boundary.PROTOCOL
        or inputs["evidence"]["split"] != "train"
    ):
        raise ValueError("expected the public training insulin-boundary protocol")
    evidence = json.loads(inputs["user"])["training_evidence"]
    witness = inputs["boundary_witness"]
    windows = evidence["ingestion_windows"]
    training_ids = {row["trajectory_id"] for row in inputs["evidence"]["trajectories"]}
    if (
        evidence["depletion"] != witness
        or not {witness["trajectory"], *(row["trajectory"] for row in windows)}
        <= training_ids
        or witness["state_at_boundary"]["I"] != 0
        or witness["inputs"] != {boundary.MEAL: 0, boundary.INFUSION: 0}
    ):
        raise ValueError("inconsistent training or depletion witness")
    parent = PrefixFitProblem.model_validate(inputs["problem"])
    point = boundary._at_witness(boundary._insulin_tree(parent), parent, witness)
    rate = point.get("rate_pmol_L_min")
    if (
        rate is None
        or rate >= 0
        or not math.isclose(rate, witness["reference_rate_pmol_L_min"], abs_tol=1e-10)
    ):
        raise ValueError("fitted equation does not reproduce the outward boundary rate")
    fitted = [
        f"At minute {witness['time_min']:g} on {witness['trajectory']}, set I=0 and "
        f"both input rates to zero at the saved state. The fitted balance gives "
        f"dI/dt={rate:.6f} pmol/L/min. Revise the complete insulin balance so that "
        "it points inward or is tangent at depletion."
    ]
    glucose = witness.get("glucose_source_pmol_L_min")
    action = witness.get("delayed_action_term_pmol_L_min")
    if (
        glucose is not None
        and action is not None
        and math.isclose(glucose + action, rate, abs_tol=1e-10)
    ):
        fitted.append(
            f"The saved glucose and delayed-action terms contribute {glucose:+.6f} "
            f"and {action:+.6f} pmol/L/min at that point."
        )
    for row in windows:
        meal = row["meal"]
        fitted.append(
            f"{row['trajectory']}, {meal['dose_g']:g} g, minutes {meal['start_min']:g}"
            f" to {meal['end_min']:g}. Insulin rose "
            f"{row['observed_rise_pmol_L']:+.2f} pmol/L "
            f"in training and {row['fitted_rise_pmol_L']:+.2f} in the fitted rollout. "
            f"The fitted direct meal term alone contributed "
            f"{row['direct_ingestion_contribution_pmol_L']:+.2f} pmol/L. "
            f"The observed and fitted peaks were at minutes "
            f"{row['observed_peak_min']:g} and {row['fitted_peak_min']:g}."
        )
    report = {"fitted_feedback": fitted, "depletion_witness": witness}
    if reply is None:
        return report
    try:
        assessment = revision.assess(inputs, reply)
    except revision.RevisionValidationError as exc:
        return {**report, "reply_status": "rejected", "diagnostic": exc.as_dict()}
    except (ValueError, ModelValidationError) as exc:
        return {
            **report,
            "reply_status": "rejected",
            "diagnostic": {"message": str(exc)},
        }
    child = PrefixFitProblem.model_validate(assessment["problem"])
    result = assessment["insulin_boundary_check"]
    messages = _clock_feedback(child, set(assessment["audit"]["new_states"]))
    evaluated = result["candidate_at_inherited_values_and_new_parameter_guesses"]
    if missing := evaluated.get("unavailable_symbols"):
        if len(missing) == 1 and missing[0] in assessment["audit"]["new_states"]:
            relation = _missing_state_term(child, witness, missing[0])
            if relation:
                messages.append(relation)
        messages.append(
            "The saved boundary state has no value for "
            + ", ".join(missing)
            + ". Supply a reachable value or establish its sign from the initializer "
            "and dynamics to determine the complete boundary rate."
        )
    coefficient = _meal_coefficient(child)
    if coefficient is not None and coefficient != 0 and windows:
        dose = windows[0]["meal"]["dose_g"]
        messages.append(
            f"At the inherited coefficients and new parameter guesses, the direct "
            f"meal term in dI/dt contributes {coefficient * dose:+.2f} pmol/L over "
            f"{dose:g} g. Check the net change after including the other terms."
        )
    return {
        **report,
        "reply_status": assessment["status"],
        "boundary_check": result,
        "revision_feedback": messages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="directory with sealed plan and inputs"
    )
    parser.add_argument(
        "--reply", type=Path, help="optional visible proposal JSON to assess offline"
    )
    args = parser.parse_args()
    plan = sealed_read(args.root / "plan.json")
    inputs = sealed_read(args.root / "inputs.json")
    if (
        plan["inputs_sha256"] != inputs["artifact_sha256"]
        or plan["protocol"] != boundary.PROTOCOL
    ):
        raise ValueError("plan does not identify these insulin-boundary inputs")
    reply = json.loads(args.reply.read_text()) if args.reply else None
    report = {
        "source_plan_sha256": plan["artifact_sha256"],
        "reply_content_sha256": revision.content_sha256(reply)
        if reply is not None
        else None,
        **feedback(inputs, reply),
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
