"""Qualified shared-initializer family certificates, without parameter fitting.

At time zero, expand the actual initializer graph and output expression. A
jointly affine dependence on shared unknown coordinates gives ``y = X theta+b``.
Exact rational elimination produces left-null vectors of X; a weighted observed
residual exceeding its explicit uncertainty proves this family inconsistent.
Uncertain conditioning, relevant trajectory roots and unsupported expressions
defer. A passed necessary condition is never a scientific-validity certificate.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from autoformalism.expressions import CompiledModel, compile_candidate
from autoformalism.fitting.initialization_graph import apply_initialization_graph
from autoformalism.research.construction_revision_consistency import (
    TARGET_COMPARISON_POLICY,
    _lower_float,
)
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.search.topology_compatibility import (
    CompleteTrajectory,
    TrainingCompatibilityEvidence,
    validate_evidence,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "construction-initial-family-consistency-1"
LIMITS = {
    "rows": 64,
    "coordinates": 64,
    "rational_bits": 4096,
    "operations": 300_000,
    "expansion_depth": 128,
}


class _Unsupported(ValueError):
    """A necessary-condition proof cannot be completed under its finite scope."""


@dataclass
class _Budget:
    remaining: int = LIMITS["operations"]
    depth: int = 0

    def check(self, value: Fraction) -> Fraction:
        """Bound exact arithmetic and work, without rounding a certificate."""
        self.remaining -= 1
        if self.remaining < 0:
            raise _Unsupported("initial-family exact arithmetic operation limit")
        if (
            max(value.numerator.bit_length(), value.denominator.bit_length())
            > LIMITS["rational_bits"]
        ):
            raise _Unsupported("initial-family exact arithmetic bit limit")
        return value


@dataclass
class _Affine:
    constant: Fraction = Fraction(0)
    coefficients: dict[str, Fraction] = field(default_factory=dict)

    def add(self, other: _Affine, budget: _Budget) -> _Affine:
        coefficients = {
            name: budget.check(
                self.coefficients.get(name, Fraction(0))
                + other.coefficients.get(name, Fraction(0))
            )
            for name in sorted(self.coefficients.keys() | other.coefficients.keys())
        }
        return _Affine(
            budget.check(self.constant + other.constant),
            {name: value for name, value in coefficients.items() if value},
        )

    def scale(self, factor: Fraction, budget: _Budget) -> _Affine:
        return _Affine(
            budget.check(self.constant * factor),
            {
                name: budget.check(coefficient * factor)
                for name, coefficient in self.coefficients.items()
                if coefficient * factor
            },
        )


def _interpret(node: ast.AST, resolve, budget: _Budget) -> _Affine:
    """Interpret restricted parsed syntax only; no generated text is executed."""
    budget.check(Fraction(0))
    budget.depth += 1
    try:
        if budget.depth > LIMITS["expansion_depth"]:
            raise _Unsupported("initial-family expression expansion depth limit")
        return _interpret_node(node, resolve, budget)
    finally:
        budget.depth -= 1


def _interpret_node(node: ast.AST, resolve, budget: _Budget) -> _Affine:
    """Apply bounded affine operations to one previously restricted AST node."""
    if isinstance(node, ast.Expression):
        return _interpret(node.body, resolve, budget)
    if isinstance(node, ast.Constant):
        return _Affine(budget.check(Fraction(node.value)))
    if isinstance(node, ast.Name):
        return resolve(node.id)
    if isinstance(node, ast.UnaryOp):
        value = _interpret(node.operand, resolve, budget)
        return (
            value.scale(Fraction(-1), budget)
            if isinstance(node.op, ast.USub)
            else value
        )
    if isinstance(node, ast.BinOp):
        left = _interpret(node.left, resolve, budget)
        right = _interpret(node.right, resolve, budget)
        if isinstance(node.op, ast.Add):
            return left.add(right, budget)
        if isinstance(node.op, ast.Sub):
            return left.add(right.scale(Fraction(-1), budget), budget)
        if isinstance(node.op, ast.Mult):
            if not left.coefficients:
                return right.scale(left.constant, budget)
            if not right.coefficients:
                return left.scale(right.constant, budget)
        if isinstance(node.op, ast.Div) and not right.coefficients and right.constant:
            return left.scale(budget.check(1 / right.constant), budget)
        if isinstance(node.op, ast.Pow) and not right.coefficients:
            if right.constant == 1:
                return left
            if right.constant == 0 and (left.coefficients or left.constant):
                return _Affine(Fraction(1))
    raise _Unsupported(
        "initializer/readout is not jointly affine in shared coordinates"
    )


class _InitialView:
    """Read one initial output using the actual lowered graph, not old metadata."""

    def __init__(
        self,
        model: CompiledModel,
        roots: set[str],
        evidence: TrainingCompatibilityEvidence,
        row: CompleteTrajectory,
        budget: _Budget,
    ) -> None:
        self.model, self.roots, self.evidence, self.row = model, roots, evidence, row
        self.budget = budget
        self.parameters = set(model.parameter_names)
        self.public_channels: set[str] = set()
        self.initial_states: set[str] = set()
        self.cache: dict[str, _Affine] = {}
        self.visiting: set[str] = set()

    def public(self, name: str) -> _Affine:
        """Only explicitly exact conditioning can enter a numerical design row."""
        if name == self.model.validated.context.time_symbol:
            return _Affine(self.budget.check(Fraction(self.row.time[0])))
        if self.evidence.contract.absolute_error_bounds.get(name) != 0:
            raise _Unsupported(f"initial conditioning {name} has nonzero/unknown error")
        self.public_channels.add(name)
        if name in self.row.covariates:
            return _Affine(self.budget.check(Fraction(self.row.covariates[name])))
        for group in (self.row.inputs, self.row.auxiliaries, self.row.targets):
            if name in group:
                return _Affine(self.budget.check(Fraction(group[name][0])))
        raise _Unsupported(f"initial conditioning {name} is unavailable")

    def initial_symbol(self, name: str) -> _Affine:
        """Graph lowering separates initial public readings from generated states."""
        if name in self.roots:
            raise _Unsupported(
                f"relevant trajectory-scoped initializer coordinate: {name}"
            )
        if name in self.parameters:
            return _Affine(coefficients={name: Fraction(1)})
        return self.public(name)

    def symbol(self, name: str) -> _Affine:
        """Expand algebraic processes, then substitute each actual state initial."""
        if name in self.cache:
            return self.cache[name]
        if name in self.visiting:
            raise _Unsupported("cyclic initial readout expansion")
        self.visiting.add(name)
        parsed = self.model.validated
        if name in self.model.state_names:
            self.initial_states.add(name)
            if name in parsed.initial_condition_expressions:
                value = _interpret(
                    parsed.initial_condition_expressions[name].tree,
                    self.initial_symbol,
                    self.budget,
                )
            else:
                initial = next(
                    item
                    for item in parsed.candidate.initial_conditions
                    if item.state == name
                )
                if initial.fixed_value is None:
                    raise _Unsupported(f"missing fully specified initial value: {name}")
                value = _Affine(self.budget.check(Fraction(initial.fixed_value)))
        elif name in parsed.process_expressions:
            value = _interpret(
                parsed.process_expressions[name].tree, self.symbol, self.budget
            )
        else:
            value = self.initial_symbol(name)
        self.visiting.remove(name)
        self.cache[name] = value
        return value


def _null_relations(
    matrix: list[list[Fraction]], budget: _Budget
) -> list[list[Fraction]]:
    """Eliminate design rows while retaining exact left-null certificates."""
    basis: dict[int, tuple[list[Fraction], list[Fraction]]] = {}
    nulls = []
    count = len(matrix)
    for index, source in enumerate(matrix):
        row = list(source)
        weights = [Fraction(int(i == index)) for i in range(count)]
        for pivot, (basis_row, basis_weights) in sorted(basis.items()):
            factor = row[pivot]
            if factor:
                row = [
                    budget.check(a - factor * b)
                    for a, b in zip(row, basis_row, strict=True)
                ]
                weights = [
                    budget.check(a - factor * b)
                    for a, b in zip(weights, basis_weights, strict=True)
                ]
        pivot = next((i for i, value in enumerate(row) if value), None)
        if pivot is None:
            norm = budget.check(sum(map(abs, weights), Fraction(0)))
            nulls.append([budget.check(value / norm) for value in weights])
        else:
            divisor = row[pivot]
            basis[pivot] = (
                [budget.check(value / divisor) for value in row],
                [budget.check(value / divisor) for value in weights],
            )
    return nulls


def _check(
    target: str,
    model: CompiledModel,
    roots: set[str],
    evidence: TrainingCompatibilityEvidence,
) -> dict[str, Any]:
    """Prove one finite necessary condition for all target initial observations."""
    result: dict[str, Any] = {
        "property": "shared_initial_output_family",
        "target": target,
        "status": "deferred",
        "comparisons": 0,
        "witnesses": [],
    }
    budget = _Budget()
    try:
        rows = evidence.trajectories
        if len(rows) > LIMITS["rows"]:
            raise _Unsupported("initial-family trajectory row limit")
        error = evidence.contract.absolute_error_bounds.get(target)
        if error is None:
            raise _Unsupported("target absolute observation error is not established")
        expansions, channels, states = [], set(), set()
        expression = model.validated.observation_expressions[target]
        for row in rows:
            view = _InitialView(model, roots, evidence, row, budget)
            expansions.append(_interpret(expression.tree, view.symbol, budget))
            channels.update(view.public_channels)
            states.update(view.initial_states)
        names = sorted({name for item in expansions for name in item.coefficients})
        if len(names) > LIMITS["coordinates"]:
            raise _Unsupported("initial-family shared-coordinate limit")
        matrix = [
            [item.coefficients.get(name, Fraction(0)) for name in names]
            for item in expansions
        ]
        observations = [Fraction(row.targets[target][0]) for row in rows]
        offsets = [item.constant for item in expansions]
        residuals = [
            budget.check(value - offset)
            for value, offset in zip(observations, offsets, strict=True)
        ]
        relations = _null_relations(matrix, budget)
        result.update(
            shared_coordinates=names,
            exact_conditioning_channels=sorted(channels),
            initialized_states=sorted(states),
            expanded_initial_readouts={
                row.trajectory_id: {
                    "constant": str(item.constant),
                    "shared_coefficients": {
                        k: str(v) for k, v in item.coefficients.items()
                    },
                }
                for row, item in zip(rows, expansions, strict=True)
            },
            comparisons=len(relations),
            status="no_conflict_found",
        )
        best, unresolved = None, None
        for weights in relations:
            gap = abs(
                budget.check(
                    sum(
                        (a * b for a, b in zip(weights, residuals, strict=True)),
                        Fraction(0),
                    )
                )
            )
            # L1-normalized weights make this a maximum absolute error lower bound.
            allowed = Fraction(error)
            guard = budget.check(
                sum(
                    (
                        abs(weight)
                        * (
                            Fraction(TARGET_COMPARISON_POLICY["absolute_tolerance"])
                            + Fraction(TARGET_COMPARISON_POLICY["relative_tolerance"])
                            * abs(value)
                        )
                        for weight, value in zip(weights, observations, strict=True)
                    ),
                    Fraction(0),
                )
            )
            if gap <= allowed:
                continue
            selected = [i for i, weight in enumerate(weights) if weight]
            witness = {
                "trajectories": [rows[i].trajectory_id for i in selected],
                "sample_indices": [0] * len(selected),
                "times": [rows[i].time[0] for i in selected],
                "observations": [float(observations[i]) for i in selected],
                "weights_exact": [str(weights[i]) for i in selected],
                "offsets_exact": [str(offsets[i]) for i in selected],
                "design_matrix_exact": [[str(v) for v in matrix[i]] for i in selected],
                "shared_coordinates": names,
                "weighted_design_is_exactly_zero": True,
                "weighted_residual_abs_exact": str(gap),
                "weighted_error_bound_exact": str(allowed),
                "weighted_numerical_guard_exact": str(guard),
                "maximum_initial_absolute_error_lower_bound": _lower_float(gap),
                "declared_absolute_error_bound": error,
                "excess_error_lower_bound": _lower_float(gap - allowed),
                "exceeds_numerical_guard": gap > allowed + guard,
            }
            if gap > allowed + guard:
                if best is None or gap > Fraction(best["weighted_residual_abs_exact"]):
                    best = witness
            elif unresolved is None:
                unresolved = witness
        if best:
            result.update(status="conditional_conflict", witnesses=[best])
        elif unresolved:
            result.update(
                status="deferred",
                numerically_unresolved=[unresolved],
                unavailable=[
                    "initial-family discrepancy is within the numerical guard"
                ],
            )
    except _Unsupported as exc:
        result.update(status="deferred", unavailable=[str(exc)], witnesses=[])
    return result


def assess(problem: PrefixFitProblem, training: dict) -> dict[str, Any]:
    """Assess actual initial graph/readouts against authenticated training only."""
    if not isinstance(problem, PrefixFitProblem):
        raise ValueError("typed PrefixFitProblem required")
    problem = PrefixFitProblem.model_validate(problem.model_dump(mode="json"))
    evidence = TrainingCompatibilityEvidence.model_validate(training)
    validate_evidence(evidence)
    if evidence.context != problem.context:
        raise ValueError("training evidence context differs from problem context")
    if problem.context.fitted_initialization or problem.context.lagged_targets:
        raise ValueError("unlowered open-loop problem required")
    if any(
        parameter.scope.value != "global"
        for parameter in problem.base_candidate.parameters
    ):
        raise ValueError("dynamics parameters must be globally shared")
    base = compile_candidate(problem.base_candidate, problem.context)
    model, _, audit = apply_initialization_graph(base, problem.graph)
    roots = set(audit["trajectory_parameter_names"])
    missing = [
        name
        for name in ("dependencies_exhaustive", "shared_parameters")
        if getattr(evidence.contract, name) is not True
    ]
    checks = (
        [
            {
                "property": "candidate_class_contract",
                "status": "deferred",
                "unavailable": missing,
            }
        ]
        if missing
        else [
            _check(target, model, roots, evidence)
            for target in sorted(problem.context.targets)
        ]
    )
    conflicts = [item for item in checks if item["status"] == "conditional_conflict"]
    deferred = [item for item in checks if item["status"] == "deferred"]
    return {
        "protocol": PROTOCOL,
        "status": "conflict"
        if conflicts
        else "deferred"
        if deferred
        else "no_conflict_found",
        "child_problem_sha256": content_hash(problem.model_dump(mode="json")),
        "evidence_sha256": evidence.sha256,
        "checks": checks,
        "conflicts": conflicts,
        "deferred": deferred,
        "target_comparison_policy": dict(TARGET_COMPARISON_POLICY),
        "resource_limits": dict(LIMITS),
        "contract_provenance": evidence.contract.provenance,
        "initialization_source": "actual_scoped_graph_and_compiled_observed_identities",
        "assumptions": [
            "Initializer/readout coordinates are shared across trajectories.",
            "Initial conditioning observations have explicit zero error.",
            "The weighted certificate permits target observation uncertainty only.",
            "Parameter domains are relaxed to all real values; incompatibility in "
            "this larger affine family implies incompatibility of the original.",
            "No dynamics, uniqueness, future observations or forecast score is used.",
        ],
        "scientific_correctness_established": False,
        "parameter_fitting_performed": False,
        "numerical_rollout_performed": False,
        "llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }


def _display(value: str) -> float | str:
    """Keep presentation finite without changing exact certificate arithmetic."""
    try:
        return float(Fraction(value))
    except OverflowError:
        return "outside float display range; see exact certificate"


def _compact_witness(witness: dict[str, Any], target: str) -> dict[str, Any]:
    """Display the relation and observations; retain exact rationals in the audit."""
    return {
        "initial_observations": [
            {
                "trajectory": trajectory,
                "time": time,
                "target": target,
                "recorded_value": observed,
                "constant_offset": _display(offset),
                "shared_coordinate_multipliers": {
                    name: _display(value)
                    for name, value in zip(
                        witness["shared_coordinates"], row, strict=True
                    )
                    if Fraction(value)
                },
            }
            for trajectory, time, observed, offset, row in zip(
                witness["trajectories"],
                witness["times"],
                witness["observations"],
                witness["offsets_exact"],
                witness["design_matrix_exact"],
                strict=True,
            )
        ],
        "relation": "target(t0) = offset + sum(multiplier * shared_coordinate)",
        "maximum_initial_absolute_error_lower_bound": witness[
            "maximum_initial_absolute_error_lower_bound"
        ],
        "declared_absolute_error_bound": witness["declared_absolute_error_bound"],
        "exceeds_numerical_guard": witness["exceeds_numerical_guard"],
        "certificate": "An exact left-null relation cancels every shared coordinate; "
        "the recorded outputs violate that relation. Full exact certificate is saved.",
        "display_numbers": "Rounded for readability; acceptance uses exact rationals.",
    }


def prompt_view(report: dict[str, Any]) -> dict[str, Any]:
    """Return concise, auditable initial-family conflicts for scoped repair."""
    return {
        "protocol": report["protocol"],
        "status": report["status"],
        "conflicts": [
            {
                "property": check["property"],
                "target": check["target"],
                "shared_coordinates": check["shared_coordinates"],
                "witnesses": [
                    _compact_witness(witness, check["target"])
                    for witness in check["witnesses"]
                ],
                "repair_direction": (
                    "No choice of the shared initializer/readout coefficients can "
                    "reproduce these initial observations within declared errors. "
                    "Change a justified initializer dependence or readout family; "
                    "changing only the ODE after time zero cannot resolve this "
                    "certificate. If the edit scope freezes those components, "
                    "report that wider initialization/readout revision is required "
                    "rather than claiming a repair."
                ),
            }
            for check in report["conflicts"]
        ],
        "deferred": [
            {
                key: check[key]
                for key in ("property", "target", "unavailable")
                if key in check
            }
            for check in report["deferred"]
        ],
        "qualification": "Time-zero necessary condition; no forecast-quality claim.",
    }
