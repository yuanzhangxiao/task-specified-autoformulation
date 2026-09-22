"""Public-specification witnesses for saved basin models, not a science score."""

from __future__ import annotations

import ast
import math
from typing import Literal

from pydantic import Field

from autoformalism.expressions import (
    ModelValidationError,
    RuntimeExpressionError,
    ValidationContext,
    compile_candidate,
)
from autoformalism.expressions.compiler import _evaluate
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.rebuttal import basin_algebra as algebra
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash

POLICY = "basin-public-equation-witnesses-1"
TOLERANCE = 1e-8
UNVERIFIED_ERRORS = (
    ValueError,
    ArithmeticError,
    ModelValidationError,
    RuntimeExpressionError,
)


class Finding(StrictSchema):
    """One scoped obligation with exact expressions or a reproducible witness."""

    code: str
    status: Literal["pass", "fail", "unverified", "not_applicable"]
    scope: str
    message: str
    evidence: dict = Field(default_factory=dict)
    universal_scientific_certification: Literal[False] = False


def finding(code: str, status: str, message: str, **evidence) -> dict:
    """Validate every emitted finding against the same reporting schema."""
    return Finding(
        code=code,
        status=status,
        scope="named_physical_depths_and_public_survey_geometry",
        message=message,
        evidence=evidence,
    ).model_dump(mode="json")


def _finite_mapping(values: dict | None) -> dict[str, float] | None:
    if values is None:
        return None
    if any(
        isinstance(v, bool) or not isinstance(v, int | float) for v in values.values()
    ):
        raise ValueError("audit values must be numeric")
    result = {k: float(v) for k, v in values.items()}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("audit values must be finite")
    return result


def _value(node: ast.AST, environment: dict) -> float:
    parsed = RestrictedParser(max_length=40000, max_nodes=4096).parse(
        ast.unparse(node), location="public static probe"
    )
    if parsed.symbols - environment.keys():
        raise ValueError("probe requires unmapped state or parameter")
    return _evaluate(parsed, environment)


def _environment(survey: dict, parameters: dict | None) -> dict:
    if parameters is None:
        raise ValueError("complete saved fitted parameters unavailable")
    return {**survey, **parameters, "inflow_up": 0.0, "inflow_down": 0.0, "t": 0.0}


def _input_check(polynomial, state, input_name, area, surveys, parameters):
    coefficient = algebra.coefficient(polynomial, input_name)
    if coefficient is None:
        return finding(
            f"inflow_conversion:{state}",
            "unverified",
            "Input occurs nonlinearly; a constant depth conversion was not proved.",
        )
    residual = algebra.add(algebra.multiply(algebra.atom(area), coefficient), {(): -1})
    evidence = {
        "required_identity": f"{area} * coefficient({input_name}) = 1",
        "coefficient": algebra.render(coefficient),
        "residual": algebra.render(residual),
        "structurally_enforced": not residual,
    }
    if not residual:
        return finding(
            f"inflow_conversion:{state}",
            "pass",
            "The external volumetric input has the required depth conversion.",
            **evidence,
        )
    samples = []
    for survey in surveys:
        env = {**survey, **(parameters or {})}
        if algebra.symbols(residual) - env.keys():
            continue
        try:
            value = algebra.number(residual, env)
            if math.isfinite(value):
                samples.append({"survey": survey, "dimensionless_residual": value})
        except UNVERIFIED_ERRORS:
            continue
    bad = next(
        (s for s in samples if abs(s["dimensionless_residual"]) > TOLERANCE), None
    )
    return finding(
        f"inflow_conversion:{state}",
        "fail" if bad else "unverified",
        "The saved coefficient violates the public inflow conversion."
        if bad
        else "Conversion is parameter/geometry dependent, not an enforced identity.",
        **evidence,
        sampled_checks=samples,
        counterexample=bad,
    )


def _transfer(up, down, up_poly, down_poly, up_name, down_name, surveys, parameters):
    total = algebra.add(
        algebra.multiply(algebra.atom("area_up"), up_poly),
        algebra.multiply(algebra.atom("area_down"), down_poly),
    )
    affected = {
        key: value
        for key, value in total.items()
        if any(up_name in algebra.factor_symbols(n) for n, _ in key)
    }
    evidence = {
        "total_storage_derivative": algebra.render(total),
        "upstream_dependent_terms": algebra.render(affected),
    }
    if up_name not in algebra.symbols(down_poly):
        return finding(
            "internal_transfer_cancellation",
            "unverified",
            "No direct upstream-depth influence was found after algebraic expansion; "
            "cancellation would be vacuous or require a different state mapping.",
            **evidence,
        )
    if not affected:
        return finding(
            "internal_transfer_cancellation",
            "pass",
            "Upstream-depth terms cancel in the area-weighted sum. This does not "
            "certify the remaining inflows, outlet or latent interpretation.",
            **evidence,
        )
    witnesses = []
    for survey in surveys:
        try:
            env = _environment(survey, parameters)
            low, high = survey["crest_up"] / 2, survey["crest_up"] * 1.25 + 0.01
            env[down_name] = survey["crest_down"] * 1.25 + 0.01
            values = []
            for height in (low, high):
                env[up_name] = height
                values.append(
                    survey["area_up"] * _value(up, env)
                    + survey["area_down"] * _value(down, env)
                )
            delta = values[1] - values[0]
            if abs(delta) > TOLERANCE * max(1.0, *map(abs, values)):
                witnesses.append(
                    {
                        "survey": survey,
                        "upstream_depths": [low, high],
                        "downstream_depth": env[down_name],
                        "total_storage_derivatives": values,
                        "difference_m3_per_min": delta,
                    }
                )
        except (KeyError, *UNVERIFIED_ERRORS):
            continue
    return finding(
        "internal_transfer_cancellation",
        "fail" if witnesses else "unverified",
        "Changing only upstream depth changes total stored-water loss under zero "
        "inflow. Under the public two-basin assumptions, internal transfer must cancel."
        if witnesses
        else "Area-weighted cancellation was not proved; free coefficients "
        "or unsupported identities may still permit it.",
        **evidence,
        counterexamples=witnesses,
    )


def _outlet(down, polynomial, down_name, up_name, surveys, parameters, states):
    symbols = algebra.symbols(polynomial)
    if down_name not in symbols:
        ambiguous = bool(symbols & (set(states) - {down_name, up_name}))
        return [
            finding(
                "storage_dependent_outlet",
                "unverified" if ambiguous else "fail",
                "Other dynamic states influence the output; their storage/outlet "
                "interpretation requires an explicit mapping."
                if ambiguous
                else "The expanded derivative has no downstream-depth dependence. "
                "A storage-dependent outlet is absent in these coordinates.",
                expanded_rhs=ast.unparse(down),
            )
        ]
    rows = [
        finding(
            "storage_dependent_outlet",
            "pass",
            "Downstream depth occurs in the derivative. Presence alone does not prove "
            "an outlet, its sign, threshold or activity.",
            expanded_rhs=ast.unparse(down),
        )
    ]
    probes, errors = [], []
    for survey in surveys:
        try:
            env = _environment(survey, parameters)
            if up_name:
                env[up_name] = 0.0  # below the public upstream crest
            crest = survey["crest_down"]
            levels = [0.0, crest / 2, crest, crest * 1.25 + 0.01]
            derivatives = []
            for height in levels:
                env[down_name] = height
                derivatives.append(_value(down, env))
            probes.append(
                {
                    "survey": survey,
                    "downstream_depths": levels,
                    "upstream_depth": 0.0 if up_name else None,
                    "inflow_up": 0.0,
                    "inflow_down": 0.0,
                    "depth_derivatives_m_per_min": derivatives,
                }
            )
        except (KeyError, *UNVERIFIED_ERRORS) as exc:
            errors.append(str(exc))
    bad = [
        p
        for p in probes
        if any(abs(v) > TOLERANCE for v in p["depth_derivatives_m_per_min"][:3])
        or p["depth_derivatives_m_per_min"][3] > TOLERANCE
        or p["depth_derivatives_m_per_min"][3] == 0.0
    ]
    near_zero = [
        p for p in probes if 0 < abs(p["depth_derivatives_m_per_min"][3]) <= TOLERANCE
    ]
    rows.append(
        finding(
            "free_outlet_threshold_probes",
            "fail"
            if bad
            else "pass"
            if probes and not errors and not near_zero
            else "unverified",
            "Zero-inflow probes require zero discharge at/below the surveyed crest "
            "and recession above it. Passing points is not a global threshold proof.",
            probes=probes,
            counterexamples=bad,
            unavailable_probes=errors,
            near_zero_above_crest_probes=near_zero,
            parameters_source="saved_training_fit",
            time_series_used=False,
        )
    )
    return rows


def assess(
    candidate: CandidateModel,
    context: ValidationContext,
    case: str,
    surveys: list[dict],
    parameters: dict | None,
) -> dict:
    """Assess final assembled equations without interpreting scientific-role prose."""
    if case not in {"coupled", "independent"}:
        raise ValueError("unknown basin public contract")
    model = compile_candidate(candidate, context)
    parameters = _finite_mapping(parameters)
    if parameters is not None and set(parameters) != set(model.parameter_names):
        raise ValueError("saved parameter vector differs from candidate")
    surveys = [_finite_mapping(s) for s in surveys]
    if any(
        s.get("area_up", 0) <= 0
        or s.get("area_down", 0) <= 0
        or s.get("crest_up", -1) < 0
        or s.get("crest_down", -1) < 0
        for s in surveys
    ):
        raise ValueError("public surveys require positive areas and nonnegative crests")
    down_name = next(
        (
            s
            for s, c in model.direct_state_observation_channels.items()
            if c == "h_down"
        ),
        None,
    )
    up_state = next((s for s in candidate.states if s.name == "h_up"), None)
    up_name = (
        "h_up"
        if up_state and up_state.unit in {"m", "unspecified"} and down_name != "h_up"
        else None
    )
    evidence = {
        "policy": POLICY,
        "candidate_sha256": content_hash(candidate.model_dump(mode="json")),
        "case": case,
        "coordinate_bindings": {
            "downstream_depth": down_name,
            "upstream_depth": up_name,
        },
        "assumptions": [
            "Identity observation binds downstream depth to the public metre channel.",
            "For coupled checks, named h_up is interpreted as physical upstream depth; "
            "this latent meaning is conditional, not independently certified.",
            "Formal identities apply where their denominators are defined and nonzero.",
            "Parameter units and latent coordinate transforms are not inferred.",
        ],
        "scientific_compliance_certified": False,
        "fitted_point_failure_proves_structural_impossibility": False,
        "automatic_rejection": False,
        "trajectory_values_used": False,
        "validation_used": False,
    }
    if not down_name:
        return {
            **evidence,
            "checks": [
                finding(
                    "physical_coordinate_mapping",
                    "unverified",
                    "No identity observation exposes downstream depth; no latent "
                    "coordinate transformation was invented.",
                )
            ],
        }
    processes = {p.name: p.expression for p in candidate.processes}
    rhs = {e.state: e.rhs for e in candidate.state_equations}
    checks = []
    try:
        down = algebra.expanded(rhs[down_name], processes)
        down_poly = algebra.formal(down)
        checks.append(
            _input_check(
                down_poly, down_name, "inflow_down", "area_down", surveys, parameters
            )
        )
        checks.extend(
            _outlet(
                down,
                down_poly,
                down_name,
                up_name,
                surveys,
                parameters,
                model.state_names,
            )
        )
        if case == "coupled" and up_name:
            up = algebra.expanded(rhs[up_name], processes)
            up_poly = algebra.formal(up)
            checks.append(
                _input_check(
                    up_poly, up_name, "inflow_up", "area_up", surveys, parameters
                )
            )
            checks.append(
                _transfer(
                    up,
                    down,
                    up_poly,
                    down_poly,
                    up_name,
                    down_name,
                    surveys,
                    parameters,
                )
            )
        elif case == "coupled":
            checks.append(
                finding(
                    "internal_transfer_cancellation",
                    "unverified",
                    "No named physical upstream depth; balance needs an "
                    "explicit latent-to-storage mapping.",
                )
            )
        # Multiple summands are a repair observation, not proof of double counting.
        for state in (down_name, up_name):
            if not state:
                continue
            node = algebra.expanded(rhs[state], processes)

            def summands(n):
                if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add | ast.Sub):
                    return summands(n.left) + summands(n.right)
                return [n]

            input_name = "inflow_down" if state == down_name else "inflow_up"
            uses = [
                ast.unparse(n)
                for n in summands(node)
                if input_name in algebra.symbols(algebra.formal(n))
            ]
            if len(uses) > 1:
                checks.append(
                    finding(
                        f"multiple_inflow_contributions:{state}",
                        "unverified",
                        "Several additive contributions use the same external inflow. "
                        "Their sum determines balance; review scientific intent.",
                        contributions=uses,
                    )
                )
    except (*UNVERIFIED_ERRORS, RecursionError) as exc:
        checks.append(finding("bounded_equation_analysis", "unverified", str(exc)))
    return {**evidence, "checks": checks}
