"""Check the internal transfer in the registered two-depth basin balance."""

import ast
from fractions import Fraction
from math import isfinite

from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.rebuttal import basin_algebra as algebra

HEAD = {"max(h_up - crest_up, 0)", "max(0, h_up - crest_up)"}
AREAS = {"area_up", "area_down"}


def _upstream_terms(formula: dict, state: str, area: str) -> algebra.Polynomial:
    node = algebra.expanded(formula["equations"][state], {})
    for call in ast.walk(node):
        if isinstance(call, ast.Call) and call.func.id in {"max", "min"}:
            call.args.sort(key=ast.dump)
    volume = algebra.multiply(algebra.atom(area), algebra.formal(node))
    return {
        key: scalar
        for key, scalar in volume.items()
        if any("h_up" in algebra.factor_symbols(name) for name, _ in key)
    }


def _head_coefficient(terms, parameters, survey):
    """Recognize one shared coefficient times positive head and known geometry."""
    if len(terms) != 1:
        return None
    ((key, scalar),) = terms.items()
    powers = dict(key)
    heads = HEAD.intersection(powers)
    gains = parameters.intersection(powers)
    if len(heads) != 1 or len(gains) != 1:
        return None
    gain = next(iter(gains))
    if powers.pop(heads.pop()) != 1 or powers.pop(gain) != 1:
        return None
    if powers.keys() - AREAS or any(abs(n) > 2 for n in powers.values()):
        return None
    for area, exponent in powers.items():
        scalar *= Fraction(str(survey[area])) ** exponent
    return gain, scalar


def check_transfer_balance(case: dict, graph: dict, formula: dict) -> dict:
    """Prove cancellation, or a contradictory nonzero flux at a surveyed geometry.

    The proof concerns named, physically initialized depths and an active shared
    transfer. It applies where the restricted RHS expressions are defined.
    """
    if case.get("provenance", {}).get("case") != "phase_c_detention_coupled_noise0_v1":
        return {"status": "not_applicable", "reason": "Different public contract."}
    initials = graph["initializers"]
    if (
        graph["variable_types"] != {"h_up": "differential", "h_down": "differential"}
        or initials.get("h_up") != {"mode": "observed_initial", "channel": "initial_up"}
        or initials.get("h_down") != {"mode": "observed_initial", "channel": "h_down"}
    ):
        return {
            "status": "unverified",
            "reason": "Both named depths need their measured initial values.",
        }
    unknown = {
        "status": "unverified",
        "reason": "The transfer identity could not be decided from the equations.",
    }
    try:
        upstream = _upstream_terms(formula, "h_up", "area_up")
        downstream = _upstream_terms(formula, "h_down", "area_down")
        if not upstream or not downstream:
            return unknown
        if not algebra.add(upstream, downstream):
            return {
                "status": "pass",
                "reason": "Upstream-depth terms cancel in the total volume rate.",
                "witness": {
                    "upstream_volume_terms": algebra.render(upstream),
                    "downstream_volume_terms": algebra.render(downstream),
                },
            }
        # A remaining symbolic term alone is inconclusive: unknown coefficients
        # may be fitted to enforce equality. Recognize only a fixed ratio for
        # the same single gain, on a surveyed event that begins above the crest.
        parameters = set(formula["parameters"])
        for row in case["evidence"].get("trajectories", ()):
            c = row["covariates"]
            if not row["trajectory_id"].startswith("train_") or not all(
                isfinite(c[n])
                for n in ("area_up", "area_down", "crest_up", "initial_up")
            ):
                continue
            if (
                min(c["area_up"], c["area_down"]) <= 0
                or c["initial_up"] <= c["crest_up"]
            ):
                continue
            up = _head_coefficient(upstream, parameters, c)
            down = _head_coefficient(downstream, parameters, c)
            if up and down and up[0] == down[0]:
                if up[1] + down[1] == 0:
                    continue
                return {
                    "status": "fail",
                    "reason": (
                        "Upstream loss and downstream gain differ for every nonzero "
                        "shared transfer coefficient. Setting it to zero removes "
                        "the required transfer."
                    ),
                    "witness": {
                        "trajectory": row["trajectory_id"],
                        "area_up": c["area_up"],
                        "area_down": c["area_down"],
                        "initial_up": c["initial_up"],
                        "crest_up": c["crest_up"],
                        "common_factor": f"{up[0]} * max(h_up - crest_up, 0)",
                        "upstream_volume_multiplier": str(up[1]),
                        "downstream_volume_multiplier": str(down[1]),
                        "net_volume_multiplier": str(up[1] + down[1]),
                    },
                }
    except (KeyError, ValueError, ArithmeticError, ModelValidationError):
        return unknown
    return unknown
