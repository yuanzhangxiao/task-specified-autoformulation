"""Exact quotient and feasible lift for the reviewed task-15 detention family.

This utility certifies an algebraic parameter redundancy; it does not change an
optimizer or certify the scientific validity of the equation.  The reduced
domain is correlated and must not be replaced by the Cartesian product of its
coordinate extrema.  Rational arithmetic keeps boundary/zero cases exact.
Floating representative parameters are for inspection; an optimizer must still
independently verify their original-equation rollout before retaining them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import isfinite
from numbers import Rational, Real

PROTOCOL = "llm-sr-detention-exact-reduction-1"
ORIGINAL_NAMES = (
    "C",
    "beta",
    "eps_s",
    "h_max",
    "Q_max",
    "dt",
    "k_loss",
    "k_gw",
    "kappa",
)
REDUCED_NAMES = ("C", "beta", "eps_s", "Q_max", "dt", "b", "d", "M")
UNCHANGED = ("C", "beta", "eps_s", "Q_max", "dt")
Number = Real | Fraction
Bounds = Mapping[str, Sequence[Number]]


def _fraction(value: Number) -> Fraction:
    """Interpret a finite supplied number exactly, rejecting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("finite real coordinates are required")
    if not isfinite(value):
        raise ValueError("finite real coordinates are required")
    if isinstance(value, Rational):
        return Fraction(value)
    return Fraction.from_float(float(value))


def _values(values: Mapping[str, Number], names: tuple[str, ...]) -> dict:
    if set(values) != set(names):
        raise ValueError("coordinate roster differs from reviewed task 15")
    return {name: _fraction(values[name]) for name in names}


def _bounds(bounds: Bounds) -> dict[str, tuple[Fraction, Fraction]]:
    if set(bounds) != set(ORIGINAL_NAMES):
        raise ValueError("bound roster differs from reviewed task 15")
    result = {}
    for name, pair in bounds.items():
        if len(pair) != 2:
            raise ValueError("each parameter requires two finite bounds")
        low, high = (_fraction(value) for value in pair)
        if low > high:
            raise ValueError("parameter bounds are reversed")
        result[name] = (low, high)
    for name in ("C", "Q_max", "k_loss", "kappa"):
        if result[name][0] < 0:
            raise ValueError("reviewed branch requires nonnegative parameter domains")
    for name in ("dt", "h_max", "eps_s", "beta"):
        if result[name][0] <= 0:
            raise ValueError("reviewed branch requires positive parameter domains")
    return result


def reduce_parameters(parameters: Mapping[str, Number]) -> dict[str, Fraction]:
    """Map nine original coordinates to eight exact rational combinations.

    Keep the rational values through a round trip.  Converting them to binary
    floats first can move a boundary point outside the exact reduced domain.
    """
    point = _values(parameters, ORIGINAL_NAMES)
    result = {name: point[name] for name in UNCHANGED}
    result.update(
        b=point["k_gw"] - point["k_loss"],
        d=point["k_gw"] * point["kappa"],
        M=point["h_max"] + point["dt"] * point["k_gw"],
    )
    return result


def feasible_g_interval(
    coordinates: Mapping[str, Number], bounds: Bounds
) -> tuple[Fraction, Fraction]:
    """Return exactly the original ``k_gw`` values lifting a reduced point.

    For T>0, linear constraints give G intersect (b+L) intersect (M-H)/T.
    The product d=g*kappa supplies one further interval because kappa>=0.
    Zero products are handled without division or an artificial sign choice.
    """
    point, domain = _values(coordinates, REDUCED_NAMES), _bounds(bounds)
    if any(not domain[n][0] <= point[n] <= domain[n][1] for n in UNCHANGED):
        raise ValueError("unchanged coordinate lies outside the original domain")
    b, d, m, t = (point[name] for name in ("b", "d", "M", "dt"))
    low = max(
        domain["k_gw"][0],
        b + domain["k_loss"][0],
        (m - domain["h_max"][1]) / t,
    )
    high = min(
        domain["k_gw"][1],
        b + domain["k_loss"][1],
        (m - domain["h_max"][0]) / t,
    )
    klo, khi = domain["kappa"]
    if d > 0:
        if khi == 0:
            raise ValueError("nonzero product cannot have zero kappa")
        low = max(low, d / khi)
        if klo > 0:
            high = min(high, d / klo)
    elif d < 0:
        if khi == 0:
            raise ValueError("nonzero product cannot have zero kappa")
        high = min(high, d / khi)
        if klo > 0:
            low = max(low, d / klo)
    elif klo > 0:
        low, high = max(low, Fraction(0)), min(high, Fraction(0))
    if low > high:
        raise ValueError("reduced coordinates have no feasible original lift")
    return low, high


def lift_parameters(
    coordinates: Mapping[str, Number],
    bounds: Bounds,
    *,
    preferred_g: Number | None = None,
    preferred_kappa: Number | None = None,
) -> dict[str, Fraction]:
    """Choose a feasible representative without freezing a family coordinate.

    Preferences choose only within the feasible interval for this reduced point.
    Their clipping does not alter the reduced coordinates or their feasible set.
    Returned rational parameters satisfy both identities and original bounds
    exactly.  The g=d=0 fibre retains the permitted kappa range.
    """
    point, domain = _values(coordinates, REDUCED_NAMES), _bounds(bounds)
    low, high = feasible_g_interval(point, domain)
    g = (low + high) / 2
    if preferred_g is not None:
        g = min(high, max(low, _fraction(preferred_g)))
    klo, khi = domain["kappa"]
    if g == 0:
        if point["d"] != 0:
            raise ValueError("nonzero product has no zero-g lift")
        kappa = (klo + khi) / 2
        if preferred_kappa is not None:
            kappa = min(khi, max(klo, _fraction(preferred_kappa)))
    else:
        kappa = point["d"] / g
    result = {name: point[name] for name in UNCHANGED}
    result.update(
        k_gw=g,
        k_loss=g - point["b"],
        kappa=kappa,
        h_max=point["M"] - point["dt"] * g,
    )
    if any(not domain[n][0] <= result[n] <= domain[n][1] for n in ORIGINAL_NAMES):
        raise ValueError("lift violates an original bound")
    if reduce_parameters(result) != point:
        raise ValueError("lift does not preserve the reduced coordinates")
    return result


def _encoded(value: Fraction) -> dict[str, str]:
    """JSON-safe exact ratio; strings avoid downstream integer precision loss."""
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def reduction_certificate(parameters: Mapping[str, Number], bounds: Bounds) -> dict:
    """Describe the exact model-15 quotient and one feasible inspection lift."""
    point, domain = _values(parameters, ORIGINAL_NAMES), _bounds(bounds)
    if any(not domain[n][0] <= point[n] <= domain[n][1] for n in ORIGINAL_NAMES):
        raise ValueError("original parameters lie outside the declared domain")
    reduced = reduce_parameters(point)
    interval = feasible_g_interval(reduced, domain)
    lifted = lift_parameters(reduced, domain)
    floating = {name: float(lifted[name]) for name in ORIGINAL_NAMES}
    rounded_reduced = reduce_parameters(floating)
    return {
        "protocol": PROTOCOL,
        "scope": "reviewed positive-capacity task-15 family and unchanged bounds",
        "original_coordinates": list(ORIGINAL_NAMES),
        "reduced_coordinates": list(REDUCED_NAMES),
        "definitions": {
            "b": "k_gw-k_loss",
            "d": "k_gw*kappa",
            "M": "h_max+dt*k_gw",
        },
        "parameters": {name: float(point[name]) for name in ORIGINAL_NAMES},
        "combinations": {name: float(reduced[name]) for name in ("b", "d", "M")},
        "exact_reduced_coordinates": {n: _encoded(v) for n, v in reduced.items()},
        "exact_feasible_k_gw_interval": [_encoded(value) for value in interval],
        "exact_representative_parameters": {n: _encoded(v) for n, v in lifted.items()},
        "representative_parameters": floating,
        "representative_roundoff_absolute": {
            name: float(abs(rounded_reduced[name] - reduced[name]))
            for name in REDUCED_NAMES
        },
        "original_bounds_satisfied": True,
        "exact_combinations_preserved": True,
        "rank_ceiling": 8,
        "declared_coordinates": 9,
        "reduced_domain_is_cartesian_box": False,
        "optimizer_coordinates_changed": False,
        "floating_representative_requires_independent_rollout": True,
        "unique_original_coefficients_certified": False,
        "scientific_validity_certified": False,
    }
