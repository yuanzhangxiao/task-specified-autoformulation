"""The wider vocabulary an external baseline's model may use.

Our own method is held to ``APPROVED_FUNCTION_ARITY``. An external baseline is
scored on what it actually produced: LLM-ODE prints SymPy, LLM-SR writes
numpy, and refusing an operator the method was free to use would measure our
grammar rather than the method. The baseline grammar therefore adds every
elementwise real function those vocabularies offer, under both spellings, and
real-valued exponents.

It is still a closed list of named float functions. Nothing here evaluates
text, and a function outside its real domain raises, so a rollout that leaves
the domain fails exactly as the method's own numpy evaluation would produce a
NaN. Branches, loops and comparisons are not functions and remain refused.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import special


def _sign(value: float) -> float:
    return float((value > 0.0) - (value < 0.0))


def _heaviside(value: float, at_zero: float = 0.5) -> float:
    """numpy's ``heaviside(x, h0)``; SymPy's one-argument form takes 1/2 at 0."""
    if value == 0.0:
        return at_zero
    return 1.0 if value > 0.0 else 0.0


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _floor(value: float) -> float:
    return float(math.floor(value))


def _ceil(value: float) -> float:
    return float(math.ceil(value))


def _mod(value: float, divisor: float) -> float:
    """numpy's ``mod`` and SymPy's ``Mod`` both take the divisor's sign."""
    return value % divisor


def _reciprocal(value: float) -> float:
    return 1.0 / value


def _square(value: float) -> float:
    return value * value


def _exp2(value: float) -> float:
    return math.pow(2.0, value)


UNARY = (1, 1)
BINARY = (2, 2)
VARIADIC = (2, 64)

#: name -> (arity bounds, implementation). ``pow`` and ``power`` are evaluated
#: by the compiler's own power rule so the operator and the call agree.
BASELINE_FUNCTIONS: dict[str, tuple[tuple[int, int], Callable[..., float] | None]] = {
    # trigonometric and hyperbolic, with numpy and SymPy spellings
    "cos": (UNARY, math.cos),
    "tan": (UNARY, math.tan),
    "asin": (UNARY, math.asin),
    "arcsin": (UNARY, math.asin),
    "acos": (UNARY, math.acos),
    "arccos": (UNARY, math.acos),
    "atan": (UNARY, math.atan),
    "arctan": (UNARY, math.atan),
    "atan2": (BINARY, math.atan2),
    "arctan2": (BINARY, math.atan2),
    "sinh": (UNARY, math.sinh),
    "cosh": (UNARY, math.cosh),
    "asinh": (UNARY, math.asinh),
    "arcsinh": (UNARY, math.asinh),
    "acosh": (UNARY, math.acosh),
    "arccosh": (UNARY, math.acosh),
    "atanh": (UNARY, math.atanh),
    "arctanh": (UNARY, math.atanh),
    # exponential and logarithmic
    "exp2": (UNARY, _exp2),
    "expm1": (UNARY, math.expm1),
    "log10": (UNARY, math.log10),
    "log2": (UNARY, math.log2),
    "log1p": (UNARY, math.log1p),
    # powers and roots
    "cbrt": (UNARY, math.cbrt),
    "square": (UNARY, _square),
    "reciprocal": (UNARY, _reciprocal),
    "hypot": (VARIADIC, math.hypot),
    "pow": (BINARY, None),
    "power": (BINARY, None),
    # piecewise
    "sign": (UNARY, _sign),
    "floor": (UNARY, _floor),
    "ceil": (UNARY, _ceil),
    "ceiling": (UNARY, _ceil),
    "heaviside": ((1, 2), _heaviside),
    "Heaviside": ((1, 2), _heaviside),
    "clip": ((3, 3), _clip),
    "mod": (BINARY, _mod),
    "Mod": (BINARY, _mod),
    "fmod": (BINARY, math.fmod),
    # special functions
    "erf": (UNARY, math.erf),
    "erfc": (UNARY, math.erfc),
    "gamma": (UNARY, math.gamma),
    "lgamma": (UNARY, math.lgamma),
    "loggamma": (UNARY, math.lgamma),
    # approved functions under the other libraries' spellings
    "Abs": (UNARY, abs),
    "absolute": (UNARY, abs),
    "fabs": (UNARY, math.fabs),
    "Max": (VARIADIC, max),
    "Min": (VARIADIC, min),
    "maximum": (BINARY, max),
    "minimum": (BINARY, min),
}


def _reduce(pairwise: Callable[[Any, Any], Any]) -> Callable[..., Any]:
    def reduced(first: Any, *rest: Any) -> Any:
        result = first
        for value in rest:
            result = pairwise(result, value)
        return result

    return reduced


def _array_heaviside(value: Any, at_zero: Any = 0.5) -> Any:
    return np.heaviside(value, at_zero)


#: The same vocabulary over numpy arrays, for the evaluators a vendored method
#: fits or rolls out with (LLM-SR's coefficient refit, D3's native rollout).
#: Outside a function's real domain numpy returns NaN, which those callers
#: already treat as a failed fit or rollout.
NUMPY_BASELINE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "arcsin": np.arcsin,
    "acos": np.arccos,
    "arccos": np.arccos,
    "atan": np.arctan,
    "arctan": np.arctan,
    "atan2": np.arctan2,
    "arctan2": np.arctan2,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "asinh": np.arcsinh,
    "arcsinh": np.arcsinh,
    "acosh": np.arccosh,
    "arccosh": np.arccosh,
    "atanh": np.arctanh,
    "arctanh": np.arctanh,
    "exp2": np.exp2,
    "expm1": np.expm1,
    "log10": np.log10,
    "log2": np.log2,
    "log1p": np.log1p,
    "cbrt": np.cbrt,
    "square": np.square,
    "reciprocal": lambda value: 1.0 / np.asarray(value, dtype=float),
    "hypot": _reduce(np.hypot),
    "pow": np.power,
    "power": np.power,
    "sign": np.sign,
    "floor": np.floor,
    "ceil": np.ceil,
    "ceiling": np.ceil,
    "heaviside": _array_heaviside,
    "Heaviside": _array_heaviside,
    "clip": np.clip,
    "mod": np.mod,
    "Mod": np.mod,
    "fmod": np.fmod,
    "erf": special.erf,
    "erfc": special.erfc,
    "gamma": special.gamma,
    "lgamma": special.gammaln,
    "loggamma": special.gammaln,
    "Abs": np.abs,
    "absolute": np.abs,
    "fabs": np.abs,
    "Max": _reduce(np.maximum),
    "Min": _reduce(np.minimum),
    "maximum": np.maximum,
    "minimum": np.minimum,
}

if set(NUMPY_BASELINE_FUNCTIONS) != set(BASELINE_FUNCTIONS):  # pragma: no cover
    raise AssertionError("the array vocabulary must match the scalar one")
