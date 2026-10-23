"""Opt-in bounded multiprecision affine transitions and Frechet derivatives.

The norm trigger is a conservative escalation heuristic, not an error bound.
Agreement between two precisions does not replace independent ODE verification.
Inputs are already compiled numeric matrices; no proposer text is evaluated.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from time import monotonic

import numpy as np
from scipy.linalg import expm, expm_frechet

from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial


@dataclass(frozen=True)
class AffinePrecisionPolicy:
    """Frozen numerical policy; all limits apply without changing parameter bounds."""

    dimensionless_norm_threshold: float = 1e6
    decimal_digits: tuple[int, int] = (60, 90)
    agreement_rtol: float = 1e-24
    agreement_atol: float = 1e-30
    maximum_dimension: int = 32
    maximum_directions: int = 128
    maximum_scalings: int = 160
    maximum_taylor_terms: int = 96
    transition_seconds: float = 60.0

    def __post_init__(self) -> None:
        for name in (
            "dimensionless_norm_threshold",
            "agreement_rtol",
            "agreement_atol",
            "transition_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"positive finite {name} required")
        if (
            not isinstance(self.decimal_digits, tuple)
            or len(self.decimal_digits) != 2
            or any(type(value) is not int for value in self.decimal_digits)
            or not 30 <= self.decimal_digits[0] < self.decimal_digits[1] <= 200
        ):
            raise ValueError("two increasing decimal precisions in [30, 200] required")
        for name in (
            "maximum_dimension",
            "maximum_directions",
            "maximum_scalings",
            "maximum_taylor_terms",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"positive integer {name} required")


class AffinePrecisionUnavailable(NumericalInvalidTrial):
    """A bounded precision attempt supplied no reliable finite transition."""


def _deadline(deadline: float) -> None:
    if monotonic() >= deadline:
        raise TimeoutError("affine precision transition wall-clock limit reached")


def _numeric_matrices(matrix, directions) -> tuple[np.ndarray, list[np.ndarray]]:
    """Reject malformed numeric input rather than masking programming errors."""
    matrix = np.asarray(matrix, dtype=float)
    directions = [np.asarray(item, dtype=float) for item in directions]
    if matrix.ndim != 2 or not matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("nonempty square affine transition generator required")
    if any(item.shape != matrix.shape for item in directions):
        raise ValueError("affine direction shape differs from generator")
    if not np.isfinite(matrix).all() or any(
        not np.isfinite(item).all() for item in directions
    ):
        raise NumericalInvalidTrial("nonfinite affine transition input")
    return matrix, directions


def _multiprecision(matrix, directions, digits, policy, deadline):
    """Differentiate the same Taylor/scaling-squaring recurrence at fixed precision.

    All Frechet directions share the base polynomial and squaring schedule. This
    avoids one expensive doubled-dimension exponential per parameter. Contexts are
    private and no global mpmath precision is changed. Each multiplication is bounded
    cooperatively before and after; a single multiplication cannot be interrupted.
    """
    import mpmath

    context = mpmath.mp.clone()
    context.dps = digits
    base = context.matrix(matrix.tolist())
    norm = context.mnorm(base, 1)
    scalings = max(0, int(context.ceil(context.log(norm, 2))) + 5) if norm else 0
    if scalings > policy.maximum_scalings:
        raise AffinePrecisionUnavailable("affine precision scaling allowance exceeded")
    # Guard digits compensate for repeated squaring; the two target precisions
    # still differ by the declared amount and are compared before float casting.
    context.dps = digits + 10 + math.ceil(scalings * math.log10(4))
    base = context.matrix(matrix.tolist()) / context.mpf(2) ** scalings
    active = [
        context.matrix(item.tolist()) / context.mpf(2) ** scalings
        for item in directions
    ]
    size = len(matrix)
    term = context.eye(size)
    value = context.eye(size)
    dterm = [context.zeros(size) for _ in active]
    derivative = [context.zeros(size) for _ in active]
    epsilon = context.mpf(10) ** -(context.dps - 5)
    # A scalar Taylor remainder bound fixes termination independently of direction
    # magnitudes. For ||A|| <= 1/32 the differentiated series decays as quickly.
    bound = context.mpf(1)
    scaled_norm = context.mnorm(base, 1)
    for order in range(1, policy.maximum_taylor_terms + 1):
        _deadline(deadline)
        next_dterm = []
        for previous, direction in zip(dterm, active, strict=True):
            _deadline(deadline)
            next_dterm.append((previous * base + term * direction) / order)
            _deadline(deadline)
        term = term * base / order
        value += term
        dterm = next_dterm
        derivative = [a + b for a, b in zip(derivative, dterm, strict=True)]
        bound *= scaled_norm / order
        if order > 1 and bound * order < epsilon:
            break
    else:
        raise AffinePrecisionUnavailable("affine precision Taylor allowance exceeded")
    for _ in range(scalings):
        _deadline(deadline)
        updated = []
        for item in derivative:
            _deadline(deadline)
            updated.append(value * item + item * value)
            _deadline(deadline)
        value = value * value
        derivative = updated
    _deadline(deadline)
    return context, value, derivative


class TransitionEvaluator:
    """Evaluate transitions while retaining observable per-backend accounting."""

    def __init__(self, policy: AffinePrecisionPolicy) -> None:
        if not isinstance(policy, AffinePrecisionPolicy):
            raise TypeError("typed affine precision policy required")
        import mpmath

        self.policy = policy
        self.audit = {
            "protocol": "affine-transition-precision-1",
            "policy": asdict(policy),
            "trigger": "dimensionless generator one-norm; heuristic, not certification",
            "comparison_scope": "numeric compiled generator and Frechet directions",
            "mpmath_version": mpmath.__version__,
            "output_representation": (
                "float64; subsequent state arithmetic remains float64"
            ),
            "symbolic_coefficient_evaluation": "original compiled float64 arithmetic",
            "independent_rollout_still_required": True,
            "double_started": 0,
            "double_completed": 0,
            "multiprecision_started": 0,
            "multiprecision_completed": 0,
            "multiprecision_passes_started": 0,
            "multiprecision_passes_completed": 0,
            "failures": 0,
            "largest_dimensionless_norm": 0.0,
            "norm_overflows": 0,
            "maximum_agreement_ratio": 0.0,
        }

    def __call__(self, matrix, directions, deadline=None):
        """Return an entire finite transition/derivative packet, or raise."""
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("finite affine precision deadline required")
        deadline = min(
            monotonic() + self.policy.transition_seconds,
            math.inf if deadline is None else deadline,
        )
        _deadline(deadline)
        matrix, directions = _numeric_matrices(matrix, directions)
        with np.errstate(over="ignore"):
            norm = float(np.linalg.norm(matrix, 1))
        if not math.isfinite(norm):
            # Preserve a JSON-finite diagnostic even when finite entries overflow
            # their float64 sum. The high-precision scaling cap still rejects it.
            norm = np.finfo(float).max
            self.audit["norm_overflows"] += 1
        self.audit["largest_dimensionless_norm"] = max(
            self.audit["largest_dimensionless_norm"], norm
        )
        high = norm > self.policy.dimensionless_norm_threshold
        backend = "multiprecision" if high else "double"
        self.audit[f"{backend}_started"] += 1
        try:
            if high:
                value, derivative = self._high(matrix, directions, deadline)
            else:
                value = expm(matrix)
                _deadline(deadline)
                derivative = []
                for direction in directions:
                    _deadline(deadline)
                    derivative.append(
                        expm_frechet(matrix, direction, compute_expm=False)
                        if np.any(direction)
                        else np.zeros_like(matrix)
                    )
                    _deadline(deadline)
            if not np.isfinite(value).all() or any(
                not np.isfinite(item).all() for item in derivative
            ):
                raise AffinePrecisionUnavailable("nonfinite affine precision result")
            _deadline(deadline)
        except BaseException:
            self.audit["failures"] += 1
            raise
        self.audit[f"{backend}_completed"] += 1
        return value, derivative

    def _high(self, matrix, directions, deadline):
        policy = self.policy
        if (
            len(matrix) > policy.maximum_dimension
            or len(directions) > policy.maximum_directions
        ):
            raise AffinePrecisionUnavailable(
                "affine precision dimension allowance exceeded"
            )
        # Exactly-zero directions (e.g. initial-only coordinates) need no matrix
        # derivative work but retain their original positions in the returned list.
        indices = [i for i, direction in enumerate(directions) if np.any(direction)]
        active = [directions[i] for i in indices]
        values = []
        for digits in policy.decimal_digits:
            _deadline(deadline)
            self.audit["multiprecision_passes_started"] += 1
            values.append(_multiprecision(matrix, active, digits, policy, deadline))
            self.audit["multiprecision_passes_completed"] += 1
        low_context, low, low_derivative = values[0]
        context, high, high_derivative = values[1]
        del low_context
        maximum = context.mpf(0)
        for first, second in zip(
            [low, *low_derivative], [high, *high_derivative], strict=True
        ):
            _deadline(deadline)
            for a, b in zip(first, second, strict=True):
                if not context.isfinite(a) or not context.isfinite(b):
                    raise AffinePrecisionUnavailable(
                        "nonfinite multiprecision transition"
                    )
                ratio = abs(context.mpf(a) - b) / (
                    context.mpf(policy.agreement_atol)
                    + context.mpf(policy.agreement_rtol) * max(abs(a), abs(b))
                )
                maximum = max(maximum, ratio)
        self.audit["maximum_agreement_ratio"] = max(
            self.audit["maximum_agreement_ratio"],
            float(min(maximum, context.mpf(np.finfo(float).max))),
        )
        if maximum > 1:
            raise AffinePrecisionUnavailable("multiprecision transition disagreement")
        result = [np.zeros_like(matrix) for _ in directions]
        for index, item in zip(indices, high_derivative, strict=True):
            result[index] = np.asarray(item.tolist(), dtype=float)
        return np.asarray(high.tolist(), dtype=float), result
