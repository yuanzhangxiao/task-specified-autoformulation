"""Small outward-rounded Decimal intervals for the bounded premise fixture.

Containment assumes Python's documented Decimal/libmpdec arithmetic contract:
basic operations obey directed rounding; exp and sqrt are correctly rounded to
nearest and are expanded by a representable neighbor on each side. Inputs are
exact decimal constants, never silently converted binary floats. This is a
restricted arithmetic implementation, not a general validated ODE solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import (
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    Clamped,
    Context,
    Decimal,
    DivisionByZero,
    FloatOperation,
    InvalidOperation,
    Overflow,
    Subnormal,
    Underflow,
)

PRECISION = 50
_EMIN = -999999
_EMAX = 999999
_TRAPS = [
    Clamped,
    DivisionByZero,
    FloatOperation,
    InvalidOperation,
    Overflow,
    Subnormal,
    Underflow,
]
_DOWN = Context(
    prec=PRECISION, rounding=ROUND_FLOOR, Emin=_EMIN, Emax=_EMAX, traps=_TRAPS
)
_UP = Context(
    prec=PRECISION, rounding=ROUND_CEILING, Emin=_EMIN, Emax=_EMAX, traps=_TRAPS
)
_NEAREST = Context(
    prec=PRECISION, rounding=ROUND_HALF_EVEN, Emin=_EMIN, Emax=_EMAX, traps=_TRAPS
)
_ZERO = Decimal(0)
_ONE = Decimal(1)
_TWO = Decimal(2)


def _validate_decimal(value: Decimal) -> None:
    """Require finite, context-range inputs without rounding their coefficients."""
    if type(value) is not Decimal or not value.is_finite():
        raise ValueError("interval endpoints must be finite Decimal values")
    if value and not _EMIN <= value.adjusted() <= _EMAX:
        raise ValueError("decimal value is outside the supported exponent range")


def _point_value(value: str | int | Decimal) -> Decimal:
    if type(value) not in (str, int, Decimal):
        raise ValueError("use an exact decimal string, integer, or Decimal")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("invalid exact decimal constant") from error
    _validate_decimal(result)
    return result


def _neighbor(value: Decimal, *, upper: bool) -> Decimal:
    """Enlarge a nearest-rounded result without using ambient context settings."""
    result = _NEAREST.next_plus(value) if upper else _NEAREST.next_minus(value)
    _validate_decimal(result)
    return result


@dataclass(frozen=True)
class Interval:
    """A closed real interval with exact finite decimal endpoints."""

    lo: Decimal
    hi: Decimal

    def __post_init__(self) -> None:
        _validate_decimal(self.lo)
        _validate_decimal(self.hi)
        if self.lo > self.hi:
            raise ValueError("interval endpoints must be ordered")

    @classmethod
    def point(cls, value: str | int | Decimal) -> Interval:
        """Construct a singleton without applying context precision to the input."""
        exact = _point_value(value)
        return cls(exact, exact)

    @classmethod
    def from_json(cls, values: list[str]) -> Interval:
        """Read exactly two decimal strings; floats and coercions are forbidden."""
        if type(values) is not list or len(values) != 2:
            raise ValueError("an interval requires a two-element string list")
        if any(type(value) is not str for value in values):
            raise ValueError("serialized interval endpoints must be strings")
        return cls(_point_value(values[0]), _point_value(values[1]))

    def to_json(self) -> list[str]:
        """Preserve exact endpoint values in portable JSON-compatible form."""
        return [str(self.lo), str(self.hi)]

    def __add__(self, other: Interval) -> Interval:
        if not isinstance(other, Interval):
            return NotImplemented
        return Interval(_DOWN.add(self.lo, other.lo), _UP.add(self.hi, other.hi))

    def __sub__(self, other: Interval) -> Interval:
        if not isinstance(other, Interval):
            return NotImplemented
        return Interval(
            _DOWN.subtract(self.lo, other.hi), _UP.subtract(self.hi, other.lo)
        )

    def __neg__(self) -> Interval:
        return Interval(self.hi.copy_negate(), self.lo.copy_negate())

    def __mul__(self, other: Interval) -> Interval:
        if not isinstance(other, Interval):
            return NotImplemented
        products = [(x, y) for x in (self.lo, self.hi) for y in (other.lo, other.hi)]
        return Interval(
            min(_DOWN.multiply(x, y) for x, y in products),
            max(_UP.multiply(x, y) for x, y in products),
        )

    def __truediv__(self, other: Interval) -> Interval:
        if not isinstance(other, Interval):
            return NotImplemented
        if other.lo <= 0 <= other.hi:
            raise ValueError("division by an interval containing zero")
        ratios = [(x, y) for x in (self.lo, self.hi) for y in (other.lo, other.hi)]
        return Interval(
            min(_DOWN.divide(x, y) for x, y in ratios),
            max(_UP.divide(x, y) for x, y in ratios),
        )

    def square(self) -> Interval:
        """Enclose the square while retaining the common-operand dependency."""
        upper = max(_UP.multiply(x, x) for x in (self.lo, self.hi))
        if self.lo <= 0 <= self.hi:
            lower = _ZERO
        else:
            lower = min(_DOWN.multiply(x, x) for x in (self.lo, self.hi))
        return Interval(lower, upper)

    def sqrt(self) -> Interval:
        """Enclose monotone square roots, expanding HALF_EVEN-rounded values."""
        if self.lo < 0:
            raise ValueError("square root requires a nonnegative interval")
        lower = (
            _ZERO if self.lo == 0 else _neighbor(_NEAREST.sqrt(self.lo), upper=False)
        )
        upper = _ZERO if self.hi == 0 else _neighbor(_NEAREST.sqrt(self.hi), upper=True)
        return Interval(lower, upper)

    def exp(self) -> Interval:
        """Enclose exponentials; exp ignores FLOOR/CEILING context rounding."""
        lower = _ONE if self.lo == 0 else _neighbor(_NEAREST.exp(self.lo), upper=False)
        upper = _ONE if self.hi == 0 else _neighbor(_NEAREST.exp(self.hi), upper=True)
        return Interval(lower, upper)

    def midpoint(self) -> Decimal:
        """Return a representable point inside this interval, not a bound."""
        middle = _NEAREST.divide(_NEAREST.add(self.lo, self.hi), _TWO)
        return min(self.hi, max(self.lo, middle))


def _kernel_point(s: Decimal, time: Decimal) -> Interval:
    rate, duration = Interval.point(s), Interval.point(time)
    numerator = (-(rate * duration)).exp() - (-duration).exp()
    return numerator / (Interval.point(1) - rate)


def kernel(s: Interval, time: Decimal) -> Interval:
    """Enclose F_(1,s)(t) for s in [0,0.6] using its monotonicity in s.

    The denominator 1-s stays strictly positive. The integral representation
    proves nonnegativity and the upper bound t, including when subtraction loses
    accuracy for very small t. No finite sampling is used to bound a rate cell.
    """
    if not isinstance(s, Interval):
        raise ValueError("rate must be an Interval")
    _validate_decimal(time)
    if s.lo < 0 or s.hi > Decimal("0.6") or time < 0:
        raise ValueError("kernel requires s within [0,0.6] and nonnegative time")
    if time == 0:
        return Interval.point(0)
    low = _kernel_point(s.hi, time).lo
    high = _kernel_point(s.lo, time).hi
    return Interval(max(_ZERO, low), min(time, high))
