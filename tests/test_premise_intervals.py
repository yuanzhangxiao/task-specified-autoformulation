"""Arithmetic containment and domain checks for the bounded Decimal fixture."""

from dataclasses import FrozenInstanceError
from decimal import (
    ROUND_DOWN,
    ROUND_UP,
    Context,
    Decimal,
    DecimalException,
    localcontext,
)
from fractions import Fraction
from itertools import product

import pytest

from autoformalism.research.premise_intervals import PRECISION, Interval, kernel


def interval(lo: str, hi: str) -> Interval:
    return Interval(Decimal(lo), Decimal(hi))


def contains_exact(result: Interval, value: Fraction) -> bool:
    return Fraction(result.lo) <= value <= Fraction(result.hi)


@pytest.mark.parametrize("left", ["-3.17", "-0.00000001", "0", "0.3", "23.17"])
@pytest.mark.parametrize("right", ["-7.13", "-0.7", "0.2", "3.11"])
def test_point_arithmetic_contains_exact_rational_results(left, right):
    a, b = Interval.point(left), Interval.point(right)
    x, y = Fraction(left), Fraction(right)
    assert contains_exact(a + b, x + y)
    assert contains_exact(a - b, x - y)
    assert contains_exact(a * b, x * y)
    assert contains_exact(a / b, x / y)


@pytest.mark.parametrize(
    "left,right",
    [
        (("-2.3", "4.7"), ("-3.8", "-0.1")),
        (("0.001", "3.2"), ("0.1", "2.7")),
        (("-5.7", "-0.2"), ("0.11", "4.9")),
    ],
)
def test_interval_arithmetic_contains_sampled_exact_rational_pairs(left, right):
    a, b = interval(*left), interval(*right)
    xs = [Fraction(a.lo), Fraction(a.hi), Fraction(a.lo + a.hi) / 2]
    ys = [Fraction(b.lo), Fraction(b.hi), Fraction(b.lo + b.hi) / 2]
    for x, y in product(xs, ys):
        assert contains_exact(a + b, x + y)
        assert contains_exact(a - b, x - y)
        assert contains_exact(a * b, x * y)
        assert contains_exact(a / b, x / y)


@pytest.mark.parametrize("bounds", [("-3", "2"), ("1.1", "2.9"), ("-3", "-1.1")])
def test_square_preserves_dependency_and_contains_exact_squares(bounds):
    source = interval(*bounds)
    result = source.square()
    for value in (source.lo, source.hi, source.midpoint()):
        assert contains_exact(result, Fraction(value) ** 2)
    if source.lo <= 0 <= source.hi:
        assert result.lo == 0
        assert result.lo > (source * source).lo


@pytest.mark.parametrize("value", ["-100", "-0.1", "0", "0.1", "1", "13.17"])
def test_exp_contains_independent_high_precision_evaluation(value):
    x = Decimal(value)
    reference = Context(prec=150).exp(x)
    result = Interval.point(x).exp()
    assert result.lo <= reference <= result.hi
    assert result.lo > 0
    if x != 0:
        rounded = Context(prec=PRECISION).exp(x)
        assert result.lo < rounded < result.hi


@pytest.mark.parametrize("value", ["0", "0.01", "2", "4", "100.1", "1e-100"])
def test_sqrt_contains_high_precision_and_exact_squared_bounds(value):
    x = Decimal(value)
    reference = Context(prec=150).sqrt(x)
    result = Interval.point(x).sqrt()
    assert result.lo <= reference <= result.hi
    assert Fraction(result.lo) ** 2 <= Fraction(x)
    assert Fraction(result.hi) ** 2 >= Fraction(x)
    assert result.lo >= 0


def test_monotone_exp_and_sqrt_bound_interval_endpoints():
    source = interval("0.2", "3.1")
    for value in (source.lo, source.midpoint(), source.hi):
        ctx = Context(prec=150)
        assert source.exp().lo <= ctx.exp(value) <= source.exp().hi
        assert source.sqrt().lo <= ctx.sqrt(value) <= source.sqrt().hi
    assert Interval.point(0).sqrt() == Interval.point(0)
    assert Interval.point(0).exp() == Interval.point(1)


def test_operations_and_sign_reversal_ignore_ambient_context():
    a = interval("0.123456789123456789", "0.987654321987654321")
    b = interval("0.112233445566778899", "0.998877665544332211")

    def operations():
        return (a + b, a - b, a * b, a / b, -a, a.exp(), a.sqrt(), a.midpoint())

    expected = operations()
    for rounding in (ROUND_UP, ROUND_DOWN):
        with localcontext(Context(prec=3, rounding=rounding)):
            assert operations() == expected
            assert kernel(interval("0.1", "0.3"), Decimal("5")) == expected_kernel()


def expected_kernel() -> Interval:
    """A separate explicit context guards the test's own reference setup."""
    with localcontext(Context(prec=150)):
        return kernel(interval("0.1", "0.3"), Decimal("5"))


def test_exact_point_and_json_round_trip_preserve_digits_and_are_immutable():
    text = "1.12345678901234567890123456789012345678901234567890123456789"
    source = Interval.point(text)
    assert source.lo == Decimal(text)
    assert source.to_json() == [text, text]
    assert Interval.from_json(source.to_json()) == source
    with pytest.raises(FrozenInstanceError):
        source.lo = Decimal(0)


@pytest.mark.parametrize("value", [True, False, 0.1, None, "NaN", "Infinity", "oops"])
def test_point_rejects_inexact_types_and_nonfinite_constants(value):
    with pytest.raises(ValueError):
        Interval.point(value)


@pytest.mark.parametrize(
    "lo,hi",
    [(1, Decimal(2)), (Decimal(2), Decimal(1)), (Decimal("NaN"), Decimal(2))],
)
def test_interval_constructor_rejects_bad_endpoints(lo, hi):
    with pytest.raises(ValueError):
        Interval(lo, hi)


@pytest.mark.parametrize("values", [("0", "1"), ["0"], ["0", "1", "2"], [0, "1"]])
def test_json_decoder_requires_exact_two_string_schema(values):
    with pytest.raises(ValueError):
        Interval.from_json(values)


def test_invalid_operations_fail_closed():
    with pytest.raises(ValueError, match="zero"):
        Interval.point(1) / interval("-1", "1")
    with pytest.raises(ValueError, match="nonnegative"):
        interval("-1", "1").sqrt()
    with pytest.raises(TypeError):
        Interval.point(1) + 1
    with pytest.raises(ValueError, match="exponent"):
        Interval.point("1e1000000")
    with pytest.raises(ValueError, match="exponent"):
        Interval.point("1e-1000000")
    with pytest.raises(DecimalException):
        Interval.point("1e999999") * Interval.point(100)
    with pytest.raises(DecimalException):
        Interval.point("1e-999999") / Interval.point(100)
    with pytest.raises(DecimalException):
        Interval.point("-10000000").exp()


@pytest.mark.parametrize("time", ["0", "0.001", "1", "5", "20", "1e-100"])
@pytest.mark.parametrize("bounds", [("0", "0.6"), ("0", "0"), ("0.2", "0.3")])
def test_kernel_encloses_high_precision_values_and_physical_bounds(time, bounds):
    duration = Decimal(time)
    rates = interval(*bounds)
    result = kernel(rates, duration)
    assert 0 <= result.lo <= result.hi <= duration
    ctx = Context(prec=180)
    for s in (rates.lo, rates.midpoint(), rates.hi):
        first = ctx.exp(ctx.minus(ctx.multiply(s, duration)))
        second = ctx.exp(ctx.minus(duration))
        expected = ctx.divide(ctx.subtract(first, second), ctx.subtract(Decimal(1), s))
        assert result.lo <= expected <= result.hi


def test_kernel_monotonic_endpoints_are_tighter_than_sampling_an_entire_box():
    time = Decimal(5)
    low_rate, high_rate = (
        kernel(Interval.point(0), time),
        kernel(Interval.point("0.6"), time),
    )
    combined = kernel(interval("0", "0.6"), time)
    assert high_rate.hi < low_rate.lo
    assert combined.lo == high_rate.lo
    assert combined.hi == low_rate.hi


@pytest.mark.parametrize(
    "rates,time",
    [
        (interval("-0.1", "0.1"), Decimal(1)),
        (interval("0", "0.61"), Decimal(1)),
        (Interval.point(1), Decimal(1)),
        (Interval.point(0), Decimal(-1)),
        (Interval.point(0), 1.0),
        (Interval.point(0), Decimal("NaN")),
    ],
)
def test_kernel_rejects_unsupported_domains(rates, time):
    with pytest.raises(ValueError):
        kernel(rates, time)
