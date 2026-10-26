"""The continuation chart preserves feasible families and evidence boundaries."""

from copy import deepcopy
from fractions import Fraction
from itertools import pairwise, product
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.fitting import llm_sr_detention_continuation as kernel
from autoformalism.fitting import llm_sr_detention_recovery as previous
from autoformalism.fitting import llm_sr_detention_reduction as reduction
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.llm_sr_instantaneous import compile_current_value
from tests.test_llm_sr_detention_refit import example


def chart(task=15, mode="reduced_model15", bounds=None):
    item, context, training = example(task)
    domain = original.domain_policy(item, context, training)
    if bounds:
        domain["bounds"] |= bounds
    return kernel.CoordinateChart(domain, item.parameters, mode)


def combinations(parameters):
    return {k: float(v) for k, v in reduction.reduce_parameters(parameters).items()}


@pytest.mark.parametrize("mode", kernel.REDUCED_MODES)
def test_reduced_chart_roundtrips_random_original_points_and_contains_boundaries(mode):
    value = chart(mode=mode)
    random = np.random.default_rng(12)
    for _ in range(100):
        original_point = {n: random.uniform(*pair) for n, pair in value.bounds.items()}
        encoded = value.encode(original_point)
        recovered = value.decode(encoded)
        for name, result in recovered.items():
            assert value.bounds[name][0] <= result <= value.bounds[name][1]
        np.testing.assert_allclose(
            list(combinations(recovered).values()),
            list(combinations(original_point).values()),
            rtol=1e-12,
            atol=1e-10,
        )
    for sides in product((0, 1), repeat=len(value.names)):
        encoded = np.asarray(
            [
                value.domain["bounds"][n][side]
                for n, side in zip(value.names, sides, strict=True)
            ]
        )
        result = value.decode(encoded)
        assert all(
            value.bounds[n][0] <= v <= value.bounds[n][1] for n, v in result.items()
        )
    assert value.decode(value.anchor_vector) == value.anchor
    assert value.domain["physical_family_preserved"]
    assert not value.domain["physical_combinations_are_independent_box"]


@pytest.mark.parametrize("mode", kernel.REDUCED_MODES)
@pytest.mark.parametrize("g,k", [(0, 0), (0, 2), (-0.25, 0), (0.25, 0), (-0.25, 0.06)])
def test_zero_and_signed_products_do_not_remove_feasible_points(g, k, mode):
    value = chart(mode=mode)
    physical = value.anchor | {"k_gw": g, "kappa": k}
    recovered = value.decode(value.encode(physical))
    np.testing.assert_allclose(
        list(combinations(recovered).values()),
        list(combinations(physical).values()),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("mode", kernel.REDUCED_MODES)
def test_collapsed_conditional_intervals_still_have_a_finite_chart(mode):
    item, context, training = example(15)
    physical = original.domain_policy(item, context, training)
    physical["bounds"] = {n: [v, v] for n, v in item.parameters.items()}
    value = kernel.CoordinateChart(physical, item.parameters, mode)
    for x in (0, 0.5, 1):
        point = value.anchor_vector.copy()
        point[-3:] = x
        assert value.decode(point) == item.parameters
    assert np.isfinite(value.units).all()


def test_fraction_scales_use_physical_units_not_arbitrary_fraction_origin():
    item, context, training = example(15)
    oracle = kernel.ChartOracle(
        item,
        context,
        training,
        item.parameters,
        "reduced_model15",
        kernel.Policy(),
        "tight",
    )
    first = oracle.vector(item.parameters)
    second = first.copy()
    second[-3:] = [0.99, 0.99, 0.99]
    np.testing.assert_equal(
        oracle.local_coordinate_scales(first)[-3:],
        oracle.local_coordinate_scales(second)[-3:],
    )
    assert np.all(oracle.units[-3:] < 1)


def test_nearby_anchor_lifts_do_not_introduce_material_discontinuities():
    value = chart()
    original_combinations = combinations(value.anchor)
    for i in range(len(value.names)):
        for sign in (-1, 1):
            point = value.anchor_vector.copy()
            point[i] = np.nextafter(point[i], np.inf if sign > 0 else -np.inf)
            recovered = value.decode(point)
            np.testing.assert_allclose(
                list(combinations(recovered).values()),
                list(original_combinations.values()),
                rtol=1e-12,
                atol=1e-12,
            )


def test_zero_product_neighbors_do_not_move_capacity_combination():
    value = chart()
    physical = value.anchor | {"k_gw": 0, "kappa": 0}
    middle = value.encode(physical)
    values = []
    index = value.names.index("d_fraction")
    for direction in (-np.inf, np.inf):
        point = middle.copy()
        point[index] = np.nextafter(point[index], direction)
        values.append(combinations(value.decode(point)))
    assert abs(values[0]["d"]) < 1e-11
    assert abs(values[1]["d"]) < 1e-11
    assert abs(values[0]["M"] - physical["h_max"]) < 1e-11
    assert values[0]["M"] == values[1]["M"]


@pytest.mark.parametrize("mode", kernel.REDUCED_MODES)
def test_chart_roundtrip_preserves_original_piecewise_rhs(mode):
    item, context, _ = example(15)
    model, value = compile_current_value(item, context), chart(mode=mode)
    random = np.random.default_rng(9)
    for _ in range(12):
        before = {n: random.uniform(*pair) for n, pair in value.bounds.items()}
        after = value.decode(value.encode(before))
        for h in (-0.1, 0, 0.2, 1, 10):
            for rate in (0, 0.1, 10):
                covariates = {
                    "inflow_down": rate * 1200,
                    "area_down": 1200,
                    "crest_down": 0.2,
                    "warning_depth": 0.6,
                }
                forcing = SimpleNamespace(value=lambda n, t, v=covariates: v[n])
                np.testing.assert_allclose(
                    model.rhs(0, [h], before, forcing),
                    model.rhs(0, [h], after, forcing),
                    rtol=2e-12,
                    atol=1e-10,
                )


def test_temporary_subset_keeps_hmax_fixed_and_rejects_wrong_task_or_point():
    value = chart(14, "subset_model14")
    assert len(value.names) == 7 and "h_max" not in value.names
    point = value.anchor_vector.copy()
    point[0] += 0.01
    assert value.decode(point)["h_max"] == value.anchor["h_max"]
    assert not value.domain["physical_family_preserved"]
    with pytest.raises(ValueError, match="fixed h_max"):
        value.encode(value.anchor | {"h_max": value.anchor["h_max"] + 1})
    with pytest.raises(ValueError, match="requires Model 15"):
        chart(14, "reduced_model15")
    point[0] = float("nan")
    with pytest.raises(ValueError, match="chart bounds"):
        value.decode(point)


def analytic_decay(monkeypatch):
    def residual(oracle, point, deadline, method="Radau"):
        beta = oracle.parameters(point)["beta"]
        return np.concatenate(
            [
                (
                    row.targets["h_down"][0] * np.exp(-beta * np.asarray(row.time))
                    - np.asarray(row.targets["h_down"])
                )
                / oracle.scale
                for row in oracle.split.trajectories
            ]
        )

    monkeypatch.setattr(original.DetentionOracle, "residual", residual)


def records(item, context, training, parameters, mode):
    return [
        kernel.direction(item, context, training, parameters, i, mode=mode, seconds=10)
        for i in range(
            len(kernel.domain(item, context, training, parameters, mode)["names"])
        )
    ]


def test_original_and_subset_audit_actual_optimizer_columns(monkeypatch):
    analytic_decay(monkeypatch)
    item, context, training = example(14)
    point = item.parameters | {"beta": 0.3}
    for mode in ("original", "subset_model14"):
        rows = records(item, context, training, point, mode)
        gate = kernel.qualification(item, context, training, point, rows, mode=mode)
        assert gate["complete"] and gate["search_qualified"]
        kernel.validate_qualification(
            item, context, training, point, rows, mode=mode, qualification=gate
        )
        fitted = kernel.fit(
            item, context, training, point, mode=mode, qualification=gate, seconds=10
        )
        assert fitted["best"]["training_nmse"] < 1e-8
        assert fitted["best"]["parameters"]["h_max"] == point["h_max"]
        assert not fitted["full_model_convergence_claimed"]


def test_saved_physical_directions_import_without_rollouts(monkeypatch):
    analytic_decay(monkeypatch)
    item, context, training = example(14)
    point = item.parameters | {"beta": 0.3}
    old = [
        previous.direction(
            item,
            context,
            training,
            point,
            i,
            purpose="diagnosis",
            precision="tight",
            seconds=10,
        )
        for i in range(8)
    ]
    monkeypatch.setattr(
        original.DetentionOracle,
        "residual",
        lambda *a, **k: pytest.fail("import performed rollout"),
    )
    for mode, count in (("original", 8), ("subset_model14", 7)):
        rows = kernel.import_original_directions(
            item, context, training, point, old, mode=mode
        )
        assert len(rows) == count
        gate = kernel.qualification(item, context, training, point, rows, mode=mode)
        assert gate["search_qualified"]
        assert all(x["imported_evidence"]["no_new_rollout"] for x in rows)
    with pytest.raises(ValueError, match="fresh derivative"):
        kernel.import_original_directions(
            item, context, training, point, old, mode="reduced_model15"
        )
    tampered = deepcopy(old)
    tampered[0]["jacobian_column"][1] += 1
    with pytest.raises(ValueError, match="digest"):
        kernel.import_original_directions(
            item, context, training, point, tampered, mode="original"
        )


def test_chart_gate_rejects_tampering_mixed_endpoint_and_missing_columns(monkeypatch):
    analytic_decay(monkeypatch)
    item, context, training = example(14)
    point = item.parameters | {"beta": 0.3}
    rows = records(item, context, training, point, "subset_model14")
    assert not kernel.qualification(
        item, context, training, point, rows[:-1], mode="subset_model14"
    )["search_qualified"]
    with pytest.raises(ValueError, match=r"another endpoint|original endpoint"):
        kernel.qualification(
            item, context, training, item.parameters, rows, mode="subset_model14"
        )
    with pytest.raises(ValueError, match="duplicate"):
        kernel.qualification(
            item, context, training, point, [*rows, rows[0]], mode="subset_model14"
        )
    gate = kernel.qualification(
        item, context, training, point, rows, mode="subset_model14"
    )
    forged = deepcopy(gate)
    forged["search_qualified"] = False
    forged["qualification_sha256"] = public.content_sha256(
        {k: v for k, v in forged.items() if k != "qualification_sha256"}
    )
    with pytest.raises(ValueError, match="reconstructed evidence"):
        kernel.fit(
            item,
            context,
            training,
            point,
            mode="subset_model14",
            qualification=forged,
            seconds=10,
        )


def test_reduced_oracle_residual_matches_original_and_full_chart_columns(monkeypatch):
    item, context, training = example(15)

    def residual(oracle, point, deadline, method="Radau"):
        reduced = combinations(oracle.parameters(point))
        return np.asarray([reduced[n] for n in reduction.REDUCED_NAMES])

    monkeypatch.setattr(original.DetentionOracle, "residual", residual)
    oracle = kernel.ChartOracle(
        item,
        context,
        training,
        item.parameters,
        "reduced_model15",
        kernel.Policy(),
        "tight",
    )
    point = oracle.vector(item.parameters)
    r, j = oracle.evaluate(point, monotonic() + 10)
    assert j.shape == (8, 8)
    np.testing.assert_allclose(
        r, list(combinations(item.parameters).values()), rtol=0, atol=0
    )
    for i in range(8):
        record = kernel.direction(
            item,
            context,
            training,
            item.parameters,
            i,
            mode="reduced_model15",
            seconds=10,
        )
        np.testing.assert_allclose(record["jacobian_column"], j[:, i], rtol=0, atol=0)


@pytest.mark.parametrize("mode", kernel.REDUCED_MODES)
def test_exact_chart_handles_nonzero_kappa_lower_and_fixed_zero_product(mode):
    value = chart(mode=mode, bounds={"kappa": [0.05, 2]})
    for g in (-0.5, 0, 0.5):
        physical = value.anchor | {"k_gw": g}
        recovered = value.decode(value.encode(physical))
        np.testing.assert_allclose(
            list(combinations(recovered).values()),
            list(combinations(physical).values()),
            rtol=1e-12,
            atol=1e-12,
        )
    item, context, training = example(15)
    point = item.parameters | {"kappa": 0}
    physical = original.domain_policy(item, context, training)
    physical["bounds"]["kappa"] = [0, 0]
    zero = kernel.CoordinateChart(physical, point, mode)
    p = zero.anchor_vector.copy()
    p[-2] = 0.9
    assert reduction.reduce_parameters(zero.decode(p))["d"] == Fraction(0)


def test_reduced_probe_policy_is_bound_and_does_not_relax_tolerances(monkeypatch):
    base = kernel.Policy()
    effective = kernel.effective_policy(base, mode="reduced_model15")
    assert effective.numerical.derivative.local_relative_step_cap == 1e-5
    assert effective.tight_fd_relative_step == base.tight_fd_relative_step
    assert effective.numerical.derivative.model_dump(
        exclude={"local_relative_step_cap"}
    ) == base.numerical.derivative.model_dump(exclude={"local_relative_step_cap"})
    assert kernel.effective_policy(base, mode="original") == base
    assert kernel.effective_policy(base, mode="subset_model14") == base
    assert kernel.effective_policy(effective, mode="reduced_model15") == effective
    assert kernel.effective_policy(base, mode="reduced_model15_consistent") == effective

    item, context, training = example(15)

    def residual(oracle, point, deadline, method="Radau"):
        value = oracle.parameters(point)["C"]
        return np.asarray([value, 2 * value, 3 * value])

    monkeypatch.setattr(original.DetentionOracle, "residual", residual)
    row = kernel.direction(
        item, context, training, item.parameters, 0, mode="reduced_model15", seconds=5
    )
    assert row["derivative"]["policy"]["local_relative_step_cap"] == 1e-5
    assert row["identity"]["policy_sha256"] == public.content_sha256(effective)
    gate = kernel.qualification(
        item, context, training, item.parameters, [row], mode="reduced_model15"
    )
    kernel.validate_qualification(
        item,
        context,
        training,
        item.parameters,
        [row],
        mode="reduced_model15",
        qualification=gate,
    )
    forged = deepcopy(row)
    forged["derivative"]["policy"]["local_relative_step_cap"] = 1e-4
    forged["evidence_sha256"] = public.content_sha256(
        {k: v for k, v in forged.items() if k != "evidence_sha256"}
    )
    with pytest.raises(ValueError, match="policy differ"):
        kernel.qualification(
            item, context, training, item.parameters, [forged], mode="reduced_model15"
        )


@pytest.mark.parametrize(
    "physical",
    [
        Fraction(0),
        Fraction(1),
        Fraction(1, 3),
        Fraction(1, 10**400),
        1 - Fraction(1, 10**400),
    ],
)
def test_consistent_fraction_map_keeps_anchor_endpoints_and_exact_inverse(physical):
    source, target = kernel._fraction_anchor(physical)
    assert kernel._map_fraction(source, source, target) == physical
    if 0 < physical < 1:
        assert 0 < float(source) < 1
    points = sorted({Fraction(0), Fraction(1, 4), source, Fraction(3, 4), Fraction(1)})
    mapped = [kernel._map_fraction(value, source, target) for value in points]
    assert mapped[0] == 0 and mapped[-1] == 1
    assert all(a < b for a, b in pairwise(mapped))
    assert [kernel._map_fraction(value, target, source) for value in mapped] == points
    with pytest.raises(ValueError, match="outside"):
        kernel._map_fraction(Fraction(-1), source, target)


@pytest.mark.parametrize("name", ["C", "beta", "eps_s", "Q_max"])
def test_consistent_chart_changes_only_requested_independent_parameter_bitwise(name):
    value = chart(mode="reduced_model15_consistent")
    index = value.names.index(name)
    anchor = value.anchor_vector[index]
    for proposed in (
        value.lower[index],
        value.upper[index],
        np.nextafter(anchor, value.lower[index]),
        np.nextafter(anchor, value.upper[index]),
        anchor * (1 - 1e-5),
        anchor * (1 + 1e-5),
    ):
        point = value.anchor_vector.copy()
        point[index] = proposed
        assert value.decode(point) == value.anchor | {name: proposed}
    assert not value.domain["exact_anchor_only_shortcut"]
    assert value.domain["fraction_mapping"] == "exact-anchor-odds-ratio-1"
    assert np.isfinite(value.units).all() and np.all(value.units > 0)


@pytest.mark.parametrize("gain,kappa", [(0, 0), (0, 2), (-0.25, 0), (-0.25, 0.06)])
def test_consistent_zero_and_signed_anchor_neighbors_preserve_original_fibre(
    gain, kappa
):
    item, context, training = example(15)
    physical = original.domain_policy(item, context, training)
    anchor = item.parameters | {"k_gw": gain, "kappa": kappa}
    value = kernel.CoordinateChart(physical, anchor, "reduced_model15_consistent")
    for name in ("C", "beta", "eps_s", "Q_max"):
        point = value.anchor_vector.copy()
        index = value.names.index(name)
        point[index] = np.nextafter(point[index], np.inf)
        assert value.decode(point) == anchor | {name: point[index]}


def test_consistent_chart_rejects_unusable_coordinate_units():
    item, context, training = example(15)
    physical = original.domain_policy(item, context, training)
    physical["units"]["C"] = 0
    with pytest.raises(ValueError, match="finite positive coordinate units"):
        kernel.CoordinateChart(physical, item.parameters, "reduced_model15_consistent")


def test_consistent_anchor_and_neighbors_match_actual_guarded_rhs_exactly():
    item, context, _ = example(15)
    model, value = compile_current_value(item, context), chart(
        mode="reduced_model15_consistent"
    )
    for name in ("C", "beta", "eps_s", "Q_max"):
        index = value.names.index(name)
        for sign in (-np.inf, np.inf):
            vector = value.anchor_vector.copy()
            vector[index] = np.nextafter(vector[index], sign)
            expected = value.anchor | {name: vector[index]}
            decoded = value.decode(vector)
            for depth in (-0.1, 0, 0.2, 1, 10):
                for rate in (0, 0.1, 10):
                    data = {
                        "inflow_down": rate * 1200,
                        "area_down": 1200,
                        "crest_down": 0.2,
                        "warning_depth": 0.6,
                    }
                    forcing = SimpleNamespace(value=lambda n, t, v=data: v[n])
                    np.testing.assert_array_equal(
                        model.rhs(0, [depth], decoded, forcing),
                        model.rhs(0, [depth], expected, forcing),
                    )


def test_consistent_inactive_column_has_no_artificial_one_sided_offset(monkeypatch):
    item, context, training = example(15)

    def residual(oracle, point, deadline, method="Radau"):
        parameters = oracle.parameters(point)
        # Q_max is absent, while tiny unwanted changes in other coefficients
        # are deliberately visible to the derivative check.
        return np.asarray(
            [
                1e8 * (parameters[name] - item.parameters[name])
                for name in ("k_loss", "kappa")
            ]
        )

    monkeypatch.setattr(original.DetentionOracle, "residual", residual)
    row = kernel.direction(
        item,
        context,
        training,
        item.parameters,
        3,
        mode="reduced_model15_consistent",
        seconds=10,
    )
    checked = row["derivative"]["directions"][0]
    assert row["jacobian_column"] == [0.0, 0.0]
    assert all(value == 0 for value in checked["one_sided_disagreement"])
    assert checked["precision_status"] != "failed"


def test_new_chart_is_opt_in_and_cannot_reuse_legacy_chart_receipts(monkeypatch):
    analytic_decay(monkeypatch)
    item, context, training = example(15)
    old = chart()
    new = chart(mode="reduced_model15_consistent")
    assert old.domain["protocol"] == "detention-preserved-family-chart-1"
    assert "fraction_maps" not in old.domain
    assert new.domain["protocol"] == "detention-preserved-family-chart-2"
    row = kernel.direction(
        item, context, training, item.parameters, 0, mode="reduced_model15", seconds=10
    )
    with pytest.raises(ValueError, match="contract"):
        kernel.qualification(
            item,
            context,
            training,
            item.parameters,
            [row],
            mode="reduced_model15_consistent",
        )
