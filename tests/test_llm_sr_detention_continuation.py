"""The continuation chart preserves feasible families and evidence boundaries."""

from copy import deepcopy
from fractions import Fraction
from itertools import product
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


def test_reduced_chart_roundtrips_random_original_points_and_contains_boundaries():
    value = chart()
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


@pytest.mark.parametrize("g,k", [(0, 0), (0, 2), (-0.25, 0), (0.25, 0), (-0.25, 0.06)])
def test_zero_and_signed_products_do_not_remove_feasible_points(g, k):
    value = chart()
    physical = value.anchor | {"k_gw": g, "kappa": k}
    recovered = value.decode(value.encode(physical))
    np.testing.assert_allclose(
        list(combinations(recovered).values()),
        list(combinations(physical).values()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_collapsed_conditional_intervals_still_have_a_finite_chart():
    item, context, training = example(15)
    physical = original.domain_policy(item, context, training)
    physical["bounds"] = {n: [v, v] for n, v in item.parameters.items()}
    value = kernel.CoordinateChart(physical, item.parameters, "reduced_model15")
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


def test_chart_roundtrip_preserves_original_piecewise_rhs():
    item, context, _ = example(15)
    model, value = compile_current_value(item, context), chart()
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


def test_exact_chart_handles_nonzero_kappa_lower_and_fixed_zero_product():
    value = chart(bounds={"kappa": [0.05, 2]})
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
    zero = kernel.CoordinateChart(physical, point, "reduced_model15")
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
