"""Coordinate recovery conserves identities, numerical gates and search budgets."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import llm_sr_detention_recovery as kernel
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import public_fitting as public
from tests.test_llm_sr_detention_refit import example


def rows(item, context, training, point, *, purpose="search", precision="standard"):
    return [
        kernel.direction(
            item,
            context,
            training,
            point,
            i,
            purpose=purpose,
            precision=precision,
            seconds=10,
        )
        for i in range(len(point))
    ]


def test_one_column_matches_original_but_avoids_full_jacobian():
    item, context, training = example()
    oracle = original.DetentionOracle(item, context, training)
    vector = oracle.vector(item.parameters)
    residual, jacobian = oracle.evaluate(vector, monotonic() + 10)
    i = oracle.names.index("beta")
    result = kernel.direction(item, context, training, item.parameters, i, seconds=10)
    assert result["status"] == "complete"
    np.testing.assert_allclose(result["base_residual"], residual, rtol=0, atol=0)
    np.testing.assert_allclose(
        result["jacobian_column"], jacobian[:, i], rtol=0, atol=0
    )
    assert result["numerical_cost"]["jacobian_calls"] == 0
    assert result["numerical_cost"]["residual_calls_started"] < 17
    assert result["derivative"]["checked_indices"] == [0]
    assert not result["parameter_information_qualified"]


def test_qualification_requires_all_coordinates_exact_endpoint_and_precision():
    item, context, training = example()
    point = item.parameters | {"beta": 0.3}
    records = rows(item, context, training, point)
    result = kernel.qualification(item, context, training, point, records)
    assert result["complete"] and result["search_qualified"]
    assert not result["parameter_information_qualified"]
    kernel.validate_qualification(
        item, context, training, point, records, qualification=result
    )
    portable = deepcopy(result)
    beta_row = next(row for row in portable["rows"] if row["name"] == "beta")
    beta_row["gradient"] = float(np.nextafter(beta_row["gradient"], np.inf))
    beta_row["search_error_limit"] = max(1e-7, 0.01 * abs(beta_row["gradient"]))
    portable["qualification_sha256"] = public.content_sha256(
        {k: v for k, v in portable.items() if k != "qualification_sha256"}
    )
    kernel.validate_qualification(
        item, context, training, point, records, qualification=portable
    )
    partial = kernel.qualification(item, context, training, point, records[:-1])
    assert not partial["search_qualified"]
    assert partial["missing_indices"] == [7]
    with pytest.raises(ValueError, match="another endpoint"):
        kernel.qualification(item, context, training, item.parameters, records)
    with pytest.raises(ValueError, match="another endpoint"):
        kernel.qualification(item, context, training, point, records, precision="tight")
    with pytest.raises(ValueError, match="duplicate"):
        kernel.qualification(item, context, training, point, [*records, records[0]])
    changed = deepcopy(records)
    changed[0]["jacobian_column"][0] += 1
    with pytest.raises(ValueError, match="digest"):
        kernel.qualification(item, context, training, point, changed)
    changed[0]["evidence_sha256"] = public.content_sha256(
        {k: v for k, v in changed[0].items() if k != "evidence_sha256"}
    )
    with pytest.raises(ValueError, match="arrays"):
        kernel.qualification(item, context, training, point, changed)


def test_search_inconclusive_zero_column_can_be_useful_but_not_stationary():
    item, context, training = example()
    records = rows(item, context, training, item.parameters | {"beta": 0.3})
    changed = deepcopy(records)
    audit = changed[0]["derivative"]["directions"][0]
    audit["precision_status"] = "inconclusive"
    audit["gradient_error_bound"] = 4e-9
    audit["projected_gradient_error_bound"] = 4e-9
    changed[0]["evidence_sha256"] = public.content_sha256(
        {k: v for k, v in changed[0].items() if k != "evidence_sha256"}
    )
    result = kernel.qualification(
        item, context, training, item.parameters | {"beta": 0.3}, changed
    )
    assert result["search_qualified"]
    assert not result["strict_gradient_qualified"]
    audit["precision_status"] = "failed"
    changed[0]["evidence_sha256"] = public.content_sha256(
        {k: v for k, v in changed[0].items() if k != "evidence_sha256"}
    )
    result = kernel.qualification(
        item, context, training, item.parameters | {"beta": 0.3}, changed
    )
    assert not result["search_qualified"] and result["failed_indices"] == [0]


def test_search_uses_qualified_settings_and_retains_improvement(monkeypatch):
    item, context, training = example()

    def analytic(oracle, point, deadline, method="Radau"):
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

    monkeypatch.setattr(original.DetentionOracle, "residual", analytic)
    point = item.parameters | {"beta": 0.3}
    records = rows(item, context, training, point, precision="tight")
    gate = kernel.qualification(
        item, context, training, point, records, precision="tight"
    )
    original_search = kernel.recovery.search
    observed = []

    def search(oracle, *args, **kwargs):
        observed.append(oracle.settings.relative_tolerance)
        assert (
            oracle.policy.finite_difference_relative_step
            == kernel.Policy().tight_fd_relative_step
        )
        return original_search(oracle, *args, **kwargs)

    monkeypatch.setattr(kernel.recovery, "search", search)
    result = kernel.fit(item, context, training, point, qualification=gate, seconds=15)
    assert result["best"]["training_nmse"] < 1e-8
    assert observed == [1e-11]
    assert not result["global_convergence_certified"]
    assert result["qualification_sha256"] == gate["qualification_sha256"]
    forged = deepcopy(gate)
    forged["search_qualified"] = False
    forged["qualification_sha256"] = public.content_sha256(
        {k: v for k, v in forged.items() if k != "qualification_sha256"}
    )
    with pytest.raises(ValueError, match="reconstructed"):
        kernel.fit(item, context, training, point, qualification=forged, seconds=15)


def test_final_diagnosis_is_pure_and_does_not_claim_all_coefficients(monkeypatch):
    item, context, training = example()
    records = rows(
        item, context, training, item.parameters, purpose="diagnosis", precision="tight"
    )
    verified = original.verify(item, context, training, item.parameters, seconds=10)
    monkeypatch.setattr(
        original.DetentionOracle,
        "residual",
        lambda *a, **k: pytest.fail("pure aggregation performed integration"),
    )
    result = kernel.diagnose(
        item,
        context,
        training,
        item.parameters,
        directions=records,
        verification=verified,
    )
    assert result["status"] == "complete"
    assert (
        result["parameter_rank_interval"][0]
        <= 1
        <= result["parameter_rank_interval"][1]
    )
    assert not result["parameter_information_qualified"]
    assert not result["coefficient_accuracy_claimed"]
    assert result["confidence_probability"] is None
    assert result["local_first_order_stationarity"] == "qualified"
    kernel.validate_diagnosis(
        item,
        context,
        training,
        item.parameters,
        directions=records,
        verification=verified,
        diagnosis=result,
    )
    flipped = deepcopy(result)
    for record in flipped["weak_combinations"]:
        record["dimensionless_direction"] = {
            n: -v for n, v in record["dimensionless_direction"].items()
        }
    kernel.validate_diagnosis(
        item,
        context,
        training,
        item.parameters,
        directions=records,
        verification=verified,
        diagnosis=flipped,
    )
    invalid = deepcopy(result)
    invalid["singular_values"][0] += 0.001
    with pytest.raises(ValueError, match="SVD roundoff"):
        kernel.validate_diagnosis(
            item,
            context,
            training,
            item.parameters,
            directions=records,
            verification=verified,
            diagnosis=invalid,
        )
    incomplete = kernel.diagnose(
        item,
        context,
        training,
        item.parameters,
        directions=records[:-1],
        verification=verified,
    )
    assert incomplete["status"] == "unavailable"
    other = kernel.diagnose(
        item,
        context,
        training,
        item.parameters,
        directions=records,
        verification=verified | {"parameters": item.parameters | {"beta": 0.3}},
    )
    assert other["status"] == "unavailable"


def test_direction_timeout_preserves_partial_accounting_and_never_qualifies(
    monkeypatch,
):
    item, context, training = example()

    def fail(*args, **kwargs):
        raise TimeoutError("test timeout")

    monkeypatch.setattr(kernel, "_column", fail)
    result = kernel.direction(item, context, training, item.parameters, 0, seconds=10)
    assert result["status"] == "unavailable"
    assert "TimeoutError" in result["derivative"]["error"]
    gate = kernel.qualification(item, context, training, item.parameters, [result])
    assert not gate["complete"] and not gate["search_qualified"]


def test_actual_kink_remains_blocked(monkeypatch):
    item, context, training = example()
    point = item.parameters | {"beta": 0.3}

    def kink(oracle, vector, deadline, method="Radau"):
        return np.full(18, 1 + abs(oracle.parameters(vector)["beta"] - 0.3))

    monkeypatch.setattr(original.DetentionOracle, "residual", kink)
    index = original.NAMES[14].index("beta")
    record = kernel.direction(item, context, training, point, index, seconds=10)
    assert record["status"] == "complete"
    assert record["derivative"]["directions"][0]["precision_status"] == "failed"
    aggregate = kernel.qualification(item, context, training, point, [record])
    assert aggregate["failed_indices"] == [index]
    assert not aggregate["search_qualified"]


def test_exact_ambiguity_survives_missing_numerical_diagnosis():
    item, context, training = example(15)
    result = kernel.diagnose(
        item, context, training, item.parameters, directions=[], verification={}
    )
    assert result["status"] == "unavailable"
    assert result["exact_ambiguity"]["rank_ceiling"] == 8
    assert result["exact_reduction"]["rank_ceiling"] == 8
    assert not result["parameter_information_qualified"]
