"""Exact raw samples survive interruption without weakening derivative evidence."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import llm_sr_detention_continuation as chart
from autoformalism.fitting import llm_sr_detention_probe_recovery as kernel
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import prefix_local_derivatives as local
from autoformalism.fitting import public_fitting as public
from tests.test_llm_sr_detention_refit import example


@pytest.fixture
def control(monkeypatch):
    """Count physical evaluations of a smooth coupled residual, without ODE work."""
    state = {"calls": [], "fail_at": None}

    def residual(oracle, point, deadline, method="Radau"):
        state["calls"].append(np.asarray(point).copy())
        oracle.cost["residual_calls_started"] += 1
        if len(state["calls"]) == state["fail_at"]:
            oracle.cost["residual_calls_failed"] += 1
            raise TimeoutError("controlled sample interruption")
        oracle.cost["residual_calls_completed"] += 1
        oracle.cost["trajectory_integrations"] += 2
        count = sum(len(row.time) for row in oracle.split.trajectories)
        weights = np.arange(1, len(point) + 1)
        return (1 + np.dot(weights, point)) * np.arange(1, count + 1) / count

    monkeypatch.setattr(original.DetentionOracle, "residual", residual)
    return state


def audit(root, *, index=0, mode="original", parameters=None, **kwargs):
    item, context, training = example(15)
    return kernel.direction(
        item,
        context,
        training,
        item.parameters if parameters is None else parameters,
        index,
        mode=mode,
        cache_root=root,
        **kwargs,
    )


def reseal_row(row):
    row["evidence_sha256"] = public.content_sha256(
        {key: value for key, value in row.items() if key != "evidence_sha256"}
    )
    return row


def replace_sealed(path, value):
    public._write(path, {"value": value, "sha256": public.content_sha256(value)})


def test_resume_and_other_coordinates_share_complete_residuals(tmp_path, control):
    first = audit(tmp_path)
    assert first["diagnostic_outcome"] == "checked"
    assert first["derivative"]["status"] == "passed"
    count = len(control["calls"])
    assert count > 1
    again = audit(tmp_path)
    assert len(control["calls"]) == count
    assert again["probe_recovery"]["missing_residual_requests"] == 0
    assert again["numerical_cost"]["trajectory_integrations"] == 0
    assert not again["parameter_information_qualified"]
    assert not again["derivative"]["parameter_information_qualified"]
    assert not again["derivative"]["gradient_precision_qualified"]
    assert not again["full_model_convergence_claimed"]
    assert again["timing"]["callback_wall_seconds"] is None
    assert again["timing"]["numerical_active_wall_seconds"] is None
    second_coordinate = audit(tmp_path, index=1)
    assert second_coordinate["diagnostic_outcome"] == "checked"
    base = control["calls"][0]
    assert sum(np.array_equal(point, base) for point in control["calls"]) == 1
    # Independent coordinate audits would each cost count evaluations.
    assert len(control["calls"]) < 2 * count


@pytest.mark.parametrize("change", ["point", "mode", "policy", "source"])
def test_changed_contract_cannot_reuse_existing_cache(
    tmp_path, control, monkeypatch, change
):
    audit(tmp_path)
    kwargs = {}
    if change == "point":
        kwargs["parameters"] = example(15)[0].parameters | {"beta": 1.001}
    elif change == "mode":
        kwargs["mode"] = "reduced_model15"
    elif change == "policy":
        kwargs["policy"] = chart.Policy(search_gradient_relative_error=0.02)
    else:
        monkeypatch.setattr(kernel, "_implementation_identity", lambda: "changed")
    with pytest.raises(ValueError, match="sealed artifact differs"):
        audit(tmp_path, **kwargs)


def test_model_and_training_are_part_of_cache_identity(tmp_path, control):
    audit(tmp_path)
    item, context, training = example(15)
    altered = item.model_copy(update={"repetition": 1})
    with pytest.raises(ValueError, match="sealed artifact differs"):
        kernel.direction(
            altered,
            context,
            training,
            item.parameters,
            0,
            mode="original",
            cache_root=tmp_path,
        )
    altered_training = training.model_copy(update={"fingerprint": "another"})
    with pytest.raises(ValueError, match="sealed artifact differs"):
        kernel.direction(
            item,
            context,
            altered_training,
            item.parameters,
            0,
            mode="original",
            cache_root=tmp_path,
        )


@pytest.mark.parametrize("tamper", ["digest", "status", "length", "request"])
def test_even_resealed_sample_tampering_is_not_a_success(tmp_path, control, tamper):
    cache = kernel.SampleCache(tmp_path, {"residual_length": 2})
    request = cache.request([1.0])
    cache.save(request, [1.0, 2.0], provenance={"kind": "test"})
    path = cache._path(request)
    value = read_seal(path)
    if tamper == "digest":
        value["values"][0] += 1
    elif tamper == "status":
        value["status"] = "unfinished"
    elif tamper == "length":
        value["values"] = [1.0]
    else:
        value["request"]["method"] = "other"
    replace_sealed(path, value)
    with pytest.raises(ValueError):
        kernel.SampleCache(tmp_path, {"residual_length": 2}).load(request)


def test_timeout_preserves_completed_samples_and_new_allowance_computes_missing(
    tmp_path, control
):
    control["fail_at"] = 4
    first = audit(tmp_path)
    assert first["diagnostic_outcome"] == "timed_out"
    assert first["status"] == "unavailable"
    assert not first["derivative"]["search_qualified"]
    assert first["probe_recovery"]["new_sample_records"] == 4  # 3 residuals + column.
    before = len(control["calls"])
    control["fail_at"] = None
    resumed = audit(tmp_path)
    assert resumed["diagnostic_outcome"] == "checked"
    assert resumed["probe_recovery"]["old_allowances_renewed"] is False
    assert resumed["probe_recovery"]["new_sample_records"] == 4
    assert len(control["calls"]) - before == 4
    final_count = len(control["calls"])
    assert audit(tmp_path)["diagnostic_outcome"] == "checked"
    assert len(control["calls"]) == final_count


def test_failed_sample_is_not_retried_in_same_allowance_or_saved(tmp_path):
    cache = kernel.SampleCache(tmp_path, {"residual_length": 1})
    calls = []

    def fails(*args):
        calls.append(True)
        raise RuntimeError("invalid integration")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="invalid integration"):
            cache.residual(fails, [0.0], monotonic() + 1)
    assert len(calls) == 1
    assert cache.load(cache.request([0.0])) is None
    assert not list(tmp_path.glob("samples/*.json"))


def test_historical_raw_arrays_can_seed_but_scalar_probe_summaries_cannot(
    tmp_path, control
):
    seed = audit(tmp_path / "old")
    seed = deepcopy(seed)
    seed["status"] = "unavailable"
    seed["derivative"]["status"] = "incomplete"
    seed["derivative"]["error"] = "TimeoutError: optional later precision probe"
    seed["derivative"]["search_qualified"] = False
    reseal_row(seed)
    before = len(control["calls"])
    new = audit(tmp_path / "new", seed_directions=[seed])
    assert new["diagnostic_outcome"] == "checked"
    assert new["probe_recovery"]["seeded_reference_columns"] == 1
    assert new["probe_recovery"]["seed_evidence_sha256"] == [seed["evidence_sha256"]]
    # The four old audit probes only have scalar summaries; they must be redone.
    assert len(control["calls"]) - before == 4
    assert new["derivative"]["error"] is None


@pytest.mark.parametrize("tamper", ["array", "endpoint"])
def test_historical_seed_rejects_false_raw_arrays_or_known_contradictions(
    tmp_path, control, tamper
):
    seed = audit(tmp_path / "old")
    if tamper == "array":
        seed["base_residual"][0] += 1
    else:
        seed["original_parameters"]["beta"] += 1
    reseal_row(seed)
    with pytest.raises(ValueError):
        audit(tmp_path / "new", seed_directions=[seed])


def test_failed_seed_does_not_block_other_coordinates_and_requires_full_recheck(
    tmp_path, control
):
    seed = audit(tmp_path / "old")
    seed["derivative"]["failed_indices"] = [0]
    seed["derivative"]["directions"][0]["precision_status"] = "failed"
    # Simulate a saved late contradiction after apparently acceptable early pairs.
    last = deepcopy(seed["derivative"]["local_probe_attempts"][-1])
    last["scale_index"] = 5
    seed["derivative"]["local_probe_attempts"].append(last)
    reseal_row(seed)
    other = audit(tmp_path / "new", index=1, seed_directions=[seed])
    assert other["diagnostic_outcome"] == "checked"
    assert other["probe_recovery"]["historical_failed_coordinates"] == [0]
    assert other["probe_recovery"]["seeded_reference_columns"] == 0
    assert other["probe_recovery"]["required_historical_probe_scales"] == 0
    before = len(control["calls"])
    repaired = audit(tmp_path / "new", seed_directions=[seed])
    assert repaired["probe_recovery"]["required_historical_probe_scales"] == 6
    assert len(repaired["derivative"]["local_probe_attempts"]) == 6
    assert len(control["calls"]) - before > 4


def test_seeded_resolved_failure_is_not_cleared_by_recovery(tmp_path, control):
    seed = audit(tmp_path / "old")
    seed["derivative"]["failed_indices"] = [0]
    seed["derivative"]["directions"][0]["precision_status"] = "failed"
    reseal_row(seed)
    control["fail_at"] = len(control["calls"]) + 1
    result = audit(tmp_path / "new", seed_directions=[seed])
    assert result["diagnostic_outcome"] == "timed_out"
    assert not result["derivative"]["search_qualified"]


def local_audit(function, *, limits=None, max_calls=None):
    calls = []

    def evaluate(point, deadline):
        calls.append(point.copy())
        if max_calls is not None and len(calls) > max_calls:
            raise TimeoutError("optional refinement exhausted")
        return function(point)

    result = local.local_derivative_check(
        evaluate,
        [1.0],
        [0.0],
        [2.0],
        monotonic() + 5,
        coordinate_units=[1.0],
        policy=chart.effective_policy(mode="original").numerical.derivative,
        search_error_limits=limits,
    )
    return result, calls


def test_search_precision_stops_before_optional_timeout_strict_default_unchanged():
    def smooth(point):
        value = np.exp(10 * point[0])
        return np.array([value]), np.array([[10 * value]])

    strict, strict_calls = local_audit(smooth)
    search, search_calls = local_audit(smooth, limits=(1e-7, 0.01), max_calls=5)
    interrupted, _ = local_audit(smooth, max_calls=5)
    assert strict.status == search.status == "passed"
    assert len(search_calls) == 5 < len(strict_calls)
    assert interrupted.status == "incomplete" and interrupted.error.startswith(
        "TimeoutError"
    )
    assert not interrupted.search_qualified
    assert search.search_qualified
    assert not search.gradient_precision_qualified
    assert not search.parameter_information_qualified
    assert strict.parameter_information_qualified


@pytest.mark.parametrize("kind", ["wrong_derivative", "kink"])
def test_search_stop_never_turns_resolved_derivative_failure_into_pass(kind):
    def bad(point):
        if kind == "wrong_derivative":
            return np.array([point[0]]), np.array([[5.0]])
        return np.array([abs(point[0] - 1.0)]), np.array([[0.0]])

    result, _ = local_audit(bad, limits=(1e-7, 0.01))
    assert result.status == "failed"
    assert result.failed_indices == (0,)
    assert not result.search_qualified


@pytest.mark.parametrize("limits", [(0.0, 0.01), (1e-7,), (1e-7, float("nan"))])
def test_invalid_search_error_contract_raises(limits):
    with pytest.raises(ValueError, match="search error limits"):
        local_audit(lambda x: (x.copy(), np.ones((1, 1))), limits=limits)
