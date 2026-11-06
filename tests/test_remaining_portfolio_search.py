"""Other-parent fitting uses legal deterministic starts and residual-only blocks."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.fitting import remaining_portfolio_search as p
from tests.test_prefix_portfolio_nonlinear_reduction import case
from tests.test_prefix_portfolio_root_profile import changed


def build(route="nonlinear", *, anchor=True):
    source = case() if route == "nonlinear" else changed()
    original, oracle, start = p.setup(
        source.problem,
        source.training,
        source.policy
        if hasattr(source, "policy")
        else p.initialization.NumericalPolicy(),
        source.start if anchor else None,
        route,
    )
    return source, original, oracle, start


@pytest.mark.parametrize("route", ["affine", "nonlinear"])
def test_setup_exact_chart_and_initial_roots_without_mutation(route):
    source = case() if route == "nonlinear" else changed()
    before = source.problem.model_dump_json(), source.training.model_dump_json()
    original, oracle, start = p.setup(
        source.problem,
        source.training,
        p.initialization.NumericalPolicy(),
        source.start,
        route,
    )
    assert original is oracle.original
    assert p._legal(oracle, start) == pytest.approx(oracle.vector(start))
    assert before == (
        source.problem.model_dump_json(),
        source.training.model_dump_json(),
    )
    assert oracle.counts["residual_solves_started"] == 0
    assert oracle.counts["jacobian_solves_started"] == 0


def test_supplied_constructor_guesses_are_supported():
    _, _, oracle, start = build(anchor=False)
    assert oracle.vector(start) == pytest.approx(oracle.vector(oracle.start))


def test_nontraining_and_wrong_routes_fail_before_rollout():
    source = case()
    with pytest.raises(ValueError, match="training only"):
        p.setup(
            source.problem,
            source.training.model_copy(update={"name": "test"}),
            p.initialization.NumericalPolicy(),
            source.start,
            "nonlinear",
        )
    with pytest.raises(ValueError, match="unknown"):
        p.setup(
            source.problem,
            source.training,
            p.initialization.NumericalPolicy(),
            source.start,
            "approximate",
        )
    _, _, oracle, start = build()
    with pytest.raises(ValueError, match="certificate"):
        p.deterministic_starts(oracle, start, "affine")


@pytest.mark.parametrize("route", ["affine", "nonlinear"])
def test_frozen_distinct_shared_starts_preserve_training_roots_and_domains(route):
    _, _, oracle, start = build(route)
    before = deepcopy(start)
    first = p.deterministic_starts(oracle, start, route)
    second = p.deterministic_starts(oracle, start, route)
    assert first == second and start == before
    assert first[0]["label"] == "warm" and first[0]["parameters"] == start
    assert len(first) == 3
    assert len({item["parameters_sha256"] for item in first}) == 3
    for item in first:
        assert item["original_domain_checked"] and not item["feasibility_established"]
        assert not item["artificial_bounds_added"]
        p._legal(oracle, item["parameters"])
        for name in oracle.names:
            if name.startswith("trajectory:"):
                assert item["parameters"][name] == start[name]
        if item["label"] != "warm":
            assert item["changed_coordinates"]
    assert oracle.counts["residual_solves_started"] == 0


def test_recovery_patterns_also_vary_latent_initial_values():
    _, _, oracle, start = build()
    starts = p.deterministic_starts(oracle, start, "nonlinear", recovery=True)
    assert len(starts) == 5
    assert len({item["parameters_sha256"] for item in starts}) == 5
    assert all(
        any(name.startswith("trajectory:") for name in row["changed_coordinates"])
        for row in starts[1:]
    )
    _, _, affine, parameters = build("affine")
    with pytest.raises(ValueError, match="nonlinear"):
        p.deterministic_starts(affine, parameters, "affine", recovery=True)


@pytest.mark.parametrize(
    "lower,upper,value",
    [
        (-2, 3, 0),
        (0, np.inf, 0),
        (-np.inf, 0, 0),
        (-np.inf, np.inf, 0),
        (-np.inf, np.inf, -2),
        (1e-12, np.inf, 1e-12),
        (-1e308, 1e308, 0),
    ],
)
@pytest.mark.parametrize("expand", [False, True])
def test_restart_arithmetic_keeps_declared_bounds(lower, upper, value, expand):
    actual = p._variant(value, lower, upper, 1.0, expand)
    assert np.isfinite(actual) and lower <= actual <= upper


def test_unrepresentable_restart_fails_explicitly():
    with pytest.raises(ValueError, match="not representable"):
        p._variant(1e308, 0, np.inf, 1.0, True)


def test_blocks_partition_all_active_shared_parameters():
    _, _, oracle, _ = build()
    blocks = p.shared_blocks(oracle, 3)
    assert all(1 <= len(block) <= 3 for block in blocks)
    assert [name for block in blocks for name in block] == [
        name
        for i, name in enumerate(oracle.names)
        if not name.startswith("trajectory:") and oracle.lower[i] < oracle.upper[i]
    ]
    for size in (0, 5, True):
        with pytest.raises(ValueError):
            p.shared_blocks(oracle, size)


def test_nonlinear_qualification_never_asks_for_derivatives(monkeypatch):
    _, original, oracle, start = build()
    direct, reduced = original.evaluate, oracle.oracle.evaluate
    calls = []

    def original_call(*args, **kwargs):
        calls.append(("original", kwargs))
        assert kwargs["jacobian"] is False
        return direct(*args, **kwargs)

    def reduced_call(*args, **kwargs):
        calls.append(("reduced", kwargs))
        assert kwargs["jacobian"] is False
        return reduced(*args, **kwargs)

    monkeypatch.setattr(original, "evaluate", original_call)
    monkeypatch.setattr(oracle.oracle, "evaluate", reduced_call)
    result = p.qualify(original, oracle, start, 10, lambda _: None, "nonlinear")
    assert result["passed"] and result["original_equations_verified"]
    assert result["parameters"] == start
    assert not result["gradient_qualified"] and not result["convergence_certified"]
    assert len(calls) == 3


def test_affine_conditional_fit_and_independent_original_replay():
    _, original, oracle, start = build("affine")
    qualified = p.qualify(original, oracle, start, 10, lambda _: None, "affine")
    assert qualified["passed"] and qualified["conditional_root_optimality_checked"]
    result = p.verify(original, oracle, qualified["parameters"], 10, lambda _: None)
    assert result["original_equations_verified"]
    assert result["training_nmse"] == pytest.approx(
        qualified["training_nmse"], abs=1e-9
    )


def test_nonlinear_blocks_freeze_complement_and_release_every_coordinate(monkeypatch):
    _, _, oracle, start = build()
    calls = []

    def search(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(p.residual, "search", search)
    block = p.shared_blocks(oracle)[0]
    p.search(oracle, start, 10, 20, lambda _: None, "nonlinear", block=block)
    assert calls[0]["fixed"] == {
        name: v for name, v in start.items() if name not in block
    }
    p.search(oracle, start, 10, 20, lambda _: None, "nonlinear")
    assert calls[1]["fixed"] is None


@pytest.mark.parametrize("block", [[], ["missing"], "a", ["a", "a"]])
def test_bad_block_never_enters_optimizer(block, monkeypatch):
    _, _, oracle, start = build()
    monkeypatch.setattr(
        p.residual, "search", lambda *a, **k: pytest.fail("bad block ran")
    )
    with pytest.raises(ValueError, match="blocks"):
        p.search(oracle, start, 10, 20, lambda _: None, "nonlinear", block=block)


def test_root_coordinates_cannot_be_selected_as_shared_blocks():
    _, _, oracle, start = build()
    root = next(name for name in oracle.names if name.startswith("trajectory:"))
    with pytest.raises(ValueError, match="blocks"):
        p.search(oracle, start, 10, 20, lambda _: None, "nonlinear", block=[root])


def test_real_residual_block_search_is_small_and_leaves_other_coordinates_fixed():
    _, _, oracle, start = build()
    block = p.shared_blocks(oracle)[0][:1]
    result = p.search(oracle, start, 10, 3, lambda _: None, "nonlinear", block=block)
    assert result["best"] is not None
    assert result["free_coordinates"] == block
    assert result["stage_backend_calls"]["jacobian_solves_started"] == 0
    for name in oracle.names:
        if name not in block:
            assert result["best"]["parameters"][name] == start[name]
    assert not result["convergence_certified"]


def test_wide_finite_bounds_do_not_set_restart_magnitude():
    assert p._variant(1e-6, 0.0, 1e9, 1.0, False) == 0.25e-6
    assert p._variant(1e-6, 0.0, 1e9, 1.0, True) == 4e-6
    assert p._variant(-1e-6, -1e9, 1e9, 1.0, True) == -4e-6
    assert p._variant(1.0, 0.0, 2.0, 1.0, True) == 2.0


def test_explicit_affine_persistence_headroom_matches_campaign_reservation(monkeypatch):
    _, _, oracle, start = build("affine")
    calls = []
    monkeypatch.setattr(p.roots, "root_profile_search", lambda *a, **k: calls.append(k))
    p.search(
        oracle, start, 10, 20, lambda _: None, "affine", persistence_headroom_seconds=0
    )
    p.search(
        oracle, start, 10, 20, lambda _: None, "affine", persistence_headroom_seconds=60
    )
    assert [row["overall_wall_seconds"] for row in calls] == [10, 70]
    for bad in (-1, 901, float("nan"), True):
        with pytest.raises(ValueError, match="headroom"):
            p.search(
                oracle,
                start,
                10,
                20,
                lambda _: None,
                "affine",
                persistence_headroom_seconds=bad,
            )


def test_qualification_and_replay_reject_an_unrelated_original():
    _, _, oracle, start = build("affine")
    with pytest.raises(ValueError, match="original oracle differs"):
        p.qualify(object(), oracle, start, 10, lambda _: None, "affine")
    with pytest.raises(ValueError, match="original oracle differs"):
        p.verify(object(), oracle, start, 10, lambda _: None)
