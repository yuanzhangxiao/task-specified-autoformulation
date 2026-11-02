"""Root-only projection preserves affine maps, data weights and original bounds."""

import json
from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting import prefix_portfolio_timescale as timescale
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit
from tests.test_prefix_portfolio_timescale import coupled, wrapped
from tests.test_prefix_structural_reduction import training


def changed(*, initializer=None, observation=None, nonnegative=False):
    source = coupled()
    payload = source.problem.model_dump(mode="json")
    if initializer is not None:
        payload["graph"]["rules"]["x"] = {"initial": initializer}
    if observation is not None:
        payload["base_candidate"]["observation_mappings"][0]["expression"] = observation
    if nonnegative:
        payload["graph"]["rules"]["z"]["initial"]["role"] = "nonnegative_coefficient"
    problem = PrefixFitProblem.model_validate(payload)
    return kernel.TrainingOracle(problem, training(source), kernel.NumericalPolicy())


def test_profile_recovers_bounded_conditional_roots_and_full_rollout():
    original, oracle = wrapped()
    profiler = roots.RootProfile(oracle)
    before = deepcopy(oracle.start)
    result = profiler.profile(oracle.start, monotonic() + 10)
    assert result["conditional_root_optimality_checked"]
    assert not result["outer_convergence_certified"]
    assert len(profiler.inner) == 2 and len(profiler.outer) == 4
    assert result["maximum_pairing_discrepancy"] < 1e-10
    initial, _ = oracle.evaluate(
        oracle.vector(before), monotonic() + 10, jacobian=False
    )
    assert result["training_nmse"] <= np.mean(initial**2)
    actual, _ = original.evaluate(
        original.vector(result["original_parameters"]),
        monotonic() + 10,
        jacobian=False,
        method="Radau",
    )
    assert result["training_nmse"] == pytest.approx(np.mean(actual**2), abs=1e-9)
    assert oracle.start == before
    assert all(
        result["parameters"][oracle.names[i]] == before[oracle.names[i]]
        for i in profiler.outer
    )
    assert profiler.counts == {
        f"{scope}_calls_{state}": (0 if state == "failed" else 1)
        for scope in ("profile", "basis", "pairing")
        for state in ("started", "completed", "failed")
    }


def test_affine_dependent_initials_and_scaled_multioutput_readouts():
    source = changed(
        initializer={
            "mode": "map",
            "expression": "2*z0+1",
            "state_initials": {"z0": "z"},
        },
        observation="2*x+3*z+1",
    )
    problem = source.problem.model_dump(mode="json")
    problem["base_candidate"]["observation_mappings"].append(
        {"channel": "y2", "expression": "x-z"}
    )
    problem["context"]["targets"].append("y2")
    data = training(source).model_dump(mode="json")
    for row in data["rows"]:
        row["targets"]["y2"] = (np.asarray(row["targets"]["y"]) * 100).tolist()
    original, oracle = timescale.make_oracle(
        PrefixFitProblem.model_validate(problem),
        PublicSplit.model_validate(data),
        kernel.NumericalPolicy(),
        source.start,
        states=("x", "z"),
    )
    profiler = roots.RootProfile(oracle)
    parameters = oracle.start
    blocks = profiler.design(parameters, monotonic() + 10)
    point = oracle.vector(parameters)
    expected = np.concatenate(
        [b["design"] @ point[b["indices"]] + b["offset"] for b in blocks]
    )
    actual, _ = oracle.evaluate(point, monotonic() + 10, jacobian=False)
    assert expected == pytest.approx(actual, abs=1e-9)
    assert original.scales["y2"] > 10 * original.scales["y"]
    result = profiler.profile(parameters, monotonic() + 10)
    assert result["maximum_pairing_discrepancy"] < 1e-9


@pytest.mark.parametrize("expression", ["z0*z0", "exp(z0)"])
def test_nonlinear_root_initializer_fails_closed(expression):
    source = changed(
        initializer={
            "mode": "map",
            "expression": expression,
            "state_initials": {"z0": "z"},
        }
    )
    _, oracle = wrapped(source)
    with pytest.raises(ValueError, match="jointly affine in roots"):
        roots.RootProfile(oracle)


def test_nonlinear_output_fails_closed():
    _, oracle = wrapped(changed(observation="x*x+z"))
    with pytest.raises(ValueError, match="readouts must be affine"):
        roots.RootProfile(oracle)


def test_nonnegative_root_bounds_preserved():
    _, oracle = wrapped(changed(nonnegative=True))
    profiler = roots.RootProfile(oracle)
    result = profiler.profile(oracle.start, monotonic() + 10)
    assert all(result["parameters"][n] >= 0 for n in profiler.root_names)
    assert all(a["bounds_satisfied"] for a in result["root_solve"])


def test_changed_root_domain_rejected():
    _, oracle = wrapped()
    root = next(i for i, n in enumerate(oracle.names) if n.startswith("trajectory:"))
    oracle.oracle.lower = oracle.lower.copy()
    oracle.oracle.lower[root] = -3
    with pytest.raises(ValueError, match="root domains differ"):
        roots.RootProfile(oracle)


def test_original_chart_must_preserve_each_root(monkeypatch):
    _, oracle = wrapped()
    original = timescale._lift_polynomials

    def changed_lift(chart):
        values = original(chart)
        name = next(n for n in values if n.startswith("trajectory:"))
        values[name] = {(): 1}
        return values

    monkeypatch.setattr(timescale, "_lift_polynomials", changed_lift)
    with pytest.raises(ValueError, match="root lift"):
        roots.RootProfile(oracle)


def test_timeout_and_failed_pairing_keep_honest_counters(monkeypatch):
    _, oracle = wrapped()
    profiler = roots.RootProfile(oracle)
    with pytest.raises(TimeoutError):
        profiler.profile(oracle.start, monotonic() - 1)
    assert profiler.counts["profile_calls_failed"] == 1
    assert profiler.counts["basis_calls_failed"] == 1
    original = oracle.evaluate

    def wrong(*args, **kwargs):
        r, j = original(*args, **kwargs)
        return r + 1, j

    monkeypatch.setattr(oracle, "evaluate", wrong)
    with pytest.raises(NumericalInvalidTrial, match="pairing differs"):
        profiler.profile(oracle.start, monotonic() + 10)
    assert profiler.counts["profile_calls_started"] == 2
    assert profiler.counts["profile_calls_completed"] == 0
    assert profiler.counts["pairing_calls_completed"] == 1


def test_search_profiles_every_root_and_retains_control_at_call_limit():
    _, oracle = wrapped()
    records = []
    result = roots.root_profile_search(oracle, oracle.start, 10, 5, records.append)
    assert result["initial_root_profile"] is not None
    assert (
        result["best"]["training_nmse"]
        <= result["unoptimized_control"]["training_nmse"]
    )
    assert result["search_calls_started"] == result["search_calls_completed"] == 5
    assert result["search_calls_failed"] == 0
    assert result["profile_execution"]["profile_calls_completed"] == 5
    assert result["budget_exhausted"]
    assert not result["derivatives_used"]
    assert all(not n.startswith("trajectory:") for n in result["outer_coordinates"])
    assert records and result["independent_original_verification_required"]
    assert "failure_diagnostics" not in result


def test_fixed_shared_parameters_are_not_optimized():
    _, oracle = wrapped()
    profiler = roots.RootProfile(oracle)
    fixed = {oracle.names[i]: oracle.start[oracle.names[i]] for i in profiler.outer}
    result = roots.root_profile_search(
        oracle, oracle.start, 10, 5, lambda _: None, fixed=fixed
    )
    assert result["stop_reason"] == "no_free_shared_coordinates"
    assert result["search_calls_completed"] == 1
    assert result["outer_coordinates"] == []


def test_balanced_log_timescale_sweep_precedes_powell(monkeypatch):
    _, oracle = wrapped()
    name = "timescale:x"
    start = oracle.start | {name: 10.0}
    seen = []
    real = roots.RootProfile.profile

    def record(self, parameters, deadline):
        seen.append(dict(parameters))
        return real(self, parameters, deadline)

    def stop_after_sweep(*args, **kwargs):
        raise TimeoutError("stop after initial sweep")

    monkeypatch.setattr(roots.RootProfile, "profile", record)
    monkeypatch.setattr(roots, "minimize", stop_after_sweep)
    result = roots.root_profile_search(oracle, start, 10, 100, lambda _: None)
    assert result["initial_sweep_completed"]
    assert result["search_calls_started"] == 1 + 2 * len(result["outer_coordinates"])
    assert set(result["outer_coordinates_varied"]) == set(result["outer_coordinates"])
    assert any(p[name] == pytest.approx(10 * np.exp(1)) for p in seen)
    assert any(p[name] == pytest.approx(10 * np.exp(-1)) for p in seen)
    assert all(p[name] > 0 for p in seen)
    assert not result["outer_coordinate_chart"]["positive_timescale_floor_added"]


def test_nonrepresentable_log_trial_is_rejected_without_losing_best(monkeypatch):
    _, oracle = wrapped()

    def extreme_trial(objective, start, **kwargs):
        point = start.copy()
        point[0] = -1e9
        assert not np.isfinite(objective(point))
        raise TimeoutError("bounded injected stop")

    monkeypatch.setattr(roots, "minimize", extreme_trial)
    result = roots.root_profile_search(oracle, oracle.start, 10, 100, lambda _: None)
    assert result["best"] is not None
    assert result["invalid_trials"] == result["search_calls_failed"] == 1
    assert "nonrepresentable logarithmic timescale" in result["last_invalid_trial"]


@pytest.mark.parametrize("seconds,calls", [(0, 1), (3601, 1), (1, 0), (1, True)])
def test_invalid_search_allowances_rejected(seconds, calls):
    with pytest.raises(ValueError, match="search needs"):
        roots.root_profile_search(None, {}, seconds, calls, lambda _: None)


def test_shared_initial_values_are_not_per_trajectory_roots():
    source = coupled()
    payload = source.problem.model_dump(mode="json")
    payload["graph"]["rules"]["z"]["initial"]["scope"] = "shared_training"
    source = kernel.TrainingOracle(
        PrefixFitProblem.model_validate(payload),
        training(source),
        kernel.NumericalPolicy(),
    )
    _, oracle = wrapped(source)
    with pytest.raises(ValueError, match="trajectory roots required"):
        roots.RootProfile(oracle)


def test_qualification_accounts_basis_and_pairing_separately():
    _, oracle = wrapped()
    records = []
    result = roots.root_profile_qualification(oracle, oracle.start, 10, records.append)
    assert result["passed"]
    assert (
        result["verification_calls_started"]
        == result["verification_calls_completed"]
        == 2
    )
    assert result["verification_calls_failed"] == 0
    assert result["profile_execution"]["profile_calls_completed"] == 1
    assert result["root_profile_certificate"]["sha256"] == result["certificate_sha256"]
    assert records[-1]["passed"]


def test_qualification_setup_failure_never_qualifies(monkeypatch):
    _, oracle = wrapped()

    def fail(*_args):
        raise ValueError("uncertified setup")

    monkeypatch.setattr(roots, "RootProfile", fail)
    result = roots.root_profile_qualification(oracle, oracle.start, 10, lambda _: None)
    assert result["passed"] is False
    assert result["verification_calls_started"] == 0
    assert "uncertified setup" in result["error"]


def boundary_oracle(lower=0):
    """A canonical shared readout offset survives the timescale chart unchanged."""
    source = coupled()
    payload = source.problem.model_dump(mode="json")
    payload["base_candidate"]["parameters"].append(
        {
            "name": "offset",
            "scope": "global",
            "role": "coefficient",
            "bounds": {"lower": lower, "upper": 2},
        }
    )
    payload["base_candidate"]["observation_mappings"][0]["expression"] = "x+z+offset"
    payload["parameter_guesses"]["offset"] = 0.25
    source = kernel.TrainingOracle(
        PrefixFitProblem.model_validate(payload),
        training(source),
        kernel.NumericalPolicy(),
    )
    return wrapped(source)


def test_zero_boundary_reprofiles_roots_without_selecting_or_moving_shared_parameters():
    original, oracle = boundary_oracle()
    before = deepcopy(oracle.start)
    records = []
    result = roots.root_boundary_control(oracle, before, "offset", 10, records.append)
    assert result["passed"] and result["feasible"]
    assert result["conditional_root_optimality_checked"]
    assert (
        result["parameters"]["offset"] == result["original_parameters"]["offset"] == 0
    )
    assert result["other_shared_coordinates_unchanged"]
    assert result["other_original_shared_parameters_unchanged"]
    assert result["independent_original_verification_required"]
    assert (
        not result["original_equations_verified"] and not result["candidate_selected"]
    )
    assert before == oracle.start
    for name, value in before.items():
        if name != "offset" and not name.startswith("trajectory:"):
            assert result["parameters"][name] == value
    residual, _ = original.evaluate(
        original.vector(result["original_parameters"]),
        monotonic() + 10,
        jacobian=False,
        method="Radau",
    )
    assert np.mean(residual**2) == pytest.approx(result["training_nmse"], abs=1e-9)
    assert (
        result["verification_calls_started"]
        == result["verification_calls_completed"]
        == 2
    )
    assert result["verification_calls_failed"] == 0
    assert result["profile_execution"]["profile_calls_completed"] == 1
    assert records[-1]["passed"]


@pytest.mark.parametrize("name", ["missing", "trajectory:first:z", "timescale:x"])
def test_boundary_rejects_unknown_root_and_noncanonical_chart_names(name):
    _, oracle = boundary_oracle()
    result = roots.root_boundary_control(oracle, oracle.start, name, 10, lambda _: None)
    assert not result["passed"] and not result["feasible"]
    assert result["verification_calls_started"] == 0
    assert "original shared coordinate" in result["error"]


def test_boundary_preserves_declared_domain_and_rejects_zero_outside_it():
    _, oracle = boundary_oracle(lower=0.1)
    result = roots.root_boundary_control(
        oracle, oracle.start, "offset", 10, lambda _: None
    )
    assert not result["passed"] and not result["feasible"]
    assert "outside the declared" in result["error"]
    assert result["profile_execution"]["profile_calls_started"] == 0


def test_boundary_failed_pairing_is_not_a_feasible_candidate(monkeypatch):
    _, oracle = boundary_oracle()
    original = oracle.evaluate

    def wrong(*args, **kwargs):
        residual, _ = original(*args, **kwargs)
        return residual + 1, None

    monkeypatch.setattr(oracle, "evaluate", wrong)
    result = roots.root_boundary_control(
        oracle, oracle.start, "offset", 10, lambda _: None
    )
    assert not result["passed"] and not result["feasible"]
    assert not result["conditional_root_optimality_checked"]
    assert result["verification_calls_completed"] == 2
    sample = result["failure_diagnostics"]["representative_trials"][0]
    assert sample["stage"] == "pairing"
    assert sample["maximum_pairing_discrepancy"] == pytest.approx(1)


def test_boundary_deadline_does_not_restart_or_claim_success(monkeypatch):
    _, oracle = boundary_oracle()
    calls = []

    def timeout(self, parameters, deadline):
        calls.append(parameters)
        raise TimeoutError("injected boundary deadline")

    monkeypatch.setattr(roots.RootProfile, "profile", timeout)
    result = roots.root_boundary_control(
        oracle, oracle.start, "offset", 10, lambda _: None
    )
    assert len(calls) == 1 and not result["passed"]
    assert result["status"] == "incomplete"
    assert "deadline" in result["error"]


def test_search_failure_diagnostics_preserve_kkt_gate_and_best(monkeypatch):
    _, oracle = wrapped()
    project = roots.projection._project
    calls = 0

    def failing(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= 2:  # The initial point profiles both trajectories successfully.
            return project(*args, **kwargs)
        raise roots.projection.ProfileUnavailable(
            "projection KKT residual unresolved",
            {
                "status": "kkt_unavailable",
                "normalized_kkt_residual": 0.01,
                "kkt_tolerance": 1e-10,
                "active_mask": [0],
                "normalized_gradient": [0.01],
                "design_rank": 1,
            },
        )

    monkeypatch.setattr(roots.projection, "_project", failing)
    records = []
    result = roots.root_profile_search(
        oracle,
        oracle.start,
        10,
        6,
        records.append,
        failure_diagnostics=True,
    )
    assert result["search_calls_completed"] == 1 and result["invalid_trials"] == 5
    assert (
        result["best"]["training_nmse"]
        <= result["unoptimized_control"]["training_nmse"]
    )
    diagnostics = result["failure_diagnostics"]
    assert sum(diagnostics["histogram"].values()) == 5
    assert len(diagnostics["representative_trials"]) == 2
    assert diagnostics["omitted_representatives"] == 3
    sample = diagnostics["representative_trials"][0]
    assert sample["stage"] == "inner_solve" and sample["trajectory_id"] == "first"
    assert sample["inner_solve"]["normalized_kkt_residual"] == 0.01
    assert sample["inner_solve"]["kkt_tolerance"] == 1e-10
    assert all(not n.startswith("trajectory:") for n in sample["shared_parameters"])
    assert records[-1]["failure_diagnostics"] == diagnostics


def test_failure_receipt_stays_bounded_and_json_finite():
    _, oracle = wrapped()
    profiler = roots.RootProfile(oracle)
    ledger = roots._failure_ledger()
    for i in range(80):
        error = roots.projection.ProfileUnavailable(
            "x" * 2000,
            {"status": f"failure-{i}", "normalized_gradient": [np.nan] * 100},
        )
        roots._record_failure(
            ledger,
            error,
            attempt=i,
            oracle=oracle,
            indices=profiler.outer,
            point=oracle.vector(oracle.start),
        )
    assert sum(ledger["histogram"].values()) == 80
    assert len(ledger["histogram"]) <= 9
    assert len(ledger["representative_trials"]) == 8
    assert ledger["omitted_representatives"] == 72
    assert len(json.dumps(ledger, allow_nan=False)) < 15000


def test_opt_in_diagnostics_do_not_change_successful_search_result():
    _, oracle = wrapped()
    plain = roots.root_profile_search(oracle, oracle.start, 10, 5, lambda _: None)
    diagnostic = roots.root_profile_search(
        oracle, oracle.start, 10, 5, lambda _: None, failure_diagnostics=True
    )
    assert plain["best"] == diagnostic["best"]
    assert plain["profile_execution"] == diagnostic["profile_execution"]
    assert "failure_diagnostics" not in plain
    assert diagnostic["failure_diagnostics"]["histogram"] == {}
