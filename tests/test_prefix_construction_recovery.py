"""Exact inactive initializer sections and bounded local precision recovery."""

from copy import deepcopy
from time import monotonic

import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_construction_recovery as recovery
from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting.prefix_affine import TrainingOracle
from autoformalism.fitting.prefix_initialization import NumericalPolicy
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from tests.test_prefix_gain_reduction import oracle_for
from tests.test_prefix_initialization import problem, row, training


def mapped_case(*, shift=False, use_time=True):
    """A dependent initialization DAG with a coefficient times public initial time."""
    payload = problem(shared=True).model_dump(mode="json")
    payload["graph"]["rules"]["r"]["initial"] = {
        "mode": "map",
        "expression": "a*clock" if use_time else "a*clock+1e-30*a",
        "parameters": [{"name": "a", "guess": 2.0}],
        "public_initials": {"clock": "t"},
    }
    model = PrefixFitProblem.model_validate(payload)
    data = training()
    if shift:
        data = data.model_copy(update={"rows": (data.rows[0], row("b", 1.7, shift=1))})
    original = TrainingOracle(model, data, NumericalPolicy())
    retained = original.start | {"init_r_a": 3.5, "init_z_phi": 0.7}
    return model, data, retained


def test_exact_training_zero_removes_full_dag_dependencies_and_preserves_anchor():
    model, data, retained = mapped_case()
    before = model.model_dump_json(), data.model_dump_json()
    original, oracle = recovery.make_oracle(model, data, NumericalPolicy(), retained)
    proof = oracle.recovery_audit["coordinate_reduction"]["inactive_initializer_proof"]
    assert set(proof["removed_names"]) == {"init_r_a", "init_z_phi"}
    assert oracle.names == ("theta",)
    vector = oracle.vector({"theta": 0.8})
    restored = continuation.to_original(original, oracle, oracle.parameters(vector))
    assert restored == retained | {"theta": 0.8}
    assert oracle.reduction.reduce(oracle.reduction.lift(vector)) == pytest.approx(
        vector
    )
    actual, derivative = oracle.evaluate(vector, monotonic() + 10)
    reference, full = original.evaluate(original.vector(restored), monotonic() + 10)
    assert actual == pytest.approx(reference, abs=1e-14)
    assert derivative == pytest.approx(full[:, [0]], abs=1e-14)
    assert not proof["future_boundary_equivalence_certified"]
    assert not oracle.recovery_audit["policy"][
        "original_parameter_identifiability_certified"
    ]
    assert before == (model.model_dump_json(), data.model_dump_json())
    # A candidate's changed active values never move the frozen source anchors.
    assert oracle.reduction.receipt["anchored_coordinates"] == {
        "init_r_a": 3.5,
        "init_z_phi": 0.7,
    }


@pytest.mark.parametrize("shift,use_time", [(True, True), (False, False)])
def test_nonzero_boundary_or_tiny_dependency_cannot_be_removed(shift, use_time):
    model, data, retained = mapped_case(shift=shift, use_time=use_time)
    _, oracle = recovery.make_oracle(model, data, NumericalPolicy(), retained)
    assert not oracle.reduction.proof["removed_names"]
    assert {"init_r_a", "init_z_phi"} <= set(oracle.names)


def test_shared_root_and_dependent_map_remain_active():
    model, data = problem(shared=True), training()
    source = TrainingOracle(model, data, NumericalPolicy())
    _, oracle = recovery.make_oracle(model, data, NumericalPolicy(), source.start)
    assert not oracle.reduction.proof["removed_names"]
    assert "init_r_value" in oracle.names
    assert "init_z_phi" in oracle.names


def test_gain_quotient_and_training_projection_compose_exactly():
    source = oracle_for(root_scope="shared_training")
    model, data = source.problem, training()
    # Ordinary gain reduction remains active without inactive initial coordinates.
    original = TrainingOracle(model, data, NumericalPolicy())
    retained = original.start | {"a": 2, "g": 3}
    original, oracle = recovery.make_oracle(model, data, NumericalPolicy(), retained)
    active = continuation.to_effective(original, oracle, retained)
    assert active["a"] == 6
    assert "g" not in active
    restored = continuation.to_original(original, oracle, active)
    assert restored["a"] == 6 and restored["g"] == 1
    actual = oracle.evaluate(oracle.vector(active), monotonic() + 10)[0]
    expected = original.evaluate(original.vector(retained), monotonic() + 10)[0]
    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "rhs,sensitive",
    [
        ("-theta*x+z+r", False),
        ("-x/theta+z+r", True),
        ("-exp(theta)*x+z+r", True),
        ("-theta**2*x+z+r", True),
    ],
)
@pytest.mark.parametrize("value", [1e-5, 1e9])
def test_structural_scales_do_not_depend_on_parameter_names(rhs, sensitive, value):
    payload = problem().model_dump(mode="json")
    payload["base_candidate"]["state_equations"][0]["rhs"] = rhs
    payload["base_candidate"]["parameters"][0]["bounds"] = {
        "lower": 1e-12,
        "upper": 1e12,
    }
    model, data = PrefixFitProblem.model_validate(payload), training()
    original = TrainingOracle(model, data, NumericalPolicy())
    retained = original.start | {"theta": value}
    _, oracle = recovery.make_oracle(model, data, NumericalPolicy(), retained)
    vector = oracle.reduction.reduce(original.vector(retained))
    assert ("theta" in oracle.sensitive_names) is sensitive
    expected = min(value, 1) if sensitive else max(value, 1)
    assert oracle.local_coordinate_scales(vector)[0] == expected
    assert oracle.units[0] == 1  # Physical probe scaling never changes optimizer units.


def test_sensitive_process_dependencies_are_expanded():
    payload = problem().model_dump(mode="json")
    payload["base_candidate"]["processes"] = [{"name": "denom", "expression": "theta"}]
    payload["base_candidate"]["state_equations"][0]["rhs"] = "-x/denom+z+r"
    model, data = PrefixFitProblem.model_validate(payload), training()
    original = TrainingOracle(model, data, NumericalPolicy())
    assert "theta" in recovery.sensitive_parameters(original)


def check_result(vector, units, *, status="passed", error=None):
    return adequacy.DerivativeCheck(
        status=status,
        reduced_vector=tuple(vector),
        coordinate_units=tuple(units),
        policy=adequacy.Policy(
            precision_aware=True, local_adaptive=True, local_bounds_aware=True
        ),
        required_indices=tuple(range(len(vector))),
        checked_indices=tuple(range(len(vector))),
        omitted_indices=(),
        calls_started=1,
        calls_completed=1,
        jacobian_calls_started=1,
        residual_only_calls_started=0,
        elapsed_seconds=0.001,
        error=error,
        search_qualified=status == "passed",
    )


def test_precision_retry_keeps_failed_evidence_cost_and_uses_same_anchor(monkeypatch):
    model, data, retained = mapped_case()
    calls = []

    def check(oracle, vector, deadline, policy):
        calls.append(
            (oracle.precision_mode, oracle.reduction.receipt["anchored_coordinates"])
        )
        return check_result(
            vector, oracle.units, status="failed" if len(calls) == 1 else "passed"
        )

    monkeypatch.setattr(recovery, "_derivative", check)
    result = recovery.qualification(
        model, data, NumericalPolicy(), retained, 10, lambda _: None
    )
    assert result["passed"]
    assert result["precision_mode"] == "multiprecision"
    assert len(result["passes"]) == 2
    assert result["passes"][0]["derivative_check"]["status"] == "failed"
    assert result["passes"][1]["derivative_check"]["status"] == "passed"
    assert calls[0][1] == calls[1][1]
    assert result["timing"]["numerical_active_wall_seconds"] >= sum(
        p["elapsed_wall_seconds"] for p in result["passes"]
    )


def test_stably_wrong_derivative_is_not_qualified_after_retry(monkeypatch):
    model, data, retained = mapped_case()
    monkeypatch.setattr(
        recovery,
        "_derivative",
        lambda o, v, d, p: check_result(v, o.units, status="failed"),
    )
    result = recovery.qualification(
        model, data, NumericalPolicy(), retained, 10, lambda _: None
    )
    assert len(result["passes"]) == 2 and not result["passed"]


@pytest.mark.parametrize("raised", [True, False])
def test_contract_errors_are_not_retried_at_higher_precision(monkeypatch, raised):
    model, data, retained = mapped_case()

    def check(oracle, vector, deadline, policy):
        if raised:
            raise ValueError("invalid residual shape")
        return check_result(
            vector,
            oracle.units,
            status="incomplete",
            error="ValueError: invalid residual shape",
        )

    monkeypatch.setattr(recovery, "_derivative", check)
    result = recovery.qualification(
        model, data, NumericalPolicy(), retained, 10, lambda _: None
    )
    assert len(result["passes"]) == 1 and not result["passed"]


def test_callback_failure_cannot_start_retry(monkeypatch):
    model, data, retained = mapped_case()
    monkeypatch.setattr(
        recovery,
        "_derivative",
        lambda o, v, d, p: check_result(v, o.units, status="inconclusive"),
    )

    def failed(_):
        raise OSError("disk full")

    result = recovery.qualification(
        model, data, NumericalPolicy(), retained, 10, failed
    )
    assert len(result["passes"]) == 1 and not result["passed"]
    assert result["timing"]["callback_failures"] == 1


def test_diagnosis_preserves_projection_and_precision(monkeypatch):
    model, data, retained = mapped_case()
    original, oracle = recovery.make_oracle(
        model, data, NumericalPolicy(), retained, precision_mode="multiprecision"
    )
    parameters = continuation.to_effective(original, oracle, retained)
    seen = []
    actual = recovery._derivative

    def check(o, v, d, p):
        seen.append((o.precision_mode, deepcopy(o.reduction.receipt)))
        return actual(o, v, d, p)

    monkeypatch.setattr(recovery, "_derivative", check)
    verification = continuation.verify_endpoint(
        original, oracle, parameters, 10, lambda _: None
    )
    result = recovery.diagnose(
        original,
        oracle,
        parameters,
        verification,
        {"stop_reason": "budget_limited", "budget_exhausted": True},
        10,
        lambda _: None,
    )
    assert result["status"] == "complete", result
    assert seen[0] == ("multiprecision", oracle.reduction.receipt)
    assert not result["original_parameter_identifiability_certified"]
    assert "Remaining active training coordinates" in result["assessment_scope"]


def test_invalid_precision_and_legacy_derivative_policy_rejected():
    with pytest.raises(ValueError, match="precision mode"):
        recovery.precision_policy("unknown")
    model, data, retained = mapped_case()
    with pytest.raises(ValueError, match="bounds-aware"):
        recovery.qualification(
            model,
            data,
            NumericalPolicy(),
            retained,
            1,
            lambda _: None,
            adequacy_policy=adequacy.Policy(),
        )


def test_dynamic_use_prevents_initial_only_elimination(monkeypatch):
    import casadi as ca

    model, data, _ = mapped_case()
    original = TrainingOracle(model, data, NumericalPolicy())
    system = original.system
    theta = ca.SX.sym("theta", len(system.names))
    state = ca.SX.sym("state", system.state_count)
    time = ca.SX.sym("time")
    inputs = ca.SX.sym("inputs", len(system.inputs))
    rhs = system.rhs(time, state, theta, inputs)
    rhs[0] += theta[system.names.index("init_r_a")]
    monkeypatch.setattr(
        system, "rhs", ca.Function("dynamic_use", [time, state, theta, inputs], [rhs])
    )
    proof = recovery.certify_inactive_initializers(original)
    assert "init_r_a" not in proof["removed_names"]
    assert {"name": "init_r_a", "reason": "used_in_dynamics_or_observations"} in proof[
        "retained"
    ]


def test_exhausted_outer_budget_starts_no_precision_retry(monkeypatch):
    model, data, retained = mapped_case()
    calls = []
    actual = recovery.make_oracle

    def build(*args, **kwargs):
        calls.append(kwargs["precision_mode"])
        return actual(*args, **kwargs)

    monkeypatch.setattr(recovery, "make_oracle", build)
    result = recovery.qualification(
        model, data, NumericalPolicy(), retained, 1e-9, lambda _: None
    )
    assert not result["passed"]
    assert result["status"] == "incomplete"
    assert not calls
