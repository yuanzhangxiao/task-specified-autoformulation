"""Fixed qualified objective, authenticated cache and distinct partial diagnostics."""

from copy import deepcopy
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import prefix_portfolio_nonlinear_search as p
from autoformalism.fitting import prefix_portfolio_prerequisites as prerequisites
from autoformalism.fitting import public_fitting as public
from tests.test_prefix_portfolio import composite_case
from tests.test_prefix_portfolio_prerequisites import source_operation
from tests.test_prefix_portfolio_qualification import LinearFixture


@pytest.fixture
def context():
    raw = LinearFixture()
    raw.original.start = deepcopy(raw.start)
    oracle = portfolio.PortfolioOracle(
        raw, {"problem_sha256": "model", "training_sha256": "training"}
    )
    point = raw.vector(raw.start)
    residual = raw.evaluate(point, monotonic() + 5, jacobian=False)[0]
    sources = [
        source_operation(oracle, method, residual) for method in ("Radau", "BDF")
    ]
    baseline = prerequisites.import_baseline(oracle, *sources, 5, lambda _: None)
    jacobian = prerequisites.prepare_jacobian(oracle, baseline, 5, lambda _: None)
    coordinates = [
        prerequisites.qualify_coordinate(
            oracle, baseline, jacobian, i, 5, lambda _: None
        )
        for i in range(2)
    ]
    qualified = prerequisites.aggregate_qualification(oracle, coordinates)
    assert qualified["search_qualified"]
    fixed = p.FixedBDFOracle(portfolio.PortfolioOracle(raw, oracle.portfolio_audit))
    raw.calls.clear()
    return fixed, raw, baseline, jacobian, coordinates, qualified


def authenticate(context):
    oracle, _, baseline, jacobian, coordinates, qualified = context
    return p.authenticate_qualification(
        oracle, baseline, jacobian, coordinates, qualified
    )


def test_import_cache_is_zero_call_and_all_new_evaluations_use_bdf(context):
    oracle, raw, *_ = context
    receipt = authenticate(context)
    assert receipt["new_numerical_calls"] == 0
    point = oracle.vector(oracle.start)
    residual, derivative = oracle.evaluate(point, monotonic() + 5)
    assert derivative.shape == (3, 2) and not raw.calls
    residual[0] += 10
    assert oracle.evaluate(point, monotonic() + 5)[0][0] != residual[0]
    oracle.evaluate(point + 0.01, monotonic() + 5, jacobian=False)
    oracle.evaluate(point + 0.01, monotonic() + 5, jacobian=True)
    assert raw.calls == [(False, "BDF"), (True, "BDF")]
    assert oracle.execution_receipt()["imported_paired_start_cache_hits"] == 2
    with pytest.raises(ValueError, match="fixed BDF"):
        oracle.evaluate(point, monotonic() + 5, method="Radau")


@pytest.mark.parametrize(
    "corruption", ["qualification", "missing", "array", "binding", "start"]
)
def test_import_authenticates_complete_checks_and_actual_arrays(context, corruption):
    oracle, raw, baseline, jacobian, coordinates, qualified = context
    if corruption == "qualification":
        qualified["search_qualified"] = False
    elif corruption == "missing":
        coordinates.pop()
    elif corruption == "array":
        jacobian["jacobian"][0][0] += 1
    elif corruption == "binding":
        # Rehashing a modified Jacobian is not enough: checked derivatives must
        # match that exact cached matrix, not merely share coordinate coverage.
        derivative = np.asarray(jacobian["jacobian"]) * 2
        jacobian.update(
            jacobian=derivative.tolist(), jacobian_sha256=p.adequacy._digest(derivative)
        )
    else:
        oracle.start["a"] += 0.001
    with pytest.raises(ValueError):
        p.authenticate_qualification(oracle, baseline, jacobian, coordinates, qualified)
    assert not raw.calls and oracle._paired_start is None


def test_search_requires_qualification_and_changed_start_never_borrows_it(context):
    oracle, raw, *_ = context
    with pytest.raises(ValueError, match="authenticated full"):
        p.search(oracle, oracle.start, 5, 20, lambda _: None, qualification={})
    receipt = authenticate(context)
    changed = oracle.start | {"a": 0.11}
    with pytest.raises(ValueError, match="authenticated full"):
        p.search(oracle, changed, 5, 20, lambda _: None, qualification=receipt)
    assert not raw.calls
    result = p.search(
        oracle, oracle.start, 5, 20, lambda _: None, qualification=receipt
    )
    assert result["best"]["training_nmse"] < 1e-12
    assert result["fixed_search_method"] == "BDF"
    assert all(method == "BDF" for _, method in raw.calls)
    assert result["lazy_jacobian"]
    assert not result["newly_verified_endpoint"]
    assert result["qualified_start_training_nmse"] == pytest.approx(0.11333333333333334)


def test_settings_change_invalidates_imported_cache(context):
    oracle, raw, *_ = context
    authenticate(context)
    raw.original.settings["atol"] *= 10
    with pytest.raises(ValueError, match="identity changed"):
        oracle.evaluate(oracle.vector(oracle.start), monotonic() + 5)
    assert not raw.calls


def test_endpoint_verifier_rejects_saved_loss_mismatch_and_uses_original_radau(
    monkeypatch,
):
    class MockOracle:
        original = object()

    oracle = MockOracle()
    values = []

    def replay(original, actual, parameters, seconds, checkpoint):
        assert original is oracle.original and actual is oracle
        result = {
            "status": "complete",
            "training_nmse": 0.1,
            "original_equations_verified": True,
        }
        checkpoint(result)
        return result

    monkeypatch.setattr(p.portfolio, "verify_endpoint", replay)
    verified = p.verify_endpoint(oracle, {}, 5, values.append, expected_nmse=0.1)
    assert verified["original_equations_verified"]
    rejected = p.verify_endpoint(oracle, {}, 5, values.append, expected_nmse=1.0)
    assert not rejected["original_equations_verified"]
    assert not values[-1]["saved_loss_replay_passed"]
    assert rejected["methods"] == ["original_BDF", "original_Radau"]
    with pytest.raises(ValueError, match="nonnegative"):
        p.verify_endpoint(oracle, {}, 5, values.append, expected_nmse=float("nan"))


class PrecisionFixture(LinearFixture):
    names = tuple(f"p{i}" for i in range(24))
    lower, upper, units = np.full(24, -2.0), np.full(24, 2.0), np.ones(24)

    def __init__(self):
        super().__init__()
        self.start = dict.fromkeys(self.names, 0.1)
        self.original.settings = {"rtol": 1e-12, "atol": 1e-14}
        self.original.start = dict(self.start)
        self.original.vector = self.vector
        self.original.parameters = self.parameters
        self.original.evaluate = self.evaluate
        self.reduction = SimpleNamespace(lift=lambda x: x)

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        self.calls.append((jacobian, method))
        return vector.copy(), np.eye(len(self.names)) if jacobian else None


@pytest.fixture
def precision():
    raw = PrecisionFixture()
    oracle = portfolio.PortfolioOracle(
        raw,
        {
            "numerical_policy": p.precision_policy()["numerical_policy"],
            "problem_sha256": "model",
            "training_sha256": "train",
        },
    )
    bdf = p.precision_replay(oracle, "BDF", 5, lambda _: None)
    radau = p.precision_replay(oracle, "Radau", 5, lambda _: None)
    baseline = p.precision_baseline(oracle, bdf, radau)
    jacobian = p.precision_jacobian(oracle, baseline, 5, lambda _: None)
    assert baseline["passed"] and jacobian["passed"]
    return oracle, raw, baseline, jacobian


def test_precision_uses_fresh_pairing_and_exactly_four_scoped_directions(precision):
    oracle, raw, baseline, jacobian = precision
    assert raw.calls == [(False, "BDF"), (False, "Radau"), (True, "BDF")]
    journals = {}

    def journal(key, vector, run):
        receipt = run(lambda _: None)
        journals[public.content_sha256(vector)] = (key, receipt)
        return receipt

    receipts = [
        p.precision_coordinate(
            oracle, baseline, jacobian, i, 5, lambda _: None, probe_runner=journal
        )
        for i in p.PRECISION_INDICES
    ]
    assert all(row["passed"] for row in receipts), receipts
    assert all(method == "BDF" and not jac for jac, method in raw.calls[3:])
    assert sum(jac for jac, _ in raw.calls) == 1
    summary = p.precision_summary(oracle, receipts)
    assert summary["targeted_complete"]
    assert summary["partial_diagnostic_only"]
    assert not summary["full_search_qualification"] and not summary["search_qualified"]
    assert summary["missing_indices"]  # Remaining old checks are not transferred.
    assert journals
    with pytest.raises(ValueError, match="four declared"):
        p.precision_coordinate(oracle, baseline, jacobian, 0, 5, lambda _: None)
    with pytest.raises(ValueError, match="240"):
        p.precision_coordinate(oracle, baseline, jacobian, 8, 241, lambda _: None)


def test_precision_rejects_old_policy_and_mixed_prerequisites(precision):
    oracle, raw, baseline, jacobian = precision
    before = list(raw.calls)
    old = deepcopy(jacobian)
    old["identity"]["protocol"] = prerequisites.PROTOCOL
    with pytest.raises(ValueError, match="tighter prerequisite"):
        p.precision_coordinate(oracle, baseline, old, 8, 5, lambda _: None)
    assert raw.calls == before
    receipt = p.precision_coordinate(oracle, baseline, jacobian, 8, 5, lambda _: None)
    old_receipt = deepcopy(receipt)
    old_receipt["identity"]["protocol"] = prerequisites.PROTOCOL
    with pytest.raises(ValueError, match="identity"):
        p.precision_summary(oracle, [old_receipt])
    oracle.portfolio_audit["numerical_policy"]["rtol"] = 1e-11
    with pytest.raises(ValueError, match="fixed tighter"):
        p.precision_replay(oracle, "BDF", 5, lambda _: None)


def test_precision_nested_probe_corruption_is_blocking(precision):
    oracle, raw, baseline, jacobian = precision

    def corrupted(key, vector, run):
        value = run()
        value["vector_sha256"] = "wrong"
        return value

    result = p.precision_coordinate(
        oracle, baseline, jacobian, 8, 5, lambda _: None, probe_runner=corrupted
    )
    assert not result["passed"]
    assert result["verification_calls_started"] == 1
    assert len(raw.calls) == 4


def test_real_nonlinear_fixed_method_and_independent_replay():
    model, data = composite_case(True)
    oracle = p.make_oracle(model, data)
    point = oracle.vector(oracle.start)
    residual, derivative = oracle.evaluate(point, monotonic() + 10)
    assert np.isfinite(residual).all() and np.isfinite(derivative).all()
    verification = p.verify_endpoint(
        oracle,
        oracle.start,
        10,
        lambda _: None,
        expected_nmse=float(np.mean(residual**2)),
    )
    assert verification["original_equations_verified"]
    assert verification["methods"] == ["original_BDF", "original_Radau"]
