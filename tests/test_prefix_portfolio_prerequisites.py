"""Durable prerequisites, strict replay import and bounded BDF probe recovery."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import prefix_portfolio_prerequisites as p
from autoformalism.fitting import prefix_portfolio_qualification as old
from autoformalism.fitting import prefix_verification as verification
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.sensitivity_probe import SymbolicRolloutFailure
from tests.test_prefix_portfolio import composite_case
from tests.test_prefix_portfolio_qualification import LinearFixture


def seal(value):
    return {"sha256": content_sha256(value), "value": value}


def source_operation(oracle, method, residual):
    """Match the authenticated historical operation envelope, including identity."""
    value = {
        "protocol": old.PROTOCOL,
        "problem_sha256": oracle.portfolio_audit["problem_sha256"],
        "training_sha256": oracle.portfolio_audit["training_sha256"],
        "parameters_sha256": content_sha256(oracle.original.start),
        "numerical_policy": old.numerical_policy().model_dump(mode="json"),
        "method": method,
        "status": "complete",
        "state_only": True,
        "replay_completed": True,
        "calls_started": 1,
        "calls_completed": 1,
        "residual": residual.tolist(),
        "residual_sha256": adequacy._digest(residual),
        "training_nmse": float(np.mean(residual**2)),
    }
    return seal(
        {
            "status": "complete",
            "wall_seconds": 2.0,
            "cpu_seconds": 1.5,
            "identity": {
                "campaign": {"task": {"problem_sha256": value["problem_sha256"]}}
            },
            "value": value,
        }
    )


@pytest.fixture
def context():
    raw = LinearFixture()
    raw.original.start = deepcopy(raw.start)
    oracle = portfolio.PortfolioOracle(
        raw, {"problem_sha256": "model", "training_sha256": "train"}
    )
    residual = raw.evaluate(raw.vector(raw.start), monotonic() + 1, jacobian=False)[0]
    raw.calls.clear()
    sources = [
        source_operation(oracle, method, residual) for method in ("Radau", "BDF")
    ]
    return oracle, raw, sources


def prerequisites(context):
    oracle, _, sources = context
    baseline = p.import_baseline(oracle, *sources, 5, lambda _: None)
    jacobian = p.prepare_jacobian(oracle, baseline, 5, lambda _: None)
    assert baseline["passed"] and jacobian["passed"]
    return baseline, jacobian


def test_import_does_not_recharge_or_reexecute_historical_replays(context):
    oracle, raw, sources = context
    before = deepcopy(sources)
    result = p.import_baseline(oracle, *sources, 5, lambda _: None)
    assert result["passed"] and not raw.calls
    assert result["verification_calls_started"] == 0
    assert result["inherited_cost"]["verification_calls_started"] == 2
    assert result["inherited_cost"]["wall_seconds"] == 4.0
    assert result["identity"]["policy"]["qualification_method"] == "BDF"
    assert sources == before


@pytest.mark.parametrize(
    "corruption",
    [
        "seal",
        "data",
        "problem",
        "parameters",
        "method",
        "precision",
        "residual",
        "loss",
        "incomplete",
        "calls",
        "agreement",
    ],
)
def test_import_rejects_untrusted_or_incompatible_source(context, corruption):
    oracle, raw, sources = context
    broken = deepcopy(sources[0]["value"])
    value = broken["value"]
    if corruption == "seal":
        sources[0]["sha256"] = "bad"
    elif corruption in {"data", "problem", "parameters"}:
        value[
            {
                "data": "training_sha256",
                "problem": "problem_sha256",
                "parameters": "parameters_sha256",
            }[corruption]
        ] = "other"
    elif corruption == "method":
        value["method"] = "BDF"
    elif corruption == "precision":
        value["numerical_policy"]["rtol"] = 1e-9
    elif corruption == "residual":
        value["residual"][0] += 1
    elif corruption == "loss":
        value["training_nmse"] += 1
    elif corruption == "incomplete":
        broken["status"] = "incomplete"
    elif corruption == "calls":
        value["calls_completed"] = 0
    else:
        residual = np.asarray(value["residual"]) + 1e-5
        value.update(
            residual=residual.tolist(),
            residual_sha256=adequacy._digest(residual),
            training_nmse=float(np.mean(residual**2)),
        )
    if corruption != "seal":
        sources[0] = seal(broken)
    result = p.import_baseline(oracle, *sources, 5, lambda _: None)
    assert not result["passed"] and not raw.calls


def test_one_shared_jacobian_then_all_coordinate_probes_are_state_only(context):
    oracle, raw, _ = context
    baseline, jacobian = prerequisites(context)
    assert raw.calls == [(True, "BDF")]
    receipts = [
        p.qualify_coordinate(oracle, baseline, jacobian, index, 5, lambda _: None)
        for index in range(2)
    ]
    assert all(value["passed"] for value in receipts)
    assert all(not jac for jac, method in raw.calls[1:])
    assert {method for _, method in raw.calls} == {"BDF"}
    aggregate = p.aggregate_qualification(oracle, receipts)
    assert aggregate["passed"] and aggregate["complete"]
    assert not p.aggregate_qualification(oracle, receipts[:1])["search_qualified"]
    assert sum(r["verification_calls_started"] for r in receipts) == len(raw.calls) - 1


def test_failed_shared_prerequisite_blocks_every_dependent_without_solves(context):
    oracle, raw, sources = context
    baseline = p.import_baseline(oracle, *sources, 5, lambda _: None)

    def fail(*args, **kwargs):
        raise SymbolicRolloutFailure(
            "integration unavailable", {"trajectory_id": "row", "time": 0.5}
        )

    raw.evaluate = fail
    jacobian = p.prepare_jacobian(oracle, baseline, 5, lambda _: None)
    assert not jacobian["passed"] and jacobian["verification_calls_failed"] == 1
    assert jacobian["failure_diagnostic"]["trajectory_id"] == "row"
    before = deepcopy(oracle.counts)
    for index in range(2):
        receipt = p.qualify_coordinate(
            oracle, baseline, jacobian, index, 5, lambda _: None
        )
        assert receipt["status"] == "blocked"
        assert receipt["verification_calls_started"] == 0
    assert oracle.counts == before


@pytest.mark.parametrize(
    "corruption", ["identity", "baseline", "array", "status", "pairing"]
)
def test_invalid_jacobian_never_starts_coordinate_work(context, corruption):
    oracle, raw, _ = context
    baseline, jacobian = prerequisites(context)
    if corruption == "identity":
        jacobian["identity"]["protocol"] = "other"
    elif corruption == "baseline":
        jacobian["baseline_sha256"] = "other"
    elif corruption == "array":
        jacobian["jacobian"][0][0] += 1
    elif corruption == "status":
        jacobian["status"] = "incomplete"
    else:
        residual = np.asarray(jacobian["residual"]) + 1
        jacobian.update(
            residual=residual.tolist(), residual_sha256=adequacy._digest(residual)
        )
    before = list(raw.calls)
    result = p.qualify_coordinate(oracle, baseline, jacobian, 0, 5, lambda _: None)
    assert result["status"] == "blocked" and raw.calls == before


def test_nested_primitive_receipts_can_be_reused_without_new_calls(context):
    oracle, raw, _ = context
    baseline, jacobian = prerequisites(context)
    saved = {}

    def runner(key, vector, run):
        if key not in saved:
            saved[key] = run(lambda _: None)
        assert saved[key]["vector"] == vector
        return deepcopy(saved[key])

    first = p.qualify_coordinate(
        oracle, baseline, jacobian, 0, 5, lambda _: None, probe_runner=runner
    )
    assert first["passed"] and len(saved) >= 4
    before = list(raw.calls)
    second = p.qualify_coordinate(
        oracle, baseline, jacobian, 0, 5, lambda _: None, probe_runner=runner
    )
    assert second["passed"] and raw.calls == before
    assert second["verification_calls_started"] == 0
    assert (
        second["derivative_check"]["directions"]
        == first["derivative_check"]["directions"]
    )
    assert all(row["value"]["method"] == "BDF" for row in second["probe_receipts"])


def test_nested_primitive_wrong_vector_is_rejected(context):
    oracle, _, _ = context
    baseline, jacobian = prerequisites(context)

    def runner(key, vector, run):
        value = run()
        value["vector"]["a"] += 1
        return value

    result = p.qualify_coordinate(
        oracle, baseline, jacobian, 0, 5, lambda _: None, probe_runner=runner
    )
    assert not result["passed"]
    assert "identity or vector" in result["derivative_check"]["error"]


def test_nested_timeout_stops_coordinate_without_retrying_other_scales(context):
    oracle, raw, _ = context
    baseline, jacobian = prerequisites(context)
    calls = []

    def runner(key, vector, run):
        calls.append(key)
        value = run()
        return value | {
            "status": "incomplete",
            "passed": False,
            "error_type": "TimeoutError",
        }

    result = p.qualify_coordinate(
        oracle, baseline, jacobian, 0, 5, lambda _: None, probe_runner=runner
    )
    assert not result["passed"] and calls == ["probe_00"]
    assert len(raw.calls) == 2


@pytest.mark.parametrize("stage", ["import", "jacobian", "coordinate"])
def test_final_callback_failure_revokes_prerequisite_or_coordinate(context, stage):
    oracle, _, sources = context
    baseline, jacobian = prerequisites(context)

    def fail(value):
        if value.get("passed"):
            raise OSError("cannot seal passed evidence")

    if stage == "import":
        result = p.import_baseline(oracle, *sources, 5, fail)
    elif stage == "jacobian":
        result = p.prepare_jacobian(oracle, baseline, 5, fail)
    else:
        result = p.qualify_coordinate(oracle, baseline, jacobian, 0, 5, fail)
    assert not result["passed"] and result["status"] == "incomplete"


def test_probe_expiry_before_start_never_calls_solver(context):
    oracle, raw, _ = context
    with pytest.raises(TimeoutError):
        p.evaluate_probe(
            oracle, oracle.start, 90, None, parent_deadline=monotonic() - 1
        )
    assert not raw.calls


def test_nested_checkpoint_failure_stops_all_further_coordinate_calls(context):
    oracle, raw, _ = context
    baseline, jacobian = prerequisites(context)
    attempted = []

    def fail(receipt):
        if receipt.get("passed"):
            raise OSError("child journal unavailable")

    def runner(key, vector, run):
        attempted.append(key)
        return run(fail)

    result = p.qualify_coordinate(
        oracle, baseline, jacobian, 0, 5, lambda _: None, probe_runner=runner
    )
    assert not result["passed"]
    assert attempted == ["probe_00"] and len(raw.calls) == 2
    assert "checkpoint failed" in result["derivative_check"]["error"]


def test_prerequisite_identity_changes_cannot_mix_at_aggregate(context):
    oracle, _, _ = context
    baseline, jacobian = prerequisites(context)
    receipts = [
        p.qualify_coordinate(oracle, baseline, jacobian, index, 5, lambda _: None)
        for index in range(2)
    ]
    receipts[1]["jacobian_prerequisite_sha256"] = "other"
    with pytest.raises(ValueError, match="prerequisite references"):
        p.aggregate_qualification(oracle, receipts)


def test_probe_is_clipped_to_parent_remaining_time(context, monkeypatch):
    oracle, raw, _ = context
    clock = [100.0]
    for module in (p, portfolio, verification):
        monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    original = raw.evaluate

    def slow(vector, deadline, **kwargs):
        assert deadline <= 102.0
        clock[0] += 3.0
        return original(vector, deadline, **kwargs)

    raw.evaluate = slow
    result = p.evaluate_probe(
        oracle, oracle.start, 90, None, parent_deadline=102.0
    )
    assert not result["passed"] and len(raw.calls) == 1
    assert result["execution_policy"]["numerical_seconds"] == 2.0
    assert result["execution_policy"]["overall_seconds"] == 2.0
    assert result["verification_calls_started"] == 1
    assert result["verification_calls_failed"] == 1


def test_real_nonlinear_smoke_retains_chart_and_imports_old_state_replays():
    model, data = composite_case(True)
    oracle = p.make_oracle(model, data)
    sources = []
    for method in ("Radau", "BDF"):
        value = old.verify_start(model, data, method, 10, lambda _: None)
        operation = source_operation(oracle, method, np.asarray(value["residual"]))[
            "value"
        ]
        operation["value"] = value
        sources.append(seal(operation))
    baseline = p.import_baseline(oracle, *sources, 10, lambda _: None)
    jacobian = p.prepare_jacobian(oracle, baseline, 10, lambda _: None)
    assert baseline["passed"] and jacobian["passed"]
    checked = p.qualify_coordinate(oracle, baseline, jacobian, 0, 10, lambda _: None)
    assert checked["passed"], checked
    assert checked["derivative_check"]["policy"] == portfolio._policy().model_dump(
        mode="json"
    )
