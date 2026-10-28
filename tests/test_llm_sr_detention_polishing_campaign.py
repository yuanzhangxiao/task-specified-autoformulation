"""Portable endpoint provenance, strict audit, and nonrenewable polishing receipts."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import llm_sr_detention_polishing_campaign as c
from tests.test_llm_sr_detention_probe_recovery_campaign import (
    direction,
    fake_gate,
    verified,
)
from tests.test_llm_sr_detention_refit import example


def checkpoint(parameters, *, usable=True):
    return verified(parameters, usable=usable) | {
        "status": "complete",
        "requested_parameters_sha256": public.content_sha256(parameters),
    }


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    item, context, training = example(15)
    monkeypatch.setattr(c, "_gate", fake_gate)

    def validate(inputs, case, gate):
        assert gate == fake_gate(inputs, case, case["parameters"], gate["directions"])

    def assess(item, context, training, parameters, *, qualification, **kwargs):
        assert kwargs["verification"]["parameters"] == parameters
        return {
            "status": "reliable_descent_remains"
            if qualification["complete"] and qualification["search_qualified"]
            else "unresolved",
            "parameters_sha256": public.content_sha256(parameters),
            "parameter_confidence_claimed": False,
        }

    monkeypatch.setattr(c, "_validate_qualification", validate)
    monkeypatch.setattr(c.polishing, "assess_endpoint", assess)
    monkeypatch.setattr(c.public, "_source_identity", lambda: "a" * 64)
    monkeypatch.setattr(c.public, "_runtime", lambda: {"testing": True})
    cases = []
    for index in range(2):
        point = {"a": 2.0 + index}
        case = {
            "index": index,
            "source_index": index,
            "start_id": "perturbed_0" if index == 0 else "generic_1",
            "mode": "original" if index == 0 else "reduced_model15_consistent",
            "parameters": point,
            "parent_verified_checkpoint": checkpoint(point),
            "source_verification_sha256": public.content_sha256(checkpoint(point)),
            "coordinate_names": ["a", "b"],
            "historical_cost": {"wall_seconds": 100},
            "source_receipts": {"row_sha256": "b" * 64},
            "numerical_policy": c.chart.Policy().model_dump(mode="json"),
        }
        case["qualification"] = fake_gate(
            None, case, point, [direction(point, coordinate) for coordinate in range(2)]
        )
        cases.append(case)
    raw = {
        "protocol": "llm-sr-detention-polishing-input-1",
        "model": item.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
        "training": training.model_dump(mode="json"),
        "cases": cases,
        "provenance": {"source_report_sha256": c.EXPECTED_SOURCE_REPORT_SHA256},
        "historical_costs": {"source": 1000},
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }
    monkeypatch.setattr(c, "EXPECTED_INPUTS_SHA256", public.content_sha256(raw))
    root = tmp_path / "polishing"
    c.prepare(root, raw)
    return root, raw


def mock_numerics(
    monkeypatch,
    *,
    improve=True,
    reject=False,
    interrupt=False,
    fit_error=False,
    unavailable_indices=(),
):
    calls = []

    def fit(item, context, training, parameters, **kwargs):
        calls.append(("fit", deepcopy(parameters), kwargs["seconds"]))
        assert kwargs["qualification"]["parameters"] == parameters
        assert kwargs["qualification"]["complete"]
        if fit_error:
            raise ValueError("search failed")
        point = {"a": parameters["a"] / 2 if improve else parameters["a"]}
        result = {
            "best": {"parameters": point, "training_nmse": point["a"] ** 2},
            "search_calls_started": 1,
            "stop_reason": "budget_exhausted",
            "initial_sample_reuse": {"residual": True, "columns": 2},
        }
        kwargs["checkpoint"](result)
        if interrupt:
            raise KeyboardInterrupt()
        return result

    def verify(item, context, training, parameters, **kwargs):
        calls.append(("verify", deepcopy(parameters), kwargs["seconds"]))
        result = checkpoint(parameters, usable=not reject)
        kwargs["checkpoint"](result)
        return result

    def probe(item, context, training, parameters, index, **kwargs):
        calls.append(
            (
                "direction",
                deepcopy(parameters),
                index,
                kwargs["seconds"],
                len(kwargs["seed_directions"]),
                kwargs["cache_root"],
            )
        )
        assert kwargs["search_only"] is False
        result = direction(
            parameters,
            index,
            status="unavailable" if index in unavailable_indices else "complete",
        )
        kwargs["checkpoint"](result)
        return result

    monkeypatch.setattr(c.polishing, "fit", fit)
    monkeypatch.setattr(c.original, "verify", verify)

    def strict_probe(*args, **kwargs):
        result = probe(*args, **kwargs)
        result["probe_recovery"] = {
            "protocol": c.recovery.PROTOCOL,
            "search_error_limits": None,
        }
        kwargs["checkpoint"](result)
        return result

    monkeypatch.setattr(c.recovery, "direction", strict_probe)
    return calls


def test_one_search_per_start_then_verify_and_strict_full_audit(prepared, monkeypatch):
    root, _ = prepared
    calls = mock_numerics(monkeypatch)
    context = c.load_run(root)
    first = c.run_task(root, 0, context=context)
    second = c.run_task(root, 1, context=context)
    assert first["retained"]["training_nmse"] == 1
    assert second["retained"]["training_nmse"] == 2.25
    assert first["diagnostic_status"] == "reliable_descent_remains"
    assert first["verified_improvement"]
    assert [r[0] for r in calls] == ["fit", "verify", "direction", "direction"] * 2
    assert [r[-1] for r in calls if r[0] == "fit"] == [600, 600]
    probes = [r for r in calls if r[0] == "direction"]
    assert all(r[3] == 360 and r[4] == 0 for r in probes)
    assert probes[0][-1] == probes[1][-1] != probes[2][-1]
    assert first["cost"]["reserved_allowance_seconds"] == 1440
    assert first["cost"]["available_initial_qualification_count"] == 1
    assert first["initial_sample_reuse"] == {"residual": True, "columns": 2}
    assert first["initial_training_nmse"] == 4
    assert not first["parameter_confidence_claimed"]
    assert c.run_task(root, 0) == first
    assert len(calls) == 8
    report = c.report(root)
    assert report["recorded"] == 2
    assert report["maximum_new_operation_budget_seconds"] == 2880


@pytest.mark.parametrize("reject,improve", [(True, True), (False, False)])
def test_retention_failure_and_no_gain_keep_own_start_and_audit_it(
    prepared, monkeypatch, reject, improve
):
    root, raw = prepared
    calls = mock_numerics(monkeypatch, reject=reject, improve=improve)
    row = c.run_task(root, 1)
    assert row["retained"] == raw["cases"][1]["parent_verified_checkpoint"]
    assert not row["verified_improvement"]
    assert row["status"] == "complete"
    probes = [r for r in calls if r[0] == "direction"]
    assert len(probes) == 2
    assert all(r[1] == {"a": 3.0} and r[4] == 2 for r in probes)


def test_search_failure_still_strictly_audits_original(prepared, monkeypatch):
    root, raw = prepared
    calls = mock_numerics(monkeypatch, fit_error=True)
    row = c.run_task(root, 0)
    assert row["retained"] == raw["cases"][0]["parent_verified_checkpoint"]
    assert [r[0] for r in calls] == ["fit", "direction", "direction"]
    assert row["operations"]["search"]["status"] == "unavailable"
    assert row["optimization_performed"] is None


def test_interrupted_search_saved_best_is_verified_without_search_renewal(
    prepared, monkeypatch
):
    root, _ = prepared
    calls = mock_numerics(monkeypatch, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 0)
    path = root / "tasks/0/operations/search/result.json"
    receipt = read_seal(path)
    assert receipt["status"] == "interrupted"
    assert c.report(root)["tasks"][0]["retained"]["training_nmse"] == 4
    later = mock_numerics(monkeypatch)
    row = c.run_task(root, 0)
    assert row["retained"]["training_nmse"] == 1
    assert len(calls) == 1
    assert [r[0] for r in later] == ["verify", "direction", "direction"]
    assert read_seal(path) == receipt
    assert c.run_task(root, 0) == row
    assert len(later) == 3


def test_interrupted_final_audit_preserves_verified_gain_and_does_not_repeat(
    prepared, monkeypatch
):
    root, _ = prepared
    calls = mock_numerics(monkeypatch)
    ordinary = c.recovery.direction

    def interrupt(*args, **kwargs):
        if args[4] == 1:
            raise KeyboardInterrupt()
        return ordinary(*args, **kwargs)

    monkeypatch.setattr(c.recovery, "direction", interrupt)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 0)
    assert c.report(root)["tasks"][0]["retained"]["training_nmse"] == 1
    later = mock_numerics(monkeypatch)
    row = c.run_task(root, 0)
    assert not later
    assert sum(r[0] == "fit" for r in calls) == 1
    assert row["status"] == "complete"
    assert row["diagnostic_status"] == "unresolved"
    assert row["final_qualification"]["missing_indices"] == [1]


def reserve(root, index, name):
    plan, _, policy = c.verify(root)
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = root / f"tasks/{index}/operations/{name}"
    seal(
        folder / "started.json",
        {
            "campaign": identity,
            "stage": name,
            "allowance_seconds": c._allowances(task, policy)[name],
        },
    )
    return folder


def test_abandoned_search_reservation_is_consumed_and_never_renews(
    prepared, monkeypatch
):
    root, _ = prepared
    reserve(root, 0, "search")
    calls = mock_numerics(monkeypatch)
    row = c.run_task(root, 0)
    assert [r[0] for r in calls] == ["direction", "direction"]
    assert row["operations"]["search"]["budget_charge_seconds"] == 600
    assert row["optimization_performed"] is None
    assert row["cost"]["wall_seconds"] is None
    assert row["cost"]["reserved_allowance_seconds"] == 1320


@pytest.mark.parametrize(
    "field,value",
    [
        ("segments", 2),
        ("segment_seconds", 601),
        ("segment_seconds", 0),
        ("verification_seconds", 121),
        ("direction_seconds", 361),
        ("direction_seconds", 239),
    ],
)
def test_budget_limits_rejected(field, value):
    with pytest.raises(ValueError):
        c.Policy(**{field: value})


@pytest.mark.parametrize("field", ["model", "training", "policy", "source", "seed"])
def test_portable_input_content_pins_cover_all_evidence(prepared, field):
    _, raw = prepared
    changed = deepcopy(raw)
    if field == "model":
        changed["model"]["parameters"]["C"] = 0.5
    elif field == "training":
        changed["training"]["rows"][0]["targets"]["h_down"][0] += 0.5
    elif field == "policy":
        changed["cases"][0]["numerical_policy"]["search_gradient_absolute_error"] *= 2
    elif field == "source":
        changed["provenance"]["source_report_sha256"] = "f" * 64
    else:
        changed["cases"][0]["qualification"]["directions"][0]["qualified"] = False
    with pytest.raises(ValueError, match="source extraction differs"):
        c.Inputs.model_validate(changed)


def test_unverified_source_checkpoint_cannot_be_rebound(prepared, monkeypatch):
    _, raw = prepared
    raw["cases"][0]["parent_verified_checkpoint"]["requested_parameters_sha256"] = (
        "f" * 64
    )
    monkeypatch.setattr(c, "EXPECTED_INPUTS_SHA256", public.content_sha256(raw))
    with pytest.raises(ValueError, match="matching independent verification"):
        c.Inputs.model_validate(raw)


def test_report_is_pure_receipt_reconstruction_and_runtime_independent(
    prepared, monkeypatch
):
    root, _ = prepared
    mock_numerics(monkeypatch)
    c.run_task(root, 0)

    def forbidden(*args, **kwargs):
        pytest.fail("numerical work during report")

    monkeypatch.setattr(c.polishing, "fit", forbidden)
    monkeypatch.setattr(c.original, "verify", forbidden)
    monkeypatch.setattr(c.recovery, "direction", forbidden)
    monkeypatch.setattr(c.public, "_source_identity", lambda: "e" * 64)
    assert c.report(root)["recorded"] == 1
    with pytest.raises(ValueError, match="source/runtime differs"):
        c.run_task(root, 1)
    assert not (root / "tasks/1").exists()


def test_receipt_progress_mutation_is_rejected(prepared, monkeypatch):
    root, _ = prepared
    mock_numerics(monkeypatch)
    c.run_task(root, 0)
    path = root / "tasks/0/operations/search/progress.json"
    value = public._read(path)
    value["best"]["parameters"]["a"] = 0.01
    public._write(path, value)
    with pytest.raises(ValueError, match="checkpoint differs"):
        c.report(root)


def test_extra_operations_and_out_of_order_verification_rejected(prepared, monkeypatch):
    root, _ = prepared
    calls = mock_numerics(monkeypatch)
    reserve(root, 0, "verification")
    with pytest.raises(ValueError, match="fixed polishing sequence"):
        c.run_task(root, 0)
    assert not calls
    (root / "tasks/1/operations/second-search").mkdir(parents=True)
    with pytest.raises(ValueError, match="unknown polishing operation"):
        c.run_task(root, 1)


def test_search_or_endpoint_binding_tampering_rejected(prepared, monkeypatch):
    root, _ = prepared
    mock_numerics(monkeypatch)
    row = c.run_task(root, 0)
    plan, inputs, _ = c.verify(root)
    operations = deepcopy(row["operations"])
    operations["search"]["value"]["source_parameters_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="approved initial endpoint"):
        c._state(inputs, plan["tasks"][0], operations)
    operations = deepcopy(row["operations"])
    operations["endpoint-direction-0"]["value"]["binding"]["strict_endpoint_audit"] = (
        False
    )
    with pytest.raises(ValueError, match="point or policy differs"):
        c._state(inputs, plan["tasks"][0], operations)


def test_verified_context_detects_portable_input_mutation(prepared, monkeypatch):
    root, _ = prepared
    context = c.load_run(root)
    context.inputs.cases[0]["parameters"]["a"] = 0
    calls = mock_numerics(monkeypatch)
    with pytest.raises(ValueError, match="verified invocation differs"):
        c.run_task(root, 0, context=context)
    assert not calls


def test_binding_only_checkpoint_does_not_claim_search_work(prepared, monkeypatch):
    root, _ = prepared
    calls = mock_numerics(monkeypatch)

    def interrupt(*args, **kwargs):
        kwargs["checkpoint"]({})
        raise KeyboardInterrupt()

    monkeypatch.setattr(c.polishing, "fit", interrupt)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 0)
    row = c.report(root)["tasks"][0]
    assert row["optimization_performed"] is None
    assert row["optimization_status"] == "search_reserved_execution_unconfirmed"
    assert row["initial_sample_reuse"] == {}
    assert not calls
    row = c.run_task(root, 0)
    assert row["optimization_performed"] is None
    assert [r[0] for r in calls] == ["direction", "direction"]


def test_verification_failure_keeps_start_and_still_runs_full_audit(
    prepared, monkeypatch
):
    root, raw = prepared
    calls = mock_numerics(monkeypatch)

    def fail(*args, **kwargs):
        raise TimeoutError("independent verification exhausted")

    monkeypatch.setattr(c.original, "verify", fail)
    row = c.run_task(root, 0)
    assert row["retained"] == raw["cases"][0]["parent_verified_checkpoint"]
    assert row["operations"]["verification"]["status"] == "unavailable"
    assert [r[0] for r in calls] == ["fit", "direction", "direction"]
    assert row["cost"]["reserved_allowance_seconds"] == 1440


def test_source_qualification_is_passed_exactly_and_never_mutated(
    prepared, monkeypatch
):
    root, raw = prepared
    mock_numerics(monkeypatch)
    ordinary = c.polishing.fit
    expected = deepcopy(raw["cases"][0]["qualification"])

    def check(*args, **kwargs):
        assert kwargs["qualification"] == expected
        return ordinary(*args, **kwargs)

    monkeypatch.setattr(c.polishing, "fit", check)
    c.run_task(root, 0)
    assert read_seal(root / "inputs.json")["cases"][0]["qualification"] == expected


def test_search_only_direction_cannot_replace_strict_final_audit(prepared, monkeypatch):
    root, _ = prepared
    mock_numerics(monkeypatch)
    row = c.run_task(root, 0)
    plan, inputs, _ = c.verify(root)
    operations = deepcopy(row["operations"])
    operations["endpoint-direction-0"]["value"]["direction"]["probe_recovery"][
        "search_error_limits"
    ] = [1e-5, 1e-2]
    with pytest.raises(ValueError, match="did not use strict auditing"):
        c._state(inputs, plan["tasks"][0], operations)


def test_export_rejects_source_report_or_receipt_mismatch(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "report/summary.json"
    path.parent.mkdir()
    public._write(path, {"status": "complete"})
    with pytest.raises(ValueError, match="source report file differs"):
        c._extract(source)
    monkeypatch.setattr(
        c, "EXPECTED_SOURCE_REPORT_FILE_SHA256", c.sha256(path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        c, "EXPECTED_SOURCE_REPORT_SHA256", public.content_sha256(public._read(path))
    )
    monkeypatch.setattr(c.source, "report", lambda _: {"status": "different"})
    with pytest.raises(ValueError, match="exactly reconstruct"):
        c._extract(source)
