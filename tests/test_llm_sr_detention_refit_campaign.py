"""Authenticated numerical controls and independent, nonrenewable refit starts."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import llm_sr_detention_refit_campaign as c
from tests.test_llm_sr_control_campaign import policy as replay_policy
from tests.test_llm_sr_instantaneous_campaign import inputs as source_inputs


@pytest.fixture
def exported(tmp_path, monkeypatch):
    original = tmp_path / "source"
    monkeypatch.setattr(public, "_source_identity", lambda: "0" * 64)
    data = source_inputs(monkeypatch)
    c.source.prepare(original, data, replay_policy())
    for index in range(4):
        c.source.run_task(original, index)
    (original / "report").mkdir()
    public._write(original / "report/summary.json", c.source.report(original))
    snapshot = {p: p.read_bytes() for p in original.rglob("*.json")}
    destination = tmp_path / "delivery/inputs.json"
    info = c.export_inputs(original, destination)
    assert info["authenticated_parent_tasks"] == 4
    assert info["selected_source_tasks"] == [14, 15]
    assert snapshot == {p: p.read_bytes() for p in snapshot}
    return original, read_seal(destination)


def fake_contract(monkeypatch):
    def starts(item, context, training, *, seed):
        assert seed == 20261025
        return [
            {"start_id": label, "start_kind": kind, "parameters": {"a": value}}
            for label, kind, value in (
                ("supplied", "supplied", 0.7),
                ("perturbed_0", "perturbed", 1.0),
                ("perturbed_1", "perturbed", 1.3),
                ("generic_0", "generic", 3.0),
                ("generic_1", "generic", 4.0),
            )
        ]

    monkeypatch.setattr(c.kernel, "deterministic_starts", starts)
    monkeypatch.setattr(c.kernel, "domain_policy", lambda *args: {"bounds": "fixed"})


@pytest.fixture
def prepared(exported, tmp_path, monkeypatch):
    _, data = exported
    fake_contract(monkeypatch)
    root = tmp_path / "refit"
    c.prepare(
        root,
        data,
        c.Policy(
            optimization_seconds=1,
            verification_seconds=1,
            diagnostic_seconds=1,
        ),
    )
    return root, data


def fake_numerics(monkeypatch, *, interrupt=False, valid=True, missing=False):
    calls = []

    def verify(item, context, training, parameters, *, seconds, checkpoint, policy):
        calls.append(("verify", dict(parameters)))
        value = {
            "parameters": parameters,
            "training_nmse": (parameters["a"] - 0.7) ** 2,
            "usable_for_retention": valid,
            "alternate_training_nmse": (parameters["a"] - 0.7) ** 2,
            "solver_loss_discrepancy": 0.0,
            "solver_prediction_discrepancy": 0.0,
        }
        checkpoint(value)
        return value

    def fit(item, context, training, start, *, seconds, checkpoint, policy):
        calls.append(("fit", dict(start)))
        value = {"best": None, "stop_reason": "maximum_calls", "calls_started": 1}
        if not missing:
            value["best"] = {
                "parameters": {"a": start["a"] - 0.1},
                "training_nmse": (start["a"] - 0.8) ** 2,
            }
        checkpoint(value)
        if interrupt:
            raise KeyboardInterrupt()
        return value

    def diagnose(item, context, training, parameters, *, seconds, checkpoint, policy):
        calls.append(("diagnose", dict(parameters)))
        value = {"assessment": "local_information_only", "parameters": parameters}
        checkpoint(value)
        return value

    monkeypatch.setattr(c.kernel, "verify", verify)
    monkeypatch.setattr(c.kernel, "fit", fit)
    monkeypatch.setattr(c.kernel, "diagnose", diagnose)
    return calls


def test_frozen_ten_starts_use_both_models_and_keep_historical_costs(prepared):
    root, data = prepared
    plan, _, _ = c.verify(root)
    assert len(plan["tasks"]) == 10
    assert [t["source_task"] for t in plan["tasks"]] == [14] * 5 + [15] * 5
    assert [t["start_kind"] for t in plan["tasks"][:5]] == [
        "supplied",
        "perturbed",
        "perturbed",
        "generic",
        "generic",
    ]
    assert all(t["domain_policy"] == {"bounds": "fixed"} for t in plan["tasks"])
    report = c.report(root)
    assert report["expected"] == 10 and report["recorded"] == 0
    assert len(report["historical_costs"]) == 4
    assert [r["identity"]["task"] for r in report["supplied_reference"]] == [14, 15]
    assert report["historical_report_sha256"] == data["source_report_sha256"]
    assert report["model_selection_performed"] is False


def test_supplied_start_retains_its_better_verified_start(prepared, monkeypatch):
    root, _ = prepared
    calls = fake_numerics(monkeypatch)
    row = c.run_task(root, 0)
    assert row["retained"]["parameters"] == {"a": 0.7}
    assert row["retained_origin"] == "start_verification"
    assert calls[-1] == ("diagnose", {"a": 0.7})
    assert row["accuracy"] == "within_predeclared_training_threshold"
    assert row["optimization"]["stop_reason"] == "maximum_calls"
    assert row["convergence_is_accuracy"] is False
    before = deepcopy(calls)
    assert c.run_task(root, 0) == row
    assert calls == before


def test_generic_start_never_falls_back_to_supplied_reference(prepared, monkeypatch):
    root, _ = prepared
    calls = fake_numerics(monkeypatch)
    row = c.run_task(root, 3)
    assert calls[0] == ("verify", {"a": 3.0})
    assert calls[1] == ("fit", {"a": 3.0})
    assert row["retained"]["parameters"] == {"a": 2.9}
    assert row["retained"]["training_nmse"] > 1
    assert row["retained_origin"] == "endpoint_verification"
    assert not row["supplied_reference_used_as_fallback"]
    report = c.report(root)
    assert report["recorded"] == 1
    assert report["tasks"][3]["supplied_reference_training_nmse"] < 1e-12
    assert report["tasks"][3]["retained_to_supplied_nmse_ratio"] > 1e10


def test_failed_verification_does_not_block_kernel_recovery_or_import_reference(
    prepared, monkeypatch
):
    root, _ = prepared
    calls = fake_numerics(monkeypatch, valid=False)
    row = c.run_task(root, 4)
    assert any(name == "fit" for name, _ in calls)
    assert not any(name == "diagnose" for name, _ in calls)
    assert row["retained"] is None and row["accuracy"] == "unavailable"


def test_missing_fit_checkpoint_preserves_only_current_verified_start(
    prepared, monkeypatch
):
    root, _ = prepared
    calls = fake_numerics(monkeypatch, missing=True)
    row = c.run_task(root, 3)
    assert row["retained"]["parameters"] == {"a": 3.0}
    assert sum(name == "verify" for name, _ in calls) == 1


def test_interrupted_attempt_retains_checkpoint_and_never_renews_budget(
    prepared, monkeypatch
):
    root, _ = prepared
    calls = fake_numerics(monkeypatch, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 3)
    row = c.run_task(root, 3)
    assert sum(name == "fit" for name, _ in calls) == 1
    assert row["operations"]["optimization"]["status"] == "interrupted"
    assert row["retained"]["parameters"] == {"a": 2.9}
    assert row["historical_allowances_renewed"] is False


def test_orphaned_operation_reservation_is_charged_without_running(
    prepared, monkeypatch
):
    root, _ = prepared
    plan, _, policy = c.verify(root)
    identity = {"plan_sha256": public.content_sha256(plan), "task": plan["tasks"][3]}
    folder = root / "tasks/3/operations/optimization"
    seal(
        folder / "started.json",
        {
            "campaign": identity,
            "stage": "optimization",
            "allowance_seconds": 1.0,
        },
    )
    calls = fake_numerics(monkeypatch)
    before = c.report(root)["tasks"][3]
    assert before["cost"]["budget_charge_seconds"] == 1.0
    row = c.run_task(root, 3)
    assert not any(name == "fit" for name, _ in calls)
    op = row["operations"]["optimization"]
    assert op["status"] == "interrupted" and not op["accounting_complete"]
    assert op["budget_charge_seconds"] == policy.optimization_seconds


@pytest.mark.parametrize(
    "mutation", ["progress", "receipt", "terminal", "report", "training", "roster"]
)
def test_source_history_tampering_is_rejected(exported, mutation):
    _, data = exported
    changed = deepcopy(data)
    parent = changed["parents"][2]
    if mutation == "progress":
        parent["operations"]["primary"]["observed_progress"]["normalized_mse"] = 999
    elif mutation == "receipt":
        parent["operations"]["primary"]["cpu_seconds"] = 999
    elif mutation == "terminal":
        parent["terminal"]["solver_agreement"]["verified"] = False
    elif mutation == "report":
        changed["source_report"]["tasks"][2]["status"] = "invented"
    elif mutation == "training":
        changed["source_plan"]["inputs"]["tasks"][2]["training"]["name"] = "test"
    else:
        changed["parents"].reverse()
    with pytest.raises(ValueError):
        c.Inputs.model_validate(changed)


def test_export_refuses_inside_source_and_changed_report(exported, tmp_path):
    original, _ = exported
    with pytest.raises(ValueError, match="inside source"):
        c.export_inputs(original, original / "new.json")
    report = public._read(original / "report/summary.json")
    report["optimizer_calls"] = 1
    public._write(original / "report/summary.json", report)
    with pytest.raises(ValueError, match="unchanged source report"):
        c.export_inputs(original, tmp_path / "corrupt.json")


def test_changed_plan_or_runtime_rejected_before_fitting(prepared, monkeypatch):
    root, _ = prepared
    monkeypatch.setattr(
        c.kernel, "fit", lambda *a, **k: pytest.fail("unexpected fitting")
    )
    monkeypatch.setattr(c.public, "_runtime", lambda: {"changed": True})
    with pytest.raises(ValueError, match="source/runtime"):
        c.run_task(root, 0)


def test_diagnosis_is_bound_to_exact_retained_point(prepared, monkeypatch):
    root, _ = prepared
    fake_numerics(monkeypatch)
    row = c.run_task(root, 0)
    altered = deepcopy(row["operations"])
    altered["retained_diagnosis"]["value"]["diagnosed_parameters_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="diagnosis differs"):
        c._assemble(row["identity"], altered, c.Policy(), terminal=True)


def test_verified_endpoint_cannot_claim_another_start(prepared, monkeypatch):
    root, _ = prepared
    fake_numerics(monkeypatch)
    row = c.run_task(root, 3)
    altered = deepcopy(row["operations"])
    altered["endpoint_verification"]["value"]["parameters"] = {"a": 0.7}
    with pytest.raises(ValueError, match="verified parameters differ"):
        c._assemble(row["identity"], altered, c.Policy(), terminal=True)


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), -1.0])
def test_verified_retention_requires_finite_score(score):
    with pytest.raises(ValueError, match="finite training evidence"):
        c._verified(
            {
                "value": {
                    "usable_for_retention": True,
                    "training_nmse": score,
                    "parameters": {"a": 1.0},
                }
            }
        )


def test_nested_search_checkpoint_survives_hard_interrupt(prepared, monkeypatch):
    root, _ = prepared
    fake_numerics(monkeypatch)
    attempts = []

    def stopped(item, context, training, start, *, seconds, checkpoint, policy):
        attempts.append(start)
        checkpoint(
            {
                "best": {"parameters": start, "training_nmse": 5.0},
                "search": {"best": {"parameters": {"a": 2.0}, "training_nmse": 1.69}},
            }
        )
        raise KeyboardInterrupt()

    monkeypatch.setattr(c.kernel, "fit", stopped)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 3)
    row = c.run_task(root, 3)
    assert len(attempts) == 1
    assert row["retained"]["parameters"] == {"a": 2.0}
    assert row["retained_origin"] == "endpoint_verification"


def test_newer_search_base_checkpoint_beats_older_nested_search_checkpoint():
    latest = {"parameters": {"a": 1.0}, "training_nmse": 0.1}
    previous = {"parameters": {"a": 1.1}, "training_nmse": 0.2}
    assert (
        c._fit_checkpoint(
            {
                "observed_progress": {
                    "best": latest,
                    "search": {"best": previous},
                }
            }
        )["best"]
        == latest
    )
