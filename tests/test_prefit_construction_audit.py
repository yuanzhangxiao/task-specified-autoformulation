"""Fresh full-model construction audits: closed data boundary and exact resume."""

import copy
import json
from pathlib import Path

import pytest

from autoformalism.data import BenchmarkLoader
from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal import prefit_construction_campaign as campaign
from autoformalism.rebuttal.prefit_construction_audit import audit_task, reconstruct
from autoformalism.schemas.staged_topology import PublicScientificBrief
from scripts.smoke_prefit_construction import CELL, client_for, synthetic_fixture


@pytest.fixture
def no_numerics(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fitting/held-out data is forbidden")

    monkeypatch.setattr(campaign, "fit_candidate", forbidden)
    monkeypatch.setattr(BenchmarkLoader, "load_development", forbidden)
    monkeypatch.setattr(BenchmarkLoader, "load_test", forbidden)
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.name not in {"validation.csv", "test.csv"}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)


def build(root, budget=32):
    plan = synthetic_fixture(root, request_budget=budget, construction_only=True)
    task = plan["tasks"][0]
    calls = []
    result = campaign.construct_task(
        root, plan, task, client_for(root, plan, task, calls)
    )
    return plan, task, result, calls


def test_protocol_config_has_twelve_tasks_and_forbids_fit_settings():
    config = campaign.ConstructionCampaignConfig.model_validate_json(
        Path("configs/prefit_construction_audit_v1.json").read_text()
    )
    assert config.fit is None and config.seeds == (0, 1, 2)
    assert len(config.public_cells) * len(config.seeds) * len(campaign.ARMS) == 12
    payload = config.model_dump(mode="json")
    payload["fit"] = campaign.FitConfig().model_dump(mode="json")
    with pytest.raises(ValueError, match="requires fit=null"):
        campaign.ConstructionCampaignConfig.model_validate(payload)


def test_freeze_and_training_loader_never_open_held_out_files(
    tmp_path, monkeypatch, no_numerics
):
    source = tmp_path / "source"
    fixture = synthetic_fixture(source, construction_only=True)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(fixture["config"]))
    monkeypatch.setattr(
        campaign,
        "build_scientific_brief",
        lambda *a, **k: PublicScientificBrief.model_validate(
            fixture["cells"][CELL]["brief"]
        ),
    )
    root = tmp_path / "frozen"
    plan = campaign.freeze(config, source / "public", root)
    assert set(plan["cells"][CELL]["assets"]) == set(campaign.TRAINING_PUBLIC_FILES)
    assert not list(root.rglob("validation.csv"))
    assert campaign.freeze(config, source / "public", root) == plan
    assert campaign.verify(root) == plan
    training = root / "public/phase_b_v1" / CELL / "train.csv"
    training.write_text(training.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen public asset differs"):
        campaign.verify(root)
    with pytest.raises(Exception, match="fingerprint mismatch"):
        campaign.load_training(root / "public", CELL)


def test_complete_audit_handoff_and_resume_without_calls(tmp_path, no_numerics):
    plan, task, result, calls = build(tmp_path)
    audit = audit_task(tmp_path, plan, task)
    assert audit["status"] == "passed", audit
    assert all(audit["certificate"]["checks"].values())
    assert audit["handoff"]["candidate"] == result["functions"]["candidate"]
    assert audit["handoff"]["parameter_values_are_optimizer_guesses"]
    assert sum(s["preserved_without_atomic_call"] for s in audit["function_slots"]) == 2
    count = len(calls)
    assert (
        campaign.construct_task(
            tmp_path, plan, task, client_for(tmp_path, plan, task, calls)
        )
        == result
    )
    path = tmp_path / "results" / task["task_id"] / "model.md"
    before = path.read_bytes()
    path.unlink()  # Interrupted report generation can be safely completed on resume.
    assert audit_task(tmp_path, plan, task) == audit
    assert path.read_bytes() == before
    assert count == len(calls) == 6
    summary = campaign.summarize(tmp_path)
    assert summary["status"] == "partial"
    assert summary["arms"]["brief_only"]["audit_passed"] == 1
    assert summary["arms"]["training_evidence"]["pending_audit"] == 1
    assert "finite_validation" not in summary["arms"]["brief_only"]
    assert not list(tmp_path.rglob("fit*.json"))


@pytest.mark.parametrize("entry", ["task", "worker"])
def test_no_fit_entrypoint_even_after_complete_construction(
    tmp_path, entry, no_numerics
):
    plan, task, _, _ = build(tmp_path)
    with pytest.raises(ValueError, match="fitting is forbidden"):
        if entry == "task":
            campaign.fit_task(tmp_path, plan, task)
        else:
            campaign.run(tmp_path, "fit")
    assert not list(tmp_path.rglob("fit*.json"))


@pytest.mark.parametrize("budget", [1, 3, 5])
def test_construction_failure_retains_denominators_and_cost(
    tmp_path, budget, no_numerics
):
    plan, task, result, calls = build(tmp_path, budget=budget)
    assert result["status"] == "construction_failed"
    audit = audit_task(tmp_path, plan, task)
    assert audit["status"] == "not_constructed"
    assert audit["handoff"] is None
    summary = campaign.summarize(tmp_path)
    arm = summary["arms"]["brief_only"]
    assert arm["expected"] == 1 and arm["not_constructed"] == 1
    assert arm["physical_requests"] == len(calls) == budget
    assert audit_task(tmp_path, plan, task) == audit


def test_deferred_cross_stage_resume_and_audit_worker(tmp_path, no_numerics):
    plan = synthetic_fixture(tmp_path, construction_only=True)
    task = plan["tasks"][0]
    calls = []
    client = client_for(tmp_path, plan, task, calls)
    client.can_start = lambda: len(calls) < 3
    with pytest.raises(DeferredCall):
        campaign.construct_task(tmp_path, plan, task, client)
    assert audit_task(tmp_path, plan, task) is None
    assert campaign.summarize(tmp_path)["arms"]["brief_only"]["physical_requests"] == 3
    assert (
        campaign.construct_task(
            tmp_path, plan, task, client_for(tmp_path, plan, task, calls)
        )["status"]
        == "complete"
    )
    result = campaign.run(tmp_path, "audit")
    assert result["arms"]["brief_only"]["audit_passed"] == 1
    assert len(calls) == 6


@pytest.mark.parametrize(
    "damage,match",
    [
        ("target", "topology"),
        ("obligation", "obligation"),
        ("draft", "draft"),
        ("initial", "initializer"),
        ("slot", "coverage"),
        ("preserved", "already-valid"),
        ("canonical", "canonical function"),
    ],
)
def test_reconstruction_rejects_inconsistent_saved_artifacts(tmp_path, damage, match):
    plan, _, original, _ = build(tmp_path)
    changed = copy.deepcopy(original)
    f = changed["functions"]
    if damage == "target":
        changed["topology"]["topology"]["candidate_id"] = "changed"
    elif damage == "obligation":
        f["accepted_functions"][0]["selected_term"]["sources"] = ["u01"]
    elif damage == "draft":
        f["draft"]["latent_initials"] = []
    elif damage == "initial":
        f["initialization"]["guesses"] = {}
    elif damage == "slot":
        f["provider_visible_accepted_functions"].pop()
    elif damage == "preserved":
        f["batch_term_audits"][0]["atomic_repair_attempted"] = True
    elif damage == "canonical":
        f["accepted_functions"][0]["expression"] = "v01"
    with pytest.raises(ValueError, match=match):
        reconstruct(plan["cells"][CELL], changed)


def test_audit_failure_is_terminal_and_visible(tmp_path, monkeypatch):
    plan, task, _, _ = build(tmp_path)
    import autoformalism.rebuttal.prefit_construction_audit as module

    def broken(*args):
        raise ValueError("fixture reconstruction failure")

    monkeypatch.setattr(module, "reconstruct", broken)
    audit = audit_task(tmp_path, plan, task)
    assert audit["status"] == "failed" and audit["handoff"] is None
    summary = campaign.summarize(tmp_path)
    assert summary["arms"]["brief_only"]["constructed"] == 1
    assert summary["arms"]["brief_only"]["audit_failed"] == 1
    assert "fixture reconstruction failure" in summary["rows"][0]["audit_error"]
    assert audit_task(tmp_path, plan, task) == audit


@pytest.mark.parametrize("damage", ["missing_call", "response", "audit"])
def test_audit_resume_rejects_changed_cache_or_seal(tmp_path, damage):
    plan, task, _, _ = build(tmp_path)
    audit_task(tmp_path, plan, task)
    directory = tmp_path / "results" / task["task_id"]
    path = next((directory / "calls").glob("*.json"))
    if damage == "missing_call":
        path.unlink()
    else:
        path = directory / "audit.json" if damage == "audit" else path
        data = json.loads(path.read_text())
        data["status"] = "modified"
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=r"missing|differ"):
        campaign.summarize(tmp_path)
    with pytest.raises(ValueError, match=r"missing|differ"):
        audit_task(tmp_path, plan, task)


def test_read_only_report_does_not_authorize_changed_runtime(tmp_path, monkeypatch):
    plan, task, _, _ = build(tmp_path)
    audit_task(tmp_path, plan, task)
    monkeypatch.setattr(campaign, "runtime_source_hash", lambda: "different")
    with pytest.raises(ValueError, match="runtime differs"):
        campaign.summarize(tmp_path)
    summary = campaign.report(tmp_path)
    assert not summary["reporting"]["execution_authorized"]
    assert summary["arms"]["brief_only"]["audit_passed"] == 1


@pytest.mark.parametrize("bad_expression", ["-a*m+b*u01", "-a*****m+b*u01"])
def test_atomic_repair_records_rhs_and_role_changes_without_scientific_verdict(
    tmp_path, bad_expression, no_numerics
):
    plan = synthetic_fixture(tmp_path, construction_only=True)
    task = plan["tasks"][0]
    calls = []
    client = client_for(tmp_path, plan, task, calls)
    original_transport = client.transport

    def transport(url, body, timeout):
        response = original_transport(url, body, timeout)
        payload = calls[-1]
        reply = json.loads(response["choices"][0]["message"]["content"])
        if payload.get("selected_lhs", {}).get("name") == "m":
            reply["terms"][0]["scientific_role"] = "nonlinear driven relaxation"
        elif payload.get("selected_equation", {}).get("lhs") == "m":
            reply["functions"][0]["expression"] = bad_expression
        elif "selected_term" in payload:
            assert payload["selected_term"]["lhs"] == "m"
            reply = {
                "expression": "-a*m**2+b*u01",
                "parameters": [
                    {"name": "a", "role": "coefficient"},
                    {"name": "b", "role": "coefficient"},
                ],
            }
        response["choices"][0]["message"]["content"] = json.dumps(reply)
        return response

    client.transport = transport
    built = campaign.construct_task(tmp_path, plan, task, client)
    assert built["status"] == "complete", built
    audit = audit_task(tmp_path, plan, task)
    assert audit["status"] == "passed", audit
    repaired = [x for x in audit["function_slots"] if x["atomic_repair_attempted"]]
    assert len(repaired) == 1
    assert repaired[0]["rhs_changed"]
    assert repaired[0]["same_name_role_changes"] == {
        "a": {"before": "rate", "after": "coefficient"}
    }
    assert audit["certificate"]["nonlinear_obligation_count"] == 1
    assert not audit["scientific_judge_called"]
    assert "not scientific verdicts" in audit["limitation"]
    assert sum("selected_term" in payload for payload in calls) == 1
    assert sum(s["preserved_without_atomic_call"] for s in audit["function_slots"]) == 1


def test_audit_becomes_terminal_only_after_readable_report_is_published(
    tmp_path, monkeypatch
):
    plan, task, _, _ = build(tmp_path)
    directory = tmp_path / "results" / task["task_id"]
    original = campaign._sealed_write

    def interrupted(path, payload):
        assert path.name == "audit.json"
        assert "Assembled equations" in (directory / "model.md").read_text()
        assert not (directory / "model.md.tmp").exists()
        raise OSError("simulated interruption before audit commit")

    monkeypatch.setattr(campaign, "_sealed_write", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        audit_task(tmp_path, plan, task)
    assert not (directory / "audit.json").exists()
    monkeypatch.setattr(campaign, "_sealed_write", original)
    assert audit_task(tmp_path, plan, task)["status"] == "passed"
