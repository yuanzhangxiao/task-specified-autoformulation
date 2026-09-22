"""Saved-parent identity, preview review, cache resume, and fitter boundary."""

import json
import shutil
from copy import deepcopy

import pytest

from autoformalism.rebuttal import basin_repair_pilot as pilot
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import smoke_basin_repair_pilot as smoke
from scripts import submit_basin_repair_pilot as submit


@pytest.fixture(scope="module")
def source_pair(tmp_path_factory):
    return smoke.fixture(tmp_path_factory.mktemp("basin-repair-history"))


@pytest.fixture
def frozen(source_pair, tmp_path):
    source, gate = source_pair
    root = tmp_path / "repair"
    plan = pilot.freeze(source, gate, root)
    return root, plan


def test_freeze_resume_missing_report_and_audit_gate(source_pair, frozen, tmp_path):
    root, plan = frozen
    source, gate = source_pair
    assert pilot.freeze(source, gate, root) == plan
    assert pilot.report(root)["status_counts"] == {"missing": 16}
    clone = tmp_path / "bad-gate"
    shutil.copytree(gate, clone)
    path = clone / "summary.json"
    raw = sealed_read(path)
    raw.pop("artifact_sha256")
    raw["rows"] = []
    path.unlink()
    sealed_write(path, raw)
    with pytest.raises(ValueError, match="audit rows"):
        pilot.freeze(source, clone, tmp_path / "bad")
    with pytest.raises(ValueError, match="separate"):
        pilot.freeze(source, gate, source / "unsafe")


def test_preview_contains_rebuilt_model_confirm_then_resume(frozen):
    root, plan = frozen
    task = next(t for t in plan["tasks"] if t["case"] == "independent")
    calls = []
    c = smoke.client(root, plan, task, calls)
    p = pilot.run_one(root, plan, task, c)
    assert p["status"] == "confirmed" and len(calls) == 2
    assert calls[0]["assembled_model"] != calls[1]["assembled_model"]
    assert calls[1]["previous_transaction"]["transaction"]["affected_components"] == [
        "h_down"
    ]
    assert all(
        "normalized_mse" not in json.dumps(x) and "validation" not in x for x in calls
    )
    assert pilot.run_one(root, plan, task, c) == p and len(calls) == 2
    assert pilot.report(root)["physical_calls"] == 2
    # Resume before proposal publication uses the same cached requests/transactions.
    (root / "results" / task["task_id"] / "proposal.json").unlink()
    fresh = smoke.client(root, plan, task, calls)
    assert pilot.run_one(root, plan, task, fresh) == p and len(calls) == 2


def test_unconfirmed_edits_do_not_fit(frozen, monkeypatch):
    root, plan = frozen
    task = next(t for t in plan["tasks"] if t["case"] == "independent")
    calls = []
    p = pilot.run_one(
        root, plan, task, smoke.client(root, plan, task, calls, accept=False)
    )
    assert p["status"] == "unconfirmed_trial" and len(calls) == 3
    monkeypatch.setattr(
        pilot.sibling_fit,
        "prepare_child_fit",
        lambda *a, **k: pytest.fail("unexpected fit"),
    )
    assert pilot.fit_task(root, task["index"])["status"] == "unconfirmed_trial"


def test_inherited_model_acceptance_is_retention_without_refit(frozen):
    root, plan = frozen
    task = plan["tasks"][0]
    calls = []
    c = smoke.client(root, plan, task, calls)
    c.transport = lambda *args: {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "hypothesis": "Retain without claiming compliance.",
                            "accept_displayed": True,
                        }
                    )
                },
            }
        ]
    }
    p = pilot.run_one(root, plan, task, c)
    assert p["status"] == "retained"
    assert pilot.fit_task(root, task["index"])["status"] == "retained"


def test_static_violation_cannot_be_approved_and_old_point_is_not_gate():
    from tests.test_basin_equation_checks import SURVEY
    from tests.test_basin_model_repair import bundle

    b = bundle(independent=True)
    bad = deepcopy(b)
    tx = pilot.repair.apply(
        b,
        {
            "hypothesis": "bad",
            "equations": [{"component": "h_down", "expression": "inflow_down-h_down"}],
        },
        "independent",
    )
    bad = tx["bundle"]
    with pytest.raises(ValueError, match="static violations"):
        pilot._transition(
            bad,
            b,
            {"hypothesis": "approve", "accept_displayed": True},
            "independent",
            [SURVEY],
        )


def test_confirmed_proposal_tampering_is_detected(frozen):
    root, plan = frozen
    task = next(t for t in plan["tasks"] if t["case"] == "independent")
    pilot.run_one(root, plan, task, smoke.client(root, plan, task, []))
    path = root / "results" / task["task_id"] / "proposal.json"
    value = sealed_read(path)
    value.pop("artifact_sha256")
    value["bundle"]["candidate"]["state_equations"][0]["rhs"] = "0"
    path.unlink()
    sealed_write(path, value)
    with pytest.raises(ValueError, match="preview differs"):
        pilot.checked_proposal(root, plan, task)


def test_scheduler_four_jobs_and_no_duplicate_submit(frozen, tmp_path, monkeypatch):
    root, _plan = frozen
    for key in (
        "AF_PYTHON",
        "AF_VLLM_IMAGE",
        "AF_HF_HOME",
        "AF_COMPUTE_CACHE_ROOT",
        "AF_IPC_TMP_ROOT",
    ):
        p = tmp_path / key
        p.touch()
        monkeypatch.setenv(key, str(p))
    monkeypatch.setenv("AF_COMMIT", "a" * 40)
    monkeypatch.setattr(submit, "source_commit", lambda _: "a" * 40)
    calls = []

    def queue(directory, key, options, worker, stage, index):
        calls.append((stage, options))
        return str(100 + len(calls))

    monkeypatch.setattr(submit, "submit_job", queue)
    result = submit.submit(root)
    assert [c[0] for c in calls] == ["prepare", "propose", "fit", "report"]
    assert result["maximum_repair_calls"] == 48 and result["maximum_child_fits"] == 16
    assert "--array=0-15%4" in calls[2][1]
    assert submit.submit(root) == result and len(calls) == 4


def test_interrupt_after_cached_reply_resumes_without_reissuing(frozen):
    from autoformalism.llm.staged_topology import DeferredCall

    root, plan = frozen
    task = next(t for t in plan["tasks"] if t["case"] == "independent")
    calls = []
    c = smoke.client(root, plan, task, calls)
    c.can_start = lambda: not calls
    with pytest.raises(DeferredCall):
        pilot.run_one(root, plan, task, c)
    assert len(calls) == 1
    assert not (root / "results" / task["task_id"] / "proposal.json").exists()
    assert (
        pilot.run_one(root, plan, task, smoke.client(root, plan, task, calls))["status"]
        == "confirmed"
    )
    assert len(calls) == 2


def test_uncertain_delivery_is_charged_and_not_retried(frozen):
    root, plan = frozen
    task = next(t for t in plan["tasks"] if t["case"] == "independent")
    calls = []
    c = smoke.client(root, plan, task, calls)

    def interrupted(*args):
        raise KeyboardInterrupt()

    c.transport = interrupted
    with pytest.raises(KeyboardInterrupt):
        pilot.run_one(root, plan, task, c)
    directory = root / "results" / task["task_id"]
    assert len(list((directory / "calls").glob("*.json"))) == 1
    result = pilot.run_one(root, plan, task, smoke.client(root, plan, task, calls))
    assert result["status"] == "retained"
    # Only the next review request; first delivery stays uncertain.
    assert len(calls) == 1
    assert "interrupted before response" in result["events"][0]["error"]
    assert pilot.report(root)["physical_calls"] == 2


def test_child_seed_preserves_compatible_parameters(frozen):
    root, plan = frozen
    task = next(t for t in plan["tasks"] if t["case"] == "independent")
    proposal = pilot.run_one(root, plan, task, smoke.client(root, plan, task, []))
    parent = sealed_read(root / "baseline.json")["parents"][task["task_id"]]
    seed = pilot.sibling_fit.compatible_seed(
        pilot.request_for(parent["bundle"], plan, task, 0),
        pilot.request_for(proposal["bundle"], plan, task, 1),
        parent["result"]["fit"]["parameters"],
        pilot.PublicSplit.model_validate(plan["cells"][task["case"]]["training"]),
    )
    assert seed["initialization_plan_unchanged"]
    assert seed["fresh_parameters"] == ["outlet_rate"]
    for name in seed["retained_parameters"]:
        assert seed["parameters"][name] == parent["result"]["fit"]["parameters"][name]
