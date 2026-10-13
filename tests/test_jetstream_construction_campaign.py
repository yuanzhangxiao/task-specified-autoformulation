"""Local serving provenance, isolated pilot plans and actual constructor resume."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import jetstream_construction_campaign as pilot
from scripts import jetstream_construction_probe as probe
from tests.test_construction_comparison import source_fixture
from tests.test_construction_ledger import variable
from tests.test_explicit_variable_bindings import response
from tests.test_jetstream_construction_probe import fixture as probe_fixture
from tests.test_jetstream_construction_probe import transport
from tests.test_phase_c_construction_baseline import tokenize


def fixture(tmp_path):
    """Create synthetic public inputs and four measured, valid compatibility calls."""
    source_fixture(tmp_path)
    inputs, server, _ = probe_fixture(tmp_path)
    work = tmp_path / "compatibility"
    work.mkdir()
    weights = work / "weights.json"
    weights.write_text("{}\n")
    record = json.loads(server.read_text())
    record["platform"] = pilot.PLATFORM
    record["weights_manifest_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    server = work / "server.json"
    server.write_text(json.dumps(record))
    probe.run(inputs, server, work / "probe", transport=transport([]))
    return tmp_path / "new", work, inputs, tmp_path / "pilot"


def reseal(path, value):
    value.pop("artifact_sha256", None)
    path.unlink()
    sealed_write(path, value)


def test_fresh_plan_changes_only_platform_and_records_provenance(tmp_path):
    args = fixture(tmp_path)
    source, work, _, root = args
    before = (source / "plan.json").read_bytes()
    old_probe = (work / "probe/plan.json").read_bytes()
    expected = pilot.campaign.freeze(
        source, tmp_path / "aces", study="focused_prompt_confirmation", seeds=[0]
    )
    plan = pilot.prepare(*args)
    assert pilot.verify(root) == plan
    assert plan["tasks"] == expected["tasks"] and len(plan["tasks"]) == 8
    assert plan["cells"] == expected["cells"]
    changed = {k for k in plan["config"] if plan["config"][k] != expected["config"][k]}
    assert changed == {"platform", "serving_image_sha256"}
    execution = sealed_read(root / "execution.json")
    assert execution["compatibility"]["physical_requests"] == 4
    assert execution["plan_sha256"] == plan["artifact_sha256"]
    assert not execution["test_data_opened"]
    assert pilot.prepare(*args) == plan
    assert (source / "plan.json").read_bytes() == before
    assert (work / "probe/plan.json").read_bytes() == old_probe


@pytest.mark.parametrize("failure", ["server", "weights", "request", "usage", "reply"])
def test_incompatible_evidence_blocks_plan_before_construction(tmp_path, failure):
    args = fixture(tmp_path)
    _, work, _, root = args
    if failure == "server":
        p = work / "server.json"
        value = json.loads(p.read_text())
        value["revision"] = "other"
        p.write_text(json.dumps(value))
    elif failure == "weights":
        (work / "weights.json").write_text('{"changed": true}')
    else:
        p = work / "probe/call-0.json"
        value = sealed_read(p)
        if failure == "request":
            value["request"]["body"]["seed"] += 1
        elif failure == "usage":
            value["raw_response"].pop("usage")
        else:
            value["raw_response"]["choices"][0]["message"]["content"] = "{}"
        reseal(p, value)
    with pytest.raises(ValueError):
        pilot.prepare(*args)
    assert not (root / "plan.json").exists()


def test_changed_settings_or_launcher_cannot_resume(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    pilot.prepare(*args)
    monkeypatch.setattr(pilot, "operation_identity", lambda: "changed")
    with pytest.raises(ValueError, match="execution identity"):
        pilot.verify(args[-1])
    with pytest.raises(ValueError, match="frozen artifact differs"):
        pilot.prepare(*args)


def test_model_settings_and_override_scope(tmp_path):
    args = fixture(tmp_path)
    source = args[0]
    old = sealed_read(source / "plan.json")
    settings = deepcopy(old["config"])
    settings["model_settings"]["temperature"] = 0.5
    with pytest.raises(ValueError, match="settings differ"):
        pilot.validate_settings(settings)
    with pytest.raises(ValueError, match="serving override"):
        pilot.campaign.freeze(
            source,
            tmp_path / "bad",
            serving_environment={"platform": pilot.PLATFORM},
        )


def test_actual_constructor_reports_bounded_failures_and_resumes_without_calls(
    tmp_path, monkeypatch
):
    args = fixture(tmp_path)
    pilot.prepare(*args)
    root = args[-1]
    calls = []

    def send(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        calls.append(body)
        return response(
            {
                **packet["response_template"],
                "variables": [variable("y")],
                "stage_complete": True,
            }
        )

    original = pilot.campaign.propose

    def propose(*a, **kw):
        return original(*a, **kw, transport=send, token_transport=tokenize)

    monkeypatch.setattr(pilot, "live_server", lambda: None)
    monkeypatch.setattr(pilot.campaign, "propose", propose)
    first = pilot.run(root)
    assert first["all_tasks_terminal"]
    assert first["status_counts"] == {"topology_incomplete": 8}
    assert first["optimizer_calls"] == 0
    assert not first["test_data_opened"]
    assert first["physical_requests"] == len(calls) > 0
    count = len(calls)
    assert pilot.run(root) == first
    assert len(calls) == count
    assert all((root / p).exists() for p in ("VARIABLES.md", "SHARED_PROCESSES.md"))


@pytest.mark.parametrize("error", [DeferredCall("draining"), OSError("storage failed")])
def test_deferred_or_failed_worker_still_reports_pending(tmp_path, monkeypatch, error):
    args = fixture(tmp_path)
    pilot.prepare(*args)
    root = args[-1]

    def fail(*_, **__):
        raise error

    monkeypatch.setattr(pilot, "live_server", lambda: None)
    monkeypatch.setattr(pilot.campaign, "propose", fail)
    if isinstance(error, DeferredCall):
        assert pilot.run(root)["status_counts"] == {"pending": 8}
    else:
        with pytest.raises(OSError, match="storage failed"):
            pilot.run(root)
    summary = json.loads((root / "summary.json").read_text())
    assert summary["status_counts"] == {"pending": 8}
    assert summary["physical_requests"] == 0


def test_launcher_does_not_install_or_launch_gpu_server():
    shell = (Path(pilot.REPO) / "scripts/jetstream/run_construction.sh").read_text()
    assert "docker run" not in shell and "apt-get" not in shell
    assert "nohup" in shell and "flock" in shell
    assert '"$AF_WORK/worker.lock"' in shell
    assert "--source" in shell and "--requests" in shell


@pytest.mark.parametrize(
    "study,policy",
    [
        ("readable_prompt_confirmation", "minimal-readable-construction-1"),
        ("completion_feedback_confirmation", "minimal-completion-feedback-1"),
        ("public_coverage_confirmation", "minimal-public-coverage-1"),
        ("coverage_decision_confirmation", "minimal-public-coverage-2"),
        ("edit_feedback_confirmation", "minimal-edit-feedback-1"),
    ],
)
def test_readable_five_seed_roster_is_frozen_and_old_plan_cannot_be_repurposed(
    tmp_path,
    study,
    policy,
):
    args = fixture(tmp_path)
    plan = pilot.prepare(*args, study=study, seeds=[0, 1, 2, 3, 4])
    assert len(plan["tasks"]) == 40
    assert plan["config"]["bookkeeping_policy"] == policy
    assert plan["config"]["repair_requests"] == 5
    assert pilot.verify(args[-1]) == plan
    assert pilot.prepare(*args, study=study, seeds=[0, 1, 2, 3, 4]) == plan
    with pytest.raises(ValueError, match="frozen artifact differs"):
        pilot.prepare(*args, study=study, seeds=[0])


@pytest.mark.parametrize(
    "study,suffix",
    [
        ("coverage_decision_confirmation", "_minimal_decisions"),
        ("edit_feedback_confirmation", "_minimal_edit_feedback"),
    ],
)
def test_decision_study_defaults_to_fresh_five_seed_roster_with_fixed_budgets(
    tmp_path, study, suffix
):
    args = fixture(tmp_path)
    plan = pilot.prepare(*args, study=study)
    control = pilot.campaign.freeze(
        args[0],
        tmp_path / "coverage-control",
        study="public_coverage_confirmation",
        seeds=[0, 1, 2, 3, 4],
        serving_environment={
            "platform": pilot.PLATFORM,
            "serving_image_sha256": plan["config"]["serving_image_sha256"],
        },
    )
    direct = pilot.campaign.freeze(args[0], tmp_path / "decisions-default", study=study)
    assert len(plan["tasks"]) == len(direct["tasks"]) == 40
    assert plan["sampling_seeds"] == direct["sampling_seeds"] == [0, 1, 2, 3, 4]
    assert all(t["arm"] == "full" for t in plan["tasks"])
    assert all(t["task_id"].endswith(suffix) for t in plan["tasks"])
    assert not any("starting_checkpoint" in t for t in plan["tasks"])
    assert plan["cells"] == control["cells"]
    assert {k for k in plan["config"] if plan["config"][k] != control["config"][k]} == {
        "bookkeeping_policy"
    }
    assert pilot.verify(args[-1]) == plan
    assert pilot.prepare(*args, study=study) == plan
    with pytest.raises(ValueError, match="frozen artifact differs"):
        pilot.prepare(
            *args, study="public_coverage_confirmation", seeds=[0, 1, 2, 3, 4]
        )
