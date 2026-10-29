"""A separately pinned continuation reuses endpoints without renewing old work."""

import os
import subprocess
from copy import deepcopy
from hashlib import sha256

import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import llm_sr_detention_staged_polishing_campaign as c
from scripts import run_llm_sr_detention_staged_polishing as launcher
from tests import test_llm_sr_detention_staged_polishing_campaign as previous

mock_numerics = previous.mock_numerics
prepared = previous.prepared


def pin_returned_report(monkeypatch, root, saved):
    """Pin this test's actual receipt-derived report, not a mocked report reader."""
    path = root / "report/summary.json"
    path.parent.mkdir(exist_ok=True)
    public._write(path, saved)
    monkeypatch.setattr(c, "STAGED_SOURCE_REPORT_SHA256", public.content_sha256(saved))
    monkeypatch.setattr(
        c, "STAGED_SOURCE_REPORT_FILE_SHA256", sha256(path.read_bytes()).hexdigest()
    )


@pytest.fixture
def completed_source(prepared, monkeypatch):
    """Build a complete v1 source using only deterministic fake numerical calls."""
    root, original = prepared
    mock_numerics(monkeypatch)
    for index in range(2):
        c.run_task(root, index)
    saved = c.report(root)
    pin_returned_report(monkeypatch, root, saved)
    raw = c._extract(root, source_stage="staged")
    monkeypatch.setattr(c, "STAGED_INPUTS_SHA256", public.content_sha256(raw))
    return root, original, saved, raw


def test_export_authenticates_own_receipts_and_preserves_original_profile(
    completed_source, tmp_path, monkeypatch
):
    root, original, saved, raw = completed_source
    old_pins = c._source_pins("polishing")

    def forbidden(*args, **kwargs):
        raise AssertionError("v2 extraction must not consult the older campaign")

    monkeypatch.setattr(c.source, "report", forbidden)
    monkeypatch.setattr(c.source, "verify", forbidden)
    destination = tmp_path / "portable.json"
    result = c.export_inputs(root, destination, source_stage="staged")
    exported = read_seal(destination)
    assert exported == raw
    assert result["sha256"] == public.content_sha256(exported)
    assert exported["protocol"] == "llm-sr-detention-staged-polishing-input-2"
    assert exported["provenance"]["source_report_exactly_reconstructed"]
    for index, case in enumerate(exported["cases"]):
        row = saved["tasks"][index]
        assert case["parameters"] == row["retained"]["parameters"]
        assert case["parent_verified_checkpoint"] == row["retained"]
        assert case["qualification"] == row["final_qualification"]
        assert case["historical_cost"] == row["cost"]
        assert case["source_receipts"]["operations_sha256"] == (
            public.content_sha256(row["operations"])
        )
    assert c._source_pins("polishing") == old_pins
    assert c.Inputs.model_validate(original).protocol.endswith("-1")
    assert c.verify(root, runtime=False)[1].model_dump(mode="json") == original
    assert c.report(root) == saved
    assert exported["historical_costs"]["source_tasks"] == [
        {"source_index": i, "cost": row["cost"]}
        for i, row in enumerate(saved["tasks"])
    ]
    assert exported["historical_costs"]["earlier_costs_sha256"] == (
        public.content_sha256(saved["historical_costs"])
    )
    assert exported["historical_costs"]["accounting_rule"] == (
        "separately labeled source scopes; not additive totals"
    )


@pytest.mark.parametrize("change", ["data", "parameters", "profile", "source"])
def test_v2_input_changes_fail_content_authentication(completed_source, change):
    _, _, _, raw = completed_source
    raw = deepcopy(raw)
    if change == "data":
        raw["training"]["rows"][0]["targets"]["h_down"][0] += 1
    elif change == "parameters":
        raw["cases"][0]["parameters"]["a"] += 1
    elif change == "profile":
        raw["protocol"] = "llm-sr-detention-staged-polishing-input-1"
    else:
        raw["provenance"]["source_report_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source extraction differs"):
        c.Inputs.model_validate(raw)


def test_export_rejects_changed_receipt_even_with_unchanged_report(completed_source):
    root, _, _, _ = completed_source
    path = root / "tasks/0/operations/search/result.json"
    receipt = read_seal(path)
    receipt["wall_seconds"] += 1
    public._write(path, {"sha256": public.content_sha256(receipt), "value": receipt})
    with pytest.raises(ValueError, match="terminal task differs"):
        c._extract(root, source_stage="staged")


def test_export_rejects_changed_report_bytes(completed_source):
    root, _, _, _ = completed_source
    path = root / "report/summary.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="source report file differs"):
        c._extract(root, source_stage="staged")


def test_report_content_pin_is_independent_of_file_pin(completed_source, monkeypatch):
    root, _, saved, _ = completed_source
    path = root / "report/summary.json"
    changed = deepcopy(saved)
    changed["tasks"][0]["retained"]["training_nmse"] *= 0.5
    public._write(path, changed)
    monkeypatch.setattr(
        c, "STAGED_SOURCE_REPORT_FILE_SHA256", sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="source report content differs"):
        c._extract(root, source_stage="staged")


def test_new_profile_cannot_overwrite_the_old_campaign(completed_source):
    root, _, saved, raw = completed_source
    with pytest.raises(ValueError, match="sealed artifact differs"):
        c.prepare(root, raw)
    assert c.report(root) == saved


def test_export_requires_two_completed_source_tasks(completed_source, monkeypatch):
    root, _, _, _ = completed_source
    (root / "tasks/1/result.json").unlink()
    incomplete = c.report(root)
    assert incomplete["recorded"] == 1
    pin_returned_report(monkeypatch, root, incomplete)
    with pytest.raises(ValueError, match="source continuation is not complete"):
        c._extract(root, source_stage="staged")


def test_v2_checkpoint_binding_is_checked_beyond_content_pin(
    completed_source, monkeypatch
):
    _, _, _, raw = completed_source
    raw = deepcopy(raw)
    raw["cases"][0]["parameters"]["a"] += 1
    monkeypatch.setattr(c, "STAGED_INPUTS_SHA256", public.content_sha256(raw))
    with pytest.raises(ValueError, match="matching independent verification"):
        c.Inputs.model_validate(raw)


def test_compact_prepare_needs_no_history_and_resume_allocates_one_new_search(
    completed_source, tmp_path, monkeypatch
):
    source_root, _, saved, raw = completed_source
    source_before = {p: p.read_bytes() for p in source_root.rglob("*.json")}
    root = tmp_path / "continuation"

    def forbidden(*args, **kwargs):
        raise AssertionError("portable preparation cannot reopen campaign history")

    monkeypatch.setattr(c, "_extract", forbidden)
    monkeypatch.setattr(c.source, "verify", forbidden)
    monkeypatch.setattr(c.source, "report", forbidden)
    identity = c.prepare(root, raw)
    assert c.prepare(root, raw) == identity
    plan, inputs, _ = c.verify(root)
    assert inputs.protocol.endswith("-2")
    assert plan["inputs_sha256"] == c.STAGED_INPUTS_SHA256
    assert plan["source_report_sha256"] == public.content_sha256(saved)
    assert [plan["operation_allowances"][str(i)]["search"] for i in range(2)] == [
        1800,
        1800,
    ]
    calls = mock_numerics(monkeypatch)
    rows = [c.run_task(root, i) for i in range(2)]
    assert [call[2] for call in calls if call[0] == "fit"] == [1800, 1800]
    assert [call[1] for call in calls if call[0] == "fit"] == [
        row["retained"]["parameters"] for row in saved["tasks"]
    ]
    count = len(calls)
    assert [c.run_task(root, i) for i in range(2)] == rows
    assert len(calls) == count
    for index, row in enumerate(rows):
        assert row["historical_cost"] == saved["tasks"][index]["cost"]
        assert row["cost"]["new_operation_count"] == 4
        assert row["cost"]["reserved_allowance_seconds"] == 2640
        assert not row["historical_allowances_renewed"]
        assert not row["strict_audit"]["performed"]
    assert c.report(root)["historical_costs"] == raw["historical_costs"]
    assert {p: p.read_bytes() for p in source_root.rglob("*.json")} == source_before


def test_interrupted_v2_search_does_not_receive_a_second_allowance(
    completed_source, tmp_path, monkeypatch
):
    _, _, _, raw = completed_source
    root = tmp_path / "continuation"
    c.prepare(root, raw)
    calls = mock_numerics(monkeypatch, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 0)
    receipt = read_seal(root / "tasks/0/operations/search/result.json")
    assert receipt["status"] == "interrupted"
    assert [call[2] for call in calls if call[0] == "fit"] == [1800]
    later = mock_numerics(monkeypatch)
    result = c.run_task(root, 0)
    assert result["status"] == "complete"
    assert [call[0] for call in later] == ["verify", "direction", "direction"]
    assert read_seal(root / "tasks/0/operations/search/result.json") == receipt


@pytest.mark.parametrize("stage", [None, "polishing", "staged"])
def test_cli_export_dispatches_explicit_stage(tmp_path, monkeypatch, capsys, stage):
    calls = []

    def export(source, destination, *, source_stage):
        calls.append((source, destination, source_stage))
        return {"tasks": 2}

    monkeypatch.setattr(c, "export_inputs", export)
    argv = [
        "export",
        "--source",
        str(tmp_path / "source"),
        "--destination",
        str(tmp_path / "input.json"),
    ]
    if stage:
        argv += ["--source-stage", stage]
    launcher.main(argv)
    assert calls == [
        (tmp_path / "source", tmp_path / "input.json", stage or "polishing")
    ]
    assert '"tasks": 2' in capsys.readouterr().out


@pytest.mark.parametrize("command", ["run", "status", "inspect"])
@pytest.mark.parametrize("custom", [False, True])
def test_continuation_wrapper_overrides_stale_campaign_environment(
    tmp_path, command, custom
):
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "AF_PYTHON": str(fake_python),
        "AF_OUTPUT_ROOT": "/unrelated/campaign",
        "AF_LLMSR_DETENTION_STAGED_POLISHING_ROOT": "/old/staged",
        "AF_LLMSR_DETENTION_STAGED_POLISHING_INPUTS": "/old/input.json",
    }
    for key in (
        "AF_LLMSR_DETENTION_STAGED_CONTINUATION_ROOT",
        "AF_LLMSR_DETENTION_STAGED_CONTINUATION_INPUTS",
    ):
        env.pop(key, None)
    if custom:
        env["AF_LLMSR_DETENTION_STAGED_CONTINUATION_ROOT"] = str(tmp_path / "new")
        env["AF_LLMSR_DETENTION_STAGED_CONTINUATION_INPUTS"] = str(
            tmp_path / "new.json"
        )
    result = subprocess.run(
        [
            "bash",
            str(
                launcher.REPO
                / "scripts/hpc/start_llm_sr_detention_staged_continuation_delta.sh"
            ),
            command,
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    argv = result.stdout.splitlines()
    root = (
        str(tmp_path / "new")
        if custom
        else "/work/hdd/bibo/yxiao2/phase_c/llm-sr-detention-staged-continuation-v1"
    )
    assert argv.count(root) == (2 if command == "run" else 1)
    assert not any(
        value in argv
        for value in ("/old/staged", "/old/input.json", "/unrelated/campaign")
    )
    if command == "run":
        assert argv.count("prepare") == argv.count("submit") == 1
        assert (
            str(tmp_path / "new.json")
            if custom
            else str(
                launcher.REPO / "inputs/llm-sr-detention-staged-continuation-input.json"
            )
        ) in argv
    else:
        assert argv.count(command) == 1
        assert "prepare" not in argv and "submit" not in argv
