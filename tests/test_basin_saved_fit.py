"""Historical context, chronology, immutable handoff and no accidental reruns."""

import shutil
from copy import deepcopy

import pytest

from autoformalism.rebuttal import basin_repair_audit as audit
from autoformalism.rebuttal import basin_saved_fit as io
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import submit_basin_saved_fit as submit
from tests.test_basin_repair_audit import saved  # noqa: F401


@pytest.fixture
def frozen(saved, tmp_path):  # noqa: F811 - imported pytest fixture
    _, source, _, task, other = saved
    gate = tmp_path / "gate"
    audit.audit(source, gate)
    root = tmp_path / "fit"
    plan = io.freeze(source, gate, root)
    return root, plan, source, gate, task, other


def test_freeze_resume_chooses_saved_edits_without_splicing(frozen):
    root, plan, source, gate, task, other = frozen
    assert io.freeze(source, gate, root) == plan
    assert plan["maximum_new_fits"] == 2
    assert plan["selections"][task["task_id"]]["attempt"] == 0
    selected = plan["selections"][other["task_id"]]
    assert selected["attempt"] == 2
    row = next(
        r
        for r in sealed_read(root / "inputs.json")["rows"]
        if r["task"] == other["task_id"]
    )
    assert len({a["context_sha256"] for a in row["attempts"]}) == 1
    assert all(
        a["context_sha256"] == io.content_hash(a["historical_context"])
        and a["raw_reply"]["accept_displayed"]
        for a in row["attempts"]
    )
    assert io.report(root)["status_counts"] == {"pending": 2, "unavailable": 14}


def test_choice_uses_chronology_not_scores_and_never_refits_confirmed(frozen):
    root, _, _, _, _, other = frozen
    row = next(
        r
        for r in sealed_read(root / "inputs.json")["rows"]
        if r["task"] == other["task_id"]
    )
    row = deepcopy(row)
    row["attempts"][0]["normalized_mse"] = 0
    row["attempts"][2]["normalized_mse"] = 100
    assert io.choose(row)["attempt"] == 2
    row["attempts"][2]["classification"] = "static_repair_required"
    assert io.choose(row)["attempt"] == 1
    row["historical_status"] = "confirmed"
    assert io.choose(row)["status"] == "historical_fit_retained"


def test_missing_or_ineligible_arm_does_not_fit(frozen, monkeypatch):
    root, plan, *_ = frozen
    monkeypatch.setattr(
        io.sibling_fit,
        "prepare_child_fit",
        lambda *a, **k: pytest.fail("unexpected fit"),
    )
    task = next(
        t
        for t in plan["tasks"]
        if plan["selections"][t["task_id"]]["status"] == "unavailable"
    )
    result = io.fit_task(root, task["index"])
    assert result["status"] == "unavailable" and not result["new_fit"]
    assert io.fit_task(root, task["index"]) == result


def test_interrupted_child_consumes_attempt_without_optimizer(frozen, monkeypatch):
    root, _plan, _, _, task, _ = frozen

    def interrupted(directory):
        frozen = io.public._read(directory / "freeze.json")
        io.public._write(directory / "started.json", {"identity": frozen["identity"]})
        return execute(directory)

    execute = io.sibling_fit.execute_child_fit
    monkeypatch.setattr(io.sibling_fit, "execute_child_fit", interrupted)
    monkeypatch.setattr(
        io.public, "_run_backend", lambda *a: pytest.fail("budget reused")
    )
    result = io.fit_task(root, task["index"])
    assert result["status"] == "interrupted"
    assert io.fit_task(root, task["index"]) == result
    assert io.report(root)["new_fit_results"] == 1
    seed = io.sibling_fit.inspect_child_fit(root / "results" / task["task_id"] / "fit")[
        "seed"
    ]
    assert seed["initialization_plan_unchanged"]


def test_gate_tampering_and_source_drift_fail(frozen, tmp_path):
    root, _, source, gate, *_ = frozen
    path = source / "plan.json"
    content = path.read_bytes()
    try:
        path.write_bytes(content + b"\n")
        with pytest.raises(ValueError, match="gate differs"):
            io.freeze(source, gate, root)
    finally:
        path.write_bytes(content)
    copied = tmp_path / "bad-gate"
    shutil.copytree(gate, copied)
    path = copied / "summary.json"
    raw = sealed_read(path)
    raw.pop("artifact_sha256")
    raw["rows"] = []
    path.unlink()
    sealed_write(path, raw)
    with pytest.raises(ValueError, match="audit rows differ"):
        io.freeze(source, copied, tmp_path / "invalid")


def test_scheduler_cpu_only_and_exact_resume(frozen, monkeypatch, tmp_path):
    _, _, source, gate, *_ = frozen
    root = tmp_path / "jobs"
    python = tmp_path / "python"
    python.touch()
    monkeypatch.setenv("AF_PYTHON", str(python))
    monkeypatch.setenv("AF_COMMIT", "a" * 40)
    monkeypatch.setattr(submit, "source_commit", lambda _: "a" * 40)
    calls = []

    def queue(directory, key, options, worker, stage, index):
        calls.append((stage, options))
        return str(100 + len(calls))

    monkeypatch.setattr(submit, "submit_job", queue)
    value = submit.submit(source, gate, root)
    assert [c[0] for c in calls] == ["prepare", "fit", "report"]
    assert "--array=0-15%4" in calls[1][1]
    assert all("--partition=cpu" in opts for _, opts in calls)
    assert value["llm_calls"] == value["gpus"] == 0
    assert submit.submit(source, gate, root) == value and len(calls) == 3


def test_scheduler_uncertainty_is_not_resubmitted(frozen, monkeypatch, tmp_path):
    _, _, source, gate, *_ = frozen
    root = tmp_path / "uncertain"
    python = tmp_path / "python"
    python.touch()
    monkeypatch.setenv("AF_PYTHON", str(python))
    monkeypatch.setenv("AF_COMMIT", "b" * 40)
    monkeypatch.setattr(submit, "source_commit", lambda _: "b" * 40)

    def fail(*args):
        raise ValueError("scheduler reply uncertain")

    monkeypatch.setattr(submit, "submit_job", fail)
    with pytest.raises(ValueError, match="scheduler reply"):
        submit.submit(source, gate, root)
    with pytest.raises(ValueError, match="partial/uncertain"):
        submit.submit(source, gate, root)
