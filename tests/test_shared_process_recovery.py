"""Exact empty-vector recovery, preserved controls, costs, boundaries and resume."""

import json
import re

import pytest

from autoformalism.fitting import fixed_model
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.rebuttal import shared_process_recovery as recovery
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import smoke_shared_process_pilot as smoke


def source_fixture(root, monkeypatch):
    plan = smoke.fixture(root)
    original = public._run_backend
    for index, task in enumerate(plan["tasks"]):
        base = smoke.transport_for([])

        def transport(*args, base=base):
            response = base(*args)
            message = response["choices"][0]["message"]
            reply = json.loads(message["content"])
            for f in reply.get("functions", []):
                f["expression"] = re.sub(r"\b(rate|gain|decay)\b", "1", f["expression"])
                f["parameters"] = []
            if "initial" in reply:
                reply["initial"].update(expression="0", parameters=[])
            message["content"] = json.dumps(reply)
            return response

        client = pipeline._client(
            root, plan, task, 0, "http://offline", lambda: True, transport
        )
        proposal = pipeline.propose_one(root, plan, task, 0, client)
        assert proposal["status"] == "constructed", proposal
        assert not public._lower(
            pipeline.request_for(proposal["bundle"], plan, task, 0)
        )[0].parameter_names

        def backend(*args, index=index):
            if index < 2:
                raise ValueError(recovery.ERROR.removeprefix("ValueError: "))
            if index == 2:
                raise ValueError("different numerical failure")
            return original(*args)

        with monkeypatch.context() as m:
            m.setattr(public, "_run_backend", backend)
            pipeline.fit_one(root, plan, task, 0)
    return plan


def test_recovery_and_next_round_preserve_costs_and_successes(tmp_path, monkeypatch):
    source, dest = tmp_path / "source", tmp_path / "recovered"
    plan = source_fixture(source, monkeypatch)
    before = {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*.json")
    }
    old_summary = reporting.report(source)
    result = recovery.recover(source, dest)
    assert result["optimizer_calls"] == result["llm_calls"] == 0
    assert len(result["rows"]) == 2
    assert all(
        r["status"] == "complete" and r["residual_packet_available"]
        for r in result["rows"]
    )
    imported = io.verify(dest)
    for i, task in enumerate(plan["tasks"]):
        old = io.read_round(source, task, 0)
        new = io.read_round(dest, task, 0)
        if i >= 2:
            assert new["selected"] == old["selected"]
            assert new["status"] == old["status"]
        else:
            assert new["trial"]["bundle"] == old["trial"]["bundle"]
            assert new["selected"]["fit"]["parameters"] == {}
            assert not new["closed"]
            assert (
                new["selected"]["packet"]["numerical_status"]["feedback_status"]
                == "fixed_model_evaluated"
            )
    summary = reporting.report(dest)
    for a, b in zip(old_summary["rows"], summary["rows"], strict=True):
        for key in (
            "cumulative_tokens",
            "cumulative_requests",
            "cumulative_fit_attempts",
        ):
            assert a[key] == b[key]
    after = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*.json")}
    assert all(after[k] == v for k, v in before.items())
    monkeypatch.setattr(
        fixed_model, "evaluate", lambda *a: pytest.fail("repeated evaluation")
    )
    assert recovery.recover(source, dest) == result
    task = plan["tasks"][0]
    client = pipeline._client(
        dest, imported, task, 1, "http://offline", lambda: True, smoke.transport_for([])
    )
    proposal = pipeline.propose_one(dest, imported, task, 1, client)
    assert proposal["status"] == "committed", proposal


def test_recovery_refuses_later_round_or_missing_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    plan = source_fixture(source, monkeypatch)
    round1 = source / "submission-intent/round-1"
    round1.mkdir(parents=True)
    with pytest.raises(ValueError, match="already submitted"):
        recovery.prepare(source, tmp_path / "new")
    round1.rmdir()
    io.round_path(source, plan["tasks"][0], 0).joinpath("result.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        recovery.prepare(source, tmp_path / "new")


def test_recovery_detects_import_and_anchor_changes(tmp_path, monkeypatch):
    source, dest = tmp_path / "source", tmp_path / "recovered"
    plan = source_fixture(source, monkeypatch)
    recovery.recover(source, dest)
    path = io.round_path(dest, plan["tasks"][-1], 0) / "result.json"
    value = sealed_read(path)
    value.pop("artifact_sha256")
    value["closed"] = not value["closed"]
    path.unlink()
    sealed_write(path, value)
    with pytest.raises(ValueError, match="differs from its source"):
        io.verify(dest)


def test_nonempty_or_unrelated_failure_is_not_eligible():
    assert not recovery._eligible(
        {"status": "fit_failed", "trial": {"fit": {"message": "unknown"}}}
    )
    assert not recovery._eligible({"status": "construction_failed", "trial": None})
