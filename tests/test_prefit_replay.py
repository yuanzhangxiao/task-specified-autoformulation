"""Historical-context replay, strict source provenance, and no numerical access."""

import copy
import json
from pathlib import Path

import pytest

from autoformalism.rebuttal.prefit_replay import ReplayCase, diagnose, replay
from scripts.smoke_prefit_replay import synthetic_source as source_fixture


def test_replay_recovers_notation_without_advancing_failed_history(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source_fixture(source)
    original = Path.read_bytes

    def guarded(path):
        assert path.suffix != ".csv" and path.name != "fit.json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    output = tmp_path / "replay.json"
    result = replay(source, output)
    assert result["counts"]["historical_complete"] == 1
    assert result["counts"]["normalization_recovered"] == 3
    assert result["counts"]["remaining_diagnostic_codes"]["SOURCE_MISMATCH"] == 2
    assert any(
        r["reason"] == "topology_stage_not_replayed"
        for r in result["unreplayed_requests"]
    )
    assert not result["trajectory_tables_opened"]
    assert replay(source, output) == result
    cases = [ReplayCase.model_validate(v) for v in result["cases"]]
    assert all(
        diagnose(c, c.original_reply).valid for c in cases if c.kind == "initializer"
    )
    assert all(
        diagnose(c, c.original_reply).protected_context_preserved
        for c in cases
        if diagnose(c, c.original_reply).valid
    )


def test_missing_or_changed_request_is_not_silently_dropped(tmp_path, monkeypatch):
    source = tmp_path / "source"
    plan = source_fixture(source)
    task = plan["tasks"][0]
    path = next((source / "results" / task["task_id"] / "calls").glob("*.json"))
    original = path.read_text()
    data = json.loads(original)
    data["request"]["namespace"] = "changed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity differs"):
        replay(source, tmp_path / "out.json")
    path.unlink()
    with pytest.raises(FileNotFoundError):
        replay(source, tmp_path / "out.json")


def test_replay_refuses_source_overwrite_and_changed_output(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source_fixture(source)
    with pytest.raises(ValueError, match="outside"):
        replay(source, source / "replay.json")
    output = tmp_path / "replay.json"
    result = replay(source, output)
    changed = copy.deepcopy(result)
    changed["counts"]["historical_complete"] = 99
    output.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="digest differs"):
        replay(source, output)


@pytest.mark.parametrize("kind", ["function", "initializer"])
def test_invalid_historical_context_is_not_a_proposer_diagnosis(tmp_path, kind):
    source = tmp_path / "source"
    source_fixture(source)
    corpus = replay(source, tmp_path / "replay.json")
    data = copy.deepcopy(next(c for c in corpus["cases"] if c["kind"] == kind))
    if kind == "function":
        data["selected_term"]["sources"] = ["unrelated"]
        message = "slot differs"
    else:
        data["public_request"]["selected_state"]["rhs"] = "0"
        message = "state RHS differs"
    case = ReplayCase.model_validate(data)
    with pytest.raises(ValueError, match=message):
        diagnose(case, case.original_reply)
