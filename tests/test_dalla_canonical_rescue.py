"""Exact assisted scope, complete provenance and unchanged paired budgets."""

import copy
import sys

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import dalla_canonical_rescue as campaign
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest
from autoformalism.staged_topology import content_hash
from scripts import submit_dalla_sign_diagnostic as submitter
from scripts.smoke_dalla_canonical_rescue import fixture
from tests.test_process_pruning_campaign import backend


def test_four_independent_equal_budget_arms_and_exact_resume(tmp_path, monkeypatch):
    source, config, root = fixture(tmp_path)
    old = sealed_read(source)
    plan = campaign.freeze(source, config, root)
    assert campaign.freeze(source, config, root) == plan
    assert plan["task_count"] == 4
    rows = sealed_read(root / "paired-inputs.json")["rows"]
    for index in (0, 2):
        repaired, control = rows[index : index + 2]
        assert control["parameters"] == old["rows"][index // 2]["parameters"]
        assert control["request"] == PublicFitRequest.model_validate(
            old["rows"][index // 2]["request"]
        ).model_dump(mode="json")
        assert (
            repaired["request"]["initialization_plan"]
            == control["request"]["initialization_plan"]
        )
        resets = repaired["warm_start_audit"]["reset_for_sign_or_domain_change"]
        assert resets and all(v == 0.1 for v in resets.values())
        assert all(repaired["parameters"][k] == v for k, v in resets.items())
    calls = []
    monkeypatch.setattr(public, "_run_backend", backend(calls))
    campaign.run_one(root, 3)
    assert campaign.report(root)["recorded"] == 1
    for index in (0, 2, 1):
        campaign.run_one(root, index)
    result = campaign.report(root)
    assert result["status"] == "complete" and len(calls) == 12
    assert all(c["settings"] == calls[0]["settings"] for c in calls)
    assert result["decision_source"] == campaign.LABEL
    assert not result["scientific_correctness_certified"]
    assert (
        sum(a is not None and a["passed"] for a in result["sign_integrity"].values())
        == 2
    )
    before = {str(p): p.read_bytes() for p in root.rglob("*.json")}
    for index in range(4):
        campaign.run_one(root, index)
    assert campaign.report(root) == result
    assert before == {str(p): p.read_bytes() for p in root.rglob("*.json")}
    assert len(calls) == 12


@pytest.mark.parametrize("field", ["state", "parameter", "original_expression"])
def test_exact_scope_rejects_wrong_term(tmp_path, field):
    source, config, _ = fixture(tmp_path)
    decision = campaign.Decisions.model_validate(public._read(config)).models[0]
    request = PublicFitRequest.model_validate(sealed_read(source)["rows"][0]["request"])
    edit = decision.edits[0].model_copy(update={field: "wrong"})
    with pytest.raises(ValueError):
        campaign.apply(request, (edit,))


@pytest.mark.parametrize(
    "kind", ["shared", "boundary", "bounded", "duplicate", "inner"]
)
def test_protected_or_non_outer_gains_rejected(tmp_path, kind):
    source, config, _ = fixture(tmp_path)
    row = sealed_read(source)["rows"][0]
    edit = campaign.Decisions.model_validate(public._read(config)).models[0].edits[0]
    raw = copy.deepcopy(row["request"])
    spec = next(
        p for p in raw["base_candidate"]["parameters"] if p["name"] == edit.parameter
    )
    if kind == "shared":
        raw["base_candidate"]["observation_mappings"][0]["expression"] = edit.parameter
    elif kind == "boundary":
        raw["base_candidate"]["initial_conditions"][0]["expression"] = edit.parameter
    elif kind == "bounded":
        spec["bounds"] = {"lower": -1.0, "upper": 1.0}
    elif kind == "inner":
        rhs = f"y / ({edit.parameter} + 1)"
        raw["base_candidate"]["state_equations"][0]["rhs"] = rhs
        edit = edit.model_copy(update={"original_expression": rhs})
    with pytest.raises(ValueError):
        request = PublicFitRequest.model_validate(raw)
        campaign.apply(request, (edit, edit) if kind == "duplicate" else (edit,))


@pytest.mark.parametrize("change", ["hash", "duplicate", "test", "intervention"])
def test_sources_fail_closed(tmp_path, change):
    source, config, root = fixture(tmp_path)
    if change in {"hash", "duplicate"}:
        raw = public._read(config)
        if change == "hash":
            raw["models"][0]["source_row_sha256"] = "0" * 64
        else:
            raw["models"].append(raw["models"][0])
        public._write(config, raw)
    else:
        raw = sealed_read(source)
        raw.pop("artifact_sha256")
        raw["test_data_opened" if change == "test" else "intervention_data_used"] = True
        source.unlink()
        sealed_write(source, raw)
    with pytest.raises(ValueError):
        campaign.freeze(source, config, root)


def test_after_freeze_change_rejected(tmp_path):
    source, config, root = fixture(tmp_path)
    campaign.freeze(source, config, root)
    raw = public._read(config)
    raw["models"][0]["assumptions"].append("changed")
    public._write(config, raw)
    with pytest.raises(ValueError, match="identity differs"):
        campaign.verify(root)


@pytest.mark.parametrize("site", ["aces", "delta"])
def test_four_cpu_tasks_no_prepare_or_gpu_dependency(tmp_path, monkeypatch, site):
    source, config, root = fixture(tmp_path)
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setenv("AF_COMMIT", "a" * 40)
    monkeypatch.setattr(submitter, "source_commit", lambda _: "a" * 40)
    calls = {}

    def queue(directory, key, options, worker, stage, index):
        calls[key] = options
        return str(300 + len(calls))

    monkeypatch.setattr(submitter, "submit_job", queue)
    result = submitter.submit(source, config, root, site=site, campaign=campaign)
    assert result["array_tasks"] == 4 and result["gpus"] == 0
    assert set(calls) == {"fit", "report"}
    assert "--array=0-3%2" in calls["fit"]
    assert not any("dependency" in x for x in calls["fit"])
    assert "--dependency=afterany:301" in calls["report"]
    assert ("--constraint=projects&work" in calls["fit"]) == (site == "delta")
    assert (
        submitter.submit(source, config, root, site=site, campaign=campaign) == result
    )


def test_changed_gain_resets_without_abs_of_saved_value(tmp_path):
    source, config, _ = fixture(tmp_path)
    raw = sealed_read(source)
    raw.pop("artifact_sha256")
    decision = public._read(config)
    name = decision["models"][0]["edits"][0]["parameter"]
    raw["rows"][0]["parameters"][name] = -17.0
    decision["models"][0]["source_row_sha256"] = content_hash(raw["rows"][0])
    source.unlink()
    sealed_write(source, raw)
    public._write(config, decision)
    rows = campaign.derive(source, config)["inputs"]["rows"]
    assert rows[0]["parameters"][name] == 0.1
    assert rows[1]["parameters"][name] == -17.0
