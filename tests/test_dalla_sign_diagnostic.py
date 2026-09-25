"""Assisted decisions never impersonate a critic or silently change frozen inputs."""

import copy
import sys

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.llm.staged_topology import StagedTopologyClient
from autoformalism.rebuttal import dalla_sign_diagnostic as campaign
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.schemas.public_fitting import PublicFitRequest
from scripts import submit_dalla_sign_diagnostic as submitter
from scripts.smoke_dalla_sign_diagnostic import fixture
from tests.test_process_pruning_campaign import backend


def test_independent_paired_fits_labels_signs_and_resume(tmp_path, monkeypatch):
    source, decisions, root = fixture(tmp_path)
    before_source = source.read_bytes()
    monkeypatch.setattr(
        StagedTopologyClient, "call", lambda *a, **k: pytest.fail("LLM call")
    )
    plan = campaign.freeze(source, decisions, root)
    assert campaign.freeze(source, decisions, root) == plan
    rows = sealed_read(root / "paired-inputs.json")["rows"]
    old = public._read(source)["models"][0]["result"]
    assert rows[1]["request"] == PublicFitRequest.model_validate(
        old["selected_request"]
    ).model_dump(mode="json")
    assert rows[1]["parameters"] == old["selected_fit"]["parameters"]
    assert (
        rows[0]["request"]["initialization_plan"]
        == rows[1]["request"]["initialization_plan"]
    )
    calls = []
    monkeypatch.setattr(public, "_run_backend", backend(calls))
    # Either arm may start first without rewriting the other arm or waiting for it.
    assert campaign.run_one(root, 1)["status"] == "complete"
    assert len(calls) == 3
    partial = campaign.report(root)
    assert partial["status"] == "partial" and partial["recorded"] == 1
    assert campaign.run_one(root, 0)["sign_integrity"]["passed"]
    assert len(calls) == 6
    assert all(c["settings"] == calls[0]["settings"] for c in calls)
    result = campaign.report(root)
    models = public._read(root / "models.json")
    assert result["status"] == "complete" and result["expected"] == 2
    assert result["live_llm_calls"] == 0 and not result["semantic_assessor_called"]
    assert not result["scientific_correctness_certified"]
    assert models["protocol"] == campaign.PROTOCOL
    assert models["decision_source"] == campaign.LABEL
    assert not models["interventions_evaluated"] and not models["test_data_opened"]
    nested = public._read(root / "fitting/models.json")
    assert all(
        m["task"]["diagnostic"]["decision_source"] == campaign.LABEL
        for m in nested["models"]
    )
    snapshot = {str(p): p.read_bytes() for p in root.rglob("*.json")}
    campaign.run_one(root, 0)
    campaign.run_one(root, 1)
    assert campaign.report(root) == result
    assert len(calls) == 6 and source.read_bytes() == before_source
    assert snapshot == {str(p): p.read_bytes() for p in root.rglob("*.json")}


@pytest.mark.parametrize(
    "key", ["source_result_sha256", "request_sha256", "public_context_sha256"]
)
def test_wrong_model_or_context_is_rejected(tmp_path, key):
    source, decisions, root = fixture(tmp_path)
    raw = public._read(decisions)
    raw[key] = "0" * 64
    public._write(decisions, raw)
    with pytest.raises(ValueError, match="differs"):
        campaign.freeze(source, decisions, root)
    assert not (root / "plan.json").exists()


@pytest.mark.parametrize(
    "change", ["role", "missing", "duplicate", "unknown", "uncertain"]
)
def test_bad_decision_scope_or_direction_is_rejected(tmp_path, change):
    source, decisions, root = fixture(tmp_path)
    raw = public._read(decisions)
    ds = raw["review"]["decisions"]
    if change == "role":
        ds[0]["outer_weight_sign"] = "positive"
    elif change == "missing":
        ds.clear()
    elif change == "duplicate":
        ds.append(copy.deepcopy(ds[0]))
    elif change == "unknown":
        ds[0]["slot_id"] = "state_999_term_0"
    else:
        ds[0]["basis"] = "undetermined"
    public._write(decisions, raw)
    with pytest.raises(ValueError):
        campaign.freeze(source, decisions, root)


@pytest.mark.parametrize("key", ["test_data_opened", "interventions_evaluated"])
def test_nonpublic_source_is_rejected(tmp_path, key):
    source, decisions, root = fixture(tmp_path)
    raw = public._read(source)
    raw[key] = True
    public._write(source, raw)
    with pytest.raises(ValueError, match="public-only"):
        campaign.freeze(source, decisions, root)


def test_optional_citation_mistake_is_not_a_semantic_gate(tmp_path):
    source, decisions, root = fixture(tmp_path)
    raw = public._read(decisions)
    raw["review"]["decisions"][0]["public_quote"] = "Not an exact quotation."
    public._write(decisions, raw)
    plan = campaign.freeze(source, decisions, root)
    assert plan["patch"]["changed"]
    assert not plan["provenance"]["semantic_assessor_called"]


def test_frozen_decisions_cannot_be_replaced(tmp_path):
    source, decisions, root = fixture(tmp_path)
    campaign.freeze(source, decisions, root)
    raw = public._read(decisions)
    raw["assumptions"].append("Changed after freezing")
    public._write(decisions, raw)
    with pytest.raises(ValueError, match="identity differs"):
        campaign.verify(root)
    with pytest.raises(ValueError):
        campaign.freeze(source, decisions, root)


@pytest.mark.parametrize("site", ["aces", "delta"])
def test_cpu_submission_no_gpu_dependency_and_exact_resume(tmp_path, monkeypatch, site):
    source, decisions, root = fixture(tmp_path)
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setenv("AF_COMMIT", "a" * 40)
    monkeypatch.setattr(submitter, "source_commit", lambda _: "a" * 40)
    calls = {}

    def queue(directory, key, options, worker, stage, index):
        calls[key] = options
        return str(200 + len(calls))

    monkeypatch.setattr(submitter, "submit_job", queue)
    result = submitter.submit(source, decisions, root, site=site)
    assert result["gpus"] == 0 and result["array_tasks"] == 2
    assert set(calls) == {"fit", "report"}
    assert "--array=0-1%2" in calls["fit"]
    assert "--time=02:15:00" in calls["fit"]
    assert not any("--dependency" in arg for arg in calls["fit"])
    assert "--dependency=afterany:201" in calls["report"]
    assert all("--partition=cpu" in opts for opts in calls.values())
    assert not any(
        arg.startswith(("--gpu", "--gres")) for opts in calls.values() for arg in opts
    )
    assert ("--constraint=projects&work" in calls["fit"]) == (site == "delta")
    assert submitter.submit(source, decisions, root, site=site) == result
    assert len(calls) == 2


def test_uncertain_submission_never_blindly_retries(tmp_path, monkeypatch):
    source, decisions, root = fixture(tmp_path)
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setenv("AF_COMMIT", "a" * 40)
    monkeypatch.setattr(submitter, "source_commit", lambda _: "a" * 40)
    calls = []

    def queue(directory, key, options, worker, stage, index):
        calls.append(key)
        public._write(
            directory / f"{key}.intent.json",
            {
                "argv": [
                    "sbatch",
                    "--parsable",
                    *options,
                    str(worker),
                    stage,
                    str(index),
                ]
            },
        )
        if key == "fit":
            raise ValueError("uncertain")
        (directory / f"{key}.id").write_text("202\n")
        return "202"

    monkeypatch.setattr(submitter, "submit_job", queue)
    with pytest.raises(ValueError, match="uncertain"):
        submitter.submit(source, decisions, root)
    with pytest.raises(ValueError, match="Unconfirmed fit"):
        submitter.submit(source, decisions, root)
    assert calls == ["fit"]
    monkeypatch.setattr(
        submitter, "scheduler_record", lambda *a, **k: {"verified": True}
    )
    result = submitter.submit(source, decisions, root, adopt={"fit": "201"})
    assert calls == ["fit", "report"] and result["jobs"]["fit"] == "201"


def test_report_before_preparation_and_invalid_index(tmp_path):
    source, decisions, root = fixture(tmp_path)
    assert campaign.report(root)["status"] == "not_prepared"
    campaign.freeze(source, decisions, root)
    with pytest.raises(ValueError, match="expected repaired"):
        campaign.run_one(root, 2)
