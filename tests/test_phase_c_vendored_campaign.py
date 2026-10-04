"""LLM-SR and LLM-ODE on Phase C public cells, against a declared endpoint.

The Phase B campaign machinery runs unchanged once the data are loaded. What is
under test is the new boundary: the release a plan was frozen against, the
endpoint kind it may resume against, and that Phase B plans freeze as before.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.baselines.models import BaselineDevelopmentResult
from autoformalism.benchmarks.audited_release import seal
from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal import llm_ode_campaign as ode
from autoformalism.rebuttal import llm_sr_campaign as sr
from autoformalism.rebuttal import phase_c_baselines as pcb
from autoformalism.rebuttal import phase_c_vendored_campaign as vendored
from autoformalism.rebuttal.phase_c_vendored_campaign import (
    JETSTREAM2_HOSTED_BASE_URL,
    PhaseCVendoredCampaignPlan,
    check_served_model,
    endpoint_environment,
    resolve_endpoint,
)
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.rebuttal.vendored_campaign import VendoredCampaignPlan

CELL = "phase_c_detention_coupled_noise0_v1"
PROMPT = "\n".join(
    [
        "# Coupled stormwater basins",
        "## A. Scientific task",
        "Predict the downstream depth from the measured inflows.",
        "## B. Observations and known forcing",
        "Target h_down; inflows inflow_up and inflow_down.",
        "## C. Surveyed geometry and preparation",
        "Covariates area_up and area_down.",
        "## D. Required mechanisms",
        "- Represent accumulation in the downstream basin.",
        "## E. Evaluation boundaries",
        "Fit parameters using training only.",
    ]
)
COVARIATES = {"area_up": 2.0, "area_down": 3.0}
UPSTREAM = {
    "llm_sr": {
        "repository": "https://github.com/deep-symbolic-mathematics/LLM-SR",
        "commit": "41c212312df6c16d936c9cb395356a62774c47e3",
        "license": "MIT",
        "search_modified": False,
    },
    "llm_ode": {
        "repository": "https://github.com/gryaklab/llm-ode",
        "commit": "31667d9ac948e8f411cd2f07726d7744784b5d5a",
        "license": "MIT",
        "search_modified": False,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trajectory(identifier: str, start: float) -> dict:
    time = np.linspace(0.0, 3.0, 31)
    return {
        "trajectory_id": identifier,
        "time": time.tolist(),
        "targets": {"h_down": (start * np.exp(-time)).tolist()},
        "auxiliaries": {},
        "external_inputs": {
            "inflow_up": np.zeros_like(time).tolist(),
            "inflow_down": np.zeros_like(time).tolist(),
        },
        "fixed_covariates": COVARIATES,
    }


def _release(root: Path, *, starts=(1.0, 2.0)) -> Path:
    """A sealed toy release with one roster cell and no test split."""
    directory = root / "public" / CELL
    directory.mkdir(parents=True)
    seal(
        directory / "specification.json",
        {
            "protocol": pcb.PROTOCOL,
            "benchmark_id": CELL,
            "targets": ["h_down"],
            "auxiliaries": [],
            "external_inputs": ["inflow_up", "inflow_down"],
            "fixed_covariates": sorted(COVARIATES),
            "public_prompt": PROMPT,
            "test_released": False,
        },
    )
    for name, values in {"train": starts, "val": (1.5,)}.items():
        rows = [_trajectory(f"{name}_{i:03d}", s) for i, s in enumerate(values)]
        seal(
            directory / f"{name}.json",
            {"name": name, "fingerprint": f"toy-{name}-{values}", "rows": rows},
        )
    (directory / "proposer_prompt.txt").write_text(PROMPT + "\n", encoding="utf-8")
    seal(
        root / "summary.json",
        {
            "protocol": pcb.PROTOCOL,
            "whole_phase_c_roster_ready": True,
            "ready_cells": pcb.RELEASE_CELLS,
            "test_generated": False,
            "files": {
                f"public/{CELL}/{name}": _sha256(directory / name)
                for name in pcb.PUBLIC_FILES
            },
        },
    )
    return root


def _payload(release: Path, method: str = "llm_sr", **overrides) -> dict:
    searched = method == "llm_sr"
    payload = {
        "schema_version": "phase-c-vendored-campaign-plan-1",
        "status": "proposed_pending_review",
        "development_only": True,
        "release_protocol": "phase-c-development-2",
        "release_summary_sha256": _sha256(release / "summary.json"),
        "method": method,
        "upstream": UPSTREAM[method],
        "budget": {
            "unit": "llm_samples" if searched else "iterations",
            "published_default": 10000 if searched else 200,
            "declared": 10000 if searched else 200,
            "rationale": "The published default, unreduced.",
        },
        "prompt_policy": {
            "supplies_public_task_specification": True,
            "upstream_withholds_specification": not searched,
            "description": (
                "LLM-SR specifications embed a description natively."
                if searched
                else "Declared adaptation: equalises task information."
            ),
        },
        "islands": 10 if searched else 4,
        "endpoint": "vm_local_vllm",
        "model": "served-model",
        "cells": [
            {
                "benchmark_id": CELL,
                "tier": "fixed",
                "public_prompt_sha256": _sha256(
                    release / "public" / CELL / "proposer_prompt.txt"
                ),
            }
        ],
        "repetitions": [0, 1],
        "execution_semantics": "continuous_ode_free_rollout",
        "derivative_provenance": "upstream_findiff_fourth_order",
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    payload.update(overrides)
    return payload


def _config(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _frozen(tmp_path: Path, method: str = "llm_sr", **overrides):
    release = _release(tmp_path / "release")
    config = _config(tmp_path / "plan.json", _payload(release, method, **overrides))
    campaign = sr if method == "llm_sr" else ode
    root = tmp_path / "campaign"
    return release, root, campaign.prepare_phase_c(config, release, root)


def _complete(calls: list) -> object:
    def searcher(**kwargs) -> dict:
        calls.append(kwargs)
        return {
            "status": "complete",
            "equations": {"h_down": "-1.0*h_down + inflow_down"},
            "development_rollout_error": 0.25,
            "training_rollout_error": 0.125,
            "accounting": {"search_seconds": 1.0},
        }

    return searcher


# --- freezing ---------------------------------------------------------------


def test_a_phase_c_plan_freezes_roster_rows_against_the_release(tmp_path):
    release, root, sealed = _frozen(tmp_path)
    receipt = _sha256(release / "summary.json")
    assert sealed["protocol"] == sr.PHASE_C_PROTOCOL
    assert sealed["release_summary_sha256"] == receipt
    assert sealed["environment"] == endpoint_environment("vm_local_vllm")
    assert sealed["test_data_opened"] is False
    assert [row["repetition"] for row in sealed["rows"]] == [0, 1]
    row = sealed["rows"][0]
    assert row["tier"] == "fixed"
    assert row["searched_targets"] == ["h_down"]
    # Only the target is searched; inputs and covariates are right-hand-side
    # variables every equation may use.
    assert row["channels"][0] == "h_down"
    assert set(row["channels"]) == {
        "h_down", "inflow_up", "inflow_down", "area_up", "area_down",
    }
    assert row["public_identity"]["release_summary_sha256"] == receipt
    assert "downstream depth" in row["prompt"]
    assert "Evaluation boundaries" not in row["prompt"]
    assert "training only" not in row["prompt"]
    assert sealed["maximum_logical_samples"] == 2 * 10000
    # Freezing the same plan again is a no-op, not a second plan.
    assert sr.prepare_phase_c(tmp_path / "plan.json", release, root) == sealed


def test_freezing_refuses_a_release_the_plan_was_not_written_for(tmp_path):
    release = _release(tmp_path / "release")
    payload = _payload(release, release_summary_sha256="0" * 64)
    config = _config(tmp_path / "plan.json", payload)
    with pytest.raises(ValueError, match="release receipt differs"):
        sr.prepare_phase_c(config, release, tmp_path / "campaign")
    assert not (tmp_path / "campaign" / "plan.json").exists()


def test_freezing_refuses_a_prompt_the_plan_was_not_written_for(tmp_path):
    release = _release(tmp_path / "release")
    payload = _payload(release)
    payload["cells"][0]["public_prompt_sha256"] = "0" * 64
    config = _config(tmp_path / "plan.json", payload)
    with pytest.raises(ValueError, match="public prompt differs"):
        sr.prepare_phase_c(config, release, tmp_path / "campaign")


def test_each_campaign_freezes_only_its_own_method(tmp_path):
    release = _release(tmp_path / "release")
    config = _config(tmp_path / "plan.json", _payload(release, "llm_ode"))
    with pytest.raises(ValueError, match="expected an llm_sr plan"):
        sr.prepare_phase_c(config, release, tmp_path / "campaign")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda p: p["cells"][0].update(
                benchmark_id="phase_c_detention_coupled_noise1_v1"
            ),
            "outside the Phase C roster",
        ),
        (lambda p: p["cells"][0].update(tier="hard"), "wrong tier"),
        (lambda p: p.update(endpoint="jetstream2_hosted"), "serves only gpt-oss-120b"),
        (lambda p: p.update(model="vllm:served-model"), "provider prefix"),
        (lambda p: p.update(repetitions=[0, 0]), "unique and nonnegative"),
        (
            lambda p: p.update(derivative_provenance="estimated_numpy_gradient"),
            "upstream_findiff_fourth_order",
        ),
        (lambda p: p.update(endpoint="openai"), "jetstream2_hosted"),
    ],
)
def test_a_plan_refuses_what_it_cannot_run_truthfully(tmp_path, change, message):
    payload = _payload(_release(tmp_path / "release"))
    change(payload)
    with pytest.raises(ValueError, match=message):
        PhaseCVendoredCampaignPlan.model_validate(payload)


def test_a_hosted_plan_states_what_the_service_cannot_prove(tmp_path):
    _, _, sealed = _frozen(
        tmp_path, endpoint="jetstream2_hosted", model="gpt-oss-120b"
    )
    assert sealed["environment"] == endpoint_environment("jetstream2_hosted")
    assert any("not verifiable" in note for note in sealed["reporting_qualifications"])


# --- running and resuming ----------------------------------------------------


def test_a_task_searches_development_data_and_seals_its_model(tmp_path):
    _, root, sealed = _frozen(tmp_path)
    calls: list = []
    result = sr.run_phase_c(root, 0, endpoint="vm_local_vllm", search=_complete(calls))
    assert result["status"] == "complete"
    assert result["protocol"] == sr.PHASE_C_PROTOCOL
    assert result["test_data_opened"] is False
    (arguments,) = calls
    assert arguments["targets"] == ("h_down",)
    assert arguments["channels"] == tuple(sealed["rows"][0]["channels"])
    assert arguments["values"].shape == (62, 5)
    assert arguments["description"] == sealed["rows"][0]["prompt"]
    train, validation = arguments["development"]
    assert train.name is SplitName.TRAIN and validation.name is SplitName.VALIDATION

    saved = json.loads((root / "results" / "0" / "native-selection.json").read_text())
    assert saved["plan_sha256"] == sealed["artifact_sha256"]
    selection = BaselineDevelopmentResult.model_validate(saved["selection"])
    assert (selection.method, selection.tier) == ("llm_sr", "fixed")
    assert selection.selected_hyperparameters["llm_samples"] == 10000

    # A finished task resumes from its seal without a second search.
    again = sr.run_phase_c(root, 0, endpoint="vm_local_vllm", search=_complete(calls))
    assert again == result and len(calls) == 1
    summary = sr.report(root)
    assert summary["protocol"] == sr.PHASE_C_PROTOCOL
    assert (summary["terminal_success"], summary["not_started"]) == (1, 1)


def test_resuming_against_another_endpoint_kind_is_refused(tmp_path):
    _, root, _ = _frozen(tmp_path, endpoint="jetstream2_hosted", model="gpt-oss-120b")
    calls: list = []
    with pytest.raises(ValueError, match="endpoint kind differs from the frozen plan"):
        sr.run_phase_c(root, 0, endpoint="vm_local_vllm", search=_complete(calls))
    assert calls == [] and not (root / "results").exists()


def test_changed_code_names_its_remedy(tmp_path, monkeypatch):
    _, root, _ = _frozen(tmp_path)
    monkeypatch.setattr(vendored, "runtime_source_hash", lambda: "changed")
    with pytest.raises(ValueError, match="delete"):
        sr.run_phase_c(root, 0, endpoint="vm_local_vllm", search=_complete([]))


def test_a_replaced_release_is_refused_on_resume(tmp_path):
    release, root, _ = _frozen(tmp_path)
    shutil.rmtree(release)
    _release(release, starts=(1.25, 2.5))
    calls: list = []
    with pytest.raises(ValueError, match="receipt differs from the frozen plan"):
        sr.run_phase_c(root, 0, endpoint="vm_local_vllm", search=_complete(calls))
    assert calls == []


def test_a_changed_public_file_is_refused_on_resume(tmp_path):
    release, root, _ = _frozen(tmp_path)
    path = release / "public" / CELL / "train.json"
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="changed or unlisted"):
        sr.run_phase_c(root, 0, endpoint="vm_local_vllm", search=_complete([]))


def test_a_phase_b_plan_is_not_run_as_a_phase_c_plan(tmp_path):
    _, root, sealed = _frozen(tmp_path)
    with pytest.raises(ValueError, match="delete"):
        sr.run(root, 0, search=_complete([]))
    with pytest.raises(ValueError, match="not a Phase C LLM-ODE plan"):
        ode.run_phase_c(root, 0, endpoint="vm_local_vllm", search=_complete([]))
    assert sealed["protocol"] == sr.PHASE_C_PROTOCOL


def test_llm_ode_runs_the_same_rows_through_its_own_search(tmp_path):
    _, root, sealed = _frozen(tmp_path, "llm_ode")
    assert sealed["protocol"] == ode.PHASE_C_PROTOCOL
    assert sealed["search_config"] == {"n_islands": 4}
    assert sealed["maximum_logical_calls"] == 2 * 200 * 4 * 1
    assert any("prompt adapted" in note for note in sealed["reporting_qualifications"])
    calls: list = []
    result = ode.run_phase_c(root, 1, endpoint="vm_local_vllm", search=_complete(calls))
    assert result["status"] == "complete"
    (arguments,) = calls
    assert arguments["targets"] == ("h_down",)
    assert arguments["train"].channels == tuple(sealed["rows"][1]["channels"])
    assert arguments["prompt"] == sealed["rows"][1]["prompt"]
    saved = json.loads((root / "results" / "1" / "native-selection.json").read_text())
    selection = BaselineDevelopmentResult.model_validate(saved["selection"])
    assert (selection.method, selection.tier, selection.seed) == ("llm_ode", "fixed", 1)


# --- interrupted and concurrent attempts ---------------------------------------


@pytest.mark.parametrize("method", ["llm_sr", "llm_ode"])
def test_an_interrupted_attempt_is_kept_aside_and_the_task_restarts_clean(
    tmp_path, method
):
    _, root, _ = _frozen(tmp_path, method)
    campaign = sr if method == "llm_sr" else ode
    # What a search stopped part-way leaves: samples and calls, no result.
    stale = root / "results" / "0" / "llmsr-h_down" / "samples"
    stale.mkdir(parents=True)
    (stale / "samples_9999.json").write_text('{"score": 0.0}', encoding="utf-8")
    (root / "results" / "0" / "llm_calls.jsonl").write_text("{}\n", encoding="utf-8")

    calls: list = []
    result = campaign.run_phase_c(
        root, 0, endpoint="vm_local_vllm", search=_complete(calls)
    )
    assert result["status"] == "complete" and len(calls) == 1
    kept = root / "results" / "0.interrupted-1"
    assert (kept / "llmsr-h_down" / "samples" / "samples_9999.json").exists()
    assert (kept / "llm_calls.jsonl").exists()
    # The restarted attempt sees none of the stopped attempt's files.
    assert not (root / "results" / "0" / "llmsr-h_down").exists()
    assert not (root / "results" / "0" / "llm_calls.jsonl").exists()

    # A finished task is never set aside; resuming returns its seal.
    again = campaign.run_phase_c(
        root, 0, endpoint="vm_local_vllm", search=_complete(calls)
    )
    assert again == result and len(calls) == 1
    assert sorted(path.name for path in (root / "results").iterdir()) == [
        "0",
        "0.interrupted-1",
    ]


def test_a_second_interruption_is_kept_beside_the_first(tmp_path):
    _, root, _ = _frozen(tmp_path)
    for number in (1, 2):
        task = root / "results" / "0"
        task.mkdir(parents=True)
        (task / "llm_calls.jsonl").write_text(f"{number}\n", encoding="utf-8")
        assert vendored.set_aside_unfinished(task) == root / "results" / (
            f"0.interrupted-{number}"
        )
    assert (root / "results" / "0.interrupted-2" / "llm_calls.jsonl").read_text() == (
        "2\n"
    )
    # Nothing to move: no attempt, or an empty one.
    assert vendored.set_aside_unfinished(root / "results" / "0") is None
    (root / "results" / "0").mkdir()
    assert vendored.set_aside_unfinished(root / "results" / "0") is None


def test_a_task_running_in_another_process_is_refused(tmp_path):
    _, root, _ = _frozen(tmp_path)
    calls: list = []
    with (
        vendored.exclusive_attempt(root, 0),
        pytest.raises(ValueError, match="already running"),
    ):
        sr.run_phase_c(root, 0, endpoint="vm_local_vllm", search=_complete(calls))
    assert calls == []
    # The lock is released with its holder; another task was never blocked.
    assert sr.run_phase_c(
        root, 1, endpoint="vm_local_vllm", search=_complete(calls)
    )["status"] == "complete"
    assert sr.run_phase_c(
        root, 0, endpoint="vm_local_vllm", search=_complete(calls)
    )["status"] == "complete"


# --- endpoints ----------------------------------------------------------------


def test_the_hosted_route_is_fixed_and_local_kinds_stay_local():
    hosted = JETSTREAM2_HOSTED_BASE_URL
    assert resolve_endpoint("jetstream2_hosted", None) == ("jetstream2_hosted", hosted)
    assert resolve_endpoint("jetstream2_hosted", hosted + "/")[1] == hosted
    assert resolve_endpoint("vm_local_vllm", "http://127.0.0.1:8000/") == (
        "vm_local_vllm",
        "http://127.0.0.1:8000",
    )
    with pytest.raises(ValueError, match="hosted endpoint is"):
        resolve_endpoint("jetstream2_hosted", "https://llm.jetstream-cloud.org/api")
    with pytest.raises(ValueError, match="not a loopback address"):
        resolve_endpoint("vm_local_vllm", hosted)
    with pytest.raises(ValueError, match="needs its base URL"):
        resolve_endpoint("job_local_vllm", None)
    with pytest.raises(ValueError, match="unknown endpoint kind"):
        resolve_endpoint("", "http://127.0.0.1:8000")


def test_the_served_model_must_be_the_plans_and_first_for_llm_ode():
    check_served_model(("other", "served-model"), "served-model", first=False)
    with pytest.raises(ValueError, match="first among"):
        check_served_model(("other", "served-model"), "served-model", first=True)
    with pytest.raises(ValueError, match="not among"):
        check_served_model(("other",), "served-model", first=False)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_the_model_list_is_read_and_a_web_page_is_refused(monkeypatch):
    replies = iter(
        [
            b'{"object": "list", "data": [{"id": "gpt-oss-120b"}]}',
            b"<!doctype html><html></html>",
        ]
    )
    monkeypatch.setattr(
        vendored.urllib.request,
        "urlopen",
        lambda request, timeout: _Response(next(replies)),
    )
    assert vendored.served_model_ids(JETSTREAM2_HOSTED_BASE_URL) == ("gpt-oss-120b",)
    with pytest.raises(ValueError, match="cannot list the models"):
        vendored.served_model_ids(JETSTREAM2_HOSTED_BASE_URL)


@pytest.mark.parametrize(
    ("method", "served", "message"),
    [
        ("llm_sr", ("other",), "not among"),
        ("llm_ode", ("other", "served-model"), "first among"),
    ],
)
def test_the_cli_checks_the_served_model_before_any_search(
    tmp_path, monkeypatch, method, served, message
):
    from scripts import phase_c_llm_ode, phase_c_llm_sr

    cli = phase_c_llm_sr if method == "llm_sr" else phase_c_llm_ode
    campaign = sr if method == "llm_sr" else ode
    _, root, _ = _frozen(tmp_path, method)
    calls: list = []
    monkeypatch.setenv("AF_LLM_SR_ROOT", str(tmp_path))
    monkeypatch.setenv("AF_LLM_ODE_ROOT", str(tmp_path))
    monkeypatch.setattr(cli, "build_searcher", lambda **kwargs: _complete(calls))
    monkeypatch.setattr(cli, "served_model_ids", lambda base_url: served)
    search = cli._searcher(root, "http://127.0.0.1:8000")
    with pytest.raises(ValueError, match=message):
        campaign.run_phase_c(root, 0, endpoint="vm_local_vllm", search=search)
    assert calls == []
    assert not (root / "results" / "0" / "result.json").exists()

    monkeypatch.setattr(cli, "served_model_ids", lambda base_url: ("served-model",))
    search = cli._searcher(root, "http://127.0.0.1:8000")
    result = campaign.run_phase_c(root, 0, endpoint="vm_local_vllm", search=search)
    assert result["status"] == "complete" and len(calls) == 1


def test_the_cli_takes_the_endpoint_from_the_environment(monkeypatch):
    from scripts import phase_c_llm_sr

    monkeypatch.setenv("AF_ENDPOINT_KIND", "jetstream2_hosted")
    monkeypatch.delenv("AF_VLLM_BASE_URL", raising=False)
    assert phase_c_llm_sr._endpoint() == (
        "jetstream2_hosted",
        JETSTREAM2_HOSTED_BASE_URL,
    )
    monkeypatch.delenv("AF_ENDPOINT_KIND")
    with pytest.raises(ValueError, match="unknown endpoint kind"):
        phase_c_llm_sr._endpoint()


# --- Phase B is unchanged -------------------------------------------------------


def _phase_b_cell():
    time = np.linspace(0.0, 1.0, 11)
    split = DatasetSplit(
        SplitName.TRAIN,
        (
            Trajectory(
                trajectory_id="a",
                time=time,
                targets={"G": time},
                auxiliaries={"I": time},
                external_inputs={},
                fixed_covariates={},
                derivatives={},
            ),
        ),
        "train",
    )
    context = ValidationContext(targets=("G",), auxiliaries=("I",))
    identity = {"train": "t", "validation": "v", "prompt": "p"}
    prompt = "\n".join(
        [
            "A. Task specification",
            "x",
            "B. Available data",
            "y",
            "C. Modeling requirements",
        ]
    )
    return SimpleNamespace(train=split), context, identity, prompt


@pytest.mark.parametrize(
    ("campaign", "config", "keys"),
    [
        (
            sr,
            "configs/phase_b_llm_sr_six_cell_v1.json",
            {"maximum_logical_samples"},
        ),
        (
            ode,
            "configs/phase_b_llm_ode_campaign_v1.json",
            {
                "search_config",
                "maximum_logical_calls",
                "test_data_opened",
                "private_reference_opened",
            },
        ),
    ],
)
def test_phase_b_plans_freeze_with_their_old_shape_and_endpoint(
    tmp_path, monkeypatch, campaign, config, keys
):
    development, context, identity, prompt = _phase_b_cell()
    monkeypatch.setattr(
        campaign, "load_public", lambda *args: (development, context, identity)
    )
    monkeypatch.setattr(campaign, "public_prompt_text", lambda *args: prompt)
    sealed = campaign.prepare(Path(config), tmp_path / "public", tmp_path / "b")
    assert set(sealed) == {
        "protocol",
        "plan",
        "public_root",
        "environment",
        "reporting_qualifications",
        "rows",
        "artifact_sha256",
        *keys,
    }
    assert sealed["protocol"] == campaign.PROTOCOL
    assert "endpoint" not in sealed["plan"]
    assert sealed["environment"] == {
        "runtime_source_sha256": runtime_source_hash(),
        "provider": "vllm",
        "endpoint_kind": "job-local vllm endpoint",
    }
    assert campaign.environment_identity() == sealed["environment"]


def test_phase_b_plans_still_refuse_a_phase_c_tier():
    raw = json.loads(Path("configs/phase_b_llm_sr_six_cell_v1.json").read_text())
    raw["cells"][0]["tier"] = "fixed"
    with pytest.raises(ValueError):
        VendoredCampaignPlan.model_validate(raw)
