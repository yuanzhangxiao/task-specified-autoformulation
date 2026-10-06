"""D3 on Phase C public cells, against a declared endpoint.

Phase B's D3 runs unchanged once the data are loaded; tests/test_phase_b_d3.py
covers the method. What is under test is the new boundary: the release and the
endpoint a plan was frozen against, the served model, and the transport that
asks a caching gateway for fresh answers and waits out a service that is down.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from autoformalism.baselines import d3
from autoformalism.baselines.d3_native import NativeD3Fit
from autoformalism.expressions import ValidationContext
from autoformalism.llm.mock import MockLLMClient
from autoformalism.rebuttal import phase_b_d3
from autoformalism.rebuttal import phase_c_d3 as campaign
from autoformalism.rebuttal.llm_call_log import CallLog
from autoformalism.rebuttal.phase_c_vendored_campaign import (
    JETSTREAM2_HOSTED_BASE_URL,
    OUTAGE_PATIENCE_SECONDS,
)
from autoformalism.schemas import CandidateModel
from tests.test_phase_c_vendored_campaign import CELL, _release, _sha256

MODEL = "served-model"
LOCAL = "http://127.0.0.1:8000"
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _payload(release: Path, **overrides) -> dict:
    payload = {
        "schema_version": "phase-c-d3-plan-1",
        "status": "frozen_before_calls",
        "development_only": True,
        "release_protocol": "phase-c-development-2",
        "release_summary_sha256": _sha256(release / "summary.json"),
        "purpose": "Tests.",
        "endpoint": "vm_local_vllm",
        "model": MODEL,
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
        "generations": 1,
        "patience": 1,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    payload.update(overrides)
    return payload


def _candidate() -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": "native",
            "parent_candidate_id": None,
            "states": [{"name": "h_down", "kind": "observed"}],
            "state_equations": [{"state": "h_down", "rhs": "-k * h_down"}],
            "observation_mappings": [{"channel": "h_down", "expression": "h_down"}],
            "parameters": [
                {
                    "name": "k",
                    "scope": "global",
                    "bounds": {"lower": 0, "upper": 1},
                    "initialization_range": {"lower": 0.1, "upper": 0.5},
                }
            ],
            "initial_conditions": [
                {"state": "h_down", "scope": "global", "expression": "h_down"}
            ],
        }
    )


def _completion() -> dict:
    """A strict-JSON chat completion holding one structured D3 proposal."""
    proposal = {
        "schema_version": "2",
        "candidate_id": "candidate_1",
        "change_summary": "Linear outflow.",
        "states": [
            {
                "name": "h_down",
                "kind": "observed",
                "observed_channel": "h_down",
                "rhs": "-k * h_down",
            }
        ],
        "algebraics": [],
        "parameters": [{"name": "k", "bounds": {"lower": 0.0, "upper": 2.0}}],
    }
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(proposal)}}
        ],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 200},
    }


class _Clock:
    """Time that passes only when the transport pauses."""

    def __init__(self) -> None:
        self.now = 0.0
        self.pauses: list[float] = []

    def __call__(self) -> float:
        return self.now

    def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)
        self.now += seconds


def _server(monkeypatch, replies) -> tuple[list[dict], _Clock]:
    """Answer each request with the next reply: an error to raise, or a pair."""
    sent: list[dict] = []
    answers = iter(replies)

    def post(url: str, body: dict, timeout: float):
        sent.append(body)
        reply = next(answers)
        if isinstance(reply, Exception):
            raise reply
        return reply

    clock = _Clock()
    monkeypatch.setattr(campaign, "_post", post)
    monkeypatch.setattr(campaign, "_clock", clock)
    monkeypatch.setattr(campaign, "_pause", clock.pause)
    return sent, clock


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        LOCAL, code, "refused", {}, io.BytesIO(b'{"error": {"message": "no"}}')
    )


@pytest.fixture
def freeze(tmp_path, monkeypatch):
    """Freeze a plan on the toy release, with D3's torch fit stood in for."""
    monkeypatch.setattr(
        campaign, "environment", lambda endpoint: {"code": "fixed", "kind": endpoint}
    )
    monkeypatch.setattr(
        d3,
        "fit_native_d3",
        lambda *a, **k: NativeD3Fit({"k": 0.1}, 0.2, 0.1, 10, {"h_down": 1.0}),
    )
    monkeypatch.setattr(
        phase_b_d3,
        "evaluate_validation",
        lambda *a, **k: {
            "saved_one_step_matches": True,
            "native_one_step": {"normalized_mse": 0.1},
            "phase_b_rollout": {"status": "complete", "normalized_mse": 0.3},
        },
    )
    release = _release(tmp_path / "release")

    def frozen(**overrides) -> tuple[Path, dict]:
        config = tmp_path / "plan.json"
        config.write_text(json.dumps(_payload(release, **overrides)))
        root = tmp_path / "campaign"
        return root, campaign.prepare_phase_c(config, release, root)

    return frozen


def _run(root: Path, index: int = 0, **options) -> dict:
    options = {
        "endpoint": "vm_local_vllm",
        "base_url": LOCAL,
        "patience_seconds": 15 * 60.0,
        **options,
    }
    return campaign.run_phase_c(root, index, **options)


# --- freezing ---------------------------------------------------------------


def test_a_plan_freezes_roster_rows_against_the_release(freeze):
    _, sealed = freeze()
    assert sealed["protocol"] == campaign.PROTOCOL
    assert sealed["model"] == MODEL
    assert sealed["adaptation"] == phase_b_d3.ADAPTATION
    assert sealed["max_attempts"] == 1
    assert [row["repetition"] for row in sealed["rows"]] == [0, 1]
    assert {row["tier"] for row in sealed["rows"]} == {"fixed"}
    assert sealed["maximum_logical_calls"] == 2  # two tasks, one generation each
    assert sealed["reporting_qualifications"] == []
    assert sealed["test_data_opened"] is False


def test_a_hosted_plan_states_what_the_endpoint_cannot_prove(freeze):
    _, sealed = freeze(endpoint="jetstream2_hosted", model="gpt-oss-120b")
    assert sealed["reporting_qualifications"] == [
        "served by the Jetstream2 hosted endpoint, whose model revision and "
        "serving software are not verifiable"
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        {"repetitions": [0, 0]},
        {"repetitions": [-1]},
        {"generations": 6, "patience": 5},
        {"model": "vllm:served-model"},
        {"endpoint": "jetstream2_hosted"},  # serves only gpt-oss-120b
        {"test_data_opened": True},
        {"temperature": 0.7},
    ],
)
def test_a_plan_outside_the_protocol_is_refused(tmp_path, mutation):
    release = _release(tmp_path / "release")
    with pytest.raises(ValueError):
        campaign.PhaseCD3Plan.model_validate(_payload(release, **mutation))


def test_a_cell_must_be_on_the_roster_with_its_own_tier(tmp_path):
    release = _release(tmp_path / "release")
    cell = _payload(release)["cells"][0]
    for changed in ({**cell, "tier": "easy"}, {**cell, "benchmark_id": "phase_c_x"}):
        with pytest.raises(ValueError):
            campaign.PhaseCD3Plan.model_validate(_payload(release, cells=[changed]))


def test_another_release_is_refused_before_freezing(freeze):
    with pytest.raises(ValueError, match="release receipt differs"):
        freeze(release_summary_sha256="0" * 64)


# --- running ------------------------------------------------------------------


def test_a_task_runs_phase_b_d3_and_resumes_without_a_call(freeze):
    root, _ = freeze()
    client = MockLLMClient(proposer_responses=[_candidate()])
    result = _run(root, client=client)
    assert result["status"] == "complete"
    assert result["protocol"] == campaign.PROTOCOL
    assert result["test_data_opened"] is False
    (call,) = client.calls
    # Phase B's native prompt: the cell's public prompt, the declared
    # clarification, and the task's own identity.
    assert "Coupled stormwater basins" in call["system_prompt"]
    assert "no dt multiplier" in call["system_prompt"]
    assert json.loads(call["user_prompt"])["campaign_task"]["task_index"] == 0

    assert _run(root, client=MockLLMClient()) == result
    summary = campaign.report(root)
    assert summary["counts"] == {"complete": 1, "pending": 1}
    assert summary["limitation"].startswith("Phase C D3-native-no-tools")


def test_a_task_resumes_only_against_its_endpoint_and_code(freeze, monkeypatch):
    root, _ = freeze()
    with pytest.raises(ValueError, match="endpoint kind differs"):
        _run(root, endpoint="jetstream2_hosted", client=MockLLMClient())
    monkeypatch.setattr(campaign, "environment", lambda endpoint: {"code": "new"})
    with pytest.raises(ValueError, match="code or dependencies changed"):
        _run(root, client=MockLLMClient())


def test_changed_inputs_are_refused_before_any_call(freeze, monkeypatch):
    root, _ = freeze()
    real = campaign.load_cell

    def drifted(release, name):
        cell = real(release, name)
        return type(cell)(cell.dataset, cell.context, cell.prompt,
                          {**cell.identity, "train": "changed"})

    monkeypatch.setattr(campaign, "load_cell", drifted)
    client = MockLLMClient()
    with pytest.raises(ValueError, match="input drift"):
        _run(root, client=client)
    assert client.calls == []


def test_the_served_model_is_checked_before_the_first_call(freeze, monkeypatch):
    root, _ = freeze()
    built: list[dict] = []
    monkeypatch.setattr(campaign, "served_model_ids", lambda base_url: ("other",))
    monkeypatch.setattr(
        campaign,
        "endpoint_client",
        lambda *a, **k: built.append(k) or MockLLMClient(
            proposer_responses=[_candidate()]
        ),
    )
    with pytest.raises(ValueError, match="not among"):
        _run(root)
    assert built == []
    assert not (root / "results" / "0" / "result.json").exists()

    # A gateway that lists other models first still serves the plan's.
    served = ("other", MODEL)
    monkeypatch.setattr(campaign, "served_model_ids", lambda base_url: served)
    assert _run(root, bypass_cache=True)["status"] == "complete"
    assert built == [
        {"base_url": LOCAL, "patience_seconds": 15 * 60.0, "bypass_cache": True}
    ]

    def down(base_url):
        raise AssertionError("a finished task must not ask the server")

    monkeypatch.setattr(campaign, "served_model_ids", down)
    assert _run(root)["status"] == "complete"


# --- the transport ----------------------------------------------------------


def _client(tmp_path: Path, **options):
    plan = campaign.PhaseCD3Plan.model_validate(_payload(_release(tmp_path / "r")))
    options = {"base_url": LOCAL, "patience_seconds": 60.0, "bypass_cache": False,
               **options}
    return campaign.endpoint_client(
        plan, ValidationContext(targets=("h_down",)), tmp_path, **options
    )


def test_requests_keep_phase_b_settings_and_ask_a_gateway_for_fresh_answers(
    tmp_path, monkeypatch
):
    sent, _ = _server(monkeypatch, [(_completion(), False)])
    result = _client(tmp_path, bypass_cache=True).propose(
        system_prompt="system", user_prompt="user"
    )
    assert [state.name for state in result.parsed.states] == ["h_down"]
    (body,) = sent
    assert body["cache"] == {"no-cache": True}
    assert body["model"] == MODEL
    assert (body["reasoning_effort"], body["temperature"]) == ("low", 0.0)
    assert body["max_tokens"] == 8192
    assert body["response_format"]["json_schema"]["strict"] is True
    assert result.raw_response["_autoformalism_gateway"] == {
        "cache_bypass": True,
        "cache_hit": False,
    }


def test_a_vllm_we_serve_is_asked_exactly_as_in_phase_b(tmp_path, monkeypatch):
    sent, _ = _server(monkeypatch, [(_completion(), False)])
    _client(tmp_path).propose(system_prompt="system", user_prompt="user")
    assert "cache" not in sent[0]


def test_a_replayed_answer_is_counted_as_a_cache_hit(tmp_path, monkeypatch):
    _server(monkeypatch, [(_completion(), True)])
    _client(tmp_path).propose(system_prompt="system", user_prompt="user")
    counted = phase_b_d3.accounting(tmp_path / "llm_calls.jsonl")
    assert (counted["cache_hits"], counted["physical_requests"]) == (1, 0)


def test_an_outage_is_waited_out_and_each_failure_counted(tmp_path, monkeypatch):
    replies = [
        urllib.error.URLError("refused"),
        _http_error(503),
        _http_error(401),  # how the hosted service refused while it was down
        TimeoutError("timed out"),
        (_completion(), False),
    ]
    _, clock = _server(monkeypatch, replies)
    result = _client(tmp_path).propose(system_prompt="system", user_prompt="user")
    assert result.parsed.states[0].name == "h_down"
    assert clock.pauses == [1.0, 2.0, 4.0, 8.0]
    counted = phase_b_d3.accounting(tmp_path / "llm_calls.jsonl")
    assert counted["physical_requests"] == 5


def test_a_refused_request_reaches_d3_at_once(tmp_path, monkeypatch):
    _, clock = _server(monkeypatch, [_http_error(400)])
    with pytest.raises(campaign.LLMProviderError, match="HTTP 400"):
        _client(tmp_path).propose(system_prompt="system", user_prompt="user")
    assert clock.pauses == []


def test_an_outage_beyond_patience_stops_the_task_and_a_rerun_resumes_it(
    freeze, monkeypatch
):
    root, _ = freeze(generations=2, patience=2, repetitions=[0])
    monkeypatch.setattr(campaign, "served_model_ids", lambda base_url: (MODEL,))
    outage = [urllib.error.URLError("refused")] * 30
    sent, clock = _server(monkeypatch, [(_completion(), False), *outage])
    with pytest.raises(campaign.EndpointUnavailable, match="failed for 15 minutes"):
        _run(root)
    # Generation 0 finished and was kept; nothing was sealed.
    assert not (root / "results" / "0" / "result.json").exists()
    checkpoint = json.loads((root / "results/0/d3_checkpoint.json").read_text())
    assert [record["generation"] for record in checkpoint["records"]] == [0]
    assert 15 * 60 <= clock.now < 16 * 60

    sent, _ = _server(monkeypatch, [(_completion(), False)])
    result = _run(root)
    assert result["status"] == "complete"
    assert len(sent) == 1  # only generation 1 was asked for
    assert json.loads(sent[0]["messages"][1]["content"])["generation"] == 1


def test_a_refused_request_is_a_failed_generation_as_in_phase_b(freeze, monkeypatch):
    root, _ = freeze()
    monkeypatch.setattr(campaign, "served_model_ids", lambda base_url: (MODEL,))
    _server(monkeypatch, [_http_error(400)])
    result = _run(root)
    assert result["status"] == "discovery_failed"
    assert "HTTP 400" in result["error"]


def test_a_gateway_replay_is_counted_apart_from_physical_requests(tmp_path):
    log = tmp_path / "calls.jsonl"
    CallLog(log).failure("HTTP 503: down")
    events = [
        {"event": "llm_response", "provider_attempts": 1,
         "usage": {"total_tokens": 7},
         "raw_response": {"_autoformalism_gateway": {"cache_hit": True}}},
        {"event": "llm_response", "provider_attempts": 1,
         "usage": {"total_tokens": 5},
         "raw_response": {"_autoformalism_gateway": {"cache_hit": False}}},
    ]
    with log.open("a") as handle:
        handle.writelines(json.dumps(event) + "\n" for event in events)
    assert phase_b_d3.accounting(log) == {
        "physical_requests": 2,
        "observed_tokens": 5,
        "unknown_usage_requests": 1,
        "cache_hits": 1,
    }


# --- the command line and the plans -------------------------------------------


def test_the_cli_gives_a_task_its_endpoints_patience_and_cache_rule(monkeypatch):
    import sys

    from scripts import phase_c_d3 as cli

    seen: dict = {}
    monkeypatch.setenv("AF_ENDPOINT_KIND", "jetstream2_hosted")
    monkeypatch.delenv("AF_VLLM_BASE_URL", raising=False)
    monkeypatch.setattr(
        cli, "run_phase_c", lambda *a, **k: seen.update(k) or {"status": "complete"}
    )
    monkeypatch.setattr(sys, "argv", ["cli.py", "run", "--root", "r", "--index", "0"])
    cli.main()
    assert seen == {
        "endpoint": "jetstream2_hosted",
        "base_url": JETSTREAM2_HOSTED_BASE_URL,
        "patience_seconds": OUTAGE_PATIENCE_SECONDS["jetstream2_hosted"],
        "bypass_cache": True,
    }


def test_the_cli_exits_resumably_when_an_outage_outlasts_patience(
    monkeypatch, capsys
):
    import sys

    from scripts import phase_c_d3 as cli

    def stopped(*args, **kwargs):
        raise campaign.EndpointUnavailable("the endpoint failed for 360 minutes")

    monkeypatch.setenv("AF_ENDPOINT_KIND", "vm_local_vllm")
    monkeypatch.setenv("AF_VLLM_BASE_URL", LOCAL)
    monkeypatch.setattr(cli, "run_phase_c", stopped)
    monkeypatch.setattr(sys, "argv", ["cli.py", "run", "--root", "r", "--index", "0"])
    with pytest.raises(SystemExit) as stop:
        cli.main()
    assert stop.value.code == cli.ENDPOINT_UNAVAILABLE_STATUS
    assert json.loads(capsys.readouterr().out)["status"] == "endpoint_unavailable"


def test_the_committed_plans_are_phase_b_settings_on_the_roster():
    roster = json.loads(
        (CONFIGS / "phase_c_public_baseline_delta_cpu_v1.json").read_text()
    )
    smoke = campaign.load_plan(CONFIGS / "phase_c_d3_smoke_v1.json")
    full = campaign.load_plan(CONFIGS / "phase_c_d3_hosted_120b_v1.json")
    assert [cell.model_dump() for cell in full.cells] == roster["cells"]
    assert full.repetitions == (0, 1)
    assert (full.generations, full.patience, full.max_output_tokens) == (5, 5, 8192)
    assert (full.trajectory_seconds, full.reasoning_effort) == (120, "low")
    assert set(smoke.cells) <= set(full.cells)
    for plan in (smoke, full):
        assert plan.release_summary_sha256 == roster["release_summary_sha256"]
        assert (plan.endpoint, plan.model) == ("jetstream2_hosted", "gpt-oss-120b")
