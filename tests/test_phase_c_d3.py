"""D3 as published on Phase C public cells, against a declared endpoint.

tests/test_d3_upstream.py covers D3's loop. What is under test here is the
boundary around it: the release and the endpoint a plan was frozen against,
the served model, the requests D3's two calls make, the replies kept for
resume, and the transport that asks a caching gateway for fresh answers and
waits out a service that is down.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from autoformalism.baselines import d3_upstream
from autoformalism.baselines.d3_native import NativeD3Fit
from autoformalism.rebuttal import phase_b_d3
from autoformalism.rebuttal import phase_c_d3 as campaign
from autoformalism.rebuttal.llm_call_log import CallLog
from autoformalism.rebuttal.phase_c_vendored_campaign import (
    JETSTREAM2_HOSTED_BASE_URL,
    OUTAGE_PATIENCE_SECONDS,
)
from tests.test_phase_c_vendored_campaign import CELL, _release, _sha256

MODEL = "served-model"
LOCAL = "http://127.0.0.1:8000"
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def _payload(release: Path, **overrides) -> dict:
    payload = {
        "schema_version": "phase-c-d3-plan-2",
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
        "generations": 2,
        "patience": 2,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    payload.update(overrides)
    return payload


def _model(rhs: str = "-k * h_down") -> dict:
    """A D3 model reply in the restricted format."""
    return {
        "model_description": f"white box: {rhs}",
        "model": {
            "schema_version": "2",
            "candidate_id": "candidate_1",
            "states": [
                {
                    "name": "h_down",
                    "kind": "observed",
                    "observed_channel": "h_down",
                    "rhs": rhs,
                }
            ],
            "algebraics": [],
            "parameters": [{"name": "k"}],
        },
        "parameter_initial_values": [{"name": "k", "value": 0.4}],
    }


def _completion(content: str | None, finish: str = "stop") -> dict:
    return {
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 200},
    }


def _model_completion(rhs: str = "-k * h_down") -> tuple[dict, bool]:
    return _completion(json.dumps(_model(rhs))), False


def _text_completion(text: str = "Use a square.") -> tuple[dict, bool]:
    return _completion(text), False


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


class _Chat:
    """D3's two calls, scripted."""

    def __init__(self, models=(), reflections=()) -> None:
        self.models = iter(models)
        self.reflections = iter(reflections)
        self.calls: list[tuple[str, int]] = []

    def reflect(self, messages, *, generation):
        self.calls.append(("reflect", generation))
        return next(self.reflections)

    def write_model(self, messages, *, generation):
        self.calls.append(("model", generation))
        return d3_upstream.D3ModelReply.model_validate(next(self.models))


@pytest.fixture
def freeze(tmp_path, monkeypatch):
    """Freeze a plan on the toy release, with D3's torch fit stood in for."""
    monkeypatch.setattr(
        campaign, "environment", lambda endpoint: {"code": "fixed", "kind": endpoint}
    )
    monkeypatch.setattr(
        d3_upstream,
        "fit_native_d3",
        lambda *a, **k: NativeD3Fit({"k": 0.1}, 0.2, 0.1, 10, {"h_down": 1.0}),
    )
    monkeypatch.setattr(
        d3_upstream, "raw_state_losses", lambda *a: (0.05, {"h_down": 0.05})
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
    assert sealed["loop"] == "upstream_reflection"
    assert sealed["departures"] == list(campaign.DEPARTURES)
    assert sealed["selection"] == d3_upstream.SELECTION
    assert sealed["max_attempts"] == 1
    assert [row["repetition"] for row in sealed["rows"]] == [0, 1]
    assert {row["tier"] for row in sealed["rows"]} == {"fixed"}
    # Two tasks: a first model, then a reflection and a model.
    assert sealed["maximum_logical_calls"] == 6
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
        {"temperature": 2.5},
        {"top_p": 0.0},
        {"keep_top_samples": 0},
        {"reasoning_effort": "max"},
        {"schema_version": "phase-c-d3-plan-1"},
        {"seed": 3},
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


def test_a_task_runs_d3s_loop_and_resumes_without_a_call(freeze):
    root, _ = freeze()
    chat = _Chat([_model(), _model("-k * h_down ** 2")], ["Use a square."])
    result = _run(root, chat=chat)
    assert chat.calls == [("model", 0), ("reflect", 1), ("model", 1)]
    assert result["status"] == "complete"
    assert result["protocol"] == campaign.PROTOCOL
    assert result["selection_metric"] == d3_upstream.SELECTION
    assert result["test_data_opened"] is False
    assert "d3_checkpoint.json" in result["source_files"]

    assert _run(root, chat=_Chat()) == result
    summary = campaign.report(root)
    assert summary["counts"] == {"complete": 1, "pending": 1}
    assert summary["limitation"].startswith("Phase C D3 as published")


def test_a_task_resumes_only_against_its_endpoint_and_code(freeze, monkeypatch):
    root, _ = freeze()
    with pytest.raises(ValueError, match="endpoint kind differs"):
        _run(root, endpoint="jetstream2_hosted", chat=_Chat())
    monkeypatch.setattr(campaign, "environment", lambda endpoint: {"code": "new"})
    with pytest.raises(ValueError, match="code or dependencies changed"):
        _run(root, chat=_Chat())


def test_changed_inputs_are_refused_before_any_call(freeze, monkeypatch):
    root, _ = freeze()
    real = campaign.load_cell

    def drifted(release, name):
        cell = real(release, name)
        return type(cell)(cell.dataset, cell.context, cell.prompt,
                          {**cell.identity, "train": "changed"})

    monkeypatch.setattr(campaign, "load_cell", drifted)
    chat = _Chat()
    with pytest.raises(ValueError, match="input drift"):
        _run(root, chat=chat)
    assert chat.calls == []


def test_the_served_model_is_checked_before_the_first_call(freeze, monkeypatch):
    root, _ = freeze()
    built: list[dict] = []
    monkeypatch.setattr(campaign, "served_model_ids", lambda base_url: ("other",))
    monkeypatch.setattr(
        campaign,
        "endpoint_chat",
        lambda *a, **k: built.append(k) or _Chat(
            [_model(), _model("-k * h_down ** 2")], ["Use a square."]
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


# --- D3's requests and the transport ------------------------------------------


def _chat(tmp_path: Path, **options) -> campaign.EndpointChat:
    overrides = options.pop("plan", {})
    release = tmp_path / "r"
    if not release.exists():
        _release(release)
    plan = campaign.PhaseCD3Plan.model_validate(_payload(release, **overrides))
    options = {"base_url": LOCAL, "patience_seconds": 60.0, "bypass_cache": False,
               **options}
    return campaign.endpoint_chat(plan, tmp_path / "task", **options)


def test_requests_carry_the_plans_sampling_settings_and_ask_for_fresh_answers(
    tmp_path, monkeypatch
):
    sent, _ = _server(monkeypatch, [_text_completion(), _model_completion()])
    chat = _chat(tmp_path, bypass_cache=True)
    assert chat.reflect(MESSAGES, generation=1) == "Use a square."
    reply = chat.write_model(MESSAGES, generation=1)
    assert reply.model.states[0].rhs == "-k * h_down"
    reflection, model = sent
    for body in sent:
        assert body["cache"] == {"no-cache": True}
        assert body["model"] == MODEL
        assert body["messages"] == MESSAGES
        assert (body["temperature"], body["top_p"]) == (0.7, 0.95)
        assert body["max_tokens"] == 8192
        assert "reasoning_effort" not in body  # the server's default applies
    assert "response_format" not in reflection
    assert model["response_format"]["json_schema"]["strict"] is True
    assert model["response_format"]["json_schema"]["name"] == "D3ModelReply"


def test_a_plan_that_names_a_reasoning_effort_sends_it(tmp_path, monkeypatch):
    sent, _ = _server(monkeypatch, [_text_completion()])
    _chat(tmp_path, plan={"reasoning_effort": "low"}).reflect(MESSAGES, generation=1)
    assert sent[0]["reasoning_effort"] == "low"
    assert "cache" not in sent[0]  # a vLLM we serve keeps no answers to replay


def test_a_kept_reply_is_read_again_instead_of_asking(tmp_path, monkeypatch):
    sent, _ = _server(monkeypatch, [_model_completion()])
    _chat(tmp_path).write_model(MESSAGES, generation=0)
    again = _chat(tmp_path).write_model(MESSAGES, generation=0)
    assert again.model.states[0].rhs == "-k * h_down"
    assert len(sent) == 1
    counted = phase_b_d3.accounting(tmp_path / "task" / "llm_calls.jsonl")
    assert (counted["physical_requests"], counted["cache_hits"]) == (1, 1)
    with pytest.raises(campaign.KeptReplyMismatch):
        _chat(tmp_path).write_model([MESSAGES[0]], generation=0)


def test_a_reply_without_usable_text_is_the_models_failure_and_is_counted(
    tmp_path, monkeypatch
):
    replies = [
        (_completion(None, finish="length"), False),
        (_completion("not json"), False),
        (_completion(json.dumps({**_model(), "code": "x"})), False),
    ]
    _server(monkeypatch, replies)
    chat = _chat(tmp_path)
    with pytest.raises(campaign.LLMResponseError, match="finish_reason='length'"):
        chat.reflect(MESSAGES, generation=1)
    with pytest.raises(campaign.LLMResponseError, match="not JSON"):
        chat.write_model(MESSAGES, generation=1)
    with pytest.raises(campaign.LLMResponseError, match="failed its format"):
        chat.write_model(MESSAGES, generation=2)
    counted = phase_b_d3.accounting(tmp_path / "task" / "llm_calls.jsonl")
    assert counted["physical_requests"] == 3
    log = (tmp_path / "task" / "llm_calls.jsonl").read_text().splitlines()
    assert json.loads(log[0])["finish_reasons"] == ["length"]


def test_a_replayed_answer_is_counted_as_a_cache_hit(tmp_path, monkeypatch):
    _server(monkeypatch, [(_completion("Use a square."), True)])
    _chat(tmp_path).reflect(MESSAGES, generation=1)
    counted = phase_b_d3.accounting(tmp_path / "task" / "llm_calls.jsonl")
    assert (counted["cache_hits"], counted["physical_requests"]) == (1, 0)


def test_an_outage_is_waited_out_and_each_failure_counted(tmp_path, monkeypatch):
    replies = [
        urllib.error.URLError("refused"),
        _http_error(503),
        _http_error(401),  # how the hosted service refused while it was down
        TimeoutError("timed out"),
        _model_completion(),
    ]
    _, clock = _server(monkeypatch, replies)
    reply = _chat(tmp_path).write_model(MESSAGES, generation=0)
    assert reply.model.states[0].name == "h_down"
    assert clock.pauses == [1.0, 2.0, 4.0, 8.0]
    counted = phase_b_d3.accounting(tmp_path / "task" / "llm_calls.jsonl")
    assert counted["physical_requests"] == 5


def test_a_refused_request_reaches_d3_at_once(tmp_path, monkeypatch):
    _, clock = _server(monkeypatch, [_http_error(400)])
    with pytest.raises(campaign.LLMProviderError, match="HTTP 400"):
        _chat(tmp_path).write_model(MESSAGES, generation=0)
    assert clock.pauses == []


def test_an_outage_beyond_patience_stops_the_task_and_a_rerun_resumes_it(
    freeze, monkeypatch
):
    root, _ = freeze(repetitions=[0])
    monkeypatch.setattr(campaign, "served_model_ids", lambda base_url: (MODEL,))
    outage = [urllib.error.URLError("refused")] * 30
    sent, clock = _server(monkeypatch, [_model_completion(), *outage])
    with pytest.raises(campaign.EndpointUnavailable, match="failed for 15 minutes"):
        _run(root)
    # Generation 0 finished and was kept; nothing was sealed.
    assert not (root / "results" / "0" / "result.json").exists()
    checkpoint = json.loads((root / "results/0/d3_checkpoint.json").read_text())
    assert [record["generation"] for record in checkpoint["records"]] == [0]
    assert 15 * 60 <= clock.now < 16 * 60

    sent, _ = _server(
        monkeypatch, [_text_completion(), _model_completion("-k * h_down ** 2")]
    )
    result = _run(root)
    assert result["status"] == "complete"
    assert len(sent) == 2  # only generation 1's reflection and model
    assert "iteration 1 out of 2" in sent[0]["messages"][-1]["content"]


def test_a_refused_request_costs_its_generation(freeze, monkeypatch):
    root, _ = freeze(generations=1, patience=1)
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


def test_the_committed_plans_are_upstreams_settings_on_the_roster():
    roster = json.loads(
        (CONFIGS / "phase_c_public_baseline_delta_cpu_v1.json").read_text()
    )
    smoke = campaign.load_plan(CONFIGS / "phase_c_d3_smoke_v2.json")
    full = campaign.load_plan(CONFIGS / "phase_c_d3_hosted_120b_v2.json")
    assert [cell.model_dump() for cell in full.cells] == roster["cells"]
    assert full.repetitions == (0, 1)
    # Upstream config.yaml, the paper's temperature, and the server's own
    # reasoning default.
    assert (full.generations, full.patience, full.keep_top_samples) == (20, 20, 16)
    assert (full.temperature, full.top_p, full.reasoning_effort) == (0.7, 0.95, None)
    assert (full.max_output_tokens, full.trajectory_seconds) == (8192, 120)
    assert full.status == "frozen_before_calls"
    assert set(smoke.cells) <= set(full.cells)
    assert (smoke.generations, smoke.patience, smoke.repetitions) == (2, 2, (0,))
    sampling = ("keep_top_samples", "temperature", "top_p", "reasoning_effort",
                "max_output_tokens", "trajectory_seconds")
    assert {key: getattr(smoke, key) for key in sampling} == {
        key: getattr(full, key) for key in sampling
    }
    for plan in (smoke, full):
        assert plan.release_summary_sha256 == roster["release_summary_sha256"]
        assert (plan.endpoint, plan.model) == ("jetstream2_hosted", "gpt-oss-120b")
    # The GPU VM's 20b smoke is the hosted smoke, served on the VM instead.
    vm = campaign.load_plan(CONFIGS / "phase_c_d3_smoke_vm_20b_v2.json")
    assert (vm.endpoint, vm.model) == ("vm_local_vllm", "openai/gpt-oss-20b")
    assert vm.reporting_qualifications() == ()
    hosted, local = smoke.model_dump(), vm.model_dump()
    assert {key for key in local if local[key] != hosted[key]} == {
        "endpoint",
        "model",
        "purpose",
    }
