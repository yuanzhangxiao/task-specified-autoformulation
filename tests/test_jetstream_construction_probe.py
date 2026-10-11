"""Bounded serving checks: complete prompts, durable failure evidence, no fitting."""

import json
from copy import deepcopy

import pytest

from autoformalism.llm.staged_topology import strict_provider_schema
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import jetstream_construction_probe as probe
from scripts.jetstream import verify_weights
from scripts.jetstream_server_record import server_record


def request(stage):
    model = probe.response_model(stage)
    return {
        "model": probe.MODEL,
        "reasoning_effort": "low",
        "temperature": 0.2,
        "max_tokens": 8192,
        "stream": False,
        "seed": 11,
        "messages": [
            {"role": "system", "content": "Construct a public model."},
            {"role": "user", "content": json.dumps({"stage": stage, "model_view": {}})},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "strict": True,
                "schema": strict_provider_schema(model.model_json_schema()),
            },
        },
    }


def fixture(tmp_path):
    rows = [
        {
            "stage": s,
            "task": "public",
            "event": 0,
            "source_request_sha256": "a" * 64,
            "after": request(s),
        }
        for s in probe.STAGES
    ]
    review = tmp_path / "review.json"
    review.write_text(json.dumps(rows))
    inputs = tmp_path / "requests.json"
    probe.prepare(review, inputs)
    server = {
        "image": probe.IMAGE,
        "model": probe.MODEL,
        "revision": probe.REVISION,
        "context_tokens": 32768,
        "max_num_seqs": 1,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "weights_manifest_sha256": "a" * 64,
        "cuda": {
            "vllm": "0.27.1",
            "kernel": "passed",
            "device_count": 1,
            "gpu": "NVIDIA H100 80GB HBM3",
            "memory_bytes": 80 * 1024**3,
        },
    }
    path = tmp_path / "server.json"
    path.write_text(json.dumps(server))
    return inputs, path, tmp_path / "run"


def transport(calls):
    def send(url, body, timeout):
        calls.append((url, body))
        if url.endswith("/tokenize"):
            return {"count": 13000, "max_model_len": 32768}
        stage = json.loads(body["messages"][1]["content"])["stage"]
        value = probe.response_model(stage)(stage_complete=True).model_dump(mode="json")
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(value)}}
            ],
            "usage": {"total_tokens": 14000},
        }

    return send


def test_complete_four_calls_and_resume_without_network(tmp_path):
    args = fixture(tmp_path)
    calls = []
    result = probe.run(*args, transport=transport(calls))
    assert len(calls) == 8  # Four tokenizations, four physical generations.
    assert result["compatibility_passed"] and result["observed_total_tokens"] == 56000
    assert not result["scientific_correctness_assessed"]
    assert probe.run(*args, transport=transport(calls)) == result
    assert len(calls) == 8
    inputs = sealed_read(args[0])
    assert [b for u, b in calls if u.endswith("completions")] == [
        r["body"] for r in inputs["requests"]
    ]


def test_selection_is_largest_per_stage_not_best_reply(tmp_path):
    args = fixture(tmp_path)
    review = tmp_path / "review.json"
    rows = json.loads(review.read_text())
    extra = deepcopy(rows[0])
    extra["after"]["messages"][0]["content"] += "More public context."
    extra["event"] = 1
    review.write_text(json.dumps([*rows, extra]))
    value = probe.prepare(review, tmp_path / "other.json")
    assert len(value["requests"]) == 4 and value["requests"][0]["event"] == 1
    with pytest.raises(ValueError, match="frozen artifact differs"):
        probe.prepare(review, args[0])


@pytest.mark.parametrize(
    "token",
    [
        {"count": 24001, "max_model_len": 32768},
        {"count": 100, "max_model_len": 16384},
        {"count": True, "max_model_len": 32768},
    ],
)
def test_token_preflight_blocks_generation(tmp_path, token):
    args = fixture(tmp_path)
    calls = []

    def send(url, body, timeout):
        assert url.endswith("tokenize")
        calls.append(url)
        return token

    result = probe.run(*args, transport=send)
    assert not result["compatibility_passed"]
    assert result["physical_generation_requests_started"] == 0
    probe.run(*args, transport=send)
    assert len(calls) == 4


def test_interrupted_generation_is_never_resent(tmp_path):
    args = fixture(tmp_path)
    calls = []
    normal = transport(calls)

    def interrupt(url, body, timeout):
        if url.endswith("completions"):
            raise KeyboardInterrupt
        return normal(url, body, timeout)

    with pytest.raises(KeyboardInterrupt):
        probe.run(*args, transport=interrupt)
    result = probe.run(*args, transport=normal)
    assert result["physical_generation_requests_started"] == 4
    assert result["usage_missing_events"] == 1
    assert result["rows"][0]["status"] == "response_unusable"
    assert len([u for u, _ in calls if u.endswith("completions")]) == 3


@pytest.mark.parametrize(
    "failure", ["length", "schema", "missing_fields", "missing_usage", "http", "shape"]
)
def test_provider_failure_retained_and_not_retried(tmp_path, failure):
    args = fixture(tmp_path)
    calls = []
    normal = transport(calls)

    def send(url, body, timeout):
        value = normal(url, body, timeout)
        if url.endswith("completions"):
            if failure == "shape":
                return []
            if failure == "http":
                raise ValueError("HTTP 500")
            if failure == "length":
                value["choices"][0]["finish_reason"] = "length"
            if failure == "schema":
                value["choices"][0]["message"]["content"] = '{"invented": true}'
            if failure == "missing_fields":
                value["choices"][0]["message"]["content"] = '{"stage_complete": true}'
            if failure == "missing_usage":
                del value["usage"]
        return value

    first = probe.run(*args, transport=send)
    assert not first["compatibility_passed"]
    assert probe.run(*args, transport=send) == first
    assert len(calls) == 8


def test_tampered_call_or_changed_server_refused(tmp_path):
    args = fixture(tmp_path)
    probe.run(*args, transport=transport([]))
    call = args[2] / "call-0.json"
    value = json.loads(call.read_text())
    value["status"] = "fake"
    call.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="digest"):
        probe.run(*args, transport=transport([]))
    value = json.loads(args[1].read_text())
    value["max_num_seqs"] = 4
    args[1].write_text(json.dumps(value))
    with pytest.raises(ValueError, match="server settings"):
        probe.run(*args, transport=transport([]))


def test_changed_sampling_or_schema_refused(tmp_path):
    args = fixture(tmp_path)
    value = sealed_read(args[0])
    value.pop("artifact_sha256")
    value["requests"][0]["body"]["temperature"] = 0.8
    other = tmp_path / "changed.json"
    sealed_write(other, value)
    with pytest.raises(ValueError, match="request settings"):
        probe.run(other, *args[1:], transport=transport([]))


def test_changed_runtime_refuses_resume(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    probe.run(*args, transport=transport([]))
    monkeypatch.setattr(probe, "source_identity", lambda: "changed")
    with pytest.raises(ValueError, match="frozen artifact differs"):
        probe.run(*args, transport=transport([]))


def test_model_and_generation_settings_match_aces():
    from pathlib import Path

    config = json.loads(Path("configs/phase_c_construction_v1.json").read_text())
    settings = config["model_settings"]
    body = request("variables")
    assert settings["model"] == probe.MODEL
    assert settings["model_revision"] == probe.REVISION
    assert body["max_tokens"] == settings["max_output_tokens"]
    assert body["temperature"] == settings["temperature"]
    assert body["reasoning_effort"] == settings["reasoning_effort"]
    assert config["served_context_tokens"] == 32768


def test_weight_hash_validation(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}")
    expected = {"config.json": verify_weights.local_hashes(path)}
    assert verify_weights.compare(tmp_path, expected) == []
    path.write_text('{"changed": true}')
    assert "not the published" in verify_weights.compare(tmp_path, expected)[0]


def test_server_record_rejects_public_binding_and_different_mounts(tmp_path):
    _, server, _ = fixture(tmp_path)
    (tmp_path / "cuda.json").write_text(
        json.dumps(json.loads(server.read_text())["cuda"])
    )
    (tmp_path / "weights.json").write_text(
        json.dumps({"repo": probe.MODEL, "revision": probe.REVISION})
    )
    inspect = {
        "Image": "sha256:verified-image-id",
        "Config": {
            "Image": probe.IMAGE,
            "Cmd": [
                "--model",
                probe.MODEL,
                "--revision",
                probe.REVISION,
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--max-model-len",
                "32768",
                "--max-num-seqs",
                "1",
                "--gpu-memory-utilization",
                "0.90",
                "--tensor-parallel-size",
                "1",
            ],
            "Env": ["HF_HOME=/af/hf", "HF_HUB_OFFLINE=1"],
        },
        "HostConfig": {
            "PortBindings": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8000"}]}
        },
        "Mounts": [
            {"Destination": "/af/hf", "Source": str(tmp_path / "hf"), "RW": False},
            {"Destination": "/af/cache", "Source": str(tmp_path / "cache"), "RW": True},
        ],
    }
    assert server_record(inspect, tmp_path)["platform"] == "jetstream2-h100x1-docker"
    inspect["HostConfig"]["PortBindings"]["8000/tcp"][0]["HostIp"] = "0.0.0.0"
    with pytest.raises(ValueError, match="loopback"):
        server_record(inspect, tmp_path)
