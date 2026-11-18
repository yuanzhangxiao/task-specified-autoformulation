"""Local judge isolation, logging, explicit authorization, and failed gates."""

import json
from pathlib import Path

import pytest

from autoformalism.llm import jetstream, local_judge
from autoformalism.llm.exceptions import LLMProviderError
from autoformalism.rebuttal import component_critic as critic
from autoformalism.rebuttal import local_judge_calibration as calibration
from autoformalism.rebuttal.prefit_replay import sealed_write


@pytest.mark.parametrize(
    "url",
    [
        jetstream.BASE_URL,
        "http://8.8.8.8:9000",
        "http://0.0.0.0:9000",
        "http://127.0.0.1:9000/elsewhere",
        "http://a:b@localhost:9000",
        "http://localhost",
    ],
)
def test_remote_and_ambiguous_endpoints_are_refused(url):
    with pytest.raises(ValueError):
        local_judge.checked_url(url)


def test_local_transport_preserves_body_records_usage_and_redacts(tmp_path):
    key = "synthetic-test-credential"

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(
                {"model": local_judge.MODEL, "echo": key, "usage": {"total_tokens": 7}}
            ).encode()

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "http://127.0.0.1:9000/v1/chat/completions"
            assert request.get_header("Authorization") == "Bearer " + key
            body = json.loads(request.data)
            assert body["model"] == local_judge.MODEL and body["temperature"] == 0.2
            return Response()

    transport = local_judge.LocalJudgeTransport(
        tmp_path, "http://127.0.0.1:9000", lambda: key
    )
    transport._opener = Opener()
    result = transport(
        transport.base_url + "/v1/chat/completions",
        {"model": local_judge.MODEL, "temperature": 0.2},
        10,
    )
    assert result["echo"] == "[REDACTED]"
    assert result["usage"]["total_tokens"] == 7
    assert all(key not in p.read_text() for p in tmp_path.rglob("*.json"))
    with pytest.raises(ValueError):
        transport(
            jetstream.BASE_URL + "/v1/chat/completions",
            {"model": local_judge.MODEL},
            10,
        )


def test_outage_is_recorded_and_never_silently_retried(tmp_path):
    class Opener:
        def open(self, *_args, **_kwargs):
            raise TimeoutError()

    transport = local_judge.LocalJudgeTransport(
        tmp_path, "http://localhost:9000", lambda: "synthetic"
    )
    transport._opener = Opener()
    with pytest.raises(LLMProviderError):
        transport(
            transport.base_url + "/v1/chat/completions", {"model": local_judge.MODEL}, 1
        )
    saved = json.loads((tmp_path / "call-000/result.json").read_text())
    assert saved["status"] == "transport_error"
    assert len(list(tmp_path.glob("call-*"))) == 1


def test_local_critic_never_requests_jetstream_credential(tmp_path, monkeypatch):
    plan = {"artifact_sha256": "p", "critic_transport": local_judge.POLICY}
    sealed_write(
        tmp_path / "critic_authorization.json",
        {
            "identity": "p",
            "known_case_gate_passed": True,
            "transport_policy": local_judge.POLICY,
        },
    )
    monkeypatch.setenv("AF_LOCAL_CRITIC_URL", "http://127.0.0.1:9000")
    monkeypatch.setenv("AF_LOCAL_CRITIC_KEY", "synthetic")

    def forbidden():
        pytest.fail("hosted credential accessed")

    url, _ = critic.transport_for(tmp_path, plan, tmp_path / "wire", forbidden)
    assert url == "http://127.0.0.1:9000"
    with pytest.raises(ValueError, match="authorization"):
        critic.checked_authorization(tmp_path, {"artifact_sha256": "p"})


def test_failed_or_incomplete_calibration_cannot_authorize_search(
    tmp_path, monkeypatch
):
    plan = {
        "artifact_sha256": "p",
        "serving_image_sha256": "a" * 64,
        "output_contract": jetstream.OUTPUT_CONTRACT,
    }
    monkeypatch.setattr(calibration, "verify", lambda _: (plan, {}))
    campaign = {
        "critic_transport": local_judge.POLICY,
        "serving_image_sha256": "a" * 64,
    }
    for status, passed in [("incomplete", None), ("complete", False)]:
        monkeypatch.setattr(
            calibration,
            "report",
            lambda _, s=status, p=passed: {"status": s, "known_case_gate_passed": p},
        )
        with pytest.raises(ValueError, match="passed gates"):
            calibration.bind(tmp_path, campaign, Path("unused"))
    assert not (tmp_path / "critic_authorization.json").exists()
