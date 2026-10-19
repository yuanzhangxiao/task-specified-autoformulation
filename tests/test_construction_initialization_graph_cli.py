"""Offline saved-topology CLI, immutable history and bounded cached execution."""

from __future__ import annotations

import copy
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from autoformalism.llm.staged_topology import StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.staged_topology import content_hash
from scripts import construction_initialization_graph as cli
from scripts import saved_initialization_aces as serving
from tests.test_topology_compatibility import draft, evidence, trajectory


def seal(value):
    return {**value, "artifact_sha256": content_hash(value)}


def historical_plan(count=2):
    """Use public synthetic trajectories with no benchmark-specific coordinates."""
    model = draft(sources=("z",), extras=[("z", "differential", ())])
    packet = evidence(trajectory("left", (0, 1, 2)), trajectory("right", (0, 2, 3)))
    extended = {**model.model_dump(mode="json"), "external_exchanges": ["retained"]}
    question = {
        "question_id": "old-pair",
        "property": "closed_subsystem",
        "target": "a",
        "status": "conditional_conflict",
        "reason": "Old shared-initial rule.",
        "witnesses": [
            {
                "trajectories": ["left", "right"],
                "sample_indices": [1, 1],
                "times": [1, 1],
                "observations": [1, 2],
            }
        ],
    }
    cost = {"physical_requests": 9, "budget_charge": 900, "observed_tokens": 800}
    cases = {}
    for seed in range(count):
        cases[f"case-{seed}"] = {
            "task": {"task_id": f"case-{seed}", "seed": seed},
            "source_result": seal({"draft": extended, "cost": cost}),
            "start": {
                "draft": extended,
                "initial_information_contract": {"permitted_symbols": ["a", "u", "t"]},
                "scientific_state": {
                    "tracked_conflicts": [question],
                    "advisory_questions": [],
                },
                "predecessor_historical_cost": {
                    "physical_requests": 4,
                    "budget_charge": 400,
                },
            },
            "cell": {
                "training_evidence": packet.model_dump(mode="json"),
                "context": packet.context.model_dump(mode="json"),
                "brief": {
                    "scientific_context": "Measured output a has latent memory z."
                },
            },
        }
    return seal(
        {
            "protocol": cli.SOURCE_PROTOCOL,
            "cases": cases,
            "parent_plan": {"settings": StagedModelSettings().model_dump(mode="json")},
            "test_data_opened": False,
            "optimizer_calls": 0,
            "automatic_followup": False,
        }
    )


def write_archive(path, plan):
    summary = {
        "plan_sha256": plan["artifact_sha256"],
        "cost": {"physical_requests": 2, "budget_charge": 200, "observed_tokens": 190},
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, value in {
            "plan.json": plan,
            "summary.json": summary,
            "results/historical/result.json": {"preserve": "original terminal result"},
        }.items():
            blob = json.dumps(value).encode()
            member = tarfile.TarInfo(name)
            member.size = len(blob)
            archive.addfile(member, io.BytesIO(blob))


@pytest.fixture
def prepared(tmp_path):
    source, root = tmp_path / "source.tar.gz", tmp_path / "new"
    write_archive(source, historical_plan())
    cli.prepare(source, root)
    return source, root


def reply(payload):
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(payload)}}
        ],
        "usage": {"total_tokens": 31},
    }


def valid_reply():
    return {
        "records": [
            {
                "state": "z",
                "rule": {"mode": "value", "scope": "trajectory"},
                "justification": "Initial hidden preparation may vary across rows.",
            }
        ]
    }


def tokenizer(*args):
    return {"count": 100, "max_model_len": 32768}


def test_prepare_preserves_full_source_and_scopes_projection_without_calls(prepared):
    source, root = prepared
    assert source.read_bytes() == (root / "historical/source.tar.gz").read_bytes()
    plan = cli.verify(root)
    assert plan["settings"]["maximum_requests"] == 2
    assert plan["settings"]["attempts_per_step"] == 2
    assert plan["settings"]["max_output_tokens"] == 8192
    assert plan["settings"]["maximum_total_tokens"] == 262144
    case = sealed_read(root / "inputs/case-0.json")
    assert case["projection"]["omitted_fields"] == {
        "external_exchanges": content_hash(["retained"])
    }
    assert "external_exchanges" not in case["draft"]
    assert case["contract"]["observed_initial_channels"] == {"a": "a"}
    assert case["contract"]["trajectory_initialization"] == "observation_prefix"
    assert case["contract"]["prefix_policy"] is None
    assert len(case["previous_questions"]) == 1
    assert cli.report(root)["cost"]["physical_requests"] == 0
    assert not (root / "results").exists()
    original = (root / "plan.json").read_bytes()
    cli.prepare(source, root)
    assert (root / "plan.json").read_bytes() == original


@pytest.mark.parametrize(
    "mutation", ("protocol", "heldout", "three_cases", "forged_seal", "unsafe_name")
)
def test_incompatible_sources_are_rejected(tmp_path, mutation):
    plan = historical_plan()
    if mutation == "protocol":
        plan["protocol"] = "other"
    elif mutation == "heldout":
        plan["test_data_opened"] = True
    elif mutation == "three_cases":
        plan["cases"]["case-2"] = copy.deepcopy(plan["cases"]["case-0"])
    elif mutation == "unsafe_name":
        plan["cases"]["../escape"] = plan["cases"].pop("case-0")
    if mutation != "forged_seal":
        plan = seal(
            {key: value for key, value in plan.items() if key != "artifact_sha256"}
        )
    else:
        plan["artifact_sha256"] = "0" * 64
    source = tmp_path / "bad.tar.gz"
    write_archive(source, plan)
    with pytest.raises(ValueError):
        cli.prepare(source, tmp_path / "new")


def test_tampered_evidence_seal_fails_before_preview(tmp_path):
    plan = historical_plan()
    plan["cases"]["case-0"]["cell"]["training_evidence"]["sha256"] = "0" * 64
    plan = seal({key: value for key, value in plan.items() if key != "artifact_sha256"})
    source = tmp_path / "source.tar.gz"
    write_archive(source, plan)
    with pytest.raises(ValueError, match="digest"):
        cli.prepare(source, tmp_path / "new")


def test_measured_binding_cannot_silently_become_an_editable_root(tmp_path):
    plan = historical_plan()
    plan["cases"]["case-0"]["start"]["initial_information_contract"][
        "measured_initial_bindings"
    ] = [{"state": "z", "source": "a"}]
    plan = seal({key: value for key, value in plan.items() if key != "artifact_sha256"})
    source = tmp_path / "source.tar.gz"
    write_archive(source, plan)
    with pytest.raises(ValueError, match="measured initial bindings"):
        cli.prepare(source, tmp_path / "new")


def test_verify_rejects_runtime_or_resealed_setting_changes(prepared, monkeypatch):
    _, root = prepared
    plan = sealed_read(root / "plan.json")
    monkeypatch.setattr(cli, "runtime_source_hash", lambda: "changed")
    with pytest.raises(ValueError, match="runtime identity"):
        cli.verify(root)
    monkeypatch.undo()
    plan["settings"]["maximum_requests"] = 3
    plan = seal({key: value for key, value in plan.items() if key != "artifact_sha256"})
    (root / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="runtime identity"):
        cli.verify(root)


def test_run_resume_retains_costs_and_keeps_scientific_deferrals(prepared):
    source, root = prepared
    before = source.read_bytes()
    requests = []

    def transport(url, body, timeout):
        requests.append(body)
        return reply(valid_reply())

    first = cli.run(
        root, "http://127.0.0.1:8000", transport=transport, token_transport=tokenizer
    )
    assert len(requests) == 2
    for index, request in enumerate(requests):
        preview = json.loads((root / "preview" / f"case-{index}.json").read_text())
        assert request["messages"] == [
            {"role": "system", "content": preview["system"]},
            {"role": "user", "content": preview["user"]},
        ]
        assert (
            request["response_format"]["json_schema"]["schema"]
            == preview["response_schema"]
        )
    assert first["cost"]["physical_requests"] == 2
    assert first["cost"]["observed_tokens"] == 62
    assert first["historical_source_diagnostic_cost"]["observed_tokens"] == 190
    assert all(
        row["historical_source_model_cost"]["observed_tokens"] == 800
        for row in first["rows"]
    )
    assert all(row["trajectory_roots"] == ["z"] for row in first["rows"])
    assert all(row["retained_unresolved_witness_checks"] == 1 for row in first["rows"])
    assert not first["scientific_correctness_established"]

    def blocked(*args):
        pytest.fail("terminal resume attempted a provider or tokenizer call")

    assert (
        cli.run(
            root, "http://127.0.0.1:8000", transport=blocked, token_transport=blocked
        )
        == first
    )
    assert source.read_bytes() == before


def test_schema_repairs_stop_at_two_calls_per_case(prepared):
    _, root = prepared
    calls = []

    def transport(*args):
        calls.append(args)
        return reply({"records": []})

    result = cli.run(
        root, "http://127.0.0.1:8000", transport=transport, token_transport=tokenizer
    )
    assert len(calls) == 4
    assert result["status_counts"] == {"repair_exhausted": 2}
    assert result["cost"]["physical_requests"] == 4
    assert not result["automatic_followup"]


def test_provider_failures_are_charged_without_hidden_retry(prepared):
    _, root = prepared
    calls = []

    def transport(*args):
        calls.append(args)
        raise OSError("synthetic unavailable provider")

    first = cli.run(
        root, "http://127.0.0.1:8000", transport=transport, token_transport=tokenizer
    )
    assert len(calls) == 2
    assert first["status_counts"] == {"unavailable": 2}
    assert first["cost"]["requests_with_unknown_usage"] == 2
    assert first["cost"]["budget_charge"] > 0
    assert (
        cli.run(
            root,
            "http://127.0.0.1:8000",
            transport=transport,
            token_transport=tokenizer,
        )
        == first
    )
    assert len(calls) == 2


def test_tokenizer_failure_is_terminal_without_generation_or_retry(prepared):
    _, root = prepared
    calls = []

    def fail(*args):
        calls.append(args)
        raise OSError("synthetic tokenizer failure")

    first = cli.run(
        root,
        "http://127.0.0.1:8000",
        transport=lambda *args: pytest.fail("generation"),
        token_transport=fail,
    )
    assert len(calls) == 2
    assert first["status_counts"] == {"unavailable": 2}
    assert first["cost"]["physical_requests"] == 0
    assert cli.run(root, "http://127.0.0.1:8000", token_transport=fail) == first
    assert len(calls) == 2


def test_interrupted_generation_stays_charged_and_is_not_resent(tmp_path):
    source, root = tmp_path / "source.tar.gz", tmp_path / "new"
    write_archive(source, historical_plan(count=1))
    cli.prepare(source, root)

    def interrupted(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        cli.run(
            root,
            "http://127.0.0.1:8000",
            transport=interrupted,
            token_transport=tokenizer,
        )
    before = cli.report(root)
    assert before["status"] == "pending"
    assert before["cost"]["physical_requests"] == 1
    charge = before["cost"]["budget_charge"]
    resumed = cli.run(
        root,
        "http://127.0.0.1:8001",
        transport=lambda *args: pytest.fail("interrupted request was resent"),
        token_transport=lambda *args: pytest.fail("cached tokenization was resent"),
    )
    assert resumed["status_counts"] == {"unavailable": 1}
    assert resumed["cost"]["physical_requests"] == 1
    assert resumed["cost"]["statuses"] == {"uncertain": 1}
    assert resumed["cost"]["budget_charge"] == charge


@pytest.mark.parametrize(
    "url",
    (
        "http://example.com:8000",
        "http://user:secret@127.0.0.1:8000",
        "http://127.0.0.1:8000/?key=value",
        "https://127.0.0.1:8000",
    ),
)
def test_nonlocal_or_authenticated_endpoints_are_rejected(prepared, url):
    _, root = prepared
    with pytest.raises(ValueError, match="loopback"):
        cli.run(root, url)
    assert not (root / "endpoint.json").exists()


def test_changed_loopback_port_preserves_cached_resume_and_address_history(prepared):
    _, root = prepared
    first = cli.run(
        root,
        "http://127.0.0.1:8000",
        transport=lambda *args: reply(valid_reply()),
        token_transport=tokenizer,
    )
    assert (
        cli.run(
            root,
            "http://127.0.0.1:8001",
            transport=lambda *args: pytest.fail("cached resume generated a reply"),
            token_transport=lambda *args: pytest.fail("cached resume counted tokens"),
        )
        == first
    )
    assert len(list((root / "endpoint_history").glob("*.json"))) == 2


def test_optional_server_receipt_uses_existing_runtime_validation(
    prepared, monkeypatch, tmp_path
):
    _, root = prepared
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    called = []
    monkeypatch.setattr(cli, "bind_server", lambda *args: called.append(args))
    cli.run(
        root,
        "http://127.0.0.1:8000",
        runtime,
        transport=lambda *args: reply(valid_reply()),
        token_transport=tokenizer,
    )
    assert called == [(root, runtime, "http://127.0.0.1:8000")]


def test_runtime_receipt_validation_never_loads_historical_campaign_protocol(
    prepared,
    monkeypatch,
    tmp_path,
):
    _, root = prepared
    metadata = {
        "visible_gpus": 1,
        "cuda_kernel_passed": True,
        "gpu_name": "NVIDIA H100",
        "vllm": "0.27.1",
        "harmony": {
            "protocol": serving.harmony_vocab.PROTOCOL,
            "filename": serving.harmony_vocab.VOCAB_FILENAME,
            "sha256": serving.harmony_vocab.VOCAB_SHA256,
            "harmony_render_passed": True,
            "openai_harmony": "fixture",
            "rendered_tokens": 7,
            "generation_requests": 0,
        },
    }
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(metadata))
    monkeypatch.setattr(serving, "verify", lambda *args: pytest.fail("old protocol"))
    monkeypatch.setattr(
        serving,
        "urlopen",
        lambda *args, **kwargs: io.StringIO(
            json.dumps({"data": [{"id": serving.MODEL}]})
        ),
    )
    cli.run(
        root,
        "http://127.0.0.1:8000",
        runtime,
        transport=lambda *args: reply(valid_reply()),
        token_transport=tokenizer,
    )
    assert sealed_read(root / "serving_runtime.json") == seal(metadata)


def test_inspection_is_offline_and_contains_history_and_prompt_preview(prepared):
    source, root = prepared
    result = cli.inspect(root)
    with tarfile.open(result["archive"]) as archive:
        names = archive.getnames()
        assert {
            "plan.json",
            "SUMMARY.md",
            "PROMPTS.html",
            "historical/source.tar.gz",
        } <= set(names)
        assert (
            archive.extractfile("historical/source.tar.gz").read()
            == source.read_bytes()
        )
    prompt = json.loads((root / "preview/case-0.json").read_text())
    assert "Measured output a" in prompt["user"]
    assert "observation_prefix" in prompt["user"]
    assert "old-pair" in prompt["user"]
    templates = json.loads((root / "preview/formula-stage-templates.json").read_text())
    assert templates["system"] == cli.FORMULA_SYSTEM_PROMPT
    assert templates["called_in_this_diagnostic"] is False
    assert templates["response_schemas"][
        "independent_root"
    ] == cli.strict_provider_schema(cli.GraphRootReply.model_json_schema())
    assert templates["response_schemas"]["dependent_map"] == cli.strict_provider_schema(
        cli.GraphMapReply.model_json_schema()
    )
    assert cli.GraphRootReply.model_validate(
        templates["illustrative_examples"][0]["reply"]
    )
    assert cli.GraphMapReply.model_validate(
        templates["illustrative_examples"][1]["reply"]
    )
    page = (root / "PROMPTS.html").read_text()
    assert "Exact saved dependency requests" in page
    assert "Later interaction/formula stage: illustrative only" in page
    assert "This stage is not called" in page
    assert "<h3>System</h3>" in page and "<h3>User</h3>" in page
    assert "<h3>Schema</h3>" in page
    assert result["status"] == "pending"


def test_html_preview_preserves_string_line_breaks_and_escapes_source_text():
    text = "first equation\nsecond equation <script>unsafe</script>"
    message = "Initialization dependencies\n" + json.dumps({"model_view": text})
    rendered = cli._html_user_message(message)
    assert "first equation\nsecond equation" in rendered
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in rendered
    assert "<script>" not in rendered
    assert "\\nsecond equation" not in rendered


def test_cli_runs_as_an_independent_script_without_notebooks(tmp_path):
    source, root = tmp_path / "source.tar.gz", tmp_path / "cli"
    write_archive(source, historical_plan(count=1))
    script = Path(cli.__file__)
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "prepare",
            "--source",
            str(source),
            "--root",
            str(root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(process.stdout)["protocol"] == cli.PROTOCOL
    subprocess.run(
        [sys.executable, str(script), "inspect", "--root", str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (root / "inspection.tar.gz").exists()
