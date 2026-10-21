"""Training-only cross-stage revisions preserve parents, scopes and durable costs."""

import json
from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import construction_feedback_revision as cli
from tests.test_construction_initialization_graph_cli import reply as response
from tests.test_construction_initialization_graph_cli import tokenizer
from tests.test_construction_revision_edits import brief, problem, reply


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    # The projection validator has its own exact real-artifact/negative tests.
    # This synthetic runner fixture substitutes only that producer boundary.
    monkeypatch.setattr(cli, "validate_training_projection", lambda value: None)
    inputs = {
        "protocol": "construction-revision-training-evidence-1",
        "parents": {
            f"parent_{index}": {
                "problem": problem().model_dump(mode="json"),
                "public_brief": brief().model_dump(mode="json"),
                "scientific_aliases": {},
                "prompt_evidence": {"id": "training:I", "status": "verified"},
            }
            for index in range(8)
        },
    }
    source = tmp_path / "source.json"
    seal(source, inputs)
    root = tmp_path / "revision"
    cli.prepare(source, root)
    return source, root


def transport(*args):
    return response(reply())


def test_all_parents_seeded_revisions_and_exact_resume(prepared, monkeypatch):
    _, root = prepared
    from autoformalism.fitting import prefix_initialization

    def forbidden(*args, **kwargs):
        pytest.fail("no fitting, simulation or repeated live requests")

    for name in ("fit_training", "infer_prefix", "forecast", "symbolic_rollout"):
        monkeypatch.setattr(prefix_initialization, name, forbidden)
    before = {p.name: p.read_bytes() for p in (root / "controls").glob("*.json")}
    result = cli.run(
        root, "http://127.0.0.1", transport=transport, token_transport=tokenizer
    )
    assert result["status_counts"] == {"revised": 24}
    assert result["cost"]["physical_requests"] == 24
    assert result["optimizer_calls"] == result["solver_rollouts"] == 0
    assert not result["model_promotion"] and not result["test_data_opened"]
    assert (
        cli.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )
        == result
    )
    assert before == {
        p.name: p.read_bytes() for p in (root / "controls").glob("*.json")
    }
    packet = sealed_read(next((root / "results").glob("*/fit_handoff.json")))
    assert (
        packet["prefix_fit_problem"]["graph"]
        == problem().model_dump(mode="json")["graph"]
    )
    assert (
        packet["prefix_fit_problem"]["base_candidate"]["processes"][0]["expression"]
        == "a*mem*Gp"
    )
    assert cli.inspect(root)["status"] == "complete"


def test_failed_transaction_has_specific_feedback_and_retry(prepared):
    _, root = prepared
    seen = []

    def edits(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        seen.append(packet)
        if "repair_feedback" not in packet:
            bad = reply()
            bad["edits"][0]["sources"] = ["mem"]
            return response(bad)
        assert packet["repair_feedback"]["diagnostic"]["code"] == "SOURCE_MISMATCH"
        assert packet["current_model"]["equations"][0]["expression"] == "-q+u"
        return response(reply())

    result = cli.run(
        root, "http://127.0.0.1", transport=edits, token_transport=tokenizer
    )
    assert result["status_counts"] == {"revised": 24}
    assert result["cost"]["physical_requests"] == 48
    assert all(p["public_path_status"] for p in seen)


def test_repeated_invalid_reply_and_unchanged_control(prepared):
    _, root = prepared
    seen = []
    bad = reply()
    bad["edits"][0]["sources"] = ["mem"]

    def repeated(url, body, timeout):
        seen.append(json.loads(body["messages"][1]["content"]))
        return response(bad)

    result = cli.run(
        root, "http://127.0.0.1", transport=repeated, token_transport=tokenizer
    )
    assert result["status_counts"] == {"rejected": 24}
    assert result["cost"]["physical_requests"] == 72
    assert seen[2]["repair_feedback"]["repeated_reply"] is True
    assert not list((root / "results").glob("*/fit_handoff.json"))


def test_preview_equals_real_first_request(prepared):
    _, root = prepared
    cli.preview(root)
    saved = json.loads((root / "preview/parent_0.json").read_text())
    seen = []

    def capture(url, body, timeout):
        seen.append(body["messages"])
        return response(reply())

    cli.run(root, "http://127.0.0.1", transport=capture, token_transport=tokenizer)
    assert seen[0] == [
        {"role": "system", "content": saved["system"]},
        {"role": "user", "content": saved["user"]},
    ]


def test_deferred_episode_resumes_same_budget(prepared):
    _, root = prepared
    calls = []

    def once(url, body, timeout):
        calls.append(body)
        if len(calls) == 2:
            raise DeferredCall("synthetic drain")
        bad = reply()
        bad["edits"][0]["sources"] = ["mem"]
        return response(bad)

    partial = cli.run(
        root, "http://127.0.0.1", transport=once, token_transport=tokenizer
    )
    assert partial["status_counts"] == {"pending": 24}
    resumed = cli.run(
        root, "http://127.0.0.1", transport=transport, token_transport=tokenizer
    )
    assert resumed["status_counts"] == {"revised": 24}
    assert resumed["cost"]["physical_requests"] == 26
    assert resumed["cost"]["requests_with_unknown_usage"] == 1


@pytest.mark.parametrize("mutation", ["settings", "protocol", "extra"])
def test_corrupted_pending_cache_fails_before_spending(prepared, mutation):
    _, root = prepared
    count = 0

    def interrupted(url, body, timeout):
        nonlocal count
        count += 1
        if count == 2:
            raise DeferredCall("synthetic drain")
        bad = reply()
        bad["edits"][0]["sources"] = ["mem"]
        return response(bad)

    cli.run(root, "http://127.0.0.1", transport=interrupted, token_transport=tokenizer)
    path = next((root / "results").glob("*/calls/*.json"))
    value = json.loads(path.read_text())
    if mutation == "settings":
        value["request"]["settings"]["temperature"] = 0.9
    else:
        from autoformalism.staged_topology import content_hash

        value["request"][mutation] = "different"
        value["request_hash"] = content_hash(value["request"])
        destination = path.with_name(value["request_hash"] + ".json")
        path.rename(destination)
        path = destination
    path.write_text(json.dumps(value))

    def forbidden(*args):
        pytest.fail("corrupted cache must fail before HTTP or tokenizer requests")

    with pytest.raises(ValueError, match=r"provenance|envelope"):
        cli.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )


@pytest.mark.parametrize("kind", ["result", "request", "control", "input"])
def test_altered_artifacts_rejected(prepared, kind):
    _, root = prepared
    cli.run(root, "http://127.0.0.1", transport=transport, token_transport=tokenizer)
    if kind == "result":
        path = next((root / "results").glob("*/result.json"))
    elif kind == "request":
        path = next((root / "results").glob("*/calls/*.json"))
    elif kind == "control":
        path = next((root / "controls").glob("*.json"))
    else:
        path = root / "inputs.json"
    item = json.loads(path.read_text())
    item["tampered"] = True
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError):
        cli.report(root)


def test_heldout_inputs_rejected(prepared, tmp_path):
    source, _ = prepared
    item = json.loads(source.read_text())["value"]
    item["validation_score"] = 42
    bad = tmp_path / "heldout.json"
    seal(bad, item)
    with pytest.raises(ValueError, match="held-out"):
        cli.prepare(bad, tmp_path / "new")


def test_missing_public_path_never_exported(prepared):
    _, root = prepared
    parent = deepcopy(sealed_read(root / "inputs.json")["parents"]["parent_0"])
    parent["public_brief"] = brief(auxiliary=True).model_dump(mode="json")

    class ToyClient:
        def call(self, **kwargs):
            return {
                "status": "responded",
                "raw_response": response(reply(edits=[], new_parameters=[])),
                "request_hash": "x",
            }

    result = cli._episode(parent, ToyClient())
    assert result["status"] == "rejected"
    assert result["problem"] is None
    assert result["attempts"][-1]["diagnostic"]["code"] == "PUBLIC_PATH_REPAIR_REQUIRED"
