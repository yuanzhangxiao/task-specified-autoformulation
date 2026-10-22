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


@pytest.fixture
def prepared_v2(prepared, tmp_path, monkeypatch):
    """Exercise orchestration with real checker and synthetic producer boundary."""
    import numpy as np

    from autoformalism.data import DatasetSplit, SplitName, Trajectory
    from autoformalism.rebuttal.prefit_replay import sealed_write
    from autoformalism.research import construction_revision_contract as contract
    from autoformalism.search.topology_compatibility import (
        EvidenceContract,
        build_evidence,
    )

    rows = tuple(
        Trajectory(
            trajectory_id=name,
            time=np.array([0.0, 1.0]),
            targets={"I": np.array(values), "Gp": np.array([2.0, 2.0])},
            external_inputs={"u": np.zeros(2)},
            auxiliaries={"aux": np.ones(2)},
            fixed_covariates={},
            derivatives={},
        )
        for name, values in (("a", [1.0, 1.0]), ("b", [1.0, 3.0]))
    )
    evidence = build_evidence(
        DatasetSplit(SplitName.TRAIN, rows, "synthetic"),
        problem().context,
        EvidenceContract(
            provenance="synthetic producer boundary",
            dependencies_exhaustive=True,
            shared_parameters=True,
            open_loop=True,
            reset_impulse_free=True,
            uniqueness_assumed=True,
            interpolation={
                "u": "linear_complete_knots",
                "aux": "linear_complete_knots",
            },
            absolute_error_bounds=dict.fromkeys(("I", "Gp", "u", "aux"), 0.0),
        ),
    ).model_dump(mode="json")
    sidecar = tmp_path / "consistency.json"
    sealed_write(sidecar, {"evidence": evidence})
    monkeypatch.setattr(
        contract, "validate", lambda value, projection: value["evidence"]
    )
    source, old = prepared
    root = tmp_path / "revision-v2"
    cli.prepare(source, root, protocol=cli.V2_PROTOCOL, consistency_source=sidecar)
    return source, root, sidecar, old


def test_v2_raw_normalization_one_call_exact_resume_and_v1_root_unchanged(prepared_v2):
    source, root, sidecar, old = prepared_v2
    original = (old / "plan.json").read_bytes()
    seen = []
    raw = reply(
        edits=[
            {
                "component": "q",
                "kind": "algebraic",
                "sources": ["a", "mem", "mem"],
                "expression": "q = a*mem*Gp",
            }
        ]
    )

    def respond(url, body, timeout):
        seen.append(body["messages"])
        return response(raw)

    cli.preview(root)
    preview = json.loads((root / "preview/parent_0.json").read_text())
    result = cli.run(
        root, "http://127.0.0.1", transport=respond, token_transport=tokenizer
    )
    assert result["protocol"] == cli.V2_PROTOCOL
    assert result["status_counts"] == {"revised": 24}
    assert result["cost"]["physical_requests"] == 24
    assert seen[0] == [
        {"role": "system", "content": preview["system"]},
        {"role": "user", "content": preview["user"]},
    ]
    saved = json.loads(next((root / "results").glob("*/result.json")).read_text())
    assert saved["reply"] == raw
    normalized = saved["audit"]["normalization"]
    assert normalized["raw_reply"] == raw
    assert normalized["canonical_reply"]["edits"][0]["sources"] == ["Gp", "mem"]
    packet = sealed_read(next((root / "results").glob("*/fit_handoff.json")))
    assert packet["protocol"] == "construction-revision-fit-handoff-2"
    assert not packet["training_consistency"]["conflicts"]
    assert not packet["optimization_adequacy_established"]

    def forbidden(*args):
        pytest.fail("completed cache must replay without requests")

    assert (
        cli.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )
        == result
    )
    assert (old / "plan.json").read_bytes() == original
    with pytest.raises(ValueError, match=r"protocol|fresh"):
        cli.prepare(source, old, protocol=cli.V2_PROTOCOL, consistency_source=sidecar)
    assert (old / "plan.json").read_bytes() == original


def test_v2_actual_candidate_conflict_feedback_and_repair(prepared_v2):
    _, root, _, _ = prepared_v2
    plan = cli.verify(root)
    inputs = sealed_read(root / "inputs.json")
    parent = cli._prepared_parent(root, inputs, "parent_0", plan)
    bad = reply(
        edits=[
            {
                "component": "I",
                "kind": "dynamic",
                "sources": ["a", "u"],
                "expression": "d(I)/dt = -a*I+u",
            }
        ]
    )

    class Client:
        def __init__(self):
            self.packets = []

        def call(self, **kwargs):
            packet = json.loads(kwargs["user"])
            self.packets.append(packet)
            payload = bad if len(self.packets) == 1 else reply()
            return {
                "status": "responded",
                "request_hash": str(len(self.packets)),
                "raw_response": response(payload),
            }

    client = Client()
    outcome = cli._episode(parent, client)
    assert outcome["status"] == "revised"
    feedback = client.packets[1]["repair_feedback"]
    assert feedback["diagnostic"]["code"] == "MODEL_DATA_CONFLICT"
    assert feedback["diagnostic"]["details"]["assessment"]["conflicts"]
    assert feedback["last_reply"] == bad
    assert "raw_reply" not in feedback["diagnostic"]
    assert parent["problem"] == problem().model_dump(mode="json")


def test_v2_retain_cannot_bypass_conflict(prepared_v2):
    _, root, _, _ = prepared_v2
    from autoformalism.research.construction_revision_edits import (
        RevisionValidationError,
    )

    parent = cli._prepared_parent(
        root, sealed_read(root / "inputs.json"), "parent_0", cli.verify(root)
    )
    parent["problem"]["base_candidate"]["state_equations"][0]["rhs"] = "-a*I+u"
    with pytest.raises(RevisionValidationError, match="conflict"):
        cli._apply(parent, reply(edits=[]))


def test_v2_evidence_and_policy_are_sealed(prepared_v2):
    _, root, _, _ = prepared_v2
    path = root / "consistency_evidence.json"
    value = json.loads(path.read_text())
    value["evidence"]["contract"]["uniqueness_assumed"] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        cli.verify(root)
