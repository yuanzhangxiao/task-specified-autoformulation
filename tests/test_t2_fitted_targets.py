"""Fitted-parent routing, independent alternatives, admission and ACES resume."""

import json
import tarfile
from copy import deepcopy

import pytest
from pydantic import ValidationError

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import t2_breadth as science
from autoformalism.staged_topology import content_hash
from scripts import submit_t2_breadth as submitter
from scripts import t2_breadth as pilot
from scripts import t2_breadth_scheduler as scheduler
from scripts import t2_fitted_targets as fitted
from tests.test_construction_initialization_graph_cli import reply, tokenizer
from tests.test_t2_breadth import INFUSION, MEAL, proposal, public

LOCAL = "http://127.0.0.1"


def edit(group, expression, *, name="I", kind="dynamic"):
    return {
        "decision": "revise",
        "hypothesis": f"Change the pathway to {name}",
        "expected_observable_effect": "Improve the displayed training residual",
        "evidence_ids": [fitted.OBJECTIVES[group]],
        "edits": [
            {"component": name, "kind": kind, "sources": [], "expression": expression}
        ],
    }


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    ctx, brief = public()
    topology, laws = proposal()
    topology["components"][0]["sources"] = [MEAL, "Gp", "U"]
    laws["formulas"][0]["expression"] = f"{MEAL}+Gp-U"
    topology["components"][1]["sources"] = ["Gp", "I", INFUSION]
    topology["components"][2]["sources"] = ["Q", "X"]
    topology["components"].append(
        {
            "name": "Q",
            "kind": "dynamic",
            "role": "basal utilization",
            "sources": [],
            "represents_auxiliary": "Uii",
        }
    )
    laws["formulas"][2]["expression"] = "Q+X"
    laws["formulas"].append({"component": "Q", "expression": "0"})
    laws["latent_initials"].append(
        {
            "state": "Q",
            "initial": {
                "mode": "map",
                "expression": "init_u",
                "public_initials": [{"alias": "init_u", "channel": "Uii"}],
            },
        }
    )
    values = {
        "protocol": fitted.INPUT_PROTOCOL,
        "brief": brief.model_dump(mode="json"),
        "aliases": {"Uii": "Q"},
        "evidence": {"split": "train"},
        "fitting_contract": {
            "training_sha256": "a" * 64,
            "prefix_policy": {
                "future_inputs": [MEAL, INFUSION],
                "future_auxiliaries": ["Uii", "EGP"],
                "duration": 8,
            },
        },
    }
    for group in fitted.GROUPS:
        p = science.assemble(
            science.Topology.model_validate(topology),
            science.Laws.model_validate(laws),
            ctx,
        )
        values[group] = {
            "problem": p.model_dump(mode="json"),
            "parent_reference": {"branch": group, "fitted_parameter_sha256": group},
            "fit": {"training_nmse": 0.13},
            "witnesses": {fitted.OBJECTIVES[group]: {"observed": 45, "fitted": 60}},
        }
        laws["formulas"][1]["expression"] = f"sigmoid(Gp)-a*I+{INFUSION}"
    path = tmp_path / "source.json"
    intake = sealed_write(path, values)
    monkeypatch.setattr(fitted, "INTAKE", intake["artifact_sha256"])
    monkeypatch.setattr(
        pilot.structural, "_assessment", lambda *a, **kw: {"status": "deferred"}
    )
    root = tmp_path / "result"
    pilot.prepare(root, path)
    return root


def test_two_pinned_parents_and_exact_schema(campaign):
    plan, data = pilot.load(campaign)
    assert plan["groups"] == ["c15-3", "c15-2"]
    assert (
        plan["maximum_generation_requests"] == 32 and plan["fresh_topology_seeds"] == []
    )
    for n in fitted.GROUPS:
        system, packet = pilot._packet(data, n)
        assert (
            packet["parent"] == n
            and packet["primary_evidence_id"] == fitted.OBJECTIVES[n]
        )
        assert "d(I)/dt" in " ".join(packet["equations"])
        assert packet["editable_initial_states"] == ["X"]
        assert "states" not in str(packet.get("frozen_fields", ""))
        assert "infusion" in system and "fitted_coefficients" in packet
        for k in (0, 4, 6):
            with pytest.raises(ValidationError):
                fitted.FiveFittedAlternatives.model_validate(
                    {"alternatives": [edit(n, "Gp")] * k}
                )
    assert pilot.report(campaign)["expected"] == 10
    preview = pilot.preview(campaign)
    page = (campaign / "PROMPTS.html").read_text()
    assert (
        preview["preview"] == str(campaign / "PROMPTS.html")
        and "Fresh function construction" not in page
    )
    assert "Required response schema" in page and "FiveFittedAlternatives" in page
    with pytest.raises(ValueError, match="own generation slots"):
        pilot.prepare(
            campaign, campaign / "inputs.json", campaign.parent / "predecessor"
        )


def test_five_distinct_each_and_immutable_resume(campaign):
    sent = []

    def transport(url, body, timeout):
        user = json.loads(body["messages"][1]["content"])
        group = user["parent"]
        sent.append(group)
        assert body["model"] == "openai/gpt-oss-120b"
        assert (
            body["response_format"]["json_schema"]["name"] == "FiveFittedAlternatives"
        )
        # Five different RHSs can be returned and fitted independently.
        return reply(
            {
                "alternatives": [
                    edit(group, f"{j + 2}*Gp-I+{INFUSION}") for j in range(5)
                ]
            }
        )

    summary = pilot.run(campaign, LOCAL, transport=transport, token_transport=tokenizer)
    assert sent == list(fitted.GROUPS)
    assert summary["groups"] == {g: {"eligible": 5} for g in fitted.GROUPS}
    assert summary["physical_requests"] == 2 and summary["eligible_first_pass"] == 10
    assert (
        len(summary["representatives"]) == 10
        and "first_27_physical_calls" not in summary
    )
    assert not summary["test_data_opened"] and summary["optimizer_calls"] == 0

    def deny(*args):
        pytest.fail("resume attempted new provider access")

    assert pilot.run(campaign, LOCAL, transport=deny, token_transport=deny) == summary
    for name in summary["representatives"]:
        row = sealed_read(campaign / "results" / f"{name}.json")
        h = sealed_read(campaign / "handoff" / f"{name}.json")
        assert (
            row["artifact_sha256"] == h["source_receipt"]
            and h["parent_reference"]["branch"] == row["group"]
        )
        assert (
            h["objective"] == fitted.OBJECTIVES[row["group"]]
            and h["fitting_status"] == "pending"
        )
    report = pilot.inspect(campaign)
    with tarfile.open(report["archive"]) as archive:
        assert len([f for f in archive.getnames() if f.startswith("handoff/")]) == 10
        assert "REPORT.html" in archive.getnames()


def test_genuine_differences_and_specific_repairs(campaign):
    root = campaign

    def transport(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        g = p["parent"]
        good = edit(g, f"Gp-2*I+{INFUSION}")
        if "previous_reply" not in p:
            if g == "c15-3":
                return reply({"alternatives": [good] * 5})
            return reply({"alternatives": [edit(g, f"sigmoid(Gp)-a*I-{INFUSION}")] * 5})
        reason = p["rejection"]["code"]
        if g == "c15-2":
            assert reason == "UNCHANGED_ON_TARGET_MEALS"
        else:
            assert reason == "DUPLICATE_ALTERNATIVE" and p["accepted_siblings"]
        return reply(edit(g, f"Gp-{3 + len(p['accepted_siblings'])}*I+{INFUSION}"))

    result = pilot.run(root, LOCAL, transport=transport, token_transport=tokenizer)
    assert (
        result["eligible_first_pass"] == 1 and result["eligible_after_correction"] == 9
    )
    assert result["physical_requests"] == 11 and result["corrections_attempted"] == 9
    assert len(result["representatives"]) == 10


def test_spelling_upstream_and_external_only_rollout(campaign):
    _, data = pilot.load(campaign)
    n = "c15-2"
    patch = edit(n, f"{MEAL}+Gp-U+τ_shape*I", name="Gp")
    patch["new_parameters"] = [{"name": "tau_shape", "role": "rate"}]
    result = pilot._assess(data, n, patch, "patch")
    assert result["status"] == "eligible" and result["checks"]["identifier_spellings"]
    assert result["problem"]["graph"] == data[n]["problem"]["graph"]
    forbidden = edit(n, f"{MEAL}+EGP-U", name="Gp")
    assert (
        "Future forcing"
        in pilot._assess(data, n, forbidden, "patch")["diagnostic"]["message"]
    )
    absent = edit(n, "undefined+I")
    assert pilot._assess(data, n, absent, "patch")["status"] == "rejected"
    other = deepcopy(patch)
    other["evidence_ids"] = ["validation_scores"]
    assert (
        pilot._assess(data, n, other, "patch")["diagnostic"]["code"]
        == "UNKNOWN_EVIDENCE"
    )


def test_batch_length_recovery_and_request_ceiling(campaign):
    def transport(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        if "previous_reply" not in p:
            broken = reply({"alternatives": []})
            broken["choices"][0]["finish_reason"] = "length"
            return broken
        if p["previous_reply"] is None:
            assert p["rejection"]["code"] == "MISSING_BATCH_OPTION"
        return reply(edit(p["parent"], "undeclared"))

    result = pilot.run(campaign, LOCAL, transport=transport, token_transport=tokenizer)
    assert result["physical_requests"] == 32 and result["recorded"] == 10
    assert result["groups"] == {n: {"rejected": 5} for n in fitted.GROUPS}
    assert [d["physical_requests"] for d in result["depth_comparison"]] == [12, 22, 32]
    assert (
        sealed_read(campaign / "first/c15-3-0.json")["initial"]["status"]
        == "unavailable"
    )


def test_uncertain_batch_is_not_reissued(campaign):
    def interrupted(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        pilot.run(campaign, LOCAL, transport=interrupted, token_transport=tokenizer)

    def remaining(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        assert p["parent"] == "c15-2"
        return reply(
            {"alternatives": [edit(p["parent"], f"Gp-{i}*I") for i in range(2, 7)]}
        )

    s = pilot.run(campaign, LOCAL, transport=remaining, token_transport=tokenizer)
    assert (
        s["physical_requests"] == 2
        and s["usage_missing"] == 1
        and s["unavailable_branches"] == 5
    )


def test_slurm_fitted_identity_and_lost_ack(campaign, monkeypatch, tmp_path):
    image = tmp_path / "fake-image"
    image.write_bytes(b"example")
    monkeypatch.setattr(submitter.aces, "_assets", lambda: (image, {}))
    monkeypatch.setattr(
        submitter.aces.harmony_vocab, "verify", lambda fld: {"sha256": "abc"}
    )

    def submit(wave, argv):
        assert "--job-name=phasec-t2-fitted" in argv and "--gres=gpu:h100:2" in argv
        assert "--dependency=singleton" in argv and argv[-1] == "t2-breadth"
        (wave / "propose.intent.json").write_text(json.dumps({"argv": argv}))
        raise ValueError("socket timeout")

    monkeypatch.setattr(submitter, "_submit_job", submit)
    lookup = []

    def query(argv):
        lookup.append(argv)
        return ""

    monkeypatch.setattr(scheduler, "_query", query)
    first = submitter.submit(campaign)
    assert first["status"] == "acknowledgment_pending" and any(
        "--name=phasec-t2-fitted" in x for x in lookup
    )
    monkeypatch.setattr(
        scheduler, "reconcile", lambda *a: {"status": "confirmed", "job": "2320001"}
    )
    assert submitter.submit(campaign)["job"] == "2320001"
    assert submitter.submit(campaign)["maximum_generation_requests"] == 32
    assert len(list((campaign / "submissions").iterdir())) == 1


def test_different_intake_fails_before_generation(campaign):
    path = campaign / "inputs.json"
    row = sealed_read(path)
    row["c15-2"]["parameters_with_training_roots"] = {}
    row.pop("artifact_sha256")
    path.write_text(json.dumps({**row, "artifact_sha256": content_hash(row)}))
    with pytest.raises(ValueError, match="reviewed"):
        pilot.load(campaign)


def test_foreign_campaign_refused_before_prepare(campaign, tmp_path):
    dest = tmp_path / "other"
    with pytest.raises(ValueError, match="different T2 campaign"):
        pilot.prepare(dest, campaign / "inputs.json", expected_protocol=pilot.PROTOCOL)
    assert not dest.exists()


def test_new_transit_state_and_initializer_spellings(campaign):
    _, data = pilot.load(campaign)
    group = "c15-3"
    change = edit(group, f"a*(Gp-I)+Transit+{INFUSION}")
    change["new_states"] = [
        {
            "name": "Transit",
            "description": "Post-meal driver",
            "rhs": f"{MEAL}-Transit",
            "sources": [MEAL, "Transit"],
            "initial": {"mode": "map", "expression": "0"},
        }
    ]
    change["initialization_edits"] = [
        {
            "state": "X",
            "initial": {
                "mode": "map",
                "expression": "\N{GREEK SMALL LETTER ALPHA}0*y0",
                "parameters": [{"name": "alpha0", "role": "coefficient"}],
                "public_initials": [{"alias": "y0", "channel": "Gp"}],
            },
        }
    ]
    result = pilot._assess(data, group, change, "patch")
    assert result["status"] == "eligible", result.get("diagnostic")
    rules = result["problem"]["graph"]["rules"]
    assert rules["Transit"]["initial"]["expression"] == "0"
    assert rules["X"]["initial"]["expression"] == "alpha0*y0"
    assert result["checks"]["identifier_spellings"][0]["field"] == [
        "initialization_edits",
        0,
        "initial",
        "expression",
    ]
