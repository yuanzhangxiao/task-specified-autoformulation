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


def test_independent_20b_breadth_costs_yields_and_resume(campaign):
    root = campaign.parent / "separate"
    plan = pilot.prepare(
        root, campaign / "inputs.json", expected_protocol=pilot.FITTED_20B
    )
    assert plan["maximum_generation_requests"] == 160
    assert plan["settings"]["temperature"] > 0
    pilot.preview(root)
    page = (root / "PROMPTS.html").read_text()
    assert "FiveFittedAlternatives" not in page and "InScopeRevisionReply" in page
    assert pilot.report(root)["independent_draws"]["c15-3"]["initial_recorded"] == 0
    sent = []
    seed_labels = {
        int(content_hash([seed, pilot._patch_slot(f"{g}-{i}", d), 0])[:8], 16)
        % (2**31): seed
        for g in fitted.GROUPS
        for i, seed in enumerate(plan["independent_initial_seeds"])
        for d in range(4)
    }
    assert len(seed_labels) == 160

    def transport(url, body, timeout):
        question = json.loads(body["messages"][1]["content"])
        group, seed = question["parent"], seed_labels[body["seed"]]
        depth = question.get("correction_number", int("rejection" in question))
        sent.append((group, seed, depth))
        assert body["model"] == "openai/gpt-oss-20b"
        assert body["response_format"]["json_schema"]["name"] == "InScopeRevisionReply"
        assert "accepted_siblings" not in question
        if seed == 103 and depth == 0:
            return {
                "choices": [{"finish_reason": "length", "message": {"content": "{"}}],
                "usage": {"total_tokens": 6},
            }
        if seed == 102 or (seed == 100 and depth < 2):
            return reply(edit(group, "unknown_signal"))
        if seed == 103:
            assert question["rejection"]["code"] == "INCOMPLETE_PATCH"
        return reply(edit(group, f"dI/dt = {4 if depth else 2}*Gp-I+{INFUSION}"))

    summary = pilot.run(root, LOCAL, transport=transport, token_transport=tokenizer)
    assert len(sent) == summary["physical_requests"] == 52
    assert len(set(sent[:40])) == 40 and all(d == 0 for _, _, d in sent[:40])
    assert [s for g, s, _ in sent[:40] if g == "c15-3"] == list(range(100, 120))
    for group in fitted.GROUPS:
        stats = summary["independent_draws"][group]
        assert (
            stats["planned"]
            == stats["initial_recorded"]
            == stats["terminal_recorded"]
            == 20
        )
        assert stats["initial_eligible"] == 17 and stats["initial_distinct"] == 1
        assert (
            stats["terminal_eligible"] == 19
            and stats["terminal_distinct_for_fitting"] == 2
        )
        assert stats["eligible_only_after_repair"] == 2
        assert stats["improvement_after_refitting"] is None
        schedules = stats["same_call_budget_replay"]
        assert schedules["breadth_first"][0]["seeds_visited"] == 5
        assert schedules["finish_one_branch_first"][0]["seeds_visited"] < 5
        assert (
            schedules["breadth_first"][-1] == schedules["finish_one_branch_first"][-1]
        )
    assert len(list((root / "handoff").glob("*.json"))) == 4

    def deny(*args):
        pytest.fail("Unexpected provider call on resume")

    assert pilot.run(root, LOCAL, transport=deny, token_transport=deny) == summary


def test_independent_20b_scheduler(campaign, monkeypatch):
    root = campaign.parent / "separate"
    pilot.prepare(root, campaign / "inputs.json", expected_protocol=pilot.FITTED_20B)

    def assets(size):
        assert size == "20b"
        return campaign / "inputs.json", {
            "model_revision": pilot.MODELS[size]["revision"]
        }

    monkeypatch.setattr(submitter.aces, "_assets", assets)
    monkeypatch.setattr(
        submitter.aces.harmony_vocab, "verify", lambda _: {"sha256": "test"}
    )
    calls = []

    def submit(wave, argv):
        calls.append(argv)
        assert "--gres=gpu:h100:1" in argv and "--time=02:00:00" in argv
        assert "--job-name=phasec-t2-fitted-20b" in argv
        return "12345"

    monkeypatch.setattr(submitter, "_submit_job", submit)
    assert submitter.submit(root)["job"] == submitter.submit(root)["job"] == "12345"
    record = submitter.verify(root, check_assets=True)
    assert len(calls) == 1 and record["maximum_generation_requests"] == 160
    assert (
        record["model"]
        == record["generation_settings"]["model"]
        == "openai/gpt-oss-20b"
    )


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
