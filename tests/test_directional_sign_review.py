"""Semantic sign support, advisory citation accounting and resumable local repair."""

import copy
import json
import subprocess

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal import dalla_sign_repair as campaign
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.schemas.public_fitting import PublicFitRequest
from autoformalism.search import directional_sign_review as direction
from autoformalism.search import sign_review as original
from autoformalism.staged_topology import content_hash
from scripts.smoke_dalla_sign_repair import fixture
from tests.test_process_pruning_campaign import backend


def setup(tmp_path, two_slots=False):
    source, config, root = fixture(tmp_path)
    settings = public._read(config)
    settings.update(
        protocol="dalla-sign-repair-2", selected_task_ids=["toy_seed0_full"]
    )
    settings["model_settings"]["maximum_requests"] = 6
    public._write(config, settings)
    packet = public._read(source)
    result = packet["models"][0]["result"]
    if two_slots:
        request = result["selected_request"]
        request["base_candidate"]["state_equations"][0]["rhs"] += "+d*x"
        request["base_candidate"]["parameters"].append(
            {"name": "d", "role": "coefficient", "scope": "global", "domain": "real"}
        )
        result["selected_fit"]["parameters"]["d"] = -0.1
        result["artifact_sha256"] = content_hash(
            {k: v for k, v in result.items() if k != "artifact_sha256"}
        )
        public._write(source, packet)
    request = PublicFitRequest.model_validate(result["selected_request"])
    return source, config, root, request, packet["cells"]["toy"]["brief"]


def proposal(slots, quote="The output has clearance."):
    return {
        "decisions": [
            {
                "slot_id": s["slot_id"],
                "outer_weight_sign": "negative",
                "basis": "public_task",
                "public_quote": quote,
                "rationale": "Public clearance removes the target quantity.",
                "mechanism_role": "sink",
                "donor": None,
                "recipient": None,
            }
            for s in slots
        ]
    }


def assessment(raw, verdict="supported", quoted=True):
    return {
        "assessments": [
            {
                "slot_id": d["slot_id"],
                "verdict": verdict,
                "quoted_text_supports_direction": quoted,
                "rationale": "The public role supports this target contribution.",
            }
            for d in raw["decisions"]
        ]
    }


def transport(url, body, timeout):
    payload = json.loads(body["messages"][1]["content"])
    reply = (
        assessment(payload["proposed"])
        if "proposed" in payload
        else proposal(payload["eligible_slots"], "Paraphrased clearance")
    )
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
        ],
        "usage": {"total_tokens": 100},
    }


@pytest.mark.parametrize(
    "quote", ["", "Paraphrased clearance", "The output has clearance."]
)
def test_bad_or_irrelevant_citation_does_not_veto_supported_direction(tmp_path, quote):
    _, _, _, request, brief = setup(tmp_path)
    raw = proposal(original.slots(request), quote)
    result = direction.finish(
        request, brief, [{"proposal": raw, "assessment": assessment(raw, quoted=False)}]
    )
    assert result["status"] == "repaired"
    audit = result["patch"]["provenance"]["citation_audit"][0]
    assert audit["citation_credited"] is False
    assert result["reply"] == raw
    assert result["patch"]["provenance"]["scientific_correctness_certified"] is False


def test_citation_requires_both_exactness_and_separate_support(tmp_path):
    _, _, _, request, brief = setup(tmp_path)
    raw = proposal(original.slots(request))
    result = direction.finish(
        request, brief, [{"proposal": raw, "assessment": assessment(raw)}]
    )
    assert result["patch"]["provenance"]["citation_audit"][0]["citation_credited"]
    raw["decisions"][0]["public_quote"] = "A wrong quote accepted by the LLM"
    result = direction.finish(
        request, brief, [{"proposal": raw, "assessment": assessment(raw)}]
    )
    assert not result["patch"]["provenance"]["citation_audit"][0]["citation_credited"]


@pytest.mark.parametrize("omit", [True, False])
def test_missing_or_null_citation_preserves_supported_decision(tmp_path, omit):
    _, _, _, request, brief = setup(tmp_path)
    raw = proposal(original.slots(request))
    if omit:
        raw["decisions"][0].pop("public_quote")
    else:
        raw["decisions"][0]["public_quote"] = None
    result = direction.finish(
        request, brief, [{"proposal": raw, "assessment": assessment(raw)}]
    )
    assert result["status"] == "repaired"
    assert result["reply"]["decisions"][0]["public_quote"] == ""
    assert not result["patch"]["provenance"]["citation_audit"][0]["citation_credited"]


@pytest.mark.parametrize("malformation", ["missing", "duplicate", "foreign"])
def test_malformed_assessment_cannot_authorize_patch(tmp_path, malformation):
    _, _, _, request, brief = setup(tmp_path, two_slots=True)
    raw = proposal(original.slots(request))
    checked = assessment(raw)
    if malformation == "missing":
        checked["assessments"].pop()
    elif malformation == "duplicate":
        checked["assessments"][1] = checked["assessments"][0]
    else:
        checked["assessments"][1]["slot_id"] = "unrequested_slot"
    with pytest.raises(ValueError, match="exactly once"):
        direction.finish(request, brief, [{"proposal": raw, "assessment": checked}])


def test_exact_generic_quote_cannot_override_insufficient_direction(tmp_path):
    _, _, _, request, brief = setup(tmp_path)
    raw = proposal(original.slots(request), brief["scientific_context"])
    result = direction.finish(
        request,
        brief,
        [
            {
                "proposal": raw,
                "assessment": assessment(raw, "insufficient", quoted=False),
            }
        ],
    )
    assert result["status"] == "attempts_exhausted"
    assert result["patch"]["request"] == request.model_dump(mode="json")
    assert result["patch"]["provenance"]["unresolved_slots"]


@pytest.mark.parametrize(
    "role,sign", [("source", "negative"), ("sink", "positive"), ("unknown", "positive")]
)
def test_mechanism_and_sign_cannot_contradict_each_other(tmp_path, role, sign):
    _, _, _, request, _ = setup(tmp_path)
    raw = proposal(original.slots(request))
    raw["decisions"][0].update(mechanism_role=role, outer_weight_sign=sign)
    with pytest.raises(ValueError):
        direction.proposals(request, raw, {raw["decisions"][0]["slot_id"]})


def test_exchange_sign_follows_declared_donor_and_recipient(tmp_path):
    _, _, _, request, _ = setup(tmp_path)
    slots = original.slots(request)
    raw = proposal(slots)
    raw["decisions"][0].update(
        mechanism_role="transfer", donor="u01", recipient=slots[0]["state"]
    )
    with pytest.raises(ValueError, match="flow direction"):
        direction.proposals(request, raw, {slots[0]["slot_id"]})
    raw["decisions"][0]["outer_weight_sign"] = "positive"
    direction.proposals(request, raw, {slots[0]["slot_id"]})
    raw["decisions"][0]["donor"] = "hidden_compartment"
    with pytest.raises(ValueError, match="endpoints"):
        direction.proposals(request, raw, {slots[0]["slot_id"]})


def test_partial_acceptance_preserves_supported_slots_and_retries_only_gap(tmp_path):
    _, _, _, request, brief = setup(tmp_path, two_slots=True)
    slots = original.slots(request)
    raw = proposal(slots)
    checks = assessment(raw)
    checks["assessments"][1]["verdict"] = "contradicted"
    locked, diagnostics = direction.assess_round(request, raw, checks, {})
    assert len(locked) == len(diagnostics) == 1
    with pytest.raises(ValueError, match="locked slots"):
        direction.assess_round(request, raw, checks, locked)
    retry = proposal(slots[1:])
    retry["decisions"][0].update(
        mechanism_role="unknown",
        outer_weight_sign="unrestricted",
        basis="undetermined",
        public_quote="",
    )
    result = direction.finish(
        request,
        brief,
        [
            {"proposal": raw, "assessment": checks},
            {
                "proposal": retry,
                "assessment": assessment(retry, "insufficient", quoted=False),
            },
        ],
    )
    assert result["reply"]["decisions"][0] == raw["decisions"][0]
    assert result["patch"]["provenance"]["unresolved_slots"] == []
    child = PublicFitRequest.model_validate(result["patch"]["request"])
    assert (
        next(p for p in child.base_candidate.parameters if p.name == "c").domain.value
        == "nonnegative"
    )
    assert (
        next(p for p in child.base_candidate.parameters if p.name == "d").domain.value
        == "real"
    )
    assert child.initialization_plan == request.initialization_plan


def test_v2_paired_fit_and_exact_resume(tmp_path, monkeypatch):
    source, config, root, _, _ = setup(tmp_path)
    before = source.read_bytes()
    plan = campaign.freeze(source, config, root)
    review = campaign.review_one(root, 0, base_url="http://mock", transport=transport)
    assert plan["protocol"] == "dalla-sign-repair-2"
    assert review["status"] == "repaired" and review["physical_requests"] == 2
    assert not review["patch"]["provenance"]["citation_audit"][0]["citation_credited"]
    calls = []
    monkeypatch.setattr(public, "_run_backend", backend(calls))
    assert campaign.fit_one(root, 0)["sign_integrity"]["passed"]
    assert len(calls) == 6
    assert source.read_bytes() == before
    assert (
        campaign.review_one(
            root,
            0,
            base_url="http://mock",
            transport=lambda *a: pytest.fail("extra call"),
        )
        == review
    )
    campaign.fit_one(root, 0)
    assert len(calls) == 6
    report = campaign.report(root)
    assert report["protocol"] == "dalla-sign-repair-2"
    assert report["rows"][0]["citation_audit"]
    saved = public._read(root / "models.json")
    assert saved["protocol"] == report["protocol"]
    warm = sealed_read(root / "fit-inputs/toy_seed0_full.json")["rows"][0][
        "warm_start_audit"
    ]
    assert warm["changed_declarations"] == ["c"]


def test_assessor_interruption_and_drain_preserve_call_budget(tmp_path):
    source, config, root, _, _ = setup(tmp_path)
    campaign.freeze(source, config, root)
    calls = []

    def interrupted(url, body, timeout):
        calls.append(body)
        if len(calls) == 2:
            raise RuntimeError("interrupted assessor")
        return transport(url, body, timeout)

    with pytest.raises(RuntimeError):
        campaign.review_one(root, 0, base_url="mock", transport=interrupted)
    with pytest.raises(DeferredCall):
        campaign.review_one(
            root, 0, base_url="mock", transport=transport, can_start=lambda: False
        )
    result = campaign.review_one(root, 0, base_url="mock", transport=transport)
    assert result["physical_requests"] == 4
    assert result["unknown_usage_requests"] == 1
    assert result["status"] == "repaired"


def test_controller_retries_only_unresolved_slot(tmp_path):
    source, config, root, _, _ = setup(tmp_path, two_slots=True)
    campaign.freeze(source, config, root)
    proposals_seen = []
    assessments = []

    def partial(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"])
        response = transport(url, body, timeout)
        if "proposed" in payload:
            checks = assessment(payload["proposed"])
            if not assessments:
                checks["assessments"][1]["verdict"] = "insufficient"
            assessments.append(checks)
            response["choices"][0]["message"]["content"] = json.dumps(checks)
        else:
            proposals_seen.append(payload)
        return response

    result = campaign.review_one(root, 0, base_url="mock", transport=partial)
    assert result["status"] == "repaired" and result["physical_requests"] == 4
    assert [len(p["eligible_slots"]) for p in proposals_seen] == [2, 1]
    assert len(proposals_seen[1]["locked_decisions"]) == 1
    locked = proposals_seen[1]["locked_decisions"][0]
    assert result["reply"]["decisions"][0] == locked
    assert result["patch"]["provenance"]["unresolved_slots"] == []


def test_all_bad_assessments_do_not_commit_or_fit(tmp_path):
    source, config, root, _, _ = setup(tmp_path)
    campaign.freeze(source, config, root)

    def unsupported(url, body, timeout):
        response = transport(url, body, timeout)
        payload = json.loads(body["messages"][1]["content"])
        if "proposed" in payload:
            response["choices"][0]["message"]["content"] = json.dumps(
                assessment(payload["proposed"], "insufficient", False)
            )
        return response

    result = campaign.review_one(root, 0, base_url="mock", transport=unsupported)
    assert result["physical_requests"] == 6
    assert result["status"] == "attempts_exhausted"
    assert campaign.fit_one(root, 0)["status"] == "not_fitted"


def test_packet_excludes_private_and_numerical_data(tmp_path):
    _, _, _, request, brief = setup(tmp_path)
    brief = {**brief, "private_reference": "HIDDEN_781", "training": [981111]}
    text = json.dumps(direction.context(request, brief))
    assert "HIDDEN_781" not in text and "981111" not in text
    assert "parameter_guesses" not in text


def test_selection_is_explicit_and_unknown_ids_fail(tmp_path):
    source, config, root, _, _ = setup(tmp_path)
    packet = public._read(source)
    second = copy.deepcopy(packet["models"][0])
    second["task"]["task_id"] = "other"
    second["result"]["task"]["task_id"] = "other"
    second["result"]["artifact_sha256"] = content_hash(
        {k: v for k, v in second["result"].items() if k != "artifact_sha256"}
    )
    packet["models"].append(second)
    public._write(source, packet)
    assert len(campaign.freeze(source, config, root)["rows"]) == 1
    settings = public._read(config)
    settings["selected_task_ids"] = ["absent"]
    public._write(config, settings)
    with pytest.raises(ValueError, match="present"):
        campaign.freeze(source, config, root)


def test_v2_server_dispatch():
    out = subprocess.check_output(
        [
            "bash",
            "scripts/hpc/run_staged_topology_server.sh",
            "--check-config",
            "configs/dalla_sign_repair_v2.json",
        ],
        text=True,
    )
    assert out.strip() == "dalla_sign_repair.py"


def test_v2_submission_is_bounded_and_resumable(tmp_path, monkeypatch):
    from scripts import submit_dalla_sign_repair as submitter

    source, config, root, _, _ = setup(tmp_path)
    image = tmp_path / "image.sif"
    image.touch()
    for name in ("AF_PYTHON", "AF_VLLM_IMAGE"):
        monkeypatch.setenv(name, str(image))
    monkeypatch.setenv("AF_HF_HOME", str(tmp_path))
    monkeypatch.setenv("AF_COMMIT", "b" * 40)
    monkeypatch.setattr(submitter, "source_commit", lambda _: "b" * 40)
    calls = []

    def submit(directory, key, options, worker, stage, index):
        calls.append((key, options))
        return str(100 + len(calls))

    monkeypatch.setattr(submitter, "submit_job", submit)
    result = submitter.submit(source, root, config=config)
    assert result["protocol"] == "dalla-sign-repair-2"
    assert result["array_tasks"] == 1 and result["maximum_llm_calls"] == 6
    assert result["review_gpus"] == 1 and result["fit_gpus"] == 0
    assert "--array=0-0%3" in dict(calls)["fit"]
    assert submitter.submit(source, root, config=config) == result
    assert len(calls) == 4
