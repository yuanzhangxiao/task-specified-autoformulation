"""Public-only proposals, sealed resume, paired allocations and scheduler receipts."""

import json
import subprocess

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import dalla_sign_repair as campaign
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.staged_topology import content_hash
from scripts import submit_dalla_sign_repair as submitter
from scripts.smoke_dalla_sign_repair import fixture, transport
from tests.test_process_pruning_campaign import backend


def test_paired_fit_and_exact_resume(tmp_path, monkeypatch):
    source, config, root = fixture(tmp_path)
    plan = campaign.freeze(source, config, root)
    before = source.read_bytes()
    assert campaign.freeze(source, config, root) == plan
    review = campaign.review_one(root, 0, base_url="http://mock", transport=transport)
    assert review["status"] == "repaired" and review["physical_requests"] == 1
    calls = []
    monkeypatch.setattr(public, "_run_backend", backend(calls))
    result = campaign.fit_one(root, 0)
    assert result["status"] == "complete" and result["sign_integrity"]["passed"]
    assert len(calls) == 6
    assert all(c["settings"] == calls[0]["settings"] for c in calls)
    warm = sealed_read(root / "fit-inputs/toy_seed0_full.json")["rows"][0][
        "warm_start_audit"
    ]
    assert warm["changed_declarations"] == ["c"] and warm["fresh_parameters"] == ["c"]
    assert warm["initialization_plan_unchanged"]
    assert warm["parameters"]["c"] >= 0
    snapshot = {str(p): p.read_bytes() for p in root.rglob("*.json")}
    assert (
        campaign.review_one(
            root,
            0,
            base_url="http://mock",
            transport=lambda *a: pytest.fail("extra request"),
        )
        == review
    )
    campaign.fit_one(root, 0)
    assert len(calls) == 6
    assert snapshot == {str(p): p.read_bytes() for p in root.rglob("*.json")}
    assert source.read_bytes() == before
    assert campaign.report(root)["rows"][0]["status"] == "complete"


def test_interrupted_request_consumes_attempt_and_is_not_resent(tmp_path):
    source, config, root = fixture(tmp_path)
    campaign.freeze(source, config, root)
    with pytest.raises(RuntimeError, match="interrupted"):
        campaign.review_one(
            root,
            0,
            base_url="http://mock",
            transport=lambda *a: (_ for _ in ()).throw(RuntimeError("interrupted")),
        )
    calls = []

    def resumed(*args):
        calls.append(args)
        return transport(*args)

    review = campaign.review_one(root, 0, base_url="http://mock", transport=resumed)
    assert len(calls) == 1 and review["physical_requests"] == 2
    assert review["unknown_usage_requests"] == 1
    assert review["attempts"][0]["accepted"] is False
    assert review["attempts"][1]["accepted"] is True


def test_repair_diagnostics_and_budget_failure_are_terminal(tmp_path):
    source, config, root = fixture(tmp_path)
    campaign.freeze(source, config, root)
    requests = []

    def bad(url, body, timeout):
        requests.append(json.loads(body["messages"][1]["content"]))
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": '{"decisions":[]}'}}
            ]
        }

    review = campaign.review_one(root, 0, base_url="http://mock", transport=bad)
    assert review["status"] == "attempts_exhausted" and len(requests) == 3
    assert requests[0]["repair_diagnostic"] is None and requests[1]["repair_diagnostic"]
    assert all(
        "original_fit" not in r and "parameters" not in r and "training" not in r
        for r in requests
    )
    assert campaign.fit_one(root, 0)["status"] == "not_fitted"
    assert campaign.report(root)["rows"][0]["status"] == "attempts_exhausted"


def test_unrestricted_is_explicit_and_not_falsely_reported_repaired(tmp_path):
    source, config, root = fixture(tmp_path)
    campaign.freeze(source, config, root)

    def uncertain(*args):
        response = transport(*args)
        reply = json.loads(response["choices"][0]["message"]["content"])
        reply["decisions"][0].update(
            outer_weight_sign="unrestricted",
            basis="undetermined",
            public_quote="",
            rationale="The symbolic state interpretation remains uncertain.",
        )
        response["choices"][0]["message"]["content"] = json.dumps(reply)
        return response

    assert (
        campaign.review_one(root, 0, base_url="http://mock", transport=uncertain)[
            "status"
        ]
        == "unchanged"
    )
    assert campaign.fit_one(root, 0)["reason"] == "unchanged"


def test_no_eligible_gains_skip_without_calls(tmp_path):
    source, config, root = fixture(tmp_path)
    packet = public._read(source)
    result = packet["models"][0]["result"]
    parameter = next(
        p
        for p in result["selected_request"]["base_candidate"]["parameters"]
        if p["name"] == "c"
    )
    parameter.update(role="nonnegative_coefficient", domain="nonnegative")
    result["selected_request"]["parameter_guesses"]["c"] = 0.1
    result["artifact_sha256"] = content_hash(
        {k: v for k, v in result.items() if k != "artifact_sha256"}
    )
    public._write(source, packet)
    campaign.freeze(source, config, root)
    review = campaign.review_one(
        root, 0, base_url="http://mock", transport=lambda *a: pytest.fail("extra call")
    )
    assert review["status"] == "no_eligible_gains" and review["physical_requests"] == 0


def test_input_and_result_tampering_fail_closed(tmp_path):
    source, config, root = fixture(tmp_path)
    campaign.freeze(source, config, root)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="identity"):
        campaign.verify(root)


def test_import_uses_rescue_seal_without_rewriting_source(tmp_path):
    source, config, root = fixture(tmp_path)
    result = sealed_read(tmp_path / "source-result.json")
    payload = {k: v for k, v in result.items() if k != "artifact_sha256"}
    assert result["artifact_sha256"] != public.content_sha256(payload)
    before = source.read_bytes()
    plan = campaign.freeze(source, config, root)
    assert plan["rows"][0]["source_result_sha256"] == result["artifact_sha256"]
    assert source.read_bytes() == before
    assert campaign.freeze(source, config, root) == plan


@pytest.mark.parametrize("alteration", ["parameter", "digest", "compact_digest"])
def test_import_rejects_changed_payload_or_wrong_seal(tmp_path, alteration):
    source, config, root = fixture(tmp_path)
    packet = public._read(source)
    result = packet["models"][0]["result"]
    if alteration == "parameter":
        result["selected_fit"]["parameters"]["c"] = 1.0
    elif alteration == "digest":
        result["artifact_sha256"] = "0" * 64
    else:
        result["artifact_sha256"] = public.content_sha256(
            {k: v for k, v in result.items() if k != "artifact_sha256"}
        )
    public._write(source, packet)
    with pytest.raises(
        ValueError, match="source result digest differs: toy_seed0_full"
    ):
        campaign.freeze(source, config, root)
    assert not (root / "plan.json").exists()


def test_nonnegative_integrity_rejects_negative_fit(tmp_path):
    source, config, root = fixture(tmp_path)
    plan = campaign.freeze(source, config, root)
    review = campaign.review_one(root, 0, base_url="http://mock", transport=transport)
    fit = {"parameters": {**plan["rows"][0]["parameters"], "c": -1}}
    with pytest.raises(ValueError, match="violated"):
        campaign.sign_integrity(review, review["patch"]["request"], fit)


@pytest.mark.parametrize("altered_term", ["c*x", "-c*x**2"])
def test_integrity_rejects_changed_outer_operator_or_inner_law(tmp_path, altered_term):
    source, config, root = fixture(tmp_path)
    plan = campaign.freeze(source, config, root)
    review = campaign.review_one(root, 0, base_url="http://mock", transport=transport)
    child = json.loads(json.dumps(review["patch"]["request"]))
    child["base_candidate"]["state_equations"][0]["rhs"] = "u01-a*x+q+r+" + altered_term
    fit = {"parameters": {**plan["rows"][0]["parameters"], "c": 0.1}}
    with pytest.raises(ValueError, match="protected inner law changed"):
        campaign.sign_integrity(review, child, fit)


def test_test_or_intervention_packet_refused(tmp_path):
    source, config, root = fixture(tmp_path)
    packet = public._read(source)
    packet["interventions_evaluated"] = True
    public._write(source, packet)
    with pytest.raises(ValueError, match="public-only"):
        campaign.freeze(source, config, root)


def test_server_dispatch_accepts_pinned_config():
    value = subprocess.check_output(
        [
            "bash",
            "scripts/hpc/run_staged_topology_server.sh",
            "--check-config",
            "configs/dalla_sign_repair_v1.json",
        ],
        text=True,
    )
    assert value.strip() == "dalla_sign_repair.py"


def test_partial_gpu_review_does_not_release_cpu_dependency(monkeypatch, capsys):
    from scripts import dalla_sign_repair as cli

    monkeypatch.setattr(
        "sys.argv", ["sign-repair", "run", "--root", "/tmp/mock", "--base-url", "mock"]
    )
    monkeypatch.setattr(
        campaign, "run_reviews", lambda *a: {"rows": [{"review_status": None}]}
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 3
    assert json.loads(capsys.readouterr().out)["rows"][0]["review_status"] is None


def test_scheduler_resume_and_verified_adoption(tmp_path, monkeypatch):
    source, config, root = fixture(tmp_path)
    python, image, cache = (
        tmp_path / "python",
        tmp_path / "image.sif",
        tmp_path / "cache",
    )
    python.touch()
    image.touch()
    cache.mkdir()
    monkeypatch.setenv("AF_PYTHON", str(python))
    monkeypatch.setenv("AF_VLLM_IMAGE", str(image))
    monkeypatch.setenv("AF_HF_HOME", str(cache))
    monkeypatch.setenv("AF_COMMIT", "a" * 40)
    monkeypatch.setattr(submitter, "source_commit", lambda _: "a" * 40)
    calls = []

    def submit(directory, key, options, worker, stage, index):
        public._write(
            directory / f"{key}.intent.json",
            {
                "argv": [
                    "sbatch",
                    "--parsable",
                    *options,
                    str(worker),
                    stage,
                    str(index),
                ]
            },
        )
        calls.append((key, options))
        if key == "review":
            raise ValueError("uncertain")
        (directory / f"{key}.id").write_text(str(100 + len(calls)))
        return str(100 + len(calls))

    monkeypatch.setattr(submitter, "submit_job", submit)
    with pytest.raises(ValueError, match="uncertain"):
        submitter.submit(source, root, config=config)
    with pytest.raises(ValueError, match="Unconfirmed review"):
        submitter.submit(source, root, config=config)
    assert len(calls) == 2
    monkeypatch.setattr(
        submitter, "scheduler_record", lambda *a, **k: {"verified": True}
    )
    result = submitter.submit(source, root, adopt={"review": "102"}, config=config)
    assert result["jobs"]["review"] == "102" and len(calls) == 4
    assert "--gres=gpu:h100:1" in calls[1][1]
    assert "--dependency=afterok:102" in calls[2][1]
    assert result == submitter.submit(source, root, config=config)
    assert len(calls) == 4
