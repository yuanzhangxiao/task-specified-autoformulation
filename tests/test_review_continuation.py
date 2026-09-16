"""Scientific edit safety, provenance-preserving import, and bounded continuation."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from autoformalism.expressions import ModelValidationError
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import review_continuation as continuation
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicFitResult
from autoformalism.search import review_revision_v3 as edits
from tests.test_review_deadline import SMOKE
from tests.test_review_deadline_v2 import example

ROOT = Path(__file__).resolve().parents[1]


def raw_reply():
    return {
        "hypothesis": "Bounded memory may address this mismatch.",
        "evidence_refs": ["E001"],
        "equations": [{"component": "f", "expression": "tanh(m)"}],
    }


def test_short_evidence_and_no_serialization_edit_targets():
    bundle, packet, params, _, _ = example()
    value = edits.payload(bundle, packet, params)
    assert "state_equations" not in value["model"]
    assert value["editable_objects"]["public_output"] == ["v01"]
    refs = set(edits.references(packet))
    assert all(r["evidence_id"] in refs for r in value["training_evidence"]["rows"])
    assert not {"action", "scope", "route", "mappings", "remove"} & set(
        edits.ScientificRevision.model_json_schema()["properties"]
    )
    assert "validation" not in value and "test" not in value


@pytest.mark.parametrize(
    "refs", [[], ["invented_samples"], ["E001", "invented_samples"]]
)
def test_citation_warnings_never_fabricate_evidence_or_block_valid_equations(refs):
    bundle, packet, _, _, _ = example()
    result = edits.apply_edits(bundle, packet, {**raw_reply(), "evidence_refs": refs})
    assert result["outcome"] == "committed"
    provenance = result["bundle"]["revision_provenance"]
    audit = provenance["citation_audit"]
    assert audit["status"] == "warning"
    expected = [edits.references(packet)["E001"]] if "E001" in refs else []
    assert audit["valid_evidence_ids"] == provenance["evidence_ids"] == expected
    assert audit["scientific_claims_verified"] is False


@pytest.mark.parametrize(
    "expression", ["missing_symbol+m", "__import__('os')", "v01+m"]
)
def test_bad_citations_do_not_bypass_equation_validation(expression):
    bundle, packet, _, _, _ = example()
    reply = {
        **raw_reply(),
        "evidence_refs": ["invented_samples"],
        "equations": [{"component": "f", "expression": expression}],
    }
    with pytest.raises((ValueError, ModelValidationError)):
        edits.apply_edits(bundle, packet, reply)


def test_identical_duplicates_and_remove_replace_are_audited():
    bundle, packet, _, _, _ = example()
    reply = raw_reply()
    reply["equations"] *= 2
    reply["remove_variables"] = ["f"]
    result = edits.apply_edits(bundle, packet, reply)
    assert result["outcome"] == "committed"
    normal = result["provenance"]["normalized_redundancies"]
    assert normal["identical_equations"] == ["f"]
    assert normal["replacement_also_removed"] == ["f"]
    reply["equations"] = [reply["equations"][0], {"component": "f", "expression": "m"}]
    with pytest.raises(ValueError, match="Conflicting"):
        edits.apply_edits(bundle, packet, reply)


def test_parameter_cleanup_only_when_unused():
    bundle, packet, _, _, _ = example()
    with pytest.raises(ValueError, match="still appears"):
        edits.apply_edits(bundle, packet, {**raw_reply(), "remove_variables": ["rate"]})
    result = edits.apply_edits(
        bundle,
        packet,
        {
            **raw_reply(),
            "equations": [{"component": "m", "expression": "-m+gain*u01"}],
            "remove_variables": ["rate"],
        },
    )
    assert "rate" not in {
        p["name"] for p in result["bundle"]["candidate"]["parameters"]
    }


@pytest.mark.parametrize("field", ["state_equations", "initial_conditions", "unknown"])
def test_json_field_removals_get_actionable_error(field):
    bundle, packet, _, _, _ = example()
    with pytest.raises(ValueError, match="Allowed variables"):
        edits.apply_edits(bundle, packet, {**raw_reply(), "remove_variables": [field]})


def test_observed_initializer_cannot_be_fitted():
    bundle, packet, _, _, _ = example()
    with pytest.raises(ValueError, match="latent dynamic state"):
        edits.apply_edits(
            bundle,
            packet,
            {**raw_reply(), "initializers": [{"state": "v01", "causal_map": None}]},
        )


def source_fixture(root):
    old = SMOKE.fixture(root)
    config = io.DeadlineConfig.model_validate(
        {**old["config"], "protocol": io.CONTENT_PROTOCOL, "rounds": 3}
    )
    (root / "plan.json").unlink()
    plan = sealed_write(
        root / "plan.json",
        {
            **{k: v for k, v in old.items() if k != "artifact_sha256"},
            "config": config.model_dump(mode="json"),
            "protocol": io.CONTENT_PROTOCOL,
            "launcher_sha256": io.launcher_hash(io.CONTENT_PROTOCOL),
        },
    )
    bundle, packet, params, _, _ = example()
    for i, task in enumerate(plan["tasks"]):
        request = pipeline.request_for(bundle, plan, task, 0)
        fit = PublicFitResult(
            status="complete",
            profile=request.profile,
            identity="0" * 64,
            request_sha256="1" * 64,
            lowered_candidate_sha256="2" * 64,
            initialization_plan_sha256="3" * 64,
            training_content_sha256="4" * 64,
            validation_content_sha256="5" * 64,
            source=request.source,
            parameters=params,
            training={"available": True, "normalized_mse": 1.0},
            validation={"available": True, "normalized_mse": 1.1},
            message="fixture",
        ).model_dump(mode="json")
        selected = (
            None
            if i == 1
            else {
                "bundle": bundle,
                "request": request.model_dump(mode="json"),
                "fit": fit,
                "packet": packet,
                "origin_round": 0,
                "certificate": {
                    "all_public_graph_requirements_certified": True,
                    "eligible_for_development_selection": True,
                    "latent_states": ["m"],
                },
            }
        )
        sealed_write(
            io.round_path(root, task, 2) / "result.json",
            {
                "task": task,
                "round": 2,
                "status": "closed_lineage",
                "selected": selected,
                "closed": True,
                "test_data_opened": False,
            },
        )
    return plan


def test_import_reopens_valid_incumbents_without_new_fit_or_source_changes(tmp_path):
    source, root = tmp_path / "source", tmp_path / "continued"
    old = source_fixture(source)
    before = {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*.json")
    }
    plan = continuation.prepare(source, root, 2, 2)
    assert plan["config"]["rounds"] == 3
    assert plan["config"]["fit_profile"] == old["config"]["fit_profile"]
    assert plan["tasks"] == old["tasks"]
    for task in plan["tasks"]:
        original, anchor = io.read_round(source, task, 2), io.read_round(root, task, 0)
        assert anchor["selected"] == original["selected"]
        assert anchor["closed"] == (anchor["selected"] is None)
    assert not list(root.rglob("started.json"))
    assert continuation.prepare(source, root, 2, 2) == plan
    assert {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*.json")
    } == before
    summary = reporting.report(root)
    assert summary["status_counts"]["imported_checkpoint"] == len(plan["tasks"])
    assert summary["rows"][0]["round"] == 2
    assert summary["rows"][1]["round"] == 3
    assert summary["rows"][0]["cumulative_requests"] == 0
    with pytest.raises(ValueError, match="budget differs"):
        continuation.prepare(source, root, 2, 3)


def test_sealed_or_incomplete_source_is_not_reopened(tmp_path):
    source = tmp_path / "source"
    plan = source_fixture(source)
    last = io.round_path(source, plan["tasks"][-1], 2) / "result.json"
    last.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        continuation.prepare(source, tmp_path / "incomplete")
    (source / "evaluation_freeze.json").write_text("{}")
    with pytest.raises(ValueError, match="sealed"):
        continuation.prepare(source, tmp_path / "sealed")


def test_anchor_tampering_rejected_even_when_resealed(tmp_path):
    source, root = tmp_path / "source", tmp_path / "continued"
    source_fixture(source)
    plan = continuation.prepare(source, root)
    path = io.round_path(root, plan["tasks"][0], 0) / "result.json"
    anchor = sealed_read(path)
    anchor.pop("artifact_sha256")
    anchor["selected"]["fit"]["parameters"]["rate"] = 99
    path.unlink()
    sealed_write(path, anchor)
    with pytest.raises(ValueError, match="differs"):
        io.verify(root)


@pytest.mark.parametrize(
    "proposal_status", ["revision_failed", "no_change", "residual_evidence_unavailable"]
)
def test_failed_revision_spends_one_existing_fit_budget_and_keeps_branch(
    tmp_path, monkeypatch, proposal_status
):
    source, root = tmp_path / "source", tmp_path / "continued"
    source_fixture(source)
    plan = continuation.prepare(source, root, visits=2)
    task = plan["tasks"][0]
    parent = io.read_round(root, task, 0)
    directory = io.round_path(root, task, 1)
    sealed_write(
        directory / "proposal.json",
        {
            "parent_sha256": parent["artifact_sha256"],
            "status": proposal_status,
            "bundle": None,
            "cost": {"physical_requests": 0},
        },
    )
    calls = []
    fit = PublicFitResult.model_validate(parent["selected"]["fit"])
    monkeypatch.setattr(
        pipeline, "certificates", lambda *a: parent["selected"]["certificate"]
    )
    monkeypatch.setattr(
        pipeline.sibling_fit,
        "prepare_child_fit",
        lambda *a, **kw: calls.append((a, kw)),
    )

    def execute(path):
        path.mkdir(parents=True, exist_ok=True)
        public._write(path / "result.json", fit.model_dump(mode="json"))
        return fit

    monkeypatch.setattr(pipeline.sibling_fit, "execute_child_fit", execute)
    monkeypatch.setattr(
        pipeline, "replay_packet", lambda *a: parent["selected"]["packet"]
    )
    result = pipeline.fit_one(root, plan, task, 1)
    assert len(calls) == 1 and calls[0][0][2] == parent["selected"]["fit"]["parameters"]
    assert not result["closed"] and result["proposal_status"] == proposal_status
    assert result["fit_trigger"] == "incumbent_fallback"
    assert pipeline.fit_one(root, plan, task, 1) == result and len(calls) == 1


def _submit_module():
    spec = importlib.util.spec_from_file_location(
        "continuation_submit", ROOT / "scripts/submit_review_continuation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_site_wrapper_receipt_and_no_wrap(tmp_path, monkeypatch):
    submitter = _submit_module()
    seen = []

    def run(argv, **kw):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, "12345\n", "")

    monkeypatch.setattr(submitter.subprocess, "run", run)
    job = submitter.submit_job(
        tmp_path, "fit-1", ["--array=0-43%16"], Path("worker.sh"), "fit", 1
    )
    assert job == "12345" and seen[0][-3:] == ["worker.sh", "fit", "1"]
    assert "--wrap" not in seen[0]
    assert (tmp_path / "fit-1.reply.json").exists()


@pytest.mark.parametrize("code,stdout", [(0, ""), (1, ""), (0, "unexpected response")])
def test_uncertain_submission_retains_receipt_without_job_id(
    tmp_path, monkeypatch, code, stdout
):
    submitter = _submit_module()
    monkeypatch.setattr(
        submitter.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], code, stdout, "detail"),
    )
    with pytest.raises(ValueError, match="unconfirmed"):
        submitter.submit_job(tmp_path, "fit-1", [], Path("worker.sh"), "fit", 1)
    assert json.loads((tmp_path / "fit-1.reply.json").read_text())["stderr"] == "detail"
    assert not (tmp_path / "fit-1.id").exists()


def test_v3_server_dispatch_and_shell_syntax(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"protocol": io.CONTINUATION_PROTOCOL, "platform": "aces-h100x1"})
    )
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/hpc/run_staged_topology_server.sh"),
            "--check-config",
            str(config),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "review_deadline.py"
    subprocess.run(
        ["bash", "-n", str(ROOT / "scripts/hpc/run_review_continuation_aces.sh")],
        check=True,
    )


def test_bounded_scheduler_chain_resume_and_partial_submission(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "continued"
    source_fixture(source)
    plan = continuation.prepare(source, root, visits=2)
    module = _submit_module()
    for name in ("AF_PYTHON", "AF_VLLM_IMAGE"):
        file = tmp_path / name
        file.write_text("fixture")
        monkeypatch.setenv(name, str(file))
    for name in ("AF_HF_HOME", "AF_COMPUTE_CACHE_ROOT", "AF_IPC_TMP_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / name))
    monkeypatch.delenv("AF_COMMIT", raising=False)
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda argv, **kw: "" if "status" in argv else "a" * 40,
    )
    jobs = []

    def sbatch(argv, **kw):
        jobs.append(argv)
        return subprocess.CompletedProcess(argv, 0, str(100 + len(jobs)), "")

    monkeypatch.setattr(module.subprocess, "run", sbatch)
    first = module.submit(root, 1)
    assert len(jobs) == 5 and not first["all_rounds_submitted"]
    assert first["global_round"] == 3
    assert module.submit(root, 1) == first and len(jobs) == 5
    assert [a[-2:] for a in jobs] == [
        ["prepare", "0"],
        ["propose", "1"],
        ["fit", "1"],
        ["finish", "1"],
        ["dispatch", "2"],
    ]
    with pytest.raises(ValueError, match="previous visit incomplete"):
        module.submit(root, 2)
    assert not (root / "submission-intent/round-2").exists()
    for task in plan["tasks"]:
        previous = io.read_round(root, task, 0)
        sealed_write(
            io.round_path(root, task, 1) / "result.json",
            {
                **{k: v for k, v in previous.items() if k != "artifact_sha256"},
                "round": 1,
            },
        )
    second = module.submit(root, 2)
    assert second["all_rounds_submitted"] and len(jobs) == 8
    assert jobs[-3][-2:] == ["propose", "2"]
    assert not any(a.startswith("--dependency") for a in jobs[-3])
    with pytest.raises(ValueError, match="outside"):
        module.submit(root, 3)

    # Another output with an uncertain reply cannot blindly resubmit.
    other = tmp_path / "other"
    continuation.prepare(source, other, visits=1)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(
            argv, 0, "", "wrapper rejection"
        ),
    )
    with pytest.raises(ValueError, match="unconfirmed"):
        module.submit(other, 1)
    with pytest.raises(ValueError, match="intent exists"):
        module.submit(other, 1)


def test_saved_response_audit_does_not_guess_mapping_or_equations(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "continuation_audit", ROOT / "scripts/audit_review_continuation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="nonpublic channel"):
        module.convert(
            {"hypothesis": "x", "mappings": [{"channel": "mem", "expression": "m"}]},
            "v01",
        )
    source, root = tmp_path / "source", tmp_path / "continued"
    original = source_fixture(source)
    for task in original["tasks"]:
        source_result = io.read_round(source, task, 2)
        parent = sealed_write(
            io.round_path(source, task, 0) / "result.json",
            {
                **{k: v for k, v in source_result.items() if k != "artifact_sha256"},
                "round": 0,
            },
        )
        if parent["selected"] is None:
            continue
        sealed_write(
            io.round_path(source, task, 1) / "proposal.json",
            {
                "task": task,
                "round": 1,
                "parent_sha256": parent["artifact_sha256"],
                "attempts": [
                    {
                        "accepted": False,
                        "raw": {
                            "hypothesis": "Change shape",
                            "evidence_ids": ["invented_samples"],
                            "equations": [
                                {"component": "f", "expression": "missing_symbol"}
                            ],
                        },
                    }
                ],
            },
        )
    continuation.prepare(source, root)
    result = module.audit(root)
    assert result["status_counts"] == {"still_blocked": 2}
    assert result["llm_calls"] == result["fitting_calls"] == 0
    assert module.audit(root) == result
