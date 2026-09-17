"""Global declarations, immutable inherited roles, and chained continuation."""

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from autoformalism.expressions import ModelValidationError
from autoformalism.rebuttal import review_continuation as continuation
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.rebuttal.revision_decision import expressions, parameter_aliases
from autoformalism.schemas import CandidateModel
from autoformalism.search import review_revision_v3 as old
from autoformalism.search import review_revision_v4 as edits
from tests.test_review_continuation import _submit_module, source_fixture
from tests.test_review_deadline_v2 import example


def reply(expression="gain*tanh(m)"):
    return {
        "hypothesis": "Revise response shape",
        "equations": [{"component": "f", "expression": expression}],
    }


def test_equations_do_not_redeclare_parameters():
    schema = edits.ScientificRevision.model_json_schema()
    assert "parameters" not in schema["$defs"]["EquationContent"]["properties"]
    assert "new_parameters" in schema["properties"]
    assert "new_parameters" in edits.SYSTEM_PROMPT


def test_inherited_conflicting_redeclarations_preserve_parent_roles():
    bundle, packet, *_ = example()
    parent = CandidateModel.model_validate(bundle["initialization"]["base_candidate"])
    alias = parameter_aliases(parent)["gain"]
    raw = reply(alias + "*tanh(m)")
    raw["new_parameters"] = [
        {"name": alias, "role": "shape"},
        {"name": alias, "role": "rate"},
    ]
    before = copy.deepcopy(bundle)
    result = edits.apply_edits(bundle, packet, raw)
    gain = next(
        p
        for p in result["bundle"]["initialization"]["base_candidate"]["parameters"]
        if p["name"] == "gain"
    )
    assert gain == next(
        p.model_dump(mode="json") for p in parent.parameters if p.name == "gain"
    )
    assert (
        len(
            result["provenance"]["parameter_declaration_audit"][
                "inherited_declarations_ignored"
            ]
        )
        == 2
    )
    assert bundle == before


def test_one_shared_new_parameter_compiles_across_equations_and_mapping():
    bundle, packet, *_ = example()
    raw = {
        **reply("amp*tanh(m)"),
        "new_parameters": [{"name": "amp"}],
        "output_expression": "amp*f",
        "initializers": [{"state": "y", "causal_map": None}],
        "equations": [
            {"component": "f", "expression": "amp*tanh(m)"},
            {"component": "m", "expression": "-rate*m+amp*u01"},
        ],
    }
    result = edits.apply_edits(bundle, packet, raw)
    params = result["bundle"]["initialization"]["base_candidate"]["parameters"]
    amp = [p for p in params if p["name"] == "amp"]
    assert len(amp) == 1 and amp[0]["role"] == "nonnegative_coefficient"
    assert (
        result["bundle"]["revision_provenance"]["protocol"]
        == "scientific-content-revision-4"
    )


@pytest.mark.parametrize("expression", ["-a*m", "(-a)*m", "a*(-m)", "-a*(m-gain)"])
def test_new_signed_gains_match_legacy_without_changing_the_equation(expression):
    bundle, packet, *_ = example()
    raw = reply(expression)
    raw["equations"][0]["parameters"] = [{"name": "a", "role": "coefficient"}]
    legacy = old.apply_edits(bundle, packet, raw)
    current = edits.apply_edits(bundle, packet, edits.migrate_saved(raw))
    for result in (legacy, current):
        model = CandidateModel.model_validate(
            result["bundle"]["initialization"]["base_candidate"]
        )
        assert expressions(model)["f"] == expression
        assert next(p for p in model.parameters if p.name == "a").role.value == (
            "nonnegative_coefficient"
        )


@pytest.mark.parametrize(
    "raw,code",
    [
        (
            {**reply("tanh(shape*m)"), "new_parameters": [{"name": "shape"}]},
            "AMBIGUOUS_INTERNAL_PARAMETER_ROLE",
        ),
        (
            {
                **reply("tanh(a*m)"),
                "new_parameters": [
                    {"name": "a", "role": "shape"},
                    {"name": "a", "role": "rate"},
                ],
            },
            "NEW_PARAMETER_DECLARATION_CONFLICT",
        ),
        ({**reply(), "new_parameters": [{"name": "unused"}]}, "UNUSED_NEW_PARAMETER"),
        (
            {
                **reply("a*tanh(m)"),
                "output_expression": "f+a",
                "new_parameters": [{"name": "a"}],
            },
            "NEW_PARAMETER_USE_CONFLICT",
        ),
    ],
)
def test_real_new_parameter_ambiguities_are_actionable(raw, code):
    bundle, packet, values, *_ = example()
    with pytest.raises(ValueError) as caught:
        edits.apply_edits(bundle, packet, raw)
    assert caught.value.code == code
    feedback = edits.feedback(bundle, packet, values, raw, caught.value)
    assert feedback["details"]["parameter"]
    assert feedback["parameter_contract"]["internal_roles"]


def test_signed_shape_and_positive_shared_shape_keep_consistent_domains():
    bundle, packet, *_ = example()
    raw = {**reply("tanh(a*m)"), "new_parameters": [{"name": "a", "role": "shape"}]}
    result = edits.apply_edits(bundle, packet, raw)
    spec = result["provenance"]["parameter_declaration_audit"]["resolved_parameters"][0]
    assert spec["role"] == "shape"
    raw.update(
        output_expression="a*f",
        initializers=[{"state": "y", "causal_map": None}],
        new_parameters=[{"name": "a", "role": "positive_shape"}],
    )
    result = edits.apply_edits(bundle, packet, raw)
    assert (
        result["provenance"]["parameter_declaration_audit"]["resolved_parameters"][0][
            "role"
        ]
        == "positive_shape"
    )


@pytest.mark.parametrize("expression", ["unknown+m", "__import__('os')", "v01+m"])
def test_global_parameter_contract_does_not_waive_model_validity(expression):
    bundle, packet, *_ = example()
    with pytest.raises((ValueError, ModelValidationError)):
        edits.apply_edits(bundle, packet, reply(expression))


def test_saved_declared_roles_migrate_without_silent_conflict_resolution():
    bundle, packet, *_ = example()
    raw = {
        **reply("newshape*tanh(m)"),
        "equations": [
            {
                "component": "f",
                "expression": "newshape*tanh(m)",
                "parameters": [{"name": "newshape", "role": "shape"}],
            },
            {
                "component": "m",
                "expression": "-rate*m+newshape*u01",
                "parameters": [{"name": "newshape", "role": "rate"}],
            },
        ],
    }
    with pytest.raises(ValueError, match="conflicting"):
        old.apply_edits(bundle, packet, raw)
    with pytest.raises(ValueError, match="one role"):
        edits.apply_edits(bundle, packet, edits.migrate_saved(raw))


def v3_fixture(path):
    source_fixture(path / "v2")
    root = path / "v3"
    plan = continuation.prepare(path / "v2", root, 2, 5)
    for task in plan["tasks"]:
        anchor = io.read_round(root, task, 0)
        if anchor["selected"]:
            anchor["selected"]["certificate"] = pipeline.certificates(
                anchor["selected"]["bundle"], plan["cells"][task["cell"]], task
            )
        for index in range(1, 6):
            sealed_write(
                io.round_path(root, task, index) / "result.json",
                {
                    **{k: v for k, v in anchor.items() if k != "artifact_sha256"},
                    "round": index,
                    "status": "complete",
                },
            )
    return root, plan


def test_chained_import_global_rounds_preserve_parameters_costs_and_resume(tmp_path):
    source, old_plan = v3_fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in source.rglob("*.json")}
    root = tmp_path / "v4"
    plan = continuation.prepare(source, root, 7, 5, protocol=io.PARAMETER_PROTOCOL)
    assert plan["continuation"]["source_phase_round"] == 5
    assert plan["continuation"]["source_round"] == 7
    assert plan["config"]["model_settings"] == old_plan["config"]["model_settings"]
    for task in plan["tasks"]:
        assert (
            io.read_round(root, task, 0)["selected"]
            == io.read_round(source, task, 5)["selected"]
        )
    assert (
        continuation.prepare(source, root, 7, 5, protocol=io.PARAMETER_PROTOCOL) == plan
    )
    summary = reporting.report(root)
    assert [r["round"] for r in summary["rows"][:6]] == list(range(7, 13))
    assert "Prediction-only (internal ID no_spec)" in (root / "SUMMARY.md").read_text()
    assert summary["rows"][0]["cumulative_fit_attempts"] == 0
    assert summary["rows"][0]["cumulative_requests"] == 0
    assert {str(p): p.read_bytes() for p in source.rglob("*.json")} == before
    with pytest.raises(ValueError, match="outside"):
        continuation.prepare(
            source, tmp_path / "bad", 8, protocol=io.PARAMETER_PROTOCOL
        )
    with pytest.raises(ValueError, match="differs"):
        continuation.prepare(source, root, 7, 4, protocol=io.PARAMETER_PROTOCOL)
    (source / "evaluation_freeze.json").write_text("{}")
    with pytest.raises(ValueError, match="sealed"):
        continuation.prepare(
            source, tmp_path / "sealed", 7, protocol=io.PARAMETER_PROTOCOL
        )


def test_v4_scheduler_uses_bounded_visits_and_real_scripts(tmp_path, monkeypatch):
    source, _ = v3_fixture(tmp_path)
    root = tmp_path / "v4"
    continuation.prepare(source, root, 7, 1, protocol=io.PARAMETER_PROTOCOL)
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

    def submit(argv, **kw):
        jobs.append(argv)
        return subprocess.CompletedProcess(argv, 0, str(100 + len(jobs)), "")

    monkeypatch.setattr(module.subprocess, "run", submit)
    result = module.submit(root, 1)
    assert result["global_round"] == 8 and result["all_rounds_submitted"]
    assert len(jobs) == 4 and all("--wrap" not in a for j in jobs for a in j)
    assert "--job-name=review-v4-propose-1" in jobs[1]
    assert module.submit(root, 1) == result and len(jobs) == 4


def test_saved_audit_exposes_inherited_conflicts_and_balance_review(tmp_path):
    source, original = v3_fixture(tmp_path)
    task = original["tasks"][0]
    parent = io.read_round(source, task, 0)
    raw = reply("gain*tanh(m)")
    raw["equations"][0]["parameters"] = [{"name": "gain", "role": "shape"}]
    raw["equations"].append(
        {
            "component": "m",
            "expression": "-rate*m+gain*u01",
            "parameters": [{"name": "gain", "role": "rate"}],
        }
    )
    sealed_write(
        io.round_path(source, task, 1) / "proposal.json",
        {
            "parent_sha256": parent["artifact_sha256"],
            "status": "revision_failed",
            "attempts": [
                {
                    "accepted": False,
                    "raw": raw,
                    "feedback": {"code": "SCIENTIFIC_CONTENT_CONTRACT"},
                }
            ],
        },
    )
    path = io.round_path(source, task, 5) / "result.json"
    selected = io.read_round(source, task, 5)
    selected.pop("artifact_sha256")
    selected["selected"]["certificate"]["mechanisms"]["predicates"] = [
        {
            "mechanism_id": "controlled_balance",
            "status": "ambiguous",
            "predicate": "graph_inference",
            "evidence": "no public driver",
        }
    ]
    selected["selected"]["certificate"]["all_public_graph_requirements_certified"] = (
        False
    )
    path.unlink()
    sealed_write(path, selected)
    root = tmp_path / "v4"
    continuation.prepare(source, root, 7, 1, protocol=io.PARAMETER_PROTOCOL)
    spec = importlib.util.spec_from_file_location(
        "parameter_audit",
        Path(__file__).resolve().parents[1] / "scripts/audit_review_parameters.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.audit(root)
    assert result["rejected_then_valid"] == 1
    assert all(p["inherited"] for p in result["records"][0]["declarations"])
    assert result["fitting_calls"] == result["llm_calls"] == 0
    assert module.audit(root) == result
    review = json.loads((root / "requirement_review.json").read_text())
    assert (
        review["cases"][0]["verdict"]
        == "manual_review_required_no_automatic_certification"
    )
    summary = reporting.report(root)
    assert summary["rows"][0]["graph_check_status"] == "unresolved"
    assert (
        io.read_round(root, task, 0)["selected"]["certificate"][
            "all_public_graph_requirements_certified"
        ]
        is False
    )
