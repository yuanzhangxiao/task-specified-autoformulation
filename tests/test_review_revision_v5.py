"""Advisory complexity cannot bypass science, initialization, or control checks."""

import copy
import importlib.util
from pathlib import Path

import pytest

from autoformalism.expressions import ModelValidationError
from autoformalism.rebuttal import review_continuation as continuation
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.search import review_revision_v4 as old
from autoformalism.search import review_revision_v5 as edits
from tests.test_review_deadline_v2 import example
from tests.test_review_parameters import reply, v3_fixture


@pytest.mark.parametrize(
    "quantity,limit",
    [("generated_variables", 2), ("terms_per_equation", 1), ("total_terms", 3)],
)
def test_only_v5_accepts_models_over_construction_references(quantity, limit):
    bundle, packet, values, *_ = example()
    bundle["brief"]["limits"][quantity] = limit
    before = copy.deepcopy(bundle)
    raw = reply("gain*tanh(m)+m")
    with pytest.raises(ValueError, match="exceeds frozen construction limits"):
        old.apply_edits(bundle, packet, raw)
    result = edits.apply_edits(bundle, packet, raw)
    assert result["outcome"] == "committed"
    audit = result["provenance"]["size_audit"]
    assert quantity in {x["quantity"] for x in audit["after"]["exceeded_references"]}
    assert audit["after"]["revision_enforcement"] == "advisory"
    assert bundle == before
    payload = edits.payload(bundle, packet, values)
    assert "limits" not in payload["public_brief"]
    assert payload["current_model_size"] == edits.model_size(bundle)
    assert "validation" not in payload and "test" not in payload


def test_unused_declaration_cleanup_preserves_all_scientific_content():
    bundle, packet, *_ = example()
    raw = {
        **reply("-a*(m-gain)"),
        "new_parameters": [
            {"name": "a", "role": "coefficient"},
            {"name": "orphan", "role": "shape"},
        ],
    }
    before = copy.deepcopy(raw)
    result = edits.apply_edits(bundle, packet, raw)
    assert raw == before
    model = result["bundle"]["initialization"]["base_candidate"]
    assert (
        next(e for e in model["processes"] if e["name"] == "f")["expression"]
        == "-a*(m-gain)"
    )
    assert (
        next(p for p in model["parameters"] if p["name"] == "a")["role"]
        == "nonnegative_coefficient"
    )
    assert "orphan" not in {p["name"] for p in model["parameters"]}
    assert (
        result["provenance"]["unused_new_declarations_removed"][0]["parameter"]
        == "orphan"
    )


def test_unused_metadata_only_does_not_manufacture_a_revision():
    bundle, packet, *_ = example()
    result = edits.apply_edits(
        bundle,
        packet,
        {"hypothesis": "No change", "new_parameters": [{"name": "unused"}]},
    )
    assert result["outcome"] == "no_change" and result["bundle"] is None
    assert len(result["provenance"]["unused_new_declarations_removed"]) == 1


def test_used_output_parameters_and_local_initial_maps_are_kept():
    bundle, packet, *_ = example()
    raw = {
        **reply(),
        "output_expression": "y+a",
        "new_parameters": [{"name": "a"}],
        "initializers": [
            {
                "state": "y",
                "causal_map": {
                    "expression": "b*v01",
                    "parameters": [{"name": "b", "role": "coefficient"}],
                },
            }
        ],
    }
    result = edits.apply_edits(bundle, packet, raw)
    model = result["bundle"]["candidate"]
    assert any(p["name"] == "a" for p in model["parameters"])
    assert (
        result["bundle"]["initialization"]["plan"]["rules"]["y"]["initial"][
            "expression"
        ]
        == "b*v01"
    )
    assert result["provenance"]["unused_new_declarations_removed"] == []


def test_initial_map_reference_is_not_mistaken_for_unused_metadata():
    bundle, packet, *_ = example()
    raw = {
        **reply(),
        "new_parameters": [{"name": "b"}],
        "initializers": [
            {
                "state": "m",
                "causal_map": {
                    "expression": "b*v01",
                    "parameters": [{"name": "b", "role": "coefficient"}],
                },
            }
        ],
    }
    with pytest.raises(ValueError) as caught:
        edits.apply_edits(bundle, packet, raw)
    assert caught.value.code == "INITIALIZER_PARAMETER_SCOPE"


@pytest.mark.parametrize(
    "expression,parameters",
    [
        ("typo*m", [{"name": "unused"}]),
        ("__import__('os')", []),
        ("v01+m", []),
        ("tanh(a*m)", [{"name": "a"}]),
    ],
)
def test_advisory_policy_never_bypasses_expression_and_role_checks(
    expression, parameters
):
    bundle, packet, *_ = example()
    with pytest.raises((ValueError, ModelValidationError)):
        edits.apply_edits(
            bundle, packet, {**reply(expression), "new_parameters": parameters}
        )


def test_missing_initials_are_actionable_and_not_automatically_invented():
    bundle, packet, values, *_ = example()
    raw = {
        "hypothesis": "Add memory",
        "equations": [
            {"component": "z", "kind": "dynamic", "expression": "-z+u01"},
            {"component": "m", "expression": "-rate*m+gain*z"},
        ],
    }
    with pytest.raises(ValueError, match="explicit initializer") as caught:
        edits.apply_edits(bundle, packet, raw)
    feedback = edits.feedback(bundle, packet, values, raw, caught.value)
    assert "causal_map=null" in feedback["initializer_guidance"]
    raw["initializers"] = [{"state": "z", "causal_map": None}]
    assert edits.apply_edits(bundle, packet, raw)["outcome"] == "committed"


def test_no_latent_options_preserve_target_alias_and_forbid_hidden_odes(tmp_path):
    bundle, packet, values, *_ = example()
    bundle["source_task"]["arm"] = "no_latent"
    caps = edits.payload(bundle, packet, values)["state_capabilities"]
    assert caps["existing_target_states"] == ["y"]
    assert not caps["new_hidden_dynamic_states_allowed"]
    assert caps["new_algebraic_processes_allowed"]
    # Actual certificate still rejects a hidden dynamic state under that control.
    _source, plan = v3_fixture(tmp_path)
    task = plan["tasks"][0]
    revised = edits.apply_edits(bundle, packet, reply())["bundle"]
    certificate = pipeline.certificates(
        revised, plan["cells"][task["cell"]], {"arm": "no_latent"}
    )
    assert not certificate["ablation_constraint_pass"]
    assert (
        "m"
        in pipeline._certificate_feedback(certificate, {"arm": "no_latent"})[
            "no_latent_forbidden_states"
        ]
    )


def load_audit():
    spec = importlib.util.spec_from_file_location(
        "revision_audit",
        Path(__file__).resolve().parents[1] / "scripts/audit_review_parameters.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_actual_v4_round_numbering_and_cannot_downgrade(tmp_path):
    source, _ = v3_fixture(tmp_path)
    v4 = tmp_path / "v4"
    original = continuation.prepare(source, v4, 7, 5, protocol=io.PARAMETER_PROTOCOL)
    for task in original["tasks"]:
        anchor = io.read_round(v4, task, 0)
        sealed_write(
            io.round_path(v4, task, 5) / "result.json",
            {
                "task": task,
                "round": 5,
                "status": "complete",
                "selected": anchor["selected"],
                "closed": anchor["selected"] is None,
                "test_data_opened": False,
            },
        )
    root = tmp_path / "v5"
    plan = continuation.prepare(v4, root, 12, 3, protocol=io.REVISION_PROTOCOL)
    assert plan["continuation"]["source_round"] == 12
    assert plan["continuation"]["source_phase_round"] == 5
    assert plan["config"]["model_settings"] == original["config"]["model_settings"]
    for task in plan["tasks"]:
        assert (
            io.read_round(root, task, 0)["selected"]
            == io.read_round(v4, task, 5)["selected"]
        )
    with pytest.raises(ValueError, match="source protocol"):
        continuation.prepare(
            root, tmp_path / "downgrade", 12, 1, protocol=io.PARAMETER_PROTOCOL
        )


def test_v5_import_audit_reporting_and_resume_preserve_source(tmp_path):
    source, original = v3_fixture(tmp_path)
    task = original["tasks"][0]
    parent = io.read_round(source, task, 0)
    # A nine-term expression fails the old eight-term limit, with no data change.
    raw = reply("+".join(["gain*m"] * 9))
    sealed_write(
        io.round_path(source, task, 1) / "proposal.json",
        {
            "parent_sha256": parent["artifact_sha256"],
            "status": "revision_failed",
            "attempts": [
                {
                    "accepted": False,
                    "raw": raw,
                    "feedback": {
                        "code": "SCIENTIFIC_CONTENT_CONTRACT",
                        "message": "revised model exceeds frozen construction limits",
                    },
                }
            ],
        },
    )
    before = {str(p): p.read_bytes() for p in source.rglob("*.json")}
    root = tmp_path / "v5"
    plan = continuation.prepare(source, root, 7, 3, protocol=io.REVISION_PROTOCOL)
    assert plan["continuation"]["source_phase_round"] == 5
    assert plan["config"]["fit_profile"] == original["config"]["fit_profile"]
    assert (
        continuation.prepare(source, root, 7, 3, protocol=io.REVISION_PROTOCOL) == plan
    )
    audit = load_audit().audit(root)
    assert audit["failed_visits_with_newly_valid_reply"] == 1
    assert audit["previously_accepted_now_blocked"] == []
    assert audit["records"][0]["size_audit"]["after"]["exceeded_references"]
    assert load_audit().audit(root) == audit
    summary = reporting.report(root)
    assert summary["rows"][0]["retained_size"]["revision_enforcement"] == "advisory"
    assert [r["round"] for r in summary["rows"][:4]] == [7, 8, 9, 10]
    assert (root / "revision_diagnostics.json").exists()
    assert {str(p): p.read_bytes() for p in source.rglob("*.json")} == before
    (source / "evaluation_freeze.json").write_text("{}")
    with pytest.raises(ValueError, match="sealed"):
        continuation.prepare(
            source, tmp_path / "sealed", 7, 3, protocol=io.REVISION_PROTOCOL
        )
