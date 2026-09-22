"""Saved-campaign identity, public-only feedback, and immutable audit resume."""

import json
import shutil

import pytest

from autoformalism.rebuttal import basin_equation_audit as audit
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import smoke_basin_equation_audit as smoke


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    return smoke.fixture(tmp_path_factory.mktemp("basin-public-audit"))


@pytest.fixture
def source(saved, tmp_path):
    destination = tmp_path / "source"
    shutil.copytree(saved, destination)
    return destination


def rewrite(path, transform):
    value = sealed_read(path)
    value.pop("artifact_sha256")
    transform(value)
    path.unlink()
    return sealed_write(path, value)


def test_reconstruct_resume_missing_rows_and_feedback_boundary(source, tmp_path):
    before = smoke.snapshot(source)
    output = tmp_path / "audit"
    result = audit.audit(source, output)
    assert result["status_counts"] == {"assessed": 2, "unavailable": 14}
    assert result["planned_constructions"] == 8
    assert result["planned_arms"] == 16
    assert result["scientific_compliance_score"] is None
    assert audit.audit(source, output) == result
    assert smoke.snapshot(source) == before
    assert (
        result["llm_calls"]
        == result["optimizer_calls"]
        == result["solver_rollouts"]
        == 0
    )
    for path in (output / "feedback").glob("*.json"):
        packet = sealed_read(path)
        assert not packet["automatic_rejection"] and not packet["critic_called"]
        text = json.dumps(packet)
        assert "normalized_mse" not in text and "PRIVATE_ORACLE_POISON" not in text
        assert "validation" not in packet


def test_signal_values_and_validation_never_affect_findings(source, tmp_path):
    first = audit.audit(source, tmp_path / "first")

    def poison(plan):
        for cell in plan["cells"].values():
            cell["validation"] = {"POISON_VALIDATION": "must not inspect"}
            for row in cell["training"]["rows"]:
                row["targets"] = {"POISON_TRAINING": "must not inspect"}
                row["external_inputs"] = "must not inspect"
                row["time"] = "must not inspect"

    rewrite(source / "plan.json", poison)
    second = audit.audit(source, tmp_path / "second")
    assert [r["assessment"] for r in first["rows"]] == [
        r["assessment"] for r in second["rows"]
    ]
    assert first["checks_by_code"] == second["checks_by_code"]


def test_refuses_drift_path_escape_and_wrong_fit_link(source, tmp_path):
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="separate"):
        audit.audit(source, source / "audit")
    audit.audit(source, output)
    path = next(source.glob("results/*/proposal.json"))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen artifact differs"):
        audit.audit(source, output)
    proposal = sealed_read(path)
    sealed_write(
        path.parent / "result.json",
        {
            "task": proposal["task"],
            "proposal_sha256": "wrong",
            "status": "complete",
            "fit": {"parameters": {}},
        },
    )
    with pytest.raises(ValueError, match="another model"):
        audit.audit(source, tmp_path / "bad-link")
    path.unlink()
    path.symlink_to(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="escapes"):
        audit.audit(source, tmp_path / "escape")


def test_tampered_gain_never_becomes_a_scientific_failure(source, tmp_path):
    path = next(source.glob("results/*/proposal.json"))
    rewrite(
        path, lambda p: p["bundle"]["candidate"]["state_equations"][0].update(rhs="0")
    )
    with pytest.raises(ValueError, match="gain assembly differs"):
        audit.audit(source, tmp_path / "bad-gain")


def test_reports_original_fallback_errors_without_new_repair_calls(source, tmp_path):
    result = audit.audit(source, tmp_path / "audit")
    for row in result["rows"]:
        if row["status"] == "assessed":
            assert row["construction_errors"] == []
            assert row["saved_fit_status"] == "unavailable"


def test_public_rule_profile_cannot_silently_change(source, tmp_path):
    rewrite(
        source / "plan.json",
        lambda p: p["cells"]["coupled"]["brief"].update(
            scientific_context="This is a different specification without conservation."
        ),
    )
    with pytest.raises(ValueError, match="rule profile"):
        audit.audit(source, tmp_path / "changed-spec")
