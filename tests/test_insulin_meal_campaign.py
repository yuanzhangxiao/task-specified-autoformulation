"""The two CPU arms and the optional proposer consume the same authenticated intake."""

import subprocess
import tarfile

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import insulin_meal_comparison as fit
from autoformalism.fitting import public_fitting as public
from autoformalism.research import insulin_meal_campaign as campaign
from scripts import run_insulin_meal_control as cli
from tests.test_construction_revision_edits import problem


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Use a tiny pinned replacement solely for testing the campaign boundary."""
    p = problem().model_dump(mode="json")
    p["context"]["external_inputs"].append(fit.MEAL)
    rhs = next(
        row for row in p["base_candidate"]["state_equations"] if row["state"] == "I"
    )
    rhs["rhs"] = f"{fit.MEAL} - I + 1"
    data = {
        "problem": p,
        "training": {"name": "train"},
        "numerical": {},
        "retained": {},
        "scientific_aliases": {},
        "compatibility": {},
        "public_brief": {},
        "prefix_policy": {"duration": 8.0},
        "public_task": {"prompt_text": "public fixture"},
    }
    monkeypatch.setattr(campaign, "SOURCE_PACKET", public.content_sha256(data))
    monkeypatch.setattr(public, "_source_identity", lambda: "test-source")
    packet = tmp_path / "source.json"
    seal(packet, data)
    root = tmp_path / "campaign"
    campaign.prepare(root, packet, {})
    return root, packet


def test_prepare_resume_and_offline_inspection(prepared):
    root, packet = prepared
    before = (root / "plan.json").read_bytes()
    assert campaign.prepare(root, packet, {})["arms"] == ["parent", "child"]
    assert (root / "plan.json").read_bytes() == before
    assert campaign.report(root)["status"] == "incomplete"
    with tarfile.open(campaign.inspect(root)) as archive:
        assert {"report/summary.json", "SUMMARY.md", "inputs.json"} <= set(
            archive.getnames()
        )
    with pytest.raises(ValueError, match="does not request"):
        campaign.proposer_inputs(root)
    with pytest.raises(ValueError, match="sealed artifact differs"):
        campaign.prepare(root, packet, {"maximum_calls": 8})


def test_resealed_foreign_packet_fails_before_dispatch(prepared):
    root, packet = prepared
    other = root.parent / "other.json"
    seal(other, read_seal(packet) | {"training": {"name": "val"}})
    with pytest.raises(ValueError, match="pinned candidate"):
        campaign.prepare(root.parent / "wrong", other, {})


@pytest.mark.parametrize("ready", [True, False])
def test_live_followup_requires_two_qualified_terminals_and_witness(
    prepared, monkeypatch, ready
):
    from autoformalism.research import meal_revision_metrics as metrics

    root, _ = prepared
    plan, data = campaign.verify(root)
    child = fit.child_problem(campaign.PrefixFitProblem.model_validate(data["problem"]))
    ops = {
        "ready_for_comparison": ready,
        "metrics": {"example": 1},
        "retained_endpoint": {
            "parameters": dict.fromkeys(child.parameter_guesses, 2.0)
        },
    }
    monkeypatch.setattr(
        campaign, "_terminals", lambda *args: {"parent": ops, "child": ops}
    )
    pair = {"routing": {"eligible_for_proposer": True}}
    monkeypatch.setattr(metrics, "compare", lambda *args: pair)
    if ready:
        value = campaign.proposer_inputs(root)
        assert value["fitted_problem"].parameter_guesses["c_I"] == 2.0
        assert (
            value["result_identity"]["child_problem_sha256"]
            == plan["problem_hashes"]["child"]
        )
        pair["routing"]["eligible_for_proposer"] = False
    with pytest.raises(ValueError, match="does not request"):
        campaign.proposer_inputs(root)


def test_foreign_binding_cannot_be_used_in_comparison(prepared):
    root, _ = prepared
    plan, _ = campaign.verify(root)
    folder = root / "results/child"
    seal(folder / "result.json", {})
    seal(
        folder / "binding.json", plan["binding_hashes"] | {"training_sha256": "foreign"}
    )
    with pytest.raises(ValueError, match="fitting inputs differ"):
        campaign.comparison(root, plan)


def test_scheduler_pins_order_budget_and_reuses_submissions(prepared, monkeypatch):
    root, _ = prepared
    calls = []
    monkeypatch.setattr(cli, "runtime_identity", lambda: {"python": "test"})

    def response(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, f"{100 + len(calls)}\n", "")

    monkeypatch.setattr(subprocess, "run", response)
    first = cli.submit(root)
    assert cli.submit(root) == first
    assert len(calls) == 2
    assert "--array=0-1%2" in calls[0]
    assert "--dependency=afterany:101" in calls[1]
    assert "--partition=cpu" in calls[0]
    assert not any("gpu" in arg for call in calls for arg in call)


def test_ambiguous_scheduler_reply_is_not_retried(prepared, monkeypatch):
    root, _ = prepared
    monkeypatch.setattr(cli, "runtime_identity", lambda: {"python": "test"})
    calls = []

    def timeout(argv, **_kwargs):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, 60)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(ValueError, match="uncertain scheduler timeout"):
        cli.submit(root)
    with pytest.raises(ValueError, match=r"uncertain .* submission"):
        cli.submit(root)
    assert len(calls) == 1
