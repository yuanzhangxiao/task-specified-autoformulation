"""Independent sampling seeds, fixed public inputs, and resumable per-seed evidence."""

import json
from collections import Counter

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.research import construction_comparison as campaign
from tests.test_construction_comparison import source_fixture
from tests.test_construction_ledger import variable
from tests.test_explicit_variable_bindings import response
from tests.test_phase_c_construction_baseline import tokenize


def test_five_seed_roster_is_balanced_fresh_and_frozen(tmp_path):
    old = source_fixture(tmp_path)
    root = tmp_path / "repeated"
    plan = campaign.freeze(
        tmp_path / "new",
        root,
        study="focused_prompt_confirmation",
        seeds=[0, 1, 2, 3, 4],
    )
    assert campaign.verify(root) == plan
    assert len(plan["tasks"]) == 40
    assert Counter(t["benchmark_id"] for t in plan["tasks"]) == dict.fromkeys(
        old["cells"], 5
    )
    assert [t["seed"] for t in plan["tasks"]] == [s for s in range(5) for _ in range(8)]
    assert len({campaign.baseline.namespace(plan, t) for t in plan["tasks"]}) == 40
    assert "inputs" not in plan
    assert not any("starting_checkpoint" in t for t in plan["tasks"])
    assert plan["config"]["model_settings"] == old["config"]["model_settings"]
    assert plan["config"]["repair_requests"] == 5
    assert plan["tasks"][0]["benchmark_id"] != plan["tasks"][8]["benchmark_id"]
    summary = campaign.report(root, plan)
    assert summary["status_counts"] == {"pending": 40}
    assert len(summary["per_case_seed_results"]) == 8
    assert all(
        r["planned"] == 5 and r["finished"] == 0 and r["tokens"]["median"] is None
        for r in summary["per_case_seed_results"]
    )
    assert "Full/seeds [0, 1, 2, 3, 4]" in (root / "SUMMARY.md").read_text()
    assert (root / "VARIABLES.md").exists()
    assert (root / "SHARED_PROCESSES.md").exists()
    with pytest.raises(ValueError):
        campaign.freeze(
            tmp_path / "new", root, study="focused_prompt_confirmation", seeds=[9]
        )


@pytest.mark.parametrize(
    "seeds", [[], [0, 0], [-1], [True], [2**31], list(range(11)), "0"]
)
def test_invalid_seed_rosters_rejected(seeds):
    with pytest.raises(ValueError, match="seeds must"):
        campaign.sampling_seeds("focused_prompt_confirmation", seeds)


def test_legacy_defaults_and_explicit_seed_scope(tmp_path):
    source_fixture(tmp_path)
    plan = campaign.freeze(
        tmp_path / "new", tmp_path / "old", study="focused_prompt_confirmation"
    )
    assert len(plan["tasks"]) == 8 and "sampling_seeds" not in plan
    assert campaign.verify(tmp_path / "old") == plan
    with pytest.raises(ValueError, match="explicit seeds require"):
        campaign.freeze(
            tmp_path / "new", tmp_path / "bad", study="prompt_comparison", seeds=[0, 1]
        )


def test_verify_rejects_missing_seed_task_even_if_resealed(tmp_path):
    source_fixture(tmp_path)
    root = tmp_path / "repeated"
    plan = campaign.freeze(
        tmp_path / "new", root, study="focused_prompt_confirmation", seeds=[1, 3]
    )
    plan.pop("artifact_sha256")
    plan["tasks"].pop()
    (root / "plan.json").unlink()
    sealed_write(root / "plan.json", plan)
    with pytest.raises(ValueError, match="exact 16-task"):
        campaign.verify(root)


def test_sampling_seed_reaches_requests_and_resume_does_not_call_again(tmp_path):
    source_fixture(tmp_path)
    root = tmp_path / "repeated"
    plan = campaign.freeze(
        tmp_path / "new", root, study="focused_prompt_confirmation", seeds=[0, 1]
    )
    case = plan["tasks"][0]["benchmark_id"]
    tasks = [t for t in plan["tasks"] if t["benchmark_id"] == case]
    calls = []

    def send(url, body, timeout):
        calls.append(body)
        packet = json.loads(body["messages"][1]["content"])
        # Deliberately incomplete public inventory exercises bounded failure too.
        return response(
            {
                **packet["response_template"],
                "variables": [variable("y")],
                "stage_complete": True,
            }
        )

    first_seeds = []
    for task in tasks:
        before = len(calls)
        result = campaign.propose(
            root, plan, task, "http://offline", transport=send, token_transport=tokenize
        )
        assert result["status"] == "topology_incomplete"
        first_seeds.append(calls[before]["seed"])
        count = len(calls)
        assert (
            campaign.propose(
                root,
                plan,
                task,
                "http://offline",
                transport=send,
                token_transport=tokenize,
            )
            == result
        )
        assert len(calls) == count
    assert len(set(first_seeds)) == 2
    summary = campaign.report(root, plan)
    selected = next(
        r for r in summary["per_case_seed_results"] if r["benchmark_id"] == case
    )
    assert selected["finished"] == 2 and selected["final_complete"] == 0
    assert selected["tokens"]["median"] is not None
    assert summary["status_counts"] == {"topology_incomplete": 2, "pending": 14}
