"""Live/preview agreement, historical provenance and escaped human review."""

import json

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import construction_comparison as campaign
from autoformalism.research import construction_focus_review as review
from autoformalism.search import construction_completion as completion
from autoformalism.search import construction_coverage as coverage
from autoformalism.search import construction_focus as focus
from autoformalism.search import construction_readable as readable
from tests.test_construction_comparison import source_fixture
from tests.test_construction_ledger import equation, variable
from tests.test_explicit_variable_bindings import response
from tests.test_phase_c_construction_baseline import tokenize


@pytest.mark.parametrize(
    "study,policy",
    [
        ("focused_prompt_confirmation", focus.POLICY),
        ("readable_prompt_confirmation", readable.POLICY),
        ("completion_feedback_confirmation", completion.POLICY),
        ("public_coverage_confirmation", coverage.POLICY),
    ],
)
def test_focused_study_live_projection_reports_and_offline_review(
    tmp_path, study, policy
):
    source_fixture(tmp_path)
    root = tmp_path / "study"
    plan = campaign.freeze(tmp_path / "new", root, study=study)
    assert campaign.verify(root) == plan
    assert plan["config"]["repair_requests"] == 5
    assert plan["config"]["bookkeeping_policy"] == policy
    assert len(plan["tasks"]) == 8
    requests = []

    def send(url, body, timeout):
        p = json.loads(body["messages"][1]["content"])
        requests.append(body)
        assert "current_draft" not in p and "bookkeeping_policy" not in p
        targets = [
            v["name"]
            for v in p["public_brief"]["public_variables"]
            if v["data_role"] == "target"
        ]
        driver = next(
            v["name"]
            for v in p["public_brief"]["public_variables"]
            if v["data_role"] == "external_input"
        )
        edit = {**p["response_template"], "stage_complete": True}
        if p["stage"] == "variables":
            edit["variables"] = [variable(n) for n in [*targets, "m"]]
            edit["mechanism_bindings"] = [
                {"requirement_id": r["id"], "memory_states": ["m"]}
                for r in p["public_brief"]["requirements"]
                if r["requires_dynamic_memory"]
            ]
        elif p["stage"] == "equations":
            edit["equations"] = [equation(n, "m", n) for n in targets] + [
                equation("m", driver, "m")
            ]
        return response(edit)

    for task in plan["tasks"]:
        result = campaign.propose(
            root, plan, task, "http://offline", transport=send, token_transport=tokenize
        )
        assert result["status"] == "topology_complete"
    campaign.report(root, plan)
    assert (
        json.loads((root / "VARIABLES.json").read_text())["completed_variable_stages"]
        == 8
    )
    assert (
        json.loads((root / "SHARED_PROCESSES.json").read_text())[
            "completed_shared_stages"
        ]
        == 8
    )
    count = len(requests)
    for task in plan["tasks"]:
        campaign.propose(
            root, plan, task, "http://offline", transport=send, token_transport=tokenize
        )
    assert len(requests) == count
    # First request has no previous receipt; preview and live must match exactly.
    task = plan["tasks"][0]
    directory = root / "results" / task["task_id"]
    event = sealed_read(directory / "construction/events/000.json")
    row = review.preview(
        plan["cells"][task["benchmark_id"]], task, event, requests[0], policy
    )
    assert row["after"] == requests[0]
    output = tmp_path / "review"
    summary = review.render(root, output, policy)
    assert summary["requests"] == count and summary["llm_calls"] == 0
    assert "New model view" in (output / "request-000.html").read_text()
    with pytest.raises(ValueError, match="must be empty"):
        review.render(root, output)
    with pytest.raises(ValueError, match="outside"):
        review.render(root, root / "review")


def test_review_escapes_untrusted_prose():
    hostile = "<script>alert('untrusted')</script>"
    rendered = review._block({"scientific_role": hostile})
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered
    assert "<script>" not in review._page(hostile, rendered)
