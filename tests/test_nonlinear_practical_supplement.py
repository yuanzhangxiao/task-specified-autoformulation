"""M42 source preservation, selection, separate budgets and portable reporting."""

import shutil
from copy import deepcopy

import pytest
from pydantic import ValidationError

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import nonlinear_practical_supplement as campaign


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    monkeypatch.setattr(campaign.public, "_source_identity", lambda: "a" * 64)


@pytest.fixture(scope="module")
def source_template(tmp_path_factory):
    """Seal endpoint stand-ins; actual numerical search has separate regressions."""
    root = tmp_path_factory.mktemp("m41-source")
    previous = campaign.previous
    previous.prepare(root)
    plan, data = previous.verify(root, runtime=False)
    for index, task in enumerate(plan["tasks"]):
        identity = {"plan_sha256": campaign.public.content_sha256(plan), "task": task}
        backend = {
            "status": "complete",
            "identity": previous.fit_identity(plan, data, task),
            "campaign_identity": identity,
            "parameters": dict(previous.training_problem(data, task).start),
            "objective_value": 10.0,
            "verification": {"passed": True},
            "cost": {
                "wall_seconds": 1.0,
                "cpu_seconds": 0.5,
                "accounting_complete": True,
            },
        }
        folder = root / "results" / task["task_id"]
        seal(folder / "backend.json", backend)
        values = {}
        for name, definition in previous._definitions(plan, data, task).items():
            stage_identity = previous.stage_identity(identity, backend, definition)
            value = {
                "identity": stage_identity,
                "status": ("partial" if index < 2 else "complete")
                if name == "profile"
                else "evaluated",
                "wall_seconds": 10.0,
                "cpu_seconds": 2.0,
                "budget_charge_seconds": 10.0,
                "accounting_complete": True,
                "calls": 4,
                "completed_calls": 4,
                "failed_calls": 0,
            }
            if name == "profile":
                value["diagnostic"] = {
                    "endpoint_qualification": {
                        "numerically_qualified": True,
                        "retained_fit_inadequate": False,
                    },
                    "supports": [],
                }
            seal(folder / f"{name}-started.json", stage_identity)
            seal(folder / f"{name}.json", value)
            values[name] = value
        seal(
            folder / "result.json",
            {
                "identity": identity,
                "status": "complete",
                "backend_sha256": campaign.public.content_sha256(backend),
                "profile_sha256": campaign.public.content_sha256(values["profile"]),
                "evaluation_sha256": campaign.public.content_sha256(
                    values["evaluation"]
                ),
            },
        )
    return root


@pytest.fixture
def source(source_template, tmp_path):
    target = tmp_path / "source"
    shutil.copytree(source_template, target)
    return target


@pytest.fixture
def prepared(source, tmp_path):
    root = tmp_path / "supplement"
    assert campaign.prepare(root, source)["tasks"] == 2
    return root


def profile_result(status="complete"):
    return {
        "status": status,
        "calls": 3,
        "completed_calls": 3,
        "failed_calls": 0,
        "budget_charge_seconds": 0.5,
        "accounting_complete": True,
        "timing": {
            "total_wall_seconds": 0.5,
            "total_cpu_seconds": 0.2,
            "numerical_wall_seconds": 0.3,
            "numerical_cpu_seconds": 0.15,
            "save_callback_wall_seconds": 0.1,
            "save_callback_cpu_seconds": 0.02,
        },
        "endpoint_qualification": {
            "numerically_qualified": status == "complete",
            "retained_fit_inadequate": status == "unqualified",
        },
        "supports": [],
    }


def test_select_only_partial_and_preserve_source_and_missing_costs(
    source, prepared, monkeypatch
):
    before = campaign.snapshot._files(source)
    monkeypatch.setattr(
        campaign.previous.measurement_fit, "run", lambda *a, **k: pytest.fail("refit")
    )
    monkeypatch.setattr(
        campaign.previous, "evaluate", lambda *a, **k: pytest.fail("evaluator")
    )
    plan, _ = campaign.verify(prepared)
    old = read_seal(prepared / "source_snapshot/plan.json")
    assert "reference_boundary_cells" not in old["diagnostic"]["policy"]
    assert plan["policy"]["diagnostic"]["reference_boundary_cells"]
    assert plan["policy"]["diagnostic"]["wall_seconds"] == 60
    summary = campaign.report(prepared)
    assert summary["source_verified_fits"] == 36
    assert summary["source_profile_status_counts"] == {"partial": 2, "complete": 34}
    assert summary["status_counts"] == {"missing": 2}
    assert summary["expected"] == 2 and summary["recorded"] == 0
    assert len(summary["rows"]) == 36
    assert summary["cost"]["extension_diagnostic"]["calls_total"] is None
    assert summary["cost"]["source_diagnostic"]["calls_total"] == 144
    assert summary["cost"]["source_fit_unchanged"]["observed"]["wall_seconds"] == 36
    assert before == campaign.snapshot._files(source)


def test_no_refitting_or_truth_and_exact_terminal_resume_and_portability(
    source, prepared, tmp_path, monkeypatch
):
    before = campaign.snapshot._files(source)
    captured = []

    def run(problem, parameters, *, policy, state, save):
        assert "truth" not in problem.model_dump()
        assert parameters == dict(problem.start)
        assert policy.reference_boundary_cells and state is None
        captured.append(parameters)
        result = profile_result()
        save(result)
        return result

    monkeypatch.setattr(campaign.profiles, "run", run)
    monkeypatch.setattr(
        campaign.previous.measurement_fit, "run", lambda *a, **k: pytest.fail("refit")
    )
    monkeypatch.setattr(
        campaign.previous, "evaluate", lambda *a, **k: pytest.fail("evaluator")
    )
    first = campaign.run_task(prepared, 0)
    files = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (prepared / "results").rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(campaign.profiles, "run", lambda *a, **k: pytest.fail("retry"))
    assert campaign.run_task(prepared, 0) == first
    assert all(
        (p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in files.items()
    )
    assert len(captured) == 1
    summary = campaign.report(prepared)
    assert summary["combined_profile_status_counts"] == {"complete": 35, "partial": 1}
    assert summary["cost"]["combined_diagnostic"]["calls_observed"] == 147
    assert summary["cost"]["combined_diagnostic"]["calls_total"] is None
    assert summary["cost"]["extension_diagnostic"]["wall_seconds_observed"] == 0.5
    assert before == campaign.snapshot._files(source)
    moved = tmp_path / "moved"
    shutil.move(str(prepared), moved)
    monkeypatch.setattr(campaign.public, "_source_identity", lambda: "other-source")
    assert campaign.report(moved) == summary
    with pytest.raises(ValueError, match="source/runtime"):
        campaign.run_task(moved, 0)


@pytest.mark.parametrize("status", ["unqualified", "failed", "partial"])
def test_adverse_supplement_is_not_hidden_by_historical_profile(
    prepared, monkeypatch, status
):
    monkeypatch.setattr(
        campaign.profiles, "run", lambda *a, **k: profile_result(status)
    )
    campaign.run_task(prepared, 0)
    summary = campaign.report(prepared)
    assert summary["rows"][0]["profile"]["status"] == status
    assert summary["rows"][0]["extension_profile_status"] == status
    assert summary["rows"][0]["source_profile_status"] == "partial"


def test_no_checkpoint_interruption_consumes_allowance_without_retry(
    prepared, monkeypatch
):
    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(campaign.profiles, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_task(prepared, 0)
    summary = campaign.report(prepared)
    assert summary["status_counts"] == {"unfinished": 1, "missing": 1}
    assert (
        summary["cost"]["extension_diagnostic"]["budget_charge_seconds_observed"] == 60
    )
    monkeypatch.setattr(campaign.profiles, "run", lambda *a, **k: pytest.fail("retry"))
    result = campaign.run_task(prepared, 0)
    assert result["status"] == "interrupted"
    assert not result["profile"]["accounting_complete"]
    assert result["profile"]["wall_seconds"] is None
    assert result["profile"]["budget_charge_seconds"] == 60
    assert campaign.run_task(prepared, 0) == result


def test_source_and_policy_changes_fail_closed(prepared, source):
    assert campaign.prepare(prepared, source)["tasks"] == 2
    policy = campaign.Policy(
        diagnostic=campaign.profiles.Policy(
            reference_boundary_cells=True, wall_seconds=70
        )
    )
    with pytest.raises(ValueError, match="source/policy"):
        campaign.prepare(prepared, source, policy)
    path = prepared / "source_snapshot/inputs.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="snapshot/identity"):
        campaign.verify(prepared)


def test_invalid_endpoint_does_not_start_supplement(source, tmp_path, monkeypatch):
    original = campaign.previous.report_row

    def wrong(*a):
        row = original(*a)
        row["profile"]["diagnostic"]["endpoint_qualification"][
            "numerically_qualified"
        ] = False
        return row

    monkeypatch.setattr(campaign.previous, "report_row", wrong)
    with pytest.raises(ValueError, match="every source endpoint"):
        campaign.prepare(tmp_path / "bad", source)


def test_selection_does_not_depend_on_evaluator_truth(source, monkeypatch):
    expected = campaign.audit(source)["selected_task_ids"]
    plan, data, rows = campaign._source_rows(source)
    for row in rows:
        row["evaluation"] = {"truth_error": 1e100}
    monkeypatch.setattr(campaign, "_source_rows", lambda _: (plan, data, rows))
    assert campaign.audit(source)["selected_task_ids"] == expected


def test_unknown_index_overlap_and_false_policy_are_rejected(prepared, source):
    for index in (-1, 2, True):
        with pytest.raises(ValueError, match="unknown"):
            campaign.run_task(prepared, index)
    with pytest.raises(ValueError, match="separate"):
        campaign.prepare(source / "inside", source)
    with pytest.raises(ValidationError, match="boundary-cell"):
        campaign.Policy(diagnostic=campaign.profiles.Policy())
    with pytest.raises(ValidationError):
        campaign.Policy(extra="unsafe")


def test_resealed_terminal_cannot_override_profile(prepared, monkeypatch):
    monkeypatch.setattr(campaign.profiles, "run", lambda *a, **k: profile_result())
    result = campaign.run_task(prepared, 0)
    path = prepared / "results" / result["task_id"] / "result.json"
    altered = deepcopy(read_seal(path))
    altered["profile_sha256"] = "0" * 64
    path.unlink()
    seal(path, altered)
    with pytest.raises(ValueError, match="terminal identity"):
        campaign.run_task(prepared, 0)
