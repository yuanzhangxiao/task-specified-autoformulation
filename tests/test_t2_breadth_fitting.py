"""Physical rollout parity, constrained retention and immutable breadth fit receipts."""

from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import t2_breadth_search as search
from autoformalism.fitting.t2_breadth_rollout import BreadthRollout
from autoformalism.research import t2_breadth_fit_intake as intake
from autoformalism.research import t2_breadth_fitting as campaign
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from scripts import run_t2_breadth_fitting as cli
from scripts import submit_review_continuation as scheduler
from tests.test_insulin_revision_fit import specimen  # noqa: F401


def example(fixture, branch):
    problem, _, training, _, warm = fixture
    raw = problem.model_dump(mode="json")
    p = dict(warm)
    eq = next(x for x in raw["base_candidate"]["state_equations"] if x["state"] == "I")
    if branch == "c15-4":
        eq["rhs"] = f"({eq['rhs']})+basal_floor*(1-tanh(I/eps))"
        for n, role in [
            ("basal_floor", "nonnegative_coefficient"),
            ("eps", "positive_shape"),
        ]:
            raw["base_candidate"]["parameters"].append(
                {"name": n, "scope": "global", "role": role}
            )
            raw["parameter_guesses"][n] = p[n] = 1.5
    if branch == "c15-0":
        eq["rhs"] = f"({eq['rhs']})+max(-I,0)"
        p["k_insulin_delay"] = -1.0
    return BreadthRollout(branch, PrefixFitProblem.model_validate(raw), training, p)


@pytest.mark.parametrize("branch", ["c15-0", "c15-4", "c15-3"])
def test_batched_search_matches_independent_original_expressions(specimen, branch):  # noqa: F811
    model = example(specimen, branch)
    value = model.fast(model.warm)
    truth, states = model.original(model.warm, 2)
    assert truth.shape == value.shape == (2, 13, 3)
    assert np.isfinite(states).all()
    np.testing.assert_allclose(value, truth, rtol=2e-4, atol=2e-4)
    assert model.affine == (branch == "c15-3")
    if branch == "c15-0":
        assert model.chart is None
    else:
        assert np.allclose(model.fast(model.lift(model.reduce(model.warm))), value)


def test_nested_zero_basal_seeds_and_release_keep_control_fixed(specimen):  # noqa: F811
    model = example(specimen, "c15-4")
    seeds = search.starts(model, control=True)
    assert list(seeds) == ["inherited"] and seeds["inherited"]["basal_floor"] == 0
    result = search.search(
        model,
        seeds,
        physical=True,
        control=True,
        seconds=10,
        calls=22,
        save=lambda _: None,
    )
    assert result["search_calls_started"] <= 22
    assert result["search_calls_completed"] > 1
    assert result["best"]["numerical"]["parameters"]["basal_floor"] == 0
    assert model.warm["basal_floor"] == 1.5


def test_wrong_root_and_nontraining_split_fail(specimen):  # noqa: F811
    problem, _, training, _, warm = specimen
    with pytest.raises(ValueError, match="training"):
        BreadthRollout(
            "c15-0", problem, training.model_copy(update={"name": "val"}), warm
        )
    missing = {k: v for k, v in warm.items() if not k.startswith("trajectory:")}
    with pytest.raises(ValueError, match="incomplete"):
        BreadthRollout("c15-0", problem, training, missing)


def test_handoff_digest_and_modified_construction_seal_fail(tmp_path):
    p = tmp_path / "foreign.tar.gz"
    p.write_bytes(b"different archive")
    with pytest.raises(ValueError, match="archive digest"):
        intake.load_archive(p)
    with pytest.raises(ValueError, match="receipt digest"):
        intake._sealed({"artifact_sha256": "bad", "status": "complete"})


def test_physical_champion_can_have_larger_error_and_absent_is_explicit():
    good = {"training_nmse": 0.21, "sampled_domain_passed": True}
    bad = {"training_nmse": 0.16, "sampled_domain_passed": False}
    records = {"warm": good, "new": bad}
    assert campaign._select(records, feasible=False)["endpoint"] == bad
    assert campaign._select(records, feasible=True)["endpoint"] == good
    assert campaign._select({"bad": bad}, feasible=True)["endpoint"] is None


def test_two_solvers_must_agree_pointwise_even_at_equal_nmse(monkeypatch):
    point = {"k": 2}
    model = SimpleNamespace(
        scales=np.ones(1),
        original=lambda *a, **k: (np.array([[[-1.0]]]), None),
        nmse=lambda arr: float(np.mean(arr**2)),
    )
    monkeypatch.setattr(
        search,
        "diagnostics",
        lambda *a, **k: {
            "training_nmse": 1.0,
            "_observed_predictions": np.array([[[1.0]]]),
        },
    )
    with pytest.raises(ValueError, match="replay"):
        campaign._verified(model, point, {}, 3)


def test_changed_plan_foreign_status_and_missing_branches(tmp_path, monkeypatch):
    plan = {"tasks": ["../other"]}
    seal(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="unexpected"):
        campaign.status(tmp_path)
    assert intake.TASKS[0] == "c15-4" and len(intake.TASKS) == 6
    assert set(intake.TASKS) == set(intake.BRANCHES) | {intake.CONTROL}


def test_task_resume_never_restarts_reserved_search(tmp_path, monkeypatch):
    policy = campaign.Policy(seconds_per_stage=1, calls_per_stage=4)
    plan = {"tasks": ["c15-0"], "policy": policy.model_dump(), "protocol": "fixture"}
    data = {
        "fixed": {
            "c15": {"all_training_nmse": 2.0},
            "c15-0": {"all_training_nmse": 3.0},
        },
        "models": {"c15-0": {"origin": "c15"}},
    }
    seal(tmp_path / "plan.json", plan)
    seal(tmp_path / "inputs.json", data)
    monkeypatch.setattr(campaign, "verify", lambda *a, **kw: (plan, data))
    model = SimpleNamespace(branch="c15-0", affine=False)
    monkeypatch.setattr(campaign, "_model", lambda *a: model)
    monkeypatch.setattr(search, "starts", lambda *a, **kw: {"inherited": {"k": 3}})

    def verified(model, point, *args, **kwargs):
        return {
            "training_nmse": float(point["k"]),
            "parameters": point,
            "two_original_solvers_verified": True,
            "sampled_domain_passed": True,
        }

    monkeypatch.setattr(campaign, "_verified", verified)
    calls = []

    def fake(model, seeds, *, physical, save, **kwargs):
        calls.append(physical)
        rec = {
            "status": "budget_exhausted",
            "search_calls_completed": 4,
            "best": {
                "admissible": {"parameters": {"k": 1}},
                "numerical": {"parameters": {"k": 1}},
            },
        }
        save(rec)
        return rec

    monkeypatch.setattr(search, "search", fake)
    row = campaign.run_task(tmp_path, 0)
    assert row["status"] == "fit_complete" and calls == [False, True]
    assert row["cost"]["search_calls_completed_total"] == 8
    assert campaign.run_task(tmp_path, 0) == row and calls == [False, True]
    assert campaign.report(tmp_path)["fitted"] == 1
    assert campaign.inspection(tmp_path).is_file()


def test_failed_replay_keeps_verified_start(tmp_path, monkeypatch):
    identity = {"task": "c15-0", "plan_sha256": "test"}
    folder = tmp_path / "results" / "c15-0"
    v = {
        "two_original_solvers_verified": True,
        "parameters": {"k": 2},
        "training_nmse": 2.0,
        "sampled_domain_passed": True,
    }
    bound = campaign._allowances(campaign.Policy())
    for name, fn in [
        ("warm", lambda _: v),
        (
            "verify_numerical_numerical",
            lambda _: (_ for _ in ()).throw(ValueError("replay failed")),
        ),
    ]:
        campaign.journal._operation(folder, identity, name, bound[name], fn)
    ops = campaign.operations._operations(folder, identity, bound)
    r = campaign._assemble(
        identity,
        {
            "models": {"c15-0": {"origin": "c15"}},
            "fixed": {"c15": {"all_training_nmse": 2.0}},
        },
        ops,
    )
    assert r["status"] == "incomplete"
    assert r["best_numerical"]["origin"] == "warm"
    assert "verify_numerical_numerical" in r["operation_errors"]


def test_delta_submission_idempotent_and_ambiguous_reply_spends_intent(
    tmp_path, monkeypatch
):
    plan = {"tasks": list(intake.TASKS), "source_sha256": "frozen"}
    monkeypatch.setattr(cli.campaign, "verify", lambda root: (plan, {}))
    monkeypatch.setattr(cli, "runtime_identity", lambda: {"mode": "native"})
    monkeypatch.delenv("AF_T2_FIT_CONCURRENCY", raising=False)
    monkeypatch.delenv("AF_REPO_ROOT", raising=False)
    monkeypatch.delenv("AF_T2_BREADTH_FIT_ROOT", raising=False)
    calls = []

    def fake(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(
            returncode=0, stdout=str(200 + len(calls)) + "\n", stderr=""
        )

    monkeypatch.setattr(scheduler.subprocess, "run", fake)
    root = tmp_path / "first"
    r = cli.submit(root)
    assert r["jobs"] == {"task": "201", "report": "202"} and "--array=0-5%2" in calls[0]
    assert "--dependency=afterany:201" in calls[1]
    assert cli.submit(root) == r and len(calls) == 2
    assert read_seal(root / "launcher_execution.json")["gpu_count"] == 0
    other = tmp_path / "ambiguous"
    count = []

    def blank(*a, **kw):
        count.append(1)
        return SimpleNamespace(returncode=0, stdout="", stderr="timeout")

    monkeypatch.setattr(scheduler.subprocess, "run", blank)
    for _ in range(2):
        with pytest.raises(ValueError):
            cli.submit(other)
    assert count == [1]


def test_export_requires_completed_report_and_keeps_training_roots(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(campaign, "report", lambda root: {"status": "incomplete"})
    with pytest.raises(ValueError, match="verified terminal"):
        cli.export_models(tmp_path)
    point = {"k": 0.25, "trajectory:train_000:init_A_value": 4.0}
    endpoint = {
        "target_nmse": {},
        "domain": {},
        "shifted_90g": [],
        "extension_150g": {},
        "parameters": point,
        "parameter_sha256": campaign.public.content_sha256(point),
        "training_nmse": 0.3,
        "sampled_domain_passed": False,
        "own_depletion_rate": -0.1,
        "common_depletion_rate": -0.2,
    }
    row = {
        "task": "c15-0",
        "branch": "c15-0",
        "is_nested_control": False,
        "best_numerical": {"endpoint": endpoint},
        "best_sampled_admissible": {"endpoint": None},
    }
    monkeypatch.setattr(
        campaign,
        "report",
        lambda root: {"status": "complete", "rows": [row], "plan_sha256": "plan"},
    )
    source = {
        "receipt": "receipt",
        "model_sha256": "model",
        "problem": {"name": "test"},
    }
    monkeypatch.setattr(
        campaign,
        "verify",
        lambda *a, **kw: (
            {"prefix": {"duration": 8}, "training_sha256": "train"},
            {"models": {"c15-0": source}},
        ),
    )
    assert cli.export_models(tmp_path)["models"] == 1
    saved = read_seal(tmp_path / "fitted-models.json")
    assert saved["models"][0]["best_numerical"]["parameters"] == point
    assert saved["models"][0]["best_sampled_admissible"] is None
    assert saved["prefix_forecast_evaluated"] is False
