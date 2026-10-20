"""M47 independent acquisitions, sealed paired policies and persistent outcomes."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("casadi")

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_noise_reliability as q


def small_policy():
    return q.Policy(
        noise_repeats=(100,),
        base=q.base.Policy(durations=(0, 0.2), root_starts=((0.3, 1.3), (1.3, 0.3))),
    )


def test_acquisitions_are_deterministic_independent_and_paired():
    policy = q.Policy()
    tasks, data = q.roster(policy), q.make_inputs(policy)
    assert len(tasks) == 36
    assert data == q.make_inputs(policy)
    q._validate(data, policy)
    a = data["records"]["separated_noisy_r100_s0"]
    b = data["records"]["separated_noisy_r100_s1"]
    c = data["records"]["separated_noisy_r101_s0"]
    for key in ("training", "validation", "followup"):
        assert a[key] == b[key] and a[key] != c[key]
        assert a["evaluation"]["clean"][key] == c["evaluation"]["clean"][key]
        assert all(row["targets"]["y"][0] == 0.4 for row in a[key]["rows"])
    assert a["problem"] != b["problem"]
    old = q.controls.fixture("separated", noise_sigma=0.01)
    assert a["evaluation"]["roots"] != old["evaluation"]["roots"]
    paired = data["records"]["near_confounded_noisy_r100_s0"]
    for key in ("training", "validation", "followup"):

        def noise(record, key=key):
            return np.array(
                [row["targets"]["y"] for row in record[key]["rows"]]
            ) - np.array(
                [
                    row["targets"]["y"]
                    for row in record["evaluation"]["clean"][key]["rows"]
                ]
            )

        np.testing.assert_allclose(noise(a), noise(paired), atol=1e-15, rtol=0)


@pytest.mark.parametrize(
    "changes",
    [
        {"noise_repeats": ()},
        {"noise_repeats": (0,)},
        {"noise_repeats": (101, 100)},
        {"noise_repeats": (100, 100)},
        {"noise_repeats": tuple(range(100, 121))},
        {"objective_rise_units": 0},
        {"support_rms_limit": float("nan")},
        {"support_seconds": -1},
    ],
)
def test_policy_rejects_invalid_design(changes):
    with pytest.raises(ValidationError):
        q.Policy(**changes)


@pytest.mark.parametrize(
    "key,value",
    [
        ("repeat", 2),
        ("scenario", "different"),
        ("noise_sigma", 0.1),
        ("test_data_opened", True),
        ("followup_scope", "test"),
    ],
)
def test_data_identity_changes_rejected(key, value):
    p = small_policy()
    data = q.make_inputs(p)
    data["records"]["separated_clean_r99_s0"][key] = value
    with pytest.raises(ValueError):
        q._validate(data, p)


def test_stale_or_unpaired_observations_rejected():
    p = small_policy()
    data = q.make_inputs(p)
    row = data["records"]["separated_clean_r99_s1"]["validation"]["rows"][0]
    row["targets"]["y"][1] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        q._validate(data, p)
    record = data["records"]["separated_clean_r99_s1"]
    record["validation"] = q.controls.fingerprint(record["validation"])
    with pytest.raises(ValueError, match="identical observations"):
        q._validate(data, p)


def test_prepare_and_source_identity_no_redraw(tmp_path, monkeypatch):
    p = small_policy()
    first = q.prepare(tmp_path, p)
    before = {f: f.read_bytes() for f in tmp_path.glob("*.json")}
    monkeypatch.setattr(q, "make_inputs", lambda *a: pytest.fail("noise redrawn"))
    assert q.prepare(tmp_path, p) == first
    assert before == {f: f.read_bytes() for f in tmp_path.glob("*.json")}
    monkeypatch.setattr(q.public, "_source_identity", lambda: "changed")
    with pytest.raises(ValueError, match="source/runtime"):
        q.verify(tmp_path)
    assert q.verify(tmp_path, runtime=False)[0]["protocol"] == q.PROTOCOL
    with pytest.raises(ValueError, match="sealed artifact differs"):
        q.prepare(tmp_path, p)


def _window(duration, *, forecast=False, bound=0.01, support_available=True):
    return {
        "duration": duration,
        "metrics": {"duration": duration, "complete": True, "adequate": forecast},
        "support_metrics": {
            "complete": support_available,
            "adequate": support_available and bound is not None and bound <= 0.1,
            "maximum_rms_bound": {"common": bound, "immediate": bound},
        },
    }


def test_rules_share_baseline_but_weak_predictive_support_delays_selection():
    p = q.Policy()
    windows = [
        _window(d, forecast=d >= 0.2, bound=0.5 if d < 1 else 0.05)
        for d in p.base.durations
    ]
    selections = q.select_policies(windows, p)
    assert selections["baseline"]["duration"] == 0.2
    assert selections["noise_aware"]["duration"] == 1
    assert q._followup_durations(selections) == (0.2, 1.0)
    assert selections["noise_aware"]["confidence_probability"] is None
    assert not selections["noise_aware"]["root_information_used_as_gate"]


def test_unbounded_is_available_but_inadequate_and_not_rank_veto():
    p = small_policy()
    windows = [_window(0, bound=None), _window(0.2, forecast=True, bound=0.02)]
    assert q.select_policies(windows, p)["noise_aware"]["duration"] == 0.2
    assert q._followup_durations(q.select_policies(windows, p)) == (0.2,)
    windows[1] = _window(0.2, forecast=True, bound=None)
    selections = q.select_policies(windows, p)
    assert selections["baseline"]["status"] == "selected"
    assert selections["noise_aware"]["status"] == "no_adequate_window"


def test_missing_diagnostic_does_not_veto_baseline_or_disappear():
    p = small_policy()
    windows = [_window(0), _window(0.2, forecast=True, support_available=False)]
    choices = q.select_policies(windows, p)
    assert choices["baseline"]["status"] == "selected"
    assert choices["noise_aware"]["status"] == "unavailable"
    assert q.select_policies(windows[1:], p)["baseline"]["status"] == "unavailable"


def test_horizon_and_prefix_do_not_include_future_observations():
    record = q.make_inputs(small_policy())["records"]["separated_clean_r99_s0"]
    row = q.PublicSplit.model_validate(record["followup"]).rows[0]
    pp = q.base.kernel.PrefixPolicy(duration=0.2)
    changed = row.model_copy(
        update={"targets": {"y": row.targets["y"][:3] + (99.0,) * 38}}
    )
    assert q.base.kernel.make_prefix(row, pp) == q.base.kernel.make_prefix(changed, pp)
    assert q._horizons(row.time, 0.2, small_policy()) == q._horizons(
        changed.time, 0.2, small_policy()
    )
    assert all(
        t > 0.2
        for ts in q._horizons(row.time, 0.2, small_policy()).values()
        for t in ts
    )


def test_missing_tasks_keep_denominators(tmp_path):
    q.prepare(tmp_path, small_policy())
    result = q.report(tmp_path)
    assert result["expected"] == 12 and result["recorded"] == 0
    assert result["status_counts"] == {"missing": 12}
    assert result["cost"]["cpu_seconds_total"] is None
    assert all(
        g["expected_runs"] == 2 and g["scored_runs"] == 0 for g in result["groups"]
    )
    assert all(g["independent_acquisitions"] == 1 for g in result["groups"])
    assert all(g["mean_followup_nmse"]["common"] is None for g in result["groups"])


def test_all_terminal_but_numerically_unavailable_stays_incomplete(
    tmp_path, monkeypatch
):
    q.prepare(tmp_path, small_policy())
    monkeypatch.setattr(q.base.execute, "fit_training", lambda *a, **kw: None)
    for i in range(12):
        result = q.run_task(tmp_path, i)
        assert result["status"] == "incomplete"
    report = q.report(tmp_path)
    assert report["recorded"] == 12 and report["all_results_recorded"]
    assert report["status"] == "incomplete"
    assert report["selection_counts"]["noise_aware"] == {"unavailable": 12}


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("m47_complete")
    q.prepare(root, small_policy())
    with pytest.MonkeyPatch.context() as patch:
        original = q.base._operation
        original_evaluate = q.base._evaluation

        def operation(folder, identity, name, seconds, work):
            if name.startswith("followup"):
                assert set(read_seal(folder / "selections.json")) == set(q.ARMS)
            return original(folder, identity, name, seconds, work)

        def evaluation(*args, **kwargs):
            assert (root / "results/separated_clean_r99_s0/selections.json").exists()
            return original_evaluate(*args, **kwargs)

        patch.setattr(q.base, "_operation", operation)
        patch.setattr(q.base, "_evaluation", evaluation)
        result = q.run_task(root, 0)
    assert result["status"] == "complete"
    return root, result


def test_completed_smoke_preserves_scopes_and_charges_shared_work_once(completed):
    _, result = completed
    assert len(result["followup"]) == 1
    for arm in q.ARMS:
        assert result["selections"][arm]["duration"] == 0.2
        assert result["outcomes"][arm]["forecast_adequate"]
        assert not result["evaluations"][arm]["truth_used_for_selection"]
    assert result["cost"]["stage_costs"]["training"]["operations"] == 1
    assert result["diagnostics_available"] == result["diagnostics_expected"] == 12
    assert result["predictive_support_available"] == 12
    assert result["cost"]["stage_costs"]["predictive_support"]["operations"] == 12
    for window in result["windows"] + result["followup"]:
        for entry in window["trajectories"]:
            assert not entry["predictive_support"]["support"][
                "future_observations_used"
            ]
            for attempt in entry["attempts"]:
                assert (
                    attempt["inference"]["shared_parameters"]
                    == (result["training_fit"]["shared_parameters"])
                )


def snapshot(root: Path):
    return {
        str(f): (f.read_bytes(), f.stat().st_mtime_ns) for f in root.rglob("*.json")
    }


def test_exact_resume_has_no_new_search_or_diagnostic(completed, monkeypatch):
    root, result = completed
    before = snapshot(root)

    def forbidden(*a, **kw):
        pytest.fail("repeated numerical work")

    monkeypatch.setattr(q.base.execute, "fit_training", forbidden)
    monkeypatch.setattr(q.base.execute, "infer_prefix", forbidden)
    monkeypatch.setattr(q.base.verification, "forecast", forbidden)
    monkeypatch.setattr(q.support, "predictive_support", forbidden)
    assert q.run_task(root, 0) == result
    assert snapshot(root) == before
    assert q.report(root)["recorded"] == 1


def test_changed_retained_score_rejected(completed):
    root, _ = completed
    path = root / "results/separated_clean_r99_s0/result.json"
    raw = path.read_bytes()
    changed = deepcopy(read_seal(path))
    changed["outcomes"]["baseline"]["followup"]["scores"]["common"]["mean_nmse"] = 1.0
    path.unlink()
    seal(path, changed)
    try:
        with pytest.raises(ValueError, match="terminal artifact"):
            q.report(root)
    finally:
        path.write_bytes(raw)


def test_support_receipt_transplant_rejected(completed):
    root, result = completed
    plan, data = q.verify(root)
    identity = {"plan_sha256": q.public.content_sha256(plan), "task": plan["tasks"][0]}
    operations = deepcopy(result["operations"])
    name = "validation_w1_val_0_support"
    operations[name]["value"]["support"]["horizon_times_sha256"] = "different"
    with pytest.raises(ValueError, match="support identity"):
        q._assemble(
            data["records"][plan["tasks"][0]["task_id"]],
            identity,
            operations,
            small_policy(),
            result["selections"],
        )


def test_extra_evaluator_arm_rejected(completed):
    root, _ = completed
    path = root / "results/separated_clean_r99_s0/result.json"
    raw = path.read_bytes()
    changed = deepcopy(read_seal(path))
    changed["evaluations"]["unexpected"] = {}
    path.unlink()
    seal(path, changed)
    try:
        with pytest.raises(ValueError, match="evaluator roster"):
            q.report(root)
    finally:
        path.write_bytes(raw)
