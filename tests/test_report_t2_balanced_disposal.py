"""Selection and validation boundaries for the matched disposal experiment."""

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import report_t2_balanced_disposal as report
from scripts.run_hidden_form_forecasts import rotations


def scored(value):
    return {
        "status": "scored",
        "nmse": value,
        "per_target_nmse": dict.fromkeys(report.TARGETS, value),
    }


def panel():
    return {
        "panel_id": "graph02-fold0",
        "graph_name": "graph02",
        "fit_ids": [f"train_{i:03}" for i in range(8)],
        "choice_rotations": rotations(
            [f"train_{i:03}" for i in range(8)],
            [f"train_{i:03}" for i in range(8, 16)],
        ),
        "native_family": "free_action",
        "candidates": {
            f"{arm}_{law}": {"arm": arm, "law": law, "dependency_eligible": True}
            for arm in ("free", "balanced")
            for law in report.LAWS
        },
    }


def test_select_all_four_use_worst_event_keep_tie_order_require_valid():
    names = ["a", "b", "c", "d"]
    metrics = {
        f"train_{i:03}": {
            "a": {"nmse": 7 if i == 3 else 0},
            "b": {"nmse": 1},
            "c": {"nmse": 1},
            "d": {"nmse": None if i == 1 else 0},
        }
        for i in range(4)
    }
    assert report.choose(names, metrics) == {
        "family": "b",
        "selection_nmse": {"a": 1.75, "b": 1, "c": 1},
    }
    assert report.choose(["d"], metrics)["family"] is None
    with pytest.raises(ValueError, match="four training"):
        report.choose(names, {**metrics, "val_000": metrics["train_001"]})
    with pytest.raises(ValueError, match="three-target"):
        report.event_metrics({**scored(1), "nmse": 0})


def test_only_eligible_families_compete_original_control_preserved():
    p = panel()
    for arm in ("free", "balanced"):
        p["candidates"][arm + "_action"]["dependency_eligible"] = False
    choices = report.menu(p)
    assert "free_action" not in choices["free"]
    assert choices["free_native"] == ["free_action"]
    del p["candidates"]["balanced_both"]
    with pytest.raises(ValueError, match="matched pair"):
        report.menu(p)


def test_score_whole_rotations_and_show_abstentions_and_events():
    p = panel()
    decision = {
        "artifact_sha256": "selected",
        "choices": {
            policy: {"family": "A" if policy.startswith("free") else "B"}
            for policy in report.POLICIES
        },
    }
    good = {str(i): {"A": scored(2), "B": scored(1)} for i in range(4)}
    bad = {**good, "0": {"A": scored(2), "B": {"status": "negative_forecast"}}}
    a = report._rotation(p, 0, decision, good)
    b = report._rotation(p, 1, decision, bad)
    summary = report.summaries([a, b])["free->balanced"]
    assert summary["coverage"] == {"free": 2, "balanced": 1}
    assert summary["common_rotations"] == 1
    assert summary["paired_event_comparisons"] == 7
    assert summary["wins"] == 1
    assert summary["means"]["free"]["I"] == 2
    assert summary["means"]["balanced"]["U"] == 1


def test_choices_sealed_and_resume_without_opening_external(tmp_path):
    root = tmp_path
    p = panel()
    fitp = sealed_write(root / "plan.json", {"panels": [p]})
    assert report.selection_plan(root)["fit_plan"] == fitp["artifact_sha256"]
    for f in p["candidates"]:
        marker = {
            "plan": fitp["artifact_sha256"],
            "panel_id": p["panel_id"],
            "family": f,
        }
        fit = sealed_write(root / "fits" / p["panel_id"] / f / "best.json", marker)
        for i in range(8, 16):
            eid = f"train_{i:03}"
            sealed_write(
                root / "curves" / p["panel_id"] / f / f"{eid}.json",
                {**marker, "fit": fit["artifact_sha256"], "event_id": eid, **scored(1)},
            )
    assert report.select(root) == report.select(root)
    inner = report.score(root, root, official=False)
    assert inner["comparisons"]["free->balanced"]["ties"] == 2
    with pytest.raises(ValueError, match="before accessing validation"):
        report.score(root, root, official=True)
    manifest = sealed_read(root / "choices.json")
    assert report.external_plan(root)["choices"] == manifest["artifact_sha256"]


def test_diagnostic_order_is_irrelevant_on_resume_actual_failures_must_match(tmp_path):
    path = tmp_path / "summary.json"
    a = {
        "panel": "A",
        "family": "x",
        "event": "train_001",
        "status": "negative_forecast",
    }
    b = {**a, "family": "y"}
    old = sealed_write(path, {"failures": [b, a], "negative_disposal_forecasts": []})
    assert (
        report._write_scores(
            path, {"failures": [a, b], "negative_disposal_forecasts": []}
        )
        == old
    )
    with pytest.raises(ValueError, match="frozen artifact differs"):
        report._write_scores(path, {"failures": [a], "negative_disposal_forecasts": []})
