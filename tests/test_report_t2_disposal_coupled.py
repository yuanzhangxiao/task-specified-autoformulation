"""Matched coupled reports preserve trajectories and feasibility outcomes."""

from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import report_t2_disposal_coupled as report
from scripts.t2_challenge_folds import aggregate, event_metrics


def example():
    result = {}
    for model, bias in zip(report.MODELS, (0.8, 1.0, 0.5), strict=True):
        events = []
        for i in range(4):
            observed = {k: [2.0, 3.0, 4.0] for k in report.TARGETS}
            predicted = {
                k: [y + bias if k != "I" else y for y in vals]
                for k, vals in observed.items()
            }
            event = {
                "trajectory_id": f"val_{i:03}",
                "time": [0.0, 1.0, 2.0],
                "observed": observed,
                "predicted": predicted,
            }
            events.append(
                event | event_metrics(event, dict.fromkeys(report.TARGETS, 1))
            )
        result[model] = {
            "events": events,
            "summary": aggregate(events, len(events)),
            "domain": {"eligible": True},
        }
    return result


def test_matched_negative_difference_denotes_lower_error_and_frozen_insulin():
    records = example()
    records["delayed_refit"]["domain"]["eligible"] = False
    untouched = deepcopy(records)
    metrics = report.summarize(records, "val")["comparisons"]["immediate_refit"]
    assert records == untouched
    assert metrics["counts"]["I"]["tied"] == 4
    assert metrics["counts"]["U"]["improved"] == 4
    assert metrics["candidate_minus_reference_whole_cohort"]["pooled_nmse"] == -0.5
    assert not metrics["candidate_domain_eligible"]
    assert metrics["candidate_minus_reference_by_event"]["val_001"][
        "per_target_nmse"
    ] == {
        "Gp": -0.75,
        "I": 0,
        "U": -0.75,
    }


def test_incomplete_replay_is_explicit_and_wrong_split_or_ids_rejected():
    records = example()
    unavailable = {"event": "val_001", "status": "grid_unresolved"}
    r = records["immediate_refit"]
    r["events"][1] = unavailable
    r["summary"] = aggregate(r["events"], 4)
    item = report.summarize(records, "val")["comparisons"]["immediate_refit"]
    assert item["events"] == 3
    assert item["candidate_minus_reference_whole_cohort"] is None
    assert item["candidate_minus_reference_by_event"]["val_001"] == {
        "status": "unresolved"
    }
    with pytest.raises(ValueError, match="training or development"):
        report.summarize(records, "test")
    r["events"].append(unavailable)
    with pytest.raises(ValueError, match="registered split"):
        report.summarize(records, "val")


def test_sealed_figure_and_mismatched_time_grid(tmp_path):
    pytest.importorskip("matplotlib")
    records = example()
    result = report.summarize(records, "val")
    sealed_write(tmp_path / "report.json", {"freeze": "example", "report": result})
    report.plot(tmp_path / "report.json", tmp_path / "plot.png")
    assert (tmp_path / "plot.png").read_bytes().startswith(b"\x89PNG")
    records["delayed_refit"]["events"][0]["time"] = [0, 2, 3]
    with pytest.raises(ValueError, match="Inconsistent"):
        report.summarize(records, "val")
