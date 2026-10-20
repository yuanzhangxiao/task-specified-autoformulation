"""Portable M46 reconstruction permits only derived analytic roundoff."""

from copy import deepcopy
from math import inf, nan, nextafter

import pytest

pytest.importorskip("casadi")

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_multiroot_campaign as q


def _result() -> dict:
    """Represent both evaluator fields alongside values that must remain exact."""
    row = {
        "trajectory_id": "synthetic-row",
        "chosen_start": 0,
        "available": True,
        "analytic_max_absolute_difference": 8.340790280669808e-10,
        "all_start_analytic_max_absolute_differences": [
            8.340790280669808e-10,
            None,
            8.339728907458266e-10,
        ],
        "clean_scores": {"common": {"normalized_mse": 0.001}},
        "root_errors": {"sum_root_error": 0.01},
    }
    return {
        "identity": {"plan_sha256": "original-plan", "task": {"seed": 0}},
        "selection": {"status": "selected", "duration": 0.2},
        "evaluation": {
            "selection_sha256": "original-selection",
            "validation": [{"rows": [deepcopy(row)]}],
            "followup": [{"rows": [deepcopy(row)]}],
        },
        "windows": [{"metrics": {"mean_nmse": 0.001}}],
        "training_fit": {"shared_parameters": {"theta": 0.7}},
        "operations": {"training": {"identity": {"stage": "training"}}},
    }


def _leaf(value: dict, path: tuple) -> tuple[dict | list, str | int]:
    for key in path[:-1]:
        value = value[key]
    return value, path[-1]


def _path(split: str, all_starts: bool) -> tuple:
    row = ("evaluation", split, 0, "rows", 0)
    return (
        (*row, "all_start_analytic_max_absolute_differences", 0)
        if all_starts
        else (*row, "analytic_max_absolute_difference")
    )


@pytest.mark.parametrize("split", ["validation", "followup"])
@pytest.mark.parametrize("all_starts", [False, True])
@pytest.mark.parametrize("direction", [-1, 1])
def test_platform_roundoff_is_symmetric_and_preserves_inputs(
    split, all_starts, direction
):
    saved = _result()
    expected = deepcopy(saved)
    parent, key = _leaf(expected, _path(split, all_starts))
    parent[key] += direction * 4.440892098500626e-16
    before = deepcopy((saved, expected))
    assert q._portable_result_matches(saved, expected)
    assert q._portable_result_matches(expected, saved)
    assert (saved, expected) == before


@pytest.mark.parametrize("all_starts", [False, True])
def test_absolute_tolerance_boundary_has_no_relative_allowance(all_starts):
    path = _path("validation", all_starts)
    assert q._portable_result_matches(0.0, 1e-15, path)
    assert not q._portable_result_matches(0.0, nextafter(1e-15, inf), path)
    assert not q._portable_result_matches(1.0, 1.0 + 1e-12, path)
    assert not q._portable_result_matches(1e-9, 2e-9, path)


@pytest.mark.parametrize(
    "path,value",
    [
        (("identity", "plan_sha256"), "changed"),
        (("identity", "task", "seed"), False),
        (("selection", "duration"), 0.2 + 4.44e-16),
        (("evaluation", "selection_sha256"), "changed"),
        (
            (
                "evaluation",
                "validation",
                0,
                "rows",
                0,
                "clean_scores",
                "common",
                "normalized_mse",
            ),
            0.001 + 4.44e-16,
        ),
        (
            ("evaluation", "followup", 0, "rows", 0, "root_errors", "sum_root_error"),
            0.01 + 4.44e-16,
        ),
        (("windows", 0, "metrics", "mean_nmse"), 0.001 + 4.44e-16),
        (("training_fit", "shared_parameters", "theta"), 0.7 + 4.44e-16),
        (("operations", "training", "identity", "stage"), "changed"),
    ],
)
def test_scores_selections_roots_and_identities_remain_exact(path, value):
    saved = _result()
    expected = deepcopy(saved)
    parent, key = _leaf(saved, path)
    parent[key] = value
    assert not q._portable_result_matches(saved, expected)


@pytest.mark.parametrize("all_starts", [False, True])
@pytest.mark.parametrize("value", [nan, inf, -inf, -1e-18, True, 0, "0.0", [], {}])
def test_analytic_fields_reject_nonfinite_negative_and_malformed_values(
    all_starts, value
):
    path = _path("validation", all_starts)
    assert not q._portable_result_matches(value, value, path)
    assert not q._portable_result_matches(value, 0.0, path)
    assert not q._portable_result_matches(0.0, value, path)


@pytest.mark.parametrize("all_starts", [False, True])
def test_missing_analytic_values_must_match_exactly(all_starts):
    path = _path("validation", all_starts)
    assert q._portable_result_matches(None, None, path)
    assert not q._portable_result_matches(None, 0.0, path)
    assert not q._portable_result_matches(0.0, None, path)


@pytest.mark.parametrize("value", [nan, inf, -inf])
def test_nonfinite_values_rejected_outside_tolerated_paths(value):
    assert not q._portable_result_matches({"score": value}, {"score": value})


@pytest.mark.parametrize(
    "path",
    [
        ("analytic_max_absolute_difference",),
        ("evaluation", "training", 0, "rows", 0, "analytic_max_absolute_difference"),
        (
            "evaluation",
            "validation",
            "0",
            "rows",
            0,
            "analytic_max_absolute_difference",
        ),
        (
            "evaluation",
            "validation",
            True,
            "rows",
            0,
            "analytic_max_absolute_difference",
        ),
        ("evaluation", "validation", -1, "rows", 0, "analytic_max_absolute_difference"),
        (
            "evaluation",
            "validation",
            0,
            "rows",
            "0",
            "analytic_max_absolute_difference",
        ),
        (
            "evaluation",
            "validation",
            0,
            "rows",
            0,
            "analytic_max_absolute_difference",
            0,
        ),
        (
            "evaluation",
            "validation",
            0,
            "rows",
            0,
            "all_start_analytic_max_absolute_differences",
            "0",
        ),
    ],
)
def test_similar_field_names_at_other_paths_never_receive_tolerance(path):
    assert not q._portable_result_matches(0.0, 4.44e-16, path)


@pytest.mark.parametrize("change", ["missing", "extra", "list_length", "container"])
def test_structure_drift_is_rejected(change):
    saved = _result()
    expected = deepcopy(saved)
    row = saved["evaluation"]["validation"][0]["rows"][0]
    if change == "missing":
        row.pop("analytic_max_absolute_difference")
    elif change == "extra":
        row["extra"] = 0
    elif change == "list_length":
        row["all_start_analytic_max_absolute_differences"].append(0.0)
    else:
        row["all_start_analytic_max_absolute_differences"] = tuple(
            row["all_start_analytic_max_absolute_differences"]
        )
    assert not q._portable_result_matches(saved, expected)


def test_completed_preserves_sealed_result_bytes_and_rejects_resealed_drift(
    tmp_path, monkeypatch
):
    saved = _result()
    expected = deepcopy(saved)
    parent, key = _leaf(expected, _path("followup", False))
    parent[key] += 4.44e-16
    seal(tmp_path / "selection.json", saved["selection"])
    seal(tmp_path / "result.json", saved)
    monkeypatch.setattr(q, "_read_operations", lambda *args: {})
    monkeypatch.setattr(q, "_assemble", lambda *args: expected)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.glob("*.json")
    }
    for _ in range(2):
        assert q._completed(tmp_path, saved["identity"], {}, q.Policy()) == saved
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.glob("*.json")
    }

    changed = read_seal(tmp_path / "result.json")
    changed["selection"]["duration"] += 4.44e-16
    (tmp_path / "result.json").unlink()
    seal(tmp_path / "result.json", changed)
    with pytest.raises(ValueError, match="terminal artifact differs"):
        q._completed(tmp_path, saved["identity"], {}, q.Policy())


def test_completed_still_checks_the_original_seal(tmp_path, monkeypatch):
    saved = _result()
    seal(tmp_path / "selection.json", saved["selection"])
    path = tmp_path / "result.json"
    seal(path, saved)
    payload = q.public._read(path)
    payload["value"]["identity"]["plan_sha256"] = "tampered"
    q.public._write(path, payload)
    monkeypatch.setattr(q, "_read_operations", lambda *args: {})
    monkeypatch.setattr(
        q, "_assemble", lambda *args: pytest.fail("unchecked seal reached assembly")
    )
    with pytest.raises(ValueError, match="sealed artifact digest differs"):
        q._completed(tmp_path, saved["identity"], {}, q.Policy())
