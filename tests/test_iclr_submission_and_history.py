"""Historical split, source completeness, and prospective scientific diagnostics."""

from __future__ import annotations

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import iclr_cstr_correction as correction
from scripts import iclr_search_history as history
from scripts import iclr_specification_results as spec
from scripts import iclr_submission_evidence as audit


def statistic_row(task, prompt, score, step=5):
    return {
        "task_id": task,
        "round": step,
        "arm": prompt,
        "cell": "case",
        "retained_validation_nmse": score,
    }


def test_coverage_and_macro_are_over_expected_runs():
    primary = [
        statistic_row(f"c_seed{seed}_full_c1v1s1", "full", n)
        for seed, n in enumerate((0.4, None))
    ]
    out = audit.per_case(primary)
    assert all(
        row["available"] == 1 and row["planned"] == 2 and row["median"] == 0.4
        for row in out
    )
    assert audit.center([]) == {"median": None, "mad": None}
    with pytest.raises(ValueError, match="duplicate"):
        audit.per_case([*primary, primary[0]])


def visit(i, selected, trial, cert):
    return {
        "task_id": "sample_c1v1s1",
        "round": i,
        "result_sha256": str(i),
        "cell": "case",
        "seed": 0,
        "arm": "full",
        "status": "complete",
        "selected_origin_round": 0,
        "retained_train_nmse": 0.1,
        "retained_validation_nmse": selected,
        "all_graph_requirements_certified": cert,
        "trial_validation_nmse": trial,
        "cumulative_requests": i + 2,
        "cumulative_tokens": 80 * (i + 1),
        "cumulative_fit_attempts": i + 1,
        "provider_calls_with_unknown_usage": 0,
    }


def test_joint_incumbent_and_independent_envelopes():
    rows = [visit(0, 0.4, 0.4, False), visit(1, 0.5, 0.2, True)]
    f = {("sample_c1v1s1", 0, "0"): 0.75, ("sample_c1v1s1", 1, "1"): 0.25}
    r = history.learning(rows, f)[1]
    assert [
        r[k]
        for k in (
            "selected_validation_nmse",
            "selected_functional_fraction",
            "best_validation_seen",
            "best_retained_functional_seen",
        )
    ] == [0.5, 0.25, 0.2, 0.75]
    with pytest.raises(ValueError, match="missing or duplicate"):
        history.learning([rows[1]], {})


def test_witness_requires_same_preparation_and_other_inputs():
    def tr(name, u, y, target0=10, feed=2):
        return {
            "trajectory_id": name,
            "time": [0, 1, 2],
            "fixed_covariates": {},
            "auxiliaries": {},
            "targets": {"T": [target0, *y]},
            "external_inputs": {"Tf": [feed] * 3, "Tjf": u},
        }

    control = tr("a", [1, 1, 1], [10, 10])
    forced = tr("b", [1, 2, 2], [11, 12])
    (row,) = spec.contradiction_pairs([control, forced], "Tjf")
    assert row["violates_invariance"] and row[
        "pair_mean_mse_lower_bound_K2"
    ] == pytest.approx(5 / 12)
    assert spec.contradiction_pairs([control, forced], None) == []
    for other in (
        tr("c", [2, 2, 2], [11, 12]),
        tr("d", [1, 2, 2], [11, 12], 11),
        tr("e", [1, 2, 2], [11, 12], feed=3),
    ):
        assert spec.contradiction_pairs([control, other], "Tjf") == []


def test_correction_refuses_to_rewrite_existing_sources(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        correction.stage(tmp_path, tmp_path / "corrected", tmp_path / "public")


def test_local_functional_join_rejects_cross_visit_scores(tmp_path):
    row = {"task": "sample_c1v1s1", "round": 0, "result_sha256": "wrong", "job": None}
    sealed_write(
        tmp_path / "plan.json",
        {"protocol": "iclr-retained-functional-1", "plan": audit.PLAN, "rows": [row]},
    )
    with pytest.raises(ValueError, match="another visit"):
        history.assessed_visits(tmp_path, [visit(0, 0.4, 0.4, False)])


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "wrong_metrics",
        "wrong_subject",
        "partial_search",
        "wrong_mode",
        "other_cohort",
    ],
)
def test_spec_confirmation_join_binds_rows_and_source_identity(
    tmp_path, monkeypatch, fault
):
    monkeypatch.setattr(spec, "development", lambda _: {"restricted": []})
    search = tmp_path / "search"
    variants = sealed_write(search / "spec_variants.json", {"variants": {}})
    task = {"task_id": "one", "cell": "sensor_00", "treatment": "full", "seed": 0}
    tasks = (
        [task, {**task, "task_id": "missing"}] if fault == "partial_search" else [task]
    )
    plan = sealed_write(
        search / "plan.json",
        {"variants_sha256": variants["artifact_sha256"], "blocks": [{"tasks": tasks}]},
    )
    r = {
        "source_id": "one",
        "status": "ready",
        "validation_nmse": 0.2,
        "benchmark_id": "sensor_00",
        "method": "full",
        "repetition": 0,
    }
    freeze = sealed_write(
        search / "evaluation_freeze.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "roster_sha256": public.content_sha256([r]),
        },
    )
    m = sealed_write(
        search / "frozen_models.json",
        {
            "protocol": "iclr-controlled-search-models-1",
            "plan_sha256": plan["artifact_sha256"],
            "selection_freeze_sha256": freeze["artifact_sha256"],
            "rows": [r],
        },
    )
    c = tmp_path / "confirmation"
    model = {
        **r,
        "benchmark_id": "corrected-cstr-hard",
        "source_benchmark_id": "sensor_00",
        "source_bundle_sha256": m["artifact_sha256"],
    }
    cohort = sealed_write(
        c / "frozen_selection_manifest.json",
        {"protocol": spec.FIDELITY_VERSION, "mode": "confirmation", "roster": [model]},
    )
    score = {
        k: model[k]
        for k in (
            "source_id",
            "benchmark_id",
            "source_benchmark_id",
            "source_bundle_sha256",
            "method",
            "repetition",
        )
    }
    score.update(subject=0, effect_rse=0.5, status="complete")
    if fault == "wrong_subject":
        score["source_id"] = "other"
    scores = sealed_write(
        c / "scored_rows.json",
        {
            "protocol": spec.FIDELITY_VERSION,
            "plan_sha256": cohort["artifact_sha256"],
            "mode": "development" if fault == "wrong_mode" else "confirmation",
            "source_bundle_hashes": [
                "other" if fault == "other_cohort" else m["artifact_sha256"]
            ],
            "metrics_sha256": "wrong"
            if fault == "wrong_metrics"
            else public.content_sha256([score]),
            "rows": [score],
        },
    )
    # The separate sealed numerical payload is authoritative; compatibility fields
    # in the outer summary need not contain copies of per-quantity results.
    sealed_write(
        c / "summary.json",
        {
            "protocol": spec.FIDELITY_VERSION,
            "plan_sha256": cohort["artifact_sha256"],
            "source_mode": "confirmation",
            "scored_rows_sha256": scores["artifact_sha256"],
            "metrics_sha256": scores["metrics_sha256"],
            "rows": 1,
            "row_count": 1,
        },
    )
    if fault != "none":
        with pytest.raises(ValueError):
            spec.report(search, tmp_path, tmp_path / "report", c)
        return
    result = spec.report(search, tmp_path, tmp_path / "report", c)
    assert result["outcomes"][0]["independent_outcomes"] == 1
    assert "0.5" in (tmp_path / "report/independent_spec_effects.csv").read_text()


def test_corrected_assets_preserve_unaffected_files_and_bind_the_original(
    tmp_path, monkeypatch
):
    import json
    from hashlib import sha256

    original = tmp_path / "campaign"
    corrected = tmp_path / "corrected"
    corrected.mkdir()
    old_cells = {}
    names = ["a", "b", "c", "d", correction.CSTR]
    for cell in names:
        folder = original / "public/phase_b_v1" / cell
        folder.mkdir(parents=True)
        assets = {}
        for n in correction.NAMES:
            (folder / n).write_text(n)
            assets[n] = sha256(n.encode()).hexdigest()
        old_cells[cell] = {"assets": assets}
    old = sealed_write(
        original / "plan.json", {"config": {"public_cells": names}, "cells": old_cells}
    )
    monkeypatch.setattr(correction, "PLAN", old["artifact_sha256"])
    wanted = {}
    for n in correction.NAMES:
        data = "corrected " + n if n.endswith("csv") else n
        if n == "manifest.json":
            data = json.dumps(
                {
                    "benchmark_id": correction.CSTR,
                    "reference_generation_protocol": "reference-events-2",
                    "private_reference_available_to_methods": False,
                }
            )
        (corrected / n).write_text(data)
        if n != "manifest.json":
            wanted[n] = sha256(data.encode()).hexdigest()
    monkeypatch.setattr(correction, "EXPECTED", wanted)
    output = tmp_path / "new"
    result = correction.stage(original, corrected, output)
    assert result["planned_search_lineages"] == 16
    assert (
        output / "phase_b_v1" / correction.CSTR / "train.csv"
    ).read_text() == "corrected train.csv"
    assert (output / "phase_b_v1/a/train.csv").read_text() == "train.csv"
    assert correction.stage(original, corrected, output) == result
    (original / "public/phase_b_v1" / correction.CSTR / "train.csv").write_text(
        "changed"
    )
    with pytest.raises(ValueError, match="original campaign assets"):
        correction.stage(original, corrected, output)
