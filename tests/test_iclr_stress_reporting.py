"""Exercise measured, unavailable and pending stress-study reports."""

import csv

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import iclr_stress_evaluation as stress


def read(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def test_terminal_solver_failures_remain_classified():
    def reject(*args):
        raise stress.SolverFailure("Radau rejected integration")

    def disagree(*args):
        raise stress.SolverDisagreement("BDF differs")

    for run, kind in [(reject, "solver_failure"), (disagree, "solver_disagreement")]:
        record = stress.score({}, {}, [{"case": "pulse"}], {}, predictor=run)
        assert record["test_nmse"] is None
        assert record["trajectories"][0]["failure_kind"] == kind
        counts = stress.numerical_coverage(record | {"search_status": "ready"})
        assert counts["solver_classified_pairs"] == counts[kind + "_pairs"] == 1
        assert counts["optimizer_status"] == "not_reported_in_endpoint"


def test_measured_fixed_dimension_cell_and_unavailable_pair(tmp_path):
    cells = [
        {
            "cell": "d8base",
            "study": "scaling",
            "dimension": 8,
            "generator_seed": 27181,
            "variant": "dimension",
        },
        {
            "cell": "d8double",
            "study": "scaling",
            "dimension": 8,
            "generator_seed": 27181,
            "variant": "double_recurrent_coupling",
        },
    ]
    runs = [
        {"benchmark_id": c["cell"], "arm": a, "repetition": s}
        for c in cells
        for a in ("full", "fixed_gamma")
        for s in (0, 1)
    ]
    m = {"artifact_sha256": "design", "cells": cells, "runs": runs}
    records = []
    for i, r in enumerate(runs):
        meta = {
            "benchmark_id": r["benchmark_id"],
            "method": r["arm"],
            "repetition": r["repetition"],
        }
        if i == 0:
            records.append(
                meta
                | {
                    "search_status": "no_retained_model",
                    "evaluation_status": "model_unavailable",
                    "last_visit_status": "fit_failed",
                    "complexity_status": "no_retained_model",
                }
            )
        elif i == 1:
            records.append(
                meta
                | {
                    "search_status": "ready",
                    "evaluation_status": "unresolved",
                    "trajectories": [
                        {
                            "case": "pulse",
                            "status": "unresolved",
                            "failure_kind": "solver_disagreement",
                            "reason": "different signed curves",
                        }
                    ],
                }
            )
        else:
            records.append(
                meta
                | {
                    "search_status": "ready",
                    "evaluation_status": "complete",
                    "test_nmse": i / 20,
                    "common_scale_test_nmse": i / 12,
                    "response_rse": i / 13,
                    "complexity_status": "complete",
                    "target_reachable_state_count": 3,
                    "expression_nodes": 16,
                    "validation_nmse": 0.5,
                    "cost": {"visit_seconds": 1.0},
                    "optimizer_converged": False,
                    "trajectories": [
                        {
                            "case": "pulse",
                            "status": "complete",
                            "channels": {
                                "v01": {
                                    "response_rse": 0.42,
                                    "reference_area": 1,
                                    "predicted_area": 2,
                                }
                            },
                        }
                    ],
                }
            )
    evaluation = tmp_path / "evaluated.json"
    sealed_write(
        evaluation,
        {
            "protocol": "iclr-stress-evaluation-1",
            "manifest_sha256": "design",
            "rows": records,
        },
    )
    out = tmp_path / "report"
    stress.report(m, [evaluation], out)
    assert (out / "fixed-dimension-sensitivity.pdf").is_file()
    assert (out / "stress-effect-curves.pdf").is_file()
    subjects = read(out / "stress_metrics.csv")
    assert subjects[0]["optimizer_status"] == "no_selected_fit"
    assert subjects[0]["expression_nodes"] == ""
    assert subjects[1]["solver_disagreement_pairs"] == "1"
    assert subjects[2]["target_reachable_state_count"] == "3"
    assert subjects[2]["optimizer_status"] == "reported_nonconvergence"
    assert subjects[1]["optimizer_status"] == "not_reported_in_endpoint"
    effects = read(out / "stress_effects.csv")
    assert effects[0]["failure_kind"] == "solver_disagreement"
    assert effects[1]["predicted_area"] == "2"
    coverage = read(out / "completion.csv")
    first = next(
        r for r in coverage if r["arm"] == "full" and r["benchmark_id"] == "d8base"
    )
    assert first["solver_disagreement_fraction_among_classified"] == "1.0"
    assert first["nonconvergence_fraction_among_reported"] == ""
    pending = tmp_path / "pending"
    stress.report(m, [], pending)
    assert not list(pending.glob("*.pdf"))
    assert (
        read(pending / "stress_metrics.csv")[0]["complexity_status"]
        == "awaiting_frozen_model"
    )
    existing = (out / "stress_metrics.csv").read_bytes()
    with pytest.raises(ValueError, match="fresh report"):
        stress.report(m, [], out)
    assert (out / "stress_metrics.csv").read_bytes() == existing
