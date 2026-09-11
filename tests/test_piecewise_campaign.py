"""Campaign reference independence, immutable resume and checkpoint-only summary."""

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.rebuttal import piecewise_campaign as campaign
from autoformalism.rebuttal.fitter_diagnostic import write_json
from autoformalism.schemas import CandidateModel


@pytest.mark.parametrize("case", campaign.CASES)
def test_reference_matches_production_and_has_no_hidden_labels(case):
    problem, reference = campaign.synthetic_problem(case, 0.0, 1)
    model = compile_candidate(
        CandidateModel.model_validate(problem["candidate"]),
        ValidationContext.model_validate(problem["context"]),
    )
    for name in ("train", "val"):
        split = campaign.unpack_split(problem["splits"][name])
        assert campaign.pack_split(split) == problem["splits"][name]
        for row in split.trajectories:
            assert not row.derivatives and tuple(row.targets) == ("v01",)
            result = simulate_trajectory(
                model,
                row,
                {"b": 10.0} if case == "fitted_threshold" else {"a": 1.0},
                {},
                FitConfig(
                    integration_method="Radau",
                    relative_tolerance=1e-10,
                    absolute_tolerance=1e-12,
                ),
                reset_observed_states=False,
            )
            assert result.success
            np.testing.assert_allclose(
                result.predictions["v01"], reference[name][row.trajectory_id], atol=2e-7
            )


def test_noise_and_data_do_not_depend_on_start():
    a, clean_a = campaign.synthetic_problem("fitted_threshold", 0.03, 0)
    b, clean_b = campaign.synthetic_problem("fitted_threshold", 0.03, 2)
    assert a["splits"] == b["splits"] and clean_a == clean_b
    assert a["start"] != b["start"]
    with pytest.raises(ValueError, match="test"):
        campaign.unpack_split({"name": "test", "rows": [], "fingerprint": "x"})


def test_prepare_resume_and_summary_never_fit(tmp_path, monkeypatch):
    plan = campaign.PiecewisePlan()
    frozen = campaign.prepare(plan, tmp_path)
    assert len(frozen["cases"]) == 24
    assert campaign.prepare(plan, tmp_path) == frozen

    def forbidden(*args, **kwargs):
        raise AssertionError("summary must not solve or fit")

    monkeypatch.setattr(campaign, "simulate_trajectory", forbidden)
    monkeypatch.setattr(campaign, "fit_collocation_forward_sensitivity", forbidden)
    report = campaign.summarize(tmp_path)
    assert len(report["rows"]) == 96
    assert {row["status"] for row in report["rows"]} == {"missing"}
    assert not report["test_data_opened"]
    write_json(tmp_path / "problems/000.json", {"tampered": True})
    with pytest.raises(ValueError, match="frozen input"):
        campaign.verify(tmp_path)


def test_bundle_traversal_and_worker_budget_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        campaign.safe_path(tmp_path, "../private.json")
    with pytest.raises(ValueError, match="scheduler"):
        campaign.PiecewisePlan.model_validate({"fit": {"refinement_seconds": 900}})
