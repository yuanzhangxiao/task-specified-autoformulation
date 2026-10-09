"""Evaluator-only coefficient correctness, rank, mesh bias and source separation."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import conditional_diagnostics as diagnostic
from autoformalism.fitting import nonlinear_comparison as fitting
from autoformalism.fitting import nonlinear_comparison_campaign as campaign
from autoformalism.fitting import public_fitting as public
from tests.profiled_fixture import TRUTH
from tests.test_nonlinear_comparison import exported as exported
from tests.test_trajectory_profile import setup


def test_exact_derivative_coefficients_recover_and_fixed_reference_nodes_refine():
    _, _, _, _, graph = setup()
    z, samples = diagnostic.reference_nodes(graph, TRUTH, monotonic() + 15)
    result = diagnostic.derivative_diagnostic(graph, TRUTH, samples)
    assert result["correctness_passed"] and result["coefficient_recovery_certified"]
    assert result["rank"] == 4 and result["maximum_relative_error"] < 1e-9
    q = (graph.oracle.vector(TRUTH) - graph.pcenter) / graph.punits
    # Dynamic derivatives are exact, whereas finite-mesh Radau defects are not.
    metrics = np.asarray(graph.components(z, q)).ravel()
    assert metrics[1] > 0


def test_rank_deficiency_is_not_a_recovery_certificate():
    a = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
    report = diagnostic.bounded_diagnostic(
        a, np.array([3.0, 6.0]), np.zeros(3), np.full(3, 10.0)
    )
    assert report["rank"] == 1 and not report["full_rank"]
    assert report["zero_columns"] == [2] and report["residual_relative_norm"] < 1e-12
    with pytest.raises(ValueError, match="nonfinite"):
        diagnostic.bounded_diagnostic(a * np.nan, [1, 2], np.zeros(3), np.ones(3))


def test_three_diagnostic_cases_do_not_mutate_original_start():
    problem, _, _, _, graph = setup()
    original = deepcopy(problem.model_dump())
    z, q = graph.guesses(problem.start)
    saved = {
        "level": 0,
        "z": z.tolist(),
        "q": q.tolist(),
        "checkpoint": {"parameters": graph.parameters(q)},
    }
    policy = fitting.ComparisonPolicy(
        node_targets=(100,),
        penalties=(100,),
        minimum_intervals=4,
        observation_anchors=2,
    )
    r = diagnostic.diagnose_checkpoint(
        problem, TRUTH, saved, policy, monotonic() + 20, lambda _: None
    )
    assert r["exact_derivative"]["correctness_passed"]
    assert [c["case"] for c in r["cases"]] == [
        "reference_nodes_reference_shapes",
        "estimated_nodes_reference_shapes",
        "estimated_nodes_estimated_shapes",
    ]
    assert not r["reference_available_to_fitter"] and not r["validation_opened"]
    assert problem.model_dump() == original
    saved["q"].pop()
    with pytest.raises(ValueError, match="layout"):
        diagnostic.diagnose_checkpoint(
            problem, TRUTH, saved, policy, monotonic() + 20, lambda _: None
        )


@pytest.mark.usefixtures("exported")
def test_measured_campaign_requires_matching_diagnostic_and_resume_is_sealed(
    tmp_path, monkeypatch
):
    inputs = tmp_path / "inputs.json"
    source = {
        "protocol": diagnostic.PROTOCOL,
        "source_plan_sha256": diagnostic.SOURCE_PLAN,
        "inputs_sha256": public.content_sha256(read_seal(inputs)),
        "rows": [
            {"common": f"alien_hard_s{i}", "level": j}
            for i in range(3)
            for j in range(2)
        ],
    }
    seal(tmp_path / "conditional-diagnostic-inputs.json", source)
    root = tmp_path / "campaign"
    with pytest.raises(ValueError, match="frozen M22 mesh"):
        campaign.prepare(
            root,
            inputs,
            fitting.ComparisonPolicy(allocation="measured", observation_anchors=7),
        )
    campaign.prepare(root, inputs, fitting.ComparisonPolicy(allocation="measured"))
    assert campaign.report(root)["protocol"] == campaign.ALLOCATION_PROTOCOL
    with pytest.raises(FileNotFoundError):
        campaign.run_task(root, 0)
    monkeypatch.setattr(
        diagnostic,
        "diagnose_checkpoint",
        lambda *a: {"exact_derivative": {"correctness_passed": True}},
    )
    # No optimizer input is extended with evaluator fields.
    assert all(
        set(p.model_dump())
        == {"request", "training", "coordinates", "start", "incumbent"}
        for p in campaign.bases(read_seal(inputs)).values()
    )
    data = read_seal(root / "inputs.json")
    # The fixture has no reference case. Run's reference access is separately
    # exercised by the numerical test above; forge no benchmark truths here.
    assert "case" not in data
    plan = read_seal(root / "plan.json")
    seal(
        root / "diagnostics/result.json",
        {"plan_sha256": "wrong", "correctness_passed": True},
    )
    with pytest.raises(ValueError, match="matching coefficient"):
        campaign.run_task(root, 0)
    assert plan["diagnostic_inputs_sha256"] == public.content_sha256(source)


@pytest.mark.usefixtures("exported")
def test_m23_cpu_submission_includes_diagnostic_dependency(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from scripts import submit_phase_c_fitting_allocation as submit
    from scripts import submit_phase_c_generic_recovery as scheduler

    inputs = tmp_path / "inputs.json"
    seal(
        tmp_path / "conditional-diagnostic-inputs.json",
        {
            "protocol": diagnostic.PROTOCOL,
            "source_plan_sha256": diagnostic.SOURCE_PLAN,
            "inputs_sha256": public.content_sha256(read_seal(inputs)),
            "rows": [
                {"common": f"alien_hard_s{i}", "level": j}
                for i in range(3)
                for j in range(2)
            ],
        },
    )
    config = tmp_path / "config.json"
    config.write_text(fitting.ComparisonPolicy(allocation="measured").model_dump_json())
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(
        scheduler.subprocess, "check_output", lambda *a, **k: "revision\n"
    )
    calls = []

    def sbatch(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(scheduler.subprocess, "run", sbatch)
    args = {
        "root": tmp_path / "run",
        "config": config,
        "inputs": inputs,
        "account": "bibo-delta-cpu",
        "concurrency": 3,
    }
    a = submit.submit(**args)
    assert submit.submit(**args) == a and len(calls) == 3
    assert "--time=00:30:00" in calls[0] and "--dependency=afterok:101" in calls[1]
    assert "--array=0-5%3" in calls[1] and a["gpus"] == 0
