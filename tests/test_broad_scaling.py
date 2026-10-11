"""Broader scaling: immutable roster, training boundary, domains and safe resume."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import broad_scaling_campaign as campaign
from autoformalism.fitting import broad_scaling_inputs as inputs
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import nonlinear_rollout
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.coordinates import training_coordinates
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit
from tests.profiled_fixture import TRUTH, make_base


@pytest.fixture
def data():
    base, validation = make_base()
    cases, commons = {}, {}
    for name in inputs.CASES:
        case = {
            "request": base["request"],
            "training": base["training"],
            "validation": validation,
            "reference_parameters": TRUTH,
        }
        cases[name] = case
        for seed in range(inputs.STARTS):
            commons[f"{name}_s{seed}"] = inputs.common(name, case, seed)
    return {
        "protocol": inputs.PROTOCOL,
        "cases": cases,
        "commons": commons,
        "test_data_opened": False,
        "evaluation_thresholds": {
            "cstr_nmse": 1e-8,
            "parameter_relative": 0.01,
            "initial_absolute": 1e-3,
        },
    }


def test_training_boundary_predetermined_distinct_starts_and_paired_roster(data):
    original = inputs.bases(data)
    for case in data["cases"].values():
        case["validation"] = "forbidden"
        case["reference_parameters"] = "forbidden"
    assert inputs.bases(data) == original
    starts = campaign.portfolios(data, campaign.Policy())
    for key, problem in original.items():
        assert "reference_parameters" not in problem.model_dump()
        assert problem.incumbent == problem.start
        assert starts[key][0] == problem.start
        assert len({public.content_sha256(p) for p in starts[key]}) == 4
        oracle = checks.Rollouts(problem, checks.ConfidencePolicy())
        for point in starts[key]:
            oracle.vector(point)
    assert len(campaign.roster(data)) == 45
    assert len({public.content_sha256(p.start) for p in original.values()}) >= 10
    for name in inputs.CASES:
        for seed in range(inputs.STARTS):
            safe = {k: data["cases"][name][k] for k in ("request", "training")}
            assert inputs.common(name, safe, seed) == data["commons"][f"{name}_s{seed}"]


def piecewise_problem():
    """Tiny coupled model with crossings; no reference or benchmark simulation."""
    base, _ = make_base()
    payload = base["request"]
    payload["base_candidate"]["state_equations"] = [
        {"state": "y", "rhs": "-b*y+c*max(z,0)-d*u"},
        {"state": "z", "rhs": "-a*z+u+0.1*y"},
    ]
    request = PublicFitRequest.model_validate(payload)
    model, start, _ = public._lower(request)
    train = public.unpack_split(PublicSplit.model_validate(base["training"]))
    return checks.PiecewiseTrainingProblem(
        request=request,
        training=base["training"],
        start=start,
        incumbent=start,
        coordinates=training_coordinates(model, train, start),
    )


def test_piecewise_is_explicit_and_joint_derivatives_match_rollout():
    p = piecewise_problem()
    plain = checks.TrainingProblem.model_validate(
        p.model_dump(exclude={"allow_piecewise"})
    )
    with pytest.raises(ValueError, match="nonsmooth"):
        checks.Rollouts(plain, checks.ConfidencePolicy())
    oracle, profile, route = nonlinear_rollout.build(p)
    assert profile is None and route["selected"] == "joint_rollout"
    assert oracle.system.has_piecewise
    point = oracle.vector(p.start)
    residual, jac = oracle.evaluate(point, monotonic() + 20)
    direction = np.linspace(0.1, 0.5, len(point))
    h = 1e-4
    plus, _ = oracle.evaluate(point + h * direction, monotonic() + 20, jacobian=False)
    minus, _ = oracle.evaluate(point - h * direction, monotonic() + 20, jacobian=False)
    np.testing.assert_allclose(
        jac @ direction, (plus - minus) / (2 * h), rtol=2e-3, atol=2e-5
    )
    assert np.isfinite(residual).all()
    check = nonlinear_rollout.verify(oracle, p.start, monotonic() + 20, lambda _: None)
    assert check["usable_for_retention"]


@pytest.fixture
def prepared(data, tmp_path):
    path = tmp_path / "inputs.json"
    seal(path, data)
    root = tmp_path / "campaign"
    campaign.prepare(root, path, campaign.Policy())
    return root


def fake_backend(problem, starts, policy, method, folder):
    result = dict.fromkeys(
        (
            "search_calls_started",
            "search_calls_completed",
            "maximum_search_calls",
            "search_budget_charge_seconds",
            "additional_wall_seconds",
            "additional_cpu_seconds",
            "additional_budget_charge_seconds",
            "blocks_used",
        ),
        0,
    )
    return result | {
        "status": "complete",
        "assessment": {},
        "accuracy_verified": False,
        "decisions": [],
        "cost_complete": True,
        "search_call_accounting_complete": True,
        "retained_parameters": problem.start,
        "operations": [],
        "routing": {"selected": "joint_rollout"},
    }


def test_backend_sealed_before_scoring_exact_resume_and_failure_denominator(
    prepared, monkeypatch
):
    seen = []

    def run(problem, starts, policy, method, folder):
        seen.append((starts, policy.model_dump(), method))
        return fake_backend(problem, starts, policy, method, folder)

    def evaluate(folder, data, common, parameters, seconds):
        assert (folder.parent / "backend.json").exists()
        assert "validation" in data["case"]
        return {"status": "no_complete_replay", "coefficients_recovered": False}

    monkeypatch.setattr(campaign.fitting, "run", run)
    monkeypatch.setattr(campaign.screening_replay, "_evaluation", evaluate)
    rows = [campaign.run_task(prepared, i) for i in range(3)]
    assert seen[0][:2] == seen[1][:2] == seen[2][:2]
    monkeypatch.setattr(
        campaign.fitting, "run", lambda *a: pytest.fail("repeated fitting")
    )
    assert campaign.run_task(prepared, 0) == rows[0]
    report = campaign.report(prepared)
    assert report["expected"] == 45 and report["recorded"] == 3
    assert len(report["groups"]) == 9
    assert all(r["expected"] == 5 for r in report["groups"])
    assert all(r["training_accuracy_verified"] == 0 for r in report["groups"])
    with pytest.raises(ValueError, match="index"):
        campaign.run_task(prepared, 45)


def test_tampered_roster_start_data_and_policy_rejected(data, prepared, tmp_path):
    bad = deepcopy(data)
    bad["commons"]["alien_hard_s0"]["start"]["a"] += 1
    with pytest.raises(ValueError, match="start differs"):
        inputs.bases(bad)
    del bad["commons"]["alien_hard_s0"]
    with pytest.raises(ValueError, match="exactly"):
        inputs.bases(bad)
    with pytest.raises(ValueError, match="sealed artifact differs"):
        campaign.prepare(
            prepared, prepared / "inputs.json", campaign.Policy(objective_multiplier=2)
        )
    raw = public._read(prepared / "inputs.json")
    raw["value"]["test_data_opened"] = True
    public._write(prepared / "inputs.json", raw)
    with pytest.raises(ValueError, match="digest"):
        campaign.verify(prepared)


def test_export_preserves_arrays_and_old_starts_without_fitted_endpoints(
    data, monkeypatch, tmp_path
):
    old = {
        "case": data["cases"]["alien_hard"],
        "commons": {k: v for k, v in data["commons"].items() if k.startswith("alien")},
        "forbidden_fitted_endpoints": {"a": 1234},
    }
    hard = {
        "protocol": "phase-c-challenging-fitting-inputs-1",
        "test_data_opened": False,
        "cases": {k: data["cases"][k] for k in ("alien_hard", "basin_coupled")},
    }
    cstr = {
        "protocol": "phase-c-fitting-strategy-inputs-1",
        "test_data_opened": False,
        "cases": {"cstr_hard": data["cases"]["cstr_hard"]},
    }
    monkeypatch.setattr(inputs.historical, "bases", lambda v: {})
    paths = [tmp_path / n for n in ("old.json", "hard.json", "cstr.json")]
    for path, value in zip(paths, (old, hard, cstr), strict=True):
        seal(path, value)
    output = tmp_path / "export.json"
    assert inputs.export_inputs(*paths, output)["starts"] == 15
    exported = read_seal(output)
    assert "forbidden_fitted_endpoints" not in str(exported)
    for name in inputs.CASES:
        assert exported["cases"][name] == data["cases"][name]
    for seed in range(3):
        assert (
            exported["commons"][f"alien_hard_s{seed}"]
            == old["commons"][f"alien_hard_s{seed}"]
        )
    hard["test_data_opened"] = True
    seal(tmp_path / "bad-hard.json", hard)
    with pytest.raises(ValueError, match="provenance"):
        inputs.export_inputs(
            paths[0], tmp_path / "bad-hard.json", paths[2], tmp_path / "bad"
        )


def test_delta_submission_idempotence_and_uncertain_receipt(
    prepared, tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    from scripts import submit_phase_c_broad_scaling as submitter

    config = tmp_path / "config.json"
    config.write_text(campaign.Policy().model_dump_json())
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", sbatch)
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "check_output", lambda *a, **k: "test"
    )
    args = {"account": "test-cpu", "concurrency": 3, "inputs": prepared / "inputs.json"}
    first = submitter.submit(prepared, config, **args)
    assert first["array_tasks"] == 45 and first["gpus"] == 0
    assert "--array=0-44%3" in calls[1] and "--time=02:30:00" in calls[1]
    assert submitter.submit(prepared, config, **args) == first and len(calls) == 3
    monkeypatch.setattr(
        submitter.scheduler.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr="timeout"),
    )
    with pytest.raises(ValueError, match="unconfirmed"):
        submitter.submit(tmp_path / "unknown", config, **args)
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "run", lambda *a, **kw: pytest.fail("resubmit")
    )
    with pytest.raises(ValueError, match="Unconfirmed"):
        submitter.submit(tmp_path / "unknown", config, **args)
