"""Same-mesh warm starts, fair budgets and evaluator-only coefficient recovery."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import identifiable_cases as controls
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import transcription_campaign as campaign
from autoformalism.fitting import transcription_fit as fit
from autoformalism.fitting import transcription_solver as solver
from tests.test_fitting_strategies import small_payload


@pytest.fixture(scope="module")
def inputs():
    return controls.make_inputs()


def test_reuse_retains_graph_and_transfers_primal_dual(inputs, tmp_path, monkeypatch):
    payload = small_payload(inputs, "collocation")
    payload["reuse_chunks"] = 8
    payload["seconds"] = 60
    constructions, solves, solver_setups = [], [], []
    native_opti = solver.ca.Opti

    class ObservedOpti:
        def __init__(self):
            self.opti = native_opti()
            constructions.append(1)

        def __getattr__(self, name):
            return getattr(self.opti, name)

        def solver(self, name, options, native):
            solver_setups.append(1)
            # Deliberately stop each chunk before convergence, so the test
            # exercises real repeated IPOPT calls on the same Opti instance.
            self.opti.solver(name, options, {**native, "max_iter": 2})

        def solve(self):
            solves.append(1)
            return self.opti.solve()

    monkeypatch.setattr(solver.ca, "Opti", ObservedOpti)
    result = solver.solve(payload, tmp_path)
    assert len(constructions) == len(solver_setups) == 1
    assert len(solves) == 8
    chunks = result["solve_chunks"]
    assert not chunks[0]["graph_reused"]
    assert all(c["primal_dual_warm_start"] and c["graph_reused"] for c in chunks[1:])
    assert sum(c["iterations"] for c in chunks) == 16
    assert result["iterations_at_last_checkpoint"] == 16
    assert public._read(tmp_path / "layout.json")["graph_builds"] == 1


@pytest.mark.parametrize("chunks", [0, 9, True, 2.5])
def test_invalid_reuse_chunk_limit_rejected(inputs, tmp_path, chunks):
    payload = small_payload(inputs, "collocation")
    payload["reuse_chunks"] = chunks
    with pytest.raises(ValueError, match="reuse_chunks"):
        solver.solve(payload, tmp_path)


def test_reuse_arm_does_not_refine_unfinished_mesh(inputs, tmp_path, monkeypatch):
    payload = small_payload(inputs, "collocation")
    payload.update(arm=fit.REUSE_ARM, policy={"seconds": 30})
    calls = []

    class FiniteOracle:
        def __init__(self, system, train, scales, settings, directory, deadline, **kw):
            self.valid_calls = 0
            self.budget = kw["budget"]
            self.names = system.names
            self.size = sum(len(r.time) for r in train.trajectories)

        def vector(self, point):
            return np.array([point[n] for n in self.names])

        def __call__(self, vector):
            self.valid_calls += 1
            self.budget.calls += 1
            return np.ones(self.size)

    def incomplete_native(data, directory, seconds, **kwargs):
        calls.append(data)
        assert data["reuse_chunks"] == 8
        assert seconds > 15  # Most remaining time, not one third per mesh.
        directory.mkdir(parents=True)
        public._write(
            directory / "checkpoints.json",
            {
                "pool": [
                    {
                        "parameters": data["start"],
                        "maximum_scaled_defect": 1,
                        "collocation_nmse": 1,
                    }
                ],
                "latest": {
                    "parameters": data["start"],
                    "maximum_scaled_defect": 1,
                    "collocation_nmse": 1,
                },
            },
        )
        return {"wall_timeout": True, "elapsed_seconds": seconds}

    monkeypatch.setattr(fit, "GuardedOracle", FiniteOracle)

    monkeypatch.setattr(fit, "invoke", incomplete_native)
    result = fit.fit(payload, tmp_path)
    assert len(calls) == 1
    assert result["stop_reason"] == "coarse_solve_incomplete_pending_replay"
    assert result["parameters"] is not None


def test_followup_roster_and_truth_separation(tmp_path, monkeypatch, inputs):
    monkeypatch.setattr(campaign.controls, "make_inputs", lambda: deepcopy(inputs))
    config = campaign.CampaignConfig(
        starts=1,
        include_cstr=False,
        include_reuse=True,
        cases=("shape",),
        strategy=fit.StrategyPolicy(seconds=600, maximum_rollout_calls=600),
    )
    assert campaign.prepare(tmp_path, config)["tasks"] == 5
    plan, cases = campaign.verify(tmp_path)
    assert plan["protocol"] == campaign.FOLLOWUP_PROTOCOL
    assert set(cases["cases"]) == {"shape"}
    workers = [campaign.worker_payload(plan, cases, t) for t in plan["tasks"]]
    for worker in workers:
        assert set(worker) == {
            "request",
            "coordinates",
            "nodes",
            "training",
            "arm",
            "policy",
        }
        assert worker["policy"]["seconds"] == 600
        assert worker["request"] == workers[0]["request"]
        assert worker["nodes"] == workers[0]["nodes"]
    assert campaign.prepare(tmp_path, config)["tasks"] == 5
    summary = campaign.report(tmp_path)
    assert summary["recorded"] == 0
    assert all(g["coefficient_passes"] == 0 for g in summary["groups"])
    assert all(
        g["median_maximum_coefficient_relative_error"] is None
        for g in summary["groups"]
    )


def test_coefficient_errors_exclude_initials_and_handle_zero(inputs):
    case = deepcopy(inputs["cases"]["linear"])
    request = controls.request("linear", 0).model_dump(mode="json")
    theta = dict(case["reference_parameters"])
    theta["a"] *= 1.2
    theta["init_z_value"] += 7
    metrics = campaign.coefficient_metrics(case, request, theta)
    assert metrics["maximum_coefficient_relative_error"] == pytest.approx(0.2)
    assert metrics["initial_parameter_absolute_errors"] == {"init_z_value": 7}
    case["reference_parameters"]["a"] = 0
    assert (
        campaign.coefficient_metrics(case, request, theta)[
            "maximum_coefficient_relative_error"
        ]
        is None
    )
    assert not campaign.coefficient_metrics(case, request, None)["available"]
    del theta["a"]
    with pytest.raises(ValueError, match="identities"):
        campaign.coefficient_metrics(case, request, theta)


def test_matched_source_imports_exact_inputs_and_generic_starts(
    tmp_path, monkeypatch, inputs
):
    monkeypatch.setattr(campaign.controls, "make_inputs", lambda: deepcopy(inputs))
    original, followup = tmp_path / "original", tmp_path / "followup"
    campaign.prepare(
        original,
        campaign.CampaignConfig(starts=1, include_cstr=False, cases=("shape",)),
    )
    old_plan, old_inputs = campaign.verify(original)
    monkeypatch.setattr(
        campaign.controls,
        "make_inputs",
        lambda: pytest.fail("must not regenerate matched data"),
    )
    config = campaign.CampaignConfig(
        starts=1, include_cstr=False, cases=("shape",), include_reuse=True
    )
    campaign.prepare(followup, config, matched_source=original)
    plan, data = campaign.verify(followup)
    assert data == old_inputs
    assert plan["commons"] == old_plan["commons"]
    assert plan["matched_source_plan_sha256"] == public.content_sha256(old_plan)
    assert campaign.prepare(followup, config, matched_source=original)["tasks"] == 5
    with pytest.raises(ValueError, match="configuration/source"):
        campaign.prepare(followup, config)
    with pytest.raises(ValueError, match="lacks requested start"):
        campaign.prepare(
            tmp_path / "wrong",
            config.model_copy(update={"starts": 2}),
            matched_source=original,
        )


def test_followup_metrics_after_endpoint_and_exact_resume(
    tmp_path, monkeypatch, inputs
):
    monkeypatch.setattr(campaign.controls, "make_inputs", lambda: deepcopy(inputs))
    campaign.prepare(
        tmp_path,
        campaign.CampaignConfig(
            starts=1, include_cstr=False, include_reuse=True, cases=("linear",)
        ),
    )
    plan, cases = campaign.verify(tmp_path)
    seal(
        tmp_path / "qualification/result.json",
        {"passed": True, "plan_sha256": public.content_sha256(plan)},
    )
    task = plan["tasks"][0]
    directory = tmp_path / "results" / task["task_id"]
    parameters = cases["cases"]["linear"]["reference_parameters"]
    seal(
        directory / "backend.json",
        {
            "parameters": parameters,
            "stop_reason": "test_endpoint",
            "total_seconds": 1,
            "budget_exhausted": False,
            "worker_payload_sha256": public.content_sha256(
                campaign.worker_payload(plan, cases, task)
            ),
        },
    )
    seal(
        directory / "replay.json",
        {
            "complete": True,
            "maximum_solver_difference": 0,
            "metrics": {"train": 0, "val": 0},
        },
    )
    monkeypatch.setattr(
        campaign, "run", lambda *a: pytest.fail("refitting completed backend")
    )
    result = campaign.run_task(tmp_path, 0)
    assert result["coefficients_recovered"]
    assert result["coefficient_recovery"]["maximum_coefficient_relative_error"] == 0
    assert campaign.run_task(tmp_path, 0) == read_seal(directory / "result.json")
    group = next(
        g for g in campaign.report(tmp_path)["groups"] if g["arm"] == task["arm"]
    )
    assert group["coefficient_vectors_available"] == group["coefficient_passes"] == 1


@pytest.mark.parametrize(
    "values",
    [
        {"cases": ()},
        {"cases": ("linear", "linear")},
        {"cases": ("cstr_hard",), "include_cstr": False},
    ],
)
def test_invalid_case_roster(values):
    with pytest.raises(ValueError):
        campaign.CampaignConfig(**values)
