"""Branch domains, causal rollouts, reliable search and recoverable checkpoints."""

from time import monotonic

import numpy as np
import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.fitting import llm_sr_detention_refit as kernel
from autoformalism.fitting import llm_sr_numerical_control as supplied
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.public_fitting import PublicSplit


def example(task=14):
    """Exact reviewed RHS families on short public synthetic decay trajectories."""
    if task == 14:
        parameters = {
            "k0": 0.04,
            "p0": 1.5,
            "kw": 0.5,
            "pw": 2.0,
            "beta": 0.2,
            "h_max": 3.0,
            "gamma_c": 0.11,
            "k_spill": -0.05,
        }
        processes = {
            "dh_dt": (
                "inflow_down / area_down * max(1-h_down/h_max,0)**gamma_c"
                "-k0*max(h_down-crest_down,0)**p0"
                "-kw*max(h_down-warning_depth,0)**pw"
                "-k_spill*max(h_down-warning_depth,0)-beta*h_down"
            )
        }
        target = "dh_dt+heaviside(-h_down,1)*max(-dh_dt,0)"
    else:
        parameters = {
            "C": 0.1,
            "beta": 1.0,
            "eps_s": 1.0,
            "h_max": 2.0,
            "Q_max": 0.13,
            "dt": 1.0,
            "k_loss": 0.2,
            "k_gw": 0.25,
            "kappa": 0.06,
        }
        processes = {
            "inflow_rate": "inflow_down/area_down",
            "outflow_capped": (
                "min(C*max(h_down-crest_down,0)**beta"
                "*(.5*(1+tanh((h_down-crest_down)/eps_s))),Q_max)"
            ),
            "outflow": (
                "max(max(outflow_capped,inflow_rate-(h_max-h_down)/dt-k_loss),0)"
            ),
            "dh0": "(inflow_rate-outflow-k_loss+k_gw*(1+kappa*h_down))*dt",
            "dh1": "dh0+heaviside(-h_down,1)*max(-dh0,0)",
        }
        target = "dh1+heaviside(-(h_down+dh1),0)*(-h_down-dh1)"
    item = supplied.Transcription.model_validate(
        {
            "task": task,
            "benchmark_id": "synthetic",
            "repetition": 0,
            "programs": {
                "h_down": {"function_sha256": "a" * 64, "parameters": [1.0] * 10}
            },
            "parameters": parameters,
            "processes": processes,
            "targets": {"h_down": target},
            "rules": ["direct"],
        }
    )
    context = ValidationContext(
        targets=("h_down",),
        external_inputs=("inflow_down",),
        fixed_covariates=("area_down", "crest_down", "warning_depth"),
    )
    times = np.linspace(0, 4, 9)
    training = PublicSplit.model_validate(
        {
            "name": "train",
            "fingerprint": "synthetic",
            "rows": [
                {
                    "trajectory_id": str(i),
                    "time": times.tolist(),
                    "targets": {"h_down": (initial * np.exp(-0.2 * times)).tolist()},
                    "external_inputs": {"inflow_down": np.zeros_like(times).tolist()},
                    "fixed_covariates": {
                        "area_down": 1200.0,
                        "crest_down": 2.0,
                        "warning_depth": 3.0,
                    },
                }
                for i, initial in enumerate((0.5, 1.0))
            ],
        }
    )
    return item, context, training


def test_domains_keep_signed_coefficients_and_all_starts_are_frozen():
    for task in (14, 15):
        item, context, training = example(task)
        domain = kernel.domain_policy(item, context, training)
        assert not domain["scientific_bounds"] and not domain["dt_is_integration_step"]
        starts = kernel.deterministic_starts(item, context, training)
        assert starts == kernel.deterministic_starts(item, context, training)
        assert [r["start_id"] for r in starts] == [
            "supplied",
            "perturbed_0",
            "perturbed_1",
            "generic_0",
            "generic_1",
        ]
        assert len({public.content_sha256(r["parameters"]) for r in starts}) == 5
        oracle = kernel.DetentionOracle(item, context, training)
        for row in starts:
            oracle.vector(row["parameters"])
        if task == 14:
            assert domain["bounds"]["k_spill"][0] < 0 < domain["bounds"]["k_spill"][1]
            assert starts[0]["parameters"]["k_spill"] < 0


def test_generic_starts_do_not_use_supplied_coefficients():
    item, context, training = example()
    changed = item.model_copy(update={"parameters": item.parameters | {"beta": 0.4}})
    original = kernel.deterministic_starts(item, context, training)
    other = kernel.deterministic_starts(changed, context, training)
    assert original[1] != other[1]
    assert original[3:] == other[3:]


def test_analytic_decay_derivatives_and_independent_verification():
    item, context, training = example()
    oracle = kernel.DetentionOracle(item, context, training)
    point = oracle.vector(item.parameters)
    residual, jacobian = oracle.evaluate(point, monotonic() + 10)
    assert np.mean(residual**2) < 1e-16
    expected = np.concatenate(
        [
            -np.asarray(row.time) * np.asarray(row.targets["h_down"]) / oracle.scale
            for row in training.rows
        ]
    )
    np.testing.assert_allclose(
        jacobian[:, oracle.names.index("beta")], expected, rtol=1e-5, atol=1e-7
    )
    verified = kernel.verify(item, context, training, item.parameters, seconds=10)
    assert verified["usable_for_retention"]
    assert verified["verification_calls_completed"] == 2
    assert not verified["initial_values_fitted"]


def test_later_targets_never_drive_rollouts():
    item, context, training = example()
    first = kernel.DetentionOracle(item, context, training)
    r0, _ = first.evaluate(
        first.vector(item.parameters), monotonic() + 5, jacobian=False
    )
    changed = training.model_dump(mode="json")
    for row in changed["rows"]:
        row["targets"]["h_down"][1:] = [10.0] * 8
    second = kernel.DetentionOracle(item, context, PublicSplit.model_validate(changed))
    r1, _ = second.evaluate(
        second.vector(item.parameters), monotonic() + 5, jacobian=False
    )
    y0 = np.concatenate([row.targets["h_down"] for row in training.rows])
    y1 = np.concatenate([row["targets"]["h_down"] for row in changed["rows"]])
    np.testing.assert_allclose(
        r0 * first.scale + y0, r1 * second.scale + y1, atol=1e-14, rtol=0
    )


def test_fit_recovers_identifiable_decay_without_claiming_all_coefficients():
    item, context, training = example()
    progress = []
    result = kernel.fit(
        item,
        context,
        training,
        item.parameters | {"beta": 0.3},
        seconds=30,
        checkpoint=progress.append,
        policy=kernel.Policy(qualification_seconds=15),
    )
    assert result["best"]["training_nmse"] < 1e-8, result
    assert result["best"]["parameters"]["beta"] == pytest.approx(0.2, rel=1e-4)
    assert result["derivative_qualification"][-1]["search_qualified"]
    assert not result["global_convergence_certified"]
    assert result["search_calls_started"] > result["calls"]
    assert progress[-1]["best"]["training_nmse"] < 1e-8


def test_successful_search_base_is_saved_before_jacobian_timeout(monkeypatch):
    item, context, training = example()
    progress = []
    oracle = kernel.DetentionOracle(item, context, training, checkpoint=progress.append)
    oracle.retain_base_checkpoints = True
    original = oracle.residual
    calls = []

    def timed(vector, deadline, method="Radau"):
        calls.append(vector.copy())
        if len(calls) > 1:
            raise TimeoutError("derivative timed out")
        return original(vector, deadline, method)

    monkeypatch.setattr(oracle, "residual", timed)
    with pytest.raises(TimeoutError):
        oracle.evaluate(oracle.vector(item.parameters), monotonic() + 5)
    saved = next(
        p["best_rollout_checkpoint"] for p in progress if "best_rollout_checkpoint" in p
    )
    assert saved["parameters"] == item.parameters
    assert saved["training_nmse"] < 1e-16
    # A residual-only diagnostic probe cannot replace the search incumbent.
    monkeypatch.setattr(oracle, "residual", original)
    oracle.evaluate(
        oracle.vector(item.parameters | {"beta": 0.3}), monotonic() + 5, jacobian=False
    )
    assert oracle.best_residual == saved


def test_interrupted_search_retains_advancing_top_level_checkpoint(monkeypatch):
    item, context, training = example()

    class Qualified:
        search_qualified = True

        def model_dump(self, **kwargs):
            return {"search_qualified": True}

    monkeypatch.setattr(kernel, "_qualify", lambda *a: Qualified())

    def interrupted(oracle, start, deadline, checkpoint, **kwargs):
        checkpoint({"best": {"parameters": item.parameters, "training_nmse": 0.0}})
        checkpoint({"best": None})
        raise TimeoutError("interrupted after improvement")

    monkeypatch.setattr(kernel.recovery, "search", interrupted)
    progress = []
    result = kernel.fit(
        item,
        context,
        training,
        item.parameters | {"beta": 0.3},
        seconds=5,
        checkpoint=progress.append,
    )
    assert result["best"]["parameters"] == item.parameters
    assert result["best"]["training_nmse"] == 0
    assert progress[-1]["best"] == result["best"]


def test_task15_gauge_preserves_predictions_and_withholds_unique_recovery():
    item, context, training = example(15)
    oracle = kernel.DetentionOracle(item, context, training)
    # Start above the fitted capacity so the h_max-dependent branch is active.
    # Matching only the ordinary inflow/loss branch would not check the M gauge.
    source = item.parameters | {"h_max": 0.2}
    delta = 0.05
    changed = source | {
        "k_gw": source["k_gw"] + delta,
        "k_loss": source["k_loss"] + delta,
        "kappa": source["kappa"] * source["k_gw"] / (source["k_gw"] + delta),
        "h_max": source["h_max"] - source["dt"] * delta,
    }
    first = oracle.evaluate(oracle.vector(source), monotonic() + 5, jacobian=False)[0]
    second = oracle.evaluate(oracle.vector(changed), monotonic() + 5, jacobian=False)[0]
    np.testing.assert_allclose(first, second, atol=1e-8, rtol=1e-8)
    diagnosis = kernel.diagnose(item, context, training, source, seconds=0.01)
    assert diagnosis["exact_ambiguity"]["rank_ceiling"] == 8
    assert not diagnosis["coefficient_accuracy_claimed"]
    assert not diagnosis["parameter_information_qualified"]


def test_nontraining_invalid_domains_and_deadlines_fail():
    item, context, training = example()
    with pytest.raises(ValueError, match="training only"):
        kernel.domain_policy(item, context, training.model_copy(update={"name": "val"}))
    oracle = kernel.DetentionOracle(item, context, training)
    with pytest.raises(ValueError, match="working box"):
        oracle.vector(item.parameters | {"h_max": 0})
    with pytest.raises(TimeoutError):
        oracle.evaluate(oracle.vector(item.parameters), monotonic() - 1)
