"""Physics, privacy, compiled semantics and immutable recovery of basin controls."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.benchmarks import detention as basin
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import detention_benchmark as campaign
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

CONFIG = Path(__file__).resolve().parents[1] / "configs/detention_benchmark_v1.json"


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    root = tmp_path_factory.mktemp("detention")
    campaign.prepare(root, CONFIG)
    return root


@pytest.mark.parametrize("case", ["coupled", "independent"])
def test_conservation_compilation_and_reference_replay(release, case):
    audit = campaign.verify(release)["audits"][case]
    assert audit["named"]["pass"] and audit["inline"]["pass"]
    assert audit["production_truth_replay"]["strict_output_recovery"]
    assert audit["max_reference_solver_difference_m"] < 1e-7
    assert audit["max_relative_reference_balance_error"] < 1e-8


def test_double_counted_transfer_fails_physical_probe():
    request = basin.reference_request("coupled", basin.DetentionConfig().starts[0])
    payload = request.model_dump(mode="json")
    payload["base_candidate"]["state_equations"][1]["rhs"] = (
        "(inflow_down + 2*q_transfer - q_outlet)/area_down"
    )
    assert not campaign.equation_audit(PublicFitRequest.model_validate(payload))["pass"]


def test_independent_control_has_no_causal_upstream_effect():
    row = basin.forcing(basin.DetentionConfig(), "train", 0)
    changed = {**row, "initial_up": 1.2, "inflow_up": 2 * row["inflow_up"]}
    a = basin.reference_rollout(row, "independent")
    b = basin.reference_rollout(changed, "independent")
    np.testing.assert_allclose(a["states"][1], b["states"][1], atol=1e-7, rtol=0)
    connected = basin.reference_rollout(row, "coupled")
    connected_change = basin.reference_rollout(changed, "coupled")
    assert (
        np.max(abs(np.array(connected["states"][1]) - connected_change["states"][1]))
        > 0.1
    )


def test_public_export_excludes_reference_and_reuses_noise_across_starts(release):
    plan = campaign.verify(release)
    assert len(plan["tasks"]) == 8 and not plan["test_generated"]
    assert not list(release.rglob("test.json"))
    for path in (release / "public").rglob("train.json"):
        split = PublicSplit.model_validate(campaign._read_seal(path))
        assert len(split.rows) == 6
        assert all(
            set(r.targets) == {"h_down"} and not r.auxiliaries for r in split.rows
        )
        raw = path.read_text()
        assert (
            "k_transfer" not in raw and '"states"' not in raw and '"truth"' not in raw
        )
    clean = campaign._read_seal(release / "public/coupled/noise0/train.json")
    noisy = campaign._read_seal(release / "public/coupled/noise1/train.json")
    for a, b in zip(clean["rows"], noisy["rows"], strict=True):
        assert a["time"] == b["time"] and a["external_inputs"] == b["external_inputs"]
        assert a["targets"]["h_down"][0] == b["targets"]["h_down"][0]
        assert a["targets"]["h_down"][1:] != b["targets"]["h_down"][1:]


def test_public_boundary_no_fitted_or_hidden_validation_initials():
    request = basin.reference_request("coupled", basin.DetentionConfig().starts[0])
    model, guesses, _ = public._lower(request)
    assert set(model.parameter_names) == {"k_transfer", "k_outlet"}
    assert guesses == {"k_transfer": 30, "k_outlet": 30}
    initials = {
        c.state: c.expression for c in model.validated.candidate.initial_conditions
    }
    assert initials == {"h_up": "initial_up", "h_down": "h_down"}
    assert not request.context.lagged_targets
    assert public._capability(request, model) is None


def test_deterministic_resume_and_no_fitting_in_report(release, monkeypatch):
    before = (release / "plan.json").read_bytes()
    monkeypatch.setattr(
        campaign, "generate_case", lambda *a: pytest.fail("regenerated")
    )
    monkeypatch.setattr(public, "execute_fit", lambda *a: pytest.fail("unexpected fit"))
    campaign.prepare(release, CONFIG)
    assert (release / "plan.json").read_bytes() == before
    assert campaign.report(release)["status_counts"] == {"missing": 8}


def test_tampering_and_changed_source_are_blocked(release, monkeypatch, tmp_path):
    import shutil

    root = tmp_path / "copy"
    shutil.copytree(release, root)
    file = root / "public/coupled/noise0/train.json"
    file.write_text(file.read_text() + " ")
    with pytest.raises(ValueError, match="release file changed"):
        campaign.verify(root)
    monkeypatch.setattr(public, "_source_identity", lambda: "a" * 64)
    with pytest.raises(ValueError, match="source changed"):
        campaign.verify(release)


def test_test_generation_and_unsupported_release_configuration_rejected():
    with pytest.raises(ValueError, match="development"):
        basin.forcing(basin.DetentionConfig(), "test", 0)
    with pytest.raises(ValidationError):
        basin.DetentionConfig(noise_fractions=(0.0, 0.2))
    with pytest.raises(ValueError, match="rating table"):
        basin.rating(5)


def test_numerical_failure_is_retained_without_fresh_budget(
    release, tmp_path, monkeypatch
):
    import shutil

    root = tmp_path / "failed_fit"
    shutil.copytree(release, root)
    calls = []

    def fail(*args):
        calls.append(1)
        raise RuntimeError("intentional numerical failure")

    monkeypatch.setattr(public, "_run_backend", fail)
    first = campaign.run_task(root, 0)
    assert first["fit"]["status"] == "fit_failed"
    assert first["diagnostic"] is None
    assert campaign.run_task(root, 0) == first
    assert calls == [1]
    assert campaign.report(root)["status_counts"] == {"fit_failed": 1, "missing": 7}


def test_interrupted_fit_does_not_restart(release, tmp_path, monkeypatch):
    import shutil

    root = tmp_path / "interrupted_fit"
    shutil.copytree(release, root)
    request = PublicFitRequest.model_validate(
        campaign._read_seal(root / "diagnostic/coupled_noise0_start0.request.json")
    )
    training = PublicSplit.model_validate(
        campaign._read_seal(root / "public/coupled/noise0/train.json")
    )
    validation = PublicSplit.model_validate(
        campaign._read_seal(root / "public/coupled/noise0/val.json")
    )
    directory = root / "results/coupled_noise0_start0/fit"
    frozen = public.prepare_fit(request, training, validation, directory)
    public._write(directory / "started.json", {"identity": frozen["identity"]})
    monkeypatch.setattr(public, "_run_backend", lambda *a: pytest.fail("budget reset"))
    result = campaign.run_task(root, 0)
    assert result["fit"]["status"] == "interrupted"
    assert campaign.run_task(root, 0) == result
    with pytest.raises(ValueError, match="invalid task"):
        campaign.run_task(root, -1)


def test_rating_units_monotonicity_threshold_and_configuration():
    head = np.linspace(-0.5, 4, 1000)
    q = np.array([basin.rating(h) for h in head])
    assert np.all(q[head <= 0] == 0) and np.all(np.diff(q) >= 0)
    np.testing.assert_array_equal(
        [basin.rating(h) for h in basin.RATING_HEADS], basin.RATING_VALUES
    )
    assert (
        basin.DetentionConfig.model_validate(json.loads(CONFIG.read_text())).fit_profile
        == "collocation-single-target-v2"
    )
