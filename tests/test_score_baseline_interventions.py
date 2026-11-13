"""External baselines go through the intervention study's own scoring code."""

from __future__ import annotations

import numpy as np
import pytest

from autoformalism.benchmarks.phase_b_generation import PhaseBProtocol
from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate
from autoformalism.schemas import CandidateModel
from scripts import iclr_anchor_fidelity as anchor
from scripts import score_baseline_interventions as scoring

CSTR = anchor.fits.CSTR
CONTEXT = ValidationContext.model_validate(
    {
        "targets": ["T"],
        "auxiliaries": [],
        "external_inputs": ["Cf", "Tf", "Tjf"],
        "fixed_covariates": [],
    }
)
#: A toy reactor law that only the baseline grammar reads (``Max``).
LAW = "-0.8*(T - 350) + 6*Tjf - 4*Cf + Max(0*Tf, -1)"


def _row(semantics="continuous_time", **model):
    candidate = equation_candidate("sindy", {"T": LAW}, CONTEXT)
    return {
        "source_id": "sindy:toy:0",
        "source_protocol": scoring.PROTOCOL,
        "source_bundle_sha256": "0" * 64,
        "benchmark_id": CSTR,
        "method": "sindy",
        "repetition": 0,
        "status": "ready",
        "direct_readouts": {},
        "model": {
            "candidate": candidate.model_dump(mode="json"),
            "context": CONTEXT.model_dump(mode="json"),
            "parameters": {},
            "initials": {},
            "semantics": semantics,
            "grammar": "baseline",
            **model,
        },
    }


def _inputs(spec, time):
    """The schedule's input deviations on a grid, for the toy reference."""
    values = np.zeros((len(time), 3))
    if spec.get("kind") == "step":
        on = (time >= spec["start"]) & (time < spec["end"])
        values[on] = spec["value"]
    return values


def _reference(item, time):
    """The toy law itself, integrated finely, as the reference response."""
    u = _inputs(item["protocol"]["specification"], time)
    temperature = np.empty_like(time)
    temperature[0] = 350.0
    for i in range(len(time) - 1):
        x = temperature[i]
        for _ in range(100):
            x += (
                (time[i + 1] - time[i])
                / 100
                * (-0.8 * (x - 350) + 6 * u[i, 2] - 4 * u[i, 0])
            )
        temperature[i + 1] = x
    hidden = np.linspace(0.0, 1.0, len(time))
    return {
        "time": time.tolist(),
        "T": temperature.tolist(),
        **{h: hidden.tolist() for h in anchor.fits.HIDDEN[CSTR]},
        **{n: u[:, j].tolist() for j, n in enumerate(("Cf", "Tf", "Tjf"))},
    }


def _trajectory(time):
    return Trajectory(
        trajectory_id="toy",
        time=time,
        targets={"T": np.full(len(time), 350.0)},
        external_inputs={n: np.zeros(len(time)) for n in ("Cf", "Tf", "Tjf")},
        auxiliaries={},
        fixed_covariates={},
        derivatives={},
    )


def test_a_continuous_baseline_is_read_with_its_own_grammar():
    row = _row()
    with pytest.raises(ModelValidationError, match="Max"):
        compile_candidate(
            CandidateModel.model_validate(row["model"]["candidate"]), CONTEXT
        )
    time = np.linspace(0.0, 2.0, 21)
    curves = scoring.predict(row, _trajectory(time), "Radau")
    assert np.allclose(curves["T"], 350.0)


def test_a_native_map_runs_only_on_its_own_sample_interval():
    time = np.linspace(0.0, 2.0, 21)
    row = _row("native_increment", increment_step=0.1)
    curves = scoring.predict(row, _trajectory(time), "Radau")
    assert len(curves["T"]) == len(time)
    with pytest.raises(ValueError, match="sample interval"):
        scoring.predict(
            _row("native_increment", increment_step=0.2), _trajectory(time), "Radau"
        )


def test_the_study_scores_a_baseline_with_its_own_metrics(tmp_path):
    schedules = anchor.cases(CSTR, "confirmation")
    time = np.round(np.arange(0.0, 30.05, 0.1), 10)
    refs = {k: _reference(item, time) for k, item in schedules.items()}
    with scoring.baseline_predictions():
        result = anchor.evaluate_one(
            (tmp_path, 0, _row(), schedules, refs, "plan", {"T": 1.0})
        )
    assert anchor.predict.__name__ == "predict"  # restored afterwards
    targets = {r["case"]: r for r in result["scores"] if r["readout"] == "target"}
    assert set(targets) == {"feed_low", "feed_high", "jacket_cool", "jacket_warm"}
    for row in targets.values():
        assert row["status"] == "complete"
        assert row["direction_correct"] is True
        assert row["effect_rse"] < 1e-3
    hidden = {r["status"] for r in result["scores"] if r["readout"] != "target"}
    assert hidden == {"physical_correspondence_unavailable", "no_correspondence"}


def test_a_rollout_past_its_time_limit_is_a_failed_rollout(monkeypatch):
    monkeypatch.setattr(scoring, "SECONDS", 0.0)
    time = np.linspace(0.0, 2.0, 21)
    with (
        scoring.baseline_predictions(),
        pytest.raises(ValueError, match="time limit"),
    ):
        anchor.predict(_row(), _trajectory(time), "Radau")


def test_a_contract_mismatch_is_refused():
    row = _row()
    row["model"]["context"] = {**row["model"]["context"], "auxiliaries": ["C"]}
    with pytest.raises(ValueError, match="contract"):
        scoring.predict(row, _trajectory(np.linspace(0.0, 1.0, 11)), "Radau")


def test_freeze_refuses_unknown_methods_and_cells(tmp_path):
    with pytest.raises(ValueError, match="unknown methods"):
        scoring.freeze(tmp_path, tmp_path, {"oracle": tmp_path}, [CSTR])
    with pytest.raises(ValueError, match="unknown methods"):
        scoring.freeze(tmp_path, tmp_path, {"d3": tmp_path}, ["phase_c_other"])


#: A trained network that HDTwinGen's map reads as ``r``.
NETWORK = {
    "name": "r",
    "inputs": ["T", "Cf"],
    "hidden_layers": 1,
    "hidden_width": 4,
    "activation": "tanh",
}


def _hybrid(bias):
    """HDTwinGen's map with zero weights, so ``r`` is its last bias everywhere."""
    weights = [
        {"weight": np.zeros(shape).tolist(), "bias": np.zeros(shape[0]).tolist()}
        for shape in ((4, 2), (1, 4))
    ]
    weights[-1]["bias"] = [bias]
    row = _row(
        "native_increment",
        increment_step=0.1,
        networks=[NETWORK],
        network_weights={"r": weights},
    )
    candidate = equation_candidate(
        "hdtwingen", {"T": "-0.8*(T - 350) + 6*Tjf + r"}, CONTEXT
    )
    row["model"]["candidate"] = candidate.model_dump(mode="json")
    return row


def test_a_hybrid_map_evaluates_its_trained_networks():
    time = np.linspace(0.0, 2.0, 21)
    hybrid = scoring.predict(_hybrid(2.0), _trajectory(time), "Radau")
    plain = _row("native_increment", increment_step=0.1)
    plain["model"]["candidate"] = equation_candidate(
        "d3", {"T": "-0.8*(T - 350) + 6*Tjf + 2.0"}, CONTEXT
    ).model_dump(mode="json")
    expected = scoring.predict(plain, _trajectory(time), "Radau")
    assert np.allclose(hybrid["T"], expected["T"], rtol=0.0, atol=1e-12)
    assert hybrid["T"][-1] > 350.0
    broken = _hybrid(2.0)
    broken["model"]["network_weights"] = {}
    with pytest.raises(ValueError, match="exactly the declared"):
        scoring.predict(broken, _trajectory(time), "Radau")


def _nine_case_run(root, results, method="spec_sindy"):
    """A sealed nine-case run on the reactor cell, one task per result."""
    from autoformalism.rebuttal.prefit_replay import sealed_write

    identity = {"benchmark_id": CSTR, "train": "public"}
    plan = sealed_write(
        root / "plan.json",
        {
            "protocol": scoring.NINE_CASE[method],
            **({} if method == "spec_sindy" else {"method": method}),
            "rows": [
                {
                    "index": index,
                    "benchmark_id": CSTR,
                    "repetition": index,
                    "tier": "hard",
                    "public_identity": identity,
                }
                for index in range(len(results))
            ],
            "test_data_opened": False,
        },
    )
    for index, result in enumerate(results):
        folder = root / "results" / str(index)
        folder.mkdir(parents=True)
        sealed_write(
            folder / "result.json",
            {
                "plan_sha256": plan["artifact_sha256"],
                "index": index,
                "test_data_opened": False,
                **result,
            },
        )
    return identity


def _public(monkeypatch, identity):
    from types import SimpleNamespace

    monkeypatch.setattr(
        scoring, "load_cell", lambda release, name: SimpleNamespace(identity=identity)
    )
    return {CSTR: {"context": CONTEXT.model_dump(mode="json")}}


def test_a_nine_case_run_is_read_with_its_failures(tmp_path, monkeypatch):
    identity = _nine_case_run(
        tmp_path,
        [
            {"status": "complete", "equations": {"T": "-0.8*(T - 350) + tanh(Cf)"}},
            {"status": "rollout_failed", "error": "diverged"},
        ],
    )
    cells = _public(monkeypatch, identity)
    ready, failed = scoring.nine_case_rows("spec_sindy", tmp_path, tmp_path, cells)
    assert (ready["status"], ready["model"]["semantics"]) == (
        "ready",
        "continuous_time",
    )
    time = np.linspace(0.0, 1.0, 11)
    curves = scoring.predict(
        {**_row(), "model": ready["model"]}, _trajectory(time), "Radau"
    )
    assert np.allclose(curves["T"], 350.0)
    assert (failed["status"], failed["failure"]) == ("method_failure", "rollout_failed")


def test_a_nine_case_pick_must_come_from_the_public_files(tmp_path, monkeypatch):
    _nine_case_run(tmp_path, [{"status": "complete", "equations": {"T": "-T"}}])
    cells = _public(monkeypatch, {"benchmark_id": CSTR, "train": "other"})
    with pytest.raises(ValueError, match="other public files"):
        scoring.nine_case_rows("spec_sindy", tmp_path, tmp_path, cells)


def test_an_outage_is_rerun_and_another_method_refused(tmp_path, monkeypatch):
    identity = _nine_case_run(
        tmp_path, [{"status": "endpoint_unavailable"}], method="hdtwingen"
    )
    cells = _public(monkeypatch, identity)
    with pytest.raises(ValueError, match="rerun"):
        scoring.nine_case_rows("hdtwingen", tmp_path, tmp_path, cells)
    with pytest.raises(ValueError, match="not a sealed spec_sindy run"):
        scoring.nine_case_rows("spec_sindy", tmp_path, tmp_path, cells)


def _latent_row(rng):
    """A small random latent ODE for the reactor cell: T plus two latents."""

    def layer(rows, cols):
        return {
            "weight": (0.1 * rng.standard_normal((rows, cols))).tolist(),
            "bias": (0.1 * rng.standard_normal(rows)).tolist(),
        }

    model = {
        "semantics": "latent_neural_ode",
        "context": CONTEXT.model_dump(mode="json"),
        "targets": ["T"],
        "inputs": ["Cf", "Tf", "Tjf"],
        "tau": 3.0,
        "standardization": {
            "target_mean": [350.0],
            "target_sd": [5.0],
            "input_mean": [0.0, 0.0, 0.0],
            "input_sd": [1.0, 1.0, 1.0],
        },
        "network": {
            "dynamics": [layer(8, 6), layer(8, 8), layer(3, 8)],
            "encoder": [layer(4, 4), layer(2, 4)],
        },
    }
    return {**_row(), "method": "latent_ode", "model": model}


def test_a_latent_ode_gives_its_targets_and_latent_coordinates():
    row = _latent_row(np.random.default_rng(0))
    time = np.linspace(0.0, 2.0, 21)
    radau = scoring.predict(row, _trajectory(time), "Radau")
    bdf = scoring.predict(row, _trajectory(time), "BDF")
    assert set(radau) == {"T", "latent.h0", "latent.h1"}
    assert radau["T"][0] == pytest.approx(350.0)
    assert np.max(np.abs(np.subtract(radau["T"], bdf["T"]))) < 1e-5


def test_the_study_probes_a_latent_ode_like_its_own_models(tmp_path):
    schedules = anchor.cases(CSTR, "confirmation")
    time = np.round(np.arange(0.0, 30.05, 0.1), 10)
    refs = {k: _reference(item, time) for k, item in schedules.items()}
    row = _latent_row(np.random.default_rng(1))
    with scoring.baseline_predictions():
        result = anchor.evaluate_one(
            (tmp_path, 0, row, schedules, refs, "plan", {"T": 1.0})
        )
    targets = [r for r in result["scores"] if r["readout"] == "target"]
    assert {r["status"] for r in targets} <= {"complete", "negligible_reference_effect"}
    probes = {r["status"] for r in result["scores"] if r["readout"] == "oracle_affine"}
    assert "calibration_unavailable" not in probes


def _latent_sindy_model():
    """Scaled T relaxes and is driven by one latent that starts at 0.5 and decays."""
    from autoformalism.baselines import latent_sindy

    terms = latent_sindy.library_terms(5)  # T, one latent, three inputs
    coefficients = np.zeros((len(terms), 2))
    coefficients[terms.index((0,)), 0] = -3.0
    coefficients[terms.index((1,)), 0] = 0.6
    coefficients[terms.index((3,)), 0] = 0.3
    coefficients[terms.index((1,)), 1] = -3.0
    return {
        "semantics": "latent_sindy",
        "context": CONTEXT.model_dump(mode="json"),
        "targets": ["T"],
        "inputs": ["Cf", "Tf", "Tjf"],
        "tau": 3.0,
        "standardization": {
            "target_mean": [350.0],
            "target_sd": [5.0],
            "input_mean": [0.0, 0.0, 0.0],
            "input_sd": [1.0, 1.0, 1.0],
        },
        "coefficients": coefficients.tolist(),
        "mask": np.ones_like(coefficients).tolist(),
        "initializer": [[0.0, 0.0, 0.0, 0.0, 0.5]],
    }


def test_a_latent_sindy_model_gives_its_target_and_latent_coordinate():
    row = {**_row(), "method": "latent_sindy", "model": _latent_sindy_model()}
    time = np.linspace(0.0, 2.0, 21)
    radau = scoring.predict(row, _trajectory(time), "Radau")
    bdf = scoring.predict(row, _trajectory(time), "BDF")
    assert set(radau) == {"T", "latent.h0"}
    assert radau["T"][0] == pytest.approx(350.0)
    assert radau["latent.h0"][0] == pytest.approx(0.5)
    assert radau["T"][-1] > 350.0  # the latent drives the temperature up
    assert np.max(np.abs(np.subtract(radau["T"], bdf["T"]))) < 1e-5


def _sindy_autoencoder_model():
    """Scaled T, read from the window's newest entry, relaxes towards a constant."""
    from autoformalism.baselines import latent_sindy

    terms = latent_sindy.library_terms(5)  # z1, z2, three inputs
    coefficients = np.zeros((len(terms), 2))
    coefficients[terms.index(()), 0] = 1.0
    coefficients[terms.index((0,)), 0] = -1.0
    coefficients[terms.index((1,)), 1] = -1.0
    return {
        "semantics": "sindy_autoencoder",
        "context": CONTEXT.model_dump(mode="json"),
        "targets": ["T"],
        "inputs": ["Cf", "Tf", "Tjf"],
        "delays": 3,
        "standardization": {
            "target_mean": [350.0],
            "target_sd": [5.0],
            "input_mean": [0.0, 0.0, 0.0],
            "input_sd": [1.0, 1.0, 1.0],
        },
        # z1 is the newest entry (the hidden unit stays where ELU is the
        # identity); z2 starts at 0.5.
        "encoder": [
            {"weight": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "bias": [10.0, 0.0]},
            {"weight": [[1.0, 0.0], [0.0, 0.0]], "bias": [-10.0, 0.5]},
        ],
        "decoder": [],
        "coefficients": coefficients.tolist(),
        "mask": np.ones_like(coefficients).tolist(),
    }


def test_a_sindy_autoencoder_model_gives_its_target_and_latent_coordinate():
    row = {**_row(), "method": "sindy_autoencoder", "model": _sindy_autoencoder_model()}
    time = np.linspace(0.0, 2.0, 21)
    radau = scoring.predict(row, _trajectory(time), "Radau")
    bdf = scoring.predict(row, _trajectory(time), "BDF")
    assert set(radau) == {"T", "latent.h0"}
    assert radau["T"][0] == pytest.approx(350.0)
    np.testing.assert_allclose(
        radau["T"], 350.0 + 5.0 * (1 - np.exp(-time)), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(radau["latent.h0"], 0.5 * np.exp(-time), atol=1e-7)
    assert np.max(np.abs(np.subtract(radau["T"], bdf["T"]))) < 1e-5


def test_a_latent_selection_is_read_only_under_its_own_protocol(tmp_path):
    from autoformalism.baselines import latent_ode, latent_sindy
    from autoformalism.rebuttal.prefit_replay import sealed_write

    def sealed(root, protocol):
        plan = sealed_write(root / "plan.json", {"protocol": protocol})
        sealed_write(
            root / "selection.json",
            {
                "plan_sha256": plan["artifact_sha256"],
                "test_data_opened": False,
                "rows": [
                    {
                        "benchmark_id": CSTR,
                        "repetition": 0,
                        "status": "complete",
                        "validation_nmse": 0.01,
                        "model": _latent_sindy_model(),
                    },
                    {"benchmark_id": CSTR, "repetition": 1, "status": "rollout_failed"},
                ],
            },
        )
        return root

    other = sealed(tmp_path / "ode", latent_ode.PROTOCOL)
    with pytest.raises(ValueError, match="latent_sindy selection belongs to another"):
        scoring.latent_rows("latent_sindy", other)
    own = sealed(tmp_path / "sindy", latent_sindy.PROTOCOL)
    ready, failed = scoring.latent_rows("latent_sindy", own)
    assert ready["method"] == "latent_sindy" and ready["status"] == "ready"
    assert ready["recorded"] == ["selection.validation_nmse", 0.01]
    assert failed["status"] == "method_failure"
    assert failed["failure"] == "rollout_failed"


def _suite():
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "cstr_mechanism_probes_v1.json"
    )
    return json.loads(path.read_text())


def test_the_probe_suite_copies_the_study_base_and_calibration():
    suite = _suite()
    table = scoring.suite_schedules(CSTR, suite)
    study = anchor.cases(CSTR, "confirmation")
    for name in suite["calibration"]:
        assert table[name] == study[name]
    base = study["train_zero"]["protocol"]
    probes = [k for k, v in table.items() if v["role"] == "contrast"]
    assert len(probes) == len(suite["probes"]) == 12
    for item in table.values():
        if item["role"] == "train_calibration":
            continue
        PhaseBProtocol.model_validate(item["protocol"])
        changed = {k for k in base if item["protocol"][k] != base[k]}
        assert changed <= {"protocol_id", "split", "specification"}
        if item["role"] == "contrast":
            assert item["control"] == "basal"
            low, high = item["window"]
            assert 0.0 <= low < high <= base["duration"]


def test_a_probe_never_shifts_a_hidden_initial_value():
    for probe_case in _suite()["probes"]:
        shift = probe_case["specification"].get("initial_shift", [0.0, 0.0, 0.0])
        assert shift[0] == shift[2] == 0.0


def test_an_unfrozen_suite_or_another_cell_is_refused():
    suite = _suite()
    with pytest.raises(ValueError, match="not frozen"):
        scoring.suite_schedules(CSTR, {**suite, "status": "draft"})
    with pytest.raises(ValueError, match="not frozen"):
        scoring.suite_schedules(anchor.fits.T2, suite)


def test_the_study_models_join_on_their_canonical_cell(tmp_path):
    from autoformalism.rebuttal.prefit_replay import sealed_write

    path = tmp_path / "frozen_models.json"
    sealed_write(
        path,
        {
            "protocol": "iclr-controlled-search-models-1",
            "rows": [
                {"benchmark_id": "sensor_00", "method": "full", "repetition": 0},
                {"benchmark_id": "acquisition_00", "method": "full", "repetition": 0},
            ],
        },
    )
    rows = scoring.study_rows(path, CSTR)
    assert [(r["benchmark_id"], r["source_benchmark_id"]) for r in rows] == [
        (CSTR, "sensor_00")
    ]
    assert rows[0]["source_protocol"] == "iclr-controlled-search-models-1"


def test_the_extrapolation_suite_leaves_the_training_range_in_four_directions():
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1] / "configs" / "cstr_extrapolation_v1.json"
    )
    suite = json.loads(path.read_text())
    table = scoring.suite_schedules(CSTR, suite)
    assert table["train_zero"] == anchor.cases(CSTR, "confirmation")["train_zero"]
    signs = set()
    for probe_case in suite["probes"]:
        spec = probe_case["specification"]
        PhaseBProtocol.model_validate(table[probe_case["name"]]["protocol"])
        assert spec["kind"] == "step" and "initial_shift" not in spec
        cf, tf, tjf = spec["value"]
        # Training's largest step is one unit in each direction.
        assert (
            tf == 0.0 and min(abs(cf), abs(tjf)) == 0.0 and max(abs(cf), abs(tjf)) >= 2
        )
        signs.add((cf > 0, cf < 0, tjf > 0, tjf < 0))
    assert len(signs) == 4


def _combined():
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "cstr_combined_interventions_v1.json"
    )
    return json.loads(path.read_text())


def test_each_probe_is_paired_with_the_control_it_names():
    suite = _combined()
    table = scoring.suite_schedules(CSTR, suite)
    controls = {c["name"] for c in suite["controls"]}
    assert {k for k, v in table.items() if v["role"] == "control"} == controls
    for probe_case in suite["probes"]:
        item = table[probe_case["name"]]
        assert item["control"] == probe_case["control"]
        assert item["events"] == probe_case["events"]
        PhaseBProtocol.model_validate(item["protocol"])
        shift = probe_case["specification"].get("initial_shift")
        own = table[item["control"]]["protocol"]["specification"].get("initial_shift")
        assert shift == own  # the control starts where the probe starts
        assert shift is None or shift[0] == shift[2] == 0.0
        for event in probe_case["events"]:
            low, high = event["window"]
            assert 0.0 <= low < high <= 30.0


def test_a_suite_must_pair_every_probe_with_one_of_its_controls():
    suite = _combined()
    stray = {**suite["probes"][0], "control": "nowhere"}
    with pytest.raises(ValueError, match="names no control"):
        scoring.suite_schedules(CSTR, {**suite, "probes": [stray]})
    with pytest.raises(ValueError, match="either one control"):
        scoring.suite_schedules(CSTR, {**suite, "control": suite["controls"][0]})
    broken = {
        **suite["probes"][0],
        "specification": {
            "kind": "composite",
            "channels": [
                [{"shape": "step", "start": 5, "end": 40, "value": 1}],
                [],
                [],
            ],
        },
    }
    with pytest.raises(ValueError, match="horizon"):
        scoring.suite_schedules(CSTR, {**suite, "probes": [broken]})


def _toy_plan(tmp_path):
    """A frozen plan with a validity rule and toy schedules of every outcome."""
    from autoformalism.rebuttal.prefit_replay import sealed_write

    flat = {"kind": "constant", "value": [0.0, 0.0, 0.0]}
    pulse = {"kind": "step", "start": 5.0, "end": 9.0, "value": [1.0, 0.0, 0.0]}
    suite = {
        "status": "frozen_before_references",
        "cell": CSTR,
        "base_protocol": "train_zero",
        "calibration": [],
        "controls": [
            {"name": "basal", "specification": flat},
            {"name": "broken_basal", "specification": flat},
        ],
        "probes": [
            {
                "name": n,
                "group": "toy",
                "control": c,
                "specification": pulse,
                "window": [5.0, 30.0],
                "events": [],
            }
            for n, c in (
                ("good", "basal"),
                ("edge", "basal"),
                ("negative", "basal"),
                ("disagree", "basal"),
                ("orphan", "broken_basal"),
            )
        ],
    }
    rule = _combined()["validity"]
    plan = {
        "protocol": scoring.PROTOCOL,
        "evaluator": scoring.evaluator_identity(),
        "schedules": {CSTR: scoring.suite_schedules(CSTR, suite)},
        "validity_rule": rule,
        "rows": [],
        "study_rows": [],
    }
    return sealed_write(tmp_path / "plan.json", plan)


def _toy_reactor(monkeypatch):
    """References and scaled simulations for the toy plan's cases."""
    time = np.round(np.arange(0.0, 30.05, 0.1), 10)
    bump = np.exp(-((time - 8.0) ** 2))

    def temperature(name, scale=None, factor=1.0):
        if name in ("basal", "broken_basal"):
            return np.full(len(time), 350.0)
        gain = 1.0 + (factor - 1.0) * (
            60.0 if name == "edge" and scale == "reaction_heat" else 1.0
        )
        return 350.0 + 5.0 * gain * bump

    def reference(cell, item, data_root, method):
        name = item["protocol"]["protocol_id"].removeprefix("test_probe_")
        shift = (
            1e-2 if method == "Radau" and name in ("disagree", "broken_basal") else 0.0
        )
        concentration = np.full(len(time), 0.5)
        if name == "negative":
            concentration[100:120] = -0.01
        return {
            "time": time.tolist(),
            "T": (temperature(name) + shift).tolist(),
            "C": concentration.tolist(),
        }

    class Run:
        state_names = ("C", "T", "Tj")

        def __init__(self, protocol, scales):
            ((scale, factor),) = scales.items()
            name = protocol.protocol_id.removeprefix("test_probe_")
            self.states = np.column_stack(
                [
                    np.full(len(time), 0.5),
                    temperature(name, scale, factor),
                    np.zeros(len(time)),
                ]
            )

    def simulate(protocol, *, data_root, private_mechanism_scales, solver):
        return Run(protocol, private_mechanism_scales)

    monkeypatch.setattr(scoring.anchor, "reference", reference)
    monkeypatch.setattr(scoring, "simulate_phase_b", simulate)


def test_the_validity_rule_drops_only_failing_trajectories(tmp_path, monkeypatch):
    _toy_reactor(monkeypatch)
    plan = _toy_plan(tmp_path)
    verdict = scoring.validity(tmp_path, tmp_path)
    assert verdict["plan"] == plan["artifact_sha256"]
    assert verdict["kept"] == ["good"]
    assert verdict["dropped"] == [
        "broken_basal",
        "disagree",
        "edge",
        "negative",
        "orphan",
    ]
    reasons = {r["case"]: " ".join(r["reasons"]) for r in verdict["rows"]}
    assert "reaction_heat" in reasons["edge"]
    assert "positive" in reasons["negative"]
    assert "solver disagreement" in reasons["disagree"]
    assert "broken_basal" in reasons["orphan"]
    good = next(r for r in verdict["rows"] if r["case"] == "good")
    assert good["knife_edge"]["rse"] == pytest.approx(1e-4, rel=1e-6)
    with pytest.raises(ValueError, match="already applied"):
        scoring.validity(tmp_path, tmp_path)


def test_no_model_is_scored_before_the_validity_rule(tmp_path, monkeypatch):
    _toy_reactor(monkeypatch)
    _toy_plan(tmp_path)
    with pytest.raises(ValueError, match="before scoring a model"):
        scoring.score(tmp_path, tmp_path)
    (tmp_path / "subjects").mkdir()
    with pytest.raises(ValueError, match="scored before the validity rule"):
        scoring.validity(tmp_path, tmp_path)
