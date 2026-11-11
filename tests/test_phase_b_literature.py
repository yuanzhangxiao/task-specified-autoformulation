"""Literature models on the nine-case Phase B comparison.

Each model is written once in the restricted grammar. What is under test: the
fit's vectorized evaluation agrees with the shared scalar evaluator, the fast
rollout scores as the shared evaluator does, the published model's basal
state is a steady state, each pair predicts exactly its case's targets from
its public channels, a fit recovers a model that generated its own data, and
the report selects on validation among starts whose replay matched.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from autoformalism.data import BenchmarkRegistry, DatasetSplit, Trajectory
from autoformalism.data.models import SplitName
from autoformalism.expressions import ValidationContext
from autoformalism.expressions.compiler import PiecewiseLinearForcing
from autoformalism.rebuttal import phase_b_literature as lit
from autoformalism.rebuttal import phase_b_nine_case as nine

ROLES = {
    "dm07_t1_easy": ("Gp",),
    "dm07_t1_hard": ("Gp",),
    "dm07_t2_easy": ("Gp", "I", "U"),
    "dm07_t2_hard": ("Gp", "I"),
    "bergman": ("Gp", "I"),
    "cstr": ("T",),
}


def _structure(model: str) -> lit.Structure:
    if model.startswith("dm07"):
        return lit.dm07(model)
    return lit.bergman() if model == "bergman" else lit.cstr(40.0)


def _split(name: SplitName, trajectories: list[Trajectory]) -> DatasetSplit:
    return DatasetSplit(name, tuple(trajectories), f"synthetic-{name.value}")


def _trajectory(identifier: str, time, targets, auxiliaries, inputs) -> Trajectory:
    def arrays(values):
        return {key: np.asarray(value, dtype=float) for key, value in values.items()}

    return Trajectory(
        trajectory_id=identifier,
        time=np.asarray(time, dtype=float),
        targets=arrays(targets),
        auxiliaries=arrays(auxiliaries),
        external_inputs=arrays(inputs),
        fixed_covariates={},
        derivatives={},
    )


def _t1_easy_cell() -> tuple[DatasetSplit, DatasetSplit, ValidationContext]:
    """Smooth synthetic T1-easy channels; one trajectory eats at t = 0."""
    time = np.arange(31.0)

    def make(identifier: str, offset: float, meal_at: int | None) -> Trajectory:
        meal = np.zeros_like(time)
        if meal_at is not None:
            meal[meal_at] = 60.0
        return _trajectory(
            identifier,
            time,
            {"Gp": 172.0 + offset + 8 * np.sin(time / 9)},
            {
                "EGP": 1.9 - 0.01 * time + 0.005 * offset,
                "Uii": np.ones_like(time),
                "E": np.zeros_like(time),
                "Gt": 130.0 + 2 * offset + 3 * np.cos(time / 7),
            },
            {lit.MEAL: meal},
        )

    train = _split(
        SplitName.TRAIN,
        [
            make("train_000", 0.0, 0),
            make("train_001", 4.0, 12),
            make("train_002", -3.0, None),
        ],
    )
    validation = _split(SplitName.VALIDATION, [make("validation_000", 2.0, 5)])
    context = ValidationContext(
        targets=("Gp",),
        auxiliaries=("EGP", "Uii", "E", "Gt"),
        external_inputs=(lit.MEAL,),
    )
    return train, validation, context


@pytest.mark.parametrize("model", sorted(ROLES))
def test_the_fits_vector_evaluation_matches_the_shared_scalar_evaluator(model):
    structure = _structure(model)
    roles = dict.fromkeys(structure.observations, "targets")
    symbols = structure.symbols()
    computed = set(structure.states) | {name for name, _ in structure.processes}
    for name in ("EGP", "Uii", "E", "Gt", "C", "Tj"):
        if name in symbols and name not in computed:
            roles[name] = "auxiliaries"
    for name in (lit.MEAL, lit.INSULIN, lit.DOSE, "Tf"):
        if name in symbols:
            roles[name] = "external_inputs"
    compiled = lit._compile(
        structure,
        lit._bare_context(roles),
        initials=dict.fromkeys(structure.latent, "0"),
    )
    fast = lit.FastModel.of(compiled)
    rng = np.random.default_rng(7)
    count = 5
    states = {state: rng.uniform(1.0, 300.0, count) for state in fast.states}
    if "T" in states:
        states["T"] = rng.uniform(340.0, 380.0, count)
    channels = {name: rng.uniform(0.0, 2.0, count) for name in fast.forcing}
    for name in (lit.DOSE, "Gt", "Tf", "Tj"):
        if name in channels:
            channels[name] = rng.uniform(100.0, 400.0, count)
    parameters = {
        name: value * rng.uniform(0.8, 1.2) for name, value in structure.starts.items()
    }
    vector = fast.derivatives(0.5, states, channels, parameters, count)
    for column in range(count):
        forcing = PiecewiseLinearForcing(
            [0.0, 1.0],
            {name: [values[column]] * 2 for name, values in channels.items()},
            allowed_channels=compiled.validated.forcing_symbols,
        )
        scalar = compiled.rhs(
            0.5,
            [states[state][column] for state in fast.states],
            parameters,
            forcing,
        )
        np.testing.assert_allclose(vector[:, column], scalar, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("pair", lit.PAIRS, ids=lambda pair: pair.key)
def test_each_pair_models_exactly_its_cases_targets_from_public_channels(pair):
    spec = BenchmarkRegistry().get(pair.benchmark_id)
    tier = nine.load_inputs().cells[pair.benchmark_id].tier
    roles = spec.tier_roles[tier]
    structure = _structure(pair.model)
    assert tuple(sorted(structure.observations)) == tuple(sorted(roles.targets))
    assert tuple(sorted(roles.targets)) == tuple(sorted(ROLES[pair.model]))
    public = set(roles.targets) | set(roles.auxiliaries) | set(spec.external_inputs)
    read = structure.symbols() - set(structure.states) - set(structure.starts)
    read -= {name for name, _ in structure.processes} | {lit.DOSE}
    assert read <= public, read - public
    for name, start in structure.starts.items():
        lower, upper = lit.bounds(name, start, structure.fractions)
        assert lower == pytest.approx(start / 10)
        assert upper == pytest.approx(
            min(start * 10, lit.FRACTION_CEILING)
            if name in structure.fractions
            else start * 10
        )
        if pair.model.startswith("dm07"):
            assert start == lit.DM07[name]


def test_the_nine_cases_are_each_covered_or_declared_not_applicable():
    covered = {pair.benchmark_id for pair in lit.PAIRS} | set(lit.NOT_APPLICABLE)
    assert covered == set(nine.load_inputs().cells)
    assert not {pair.benchmark_id for pair in lit.PAIRS} & set(lit.NOT_APPLICABLE)


def test_supplied_auxiliaries_replace_their_own_equations():
    for variant, supplied in lit.DM07_SUPPLIED.items():
        structure = lit.dm07(variant)
        processes = {name for name, _ in structure.processes}
        assert not supplied & (set(structure.states) | processes)
    assert "Gt" in lit.dm07("dm07_t2_hard").latent
    assert {"I1", "Id"} <= set(lit.dm07("dm07_t1_hard").latent)
    assert "I" in lit.dm07("dm07_t1_hard").latent


def test_the_dose_is_the_last_meal_in_mg_and_never_looks_ahead():
    meal = np.array([0.0, 0.0, 30.0, 0.0, 0.0, 45.0, 0.0])
    assert lit.dose_series(meal).tolist() == [0, 0, 30000, 30000, 30000, 45000, 45000]
    assert lit.dose_series(meal[:4]).tolist() == lit.dose_series(meal)[:4].tolist()


def test_the_published_basal_state_is_a_steady_state():
    basal = lit.dm07_basal()
    assert basal["I1"] == pytest.approx(basal["I"])
    assert basal["Id"] == pytest.approx(basal["I"])
    assert basal["X"] == pytest.approx(basal["I"] - lit.DM07["Ib"])
    assert basal["Y"] == pytest.approx(
        lit.DM07["beta"] * (basal["Gp"] / lit.VG - lit.DM07["Gb"])
    )
    assert basal["U"] == pytest.approx(1.0 + basal["Uid"])
    assert basal["Qsto1"] == basal["Qsto2"] == basal["Qgut"] == 0.0


def test_the_fast_rollout_scores_as_the_shared_evaluator_does():
    train, validation, context = _t1_easy_cell()
    pair = lit.PAIR_BY_KEY["dm07_t1_easy_canonical"]
    prepared = lit.assemble(pair, train, validation, context)
    assert set(prepared.description["readings"]) == {"Gp", "EGP", "Gt", lit.MEAL}
    rng = np.random.default_rng(3)
    parameters = {
        **{
            name: value * rng.uniform(0.7, 1.3)
            for name, value in prepared.structure.starts.items()
        },
        **{name: rng.normal(0.0, 0.2) for name in prepared.weights},
    }
    for split, stack in (
        (prepared.train, prepared.train_stack),
        (prepared.validation, prepared.validation_stack),
    ):
        shared = lit.score(prepared, split, parameters)
        assert shared["failed_trajectories"] == []
        fast = lit.fast_nmse(prepared, parameters, stack)
        assert fast == pytest.approx(shared["normalized_mse"], rel=1e-6)


def test_the_meal_at_the_first_sample_enters_the_initial_map():
    train, validation, context = _t1_easy_cell()
    prepared = lit.assemble(
        lit.PAIR_BY_KEY["dm07_t1_easy_canonical"], train, validation, context
    )
    expression = prepared.description["initial_conditions"]["Qsto1"]
    assert expression.startswith("(0.0) + ")
    assert f"w_Qsto1_{lit.MEAL} * ({lit.MEAL} - (0.0))" in expression


def _cstr_cell(
    truth: dict[str, float],
) -> tuple[DatasetSplit, DatasetSplit, ValidationContext]:
    """CSTR data generated by the textbook balance itself."""
    time = np.round(np.arange(0.0, 6.05, 0.1), 10)
    context = ValidationContext(
        targets=("T",), auxiliaries=("C", "Tj"), external_inputs=("Cf", "Tf", "Tjf")
    )
    structure = lit.cstr(truth["r_ref"])
    compiled = lit._compile(
        structure,
        lit._bare_context(
            {
                "T": "targets",
                "C": "auxiliaries",
                "Tj": "auxiliaries",
                "Tf": "external_inputs",
            }
        ),
        initials={},
    )
    fast = lit.FastModel.of(compiled)

    def make(split_name: str, index: int) -> Trajectory:
        channels = {
            "C": 0.25 + 0.02 * index + 0.01 * time,
            "Tj": 346.0 + index - 0.3 * time,
            "Tf": np.full_like(time, 349.0 + index),
        }
        stack = lit.Stack(
            ("x",),
            time,
            {
                **{k: v[None, :] for k, v in channels.items()},
                "T": np.full((1, len(time)), 355.0 + 2 * index),
            },
        )
        temperature = fast.rollout(stack, truth, max_step=0.1)["T"][0]
        return _trajectory(
            f"{split_name}_{index:03d}",
            time,
            {"T": temperature},
            {"C": channels["C"], "Tj": channels["Tj"]},
            {
                "Cf": np.ones_like(time),
                "Tf": channels["Tf"],
                "Tjf": np.full_like(time, 330.0),
            },
        )

    train = _split(SplitName.TRAIN, [make("train", index) for index in range(4)])
    validation = _split(SplitName.VALIDATION, [make("validation", 5)])
    return train, validation, context


def test_a_fit_recovers_a_model_that_generated_its_data():
    truth = {"q": 1.3, "h": 0.8, "ER": 9000.0, "r_ref": 8.0}
    train, validation, context = _cstr_cell(truth)
    prepared = lit.assemble(lit.PAIR_BY_KEY["cstr_balance"], train, validation, context)
    result = lit.fit(prepared, 0, wall_seconds=120)
    assert result["replay"]["matches"]
    assert result["train"]["normalized_mse"] < 1e-6
    assert result["validation"]["normalized_mse"] < 1e-5
    assert result["parameters"]["q"] + result["parameters"]["h"] == pytest.approx(
        truth["q"] + truth["h"], rel=1e-2
    )


def test_perturbed_starts_stay_in_bounds_and_differ_by_seed():
    train, validation, context = _cstr_cell(
        {"q": 1.0, "h": 1.0, "ER": 1e4, "r_ref": 5.0}
    )
    prepared = lit.assemble(lit.PAIR_BY_KEY["cstr_balance"], train, validation, context)
    vectorizer = lit.Vectorizer.of(prepared)
    lower, upper = vectorizer.lower_upper()
    starts = [
        vectorizer.start(prepared.structure.starts, index) for index in lit.STARTS
    ]
    assert np.allclose(
        np.exp(starts[0]), [prepared.structure.starts[n] for n in vectorizer.names]
    )
    assert not np.allclose(starts[1], starts[2])
    for start in starts:
        assert (start >= lower).all() and (start <= upper).all()
        assert np.abs(start - starts[0]).max() <= lit.PERTURBATION + 1e-12


def _seal(path, start: int, validation: float, matches: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "start": start,
                "train": {"normalized_mse": validation / 2},
                "validation": {"normalized_mse": validation, "per_target": {"Gp": 1.0}},
                "replay": {"matches": matches},
                "least_squares": {"status": 2},
                "fit_seconds": 1.0,
            }
        )
    )


def test_the_report_selects_on_validation_among_starts_whose_replay_matched(tmp_path):
    directory = tmp_path / "results" / "bergman_t2_hard"
    _seal(directory / "start_0.json", 0, 0.10, True)
    _seal(directory / "start_1.json", 1, 0.05, False)
    _seal(directory / "start_2.json", 2, 0.08, True)
    summary = lit.report(tmp_path)
    row = next(row for row in summary["rows"] if row["key"] == "bergman_t2_hard")
    assert row["selected_start"] == 2
    assert row["validation_nmse"] == 0.08
    assert row["complete"] is True
    assert summary["status"] == "pending"
    assert (tmp_path / "summary.json").exists()
    assert ("bergman_t2_hard", "start_0") not in lit.jobs(tmp_path)
    assert ("dm07_t2_hard", "unfitted") in lit.jobs(tmp_path)


def test_an_infinite_score_is_kept_when_no_rollout_finishes():
    train, validation, context = _cstr_cell(
        {"q": 1.0, "h": 1.0, "ER": 1e4, "r_ref": 5.0}
    )
    prepared = lit.assemble(lit.PAIR_BY_KEY["cstr_balance"], train, validation, context)
    runaway = {"q": 0.1, "h": 0.1, "ER": 1e5, "r_ref": 50.0}
    scored = lit.score(prepared, prepared.validation, runaway)
    assert math.isinf(scored["normalized_mse"]) or scored["normalized_mse"] > 1e6
