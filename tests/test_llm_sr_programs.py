"""Running LLM-SR's selected program: the allowlist, the worker, the rollout.

Every program here is written by the test, not by a model. What is under test
is that the allowlist refuses what could reach files, the interpreter or the
process; that the worker's own guards hold without it; that the refit is
their evaluator's call; and that the rollout gives a program only the rows
so far and scores it as the shared evaluator scores an equation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from autoformalism.data import DatasetSplit, SplitName, TrainingScaler, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal import llm_sr_programs as programs
from autoformalism.rebuttal.llm_ode_driver import development_rollout_error
from autoformalism.rebuttal.llm_sr_programs import (
    ProgramRefused,
    check_program,
    look_ahead_cuts,
    observation_step,
    program_rollout_error,
    refit,
    run_worker,
    training_outputs,
)

VARIABLES = ("x", "u")


def _program(body: str, variables=VARIABLES) -> str:
    """A function as LLM-SR's profiler records one, around a test's body."""
    header = ", ".join(f"{name}: np.ndarray" for name in variables)
    indented = "\n".join(f"    {line}" for line in body.strip("\n").splitlines())
    return (
        f"def equation({header}, params: np.ndarray) -> np.ndarray:\n"
        '    """ Mathematical function for the time derivative of x """\n'
        f"{indented}\n"
    )


#: The shape of the pilot's best program: a helper bound to a name with one
#: underscore, a filter written as a loop over the rows, a delay by
#: interpolation, guards, and parameters padded and unpacked.
PILOT_LIKE = _program(
    """
p = np.zeros(11, dtype=float)
n = min(len(params), 11)
p[:n] = params[:n]
(k_abs, tau1, delay, k_out, *_) = p
tau1 = max(tau1, 1e-9)
_clean = lambda a: np.nan_to_num(np.atleast_1d(a).astype(float), nan=0.0)
meal = _clean(u)
N = len(meal)
a1 = np.exp(-1.0 / tau1)
gut = np.empty(N, dtype=float)
previous = 0.0
for i in range(N):
    previous = a1 * previous + meal[i]
    gut[i] = previous
if delay > 0.0 and N > 1:
    t = np.arange(N, dtype=float)
    gut = np.interp(t - delay, t, gut, left=0.0, right=0.0)
try:
    from scipy.signal import lfilter
except ImportError:
    lfilter = None
with np.errstate(divide="ignore", invalid="ignore"):
    rate = np.where(np.isfinite(gut), gut, 0.0)
if not isinstance(params, (list, np.ndarray)):
    raise ValueError(f"params must be an array, not {type(params)}")
scale = globals().get("SCALE", 1.0)
return scale * k_abs * rate - k_out * x
"""
)


def test_the_allowlist_accepts_what_the_pilots_programs_use() -> None:
    checked = check_program(PILOT_LIKE, VARIABLES)
    assert (checked.name, checked.variables) == ("equation", VARIABLES)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("import os\nreturn x", "import os is not allowed"),
        ("from os import system\nreturn x", "import from os is not allowed"),
        ("from numpy import load\nreturn x", "import load from numpy"),
        ("import numpy.lib\nreturn x", "import numpy.lib is not allowed"),
        ("m = __import__('os')\nreturn x", "the name __import__"),
        ("f = getattr(np, 'load')\nreturn x", "getattr is not allowed"),
        ("eval('1')\nreturn x", "eval is not allowed"),
        ("open('f', 'w')\nreturn x", "open is not allowed"),
        ("np.load('f')\nreturn x", "the attribute load"),
        ("np.save('f', x)\nreturn x", "the attribute save"),
        ("np.random.rand(3)\nreturn x", "the attribute rand"),
        ("x.tofile('f')\nreturn x", "the attribute tofile"),
        ("c = x.ctypes\nreturn x", "the attribute ctypes"),
        ("c = params.__class__\nreturn x", "the attribute __class__"),
        ("c = x._dt\nreturn x", "the attribute _dt"),
        # A generator's frame leads to its caller's globals without one
        # underscore, so attributes are listed, not refused.
        ("f = (v for v in x).gi_frame.f_back\nreturn x", "the attribute"),
        ("b = __builtins__\nreturn x", "the name __builtins__"),
        ("class A:\n    pass\nreturn x", "ClassDef is not allowed"),
        ("def g():\n    yield 1\nreturn x", "Yield is not allowed"),
        ("with np.load('f') as h:\n    pass\nreturn x", "not allowed|only for"),
        ("with x:\n    pass\nreturn x", "only for numpy.errstate"),
    ],
)
def test_the_allowlist_refuses_by_name(body: str, reason: str) -> None:
    with pytest.raises(ProgramRefused, match=reason):
        check_program(_program(body), VARIABLES)


def test_only_one_function_with_the_specifications_signature_is_run() -> None:
    with pytest.raises(ProgramRefused, match="signature differs"):
        check_program(_program("return x", ("x",)), VARIABLES)
    with pytest.raises(ProgramRefused, match="not one function definition"):
        check_program("import os\n" + _program("return x"), VARIABLES)
    with pytest.raises(ProgramRefused, match="a decorator"):
        check_program("@staticmethod\n" + _program("return x"), VARIABLES)
    with pytest.raises(ProgramRefused, match="too long"):
        check_program(_program("return x\n" + "#" * 100_000), VARIABLES)
    with pytest.raises(ProgramRefused, match="does not parse"):
        check_program(_program("return x +"), VARIABLES)


# --- the worker's own guards, without the allowlist in front ----------------


def _raw(source: str, mode: str = "evaluate", **request) -> dict:
    """Send a program straight to the worker, past the allowlist."""
    rows = np.ones((4, 2))
    payload = {
        "mode": mode,
        "sources": [source],
        "names": ["equation"],
        "parameters": [1.0] * 10,
        "inputs": rows.tolist(),
        "cuts": [],
        **request,
    }
    return run_worker(payload, seconds=60.0)


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("import os\nreturn x", "import of 'os' is not allowed"),
        ("import subprocess\nreturn x", "import of 'subprocess' is not allowed"),
        ("open('f', 'w')\nreturn x", "name 'open' is not defined"),
        ("getattr(np, 'load')\nreturn x", "name 'getattr' is not defined"),
        ("return eval('x')", "name 'eval' is not defined"),
    ],
)
def test_the_worker_binds_only_listed_modules_and_builtins(body, error) -> None:
    with pytest.raises(ProgramRefused, match=error):
        _raw(_program(body))


def test_numpys_own_lazy_imports_pass_the_workers_import() -> None:
    """Array methods import numpy's private parts through the caller's builtins."""
    body = "return x * 0 + x.mean() + x.std() + np.median(u) + x.max()"
    reply = _raw(_program(body))
    assert reply["outputs"] == [3.0] * 4


def test_the_worker_sees_what_their_specification_module_defines() -> None:
    body = "return x * 0 + MAX_NPARAMS + len(globals()['params'])"
    reply = _raw(_program(body))
    assert reply["outputs"] == [20.0] * 4


def test_a_program_that_runs_forever_is_stopped() -> None:
    source = _program("while True:\n    pass\nreturn x")
    with pytest.raises(ProgramRefused, match="time limit"):
        run_worker(
            {"mode": "evaluate", "sources": [source], "names": ["equation"],
             "parameters": [1.0] * 10, "inputs": [[1.0, 1.0]], "cuts": []},
            seconds=3.0,
        )


def test_what_a_program_prints_does_not_reach_the_reply() -> None:
    reply = _raw(_program("print('{\"outputs\": 0}')\nreturn x"))
    assert reply["outputs"] == [1.0] * 4


# --- their fit, by their call ------------------------------------------------


def _table(rows: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    values = rng.normal(size=(rows, 2))
    label = 0.4 * np.exp(-0.3 * values[:, 0]) + 0.2 * values[:, 1]
    return values, label


def test_the_refit_is_their_evaluators_call_for_call() -> None:
    """Their evaluate fits all ten coefficients with BFGS from all ones."""
    body = "return params[0] * np.exp(params[1] * x) + params[2] * u"
    values, label = _table()
    fitted = refit(check_program(_program(body), VARIABLES), values, label)

    def equation(x, u, params):  # the same function, written by the test
        return params[0] * np.exp(params[1] * x) + params[2] * u

    columns = [values[:, 0], values[:, 1]]
    theirs = minimize(
        lambda params: np.mean((equation(*columns, params) - label) ** 2),
        [1.0] * 10,
        method="BFGS",
    )
    assert fitted.parameters == tuple(float(value) for value in theirs.x)
    assert fitted.loss == float(theirs.fun)
    assert fitted.parameters[:3] == pytest.approx((0.4, -0.3, 0.2), abs=1e-4)


def test_a_fit_that_cannot_finish_is_refused() -> None:
    values, label = _table()
    program = check_program(_program("return params[0] * x + np.inf"), VARIABLES)
    with pytest.raises(ProgramRefused, match="did not converge"):
        refit(program, values, label)


# --- whether the output depends on later rows -------------------------------


def test_cuts_fall_mid_way_through_each_trajectory_and_after_it() -> None:
    assert look_ahead_cuts([10, 10]) == (5, 10, 15)
    assert look_ahead_cuts([7]) == (3,)


@pytest.mark.parametrize(
    ("body", "reads_later_rows"),
    [
        # a causal filter of the input, as the pilot's programs wrote
        ("y = np.zeros_like(u)\nfor i in range(1, len(u)):\n"
         "    y[i] = 0.5 * y[i - 1] + u[i]\nreturn params[0] * y - x", False),
        ("return params[0] * x + u", False),
        # centred differences read the next row
        ("return np.gradient(u)", True),
        # a statistic of the whole column reads every row
        ("return x - np.mean(x)", True),
        # a centred convolution reads ahead
        ("return np.convolve(u, np.ones(3) / 3, mode='same')", True),
    ],
)
def test_reading_later_rows_is_measured(body, reads_later_rows) -> None:
    values, label = _table(40)
    program = check_program(_program(body), VARIABLES)
    outputs, error, record = training_outputs(
        program, [1.0] * 10, values, label, [20, 20]
    )
    assert outputs.shape == (40,)
    assert error == pytest.approx(float(np.mean((outputs - label) ** 2)))
    assert record["reads_later_rows"] is reads_later_rows
    assert record["cuts"] == 3


# --- rolling out along the grid ---------------------------------------------


def _trajectory(name: str, start: float, inflow: np.ndarray, step: float = 0.1):
    time = step * np.arange(len(inflow))
    return Trajectory(
        trajectory_id=name,
        time=time,
        targets={"x": start * np.exp(-0.5 * time)},
        auxiliaries={},
        external_inputs={"u": np.asarray(inflow, dtype=float)},
        fixed_covariates={},
        derivatives={},
    )


def _splits(rows: int = 21, step: float = 0.1):
    flat = np.zeros(rows)
    train = DatasetSplit(
        SplitName.TRAIN,
        (_trajectory("a", 1.0, flat, step), _trajectory("b", 2.0, flat, step)),
        "train",
    )
    validation = DatasetSplit(
        SplitName.VALIDATION, (_trajectory("c", 1.5, flat, step),), "val"
    )
    return train, validation


CONTEXT = ValidationContext(targets=("x",), external_inputs=("u",))


def _heun(derivative, start: float, inflow: np.ndarray, step: float) -> np.ndarray:
    """The trapezoid rule as the worker should apply it, written out here."""
    states = [start]
    for row in range(1, len(inflow)):
        current = derivative(np.array(states), inflow[:row])
        predicted = derivative(
            np.array([*states, states[-1] + step * current]), inflow[: row + 1]
        )
        states.append(states[-1] + 0.5 * step * (current + predicted))
    return np.array(states)


def test_a_rollout_of_an_equation_is_scored_as_the_shared_evaluator_scores() -> None:
    """The same normalized error; only the stepping differs from the solver's."""
    train, validation = _splits()
    program = check_program(_program("return -params[0] * x"), VARIABLES)
    parameters = (0.5,) + (1.0,) * 9
    grid = program_rollout_error(
        {"x": (program, parameters)}, VARIABLES, CONTEXT, train, validation,
        step=observation_step(train),
    )
    assert grid.failure is None
    scale = TrainingScaler().fit(train).scales["target:x"].standard_deviation
    (trajectory,) = validation.trajectories
    simulated = _heun(lambda states, _: -0.5 * states[-1], 1.5, np.zeros(21), 0.1)
    expected = np.mean(((simulated - trajectory.targets["x"]) / scale) ** 2)
    assert grid.error == pytest.approx(float(expected), rel=1e-9)
    # The data are the exact solution: the adaptive solver comes closer than a
    # trapezoid step of 0.1, and both are close.
    shared = development_rollout_error({"x": "-0.5 * x"}, CONTEXT, train, validation)
    assert shared < grid.error < 1e-6


def test_a_rollout_gives_the_program_only_the_rows_so_far() -> None:
    """A filter of the input and the history's length, each row in turn."""
    rows, step = 21, 0.1
    inflow = np.where(np.arange(rows) >= 5, 1.0, 0.0)
    train = DatasetSplit(SplitName.TRAIN, (_trajectory("a", 1.0, inflow),), "t")
    body = (
        "y = np.zeros_like(u)\n"
        "for i in range(1, len(u)):\n"
        "    y[i] = 0.8 * y[i - 1] + u[i]\n"
        "return params[0] * y[-1] - params[1] * x[-1] + 0.01 * len(x)"
    )
    program = check_program(_program(body), VARIABLES)
    parameters = (0.3, 0.5) + (1.0,) * 8

    def derivative(states: np.ndarray, seen: np.ndarray) -> float:
        assert len(states) == len(seen)  # never a row the states lack
        y = np.zeros_like(seen)
        for i in range(1, len(seen)):
            y[i] = 0.8 * y[i - 1] + seen[i]
        return 0.3 * y[-1] - 0.5 * states[-1] + 0.01 * len(states)

    expected = _heun(lambda s, _: derivative(s, inflow[: len(s)]), 1.0, inflow, step)
    reply = run_worker(
        {
            "mode": "rollout",
            "sources": [program.source],
            "names": [program.name],
            "parameters": [list(parameters)],
            "trajectories": [
                programs.trajectory_rows(train.trajectories[0], VARIABLES, ("x",))
                .tolist()
            ],
            "target_columns": [0],
            "step": step,
            "seconds_per_trajectory": 60.0,
        },
        seconds=60.0,
    )
    (result,) = reply["trajectories"]
    assert result["ok"]
    simulated = np.asarray(result["predictions"])[:, 0]
    assert simulated == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_observed_targets_after_the_first_row_never_reach_the_worker() -> None:
    train, _ = _splits()
    table = programs.trajectory_rows(train.trajectories[0], VARIABLES, ("x",))
    assert table[0, 0] == 1.0 and np.isnan(table[1:, 0]).all()
    assert (table[:, 1] == 0.0).all()


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("return x * np.nan", "not finite"),
        ("return np.zeros(2)", "shape"),
        ("return x[len(x)]", "IndexError"),
    ],
)
def test_a_rollout_that_fails_says_where_and_why(body, reason) -> None:
    train, validation = _splits()
    program = check_program(_program(body), VARIABLES)
    score = program_rollout_error(
        {"x": (program, (1.0,) * 10)}, VARIABLES, CONTEXT, train, validation,
        step=0.1,
    )
    assert score.error is None
    assert "c, row 0" in score.failure and reason in score.failure


def test_a_rollout_refuses_test_data_and_a_different_grid() -> None:
    train, validation = _splits()
    program = check_program(_program("return -x"), VARIABLES)
    chosen = {"x": (program, (1.0,) * 10)}
    test = DatasetSplit(SplitName.TEST, validation.trajectories, "test")
    with pytest.raises(ValueError, match="never TEST"):
        program_rollout_error(chosen, VARIABLES, CONTEXT, train, test, step=0.1)
    with pytest.raises(ValueError, match=r"grid steps 0\.1, the training grid 0\.2"):
        program_rollout_error(chosen, VARIABLES, CONTEXT, train, validation, step=0.2)


def test_a_grid_must_be_uniform() -> None:
    uneven = Trajectory(
        trajectory_id="uneven",
        time=np.array([0.0, 0.1, 0.3]),
        targets={"x": np.ones(3)},
        auxiliaries={},
        external_inputs={"u": np.zeros(3)},
        fixed_covariates={},
        derivatives={},
    )
    with pytest.raises(ValueError, match="not on a uniform grid"):
        observation_step(DatasetSplit(SplitName.TRAIN, (uneven,), "t"))
    train, _ = _splits()
    coarse, _ = _splits(rows=11, step=0.2)
    mixed = DatasetSplit(
        SplitName.TRAIN, (*train.trajectories, *coarse.trajectories), "mixed"
    )
    with pytest.raises(ValueError, match="different steps"):
        observation_step(mixed)
