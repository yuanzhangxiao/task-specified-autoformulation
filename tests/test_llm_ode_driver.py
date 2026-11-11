"""Driving the pinned LLM-ODE checkout over one Phase-B cell.

The checkout is not available here, so upstream is replaced by a fake that
reproduces the parts the driver actually touches: a per-variable searcher with
`step` and `get_pareto_frontier`, and a module-level `generate_prompt` the
driver extends. What is under test is our binding, not their search.
"""

from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from autoformalism.rebuttal import llm_ode_driver as driver
from autoformalism.rebuttal.llm_ode_campaign import CellArrays, SearcherFactory


class _Program:
    def __init__(self, equation: str) -> None:
        self.equation = equation


class _FakeSearcher:
    """Returns a fixed frontier; records how many iterations it was stepped."""

    def __init__(self, frontier: list[str], **kwargs) -> None:
        self.frontier = frontier
        self.steps = 0
        self.kwargs = kwargs

    def step(self) -> None:
        self.steps += 1

    def get_pareto_frontier(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "complexity": list(range(len(self.frontier))),
                "mse_val": [1.0 / (index + 1) for index in range(len(self.frontier))],
                "program": [_Program(item) for item in self.frontier],
            }
        )


class _Raw:
    """What the SDK's raw-response call returns: the headers and the reply."""

    def __init__(self, reply, headers: dict | None = None) -> None:
        import httpx

        self.headers = httpx.Headers(headers or {})
        self._reply = reply

    def parse(self):
        return self._reply


def _client(create) -> types.SimpleNamespace:
    """An SDK client whose raw-response `responses.create` is `create`."""
    return types.SimpleNamespace(
        responses=types.SimpleNamespace(
            with_raw_response=types.SimpleNamespace(create=create)
        )
    )


class _FakeLlm:
    """Stands in for upstream's client, including what the driver wraps."""

    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.n_queries = 7
        self.model_name = "m"
        self.request_kwargs = {"max_output_tokens": 1024}  # upstream's
        self.client = _client(
            lambda **kwargs: _Raw(
                types.SimpleNamespace(output_text="x_0", usage={"total_tokens": 11})
            )
        )

    def make_request(self, prompt):
        return "x_0"


def _install_fake_upstream(monkeypatch, frontiers: dict[str, list[str]]):
    """Stand in for the pinned checkout, including its module-level prompt."""
    module = types.ModuleType("llmode.llmode")
    module.generate_prompt = lambda examples_str, n_new, n_variables: [
        {"role": "system", "content": "upstream instructions"},
        {"role": "user", "content": "\n".join(examples_str)},
    ]
    package = types.ModuleType("llmode")
    monkeypatch.setitem(sys.modules, "llmode", package)
    monkeypatch.setitem(sys.modules, "llmode.llmode", module)

    order = list(frontiers)
    built: list[_FakeSearcher] = []

    def equation_searcher(**kwargs):
        searcher = _FakeSearcher(frontiers[order[len(built)]], **kwargs)
        built.append(searcher)
        return searcher

    monkeypatch.setattr(
        driver,
        "load_upstream",
        lambda root: types.SimpleNamespace(
            equation_searcher=equation_searcher,
            llm=_FakeLlm,
            system=object,
            generate_prompt=module.generate_prompt,
        ),
    )
    return module, built


def _arrays(channels: tuple[str, ...]) -> CellArrays:
    import numpy as np

    samples = 10
    return CellArrays(
        time=np.linspace(0.0, 1.0, samples),
        states=np.ones((samples, len(channels))),
        derivatives=np.ones((samples, len(channels))),
        channels=channels,
        trajectory_bounds=((0, samples),),
    )


def _scoring(monkeypatch, error=lambda equations: 0.5) -> None:
    """Stand in for the development rollouts: `error` gives a system's error.

    `error` returns None for a system whose rollout fails and may raise
    InexpressibleEquation for one the grammar refuses, as the real scorer
    finds them. The searcher passes no development data, so none is read.
    """

    def score(system, development, context, channels, *, seconds, bound=None):
        targets = channels[: len(system)]
        try:
            equations = driver.to_state_equations(system, channels, targets)
            value = error(equations)
        except driver.InexpressibleEquation as exc:
            return driver.SystemScore("inexpressible", functions=exc.functions)
        if value is None:
            return driver.SystemScore("rollout_failed", equations=equations)
        return driver.SystemScore(
            "scored",
            equations=equations,
            train_val_rollout_mse=value,
            training_rollout_error=value / 2,
            validation_rollout_error=value,
        )

    monkeypatch.setattr(driver, "score_system", score)


def test_the_driver_accepts_exactly_what_the_campaign_passes(monkeypatch) -> None:
    """The searcher and the caller must not drift apart.

    run() calls the searcher by keyword. When the driver grew the arguments it
    needs for development scoring, nothing tied the two together and the
    protocol silently described a signature no longer in use.
    """
    _install_fake_upstream(monkeypatch, {"y": ["x_0"]})
    search = driver.build_searcher(
        upstream_root=Path("/nonexistent"),
        base_url="http://127.0.0.1:1/v1",
        iterations=1,
        islands=2,
    )
    declared = inspect.signature(SearcherFactory.__call__).parameters
    actual = inspect.signature(search).parameters
    assert set(actual) == {name for name in declared if name != "self"}
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in actual.values()
    )


def test_every_target_is_stepped_for_the_declared_iterations(
    monkeypatch, tmp_path: Path
) -> None:
    """Upstream advances all equations once per iteration; so must we."""
    _install_fake_upstream(monkeypatch, {"y": ["x_0"], "w": ["x_1"]})
    _scoring(monkeypatch, lambda equations: 0.5)
    search = driver.build_searcher(
        upstream_root=Path("/nonexistent"),
        base_url="http://127.0.0.1:1/v1",
        iterations=3,
        islands=2,
    )
    outcome = search(
        train=_arrays(("y", "w")),
        validation=_arrays(("y", "w")),
        targets=("y", "w"),
        prompt="",
        directory=tmp_path,
        development=(object(), object()),
        context=object(),
    )
    assert outcome["status"] == "complete"
    assert outcome["equations"] == {"y": "y", "w": "w"}
    assert outcome["accounting"]["llm_queries"] == 7


def test_the_specification_is_installed_and_then_removed(
    monkeypatch, tmp_path: Path
) -> None:
    """A leaked patch would silently alter any later search in the process."""
    module, _ = _install_fake_upstream(monkeypatch, {"y": ["x_0"]})
    original = module.generate_prompt
    _scoring(monkeypatch, lambda equations: 1.0)
    seen: list[str] = []

    class _Recording(_FakeSearcher):
        def step(self) -> None:
            super().step()
            seen.append(module.generate_prompt([], 1, 1)[-1]["content"])

    monkeypatch.setattr(driver, "_frontier_equations", lambda s: ("x_0",))
    search = driver.build_searcher(
        upstream_root=Path("/nonexistent"),
        base_url="http://127.0.0.1:1/v1",
        iterations=1,
        islands=1,
    )
    search(
        train=_arrays(("y",)),
        validation=_arrays(("y",)),
        targets=("y",),
        prompt="\n\nTask specification: recover dy/dt.",
        directory=tmp_path,
        development=(object(), object()),
        context=object(),
    )
    assert module.generate_prompt is original


def test_an_inexpressible_sweep_is_reported_with_the_operator_named(
    monkeypatch, tmp_path: Path
) -> None:
    """The cost of our restricted grammar must be countable, not estimated.

    The baseline grammar reads every function SymPy and numpy offer, so the
    name here is one nothing defines; the counting mechanism is under test.
    """
    _install_fake_upstream(
        monkeypatch, {"y": ["besselj(x_0)", "besselj(x_0) + x_0"]}
    )
    _scoring(monkeypatch, lambda equations: 0.1)
    search = driver.build_searcher(
        upstream_root=Path("/nonexistent"),
        base_url="http://127.0.0.1:1/v1",
        iterations=1,
        islands=1,
    )
    outcome = search(
        train=_arrays(("y",)),
        validation=_arrays(("y",)),
        targets=("y",),
        prompt="",
        directory=tmp_path,
        development=(object(), object()),
        context=object(),
    )
    assert outcome["status"] == "inexpressible"
    assert outcome["accounting"]["inexpressible_operators"] == ["besselj"]
    assert outcome["accounting"]["inexpressible_systems"] == 2
    assert outcome["accounting"]["scored_systems"] == 0
    assert "besselj" in outcome["error"]


def test_a_large_product_is_scored_whole_as_upstream_scores_it(
    monkeypatch, tmp_path: Path
) -> None:
    """Upstream has no cap; the old one would have lost a finished search.

    30 x 30 x 12 systems exceed the 10,000 the adapter used to refuse past.
    """
    sizes = {"a": 30, "b": 30, "c": 12}
    wide = {
        name: [f"x_0 + {index}" for index in range(size)]
        for name, size in sizes.items()
    }
    _install_fake_upstream(monkeypatch, wide)
    # The last system in product order is the best one.
    _scoring(
        monkeypatch, lambda equations: 2.0 - float(equations["c"].split()[-1]) / 11
    )
    search = driver.build_searcher(
        upstream_root=Path("/nonexistent"),
        base_url="http://127.0.0.1:1/v1",
        iterations=1,
        islands=1,
    )
    outcome = search(
        train=_arrays(("a", "b", "c")),
        validation=_arrays(("a", "b", "c")),
        targets=("a", "b", "c"),
        prompt="",
        directory=tmp_path,
        development=(object(), object()),
        context=object(),
    )
    assert outcome["status"] == "complete"
    assert outcome["accounting"]["systems"] == 30 * 30 * 12
    assert outcome["accounting"]["scored_systems"] == 30 * 30 * 12
    # Best derivative fits first, so the first of the tied best systems is
    # the one built from each frontier's first equation.
    assert outcome["equations"] == {"a": "a + 29", "b": "a + 29", "c": "a + 11"}
    assert outcome["train_val_rollout_mse"] == 1.0
    record = json.loads((tmp_path / "systems.json").read_text())
    assert record["selection_metric"] == "train_val_rollout_mse"
    assert len(record["systems"]) == 30 * 30 * 12


def test_selection_refuses_a_held_out_split() -> None:
    """The selection metric must never be computable against test data."""
    from autoformalism.data import SplitName

    class _Split:
        def __init__(self, name) -> None:
            self.name = name

    with pytest.raises(ValueError, match="never TEST"):
        driver.development_rollout_error(
            {},
            object(),
            _Split(SplitName.TRAIN),
            _Split(SplitName.TEST),
        )


def test_an_interpreter_older_than_upstream_requires_is_refused() -> None:
    """A two-hour job that discovers nothing must not be possible silently.

    Upstream pins Python 3.13.5 and builds programs with
    str.replace(count=1). On 3.12 every construction raises TypeError, which
    upstream logs as a warning and continues past, so the islands stay empty
    for the whole run and the job dies at its walltime having produced
    nothing.
    """
    from autoformalism.rebuttal import llm_ode_upstream

    with (
        mock.patch.object(llm_ode_upstream.sys, "version_info", (3, 12, 10)),
        pytest.raises(RuntimeError, match="requires Python"),
    ):
        llm_ode_upstream.load_upstream(Path("/nonexistent"))


def test_a_search_that_never_produces_a_candidate_stops_early(
    monkeypatch, tmp_path: Path
) -> None:
    """Whatever the cause, an empty search must not spend the allocation."""
    _install_fake_upstream(monkeypatch, {"y": []})
    _scoring(monkeypatch, lambda equations: 1.0)
    search = driver.build_searcher(
        upstream_root=Path("/nonexistent"),
        base_url="http://127.0.0.1:1/v1",
        iterations=200,
        islands=4,
    )
    outcome = search(
        train=_arrays(("y",)),
        validation=_arrays(("y",)),
        targets=("y",),
        prompt="",
        directory=tmp_path,
        development=(object(), object()),
        context=object(),
    )
    assert outcome["status"] == "no_candidates"
    assert "not running" in outcome["error"]
    # stopped at the barren threshold rather than running all 200
    assert outcome["accounting"]["llm_queries"] == 7


def test_what_the_baseline_grammar_still_refuses_is_named() -> None:
    """A fitted exponent is read now; a comparison is still not an expression."""
    from autoformalism.data import SplitName
    from autoformalism.expressions import ValidationContext

    class _Split:
        def __init__(self, name) -> None:
            self.name = name

    context = ValidationContext(
        targets=("y",), auxiliaries=(), external_inputs=(), fixed_covariates=()
    )
    with pytest.raises(driver.InexpressibleEquation) as refused:
        driver.development_rollout_error(
            {"y": "-0.4*y**0.73 * (y > 1.0)"},
            context,
            _Split(SplitName.TRAIN),
            _Split(SplitName.VALIDATION),
        )
    assert refused.value.functions == ("UNSUPPORTED_SYNTAX",)


def test_a_refused_system_is_counted_and_the_search_is_kept(
    monkeypatch, tmp_path: Path
) -> None:
    """One unreadable frontier member used to discard all 18 paid-for searches."""
    _install_fake_upstream(monkeypatch, {"y": ["x_0*(x_0 > 1)", "x_0"]})

    def rollout(equations):
        if ">" in equations["y"]:
            raise driver.InexpressibleEquation(equations["y"], ("UNSUPPORTED_SYNTAX",))
        return 0.3

    _scoring(monkeypatch, rollout)
    search = driver.build_searcher(
        upstream_root=Path("/nonexistent"),
        base_url="http://127.0.0.1:1/v1",
        iterations=1,
        islands=1,
    )
    outcome = search(
        train=_arrays(("y",)),
        validation=_arrays(("y",)),
        targets=("y",),
        prompt="",
        directory=tmp_path,
        development=(object(), object()),
        context=object(),
    )
    assert outcome["status"] == "complete"
    assert outcome["equations"] == {"y": "y"}
    assert outcome["development_rollout_error"] == 0.3
    assert outcome["accounting"]["inexpressible_systems"] == 1
    assert outcome["accounting"]["inexpressible_operators"] == ["UNSUPPORTED_SYNTAX"]
    assert outcome["accounting"]["scored_systems"] == 1


# --- choosing the system as the released code does ----------------------------


def _cell(targets: dict[str, tuple[float, float]], train=(1.0, 3.0), validation=(1.5,)):
    """Targets decaying as size * start * exp(-rate t), and an input left at zero.

    Returns the development splits, the context and the channels, in the
    order upstream's variables x_0, x_1, ... name them.
    """
    from types import SimpleNamespace

    import numpy as np

    from autoformalism.baselines.core import baseline_validation_context
    from autoformalism.data import DatasetSplit, SplitName, Trajectory
    from autoformalism.data.models import DevelopmentDataset, TierRoles

    time = np.linspace(0.0, 3.0, 31)

    def trajectory(identifier: str, start: float) -> Trajectory:
        return Trajectory(
            trajectory_id=identifier,
            time=time,
            targets={
                name: size * start * np.exp(-rate * time)
                for name, (size, rate) in targets.items()
            },
            auxiliaries={},
            external_inputs={"inflow": np.zeros_like(time)},
            fixed_covariates={},
            derivatives={},
        )

    splits = tuple(
        DatasetSplit(
            name,
            tuple(trajectory(f"{name.value}_{i}", s) for i, s in enumerate(starts)),
            name.value,
        )
        for name, starts in (
            (SplitName.TRAIN, train),
            (SplitName.VALIDATION, validation),
        )
    )
    context = baseline_validation_context(
        DevelopmentDataset(
            benchmark_id="toy",
            tier="easy",
            roles=TierRoles(targets=tuple(targets), auxiliaries=()),
            train=splits[0],
            validation=splits[1],
        ),
        SimpleNamespace(
            external_inputs=("inflow",),
            fixed_covariates=(),
            one_step_target_history=False,
        ),
    )
    return splits, context, (*targets, "inflow")


#: Exact on the validation trajectory, which stays below 1.5, and wrong on the
#: training one that starts at 3; against a system slightly wrong everywhere.
EXACT_ON_VALIDATION = "-x_0 - 0.5*Abs(x_0 - 1.5) - 0.5*(x_0 - 1.5)"
SLIGHTLY_WRONG = "-0.97*x_0"


def test_the_system_is_ranked_over_training_and_validation_not_validation_alone():
    """Upstream's mse_train_val covers the fitted part of the trajectory too.

    The adapter used to rank by the validation error alone, which picks the
    other system here.
    """
    development, context, channels = _cell({"h": (1.0, 1.0)})
    systems, scores = driver.rank_systems(
        ((EXACT_ON_VALIDATION, SLIGHTLY_WRONG),),
        development,
        context,
        channels,
        seconds=30.0,
    )
    exact, slight = scores
    assert exact.status == slight.status == "scored"
    assert exact.validation_rollout_error < slight.validation_rollout_error
    assert slight.train_val_rollout_mse < exact.train_val_rollout_mse
    assert systems[driver.best_system(scores)] == (SLIGHTLY_WRONG,)


def test_the_error_is_pooled_over_targets_in_their_own_units():
    """As upstream pools over its variables: a large target counts for more."""
    import numpy as np

    from autoformalism.rebuttal.llm_ode_upstream import to_state_equations

    development, context, channels = _cell({"a": (100.0, 1.0), "b": (0.1, 2.0)})
    score = driver.score_system(
        ("-0.5*x_0", "-2*x_1"), development, context, channels, seconds=30.0
    )
    assert score.status == "scored"
    time = development[0].trajectories[0].time
    starts = (1.0, 3.0, 1.5)
    # b is exact, so only a's raw error is left, averaged over both targets.
    expected = np.mean(
        [
            np.mean(np.square(100.0 * s * (np.exp(-0.5 * time) - np.exp(-time)))) / 2
            for s in starts
        ]
    )
    assert score.train_val_rollout_mse == pytest.approx(expected, rel=1e-5)
    # The normalized errors recorded beside it are the evaluator's.
    equations = to_state_equations(("-0.5*x_0", "-2*x_1"), channels, ("a", "b"))
    assert score.equations == equations
    assert score.validation_rollout_error == driver.development_rollout_error(
        equations, context, *development
    )
    assert score.training_rollout_error == driver.development_rollout_error(
        equations, context, development[0], development[0]
    )


def test_a_system_stopped_early_could_not_have_been_chosen():
    """Stopping past the bound saves rollouts and changes no choice."""
    development, context, channels = _cell({"h": (1.0, 1.0)})
    frontier = ("-0.5*x_0", "-0.97*x_0", "-2*x_0", "-x_0", "-0.99*x_0", "-3*x_0")
    systems, scores = driver.rank_systems(
        (frontier,), development, context, channels, seconds=30.0
    )
    full = [
        driver.score_system(system, development, context, channels, seconds=30.0)
        for system in systems
    ]
    assert all(score.status == "scored" for score in full)
    assert driver.best_system(scores) == driver.best_system(full) == 3
    pruned = [index for index, score in enumerate(scores) if score.status == "pruned"]
    assert pruned  # the bound was used
    lowest = full[3].train_val_rollout_mse
    assert all(full[index].train_val_rollout_mse > lowest for index in pruned)
    for score, unpruned in zip(scores, full, strict=True):
        if score.status == "scored":
            assert score == unpruned


def test_scoring_in_several_processes_chooses_what_one_process_chooses():
    development, context, channels = _cell({"a": (10.0, 1.0), "b": (1.0, 2.0)})
    frontiers = (("-0.5*x_0", "-x_0", "-0.9*x_0"), ("-2*x_1", "-x_1"))
    one = driver.rank_systems(frontiers, development, context, channels, seconds=30.0)
    two = driver.rank_systems(
        frontiers, development, context, channels, seconds=30.0, workers=2
    )
    assert one[0] == two[0]
    assert driver.best_system(one[1]) == driver.best_system(two[1]) == 2 * 1 + 0
    best = driver.best_system(one[1])
    assert one[1][best] == two[1][best]


def test_a_rollout_over_its_limit_leaves_the_system_unchosen():
    """The simulator raises past its deadline; that must not end the search."""
    development, context, channels = _cell({"h": (1.0, 1.0)})
    score = driver.score_system(
        ("-x_0",), development, context, channels, seconds=1e-9
    )
    assert score.status == "rollout_failed"
    assert "over 1e-09 seconds" in score.message
    _, scores = driver.rank_systems(
        (("-x_0",),), development, context, channels, seconds=1e-9
    )
    assert driver.best_system(scores) is None


def test_selection_never_rolls_out_test_data():
    from dataclasses import replace

    from autoformalism.data import SplitName

    (train, validation), context, channels = _cell({"h": (1.0, 1.0)})
    held_out = replace(validation, name=SplitName.TEST)
    with pytest.raises(ValueError, match="never TEST"):
        driver.score_system(
            ("-x_0",), (train, held_out), context, channels, seconds=30.0
        )


def test_a_system_the_grammar_refuses_is_counted_not_scored():
    development, context, channels = _cell({"h": (1.0, 1.0)})
    refused = driver.score_system(
        ("-0.4*x_0**0.73*(x_0 > 1.0)",), development, context, channels, seconds=30.0
    )
    assert refused.status == "inexpressible"
    assert refused.functions == ("UNSUPPORTED_SYNTAX",)
    beyond = driver.score_system(
        ("-x_7",), development, context, channels, seconds=30.0
    )
    assert beyond.status == "inexpressible"
    assert beyond.functions == ("unknown_variable",)


def test_the_search_logs_as_upstream_logs_and_then_stops(monkeypatch, tmp_path: Path):
    """Upstream's main.py logs every prompt and reply to llm_ode.log at INFO."""
    import logging

    _install_fake_upstream(monkeypatch, {"y": ["x_0"]})
    _scoring(monkeypatch)
    search = driver.build_searcher(
        upstream_root=Path("/nonexistent"),
        base_url="http://127.0.0.1:1/v1",
        iterations=10,
        islands=1,
    )
    level = logging.getLogger().level
    handlers = list(logging.getLogger().handlers)
    search(
        train=_arrays(("y",)),
        validation=_arrays(("y",)),
        targets=("y",),
        prompt="",
        directory=tmp_path,
        development=(object(), object()),
        context=object(),
    )
    lines = (tmp_path / "llm_ode.log").read_text().splitlines()
    assert any(" INFO iteration 10 of 10 after " in line for line in lines)
    assert logging.getLogger().level == level
    assert logging.getLogger().handlers == handlers


# --- a failing endpoint --------------------------------------------------------


class _Clock:
    """Time that passes only when the driver pauses."""

    def __init__(self) -> None:
        self.now = 0.0
        self.pauses: list[float] = []

    def __call__(self) -> float:
        return self.now

    def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)
        self.now += seconds


def _openai_error(kind: str) -> Exception:
    import httpx
    import openai

    request = httpx.Request("POST", "http://127.0.0.1:1/v1/responses")
    if kind == "offline":
        return openai.APIConnectionError(request=request)
    status, cls = {
        "503": (503, openai.InternalServerError),
        "401": (401, openai.AuthenticationError),
        "400": (400, openai.BadRequestError),
    }[kind]
    return cls(kind, response=httpx.Response(status, request=request), body=None)


def _install_endpoint(monkeypatch, replies, requests: list | None = None) -> _Clock:
    """A one-target upstream whose every step asks the endpoint once.

    A reply is an error to raise, a `_Raw` to return as it is, or anything
    else for a plain answer; `requests` collects what each call sent.
    """
    _install_fake_upstream(monkeypatch, {"y": ["x_0"]})
    modules = driver.load_upstream(Path("/nonexistent"))
    answers = iter(replies)

    class _Llm(_FakeLlm):
        def __init__(self, api_key: str, base_url: str) -> None:
            super().__init__(api_key, base_url)

            def create(**kwargs):
                if requests is not None:
                    requests.append(kwargs)
                reply = next(answers)
                if isinstance(reply, Exception):
                    raise reply
                if isinstance(reply, _Raw):
                    return reply
                return _Raw(types.SimpleNamespace(output_text="x_0", usage={}))

            self.client = _client(create)

    class _Asking(_FakeSearcher):
        def step(self) -> None:
            super().step()
            self.kwargs["llm"].make_request("propose")

    monkeypatch.setattr(
        driver,
        "load_upstream",
        lambda root: types.SimpleNamespace(
            **{
                **vars(modules),
                "llm": _Llm,
                "equation_searcher": lambda **kwargs: _Asking(["x_0"], **kwargs),
            }
        ),
    )
    _scoring(monkeypatch, lambda equations: 0.5)
    clock = _Clock()
    monkeypatch.setattr(driver, "_clock", clock)
    monkeypatch.setattr(driver, "_pause", clock.pause)
    return clock


def _search(tmp_path: Path, **options) -> dict:
    search = driver.build_searcher(
        upstream_root=Path("/nonexistent"),
        base_url="http://127.0.0.1:1/v1",
        iterations=2,
        islands=1,
        **options,
    )
    return search(
        train=_arrays(("y",)),
        validation=_arrays(("y",)),
        targets=("y",),
        prompt="",
        directory=tmp_path,
        development=(object(), object()),
        context=object(),
    )


def test_a_brief_outage_is_waited_out(monkeypatch, tmp_path: Path) -> None:
    """Upstream's client gives up within seconds; the search should not."""
    clock = _install_endpoint(
        monkeypatch,
        [
            _openai_error("503"),
            _openai_error("offline"),
            _openai_error("401"),
            "ok",
            "ok",
        ],
    )
    outcome = _search(tmp_path)
    assert outcome["status"] == "complete"
    assert outcome["accounting"]["search_error"] is None
    assert clock.pauses == [1.0, 2.0, 4.0]
    failures = (tmp_path / "llm_calls.jsonl").read_text().count("llm_failure")
    assert failures == 3


def test_an_outage_beyond_patience_is_an_infrastructure_failure(
    monkeypatch, tmp_path: Path
) -> None:
    """What the search had found before the outage is not the method's result."""
    import itertools

    clock = _install_endpoint(monkeypatch, itertools.repeat(_openai_error("offline")))
    outcome = _search(tmp_path)
    assert outcome["status"] == "endpoint_unavailable"
    assert "failed for 15 minutes" in outcome["error"]
    assert "equations" not in outcome
    assert max(clock.pauses) == 60.0
    patience = driver.DEFAULT_PATIENCE_SECONDS
    assert patience <= clock.now < patience + 60


def test_a_longer_patience_outlasts_a_longer_outage(
    monkeypatch, tmp_path: Path
) -> None:
    outage = [_openai_error("503")] * 40  # about 35 minutes at a minute a try
    clock = _install_endpoint(monkeypatch, [*outage, "ok", "ok"])
    outcome = _search(tmp_path, patience_seconds=6 * 3600.0)
    assert outcome["status"] == "complete"
    assert 30 * 60 < clock.now < 40 * 60


def test_a_refused_request_is_not_retried(monkeypatch, tmp_path: Path) -> None:
    """A malformed request fails the same way each time; upstream's handling stands."""
    clock = _install_endpoint(monkeypatch, [_openai_error("400")])
    outcome = _search(tmp_path)
    assert clock.pauses == []
    assert outcome["status"] == "complete"  # upstream keeps what it had found
    assert outcome["accounting"]["search_error"].startswith("BadRequestError")


# --- what a request carries ----------------------------------------------------


def test_without_options_the_requests_are_upstreams(
    monkeypatch, tmp_path: Path
) -> None:
    """The model the server listed first, upstream's output limit, nothing added."""
    sent: list = []
    _install_endpoint(monkeypatch, ["ok", "ok"], sent)
    _search(tmp_path)
    assert [request["model"] for request in sent] == ["m", "m"]
    assert all(request["max_output_tokens"] == 1024 for request in sent)
    assert all("extra_body" not in request for request in sent)


def test_a_campaign_names_its_model_and_asks_for_fresh_answers(
    monkeypatch, tmp_path: Path
) -> None:
    """The hosted gateway lists a dozen models and replays a repeated request."""
    sent: list = []
    _install_endpoint(monkeypatch, ["ok", "ok"], sent)
    _search(tmp_path, model="gpt-oss-120b", bypass_cache=True)
    assert [request["model"] for request in sent] == ["gpt-oss-120b"] * 2
    assert all(
        request["extra_body"] == {"cache": {"no-cache": True}} for request in sent
    )
    # The prompt and the output limit stay upstream's.
    assert all(request["input"] == "propose" for request in sent)
    assert all(request["max_output_tokens"] == 1024 for request in sent)


def test_a_replayed_answer_and_a_cut_off_reply_are_logged(
    monkeypatch, tmp_path: Path
) -> None:
    """The gateway marks a replay in a header; a reply says why it stopped."""
    replayed = _Raw(
        types.SimpleNamespace(
            output_text="x_0",
            usage={"total_tokens": 9},
            status="completed",
            incomplete_details=None,
        ),
        {"x-litellm-cache-key": "4f2a"},
    )
    cut_off = _Raw(
        types.SimpleNamespace(
            output_text="",
            usage={"total_tokens": 1024},
            status="incomplete",
            incomplete_details=types.SimpleNamespace(reason="max_output_tokens"),
        )
    )
    _install_endpoint(monkeypatch, [replayed, cut_off])
    outcome = _search(tmp_path, bypass_cache=True)
    events = [
        json.loads(line)
        for line in (tmp_path / "llm_calls.jsonl").read_text().splitlines()
    ]
    assert [(event["cache_hit"], event["finish_reasons"]) for event in events] == [
        (True, ["completed"]),
        (False, ["max_output_tokens"]),
    ]
    assert outcome["accounting"]["cache_hits"] == 1
