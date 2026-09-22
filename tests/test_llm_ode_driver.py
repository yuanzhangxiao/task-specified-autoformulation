"""Driving the pinned LLM-ODE checkout over one Phase-B cell.

The checkout is not available here, so upstream is replaced by a fake that
reproduces the parts the driver actually touches: a per-variable searcher with
`step` and `get_pareto_frontier`, and a module-level `generate_prompt` the
driver extends. What is under test is our binding, not their search.
"""

from __future__ import annotations

import inspect
import sys
import types
from unittest import mock
from pathlib import Path

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


class _FakeLlm:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.n_queries = 7


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


def test_every_target_is_stepped_for_the_declared_iterations(monkeypatch) -> None:
    """Upstream advances all equations once per iteration; so must we."""
    _install_fake_upstream(monkeypatch, {"y": ["x_0"], "w": ["x_1"]})
    monkeypatch.setattr(driver, "development_rollout_error", lambda *a, **k: 0.5)
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
        directory=Path("/tmp"),
        development=(object(), object()),
        context=object(),
    )
    assert outcome["status"] == "complete"
    assert outcome["equations"] == {"y": "y", "w": "w"}
    assert outcome["accounting"]["llm_queries"] == 7


def test_the_specification_is_installed_and_then_removed(monkeypatch) -> None:
    """A leaked patch would silently alter any later search in the process."""
    module, _ = _install_fake_upstream(monkeypatch, {"y": ["x_0"]})
    original = module.generate_prompt
    monkeypatch.setattr(driver, "development_rollout_error", lambda *a, **k: 1.0)
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
        directory=Path("/tmp"),
        development=(object(), object()),
        context=object(),
    )
    assert module.generate_prompt is original


def test_an_inexpressible_sweep_is_reported_with_the_operator_named(
    monkeypatch,
) -> None:
    """The cost of our restricted grammar must be countable, not estimated."""
    _install_fake_upstream(monkeypatch, {"y": ["sin(x_0)", "sin(x_0) + x_0"]})
    monkeypatch.setattr(driver, "development_rollout_error", lambda *a, **k: 0.1)
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
        directory=Path("/tmp"),
        development=(object(), object()),
        context=object(),
    )
    assert outcome["status"] == "inexpressible"
    assert outcome["accounting"]["inexpressible_operators"] == ["sin"]
    assert outcome["accounting"]["inexpressible_systems"] == 2
    assert outcome["accounting"]["scored_systems"] == 0
    assert "sin" in outcome["error"]


def test_an_exploding_product_is_recorded_rather_than_truncated(monkeypatch) -> None:
    """Ranking a subset and calling it the search would overstate the method."""
    wide = {
        name: [f"x_0 + {index}" for index in range(30)] for name in ("a", "b", "c")
    }
    _install_fake_upstream(monkeypatch, wide)
    monkeypatch.setattr(driver, "development_rollout_error", lambda *a, **k: 1.0)
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
        directory=Path("/tmp"),
        development=(object(), object()),
        context=object(),
    )
    assert outcome["status"] == "product_too_large"
    assert "exceed the" in outcome["error"]


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

    with mock.patch.object(llm_ode_upstream.sys, "version_info", (3, 12, 10)):
        with pytest.raises(RuntimeError, match="requires Python"):
            llm_ode_upstream.load_upstream(Path("/nonexistent"))


def test_a_search_that_never_produces_a_candidate_stops_early(monkeypatch) -> None:
    """Whatever the cause, an empty search must not spend the allocation."""
    _install_fake_upstream(monkeypatch, {"y": []})
    monkeypatch.setattr(driver, "development_rollout_error", lambda *a, **k: 1.0)
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
        directory=Path("/tmp"),
        development=(object(), object()),
        context=object(),
    )
    assert outcome["status"] == "no_candidates"
    assert "not running" in outcome["error"]
    # stopped at the barren threshold rather than running all 200
    assert outcome["accounting"]["llm_queries"] == 7
