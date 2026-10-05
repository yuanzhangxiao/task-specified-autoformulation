"""Driving the pinned LLM-SR checkout, with upstream replaced by a fake.

What is under test is our binding: the transport substitution, the guard on
their unbounded retry loop, and the recovery of a model from what their
profiler wrote. Their search is not reimplemented here.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from autoformalism.rebuttal import llm_sr_driver as driver
from autoformalism.rebuttal.llm_sr_shim import ShimAccounting, UpstreamEndpointError


class _FakeLocalLLM:
    """Stands in for their sampler class; only _do_request is replaced."""

    def __init__(self) -> None:
        self._samples_per_prompt = 4
        self._batch_inference = True

    def _do_request(self, content: str) -> list[str]:
        raise AssertionError("the original transport must not be reached")


def _fake_upstream(monkeypatch, tmp_path: Path, *, best: str | None, score: float):
    """A checkout whose pipeline.main writes what the real profiler writes."""
    checkout = tmp_path / "LLM-SR"
    (checkout / "llmsr").mkdir(parents=True)
    (checkout / "llmsr" / "pipeline.py").write_text("", encoding="utf-8")

    sampler = types.ModuleType("llmsr.sampler")
    sampler.LocalLLM = _FakeLocalLLM
    evaluator = types.ModuleType("llmsr.evaluator")
    evaluator.LocalSandbox = object
    config = types.ModuleType("llmsr.config")
    config.Config = lambda **kwargs: types.SimpleNamespace(**kwargs)
    config.ClassConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)

    pipeline = types.ModuleType("llmsr.pipeline")

    def main(*, specification, inputs, config, max_sample_nums, class_config, log_dir):
        # exercise the substituted transport exactly as their sampler would
        llm = sampler.LocalLLM()
        llm._do_request("a prompt")
        if best is None:
            return
        samples = Path(log_dir) / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        (samples / "samples_1.json").write_text(
            json.dumps({"sample_order": 1, "function": best, "score": score})
        )

    pipeline.main = main
    package = types.ModuleType("llmsr")
    for name, module in (
        ("llmsr", package), ("llmsr.sampler", sampler),
        ("llmsr.evaluator", evaluator), ("llmsr.config", config),
        ("llmsr.pipeline", pipeline),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return checkout, sampler


def _arrays(rows: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(rows, 2))
    derivatives = np.column_stack(
        [-0.7 * values[:, 0] + 0.3 * values[:, 1], np.zeros(rows)]
    )
    return values, derivatives


def _run(driver_search, **overrides):
    values, derivatives = _arrays()
    kwargs = {
        "channels": ("G", "I"),
        "targets": ("G",),
        "values": values,
        "derivatives": derivatives,
        "description": "Recover the flux.",
        "development": (object(), object()),
        "context": object(),
        "score_rollout": lambda *a, **k: 0.25,
    }
    kwargs.update(overrides)
    return driver_search(**kwargs)


def test_a_recovered_model_carries_refitted_coefficients(
    monkeypatch, tmp_path: Path
) -> None:
    """Their evaluate discards the values that produced the score."""
    best = (
        "def equation(G, I, params):\n"
        "    return params[0] * G + params[1] * I\n"
    )
    checkout, _ = _fake_upstream(monkeypatch, tmp_path, best=best, score=-0.01)
    search = driver.build_searcher(
        upstream_root=checkout, base_url="http://127.0.0.1:1", model="m", samples=20
    )
    monkeypatch.setattr(
        driver, "complete", lambda *a, **k: {"content": ["x"] * 4}
    )
    outcome = _run(search, directory=tmp_path / "out")
    assert outcome["status"] == "complete"
    # the coefficients used to generate the data are recovered, not left at 1.0
    expression = outcome["equations"]["G"]
    assert "-0.7" in expression or "-0.69" in expression
    assert outcome["accounting"]["llm_requests"] == 0  # complete() was stubbed


class _Clock:
    """Time that passes only when the transport pauses."""

    def __init__(self) -> None:
        self.now = 0.0
        self.pauses: list[float] = []

    def __call__(self) -> float:
        return self.now

    def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)
        self.now += seconds


def _failing(monkeypatch, outcomes):
    """Replace the endpoint with one that answers "ok" or fails, in turn."""
    replies = iter(outcomes)

    def complete(*args, **kwargs):
        if next(replies) == "fail":
            raise UpstreamEndpointError("offline")
        return {"content": ["x"]}

    monkeypatch.setattr(driver, "complete", complete)
    clock = _Clock()
    monkeypatch.setattr(driver, "_clock", clock)
    monkeypatch.setattr(driver, "_pause", clock.pause)
    return clock


def test_an_unbounded_retry_loop_is_stopped(monkeypatch, tmp_path: Path) -> None:
    """Their sampler catches Exception and retries forever.

    A wedged endpoint would otherwise consume the whole walltime with nothing
    recorded, which is how the first LLM-ODE probe was lost.
    """
    _, sampler = _fake_upstream(monkeypatch, tmp_path, best=None, score=0.0)
    clock = _failing(monkeypatch, ["fail"] * 1000)
    accounting = ShimAccounting()
    with driver._transport(sampler, "http://127.0.0.1:1", "m", accounting):
        llm = sampler.LocalLLM()
        # their loop swallows ordinary failures, until patience runs out
        with pytest.raises(driver.SamplerStalled, match="15 minutes"):
            while True:
                with pytest.raises(UpstreamEndpointError):
                    llm._do_request("p")
    assert not isinstance(driver.SamplerStalled("x"), Exception)
    # Each retry waited, doubling to a minute, and the search gave up once the
    # endpoint had been failing for the default patience of a quarter hour.
    assert clock.pauses[:7] == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0]
    assert max(clock.pauses) == 60.0
    patience = driver.DEFAULT_PATIENCE_SECONDS
    assert patience <= clock.now < patience + 60.0


def test_a_long_outage_is_waited_out_when_patience_allows(
    monkeypatch, tmp_path: Path
) -> None:
    """A hosted service that is down for two hours does not end the search."""
    _, sampler = _fake_upstream(monkeypatch, tmp_path, best=None, score=0.0)
    failures = 130  # about two hours at one retry a minute
    clock = _failing(monkeypatch, ["fail"] * failures + ["ok"])
    with driver._transport(
        sampler, "http://127.0.0.1:1", "m", ShimAccounting(), patience=6 * 3600.0
    ):
        llm = sampler.LocalLLM()
        for _ in range(failures):
            with pytest.raises(UpstreamEndpointError):
                llm._do_request("p")
        assert llm._do_request("p") == ["x"]
    assert 2 * 3600 - 300 < clock.now < 6 * 3600
    assert max(clock.pauses) == 60.0


def test_patience_counts_one_outage_not_the_sum_of_several(
    monkeypatch, tmp_path: Path
) -> None:
    """Two nine-minute outages a success apart are each within a quarter hour."""
    _, sampler = _fake_upstream(monkeypatch, tmp_path, best=None, score=0.0)
    outage = ["fail"] * 14  # 1+2+...+32 seconds, then a minute each: 9 minutes
    clock = _failing(monkeypatch, [*outage, "ok", *outage, "ok"])
    with driver._transport(sampler, "http://127.0.0.1:1", "m", ShimAccounting()):
        llm = sampler.LocalLLM()
        for _ in range(2):
            for _ in outage:
                with pytest.raises(UpstreamEndpointError):
                    llm._do_request("p")
            assert llm._do_request("p") == ["x"]
    assert clock.now > driver.DEFAULT_PATIENCE_SECONDS


def test_a_success_resets_the_pause(monkeypatch, tmp_path: Path) -> None:
    """A brief outage is waited out, and the next one starts short again."""
    _, sampler = _fake_upstream(monkeypatch, tmp_path, best=None, score=0.0)
    clock = _failing(monkeypatch, ["fail", "fail", "ok", "fail"])
    with driver._transport(sampler, "http://127.0.0.1:1", "m", ShimAccounting()):
        llm = sampler.LocalLLM()
        for _ in range(2):
            with pytest.raises(UpstreamEndpointError):
                llm._do_request("p")
        assert llm._do_request("p") == ["x"]
        with pytest.raises(UpstreamEndpointError):
            llm._do_request("p")
    assert clock.pauses == [1.0, 2.0, 1.0]


def test_the_transport_is_restored_afterwards(monkeypatch, tmp_path: Path) -> None:
    _, sampler = _fake_upstream(monkeypatch, tmp_path, best=None, score=0.0)
    original = sampler.LocalLLM._do_request
    with driver._transport(sampler, "http://127.0.0.1:1", "m", ShimAccounting()):
        assert sampler.LocalLLM._do_request is not original
    assert sampler.LocalLLM._do_request is original


def test_a_search_that_scored_nothing_is_reported(monkeypatch, tmp_path: Path) -> None:
    checkout, _ = _fake_upstream(monkeypatch, tmp_path, best=None, score=0.0)
    monkeypatch.setattr(driver, "complete", lambda *a, **k: {"content": ["x"]})
    search = driver.build_searcher(
        upstream_root=checkout, base_url="http://127.0.0.1:1", model="m", samples=20
    )
    outcome = _run(search, directory=tmp_path / "empty")
    assert outcome["status"] == "no_candidates"


def test_a_program_outside_the_grammar_is_reported_with_its_reason(
    monkeypatch, tmp_path: Path
) -> None:
    """A loop over timesteps is a real discovery we cannot score."""
    best = (
        "def equation(G, I, params):\n"
        "    out = 0\n"
        "    for i in range(3):\n"
        "        out = out + params[i] * G\n"
        "    return out\n"
    )
    checkout, _ = _fake_upstream(monkeypatch, tmp_path, best=best, score=-0.01)
    monkeypatch.setattr(driver, "complete", lambda *a, **k: {"content": ["x"]})
    search = driver.build_searcher(
        upstream_root=checkout, base_url="http://127.0.0.1:1", model="m", samples=20
    )
    outcome = _run(search, directory=tmp_path / "loop")
    assert outcome["status"] == "inexpressible"
    assert "no expression equivalent" in outcome["error"]
    assert outcome["accounting"]["inexpressible_targets"]


def test_a_refitted_exponent_the_grammar_refuses_is_reported(
    monkeypatch, tmp_path: Path
) -> None:
    """Conversion can succeed where the evaluator's grammar refuses.

    A refitted exponent, `G ** 0.73`, is the case met in practice. The refusal
    is stood in for here; the real grammar is exercised in the LLM-ODE tests.
    """
    from autoformalism.rebuttal.llm_ode_upstream import InexpressibleEquation

    best = (
        "def equation(G, I, params):\n"
        "    return params[0] * G + params[1] * I\n"
    )
    checkout, _ = _fake_upstream(monkeypatch, tmp_path, best=best, score=-0.01)
    monkeypatch.setattr(driver, "complete", lambda *a, **k: {"content": ["x"]})
    search = driver.build_searcher(
        upstream_root=checkout, base_url="http://127.0.0.1:1", model="m", samples=20
    )

    def refuse(equations, *args, **kwargs):
        raise InexpressibleEquation(equations["G"], ("UNSUPPORTED_POWER",))

    outcome = _run(search, directory=tmp_path / "power", score_rollout=refuse)
    assert outcome["status"] == "inexpressible"
    assert "UNSUPPORTED_POWER" in outcome["error"]
    assert outcome["accounting"]["inexpressible_targets"]


def test_a_missing_checkout_is_refused_before_anything_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not an LLM-SR checkout"):
        driver.build_searcher(
            upstream_root=tmp_path, base_url="http://127.0.0.1:1", model="m", samples=1
        )
