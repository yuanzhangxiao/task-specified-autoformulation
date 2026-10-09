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
    # Their sample counter, kept on the class as upstream keeps it.
    sampler.Sampler = type("Sampler", (), {"_global_samples_nums": 1})
    sampler.counts_at_start = []
    sampler.configs = []
    evaluator = types.ModuleType("llmsr.evaluator")
    evaluator.LocalSandbox = object
    config = types.ModuleType("llmsr.config")
    config.Config = lambda **kwargs: types.SimpleNamespace(**kwargs)
    config.ExperienceBufferConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    config.ClassConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)

    pipeline = types.ModuleType("llmsr.pipeline")

    def main(*, specification, inputs, config, max_sample_nums, class_config, log_dir):
        sampler.configs.append(config)
        # exercise the substituted transport exactly as their sampler would
        llm = sampler.LocalLLM()
        llm._do_request("a prompt")
        # their sampler stops once the class counter reaches the budget
        sampler.counts_at_start.append(sampler.Sampler._global_samples_nums)
        sampler.Sampler._global_samples_nums = max_sample_nums + 1
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
    """Rows over the channels (G, I), and the derivative of each channel."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=(rows, 2))
    derivatives = np.column_stack(
        [-0.7 * values[:, 0] + 0.3 * values[:, 1], np.zeros(rows)]
    )
    return values, derivatives


def _run(driver_search, **overrides):
    values, derivatives = _arrays()
    targets = overrides.get("targets", ("G",))
    kwargs = {
        "channels": ("G", "I"),
        "targets": targets,
        "values": values,
        # one label column per searched target
        "labels": derivatives[:, [("G", "I").index(name) for name in targets]],
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
    # the profiler's one sample was the model's, and it scored
    assert outcome["accounting"]["model_samples"] == 1
    assert outcome["accounting"]["model_samples_scored"] == 1
    # LLM-SR's own score is kept beside the refitted error, for comparison
    record = outcome["selected_samples"]["G"]
    assert (record["sample_order"], record["llm_sr_score"]) == (1, -0.01)
    assert record["refitted_training_mse"] < 1e-10


def _capturing(monkeypatch) -> list[dict]:
    """Replace the endpoint call with one that records how it was asked."""
    seen: list[dict] = []

    def complete(payload, **kwargs):
        seen.append({**kwargs, "payload": payload})
        return {"content": ["x"] * 4}

    monkeypatch.setattr(driver, "complete", complete)
    return seen


def test_the_declared_adaptation_reaches_every_request(
    monkeypatch, tmp_path: Path
) -> None:
    """A plan's limit, header reading and cache bypass are what is sent."""
    checkout, _ = _fake_upstream(monkeypatch, tmp_path, best=None, score=0.0)
    seen = _capturing(monkeypatch)
    search = driver.build_searcher(
        upstream_root=checkout, base_url="http://127.0.0.1:1", model="m",
        samples=20, max_new_tokens=4096, join_split_headers=True,
        bypass_cache=True,
    )
    _run(search, directory=tmp_path / "adapted")
    assert seen and all(
        (call["max_tokens"], call["join_headers"], call["bypass_cache"])
        == (4096, True, True)
        for call in seen
    )


def test_without_a_declaration_the_transport_is_upstreams(
    monkeypatch, tmp_path: Path
) -> None:
    checkout, sampler = _fake_upstream(monkeypatch, tmp_path, best=None, score=0.0)
    seen = _capturing(monkeypatch)
    search = driver.build_searcher(
        upstream_root=checkout, base_url="http://127.0.0.1:1", model="m", samples=20
    )
    _run(search, directory=tmp_path / "plain")
    assert [
        (call["max_tokens"], call["join_headers"], call["bypass_cache"])
        for call in seen
    ] == [(512, False, False)]
    # their client sends no temperature, and their Config keeps its defaults
    assert seen[0]["payload"]["params"]["temperature"] is None
    assert [vars(config) for config in sampler.configs] == [{}]


def test_the_papers_settings_reach_every_request_and_their_configuration(
    monkeypatch, tmp_path: Path
) -> None:
    """The temperature is sent; the period and evaluators are their Config's."""
    checkout, sampler = _fake_upstream(monkeypatch, tmp_path, best=None, score=0.0)
    seen = _capturing(monkeypatch)
    search = driver.build_searcher(
        upstream_root=checkout, base_url="http://127.0.0.1:1", model="m",
        samples=20, temperature=0.8, cluster_sampling_temperature_period=10_000,
        num_evaluators=4,
    )
    _run(search, directory=tmp_path / "paper")
    assert seen and all(
        call["payload"]["params"]["temperature"] == 0.8 for call in seen
    )
    (config,) = sampler.configs
    assert config.num_evaluators == 4
    assert config.experience_buffer.cluster_sampling_temperature_period == 10_000


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
        "    out = np.zeros_like(G)\n"
        "    for t in range(1, len(G)):\n"
        "        out[t] = out[t - 1] + params[0] * (G[t] - out[t - 1])\n"
        "    return out\n"
    )
    checkout, _ = _fake_upstream(monkeypatch, tmp_path, best=best, score=-0.01)
    monkeypatch.setattr(driver, "complete", lambda *a, **k: {"content": ["x"]})
    search = driver.build_searcher(
        upstream_root=checkout, base_url="http://127.0.0.1:1", model="m", samples=20
    )
    outcome = _run(search, directory=tmp_path / "loop")
    assert outcome["status"] == "inexpressible"
    assert "a range over the rows" in outcome["error"]
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


# --- one search per target, kept once finished --------------------------------

LINEAR = "def equation(G, I, params):\n    return params[0] * G + params[1] * I\n"


def _counting(monkeypatch, tmp_path: Path, best: str | None = LINEAR):
    """The fake checkout, recording which target each run of their pipeline searched."""
    checkout, _ = _fake_upstream(monkeypatch, tmp_path, best=best, score=-0.01)
    pipeline = sys.modules["llmsr.pipeline"]
    original = pipeline.main
    runs: list[str] = []

    def main(**kwargs):
        runs.append(Path(kwargs["log_dir"]).name)
        original(**kwargs)

    monkeypatch.setattr(pipeline, "main", main)
    monkeypatch.setattr(driver, "complete", lambda *a, **k: {"content": ["x"] * 4})
    return checkout, runs


def test_each_target_is_searched_once_and_the_model_assembled_from_all(
    monkeypatch, tmp_path: Path
) -> None:
    """Separate processes can search a cell's targets; assembly reuses them."""
    checkout, runs = _counting(monkeypatch, tmp_path)
    checks: list[str] = []
    search = driver.build_searcher(
        upstream_root=checkout, base_url="http://127.0.0.1:1", model="m",
        samples=20, before_search=lambda: checks.append("checked"),
    )
    task = tmp_path / "task"
    first = _run(search, directory=task, targets=("G", "I"), only="I")
    assert (first["status"], first["target"]) == ("searched", "I")
    assert runs == ["llmsr-I"] and checks == ["checked"]
    record = json.loads((task / "llmsr-I" / driver.SEARCH_RECORD).read_text())
    assert record["target"] == "I" and record["started_utc"] <= record["finished_utc"]
    assert not (task / "llmsr-G").exists()

    # Without `only`, what is missing is searched and the model takes both.
    outcome = _run(search, directory=task, targets=("G", "I"))
    assert outcome["status"] == "complete"
    assert runs == ["llmsr-I", "llmsr-G"] and len(checks) == 2
    assert set(outcome["equations"]) == {"G", "I"}
    assert outcome["accounting"]["model_samples"] == 2
    assert "wall_seconds" in outcome["accounting"]

    # Assembling again searches nothing and contacts no endpoint.
    again = _run(search, directory=task, targets=("G", "I"))
    assert again["equations"] == outcome["equations"]
    assert runs == ["llmsr-I", "llmsr-G"] and len(checks) == 2
    with pytest.raises(ValueError, match="not among the searched targets"):
        _run(search, directory=task, targets=("G", "I"), only="X")


def test_finished_searches_are_counted_from_their_records(
    monkeypatch, tmp_path: Path
) -> None:
    """A search another process finished costs what its record says."""
    checkout, runs = _counting(monkeypatch, tmp_path)
    task = tmp_path / "task"
    times = {
        "G": ("2026-10-09T00:00:00+00:00", "2026-10-10T00:00:00+00:00", 3, 1),
        "I": ("2026-10-09T01:00:00+00:00", "2026-10-11T00:00:00+00:00", 2, 0),
    }
    for target, (started, finished, requests, failures) in times.items():
        samples = task / f"llmsr-{target}" / "samples"
        samples.mkdir(parents=True)
        (samples / "samples_1.json").write_text(
            json.dumps({"sample_order": 1, "function": LINEAR, "score": -0.01})
        )
        reasons = {"timeout": failures} if failures else {}
        (task / f"llmsr-{target}" / driver.SEARCH_RECORD).write_text(json.dumps({
            "target": target, "llm_requests": requests, "llm_samples": 4 * requests,
            "transport_failures": failures, "transport_failure_reasons": reasons,
            "search_seconds": 100.0 * requests, "started_utc": started,
            "finished_utc": finished,
        }))
    search = driver.build_searcher(
        upstream_root=checkout, base_url="http://127.0.0.1:1", model="m", samples=20
    )
    outcome = _run(search, directory=task, targets=("G", "I"))
    assert outcome["status"] == "complete" and runs == []
    accounting = outcome["accounting"]
    assert (accounting["llm_requests"], accounting["llm_samples"]) == (5, 20)
    assert accounting["transport_failure_reasons"] == {"timeout": 1}
    # Durations add up; the wall time spans the first start to the last finish.
    assert accounting["search_seconds"] == 500.0
    assert accounting["wall_seconds"] == 48 * 3600.0


def test_a_target_search_that_stopped_part_way_starts_over(
    monkeypatch, tmp_path: Path
) -> None:
    """Their search cannot resume, and a stopped one's samples must not count."""
    checkout, runs = _counting(monkeypatch, tmp_path)
    stale = tmp_path / "task" / "llmsr-G" / "samples"
    stale.mkdir(parents=True)
    unfinished = "def equation(G, I, params):\n    return params[0] * I\n"
    (stale / "samples_9.json").write_text(
        json.dumps({"sample_order": 9, "function": unfinished, "score": 0.0})
    )
    search = driver.build_searcher(
        upstream_root=checkout, base_url="http://127.0.0.1:1", model="m", samples=20
    )
    outcome = _run(search, directory=tmp_path / "task")
    assert outcome["status"] == "complete" and runs == ["llmsr-G"]
    kept = tmp_path / "task" / "llmsr-G.interrupted-1" / "samples" / "samples_9.json"
    assert kept.exists()
    assert not (tmp_path / "task" / "llmsr-G" / "samples" / "samples_9.json").exists()
    assert "G" in outcome["equations"]["G"]  # the new search's model, not the stale one
    # Nothing to set aside: a finished search, an empty or a missing directory.
    assert driver.set_aside_unfinished_search(tmp_path / "task" / "llmsr-G") is None
    assert driver.set_aside_unfinished_search(tmp_path / "missing") is None


def test_a_stalled_target_search_leaves_no_record(monkeypatch, tmp_path: Path) -> None:
    """A search the endpoint stopped is searched again, from the start."""
    checkout, _ = _counting(monkeypatch, tmp_path)

    def stalled(**kwargs):
        raise driver.SamplerStalled("the endpoint failed 360 times in a row")

    monkeypatch.setattr(sys.modules["llmsr.pipeline"], "main", stalled)
    search = driver.build_searcher(
        upstream_root=checkout, base_url="http://127.0.0.1:1", model="m", samples=20
    )
    outcome = _run(search, directory=tmp_path / "task", only="G")
    assert outcome["status"] == "endpoint_unavailable"
    assert "360 times" in outcome["error"]
    assert not (tmp_path / "task" / "llmsr-G" / driver.SEARCH_RECORD).exists()
    assert "wall_seconds" not in outcome["accounting"]


def test_each_search_in_one_process_gets_its_whole_budget(
    monkeypatch, tmp_path: Path
) -> None:
    """Their sample counter lives on a class, so only a new process resets it."""
    checkout, sampler = _fake_upstream(monkeypatch, tmp_path, best=LINEAR, score=-0.01)
    monkeypatch.setattr(driver, "complete", lambda *a, **k: {"content": ["x"] * 4})
    search = driver.build_searcher(
        upstream_root=checkout, base_url="http://127.0.0.1:1", model="m", samples=20
    )
    outcome = _run(search, directory=tmp_path / "task", targets=("G", "I"))
    assert outcome["status"] == "complete"
    # Without the reset the second search would start at 21 and stop at once.
    assert sampler.counts_at_start == [1, 1]


def test_a_branch_on_a_coefficient_is_refitted_as_their_evaluator_runs_it(
    monkeypatch, tmp_path: Path
) -> None:
    """Their fit follows whichever branch each trial vector takes; so does ours."""
    best = (
        "def equation(G, I, params):\n"
        "    k = params[0] if len(params) > 0 else 1.0\n"
        "    if params[1] == 0:\n"
        "        return k * G\n"
        "    return k * G + params[1] * I\n"
    )
    checkout, _ = _fake_upstream(monkeypatch, tmp_path, best=best, score=-0.01)
    monkeypatch.setattr(driver, "complete", lambda *a, **k: {"content": ["x"] * 4})
    search = driver.build_searcher(
        upstream_root=checkout, base_url="http://127.0.0.1:1", model="m", samples=20
    )
    outcome = _run(search, directory=tmp_path / "branch")
    assert outcome["status"] == "complete"
    assert outcome["selected_samples"]["G"]["refitted_training_mse"] < 1e-10
