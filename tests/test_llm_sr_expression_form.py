"""LLM-SR in its paper's "w/o Program" form: one expression per hypothesis."""

from __future__ import annotations

import json
import sys
import types
from collections import Counter
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from autoformalism.rebuttal import llm_sr_driver as driver
from autoformalism.rebuttal.llm_sr_upstream import (
    SINGLE_EXPRESSION_INSTRUCTION,
    build_specification,
    single_expression_refusal,
)
from autoformalism.rebuttal.phase_c_vendored_campaign import (
    PhaseCVendoredCampaignPlan,
    llm_sr_hypothesis_settings,
    llm_sr_method_label,
    load_phase_c_vendored_plan,
)

REPO = Path(__file__).resolve().parents[1]
CHANNELS = ("Gp", "meal_rate_g_per_min")
INPUTS = {"Gp": "Gp", "meal_rate_g_per_min": "meal_rate_g_per_min"}


def function(body: str) -> str:
    return "def equation(Gp, meal_rate_g_per_min, params):\n" + body


def starting_function(specification: str) -> str:
    return specification[specification.index("def equation("):]


def test_the_form_starts_from_one_expression_and_asks_for_one():
    single, mapping = build_specification("Task.", CHANNELS, "Gp",
                                          single_expression=True)
    program, same = build_specification("Task.", CHANNELS, "Gp")
    assert mapping == same == INPUTS
    assert SINGLE_EXPRESSION_INSTRUCTION in single
    assert SINGLE_EXPRESSION_INSTRUCTION not in program
    assert single_expression_refusal(starting_function(single), mapping) is None
    # The program form's start assigns before it returns, so it is refused.
    assert "one return statement" in single_expression_refusal(
        starting_function(program), mapping
    )
    assert "    dGp = params[0] * Gp" in program


@pytest.mark.parametrize(
    ("body", "refusal"),
    [
        ("    return params[0] * Gp + params[1] * meal_rate_g_per_min\n", None),
        ('    """Glucose."""\n    return np.exp(params[0] * Gp) - params[1]\n', None),
        ("    return params[0] * np.maximum(Gp - params[1], 0.0)\n", None),
        ("    d = params[0] * Gp\n    return d\n", "one return statement"),
        ("    return params[0] * np.cumsum(meal_rate_g_per_min)\n",
         "not an expression of the current values"),
        ("    return params[0] * meal_rate_g_per_min[:-1]\n",
         "not an expression of the current values"),
        ("    return np.where(Gp > 100, params[0], params[1])\n",
         "not an expression of the current values"),
        ("    return params[0] * (Gp - np.mean(Gp))\n",
         "not an expression of the current values"),
        ("    return\n", "no expression"),
    ],
)
def test_only_one_expression_of_the_present_is_a_hypothesis(body, refusal):
    found = single_expression_refusal(function(body), INPUTS)
    if refusal is None:
        assert found is None
    else:
        assert refusal in found


class _Base:
    """Stands in for their sandbox; records what it was asked to run."""

    calls: ClassVar[list] = []

    def run(self, program, function_to_run, function_to_evolve, inputs,
            test_input, timeout_seconds, **kwargs):
        type(self).calls.append(program)
        return -0.5, True


def _program(body: str) -> str:
    """Their runnable program: the specification with one evolved function."""
    specification, _ = build_specification("Task.", CHANNELS, "Gp",
                                           single_expression=True)
    return specification[: specification.index("def equation(")] + function(body)


def test_the_sandbox_refuses_before_running_and_counts_why():
    _Base.calls = []
    refused: Counter = Counter()
    sandbox = driver.single_expression_sandbox(_Base, INPUTS, refused)()
    ok = _program("    return params[0] * Gp\n")
    assert sandbox.run(ok, "evaluate", "equation", {}, "data", 30) == (-0.5, True)
    many = _program("    d = params[0] * Gp\n    return d\n")
    assert sandbox.run(many, "evaluate", "equation", {}, "data", 30) == (None, False)
    rows = _program("    return params[0] * np.cumsum(Gp)\n")
    assert sandbox.run(rows, "evaluate", "equation", {}, "data", 30) == (None, False)
    assert _Base.calls == [ok]
    assert refused == {
        "the body is not one return statement": 1,
        "not an expression of the current values": 1,
    }


def _fake_upstream(monkeypatch, tmp_path: Path, seen: dict):
    """A checkout whose pipeline records its specification and sandbox class."""
    checkout = tmp_path / "LLM-SR"
    (checkout / "llmsr").mkdir(parents=True)
    (checkout / "llmsr" / "pipeline.py").write_text("", encoding="utf-8")
    sampler = types.ModuleType("llmsr.sampler")
    sampler.LocalLLM = type("LocalLLM", (), {"_do_request": lambda self, c: []})
    sampler.Sampler = type("Sampler", (), {"_global_samples_nums": 1})
    evaluator = types.ModuleType("llmsr.evaluator")
    evaluator.LocalSandbox = _Base
    config = types.ModuleType("llmsr.config")
    config.Config = lambda **kwargs: types.SimpleNamespace(**kwargs)
    config.ExperienceBufferConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    config.ClassConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    pipeline = types.ModuleType("llmsr.pipeline")

    def main(*, specification, inputs, config, max_sample_nums, class_config,
             log_dir):
        seen["specification"] = specification
        sandbox = class_config.sandbox_class()
        seen["sandbox"] = type(sandbox)
        # Their evaluator calls the sandbox once per sample.
        sandbox.run(_program("    d = Gp\n    return d\n"), "evaluate",
                    "equation", inputs, "data", 30)
        sampler.Sampler._global_samples_nums = max_sample_nums + 1

    pipeline.main = main
    package = types.ModuleType("llmsr")
    for name, module in (
        ("llmsr", package), ("llmsr.sampler", sampler),
        ("llmsr.evaluator", evaluator), ("llmsr.config", config),
        ("llmsr.pipeline", pipeline),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return checkout


@pytest.mark.parametrize("single_expression", [False, True])
def test_the_driver_uses_the_form_only_when_declared(
    monkeypatch, tmp_path: Path, single_expression
):
    seen: dict = {}
    checkout = _fake_upstream(monkeypatch, tmp_path, seen)
    _Base.calls = []
    search = driver.build_searcher(
        upstream_root=checkout, base_url="http://127.0.0.1:1", model="m",
        samples=4, single_expression=single_expression,
    )
    rng = np.random.default_rng(0)
    values = rng.normal(size=(20, 2))
    outcome = search(
        channels=CHANNELS, targets=("Gp",), values=values,
        labels=values[:, :1], description="Task.",
        directory=tmp_path / "out", development=(object(), object()),
        context=object(), score_rollout=lambda *a, **k: 0.0,
    )
    record = json.loads(
        (tmp_path / "out" / "llmsr-Gp" / driver.SEARCH_RECORD).read_text()
    )
    if single_expression:
        assert SINGLE_EXPRESSION_INSTRUCTION in seen["specification"]
        assert issubclass(seen["sandbox"], _Base) and seen["sandbox"] is not _Base
        assert _Base.calls == []  # the two-statement sample never ran
        assert record["single_expression_refusals"] == {
            "the body is not one return statement": 1
        }
        assert outcome["accounting"]["single_expression_refusals"] == {
            "the body is not one return statement": 1
        }
    else:
        assert SINGLE_EXPRESSION_INSTRUCTION not in seen["specification"]
        assert seen["sandbox"] is _Base and len(_Base.calls) == 1
        assert "single_expression_refusals" not in record


def _load(name: str) -> dict:
    return load_phase_c_vendored_plan(REPO / "configs" / name).model_dump(mode="json")


def test_the_plans_change_only_the_form_and_its_description():
    smoke, base_smoke = (_load("phase_c_llm_sr_expression_smoke_v1.json"),
                         _load("phase_c_llm_sr_smoke_v5.json"))
    assert {k for k in smoke if smoke[k] != base_smoke[k]} == {
        "hypothesis_form", "budget", "scope_note",
    }
    assert smoke["budget"]["declared"] == base_smoke["budget"]["declared"]
    assert smoke["status"] == "frozen_before_calls"
    full, base_full = (_load("phase_c_llm_sr_expression_hosted_120b_v1.json"),
                       _load("phase_c_llm_sr_hosted_120b_v1.json"))
    assert {k for k in full if full[k] != base_full[k]} == {
        "hypothesis_form", "scope_note",
    }
    assert full["status"] == "frozen_before_calls"
    for plan in (smoke, full):
        assert llm_sr_hypothesis_settings(plan) == {"single_expression": True}
        assert llm_sr_method_label(plan) == "llm_sr_expression"
    assert llm_sr_hypothesis_settings(base_full) == {"single_expression": False}
    assert llm_sr_method_label(base_full) == "llm_sr"


def test_the_seeds_plan_changes_only_the_seeds_and_their_description():
    seeds, frozen = (_load("phase_c_llm_sr_expression_hosted_120b_v1_seeds_2_4.json"),
                     _load("phase_c_llm_sr_expression_hosted_120b_v1.json"))
    assert {k for k in seeds if seeds[k] != frozen[k]} == {
        "repetitions", "budget", "scope_note", "status",
    }
    budget, frozen_budget = seeds["budget"], frozen["budget"]
    assert {k for k in budget if budget[k] != frozen_budget[k]} == {"rationale"}
    assert seeds["status"] == "proposed_pending_review"
    assert sorted(frozen["repetitions"] + seeds["repetitions"]) == [0, 1, 2, 3, 4]
    assert llm_sr_method_label(seeds) == "llm_sr_expression"


def test_only_llm_sr_declares_a_hypothesis_form():
    llm_ode = json.loads(
        (REPO / "configs" / "phase_c_llm_ode_hosted_120b_v1.json").read_text()
    )
    form = _load("phase_c_llm_sr_expression_smoke_v1.json")["hypothesis_form"]
    with pytest.raises(ValueError, match="LLM-SR only"):
        PhaseCVendoredCampaignPlan.model_validate({**llm_ode, "hypothesis_form": form})
