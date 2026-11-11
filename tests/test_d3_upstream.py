"""D3's own propose-reflect loop, ported to the restricted model format.

The model and the torch fit are stood in for: what is under test is the loop
upstream's ``D3._run`` defines, meaning which request each generation makes,
what the reflection shows, what is kept, when the search stops and which model
it returns, plus the ported prompts and the reply format.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from autoformalism.baselines import d3_upstream as loop
from autoformalism.baselines.d3 import D3AdapterError
from autoformalism.baselines.d3_native import NativeD3Error, NativeD3Fit
from autoformalism.llm.exceptions import LLMResponseError
from autoformalism.llm.ollama import _ollama_compatible_schema
from autoformalism.rebuttal.phase_c_baselines import load_cell
from tests.test_phase_c_vendored_campaign import CELL, _release

#: The validation loss the stand-in fit gives each right-hand side.
LOSSES = {
    "-k * h_down": 0.5,
    "-k * h_down ** 2": 0.2,
    "-k * sqrt(h_down)": 0.3,
    "-k * h_down + area_down": 0.7,
    "-k * h_down - area_up": 0.6,
}


class Down(Exception):
    """An endpoint that stays down: not a generation's failure."""


def _reply(rhs: str, *, start: float = 0.5, description: str | None = None):
    return loop.D3ModelReply.model_validate(
        {
            "model_description": description or f"white box: {rhs}",
            "model": {
                "candidate_id": "m",
                "states": [
                    {
                        "name": "h_down",
                        "kind": "observed",
                        "observed_channel": "h_down",
                        "rhs": rhs,
                    }
                ],
                "parameters": [{"name": "k"}],
            },
            "parameter_initial_values": [{"name": "k", "value": start}],
        }
    )


class Chat:
    """Answers each call with the next scripted reply, or raises it."""

    def __init__(self, models, reflections=()) -> None:
        self.models = iter(models)
        self.reflections = iter(reflections)
        self.calls: list[tuple[str, int, list[dict]]] = []

    def _next(self, items):
        item = next(items)
        if isinstance(item, Exception):
            raise item
        return item

    def reflect(self, messages, *, generation):
        self.calls.append(("reflect", generation, copy.deepcopy(messages)))
        return self._next(self.reflections)

    def write_model(self, messages, *, generation):
        self.calls.append(("model", generation, copy.deepcopy(messages)))
        return self._next(self.models)


@pytest.fixture
def cell(tmp_path):
    return load_cell(_release(tmp_path / "release"), CELL)


@pytest.fixture
def fits(monkeypatch):
    """Stand in for the torch fit; record what each fit started from."""
    started: list[dict] = []

    def fit(candidate, training, validation, *, targets, seed, initial_values):
        started.append(dict(initial_values))
        return NativeD3Fit({"k": 0.9}, 0.11, 0.12, 40, {"h_down": 1.0})

    def losses(candidate, split, parameters):
        loss = LOSSES[candidate.state_equations[0].rhs]
        return loss, {"h_down": loss}

    monkeypatch.setattr(loop, "fit_native_d3", fit)
    monkeypatch.setattr(loop, "raw_state_losses", losses)
    return started


def _run(cell, chat, directory: Path, **settings):
    return loop.run_d3_upstream(
        cell.dataset,
        cell.context,
        task_prompt=cell.prompt,
        chat=chat,
        settings=loop.D3Settings(**settings),
        seed=0,
        work_directory=directory,
        identity={"task": 0},
    )


def test_each_generation_reflects_then_writes_and_the_lowest_loss_is_kept(
    cell, fits, tmp_path
):
    chat = Chat(
        [_reply("-k * h_down"), _reply("-k * h_down ** 2", start=2.0),
         _reply("-k * sqrt(h_down)")],
        ["Use a square.", "Try a root."],
    )
    result = _run(cell, chat, tmp_path, generations=3, patience=3)

    assert [(kind, generation) for kind, generation, _ in chat.calls] == [
        ("model", 0), ("reflect", 1), ("model", 1), ("reflect", 2), ("model", 2)
    ]
    first, reflect_1, model_1, reflect_2, _ = (call[2] for call in chat.calls)
    assert [message["role"] for message in first] == ["system", "user"]
    assert "Coupled stormwater basins" in first[1]["content"]
    assert "iteration 0 out of 3" in first[1]["content"]
    # One conversation per generation: the first request, the reflection
    # request, the reflection itself, then the request for a new model.
    assert reflect_1 == model_1[:3] and model_1[:2] == first
    assert [message["role"] for message in model_1] == [
        "system", "user", "user", "assistant", "user"
    ]
    assert model_1[3]["content"] == "Use a square."
    assert "regenerate the model" in model_1[4]["content"]
    assert "iteration 1 out of 3" in model_1[4]["content"]

    # The second reflection shows both models, the lowest loss last, with
    # their fitted values, and the best model after each generation.
    shown = reflect_2[2]["content"]
    completions = shown.split("sorted for the lowest validation loss last")[1]
    assert completions.index("Val Loss: 0.5 (") < completions.index("Val Loss: 0.2 (")
    assert "h_down val loss: 0.2" in shown
    assert "optimized_parameters = {'k': 0.9}" in shown
    assert (
        "Iteration 0. Best Val Loss: 0.5. Model description: white box: -k * h_down"
        in shown
    )
    assert "Iteration 1. Best Val Loss: 0.2." in shown

    assert result.selection_payload["selected_generation"] == 1
    assert result.selection_payload["validation_loss"] == 0.2
    assert result.selection_payload["initial_values"] == {"k": 2.0}
    assert result.selected_hyperparameters["selection"] == loop.SELECTION
    assert result.equations == {"h_down": "-k * h_down ** 2"}
    assert result.validation_normalized_mse == 0.12
    assert fits == [{"k": 0.5}, {"k": 2.0}, {"k": 0.5}]


def test_only_the_lowest_losses_are_kept_and_patience_counts_from_generation_1():
    def record(generation, fitness):
        return {**loop._record(generation, "reflect"), "fitness": fitness,
                "reply": {"model_description": str(fitness)}}

    records = [record(0, 0.5), record(1, 0.4), record(2, 0.3), record(3, 0.6)]
    programs, history, stagnant = loop.replay(records, keep_top_samples=2)
    assert [item["fitness"] for item in programs] == [0.3, 0.4]
    assert [item["fitness"] for item in history] == [0.5, 0.4, 0.3, 0.3]
    assert stagnant == 1


def test_a_failed_or_repeated_model_costs_its_generation(cell, fits, tmp_path):
    chat = Chat(
        [_reply("-k * h_down"), LLMResponseError("cut off"), _reply("-k * h_down")],
        ["Again.", "Once more."],
    )
    result = _run(cell, chat, tmp_path, generations=3, patience=3)
    assert len(fits) == 1  # the repeated model was not fitted again
    records = loop._load(tmp_path / "d3_checkpoint.json",
                         loop._fingerprint(cell.dataset, loop.D3Settings(3, 3), 0,
                                           {"task": 0}))
    assert records[1]["error"].startswith("LLMResponseError: cut off")
    assert records[2]["duplicate_of"] == 0
    assert result.selection_payload["selected_generation"] == 0


def test_without_a_model_the_next_generation_asks_for_a_first_one_again(
    cell, fits, tmp_path
):
    chat = Chat([_reply("-k * h_down + undeclared"), _reply("-k * h_down")])
    result = _run(cell, chat, tmp_path, generations=2, patience=2)
    assert [(kind, generation) for kind, generation, _ in chat.calls] == [
        ("model", 0), ("model", 1)
    ]
    again = chat.calls[1][2]
    assert [message["role"] for message in again] == ["system", "user"]
    assert "iteration 1 out of 2" in again[1]["content"]
    assert result.selection_payload["selected_generation"] == 1


def test_no_usable_model_is_a_discovery_failure(cell, fits, tmp_path):
    chat = Chat([LLMResponseError("empty"), LLMResponseError("empty")])
    with pytest.raises(D3AdapterError, match="no valid fitted candidates"):
        _run(cell, chat, tmp_path, generations=2, patience=2)


def test_patience_ends_the_search_as_upstream_counts_it(cell, fits, tmp_path):
    chat = Chat(
        [_reply("-k * h_down"), _reply("-k * h_down ** 2"),
         _reply("-k * h_down - area_up"), _reply("-k * h_down + area_down")],
        ["r1", "r2", "r3"],
    )
    result = _run(cell, chat, tmp_path, generations=6, patience=2)
    assert [generation for kind, generation, _ in chat.calls if kind == "model"] == [
        0, 1, 2, 3
    ]
    assert result.selected_hyperparameters["generations_completed"] == 4


def test_a_resumed_search_asks_only_for_what_is_left(cell, fits, tmp_path):
    with pytest.raises(Down):
        _run(cell, Chat([_reply("-k * h_down")], [Down()]), tmp_path,
             generations=2, patience=2)
    chat = Chat([_reply("-k * h_down ** 2")], ["Use a square."])
    result = _run(cell, chat, tmp_path, generations=2, patience=2)
    assert [(kind, generation) for kind, generation, _ in chat.calls] == [
        ("reflect", 1), ("model", 1)
    ]
    assert result.selection_payload["selected_generation"] == 1
    with pytest.raises(D3AdapterError, match="fingerprint"):
        loop.run_d3_upstream(
            cell.dataset, cell.context, task_prompt=cell.prompt, chat=Chat([]),
            settings=loop.D3Settings(2, 2), seed=0, work_directory=tmp_path,
            identity={"task": 1},
        )


def test_starting_values_name_declared_parameters_once():
    assert loop.initial_values(_reply("-k * h_down", start=3.0)) == {"k": 3.0}
    reply = _reply("-k * h_down").model_dump(mode="json")
    for values in (
        [{"name": "q", "value": 1.0}],
        [{"name": "k", "value": 1.0}, {"name": "k", "value": 2.0}],
    ):
        with pytest.raises(NativeD3Error):
            loop.initial_values(
                loop.D3ModelReply.model_validate(
                    {**reply, "parameter_initial_values": values}
                )
            )
    left_out = loop.D3ModelReply.model_validate(
        {**reply, "parameter_initial_values": []}
    )
    assert loop.initial_values(left_out) == {}


def test_the_prompts_keep_upstreams_wording_except_about_code(cell):
    form = loop.model_format(cell.context, ("inflow_up", "area_down"))
    first = loop.first_task_prompt("A system.", form, generations=20)
    reflection = loop.reflection_prompt([], [], iteration=1, generations=20)
    regenerate = loop.regenerate_prompt(form, iteration=1, generations=20)
    for kept in (
        "You must act autonomously and you will receive no human input at any stage.",
        "You cannot visualize any graphical output. You exist within a machine.",
    ):
        assert kept in loop.SYSTEM_PROMPT
    for kept in (
        "* The observed training dataset has very few samples, and the model must "
        "be able to generalize to unseen data.",
        "* It is preferable to decompose the system into differential equations "
        "(compartments) if possible.",
        "You are generating a model for iteration 0 out of 20.",
    ):
        assert kept in first
    assert "sorted for the lowest validation loss last" in reflection
    assert "Provide only actionable feedback" in reflection
    assert "You cannot change the model format, or input variables." in regenerate
    for prompt in (loop.SYSTEM_PROMPT, first, reflection, regenerate):
        assert "PyTorch" not in prompt and "multi-layer" not in prompt
        assert "black box models to the white box" not in prompt
    assert "x_next = x + rhs" in form and "no dt" in form
    assert "States, each needing exactly one equation: h_down" in form
    assert "Inputs every equation may use: h_down, inflow_up, area_down, t" in form


def test_the_reply_format_is_a_strict_schema_a_server_can_enforce():
    schema = _ollama_compatible_schema(
        loop.D3ModelReply.model_json_schema(mode="validation")
    )
    assert set(schema["properties"]) == {
        "model_description", "model", "parameter_initial_values"
    }
    with pytest.raises(ValueError):
        loop.D3ModelReply.model_validate(
            {**_reply("-k").model_dump(mode="json"), "code": "import os"}
        )
