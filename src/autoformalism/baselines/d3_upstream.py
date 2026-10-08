"""D3's own propose-reflect loop, with models written in our restricted format.

Phase B ran D3 through our one-call adapter (``d3.py``). This module runs the
loop D3's authors released instead: ``agents.py``, class ``D3``, at the
revision ``D3_UPSTREAM_REVISION`` of github.com/samholt/DataDrivenDiscovery,
with its prompts and its search settings.

* Generation 0 asks for a first model. Every later generation makes two calls
  in one conversation. The model first reflects on the best models so far,
  each shown with its validation loss, the loss on every state, and its fitted
  parameter values, sorted so the lowest loss comes last, and on the best model
  after every earlier generation. It then writes a new model using that
  reflection.
* Each new model's parameters are fitted on the training data. Its validation
  loss is the one-step squared error over every state it models. A model
  already fitted is not fitted again. After each generation only the
  ``keep_top_samples`` lowest losses are kept.
* The search ends after ``generations`` generations, or after ``patience``
  generations without a lower best loss. It returns the model with the lowest
  validation loss: D3's own selection rule.

Three things cannot be upstream's, and the module docstring is their record:

1. D3 writes PyTorch code that its harness executes. We never execute
   model-written code, so the model writes the same content as equations in
   our restricted grammar. The neural-network components D3 may add cannot be
   expressed, so this is D3's white-box mode. Its prompts change only where
   they describe code or such components.
2. Upstream's prompt says the model is used with an ODE solver, while its
   fitting adds the model's output to the current state with no dt. The
   prompts here state the update the fitting actually uses.
3. Upstream cannot continue when its first model fails to run. Here a
   generation that yields no usable model is recorded as failed, and while no
   model exists the next generation asks for a first model again.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field

from autoformalism.baselines.d3 import (
    D3_UPSTREAM_REVISION,
    D3AdapterError,
    _numeric_available_inputs,
)
from autoformalism.baselines.d3_native import (
    NativeD3Error,
    fit_native_d3,
    raw_state_losses,
    validate_native_candidate,
)
from autoformalism.baselines.models import BaselineDevelopmentResult
from autoformalism.data import DevelopmentDataset
from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.expressions.parser import BASELINE_FUNCTION_ARITY
from autoformalism.llm.exceptions import LLMProviderError, LLMResponseError
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.schemas import (
    CandidateModel,
    ProposerCandidateV2,
    enrich_proposal_v2,
)
from autoformalism.schemas.base import (
    FiniteFloat,
    Identifier,
    NonEmptyText,
    StrictSchema,
)

#: Names the selection rule in sealed records.
SELECTION = "lowest_one_step_validation_loss_over_modeled_states"

#: What a generation's failure may be; anything else, such as an endpoint that
#: stays down, stops the task instead of costing it a generation.
GENERATION_FAILURES = (
    ArithmeticError,
    D3AdapterError,
    NativeD3Error,
    LLMProviderError,
    LLMResponseError,
    ModelValidationError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True)
class D3Settings:
    """Upstream ``config.yaml``: generations, d3_patience, keep_top_samples."""

    generations: int = 20
    patience: int = 20
    keep_top_samples: int = 16


class D3InitialValue(StrictSchema):
    """The value one parameter's fitting starts from."""

    name: Identifier
    value: FiniteFloat


class D3ModelReply(StrictSchema):
    """D3's code-writing call, with the code in our restricted format.

    Upstream's function takes the model's code and a concise description of
    it. The code also sets each parameter's starting value, which D3's prompts
    use: the fitted values are shown back so that later models can start
    nearer them. The format carries those starting values explicitly.
    """

    model_description: NonEmptyText
    model: ProposerCandidateV2
    parameter_initial_values: tuple[D3InitialValue, ...] = Field(
        default=(), max_length=256
    )


class D3Chat(Protocol):
    """The two calls D3 makes; each is cached and logged by its implementation."""

    def reflect(self, messages: list[dict[str, str]], *, generation: int) -> str:
        """Return the model's free-text reflection."""

    def write_model(
        self, messages: list[dict[str, str]], *, generation: int
    ) -> D3ModelReply:
        """Return the model's next structured model."""


# --- prompts: upstream's wording, changed only where it describes code ------

#: ``utils/prompts.py: system_prompt()``.
SYSTEM_PROMPT = "\n".join(
    [
        "",
        "Objective: Write a model to create an effective differential equation "
        "simulator for a given task.",
        "Please note that the model should be fully functional. No placeholders.",
        "",
        "You must act autonomously and you will receive no human input at any "
        "stage. You have to return as output the complete model for completing "
        "this task, and correctly improve the model to create the most accurate "
        "and realistic simulator possible.",
        "You always write out the model contents.",
        "You cannot visualize any graphical output. You exist within a machine.",
        "",
        "When asked for a model, only provide a RFC8259 compliant JSON object "
        "following the required format without deviation.",
        "",
    ]
)


def model_format(
    context: ValidationContext, available_inputs: Sequence[str]
) -> str:
    """What upstream's code skeleton states: the inputs, and one output per state."""
    states = (*context.targets, *context.auxiliaries)
    inputs = tuple(dict.fromkeys((*states, *available_inputs, "t")))
    functions = ", ".join(sorted(BASELINE_FUNCTION_ARITY))
    return (
        f"States, each needing exactly one equation: {', '.join(states)}\n"
        f"Inputs every equation may use: {', '.join(inputs)}\n"
        "Each equation is its state's change from one sample to the next: the "
        "fitting computes x_next = x + rhs from the measured state x, with no dt "
        "multiplier.\n"
        "Write the model as JSON: one state of kind \"observed\" for every state "
        "above, with observed_channel equal to its name and rhs its equation; any "
        "named intermediate quantity as an algebraic; every parameter as a global "
        "parameter, with the value its fitting starts from in "
        "parameter_initial_values. Do not add other states.\n"
        "Equations may use + - * / ** and these functions: "
        f"{functions}."
    )


def first_task_prompt(
    system_description: str,
    form: str,
    *,
    generations: int,
    iteration: int = 0,
) -> str:
    """``utils/prompts.py: first_task_prompt()``."""
    return "\n".join(
        [
            "",
            "You will get a system description to write a differential equation "
            "simulator for.",
            "",
            "System Description:```",
            system_description,
            "```",
            "",
            "Modelling goals:```",
            "* The parameters of the model will be optimized to an observed "
            "training dataset with the given simulator.",
            "* The observed training dataset has very few samples, and the model "
            "must be able to generalize to unseen data.",
            "```",
            "",
            "Requirement Specification:```",
            "* The model generated should achieve the lowest possible validation "
            "loss, of 1e-10 or less.",
            "* The model generated should be interpretable, and fit the dataset as "
            "accurately as possible.",
            "```",
            "",
            "Model format to fill in:```",
            form,
            "```",
            "",
            "Useful to know:```",
            "* You are a model evolving machine, and you will be called "
            f"{generations} times to generate a model, and improve the model to "
            "achieve the lowest possible validation loss.",
            "* The model defines each state's change from one sample to the next, "
            "and its parameters are fitted to the observed training dataset by "
            "one-step predictions from the measured states.",
            "* You can use any parameters you want; however, you have to define "
            "these.",
            "* It is preferable to decompose the system into differential "
            "equations (compartments) if possible.",
            "* You can use any of the listed functions, for example log, exp, "
            "power etc.",
            "* Under no circumstance can you change the model format, only fill in "
            "the model.",
            "* Use white box models; black box components cannot be expressed in "
            "this format.",
            "* Make sure your model follows the exact model format specification.",
            "```",
            "",
            "Think step-by-step, and then give the complete full working model. "
            f"You are generating a model for iteration {iteration} out of "
            f"{generations}.",
            "",
        ]
    )


def _completion(record: Mapping[str, Any]) -> str:
    """``generate_reflection_competition_for_generation_dict()``."""
    shares = ", ".join(
        f"{state} val loss: {value:.3g}"
        for state, value in record["fitness_per_state"].items()
    )
    # Upstream shows the code as the model wrote it. The model as written is
    # shown here without the fields left at their defaults, which keeps 16 of
    # them well inside a served context window.
    reply = D3ModelReply.model_validate(record["reply"]).model_dump(
        mode="json", exclude_defaults=True
    )
    model = json.dumps(
        {
            "model": reply["model"],
            "parameter_initial_values": reply.get("parameter_initial_values", []),
        },
        ensure_ascii=False,
    )
    return "\n".join(
        [
            "",
            f"Val Loss: {record['fitness']:.3g} (Where the val loss per dimension "
            f"is {shares}) Iteration: {record['generation']}",
            "###",
            "```",
            model,
            "```",
            f"optimized_parameters = {dict(record['parameters'])}",
            "###",
            "",
            "",
        ]
    )


def reflection_prompt(
    programs: Sequence[Mapping[str, Any]],
    history_best: Sequence[Mapping[str, Any]],
    *,
    iteration: int,
    generations: int,
) -> str:
    """``generate_reflection_prompt_with_group()`` for method ``D3``."""
    history = "\n".join(
        f"Iteration {index}. Best Val Loss: {item['fitness']}. "
        f"Model description: {item['reply']['model_description']}"
        for index, item in enumerate(history_best)
    )
    completions = "\n".join(
        _completion(item)
        for item in sorted(programs, key=lambda item: item["fitness"], reverse=True)
    )
    return "\n".join(
        [
            "",
            "You generated the following model completions, which then had their "
            "parameters optimized to the training dataset. Please reflect on how "
            "you can improve the model to minimize the validation loss to 1e-6 or "
            "less. The model examples are delineated by ###.",
            "",
            "Here are your previous iterations the best programs generated. Use it "
            "to see if you have exhausted white box models, i.e. when a white box "
            "model repeats with the same val loss:```",
            history,
            "```",
            "",
            "Here are the top model completions so far that you have generated, "
            "sorted for the lowest validation loss last:```",
            completions,
            "```",
            "",
            "Please reflect on how you can improve the model to fit the dataset as "
            "accurately as possible, and be interpretable. Think step-by-step. "
            "Provide only actionable feedback, that has direct changes to the "
            "model. Do not write out the model, only describe how it can be "
            "improved. Where applicable use the values of the optimized parameters "
            "to reason how the model can be improved to fit the dataset as "
            "accurately as possible. This is for generating a new model for the "
            f"next iteration {iteration} out of {generations}.",
            "",
        ]
    )


def regenerate_prompt(form: str, *, iteration: int, generations: int) -> str:
    """The code request ``D3._run`` sends after each reflection."""
    return "\n".join(
        [
            "",
            "Please now regenerate the model, with the aim to improve the model to "
            "achieve a lower validation error. Use the feedback where applicable. "
            f"You are generating a model for iteration {iteration} out of "
            f"{generations} total iterations. When generating the model if you are "
            "unsure about something, take your best guess. You have to generate a "
            "model, and cannot give an empty answer.",
            "",
            "Please always only fill in the following model format:```",
            form,
            "```",
            "You cannot change the model format, or input variables.",
            "",
        ]
    )


# --- the search -------------------------------------------------------------


def initial_values(reply: D3ModelReply) -> dict[str, float]:
    """Starting values by parameter; a parameter left out starts from 1."""
    names = [item.name for item in reply.parameter_initial_values]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise NativeD3Error(f"parameters given two starting values: {duplicated}")
    undeclared = sorted(set(names) - {item.name for item in reply.model.parameters})
    if undeclared:
        raise NativeD3Error(f"starting values for undeclared parameters: {undeclared}")
    return {item.name: float(item.value) for item in reply.parameter_initial_values}


def replay(
    records: Sequence[Mapping[str, Any]], keep_top_samples: int
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], int]:
    """D3's memory and patience after the given generations, as ``D3._run`` keeps them.

    Returns the kept models (lowest loss first), the best model after every
    generation that had one, and the generations since the best loss last
    fell, counted from generation 1 as upstream counts them.
    """
    programs: list[Mapping[str, Any]] = []
    history: list[Mapping[str, Any]] = []
    best = math.inf
    stagnant = 0
    for record in records:
        if record["fitness"] is not None and record["duplicate_of"] is None:
            programs.append(record)
        programs = sorted(programs, key=lambda item: item["fitness"])
        if record["generation"] >= 1:
            programs = programs[:keep_top_samples]
        if programs:
            history.append(programs[0])
        if record["generation"] >= 1:
            top = programs[0]["fitness"] if programs else math.inf
            if top < best:
                best, stagnant = top, 0
            else:
                stagnant += 1
    return programs, history, stagnant


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _record(generation: int, request: str) -> dict[str, Any]:
    return {
        "generation": generation,
        "request": request,
        "reflection": None,
        "reply": None,
        "candidate": None,
        "initial_values": {},
        "parameters": {},
        "fitness": None,
        "fitness_per_state": {},
        "training_mse": None,
        "validation_mse": None,
        "target_scales": {},
        "epochs_completed": None,
        "duplicate_of": None,
        "error": None,
    }


def run_d3_upstream(
    dataset: DevelopmentDataset,
    context: ValidationContext,
    *,
    task_prompt: str,
    chat: D3Chat,
    settings: D3Settings,
    seed: int,
    work_directory: Path,
    identity: Mapping[str, object],
) -> BaselineDevelopmentResult:
    """Run D3's loop on development data and return its selected model.

    Each finished generation is checkpointed, so an interrupted task resumes
    after its last one; `identity` names the run and must match on resume.
    Test data are never read.
    """
    work_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = work_directory / "d3_checkpoint.json"
    fingerprint = _fingerprint(dataset, settings, seed, identity)
    records = _load(checkpoint, fingerprint)
    observed = (*context.targets, *context.auxiliaries)
    available = _numeric_available_inputs(dataset, context)
    form = model_format(context, available)
    first = first_task_prompt(task_prompt, form, generations=settings.generations)
    base = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": first},
    ]
    while True:
        programs, history, stagnant = replay(records, settings.keep_top_samples)
        generation = len(records)
        if generation >= settings.generations:
            break
        if generation >= 2 and stagnant >= settings.patience:
            break
        record = _record(generation, "reflect" if programs else "first")
        fitted = {
            _canonical(item["reply"]): item["generation"]
            for item in records
            if item["fitness"] is not None and item["duplicate_of"] is None
        }
        try:
            if programs:
                messages = [
                    *base,
                    {
                        "role": "user",
                        "content": reflection_prompt(
                            programs,
                            history,
                            iteration=generation,
                            generations=settings.generations,
                        ),
                    },
                ]
                record["reflection"] = chat.reflect(
                    list(messages), generation=generation
                )
                messages += [
                    {"role": "assistant", "content": record["reflection"]},
                    {
                        "role": "user",
                        "content": regenerate_prompt(
                            form,
                            iteration=generation,
                            generations=settings.generations,
                        ),
                    },
                ]
            elif generation == 0:
                messages = list(base)
            else:
                again = first_task_prompt(
                    task_prompt,
                    form,
                    generations=settings.generations,
                    iteration=generation,
                )
                messages = [base[0], {"role": "user", "content": again}]
            reply = chat.write_model(messages, generation=generation)
            record["reply"] = reply.model_dump(mode="json")
            key = _canonical(record["reply"])
            if key in fitted:
                record["duplicate_of"] = fitted[key]
            else:
                _fit(record, reply, dataset, context, observed, available, seed)
        except GENERATION_FAILURES as exc:
            record["error"] = f"{type(exc).__name__}: {str(exc)[:2000]}"
        records.append(record)
        atomic_json(
            checkpoint,
            {
                "protocol_version": "d3-upstream-1",
                "fingerprint": fingerprint,
                "records": records,
            },
        )
    return _selected(records, dataset, context, settings, seed, observed)


def _fit(
    record: dict[str, Any],
    reply: D3ModelReply,
    dataset: DevelopmentDataset,
    context: ValidationContext,
    observed: tuple[str, ...],
    available: tuple[str, ...],
    seed: int,
) -> None:
    """Fit one new model and score it as D3 does, filling in its record."""
    candidate = enrich_proposal_v2(reply.model, context.targets)
    record["candidate"] = candidate.model_dump(mode="json")
    starts = initial_values(reply)
    record["initial_values"] = starts
    validate_native_candidate(candidate, observed, available)
    fit = fit_native_d3(
        candidate,
        dataset.train,
        dataset.validation,
        targets=context.targets,
        seed=seed + record["generation"],
        initial_values=starts,
    )
    loss, per_state = raw_state_losses(candidate, dataset.validation, fit.parameters)
    record.update(
        parameters=dict(fit.parameters),
        fitness=loss,
        fitness_per_state=per_state,
        training_mse=fit.training_mse,
        validation_mse=fit.validation_mse,
        target_scales=dict(fit.target_scales),
        epochs_completed=fit.epochs_completed,
    )


def _selected(
    records: Sequence[Mapping[str, Any]],
    dataset: DevelopmentDataset,
    context: ValidationContext,
    settings: D3Settings,
    seed: int,
    observed: tuple[str, ...],
) -> BaselineDevelopmentResult:
    """D3's choice: the kept model with the lowest validation loss."""
    programs, _, _ = replay(records, settings.keep_top_samples)
    if not programs:
        failures = "; ".join(
            f"generation {item['generation']}: {item['error']}" for item in records
        )
        raise D3AdapterError(
            "D3 produced no valid fitted candidates. " + failures[-6000:]
        )
    best = programs[0]
    candidate = CandidateModel.model_validate(best["candidate"])
    hyperparameters: dict[str, float | int | str] = {
        "loop": "upstream_reflection",
        "upstream_revision": D3_UPSTREAM_REVISION,
        "generations_completed": len(records),
        "selected_generation": int(best["generation"]),
        "selection": SELECTION,
        "selected_validation_loss": float(best["fitness"]),
        "keep_top_samples": settings.keep_top_samples,
        "external_tools_enabled": "false",
        "parameter_fitting": "pytorch_adam",
        "state_update": "teacher_forced_forward_euler",
        "learning_rate": 1e-2,
        "maximum_epochs": 2_000,
        "validation_interval": 10,
        "early_stopping_patience_checks": 100,
        "optimizer_device": "cpu",
        "parameter_initialization": "model_stated_starting_value_else_one",
        "selected_parameters": json.dumps(dict(best["parameters"]), sort_keys=True),
        "modeled_observed_channels": json.dumps(observed),
        "selected_epochs_completed": int(best["epochs_completed"]),
    }
    return BaselineDevelopmentResult(
        method="d3_native_no_tools",
        benchmark_id=dataset.benchmark_id,
        tier=dataset.tier,
        seed=seed,
        equations={item.state: item.rhs for item in candidate.state_equations},
        selected_hyperparameters=hyperparameters,
        selection_payload={
            "candidate": best["candidate"],
            "parameters": dict(best["parameters"]),
            "target_scales": dict(best["target_scales"]),
            "selected_generation": int(best["generation"]),
            "initial_values": dict(best["initial_values"]),
            "validation_loss": float(best["fitness"]),
            "model_description": best["reply"]["model_description"],
        },
        training_normalized_mse=float(best["training_mse"]),
        validation_normalized_mse=float(best["validation_mse"]),
        test_data_opened=False,
    )


def _fingerprint(
    dataset: DevelopmentDataset,
    settings: D3Settings,
    seed: int,
    identity: Mapping[str, object],
) -> str:
    payload = {
        "benchmark_id": dataset.benchmark_id,
        "tier": dataset.tier,
        "train": dataset.train.fingerprint,
        "validation": dataset.validation.fingerprint,
        "settings": asdict(settings),
        "seed": seed,
        "identity": dict(identity),
        "loop_revision": "d3-upstream-1",
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _load(path: Path, fingerprint: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint:
        raise D3AdapterError("D3 checkpoint fingerprint does not match this run")
    return list(payload["records"])
