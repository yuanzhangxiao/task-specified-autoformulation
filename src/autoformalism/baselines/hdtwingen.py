"""HDTwinGen's loop: D3's, with black-box networks allowed beside white-box terms.

HDTwinGen (Holt et al., NeurIPS 2024; github.com/samholt/HDTwinGen, revision
``HDTWINGEN_UPSTREAM_REVISION``) runs the loop D3's release also runs: class
``NSDT`` in its ``agents.py`` has the same first model, reflection and rewrite
per generation, keeps the 16 lowest validation losses, stops after 20
generations or 20 without improvement, and returns the lowest validation
loss. Its ``config.yaml`` has the same settings as D3's (temperature 0.7,
top-p 0.95, Adam at 1e-2 for 2,000 epochs, validation every 10 epochs,
patience 100), and its fitting is the same one-step update, ``x + f`` with
no dt. So this module reuses ``d3_upstream.run_d3_upstream`` and changes
what HDTwinGen's prompts change.

HDTwinGen's model may add black-box multilayer perceptrons for the residuals
of a white-box model: its prompts say so, and that it should use white-box
models first and switch to hybrid ones only once white-box models stop
improving. Its model writes PyTorch code; ours writes the restricted format,
and declares each network in the fixed form of ``hybrid_networks``, which the
harness builds and fits with the parameters. The prompts change only where
they describe code, and they state the update the fitting uses, as in
``d3_upstream``. D3's released prompts carry the same black-box wording; the
D3 runs here keep its white-box mode, so this module is the hybrid one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field

from autoformalism.baselines.d3_upstream import (
    D3ModelReply,
    LoopVariant,
    completions_text,
    regenerate_prompt,
)
from autoformalism.baselines.d3_upstream import model_format as white_box_format
from autoformalism.baselines.hybrid_networks import MAX_NETWORKS, NetworkSpec
from autoformalism.expressions import ValidationContext

#: The HDTwinGen revision whose prompts and settings this module follows.
HDTWINGEN_UPSTREAM_REVISION = "a7cd335"


class HybridModelReply(D3ModelReply):
    """A model reply that may also declare black-box networks."""

    networks: tuple[NetworkSpec, ...] = Field(default=(), max_length=MAX_NETWORKS)


#: ``utils/prompts.py: system_prompt()``, with "code" as "model".
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
        "You cannot visualize any graphical output. You exist within a machine. "
        "The model can include black box multi-layer perceptions where required.",
        "",
        "When asked for a model, only provide a RFC8259 compliant JSON object "
        "following the required format without deviation.",
        "",
    ]
)


def model_format(context: ValidationContext, available_inputs: Sequence[str]) -> str:
    """D3's model format, plus how a black-box network is declared."""
    return "\n".join(
        [
            white_box_format(context, available_inputs),
            "Black box components: declare each multi-layer perceptron in "
            "networks, with a name; its inputs, as expressions of the states, "
            "inputs, parameters and t above, used exactly as written with no "
            "rescaling; hidden_layers (1 to 3); hidden_width (2 to 64); and "
            "activation (tanh, relu, sigmoid, softplus, elu or silu). A network "
            "has one output, which equations and algebraics use as a plain "
            "symbol with the network's name. Its weights are fitted together "
            f"with the parameters. At most {MAX_NETWORKS} networks. A white box "
            "model declares none.",
        ]
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
            "* You can use any parameters you want and any black box neural "
            "network components (multi-layer perceptrons); however, you have to "
            "define these.",
            "* It is preferable to decompose the system into differential "
            "equations (compartments) if possible.",
            "* You can use any of the listed functions, for example log, exp, "
            "power etc.",
            "* Under no circumstance can you change the model format, only fill in "
            "the model.",
            "* Use initially white box models first and then switch to hybrid "
            "white and black box models for the residuals, only after no further "
            "best program iteration improvement with white box models.",
            "* Make sure your model follows the exact model format specification.",
            "```",
            "",
            "Think step-by-step, and then give the complete full working model. "
            f"You are generating a model for iteration {iteration} out of "
            f"{generations}.",
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
    """``generate_reflection_prompt_with_group()`` for method ``NSDT``."""
    history = "\n".join(
        f"Iteration {index}. Best Val Loss: {item['fitness']}. "
        f"Model description: {item['reply']['model_description']}"
        for index, item in enumerate(history_best)
    )
    completions = completions_text(programs, HybridModelReply)
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
            "model repeats with the same val loss and then only add black box "
            "models to the white box models:```",
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


#: HDTwinGen's loop: D3's search with HDTwinGen's prompts and hybrid replies.
HDTWINGEN = LoopVariant(
    name="hdtwingen_nsdt",
    revision="hdtwingen-1",
    system_prompt=SYSTEM_PROMPT,
    reply_type=HybridModelReply,
    model_format=model_format,
    first_task_prompt=first_task_prompt,
    reflection_prompt=reflection_prompt,
    regenerate_prompt=regenerate_prompt,
)
