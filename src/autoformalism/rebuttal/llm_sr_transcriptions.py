"""Reviewed ODE transcriptions of LLM-SR's sealed Phase C programs.

An LLM-SR pick is a numpy program, not an equation. Most picks read the
history of their inputs -- row recursions that filter, delay or simulate a
part of the system the data do not show -- so no equation of the current
values expresses them. The public mechanism tests probe the states of an ODE,
so each pick is transcribed by hand into one: every recursion becomes a hidden
state, under the fixed rules in ``RULES`` (explained in
docs/PHASE_C_BASELINE_MECHANISMS.md).

A transcription names the exact sealed program text (by SHA-256) and fitted
coefficients it transcribes, so it cannot silently stand in for another pick.
Transcriptions are data, read with the external-baseline grammar; nothing in
this module executes a program.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.expressions import ValidationContext
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema

PROTOCOL = "phase-c-llm-sr-ode-transcriptions-1"


class ProgramBinding(StrictSchema):
    """The sealed program and coefficients one target's transcription reads."""

    function_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameters: tuple[float, ...] = Field(min_length=1, max_length=64)


class HiddenState(StrictSchema):
    """One recursion of a program, as an ODE state."""

    rhs: str = Field(min_length=1)
    initial: str = Field(min_length=1)
    program_name: str = Field(min_length=1)
    rule: str = Field(min_length=1)


class Transcription(StrictSchema):
    """One sealed LLM-SR pick as an ODE: targets, hidden states and processes."""

    task: int = Field(ge=0)
    benchmark_id: str
    repetition: int = Field(ge=0)
    programs: dict[str, ProgramBinding] = Field(min_length=1)
    parameters: dict[str, float] = Field(default_factory=dict)
    processes: dict[str, str] = Field(default_factory=dict)
    hidden_states: dict[str, HiddenState] = Field(default_factory=dict)
    targets: dict[str, str] = Field(min_length=1)
    rules: tuple[str, ...] = Field(min_length=1)
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def one_equation_per_program(self) -> Transcription:
        """Every transcribed target has its program, and nothing else does."""
        if set(self.programs) != set(self.targets):
            raise ValueError("each target needs exactly one sealed program binding")
        names = [*self.parameters, *self.processes, *self.hidden_states]
        if len(set(names)) != len(names) or set(names) & set(self.targets):
            raise ValueError("parameter, process and state names must be distinct")
        used = {h.rule for h in self.hidden_states.values()}
        if not used <= set(self.rules):
            raise ValueError("a hidden state cites a rule its transcription omits")
        return self


class TranscriptionSet(StrictSchema):
    """The reviewed file: its rules, and one transcription per sealed pick."""

    protocol: Literal["phase-c-llm-sr-ode-transcriptions-1"]
    status: Literal["proposed_pending_review", "reviewed"]
    source_run: str
    source_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rules: dict[str, str] = Field(min_length=1)
    transcriptions: tuple[Transcription, ...]

    @model_validator(mode="after")
    def rules_are_declared(self) -> TranscriptionSet:
        """Cite only declared rules, and transcribe each task once."""
        tasks = [t.task for t in self.transcriptions]
        if len(set(tasks)) != len(tasks):
            raise ValueError("a task is transcribed twice")
        for item in self.transcriptions:
            unknown = set(item.rules) - set(self.rules)
            if unknown:
                raise ValueError(f"task {item.task} cites undeclared rules {unknown}")
        return self

    def for_task(self, task: int) -> Transcription | None:
        """The transcription of one campaign task, if there is one."""
        return next((t for t in self.transcriptions if t.task == task), None)


def load(path: Path) -> TranscriptionSet:
    """Read and validate a transcription file."""
    return TranscriptionSet.model_validate(json.loads(path.read_text("utf-8")))


def function_sha256(source: str) -> str:
    """The digest a binding names: the sealed function text, byte for byte."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def bind(item: Transcription, programs: dict) -> None:
    """Refuse a transcription of any program but the sealed one.

    ``programs`` is the sealed selection's ``programs`` mapping: per target,
    the function text and the coefficients LLM-SR's own fit gave it.
    """
    if set(programs) != set(item.programs):
        raise ValueError(
            f"task {item.task}: the transcription covers {sorted(item.programs)}, "
            f"the sealed pick {sorted(programs)}"
        )
    for target, binding in item.programs.items():
        record = programs[target]
        if function_sha256(record["function"]) != binding.function_sha256:
            raise ValueError(f"task {item.task}: {target}'s program text differs")
        if tuple(float(v) for v in record["parameters"]) != binding.parameters:
            raise ValueError(f"task {item.task}: {target}'s coefficients differ")


def context_for(context: ValidationContext) -> ValidationContext:
    """The cell's context, letting hidden states start from first-row values.

    A program's recursion starts from its own first-row value, which may read
    a target's first observation (``Gp[0]``) or a coefficient. Both are known
    at the prediction boundary; the default context admits only forcing
    channels in a latent initializer, so this opts in, as fitting does.
    """
    return context.model_copy(update={"fitted_initialization": True})


def candidate(item: Transcription, context: ValidationContext) -> CandidateModel:
    """The transcription as our model schema: identity-observed targets."""
    if list(item.targets) != list(context.targets):
        raise ValueError(
            f"task {item.task}: transcribed targets {list(item.targets)} differ "
            f"from the cell's {list(context.targets)}"
        )
    states = [
        {"name": name, "kind": "observed"} for name in item.targets
    ] + [
        {
            "name": name,
            "kind": "latent",
            "description": f"program recursion {state.program_name} ({state.rule})",
        }
        for name, state in item.hidden_states.items()
    ]
    return CandidateModel.model_validate(
        {
            "candidate_id": f"llm_sr_task{item.task}_transcription",
            "parent_candidate_id": None,
            "change_summary": "reviewed ODE transcription of the sealed program",
            "states": states,
            "processes": [
                {"name": name, "expression": expression}
                for name, expression in item.processes.items()
            ],
            "state_equations": [
                {"state": name, "rhs": rhs} for name, rhs in item.targets.items()
            ]
            + [
                {"state": name, "rhs": state.rhs}
                for name, state in item.hidden_states.items()
            ],
            "observation_mappings": [
                {"channel": name, "expression": name} for name in item.targets
            ],
            "parameters": [
                {"name": name, "scope": "global"} for name in item.parameters
            ],
            "initial_conditions": [
                {"state": name, "scope": "global", "expression": name}
                for name in item.targets
            ]
            + [
                {"state": name, "scope": "global", "expression": state.initial}
                for name, state in item.hidden_states.items()
            ],
        }
    )
