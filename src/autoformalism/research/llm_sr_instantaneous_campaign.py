"""Frozen current-value LLM-SR controls; supplied coefficients, training only.

The roster is the four history-free programs in the supplied native campaign,
selected by representation before replay. Native source is authenticated but
never executed. Solver agreement does not certify native program equivalence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.expressions import ValidationContext
from autoformalism.fitting import llm_sr_numerical_control as control
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.llm_sr_instantaneous import compile_current_value
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import llm_sr_control_campaign as common
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.public_fitting import PublicSplit, Sha256

PROTOCOL = "llm-sr-instantaneous-controls-1"
SOURCE_PLAN = "ba3d23cdbad28bd4c165154a3ea86a67a709df2b438e88d0f3b40af1a40f3bd4"
CSTR = "phase_c_cstr_controlled_reactor_mechanism_canonical_named_easy_reset_v1"
DETENTION = "phase_c_detention_independent_noise0_v1"
ROSTER = ((8, CSTR, 0), (9, CSTR, 1), (14, DETENTION, 0), (15, DETENTION, 1))
# Pin the inspected current-value rewrites, including coefficients and guards.
# A different expression requires its own review/protocol, not a new hash field.
TRANSCRIPTIONS = {
    8: "57ca34ee2bc11617e9012d61aa20f4db782bae69e9b8b51a0ed183d8683bfeee",
    9: "71dad188a4636441c568ac99851a6287b688b1434ab88db6f7b0727ba395f858",
    14: "f116ae19f10bae0f1aa8c1c3537aabfb3094371c81b8620644989ba265333d30",
    15: "ca9ccce691b28424ad0880706835840241d5ba4852773f8fc06f235138744de1",
}
TRAINING = {
    CSTR: "83da61198892af209bfb65307332fe43f0ebab1657690e526e6b571fa0ca0057",
    DETENTION: "d8e65a9b55f7ea5b8302e9dc5872f382f4deab9a92e399fcb0a1b3e3b8452da3",
}
CONTEXTS = {
    CSTR: {
        "targets": ("T",),
        "auxiliaries": ("C", "Tj"),
        "external_inputs": ("Cf", "Tf", "Tjf"),
    },
    DETENTION: {
        "targets": ("h_down",),
        "external_inputs": ("inflow_up", "inflow_down"),
        "fixed_covariates": (
            "area_up",
            "area_down",
            "crest_up",
            "crest_down",
            "warning_depth",
            "initial_up",
        ),
    },
}
Policy = common.Policy


class Task(StrictSchema):
    """One native repetition and its complete public training handoff."""

    task: int
    benchmark_id: str
    repetition: int
    programs: dict[Identifier, control.SourceProgram]
    context: ValidationContext
    training: PublicSplit


class Inputs(StrictSchema):
    """Self-contained fixed-value inputs; no downstream scores or private data."""

    protocol: Literal["llm-sr-instantaneous-controls-input-1"]
    source_plan_sha256: Sha256
    transcription_set: control.TranscriptionSet
    tasks: tuple[Task, Task, Task, Task]

    @model_validator(mode="after")
    def current_value_roster(self) -> Inputs:
        """Reject history approximations, altered coefficients and guard misuse."""
        if tuple((t.task, t.benchmark_id, t.repetition) for t in self.tasks) != ROSTER:
            raise ValueError("instantaneous controls require the four frozen tasks")
        if self.source_plan_sha256 != SOURCE_PLAN or (
            self.transcription_set.source_plan_sha256 != self.source_plan_sha256
        ):
            raise ValueError("instantaneous control source plan differs")
        if tuple(t.task for t in self.transcription_set.transcriptions) != (
            8,
            9,
            14,
            15,
        ):
            raise ValueError("transcription roster differs from current-value controls")
        for task in self.tasks:
            if public.content_sha256(task.training) != TRAINING[task.benchmark_id]:
                raise ValueError("training differs from the native plan public release")
            if task.context != ValidationContext(**CONTEXTS[task.benchmark_id]):
                raise ValueError("public context differs from the native plan release")
            item = self.transcription_set.for_task(task.task)
            assert item is not None
            if public.content_sha256(item) != TRANSCRIPTIONS[task.task]:
                raise ValueError("inspected current-value transcription differs")
            if item.hidden_states or set(item.rules) - {
                "direct",
                "guard",
                "coefficients",
            }:
                raise ValueError("history or recurrence approximations are excluded")
            control.checked_training(task.training, task.context)
            control.bind_programs(item, task.programs)
            channels = (
                *task.context.targets,
                *task.context.auxiliaries,
                *task.context.external_inputs,
                *task.context.fixed_covariates,
            )
            if any(p.variables != channels for p in task.programs.values()):
                raise ValueError("native variable order differs from public context")
            if task.task in (14, 15) and any(
                row.fixed_covariates.get("area_down", 0) <= 1
                for row in task.training.rows
            ):
                raise ValueError("dropped native area guard requires area_down > 1")
            compile_current_value(item, task.context)
        return self


def prepare(root: Path, inputs: dict | Inputs, policy: Policy | None = None) -> dict:
    """Seal the four representations and budgets without running numerical work."""
    inputs = Inputs.model_validate(
        inputs.model_dump(mode="json") if isinstance(inputs, Inputs) else inputs
    )
    policy = Policy.model_validate((policy or Policy()).model_dump(mode="json"))
    body = {
        "protocol": PROTOCOL,
        "inputs": inputs.model_dump(mode="json"),
        "policy": policy.model_dump(mode="json"),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "input_sha256": public.content_sha256(inputs),
        "selection_rule": "all four supplied programs with current-value rewrites",
        "training_content_sha256": {
            str(task.task): public.content_sha256(task.training)
            for task in inputs.tasks
        },
    }
    with public._lock(root):
        if not (root / "plan.json").exists() and any(
            p.name != ".lock" for p in root.iterdir()
        ):
            raise ValueError("cannot prepare a nonempty unsealed control directory")
        seal(root / "plan.json", body)
    return report(root)


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, Inputs, Policy]:
    """Authenticate input, current-value rewrite and source/runtime identities."""
    plan = read_seal(root / "plan.json")
    if plan["protocol"] != PROTOCOL:
        raise ValueError("unknown instantaneous-control protocol")
    inputs = Inputs.model_validate(plan["inputs"])
    policy = Policy.model_validate(plan["policy"])
    if plan["input_sha256"] != public.content_sha256(inputs) or plan[
        "training_content_sha256"
    ] != {str(t.task): public.content_sha256(t.training) for t in inputs.tasks}:
        raise ValueError("instantaneous control input identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("instantaneous control execution source/runtime differs")
    return plan, inputs, policy


def run_task(root: Path, index: int) -> dict:
    """Replay one fixed model twice, preserving interrupted stage allowances."""
    plan, inputs, policy = verify(root)
    if index not in range(4):
        raise ValueError("task index must be 0 through 3")
    task = inputs.tasks[index]
    item = inputs.transcription_set.for_task(task.task)
    assert item is not None
    identity = common._identity(plan, task)
    directory = root / "tasks" / str(index)
    with public._lock(directory):
        result_path = directory / "result.json"
        if result_path.exists():
            return common._task_result(directory, identity)
        operations = {}
        for name, settings in (
            ("primary", policy.primary),
            ("verification", policy.verification),
        ):
            operations[name] = journal._operation(
                directory,
                identity,
                name,
                settings.seconds,
                lambda save, settings=settings: control.replay_training(
                    item,
                    task.context,
                    task.training,
                    settings,
                    checkpoint=save,
                    model_factory=compile_current_value,
                ),
            )
        agreement = common._agreement(operations, policy)
        result = {
            "identity": identity,
            "status": "replay_verified"
            if agreement["verified"]
            else "replay_unverified",
            "operations": {
                name: public.content_sha256(value) for name, value in operations.items()
            },
            "solver_agreement": agreement,
            "stage_cost": journal._cost(operations),
            "transcription_review_status": inputs.transcription_set.status,
            "representation": "current_value_rewrite_at_supplied_coefficients",
            "native_program_executed": False,
            "native_equivalence_certified": False,
            "validation_or_test_access": False,
            "optimizer_calls": 0,
            "refit_ready": False,
            "refit_limitation": control.REFIT_LIMIT,
            "model_selected": False,
            "interpretation": (
                "Current-value rewrites have no history approximation. Two ODE solvers "
                "check numerical agreement; native Euler scores do not transfer."
            ),
        }
        seal(result_path, result)
        return common._task_result(directory, identity)


def report(root: Path) -> dict:
    """Keep all four rows, failures and per-stage costs without selecting models."""
    plan, inputs, policy = verify(root, runtime=False)
    rows = []
    for index, task in enumerate(inputs.tasks):
        identity = common._identity(plan, task)
        directory = root / "tasks" / str(index)
        rows.append(
            common._task_result(directory, identity)
            if (directory / "result.json").exists()
            else {"identity": identity, "status": "not_started"}
        )
    return {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "training_content_sha256": plan["training_content_sha256"],
        "policy": policy.model_dump(mode="json"),
        "tasks": rows,
        "status": "complete"
        if all(r["status"] != "not_started" for r in rows)
        else "pending",
        "optimizer_calls": 0,
        "model_selection_performed": False,
        "validation_or_test_access": False,
    }
