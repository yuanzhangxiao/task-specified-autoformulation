"""Public-prompt-derived mechanism contracts for the Phase-B suite."""

from __future__ import annotations

import hashlib

from autoformalism.benchmarks.phase_b_public import (
    PhaseBPublicSpec,
    phase_b_task_mechanism_lines,
    render_phase_b_prompts,
)
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    MechanismRequirement,
)


def phase_b_public_mechanism_spec(
    public_spec: PhaseBPublicSpec,
) -> MechanismEvaluationSpec:
    """Derive graph predicates solely from one frozen public prompt contract."""
    proposer_prompt, _ = render_phase_b_prompts(public_spec)
    requirements = _requirements(public_spec)
    public_lines = phase_b_task_mechanism_lines(public_spec)
    if len(requirements) != len(public_lines):
        raise ValueError(
            "public mechanism specification must preserve task-bullet cardinality"
        )
    return MechanismEvaluationSpec(
        source="public_prompt",
        benchmark_id=public_spec.benchmark_id,
        tier=public_spec.tier,
        public_prompt_sha256=hashlib.sha256(
            proposer_prompt.encode("utf-8")
        ).hexdigest(),
        required_mechanisms=requirements,
    )


def _requirements(
    spec: PhaseBPublicSpec,
) -> tuple[MechanismRequirement, ...]:
    lines = phase_b_task_mechanism_lines(spec)
    targets = tuple(item.public_name for item in spec.channels if item.role == "target")
    inputs = tuple(
        item.public_name for item in spec.channels if item.role == "external_input"
    )
    if spec.family == "dalla_man":
        return _dalla_requirements(spec, lines, targets, inputs)
    if spec.family == "cstr":
        aliases = (
            (
                "reactor_temperature_balance",
                "feed_transport",
                "reaction_heat_generation",
                "jacket_heat_exchange",
            )
            if spec.semantic_variant == "named"
            else (
                "primary_target_balance",
                "external_transport",
                "state_dependent_source",
                "coupled_exchange",
            )
        )
        return (
            _requirement(
                "controlled_balance"
                if spec.semantic_variant == "named"
                else "coupled_balance",
                lines[0],
                aliases=aliases,
                targets=(targets[0],),
            ),
        )
    aliases = (
        "input_memory_output_pathway",
        "input_driven_memory",
        "dynamic_memory",
        "persistent_coupling",
        "nonlinear_feedback",
        "output_generation",
    )
    return (
        _requirement(
            "input_memory_output_pathway",
            lines[0],
            aliases=aliases,
            drivers=(inputs[0],),
            targets=(targets[0],),
            memory=True,
        ),
    )


def _dalla_requirements(
    spec: PhaseBPublicSpec,
    lines: tuple[str, ...],
    targets: tuple[str, ...],
    inputs: tuple[str, ...],
) -> tuple[MechanismRequirement, ...]:
    named = spec.semantic_variant == "named"
    if spec.task == "T1":
        return (
            _requirement(
                "meal_response" if named else "input_response",
                lines[0],
                aliases=(
                    (
                        "meal_pathway",
                        "meal_appearance",
                        "post_meal_response",
                    )
                    if named
                    else (
                        "causal_input_response",
                        "delayed_input_response",
                        "input_memory",
                    )
                ),
                drivers=(inputs[0],),
                targets=(targets[0],),
            ),
        )
    if spec.task == "T2":
        second_target = targets[2] if len(targets) >= 3 else targets[0]
        return (
            _requirement(
                "meal_pathway" if named else "delayed_input_pathway",
                lines[0],
                aliases=(
                    ("meal_response", "meal_appearance")
                    if named
                    else ("input_response", "delayed_input_response", "input_memory")
                ),
                drivers=(inputs[0],),
                targets=(targets[0],),
                memory=not named,
            ),
            _requirement(
                "delayed_insulin_action" if named else "regulator_dependent_removal",
                lines[1],
                aliases=(
                    (
                        "insulin_action",
                        "insulin_dependent_disposal",
                        "delayed_disposal",
                    )
                    if named
                    else (
                        "delayed_regulatory_removal",
                        "regulatory_removal",
                        "delayed_removal",
                    )
                ),
                drivers=(targets[1],),
                targets=(second_target,),
                memory=True,
            ),
        )
    if spec.task == "T3":
        return (
            _requirement(
                "meal_appearance" if named else "delayed_input_response",
                lines[0],
                aliases=(
                    ("meal_pathway", "meal_response")
                    if named
                    else ("delayed_input_pathway", "input_response", "input_memory")
                ),
                drivers=(inputs[0],),
                targets=(targets[0],),
                memory=not named,
            ),
            _requirement(
                "peripheral_insulin_action"
                if named
                else "peripheral_regulatory_removal",
                lines[1],
                aliases=(
                    ("delayed_disposal", "insulin_dependent_disposal")
                    if named
                    else ("delayed_regulatory_removal", "regulatory_removal")
                ),
                drivers=(targets[1],),
                targets=(targets[0],),
                memory=True,
            ),
            _requirement(
                "hepatic_regulation" if named else "source_regulation",
                lines[2],
                aliases=(
                    ("delayed_hepatic_regulation", "endogenous_production")
                    if named
                    else ("delayed_source_regulation", "regulated_internal_source")
                ),
                drivers=(targets[1],) if not named else (),
                targets=(targets[2],),
                memory=True,
            ),
        )
    aliases = (
        (
            "flux_portrait",
            "meal_appearance",
            "glucose_utilization",
            "endogenous_production",
            "insulin_secretion",
        )
        if named
        else (
            "flux_portrait",
            "delayed_input_response",
            "regulator_dependent_removal",
            "regulated_internal_source",
            "exchange",
            "secondary_target_generation",
        )
    )
    return (
        _requirement(
            "coupled_flux_portrait",
            lines[0],
            aliases=aliases,
            drivers=(inputs[0],),
            targets=targets,
            memory=not named,
        ),
    )


def _requirement(
    identifier: str,
    public_requirement: str,
    *,
    aliases: tuple[str, ...],
    drivers: tuple[str, ...] = (),
    targets: tuple[str, ...],
    memory: bool = False,
) -> MechanismRequirement:
    return MechanismRequirement(
        id=identifier,
        public_requirement=public_requirement,
        tag_aliases=aliases,
        required_drivers=drivers,
        required_targets=targets,
        must_be_generated=True,
        requires_dynamic_memory=memory,
        required_sign="unspecified",
        forbid_current_future_target_input=True,
    )
