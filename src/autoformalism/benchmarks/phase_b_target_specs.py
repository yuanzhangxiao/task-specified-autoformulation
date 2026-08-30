"""Public-prompt-derived target-generation contracts for Phase B."""

from __future__ import annotations

import hashlib

from autoformalism.benchmarks.phase_b_public import (
    PhaseBPublicSpec,
    render_phase_b_prompts,
)
from autoformalism.targets import (
    PublicTargetContract,
    PublicTargetRequirement,
    RequiredTargetDependency,
)


def phase_b_public_target_contract(
    public_spec: PhaseBPublicSpec,
) -> PublicTargetContract:
    """Derive the shared target schema solely from one frozen public prompt."""
    proposer_prompt, _ = render_phase_b_prompts(public_spec)
    targets = tuple(
        _target_requirement(public_spec, item.public_name, item.description)
        for item in public_spec.channels
        if item.role == "target"
    )
    return PublicTargetContract(
        benchmark_id=public_spec.benchmark_id,
        tier=public_spec.tier,
        public_prompt_sha256=hashlib.sha256(
            proposer_prompt.encode("utf-8")
        ).hexdigest(),
        targets=targets,
    )


def _target_requirement(
    spec: PhaseBPublicSpec,
    target: str,
    description: str,
) -> PublicTargetRequirement:
    dependencies: tuple[RequiredTargetDependency, ...] = ()
    public_symbols = {item.public_name for item in spec.channels}
    if (
        spec.family == "dalla_man"
        and spec.semantic_variant == "named"
        and target == "U"
        and "Uii" in public_symbols
    ):
        dependencies = (
            RequiredTargetDependency(
                dependency_id="insulin_independent_contribution",
                acceptable_symbols=("Uii",),
                public_requirement=(
                    "U(t): total glucose utilization/disposal rate, including "
                    "insulin-independent and insulin-dependent contributions; "
                    "Uii(t): supplied insulin-independent contribution to "
                    "glucose utilization"
                ),
            ),
        )
    return PublicTargetRequirement(
        target_channel=target,
        public_requirement=f"generate target {target}(t): {description}",
        required_dependencies=dependencies,
    )
