"""Reviewed Phase C role corrections and one early/final target-contract view."""

from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import review_multi_construction
from autoformalism.staged_topology import content_hash
from autoformalism.targets import PublicTargetContract

POLICY = "phase-c-public-contract-consistency-1"
# These exact legacy descriptions specify physical quantities, not mandatory
# mathematical representations. Do not generalize this correction to arbitrary
# future contracts or reinterpret their scientific text at runtime.
LEGACY_ROLE_INFERENCES = {
    "Gp": (
        "dynamic_state",
        "Gp(t): primary nonnegative target representing plasma glucose mass",
    ),
    "U": (
        "instantaneous_process",
        "U(t): total glucose utilization/disposal rate, including "
        "insulin-independent and insulin-dependent contributions",
    ),
}


def correct_roles(cell: dict) -> dict:
    """Correct only reviewed keyword inferences; retain dependencies and provenance."""
    original = PublicTargetContract.model_validate(cell["target_contract"])
    targets, changes = [], []
    for target in original.targets:
        before = target.model_dump(mode="json")
        if original.benchmark_id.startswith(
            "phase_c_dalla_man_"
        ) and LEGACY_ROLE_INFERENCES.get(target.target_channel) == (
            before["expected_representation"],
            before["representation_requirement"],
        ):
            target = type(target).model_validate(
                {
                    **before,
                    "expected_representation": "unspecified",
                    "representation_requirement": None,
                }
            )
            changes.append(
                {
                    "target": target.target_channel,
                    "before": before,
                    "after": target.model_dump(mode="json"),
                    "reason": (
                        "Physical mass/rate description does not mandate "
                        "a state/algebraic representation."
                    ),
                }
            )
        targets.append(target)
    corrected = original.model_copy(update={"targets": tuple(targets)})
    return {
        **cell,
        "target_contract": corrected.model_dump(mode="json"),
        "construction_contract_policy": {
            "policy": POLICY,
            "source_contract_sha256": content_hash(original.model_dump(mode="json")),
            "changes": changes,
        },
    }


def target_definitions(cell: dict) -> dict[str, str]:
    """Project explicit role restrictions for early variable admission."""
    contract = PublicTargetContract.model_validate(cell["target_contract"])
    roles = {"dynamic_state": "differential", "instantaneous_process": "algebraic"}
    return {
        t.target_channel: roles[t.expected_representation.value]
        for t in contract.targets
        if t.expected_representation.value in roles
    }


def visible_brief(cell: dict, task: dict) -> PublicScientificBrief:
    """Reuse the Phase B wrapper: the exact final contract is visible from call one."""
    return review_multi_construction.construction_brief(cell, task)
