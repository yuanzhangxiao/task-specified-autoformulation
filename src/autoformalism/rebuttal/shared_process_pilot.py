"""Matched guidance pilot and descriptive equation-derived reporting.

No new fitter, judge or automatic law tying. Missing models stay unavailable;
syntax reuse and deterministic predicates are not scientific certification.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.shared_process_audit import inventory
from autoformalism.schemas import CandidateModel


def tasks(config) -> list[dict]:
    """Pair seeds/prompts, alternating guidance order to balance server ordering."""
    result = []
    for ci, cell in enumerate(config.public_cells):
        for seed in config.seeds:
            arms = ["full", "brief_only"]
            if (ci + seed) % 2:
                arms.reverse()
            for arm in arms:
                policies = ["off", "on"]
                if (ci + seed + (arm == "brief_only")) % 2:
                    policies.reverse()
                for guidance in policies:
                    result.append(
                        {
                            "index": len(result),
                            "cell": cell,
                            "seed": seed,
                            "arm": arm,
                            "process_guidance": guidance,
                            "task_id": (
                                f"cell{ci:02d}_seed{seed}_{arm}_shared_{guidance}"
                            ),
                            "shared_round_zero": None,
                        }
                    )
    return result


def model_evidence(selected: dict | None) -> dict | None:
    """Inspect actual retained equations without trajectories or hidden references."""
    if selected is None:
        return None
    bundle = selected["bundle"]
    candidate = CandidateModel.model_validate(bundle["candidate"])
    context = ValidationContext.model_validate(bundle["initialization"]["context"])
    evidence = inventory(candidate, context)
    shared = evidence["existing_shared_processes"]
    evidence["counts"].update(
        named_processes_shared_between_governing_definitions=sum(
            sum(not c.startswith("output:") for c in p["consumers"]) >= 2
            for p in shared
        ),
        named_processes_shared_between_state_equations=sum(
            sum(c.startswith("state:") for c in p["consumers"]) >= 2 for p in shared
        ),
    )
    certificate = selected["certificate"]
    evidence["public_mechanism_predicates"] = certificate["mechanisms"].get(
        "predicates", []
    )
    evidence["public_target_predicates"] = certificate["targets"].get("predicates", [])
    evidence["all_public_graph_requirements_certified"] = certificate[
        "all_public_graph_requirements_certified"
    ]
    evidence["scientific_transfer_certification"] = "not_performed"
    return evidence


def write_report(root: Path, summary: dict) -> None:
    """Write every planned row; never pick a winner from incomplete denominators."""
    rows = summary["rows"]
    value = {
        "plan_sha256": summary["plan_sha256"],
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "rows": rows,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_parameter_tying": False,
        "scientific_correctness": "not_inferred_from_reuse_or_graph_predicates",
    }
    public._write(root / "shared_process_summary.json", value)
    lines = [
        "# Shared-process guidance pilot",
        "",
        (
            "Same seeds, data, prompt variants, construction capacities, "
            "revision schema and frozen fitter."
        ),
        (
            "Guidance on/off differs only in scientific scaffolding at "
            "construction and revision."
        ),
        (
            "Round 0 is construction; round 1 is one whole-model revision "
            "(or recorded fallback refit)."
        ),
        (
            "Named reuse is syntax evidence, not a conservation or "
            "scientific correctness score."
        ),
        (
            "Missing models are unavailable, never zero-complexity "
            "successes. No test data."
        ),
        "",
        (
            "| Task | Round | Status | Proposal | Train NMSE | Validation "
            "NMSE | Parameters | States | Algebraics | "
            "Shared governing laws | Tokens cumulative |"
        ),
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        counts = (r.get("shared_process_evidence") or {}).get("counts", {})
        fields = [
            r["task_id"],
            r["round"],
            r["status"],
            r["proposal_status"],
            r["retained_train_nmse"],
            r["retained_validation_nmse"],
            counts.get("parameters"),
            counts.get("dynamic_states"),
            counts.get("algebraic_processes"),
            counts.get("named_processes_shared_between_governing_definitions"),
            r["cumulative_tokens"],
        ]
        lines.append(
            "| " + " | ".join("—" if x is None else str(x) for x in fields) + " |"
        )
    (root / "SHARED_PROCESS_SUMMARY.md").write_text("\n".join(lines) + "\n")
