"""Compare exact and material-gap pre-topology screens on archived public train data."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import preproposal_cases as cases
from autoformalism.search.preproposal_screen import assess_options, screen_target
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    _value,
)
from autoformalism.staged_topology import content_hash

ALIEN = (
    "phase_c_alien_device_unknown_device_mechanism_canonical_functional_easy_reset_v1"
)
DATASETS = {
    "phase_c_cstr_controlled_reactor_mechanism_canonical_named_easy_reset_v1": (
        4,
        "T",
        "K",
        "5bce0d273c5bc5647ed3a196a60ec0b4226b1093b8853b1a5b5ad58bee3ed390",
    ),
    ALIEN: (
        5,
        "v01",
        "device output units",
        "0347e3c8d23d068c5cb060609bfee76d37073fc1f84a871be479390287074978",
    ),
}


def all_time_conflicts(packet, target, roots):
    """Strongest equal-root witness over every time on a common exact linear grid."""
    target_bounds = packet.contract.absolute_error_bounds
    if any(bound != 0 for bound in target_bounds.values()):
        raise ValueError("The exploratory envelope requires exact recorded channels")
    histories = [v[8:] for v in roots if v.startswith("history:")]
    initials = [v[8:] for v in roots if v.startswith("initial:")]
    if any(
        packet.contract.interpolation[n] != "linear_complete_knots" for n in histories
    ):
        raise ValueError("The exploratory envelope requires complete linear knots")
    n_conflicts, largest, witness = 0, 0.0, None
    for left, right in combinations(packet.trajectories, 2):
        if left.time != right.time:
            raise ValueError("The exploratory envelope requires a common grid")
        if any(
            _value(left, n, 0, packet.context.time_symbol)
            != _value(right, n, 0, packet.context.time_symbol)
            for n in initials
        ):
            continue
        pair_conflicts = False
        for i, t in enumerate(left.time):
            # A knot difference in a linear spline starts on its preceding
            # interval. The differential target is fixed to its reading at t0.
            if any(
                _value(left, n, i, packet.context.time_symbol)
                != _value(right, n, i, packet.context.time_symbol)
                for n in histories
            ):
                break
            a, b = left.targets[target][i], right.targets[target][i]
            gap = abs(a - b)
            pair_conflicts |= gap != 0.0
            if gap > largest:
                largest, witness = (
                    gap,
                    {
                        "trajectories": [left.trajectory_id, right.trajectory_id],
                        "time": t,
                        "values": [a, b],
                    },
                )
        n_conflicts += pair_conflicts
    return {
        "roots": sorted(roots),
        "conflicting_pairs": n_conflicts,
        "max_equal_prefix_gap": largest,
        "pair_absolute_error_floor": largest / 2,
        "witness": witness,
    }


def audit_inventory(case, dataset, units, margins):
    """Retain declared assumptions, selection yields and recorded absolute scales."""
    packet = TrainingCompatibilityEvidence.model_validate(case["evidence"])
    target = case["target"]
    sensitivities, scenarios = [], {}
    for gap in margins:
        result = screen_target(
            packet,
            target,
            histories=tuple(case["histories"]),
            initials=tuple(case["initials"]),
            required_histories=tuple(case["required_histories"]),
            minimum_target_gap=gap,
        )
        sensitivities.append(
            {
                "minimum_target_gap": gap,
                "pair_count": result.pair_count,
                "witnesses": len(result.witnesses),
                "baseline": assess_options(result, ()).model_dump(),
                "subset_counts": result.subset_counts,
                "subsets_examined": result.subsets_examined,
                "minimal_options": [c.options for c in result.candidates],
            }
        )
        if gap == 0:
            mandatory = set(result.required_options)
            all_roots = mandatory | set(result.optional_options)
            scenarios["mandatory_only"] = mandatory
            scenarios["whole_catalogue"] = all_roots
            if len(result.candidates) == 1 and len(result.candidates[0].options) == 1:
                only = result.candidates[0].options[0]
                scenarios[f"all_except_{only}"] = all_roots - {only}
    return {
        "dataset": dataset,
        "target": target,
        "units": units,
        "inventory": case["inventory"],
        "inventory_sha256": content_hash(case["inventory"]),
        "source_events": [],
        "evidence_sha256": packet.sha256,
        "n_train": len(packet.trajectories),
        "contract_provenance": packet.contract.provenance,
        "sensitivities": sensitivities,
        "all_time_conflicts": {
            k: all_time_conflicts(packet, target, roots)
            for k, roots in scenarios.items()
        },
    }


def run(root: Path) -> dict:
    """Authenticate two saved stage-exit series and the seed-102 projected inventory."""
    plan = cases._pinned(root / cases.STAGED_DIR / "plan.json", cases.STAGED_PLAN)
    rows = {}
    for dataset, (cell, target, units, expected) in DATASETS.items():
        brief = plan["cells"][dataset]["brief"]
        evidence = json.loads(
            (root / cases.EVIDENCE_DIR / f"{dataset}.evidence.json").read_text()
        )
        if evidence["sha256"] != expected:
            raise ValueError(f"Unexpected evidence for {dataset}")
        histories = [
            v["name"]
            for v in brief["public_variables"]
            if v["data_role"] in {"external_input", "auxiliary"}
        ]
        initials = [
            n
            for n, r in evidence["contract"]["initial_rules"].items()
            if r["mode"] == "observed_initial"
        ]
        for seed in range(5):
            path = (
                cases.STAGED_DIR
                / "results"
                / f"cell{cell:02d}_seed{seed}_full_minimal_ownership_feedback"
                / "construction/events/000.json"
            )
            event = sealed_read(root / path)
            case = cases.from_variable_exit(
                case_id=f"cell{cell}_seed{seed}",
                target=target,
                event=event,
                public_brief=brief,
                evidence=evidence,
                histories=histories,
                initials=initials,
            )
            key = dataset, content_hash(case["inventory"])
            if key not in rows:
                rows[key] = audit_inventory(case, dataset, units, (0.0, 1e-8))
            rows[key]["source_events"].append(
                {
                    "seed": seed,
                    "path": str(path),
                    "sha256": event["artifact_sha256"],
                    "stage": "variable_exit",
                }
            )
    t2 = cases._fresh(root)
    t2row = audit_inventory(t2, cases.T2, "pmol/L", (0.0, 1e-8, 1.0))
    t2row["source_events"] = [dict(t2["provenance"])]
    return {
        "protocol": "preproposal-transfer-audit-1",
        "public_plan_sha256": cases.STAGED_PLAN,
        "margin_interpretation": (
            "Two target observation intervals must be separated by more than the "
            "selected margin in target units. The margins are illustrative. "
            "Conditioning histories and measured target initials retain their "
            "exact recorded values. Simulator error is unknown."
        ),
        "rows": [*rows.values(), t2row],
        "optimizer_calls": 0,
        "llm_calls": 0,
        "test_data_opened": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.project_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, report)


if __name__ == "__main__":
    main()
