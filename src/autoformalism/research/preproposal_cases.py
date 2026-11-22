"""Freeze train-only inventories and permitted evidence before building new topology."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search.preproposal_screen import (
    ScreenResult,
    assess_options,
    screen_target,
)
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    validate_evidence,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "preproposal-public-cases-v2"
T2_CASE = "t2_seed102_insulin"
T2_GAP_CASE = "t2_seed102_insulin_gap1"
BASIN_CASE = "basin_seed0_downstream"
STRICT_T2_CASE = "t2_seed0_insulin_variable_exit"
FRESH_DIR = Path("output/t2-fresh-20b-results-20261119/returned")
STAGED_DIR = Path("output/jetstream-ownership-inspection-20261013")
EVIDENCE_DIR = Path("output/topology-diagnostics-review-20261014")
T2 = "phase_c_dalla_man_t2_canonical_named_easy_rates_v1"
BASIN = "phase_c_detention_coupled_noise0_v1"
FRESH_RECEIPT = FRESH_DIR / "steps/fresh-102-topology_correction-2/reply.json"
INPUTS_DIGEST = "db0b5e78406147a8482c1b05f9d2463dacddbb2937f4f0f542ba9d74700a064c"
RECEIPT_DIGEST = "68a8af3a65e892549fb55c31f95807c370dc0572fcddcc39f83b2f47ae59e125"
STAGED_PLAN = "dd2ed5be34a26b5988c77e51c1ff1bd63646643bf97fed5d1e064de7a82c0c1a"
STAGED = {
    BASIN_CASE: (
        BASIN,
        6,
        "h_down",
        "35f40542d2c2bf6c832553346f095019b353e453066e301ac902c2e96e517dbe",
        "83c30ff97ac5add9c007f646b3e72c191a43689332f4488ed822da87997c571d",
    ),
    STRICT_T2_CASE: (
        T2,
        2,
        "I",
        "13f812e82bc1c8c75394904df4b7b5d64bcf6fa33e5cfa2690732fafdca5e502",
        "27511894c26dea71c1995de15f4b2efb38a7501c5f5133ef4ea8c0f103625864",
    ),
}


def _pinned(path: Path, digest: str) -> dict:
    value = sealed_read(path)
    if value["artifact_sha256"] != digest:
        raise ValueError(f"Unexpected source artifact at {path}")
    return value


def _inventory(pairs) -> list[dict]:
    variables = [{"name": n, "kind": k} for n, k in pairs]
    if len({v["name"] for v in variables}) != len(variables):
        raise ValueError("Duplicate inventory declaration")
    if any(v["kind"] not in {"algebraic", "differential"} for v in variables):
        raise ValueError("Unrecognized inventory kind")
    return sorted(variables, key=lambda v: v["name"])


def _variable_exit(event: dict) -> list[dict]:
    if (event["stage"], event["accepted"], event["stage_complete"]) != (
        "variables",
        True,
        True,
    ):
        raise ValueError("An accepted variable-stage exit is required")
    if any(event["after"].get(k) for k in ("equations", "processes")):
        raise ValueError("Expected a variable-only draft before constructing topology")
    return _inventory((v["name"], v["definition"]) for v in event["after"]["variables"])


def _joint_inventory(value: dict) -> list[dict]:
    declarations = [(name, item["kind"]) for name, item in value["targets"].items()]
    declarations.extend((v["name"], v["kind"]) for v in value["additional_components"])
    return _inventory(
        (n, "differential" if k == "dynamic" else k) for n, k in declarations
    )


def _case(
    case_id: str,
    target: str,
    inventory: list[dict],
    brief: dict,
    evidence: dict,
    future: set[str],
    baseline_readings: set[str],
    provenance: dict,
    required_histories: Iterable[str] | None = None,
    minimum_target_gap: float = 0.0,
) -> dict:
    typed = TrainingCompatibilityEvidence.model_validate(evidence)
    validate_evidence(typed)
    public = PublicScientificBrief.model_validate(brief)
    generated = {v["name"] for v in inventory}
    if target not in {v["name"] for v in inventory if v["kind"] == "differential"}:
        raise ValueError("Study target requires a declared differential state")
    if not set(typed.context.targets) <= generated:
        raise ValueError("Absent public target in the frozen inventory")
    permitted = set(typed.context.external_inputs + typed.context.auxiliaries)
    available_initials = permitted | set(
        typed.context.targets + typed.context.fixed_covariates
    )
    if not future <= permitted or not baseline_readings <= available_initials:
        raise ValueError("Forbidden source in forecast permission contract")
    if target not in baseline_readings:
        raise ValueError(
            "This screen requires permission to use the target at time zero"
        )
    if target not in typed.context.targets:
        raise ValueError("The selected variable must be a public measured target")
    declared_public = {v.name: v.data_role for v in public.public_variables}
    for names, role in (
        (typed.context.targets, "target"),
        (typed.context.auxiliaries, "auxiliary"),
        (typed.context.external_inputs, "external_input"),
        (typed.context.fixed_covariates, "covariate"),
    ):
        if any(declared_public.get(name) != role for name in names):
            raise ValueError("Training evidence roles disagree with the public brief")
    histories = future - generated
    initials = baseline_readings - {target}
    # A mandatory upstream input may reach the target through latent processes.
    # Drivers listed together in a requirement are not all forced individually.
    required = {
        r.drivers[0]
        for r in public.requirements
        if target in r.targets and len(r.drivers) == 1 and r.drivers[0] in histories
    }
    if required_histories is not None:
        required.update(required_histories)
        if not required <= histories:
            raise ValueError("Required histories must be permitted upstream histories")
    return {
        "case_id": case_id,
        "target": target,
        "minimum_target_gap": minimum_target_gap,
        "inventory": inventory,
        "public_brief": brief,
        "evidence": evidence,
        "histories": sorted(histories),
        "initials": sorted(initials),
        "required_histories": sorted(required),
        "provenance": provenance,
    }


def from_variable_exit(
    *,
    case_id: str,
    target: str,
    event: dict,
    public_brief: PublicScientificBrief | dict,
    evidence: TrainingCompatibilityEvidence | dict,
    histories: Iterable[str],
    initials: Iterable[str],
    required_histories: Iterable[str] | None = None,
    minimum_target_gap: float = 0.0,
) -> dict:
    """Start a new case between an accepted variable stage and its first topology.

    The caller supplies positive forecast permissions. Initial readings include the
    target's own measured initial. Required histories can add public upstream paths
    to the unambiguous single-driver requirements already in the scientific brief.
    The result contains the complete training evidence for local verification.
    """
    inventory = _variable_exit(event)
    if (
        "artifact_sha256" in event
        and content_hash({k: v for k, v in event.items() if k != "artifact_sha256"})
        != event["artifact_sha256"]
    ):
        raise ValueError("Variable-stage event digest differs")
    return _case(
        case_id,
        target,
        inventory,
        public_brief.model_dump(mode="json")
        if isinstance(public_brief, PublicScientificBrief)
        else public_brief,
        evidence.model_dump(mode="json")
        if isinstance(evidence, TrainingCompatibilityEvidence)
        else evidence,
        set(histories),
        set(initials),
        {
            "inventory_origin": "accepted_variable_stage_exit",
            "receipt_sha256": event.get("artifact_sha256") or content_hash(event),
            "initialization_scope": "observed_time_zero_only",
        },
        required_histories,
        minimum_target_gap,
    )


def _observed_readings(evidence: dict) -> set[str]:
    return {
        name
        for name, rule in evidence["contract"]["initial_rules"].items()
        if rule["mode"] == "observed_initial"
    }


def _fresh(root: Path) -> dict:
    inputs = _pinned(root / FRESH_DIR / "inputs.json", INPUTS_DIGEST)
    receipt = _pinned(root / FRESH_RECEIPT, RECEIPT_DIGEST)
    forecast = inputs["fitting_contract"]["prefix_policy"]
    return _case(
        T2_CASE,
        "I",
        _joint_inventory(receipt["value"]),
        inputs["brief"],
        inputs["evidence"],
        set(forecast["future_inputs"] + forecast["future_auxiliaries"]),
        _observed_readings(inputs["evidence"]),
        {
            "inventory_origin": "projected_from_joint_declaration",
            "seed": 102,
            "case": T2,
            "receipt_sha256": RECEIPT_DIGEST,
            "public_inputs_sha256": INPUTS_DIGEST,
            "receipt_path": str(FRESH_RECEIPT),
            "initialization_scope": "observed_time_zero_only",
        },
    )


def _staged(root: Path, case_id: str) -> dict:
    dataset, cell, target, event_hash, evidence_hash = STAGED[case_id]
    plan = _pinned(root / STAGED_DIR / "plan.json", STAGED_PLAN)
    path = (
        STAGED_DIR
        / "results"
        / f"cell{cell:02d}_seed0_full_minimal_ownership_feedback"
        / "construction/events/000.json"
    )
    event = _pinned(root / path, event_hash)
    evidence = json.loads(
        (root / EVIDENCE_DIR / f"{dataset}.evidence.json").read_text()
    )
    if evidence["sha256"] != evidence_hash:
        raise ValueError("Unexpected training evidence")
    brief = plan["cells"][dataset]["brief"]
    # For these two public specifications the named external inputs and supplied
    # auxiliaries are given at forecast time. Ground-truth state histories are not.
    future = {
        v["name"]
        for v in brief["public_variables"]
        if v["data_role"] in {"external_input", "auxiliary"}
    }
    case = from_variable_exit(
        case_id=case_id,
        target=target,
        event=event,
        public_brief=brief,
        evidence=evidence,
        histories=future,
        initials=_observed_readings(evidence)
        | ({"initial_up"} if case_id == BASIN_CASE else set()),
    )
    case["provenance"].update(
        seed=0, case=dataset, receipt_path=str(path), public_plan_sha256=STAGED_PLAN
    )
    return case


def load_inputs(project_root: Path) -> dict[str, dict]:
    """Build two T2 sensitivity arms on identical data and a separate basin case."""
    exact = _fresh(project_root)
    return {
        T2_CASE: exact,
        T2_GAP_CASE: {**exact, "case_id": T2_GAP_CASE, "minimum_target_gap": 1.0},
        BASIN_CASE: _staged(project_root, BASIN_CASE),
    }


def load_strict_t2(project_root: Path) -> dict:
    """Additional offline check with an accepted variable-only inventory from T2."""
    return _staged(project_root, STRICT_T2_CASE)


def screen_case(case: dict) -> ScreenResult:
    """Apply the catalogue frozen for this inventory to its training observations."""
    return screen_target(
        TrainingCompatibilityEvidence.model_validate(case["evidence"]),
        case["target"],
        histories=tuple(case["histories"]),
        initials=tuple(case["initials"]),
        required_histories=tuple(case["required_histories"]),
        minimum_target_gap=case.get("minimum_target_gap", 0.0),
    )


def selection_case(case: dict, screen: ScreenResult | None = None) -> dict:
    """Make a scientific menu from public text and the training-only screen."""
    screen = screen if screen is not None else screen_case(case)
    baseline = assess_options(screen, ())
    margin = screen.minimum_target_gap
    cutoff = (
        f"This experimental sensitivity uses an additional target gap of {margin:g} "
        "in the target's units after accounting for measurement bounds. It is a "
        "screening setting only. "
        if margin
        else "This screen uses the recorded measurement bounds. "
    )
    summary = cutoff + (
        f"{len(case['evidence']['trajectories'])} training trajectories, "
        f"{screen.pair_count} pairs. {len(screen.witnesses)} pairs have an observed "
        f"{screen.target} gap greater than the cutoff. With the required upstream "
        f"information, {baseline.conflicting_pairs} pairs still contradict a shared "
        f"deterministic subsystem and {baseline.unresolved_pairs} are unresolved. "
        "Each candidate lists additional information that could reach the target "
        "directly or through other states. The target uses its own recorded initial "
        "value. A source tagged initial can enter only at time zero."
    )
    context = case["evidence"]["context"]
    option_names = {
        o.removeprefix("history:")
        for c in screen.candidates
        for o in c.options
        if o.startswith("history:")
    }
    for field, role in (
        ("auxiliaries", "supplied auxiliary"),
        ("external_inputs", "external input"),
    ):
        for name in sorted(option_names.intersection(context[field])):
            summary += f" {name} has the forecast role {role}."
    if case["required_histories"]:
        names = ", ".join(case["required_histories"])
        summary += f" The public task already requires upstream paths from {names}."
    fixed = set(screen.required_options)
    example = max(
        (w for w in screen.witnesses if fixed.isdisjoint((*w.different, *w.unknown))),
        key=lambda w: w.pair_error_floor,
        default=None,
    )
    if example:
        summary += (
            f" For {example.trajectories[0]} and {example.trajectories[1]}, "
            f"{screen.target}({example.time:g}) is {example.observations[0]} "
            f"versus {example.observations[1]}, with equal required "
            "information through that time."
        )
    return {
        "case_id": case["case_id"],
        "target": case["target"],
        "inventory": case["inventory"],
        "public_task": case["public_brief"]["scientific_context"],
        "evidence_sha256": screen.evidence_sha256,
        "evidence_summary": summary,
        "required_options": list(screen.required_options),
        "candidates": [
            {"candidate_id": c.id, "sources": list(c.options)}
            for c in screen.candidates
        ],
    }


def load_frozen_cases(path: Path) -> dict[str, dict]:
    """Verify the public projection and its sealed provenance after upload."""
    payload = sealed_read(path)
    cases = {c["case_id"]: c for c in payload["cases"]}
    if (
        payload["protocol"] != PROTOCOL
        or len(payload["cases"]) != len(cases)
        or set(cases) != set(PUBLIC_DIGESTS)
    ):
        raise ValueError("Unexpected frozen case set")
    for key, value in cases.items():
        if content_hash(value) != PUBLIC_DIGESTS[key]:
            raise ValueError(f"Public case projection differs for {key}")
    return cases


PUBLIC_DIGESTS = {
    T2_GAP_CASE: "df014daf3ba37a5bd28bba9bc7589453f8c2a7117e7443799180dc404a56ad65",
    T2_CASE: "6361e21223661c92fa8c7f91cce9e642caf0e86e96572aea7abe2973cfb7112d",
    BASIN_CASE: "3f71400ed6befbac43840cbef97b63efa716497c9655ce0c6d5c0435bd5b2a71",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = load_inputs(args.project_root)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "frozen-cases.json"
    sealed_write(path, {"protocol": PROTOCOL, "cases": list(cases.values())})
    load_frozen_cases(path)
    intake = []
    for case in (*cases.values(), load_strict_t2(args.project_root)):
        screen = screen_case(case)
        atomic_json(
            args.out / f"{case['case_id']}-screen.json", screen.model_dump(mode="json")
        )
        if case["case_id"] in cases:
            intake.append(selection_case(case, screen))
    atomic_json(args.out / "selection-intake.json", {"cases": intake})


if __name__ == "__main__":
    main()
