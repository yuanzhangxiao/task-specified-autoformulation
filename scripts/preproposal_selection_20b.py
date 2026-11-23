"""Run bounded 20B scientific selection and topology realization on public screens."""

from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.staged_topology import atomic_json, strict_provider_schema
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research import preproposal_selection as selection
from autoformalism.staged_topology import content_hash
from scripts import t2_breadth as pilot
from scripts.construction_functions import file_hash
from scripts.topology_repair_20b import SEEDS, SETTINGS

PROTOCOL = pilot.PREPROPOSAL_20B
ENTRYPOINTS = (
    *pilot.ENTRYPOINTS,
    "scripts/topology_repair_20b.py",
    "scripts/preproposal_selection_20b.py",
    "scripts/hpc/start_topology_repair_20b_aces.sh",
    "scripts/hpc/start_preproposal_selection_aces.sh",
)


def input_payload(source: Path) -> dict:
    """Accept the two verified local reports or their single packaged payload."""
    from autoformalism.research import preproposal_cases as cases

    payload = json.loads(source.read_text())
    if "frozen_cases" not in payload:
        payload = {
            "selection": payload,
            "frozen_cases": cases.load_frozen_cases(
                source.with_name("frozen-cases.json")
            ),
        }
    intake = selection.SelectionIntake.model_validate(payload["selection"])
    frozen = payload["frozen_cases"]
    if {c.case_id for c in intake.cases} != set(frozen) or {
        key: content_hash(v) for key, v in frozen.items()
    } != cases.PUBLIC_DIGESTS:
        raise ValueError("screened selection cases and full training cases differ")
    for case in intake.cases:
        derived = selection.SelectionCase.model_validate(
            cases.selection_case(frozen[case.case_id])
        )
        if case != derived:
            raise ValueError(f"frozen public screen differs for {case.case_id}")
    return {"selection": intake.model_dump(mode="json"), "frozen_cases": frozen}


def _plan(payload: dict) -> dict:
    from autoformalism.research import preproposal_topology as topology

    intake = selection.SelectionIntake.model_validate(payload["selection"])
    active = [c for c in intake.cases if selection.needs_selection(c)]
    matched = "source_checklists" in payload
    branches = sum(min(selection.MAX_CHOICES, len(c.candidates)) for c in active)
    if matched:
        branches = sum(
            min(2 * selection.matched_count(c), len(c.candidates)) for c in active
        ) + len(active)
    entrypoints = ENTRYPOINTS
    if matched:
        entrypoints += ("scripts/hpc/start_preproposal_matched_aces.sh",)
    plan = {
        "protocol": pilot.PREPROPOSAL_MATCHED_20B if matched else PROTOCOL,
        "intake_sha256": content_hash(payload),
        "maximum_generation_requests": len(SEEDS) * (len(active) + branches),
        "maximum_selection_requests": len(SEEDS) * len(active),
        "maximum_topology_requests": len(SEEDS) * branches,
        "settings": SETTINGS.model_dump(mode="json"),
        "seeds": list(SEEDS),
        "sampling_design": "common_provider_seed_by_base_seed_stage_and_rank",
        "cases": [c.case_id for c in intake.cases],
        "request_sha256": {
            c.case_id: content_hash(
                [
                    selection.messages(
                        c,
                        source_checklist=payload.get("source_checklists", {}).get(
                            c.case_id
                        ),
                    ),
                    selection.selection_schema(c, matched=matched).model_json_schema(),
                    topology.topology_schema(
                        c,
                        shared_hidden_initials=matched
                        and payload["frozen_cases"][c.case_id].get(
                            "shared_hidden_initials", False
                        ),
                    ).model_json_schema(),
                ]
            )
            for c in active
        },
        "runtime_source": runtime_source_hash(),
        "entrypoints": {p: file_hash(pilot.REPO / p) for p in entrypoints},
        "test_data_opened": False,
        "optimizer_calls": 0,
        "automatic_followup": False,
    }
    if matched:
        plan.update(
            sampling_design="one_candidate_topology_per_case_seed_and_source_set",
            choices_per_arm={c.case_id: selection.matched_count(c) for c in active},
            uniform_selections={
                c.case_id: {str(s): selection.uniform_ids(c, s) for s in SEEDS}
                for c in active
            },
            maximum_graph_first_requests=len(SEEDS) * len(active),
            maximum_selected_topology_requests=len(SEEDS) * (branches - len(active)),
        )
    return plan


def prepare(root: Path, source: Path, *, matched: bool = False) -> dict:
    """Rebuild public menus from frozen train-only evidence before submission."""
    payload = input_payload(source)
    if matched:
        from autoformalism.research.preproposal_cases import source_checklist

        payload["source_checklists"] = {
            key: source_checklist(frozen)
            for key, frozen in payload["frozen_cases"].items()
        }
    plan = _plan(payload)
    with _lock(root / "prepare-lock"):
        sealed_write(root / "intake.json", payload)
        return sealed_write(root / "plan.json", plan)


def load(root: Path) -> tuple[dict, dict]:
    payload = sealed_read(root / "intake.json")
    payload.pop("artifact_sha256")
    plan = sealed_read(root / "plan.json")
    if plan["artifact_sha256"] != content_hash(_plan(payload)):
        raise ValueError("selection plan, intake, runtime or messages changed")
    return plan, payload


def _read_stage(root: Path, plan: dict, slot: str) -> dict | None:
    path = root / "results" / f"{slot}.json"
    if not path.exists():
        return None
    saved = sealed_read(path)
    directory = root / "steps" / slot
    reply = sealed_read(directory / "reply.json")
    if (
        saved["slot"] != slot
        or saved["plan_sha256"] != plan["artifact_sha256"]
        or saved["reply_sha256"] != reply["artifact_sha256"]
    ):
        raise ValueError(f"checkpoint mismatch at {slot}")
    if "request_hash" in reply and (
        file_hash(directory / "calls" / (reply["request_hash"] + ".json"))
        != reply["physical_receipt_sha256"]
    ):
        raise ValueError(f"changed provider receipt at {slot}")
    return saved["assessment"]


def _generate(
    root,
    plan,
    slot,
    sampling_step,
    seed,
    messages,
    schema,
    assessor,
    transport,
    token_transport,
    endpoint,
):
    saved = _read_stage(root, plan, slot)
    if saved is not None:
        return saved
    reply = pilot._ask(
        root,
        plan,
        slot,
        seed,
        *messages,
        schema,
        endpoint,
        transport,
        token_transport,
        {"physical_requests": 0, "slots": {}},
        plan["maximum_generation_requests"],
        sampling_step=sampling_step,
    )
    assessment = {"status": reply["status"]}
    if reply["status"] == "visible":
        try:
            value = schema.model_validate(reply["value"]).model_dump(
                mode="json", by_alias=True
            )
        except ValidationError as exc:
            assessment = {"status": "invalid_reply", "reason": str(exc)}
        else:
            assessment = assessor(value)
    sealed_write(
        root / "results" / f"{slot}.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "slot": slot,
            "reply_sha256": reply["artifact_sha256"],
            "assessment": assessment,
        },
    )
    return assessment


def _prefix(case, seed):
    return f"{case.case_id}--seed-{seed}"


def _candidate_step(candidate_id: str) -> str:
    return "source-" + content_hash(candidate_id)


def _matched_draw(case, frozen, checklist, plan, seed, prefix, ask):
    """Generate a blind graph per unique choice, including failed selections."""
    from autoformalism.research import preproposal_graph_repair as repair
    from autoformalism.research import preproposal_topology as topology

    graph_schema = topology.topology_schema(
        case, shared_hidden_initials=frozen.get("shared_hidden_initials", False)
    )
    ask(
        prefix + "--graph-first",
        "graph-first",
        repair.graph_first_messages(case, frozen),
        graph_schema,
        lambda value: repair.assess_and_repair(case, value, frozen),
    )
    chosen = ask(
        prefix + "--select",
        "select",
        selection.messages(case, source_checklist=checklist),
        selection.selection_schema(case, matched=True),
        lambda v: selection.assess(case, selection.SelectionReply.model_validate(v)),
    )
    selected_ids = {c["candidate_id"] for c in chosen.get("selected", [])}
    selected_ids.update(plan["uniform_selections"][case.case_id][str(seed)])
    for option in case.candidates:
        if option.candidate_id not in selected_ids:
            continue
        choice = selection.SelectionChoice(
            **option.model_dump(), rationale="Selected source set."
        )
        step = _candidate_step(option.candidate_id)
        ask(
            prefix + "--" + step,
            step,
            topology.topology_messages(
                case, choice, frozen, source_checklist=checklist
            ),
            graph_schema,
            lambda v, choice=choice: topology.assess_topology(case, choice, v, frozen),
        )


def run(root: Path, endpoint: str, *, transport=None, token_transport=None) -> dict:
    """Commit ranked choices, then realize each one in a separate cached call."""
    from autoformalism.research import preproposal_topology as topology

    with _lock(root / "worker-lock"):
        plan, payload = load(root)
        intake = selection.SelectionIntake.model_validate(payload["selection"])
        for case in intake.cases:
            if not selection.needs_selection(case):
                continue
            for seed in plan["seeds"]:
                prefix = _prefix(case, seed)
                if "source_checklists" in payload:

                    def ask(slot, step, messages, schema, assessor, seed=seed):
                        return _generate(
                            root,
                            plan,
                            slot,
                            step,
                            seed,
                            messages,
                            schema,
                            assessor,
                            transport,
                            token_transport,
                            endpoint,
                        )

                    _matched_draw(
                        case,
                        payload["frozen_cases"][case.case_id],
                        payload["source_checklists"][case.case_id],
                        plan,
                        seed,
                        prefix,
                        ask,
                    )
                    continue
                chosen = _generate(
                    root,
                    plan,
                    prefix + "--select",
                    "select",
                    seed,
                    selection.messages(case),
                    selection.selection_schema(case),
                    lambda v, case=case: selection.assess(
                        case, selection.SelectionReply.model_validate(v)
                    ),
                    transport,
                    token_transport,
                    endpoint,
                )
                for rank, item in enumerate(chosen.get("selected", []), 1):
                    choice = selection.SelectionChoice.model_validate(item)
                    frozen = payload["frozen_cases"][case.case_id]
                    _generate(
                        root,
                        plan,
                        f"{prefix}--topology-{rank}",
                        f"topology-{rank}",
                        seed,
                        topology.topology_messages(case, choice, frozen),
                        topology.topology_schema(case),
                        lambda v, case=case, choice=choice: topology.assess_topology(
                            case, choice, v, payload["frozen_cases"][case.case_id]
                        ),
                        transport,
                        token_transport,
                        endpoint,
                    )
        return status(root)


def _matched_status(root, plan, case, group):
    """Compare paired seed outcomes and track checker/selection failures separately."""
    portfolios = {"scientific": [], "uniform": []}
    paired = Counter()
    finished = 0
    for seed in plan["seeds"]:
        prefix = _prefix(case, seed)
        first = _read_stage(root, plan, prefix + "--graph-first") or {
            "status": "pending"
        }
        picked = _read_stage(root, plan, prefix + "--select") or {"status": "pending"}
        ids = {
            "scientific": [item["candidate_id"] for item in picked.get("selected", [])],
            "uniform": plan["uniform_selections"][case.case_id][str(seed)],
        }
        draw = {"seed": seed, "graph_first": first, "scientific_selection": picked}
        passed = {}
        complete = first["status"] != "pending" and picked["status"] != "pending"
        for arm, choices in ids.items():
            branches = []
            for candidate_id in choices:
                slot = prefix + "--" + _candidate_step(candidate_id)
                graph = _read_stage(root, plan, slot) or {"status": "pending"}
                branches.append(
                    {"candidate_id": candidate_id, "slot": slot, "topology": graph}
                )
                complete &= graph["status"] != "pending"
            portfolios[arm].append(branches)
            draw[arm] = branches
            passed[arm] = any(
                b["topology"]["status"] == "topology_screen_passed" for b in branches
            )
        if complete:
            finished += 1
            if picked["status"] != "selected":
                paired["scientific_selection_failed"] += 1
            elif passed["scientific"] and passed["uniform"]:
                paired["both_passed"] += 1
            elif passed["scientific"]:
                paired["scientific_only"] += 1
            elif passed["uniform"]:
                paired["uniform_only"] += 1
            else:
                paired["neither_passed"] += 1
        group["draws"].append(draw)
    arm_summary = {}
    for arm, draws in portfolios.items():
        branches = [b for items in draws for b in items]
        passing = [
            b for b in branches if b["topology"]["status"] == "topology_screen_passed"
        ]
        arm_summary[arm] = {
            "selection_frequencies": dict(Counter(b["candidate_id"] for b in branches)),
            "topology_status_counts": dict(
                Counter(b["topology"]["status"] for b in branches)
            ),
            "draws_with_passing_topology": sum(
                any(b["topology"]["status"] == "topology_screen_passed" for b in items)
                for items in draws
            ),
            "distinct_passing_topologies": len(
                {b["topology"]["structure_sha256"] for b in passing}
            ),
        }
    group.update(
        seed_count=len(plan["seeds"]),
        selections_per_arm=selection.matched_count(case),
        scientific_selection_status_counts=dict(
            Counter(d["scientific_selection"]["status"] for d in group["draws"])
        ),
        paired_draws=dict(paired),
        selectors=arm_summary,
        graph_first_initial_status_counts=dict(
            Counter(d["graph_first"]["status"] for d in group["draws"])
        ),
        graph_first_repair_status_counts=dict(
            Counter(
                d["graph_first"].get("repair_status", "unassessed")
                for d in group["draws"]
            )
        ),
    )
    return finished


def status(root: Path) -> dict:
    """Keep model selection yield, topology admission and provider errors separate."""
    plan, payload = load(root)
    intake = selection.SelectionIntake.model_validate(payload["selection"])
    groups = []
    finished = 0
    expected = 0
    for case in intake.cases:
        active = selection.needs_selection(case)
        group = {
            "case_id": case.case_id,
            "target": case.target,
            "candidate_count": len(case.candidates),
            "evidence_sha256": case.evidence_sha256,
            "screen_status": "selection_required"
            if active
            else selection.skipped_outcome(case)["status"],
            "draws": [],
        }
        if "source_checklists" in payload:
            if active:
                expected += len(plan["seeds"])
                finished += _matched_status(root, plan, case, group)
            groups.append(group)
            continue
        choices = Counter()
        ranked = Counter()
        first_choices = Counter()
        draws_with_passing_topology = 0
        topologies = Counter()
        passing_structures = set()
        distinct = set()
        for seed in plan["seeds"] if active else ():
            expected += 1
            prefix = _prefix(case, seed)
            picked = _read_stage(root, plan, prefix + "--select")
            if picked is None:
                choices["pending"] += 1
                continue
            choice_status = picked["status"]
            choices[choice_status] += 1
            branches = []
            for rank, child in enumerate(picked.get("selected", []), 1):
                outcome = _read_stage(root, plan, f"{prefix}--topology-{rank}") or {
                    "status": "pending"
                }
                ranked[child["candidate_id"]] += 1
                topologies[outcome["status"]] += 1
                if outcome["status"] == "topology_screen_passed":
                    passing_structures.add(outcome["structure_sha256"])
                branches.append({"rank": rank, **child, "topology": outcome})
            if branches:
                first_choices[branches[0]["candidate_id"]] += 1
                draws_with_passing_topology += int(
                    any(
                        b["topology"]["status"] == "topology_screen_passed"
                        for b in branches
                    )
                )
                distinct.add(tuple(b["candidate_id"] for b in branches))
            finished += int(all(b["topology"]["status"] != "pending" for b in branches))
            group["draws"].append(
                {"seed": seed, "selection": picked, "branches": branches}
            )
        group.update(
            selection_status_counts=dict(choices),
            selection_frequencies=dict(ranked),
            first_choice_frequencies=dict(first_choices),
            draws_with_passing_topology=draws_with_passing_topology,
            distinct_ranked_portfolios=len(distinct),
            distinct_passing_topologies=len(passing_structures),
            topology_status_counts=dict(topologies),
        )
        groups.append(group)
    records = [
        json.loads(p.read_text()) for p in sorted(root.glob("steps/*/calls/*.json"))
    ]
    return {
        "status": "complete" if expected == finished else "incomplete",
        "expected_active_draws": expected,
        "completed_active_draws": finished,
        "plan_sha256": plan["artifact_sha256"],
        "maximum_generation_requests": plan["maximum_generation_requests"],
        "physical_generation_requests": len(records),
        "provider_status_counts": dict(Counter(r["status"] for r in records)),
        "observed_tokens": sum(r.get("observed_total_tokens") or 0 for r in records),
        "unknown_usage": sum(r.get("observed_total_tokens") is None for r in records),
        "optimizer_calls": 0,
        "test_data_opened": False,
        "cases": groups,
    }


def inspect(root: Path) -> dict:
    with _lock(root / "worker-lock"), _lock(root / "inspection-lock"):
        summary = status(root)
        atomic_json(root / "summary.json", summary)
        archive = root / "inspection.tar.gz"
        temporary = archive.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz") as tar:
            for name in (
                "plan.json",
                "intake.json",
                "summary.json",
                "steps",
                "results",
                "execution.json",
                "submissions",
                "runtime",
                "logs",
            ):
                source = root / name
                if source.exists() and not source.is_symlink():
                    tar.add(source, arcname=name)
        temporary.replace(archive)
    return {
        "archive": str(archive),
        **{k: v for k, v in summary.items() if k != "cases"},
    }


def preview(root: Path) -> list[dict]:
    _, payload = load(root)
    values = []
    for case in selection.SelectionIntake.model_validate(payload["selection"]).cases:
        if not selection.needs_selection(case):
            values.append({"case_id": case.case_id, **selection.skipped_outcome(case)})
            continue
        checklist = payload.get("source_checklists", {}).get(case.case_id)
        system, user = selection.messages(case, source_checklist=checklist)
        entry = {
            "case_id": case.case_id,
            "system": system,
            "user": json.loads(user),
            "schema": selection.selection_schema(
                case, matched=checklist is not None
            ).model_json_schema(),
        }
        if checklist is not None:
            from autoformalism.research import preproposal_graph_repair as repair
            from autoformalism.research import preproposal_topology as topology

            frozen = payload["frozen_cases"][case.case_id]
            choice = selection.SelectionChoice(
                **case.candidates[0].model_dump(), rationale="Selected source set."
            )
            for key, (system, user) in {
                "graph_example": topology.topology_messages(
                    case, choice, frozen, source_checklist=checklist
                ),
                "graph_first": repair.graph_first_messages(case, frozen),
            }.items():
                entry[key] = {
                    "system": system,
                    "user": json.loads(user),
                    "schema": topology.topology_schema(
                        case,
                        shared_hidden_initials=frozen.get(
                            "shared_hidden_initials", False
                        ),
                    ).model_json_schema(),
                }
        values.append(entry)
    return values


def preflight(root: Path) -> dict:
    """Exercise every selectable prompt against the shipped CPU-only inputs."""
    from autoformalism.research import preproposal_topology as topology

    plan, payload = load(root)
    arms = []
    for case in selection.SelectionIntake.model_validate(payload["selection"]).cases:
        active = selection.needs_selection(case)
        arms.append(
            {
                "case_id": case.case_id,
                "target": case.target,
                "candidate_count": len(case.candidates),
                "active": active,
            }
        )
        if not active:
            continue
        checklist = payload.get("source_checklists", {}).get(case.case_id)
        if checklist is not None:
            from autoformalism.research import preproposal_graph_repair as repair

            repair.graph_first_messages(case, payload["frozen_cases"][case.case_id])
        selection.messages(case, source_checklist=checklist)
        strict_provider_schema(
            selection.selection_schema(
                case, matched=checklist is not None
            ).model_json_schema()
        )
        strict_provider_schema(
            topology.topology_schema(
                case,
                shared_hidden_initials=checklist is not None
                and payload["frozen_cases"][case.case_id].get(
                    "shared_hidden_initials", False
                ),
            ).model_json_schema()
        )
        for option in case.candidates:
            choice = selection.SelectionChoice(
                **option.model_dump(), rationale="Check assembly of this request."
            )
            topology.topology_messages(
                case,
                choice,
                payload["frozen_cases"][case.case_id],
                **({"source_checklist": checklist} if checklist is not None else {}),
            )
    return {
        "status": "preflight_passed",
        "plan_sha256": plan["artifact_sha256"],
        "arms": arms,
        **{
            k: plan[k]
            for k in (
                "maximum_selection_requests",
                "maximum_topology_requests",
                "maximum_generation_requests",
                "seeds",
                "sampling_design",
            )
        },
        "generation_requests_started": 0,
        "optimizer_calls": 0,
        **{
            k: plan[k]
            for k in (
                "choices_per_arm",
                "maximum_graph_first_requests",
                "maximum_selected_topology_requests",
            )
            if k in plan
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=("prepare", "preflight", "preview", "run", "status", "inspect"),
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument(
        "--matched",
        action="store_true",
        help="Prepare the paired selector and graph-first experiment.",
    )
    p.add_argument("--endpoint")
    p.add_argument("--runtime-record", type=Path)
    args = p.parse_args()
    if args.action == "prepare":
        if not args.source:
            p.error("prepare needs --source")
        result = prepare(args.root, args.source, matched=args.matched)
    elif args.action == "run":
        if not args.endpoint or not args.runtime_record:
            p.error("run needs a bound ACES runtime and endpoint")
        from scripts.submit_t2_breadth import bind

        bind(args.root, args.runtime_record, args.endpoint)
        result = run(args.root, args.endpoint)
    else:
        result = {
            "preflight": preflight,
            "preview": preview,
            "inspect": inspect,
            "status": status,
        }[args.action](args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
