"""Frozen nine-benchmark component matrix, reused search and paired final pruning."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import fresh_shared, process_pruning
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest
from autoformalism.search import scientific_verification
from autoformalism.search.review_integrity import selection_policy

CELLS = (
    "phase_b_dalla_man_t1_canonical_named_easy",
    "phase_b_anonymous_system_t1_canonical_obfuscated_easy",
    "phase_b_dalla_man_t1_perturbed_named_easy",
    "phase_b_anonymous_system_t1_perturbed_obfuscated_easy",
    "phase_b_cstr_controlled_reactor_mechanism_canonical_named_easy",
    "phase_b_alien_device_unknown_device_mechanism_canonical_functional_easy",
    "phase_b_dalla_man_t1_canonical_named_hard",
    "phase_b_dalla_man_t2_canonical_named_easy",
    "phase_b_dalla_man_t2_canonical_named_hard",
)
SECONDARY_CELLS = (CELLS[0], *CELLS[6:])
ARMS = ((True, True), (False, True), (True, False), (False, False))


def policy() -> dict:
    """Separate scientific gates/advice from compiler contracts and final scoring."""
    return {
        "version": io.ABLATION_PROTOCOL,
        "primary_cells": list(CELLS),
        "secondary_cells": list(SECONDARY_CELLS),
        "primary_lineages": 144,
        "secondary_additional_lineages": 16,
        "rounds": 15,
        "fresh_starts": True,
        "revision_patch_count_limits": None,
        "selection": selection_policy(),
        "verifier_off": [
            "no required-driver inventory repairs",
            "no dynamic-memory path repairs",
            "no public topology/pathway gates",
            "no nonlinear-role syntax gate",
            "no target representation/composition or mechanism eligibility gate",
            "no scientific certificate feedback",
            "no public-predicate pruning gate",
        ],
        "always_on": [
            "complete generated targets",
            "permitted channels and causal initials",
            "restricted expressions",
            "algebraic cycle checks",
            "parameter contracts",
            "declared shared-law/consumer/sign/conversion consistency",
            "finite numerical evaluation",
            "immutable budgets and resume",
        ],
        "same_public_specification_in_all_primary_arms": True,
        "critic": "corrected paired GPT-OSS-120B; advisory only; no selection score",
        "critic_placement": "before fit, after pruning; advice before revision",
        "critic_transport": "jetstream-authenticated-proxy-2",
        "shared_off": "dedicated process proposal/guidance off; algebraics allowed",
        "pruning": {
            **process_pruning.POLICY,
            "fit_profile": "collocation-multi-target-v1",
        },
        "pruning_off": "same search checkpoint; unchanged-refit control also reported",
        "test_access": "explicit freeze, separate evaluation; never automatic",
    }


def validate_config(config) -> None:
    """The deadline cannot silently shrink benchmarks, seeds, variants or rounds."""
    if (
        tuple(config.public_cells) != CELLS
        or tuple(config.seeds) != (0, 1)
        or config.full_only
        or config.no_latent_cells
        or config.no_spec_cells
        or config.rounds != 15
        or config.fit_profile != "collocation-multi-target-v1"
        or config.scientific_judge != "advisory-jetstream"
    ):
        raise ValueError(
            "component campaign requires nine cells, two seeds/variants, 15 visits"
        )
    blocks = config.campaign_blocks
    if blocks is not None and (
        not blocks
        or len(set(blocks)) != len(blocks)
        or any(type(i) is not int or i not in range(36) for i in blocks)
    ):
        raise ValueError("select distinct matched blocks from 0 through 35")


def tasks(config) -> list[dict]:
    """Place every competing arm in the same cell/seed/prompt hardware block."""
    result = []
    for ci, cell in enumerate(config.public_cells):
        for seed in config.seeds:
            for pi, variant in enumerate(("full", "brief_only")):
                block = ci * 4 + seed * 2 + pi
                if (
                    config.campaign_blocks is not None
                    and block not in config.campaign_blocks
                ):
                    continue
                arms = list(ARMS)
                offset = block % len(arms)
                arms = arms[offset:] + arms[:offset]
                for critic, verifier in arms:
                    result.append(
                        {
                            "index": len(result),
                            "cell": cell,
                            "seed": seed,
                            "arm": variant,
                            "block": block,
                            "task_id": (
                                f"cell{ci:02d}_seed{seed}_{variant}_"
                                f"c{int(critic)}v{int(verifier)}s1"
                            ),
                            "critic": critic,
                            "scientific_verifier": verifier,
                            "shared_processes": True,
                            "secondary": False,
                            "shared_round_zero": None,
                        }
                    )
                if config.secondary_shared_comparison and cell in SECONDARY_CELLS:
                    result.append(
                        {
                            "index": len(result),
                            "cell": cell,
                            "seed": seed,
                            "arm": variant,
                            "block": block,
                            "task_id": f"cell{ci:02d}_seed{seed}_{variant}_c1v1s0",
                            "critic": True,
                            "scientific_verifier": True,
                            "shared_processes": False,
                            "secondary": True,
                            "shared_round_zero": None,
                        }
                    )
    return result


def verify(root: Path, *, execution=True) -> dict:
    """Do not accept an older pilot, even if its files have familiar names."""
    plan = io.verify(root, execution=execution)
    if plan["protocol"] != io.ABLATION_PROTOCOL:
        raise ValueError("requires the nine-benchmark component campaign")
    return plan


def freeze(config_path: Path, public_root: Path, root: Path) -> dict:
    """Use the existing public-only snapshot and frozen runtime contracts."""
    config = io.DeadlineConfig.model_validate_json(config_path.read_text())
    if config.protocol != io.ABLATION_PROTOCOL:
        raise ValueError("requires component campaign config")
    # Fail before creating a partial campaign. No private/test file is inspected.
    for cell in config.public_cells:
        for name in io.FILES:
            if not (public_root / "phase_b_v1" / cell / name).is_file():
                raise ValueError(f"missing public asset: {cell}/{name}")
    return io.freeze(config_path, public_root, root)


def prune_one(root: Path, plan: dict, task: dict) -> dict:
    """Same training ranking/refits; apply the assigned verifier to pruning too."""
    io.require_open(root)
    row = fresh_shared.final_row(root, plan, task)
    if row is None:
        raise ValueError("last search visit is not complete")
    directory = root / "pruning" / task["task_id"]
    sealed_write(directory / "parent.json", fresh_shared._binding(plan, row))
    if not row["parent"] or process_pruning.score(row["parent"]["fit"]) is None:
        with public._lock(directory):
            return sealed_write(
                directory / "result.json", fresh_shared._skipped(plan, row)
            )
    with scientific_verification.scope(task["scientific_verifier"]):
        return process_pruning.execute_row(directory, plan, row, task["block"])


def endpoints(root: Path, plan: dict, task: dict) -> list[dict]:
    """Final/pruning-off endpoints are predetermined, never chosen from test scores."""
    row = fresh_shared.final_row(root, plan, task)
    parent = row["parent"] if row else None
    path = root / "pruning" / task["task_id"] / "result.json"
    pruned = None
    if row and path.exists():
        with scientific_verification.scope(task["scientific_verifier"]):
            pruned = fresh_shared._checked(path.parent, plan, row)
    selection = (pruned or {}).get("selection")
    final_request = (parent or {}).get("request")
    final_fit = (parent or {}).get("fit")
    if selection:
        final_fit = selection["fit"]
        if selection["selected"] == "pruned":
            final_request = pruned["choice"]["request"]
    critic_path = root / "pruning" / task["task_id"] / "critic.json"
    final_review_missing = bool(
        task["critic"] and parent and pruned and not critic_path.exists()
    )
    if critic_path.exists():
        review = sealed_read(critic_path)
        if (
            review["identity"] != plan["artifact_sha256"]
            or review["pruning_sha256"] != pruned["artifact_sha256"]
        ):
            raise ValueError("final review belongs to another pruning result")
    values = [
        {
            "endpoint": "final",
            "request": final_request,
            "fit": final_fit,
            "status": "missing"
            if not pruned or final_review_missing
            else pruned["status"],
            "source_sha256": (pruned or {}).get("artifact_sha256"),
            "final_review_sha256": sealed_read(critic_path)["artifact_sha256"]
            if critic_path.exists()
            else None,
            "pruning_selection": (selection or {}).get("selected"),
        }
    ]
    if (
        task["cell"] in SECONDARY_CELLS
        and task["critic"]
        and task["scientific_verifier"]
        and task["shared_processes"]
    ):
        values.append(
            {
                "endpoint": "no_pruning",
                "request": (parent or {}).get("request"),
                "fit": (parent or {}).get("fit"),
                "status": "complete"
                if parent
                else "model_unavailable"
                if row
                else "missing",
                "source_sha256": (row or {}).get("source_result_sha256"),
            }
        )
        values.append(
            {
                "endpoint": "unchanged_refit_control",
                "request": (parent or {}).get("request"),
                "fit": (pruned or {}).get("fits", {}).get("control"),
                "status": "complete"
                if (pruned or {}).get("fits", {}).get("control")
                else "model_unavailable"
                if pruned
                else "missing",
                "source_sha256": (pruned or {}).get("artifact_sha256"),
            }
        )
    return values


def report(root: Path) -> dict:
    """Keep complete denominators and separate scientific evidence from eligibility."""
    plan = verify(root, execution=False)
    search = reporting.report(root)
    rows = []
    for task in plan["tasks"]:
        history = [r for r in search["rows"] if r["task_id"] == task["task_id"]]
        last = history[-1] if history else {}
        for item in endpoints(root, plan, task):
            request, fit = item.pop("request"), item.pop("fit")
            certificate = None
            if request:
                # Offline assessment is always enabled, including verifier-off arms.
                certificate = process_pruning.certificate(
                    PublicFitRequest.model_validate(request),
                    plan["cells"][task["cell"]],
                    {**task, "scientific_verifier": True},
                )
            rows.append(
                {
                    "task": task,
                    **item,
                    "proposer_requests": last.get("cumulative_requests"),
                    "proposer_tokens": last.get("cumulative_tokens"),
                    "fit_status": (fit or {}).get("status"),
                    "model_available": bool(
                        request and process_pruning.score(fit) is not None
                    ),
                    "training": (fit or {}).get("training"),
                    "validation": (fit or {}).get("validation"),
                    "complexity": process_pruning.pruning.complexity(
                        PublicFitRequest.model_validate(request)
                    )
                    if request
                    else None,
                    "public_assessment": certificate,
                }
            )
    from autoformalism.rebuttal.component_critic import cost

    value = {
        "identity": plan["artifact_sha256"],
        "protocol": io.ABLATION_PROTOCOL,
        "source_sha256": plan["source_sha256"],
        "config": plan["config"],
        "public_assets": {
            cell: value["assets"] for cell, value in plan["cells"].items()
        },
        "frozen_for_evaluation": (root / "evaluation_freeze.json").exists(),
        "planned_lineages": len(plan["tasks"]),
        "planned_round_rows": len(search["rows"]),
        "search_status_counts": search["status_counts"],
        "endpoint_status_counts": dict(Counter(r["status"] for r in rows)),
        "judge_cost": cost(root),
        "rows": rows,
        "test_data_opened": False,
    }
    public._write(root / "component_summary.json", value)
    return value


def export(root: Path) -> dict:
    """Freeze all search and final pruning endpoints before any test access."""
    from autoformalism.rebuttal.final_evaluation import FrozenEvaluationSubject

    with io.execution_lease(root, exclusive=True), public._lock(root):
        plan = verify(root)
        receipt = root / "evaluation_freeze.json"
        if receipt.exists():
            saved = sealed_read(receipt)
            if (
                hashlib.sha256((root / "subjects.jsonl").read_bytes()).hexdigest()
                != saved["subjects_sha256"]
            ):
                raise ValueError("frozen subjects changed")
            return saved
        summary = report(root)
        if summary["search_status_counts"].get("missing") or summary[
            "endpoint_status_counts"
        ].get("missing"):
            raise ValueError("unfinished planned work; no implicit partial freeze")
        subjects, roster = [], []
        for task in plan["tasks"]:
            for item in endpoints(root, plan, task):
                sid = task["task_id"] + "_" + item["endpoint"]
                row = {
                    "subject_id": sid,
                    "task": task,
                    "endpoint": item["endpoint"],
                    "source_sha256": item["source_sha256"],
                    "model_available": False,
                }
                if item["request"] and process_pruning.score(item["fit"]) is not None:
                    model, _, _ = public._lower(
                        PublicFitRequest.model_validate(item["request"])
                    )
                    candidate = model.validated.candidate
                    subject = FrozenEvaluationSubject.model_validate(
                        {
                            "subject_id": sid,
                            "method": method_name(task, item["endpoint"]),
                            "benchmark_id": task["cell"],
                            "tier": plan["cells"][task["cell"]]["target_contract"][
                                "tier"
                            ],
                            "repetition": task["seed"],
                            "private_metrics_opened_after_freeze": False,
                            "source_provenance": {
                                "adapter": "direct_candidate",
                                "request_id": sid,
                                "source_path": str(root / "pruning" / task["task_id"]),
                                "source_sha256": item["source_sha256"],
                                "candidate_sha256": hashlib.sha256(
                                    candidate.model_dump_json().encode()
                                ).hexdigest(),
                            },
                            "candidate": candidate.model_dump(mode="json"),
                            "parameterization": {
                                "status": "available",
                                "global_parameters": item["fit"]["parameters"],
                            },
                            "validation_context": model.validated.context.model_dump(
                                mode="json"
                            ),
                            "target_prediction": {
                                "status": "missing",
                                "message": "Frozen before test access",
                            },
                        }
                    )
                    subjects.append(subject.model_dump_json())
                    row["model_available"] = True
                roster.append(row)
        payload = "".join(s + "\n" for s in subjects).encode()
        (root / "subjects.jsonl").write_bytes(payload)
        return sealed_write(
            receipt,
            {
                "plan_sha256": plan["artifact_sha256"],
                "subjects_sha256": hashlib.sha256(payload).hexdigest(),
                "subject_count": len(subjects),
                "roster": roster,
                "further_search_permitted": False,
                "test_data_opened": False,
            },
        )


def stage_public(sources: list[Path], root: Path) -> dict:
    """Copy only allowlisted development files; conflicting exports require a choice."""
    entries = []
    for cell in CELLS:
        for name in io.FILES:
            candidates = [base / "phase_b_v1" / cell / name for base in sources]
            existing = [p for p in candidates if p.is_file()]
            if not existing:
                raise ValueError(
                    f"missing public asset: {cell}/{name}; no files copied"
                )
            contents = {hashlib.sha256(p.read_bytes()).hexdigest(): p for p in existing}
            if len(contents) != 1:
                raise ValueError(
                    f"conflicting public exports: {cell}/{name}; provide one version"
                )
            digest, path = next(iter(contents.items()))
            target = root / "phase_b_v1" / cell / name
            if (
                target.exists()
                and hashlib.sha256(target.read_bytes()).hexdigest() != digest
            ):
                raise ValueError(f"destination differs: {target}")
            entries.append((target, path, digest))
    for target, path, _digest in entries:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(path.read_bytes())
    return {
        "public_cells": len(CELLS),
        "files": len(entries),
        "test_data_opened": False,
    }


def evaluation_report(root: Path) -> dict:
    """Assess every ablation with the same independent endpoint evaluator."""
    from autoformalism.rebuttal.final_evaluation import (
        FrozenEvaluationSubject,
        evaluate_frozen_subject,
    )
    from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec

    plan = verify(root, execution=False)
    receipt = sealed_read(root / "evaluation_freeze.json")
    if receipt["plan_sha256"] != plan["artifact_sha256"]:
        raise ValueError("evaluation freeze belongs to another plan")
    raw = (root / "subjects.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["subjects_sha256"]:
        raise ValueError("frozen subjects changed")
    frozen = {
        s.subject_id: s
        for s in (
            FrozenEvaluationSubject.model_validate_json(line)
            for line in raw.decode().splitlines()
        )
    }
    rows = []
    for row in receipt["roster"]:
        path = root / "evaluation" / row["subject_id"] / "result.json"
        value = sealed_read(path) if path.exists() else None
        common = None
        if value:
            execution = sealed_read(root / "evaluation-runtime/identity.json")
            subject = frozen[row["subject_id"]]
            reporting.checked_evaluation(
                value,
                subject,
                reporting.evaluation_binding(receipt, subject, execution),
            )
            if "subject" in value:
                updated = FrozenEvaluationSubject.model_validate(value["subject"])
                spec = MechanismEvaluationSpec.model_validate(
                    plan["cells"][row["task"]["cell"]]["mechanism_spec"]
                )
                common = evaluate_frozen_subject(updated, spec).model_dump(mode="json")
        rows.append(
            {
                **row,
                "status": value["status"]
                if value
                else "evaluation_missing"
                if row["model_available"]
                else "model_unavailable",
                "test_nmse": value.get("nmse") if value else None,
                "common_endpoints": common,
            }
        )
    result = {
        "freeze_sha256": receipt["artifact_sha256"],
        "planned_endpoints": len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "rows": rows,
        "private_scores_for_reporting_only": True,
    }
    public._write(root / "component_evaluation.json", result)
    return result


def method_name(task: dict, endpoint: str) -> str:
    """Stable across benchmark/seed; includes the actual treatment and endpoint."""
    return (
        f"autoformalism:{task['arm']}:c{int(task['critic'])}"
        f"v{int(task['scientific_verifier'])}s{int(task['shared_processes'])}:{endpoint}"
    )


def merge(summaries: list[Path], output: Path) -> dict:
    """Join disjoint site blocks; never average duplicated or incompatible campaigns."""
    if not summaries:
        raise ValueError("provide --source component_summary.json for every site")
    values = [json.loads(path.read_text()) for path in summaries]
    reference = values[0]
    ignore = {"campaign_blocks", "platform"}
    expected = {k: v for k, v in reference["config"].items() if k not in ignore}
    rows, blocks, seen = [], set(), set()
    for value in values:
        if (
            value["source_sha256"] != reference["source_sha256"]
            or value["public_assets"] != reference["public_assets"]
            or {k: v for k, v in value["config"].items() if k not in ignore} != expected
        ):
            raise ValueError(
                "site code, public assets or scientific configuration differ"
            )
        site_blocks = {r["task"]["block"] for r in value["rows"]}
        if blocks & site_blocks:
            raise ValueError("duplicate matched block across sites")
        blocks |= site_blocks
        for row in value["rows"]:
            key = (row["task"]["task_id"], row["endpoint"])
            if key in seen:
                raise ValueError("duplicate endpoint")
            seen.add(key)
            rows.append(
                {
                    **row,
                    "site_identity": value["identity"],
                    "platform": value["config"]["platform"],
                }
            )
    result = {
        "source_sha256": reference["source_sha256"],
        "site_identities": [v["identity"] for v in values],
        "present_blocks": sorted(blocks),
        "missing_blocks": sorted(set(range(36)) - blocks),
        "all_sites_frozen": all(v["frozen_for_evaluation"] for v in values),
        "endpoint_status_counts": dict(Counter(r["status"] for r in rows)),
        "rows": rows,
        "test_data_opened": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    public._write(output / "merged_component_summary.json", result)
    return {k: v for k, v in result.items() if k != "rows"}
