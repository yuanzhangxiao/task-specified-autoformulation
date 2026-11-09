"""Frozen, development-only, paired representation and specification experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from time import monotonic

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.expressions import ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import component_campaign as components
from autoformalism.rebuttal import component_critic as critic
from autoformalism.rebuttal import iclr_search_controls as controls
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit
from autoformalism.search import scientific_verification

PROTOCOL = "iclr-controlled-search-1"


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_cell(directory: Path, root: Path, cfg: dict) -> dict:
    """Ingest only the four public development assets through their native loader."""
    name = directory.name
    if (directory / "specification.json").is_file():
        spec = read_seal(directory / "specification.json")
        prompt = (directory / "proposer_prompt.txt").read_text()
        if spec["benchmark_id"] != name or spec["public_prompt"].rstrip(
            "\n"
        ) != prompt.rstrip("\n"):
            raise ValueError(f"mismatched public specification: {name}")
        fields = ("targets", "auxiliaries", "external_inputs", "fixed_covariates")
        context = ValidationContext.model_validate({k: spec[k] for k in fields})
        splits = [
            PublicSplit.model_validate(read_seal(directory / f"{k}.json"))
            for k in ("train", "val")
        ]
        files = ("specification.json", "proposer_prompt.txt", "train.json", "val.json")
    else:
        files = io.FILES
        staged = root / "registered-inputs" / "phase_b_v1" / name
        staged.mkdir(parents=True, exist_ok=True)
        for f in files:
            destination = staged / f
            if not destination.exists():
                shutil.copyfile(directory / f, destination)
            elif file_digest(destination) != file_digest(directory / f):
                raise ValueError("partially staged public file differs")
        development = io.load_development(root / "registered-inputs", name)
        context = io.public_validation_context(name)
        prompt = (directory / "proposer_prompt.txt").read_text()
        splits = [
            public.pack_split(s) for s in (development.train, development.validation)
        ]
    if [s.name for s in splits] != ["train", "val"]:
        raise ValueError("split role changed")
    for s in splits:
        for row in s.rows:
            for role in (
                "targets",
                "auxiliaries",
                "external_inputs",
                "fixed_covariates",
            ):
                if set(getattr(row, role)) != set(getattr(context, role)):
                    raise ValueError(f"split has incorrect public {role} in {name}")
    evidence = io.build_training_evidence(
        public.unpack_split(splits[0]),
        context,
        io.EvidenceSettings.model_validate(cfg["evidence"]),
    )
    hashes = {f: file_digest(directory / f) for f in files}
    return {
        "benchmark_id": name,
        "tier": "hard" if not context.auxiliaries or "hard" in name else "public",
        "prompt": prompt,
        "public_spec_sha256": hashes.get(
            "specification.json", hashes["proposer_prompt.txt"]
        ),
        "files": hashes,
        "context": context.model_dump(mode="json"),
        "limits": cfg["limits"],
        "meal_input": next((n for n in context.external_inputs if "meal" in n), None),
        "training": splits[0].model_dump(mode="json"),
        "validation": splits[1].model_dump(mode="json"),
        "evidence": evidence.model_dump(mode="json"),
    }


def prepare(
    root: Path,
    public_root: Path,
    names: list[str],
    arms: list[str],
    seeds: list[int],
    visits: int,
    critic_on: bool,
    prune: bool,
    calibration: tuple | None = None,
    max_parameters: int | None = None,
) -> dict:
    """Seal the roster, public inputs, resource caps and runtime."""
    if max_parameters is not None and max_parameters < 1:
        raise ValueError("positive parameter cap required")
    if not names or not arms or not seeds or not 1 <= visits <= 15:
        raise ValueError("nonempty roster and one to fifteen visits required")
    if any(len(set(v)) != len(v) for v in (names, arms, seeds)) or set(arms) - set(
        controls.ARMS
    ):
        raise ValueError("duplicate or unsupported treatment, cell or seed")
    if any(s < 0 or s > 999999 for s in seeds) or any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", n) for n in names
    ):
        raise ValueError("invalid public cell or seed")
    if "fixed_gamma" in arms and "full" not in arms:
        raise ValueError("fixed_gamma requires full in the same plan")
    if root.resolve().is_relative_to(
        public_root.resolve()
    ) or public_root.resolve().is_relative_to(root.resolve()):
        raise ValueError("development release and campaign output must be disjoint")
    cfg = json.loads((io.REPO / "configs/final_component_campaign_v1.json").read_text())
    cfg["rounds"] = visits
    blocks = []
    cells = {}
    variants = {}
    ordered = sorted(arms, key=lambda s: controls.ARMS.index(s))
    with public._lock(root):
        for name in names:
            cell = read_cell(public_root / name, root, cfg)
            cell["max_parameters"] = max_parameters
            seal = sealed_write(root / "cells" / f"{name}.json", cell)
            cells[name] = seal["artifact_sha256"]
            variants[name] = {}
            for arm in ordered:
                active = controls.treatment(cell, arm)
                variants[name][arm] = {
                    "prompt_sha256": active["target_contract"]["public_prompt_sha256"],
                    "source_clauses": controls.clauses(cell["prompt"]),
                    "active_clauses": active["active_clauses"],
                    "prompt": active["active_prompt"],
                }
            for seed in seeds:
                bi = len(blocks)
                blocks.append(
                    {
                        "block": bi,
                        "cell": name,
                        "seed": seed,
                        "tasks": [
                            {
                                "index": bi * len(arms) + i,
                                "cell": name,
                                "seed": seed,
                                "arm": "full",
                                "treatment": arm,
                                "block": bi,
                                "task_id": f"block{bi:04d}_{arm}",
                                "critic": critic_on,
                                "scientific_verifier": arm != "observed_only",
                                "shared_processes": True,
                                "secondary": False,
                                "shared_round_zero": None,
                            }
                            for i, arm in enumerate(ordered)
                        ],
                    }
                )
        variants_file = sealed_write(
            root / "spec_variants.json", {"variants": variants}
        )
        switches = sealed_write(
            root / "ablation_switch_manifest.json",
            {
                "arms": ordered,
                "seeds": seeds,
                "visits": visits,
                "max_parameters": max_parameters,
                "initial_construction": "Full and fixed share the fitted round zero.",
                "constraint": "Preserve state identities, roles and direct mappings.",
                "no_latent": "Target-only states with the public scientific gate.",
                "observed_only": "Target-only states; relaxed task admission.",
                "no_spec": "Keep data roles and preparation; remove mechanism clauses.",
                "selection": "Validation fit and model complexity.",
                "mechanism_assessment": "Public witnesses; fidelity after freeze.",
                "model_settings": cfg["model_settings"],
                "fit_profile": cfg["fit_profile"],
                "critic": critic_on,
                "pruning": prune,
            },
        )
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "cells": cells,
                "blocks": blocks,
                "switch_manifest_sha256": switches["artifact_sha256"],
                "variants_sha256": variants_file["artifact_sha256"],
                "visits": visits,
                "max_parameters": max_parameters,
                "arms": ordered,
                "seeds": seeds,
                "critic_enabled": critic_on,
                "pruning_enabled": prune,
                "serving_image_sha256": cfg["serving_image_sha256"],
                "model_settings": cfg["model_settings"],
                "fit_profile": cfg["fit_profile"],
                "source_sha256": public._source_identity(),
                "runtime": public._runtime(),
                "study": "Prospectively paired development seeds.",
                "test_data_opened": False,
                "private_reference_opened": False,
            },
        )
        if critic_on and calibration:
            critic.bind_calibration(root, plan, *calibration)
        return plan


def view(plan: dict, cell: dict) -> dict:
    """Use the actual component constructor, revision policy and fitter budgets."""
    cfg = {
        "fit_profile": plan["fit_profile"],
        "rounds": plan["visits"],
        "model_settings": plan["model_settings"],
    }
    return {
        "artifact_sha256": plan["artifact_sha256"],
        "protocol": io.ABLATION_PROTOCOL,
        "control_campaign": True,
        "config": cfg,
        "cells": {cell["benchmark_id"]: cell},
    }


def share_initial(root: Path, parent: dict, task: dict) -> dict:
    """Reuse the actual proposal, fitted parameters, score, fit and critic receipts."""
    src = io.round_path(root, parent, 0)
    dest = io.round_path(root, task, 0)
    original = sealed_read(src / "result.json")
    selected = original["selected"]
    if selected:
        selected = {**selected, "origin_task": task["task_id"]}
        for name in (
            ("fit/result.json", "critic.json")
            if task["critic"]
            else ("fit/result.json",)
        ):
            source, target = src / name, dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                public._write(target, public._read(source))
            if public._read(source) != public._read(target):
                raise ValueError("shared initial fit/critic receipt differs")
    return sealed_write(
        dest / "result.json",
        {
            **{k: v for k, v in original.items() if k != "artifact_sha256"},
            "task": task,
            "selected": selected,
            "trial": None,
            "status": "shared_full_round_zero",
            "shared_source_sha256": original["artifact_sha256"],
            "cost": {},
        },
    )


def run_block(
    root: Path,
    number: int,
    url: str,
    seconds: int,
    *,
    transport=None,
    token_transport=None,
) -> dict:
    """Run one bounded cell/seed block with exclusive ownership and immutable visits."""
    plan = sealed_read(root / "plan.json")
    if (
        plan["protocol"] != PROTOCOL
        or plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("controlled-search implementation or runtime changed")
    if not 0 <= number < len(plan["blocks"]) or seconds <= 0:
        raise ValueError("invalid block index or wall allowance")
    block = plan["blocks"][number]
    tasks = block["tasks"]
    raw = sealed_read(root / "cells" / (block["cell"] + ".json"))
    if raw["artifact_sha256"] != plan["cells"][block["cell"]]:
        raise ValueError("public cell snapshot changed")
    deadline = monotonic() + seconds

    def key() -> str:
        return os.environ["AF_JETSTREAM_API_KEY"]

    with io.execution_lease(root), public._lock(root / "locks" / str(number)):
        if plan["critic_enabled"]:
            critic.checked_authorization(root, plan)
        for i in range(plan["visits"]):
            for task in tasks:
                if io.read_round(root, task, i) is not None:
                    continue
                if monotonic() >= deadline:
                    return {"status": "pending", "block": number}
                arm = task["treatment"]
                if i == 0 and arm == "fixed_gamma":
                    share_initial(root, tasks[0], task)
                    continue
                paired = "fixed_gamma" in plan["arms"] and arm in {
                    "fixed_gamma",
                    "full",
                }
                start = io.read_round(root, task, 0) if i and paired else None
                if start and start["selected"] is None:
                    # A paired representation comparison requires a common usable start.
                    sealed_write(
                        io.round_path(root, task, i) / "result.json",
                        {
                            "task": task,
                            "round": i,
                            "status": "shared_initial_unavailable",
                            "selected": None,
                            "trial": None,
                            "cost": {},
                            "shared_initial_sha256": start["artifact_sha256"],
                        },
                    )
                    continue
                frozen = (
                    controls.gamma(start["selected"]["bundle"])
                    if start and arm == "fixed_gamma"
                    else None
                )
                cell = controls.treatment(raw, arm, frozen)
                current = view(plan, cell)
                directory = io.round_path(root, task, i)
                started = monotonic()
                with scientific_verification.scope(task["scientific_verifier"]):
                    client = pipeline._client(
                        root,
                        current,
                        task,
                        i,
                        url,
                        lambda: monotonic() < deadline,
                        transport,
                    )
                    if token_transport is not None:
                        client.token_transport = token_transport
                    proposal = pipeline.propose_one(root, current, task, i, client)
                    if not proposal:
                        raise RuntimeError("missing parent for scheduled visit")
                    if task["critic"]:
                        critic.review_one(root, current, task, i, key)
                    fit_started = monotonic()
                    pipeline.fit_one(root, current, task, i)
                sealed_write(
                    directory / "cost.json",
                    {
                        "fit_seconds": monotonic() - fit_started,
                        "visit_seconds": monotonic() - started,
                        "plan_sha256": plan["artifact_sha256"],
                    },
                )
        if plan["pruning_enabled"]:
            for task in tasks:
                if monotonic() >= deadline:
                    return {"status": "pending", "block": number}
                first = io.read_round(root, task, 0)["selected"]
                anchor = (
                    controls.gamma(first["bundle"])
                    if first and task["treatment"] == "fixed_gamma"
                    else None
                )
                current = view(plan, controls.treatment(raw, task["treatment"], anchor))
                pruning_started = monotonic()
                existing_prune = (
                    root / "pruning" / task["task_id"] / "result.json"
                ).exists()
                result = components.prune_one(root, current, task)
                if task["critic"] and result["status"] != "skipped_no_finite_parent":
                    critic.review_final(root, current, task, key)
                cp = root / "pruning" / task["task_id"] / "cost.json"
                if not cp.exists():
                    sealed_write(
                        cp,
                        {
                            "pruning_seconds": None
                            if existing_prune
                            else monotonic() - pruning_started
                        },
                    )
        return {"status": "complete", "block": number}


def add_review_cost(root: Path, review: dict, counted: set, totals: Counter) -> None:
    """Count referenced critic transport records once within this task."""
    digest = review.get("request_sha256")
    if not digest or review.get("reused") or digest in counted:
        return
    counted.add(digest)
    for marker in (root / "judge_cache" / digest / "wire").glob("call-*/started.json"):
        totals["critic_calls"] += 1
        result = marker.with_name("result.json")
        data = public._read(result) if result.exists() else {}
        response = data.get("response")
        usage = (response.get("usage") or {}) if isinstance(response, dict) else {}
        if type(usage.get("total_tokens")) is int:
            totals["observed_critic_tokens"] += usage["total_tokens"]
        else:
            totals["unknown_critic_token_counts"] += 1


def summarize_row(root: Path, plan: dict, raw: dict, task: dict) -> tuple[dict, list]:
    arm = task["treatment"]
    row = {
        "method": arm,
        "repetition": task["seed"],
        "benchmark_id": task["cell"],
        "public_spec_sha256": raw["public_spec_sha256"],
        "public_file_sha256": raw["files"],
        "source_id": task["task_id"],
    }
    trace = []
    totals = Counter()
    counted_reviews = set()
    for i in range(plan["visits"]):
        result = io.read_round(root, task, i)
        if not result:
            return {**row, "status": "pending", "cost": dict(totals)}, trace
        selected = result["selected"]
        directory = io.round_path(root, task, i)
        cost = result["cost"]
        clock_path = directory / "cost.json"
        totals.update(
            {
                "proposal_calls": cost.get("physical_requests", 0),
                "observed_proposal_tokens": cost.get("observed_tokens", 0),
                "unknown_proposal_token_counts": cost.get(
                    "requests_with_unknown_usage", 0
                ),
            }
        )
        if clock_path.exists():
            clock = sealed_read(clock_path)
            totals["fit_seconds"] += clock["fit_seconds"]
            totals["visit_seconds"] += clock["visit_seconds"]
        elif result["status"] not in {
            "shared_full_round_zero",
            "shared_initial_unavailable",
        }:
            totals["unknown_visit_costs"] += 1
        review_path = directory / "critic.json"
        if (
            task["critic"]
            and review_path.exists()
            and result["status"] != "shared_full_round_zero"
        ):
            review = sealed_read(review_path)
            add_review_cost(root, review, counted_reviews, totals)
        proposal_path = directory / "proposal.json"
        if proposal_path.exists():
            proposal = sealed_read(proposal_path)
            diagnostics = proposal.get("attempts") or []
            totals["recorded_rejected_attempts"] += sum(
                item.get("accepted") is False
                for item in diagnostics
                if isinstance(item, dict)
            )
        critic_data = sealed_read(review_path) if review_path.exists() else {}
        feedback = critic_data.get("review") or {}
        findings = feedback.get("findings") or []
        rejection = []
        if proposal_path.exists():
            if proposal.get("error"):
                rejection.append(str(proposal["error"])[:600])
            for entry in diagnostics:
                if not isinstance(entry, dict) or entry.get("accepted") is True:
                    continue
                reason = (
                    entry.get("error")
                    or (entry.get("feedback") or {}).get("message")
                    or entry.get("code")
                )
                if reason:
                    rejection.append(str(reason)[:600])
        trace.append(
            {
                **row,
                "visit": i,
                "critic_status": critic_data.get(
                    "status", "off" if not task["critic"] else "unavailable"
                ),
                "critic_findings_json": json.dumps(findings, sort_keys=True),
                "rejections_json": json.dumps(rejection, sort_keys=True),
                "status": result["status"],
                "proposal_status": result.get("proposal_status"),
                "selected_origin": selected["origin_round"] if selected else None,
                "validation_nmse": selected["fit"]["validation"]["normalized_mse"]
                if selected
                else None,
                "states": len(selected["bundle"]["candidate"]["states"])
                if selected
                else None,
                "parameters": len(selected["fit"]["parameters"]) if selected else None,
                "requirements": selected["certificate"][
                    "all_public_graph_requirements_certified"
                ]
                if selected
                else None,
            }
        )
    first = io.read_round(root, task, 0)["selected"]
    anchor = controls.gamma(first["bundle"]) if first and arm == "fixed_gamma" else None
    current = view(plan, controls.treatment(raw, arm, anchor))
    if plan["pruning_enabled"]:
        endpoint = components.endpoints(root, current, task)[0]
        if endpoint["status"] == "missing":
            return {**row, "status": "pending", "cost": dict(totals)}, trace
        request, fit = endpoint["request"], endpoint["fit"]
        receipt = endpoint["source_sha256"]
    else:
        request, fit = (
            (selected["request"], selected["fit"]) if selected else (None, None)
        )
        receipt = result["artifact_sha256"]
    prune_cost = root / "pruning" / task["task_id"] / "cost.json"
    if prune_cost.exists():
        duration = sealed_read(prune_cost)["pruning_seconds"]
        if duration is None:
            totals["unknown_pruning_costs"] += 1
        else:
            totals["pruning_seconds"] = duration
    final_critic = root / "pruning" / task["task_id"] / "critic.json"
    if final_critic.exists():
        add_review_cost(root, sealed_read(final_critic), counted_reviews, totals)
    totals["search_and_pruning_wall_seconds"] = (
        totals["visit_seconds"] + totals["pruning_seconds"]
    )
    if arm == "fixed_gamma":
        others = next(
            t
            for t in plan["blocks"][task["block"]]["tasks"]
            if t["treatment"] == "full"
        )
        initial = io.read_round(root, others, 0)
        row["shared_initial_result_sha256"] = initial["artifact_sha256"]
        d = io.round_path(root, others, 0)
        cp = d / "cost.json"
        if cp.exists():
            totals["shared_initial_fit_seconds"] = sealed_read(cp)["fit_seconds"]
            totals["shared_initial_visit_seconds"] = sealed_read(cp)["visit_seconds"]
        cp = d / "critic.json"
        if cp.exists():
            charges = Counter()
            add_review_cost(root, sealed_read(cp), set(), charges)
            totals.update({"shared_initial_" + k: v for k, v in charges.items()})
        totals["shared_initial_proposal_calls"] = initial["cost"].get(
            "physical_requests", 0
        )
        totals["shared_initial_observed_tokens"] = initial["cost"].get(
            "observed_tokens", 0
        )
    row.update(
        source_sha256=receipt,
        cost=dict(totals),
        shared_initial=arm == "fixed_gamma",
        semantics="continuous_time",
    )
    if not fit or fit["status"] != "complete" or not fit["validation"]["available"]:
        return {
            **row,
            "status": "no_retained_model",
            "last_visit_status": result["status"],
        }, trace
    model, _, _ = public._lower(PublicFitRequest.model_validate(request))
    return {
        **row,
        "status": "ready",
        "model_sha256": public.content_sha256(model.validated.candidate),
        "candidate": model.validated.candidate.model_dump(mode="json"),
        "context": model.validated.context.model_dump(mode="json"),
        "parameters": fit["parameters"],
        "initials": {},
        "training_nmse": fit["training"]["normalized_mse"],
        "validation_nmse": fit["validation"]["normalized_mse"],
    }, trace


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
        )
        writer.writeheader()
        writer.writerows(rows)


def report(root: Path, freeze: bool) -> dict:
    """Report all planned denominators; close campaign before exposing frozen models."""
    with io.execution_lease(root, exclusive=True):
        plan = sealed_read(root / "plan.json")
        rows = []
        trace = []
        for block in plan["blocks"]:
            raw = sealed_read(root / "cells" / (block["cell"] + ".json"))
            if raw["artifact_sha256"] != plan["cells"][block["cell"]]:
                raise ValueError("public cell seal mismatch")
            for task in block["tasks"]:
                row, history = summarize_row(root, plan, raw, task)
                rows.append(row)
                trace.extend(history)
        correct = {
            (r["benchmark_id"], r["repetition"]): r
            for r in rows
            if r["method"] == "full"
        }
        for r in rows:
            q = correct.get((r["benchmark_id"], r["repetition"]), {})
            if "validation_nmse" in r and "validation_nmse" in q:
                r["validation_nmse_delta_from_full"] = (
                    r["validation_nmse"] - q["validation_nmse"]
                )
            r["specification_verdict"] = "unresolved"
            r["diagnostic_observation"] = {
                "ready": "admitted fit; scientific premise remains unverified",
                "pending": "incomplete computation",
                "no_retained_model": "search or fit unsuccessful under budget",
            }[r["status"]]
        summary = [
            {
                k: v
                for k, v in r.items()
                if k not in {"candidate", "context", "parameters", "initials", "cost"}
            }
            | r["cost"]
            for r in rows
        ]
        for item in summary:
            item["public_file_sha256"] = json.dumps(
                item["public_file_sha256"], sort_keys=True
            )
        write_csv(root / "paired_ablation_results.csv", summary)
        write_csv(root / "search_visits.csv", trace)
        write_csv(
            root / "spec_sensitivity.csv",
            [
                r
                for r in summary
                if r["method"] in {"full", "no_spec", *controls.MUTATIONS}
            ],
        )
        if not freeze:
            return {"status_counts": dict(Counter(r["status"] for r in rows))}
        if any(r["status"] == "pending" for r in rows):
            raise ValueError("complete every planned endpoint before freezing")
        receipt = sealed_write(
            root / "evaluation_freeze.json",
            {
                "plan_sha256": plan["artifact_sha256"],
                "roster_sha256": public.content_sha256(rows),
            },
        )
        return sealed_write(
            root / "frozen_models.json",
            {
                "protocol": "iclr-controlled-search-models-1",
                "plan_sha256": plan["artifact_sha256"],
                "selection_freeze_sha256": receipt["artifact_sha256"],
                "rows": rows,
                "test_data_opened": False,
                "private_reference_opened": False,
            },
        )
