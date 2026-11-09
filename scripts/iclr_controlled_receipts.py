"""Export public fitting and critic evidence for a frozen controlled campaign."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import iclr_controlled_search as search
from autoformalism.rebuttal import iclr_search_controls as controls
from autoformalism.rebuttal import process_pruning
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.review_deadline_io import execution_lease, round_path
from autoformalism.staged_topology import content_hash
from scripts.iclr_anchor_fidelity import verify_controlled_bundle

HASH = public.content_sha256
MAX_QUOTE = 4096


def csv_export(path: Path, rows: list[dict], columns: tuple[str, ...]):
    """Write CSVs with JSON-valued cells and explicit empty headers."""
    fields = list(dict.fromkeys([*columns, *(k for row in rows for k in row)]))
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True)
                    if isinstance(v, (list, dict))
                    else v
                    for k, v in row.items()
                }
            )
    tmp.replace(path)


def fit_receipt(directory, bind, expected, envelope_hash=None):
    """Authenticate the fit envelope and bound numerical backend."""
    path = directory / "result.json"
    if not path.is_file():
        if expected is not None:
            raise ValueError("missing receipt for reported fit")
        return None
    env = public._read(path)
    fit = env["result"]
    digest = HASH(env)
    if (
        env["sha256"] != HASH(fit)
        or (expected is not None and fit != expected)
        or (envelope_hash is not None and envelope_hash != digest)
    ):
        raise ValueError("fit envelope differs from selection")
    backend = {}
    bsha = fit.get("backend_result_sha256")
    if bsha:
        backend = public._read(directory / "backend_result.json")
        if HASH(backend) != bsha:
            raise ValueError("fit numerical backend differs")
    init = backend.get("initializer") or {}
    ref = backend.get("refinement") or {}
    stages = [r.get("result") or {} for r in ref.get("stages") or []]
    nfev = [r.get("nfev") for r in stages]
    portfolio = [r.get("iterations") for r in init.get("portfolio") or []]
    return {
        **bind,
        "fit_envelope_sha256": digest,
        "backend_sha256": bsha,
        "fit_identity": fit["identity"],
        "candidate_sha256": fit.get("lowered_candidate_sha256"),
        "request_sha256": fit.get("request_sha256"),
        "fit_status": fit["status"],
        "actual_residual_calls": fit.get("actual_residual_calls"),
        "initializer_iterations": init.get("iterations"),
        "refinement_residual_calls": ref.get("actual_residual_calls"),
        "optimizer_nfev": sum(nfev) if nfev and None not in nfev else ref.get("nfev"),
        "refinement_stage_nfev": nfev,
        "initializer_portfolio_iterations": portfolio,
        "budget_exhausted": fit.get("budget_exhausted"),
        "metric_note": "null means unavailable; stages have distinct work units",
    }


def review_rows(receipt, bind, plan, parent_field, parent_sha):
    """Export bounded verbatim critic findings and explicit empty reviews."""
    if (
        receipt["identity"] != plan
        or receipt["task"]["task_id"] != bind["task"]
        or (receipt.get(parent_field) != parent_sha)
    ):
        raise ValueError("critic receipt does not match its source")
    if receipt.get("used_for_selection") or receipt.get("selection_uses_critic"):
        raise ValueError("unexpected critic selection behavior")
    review = receipt.get("review") or {}
    request = receipt.get("request_sha256")
    if request and review.get("request_sha256") != request:
        raise ValueError("critic finding belongs to another request")
    findings = review.get("findings") or []
    common = {
        **bind,
        "critic_receipt_sha256": receipt["artifact_sha256"],
        "request_sha256": request,
        "critic_status": receipt["status"],
        "reused": receipt.get("reused"),
        "findings_count": len(findings),
        "parent_sha256": parent_sha,
    }
    if not findings:
        return [common]
    result = []
    for idx, f in enumerate(findings):
        verbatim = json.dumps(f, ensure_ascii=False, sort_keys=True)
        result.append(
            {
                **common,
                "ordinal": idx,
                "finding_sha256": HASH(f),
                "finding_json_text": verbatim[:MAX_QUOTE],
                "truncated": len(verbatim) > MAX_QUOTE,
            }
        )
    return result


def shared(root, full, fixed):
    """Compare the selected candidate, coefficients, initializer, fit and critic."""
    a, b = (sealed_read(round_path(root, t, 0) / "result.json") for t in (full, fixed))
    if (
        a["task"] != full
        or b["task"] != fixed
        or b["status"] != "shared_full_round_zero"
        or b["shared_source_sha256"] != a["artifact_sha256"]
        or b["cost"]
        or b["trial"] is not None
    ):
        raise ValueError("shared initial origin differs")
    sa, sb = a["selected"], b["selected"]
    record = {
        "block": full["block"],
        "cell": full["cell"],
        "seed": full["seed"],
        "full_task": full["task_id"],
        "fixed_task": fixed["task_id"],
        "full_result_sha256": a["artifact_sha256"],
        "fixed_result_sha256": b["artifact_sha256"],
        "proposer_and_fit_charged_to": full["task_id"],
    }
    if sa is None and sb is None:
        return {**record, "status": "no_shared_retained_model"}
    if not sa or not sb or sb != {**sa, "origin_task": fixed["task_id"]}:
        raise ValueError("shared selected candidate or fitted state differs")
    dirs = [round_path(root, t, 0) for t in (full, fixed)]
    left, right = (public._read(d / "fit/result.json") for d in dirs)
    if (
        left != right
        or HASH(left) != sa["fit_result_sha256"]
        or left["result"] != sa["fit"]
    ):
        raise ValueError("shared fit receipts differ")
    critic_hash = None
    if full["critic"]:
        ra, rb = (sealed_read(d / "critic.json") for d in dirs)
        if ra != rb:
            raise ValueError("shared critic receipts differ")
        critic_hash = ra["artifact_sha256"]
    return {
        **record,
        "status": "verified_shared_model",
        "candidate_sha256": HASH(sa["bundle"]["candidate"]),
        "state_representation_sha256": HASH(controls.gamma(sa["bundle"])),
        "initialization_plan_sha256": HASH(sa["request"]["initialization_plan"]),
        "parameters_sha256": HASH(sa["fit"]["parameters"]),
        "training_metrics_sha256": HASH(sa["fit"]["training"]),
        "validation_metrics_sha256": HASH(sa["fit"]["validation"]),
        "fit_envelope_sha256": HASH(left),
        "critic_sha256": critic_hash,
        "critic_enabled": full["critic"],
        "proposal_sha256": a["proposal_sha256"],
    }


def visit_records(root, plan, task, outputs):
    """Verify source links for each visit, including failed searches."""
    parent = None
    anchor = None
    for i in range(plan["visits"]):
        directory = round_path(root, task, i)
        r = sealed_read(directory / "result.json")
        if r["task"] != task or r["round"] != i:
            raise ValueError("visit identity mismatch")
        initial = r["status"] == "shared_full_round_zero"
        unavailable = r["status"] == "shared_initial_unavailable"
        if not unavailable and r.get("parent_sha256") != parent:
            raise ValueError("visit parent differs")
        if unavailable and r.get("shared_initial_sha256") != anchor:
            raise ValueError("unavailable start belongs to another lineage")
        parent = r["artifact_sha256"]
        if i == 0:
            anchor = parent
        bind = {
            "task": task["task_id"],
            "cell": task["cell"],
            "treatment": task["treatment"],
            "seed": task["seed"],
            "visit": i + 1,
            "stage": "search",
        }
        trial = r["trial"]
        selected = r["selected"]
        prop = {}
        ppath = directory / "proposal.json"
        if not initial and not unavailable:
            prop = sealed_read(ppath)
            if (
                prop["artifact_sha256"] != r["proposal_sha256"]
                or prop["task"] != task
                or prop["round"] != i
                or prop["parent_sha256"] != r["parent_sha256"]
            ):
                raise ValueError("proposal disagrees with visit")
            for event in _cache_records(
                directory / "calls", content_hash([plan["artifact_sha256"], task, i])
            ):
                outputs["proposer_requests"].append(
                    {
                        **bind,
                        "request_sha256": event["request_hash"],
                        "step": event["step"],
                        "status": event["status"],
                        "response_sha256": HASH(event["raw_response"])
                        if event.get("raw_response") is not None
                        else None,
                        "attempt": event.get("attempt"),
                        "tokens": event.get("observed_total_tokens"),
                    }
                )
            rec = fit_receipt(
                directory / "fit",
                bind,
                trial["fit"] if trial else None,
                trial["fit_result_sha256"] if trial else None,
            )
            if rec:
                outputs["fit_evaluations"].append(rec)
            review_path = directory / "critic.json"
            if task["critic"]:
                review = sealed_read(review_path)
                if review["round"] != i:
                    raise ValueError("wrong critic visit")
                outputs["critic_findings"].extend(
                    review_rows(
                        review,
                        bind,
                        plan["artifact_sha256"],
                        "proposal_sha256",
                        prop["artifact_sha256"],
                    )
                )
        attempts = prop.get("attempts") or []
        provenance = (prop.get("decision") or {}).get("provenance") or {}
        outputs["visits"].append(
            {
                **bind,
                "visit_sha256": parent,
                "parent_sha256": r.get("parent_sha256"),
                "visit_status": r["status"],
                "proposal_status": r.get("proposal_status"),
                "proposal_sha256": r.get("proposal_sha256"),
                "shared_source_sha256": r.get("shared_source_sha256"),
                "fit_trigger": r.get("fit_trigger"),
                "physical_fit_receipt_status": (
                    "shared_charged_to_full"
                    if initial
                    else "shared_start_unavailable"
                    if unavailable
                    else "observed"
                    if (directory / "fit/result.json").exists()
                    else "no_recorded_fit"
                ),
                "selected_new_trial": r.get("selected_new_trial"),
                "trial_model_sha256": HASH(trial["bundle"]["candidate"])
                if trial
                else None,
                "selected_model_sha256": HASH(selected["bundle"]["candidate"])
                if selected
                else None,
                "trial_fit_sha256": trial["fit_result_sha256"] if trial else None,
                "selected_fit_sha256": selected["fit_result_sha256"]
                if selected
                else None,
                "revision_provenance_sha256": HASH(provenance) if provenance else None,
                "recorded_revision_effects": (provenance.get("audit") or {}).get(
                    "actual_effects"
                ),
                "recorded_attempts": [
                    {
                        "ordinal": j,
                        "accepted": item.get("accepted"),
                        "request_sha256": item.get("request_hash"),
                        "attempt_sha256": HASH(item),
                        "error": str(item.get("error", ""))[:512],
                    }
                    for j, item in enumerate(attempts)
                    if isinstance(item, dict)
                ],
                "attempt_sequence_sha256": HASH(attempts) if attempts else None,
                "selected_origin_task": selected.get("origin_task")
                if selected
                else None,
                "selected_origin_round": selected.get("origin_round")
                if selected
                else None,
            }
        )


def public_check(raw, task, frozen, start, arm):
    """Evaluate the frozen equations against assigned public predicates."""
    data = {
        "task": task["task_id"],
        "cell": task["cell"],
        "seed": task["seed"],
        "treatment": task["treatment"],
        "evaluated_task": arm,
        "source_sha256": frozen["source_sha256"],
        "endpoint_status": frozen["status"],
    }
    if frozen["status"] != "ready":
        return data
    anchor = controls.gamma(start["bundle"]) if start and arm == "fixed_gamma" else None
    contract = controls.treatment(raw, arm, anchor)
    b = {
        "candidate": frozen["candidate"],
        "initialization": {"context": frozen["context"]},
        "lowered_pruning_request": True,
    }
    working = {**task, "scientific_verifier": arm != "observed_only"}
    cert = controls.certificate(b, contract, working)
    return {
        **data,
        "model_sha256": frozen["model_sha256"],
        "mechanism_requirements": len(cert["mechanisms"]["mechanism_results"]),
        "requirements": cert["mechanisms"]["mechanism_results"],
        "target_results": cert["targets"],
        "controlled_checks": cert["controlled_checks"],
        "all_graph_requirements_certified": cert[
            "all_public_graph_requirements_certified"
        ],
        "eligible_for_development_selection": cert[
            "eligible_for_development_selection"
        ],
        "scientific_correctness_certified": False,
    }


def bind_endpoint(frozen, task, raw, last, prune):
    """Match the frozen endpoint to the original pruning or final-visit fit."""
    source = prune if prune is not None else last
    selected = last["selected"] or {}
    request = selected.get("request")
    fit = selected.get("fit")
    if prune and prune.get("selection"):
        decision = prune["selection"]
        fit = decision["fit"]
        if decision["selected"] == "pruned":
            request = prune["choice"]["request"]
    if (
        frozen["source_sha256"] != source["artifact_sha256"]
        or frozen["public_spec_sha256"] != raw["public_spec_sha256"]
        or frozen["public_file_sha256"] != raw["files"]
        or (frozen["method"], frozen["repetition"], frozen["benchmark_id"])
        != (task["treatment"], task["seed"], task["cell"])
    ):
        raise ValueError("source endpoint differs from frozen assignment")
    if fit and fit["status"] == "complete" and fit["validation"]["available"]:
        if (
            frozen["status"] != "ready"
            or HASH(frozen["candidate"]) != frozen["model_sha256"]
            or fit["lowered_candidate_sha256"] != frozen["model_sha256"]
            or fit["request_sha256"] != HASH(request)
            or frozen["parameters"] != fit["parameters"]
            or fit["validation"]["normalized_mse"] != frozen["validation_nmse"]
            or fit["training"]["normalized_mse"] != frozen["training_nmse"]
        ):
            raise ValueError("frozen endpoint differs from final fit")
    elif frozen["status"] != "no_retained_model":
        raise ValueError("frozen endpoint has no validated selection")


def export(root: Path, out: Path) -> dict:
    """Export portable evidence from a complete frozen production roster."""
    root = root.resolve()
    if out.resolve() == root:
        raise ValueError("use a separate audit output directory")
    with execution_lease(root, exclusive=True), public._lock(out):
        plan = sealed_read(root / "plan.json")
        bundle = sealed_read(root / "frozen_models.json")
        if (
            plan["protocol"] != search.PROTOCOL
            or bundle["protocol"] != "iclr-controlled-search-models-1"
            or bundle["test_data_opened"] is not False
            or bundle["private_reference_opened"] is not False
            or plan["test_data_opened"]
            or plan["private_reference_opened"]
        ):
            raise ValueError("require a public controlled search")
        verify_controlled_bundle(root / "frozen_models.json", bundle)
        by_task = {r["source_id"]: r for r in bundle["rows"]}
        names = (
            "visits",
            "fit_evaluations",
            "proposer_requests",
            "critic_findings",
            "shared_initials",
            "final_public_checks",
        )
        outputs = {n: [] for n in names}
        for block in plan["blocks"]:
            raw = sealed_read(root / "cells" / (block["cell"] + ".json"))
            if raw["artifact_sha256"] != plan["cells"][block["cell"]]:
                raise ValueError("incorrect cell")
            by_arm = {t["treatment"]: t for t in block["tasks"]}
            if "fixed_gamma" in by_arm:
                outputs["shared_initials"].append(
                    shared(root, by_arm["full"], by_arm["fixed_gamma"])
                )
            for task in block["tasks"]:
                frozen = by_task[task["task_id"]]
                last = sealed_read(
                    round_path(root, task, plan["visits"] - 1) / "result.json"
                )
                rec = None
                visit_records(root, plan, task, outputs)
                if plan["pruning_enabled"]:
                    folder = root / "pruning" / task["task_id"]
                    rec = sealed_read(folder / "result.json")
                    if (
                        rec["identity"] != plan["artifact_sha256"]
                        or rec["artifact_sha256"] != frozen["source_sha256"]
                        or rec["task"] != task
                    ):
                        raise ValueError("pruning endpoint mismatch")
                    if rec.get("choice") is not None:
                        if sealed_read(folder / "choice.json") != rec["choice"]:
                            raise ValueError("pruning choice differs")
                        if rec.get("selection") != process_pruning.decide(
                            last["selected"]["fit"],
                            rec["fits"]["control"],
                            rec["fits"]["pruned"],
                            rec["choice"],
                        ):
                            raise ValueError("pruning selection differs")
                    elif rec.get("source_result_sha256") != last["artifact_sha256"]:
                        raise ValueError("skipped pruning belongs to another search")
                    for stage, fit in (rec.get("fits") or {}).items():
                        if stage not in ("control", "pruned"):
                            raise ValueError("unknown pruning fit role")
                        bind = {"task": task["task_id"], "stage": "pruning_" + stage}
                        fc = fit_receipt(folder / stage, bind, fit)
                        if fc:
                            outputs["fit_evaluations"].append(fc)
                    if (folder / "critic.json").is_file():
                        crit = sealed_read(folder / "critic.json")
                        outputs["critic_findings"].extend(
                            review_rows(
                                crit,
                                {"task": task["task_id"], "stage": "pruning"},
                                plan["artifact_sha256"],
                                "pruning_sha256",
                                rec["artifact_sha256"],
                            )
                        )
                bind_endpoint(frozen, task, raw, last, rec)
                start = sealed_read(round_path(root, task, 0) / "result.json")[
                    "selected"
                ]
                for purpose, arm in (
                    ("assigned", task["treatment"]),
                    ("original", "full"),
                ):
                    outputs["final_public_checks"].append(
                        {
                            "perspective": purpose,
                            **public_check(raw, task, frozen, start, arm),
                        }
                    )
        report = sealed_write(
            out / "endpoint_audit.json",
            {
                "protocol": "iclr-production-receipts-1",
                "plan_sha256": plan["artifact_sha256"],
                "frozen_bundle_sha256": bundle["artifact_sha256"],
                "source_sha256": search.file_digest(Path(__file__)),
                "source_freeze_sha256": bundle["selection_freeze_sha256"],
                "endpoint_status": dict(Counter(r["status"] for r in by_task.values())),
                "advice_interpretation": "recorded advisory findings",
                "public_checks_scope": "graph predicates of the frozen equations",
                "counts": {k: len(v) for k, v in outputs.items()},
                "records": outputs,
            },
        )
        for name, rows in outputs.items():
            fields = (
                ("cell", "seed", "full_task", "fixed_task")
                if name == "shared_initials"
                else ("task",)
            )
            csv_export(out / (name + ".csv"), rows, fields)
        return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(export(a.root, a.output)["counts"]))
