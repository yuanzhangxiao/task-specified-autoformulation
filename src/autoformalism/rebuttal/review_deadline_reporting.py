"""Complete denominators, immutable model export and common held-out evaluation."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry, FrozenTestAccess
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.models import FitConfig
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal.final_evaluation import FrozenEvaluationSubject
from autoformalism.rebuttal.postfreeze_evaluation import evaluate_subject_on_test
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records, _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest
from autoformalism.search import review_revision_v5
from autoformalism.staged_topology import content_hash


def report(root: Path) -> dict:
    """Every predeclared task/round stays in the denominator, including omissions."""
    plan = io.verify(root, execution=False)
    rows = []
    for task in plan["tasks"]:
        calls, fits, tokens = 0, 0, 0
        for index in range(plan["config"]["rounds"]):
            result = io.read_round(root, task, index)
            selected = result.get("selected") if result else None
            trial = result.get("trial") if result else None
            directory = io.round_path(root, task, index)
            records = _cache_records(
                directory / "calls",
                content_hash([plan["artifact_sha256"], task, index]),
            )
            cost = _cost(records)
            imported = (
                plan.get("parameter_free_recovery", {})
                .get("entries", {})
                .get(task["task_id"])
                if index == 0
                else None
            )
            if imported is not None:
                if records:
                    raise ValueError("imported construction cannot issue fresh calls")
                cost = imported["cost"]
            if (
                result
                and result.get("cost", {}).get("physical_requests", 0)
                != cost["physical_requests"]
            ):
                raise ValueError("round provider accounting differs from request cache")
            started_path = directory / "fit" / "started.json"
            fit_started = started_path.exists()
            if fit_started:
                marker = public._read(started_path)
                frozen_fit = public._read(directory / "fit" / "freeze.json")
                if marker["identity"] != frozen_fit["identity"]:
                    raise ValueError("fit accounting marker differs from frozen fit")
            calls += cost.get("physical_requests", 0)
            tokens += cost.get("observed_tokens", 0)
            fits += (
                imported["historical_fit_attempts"] if imported else int(fit_started)
            )
            metric = selected["fit"] if selected else {}
            rows.append(
                {
                    "task_id": task["task_id"],
                    "cell": task["cell"],
                    "seed": task["seed"],
                    "arm": task["arm"],
                    "round": index
                    + plan.get("continuation", {}).get("source_round", 0),
                    "phase_round": index,
                    "status": result["status"] if result else "missing",
                    "trial_train_nmse": (trial or {})
                    .get("fit", {})
                    .get("training", {})
                    .get("normalized_mse"),
                    "trial_validation_nmse": (trial or {})
                    .get("fit", {})
                    .get("validation", {})
                    .get("normalized_mse"),
                    "retained_train_nmse": metric.get("training", {}).get(
                        "normalized_mse"
                    ),
                    "retained_validation_nmse": metric.get("validation", {}).get(
                        "normalized_mse"
                    ),
                    "retained_train_per_target_nmse": metric.get("training", {}).get(
                        "per_target_normalized_mse", {}
                    ),
                    "retained_validation_per_target_nmse": metric.get(
                        "validation", {}
                    ).get("per_target_normalized_mse", {}),
                    "trial_train_per_target_nmse": (trial or {})
                    .get("fit", {})
                    .get("training", {})
                    .get("per_target_normalized_mse", {}),
                    "trial_validation_per_target_nmse": (trial or {})
                    .get("fit", {})
                    .get("validation", {})
                    .get("per_target_normalized_mse", {}),
                    "native_converged": metric.get("native_optimizer_converged"),
                    "budget_exhausted": metric.get("budget_exhausted"),
                    "all_graph_requirements_certified": selected["certificate"][
                        "all_public_graph_requirements_certified"
                    ]
                    if selected
                    else None,
                    "latent_states": len(selected["certificate"]["latent_states"])
                    if selected
                    else None,
                    "selected_origin_round": selected["origin_round"]
                    if selected
                    else None,
                    "cumulative_requests": calls,
                    "cumulative_tokens": tokens,
                    "cumulative_fit_attempts": fits,
                    "worker_started": (directory / "worker_started.json").exists(),
                    **(
                        {
                            "source_result_sha256": imported["result_sha256"],
                            "execution_recovery": (result or {}).get(
                                "execution_recovery"
                            ),
                            "fixed_model_evaluations": int(
                                fit_started and imported["eligible"]
                            ),
                            "construction_usage_imported": True,
                        }
                        if imported
                        else {}
                    ),
                    "provider_reserved_tokens": cost["budget_charge"],
                    "provider_calls_with_unknown_usage": cost[
                        "requests_with_unknown_usage"
                    ],
                    "shared_round_zero_cost": task["arm"] == "refit_only",
                    "result_sha256": result["artifact_sha256"] if result else None,
                    "proposal_status": result.get("proposal_status")
                    if result
                    else None,
                    "fit_trigger": result.get("fit_trigger") if result else None,
                    "citation_status": (result.get("citation_audit") or {}).get(
                        "status"
                    )
                    if result
                    else None,
                }
            )
            if plan["protocol"] in io.PARAMETER_PROTOCOLS:
                certificate = (selected or {}).get("certificate", {})
                states = {
                    p["status"]
                    for p in certificate.get("mechanisms", {}).get("predicates", [])
                }
                rows[-1].update(
                    arm_label="prediction_only"
                    if task["arm"] == "no_spec"
                    else task["arm"],
                    graph_check_status=(
                        "failed"
                        if "failed" in states
                        else "unresolved"
                        if "ambiguous" in states
                        else "certified"
                        if certificate.get("all_public_graph_requirements_certified")
                        else "unavailable"
                        if selected is None
                        else "uncertified"
                    ),
                )
            if plan["protocol"] in {io.REVISION_PROTOCOL, *io.SHARED_PROTOCOLS}:
                proposal_path = directory / "proposal.json"
                proposal = sealed_read(proposal_path) if proposal_path.exists() else {}
                attempts = proposal.get("attempts", [])
                feedback = (attempts[-1].get("feedback") or {}) if attempts else {}
                failed = proposal.get("status") == "revision_failed"
                provenance = (proposal.get("decision") or {}).get("provenance", {})
                rows[-1].update(
                    terminal_revision_code=(
                        feedback.get("code", "delivery_or_budget") if failed else None
                    ),
                    terminal_revision_message=(
                        proposal.get("error") or feedback.get("message")
                    )
                    if failed
                    else None,
                    selected_new_trial=result.get("selected_new_trial", False)
                    if result
                    else False,
                    trial_size=review_revision_v5.model_size(trial["bundle"])
                    if trial
                    else None,
                    retained_size=review_revision_v5.model_size(selected["bundle"])
                    if selected
                    else None,
                    unused_new_declarations_removed=provenance.get(
                        "unused_new_declarations_removed", []
                    ),
                )
            if plan["protocol"] in io.SHARED_PROTOCOLS:
                from autoformalism.rebuttal.shared_process_pilot import model_evidence

                rows[-1].update(
                    process_guidance=task["process_guidance"],
                    arm_label=task["arm"] + "/shared_" + task["process_guidance"],
                    shared_process_evidence=model_evidence(selected),
                    trial_shared_process_evidence=model_evidence(trial),
                )
    value = {
        "plan_sha256": plan["artifact_sha256"],
        "planned_rounds": len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "rows": rows,
        "test_metrics_used": False,
        "fitter_research_reopened": False,
        "continuation": plan.get("continuation"),
        "proposal_status_counts": dict(
            Counter(
                r["proposal_status"] for r in rows if r["proposal_status"] is not None
            )
        ),
        "fallback_fits": sum(r["fit_trigger"] == "incumbent_fallback" for r in rows),
    }
    public._write(root / "summary.json", value)
    if plan["protocol"] in {io.REVISION_PROTOCOL, *io.SHARED_PROTOCOLS}:
        public._write(
            root / "revision_diagnostics.json",
            {
                "plan_sha256": plan["artifact_sha256"],
                "terminal_errors": [
                    {"code": code, "message": message, "visits": count}
                    for (code, message), count in sorted(
                        Counter(
                            (
                                r["terminal_revision_code"],
                                r["terminal_revision_message"] or "",
                            )
                            for r in rows
                            if r["proposal_status"] == "revision_failed"
                        ).items()
                    )
                ],
                "accepted_revision_visits": sum(
                    r["proposal_status"] == "committed" for r in rows
                ),
                "accepted_revision_trials_retained": sum(
                    r["proposal_status"] == "committed" and r["selected_new_trial"]
                    for r in rows
                ),
                "trial_visits_above_original_size_references": sum(
                    bool((r["trial_size"] or {}).get("exceeded_references"))
                    for r in rows
                ),
                "test_data_opened": False,
            },
        )
    with (root / "rounds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Internal deadline campaign",
        "",
        f"Planned round rows: {len(rows)}. Status counts: {value['status_counts']}.",
        "",
        (
            "Retained models use validation selection; parameter fitting "
            "and proposer evidence use training only."
        ),
        (
            "Scores describe observed outputs. Finite replay and "
            "optimizer stopping do not certify scientific correctness."
        ),
        (
            "Refit-only shares full round zero. Its shared "
            "construction/fit cost is excluded from incremental counts."
        ),
        (
            "Few-round curves measure budgeted progress, not convergence."
            " Missing rounds remain missing."
        ),
        "",
        (
            "| Cell | Seed | Arm | Round | Status | Retained train | "
            "Retained validation | Graph certified |"
        ),
        "| --- | ---: | --- | ---: | --- | ---: | ---: | --- |",
    ]
    if plan.get("continuation"):
        lines[4:4] = [
            f"Imported round {plan['continuation']['source_round']} is an anchor, "
            "not a new construction or fit. Costs below count this phase only.",
            f"Proposal outcomes: {value['proposal_status_counts']}; "
            f"incumbent fallback fits: {value['fallback_fits']}.",
            "Citation warnings remain visible in rounds.csv; they do not certify "
            "scientific claims or bypass equation/public-requirement checks.",
            "Protocol changed after the anchor; this is continued development, "
            "not one unchanged-protocol convergence experiment.",
            "",
        ]
    if plan["protocol"] in io.PARAMETER_PROTOCOLS:
        lines = [line.replace("Graph certified |", "Graph check |") for line in lines]
        lines[4:4] = [
            "Prediction-only (internal ID no_spec) omits scientific requirements "
            "and their enforcement; evaluate task satisfaction alongside NMSE.",
            "Unresolved graph checks are distinct from failed predicates. "
            "Certificates are unchanged.",
            "",
        ]
    if plan["protocol"] == io.REVISION_PROTOCOL:
        lines[4:4] = [
            "Revision size is advisory; original construction references "
            "are not acceptance gates. "
            "Actual counts are in summary.json/rounds.csv; rejection details "
            "are in revision_diagnostics.json.",
            "Per-reply limits, call/fit budgets and scientific/ablation checks "
            "remain enforced.",
            "",
        ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                str(r[k])
                for k in (
                    "cell",
                    "seed",
                    "arm_label"
                    if plan["protocol"] in io.PARAMETER_PROTOCOLS
                    else "arm",
                    "round",
                    "status",
                    "retained_train_nmse",
                    "retained_validation_nmse",
                    "graph_check_status"
                    if plan["protocol"] in io.PARAMETER_PROTOCOLS
                    else "all_graph_requirements_certified",
                )
            )
            + " |"
        )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    if plan["protocol"] == io.SHARED_PROTOCOL:
        from autoformalism.rebuttal.shared_process_pilot import write_report

        write_report(root, value)
    if plan["protocol"] == io.INTEGRATION_PROTOCOL:
        from autoformalism.rebuttal.shared_process_integration import write_report

        write_report(root, value)
    return value


def export(root: Path, *, allow_partial: bool = False) -> dict:
    """Seal all round selections before any test access; prohibit further search."""
    with public._lock(root):
        plan = io.verify(root)
        receipt_path = root / "evaluation_freeze.json"
        if receipt_path.exists():
            receipt = sealed_read(receipt_path)
            if (
                hashlib.sha256((root / "subjects.jsonl").read_bytes()).hexdigest()
                != receipt["subjects_sha256"]
            ):
                raise ValueError("exported subjects differ from evaluation freeze")
            return receipt
        summary = report(root)
        missing = summary["status_counts"].get("missing", 0)
        if missing and not allow_partial:
            raise ValueError(
                "unfinished rounds: explicit --allow-partial required to "
                "freeze at the deadline"
            )
        subjects = []
        for task in plan["tasks"]:
            for index in range(plan["config"]["rounds"]):
                result = io.read_round(root, task, index)
                if not result or result["selected"] is None:
                    continue
                selected = result["selected"]
                model, _, _ = public._lower(
                    PublicFitRequest.model_validate(selected["request"])
                )
                candidate = model.validated.candidate
                subject = FrozenEvaluationSubject.model_validate(
                    {
                        "subject_id": f"{task['task_id']}_round{index}",
                        "method": f"deadline_{task['arm']}",
                        "benchmark_id": task["cell"],
                        "tier": plan["cells"][task["cell"]]["target_contract"]["tier"],
                        "repetition": task["seed"],
                        "private_metrics_opened_after_freeze": False,
                        "source_provenance": {
                            "adapter": "direct_candidate",
                            "request_id": f"{task['task_id']}/{index}",
                            "source_path": str(
                                io.round_path(root, task, index) / "result.json"
                            ),
                            "source_sha256": result["artifact_sha256"],
                            "candidate_sha256": hashlib.sha256(
                                candidate.model_dump_json().encode()
                            ).hexdigest(),
                        },
                        "candidate": candidate.model_dump(mode="json"),
                        "parameterization": {
                            "status": "available",
                            "global_parameters": selected["fit"]["parameters"],
                        },
                        "validation_context": model.validated.context.model_dump(
                            mode="json"
                        ),
                        "target_prediction": {
                            "status": "missing",
                            "message": "No test opened before global campaign freeze",
                        },
                    }
                )
                subjects.append(subject.model_dump_json())
        payload = "".join(s + "\n" for s in subjects).encode()
        (root / "subjects.jsonl").write_bytes(payload)
        return sealed_write(
            receipt_path,
            {
                "plan_sha256": plan["artifact_sha256"],
                "subjects_sha256": hashlib.sha256(payload).hexdigest(),
                "subject_count": len(subjects),
                "planned_round_count": len(summary["rows"]),
                "round_hashes": {
                    f"{r['task_id']}/{r['round']}": r["result_sha256"]
                    for r in summary["rows"]
                },
                "missing_round_count": missing,
                "allow_partial": allow_partial,
                "test_data_opened": False,
                "further_search_permitted": False,
            },
        )


def evaluate(root: Path, public_root: Path, shard: int, shards: int) -> dict:
    """Use the existing free-rollout/intervention endpoint, with no parameter refit."""
    plan = io.verify(root, execution=False)
    receipt = sealed_read(root / "evaluation_freeze.json")
    if plan["source_sha256"] != public._source_identity():
        raise ValueError("evaluation source differs from frozen campaign")
    with public._lock(root / "evaluation-runtime"):
        execution = sealed_write(
            root / "evaluation-runtime" / "identity.json",
            {
                "source_sha256": public._source_identity(),
                "runtime": public._runtime(),
                "solver": "Radau",
                "rtol": 1e-7,
                "atol": 1e-9,
                "maximum_trajectory_seconds": 300,
                "refitting": False,
            },
        )
    raw = (root / "subjects.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["subjects_sha256"]:
        raise ValueError("subjects changed after freeze")
    if not 0 <= shard < shards:
        raise ValueError("invalid evaluation shard")
    for cell, value in plan["cells"].items():
        for filename, digest in value["assets"].items():
            if (
                hashlib.sha256(
                    (public_root / "phase_b_v1" / cell / filename).read_bytes()
                ).hexdigest()
                != digest
            ):
                raise ValueError(
                    "evaluation public assets differ from development campaign"
                )
    loader = BenchmarkLoader(BenchmarkRegistry())
    rows = []
    for line in raw.decode().splitlines()[shard::shards]:
        subject = FrozenEvaluationSubject.model_validate_json(line)
        directory = root / "evaluation" / subject.subject_id
        binding = evaluation_binding(receipt, subject, execution)
        with public._lock(directory):
            if (directory / "result.json").exists():
                rows.append(
                    checked_evaluation(
                        sealed_read(directory / "result.json"), subject, binding
                    )
                )
                continue
            # No new replay allowance if a worker died after opening the held-out case.
            if (directory / "started.json").exists():
                marker = sealed_read(directory / "started.json")
                if any(marker.get(k) != v for k, v in binding.items()):
                    raise ValueError(
                        "evaluation start belongs to another frozen subject"
                    )
                rows.append(
                    sealed_write(
                        directory / "result.json",
                        {
                            **binding,
                            "status": "interrupted",
                            "nmse": None,
                        },
                    )
                )
                continue
            sealed_write(directory / "started.json", binding)
            config = DataConfig(
                root=public_root, benchmark_id=subject.benchmark_id, tier=subject.tier
            )
            development = loader.load_development(config)
            test = loader.load_test(
                config,
                access=FrozenTestAccess(
                    benchmark_id=subject.benchmark_id,
                    tier=subject.tier,
                    selection_hash=receipt["subjects_sha256"],
                ),
            )
            updated = evaluate_subject_on_test(
                subject,
                training_split=development.train,
                test_split=test,
                fit_config=FitConfig(
                    integration_method="Radau",
                    allow_derivative_regression=False,
                    relative_tolerance=1e-7,
                    absolute_tolerance=1e-9,
                    maximum_wall_time_seconds=300,
                ),
            )
            rows.append(
                sealed_write(
                    directory / "result.json",
                    {
                        **binding,
                        "status": updated.target_prediction.status,
                        "nmse": updated.target_prediction.normalized_mse,
                        "subject": updated.model_dump(mode="json"),
                        "parameter_refit": False,
                    },
                )
            )
    return {"shard": shard, "shards": shards, "evaluated": len(rows)}


def evaluation_binding(receipt, subject, execution) -> dict:
    """Bind each prediction to the exact globally frozen model and evaluator."""
    return {
        "subject_id": subject.subject_id,
        "freeze_sha256": receipt["artifact_sha256"],
        "frozen_subject_sha256": public.content_sha256(subject),
        "evaluator_sha256": execution["artifact_sha256"],
    }


def checked_evaluation(value, subject, binding) -> dict:
    """Reject validly sealed but unrelated cached predictions or parameters."""
    if any(value.get(k) != v for k, v in binding.items()):
        raise ValueError("evaluation belongs to another frozen subject or evaluator")
    if "subject" in value:
        updated = FrozenEvaluationSubject.model_validate(value["subject"])
        if (
            not updated.private_metrics_opened_after_freeze
            or updated.target_prediction.status not in {"available", "failed"}
        ):
            raise ValueError("cached evaluation has no completed held-out endpoint")
        mutable = {
            "target_prediction",
            "interventions",
            "hidden_mechanisms",
            "private_metrics_opened_after_freeze",
        }
        original = subject.model_dump(mode="json", exclude=mutable)
        if updated.model_dump(mode="json", exclude=mutable) != original:
            raise ValueError("evaluation changed the frozen model or parameters")
        if (
            value["status"] != updated.target_prediction.status
            or value["nmse"] != updated.target_prediction.normalized_mse
        ):
            raise ValueError("evaluation summary disagrees with its endpoint")
    elif value.get("status") != "interrupted" or value.get("nmse") is not None:
        raise ValueError("evaluation without a subject must be interrupted")
    return value


def evaluation_report(root: Path) -> dict:
    """Join held-out endpoints to all planned rows, including missing models."""
    plan = io.verify(root, execution=False)
    receipt = sealed_read(root / "evaluation_freeze.json")
    if (
        hashlib.sha256((root / "subjects.jsonl").read_bytes()).hexdigest()
        != receipt["subjects_sha256"]
    ):
        raise ValueError("subjects changed after freeze")
    from autoformalism.rebuttal.final_evaluation import evaluate_frozen_subject
    from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec

    frozen_subjects = {
        item.subject_id: item
        for item in (
            FrozenEvaluationSubject.model_validate_json(line)
            for line in (root / "subjects.jsonl").read_text().splitlines()
        )
    }
    rows = []
    for task in plan["tasks"]:
        for index in range(plan["config"]["rounds"]):
            subject_id = f"{task['task_id']}_round{index}"
            source = io.read_round(root, task, index)
            if (source["artifact_sha256"] if source else None) != receipt[
                "round_hashes"
            ][f"{task['task_id']}/{index}"]:
                raise ValueError("round changed after evaluation freeze")
            path = root / "evaluation" / subject_id / "result.json"
            value = sealed_read(path) if path.exists() else None
            final = None
            if value:
                execution = sealed_read(root / "evaluation-runtime" / "identity.json")
                frozen_subject = frozen_subjects[subject_id]
                checked_evaluation(
                    value,
                    frozen_subject,
                    evaluation_binding(receipt, frozen_subject, execution),
                )
            if value and "subject" in value:
                subject = FrozenEvaluationSubject.model_validate(value["subject"])
                spec = MechanismEvaluationSpec.model_validate(
                    plan["cells"][task["cell"]]["mechanism_spec"]
                )
                final = evaluate_frozen_subject(subject, spec).model_dump(mode="json")
            rows.append(
                {
                    "task_id": task["task_id"],
                    "cell": task["cell"],
                    "arm": task["arm"],
                    "seed": task["seed"],
                    "round": index,
                    "status": value["status"]
                    if value
                    else "model_unavailable"
                    if not source or not source["selected"]
                    else "evaluation_missing",
                    "test_nmse": value.get("nmse") if value else None,
                    "common_endpoints": final,
                }
            )
    result = {
        "freeze_sha256": receipt["artifact_sha256"],
        "planned_rounds": len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "rows": rows,
        "private_scores_for_reporting_only": True,
    }
    public._write(root / "evaluation_summary.json", result)
    lines = [
        "# Held-out review campaign",
        "",
        f"Planned rows: {len(rows)}. Status counts: {result['status_counts']}.",
        "",
        "No refitting, target resets, or selection using held-out scores.",
        "",
        "| Cell | Seed | Arm | Round | Status | Test NMSE |",
        "| --- | ---: | --- | ---: | --- | ---: |",
    ]
    lines += [
        "| "
        + " | ".join(
            str(r[k]) for k in ("cell", "seed", "arm", "round", "status", "test_nmse")
        )
        + " |"
        for r in rows
    ]
    (root / "EVALUATION.md").write_text("\n".join(lines) + "\n")
    return result
