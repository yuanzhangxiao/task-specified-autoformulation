"""Offline, evaluator-only coverage and accuracy for the portable judge suite."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from autoformalism.judging import HybridScoringConfig
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal import judge_jetstream_calibration as campaign
from autoformalism.rebuttal.hybrid_labels import HybridCalibrationLabels
from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts.analyze_hybrid_consensus_validation import evaluate_validation
from scripts.analyze_hybrid_symmetric_aggregation import analyze


def _accuracy(symmetric: dict, labels: dict) -> dict:
    """Count missing/indeterminate known predicates, including absent paired trials."""
    trials = {(t["pair_id"], t["repetition"]): t for t in symmetric["trials"]}
    outcomes = Counter()
    for pair_id, label in labels.items():
        for repetition in range(5):
            trial = trials.get((pair_id, repetition), {})
            observed = {
                (item["criterion"], item["subject_id"]): item
                for item in trial.get("consensus_absolute_assessments", [])
            }
            for unit in label.absolute_labels:
                if not unit.label_source.startswith("mutation_contract:"):
                    continue
                item = observed.get((unit.criterion.value, unit.subject_id), {})
                for side, gold in (
                    ("candidate_a", unit.baseline),
                    ("candidate_b", unit.mutated),
                ):
                    if gold.value == "unlabeled":
                        continue
                    value = item.get(side, {}).get("verdict", "missing")
                    outcomes["expected_labeled_units"] += 1
                    outcomes["correct_units"] += value == gold.value
                    outcomes["missing_units"] += value == "missing"
                    outcomes["observed_indeterminate_units"] += value == "indeterminate"
                    if gold.value == "fail":
                        outcomes["known_defect_units"] += 1
                        outcomes["known_defects_answered_pass"] += value == "pass"
                        outcomes["known_defects_detected"] += value == "fail"
                        outcomes["known_defects_missing"] += value == "missing"
                        outcomes["known_defects_indeterminate"] += (
                            value == "indeterminate"
                        )
    return dict(outcomes)


def _evaluation(
    rows: list, failures: list, bundle: dict, scoring: HybridScoringConfig
) -> dict:
    labels = {
        r["pair_id"]: HybridCalibrationLabels.model_validate(r)
        for r in bundle["labels"]
    }
    symmetric = analyze(rows, failures, labels, config=scoring)
    gates = evaluate_validation(
        symmetric,
        labels,
        response_success=len(rows) / 140,
        tie_threshold=scoring.tie_threshold,
        gates=bundle["source_config"]["validation_gate"],
    )
    return {
        "gate_assessment": gates,
        "aggregation": symmetric,
        "absolute_coverage": _accuracy(symmetric, labels),
    }


def _cost(root: Path) -> dict:
    calls = []
    for marker in sorted(
        (root / "orientations").glob("pair-*/seed-*/*/calls/call-*/started.json")
    ):
        path = marker.with_name("result.json")
        result = json.loads(path.read_text()) if path.exists() else {}
        response = result.get("response")
        response = response if isinstance(response, dict) else {}
        usage = response.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        total = usage.get("total_tokens")
        valid = isinstance(total, int) and not isinstance(total, bool) and total >= 0
        calls.append(
            {
                "call": str(marker.parent.relative_to(root)),
                "status": result.get("status", "interrupted"),
                "http_status": result.get("http_status"),
                "seconds": result.get("seconds"),
                "total_tokens": total if valid else None,
            }
        )
    return {
        "physical_requests_started": len(calls),
        "observed_total_tokens": sum(c["total_tokens"] or 0 for c in calls),
        "usage_missing_events": sum(c["total_tokens"] is None for c in calls),
        "usage_complete": bool(calls)
        and all(c["total_tokens"] is not None for c in calls),
        "scope": (
            "All started HTTP attempts, including failed parsing and interruptions; "
            "unknown usage is not zero."
        ),
        "calls": calls,
    }


def report(root: Path) -> dict:
    """Report all planned trials; never treat missing responses as a passed gate."""
    plan, bundle = campaign.verify(root)
    rows, failures = [], []
    statuses = Counter()
    for name, raw, repetition, order in campaign.units(bundle):
        work = root / "orientations" / name
        result = (
            sealed_read(work / "result.json")
            if (work / "result.json").exists()
            else None
        )
        if result and result["identity"] != {
            "plan": plan["artifact_sha256"],
            "unit": name,
        }:
            raise ValueError("orientation belongs to another plan")
        status = (
            result["status"]
            if result
            else "started_without_result"
            if (work / "started.json").exists()
            else "pending"
        )
        statuses[status] += 1
        if status == "reviewed":
            rows.append(result["row"])
        else:
            failures.append(
                {
                    "pair_id": raw["pair_id"],
                    "judge_model": campaign.MODEL,
                    "repetition": repetition,
                    "order": order,
                    "status": status,
                }
            )
    current = _evaluation(
        rows,
        failures,
        bundle,
        HybridScoringConfig(**plan["scientific_protocol"]["scoring"]),
    )
    historical = _evaluation(
        bundle["historical_rows"],
        bundle["historical_failures"],
        bundle,
        HybridScoringConfig(**bundle["source_config"]["protocol"]["scoring"]),
    )
    finished = not (statuses["pending"] or statuses["started_without_result"])
    value = {
        "protocol": campaign.PROTOCOL,
        "identity": plan["artifact_sha256"],
        "status": "complete" if finished else "incomplete",
        "status_counts": dict(statuses),
        "unique_pairs": 14,
        "planned_paired_trials": 70,
        "planned_orientations": 140,
        "known_case_gate_passed": current["gate_assessment"]["passed"]
        if finished
        else None,
        "fresh_holdout_calibration_established": False,
        "current": current,
        "historical_original_protocol": historical,
        "historical_comparison_limits": [
            "Known calibration cases have previously been inspected; "
            "this is not a fresh holdout.",
            "Current corrected outer-factor signs, blinded metadata and fixed "
            "indeterminate denominator differ from the historical original protocol.",
            "Managed model revision and setting forwarding are unverified; "
            "differences cannot be attributed to hardware alone.",
            "Unlabeled tradeoff winners never contribute to accuracy; scoped mutation "
            "labels do not establish broad scientific correctness.",
        ],
        "cost": _cost(root),
        "llm_calls_in_report": 0,
        "optimizer_calls": 0,
        "model_changes": 0,
        "automatic_followup": False,
        "test_data_opened": False,
    }
    atomic_json(root / "summary.json", value)
    lines = [
        "# Jetstream known-case judge revalidation",
        "",
        *value["historical_comparison_limits"],
        "",
        f"Status: {value['status']}. Orientation counts: {dict(statuses)}.",
        f"Known-case gates passed: {value['known_case_gate_passed']}.",
        f"HTTP attempts: {value['cost']['physical_requests_started']}; "
        f"observed tokens: {value['cost']['observed_total_tokens']}; "
        f"missing usage: {value['cost']['usage_missing_events']}.",
        "",
        "| Gate | Jetstream | Historical original | Required | Pass (Jetstream) |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, check in current["gate_assessment"]["checks"].items():
        old = historical["gate_assessment"]["checks"][name]
        lines.append(
            f"| {name} | {check['observed']} | {old['observed']} | "
            f"{check['threshold']} | {check['passed']} |"
        )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return value
