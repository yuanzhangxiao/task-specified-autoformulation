"""Audit evidence attached to incorrect certified hybrid-judge answers."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.rebuttal.hybrid_labels import (
    ExpectedPairPreference,
    ExpectedVerdict,
    HybridCalibrationLabels,
)
from autoformalism.search.controller import _structural_hash

SCHEMA_VERSION = "hybrid-judge-evidence-audit-1"


def _normalize_absolute(
    items: list[dict[str, Any]], baseline_position: str
) -> dict[tuple[str, str], tuple[dict[str, str], dict[str, str]]]:
    output = {}
    for item in items:
        key = (str(item["criterion"]), str(item["subject_id"]))
        left = {
            "verdict": str(item["candidate_a"]["verdict"]),
            "evidence": str(item["candidate_a"]["evidence"]),
        }
        right = {
            "verdict": str(item["candidate_b"]["verdict"]),
            "evidence": str(item["candidate_b"]["evidence"]),
        }
        output[key] = (
            (left, right) if baseline_position == "A" else (right, left)
        )
    return output


def _normalize_comparative(
    items: list[dict[str, Any]], baseline_position: str
) -> dict[str, dict[str, str]]:
    output = {}
    for item in items:
        verdict = str(item["verdict"])
        if verdict in {"candidate_a", "candidate_b"}:
            position = "A" if verdict == "candidate_a" else "B"
            verdict = "baseline" if position == baseline_position else "mutated"
        output[str(item["criterion"])] = {
            "verdict": verdict,
            "evidence": str(item["evidence"]),
        }
    return output


def audit_evidence(
    rows: list[dict[str, object]],
    labels: dict[str, HybridCalibrationLabels],
    pair_metadata: dict[str, tuple[str, str]],
) -> dict[str, object]:
    """Return certified-question performance and every incorrect rationale."""
    errors: list[dict[str, object]] = []
    totals: Counter[tuple[str, str, str]] = Counter()
    correct: Counter[tuple[str, str, str]] = Counter()
    seen_keys = set()
    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id not in labels or pair_id not in pair_metadata:
            raise ValueError(f"score row has unknown pair: {pair_id}")
        key = (
            pair_id,
            str(row["judge_model"]),
            int(row["repetition"]),
            str(row["order"]),
        )
        if key in seen_keys:
            raise ValueError(f"duplicate score outcome key: {key}")
        seen_keys.add(key)
        mutation_type, structure_id = pair_metadata[pair_id]
        if str(row["mutation_type"]) != mutation_type:
            raise ValueError(f"mutation mismatch for pair {pair_id}")
        position = str(row["baseline_position"])
        absolute = _normalize_absolute(
            json.loads(str(row["absolute_assessments"])), position
        )
        comparative = _normalize_comparative(
            json.loads(str(row["comparative_assessments"])), position
        )
        gold = labels[pair_id]
        for item in gold.absolute_labels:
            if item.label_source == "deterministic_runtime":
                continue
            label_key = (item.criterion.value, item.subject_id)
            if label_key not in absolute:
                raise ValueError(
                    f"missing absolute assessment {label_key} for pair {pair_id}"
                )
            baseline, mutated = absolute[label_key]
            expected = {
                "baseline": item.baseline.value,
                "mutated": item.mutated.value,
            }
            actual = {
                "baseline": baseline["verdict"],
                "mutated": mutated["verdict"],
            }
            scored_sides = [
                side
                for side in ("baseline", "mutated")
                if expected[side] != ExpectedVerdict.UNLABELED.value
            ]
            group = ("absolute", item.criterion.value, mutation_type)
            for side in scored_sides:
                totals[group] += 1
                correct[group] += actual[side] == expected[side]
            incorrect_sides = [
                side for side in scored_sides if actual[side] != expected[side]
            ]
            if incorrect_sides:
                errors.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "absolute",
                        "pair_id": pair_id,
                        "structure_id": structure_id,
                        "mutation_type": mutation_type,
                        "judge_model": str(row["judge_model"]),
                        "repetition": int(row["repetition"]),
                        "order": str(row["order"]),
                        "baseline_position": position,
                        "criterion": item.criterion.value,
                        "subject_id": item.subject_id,
                        "expected_baseline": expected["baseline"],
                        "actual_baseline": actual["baseline"],
                        "baseline_evidence": baseline["evidence"],
                        "expected_mutated": expected["mutated"],
                        "actual_mutated": actual["mutated"],
                        "mutated_evidence": mutated["evidence"],
                        "incorrect_sides": incorrect_sides,
                        "label_source": item.label_source,
                        "gold_rationale": item.rationale,
                    }
                )
        for item in gold.comparative_labels:
            if item.preference is ExpectedPairPreference.UNLABELED:
                continue
            criterion = item.criterion.value
            if criterion not in comparative:
                raise ValueError(
                    f"missing comparative assessment {criterion} for pair {pair_id}"
                )
            assessment = comparative[criterion]
            group = ("comparative", criterion, mutation_type)
            totals[group] += 1
            is_correct = assessment["verdict"] == item.preference.value
            correct[group] += is_correct
            if not is_correct:
                errors.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "comparative",
                        "pair_id": pair_id,
                        "structure_id": structure_id,
                        "mutation_type": mutation_type,
                        "judge_model": str(row["judge_model"]),
                        "repetition": int(row["repetition"]),
                        "order": str(row["order"]),
                        "baseline_position": position,
                        "criterion": criterion,
                        "expected": item.preference.value,
                        "actual": assessment["verdict"],
                        "evidence": assessment["evidence"],
                        "label_source": item.label_source,
                        "gold_rationale": item.rationale,
                    }
                )
    performance = []
    for group, total in sorted(totals.items()):
        kind, criterion, mutation = group
        performance.append(
            {
                "kind": kind,
                "criterion": criterion,
                "mutation_type": mutation,
                "correct": correct[group],
                "total": total,
                "accuracy": correct[group] / total,
                "error_count": total - correct[group],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "frozen_calls_no_new_llm_requests",
        "score_row_count": len(rows),
        "certified_performance": performance,
        "error_count": len(errors),
        "errors": errors,
    }


def _error_pattern(error: dict[str, object]) -> str:
    if error["kind"] == "absolute":
        return (
            f"baseline={error['actual_baseline']};"
            f"mutated={error['actual_mutated']}"
        )
    return f"preference={error['actual']}"


def render_markdown(payload: dict[str, object], *, examples_per_group: int = 4) -> str:
    """Render aggregate error patterns plus representative stored rationales."""
    lines = [
        "# Hybrid judge evidence audit",
        "",
        "This report uses frozen structured responses and certified mutation "
        "contracts. It makes no LLM calls and does not infer from hidden reasoning.",
        "",
        "## Certified-question performance",
        "",
        "| Kind | Criterion | Mutation | Correct | Total | Accuracy | Errors |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in payload["certified_performance"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(item["kind"]),
                    str(item["criterion"]),
                    str(item["mutation_type"]),
                    str(item["correct"]),
                    str(item["total"]),
                    f"{float(item['accuracy']):.3f}",
                    str(item["error_count"]),
                )
            )
            + " |"
        )
    errors = payload["errors"]
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for error in errors:
        grouped[
            (
                str(error["kind"]),
                str(error["criterion"]),
                str(error["mutation_type"]),
            )
        ].append(error)
    lines.extend(["", "## Incorrect-answer patterns"])
    for group, group_errors in sorted(grouped.items()):
        kind, criterion, mutation = group
        patterns = Counter(_error_pattern(item) for item in group_errors)
        orders = Counter(str(item["order"]) for item in group_errors)
        lines.extend(
            [
                "",
                f"### {kind}: {criterion} / {mutation}",
                "",
                f"Errors: {len(group_errors)}. Orders: "
                + ", ".join(f"{key}={value}" for key, value in sorted(orders.items()))
                + ".",
                "",
                "Verdict patterns: "
                + ", ".join(
                    f"`{key}` x {value}" for key, value in patterns.most_common()
                )
                + ".",
            ]
        )
        representatives = []
        seen = set()
        for error in group_errors:
            identity = (_error_pattern(error), error["order"], error["pair_id"])
            if identity in seen:
                continue
            seen.add(identity)
            representatives.append(error)
            if len(representatives) == examples_per_group:
                break
        for index, error in enumerate(representatives, start=1):
            lines.extend(
                [
                    "",
                    f"Example {index}: `{error['pair_id']}`, repetition "
                    f"{error['repetition']}, `{error['order']}`.",
                    "",
                ]
            )
            if error["kind"] == "absolute":
                lines.extend(
                    [
                        f"- Expected baseline/actual: `{error['expected_baseline']}` / "
                        f"`{error['actual_baseline']}`",
                        f"- Baseline evidence: {error['baseline_evidence']}",
                        f"- Expected mutated/actual: `{error['expected_mutated']}` / "
                        f"`{error['actual_mutated']}`",
                        f"- Mutated evidence: {error['mutated_evidence']}",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"- Expected/actual: `{error['expected']}` / "
                        f"`{error['actual']}`",
                        f"- Evidence: {error['evidence']}",
                    ]
                )
            lines.append(f"- Certified rationale: {error['gold_rationale']}")
    return "\n".join(lines) + "\n"


def _load_pair_metadata(path: Path) -> dict[str, tuple[str, str]]:
    pairs = tuple(
        AdversarialPair.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return {
        pair.pair_id: (pair.mutation_type, _structural_hash(pair.valid_candidate)[:12])
        for pair in pairs
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    with args.scores.open(encoding="utf-8", newline="") as handle:
        rows: list[dict[str, object]] = list(csv.DictReader(handle))
    labels = {
        item.pair_id: item
        for item in (
            HybridCalibrationLabels.model_validate_json(line)
            for line in args.labels.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    payload = audit_evidence(rows, labels, _load_pair_metadata(args.pairs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(item, sort_keys=True) + "\n" for item in payload["errors"]
        ),
        encoding="utf-8",
    )
    summary = args.summary or args.output.with_suffix(".md")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"audited {payload['score_row_count']} score rows; wrote "
        f"{payload['error_count']} certified errors to {args.output} and {summary}"
    )


if __name__ == "__main__":
    main()
