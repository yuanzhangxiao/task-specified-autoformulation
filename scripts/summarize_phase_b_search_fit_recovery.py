"""Summarize all frozen search fit-recovery tasks without opening test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autoformalism.rebuttal.search_fit_recovery import (
    canonical_plan_sha256,
    load_search_fit_recovery_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = load_search_fit_recovery_plan(args.config)
    rows: list[dict[str, Any]] = []
    case_summaries = []
    expected_hash = canonical_plan_sha256(plan)
    for index, case in enumerate(plan.cases):
        path = args.input_root / f"task_{index}.json"
        payload = _read_object(path)
        if payload.get("plan_sha256") != expected_hash:
            raise ValueError(f"task plan hash differs: {path}")
        task_rows = payload.get("results")
        if not isinstance(task_rows, list):
            raise ValueError(f"task results are absent: {path}")
        rows.extend(task_rows)
        successful_rounds = sorted(
            {
                int(row["round_index"])
                for row in task_rows
                if row.get("success") is True
            }
        )
        case_summaries.append(
            {
                "case_id": case.case_id,
                "mode": case.mode,
                "candidate_count": len(case.round_indices),
                "recovered_candidate_count": len(successful_rounds),
                "recovered_round_indices": successful_rounds,
            }
        )

    report = {
        "schema_version": "phase-b-search-fit-recovery-summary-1",
        "status": "complete",
        "development_only": True,
        "new_llm_calls": False,
        "test_data_opened": False,
        "plan_sha256": expected_hash,
        "attempted_fit_count": len(rows),
        "successful_fit_count": sum(row.get("success") is True for row in rows),
        "cases": case_summaries,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for item in case_summaries:
        print(
            f"{item['case_id']}: recovered="
            f"{item['recovered_candidate_count']}/{item['candidate_count']}"
        )


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing recovery result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"recovery result is not an object: {path}")
    return payload


if __name__ == "__main__":
    main()
