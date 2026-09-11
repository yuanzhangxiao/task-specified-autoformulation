#!/usr/bin/env python3
"""Export frozen public multiround candidates using Python 3.6+ standard library."""

# ACES system Python predates postponed annotations and newer typing syntax.
# ruff: noqa: UP045

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

PUBLIC = ("manifest.json", "proposer_prompt.txt", "train.csv", "validation.csv")


def digest(value: object) -> str:
    """Match the repository's canonical content hash."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def within(path: Path, root: Path) -> bool:
    """Check resolved containment without the Python 3.9 is_relative_to API."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def checked(root: Path, relative: str) -> Path:
    """Reject absolute paths, traversal and symlink escapes from the source."""
    path = root / relative
    if Path(relative).is_absolute() or not within(path, root):
        raise ValueError("artifact path escapes its root")
    return path


def export_cases(source: Path, output: Path) -> dict:
    """Copy every frozen parent and committed round, without selecting on metrics."""
    if within(source, output) or within(output, source):
        raise ValueError("source and export must be separate")
    plan = json.loads((source / "plan.json").read_text())
    accepted_plan_schemas = {
        "scientific-staged-multiround-feedback-plan-3",
        "scientific-staged-multiround-feedback-plan-4",
    }
    if (
        plan.get("schema_version") not in accepted_plan_schemas
        or plan.get("plan_sha256")
        != digest({k: v for k, v in plan.items() if k != "plan_sha256"})
        or plan.get("test_data_opened") is not False
        or plan.get("private_reference_opened") is not False
        or plan.get("public_asset_ledger_sha256") != digest(plan["public_asset_ledger"])
    ):
        raise ValueError("source multiround plan or public-data declaration differs")
    files = {}
    cases = []
    skipped_uncommitted_rounds = []

    def copy(relative: str, destination: str, expected: Optional[str] = None):
        payload = checked(source, relative).read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        if expected is not None and expected != sha:
            raise ValueError(f"source hash differs: {relative}")
        target = checked(output, destination)
        if target.exists() and target.read_bytes() != payload:
            raise ValueError("export differs; use a new output directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(payload)
        files[destination] = sha

    copy("plan.json", "provenance/plan.json")
    for task in plan["tasks"]:
        benchmark = task["benchmark_id"]
        for name in PUBLIC:
            relative = f"frozen/public/phase_b_v1/{benchmark}/{name}"
            copy(
                relative,
                relative[len("frozen/") :],
                plan["public_asset_ledger"][relative],
            )
        ordinal = len(cases)
        destination = f"candidates/case_{ordinal:03d}.json"
        copy(task["candidate_path"], destination, task["candidate_file_sha256"])
        parent = json.loads(checked(source, task["candidate_path"]).read_text())
        base = {
            "benchmark_id": benchmark,
            "tier": task["tier"],
            "source_task": task["task_id"],
        }
        cases.append(
            {
                **base,
                "name": f"seed{task['seed']}_parent",
                "candidate": destination,
                "source_round": 0,
            }
        )
        roots = [
            relative
            for relative in (f"results/{task['task_id']}", task["task_id"])
            if checked(source, relative + "/terminal.json").exists()
        ]
        if len(roots) != 1:
            raise ValueError("source must have exactly one terminal record per task")
        relative = roots[0]
        terminal = json.loads(checked(source, relative + "/terminal.json").read_text())
        if terminal.get("identity") != digest([plan["plan_sha256"], task]):
            raise ValueError("source terminal identity differs")
        result = terminal["result"]
        if (
            result.get("test_data_opened") is not False
            or result.get("private_reference_opened") is not False
        ):
            raise ValueError("source result opened nonpublic data")
        copy(
            relative + "/terminal.json", f"provenance/seed{task['seed']}_terminal.json"
        )
        for expected_index, record in enumerate(result.get("rounds", []), 1):
            round_index = record["round_index"]
            if round_index != expected_index or record[
                "parent_candidate_sha256"
            ] != digest(parent):
                raise ValueError("committed round parent chain differs")
            candidate_path = f"{relative}/round_{round_index:03d}/candidate.json"
            if not checked(source, candidate_path).is_file():
                failure_class = (record.get("fit") or {}).get("failure_class")
                if (
                    failure_class == "revision_contract"
                    and record.get("candidate_sha256") == digest(parent)
                ):
                    skipped_uncommitted_rounds.append(
                        {
                            "source_task": task["task_id"],
                            "source_round": round_index,
                            "failure_class": failure_class,
                        }
                    )
                    continue
                raise ValueError("committed round candidate is missing")
            candidate = json.loads(checked(source, candidate_path).read_text())
            if record["candidate_sha256"] != digest(candidate):
                raise ValueError("committed round candidate differs")
            destination = f"candidates/case_{len(cases):03d}.json"
            copy(candidate_path, destination)
            cases.append(
                {
                    **base,
                    "name": f"seed{task['seed']}_round{round_index}",
                    "candidate": destination,
                    "source_round": round_index,
                }
            )
            parent = candidate
    names = [c["name"] for c in cases]
    if len(set(names)) != len(names) or not cases:
        raise ValueError("case names must be nonempty and unique")
    manifest = {
        "protocol": "collocation-node-cases-1",
        "source_plan_sha256": plan["plan_sha256"],
        "source_plan_schema_version": plan["schema_version"],
        "selection": "all_frozen_parents_and_all_committed_rounds",
        "skipped_uncommitted_rounds": skipped_uncommitted_rounds,
        "cases": cases,
        "files": files,
        "llm_calls": 0,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    manifest["identity"] = digest(manifest)
    path = output / "bundle.json"
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError("existing bundle manifest differs")
    path.write_text(payload)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_cases(args.source_root, args.output)
    print(json.dumps({"cases": len(result["cases"]), "identity": result["identity"]}))


if __name__ == "__main__":
    main()
