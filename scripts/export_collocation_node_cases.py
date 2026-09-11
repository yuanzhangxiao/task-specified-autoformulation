#!/usr/bin/env python3
"""Export frozen public multiround candidates with Python's standard library only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PUBLIC = ("manifest.json", "proposer_prompt.txt", "train.csv", "validation.csv")


def digest(value: object) -> str:
    """Match the repository's canonical content hash."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checked(root: Path, relative: str) -> Path:
    """Reject absolute paths, traversal and symlink escapes from the source."""
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError("artifact path escapes its root")
    return path


def export_cases(source: Path, output: Path) -> dict:
    """Copy every frozen parent and committed round, without selecting on metrics."""
    if source.resolve().is_relative_to(
        output.resolve()
    ) or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("source and export must be separate")
    plan = json.loads((source / "plan.json").read_text())
    if (
        plan.get("schema_version") != "scientific-staged-multiround-feedback-plan-3"
        or plan.get("plan_sha256")
        != digest({k: v for k, v in plan.items() if k != "plan_sha256"})
        or plan.get("test_data_opened") is not False
        or plan.get("private_reference_opened") is not False
        or plan.get("public_asset_ledger_sha256") != digest(plan["public_asset_ledger"])
    ):
        raise ValueError("source multiround plan or public-data declaration differs")
    files: dict[str, str] = {}
    cases = []

    def copy(relative: str, destination: str, expected: str | None = None):
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
                relative.removeprefix("frozen/"),
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
        for index, record in enumerate(result.get("rounds", []), 1):
            if record["round_index"] != index or record[
                "parent_candidate_sha256"
            ] != digest(parent):
                raise ValueError("committed round parent chain differs")
            candidate_path = f"{relative}/round_{index:03d}/candidate.json"
            candidate = json.loads(checked(source, candidate_path).read_text())
            if record["candidate_sha256"] != digest(candidate):
                raise ValueError("committed round candidate differs")
            destination = f"candidates/case_{len(cases):03d}.json"
            copy(candidate_path, destination)
            cases.append(
                {
                    **base,
                    "name": f"seed{task['seed']}_round{index}",
                    "candidate": destination,
                    "source_round": index,
                }
            )
            parent = candidate
    names = [c["name"] for c in cases]
    if len(set(names)) != len(names) or not cases:
        raise ValueError("case names must be nonempty and unique")
    manifest = {
        "protocol": "collocation-node-cases-1",
        "source_plan_sha256": plan["plan_sha256"],
        "selection": "all_frozen_parents_and_all_committed_rounds",
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
