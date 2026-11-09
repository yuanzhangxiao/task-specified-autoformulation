"""Reconstruct the submitted method's nine-case evidence from archived receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from autoformalism.rebuttal.prefit_replay import sealed_write

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = (
    REPO / "outputs/all-benchmarks-mad-20260925/all-benchmarks-validation-data.zip"
)
ASSESSMENT = REPO / "output/ours-mechanism-status/assessment_received.json"
INTEGRITY = REPO / "artifacts/reference-integrity-audit-2026-09-26-v3/summary.json"
HISTORY = REPO / "artifacts/reference-integrity-history-2026-09-26/comparison.json"
PLAN = "cdfbd68d07a88f510a869c595c457dfcdf6595d42cf6fb8983bf10a901d5b950"
CUTOFF = 5
MAIN = ("c1v1s1", "c0v1s1", "c1v0s1", "c0v0s1")
SOURCE_HASHES = {
    "summary": "6a4b873da258369feedef4f70937222124008f671a0caab3b6daa7f4d7815241",
    "component_summary": (
        "88bd4775182a0de952a7e497475ef265a67bce9dd43b9cdee4aff230d145209e"
    ),
}


def digest(path: Path) -> str:
    """Hash one local source without incorporating its machine-specific location."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path: Path, rows: list[dict]) -> None:
    """Write a deterministic, machine-readable table with explicit empty cells."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def load_archive(path: Path) -> tuple[dict, dict]:
    """Authenticate the two original JSON files without executing archived code."""
    sources = {}
    with ZipFile(path) as archive:
        for key, expected in SOURCE_HASHES.items():
            member = archive.getinfo(f"sources/{key}.json")
            if member.file_size > 32_000_000:
                raise ValueError("oversized campaign file")
            raw = archive.read(member)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"unexpected archived {key} identity")
            sources[key] = json.loads(raw)
    search, campaign = sources["summary"], sources["component_summary"]
    if (
        campaign["identity"] != PLAN
        or search["plan_sha256"] != PLAN
        or campaign["test_data_opened"] is not False
        or search["test_metrics_used"] is not False
    ):
        raise ValueError("incompatible campaign or data split")
    return search, campaign


def center(values: list[float]) -> dict:
    """Return descriptive median and unscaled MAD over an explicit finite cohort."""
    if not values:
        return {"median": None, "mad": None}
    if not all(math.isfinite(v) for v in values):
        raise ValueError("archive contains a nonfinite numeric outcome")
    med = statistics.median(values)
    return {"median": med, "mad": statistics.median(abs(x - med) for x in values)}


def per_case(rows: list[dict]) -> list[dict]:
    """Summarize validation incumbents with missing entries kept in the denominator."""
    groups = defaultdict(list)
    seen = set()
    for r in rows:
        if r["round"] != CUTOFF:
            continue
        task = r["task_id"]
        if task in seen:
            raise ValueError(f"duplicate common-visit task {task}")
        seen.add(task)
        arm = task.rsplit("_", 1)[1]
        for prompt in (r["arm"], "pooled"):
            groups[r["cell"], arm, prompt].append(r["retained_validation_nmse"])
    return [
        {
            "cell": cell,
            "configuration": arm,
            "prompt": prompt,
            "planned": len(values),
            "available": sum(v is not None for v in values),
            **center([v for v in values if v is not None]),
        }
        for (cell, arm, prompt), values in sorted(groups.items())
    ]


def historical_files(campaign: dict, roots: list[Path]) -> list[dict]:
    """Match exact public input hashes across declared historical projections."""
    out = []
    for cell, files in sorted(campaign["public_assets"].items()):
        for name, expected in sorted(files.items()):
            matches = []
            for i, root in enumerate(roots):
                path = root / cell / name
                if path.is_file() and digest(path) == expected:
                    matches.append(i)
            out.append(
                {
                    "cell": cell,
                    "file": name,
                    "sha256": expected,
                    "status": "matched" if matches else "receipt_only",
                    "matching_source_indices": ",".join(map(str, matches)),
                }
            )
    return out


def audit(archive: Path, root: Path, public_roots: list[Path]) -> dict:
    """Emit the original run inventory, manuscript corrections and reference audit."""
    search, campaign = load_archive(archive)
    accounting = json.loads(ASSESSMENT.read_text())
    integrity = json.loads(INTEGRITY.read_text())
    changes = json.loads(HISTORY.read_text())
    rows = search["rows"]
    checks = historical_files(campaign, public_roots)
    identity = sealed_write(
        root / "audit_identity.json",
        {
            "protocol": "iclr-method-submission-audit-1",
            "submission_plan": PLAN,
            "inputs": {
                p.name: digest(p) for p in (archive, ASSESSMENT, INTEGRITY, HISTORY)
            },
            "source_sha256": digest(Path(__file__)),
            "verified_public_assets": checks,
        },
    )
    table(root / "data_versions.csv", checks)
    summaries = per_case(rows)
    table(root / "per_case_results.csv", summaries)
    macro = []
    for arm in MAIN:
        groups = [
            r
            for r in summaries
            if r["configuration"] == arm and r["prompt"] == "pooled"
        ]
        if len(groups) != 9 or any(r["median"] is None for r in groups):
            raise ValueError("nine-case submitted macro is unavailable")
        macro.append(
            {
                "configuration": arm,
                "fit_split": "validation",
                "completed": sum(r["available"] for r in groups),
                "planned": sum(r["planned"] for r in groups),
                "visits": 6,
                **center([r["median"] for r in groups]),
            }
        )
    table(root / "reconstructed_nmse.csv", macro)
    inventory = []
    for r in rows:
        if r["round"] != CUTOFF:
            continue
        files = campaign["public_assets"][r["cell"]]
        inventory.append(
            {
                "cell": r["cell"],
                "task": r["task_id"],
                "seed": r["seed"],
                "prompt": r["arm"],
                "configuration": r["task_id"].rsplit("_", 1)[1],
                "round": CUTOFF,
                "origin_round": r["selected_origin_round"],
                "status": r["status"],
                "source_result_sha256": r["result_sha256"],
                "prompt_sha256": files["proposer_prompt.txt"],
                "training_sha256": files["train.csv"],
                "validation_sha256": files["validation.csv"],
                "training_nmse": r["retained_train_nmse"],
                "validation_nmse": r["retained_validation_nmse"],
                "validation_per_target": json.dumps(
                    r["retained_validation_per_target_nmse"], sort_keys=True
                ),
                "graph_certified": r["all_graph_requirements_certified"],
                "proposer_requests": r["cumulative_requests"],
                "proposer_tokens": r["cumulative_tokens"],
                "fit_attempts": r["cumulative_fit_attempts"],
                "latent_states": r["latent_states"],
                "test_status": "unavailable_prepruning",
                "test_nmse": None,
            }
        )
    table(root / "run_inventory.csv", inventory)
    bindings = [
        ("both", "brief_only", "c1v1s1", "graph_requirements"),
        ("no_critic", "full", "c0v1s1", "functional"),
        ("no_verifier", "full", "c1v0s1", "functional"),
        ("neither", "brief_only", "c0v0s1", "functional"),
    ]
    printed = []
    for label, prompt, arm, metric in bindings:
        method = f"autoformalism:{prompt}:{arm}"
        (entry,) = [x for x in accounting["tables"][metric] if x["method"] == method]
        printed.append(
            {
                "printed_arm": label,
                "source_variant": prompt,
                "endpoint": metric,
                "mean": entry["confirmed_mean_sd"][0],
                "across_case_sd": entry["confirmed_mean_sd"][1],
            }
        )
    table(root / "submitted_compliance_provenance.csv", printed)
    comparable = [
        {
            "method": r["method"],
            "cases": r["cases"],
            "mean": r["confirmed_mean_sd"][0],
            "across_case_sd": r["confirmed_mean_sd"][1],
            **dict(
                zip(
                    ("pass", "fail", "unresolved"),
                    r["pass_fail_unresolved"],
                    strict=True,
                )
            ),
        }
        for r in accounting["tables"]["functional"]
    ]
    table(root / "functional_compliance.csv", comparable)
    cstr = [r for r in changes["rows"] if r["family"] == "cstr"]
    table(
        root / "cstr_reference_differences.csv",
        [
            {
                "schedule": r["protocol"],
                "max_temperature_difference_K": r["state_max_difference"]["T"],
                "max_scaled_state_difference": r["state_scaled_max_difference"],
            }
            for r in cstr
        ],
    )
    affected = Counter(
        r["protocol"].split("_")[0]
        for r in cstr
        if r["state_scaled_max_difference"] > 2e-6
    )
    summary = sealed_write(
        root / "audit.json",
        {
            "audit_identity": identity["artifact_sha256"],
            "nmse": macro,
            "planned_seeds": campaign["config"]["seeds"],
            "planned_visits": campaign["config"]["rounds"],
            "public_asset_matches": dict(Counter(r["status"] for r in checks)),
            "cutoff": CUTOFF,
            "pruning_at_snapshot": campaign["endpoint_status_counts"],
            "judge_cost_whole_snapshot": campaign["judge_cost"],
            "cstr_affected_protocols": dict(affected),
            "cstr_max_temperature_change_K": max(
                r["state_max_difference"]["T"] for r in cstr
            ),
            "verified_reference_protocols": integrity["numerical_passed"],
            "reference_protocols": integrity["numerical_protocols"],
            "public_projections": integrity["public_cells_passed"],
            "test_used_for_selection": False,
            "external_baselines_evaluated": False,
            "text_85_percent": "untraced",
            "common_test_ranking": "unavailable",
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--public-root", action="append", type=Path, default=[])
    args = parser.parse_args()
    roots = args.public_root or [
        REPO,
        REPO / "work/phase_b_prompt_review",
        REPO / "artifacts/reference-integrity-audit-2026-09-26-v3/public-development",
    ]
    print(json.dumps(audit(args.archive, args.output, roots), indent=2))


if __name__ == "__main__":
    main()
