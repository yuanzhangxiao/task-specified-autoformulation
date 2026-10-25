#!/usr/bin/env python3
"""Audit saved replies with the new bookkeeping, without simulating new decisions."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.staged_topology import atomic_json, visible_response
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research import construction_search_policy as engine
from autoformalism.staged_topology import content_hash
from scripts import replay_structural_revision as historical
from scripts import structural_revision_campaign as base

PROTOCOL = "saved-search-bookkeeping-replay-1"
# Original Downloads were removed. These are deterministic repacks of the
# preserved inspection members; all 108 call hashes match the prior replay.
# Inner original plan, parent, request, tokenization and completion pins remain.
PRESERVED_ARCHIVES = {
    "20b": "7ab2e342168ab3b45eba677b5b4b0521784f56e7cedc7416bba8a3765bd88615",
    "120b": "d266f2246f92e37b59ebfba6ac7153b3847e8bfad28ee439199c1282cb2664f4",
}


def _load(path: Path, arm: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("bounded regular source archive required")
    if path.stat().st_size > historical.MAX_BYTES:
        raise ValueError("source archive exceeds bound")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest not in {
        historical.SOURCE_PINS[arm]["archive_sha256"],
        PRESERVED_ARCHIVES[arm],
    }:
        raise ValueError("source archive identity differs")
    result = historical.load_source(path, arm, archive_sha256=digest)
    return {**result, "actual_archive_sha256": digest}


def replay(archives: dict[str, Path], root: Path) -> dict:
    """Check each historic reply against its actual parent; preserve source files."""
    root = root.resolve()
    if any(path.resolve().is_relative_to(root) for path in archives.values()):
        raise ValueError("replay output must be separate from source archive storage")
    loaded = [_load(archives[arm], arm) for arm in sorted(archives)]
    if len({v["plan"]["comparison_sha256"] for v in loaded}) != 1:
        raise ValueError("historical comparison identities differ")
    identity = {
        "protocol": PROTOCOL,
        "sources": {v["arm"]: historical.SOURCE_PINS[v["arm"]] for v in loaded},
        "actual_archive_sha256": {v["arm"]: v["actual_archive_sha256"] for v in loaded},
        "runtime_source_sha256": runtime_source_hash(),
        "entrypoints": base._entrypoints(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "llm_calls": 0,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "adopted_models": 0,
        "fitting_exports": 0,
    }
    rows, arms = [], {}
    with _lock(root / "replay-lock"):
        plan = sealed_write(root / "plan.json", identity)
        for source in loaded:
            history = {}
            for call in source["calls"]:
                case = call["case_id"]
                prior = history.setdefault(case, [])
                parent = base._parent(source["selected"], call["parent_id"])
                receipt = {
                    "plan_sha256": plan["artifact_sha256"],
                    "arm": source["arm"],
                    **{k: v for k, v in call.items() if k != "record"},
                    "request_sha256": call["record"]["request_hash"],
                    "record_sha256": content_hash(call["record"]),
                    "history_sha256": content_hash(prior),
                }
                path = root / "rows" / f"{source['arm']}-{case}-{call['attempt']}.json"
                if path.exists():
                    row = sealed_read(path)
                    if row["receipt"] != receipt:
                        raise ValueError("saved bookkeeping replay identity differs")
                else:
                    try:
                        raw = visible_response(call["record"])
                    except ValueError:
                        raw = None
                    candidate = engine.assess_candidate(
                        parent,
                        raw,
                        candidate_id=f"call{call['attempt']}-option0",
                        call_index=call["attempt"],
                    )
                    candidate = engine.annotate_repetition(candidate, prior)
                    row = sealed_write(
                        path,
                        {
                            "receipt": receipt,
                            "candidate": candidate,
                            "adopted": False,
                            "fitting_exported": False,
                        },
                    )
                prior.append(row["candidate"])
                rows.append(
                    {
                        "arm": source["arm"],
                        "case_id": case,
                        "parent_id": call["parent_id"],
                        "seed": call["seed"],
                        "attempt": call["attempt"],
                        "historical_outcome": call["historical_outcome"],
                        "status": row["candidate"]["status"],
                        "diagnostic": (row["candidate"].get("diagnostic") or {}).get(
                            "code"
                        ),
                        "family_identity": row["candidate"]["family_identity"],
                        "repeated_unchanged_failure": row["candidate"][
                            "repeated_unchanged_failure"
                        ],
                        "assessment_status": row["candidate"]
                        .get("audit", {})
                        .get("training_consistency", {})
                        .get("status"),
                        "evidence_file": str(path.relative_to(root)),
                    }
                )
            arm_rows = [r for r in rows if r["arm"] == source["arm"]]
            arms[source["arm"]] = {
                "saved_replies": len(arm_rows),
                "eligible_replies": sum(r["status"] == "eligible" for r in arm_rows),
                "eligible_episodes": len(
                    {r["case_id"] for r in arm_rows if r["status"] == "eligible"}
                ),
                "identified_model_families": len(
                    {
                        (r["parent_id"], r["family_identity"]["sha256"])
                        for r in arm_rows
                        if r["family_identity"].get("sha256")
                    }
                ),
                "renaming_identity_deferred": sum(
                    r["family_identity"].get("sha256") is None for r in arm_rows
                ),
                "repeated_unchanged_failures": sum(
                    r["repeated_unchanged_failure"] for r in arm_rows
                ),
                "outcomes": dict(
                    Counter(r["diagnostic"] or r["status"] for r in arm_rows)
                ),
            }
        summary = {
            "protocol": PROTOCOL,
            "plan_sha256": plan["artifact_sha256"],
            "arms": arms,
            "rows": rows,
            "llm_calls": 0,
            "optimizer_calls": 0,
            "solver_rollouts": 0,
            "test_data_opened": False,
            "adopted_models": 0,
            "fitting_exports": 0,
            "interpretation": "Independent saved-reply rechecks and retrospective "
            "history only. Later replies did not receive the new prompts; this is "
            "not live policy performance. Family counts exclude unsupported/invalid "
            "candidates; deferred checks are not scientific approval.",
        }
        atomic_json(root / "summary.json", summary)
        parts = [
            "<!doctype html><meta charset='utf-8'>"
            "<title>Search bookkeeping replay</title>",
            "<style>body{max-width:1150px;margin:2rem auto;font:16px/1.5 sans-serif}"
            "table{border-collapse:collapse}td,th{padding:.5rem;border:1px solid #ccc}"
            "pre{white-space:pre-wrap}</style>",
            "<h1>Saved replies: search bookkeeping audit</h1><p>"
            + html.escape(summary["interpretation"])
            + "</p>",
            "<h2>Arm totals</h2><pre>"
            + html.escape(json.dumps(arms, indent=2))
            + "</pre>",
            "<h2>All replies</h2><table><tr><th>Model</th>"
            "<th>Parent / seed / attempt</th><th>Outcome</th>"
            "<th>Same family and failure</th><th>Evidence</th></tr>",
        ]
        for row in rows:
            parts.append(
                f"<tr><td>{row['arm']}</td><td>{row['parent_id']} / "
                f"{row['seed']} / {row['attempt']}</td>"
                f"<td>{html.escape(row['diagnostic'] or row['status'])}</td>"
                f"<td>{row['repeated_unchanged_failure']}</td>"
                f"<td><a href='{row['evidence_file']}'>Full candidate and checks"
                "</a></td></tr>"
            )
        parts.append("</table>")
        (root / "REPORT.html").write_text("\n".join(parts))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-20b", type=Path, required=True)
    parser.add_argument("--archive-120b", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    value = replay({"20b": args.archive_20b, "120b": args.archive_120b}, args.root)
    print(json.dumps({k: v for k, v in value.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
