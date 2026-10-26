#!/usr/bin/env python3
"""Checkpointed three-policy construction experiment on six frozen parents."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import signal
import statistics
import tarfile
import time
from collections import Counter
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.construction import MEASURED_RESERVATION, ConstructionClient
from autoformalism.llm.response_revision import PromptPreflightError
from autoformalism.llm.staged_topology import (
    DeferredCall,
    atomic_json,
    strict_provider_schema,
)
from autoformalism.rebuttal.prefit_construction_campaign import _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.repair_comparison import RepairBudgetExceeded
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research import construction_search_policy as engine
from autoformalism.research.construction_structural_revision import (
    StructuralRevisionReply,
)
from autoformalism.staged_topology import content_hash
from scripts import construction_functions as receipts
from scripts import structural_revision_campaign as base

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = engine.PROTOCOL
SEEDS = (0, 1, 2)
ENTRYPOINTS = tuple(
    dict.fromkeys(
        (
            *base.ENTRYPOINTS,
            "scripts/construction_search_campaign.py",
            "scripts/submit_construction_search_aces.py",
            "scripts/hpc/run_construction_search_aces.sh",
            "scripts/hpc/start_construction_search_aces.sh",
        )
    )
)


def _entrypoints() -> dict:
    return {
        name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
        for name in ENTRYPOINTS
    }


def _roster(selected: dict, size: str) -> list[dict]:
    """Rotate policy order across seed bundles; match order between model arms."""
    rows = []
    for seed in SEEDS:
        for position, (name, parent) in enumerate(sorted(selected["parents"].items())):
            offset = (position + seed) % len(engine.POLICIES)
            policies = engine.POLICIES[offset:] + engine.POLICIES[:offset]
            for policy in policies:
                identity = {
                    "parent_id": name,
                    "seed": seed,
                    "policy": policy,
                    "input_sha256": content_hash(parent),
                }
                case = f"{name}__seed{seed}__{policy}"
                rows.append(
                    {
                        **identity,
                        "name": case,
                        "case_id": case,
                        "namespace": f"{PROTOCOL}:{size}:{content_hash(identity)}",
                    }
                )
    return rows


def smoke_case_ids(plan: dict) -> list[str]:
    """The first parent's seed-zero policies are an included operational subset."""
    first = min(row["parent_id"] for row in plan["cases"])
    return [
        row["case_id"]
        for row in plan["cases"]
        if row["parent_id"] == first and row["seed"] == SEEDS[0]
    ]


def _plan(selected: dict, size: str) -> dict:
    settings = base.settings_for(size).model_dump(mode="json")
    common = {
        "protocol": PROTOCOL,
        "source_archive_sha256": base.sources.SOURCE_ARCHIVE_SHA256,
        "selected_sources_sha256": content_hash(selected),
        "runtime_source_sha256": runtime_source_hash(),
        "entrypoints": _entrypoints(),
        "generation_policy": {
            k: v for k, v in settings.items() if k not in {"model", "model_revision"}
        },
        "policies": list(engine.POLICIES),
        "seeds": list(SEEDS),
        "system_sha256": content_hash(engine.SYSTEM),
        "schema_sha256": {
            cls.__name__: content_hash(cls.model_json_schema())
            for cls in (StructuralRevisionReply, engine.OptionsReply)
        },
        "gate_policy": {
            **base.old.V4_POLICIES,
            "symbolic_initials": base.SYMBOLIC_INITIAL_POLICY,
            "weighted_symbolic_initials": "weighted-shared-symbolic-initials-1",
            "parent_declarations": "parent-relative-declaration-normalization-1",
        },
        "maximum_calls_per_episode": 3,
        "maximum_options_in_first_call": 3,
        "required_options_in_first_call": 3,
        "option_output_budget": "three alternatives share one 8192-token completion",
        "smoke_policy": "first-parent-seed0-all-policies-included-in-full-1",
        "selection": "six frozen T2-easy defect cases; no NMSE-based selection",
        "automatic_followup": False,
        "test_data_opened": False,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "model_promotion": False,
    }
    return {
        **common,
        "comparison_sha256": content_hash(common),
        "model_size": size,
        "settings": settings,
        "cases": _roster(selected, size),
    }


def prepare(source: Path, root: Path, model_size: str) -> dict:
    selected = base.sources.load(source)
    base.old._validate_inputs(selected["inputs"])
    plan = _plan(selected, model_size)
    with _lock(root / "prepare-lock"):
        target = root / "saved-source.tar.gz"
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError("saved source archive differs")
        if not target.exists():
            target.write_bytes(source.read_bytes())
        sealed_write(root / "selected_sources.json", selected)
        for name, parent in selected["parents"].items():
            sealed_write(
                root / "controls" / f"{name}.json",
                {
                    "problem": parent["problem"],
                    "control_unchanged": True,
                    "problem_sha256": content_hash(parent["problem"]),
                },
            )
        sealed_write(root / "plan.json", plan)
    return {
        "identity": content_hash(plan),
        "model_size": model_size,
        "comparison_sha256": plan["comparison_sha256"],
        "cases": len(plan["cases"]),
        "maximum_generation_requests": 3 * len(plan["cases"]),
        "smoke_cases": smoke_case_ids(plan),
    }


def verify(root: Path) -> dict:
    plan = sealed_read(root / "plan.json")
    selected = base.sources.load(root / "saved-source.tar.gz")
    if sealed_read(root / "selected_sources.json") != {
        **selected,
        "artifact_sha256": content_hash(selected),
    }:
        raise ValueError("selected sources differ")
    expected = _plan(selected, plan["model_size"])
    if plan != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError("search plan, executable source, or settings differ")
    for name, parent in selected["parents"].items():
        expected = {
            "problem": parent["problem"],
            "control_unchanged": True,
            "problem_sha256": content_hash(parent["problem"]),
        }
        if sealed_read(root / "controls" / f"{name}.json") != {
            **expected,
            "artifact_sha256": content_hash(expected),
        }:
            raise ValueError("unchanged parent control differs")
    return plan


def _handoffs(plan: dict, entry: dict, parent: dict, result: dict) -> dict:
    """Retain one packet per known eligible family, without fitting or ranking."""
    files, seen = {}, set()
    for candidate in result.get("candidates", []):
        if candidate["status"] != "eligible":
            continue
        key = candidate.get("family_identity", {}).get("sha256") or content_hash(
            candidate["problem"]
        )
        if key in seen:
            continue
        seen.add(key)
        identifier = candidate["id"]
        if not identifier or any(
            ch not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for ch in identifier
        ):
            raise ValueError("invalid internal candidate id")
        files[f"fit-{identifier}.json"] = {
            "protocol": "construction-search-fit-handoff-1",
            "plan_sha256": plan["artifact_sha256"],
            "case_id": entry["case_id"],
            "policy": entry["policy"],
            "parent_id": entry["parent_id"],
            "candidate_id": identifier,
            "family_identity": candidate.get("family_identity"),
            "parent_problem_sha256": content_hash(parent["problem"]),
            "prefix_fit_problem": candidate["problem"],
            "training_consistency": candidate["audit"]["training_consistency"],
            "fitting_prefix_policy": parent.get("fixed_fitting_prefix_policy"),
            "fitting_launched": False,
            "test_data_opened": False,
            "validation_used": False,
            "scientific_compliance_assessed": False,
            "model_promotion": False,
        }
    return files


def _completion(root: Path, entry: dict, plan: dict, parent: dict) -> dict | None:
    directory = root / "results" / entry["case_id"]
    if not (directory / "completion.json").exists():
        return None
    receipt = sealed_read(directory / "completion.json")
    records = receipts._calls(root, entry, plan)
    result = json.loads((directory / "result.json").read_text())
    handoffs = _handoffs(plan, entry, parent, result)
    files = ["result.json", *handoffs]
    expected = {
        "plan_sha256": plan["artifact_sha256"],
        "input_sha256": entry["input_sha256"],
        "calls_sha256": content_hash(records),
        "status": result["status"],
        "tokenizations_sha256": content_hash(
            receipts._tokenizations(root, entry, plan)
        ),
        "files": {name: base._hash(directory / name) for name in files},
    }
    if receipt != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError("search completion receipt differs")
    replay = base._ReplayClient(records, base.settings_for(plan["model_size"]))
    if engine.run_episode(
        parent, replay, replay.settings, entry["policy"], seed=entry["seed"]
    ) != result or len(replay.consumed) != len(records):
        raise ValueError(
            "search result differs from exact request/reply reconstruction"
        )
    for name, value in handoffs.items():
        if sealed_read(directory / name) != {
            **value,
            "artifact_sha256": content_hash(value),
        }:
            raise ValueError("search fitting handoff differs")
    return result


def run(
    root: Path,
    endpoint: str,
    runtime_record: Path | None = None,
    *,
    seconds: int = 12600,
    phase: str = "smoke",
    transport=None,
    token_transport=None,
) -> dict:
    """Resume one measured client budget per episode, including multi-option calls."""
    if phase not in {"smoke", "full"}:
        raise ValueError("phase must be smoke or full")
    plan = verify(root)
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("construction server must use loopback HTTP")
    if runtime_record is not None:
        from scripts import submit_construction_search_aces as launcher

        launcher.bind_server(root, runtime_record, endpoint)
    elif transport is None:
        raise ValueError("live generation requires verified serving identity")
    selected = sealed_read(root / "selected_sources.json")
    smoke = set(smoke_case_ids(plan))
    if phase == "full" and any(
        _completion(root, e, plan, base._parent(selected, e["parent_id"])) is None
        for e in plan["cases"]
        if e["case_id"] in smoke
    ):
        raise ValueError("complete and inspect the smoke phase before full submission")
    stopped, deadline = False, time.monotonic() + seconds

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    handlers = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        with _lock(root / "worker-lock"):
            for entry in plan["cases"]:
                if phase == "smoke" and entry["case_id"] not in smoke:
                    continue
                parent = base._parent(selected, entry["parent_id"])
                if _completion(root, entry, plan, parent) is not None:
                    continue
                directory = root / "results" / entry["case_id"]
                saved = receipts._calls(root, entry, plan)
                receipts._tokenizations(root, entry, plan)
                # A consumed immutable allowance cannot be renewed by another job.
                if (directory / "budget_blocked.json").exists():
                    _blocked(root, entry, plan, saved, "budget_blocked")
                    continue
                replay = base._ReplayClient(
                    saved, base.settings_for(plan["model_size"]), pending=True
                )
                with suppress(DeferredCall):
                    engine.run_episode(
                        parent,
                        replay,
                        replay.settings,
                        entry["policy"],
                        seed=entry["seed"],
                    )
                if len(replay.consumed) != len(saved):
                    raise ValueError("pending calls are not an exact request prefix")
                settings = base.settings_for(plan["model_size"])
                client = ConstructionClient(
                    settings=settings,
                    base_url=endpoint,
                    directory=directory / "calls",
                    namespace=entry["namespace"],
                    seed=entry["seed"],
                    reservation_policy=MEASURED_RESERVATION,
                    token_transport=token_transport,
                    can_start=lambda settings=settings: not stopped
                    and time.monotonic() + settings.timeout_seconds < deadline,
                    **({"transport": transport} if transport else {}),
                )
                client.budget_diagnostics_enabled = True
                try:
                    result = engine.run_episode(
                        parent, client, settings, entry["policy"], seed=entry["seed"]
                    )
                except DeferredCall:
                    break
                except (PromptPreflightError, RepairBudgetExceeded) as exc:
                    status = (
                        "budget_blocked"
                        if isinstance(exc, RepairBudgetExceeded)
                        else "preflight_blocked"
                    )
                    sealed_write(
                        directory / f"{status}.json",
                        {
                            "plan_sha256": plan["artifact_sha256"],
                            "calls_sha256": content_hash(
                                receipts._calls(root, entry, plan)
                            ),
                            "tokenizations_sha256": content_hash(
                                receipts._tokenizations(root, entry, plan)
                            ),
                            "status": status,
                            "error": str(exc)[:4000],
                        },
                    )
                    continue
                atomic_json(directory / "result.json", result)
                handoffs = _handoffs(plan, entry, parent, result)
                for name, value in handoffs.items():
                    sealed_write(directory / name, value)
                files = ["result.json", *handoffs]
                sealed_write(
                    directory / "completion.json",
                    {
                        "plan_sha256": plan["artifact_sha256"],
                        "input_sha256": entry["input_sha256"],
                        "calls_sha256": content_hash(
                            receipts._calls(root, entry, plan)
                        ),
                        "tokenizations_sha256": content_hash(
                            receipts._tokenizations(root, entry, plan)
                        ),
                        "status": result["status"],
                        "files": {f: base._hash(directory / f) for f in files},
                    },
                )
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return report(root)


def _spread(values: list) -> dict:
    if not values:
        return {"n": 0, "median": None, "mad": None}
    middle = statistics.median(values)
    return {
        "n": len(values),
        "median": middle,
        "mad": statistics.median(abs(v - middle) for v in values),
    }


def _blocked(root: Path, entry: dict, plan: dict, calls: list, status: str) -> dict:
    """Authenticate an admission failure without pretending the episode finished."""
    path = root / "results" / entry["case_id"] / f"{status}.json"
    blocked = sealed_read(path)
    if (
        blocked["plan_sha256"] != plan["artifact_sha256"]
        or blocked["calls_sha256"] != content_hash(calls)
        or blocked["tokenizations_sha256"]
        != content_hash(receipts._tokenizations(root, entry, plan))
        or blocked["status"] != status
    ):
        raise ValueError("admission blocked identity differs")
    return blocked


def report(root: Path) -> dict:
    plan, rows, all_calls = verify(root), [], []
    selected = sealed_read(root / "selected_sources.json")
    smoke = set(smoke_case_ids(plan))
    for entry in plan["cases"]:
        parent = base._parent(selected, entry["parent_id"])
        result = _completion(root, entry, plan, parent)
        calls = receipts._calls(root, entry, plan)
        all_calls.extend(calls)
        status = result["status"] if result else "pending"
        if not result:
            for blocked_status in ("preflight_blocked", "budget_blocked"):
                marker = root / "results" / entry["case_id"] / f"{blocked_status}.json"
                if marker.exists():
                    _blocked(root, entry, plan, calls, blocked_status)
                    status = blocked_status
        candidates = (result or {}).get("candidates", [])
        eligible = [c for c in candidates if c["status"] == "eligible"]
        first = min((c["call_index"] for c in eligible), default=None)
        prefix = [r for r in calls if first is not None and r["attempt"] <= first]
        truncations = sum(
            choice.get("finish_reason") == "length"
            for record in calls
            for choice in record.get("raw_response", {}).get("choices", [])
        )
        rows.append(
            {
                **{k: entry[k] for k in ("case_id", "parent_id", "seed", "policy")},
                "status": status,
                "completed": result is not None,
                "smoke": entry["case_id"] in smoke,
                "eligible_candidates": len(eligible),
                "distinct_known_families": len(
                    {
                        c["family_identity"]["sha256"]
                        for c in candidates
                        if c.get("family_identity", {}).get("sha256")
                    }
                ),
                "eligible_known_families": len(
                    {
                        c["family_identity"]["sha256"]
                        for c in eligible
                        if c.get("family_identity", {}).get("sha256")
                    }
                ),
                "candidate_outcomes": dict(Counter(c["status"] for c in candidates)),
                "candidate_diagnostics": dict(
                    Counter(
                        (c.get("diagnostic") or {}).get("code", c["status"])
                        for c in candidates
                    )
                ),
                "repeated_unchanged_failures": sum(
                    c.get("repeated_unchanged_failure", False) for c in candidates
                ),
                "renaming_identity_unresolved": sum(
                    c.get("problem") is not None
                    and c.get("family_identity", {}).get("sha256") is None
                    for c in candidates
                ),
                "omitted_prior_actions": sum(
                    len(c.get("omitted_previous_actions", [])) for c in candidates
                ),
                "output_truncations": truncations,
                "eligible_check_statuses": dict(
                    Counter(
                        c["audit"]["training_consistency"]["status"] for c in eligible
                    )
                ),
                "calls_to_first_eligible": first + 1 if first is not None else None,
                "tokens_to_first_eligible": _cost(prefix)["observed_tokens"]
                if first is not None
                and not _cost(prefix)["requests_with_unknown_usage"]
                else None,
                "cost": _cost(calls),
            }
        )
    aggregates = {}
    for policy in engine.POLICIES:
        selected_rows = [r for r in rows if r["policy"] == policy]
        completed = [r for r in selected_rows if r["completed"]]
        aggregates[policy] = {
            "planned_episodes": len(selected_rows),
            "completed_episodes": len(completed),
            "eligible_episodes": sum(r["eligible_candidates"] > 0 for r in completed),
            "parents_with_eligible_candidate": len(
                {r["parent_id"] for r in completed if r["eligible_candidates"]}
            ),
            "repeated_unchanged_failures": sum(
                r["repeated_unchanged_failures"] for r in completed
            ),
            "output_truncations": sum(r["output_truncations"] for r in selected_rows),
            "omitted_prior_actions": sum(r["omitted_prior_actions"] for r in completed),
            "observed_tokens_completed": _spread(
                [r["cost"]["observed_tokens"] for r in completed]
            ),
            "calls_to_first_eligible_successes_only": _spread(
                [
                    r["calls_to_first_eligible"]
                    for r in completed
                    if r["calls_to_first_eligible"] is not None
                ]
            ),
            "tokens_to_first_eligible_successes_only": _spread(
                [
                    r["tokens_to_first_eligible"]
                    for r in completed
                    if r["tokens_to_first_eligible"] is not None
                ]
            ),
        }
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "comparison_sha256": plan["comparison_sha256"],
        "model_size": plan["model_size"],
        "status": "complete" if all(r["completed"] for r in rows) else "incomplete",
        "smoke_complete": all(r["completed"] for r in rows if r["smoke"]),
        "smoke_eligible_episodes": sum(
            r["eligible_candidates"] > 0 for r in rows if r["smoke"]
        ),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "by_policy": aggregates,
        "cost": _cost(all_calls),
        "rows": rows,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "automatic_followup": False,
        "model_promotion": False,
        "interpretation": "Finite gate eligibility is not scientific validity or "
        "fitted accuracy. Deferred checks remain unresolved. Token summaries use "
        "unscaled MAD; unavailable episodes remain in the planned roster.",
    }
    atomic_json(root / "summary.json", summary)
    lines = [
        "# Construction search policies",
        "",
        f"Model: {plan['model_size']}; status: {summary['status']}; "
        f"smoke complete: {summary['smoke_complete']}.",
        "",
        summary["interpretation"],
        "",
        "| Policy | Completed / planned | Episodes with eligible model | "
        "Parents with eligible model | Tokens median [MAD] |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy, row in aggregates.items():
        spread = row["observed_tokens_completed"]
        lines.append(
            f"| {policy} | {row['completed_episodes']} / {row['planned_episodes']} | "
            f"{row['eligible_episodes']} | {row['parents_with_eligible_candidate']} | "
            f"{spread['median']} [{spread['mad']}] |"
        )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return summary


def preview(root: Path) -> dict:
    """Render exact initial requests, plus clearly marked offline repair examples."""
    plan = verify(root)
    selected = sealed_read(root / "selected_sources.json")
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Three-policy prompts</title>",
        "<style>body{max-width:1100px;margin:2rem auto;font:16px/1.5 sans-serif}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;"
        "padding:1rem}mark{background:#fff0a8}summary{cursor:pointer;font-weight:bold}"
        "section{margin:2rem 0}</style>",
        "<h1>Construction search: 20B and 120B</h1><p><mark>New:</mark> compact "
        "exactly three initial options (mandatory), parent-relative changes only, "
        "safe redundant-declaration normalization, and weighted symbolic initial "
        "checks. Both models receive the same scientific prompts. "
        "No fitting or test-data access.</p>",
        "<p>Sequential repair restarts after an unchanged failure repeats; independent "
        "proposals receive no previous failures; the option policy requires exactly "
        "three complete alternatives sharing one output budget, then repairs two "
        "distinct failed families at most once each.</p>",
        "<p>Each arm has 6 parents &times; 3 seeds &times; 3 policies = 54 episodes, "
        "capped at 162 calls. Smoke uses the first parent's seed-zero policies and "
        "is included in those totals.</p>",
        "<h2>System instruction</h2><pre>" + html.escape(engine.SYSTEM) + "</pre>",
    ]
    for name in sorted(selected["parents"]):
        parent = base._parent(selected, name)
        for policy in engine.POLICIES:
            packet = {
                "system": engine.SYSTEM,
                "user": engine.request(
                    parent,
                    policy,
                    batch=policy == "options_shallow",
                    fresh=policy == "independent",
                ),
                "response_schema": strict_provider_schema(
                    (
                        engine.OptionsReply
                        if policy == "options_shallow"
                        else StructuralRevisionReply
                    ).model_json_schema()
                ),
            }
            atomic_json(root / "preview" / f"{name}-{policy}.json", packet)
            parts.append(
                f"<details><summary>{html.escape(name)} · {policy} · initial request"
                "</summary><pre>"
                + html.escape(json.dumps(json.loads(packet["user"]), indent=2))
                + "</pre></details>"
            )
    parent = base._parent(selected, sorted(selected["parents"])[0])
    raw = {
        "decision": "retain",
        "hypothesis": "Retain for this offline illustration.",
        "expected_observable_effect": "Unchanged",
        "evidence_ids": ["training_consistency"],
    }
    first = engine.assess_candidate(parent, raw, candidate_id="call0-option0")
    history = [first]
    for title, fresh in (
        ("Repair with full intended patch", False),
        ("Fresh branch after the unchanged failure repeats", True),
    ):
        if fresh:
            second = engine.assess_candidate(
                parent, raw, candidate_id="call1-option0", call_index=1
            )
            history.append(engine.annotate_repetition(second, history))
        packet = {
            "system": engine.SYSTEM,
            "user": engine.request(
                parent,
                "sequential_restart",
                history=history,
                intended=history[-1],
                fresh=fresh,
            ),
        }
        file = "restart-example.json" if fresh else "repair-example.json"
        atomic_json(root / "preview" / file, packet)
        parts.append(
            f"<h2>{title}</h2><p><mark>Offline illustration, not a generated reply:"
            "</mark> the unchanged saved parent is checked using a hypothetical "
            "retain response.</p><pre>"
            + html.escape(json.dumps(json.loads(packet["user"]), indent=2))
            + "</pre>"
        )
    for schema in (StructuralRevisionReply, engine.OptionsReply):
        parts.append(
            f"<details><summary>{schema.__name__} response schema</summary><pre>"
            + html.escape(
                json.dumps(strict_provider_schema(schema.model_json_schema()), indent=2)
            )
            + "</pre></details>"
        )
    (root / "PROMPTS.html").write_text("\n".join(parts))
    return {
        "path": str(root / "PROMPTS.html"),
        "llm_calls": 0,
        "cases": len(plan["cases"]),
    }


def inspect(root: Path) -> dict:
    result = report(root)
    preview(root)
    path = root / "inspection.tar.gz"
    allowed = {
        "plan.json",
        "selected_sources.json",
        "summary.json",
        "SUMMARY.md",
        "PROMPTS.html",
        "saved-source.tar.gz",
        "execution.json",
        "serving_runtime.json",
    }
    with tarfile.open(path, "w:gz") as archive:
        for item in sorted(root.rglob("*")):
            rel = item.relative_to(root)
            if (
                not item.is_symlink()
                and item.is_file()
                and (
                    str(rel) in allowed
                    or rel.parts[0]
                    in {
                        "results",
                        "controls",
                        "preview",
                        "logs",
                        "submission",
                        "runtime",
                    }
                )
            ):
                archive.add(item, arcname=str(rel), recursive=False)
    return {**result, "archive": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "verify", "run", "report", "preview", "inspect")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model-size", choices=tuple(base.MODEL_SPECS), default="20b")
    parser.add_argument("--endpoint")
    parser.add_argument("--runtime-record", type=Path)
    parser.add_argument("--seconds", type=int, default=12600)
    parser.add_argument("--phase", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    if args.command == "prepare":
        if not args.source:
            parser.error("prepare requires --source")
        value = prepare(args.source, args.root, args.model_size)
    elif args.command == "run":
        if not args.endpoint:
            parser.error("run requires --endpoint")
        value = run(
            args.root,
            args.endpoint,
            args.runtime_record,
            seconds=args.seconds,
            phase=args.phase,
        )
    else:
        value = {
            "verify": verify,
            "report": report,
            "preview": preview,
            "inspect": inspect,
        }[args.command](args.root)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
