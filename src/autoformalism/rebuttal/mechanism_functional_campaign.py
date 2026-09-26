"""Immutable public-mechanism tests and coverage-aware robust aggregation."""

from __future__ import annotations

import fcntl
import hashlib
import platform
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np
import scipy

from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.baseline_validation import load_public
from autoformalism.rebuttal.mechanism_audit import rubric
from autoformalism.rebuttal.mechanism_audit_campaign import _public_prompt
from autoformalism.rebuttal.mechanism_functional import (
    Rule,
    Settings,
    assess,
    structural_check,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.staged_topology import content_hash

PROTOCOL = "fitted-public-mechanism-tests-1"
REPO = Path(__file__).resolve().parents[3]


def code_identity():
    """Resume requires identical implementation, including the package adapter CLI."""
    return {
        "runtime": runtime_source_hash(),
        "numerics": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "cli": content_hash(
            [
                (REPO / "scripts" / name).read_text()
                for name in (
                    "assess_functional_mechanisms.py",
                    "assess_results_package_mechanisms.py",
                    "audit_experiments_results_package.py",
                )
            ]
        ),
    }


def prepare(rows, config, root, public_root, input_identity):
    """Pin common public obligations and development identities before scoring."""
    if config["protocol"] != PROTOCOL:
        raise ValueError("unknown functional mechanism protocol")
    Settings.model_validate(config["numerical_settings"])
    for binding in config["cells"].values():
        rules = [Rule.model_validate(r) for r in binding["mechanisms"]]
        if not rules or len({r.id for r in rules}) != len(rules):
            raise ValueError("mechanism IDs must be nonempty and unique")
    sources, identities, seen = [], {}, set()
    for source in rows:
        row = dict(source)
        cell = row["benchmark_id"]
        if cell not in config["cells"]:
            continue
        key = (row["method"], cell, row["repetition"])
        if key in seen:
            raise ValueError(f"duplicate method/case/repetition: {key}")
        seen.add(key)
        row["index"] = len(sources)
        binding = config["cells"][cell]
        row["mechanisms"] = binding["mechanisms"]
        # Check every cell's public contract, including cells with missing models.
        if (cell, row["tier"]) not in identities:
            prompt = _public_prompt(public_root, cell)
            if (
                hashlib.sha256(prompt.encode()).hexdigest()
                != binding["public_prompt_sha256"]
            ):
                raise ValueError(f"public prompt differs: {cell}")
            quotes = {
                r.text.rstrip(".")
                for r in rubric(prompt)
                if r.category == "task_mechanism"
            }
            if {
                r["public_requirement"].rstrip(".") for r in row["mechanisms"]
            } != quotes:
                raise ValueError("require exact coverage of public mechanism bullets")
            _, context, identity = load_public(public_root, cell, row["tier"])
            identities[cell, row["tier"]] = (context, identity)
        actual, identity = identities[cell, row["tier"]]
        row["data_identity"] = identity
        if row["status"] == "ready":
            context = ValidationContext.model_validate(row["context"])
            for role in (
                "targets",
                "auxiliaries",
                "external_inputs",
                "fixed_covariates",
            ):
                if set(getattr(context, role)) != set(getattr(actual, role)):
                    raise ValueError(f"public channel role mismatch: {key}/{role}")
            if row["public_prompt_sha256"] != identity["prompt"]:
                raise ValueError("source/config public prompt differs")
        public_names = {
            *actual.targets,
            *actual.auxiliaries,
            *actual.external_inputs,
            *actual.fixed_covariates,
        }
        for r in row["mechanisms"]:
            if r["target"] not in actual.targets:
                raise ValueError("rule target must be a public generated target")
            if any(
                r[k] not in public_names
                for k in ("driver", "feed", "jacket", "reactant")
                if r.get(k)
            ):
                raise ValueError("rule drivers must be public channels")
        sources.append(row)
    if not sources:
        raise ValueError("empty mechanism assessment roster")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "code_identity": code_identity(),
                "config": config,
                "rows": sources,
                "input_identity": input_identity,
                "public_root": str(public_root.resolve()),
                "test_data_opened": False,
                "llm_calls": 0,
                "optimizer_calls": 0,
                "model_changes": 0,
                "selection_changes": 0,
            },
        )
        report(root)
        return plan


def unavailable(row, reason):
    """Known model failures score zero; missing scientific evidence is unresolved."""
    failed = row["status"] == "unavailable" and row.get("source_terminal_status") in {
        "failed",
        "timed_out",
        "construction_failed",
        "fit_failed",
    }
    return [
        {
            "id": r["id"],
            "public_requirement": r["public_requirement"],
            "status": "fail" if failed else "unresolved",
            "reason": reason,
            "method_model_failure": failed,
            "points": [],
        }
        for r in row["mechanisms"]
    ]


def run(root, index):
    """One immutable task per model; no refitting or provider access."""
    plan = sealed_read(root / "plan.json")
    if plan["code_identity"] != code_identity():
        raise ValueError("assessment source changed; use the pinned checkout")
    if not 0 <= index < len(plan["rows"]):
        raise ValueError("index outside frozen roster")
    row = plan["rows"][index]
    directory = root / "results" / f"{index:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / "result.json"
        if path.exists():
            saved = sealed_read(path)
            if (
                saved["plan_sha256"] != plan["artifact_sha256"]
                or saved["index"] != index
            ):
                raise ValueError("saved result identity differs")
            return saved
        findings = unavailable(row, row.get("error") or "model unavailable")
        status = "assessed" if row["status"] != "ready" else "assessment_failed"
        error = None
        if row["status"] == "ready":
            try:
                data, _, identity = load_public(
                    Path(plan["public_root"]), row["benchmark_id"], row["tier"]
                )
                if identity != row["data_identity"]:
                    raise ValueError("frozen public development data changed")
                findings = assess(
                    row,
                    row["mechanisms"],
                    data.train,
                    Settings.model_validate(plan["config"]["numerical_settings"]),
                    directory / "rollouts",
                )
                status = "assessed"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                findings = unavailable(row, error)
                for rule, finding in zip(row["mechanisms"], findings, strict=True):
                    try:
                        known = structural_check(row, rule)
                    except Exception:
                        known = None
                    if known:
                        finding.update(known)
        return sealed_write(
            path,
            {
                "plan_sha256": plan["artifact_sha256"],
                "index": index,
                "status": status,
                "error": error,
                "mechanisms": findings,
                "llm_calls": 0,
                "optimizer_calls": 0,
                "test_data_opened": False,
                "free_rollouts_started": len(
                    list((directory / "rollouts").glob("*.started.json"))
                ),
            },
        )


def robust(values):
    """Unscaled median absolute deviation, not standard deviation of outliers."""
    center = median(values)
    return {"median": center, "mad": median([abs(x - center) for x in values])}


def aggregate(rows, cells):
    """Average mechanisms within run; median runs then cases; retain uncertainty."""
    methods = []
    for method in sorted({r["method"] for r in rows}):
        cases = []
        totals = Counter()
        for cell in cells:
            selected = [
                r for r in rows if r["method"] == method and r["benchmark_id"] == cell
            ]
            counts = Counter(m["status"] for r in selected for m in r["mechanisms"])
            totals.update(counts)
            lower, upper = [], []
            for row in selected:
                n = len(row["mechanisms"])
                p = sum(m["status"] == "pass" for m in row["mechanisms"])
                u = sum(m["status"] == "unresolved" for m in row["mechanisms"])
                lower.append(p / n)
                upper.append((p + u) / n)
            cases.append(
                {
                    "benchmark_id": cell,
                    "runs": len(selected),
                    "counts": dict(counts),
                    "confirmed": robust(lower or [0.0]),
                    "possible": robust(upper or [1.0]),
                }
            )
        methods.append(
            {
                "method": method,
                "cases": cases,
                "counts": dict(totals),
                "case_coverage": sum(c["runs"] > 0 for c in cases),
                "fully_confirmed_runs": sum(
                    all(m["status"] == "pass" for m in r["mechanisms"])
                    for r in rows
                    if r["method"] == method
                ),
                "runs_with_counterexample_or_model_failure": sum(
                    any(m["status"] == "fail" for m in r["mechanisms"])
                    for r in rows
                    if r["method"] == method
                ),
                "runs_with_unresolved_evidence": sum(
                    any(m["status"] == "unresolved" for m in r["mechanisms"])
                    for r in rows
                    if r["method"] == method
                ),
                "confirmed": robust([c["confirmed"]["median"] for c in cases]),
                "possible": robust([c["possible"]["median"] for c in cases]),
            }
        )
    return methods


def report(root):
    """Do not turn incomplete jobs or unidentifiable pathways into passes."""
    plan = sealed_read(root / "plan.json")
    if plan["code_identity"] != code_identity():
        raise ValueError("assessment source changed; use the pinned checkout")
    rows = []
    for row in plan["rows"]:
        value = {
            k: row.get(k)
            for k in (
                "index",
                "method",
                "benchmark_id",
                "repetition",
                "source_public_prompt_verified",
                "semantics_basis",
            )
        }
        path = root / "results" / f"{row['index']:04d}" / "result.json"
        if path.exists():
            saved = sealed_read(path)
            if (
                saved["plan_sha256"] != plan["artifact_sha256"]
                or saved["index"] != row["index"]
            ):
                raise ValueError("result/plan mismatch")
            value.update(saved)
        else:
            value.update(
                status="pending", mechanisms=unavailable(row, "assessment not executed")
            )
        statuses = [m["status"] for m in value["mechanisms"]]
        value["confirmed_fraction"] = statuses.count("pass") / len(statuses)
        value["possible_fraction"] = (
            statuses.count("pass") + statuses.count("unresolved")
        ) / len(statuses)
        rows.append(value)
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "all_workers_assessed": all(r["status"] == "assessed" for r in rows),
        "rows": rows,
        "methods": aggregate(rows, plan["config"]["cells"]),
        "free_rollouts_started": sum(r.get("free_rollouts_started", 0) for r in rows),
        "score_name": "confirmed_public_mechanism_test_pass_rate",
        "universal_scientific_compliance": False,
        "interval_interpretation": (
            "evidence bounds, not statistical confidence intervals"
        ),
        "llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }
    from autoformalism.llm.staged_topology import atomic_json

    atomic_json(root / "summary.json", result)
    lines = [
        "# Fitted public-mechanism tests",
        "",
        "Finite public tests; no claim of universal scientific correctness.",
        "Case medians then macro median; brackets contain unscaled MAD.",
        (
            "Unresolved evidence is retained. Possible score is an upper bound, "
            "not a confidence interval."
        ),
        (
            "T2-hard disposal/source identification remains unresolved "
            "when its necessary proxy passes."
        ),
        "A macro median of 100% does not mean every mechanism or model passed.",
        "",
        f"Status counts: {result['status_counts']}",
        f"All workers assessed: {result['all_workers_assessed']}",
        "",
        "| Method | Confirmed % [MAD pp] | Possible % | Pass / fail / unresolved |",
        "| --- | ---: | ---: | --- |",
    ]
    for m in result["methods"]:
        c = m["counts"]
        lines.append(
            f"| {m['method']} | {100 * m['confirmed']['median']:.1f} "
            f"[{100 * m['confirmed']['mad']:.1f}] | "
            f"{100 * m['possible']['median']:.1f} "
            f"| {c.get('pass', 0)} / {c.get('fail', 0)} / {c.get('unresolved', 0)} |"
        )
    lines += [
        "",
        "## Per-case results",
        "",
        "| Method | Benchmark | Confirmed % [MAD pp] | Possible % | P/F/U |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for m in result["methods"]:
        for c in m["cases"]:
            counts = c["counts"]
            lines.append(
                f"| {m['method']} | {c['benchmark_id']} | "
                f"{100 * c['confirmed']['median']:.1f} "
                f"[{100 * c['confirmed']['mad']:.1f}] | "
                f"{100 * c['possible']['median']:.1f} | "
                f"{counts.get('pass', 0)}/{counts.get('fail', 0)}/"
                f"{counts.get('unresolved', 0)} |"
            )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return result
