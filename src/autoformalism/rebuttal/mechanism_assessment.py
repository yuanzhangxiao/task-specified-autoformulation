"""Checkpointed CPU-only assessment of three distinct public evidence layers."""

from __future__ import annotations

import fcntl
import hashlib
import json
from collections import Counter
from pathlib import Path

from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.baseline_validation import load_public
from autoformalism.rebuttal.mechanism_audit import rubric
from autoformalism.rebuttal.mechanism_audit_campaign import _public_prompt
from autoformalism.rebuttal.mechanism_audit_sources import BUNDLE_PROTOCOL
from autoformalism.rebuttal.mechanism_checks import (
    EquationRequirement,
    equation_evidence,
)
from autoformalism.rebuttal.mechanism_probes import (
    ProbeSettings,
    _native,
    numerical_assessment,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas import CandidateModel
from autoformalism.staged_topology import content_hash

PROTOCOL = "public-mechanism-assessment-1"
REPO = Path(__file__).resolve().parents[3]


def source_identity() -> dict:
    """Pin both numerical implementation and standalone CLI."""
    return {
        "package": runtime_source_hash(),
        "cli": hashlib.sha256(
            (REPO / "scripts/assess_mechanisms.py").read_bytes()
        ).hexdigest(),
    }


def prepare(
    bundles: list[Path], public_root: Path, root: Path, config_path: Path
) -> dict:
    """Freeze public data fingerprints and exact requirement interpretations."""
    config = json.loads(config_path.read_text())
    settings = ProbeSettings.model_validate(config["numerical_settings"])
    rows, hashes, seen, data_cache = [], [], set(), {}
    for path in bundles:
        bundle = sealed_read(path)
        if bundle["protocol"] != BUNDLE_PROTOCOL:
            raise ValueError("unknown public model bundle")
        hashes.append(bundle["artifact_sha256"])
        for source in bundle["rows"]:
            cell = source["benchmark_id"]
            if cell not in config["cells"]:
                continue
            key = (
                source["method"],
                cell,
                source["tier"],
                source["repetition"],
                source.get("round"),
            )
            if key in seen:
                raise ValueError(f"duplicate source identity: {key}")
            seen.add(key)
            row = {**source, "index": len(rows)}
            rules = [
                EquationRequirement.model_validate(r)
                for r in config["cells"][cell]["requirements"]
            ]
            if not rules or len({r.id for r in rules}) != len(rules):
                raise ValueError("require unique nonempty public predicates")
            row["requirements"] = [r.model_dump() for r in rules]
            if row["status"] == "ready":
                try:
                    prompt = _public_prompt(public_root, cell)
                    digest = hashlib.sha256(prompt.encode()).hexdigest()
                    if (
                        digest != source["public_prompt_sha256"]
                        or digest != config["cells"][cell]["public_prompt_sha256"]
                    ):
                        raise ValueError(
                            "public prompt differs from source or "
                            "reviewed predicate binding"
                        )
                    quotes = {
                        r.text.rstrip(".")
                        for r in rubric(prompt)
                        if r.category == "task_mechanism"
                    }
                    if any(
                        r.public_requirement.rstrip(".") not in quotes for r in rules
                    ):
                        raise ValueError(
                            "predicate requirement is absent from public prompt"
                        )
                    if quotes - {r.public_requirement.rstrip(".") for r in rules}:
                        raise ValueError("not every public mechanism has a binding")
                    if (cell, row["tier"]) not in data_cache:
                        data_cache[cell, row["tier"]] = load_public(
                            public_root, cell, row["tier"]
                        )
                    data, actual_context, identity = data_cache[cell, row["tier"]]
                    context = ValidationContext.model_validate(source["context"])
                    for role in (
                        "targets",
                        "auxiliaries",
                        "external_inputs",
                        "fixed_covariates",
                    ):
                        if set(getattr(context, role)) != set(
                            getattr(actual_context, role)
                        ):
                            raise ValueError(
                                f"saved/public channel role mismatch: {role}"
                            )
                    if (
                        source.get("data_identity")
                        and source["data_identity"] != identity
                    ):
                        raise ValueError("source train/validation identity differs")
                    row["data_identity"] = identity
                    row["source_data_identity_checked"] = bool(
                        source.get("data_identity")
                    )
                    row["expected_response_rows"] = len(
                        data.validation.trajectories
                    ) * len(context.targets)
                    for rule in rules:
                        if rule.target not in context.targets or rule.driver not in (
                            *context.external_inputs,
                            *context.auxiliaries,
                        ):
                            raise ValueError(
                                "predicate references unavailable public channel"
                            )
                    model = CandidateModel.model_validate(source["candidate"])
                    row["model_sha256"] = content_hash(
                        [
                            source[k]
                            for k in (
                                "candidate",
                                "parameters",
                                "initials",
                                "semantics",
                            )
                        ]
                    )
                    if source["semantics"] == "continuous_time":
                        compile_candidate(model, context)
                    elif source["semantics"] == "native_increment":
                        _native(row)
                    else:
                        raise ValueError("unknown saved equation semantics")
                    row["equation_requirements"] = equation_evidence(
                        model, context, rules, semantics=source["semantics"]
                    )
                    row["fitted_equation_requirements"] = equation_evidence(
                        model,
                        context,
                        rules,
                        parameters=source["parameters"],
                        semantics=source["semantics"],
                    )
                except Exception as exc:
                    row.update(
                        status="input_failed", error=f"{type(exc).__name__}: {exc}"
                    )
            rows.append(row)
    if not rows:
        raise ValueError("no source rows match configured public cells")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "source_identity": source_identity(),
                "bundle_hashes": hashes,
                "config": config,
                "settings": settings.model_dump(),
                "public_root": str(public_root.resolve()),
                "rows": rows,
                "test_data_opened": False,
                "private_reference_opened": False,
                "live_llm_calls": 0,
                "parameter_refit_applied": False,
            },
        )
        report(root)
        return plan


def run(root: Path, index: int) -> dict:
    """One independent saved model per CPU task; each rollout checkpoints."""
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] != PROTOCOL or plan["source_identity"] != source_identity():
        raise ValueError("assessment code changed; use pinned checkout")
    if not 0 <= index < len(plan["rows"]):
        raise ValueError("row index outside frozen plan")
    row = plan["rows"][index]
    directory = root / "results" / f"{index:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "task.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / "result.json"
        if path.exists():
            saved = sealed_read(path)
            if (
                saved["plan_sha256"] != plan["artifact_sha256"]
                or saved["index"] != index
            ):
                raise ValueError("result belongs to another frozen model/plan")
            return saved
        result = {"status": row["status"], "error": row.get("error")}
        if row["status"] == "ready":
            try:
                data, _, identity = load_public(
                    Path(plan["public_root"]), row["benchmark_id"], row["tier"]
                )
                if identity != row["data_identity"]:
                    raise ValueError("frozen public data changed")
                result = {
                    "status": "complete",
                    **numerical_assessment(
                        row,
                        data.train,
                        data.validation,
                        ProbeSettings.model_validate(plan["settings"]),
                        directory / "rollouts",
                    ),
                }
            except Exception as exc:
                result = {
                    "status": "assessment_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return sealed_write(
            path, {"plan_sha256": plan["artifact_sha256"], "index": index, **result}
        )


def report(root: Path) -> dict:
    """Keep the three layers separate and retain every requested model."""
    plan = sealed_read(root / "plan.json")
    rows = []
    for row in plan["rows"]:
        record = {
            k: row.get(k)
            for k in (
                "index",
                "method",
                "benchmark_id",
                "tier",
                "repetition",
                "round",
                "status",
                "error",
                "validation_nmse",
                "equation_requirements",
                "fitted_equation_requirements",
                "model_sha256",
                "source_id",
                "source_plan_sha256",
                "source_data_identity_checked",
            )
        }
        path = root / "results" / f"{row['index']:04d}" / "result.json"
        if path.exists():
            saved = sealed_read(path)
            if (
                saved["plan_sha256"] != plan["artifact_sha256"]
                or saved["index"] != row["index"]
            ):
                raise ValueError("result identity differs")
            record.update(saved)
        for layer, expected in (
            ("equation_requirements", len(row["requirements"])),
            ("fitted_equation_requirements", len(row["requirements"])),
            ("fitted_activity", len(row["requirements"])),
            ("response_behavior", row.get("expected_response_rows")),
        ):
            if not record.get(layer):
                record[layer] = {
                    "status": "unavailable",
                    "counts": {
                        "total": expected,
                        "pass": 0,
                        "fail": 0,
                        "unresolved": expected,
                        "passed_fraction": 0.0 if expected else None,
                    },
                }
        rows.append(record)
    groups = []
    for method, cell in sorted({(r["method"], r["benchmark_id"]) for r in rows}):
        selected = [
            r for r in rows if (r["method"], r["benchmark_id"]) == (method, cell)
        ]
        group = {
            "method": method,
            "benchmark_id": cell,
            "models": len(selected),
            "completed": sum(r["status"] == "complete" for r in selected),
        }
        for layer in (
            "equation_requirements",
            "fitted_equation_requirements",
            "fitted_activity",
            "response_behavior",
        ):
            values = [r[layer]["counts"] for r in selected]
            group[layer] = {
                k: sum(v[k] or 0 for v in values)
                for k in ("total", "pass", "fail", "unresolved")
            }
            group[layer]["unknown_denominator_models"] = sum(
                v["total"] is None for v in values
            )
        representation = Counter(
            r["equation_requirements"].get(
                "continuous_time_representation", "unresolved"
            )
            for r in selected
        )
        group["continuous_time_representation"] = {
            s: representation[s] for s in ("pass", "fail", "unresolved")
        }
        groups.append(group)
    value = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "rows": rows,
        "groups": groups,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "thresholds": plan["settings"],
        "broader_scientific_interpretation": "optional_not_run",
        "test_data_opened": False,
        "live_llm_calls": 0,
        "parameter_refit_applied": False,
        "scientific_correctness_certified": False,
    }
    atomic_json(root / "summary.json", value)
    lines = [
        "# Deterministic public mechanism assessment",
        "",
        "Equation predicates, fitted influence, and response agreement are separate.",
        "Counts: pass/fail/unresolved. Unknown denominators remain explicit in JSON.",
        "Completion does not mean a good model. No overall scientific score.",
        "Validation is reused development data, not a pristine causal reference.",
        "",
        "| Method | Cell | Seed | Status | Equations | Fitted | Activity | Responses |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        totals = [
            "/".join(
                str(row[layer]["counts"][k]) for k in ("pass", "fail", "unresolved")
            )
            for layer in (
                "equation_requirements",
                "fitted_equation_requirements",
                "fitted_activity",
                "response_behavior",
            )
        ]
        lines.append(
            f"| {row['method']} | {row['benchmark_id']} | "
            f"{row['repetition']} | {row['status']} | " + " | ".join(totals) + " |"
        )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return value
