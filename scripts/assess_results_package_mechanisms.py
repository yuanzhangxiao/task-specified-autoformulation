#!/usr/bin/env python3
"""Inspect fitted public equation predicates and export CPU replay inputs.

This is a necessary-feature assessment, not a complete scientific compliance
score. It does not read trajectories, fit models, or call a scientific judge.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from audit_experiments_results_package import (
    METHODS,
    load_package,
    read_roster,
    statistics,
    summarize,
)

from autoformalism.baselines.d3_rollout import NativeMap
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.mechanism_audit_sources import BUNDLE_PROTOCOL
from autoformalism.rebuttal.mechanism_checks import (
    EquationRequirement,
    equation_evidence,
)
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.schemas import (
    CandidateModel,
    InitialConditionSpec,
    ObservationMapping,
)

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "results-package-fitted-equations-1"
HARD_T1 = "phase_b_dalla_man_t1_canonical_named_hard"
HARD_T1_PROMPT = "ede5a1db0a828a8d6582bb80614d536897941b6e6164f59e2a426b0c905a2fe3"
LEGACY_CONTINUOUS_ADAPTERS = {
    "sindy": "sindy_result",
    "pysr": "pysr_result",
    "raw_data_agent:gpt-5.6-sol": "raw_data_agent_run",
}


def assessment_config() -> dict:
    """Reuse reviewed six-case rules; T1-hard has the identical meal obligation."""
    config = json.loads((REPO / "configs/mechanism_assessment_v1.json").read_text())
    hard = copy.deepcopy(config["cells"]["phase_b_dalla_man_t1_canonical_named_easy"])
    hard["public_prompt_sha256"] = HARD_T1_PROMPT
    config["cells"][HARD_T1] = hard
    return config


def resolve_semantics(source: dict) -> tuple[str, str]:
    """Recognize explicit execution or documented legacy continuous adapters."""
    model = source["model"]
    values = {
        v
        for v in (source.get("execution_semantics"), model.get("execution_semantics"))
        if v is not None
    }
    if len(values) > 1:
        raise ValueError("conflicting execution semantics")
    value = next(iter(values), None)
    if source["method"] == "d3_native_no_tools":
        if value != "discrete_increment_recursive_rollout":
            raise ValueError("D3 requires explicit native increment semantics")
        return "native_increment", "explicit_saved_execution"
    if value == "continuous_ode_free_rollout":
        return "continuous_time", "explicit_saved_execution"
    adapter = model.get("source_provenance", {}).get("adapter")
    if (
        value is None
        and adapter is not None
        and adapter == LEGACY_CONTINUOUS_ADAPTERS.get(source["method"])
    ):
        return "continuous_time", "legacy_adapter_contract_inference"
    raise ValueError("unknown or incompatible saved execution semantics")


def declaration_violations(candidate: CandidateModel, parameters: dict) -> list[dict]:
    """Audit declarations separately; they are not a common scientific rubric."""
    result = []
    for p in candidate.parameters:
        value = parameters[p.name]
        domain_bad = (p.domain == "positive" and value <= 0) or (
            p.domain == "nonnegative" and value < 0
        )
        bounds_bad = (
            p.bounds is not None and not p.bounds.lower <= value <= p.bounds.upper
        )
        if domain_bad or bounds_bad:
            result.append(
                {
                    "parameter": p.name,
                    "value": value,
                    "domain": p.domain,
                    "bounds": p.bounds.model_dump() if p.bounds else None,
                }
            )
    return result


def adapt(source: dict, config: dict) -> dict:
    """Preserve fitted values and report absent source provenance explicitly."""
    row = {k: source[k] for k in ("method", "benchmark_id", "tier", "repetition")}
    row.update(
        status="unavailable",
        error=None,
        source_id=source["request_id"],
        historical_runtime_valid=source.get("runtime_valid"),
        historical_runtime_diagnostics=source.get("runtime_diagnostics"),
        source_terminal_status=source.get("terminal_status"),
    )
    rule = config["cells"].get(row["benchmark_id"])
    if rule is None:
        row["error"] = (
            "T2 delayed-insulin-action requirement has no reviewed binding here"
        )
        row["status"] = "requirements_unbound"
        return row
    if not source.get("model"):
        row["error"] = "no saved model; not an assessed equation failure"
        return row
    try:
        model = source["model"]
        candidate = CandidateModel.model_validate(model["candidate"])
        context = ValidationContext.model_validate(model["validation_context"])
        parameters = model["parameterization"]["global_parameters"]
        semantics, basis = resolve_semantics(source)
        source_prompt = (source.get("public_mechanism") or {}).get(
            "public_prompt_sha256"
        )
        if source_prompt and source_prompt != rule["public_prompt_sha256"]:
            raise ValueError("saved public prompt differs from reviewed rule")
        row.update(
            source_public_prompt_sha256=source_prompt,
            source_public_prompt_verified=source_prompt is not None,
            public_prompt_sha256=rule["public_prompt_sha256"],
            prompt_binding_basis="saved_hash" if source_prompt else "case_roster_only",
            semantics=semantics,
            semantics_basis=basis,
            source_provenance=model.get("source_provenance"),
            parameters=parameters,
            initials=model["parameterization"]["global_initial_conditions"],
            context=context.model_dump(mode="json"),
        )
        if semantics == "native_increment":
            NativeMap.build(
                candidate,
                parameters,
                (*context.targets, *context.auxiliaries),
                (*context.external_inputs, *context.fixed_covariates),
            )
            row["native_auxiliary_equations_ignored"] = [
                e.model_dump()
                for e in candidate.state_equations
                if e.state in context.auxiliaries
            ]
            candidate = candidate.model_copy(
                update={
                    "state_equations": tuple(
                        e
                        for e in candidate.state_equations
                        if e.state in context.targets
                    ),
                    "observation_mappings": tuple(
                        ObservationMapping(channel=t, expression=t)
                        for t in context.targets
                    ),
                    "initial_conditions": tuple(
                        InitialConditionSpec(state=t, scope="global", expression=t)
                        for t in context.targets
                    ),
                }
            )
            row["initials"] = {}
        requirements = [
            EquationRequirement.model_validate(r) for r in rule["requirements"]
        ]
        row["fitted_equation_requirements"] = equation_evidence(
            candidate, context, requirements, parameters=parameters, semantics=semantics
        )
        row["declaration_violations"] = declaration_violations(candidate, parameters)
        row["candidate"] = candidate.model_dump(mode="json")
        row["status"] = "ready"
    except Exception as exc:
        row.update(status="input_failed", error=f"{type(exc).__name__}: {exc}")
    return row


def aggregate(rows: list[dict], roster: tuple[str, ...], config: dict) -> list[dict]:
    """Median repetitions, then cases; unresolved evidence is never dropped."""
    result = []
    for method, label in METHODS.items():
        cases = []
        for cell in roster:
            subset = sorted(
                (
                    r
                    for r in rows
                    if r["method"] == method and r["benchmark_id"] == cell
                ),
                key=lambda r: r["repetition"],
            )
            if [r["repetition"] for r in subset] != [0, 1, 2]:
                raise ValueError("require complete three-repetition roster")
            if cell not in config["cells"]:
                continue
            scores, evidence_counts = [], Counter()
            for row in subset:
                evidence = row.get("fitted_equation_requirements", {}).get("counts")
                if evidence:
                    scores.append(evidence["passed_fraction"])
                    evidence_counts.update(
                        {k: evidence[k] for k in ("pass", "fail", "unresolved")}
                    )
                else:
                    # Zero confirmed evidence, not a claim all predicates are false.
                    scores.append(0.0)
                    evidence_counts["unassessed"] += len(
                        config["cells"][cell]["requirements"]
                    )
            cases.append(
                {
                    "benchmark_id": cell,
                    "confirmed_fraction": statistics(scores),
                    "repetition_fractions": scores,
                    "evidence_counts": dict(evidence_counts),
                    "assessed_runs": sum(r["status"] == "ready" for r in subset),
                }
            )
        selected = [r for r in rows if r["method"] == method]
        result.append(
            {
                "method": method,
                "label": label,
                "cases": cases,
                "supported_case_count": len(cases),
                "planned_case_count": len(roster),
                "confirmed_fraction_macro": statistics(
                    [c["confirmed_fraction"]["median"] for c in cases]
                ),
                "all_predicates_pass_runs": sum(
                    r.get("fitted_equation_requirements", {})
                    .get("counts", {})
                    .get("passed_fraction")
                    == 1
                    for r in selected
                ),
                "status_counts": dict(Counter(r["status"] for r in selected)),
                "declaration_violation_models": sum(
                    bool(r.get("declaration_violations")) for r in selected
                ),
            }
        )
    return result


def write_report(report: dict, root: Path) -> None:
    """Keep the scope next to every numerical result."""
    lines = [
        "# Fitted public-equation evidence",
        "",
        "Necessary equation predicates only; not overall mechanism compliance.",
        "Seven supported cases; T2-easy/hard are unbound and excluded explicitly.",
        "No fitting, LLM calls, numerical rollouts or trajectory reads.",
        "",
        "Case medians over three repetitions, then median and unscaled MAD "
        "over seven cases.",
        "Unresolved/missing evidence contributes zero confirmed passes, "
        "not a scientific failure verdict.",
        "",
        "| Method | Confirmed predicates: median [MAD] "
        "| All predicates pass / 21 planned runs |",
        "| --- | ---: | ---: |",
    ]
    for m in report["methods"]:
        s = m["confirmed_fraction_macro"]
        lines.append(
            f"| {m['label']} | {100 * s['median']:.1f}% [{100 * s['mad']:.1f} pp] | "
            f"{m['all_predicates_pass_runs']}/21 |"
        )
    lines += [
        "",
        "| Case | SINDy | PySR | D3 | GPT-5.6 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for i, case in enumerate(report["methods"][0]["cases"]):
        values = [m["cases"][i] for m in report["methods"]]
        cells = [case["benchmark_id"].removeprefix("phase_b_")]
        cells += [
            f"{100 * v['confirmed_fraction']['median']:.1f}% "
            f"[{100 * v['confirmed_fraction']['mad']:.1f} pp]; "
            f"{v['assessed_runs']}/3 assessed"
            for v in values
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Limits", ""] + [f"- {x}" for x in report["limitations"]]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, manifest, integrity = load_package(args.package)
    if manifest["schema_version"] != "phase-b-results-package-2":
        raise ValueError("require v2 fitted-model package")
    roster = read_roster(
        REPO / "scripts/build_experiments_table_results.py",
        REPO / "configs/final_component_campaign_v1.json",
    )
    # Validate the full roster independently of equation-assessment coverage.
    summarize(rows, roster)
    config = assessment_config()
    adapted = [adapt(r, config) for r in rows if r["benchmark_id"] in roster]
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = sealed_write(
        args.output / "models.json",
        {
            "protocol": BUNDLE_PROTOCOL,
            "rows": adapted,
            "package_sha256": integrity["archive_sha256"],
            "test_data_opened": False,
            "parameter_refit_applied": False,
        },
    )
    report = sealed_write(
        args.output / "summary.json",
        {
            "protocol": PROTOCOL,
            "bundle_sha256": bundle["artifact_sha256"],
            "integrity": integrity,
            "roster": list(roster),
            "status_counts": dict(Counter(r["status"] for r in adapted)),
            "methods": aggregate(adapted, roster, config),
            "overall_mechanism_compliance": None,
            "llm_calls": 0,
            "optimizer_calls": 0,
            "solver_rollouts": 0,
            "test_data_opened": False,
            "limitations": [
                "T2 requires a reviewed delayed-insulin-action binding; treating "
                "generated I as supplied forcing would be wrong.",
                "Nonzero dependencies, additive CSTR channels and dynamic-memory "
                "witnesses do not certify signs, balance, or scientific correctness.",
                "D3 uses native increments with clamped auxiliaries; their "
                "discarded equations cannot supply fictitious paths.",
                "Runtime validity and parameter-declaration violations are separate "
                "diagnostics. Declaration sets differ across methods.",
                "Absent source prompt hashes are marked case_roster_only; numerical "
                "preparation must verify the pinned public prompt and channel roles.",
                "Legacy SINDy/PySR/agent execution is inferred from recognized "
                "adapter contracts, not asserted as an explicit saved field.",
                "Training/validation activity probes and response checks remain "
                "unexecuted. This archive contains no trajectories.",
                "Existing held-out NMSE records are not used to choose models, "
                "predicates, or numerical probes.",
            ],
        },
    )
    (args.output / "assessment_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    write_report(report, args.output)
    print(
        json.dumps(
            {
                "status_counts": report["status_counts"],
                "supported_cases": len(config["cells"]),
                "overall_mechanism_compliance": None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
