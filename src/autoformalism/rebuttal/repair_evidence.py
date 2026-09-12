"""Versioned observations for repair, never deterministic scientific explanations."""

from __future__ import annotations

import ast
import math
from typing import Any, Literal

from pydantic import Field

from autoformalism.expressions import RestrictedParser, ValidationContext
from autoformalism.expressions.intervals import (
    UNKNOWN_INTERVAL,
    Interval,
    analyze_interval,
)
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash


class Finding(StrictSchema):
    """A source-labelled, candidate-bound observation and its recheck contract."""

    source: Literal["runtime", "scientific_judge", "fitter"]
    stage: Literal["prefit", "postfit"]
    candidate_sha256: str
    category: str
    code: str
    component: str | None = None
    certainty: Literal["certified", "observed", "unresolved", "advisory"]
    observation: str
    recheck: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    blocking: bool = False


def model_hash(candidate: CandidateModel) -> str:
    """Content identity includes initial conditions, not just the RHS."""
    payload = candidate.model_dump(mode="json")
    for field in ("candidate_id", "parent_candidate_id", "change_summary"):
        payload.pop(field, None)
    for field, key in (
        ("states", "name"),
        ("processes", "name"),
        ("parameters", "name"),
        ("state_equations", "state"),
        ("initial_conditions", "state"),
        ("observation_mappings", "channel"),
    ):
        for entry in payload[field]:
            for expression in ("rhs", "expression"):
                if entry.get(expression):
                    entry[expression] = ast.dump(
                        RestrictedParser()
                        .parse(entry[expression], location="identity")
                        .tree,
                        include_attributes=False,
                    )
        payload[field].sort(key=lambda e: e[key])
    return content_hash(payload)


def memory_path_findings(
    candidate: CandidateModel, context: ValidationContext, targets: tuple[str, ...]
) -> list[Finding]:
    """Necessary graph condition only: an internal dynamic input-to-target path.

    Call only for targets whose frozen public requirement explicitly asks for
    internal dynamic memory. This is not a scientific mechanism-compliance score.
    """
    definitions = {e.state: e.rhs for e in candidate.state_equations} | {
        p.name: p.expression for p in candidate.processes
    }
    edges = {
        name: set(RestrictedParser().parse(text, location=name).symbols)
        for name, text in definitions.items()
    }

    def ancestors(symbols):
        found, pending = set(), list(symbols)
        while pending:
            symbol = pending.pop()
            if symbol not in found:
                found.add(symbol)
                pending.extend(edges.get(symbol, set()) - found)
        return found

    direct = {
        m.expression
        for m in candidate.observation_mappings
        if m.expression in definitions
    }
    internal_states = {s.name for s in candidate.states} - direct
    findings = []
    for mapping in candidate.observation_mappings:
        if mapping.channel not in targets:
            continue
        upstream = ancestors(
            RestrictedParser()
            .parse(mapping.expression, location=mapping.channel)
            .symbols
        )
        witnesses = [
            s
            for s in internal_states & upstream
            if set(context.external_inputs) & ancestors({s})
        ]
        if not witnesses:
            findings.append(
                Finding(
                    source="runtime",
                    stage="prefit",
                    candidate_sha256=model_hash(candidate),
                    category="public_graph_obligation",
                    code="INTERNAL_MEMORY_PATH_ABSENT",
                    component=mapping.channel,
                    certainty="certified",
                    blocking=True,
                    observation=(
                        "No internal dynamic state lies on a declared input-to-target "
                        "path; the frozen public task requires internal memory."
                    ),
                    recheck=(
                        "dependency closure and presence of an internal dynamic state, "
                        "not scientific correctness"
                    ),
                )
            )
    return findings


def domain_findings(
    candidate: CandidateModel, context: ValidationContext
) -> list[Finding]:
    """Audit division/log/sqrt/integer powers with explicit uncertainty.

    Unknown generated ranges are not inferred from training samples. A possible
    pole is not asserted reachable. The restricted grammar disallows variable
    and fractional powers; their parser errors remain contract findings.
    """
    identity = model_hash(candidate)
    intervals = {k: Interval(*v) for k, v in context.forcing_bounds.items()}
    for p in candidate.parameters:
        intervals[p.name] = (
            Interval(0, math.inf) if p.domain.value != "real" else UNKNOWN_INTERVAL
        )
    expressions = [
        *((p.name, p.expression) for p in candidate.processes),
        *((e.state, e.rhs) for e in candidate.state_equations),
        *(
            (f"mapping:{m.channel}", m.expression)
            for m in candidate.observation_mappings
        ),
        *(
            (f"initial:{i.state}", i.expression)
            for i in candidate.initial_conditions
            if i.expression
        ),
    ]
    result = []
    for component, expression in expressions:
        parsed = RestrictedParser().parse(expression, location=component)
        for node in ast.walk(parsed.tree):
            subject = None
            rule = ""
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                subject, rule = node.right, "nonzero"
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                exponent = ast.unparse(node.right)
                if exponent.startswith("-"):
                    subject, rule = node.left, "nonzero"
            elif isinstance(node, ast.Call) and node.func.id in {"log", "sqrt"}:
                subject = node.args[0]
                rule = "positive" if node.func.id == "log" else "nonnegative"
            if subject is None:
                continue
            try:
                span = analyze_interval(
                    subject, intervals, location=component, diagnostics=[]
                )
            except (ArithmeticError, ValueError):
                span = UNKNOWN_INTERVAL
            safe = (
                span.lower > 0 or span.upper < 0
                if rule == "nonzero"
                else span.lower > 0
                if rule == "positive"
                else span.lower >= 0
            )
            invalid = (
                span.lower == span.upper == 0
                if rule == "nonzero"
                else span.upper <= 0
                if rule == "positive"
                else span.upper < 0
            )
            if safe:
                continue
            formula = ast.unparse(subject)
            result.append(
                Finding(
                    source="runtime",
                    stage="prefit",
                    candidate_sha256=identity,
                    category="domain",
                    code="DOMAIN_INVALID" if invalid else "DOMAIN_UNRESOLVED",
                    component=component,
                    certainty="certified" if invalid else "unresolved",
                    observation=f"{formula} must be {rule}; "
                    + (
                        "its certified interval violates this requirement"
                        if invalid
                        else "domain not certified; reachability is unknown"
                    ),
                    recheck="restricted-domain-audit plus actual numerical evaluation",
                    evidence={
                        "expression": ast.unparse(node),
                        "subject": formula,
                        "requires": rule,
                        "interval": [str(span.lower), str(span.upper)],
                    },
                    blocking=invalid,
                )
            )
    return result


def numerical_findings(candidate: CandidateModel, fit: dict) -> list[Finding]:
    """Summarize observations; timeouts cannot be relabelled mechanism defects."""
    result = []
    identity = model_hash(candidate)
    init, refinement = fit.get("initializer") or {}, fit.get("refinement") or {}
    for code, observation, evidence in (
        ("INITIALIZER_UNAVAILABLE", init.get("message"), init),
        ("REFINEMENT_STATUS", refinement.get("message"), refinement),
    ):
        if observation:
            result.append(
                Finding(
                    source="fitter",
                    stage="postfit",
                    candidate_sha256=identity,
                    category="numerical",
                    code=code,
                    certainty="observed",
                    observation=observation,
                    recheck="same frozen fitter and training data",
                    evidence=compact_numerics(evidence),
                )
            )
    result.append(
        Finding(
            source="fitter",
            stage="postfit",
            candidate_sha256=identity,
            category="prediction" if fit.get("status") == "complete" else "numerical",
            code="FINITE_ROLLOUT"
            if fit.get("status") == "complete"
            else "FIT_UNAVAILABLE",
            certainty="observed",
            observation=(
                "Fresh causal rollouts are finite, not scientific recovery."
                if fit.get("status") == "complete"
                else "No verified fit under this budget; the cause is unresolved."
            ),
            recheck="fresh causal training and validation rollout",
            evidence={
                k: compact_numerics(fit.get(k))
                if isinstance(fit.get(k), dict)
                else fit.get(k)
                for k in ("status", "failure_class", "error", "training", "validation")
            },
        )
    )
    return result


def compact_numerics(raw: dict) -> dict:
    """Bound the next request; full solver journals remain in fit artifacts."""
    fields = (
        "status",
        "success",
        "message",
        "numerical_status",
        "optimizer_success",
        "optimizer_native_success",
        "production_training_rollout_verified",
        "valid_residual_evaluations",
        "actual_residual_calls",
        "selected_training_rollout_verified",
        "normalized_mse",
        "failed_trajectories",
        "parameters",
        "error",
        "failure_class",
        "wall_seconds",
        "iterations",
    )
    result = {k: raw[k] for k in fields if k in raw}
    for key, value in list(result.items()):
        if isinstance(value, list):
            result[key] = value[:16]
            result[key + "_total_count"] = len(value)
        elif isinstance(value, str):
            result[key] = value[:2000]
    return result


def decision_report(
    candidate: CandidateModel,
    runtime: list[Finding],
    science: list[Finding],
    last_numerical: list[Finding],
    history: list[dict],
) -> dict:
    """Choose one problem category, not a causal equation or mandatory edit."""
    categories = (
        ("runtime_contract", [f for f in runtime if f.blocking]),
        ("scientific_requirement", [f for f in science if f.certainty == "advisory"]),
        ("domain_uncertainty", [f for f in runtime if f.category == "domain"]),
        (
            "numerical_feasibility",
            [f for f in last_numerical if f.code == "FIT_UNAVAILABLE"],
        ),
        ("prediction_refinement", last_numerical),
    )
    category, focus = next(
        ((c, fs) for c, fs in categories if fs), ("model_review", [])
    )
    unchanged = sum(r.get("outcome") == "no_change" for r in history[-2:])
    return {
        "schema_version": "repair-decision-report-1",
        "candidate_sha256": model_hash(candidate),
        "objective_category": category,
        "scope_policy": "consider_other_components_or_explicit_topology"
        if unchanged
        else "small_coherent_edit",
        "focus": [f.model_dump(mode="json") for f in focus],
        "runtime_findings": [f.model_dump(mode="json") for f in runtime],
        "scientific_findings": [f.model_dump(mode="json") for f in science],
        "last_numerical_evidence": [f.model_dump(mode="json") for f in last_numerical],
        "recent_actions": history[-4:],
        "cause_is_not_determined_by_runtime": True,
        "hypothesis_and_repair_owned_by_proposer": True,
    }
