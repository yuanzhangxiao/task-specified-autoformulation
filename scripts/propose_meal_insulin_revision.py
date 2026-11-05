#!/usr/bin/env python3
"""One explicit, cached insulin-law revision after a verified child training fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from html import escape
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from pydantic import Field

from autoformalism.fitting.public_fitting import _lock, content_sha256
from autoformalism.llm.construction import MEASURED_RESERVATION, ConstructionClient
from autoformalism.llm.response_revision import PromptPreflightError
from autoformalism.llm.staged_topology import (
    StagedModelSettings,
    strict_provider_schema,
    visible_response,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research import meal_revision_metrics as metrics
from autoformalism.research.construction_repair_identity import family_identity
from autoformalism.research.construction_revision_edits import (
    LawEdit,
    RevisionValidationError,
    model_view,
)
from autoformalism.research.construction_structural_revision import (
    NewLatentState,
    NewProcess,
    apply_structural_revision,
)
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit
from autoformalism.schemas.staged_functions import FunctionParameter
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    validate_evidence,
)
from autoformalism.staged_topology import content_hash
from scripts import structural_revision_campaign as structural

PROTOCOL = "candidate-15-postmeal-revision-1"
SYSTEM = """Revise one law in the post-meal glucose and insulin model.
The direct ingestion coefficient in the insulin balance has already been freed and
refitted. Use the fitted child's training residuals to motivate your change.
When ingestion ends, what could sustain insulin production until the observed peak?
Replace one existing right-hand side (I or insulin_production). You choose the
scientific dependence and function. If needed, include one connected new latent
state with a causal initializer, one connected algebraic process, and shared parameters.
Preserve the meal-to-glucose path and delayed insulin-to-disposal path. Explain
the mechanism you preserve and predict changes in insulin peak timing and height.
Use only current supplied inputs/auxiliaries and generated model state in the
dynamics. The observations and conditional replay, when present, are evidence.
Return the complete law and its sources using the response schema. Return empty
lists for unnecessary supporting declarations. The estimated parameters shown
describe the fitted model and can be refitted during a later comparison."""


class InsulinRevision(StrictSchema):
    """One existing law and its supporting declarations form one scientific edit."""

    hypothesis: str = Field(min_length=1, max_length=1500)
    mechanism_preserved: str = Field(min_length=1, max_length=1000)
    expected_peak_timing: str = Field(min_length=1, max_length=800)
    expected_peak_height: str = Field(min_length=1, max_length=800)
    edit: LawEdit
    new_states: tuple[NewLatentState, ...] = Field(max_length=1)
    new_processes: tuple[NewProcess, ...] = Field(max_length=1)
    new_parameters: tuple[FunctionParameter, ...] = Field(max_length=8)


def _source() -> dict:
    paths = [Path(__file__), Path(structural.__file__), Path(structural.old.__file__)]
    return {
        "package": runtime_source_hash(),
        **{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }


def _number(value) -> float:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("finite training measurement required")
    return float(value)


def _training_feedback(comparison: dict, training_ids: set[str]) -> dict:
    """Project measured public training values and exclude auxiliary report fields."""
    routing = comparison["routing"]
    if (
        routing["eligible_for_proposer"] is not True
        or routing["status"] != "persistent_early_peak"
    ):
        raise ValueError("child training evidence does not trigger a revision")
    metrics = comparison["child_metrics"]
    witnesses = []
    for row in routing["witnesses"]:
        trajectory = row["trajectory"]
        if trajectory not in training_ids or row["external_insulin_zero"] is not True:
            raise ValueError(
                "witness must be a training row with zero external insulin"
            )
        (event,) = row["meal_events"]
        witnesses.append(
            {
                "trajectory": trajectory,
                "meal": {
                    k: _number(event[k]) for k in ("start_min", "end_min", "dose_g")
                },
                **{
                    k: _number(row[k])
                    for k in (
                        "observed_insulin_peak_min",
                        "predicted_insulin_peak_min",
                        "observed_insulin_max",
                        "predicted_insulin_max",
                        "grid_tolerance_min",
                    )
                },
            }
        )
    if len({w["trajectory"] for w in witnesses}) != 3 or len(witnesses) != 3:
        raise ValueError("three distinct shifted-meal witnesses required")
    result = {
        "scope": "Training reconstruction with previously fitted training initials.",
        "child_training_nmse": _number(metrics["training_nmse"]),
        "child_target_training_nmse": {
            k: _number(metrics["target_training_nmse"][k]) for k in ("Gp", "I", "U")
        },
        "shifted_meals_with_zero_external_insulin": witnesses,
    }
    diagnostic = metrics.get("conditional_postmeal")
    if diagnostic and diagnostic.get("rows"):
        entries = []
        for row in diagnostic["rows"]:
            if row["trajectory"] not in training_ids:
                raise ValueError("conditional replay contains nontraining trajectory")
            entries.append(
                {
                    "trajectory": row["trajectory"],
                    **{
                        k: _number(row[k])
                        for k in (
                            "meal_end_min",
                            "observed_peak_min",
                            "observed_I_at_meal_end",
                            "observed_I_at_peak",
                            "conditional_I_at_peak",
                        )
                    },
                }
            )
        result["conditional_postmeal"] = {
            "interpretation": "At the observed insulin value at meal end, integrate "
            "the child's fitted insulin balance using interpolated observed glucose "
            "until the observed insulin peak; external inputs are zero.",
            "rows": entries,
        }
    return result


def prompt_payload(problem, brief, scientific_aliases, feedback) -> dict:
    """Render the prompt fields for a measured request or a labelled offline preview."""
    model = model_view(problem, scientific_aliases=scientific_aliases)
    model.pop("frozen_fields")
    return {
        "public_task": brief.scientific_context,
        "current_model": model,
        "fitted_shared_coefficients": dict(problem.parameter_guesses),
        "training_evidence": feedback,
    }


def write_prompt_html(path: Path, request: dict, *, illustrative=False) -> None:
    """Render the visible messages and response schema without creating a run plan."""
    note = (
        "Illustrative preview using the parent fit at the nested starting point "
        "c_I=1. The actual request uses the verified refitted child."
        if illustrative
        else "Prepared from the verified fitted child."
    )
    schema = strict_provider_schema(InsulinRevision.model_json_schema())
    sections = [
        ("System", SYSTEM),
        ("Training evidence", request["training_evidence"]),
        ("Full user message", request),
        ("Response schema", schema),
    ]
    body = "".join(
        "<details"
        + (" open" if i < 2 else "")
        + "><summary>"
        + escape(label)
        + "</summary><pre>"
        + escape(text if isinstance(text, str) else json.dumps(text, indent=2))
        + "</pre></details>"
        for i, (label, text) in enumerate(sections)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<!doctype html><meta charset="utf-8"><title>Insulin revision prompt</title>'
        "<style>body{font:16px/1.5 system-ui;max-width:960px;margin:3rem auto}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f6;padding:1rem}"
        "summary{cursor:pointer;font-weight:bold}</style>"
        "<h1>Insulin revision prompt</h1>"
        "<p>" + note + "</p>" + body
    )


def prepare(
    root: Path,
    *,
    fitted_problem,
    public_brief,
    scientific_aliases,
    consistency_evidence,
    comparison,
    result_identity,
    fitting_contract,
    model_size="120b",
) -> dict:
    """Freeze the completed fit's eligible training evidence as a portable packet."""
    problem = PrefixFitProblem.model_validate(fitted_problem)
    evidence = TrainingCompatibilityEvidence.model_validate(consistency_evidence)
    validate_evidence(evidence)
    if evidence.context != problem.context:
        raise ValueError("fit problem differs from the training evidence context")
    brief = PublicScientificBrief.model_validate(public_brief)
    feedback = _training_feedback(
        comparison, {t.trajectory_id for t in evidence.trajectories}
    )
    request = prompt_payload(problem, brief, scientific_aliases, feedback)
    if (
        fitting_contract["training_sha256"]
        != comparison["child_metrics"]["training_sha256"]
    ):
        raise ValueError("refit contract and comparison training identities differ")
    settings = StagedModelSettings(
        model=structural.MODEL_SPECS[model_size][0],
        model_revision="loopback_alias_revision_unverified",
        maximum_requests=1,
        attempts_per_step=1,
        max_output_tokens=4096,
        maximum_total_tokens=32768,
    )
    with _lock(root):
        inputs = sealed_write(
            root / "inputs.json",
            {
                "problem": problem.model_dump(mode="json"),
                "brief": brief.model_dump(mode="json"),
                "aliases": scientific_aliases,
                "evidence": evidence.model_dump(mode="json"),
                "result_identity": result_identity,
                "fitting_contract": fitting_contract,
                "fitted_child_metrics": comparison["child_metrics"],
                "system": SYSTEM,
                "user": json.dumps(request, ensure_ascii=False, separators=(",", ":")),
                "response_schema": strict_provider_schema(
                    InsulinRevision.model_json_schema()
                ),
            },
        )
        return sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "source": _source(),
                "inputs_sha256": inputs["artifact_sha256"],
                "settings": settings.model_dump(mode="json"),
                "seed": 0,
            },
        )


def _load(root: Path):
    plan = sealed_read(root / "plan.json")
    inputs = sealed_read(root / "inputs.json")
    if (
        plan["protocol"] != PROTOCOL
        or plan["source"] != _source()
        or inputs["artifact_sha256"] != plan["inputs_sha256"]
        or inputs["response_schema"]
        != strict_provider_schema(InsulinRevision.model_json_schema())
        or inputs["system"] != SYSTEM
    ):
        raise ValueError("frozen proposer plan or source differs")
    return plan, inputs


def assess(inputs: dict, raw: dict) -> dict:
    """Check the single proposed revision using the current construction gate."""
    response = InsulinRevision.model_validate(raw)
    if response.edit.component not in {"I", "insulin_production"}:
        raise ValueError("one existing edit to I or insulin_production required")
    problem = PrefixFitProblem.model_validate(inputs["problem"])
    reply = {
        "decision": "revise",
        "hypothesis": response.hypothesis,
        "evidence_ids": ["public_task", "postmeal_training_replay"],
        "expected_observable_effect": response.expected_peak_timing
        + " "
        + response.expected_peak_height,
        "edits": [response.edit.model_dump(mode="json")],
        **response.model_dump(
            mode="json", include={"new_states", "new_processes", "new_parameters"}
        ),
    }
    applied = apply_structural_revision(
        problem,
        reply,
        evidence_ids=reply["evidence_ids"],
        brief=PublicScientificBrief.model_validate(inputs["brief"]),
        scientific_aliases=inputs["aliases"],
        normalize_parent_declarations=True,
    )
    child, audit = applied["problem"], applied["audit"]
    if audit["static_eligibility"] != "eligible":
        raise ValueError("candidate lacks a checked public mechanism path")
    equations = model_view(child)["equations"]
    dependencies = {e["component"]: e["sources"] for e in equations}
    active, pending = set(), ["I"]
    while pending:
        symbol = pending.pop()
        if symbol not in active:
            active.add(symbol)
            pending.extend(dependencies.get(symbol, []))
    if {s.name for s in (*response.new_states, *response.new_processes)} - active:
        raise ValueError("new components must contribute to the insulin balance")
    old_id = family_identity(problem, problem).get("sha256")
    if old_id and old_id == family_identity(problem, child).get("sha256"):
        raise ValueError("candidate repeats the original model")
    consistency = structural._assessment(
        {"_consistency_evidence": inputs["evidence"]},
        child,
        weighted_symbolic_initials=True,
        combined_symbolic_initials=True,
        common_zero_forcing=True,
        admission_guard=True,
        scaled_readout_initials=True,
    )
    if consistency["status"] == "conflict":
        raise RevisionValidationError(
            "QUALIFIED_MODEL_DATA_CONFLICT",
            "a qualified training witness contradicts the revision",
            assessment=consistency,
        )
    return {
        "status": "eligible_for_refit",
        "problem": child.model_dump(mode="json"),
        "audit": audit,
        "training_consistency": consistency,
        "proposal": response.model_dump(mode="json"),
    }


def _server(endpoint: str, expected: str) -> dict:
    """Check the alias and usable context advertised by the running local server."""
    with urlopen(endpoint + "/v1/models", timeout=15) as stream:
        models = json.load(stream)["data"]
    with urlopen(endpoint + "/version", timeout=15) as stream:
        version = json.load(stream)["version"]
    if (
        len(models) != 1
        or models[0]["id"] != expected
        or models[0].get("max_model_len") != 32768
    ):
        raise ValueError("loopback model alias or context differs from the plan")
    return {
        "alias": expected,
        "context": models[0]["max_model_len"],
        "vllm_version": version,
        "model_revision_verified": False,
    }


def _one_call(root, plan, inputs, endpoint, transport, token_transport):
    """Recover an exact call or durably reserve the sole physical generation."""
    settings = StagedModelSettings.model_validate(plan["settings"])
    key = plan["artifact_sha256"]
    directory = root / "calls"
    prior = root / "attempt.json"
    files = list(directory.glob("*.json"))
    arguments = {
        "system": inputs["system"],
        "user": inputs["user"],
        "response_model": InsulinRevision,
        "step": "insulin_law",
        "attempt": 0,
    }
    if prior.exists():
        if sealed_read(prior)["plan_sha256"] != key:
            raise ValueError("reservation plan differs")
        if not files:
            return {"status": "interrupted_after_reservation"}
        if len(files) != 1:
            raise ValueError("one-call budget exceeded")
        record = json.loads(files[0].read_text())
        identity = record["request"]
        if (
            files[0].stem != content_hash(identity)
            or record["request_hash"] != files[0].stem
            or identity["namespace"] != key
            or identity["settings"] != plan["settings"]
            or identity["body"]["seed"]
            != int(content_hash([plan["seed"], "insulin_law", 0])[:8], 16) % 2**31
        ):
            raise ValueError("cached request differs from sealed input")
        return structural._ReplayClient([record], settings).call(**arguments)
    if files:
        raise ValueError("cached generation lacks a reservation")
    serving = (
        _server(endpoint, settings.model)
        if transport is None
        else {"alias": settings.model, "test_transport": True}
    )
    sealed_write(prior, {"plan_sha256": key, "serving": serving})
    client = ConstructionClient(
        settings=settings,
        base_url=endpoint,
        directory=directory,
        namespace=key,
        seed=plan["seed"],
        reservation_policy=MEASURED_RESERVATION,
        token_transport=token_transport,
        **({"transport": transport} if transport else {}),
    )
    try:
        return client.call(**arguments)
    except PromptPreflightError as exc:
        return {"status": "preflight_blocked", "error": str(exc)[:3000]}


def run(root: Path, endpoint: str, *, transport=None, token_transport=None) -> dict:
    """Publish one cached physical outcome and any admitted structural patch."""
    url = urlsplit(endpoint)
    if (
        url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or url.username is not None
        or url.query
        or url.fragment
        or url.path not in ("", "/")
    ):
        raise ValueError("requires an unauthenticated loopback HTTP endpoint")
    with _lock(root):
        plan, inputs = _load(root)
        key = plan["artifact_sha256"]
        terminal = root / "result.json"
        directory = root / "calls"
        if terminal.exists():
            result = sealed_read(terminal)
            if result["plan_sha256"] != key or result["call_files"] != _calls(
                directory
            ):
                raise ValueError("cached result or calls differ from the plan")
            return result
        record = _one_call(
            root, plan, inputs, endpoint.rstrip("/"), transport, token_transport
        )
        if record["status"] == "responded":
            try:
                outcome = assess(inputs, visible_response(record))
            except (ValueError, KeyError, TypeError) as exc:
                outcome = {
                    "status": "rejected",
                    "problem": None,
                    "error": str(exc)[:3000],
                    "details": getattr(exc, "details", {}),
                }
        else:
            outcome = {
                "status": record["status"]
                if record["status"] != "inflight"
                else "interrupted_provider_outcome",
                "problem": None,
                "error": record.get("error"),
            }
        if outcome["status"] == "eligible_for_refit":
            outcome["refit_handoff"] = {
                "problem_sha256": content_sha256(outcome["problem"]),
                "fitted_parent_reference": inputs["result_identity"],
                "fitting_contract": inputs["fitting_contract"],
                "numerical_status": "pending_general_refit",
                "comparison": "Use a qualified backend for the revised dynamics. "
                "Preserve "
                "the public training rows, initialization scopes, matched resource "
                "limits and original-equation replay. Compare the new fit with the "
                "retained nested child using compare_refit_replay.",
            }
        return sealed_write(
            terminal,
            {
                **outcome,
                "plan_sha256": key,
                "call_files": _calls(directory),
                "automatic_fit": False,
                "test_data_opened": False,
            },
        )


def compare_refit_replay(
    root: Path,
    *,
    problem_sha256: str,
    training: PublicSplit,
    channels,
    scales,
    residual,
    verified_training_nmse: float,
) -> dict:
    """Score an independently verified general-backend original-equation residual."""
    plan = sealed_read(root / "plan.json")
    inputs = sealed_read(root / "inputs.json")
    accepted = sealed_read(root / "result.json")
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != inputs["artifact_sha256"]
        or accepted["plan_sha256"] != plan["artifact_sha256"]
        or accepted["status"] != "eligible_for_refit"
        or problem_sha256 != content_sha256(accepted["problem"])
        or content_sha256(training) != inputs["fitting_contract"]["training_sha256"]
    ):
        raise ValueError("refit replay and admitted training problem differ")
    revised = metrics.summarize(training, channels, scales, residual)
    if not math.isclose(
        revised["training_nmse"],
        _number(verified_training_nmse),
        rel_tol=1e-8,
        abs_tol=1e-10,
    ):
        raise ValueError("residual and verified training NMSE differ")
    baseline = inputs["fitted_child_metrics"]
    return {
        "status": "scored_training_replay",
        "refit_problem_sha256": problem_sha256,
        "fitted_parent_reference": inputs["result_identity"],
        "metrics": revised,
        "comparison": metrics.compare(baseline, revised),
        "report": metrics.render_comparison(baseline, revised),
        "automatic_followup": False,
    }


def _calls(directory: Path) -> dict:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.glob("*.json")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preview", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument(
        "--model-size", choices=tuple(structural.MODEL_SPECS), default="120b"
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--html", type=Path, help="optional HTML path for preview")
    args = parser.parse_args()
    if args.html and args.action != "preview":
        parser.error("--html belongs to preview")
    if args.action == "prepare":
        if args.campaign is None:
            parser.error("prepare requires --campaign")
        from autoformalism.research.insulin_meal_campaign import proposer_inputs

        result = prepare(
            args.root, model_size=args.model_size, **proposer_inputs(args.campaign)
        )
    elif args.action == "preview":
        _, inputs = _load(args.root)
        result = {k: inputs[k] for k in ("system", "user", "response_schema")}
        result["user"] = json.loads(result["user"])
        if args.html:
            write_prompt_html(args.html, result["user"])
    else:
        receipt = run(args.root, args.endpoint)
        result = {
            k: receipt[k]
            for k in ("status", "plan_sha256", "call_files", "automatic_fit")
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
