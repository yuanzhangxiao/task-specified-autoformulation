#!/usr/bin/env python3
"""One explicit, cached insulin-law revision after a verified child training fit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import tarfile
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
from autoformalism.research import insulin_boundary_revision as boundary
from autoformalism.research import meal_revision_metrics as metrics
from autoformalism.research import revision_spelling
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
INTAKE_PROTOCOL = "candidate-15-reviewed-revision-intake-1"
REVIEWED_INTAKE_SHA256 = (
    "fc5372e439b37f4ff7f0a93dce071f0480013e3b612fa00c4045d93c95bddbfe"
)
_MEAL_INSTRUCTION = """Revise one law in the post-meal glucose and insulin model.
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
_EDIT_FORMAT = """\
edit.expression replaces the whole named RHS; keep every contribution you intend
to retain. Write only the RHS using + - * / **, parentheses, and the supported
functions abs, exp, log, max, min, sigmoid, sin, softplus, sqrt, tanh.
sources lists variable names, not coefficient names. Declare new dynamics
coefficients in new_parameters and preserve existing parameter names and roles.
New states and processes need distinct names. Initial maps use explicit aliases
for public time-zero readings or other state initials; their parameters are
initializer-local, separate from dynamics parameters with the same spelling.
For a fixed initial constant, use mode=map, a numeric expression and empty lists.
Initialization scopes and the permitted observation prefix are shown below."""
SYSTEM = _MEAL_INSTRUCTION + "\n" + _EDIT_FORMAT
BOUNDARY_SYSTEM = boundary.INSTRUCTION + "\n" + _EDIT_FORMAT


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


class BoundaryRevision(InsulinRevision):
    """Explain the boundary and ingestion behavior of the proposed equations."""

    insulin_rate_at_zero: str = Field(min_length=1, max_length=1200)
    expected_ingestion_window_change: str = Field(min_length=1, max_length=800)


def _contract(protocol: str) -> tuple[str, type[InsulinRevision]]:
    if protocol == PROTOCOL:
        return SYSTEM, InsulinRevision
    if protocol == boundary.PROTOCOL:
        return BOUNDARY_SYSTEM, BoundaryRevision
    if protocol == boundary.REPAIR_PROTOCOL:
        return boundary.REPAIR_INSTRUCTION + "\n" + _EDIT_FORMAT, BoundaryRevision
    raise ValueError(f"unknown insulin revision protocol: {protocol}")


def _source() -> dict:
    paths = [
        Path(__file__),
        Path(__file__).with_name("describe_insulin_boundary.py"),
        Path(structural.__file__),
        Path(structural.old.__file__),
    ]
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


def write_prompt_html(
    path: Path, request: dict, *, illustrative=False, protocol=PROTOCOL
) -> None:
    """Render the visible messages and response schema without creating a run plan."""
    note = (
        "Illustrative preview using the parent fit at the nested starting point "
        "c_I=1. The actual request uses the verified refitted child."
        if illustrative
        else "Prepared from the verified fitted child."
    )
    system, response_model = _contract(protocol)
    schema = strict_provider_schema(response_model.model_json_schema())
    if protocol in boundary.BOUNDARY_PROTOCOLS:
        note = "Prepared from the primary concentration-constrained training fit."
    sections = [
        ("System", system),
        ("Training evidence", request["training_evidence"]),
        ("Full user message", request),
        ("Response schema", schema),
    ]
    if protocol == boundary.REPAIR_PROTOCOL:
        sections.insert(
            2,
            (
                "Previous attempt and deterministic feedback",
                request["previous_attempt"],
            ),
        )
    body = "".join(
        "<details"
        + (
            " open"
            if i < 2 or label == "Previous attempt and deterministic feedback"
            else ""
        )
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
    boundary_feedback=None,
    repair_history=None,
) -> dict:
    """Freeze the completed fit's eligible training evidence as a portable packet."""
    problem = PrefixFitProblem.model_validate(fitted_problem)
    evidence = TrainingCompatibilityEvidence.model_validate(consistency_evidence)
    validate_evidence(evidence)
    if evidence.context != problem.context:
        raise ValueError("fit problem differs from the training evidence context")
    brief = PublicScientificBrief.model_validate(public_brief)
    protocol = boundary.PROTOCOL if boundary_feedback is not None else PROTOCOL
    if repair_history is not None:
        if boundary_feedback is None or model_size != "120b":
            raise ValueError("feedback repair requires the 120B boundary comparison")
        protocol = boundary.REPAIR_PROTOCOL
    system, response_model = _contract(protocol)
    training_ids = {t.trajectory_id for t in evidence.trajectories}
    if boundary_feedback is None:
        feedback = _training_feedback(comparison, training_ids)
    else:
        witness = boundary_feedback["depletion"]
        feedback_ids = {witness["trajectory"]} | {
            row["trajectory"] for row in boundary_feedback["ingestion_windows"]
        }
        if not feedback_ids <= training_ids:
            raise ValueError("boundary feedback requires public training trajectories")
        feedback = boundary_feedback
    request = prompt_payload(problem, brief, scientific_aliases, feedback)
    if boundary_feedback is not None:
        # Put the actual witnesses first in the user message sent on this repair.
        request = {"training_evidence": feedback, **request}
    if repair_history is not None:
        previous = repair_history["previous_attempt"]
        if (
            repair_history["fitted_problem_sha256"] != content_sha256(problem)
            or previous["fixed_parameter_replay"]["trajectory"] not in training_ids
        ):
            raise ValueError("previous feedback differs from the fitted training case")
        request["training_evidence"] = {
            "deterministic_feedback": repair_history["fitted_feedback"],
            "depleted_state": witness["state_at_boundary"],
            "measured_I_at_300": witness["observed_I_pmol_L"],
            "fitted_I_at_300": witness["constrained_fit_I_pmol_L"],
            "zero_input_extension": feedback["zero_input_extension"],
        }
        # Keep the fitted parent and the previous, unfitted attempt distinguishable.
        request = {
            "training_evidence": request.pop("training_evidence"),
            "previous_attempt": previous,
            **request,
        }
    request["initialization_contract"] = {
        "shared_training": "One learned initial constant shared across trajectories.",
        "trajectory": "Independent training roots may be estimated from training "
        "trajectories. At validation/test, shared dynamics are frozen and these "
        "roots may use only the fixed observation prefix, never the full "
        "target history.",
        "map": "An explicit causal map of available time-zero readings and state "
        "initials. Initializer-local parameters are shared across trajectories.",
        "fixed_prefix_policy": fitting_contract["prefix_policy"],
    }
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
                **(
                    {
                        "repair_history": repair_history,
                        "boundary_feedback": boundary_feedback,
                    }
                    if repair_history is not None
                    else {}
                ),
                **(
                    {"revision_protocol": protocol, "boundary_witness": witness}
                    if boundary_feedback is not None
                    else {}
                ),
                "system": system,
                "user": json.dumps(request, ensure_ascii=False, separators=(",", ":")),
                "response_schema": strict_provider_schema(
                    response_model.model_json_schema()
                ),
            },
        )
        return sealed_write(
            root / "plan.json",
            {
                "protocol": protocol,
                "model_size": model_size,
                "source": _source(),
                "inputs_sha256": inputs["artifact_sha256"],
                "settings": settings.model_dump(mode="json"),
                "seed": 0,
                "optimizer_calls": 0,
                "test_data_opened": False,
                "automatic_followup": False,
            },
        )


def prepare_source(root: Path, source: Path, model_size="120b") -> dict:
    """Bind reviewed fit evidence to this launcher without rewriting old receipts."""
    from autoformalism.benchmarks.audited_release import read_seal

    intake = read_seal(source)
    if intake.get("protocol") == boundary.REPAIR_INTAKE_PROTOCOL:
        if content_sha256(intake) != boundary.REPAIR_INTAKE_SHA256:
            raise ValueError(
                "expected the reviewed deterministic feedback repair intake"
            )
        return prepare(root, model_size=model_size, **intake["arguments"])
    if intake.get("protocol") == boundary.INTAKE_PROTOCOL:
        if content_sha256(intake) != boundary.INTAKE_SHA256:
            raise ValueError("expected the reviewed insulin boundary intake")
        return prepare(root, model_size=model_size, **intake["arguments"])
    if (
        intake.get("protocol") != INTAKE_PROTOCOL
        or content_sha256(intake) != REVIEWED_INTAKE_SHA256
    ):
        raise ValueError("expected the reviewed candidate-15 fitting intake")
    return prepare(root, model_size=model_size, **intake["arguments"])


def _load(root: Path):
    plan = sealed_read(root / "plan.json")
    inputs = sealed_read(root / "inputs.json")
    system, response_model = _contract(plan["protocol"])
    if (
        plan["protocol"] != inputs.get("revision_protocol", PROTOCOL)
        or plan["source"] != _source()
        or inputs["artifact_sha256"] != plan["inputs_sha256"]
        or inputs["response_schema"]
        != strict_provider_schema(response_model.model_json_schema())
        or inputs["system"] != system
    ):
        raise ValueError("frozen proposer plan or source differs")
    return plan, inputs


def assess(inputs: dict, raw: dict) -> dict:
    """Check the single proposed revision using the current construction gate."""
    if inputs.get("revision_protocol") != boundary.REPAIR_PROTOCOL:
        return _assess(inputs, raw)
    from scripts.describe_insulin_boundary import revision_messages

    parent = PrefixFitProblem.model_validate(inputs["problem"])
    patch, spellings = revision_spelling.normalize(parent, raw, BoundaryRevision)
    try:
        outcome = _assess(inputs, patch)
        history = inputs["repair_history"]
        child = PrefixFitProblem.model_validate(outcome["problem"])
        if boundary.previous_counterexample_applies(parent, child, history):
            raise RevisionValidationError(
                "PREVIOUS_REPAIR_REPEATED",
                "the returned equations and initialization reproduce the previous "
                "negative-insulin counterexample at admissible parameter values",
                previous_family_sha256=history["previous_family_sha256"],
                counterexample=history["previous_attempt"]["fixed_parameter_replay"],
            )
    except RevisionValidationError as exc:
        exc.details["identifier_spellings"] = spellings
        raise
    except ValueError as exc:
        if not spellings:
            raise
        raise RevisionValidationError(
            "NORMALIZED_PATCH_REJECTED", str(exc), identifier_spellings=spellings
        ) from exc
    outcome["audit"]["identifier_spellings"] = spellings
    outcome["revision_feedback"] = revision_messages(
        outcome,
        inputs["boundary_witness"],
        inputs["boundary_feedback"]["ingestion_windows"],
    )
    return outcome


def _assess(inputs: dict, raw: dict) -> dict:
    """Apply the typed edit to the fitted model and check the resulting equations."""
    protocol = inputs.get("revision_protocol", PROTOCOL)
    _, response_model = _contract(protocol)
    response = response_model.model_validate(raw)
    if response.edit.component not in {"I", "insulin_production"}:
        raise ValueError("one existing edit to I or insulin_production required")
    problem = PrefixFitProblem.model_validate(inputs["problem"])
    reply = {
        "decision": "revise",
        "hypothesis": response.hypothesis,
        "evidence_ids": (
            ["public_task", "depletion", "ingestion_windows"]
            if protocol in boundary.BOUNDARY_PROTOCOLS
            else ["public_task", "postmeal_training_replay"]
        ),
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
    if (set(audit["new_states"]) | set(audit["new_processes"])) - active:
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
    boundary_result = {}
    if protocol in boundary.BOUNDARY_PROTOCOLS:
        boundary_result["insulin_boundary_check"] = boundary.check(
            problem, child, inputs["boundary_witness"]
        )
    return {
        **boundary_result,
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
        "response_model": _contract(plan["protocol"])[1],
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
            return _terminal(root, plan)
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
            if plan["protocol"] in boundary.BOUNDARY_PROTOCOLS:
                outcome["refit_handoff"]["comparison"] = (
                    "Compare against the primary constrained endpoint. Refit using "
                    "the declared concentration control and replay original equations. "
                    "Report dense minima, boundary rates, one-minute zero-input "
                    "extension, ingestion-window rises, shifted peaks, no-meal "
                    "controls and per-trajectory NMSE median/MAD on the same rows."
                )
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
        plan["protocol"] not in {PROTOCOL, *boundary.BOUNDARY_PROTOCOLS}
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


def _terminal(root: Path, plan: dict) -> dict:
    """Read a terminal only if its source plan and exact call files still match."""
    result = sealed_read(root / "result.json")
    if result["plan_sha256"] != plan["artifact_sha256"] or result[
        "call_files"
    ] != _calls(root / "calls"):
        raise ValueError("cached result or calls differ from the plan")
    return result


def status(root: Path) -> dict:
    """Report cached outcomes without contacting a server or reserving a call."""
    plan, inputs = _load(root)
    terminal = _terminal(root, plan) if (root / "result.json").exists() else None
    records = [
        json.loads(p.read_text()) for p in sorted((root / "calls").glob("*.json"))
    ]
    if len(records) > plan["settings"]["maximum_requests"]:
        raise ValueError("one-call budget exceeded")
    observed, missing = 0, 0
    for record in records:
        tokens = record.get("observed_total_tokens")
        if type(tokens) is int and tokens >= 0:
            observed += tokens
        else:
            missing += 1
    uncertain = (
        (root / "attempt.json").exists()
        and not records
        and (terminal is None or terminal["status"] == "interrupted_after_reservation")
    )
    return {
        "protocol": plan["protocol"],
        "plan_sha256": plan["artifact_sha256"],
        "status": terminal["status"]
        if terminal
        else (
            "reserved_without_terminal"
            if (root / "attempt.json").exists()
            else "pending"
        ),
        "terminal_available": terminal is not None,
        "physical_requests_started": len(records),
        "unrecorded_request_outcome_possible": uncertain,
        "observed_total_tokens": observed,
        "usage_missing_events": missing,
        "usage_complete": missing == 0 and not uncertain,
        "static_eligibility": (terminal or {})
        .get("audit", {})
        .get("static_eligibility"),
        "training_consistency": (terminal or {})
        .get("training_consistency", {})
        .get("status"),
        **(
            {
                "insulin_boundary_check": (terminal or {}).get("insulin_boundary_check")
                or (terminal or {}).get("details", {}).get("boundary_check")
            }
            if plan["protocol"] in boundary.BOUNDARY_PROTOCOLS
            else {}
        ),
        **(
            {
                "identifier_spellings": (terminal or {})
                .get("audit", {})
                .get(
                    "identifier_spellings",
                    (terminal or {}).get("details", {}).get("identifier_spellings", []),
                ),
                "revision_feedback": (terminal or {}).get("revision_feedback", []),
                "prior_attempt_plan_sha256": inputs["repair_history"][
                    "source_plan_sha256"
                ],
            }
            if plan["protocol"] == boundary.REPAIR_PROTOCOL
            else {}
        ),
        "error": (terminal or {}).get("error"),
        "optimizer_calls": 0,
        "test_data_opened": False,
        "automatic_followup": False,
    }


def report(root: Path) -> dict:
    """Render the complete reply and equation changes alongside admission evidence."""
    from autoformalism.llm.staged_topology import atomic_json

    summary = status(root)
    plan, inputs = _load(root)
    terminal = _terminal(root, plan) if summary["terminal_available"] else None
    before = model_view(
        PrefixFitProblem.model_validate(inputs["problem"]),
        scientific_aliases=inputs["aliases"],
    )
    sections = [
        ("Completion and cost", summary),
        ("Fitted model before revision", before),
    ]
    if terminal:
        sections.append(("Revision and deterministic assessment", terminal))
        if terminal.get("problem"):
            sections.append(
                (
                    "Model after revision",
                    model_view(
                        PrefixFitProblem.model_validate(terminal["problem"]),
                        scientific_aliases=inputs["aliases"],
                    ),
                )
            )
    for path in sorted((root / "calls").glob("*.json")):
        record = json.loads(path.read_text())
        sections.append(
            ("Complete recorded provider response", record.get("raw_response", record))
        )
    body = "".join(
        "<details open><summary>"
        + escape(title)
        + "</summary><pre>"
        + escape(json.dumps(value, indent=2))
        + "</pre></details>"
        for title, value in sections
    )
    repair_result = ""
    if plan["protocol"] == boundary.REPAIR_PROTOCOL:
        sign = (summary["insulin_boundary_check"] or {}).get(
            "boundary_sign", "unassessed"
        )
        repair_result = (
            f"The zero-concentration boundary assessment is {sign}. "
            f"Resolved {len(summary['identifier_spellings'])} identifier spellings.\n\n"
        )
        if summary["error"]:
            repair_result += summary["error"] + "\n\n"
    with _lock(root / "report"):
        atomic_json(root / "summary.json", summary)
        (root / "SUMMARY.md").write_text(
            "# Candidate 15 insulin revision\n\n"
            f"Status: {summary['status']}. Generation attempts recorded: "
            f"{summary['physical_requests_started']}. Observed tokens: "
            f"{summary['observed_total_tokens']}.\n\n{repair_result}"
            "Admission means the current deterministic checks passed, with unresolved "
            "checks retained. Peak improvement and scientific adequacy require an "
            "independently verified refit. No fitting or test evaluation runs here.\n\n"
            "See REPORT.html for the reply, equations and assessment, PROMPT.html for "
            "the complete request, and result.json for any typed model and Orion refit "
            "handoff.\n"
        )
        (root / "REPORT.html").write_text(
            '<!doctype html><meta charset="utf-8">'
            "<title>Insulin revision result</title>"
            "<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:2rem auto}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f6;padding:1rem}"
            "summary{font-weight:bold;cursor:pointer}</style>"
            "<h1>Candidate 15 insulin revision</h1><p>One bounded proposal followed by "
            "deterministic admission. An admitted model still needs fitting.</p>" + body
        )
        write_prompt_html(
            root / "PROMPT.html", json.loads(inputs["user"]), protocol=plan["protocol"]
        )
    return summary


def inspect(root: Path) -> dict:
    """Package only this campaign's regular evidence files, never following links."""
    summary = report(root)
    paths = [
        root / name
        for name in (
            "plan.json",
            "inputs.json",
            "attempt.json",
            "result.json",
            "execution.json",
            "serving_runtime.json",
            "summary.json",
            "SUMMARY.md",
            "REPORT.html",
            "PROMPT.html",
        )
    ]
    for name in ("calls", "submission", "logs", "runtime"):
        folder = root / name
        if folder.is_symlink():
            raise ValueError("inspection cannot include a symlink directory")
        paths.extend(p for p in folder.rglob("*") if p.is_file())
    entries = []
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("inspection cannot include external files or symlinks")
        if path.is_file() and not path.name.startswith("."):
            entries.append(path)
    target, temporary = root / "inspection.tar.gz", root / "inspection.tar.gz.tmp"
    with _lock(root / "report"), temporary.open("wb") as stream:
        with (
            gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped,
            tarfile.open(fileobj=zipped, mode="w") as archive,
        ):
            for path in sorted(entries):
                value = path.read_bytes()
                info = tarfile.TarInfo(path.relative_to(root).as_posix())
                info.size, info.mode, info.mtime = len(value), 0o644, 0
                archive.addfile(info, io.BytesIO(value))
        temporary.replace(target)
    return {**summary, "inspection_archive": str(target)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "preview", "run", "status", "inspect")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--source", type=Path, help="pinned reviewed fitting intake")
    parser.add_argument("--runtime-record", type=Path, help="ACES serving probe")
    parser.add_argument(
        "--model-size", choices=tuple(structural.MODEL_SPECS), default="120b"
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--html", type=Path, help="optional HTML path for preview")
    args = parser.parse_args()
    if args.html and args.action != "preview":
        parser.error("--html belongs to preview")
    if args.action == "prepare":
        if (args.campaign is None) == (args.source is None):
            parser.error("prepare requires exactly one of --campaign or --source")
        if args.source:
            result = prepare_source(args.root, args.source, args.model_size)
        else:
            from autoformalism.research.insulin_meal_campaign import proposer_inputs

            result = prepare(
                args.root, model_size=args.model_size, **proposer_inputs(args.campaign)
            )
    elif args.action == "preview":
        plan, inputs = _load(args.root)
        result = {k: inputs[k] for k in ("system", "user", "response_schema")}
        result["user"] = json.loads(result["user"])
        if args.html:
            write_prompt_html(args.html, result["user"], protocol=plan["protocol"])
    elif args.action in {"status", "inspect"}:
        result = status(args.root) if args.action == "status" else inspect(args.root)
    else:
        if args.runtime_record:
            from scripts.submit_meal_insulin_revision_aces import bind_server

            bind_server(args.root, args.runtime_record, args.endpoint)
        receipt = run(args.root, args.endpoint)
        result = {
            k: receipt[k]
            for k in ("status", "plan_sha256", "call_files", "automatic_fit")
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
