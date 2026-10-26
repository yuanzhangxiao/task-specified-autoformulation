#!/usr/bin/env python3
"""Matched model-size experiment with atomic cross-stage scientific repairs."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import signal
import tarfile
import time
from collections import Counter
from contextlib import suppress
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.construction import MEASURED_RESERVATION, ConstructionClient
from autoformalism.llm.response_revision import PromptPreflightError
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    atomic_json,
    strict_provider_schema,
    visible_response,
)
from autoformalism.rebuttal.prefit_construction_campaign import _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research import construction_parameter_audit as parameters
from autoformalism.research import construction_saved_revision_sources as sources
from autoformalism.research import construction_structural_feedback as feedback
from autoformalism.research.construction_directional_feedback import normalize_citations
from autoformalism.research.construction_revision_edits import (
    RevisionValidationError,
    model_view,
)
from autoformalism.research.construction_structural_revision import (
    StructuralRevisionReply,
    apply_structural_revision,
    editable_initial_states,
)
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.staged_topology import content_hash
from scripts import construction_feedback_revision as old
from scripts import construction_functions as receipts

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "construction-structural-revision-2"
SYMBOLIC_INITIAL_POLICY = "shared-symbolic-initial-readout-1"
MODEL_SPECS = {
    "20b": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee"),
    "120b": ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"),
}
SEEDS = (0, 1, 2)
ENTRYPOINTS = (
    "scripts/structural_revision_campaign.py",
    "scripts/submit_structural_revision_aces.py",
    "scripts/hpc/run_structural_revision_aces.sh",
    "scripts/hpc/start_structural_revision_aces.sh",
    "scripts/construction_feedback_revision.py",
    "scripts/construction_functions.py",
    "scripts/saved_initialization_aces.py",
    "scripts/submit_review_continuation.py",
    "scripts/harmony_vocab.py",
)
SYSTEM = """Revise this ODE/algebraic model using the public task and training evidence.
Choose a coherent scientific repair. You may edit complete right-hand sides and
their dependencies, add/remove latent states and algebraic processes, and change
editable latent initializers. Include a typed initializer for every new state.
Dependencies are derived from expressions; no separate topology copy is needed.
Keep public targets, output mappings, observed initial values, supplied data,
parameter roles/domains and the experiment's observation-window policy fixed.
Retain useful mechanisms and sharing. Adding a shared constant cannot distinguish
trajectories whose relevant initial conditions and forcing are identical.
Use existing parameter names when sharing is intended; declare each new object
once. Formula labels with an unambiguous matching left-hand side are accepted.
Put a new state's complete equation and initializer in new_states only; edits
and initialization_edits are for existing components. Identical redundant records
are consolidated and logged; conflicting definitions cannot be merged.
A reported qualified model-data conflict prevents acceptance. Change the model;
acknowledging a conflict or deleting its evidence cannot clear it. A deferred check
is not evidence that the model is scientifically correct or fits well.
Return decision=revise with a complete patch relative to the unchanged parent.
Rejected patches are not partially installed. Use retain only if no change is
needed: the unchanged model still undergoes all checks. If a necessary change is
outside the documented editable scope, return requires_wider_revision, explain
it, and supply no edits; this ends the episode without accepting a model.
Initializers use shared values, trajectory-specific roots, or acyclic maps of
allowed initial quantities. The proposer never chooses the observation window.
Do not include parameter values from held-out data or request fitting here.
""".strip()


def settings_for(model_size: str) -> StagedModelSettings:
    """Change only pinned model identity between the two inference arms."""
    model, revision = MODEL_SPECS[model_size]
    return StagedModelSettings(
        model=model,
        model_revision=revision,
        maximum_requests=3,
        maximum_total_tokens=98304,
        max_output_tokens=8192,
        attempts_per_step=3,
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entrypoints() -> dict:
    return {name: _hash(REPO / name) for name in ENTRYPOINTS}


@lru_cache(maxsize=64)
def _assessment_cached(
    problem_json: str,
    evidence_json: str,
    assess,
    weighted_symbolic_initials: bool = False,
    combined_symbolic_initials: bool = False,
) -> dict:
    """Memoize a pure gate result; public and model bytes are both in its key."""
    from autoformalism.research import construction_affine_family_consistency

    evidence = json.loads(evidence_json)
    problem = PrefixFitProblem.model_validate_json(problem_json)
    result = assess(
        {
            "_revision_protocol": old.V4_PROTOCOL,
            "_consistency_evidence": evidence,
        },
        problem,
    )
    # Historical V4 campaigns keep their original gate. Only this new namespace
    # enables symbolic initial equalities, including unknown common offsets.
    options = {"symbolic_initials": True}
    if weighted_symbolic_initials:
        options["weighted_symbolic_initials"] = True
    if combined_symbolic_initials:
        options["combined_symbolic_initials"] = True
    result["affine_family_consistency"] = construction_affine_family_consistency.assess(
        problem, evidence, **options
    )
    statuses = [
        result["equality_status"],
        result["function_consistency"]["status"],
        result["initial_family_consistency"]["status"],
        result["affine_family_consistency"]["status"],
    ]
    result["status"] = (
        "conflict"
        if "conflict" in statuses
        else "deferred"
        if "deferred" in statuses
        else "no_conflict_found"
    )
    result["symbolic_initial_policy"] = SYMBOLIC_INITIAL_POLICY
    return result


def _assessment(
    parent: dict,
    problem: PrefixFitProblem,
    *,
    weighted_symbolic_initials: bool = False,
    combined_symbolic_initials: bool = False,
) -> dict:
    return deepcopy(
        _assessment_cached(
            problem.model_dump_json(),
            json.dumps(parent["_consistency_evidence"], sort_keys=True),
            old._consistency,
            weighted_symbolic_initials,
            combined_symbolic_initials,
        )
    )


def _parent(selected: dict, name: str) -> dict:
    return {
        **selected["parents"][name],
        "fixed_fitting_prefix_policy": selected["inputs"].get("fitting_prefix_policy"),
        "_revision_protocol": old.V4_PROTOCOL,
        "_consistency_evidence": selected["consistency_source"]["evidence"],
    }


def _roster(selected: dict, model_size: str) -> list[dict]:
    cases = []
    for name, parent in sorted(selected["parents"].items()):
        for seed in SEEDS:
            identity = {
                "parent_id": name,
                "seed": seed,
                "input_sha256": content_hash(parent),
            }
            case = f"{name}__revision_seed{seed}"
            cases.append(
                {
                    **identity,
                    "name": case,
                    "case_id": case,
                    "namespace": f"{PROTOCOL}:{model_size}:{content_hash(identity)}",
                }
            )
    return cases


def _plan(selected: dict, model_size: str) -> dict:
    settings = settings_for(model_size).model_dump(mode="json")
    common = {
        "protocol": PROTOCOL,
        "selected_sources_sha256": content_hash(selected),
        "source_archive_sha256": sources.SOURCE_ARCHIVE_SHA256,
        "seeds": list(SEEDS),
        "runtime_source_sha256": runtime_source_hash(),
        "entrypoints": _entrypoints(),
        "generation_policy": {
            k: v for k, v in settings.items() if k not in {"model", "model_revision"}
        },
        "system_sha256": content_hash(SYSTEM),
        "response_schema_sha256": content_hash(
            StructuralRevisionReply.model_json_schema()
        ),
        "gate_policy": {
            **old.V4_POLICIES,
            "symbolic_initials": SYMBOLIC_INITIAL_POLICY,
        },
        "edit_policy": "atomic-cross-stage-redundancy-normalization-2",
        "test_data_opened": False,
        "automatic_followup": False,
        "model_promotion": False,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
    }
    return {
        **common,
        "comparison_sha256": content_hash(common),
        "model_size": model_size,
        "settings": settings,
        "cases": _roster(selected, model_size),
    }


def prepare(source: Path, root: Path, model_size: str) -> dict:
    """Authenticate the same six historical models; never select by fitted error."""
    selected = sources.load(source)
    old._validate_inputs(selected["inputs"])
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
        "comparison_sha256": plan["comparison_sha256"],
        "model_size": model_size,
        "cases": len(plan["cases"]),
        "maximum_generation_requests": len(plan["cases"]) * 3,
    }


def verify(root: Path) -> dict:
    """Reconstruct source provenance, settings, code and unchanged controls."""
    plan = sealed_read(root / "plan.json")
    selected = sources.load(root / "saved-source.tar.gz")
    saved = sealed_read(root / "selected_sources.json")
    if saved != {**selected, "artifact_sha256": content_hash(selected)}:
        raise ValueError("selected source content differs")
    expected = _plan(selected, plan["model_size"])
    if plan != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError(
            "structural revision plan, executable source or settings differs"
        )
    for name, parent in selected["parents"].items():
        control = sealed_read(root / "controls" / f"{name}.json")
        expected_control = {
            "problem": parent["problem"],
            "control_unchanged": True,
            "problem_sha256": content_hash(parent["problem"]),
        }
        if control != {
            **expected_control,
            "artifact_sha256": content_hash(expected_control),
        }:
            raise ValueError("unchanged control differs")
    return plan


def request(
    parent: dict,
    repair: dict | None = None,
    *,
    weighted_symbolic_initials: bool = False,
    combined_symbolic_initials: bool = False,
) -> str:
    """Present one model view and target-specific witnesses, with explicit scope."""
    problem = PrefixFitProblem.model_validate(parent["problem"])
    view = model_view(problem, scientific_aliases=parent["scientific_aliases"])
    view["frozen_fields"] = [
        "public context",
        "target output mappings",
        "observed initial identities",
        "existing parameter domains",
    ]
    view["editable_initial_states"] = editable_initial_states(
        problem, scientific_aliases=parent["scientific_aliases"]
    )
    assessment = _assessment(
        parent,
        problem,
        weighted_symbolic_initials=weighted_symbolic_initials,
        combined_symbolic_initials=combined_symbolic_initials,
    )
    packet = {
        "public_task": parent["public_brief"],
        "current_model": view,
        "fitting_prefix_policy": parent.get("fixed_fitting_prefix_policy"),
        "allowed_evidence_ids": [
            "public_task",
            "training_consistency",
            "parameter_audit",
        ],
        "parameter_audit": parameters.prompt_view(parameters.assess(problem)),
        "training_consistency": feedback.consistency_view(assessment, parent),
        "editing_scope": {
            "edits": "Existing component name, dynamic or algebraic, complete RHS. "
            "Do not repeat a new_states/new_processes definition here.",
            "new_states": "New latent differential state, RHS, and typed initial rule.",
            "new_processes": "New algebraic process and complete expression.",
            "initialization_edits": "Replace an existing editable latent state's "
            "initial rule. A new state's initializer belongs inside new_states.",
            "remove_states_and_processes": "Explicit names; rewrite dependent uses.",
            "new_parameters": "New shared dynamics parameters and admissible domains.",
            "unsupported": "Changing public data/targets/observed initials, parameter "
            "role/domain, or converting an existing component's kind.",
        },
        "initializer_map_example": {
            "note": "Syntax example only, not a prescribed scientific choice. "
            "Aliases are local symbols in the initializer expression.",
            "state": "editable_latent_name",
            "initial": {
                "mode": "map",
                "expression": "a*z0+b*y0",
                "parameters": [
                    {"name": "a", "role": "coefficient"},
                    {"name": "b", "role": "coefficient"},
                ],
                "state_initials": [{"alias": "z0", "state": "other_state_name"}],
                "public_initials": [{"alias": "y0", "channel": "public_channel_name"}],
            },
        },
    }
    if repair:
        packet["repair_feedback"] = repair
        if repair.get("diagnostic", {}).get("details", {}).get("assessment"):
            packet["training_consistency"] = {
                "starting_model_status": assessment["status"],
                "trial_evidence_location": (
                    "repair_feedback.diagnostic.details.assessment"
                ),
                "model_reference": "Trial evidence refers to last_reply; "
                "current_model is unchanged.",
            }
    return json.dumps(packet, ensure_ascii=False, separators=(",", ":"))


def _normalize_displayed_citations(
    parent: dict,
    reply: dict,
    *,
    weighted_symbolic_initials: bool = False,
    combined_symbolic_initials: bool = False,
) -> tuple[dict, list]:
    """Resolve exact property labels only from this model's displayed evidence."""
    reply, changes = normalize_citations(reply)
    identifiers = reply.get("evidence_ids")
    if not isinstance(identifiers, list) or any(
        not isinstance(item, str) for item in identifiers
    ):
        return reply, changes
    allowed = {"public_task", "training_consistency", "parameter_audit"}
    if set(identifiers) <= allowed:
        return reply, changes
    report = feedback.consistency_view(
        _assessment(
            parent,
            PrefixFitProblem.model_validate(parent["problem"]),
            weighted_symbolic_initials=weighted_symbolic_initials,
            combined_symbolic_initials=combined_symbolic_initials,
        ),
        parent,
    )
    properties = set()

    def collect(value):
        if isinstance(value, dict):
            if isinstance(value.get("property"), str):
                properties.add(value["property"])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(report)
    normalized = []
    for identifier in identifiers:
        target = "training_consistency" if identifier in properties else identifier
        if target != identifier:
            changes.append({"from": identifier, "to": target})
        if target not in normalized:
            normalized.append(target)
    reply["evidence_ids"] = normalized
    return reply, changes


def _apply(
    parent: dict,
    reply: dict,
    *,
    normalize_parent_declarations: bool = False,
    weighted_symbolic_initials: bool = False,
    combined_symbolic_initials: bool = False,
) -> dict:
    reply, changes = _normalize_displayed_citations(
        parent,
        reply,
        weighted_symbolic_initials=weighted_symbolic_initials,
        combined_symbolic_initials=combined_symbolic_initials,
    )
    options = {}
    if normalize_parent_declarations:
        options["normalize_parent_declarations"] = True
    applied = apply_structural_revision(
        PrefixFitProblem.model_validate(parent["problem"]),
        reply,
        evidence_ids=["public_task", "training_consistency", "parameter_audit"],
        brief=PublicScientificBrief.model_validate(parent["public_brief"]),
        scientific_aliases=parent["scientific_aliases"],
        **options,
    )
    if applied["status"] == "requires_wider_revision":
        return {**applied, "problem": None}
    problem = applied["problem"]
    assessment = _assessment(
        parent,
        problem,
        weighted_symbolic_initials=weighted_symbolic_initials,
        combined_symbolic_initials=combined_symbolic_initials,
    )
    applied["audit"].update(
        training_consistency=assessment,
        parameter_audit=parameters.assess(problem),
        citation_normalization=changes,
    )
    if assessment["status"] == "conflict":
        raise RevisionValidationError(
            "QUALIFIED_MODEL_DATA_CONFLICT",
            "This candidate contradicts qualified training witnesses and cannot "
            "be accepted. Revise equations, dependencies or initial conditions within "
            "the displayed scope. Returning no edits cannot clear this conflict.",
            assessment=feedback.consistency_view(assessment, parent),
            public_path_checks=applied["audit"].get("public_path_checks_after"),
        )
    if applied["audit"].get("static_eligibility") != "eligible":
        raise RevisionValidationError(
            "PUBLIC_PATH_REPAIR_REQUIRED",
            "The complete candidate lacks a checked public pathway.",
            checks=applied["audit"].get("public_path_checks_after"),
        )
    return applied


def _episode(parent: dict, client, settings: StagedModelSettings) -> dict:
    attempts, repair = [], None
    available = False
    for attempt in range(settings.maximum_requests):
        record = client.call(
            system=SYSTEM,
            user=request(parent, repair),
            response_model=StructuralRevisionReply,
            step="structural_revision",
            attempt=attempt,
        )
        raw = None
        try:
            raw = visible_response(record)
            available = True
            result = _apply(parent, raw)
            problem = result["problem"]
            return {
                **result,
                "problem": problem.model_dump(mode="json") if problem else None,
                "reply": raw,
                "availability": "available",
                "attempts": [
                    *attempts,
                    {
                        "attempt": attempt,
                        "request_hash": record["request_hash"],
                        "status": result["status"],
                    },
                ],
                "interpretation": "Qualified deterministic checks only; "
                "no fitting or promotion.",
            }
        except ValueError as exc:
            diagnostic = (
                exc.as_dict()
                if isinstance(exc, RevisionValidationError)
                else {
                    "code": "RESPONSE_SCHEMA_OR_AVAILABILITY",
                    "message": str(exc)[:4000],
                }
            )
            guidance = feedback.feedback_guidance(
                diagnostic["code"], diagnostic.get("details", {}), parent, raw
            )
            # The diagnostic already contains this evidence; show it only once.
            guidance.pop("details", None)
            signature = feedback.patch_signature(raw)
            previous = [
                r["attempt"]
                for r in attempts
                if signature and r.get("patch_sha256") == signature
            ]
            repair = {
                "status": "rejected_in_full_parent_unchanged",
                "diagnostic": diagnostic,
                "guidance": guidance,
                "last_reply": raw,
                "repeated_equation_patch": bool(previous),
                "repeated_patch_attempts": previous,
                "instruction": (
                    "This repeats a rejected edit. Changing explanation or citations "
                    "does not change its behavior. Return a different complete patch."
                    if previous
                    else "Return a complete corrected patch relative to current_model."
                ),
                "attempts_remaining": settings.maximum_requests - attempt - 1,
            }
            attempts.append(
                {
                    "attempt": attempt,
                    "request_hash": record["request_hash"],
                    "status": "rejected",
                    "diagnostic": diagnostic,
                    "reply_sha256": content_hash(raw),
                    "patch_sha256": signature,
                }
            )
    return {
        "status": "rejected" if available else "unavailable",
        "problem": None,
        "availability": "available" if available else "unavailable",
        "attempts": attempts,
    }


class _ReplayClient:
    """Authenticate exact prompts, schema, generation settings and call order."""

    def __init__(
        self,
        records: list[dict],
        settings: StagedModelSettings,
        *,
        pending: bool = False,
    ):
        self.records, self.settings, self.pending = records, settings, pending
        self.consumed: set[str] = set()

    def call(self, *, system, user, response_model, step, attempt):
        matches = [
            r for r in self.records if r["step"] == step and r["attempt"] == attempt
        ]
        if not matches and self.pending:
            raise DeferredCall("saved prefix ends here")
        if len(matches) != 1:
            raise ValueError("missing or duplicate replay call")
        record = matches[0]
        body = record["request"]["body"]
        expected = {
            "model": self.settings.model,
            "stream": False,
            "reasoning_effort": self.settings.reasoning_effort,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_output_tokens,
            "seed": body["seed"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": strict_provider_schema(
                        response_model.model_json_schema()
                    ),
                },
            },
        }
        if (
            body != expected
            or set(record["request"])
            != {
                "protocol",
                "namespace",
                "settings",
                "body",
                "budget_reservation_policy",
            }
            or record["request"].get("budget_reservation_policy")
            != MEASURED_RESERVATION
            or record["request"].get("protocol") != "scientific-staged-topology-1"
        ):
            raise ValueError("saved structural revision request differs")
        self.consumed.add(record["request_hash"])
        return record


def _handoff(plan: dict, entry: dict, parent: dict, result: dict) -> dict:
    return {
        "protocol": "construction-structural-fit-handoff-1",
        "plan_sha256": plan["artifact_sha256"],
        "case_id": entry["case_id"],
        "parent_id": entry["parent_id"],
        "parent_problem_sha256": content_hash(parent["problem"]),
        "prefix_fit_problem": result["problem"],
        "fitting_prefix_policy": parent.get("fixed_fitting_prefix_policy"),
        "original_initialization_unchanged": result["problem"]["graph"]
        == parent["problem"]["graph"],
        "training_consistency": result["audit"]["training_consistency"],
        "fitting_launched": False,
        "test_data_opened": False,
        "validation_used": False,
        "scientific_compliance_assessed": False,
        "selection": "candidate_for_independent_fitting_not_promoted",
    }


def _completion(root: Path, entry: dict, plan: dict, parent: dict) -> dict | None:
    directory = root / "results" / entry["case_id"]
    if not (directory / "completion.json").exists():
        return None
    receipt = sealed_read(directory / "completion.json")
    records = receipts._calls(root, entry, plan)
    result = json.loads((directory / "result.json").read_text())
    files = ["result.json"] + (["fit_handoff.json"] if result.get("problem") else [])
    expected = {
        "plan_sha256": plan["artifact_sha256"],
        "input_sha256": entry["input_sha256"],
        "calls_sha256": content_hash(records),
        "status": result["status"],
        "tokenizations_sha256": content_hash(
            receipts._tokenizations(root, entry, plan)
        ),
        "files": {name: _hash(directory / name) for name in files},
    }
    if receipt != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError("completion receipt differs")
    replay = _ReplayClient(records, settings_for(plan["model_size"]))
    if _episode(parent, replay, replay.settings) != result or len(
        replay.consumed
    ) != len(records):
        raise ValueError("saved result differs from exact request/reply reconstruction")
    if result.get("problem"):
        expected_handoff = _handoff(plan, entry, parent, result)
        if sealed_read(directory / "fit_handoff.json") != {
            **expected_handoff,
            "artifact_sha256": content_hash(expected_handoff),
        }:
            raise ValueError("fitting handoff differs")
    return result


def run(
    root: Path,
    endpoint: str,
    runtime_record: Path | None = None,
    *,
    seconds: int = 5400,
    transport=None,
    token_transport=None,
) -> dict:
    """Resume cached attempts with one lock; never discard an ambiguous call."""
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
        raise ValueError("construction server must be a loopback HTTP endpoint")
    if runtime_record is not None:
        from scripts import submit_structural_revision_aces as launcher

        launcher.bind_server(root, runtime_record, endpoint)
    elif transport is None:
        raise ValueError("live generation requires verified serving identity")
    selected = sealed_read(root / "selected_sources.json")
    settings = settings_for(plan["model_size"])
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
                parent = _parent(selected, entry["parent_id"])
                if _completion(root, entry, plan, parent) is not None:
                    continue
                directory = root / "results" / entry["case_id"]
                saved = receipts._calls(root, entry, plan)
                receipts._tokenizations(root, entry, plan)
                replay = _ReplayClient(saved, settings, pending=True)
                with suppress(DeferredCall):
                    _episode(parent, replay, settings)
                if len(replay.consumed) != len(saved):
                    raise ValueError("pending calls are not an exact request prefix")
                client = ConstructionClient(
                    settings=settings,
                    base_url=endpoint,
                    directory=directory / "calls",
                    namespace=entry["namespace"],
                    seed=entry["seed"],
                    reservation_policy=MEASURED_RESERVATION,
                    token_transport=token_transport,
                    can_start=lambda: not stopped
                    and time.monotonic() + settings.timeout_seconds < deadline,
                    **({"transport": transport} if transport else {}),
                )
                client.budget_diagnostics_enabled = True
                try:
                    result = _episode(parent, client, settings)
                except DeferredCall:
                    break
                except PromptPreflightError as exc:
                    sealed_write(
                        directory / "preflight_blocked.json",
                        {
                            "plan_sha256": plan["artifact_sha256"],
                            "calls_sha256": content_hash(
                                receipts._calls(root, entry, plan)
                            ),
                            "tokenizations_sha256": content_hash(
                                receipts._tokenizations(root, entry, plan)
                            ),
                            "status": "preflight_blocked",
                            "problem": None,
                            "error": str(exc)[:4000],
                        },
                    )
                    continue
                atomic_json(directory / "result.json", result)
                files = ["result.json"]
                if result.get("problem"):
                    sealed_write(
                        directory / "fit_handoff.json",
                        _handoff(plan, entry, parent, result),
                    )
                    files.append("fit_handoff.json")
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
                        "files": {f: _hash(directory / f) for f in files},
                    },
                )
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return report(root)


def report(root: Path) -> dict:
    """Report acceptance, unresolved checks and cost without numerical ranking."""
    plan, rows, records = verify(root), [], []
    selected = sealed_read(root / "selected_sources.json")
    for entry in plan["cases"]:
        parent = _parent(selected, entry["parent_id"])
        result = _completion(root, entry, plan, parent) or {}
        calls = receipts._calls(root, entry, plan)
        blocked = root / "results" / entry["case_id"] / "preflight_blocked.json"
        if not result and blocked.exists():
            marker = sealed_read(blocked)
            if (
                marker["plan_sha256"] != plan["artifact_sha256"]
                or marker["calls_sha256"] != content_hash(calls)
                or marker["tokenizations_sha256"]
                != content_hash(receipts._tokenizations(root, entry, plan))
                or marker["status"] != "preflight_blocked"
                or marker["problem"] is not None
            ):
                raise ValueError("preflight blocked receipt differs")
            result = marker
        records.extend(calls)
        rows.append(
            {
                "case_id": entry["case_id"],
                "parent_id": entry["parent_id"],
                "seed": entry["seed"],
                "status": result.get("status", "pending"),
                "fitting_exported": bool(result.get("problem")),
                "training_consistency_status": result.get("audit", {})
                .get("training_consistency", {})
                .get("status"),
                "cost": _cost(calls),
            }
        )
    counts = dict(Counter(row["status"] for row in rows))
    summary = {
        "protocol": PROTOCOL,
        "model_size": plan["model_size"],
        "plan_sha256": plan["artifact_sha256"],
        "comparison_sha256": plan["comparison_sha256"],
        "status": "incomplete"
        if counts.get("pending") or counts.get("preflight_blocked")
        else "complete",
        "status_counts": counts,
        "rows": rows,
        "cost": _cost(records),
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "automatic_followup": False,
        "model_promotion": False,
    }
    atomic_json(root / "summary.json", summary)
    lines = [
        "# Cross-stage construction repair",
        "",
        f"Model: {plan['model_size']}. Status: {summary['status']}.",
        f"Counts: {counts}. Requests: {summary['cost']['physical_requests']}.",
        "",
        "Deterministic eligibility is not scientific compliance or fitting adequacy.",
        "",
        "| Parent | Seed | Status | Data checks | Fitting packet |",
        "|---|---:|---|---|---|",
    ]
    lines += [
        f"|{r['parent_id']}|{r['seed']}|{r['status']}|"
        f"{r['training_consistency_status']}|{r['fitting_exported']}|"
        for r in rows
    ]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return summary


def preview(root: Path) -> dict:
    """Save exact initial requests and readable prompts, with no generation."""
    plan = verify(root)
    selected = sealed_read(root / "selected_sources.json")
    parts = [
        "<!doctype html><meta charset='utf-8'>"
        "<title>Cross-stage repair prompts</title>",
        "<style>body{max-width:1100px;margin:2rem auto;font:16px/1.5 sans-serif}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;"
        "padding:1rem}mark{background:#fff0a8}</style>",
        "<h1>Cross-stage repair: matched 20B / 120B</h1><p><mark>Changed:</mark> "
        "explicit latent-state, process and initializer edits; target-specific "
        "witnesses; no fixed-roster repair suggestions. Same cases, seeds, calls "
        "and token ceilings in both arms.</p>",
        "<p>The system and scientific payload are identical between model arms. "
        "Unsupported changes terminate with a distinct outcome; this is not a "
        "narrow-versus-wide ablation.</p>",
    ]
    for name in sorted(selected["parents"]):
        packet = {
            "system": SYSTEM,
            "user": request(_parent(selected, name)),
            "response_schema": strict_provider_schema(
                StructuralRevisionReply.model_json_schema()
            ),
        }
        atomic_json(root / "preview" / f"{name}.json", packet)
        parts.append(
            f"<h2>{html.escape(name)}</h2><pre>{html.escape(SYSTEM)}</pre><pre>"
            f"{html.escape(json.dumps(json.loads(packet['user']), indent=2))}</pre>"
        )
    example_parent = _parent(selected, sorted(selected["parents"])[0])
    hypothetical_reply = {
        "decision": "retain",
        "hypothesis": "Retain the current model.",
        "expected_observable_effect": "Unchanged",
        "evidence_ids": ["training_consistency"],
    }
    try:
        _apply(example_parent, hypothetical_reply)
    except RevisionValidationError as exc:
        diagnostic = exc.as_dict()
        guidance = feedback.feedback_guidance(
            diagnostic["code"],
            diagnostic.get("details", {}),
            example_parent,
            hypothetical_reply,
        )
        guidance.pop("details", None)
        example = {
            "status": "rejected_in_full_parent_unchanged",
            "diagnostic": diagnostic,
            "last_reply": hypothetical_reply,
            "repeated_equation_patch": False,
            "guidance": guidance,
            "attempts_remaining": 2,
        }
        example_packet = {"system": SYSTEM, "user": request(example_parent, example)}
        atomic_json(root / "preview" / "repair-example.json", example_packet)
        parts.append(
            "<h2>Repair feedback example</h2><p><mark>Offline illustration:</mark> "
            "a hypothetical retain reply was checked against the actual saved model. "
            "No LLM call was made. This shows the complete next request.</p><pre>"
            + html.escape(json.dumps(json.loads(example_packet["user"]), indent=2))
            + "</pre>"
        )
    parts.append(
        "<h2>Reply schema</h2><pre>"
        + html.escape(json.dumps(packet["response_schema"], indent=2))
        + "</pre>"
    )
    (root / "PROMPTS.html").write_text("\n".join(parts))
    return {
        "path": str(root / "PROMPTS.html"),
        "cases": len(plan["cases"]),
        "llm_calls": 0,
    }


def inspect(root: Path) -> dict:
    """Package only the bounded campaign evidence and rendered prompt review."""
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
            if item.is_symlink():
                continue
            if item.is_file() and (
                str(rel) in allowed
                or rel.parts[0]
                in {"results", "controls", "preview", "logs", "submission", "runtime"}
            ):
                archive.add(item, arcname=str(rel), recursive=False)
    return {**result, "archive": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "run", "report", "inspect", "preview")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model-size", choices=tuple(MODEL_SPECS), default="20b")
    parser.add_argument("--endpoint")
    parser.add_argument("--runtime-record", type=Path)
    parser.add_argument("--seconds", type=int, default=5400)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        value = prepare(args.source, args.root, args.model_size)
    elif args.command == "run":
        if not args.endpoint:
            parser.error("run requires --endpoint")
        value = run(args.root, args.endpoint, args.runtime_record, seconds=args.seconds)
    else:
        value = {"report": report, "inspect": inspect, "preview": preview}[
            args.command
        ](args.root)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
