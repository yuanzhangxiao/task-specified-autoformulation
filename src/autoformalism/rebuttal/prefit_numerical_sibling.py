"""One frozen residual-feedback sibling beside continued fitting of its parent."""

from __future__ import annotations

import hashlib
import json
import signal
import time
from contextlib import suppress
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.expressions import ModelValidationError
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit as child_backend
from autoformalism.fitting.sibling_fit import (
    execute_child_fit,
    inspect_child_fit,
    prepare_child_fit,
)
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    strict_provider_schema,
    visible_response,
)
from autoformalism.rebuttal.prefit_feedback import (
    EpisodeClient,
    _check_events,
    _records,
    _save_state,
)
from autoformalism.rebuttal.prefit_fit_handoff import (
    HandoffSelection,
    _disjoint,
    verify_source,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit, Sha256
from autoformalism.schemas.residual_feedback import FeedbackSelection
from autoformalism.search import numerical_sibling as review
from autoformalism.search.requirement_feedback import diagnose_requirements
from autoformalism.search.residual_evidence import validate_residual_evidence
from autoformalism.staged_topology import content_hash

PROTOCOL = "prefit-numerical-sibling-1"
STEP = "numerical_sibling_review"
LIMITATION = (
    "One exploratory sibling from an unfinished fit; no automatic winner, pruning, "
    "or scientific certification. Parent continuation and child fitting have "
    "different additional budgets, so this is not an equal-compute comparison."
)


class SiblingConfig(StrictSchema):
    """Pin one completed parent and one bounded optional proposer episode."""

    protocol: Literal["prefit-numerical-sibling-1"] = PROTOCOL
    feedback_policy: review.FeedbackPolicy = review.LEGACY_POLICY
    platform: Literal["aces-h100x1"] = "aces-h100x1"
    selection: FeedbackSelection = Field(default_factory=FeedbackSelection)
    serving_image_sha256: Sha256
    model_settings: StagedModelSettings
    seed: Literal[0] = 0
    served_context_tokens: Literal[32768] = 32768
    wall_seconds: Literal[3600] = 3600
    shutdown_margin_seconds: Literal[300] = 300

    @model_validator(mode="after")
    def bounded(self):
        settings = self.model_settings
        if settings.maximum_requests != 3 or settings.attempts_per_step != 3:
            raise ValueError("one sibling episode has exactly three request slots")
        if settings.timeout_seconds >= self.shutdown_margin_seconds:
            raise ValueError("provider timeout must fit inside shutdown margin")
        return self


def _exporter():
    from autoformalism.fitting import fit_residual_feedback

    return fit_residual_feedback


def launcher_hash() -> str:
    repo = Path(__file__).resolve().parents[3]
    paths = (
        "scripts/prefit_numerical_sibling.py",
        "scripts/hpc/run_prefit_numerical_sibling_aces.sh",
        "scripts/hpc/submit_prefit_numerical_sibling_aces.sh",
        "scripts/hpc/run_staged_topology_server.sh",
    )
    return content_hash(
        {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in paths}
    )


def _reservation_directory(
    parent_path: Path, selection: FeedbackSelection, policy: str
) -> Path:
    """One allocation per explicitly versioned experiment, never per output path."""
    ledger = (
        parent_path.parent.parent
        / ".numerical-siblings"
        / selection.continuation_identity
    )
    return ledger if policy == review.LEGACY_POLICY else ledger / policy


def _reserve(root: Path, parent_path: Path, config: SiblingConfig) -> None:
    ledger = _reservation_directory(
        parent_path, config.selection, config.feedback_policy
    )
    with public._lock(ledger):
        sealed_write(
            ledger / "reservation.json",
            {
                "protocol": PROTOCOL,
                "output": str(root.resolve()),
                "selection": config.selection.model_dump(mode="json"),
                **(
                    {"feedback_policy": config.feedback_policy}
                    if config.feedback_policy != review.LEGACY_POLICY
                    else {}
                ),
            },
        )


def _check_reservation(root: Path, plan: dict) -> None:
    config = SiblingConfig.model_validate(plan["config"])
    ledger = _reservation_directory(
        Path(plan["paths"]["parent"]), config.selection, config.feedback_policy
    )
    reservation = sealed_read(ledger / "reservation.json")
    expected = {
        "protocol": PROTOCOL,
        "output": str(root.resolve()),
        "selection": plan["config"]["selection"],
        **(
            {"feedback_policy": config.feedback_policy}
            if config.feedback_policy != review.LEGACY_POLICY
            else {}
        ),
    }
    if {k: v for k, v in reservation.items() if k != "artifact_sha256"} != expected:
        raise ValueError("sibling reservation differs; no new allocation")


def prepare(
    parent_path: Path,
    continuation_path: Path,
    source: Path,
    construction_root: Path,
    config_path: Path,
    root: Path,
) -> dict:
    """Reconstruct the accepted model and seal a separate, single sibling allocation."""
    _disjoint(
        root, parent_path.parent, continuation_path.parent, source, construction_root
    )
    config = SiblingConfig.model_validate_json(config_path.read_text())
    handoff = sealed_read(parent_path.parent / "handoff.json")
    selection = HandoffSelection.model_validate(handoff["selection"])
    checked = verify_source(source, construction_root, selection)
    source_plan = sealed_read(source / "plan.json")
    for key in (
        "model",
        "model_revision",
        "reasoning_effort",
        "temperature",
        "max_output_tokens",
    ):
        if source_plan["config"]["model_settings"][key] != getattr(
            config.model_settings, key
        ):
            raise ValueError(
                f"proposer inference differs from historical source: {key}"
            )
    bundle = {
        **checked["bundle"],
        "candidate": checked["final"]["candidate"],
        "initialization": checked["final"]["initialization"],
        "slots": [
            checked["final"]["selected_function"]
            if s["interaction_id"] == checked["final"]["selected_interaction"]
            else s
            for s in checked["bundle"]["slots"]
        ],
    }
    bindings = [
        b for b in source_plan["config"]["bindings"] if b["cell"] == selection.cell
    ]
    if not bindings or diagnose_requirements(bundle, bindings)["requirement_gap"]:
        raise ValueError("sibling parent needs its bound requirements satisfied")
    if (
        handoff["accepted_candidate"] != bundle["candidate"]
        or handoff["public_brief"] != bundle["brief"]
    ):
        raise ValueError("accepted source differs from original public handoff")
    with public._lock(root):
        residual = _exporter().prepare_residual_feedback(
            parent_path,
            continuation_path,
            root / "evidence",
            selection=config.selection,
        )
        seed = residual["seed"]
        original = seed["parent"]["freeze"]
        if (
            original["request"] != handoff["request"]
            or original["lowered_candidate"] != bundle["candidate"]
        ):
            raise ValueError("residual parent differs from accepted scientific model")
        plan = {
            "protocol": PROTOCOL,
            "config": config.model_dump(mode="json"),
            "bundle": bundle,
            "bindings": bindings,
            "paths": {
                "parent": str(parent_path.resolve()),
                "continuation": str(continuation_path.resolve()),
                "source": str(source.resolve()),
                "construction": str(construction_root.resolve()),
            },
            "residual_identity": residual["identity"],
            "seed_sha256": public.content_sha256(seed),
            "handoff_sha256": handoff["artifact_sha256"],
            "source_sha256": public._source_identity(),
            "runtime": public._runtime(),
            "launcher_sha256": launcher_hash(),
        }
        _reserve(root, parent_path, config)
        return sealed_write(root / "plan.json", plan)


def verify(root: Path) -> tuple[dict, dict]:
    """Verify the fixed sibling lineage without reading the running continuation."""
    plan = sealed_read(root / "plan.json")
    SiblingConfig.model_validate(plan["config"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
        or plan["launcher_sha256"] != launcher_hash()
    ):
        raise ValueError("sibling source, runtime or launcher differs")
    _check_reservation(root, plan)
    residual = _exporter().load_residual_feedback(root / "evidence")
    if (
        residual["identity"] != plan["residual_identity"]
        or public.content_sha256(residual["seed"]) != plan["seed_sha256"]
    ):
        raise ValueError("sibling residual seed differs")
    if residual["status"] == "ready":
        original = residual["seed"]["parent"]["freeze"]
        validate_residual_evidence(
            residual["packet"],
            candidate_sha256=public.content_sha256(plan["bundle"]["candidate"]),
            parameters=residual["seed"]["state"]["parameters"],
            training_content_sha256=public.content_sha256(original["training"]),
        )
    return plan, residual


def replay(root: Path) -> dict:
    """Measure the frozen points on training only; no proposer or optimization."""
    with public._lock(root):
        verify(root)
        return _exporter().run_residual_feedback(root / "evidence")


def _namespace(plan: dict) -> str:
    return content_hash([plan["artifact_sha256"], STEP, 0])


def _user(plan, residual, retry):
    policy = SiblingConfig.model_validate(plan["config"]).feedback_policy
    title = (
        "Optional numerical sibling review"
        if policy == review.LEGACY_POLICY
        else "Routed structural hypothesis"
    )
    return (
        title
        + "\n"
        + json.dumps(
            review.payload(
                plan["bundle"],
                plan["bindings"],
                residual["packet"],
                residual["seed"]["state"]["parameters"],
                retry,
                policy=policy,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _evaluate(plan, packet, record):
    raw, result = None, None
    try:
        raw = visible_response(record)
        result = review.apply_revision(
            plan["bundle"],
            plan["bindings"],
            packet,
            raw,
            policy=SiblingConfig.model_validate(plan["config"]).feedback_policy,
        )
        feedback = {
            "stage": "revision_contract",
            "error": None,
            "transaction": result["outcome"],
        }
    except (ValueError, KeyError, TypeError, ModelValidationError) as error:
        feedback = review.failure_feedback(plan["bundle"], record, raw, error)
    return raw, result, feedback


def _state(root, plan, residual):
    path = root / "results/state.json"
    state = (
        sealed_read(path)
        if path.exists()
        else {
            "identity": _namespace(plan),
            "status": "pending",
            "attempts": [],
            "decision": None,
            "stop_reason": None,
            "packet_sha256": (residual.get("packet") or {}).get("packet_sha256"),
        }
    )
    state.pop("artifact_sha256", None)
    if state["identity"] != _namespace(plan) or state["packet_sha256"] != (
        residual.get("packet") or {}
    ).get("packet_sha256"):
        raise ValueError("sibling episode identity differs")
    records = _records(root / "results/calls", _namespace(plan))
    _check_events(state, records)
    config = SiblingConfig.model_validate(plan["config"])
    settings = config.model_settings
    if (
        len(records) > settings.maximum_requests
        or len(state["attempts"]) > settings.attempts_per_step
    ):
        raise ValueError("sibling episode exceeds frozen budget")
    by_key, seen = {r["request_hash"]: r for r in records}, set()
    final = None
    for index, event in enumerate(state["attempts"]):
        if (
            event["attempt"] != index
            or event["request_hash"] in seen
            or final is not None
        ):
            raise ValueError("sibling event order differs")
        seen.add(event["request_hash"])
        raw, final, feedback = _evaluate(
            plan, residual["packet"], by_key[event["request_hash"]]
        )
        if (
            event["response"] != raw
            or event["feedback"] != feedback
            or event["accepted"] != (final is not None)
        ):
            raise ValueError("saved sibling decision differs from deterministic replay")
    if state["decision"] != final:
        raise ValueError("sibling final decision differs from replay")
    seen_attempts = set()
    for record in records:
        index = record["attempt"]
        if (
            index in seen_attempts
            or not 0 <= index <= len(state["attempts"])
            or index >= settings.attempts_per_step
        ):
            raise ValueError("orphan or duplicate sibling provider request")
        seen_attempts.add(index)
        retry = state["attempts"][index - 1]["feedback"] if index else None
        body = {
            "model": settings.model,
            "messages": [
                {
                    "role": "system",
                    "content": review.system_prompt(config.feedback_policy),
                },
                {"role": "user", "content": _user(plan, residual, retry)},
            ],
            "stream": False,
            "reasoning_effort": settings.reasoning_effort,
            "temperature": settings.temperature,
            "max_tokens": settings.max_output_tokens,
            "seed": int(content_hash([0, STEP, index])[:8], 16) % (2**31),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": review.response_model(config.feedback_policy).__name__,
                    "strict": True,
                    "schema": strict_provider_schema(
                        review.response_model(
                            config.feedback_policy
                        ).model_json_schema()
                    ),
                },
            },
        }
        expected = {
            "protocol": "scientific-staged-topology-1",
            "namespace": _namespace(plan),
            "settings": settings.model_dump(mode="json"),
            "body": body,
        }
        if record["request"] != expected or record["step"] != STEP:
            raise ValueError("provider request differs from frozen sibling contract")
    if (
        state["status"] == "complete"
        and final is None
        and state["stop_reason"]
        not in ("attempts_exhausted", "budget_exhausted", "replay_unavailable")
    ):
        raise ValueError("invalid terminal sibling reason")
    return state, records


def run_episode(root: Path, plan: dict, residual: dict, client: EpisodeClient) -> dict:
    """One bounded episode; cached responses resume without replacing the parent."""
    config = SiblingConfig.model_validate(plan["config"])
    if (
        client.namespace != _namespace(plan)
        or client.settings != config.model_settings
        or client.seed != 0
        or client.directory.resolve() != (root / "results/calls").resolve()
    ):
        raise ValueError("client differs from frozen sibling episode")
    state, records = _state(root, plan, residual)
    if state["status"] == "complete":
        return state
    path = root / "results/state.json"
    if residual["status"] != "ready":
        if records:
            raise ValueError("unavailable replay contains provider work")
        if residual["status"] == "prepared":
            raise ValueError("training replay has not run")
        state.update(status="complete", stop_reason="replay_unavailable")
        _save_state(path, state)
        return state
    for attempt in range(
        len(state["attempts"]), config.model_settings.attempts_per_step
    ):
        retry = state["attempts"][-1]["feedback"] if state["attempts"] else None
        try:
            record = client.call(
                system=review.system_prompt(config.feedback_policy),
                user=_user(plan, residual, retry),
                response_model=review.response_model(config.feedback_policy),
                step=STEP,
                attempt=attempt,
            )
        except ValueError as error:
            if str(error) not in {
                "total provider request budget exhausted",
                "total token budget cannot accommodate the next request",
            }:
                raise
            state["stop_reason"] = "budget_exhausted"
            break
        raw, decision, feedback = _evaluate(plan, residual["packet"], record)
        state["attempts"].append(
            {
                "attempt": attempt,
                "request_hash": record["request_hash"],
                "record_sha256": content_hash(record),
                "response": raw,
                "accepted": decision is not None,
                "feedback": feedback,
            }
        )
        state["decision"] = decision
        if decision is not None:
            state.update(status="complete", stop_reason=decision["outcome"])
        else:
            state["status"] = "running"
        _save_state(path, state)
        if decision is not None:
            return state
    state.update(
        status="complete", stop_reason=state["stop_reason"] or "attempts_exhausted"
    )
    _save_state(path, state)
    return state


def run(root: Path, base_url: str, *, wall_seconds: float | None = None) -> dict:
    """GPU worker; signal/deadline draining cannot create another episode."""
    with public._lock(root):
        plan, residual = verify(root)
        config = SiblingConfig.model_validate(plan["config"])
        deadline = time.monotonic() + (wall_seconds or config.wall_seconds)
        stopped = False

        def stop(*_):
            nonlocal stopped
            stopped = True

        prior = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            client = EpisodeClient(
                settings=config.model_settings,
                base_url=base_url,
                directory=root / "results/calls",
                namespace=_namespace(plan),
                seed=0,
                can_start=lambda: not stopped
                and time.monotonic() + config.shutdown_margin_seconds < deadline,
            )
            with suppress(DeferredCall):
                run_episode(root, plan, residual, client)
        finally:
            for sig, handler in prior.items():
                signal.signal(sig, handler)
        return _report(root, plan, residual)


def _child_contract(plan, residual, decision):
    """Derive one expected child contract for execution and read-only reporting."""
    frozen = residual["seed"]["parent"]["freeze"]
    parent = PublicFitRequest.model_validate(frozen["request"])
    final = decision["final"]
    child = PublicFitRequest.model_validate(
        {
            **parent.model_dump(mode="json"),
            "base_candidate": final["initialization"]["base_candidate"],
            "parameter_guesses": {},
            "source": {
                "stage": "controller_revision",
                "task_id": _namespace(plan),
                "artifact_sha256": public.content_sha256(decision),
            },
        }
    )
    if (
        public._lower(child)[0].validated.candidate.model_dump(mode="json")
        != final["candidate"]
    ):
        raise ValueError("child fit lowering differs from committed revision")
    return {
        "parent": parent,
        "child": child,
        "parameters": residual["seed"]["state"]["parameters"],
        "training": PublicSplit.model_validate(frozen["training"]),
        "validation": PublicSplit.model_validate(frozen["validation"]),
        "lineage": {
            "plan_sha256": plan["artifact_sha256"],
            "packet_sha256": residual["packet"]["packet_sha256"],
            "decision_sha256": public.content_sha256(decision),
            "parent_continuation": plan["config"]["selection"]["continuation_identity"],
        },
    }


def fit_child(root: Path) -> dict:
    """Allocate the unchanged fitter only to a real, accepted child revision."""
    with public._lock(root):
        plan, residual = verify(root)
        state, _ = _state(root, plan, residual)
        decision = state["decision"]
        if state["status"] != "complete":
            raise ValueError("proposer episode is not terminal")
        if not decision or decision["outcome"] != "committed":
            return _report(root, plan, residual)
        prepare_child_fit(
            directory=root / "child_fit", **_child_contract(plan, residual, decision)
        )
        execute_child_fit(root / "child_fit")
        return _report(root, plan, residual)


def _report(root, plan, residual):
    state, records = _state(root, plan, residual)
    child = (
        inspect_child_fit(root / "child_fit")
        if (root / "child_fit/freeze.json").exists()
        else None
    )
    committed = bool(state["decision"] and state["decision"]["outcome"] == "committed")
    if child and not committed:
        raise ValueError("uncommitted sibling contains a child fit")
    if child:
        expected = child_backend._freeze(
            **_child_contract(plan, residual, state["decision"])
        )
        if public._read(root / "child_fit/freeze.json") != expected:
            raise ValueError("child fit belongs to a different sibling decision")
    complete = state["status"] == "complete" and (
        not committed or (child and child["result"] is not None)
    )
    report = {
        "protocol": PROTOCOL,
        "feedback_policy": SiblingConfig.model_validate(plan["config"]).feedback_policy,
        "plan_sha256": plan["artifact_sha256"],
        "status": "complete" if complete else "partial",
        "replay_status": residual["status"],
        "proposal_outcome": state["stop_reason"],
        "parent": {
            "continuation_identity": plan["config"]["selection"][
                "continuation_identity"
            ],
            **residual["seed"]["state"],
        },
        "decision": state["decision"],
        "child": child,
        "physical_requests": len(records),
        "observed_tokens": sum(r.get("observed_total_tokens") or 0 for r in records),
        "budget_charge": sum(r.get("budget_charge", 0) for r in records),
        "unknown_usage_requests": sum(
            r.get("observed_total_tokens") is None for r in records
        ),
        "provider_seconds": sum(r.get("latency_seconds") or 0 for r in records),
        "unknown_latency_requests": sum(
            r.get("latency_seconds") is None for r in records
        ),
        "automatic_branch_selection": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "validation_evidence_sent_to_proposer": False,
        "limitation": LIMITATION,
    }
    public._write(root / "summary.json", report)
    return report


def report(root: Path) -> dict:
    """Publish a consistent read-only numerical report under the campaign lock."""
    with public._lock(root):
        return _report(root, *verify(root))


def audit_saved_citations(root: Path) -> dict:
    """Read historical replies under both citation policies; never resume a run."""
    state = sealed_read(root / "results/state.json")
    envelope = public._read(root / "evidence/result.json")
    result = envelope["result"]
    if public.content_sha256(result) != envelope["sha256"]:
        raise ValueError("saved evidence result digest differs")
    packet = result["packet"]
    if (
        public.content_sha256({k: v for k, v in packet.items() if k != "packet_sha256"})
        != packet["packet_sha256"]
        or state["packet_sha256"] != packet["packet_sha256"]
    ):
        raise ValueError("saved packet digest differs")
    rows = []
    for event in state["attempts"]:
        entry = {"attempt": event["attempt"], "historical_accepted": event["accepted"]}
        for policy in (review.LEGACY_POLICY, review.ROUTED_POLICY):
            try:
                reply, normalizations = review.checked_reply(
                    event["response"], packet, policy
                )
                outcome = {
                    "citation_contract_valid": True,
                    "evidence_ids": list(reply.evidence_ids),
                    "normalizations": normalizations,
                }
            except (ValueError, TypeError) as error:
                outcome = {"citation_contract_valid": False, "error": str(error)}
                if isinstance(error, review.EvidenceReferenceError):
                    outcome.update(
                        absent_evidence_ids=error.absent,
                        normalizations=error.normalizations,
                    )
            entry[policy] = outcome
        rows.append(entry)
    return {
        "protocol": "sibling-citation-replay-1",
        "source_state_sha256": state["artifact_sha256"],
        "packet_sha256": packet["packet_sha256"],
        "available_evidence_ids": sorted(review.evidence_ids(packet)),
        "attempts": rows,
        "live_llm_calls": 0,
        "parameter_fitting_performed": False,
        "historical_state_modified": False,
        "limitation": (
            "Saved reply/citation contracts only; no mathematical or scientific "
            "acceptance and no counterfactual fresh-run outcome."
        ),
    }
