"""Bounded staged construction, fixed fitting and two public-feedback visits."""

from __future__ import annotations

import json
import math
import signal
from pathlib import Path
from time import monotonic

from autoformalism.expressions import (
    ModelValidationError,
    ValidationContext,
    compile_candidate,
)
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.llm.review_revision import RevisionClient
from autoformalism.llm.staged_topology import DeferredCall, visible_response
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    evaluate_mechanisms,
)
from autoformalism.rebuttal.prefit_construction_audit import reconstruct
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records, _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.repair_comparison import (
    BudgetedRepairClient,
    RepairBudgetExceeded,
)
from autoformalism.rebuttal.staged_multiround_feedback_campaign import (
    RevisionContractError,
)
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.public_fitting import (
    PublicFitRequest,
    PublicSplit,
)
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search import numerical_sibling as revision
from autoformalism.search import review_model_edits as content_edits
from autoformalism.search.residual_evidence import build_residual_evidence
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.search.staged_topology_runner import run_staged_topology
from autoformalism.search.training_evidence import TrainingEvidence
from autoformalism.staged_topology import content_hash
from autoformalism.targets import PublicTargetContract, evaluate_public_targets


def _brief(cell: dict, task: dict) -> PublicScientificBrief:
    value = PublicScientificBrief.model_validate(cell["brief"])
    if task["arm"] == "no_latent":
        value = value.model_copy(
            update={
                "scientific_context": value.scientific_context
                + (
                    "\nExperimental no-persistent-latent ablation: only public target "
                    "variables may have differential equations. Other generated "
                    "variables must be algebraic processes; do not introduce "
                    "additional persistent states. Keep the public scientific task "
                    "and all supplied-channel rules."
                )
            }
        )
    if task["arm"] == "no_spec":
        value = value.model_copy(
            update={
                "scientific_context": (
                    "Construct a continuous-time model of the supplied target channels "
                    "from the permitted inputs and auxiliaries. Use only training "
                    "observations. Target trajectories cannot be model inputs. "
                    "Persistent hidden states and algebraic processes are permitted."
                ),
                "requirements": (),
                "target_dependencies": (),
            }
        )
    return value


def _bundle(cell, task, brief, topology, functions):
    saved = {"topology": topology, "functions": functions}
    audit = reconstruct({**cell, "brief": brief.model_dump(mode="json")}, saved)
    if not audit["certificate"]["passed"]:
        raise ValueError("construction reconstruction did not pass")
    return {
        "source_task": task,
        "brief": brief.model_dump(mode="json"),
        "context": cell["context"],
        "topology": {k: topology[k] for k in ("inventory", "equations", "topology")},
        "candidate": audit["handoff"]["candidate"],
        "initialization": audit["handoff"]["initialization"],
        "slots": audit["function_slots"],
        "scientific_review_facts": audit["scientific_review_facts"],
    }


def certificates(bundle: dict, cell: dict, task: dict) -> dict:
    """Actual equations determine checks; unknown semantics are never certified."""
    candidate = CandidateModel.model_validate(bundle["candidate"])
    context = ValidationContext.model_validate(bundle["initialization"]["context"])
    model = compile_candidate(candidate, context)
    mechanisms = evaluate_mechanisms(
        candidate, MechanismEvaluationSpec.model_validate(cell["mechanism_spec"])
    )
    target = evaluate_public_targets(
        candidate, PublicTargetContract.model_validate(cell["target_contract"])
    )
    hidden = sorted(
        set(model.state_names) - set(model.direct_state_observation_channels)
    )
    target_payload = target.model_dump(mode="json")
    target_failed = any(p["status"] == "failed" for p in target_payload["predicates"])
    mechanism_failed = any(p.status == "failed" for p in mechanisms.mechanism_results)
    non_target_states = sorted(
        state
        for state in model.state_names
        if model.direct_state_observation_channels.get(state) not in context.targets
    )
    ablation_pass = task["arm"] != "no_latent" or not non_target_states
    eligible = not target_failed and not mechanism_failed and ablation_pass
    if task["arm"] == "no_spec":
        # Withheld scientific requirements are scored only after construction;
        # they must not secretly constrain the ablated model's selection.
        eligible = all(
            p["status"] != "failed"
            for p in target_payload["predicates"]
            if p["predicate"]
            in {"explicit_observation_mapping", "generated_model_path"}
        )
    return {
        "runtime_valid": True,
        "targets": target_payload,
        "mechanisms": mechanisms.model_dump(mode="json"),
        "latent_states": hidden,
        "differential_states_outside_targets": non_target_states,
        "ablation_constraint_pass": ablation_pass,
        "eligible_for_development_selection": eligible,
        "all_public_graph_requirements_certified": (
            mechanisms.mechanism_compliance_complete
            and mechanisms.mechanism_compliance == 1
        ),
        "scientific_correctness_certified": False,
    }


def request_for(
    bundle: dict, plan: dict, task: dict, round_index: int
) -> PublicFitRequest:
    initial = bundle["initialization"]
    base_names = {p["name"] for p in initial["base_candidate"]["parameters"]}
    return PublicFitRequest.model_validate(
        {
            "base_candidate": initial["base_candidate"],
            "context": bundle["context"],
            "initialization_plan": initial["plan"],
            "parameter_guesses": {
                k: v for k, v in initial["guesses"].items() if k in base_names
            },
            "profile": plan["config"]["fit_profile"],
            "random_seed": 20260916 + task["seed"],
            "source": {
                "stage": "construction" if round_index == 0 else "controller_revision",
                "task_id": f"{task['task_id']}/round_{round_index}",
                "artifact_sha256": content_hash(bundle),
            },
        }
    )


def revision_prompt() -> str:
    """Only the requirements actually present in this visit's brief are active."""
    return (
        revision.system_prompt(revision.ROUTED_POLICY)
        .replace(
            "Retain the bound public nonlinear feedback\nrequirement.",
            "Retain only the scientific obligations explicitly present in the supplied "
            "public brief. An empty requirement list adds no hidden "
            "scientific obligation.",
        )
        .replace("anonymous public task", "supplied public task")
    )


def _client(root, plan, task, index, base_url, can_start, transport=None):
    directory = io.round_path(root, task, index) / "calls"
    namespace = content_hash([plan["artifact_sha256"], task, index])
    _cache_records(directory, namespace)
    settings = io.DeadlineConfig.model_validate(plan["config"]).model_settings
    if index and plan["protocol"] != io.CONTENT_PROTOCOL:
        settings = settings.model_copy(
            update={
                "maximum_requests": 3,
                "attempts_per_step": 3,
                "maximum_total_tokens": 98304,
            }
        )
    kwargs = {} if transport is None else {"transport": transport}
    client_type = (
        RevisionClient
        if index and plan["protocol"] == io.CONTENT_PROTOCOL
        else BudgetedRepairClient
    )
    return client_type(
        settings=settings,
        seed=task["seed"],
        namespace=namespace,
        directory=directory,
        base_url=base_url,
        can_start=can_start,
        **kwargs,
    )


def _certificate_feedback(certificate: dict, task: dict) -> dict:
    """Expose failed public predicates without leaking withheld ablation criteria."""
    targets = [
        p
        for p in certificate["targets"]["predicates"]
        if p["status"] == "failed"
        and (
            task["arm"] != "no_spec"
            or p["predicate"]
            in {"explicit_observation_mapping", "generated_model_path"}
        )
    ]
    result = {"failed_target_predicates": targets}
    if task["arm"] != "no_spec":
        result["failed_mechanisms"] = [
            p
            for p in certificate["mechanisms"]["mechanism_results"]
            if p["status"] == "failed"
        ]
    if not certificate["ablation_constraint_pass"]:
        result["no_latent_forbidden_states"] = certificate[
            "differential_states_outside_targets"
        ]
    return result


def _content_revision(plan: dict, task: dict, parent: dict, client) -> dict:
    """Three complete content patches; infer routes and recheck each closure."""
    selected = parent["selected"]
    packet = selected.get("packet")
    if packet is None:
        return {"status": "residual_evidence_unavailable"}
    bundle = selected["bundle"]
    attempts, feedback = [], None
    for attempt in range(3):
        raw, record = None, None
        user = content_edits.payload(
            bundle, packet, selected["fit"]["parameters"], feedback
        )
        try:
            record = client.call(
                system=content_edits.SYSTEM_PROMPT,
                user=json.dumps(user, sort_keys=True, separators=(",", ":")),
                response_model=content_edits.ModelEdits,
                step="review_model_content",
                attempt=attempt,
            )
            raw = visible_response(record)
            decision = content_edits.apply_edits(bundle, packet, raw)
            certificate = None
            if decision["bundle"] is not None:
                certificate = certificates(
                    decision["bundle"], plan["cells"][task["cell"]], task
                )
                if not certificate["eligible_for_development_selection"]:
                    raise RevisionContractError(
                        "PUBLIC_MODEL_REQUIREMENTS",
                        "Reconstructed model fails public target/graph or ablation "
                        "requirements. Preserve the active brief's requirements and "
                        "supply complete target-generating definitions.",
                        **_certificate_feedback(certificate, task),
                    )
            attempts.append(
                {
                    "request_hash": record["request_hash"],
                    "record_sha256": content_hash(record),
                    "accepted": True,
                    "raw": raw,
                }
            )
            return {
                "status": decision["outcome"],
                "bundle": decision["bundle"],
                "certificate": certificate,
                "decision": decision,
                "attempts": attempts,
                "revision_policy": content_edits.POLICY,
            }
        except RepairBudgetExceeded as error:
            return {
                "status": "revision_failed",
                "error": str(error),
                "attempts": attempts,
            }
        except (ValueError, KeyError, TypeError, ModelValidationError) as error:
            if record is None:
                raise
            feedback = {
                "code": getattr(error, "code", "MODEL_CONTENT_CONTRACT"),
                "stage": "model_construction"
                if raw is not None
                else "provider_delivery",
                "message": str(error)[:6000],
                "rejected_patch": raw,
                "incumbent_unchanged": True,
                "attempts_remaining": 2 - attempt,
            }
            if isinstance(error, RevisionContractError):
                feedback["details"] = error.details
            elif isinstance(error, ModelValidationError):
                feedback["details"] = [
                    {"code": d.code, "location": d.location, "message": d.message}
                    for d in error.diagnostics
                ]
            attempts.append(
                {
                    "request_hash": record["request_hash"],
                    "record_sha256": content_hash(record),
                    "accepted": False,
                    "feedback": feedback,
                    "raw": raw,
                }
            )
    return {
        "status": "revision_failed",
        "attempts": attempts,
        "revision_policy": content_edits.POLICY,
    }


def propose_one(root: Path, plan: dict, task: dict, index: int, client) -> dict | None:
    """Resume cached calls at exact stage inputs; never reallocate spent attempts."""
    io.require_open(root)
    directory = io.round_path(root, task, index)
    path = directory / "proposal.json"
    if path.exists():
        return sealed_read(path)
    cell = plan["cells"][task["cell"]]
    parent = io.read_round(root, task, index - 1) if index else None
    if index and parent is None:
        return None
    payload = {
        "task": task,
        "round": index,
        "parent_sha256": parent["artifact_sha256"] if parent else None,
        "bundle": None,
        "status": "construction_failed",
        "decision": None,
    }
    if index and (parent["selected"] is None or parent["closed"]):
        payload["status"] = "closed_lineage"
    elif task["arm"] == "refit_only":
        if index:
            payload.update(
                status="unchanged_refit", bundle=parent["selected"]["bundle"]
            )
        else:
            payload["status"] = "shared_full_round_zero"
    elif index == 0:
        brief = _brief(cell, task)
        evidence = (
            None
            if task["arm"] == "brief_only"
            else TrainingEvidence.model_validate(cell["evidence"])
        )
        topology = run_staged_topology(
            brief,
            ValidationContext.model_validate(cell["context"]),
            client,
            directory / "topology",
            hybrid_variable_construction=True,
            audit_public_polarity_policy=True,
            proposer_owns_unfixed_signs=True,
            training_evidence=evidence,
        )
        functions = None
        if topology["complete_topology"] and topology["public_structure_checks_passed"]:
            functions = run_staged_functions(
                brief,
                ValidationContext.model_validate(cell["context"]),
                topology,
                client,
                directory / "functions",
                generation_granularity="equation_batch_atomic_repair",
                function_repair_policy="certified_outer_gain",
                initialization_policy="causal_training",
                training_evidence=evidence,
            )
        payload.update(topology=topology, functions=functions)
        if functions and functions["complete_model"]:
            try:
                bundle = _bundle(cell, task, brief, topology, functions)
                certificate = certificates(bundle, cell, task)
                payload.update(
                    bundle=bundle,
                    certificate=certificate,
                    status="constructed"
                    if certificate["eligible_for_development_selection"]
                    else "requirement_failed",
                )
            except (ValueError, KeyError, ModelValidationError) as error:
                payload.update(status="contract_failed", error=str(error))
    elif plan["protocol"] == io.CONTENT_PROTOCOL:
        payload.update(_content_revision(plan, task, parent, client))
    else:
        selected = parent["selected"]
        packet = selected.get("packet")
        if packet is None:
            payload["status"] = "residual_evidence_unavailable"
        else:
            bundle = selected["bundle"]
            attempts, feedback = [], None
            for attempt in range(3):
                raw, record = None, None
                user = revision.payload(
                    bundle,
                    [],
                    packet,
                    selected["fit"]["parameters"],
                    feedback,
                    policy=revision.ROUTED_POLICY,
                )
                try:
                    record = client.call(
                        system=revision_prompt(),
                        user=json.dumps(user, sort_keys=True, separators=(",", ":")),
                        response_model=revision.response_model(revision.ROUTED_POLICY),
                        step="review_deadline_revision",
                        attempt=attempt,
                    )
                    raw = visible_response(record)
                    decision = revision.apply_revision(
                        bundle,
                        [],
                        packet,
                        raw,
                        policy=revision.ROUTED_POLICY,
                        inherit_existing_parameters=True,
                    )
                    attempts.append(
                        {
                            "request_hash": record["request_hash"],
                            "record_sha256": content_hash(record),
                            "accepted": True,
                            "raw": raw,
                        }
                    )
                    payload.update(status=decision["outcome"], decision=decision)
                    if decision["final"] is not None:
                        final = decision["final"]
                        new_bundle = {
                            **bundle,
                            "candidate": final["candidate"],
                            "initialization": final["initialization"],
                            "slots": [
                                final["selected_function"]
                                if s["interaction_id"] == final["selected_interaction"]
                                else s
                                for s in bundle["slots"]
                            ],
                        }
                        certificate = certificates(new_bundle, cell, task)
                        payload.update(bundle=new_bundle, certificate=certificate)
                        if not certificate["eligible_for_development_selection"]:
                            payload["status"] = "requirement_failed"
                    break
                except RepairBudgetExceeded as error:
                    payload.update(status="provider_budget_exhausted", error=str(error))
                    break
                except (ValueError, KeyError, TypeError, ModelValidationError) as error:
                    if record is None:
                        raise
                    feedback = revision.failure_feedback(bundle, record, raw, error)
                    attempts.append(
                        {
                            "request_hash": record["request_hash"],
                            "record_sha256": content_hash(record),
                            "accepted": False,
                            "feedback": feedback,
                            "raw": raw,
                        }
                    )
                    payload["status"] = "revision_failed"
            payload["attempts"] = attempts
    payload["cost"] = _cost(client.records)
    payload["test_data_opened"] = False
    return sealed_write(path, payload)


def replay_packet(root, directory, request, result, training):
    """Fixed-parameter training evidence only, with its own consumed replay marker."""
    path = directory / "packet.json"
    if path.exists():
        return sealed_read(path)["packet"]
    marker = directory / "replay_started.json"
    if marker.exists():
        return None
    sealed_write(
        marker, {"result_sha256": content_hash(result.model_dump(mode="json"))}
    )
    model, _, _ = public._lower(request)
    split = public.unpack_split(training)
    settings = FitConfig(
        integration_method="Radau",
        relative_tolerance=1e-7,
        absolute_tolerance=1e-9,
        allow_derivative_regression=False,
        maximum_wall_time_seconds=300,
    )
    deadline, predictions = monotonic() + 300, {}
    for row in split.trajectories:
        sim = simulate_trajectory(
            model,
            row,
            result.parameters,
            {},
            settings,
            deadline=deadline,
            reset_observed_states=False,
        )
        if not sim.success:
            sealed_write(path, {"packet": None, "error": sim.message})
            return None
        predictions[row.trajectory_id] = {
            "time": row.time.tolist(),
            "predictions": {k: v.tolist() for k, v in sim.predictions.items()},
        }
    packet = build_residual_evidence(
        split,
        model.validated.context,
        predictions,
        candidate_sha256=content_hash(
            model.validated.candidate.model_dump(mode="json")
        ),
        parameters=result.parameters,
        numerical_status={
            "feedback_status": "budget_limited_unresolved"
            if result.budget_exhausted
            else "local_optimizer_stopped",
            "native_optimizer_converged": result.native_optimizer_converged,
            "budget_exhausted": result.budget_exhausted,
        },
    )
    if not math.isclose(
        packet.normalized_mse,
        result.training.normalized_mse,
        rel_tol=1e-5,
        abs_tol=1e-8,
    ):
        sealed_write(
            path,
            {"packet": None, "error": "training replay disagrees with retained score"},
        )
        return None
    sealed_write(path, {"packet": packet.model_dump(mode="json"), "error": None})
    return packet.model_dump(mode="json")


def selection_key(value):
    if value is None or not value["certificate"]["eligible_for_development_selection"]:
        return (math.inf, math.inf)
    fit = value["fit"]
    if not fit["training"]["available"] or not fit["validation"]["available"]:
        return (math.inf, math.inf)
    from autoformalism.rebuttal.final_evaluation import _additive_term_count

    candidate = value["bundle"]["candidate"]
    terms = sum(_additive_term_count(e["rhs"]) for e in candidate["state_equations"])
    terms += sum(_additive_term_count(e["expression"]) for e in candidate["processes"])
    return (fit["validation"]["normalized_mse"], terms)


def fit_one(root: Path, plan: dict, task: dict, index: int) -> dict | None:
    """Preserve the incumbent; changes and unchanged controls get the same profile."""
    io.require_open(root)
    directory = io.round_path(root, task, index)
    with public._lock(directory):
        old = io.read_round(root, task, index)
        if old is not None:
            return old
        if task["arm"] == "refit_only" and index == 0:
            shared_task = next(
                t for t in plan["tasks"] if t["task_id"] == task["shared_round_zero"]
            )
            shared = io.read_round(root, shared_task, 0)
            if shared is None:
                return None
            return sealed_write(
                directory / "result.json",
                {
                    "task": task,
                    "round": 0,
                    "status": "shared_full_round_zero",
                    "selected": shared["selected"],
                    "closed": shared["selected"] is None,
                    "trial": None,
                    "source_sha256": shared["artifact_sha256"],
                    "test_data_opened": False,
                    "cost": {"requests": 0},
                },
            )
        parent = io.read_round(root, task, index - 1) if index else None
        if index and parent is None:
            return None
        proposal_path = directory / "proposal.json"
        if not proposal_path.exists():
            return None
        proposal = sealed_read(proposal_path)
        if proposal["parent_sha256"] != (parent["artifact_sha256"] if parent else None):
            raise ValueError("proposal names a stale parent round")
        incumbent = parent["selected"] if parent else None
        trial = None
        status = proposal["status"]
        if status in {"constructed", "committed", "unchanged_refit"}:
            bundle = proposal["bundle"]
            certificate = certificates(bundle, plan["cells"][task["cell"]], task)
            request = request_for(bundle, plan, task, index)
            cell = plan["cells"][task["cell"]]
            training = PublicSplit.model_validate(cell["training"])
            validation = PublicSplit.model_validate(cell["validation"])
            fit_directory = directory / "fit"
            if incumbent is None:
                public.prepare_fit(request, training, validation, fit_directory)
                result = public.execute_fit(fit_directory)
            else:
                sibling_fit.prepare_child_fit(
                    PublicFitRequest.model_validate(incumbent["request"]),
                    request,
                    incumbent["fit"]["parameters"],
                    training,
                    validation,
                    fit_directory,
                    lineage={
                        "campaign": plan["artifact_sha256"],
                        "proposal": proposal["artifact_sha256"],
                    },
                    **(
                        {"allow_initialization_changes": True}
                        if plan["protocol"] == io.CONTENT_PROTOCOL
                        else {}
                    ),
                )
                result = sibling_fit.execute_child_fit(fit_directory)
            packet = None
            if result.training.available and result.parameters is not None:
                packet = replay_packet(root, directory, request, result, training)
            trial = {
                "bundle": bundle,
                "request": request.model_dump(mode="json"),
                "certificate": certificate,
                "fit": result.model_dump(mode="json"),
                "packet": packet,
                "origin_task": task["task_id"],
                "origin_round": index,
                "fit_result_sha256": public.content_sha256(
                    public._read(fit_directory / "result.json")
                ),
            }
            status = result.status
        selected = incumbent
        if selection_key(trial) < selection_key(incumbent):
            selected = trial
        closed = selected is None or proposal["status"] in {
            "closed_lineage",
            "no_change",
            "topology_revision_needed",
            "residual_evidence_unavailable",
            "unchanged_canonical_function",
            "revision_failed",
            "provider_budget_exhausted",
            "requirement_failed",
        }
        return sealed_write(
            directory / "result.json",
            {
                "task": task,
                "round": index,
                "status": status,
                "trial": trial,
                "selected": selected,
                "closed": closed,
                "proposal_sha256": proposal["artifact_sha256"],
                "parent_sha256": parent["artifact_sha256"] if parent else None,
                "selection_uses": "validation_only_then_complexity",
                "cost": proposal["cost"],
                "test_data_opened": False,
                "selected_new_trial": selected is not None and selected is trial,
            },
        )


def run_proposals(root: Path, index: int, base_url: str, *, wall_seconds=None) -> None:
    io.require_open(root)
    plan = io.verify(root)
    config = io.DeadlineConfig.model_validate(plan["config"])
    if index not in range(config.rounds):
        raise ValueError("round not in frozen matrix")
    deadline = monotonic() + (wall_seconds or config.wall_seconds)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for task in plan["tasks"]:
            if stopped or monotonic() >= deadline - config.shutdown_margin_seconds:
                break
            directory = io.round_path(root, task, index)
            with public._lock(directory):
                client = _client(
                    root,
                    plan,
                    task,
                    index,
                    base_url,
                    lambda: not stopped
                    and monotonic() < deadline - config.shutdown_margin_seconds,
                )
                try:
                    propose_one(root, plan, task, index, client)
                except DeferredCall:
                    break
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
