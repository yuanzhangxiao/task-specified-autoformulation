"""Public-only audit and localized feedback routing before numerical fitting."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import signal
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from autoformalism.construction import finalize_functional_draft
from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedTopologyClient,
    atomic_json,
    visible_response,
)
from autoformalism.rebuttal.staged_function_hybrid_campaign import (
    HybridFunctionCampaignConfig,
    summarize_hybrid,
)
from autoformalism.rebuttal.staged_topology_campaign import (
    StagedCampaignConfig,
    runtime_source_hash,
)
from autoformalism.schemas.candidate import CandidateModel, StateKind
from autoformalism.schemas.construction import FunctionalDraft
from autoformalism.schemas.staged_functions import (
    InteractionFunctionObligation,
    InteractionFunctionReply,
    LatentInitialReply,
    PrefitReviewCategory,
    PrefitReviewFinding,
    PrefitScientificReviewReply,
)
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search.staged_function_prompts import (
    render_interaction_function_system_prompt,
    render_latent_initial_system_prompt,
)
from autoformalism.search.staged_prefit_prompts import (
    render_prefit_function_revision_user_prompt,
    render_prefit_initial_revision_user_prompt,
    render_prefit_review_system_prompt,
    render_prefit_review_user_prompt,
)
from autoformalism.staged_functions import (
    apply_function_reply,
    apply_initial_reply,
    has_nonlinear_source_dependence,
    initial_symbols,
    repair_certified_outer_gain_role,
)
from autoformalism.staged_topology import content_hash, lower_topology


class PrefitFeedbackCampaignConfig(StagedCampaignConfig):
    """One source-bound critique and localized-revision experiment."""

    protocol: Literal["scientific-staged-prefit-feedback-1"]
    source_hybrid_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_categories: tuple[PrefitReviewCategory, ...] = Field(
        min_length=6,
        max_length=6,
    )
    maximum_revised_interactions: int = Field(default=3, ge=1, le=3)

    @model_validator(mode="after")
    def complete_review_contract(self) -> PrefitFeedbackCampaignConfig:
        """Require the fixed rubric once and keep the pilot small."""
        if set(self.review_categories) != set(PrefitReviewCategory):
            raise ValueError("prefit campaign requires every fixed review category")
        if len(self.review_categories) != len(set(self.review_categories)):
            raise ValueError("duplicate prefit review category")
        if len(self.public_cells) != 1 or self.diagnostic_fixtures:
            raise ValueError("prefit feedback pilot requires one public cell only")
        return self


_CATEGORY_PRIORITY = {
    PrefitReviewCategory.MECHANISM_TOPOLOGY: 0,
    PrefitReviewCategory.DIMENSIONAL_CONSISTENCY: 1,
    PrefitReviewCategory.DYNAMIC_PLAUSIBILITY: 2,
    PrefitReviewCategory.FUNCTIONAL_SEMANTICS: 3,
    PrefitReviewCategory.PARAMETER_PARSIMONY: 4,
    PrefitReviewCategory.LATENT_INITIALIZATION: 5,
}

_FUNCTION_CATEGORIES = {
    PrefitReviewCategory.DIMENSIONAL_CONSISTENCY,
    PrefitReviewCategory.DYNAMIC_PLAUSIBILITY,
    PrefitReviewCategory.FUNCTIONAL_SEMANTICS,
    PrefitReviewCategory.PARAMETER_PARSIMONY,
}


def prefit_launcher_hash() -> str:
    """Bind the worker, CLI, and cluster wrappers to every frozen plan."""
    repository = Path(__file__).resolve().parents[3]
    paths = (
        "scripts/staged_prefit_feedback_campaign.py",
        "scripts/hpc/run_staged_topology_server.sh",
        "scripts/hpc/staged_prefit_feedback_aces.slurm",
        "scripts/hpc/submit_staged_prefit_feedback_aces.sh",
    )
    return content_hash(
        {
            path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
            for path in paths
        }
    )


def freeze_prefit_campaign(
    config_path: Path,
    source_hybrid_plan_path: Path,
    source_results: Path,
    output: Path,
) -> dict[str, Any]:
    """Freeze exact completed source candidates before any new model call."""
    config = PrefitFeedbackCampaignConfig.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )
    source_plan = json.loads(source_hybrid_plan_path.read_text(encoding="utf-8"))
    source_digest = content_hash(
        {key: value for key, value in source_plan.items() if key != "plan_sha256"}
    )
    if (
        source_digest != source_plan.get("plan_sha256")
        or source_digest != config.source_hybrid_plan_sha256
    ):
        raise ValueError("source hybrid plan digest mismatch")
    if (
        source_plan.get("test_data_opened") is not False
        or source_plan.get("private_reference_opened") is not False
    ):
        raise ValueError("source hybrid plan crossed the public-only boundary")
    source_config = HybridFunctionCampaignConfig.model_validate(source_plan["config"])
    _require_matched_settings(config, source_config)

    records = []
    tasks = []
    for source_task in source_plan["tasks"]:
        terminal_path = source_results / source_task["task_id"] / "terminal.json"
        if not terminal_path.is_file():
            raise ValueError(
                f"missing source terminal result: {source_task['task_id']}"
            )
        record = json.loads(terminal_path.read_text(encoding="utf-8"))
        expected_identity = content_hash([source_digest, source_task])
        if record.get("identity") != expected_identity:
            raise ValueError("source terminal identity mismatch")
        result = record.get("result")
        if not isinstance(result, dict) or not result.get("complete_model"):
            raise ValueError("prefit campaign requires complete source candidates")
        if (
            result.get("parameter_fitting_performed") is not False
            or result.get("test_data_opened") is not False
            or result.get("private_reference_opened") is not False
        ):
            raise ValueError("source result crossed the pre-fitting boundary")
        audit = audit_prefit_candidate(
            PublicScientificBrief.model_validate(source_task["brief"]),
            source_task["source"],
            result,
        )
        records.append(record)
        tasks.append(
            {
                "task_id": f"{source_task['task_id']}_prefit_feedback",
                "source_task_id": source_task["task_id"],
                "seed": source_task["seed"],
                "brief": source_task["brief"],
                "context": source_task["context"],
                "source": source_task["source"],
                "source_result": result,
                "source_terminal_sha256": content_hash(record),
                "source_audit": audit,
            }
        )
    if len(tasks) != len(config.seeds):
        raise ValueError("source result count differs from frozen seed count")

    source_summary_path = source_results / "summary.json"
    if not source_summary_path.is_file():
        raise ValueError("missing source hybrid summary")
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary != summarize_hybrid(source_plan, records):
        raise ValueError("source hybrid summary differs from terminal results")

    plan = {
        "schema_version": "scientific-staged-prefit-feedback-plan-1",
        "config": config.model_dump(mode="json"),
        "source_hybrid_plan_sha256": source_digest,
        "source_hybrid_summary_sha256": content_hash(source_summary),
        "runtime_source_sha256": runtime_source_hash(),
        "launcher_sha256": prefit_launcher_hash(),
        "tasks": tasks,
        "routing_policy": (
            "Runtime selects the highest-priority failed category; function and "
            "initializer findings receive localized revisions, while mechanism-"
            "topology findings route backward without a forced function edit"
        ),
        "source_candidates_are_paired_controls": True,
        "automatic_winner_defined": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    plan["plan_sha256"] = content_hash(plan)
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != plan:
            raise ValueError("existing frozen prefit plan differs")
    else:
        atomic_json(output, plan)
    return plan


def _require_matched_settings(
    config: PrefitFeedbackCampaignConfig,
    source: HybridFunctionCampaignConfig,
) -> None:
    comparisons = (
        (config.model_settings, source.model_settings, "model settings"),
        (config.platform, source.platform, "platform"),
        (config.serving_image_sha256, source.serving_image_sha256, "image"),
        (config.served_context_tokens, source.served_context_tokens, "context"),
        (config.public_cells, source.public_cells, "public cells"),
        (config.seeds, source.seeds, "seeds"),
        (config.limits, source.limits, "modeling limits"),
    )
    for actual, expected, label in comparisons:
        if actual != expected:
            raise ValueError(f"prefit campaign {label} differ from source")


def audit_prefit_candidate(
    brief: PublicScientificBrief,
    source: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Expose deterministic facts without inventing scientific verdicts."""
    candidate = CandidateModel.model_validate(result["candidate"])
    draft = FunctionalDraft.model_validate(result["draft"])
    # Historical selected terms do not include an ID; their stable order follows
    # the frozen equation order and is reconstructed explicitly below.
    catalog = interaction_catalog(source, result)
    accepted_by_id = {item["interaction_id"]: item for item in catalog}
    expected_interactions = sum(
        len(EquationDefinition.model_validate(item).terms)
        for item in source["equations"]
    )
    target_channels = {
        item.name for item in brief.public_variables if item.data_role == "target"
    }
    mapped_channels = [item.channel for item in candidate.observation_mappings]
    latent_states = {
        item.name for item in candidate.states if item.kind is StateKind.LATENT
    }
    initialized_states = {item.state for item in candidate.initial_conditions}
    nonlinear_required = [
        item
        for item in catalog
        if item["functional_obligation"]["requires_nonlinear_source_dependence"]
    ]
    nonlinear_pass = all(item["nonlinear_source_syntax"] for item in nonlinear_required)
    parameter_uses: dict[str, list[str]] = {}
    for item in draft.interaction_functions:
        for parameter in item.parameters:
            parameter_uses.setdefault(parameter.name, []).append(item.interaction_id)
    checks = {
        "source_complete_model": bool(result.get("complete_model")),
        "source_public_structure_checks_passed": bool(
            source.get("public_structure_checks_passed")
        ),
        "target_coverage_exact": (
            len(mapped_channels) == len(set(mapped_channels))
            and set(mapped_channels) == target_channels
        ),
        "interaction_function_coverage_exact": (
            len(draft.interaction_functions) == expected_interactions
            and len(accepted_by_id) == expected_interactions
        ),
        "latent_initialization_coverage_exact": (
            {state for state in initialized_states if state in latent_states}
            == latent_states
        ),
        "nonlinear_obligations_satisfied": nonlinear_pass,
        "source_and_expression_contracts_validated": True,
    }
    return {
        "schema_version": "deterministic-prefit-fact-audit-1",
        "checks": checks,
        "deterministic_prefit_pass": all(checks.values()),
        "target_channels": sorted(target_channels),
        "mapped_target_channels": mapped_channels,
        "interaction_count": expected_interactions,
        "nonlinear_obligation_count": len(nonlinear_required),
        "parameter_count": len(candidate.parameters),
        "parameter_roles": {
            item.name: item.role.value for item in candidate.parameters
        },
        "parameter_interaction_uses": parameter_uses,
        "cross_interaction_parameter_count": sum(
            len(set(uses)) > 1 for uses in parameter_uses.values()
        ),
        "latent_states": sorted(latent_states),
        "interaction_catalog": catalog,
        "unresolved_scientific_categories": [
            item.value for item in PrefitReviewCategory
        ],
        "scientific_scope_note": (
            "Units, scientific adequacy, dynamic plausibility, parsimony, and "
            "initializer meaning are not certified by these deterministic facts"
        ),
        "parameter_fitting_performed": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }


def interaction_catalog(
    source: dict[str, Any], result: dict[str, Any]
) -> list[dict[str, Any]]:
    """Join immutable topology slots to accepted functions deterministically."""
    accepted = result.get("accepted_functions", [])
    provider_visible = result.get("provider_visible_accepted_functions", accepted)
    if len(provider_visible) != len(accepted):
        raise ValueError("provider-visible function count differs from compiled count")
    catalog = []
    offset = 0
    for equation_index, raw_equation in enumerate(source["equations"]):
        equation = EquationDefinition.model_validate(raw_equation)
        for term_index, term in enumerate(equation.terms):
            interaction_id = f"term_{equation_index}_{term_index}"
            item = accepted[offset]
            visible_item = provider_visible[offset]
            selected = item["selected_term"]
            if (
                selected["lhs"] != equation.name
                or tuple(selected["sources"]) != term.sources
                or selected["outer_sign"] != term.outer_sign
            ):
                raise ValueError("accepted function order differs from frozen topology")
            selected_fields = (
                "lhs",
                "definition",
                "sources",
                "outer_sign",
                "scientific_role",
                "functional_obligation",
            )
            if any(
                visible_item["selected_term"].get(key) != selected.get(key)
                for key in selected_fields
            ):
                raise ValueError(
                    "provider-visible function order differs from compiled topology"
                )
            expression = visible_item["expression"]
            tree = ast.parse(expression, mode="eval")
            obligation = selected["functional_obligation"]
            catalog.append(
                {
                    "interaction_id": interaction_id,
                    "lhs": equation.name,
                    "definition": equation.definition,
                    "sources": list(term.sources),
                    "outer_sign": term.outer_sign,
                    "scientific_role": term.scientific_role,
                    "expression": expression,
                    "parameters": visible_item["parameters"],
                    "compiled_expression": item["expression"],
                    "compiled_parameters": item["parameters"],
                    "functional_obligation": obligation,
                    "nonlinear_source_syntax": has_nonlinear_source_dependence(
                        tree, set(term.sources)
                    ),
                }
            )
            offset += 1
    if offset != len(accepted):
        raise ValueError("accepted function count differs from frozen topology")
    return catalog


def audit_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Summarize source audits without any provider or numerical call."""
    rows = [
        {
            "task_id": task["task_id"],
            "source_task_id": task["source_task_id"],
            "seed": task["seed"],
            "source_terminal_sha256": task["source_terminal_sha256"],
            "audit": task["source_audit"],
        }
        for task in plan["tasks"]
    ]
    return {
        "schema_version": "scientific-staged-prefit-source-audit-1",
        "status": "complete",
        "plan_sha256": plan["plan_sha256"],
        "candidate_count": len(rows),
        "deterministic_prefit_pass_rate": _ratio(
            sum(row["audit"]["deterministic_prefit_pass"] for row in rows),
            len(rows),
        ),
        "rows": rows,
        "new_llm_calls_made": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }


def run_prefit_campaign(
    plan_path: Path,
    output: Path,
    base_url: str,
    *,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Review and route each source candidate with deterministic resume."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if content_hash(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ) != plan.get("plan_sha256"):
        raise ValueError("frozen prefit plan digest mismatch")
    if runtime_source_hash() != plan["runtime_source_sha256"]:
        raise ValueError("runtime source differs from frozen prefit campaign")
    if prefit_launcher_hash() != plan["launcher_sha256"]:
        raise ValueError("launcher differs from frozen prefit campaign")
    config = PrefitFeedbackCampaignConfig.model_validate(plan["config"])
    deadline = time.monotonic() + (wall_seconds or config.wall_seconds)
    stop = False

    def drain(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    previous = {
        sig: signal.signal(sig, drain) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    records: list[dict[str, Any]] = []
    try:
        for task in plan["tasks"]:
            root = output / task["task_id"]
            root.mkdir(parents=True, exist_ok=True)
            with (root / "worker.lock").open("w") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                identity = content_hash([plan["plan_sha256"], task])
                terminal = root / "terminal.json"
                if terminal.exists():
                    record = json.loads(terminal.read_text(encoding="utf-8"))
                    if record["identity"] != identity:
                        raise ValueError("terminal result belongs to another task")
                else:
                    client = StagedTopologyClient(
                        settings=config.model_settings,
                        base_url=base_url,
                        directory=root / "calls",
                        namespace=identity,
                        seed=task["seed"],
                        can_start=lambda: not stop
                        and time.monotonic()
                        < deadline - config.shutdown_margin_seconds,
                    )
                    try:
                        result = _run_prefit_task(task, client, root, config)
                    except DeferredCall:
                        break
                    record = {
                        "identity": identity,
                        "task_id": task["task_id"],
                        "seed": task["seed"],
                        "result": result,
                    }
                    atomic_json(terminal, record)
                records.append(record)
                atomic_json(output / "summary.json", summarize_prefit(plan, records))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    summary = summarize_prefit(plan, records)
    atomic_json(output / "summary.json", summary)
    return summary


def _run_prefit_task(
    task: dict[str, Any],
    client: StagedTopologyClient,
    output: Path,
    config: PrefitFeedbackCampaignConfig,
) -> dict[str, Any]:
    brief = PublicScientificBrief.model_validate(task["brief"])
    context = ValidationContext.model_validate(task["context"])
    source_result = task["source_result"]
    inventory = tuple(
        ScientificVariable.model_validate(item) for item in task["source"]["inventory"]
    )
    equations = tuple(
        EquationDefinition.model_validate(item) for item in task["source"]["equations"]
    )
    topology, aliases = lower_topology(brief, inventory, equations, context)
    draft = FunctionalDraft.model_validate(source_result["draft"])
    source_draft = draft.model_dump(mode="json")
    catalog = interaction_catalog(task["source"], source_result)
    catalog_by_id = {item["interaction_id"]: item for item in catalog}
    catalog_index = {
        item["interaction_id"]: index for index, item in enumerate(catalog)
    }
    provider_visible_functions = list(
        source_result["provider_visible_accepted_functions"]
    )
    latent_states = {
        item.name for item in topology.states if item.kind is StateKind.LATENT
    }
    events: list[dict[str, Any]] = []

    def checkpoint(extra: dict[str, Any]) -> None:
        atomic_json(
            output / "progress.json",
            {
                "source_terminal_sha256": task["source_terminal_sha256"],
                "events": events,
                **extra,
            },
        )

    def request(
        step: str,
        system: str,
        user: str,
        response_model: type[Any],
        validate: Any,
    ) -> Any:
        diagnostic = None
        for attempt in range(client.settings.attempts_per_step):
            rendered_user = user
            if diagnostic is not None:
                rendered_user += "\nRUNTIME_CONTRACT_DIAGNOSTIC\n" + diagnostic
            record = client.call(
                system=system,
                user=rendered_user,
                response_model=response_model,
                step=step,
                attempt=attempt,
            )
            rejected: object = None
            try:
                rejected = visible_response(record)
                reply = response_model.model_validate(rejected)
                validate(reply)
            except (ValueError, TypeError, KeyError, ModelValidationError) as exc:
                diagnostic = json.dumps(
                    {"rejected_response": rejected, "error": str(exc)[:6000]}
                )
                events.append(
                    {
                        "step": step,
                        "attempt": attempt,
                        "accepted": False,
                        "request_hash": record["request_hash"],
                        "error": str(exc)[:6000],
                    }
                )
                checkpoint({"draft": draft.model_dump(mode="json")})
                continue
            events.append(
                {
                    "step": step,
                    "attempt": attempt,
                    "accepted": True,
                    "request_hash": record["request_hash"],
                }
            )
            return reply
        raise ValueError(f"bounded local repair exhausted for {step}")

    review_payload = _review_payload(task, source_result, task["source_audit"])
    review = request(
        "prefit_review_before",
        render_prefit_review_system_prompt(),
        render_prefit_review_user_prompt(review_payload),
        PrefitScientificReviewReply,
        lambda reply: _validate_review(reply, catalog_by_id, latent_states),
    )
    selected = select_prefit_finding(review)
    route = _route(selected)
    changed_interactions: list[str] = []
    changed_initials: list[str] = []
    deterministic_repairs: list[dict[str, Any]] = []
    error = None
    revised_candidate = None
    revised_audit = None
    review_after = None
    try:
        if route == "function_revision":
            assert selected is not None
            for interaction_id in selected.interaction_ids[
                : config.maximum_revised_interactions
            ]:
                catalog_item = catalog_by_id[interaction_id]
                accepted_context = list(provider_visible_functions)
                user_payload = {
                    "schema_version": "prefit-function-revision-request-1",
                    "public_brief": task["brief"],
                    "frozen_inventory": task["source"]["inventory"],
                    "frozen_equation_sketch": task["source"]["equations"],
                    "selected_term": {
                        key: catalog_item[key]
                        for key in (
                            "interaction_id",
                            "lhs",
                            "definition",
                            "sources",
                            "outer_sign",
                            "scientific_role",
                            "functional_obligation",
                        )
                    },
                    "incumbent_function": {
                        "expression": catalog_item["expression"],
                        "parameters": catalog_item["parameters"],
                    },
                    "accepted_functions": accepted_context,
                    "parameter_registry": {},
                    "selected_prefit_feedback": selected.model_dump(mode="json"),
                    "runtime_rules": {
                        "topology_is_immutable": True,
                        "only_selected_interaction_may_change": True,
                        "outer_sign_is_runtime_owned": True,
                        "parameter_identity_policy": "interaction_local",
                    },
                }

                def validate_function(
                    reply: InteractionFunctionReply,
                    selected_catalog_item: dict[str, Any] = catalog_item,
                    current_draft: FunctionalDraft = draft,
                    selected_interaction_id: str = interaction_id,
                ) -> None:
                    local, _ = repair_certified_outer_gain_role(
                        reply, set(selected_catalog_item["sources"])
                    )
                    apply_function_reply(
                        topology,
                        current_draft,
                        selected_interaction_id,
                        local,
                        context,
                        aliases,
                        InteractionFunctionObligation.model_validate(
                            selected_catalog_item["functional_obligation"]
                        ),
                    )

                reply = request(
                    f"prefit_function_revision_{interaction_id}",
                    render_interaction_function_system_prompt()
                    + "\nThis is a localized pre-fitting revision. Correct only "
                    "the selected feedback while preserving the frozen topology.",
                    render_prefit_function_revision_user_prompt(user_payload),
                    InteractionFunctionReply,
                    validate_function,
                )
                local, repairs = repair_certified_outer_gain_role(
                    reply, set(catalog_item["sources"])
                )
                draft = apply_function_reply(
                    topology,
                    draft,
                    interaction_id,
                    local,
                    context,
                    aliases,
                    InteractionFunctionObligation.model_validate(
                        catalog_item["functional_obligation"]
                    ),
                )
                changed_interactions.append(interaction_id)
                deterministic_repairs.extend(
                    item.model_dump(mode="json") for item in repairs
                )
                visible_index = catalog_index[interaction_id]
                provider_visible_functions[visible_index] = {
                    "selected_term": provider_visible_functions[visible_index][
                        "selected_term"
                    ],
                    "expression": local.expression,
                    "parameters": [
                        item.model_dump(mode="json") for item in local.parameters
                    ],
                }
        elif route == "initializer_revision":
            assert selected is not None
            for state in selected.latent_states:
                user_payload = {
                    "schema_version": "prefit-initial-revision-request-1",
                    "public_brief": task["brief"],
                    "frozen_inventory": task["source"]["inventory"],
                    "frozen_equation_sketch": task["source"]["equations"],
                    "selected_state": {"name": state},
                    "allowed_symbols": list(initial_symbols(context, aliases)),
                    "selected_prefit_feedback": selected.model_dump(mode="json"),
                    "runtime_rules": {
                        "topology_and_functions_are_immutable": True,
                        "only_selected_initializer_may_change": True,
                    },
                }

                def validate_initial(
                    reply: LatentInitialReply,
                    current_draft: FunctionalDraft = draft,
                    selected_state: str = state,
                ) -> None:
                    apply_initial_reply(
                        topology,
                        current_draft,
                        selected_state,
                        reply,
                        context,
                        aliases,
                    )

                reply = request(
                    f"prefit_initial_revision_{state}",
                    render_latent_initial_system_prompt()
                    + "\nThis is a localized pre-fitting revision. Correct only "
                    "the selected initializer feedback.",
                    render_prefit_initial_revision_user_prompt(user_payload),
                    LatentInitialReply,
                    validate_initial,
                )
                draft = apply_initial_reply(
                    topology, draft, state, reply, context, aliases
                )
                changed_initials.append(state)

        if route in {"function_revision", "initializer_revision"}:
            expansion = finalize_functional_draft(topology, draft, context)
            revised_candidate = expansion.candidate.model_dump(mode="json")
            revised_result = {
                **source_result,
                "candidate": revised_candidate,
                "draft": draft.model_dump(mode="json"),
                "accepted_functions": _accepted_from_draft(catalog, draft),
                "provider_visible_accepted_functions": provider_visible_functions,
            }
            revised_audit = audit_prefit_candidate(
                brief, task["source"], revised_result
            )
            _validate_locality(
                FunctionalDraft.model_validate(source_draft),
                draft,
                set(changed_interactions),
                set(changed_initials),
            )
            review_after = request(
                "prefit_review_after",
                render_prefit_review_system_prompt(),
                render_prefit_review_user_prompt(
                    _review_payload(task, revised_result, revised_audit)
                ),
                PrefitScientificReviewReply,
                lambda reply: _validate_review(
                    reply,
                    {
                        item["interaction_id"]: item
                        for item in revised_audit["interaction_catalog"]
                    },
                    latent_states,
                ),
            )
    except (ValueError, TypeError, KeyError, ModelValidationError) as exc:
        error = str(exc)[:6000]

    result = {
        "schema_version": "scientific-staged-prefit-feedback-result-1",
        "status": "complete" if error is None else "failed",
        "error": error,
        "source_task_id": task["source_task_id"],
        "source_terminal_sha256": task["source_terminal_sha256"],
        "source_candidate": source_result["candidate"],
        "source_audit": task["source_audit"],
        "review_before": review.model_dump(mode="json"),
        "selected_finding": (
            selected.model_dump(mode="json") if selected is not None else None
        ),
        "revision_route": route,
        "changed_interaction_ids": changed_interactions,
        "changed_initial_states": changed_initials,
        "deterministic_role_repairs": deterministic_repairs,
        "revised_candidate": revised_candidate,
        "revised_audit": revised_audit,
        "review_after": (
            review_after.model_dump(mode="json") if review_after is not None else None
        ),
        "events": events,
        "physical_requests": len(client.records),
        "observed_total_tokens": sum(
            item.get("observed_total_tokens") or 0 for item in client.records
        ),
        "provider_seconds": sum(
            item.get("latency_seconds", 0) for item in client.records
        ),
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "scientific_critic_called": True,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    checkpoint({"result": result})
    atomic_json(output / "result.json", result)
    return result


def _accepted_from_draft(
    catalog: list[dict[str, Any]],
    draft: FunctionalDraft,
) -> list[dict[str, Any]]:
    """Reconstruct ordered accepted functions for a deterministic post-audit."""
    functions = {item.interaction_id: item for item in draft.interaction_functions}
    return [
        {
            "selected_term": {
                key: item[key]
                for key in (
                    "lhs",
                    "definition",
                    "sources",
                    "outer_sign",
                    "scientific_role",
                    "functional_obligation",
                )
            },
            "expression": functions[item["interaction_id"]].expression,
            "parameters": [
                parameter.model_dump(mode="json")
                for parameter in functions[item["interaction_id"]].parameters
            ],
        }
        for item in catalog
    ]


def _review_payload(
    task: dict[str, Any], result: dict[str, Any], audit: dict[str, Any]
) -> dict[str, object]:
    candidate = CandidateModel.model_validate(result["candidate"])
    return {
        "schema_version": "prefit-scientific-review-request-1",
        "public_brief": task["brief"],
        "frozen_inventory": task["source"]["inventory"],
        "frozen_equation_sketch": task["source"]["equations"],
        "compiled_candidate": {
            "states": [item.model_dump(mode="json") for item in candidate.states],
            "state_equations": [
                item.model_dump(mode="json") for item in candidate.state_equations
            ],
            "processes": [item.model_dump(mode="json") for item in candidate.processes],
            "observation_mappings": [
                item.model_dump(mode="json") for item in candidate.observation_mappings
            ],
            "parameters": [
                item.model_dump(mode="json") for item in candidate.parameters
            ],
            "initial_conditions": [
                item.model_dump(mode="json") for item in candidate.initial_conditions
            ],
        },
        "interaction_catalog": audit["interaction_catalog"],
        "deterministic_prefit_facts": {
            key: value for key, value in audit.items() if key != "interaction_catalog"
        },
        "category_priority": [
            item.value
            for item in sorted(PrefitReviewCategory, key=_CATEGORY_PRIORITY.__getitem__)
        ],
    }


def _validate_review(
    review: PrefitScientificReviewReply,
    catalog: dict[str, dict[str, Any]],
    latent_states: set[str],
) -> None:
    """Reject invented anchors and unrouteable failed findings."""
    for finding in review.findings:
        unknown_interactions = set(finding.interaction_ids) - set(catalog)
        unknown_states = set(finding.latent_states) - latent_states
        if unknown_interactions:
            raise ValueError(
                f"review invented interaction anchors: {sorted(unknown_interactions)}"
            )
        if unknown_states:
            raise ValueError(
                f"review invented latent-state anchors: {sorted(unknown_states)}"
            )
        if finding.status != "fail":
            continue
        if finding.category in _FUNCTION_CATEGORIES and not finding.interaction_ids:
            raise ValueError("failed function category requires interaction anchors")
        if (
            finding.category is PrefitReviewCategory.LATENT_INITIALIZATION
            and not finding.latent_states
        ):
            raise ValueError("failed initialization requires latent-state anchors")


def select_prefit_finding(
    review: PrefitScientificReviewReply,
) -> PrefitReviewFinding | None:
    """Select one category deterministically; the critic selects only anchors."""
    failed = [item for item in review.findings if item.status == "fail"]
    if failed:
        return min(failed, key=lambda item: _CATEGORY_PRIORITY[item.category])
    uncertain = [item for item in review.findings if item.status == "uncertain"]
    if uncertain:
        return min(uncertain, key=lambda item: _CATEGORY_PRIORITY[item.category])
    return None


def _route(finding: PrefitReviewFinding | None) -> str:
    if finding is None:
        return "no_revision_review_passed"
    if finding.status == "uncertain":
        return "no_revision_review_uncertain"
    if finding.category is PrefitReviewCategory.MECHANISM_TOPOLOGY:
        return "topology_backtrack"
    if finding.category is PrefitReviewCategory.LATENT_INITIALIZATION:
        return "initializer_revision"
    return "function_revision"


def _validate_locality(
    before: FunctionalDraft,
    after: FunctionalDraft,
    allowed_interactions: set[str],
    allowed_initials: set[str],
) -> None:
    before_functions = {
        item.interaction_id: item.model_dump(mode="json")
        for item in before.interaction_functions
    }
    after_functions = {
        item.interaction_id: item.model_dump(mode="json")
        for item in after.interaction_functions
    }
    changed_functions = {
        key
        for key in set(before_functions) | set(after_functions)
        if before_functions.get(key) != after_functions.get(key)
    }
    before_initials = {
        item.state: item.model_dump(mode="json") for item in before.latent_initials
    }
    after_initials = {
        item.state: item.model_dump(mode="json") for item in after.latent_initials
    }
    changed_initials = {
        key
        for key in set(before_initials) | set(after_initials)
        if before_initials.get(key) != after_initials.get(key)
    }
    if not changed_functions <= allowed_interactions:
        raise ValueError("localized revision changed an unselected interaction")
    if not changed_initials <= allowed_initials:
        raise ValueError("localized revision changed an unselected initializer")


def summarize_prefit(
    plan: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Keep source controls, routing, completion, and resources separate."""
    by_id = {record["task_id"]: record for record in records}
    rows = []
    for task in plan["tasks"]:
        record = by_id.get(task["task_id"])
        result = record["result"] if record is not None else None
        before = result.get("review_before") if result else None
        after = result.get("review_after") if result else None
        selected = result.get("selected_finding") if result else None
        rows.append(
            {
                "task_id": task["task_id"],
                "source_task_id": task["source_task_id"],
                "seed": task["seed"],
                "result_present": record is not None,
                "status": result.get("status") if result else None,
                "error": result.get("error") if result else None,
                "source_deterministic_prefit_pass": task["source_audit"][
                    "deterministic_prefit_pass"
                ],
                "review_before_failure_count": _review_failure_count(before),
                "selected_category": selected.get("category") if selected else None,
                "revision_route": result.get("revision_route") if result else None,
                "changed_interaction_ids": (
                    result.get("changed_interaction_ids", []) if result else []
                ),
                "changed_initial_states": (
                    result.get("changed_initial_states", []) if result else []
                ),
                "revised_candidate_complete": bool(
                    result and result.get("revised_candidate")
                ),
                "revised_deterministic_prefit_pass": (
                    result.get("revised_audit", {}).get("deterministic_prefit_pass")
                    if result and result.get("revised_audit")
                    else None
                ),
                "review_after_failure_count": _review_failure_count(after),
                "selected_category_passed_after": _selected_passed_after(
                    selected, after
                ),
                "physical_requests": result.get("physical_requests", 0)
                if result
                else 0,
                "observed_total_tokens": (
                    result.get("observed_total_tokens", 0) if result else 0
                ),
                "provider_seconds": result.get("provider_seconds", 0) if result else 0,
            }
        )
    terminal = sum(row["result_present"] for row in rows)
    revised = [row for row in rows if row["revised_candidate_complete"]]
    return {
        "schema_version": "scientific-staged-prefit-feedback-summary-1",
        "status": "complete" if terminal == len(rows) else "incomplete",
        "plan_sha256": plan["plan_sha256"],
        "source_hybrid_plan_sha256": plan["source_hybrid_plan_sha256"],
        "planned_tasks": len(rows),
        "terminal_results": terminal,
        "source_deterministic_prefit_pass_rate": _ratio(
            sum(row["source_deterministic_prefit_pass"] for row in rows), len(rows)
        ),
        "topology_backtrack_count": sum(
            row["revision_route"] == "topology_backtrack" for row in rows
        ),
        "function_revision_count": sum(
            row["revision_route"] == "function_revision" for row in rows
        ),
        "initializer_revision_count": sum(
            row["revision_route"] == "initializer_revision" for row in rows
        ),
        "revised_candidate_count": len(revised),
        "revised_deterministic_prefit_pass_rate": _ratio(
            sum(row["revised_deterministic_prefit_pass"] is True for row in revised),
            len(revised),
        ),
        "selected_category_postreview_pass_rate": _ratio(
            sum(row["selected_category_passed_after"] is True for row in revised),
            len(revised),
        ),
        "physical_requests": sum(row["physical_requests"] for row in rows),
        "observed_total_tokens": sum(row["observed_total_tokens"] for row in rows),
        "provider_seconds": sum(row["provider_seconds"] for row in rows),
        "rows": rows,
        "source_candidates_are_paired_controls": True,
        "automatic_winner_defined": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }


def _review_failure_count(review: dict[str, Any] | None) -> int | None:
    if review is None:
        return None
    return sum(item["status"] == "fail" for item in review["findings"])


def _selected_passed_after(
    selected: dict[str, Any] | None, after: dict[str, Any] | None
) -> bool | None:
    if selected is None or after is None:
        return None
    match = next(
        item for item in after["findings"] if item["category"] == selected["category"]
    )
    return match["status"] == "pass"


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


__all__ = [
    "PrefitFeedbackCampaignConfig",
    "audit_from_plan",
    "audit_prefit_candidate",
    "freeze_prefit_campaign",
    "prefit_launcher_hash",
    "run_prefit_campaign",
    "select_prefit_finding",
    "summarize_prefit",
]
