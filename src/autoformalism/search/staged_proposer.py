"""Checkpointed two-stage proposer orchestration with routed feedback."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import Field

from autoformalism.expressions import ValidationContext
from autoformalism.llm.models import LLMCallResult, StagedLLMClient
from autoformalism.schemas import FunctionalCandidate, TopologyCandidate
from autoformalism.schemas.base import NonEmptyText, StrictSchema
from autoformalism.schemas.staged import Sha256Digest
from autoformalism.search.feedback_routing import (
    RevisionStage,
    RoutedProposerFeedback,
)
from autoformalism.staging import (
    StagedCandidateExpansion,
    expand_staged_candidate,
    topology_commitment_sha256,
)


class StagedProposerConfig(StrictSchema):
    """Immutable prompts and cache namespace for staged construction."""

    schema_version: Literal["staged-proposer-config-3"] = (
        "staged-proposer-config-3"
    )
    checkpoint_directory: Path
    run_fingerprint: Sha256Digest
    topology_system_prompt: NonEmptyText
    functional_system_prompt: NonEmptyText
    cache_only: bool = False


class StageCallReceipt(StrictSchema):
    """Resource and reproducibility record for one staged LLM operation."""

    request_hash: Sha256Digest
    checkpoint_hit: bool
    provider_cache_hit: bool
    logical_calls: int = Field(ge=0)
    provider_attempts: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)


class StagedProposalResult(StrictSchema):
    """Validated topology, functions, and executable expansion."""

    schema_version: Literal["staged-proposal-result-1"] = (
        "staged-proposal-result-1"
    )
    topology: TopologyCandidate
    functional: FunctionalCandidate
    expansion: StagedCandidateExpansion
    topology_call: StageCallReceipt | None = None
    functional_call: StageCallReceipt
    topology_reused: bool
    routed_feedback: RoutedProposerFeedback


class StagedProposer:
    """Construct a model using separate topology and function LLM calls.

    Passing ``fixed_topology`` creates a function-only refinement: the topology
    is not sent back through the topology proposer and its exact commitment is
    included in the function request.  Omitting it constructs both stages.
    """

    def __init__(
        self,
        *,
        client: StagedLLMClient,
        config: StagedProposerConfig,
    ) -> None:
        self._client = client
        self._config = config

    def construct(
        self,
        *,
        public_problem: str,
        context: ValidationContext,
        feedback: RoutedProposerFeedback,
        fixed_topology: TopologyCandidate | None = None,
        incumbent_topology: TopologyCandidate | None = None,
        cache_only: bool | None = None,
    ) -> StagedProposalResult:
        """Build or functionally refine one validated executable candidate."""
        if not public_problem.strip():
            raise ValueError("public_problem must not be empty")
        if fixed_topology is not None and incumbent_topology is not None:
            raise ValueError(
                "fixed_topology and incumbent_topology are mutually exclusive"
            )

        topology_call: StageCallReceipt | None = None
        if fixed_topology is None:
            topology_prompt = _topology_prompt(
                public_problem,
                context,
                feedback,
                incumbent_topology=incumbent_topology,
            )
            topology, topology_call = self._topology_stage(
                topology_prompt,
                context,
                cache_only=cache_only,
            )
            topology_reused = False
        else:
            topology = fixed_topology
            topology_reused = True

        functional_prompt = _functional_prompt(
            public_problem,
            topology,
            feedback,
        )
        functional, functional_call = self._functional_stage(
            functional_prompt,
            topology,
            context,
            cache_only=cache_only,
        )
        expansion = expand_staged_candidate(topology, functional, context)
        return StagedProposalResult(
            topology=topology,
            functional=functional,
            expansion=expansion,
            topology_call=topology_call,
            functional_call=functional_call,
            topology_reused=topology_reused,
            routed_feedback=feedback,
        )

    def _topology_stage(
        self,
        user_prompt: str,
        context: ValidationContext,
        *,
        cache_only: bool | None,
    ) -> tuple[TopologyCandidate, StageCallReceipt]:
        input_hash = self._stage_input_hash(
            "topology", self._config.topology_system_prompt, user_prompt
        )
        checkpoint = self._checkpoint_path("topology", input_hash)
        restored = _load_checkpoint(checkpoint, TopologyCandidate)
        if restored is not None:
            topology, request_hash = restored
            return topology, _checkpoint_receipt(request_hash)
        result = self._client.propose_topology(
            system_prompt=self._config.topology_system_prompt,
            user_prompt=user_prompt,
            context=context,
            cache_only=(
                self._config.cache_only if cache_only is None else cache_only
            ),
        )
        _write_checkpoint(checkpoint, input_hash, result)
        return result.parsed, _call_receipt(result)

    def _functional_stage(
        self,
        user_prompt: str,
        topology: TopologyCandidate,
        context: ValidationContext,
        *,
        cache_only: bool | None,
    ) -> tuple[FunctionalCandidate, StageCallReceipt]:
        input_hash = self._stage_input_hash(
            "functional", self._config.functional_system_prompt, user_prompt
        )
        checkpoint = self._checkpoint_path("functional", input_hash)
        restored = _load_checkpoint(checkpoint, FunctionalCandidate)
        if restored is not None:
            functional, request_hash = restored
            return functional, _checkpoint_receipt(request_hash)
        result = self._client.propose_functions(
            system_prompt=self._config.functional_system_prompt,
            user_prompt=user_prompt,
            topology=topology,
            context=context,
            cache_only=(
                self._config.cache_only if cache_only is None else cache_only
            ),
        )
        _write_checkpoint(checkpoint, input_hash, result)
        return result.parsed, _call_receipt(result)

    def _stage_input_hash(
        self,
        stage: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        payload = {
            "run_fingerprint": self._config.run_fingerprint,
            "protocol": self._config.schema_version,
            "stage": stage,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
        return _canonical_sha256(payload)

    def _checkpoint_path(self, stage: str, input_hash: str) -> Path:
        return self._config.checkpoint_directory / f"{stage}-{input_hash}.json"


def _topology_prompt(
    public_problem: str,
    context: ValidationContext,
    feedback: RoutedProposerFeedback,
    *,
    incumbent_topology: TopologyCandidate | None,
) -> str:
    payload = {
        "schema_version": "staged-topology-request-3",
        "task": (
            "Propose only the dynamic interaction graph. Do not provide equations, "
            "interaction functions, parameters, units, scopes, or numeric ranges. "
            "Separate direct auxiliary-state measurements from prediction-target "
            "mappings."
        ),
        "public_problem": public_problem,
        "runtime_contract": {
            "target_channels": list(context.targets),
            "required_target_mapping_count": len(context.targets),
            "required_target_mapping_channels": list(context.targets),
            "state_measurement_channels": list(context.auxiliaries),
            "available_forcing_channels": sorted(context.forcing_channels),
            "time_symbol": context.time_symbol,
            "observability_rule": (
                "A state is observed only through an identity measurement: either "
                "a target maps directly to that state, or state_measurements binds "
                "it to a supplied auxiliary channel. A target produced through an "
                "algebraic process does not reveal its internal states."
            ),
            "mapping_rule": (
                "Emit exactly required_target_mapping_count target_mappings: "
                "one for each required_target_mapping_channels entry, with no "
                "duplicate channel and no extra channel. "
                "state_measurements may bind states only to listed auxiliary "
                "channels; inputs and covariates remain external forcing."
            ),
            "interaction_target_rule": (
                "For every interaction, target_kind must be state_derivative "
                "exactly when target names a declared state, and must be "
                "algebraic_process exactly when target names a declared process. "
                "Do not declare an available forcing or auxiliary channel as a "
                "generated state unless it has genuine proposed dynamics."
            ),
        },
        "routed_feedback": feedback.for_stage(
            RevisionStage.TOPOLOGY,
            include_integrated_repairs=True,
        ),
        "incumbent_topology": (
            None
            if incumbent_topology is None
            else incumbent_topology.model_dump(mode="json")
        ),
        "incumbent_topology_commitment_sha256": (
            None
            if incumbent_topology is None
            else topology_commitment_sha256(incumbent_topology)
        ),
        "revision_rule": (
            "Construct an initial topology."
            if incumbent_topology is None
            else (
                "Revise the incumbent topology only where the routed graph "
                "feedback requires it; preserve unaffected nodes and interactions."
            )
        ),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _functional_prompt(
    public_problem: str,
    topology: TopologyCandidate,
    feedback: RoutedProposerFeedback,
) -> str:
    payload = {
        "schema_version": "staged-functional-request-3",
        "task": (
            "Assign one restricted expression to every interaction without changing "
            "the topology. Declare parameter names and qualitative roles only; omit "
            "parameter scopes and numeric ranges. The topology owns each outer plus "
            "or minus sign, so scalar interaction weights must use a nonnegative or "
            "positive role rather than the signed coefficient role. Offsets and "
            "shape parameters may use their corresponding roles. Initialize every "
            "latent state. Each expression is a right-hand-side value only: do not "
            "emit equations, assignments, derivative notation, or prime notation."
        ),
        "public_problem": public_problem,
        "immutable_topology": topology.model_dump(mode="json"),
        "topology_commitment_sha256": topology_commitment_sha256(topology),
        "routed_feedback": feedback.for_stage(
            RevisionStage.FUNCTIONAL_FORM,
            include_integrated_repairs=True,
        ),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _canonical_sha256(payload: object) -> str:
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _write_checkpoint(
    path: Path,
    input_hash: str,
    result: LLMCallResult[TopologyCandidate] | LLMCallResult[FunctionalCandidate],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    payload = {
        "schema_version": "staged-provider-checkpoint-1",
        "input_sha256": input_hash,
        "request_hash": result.request_hash,
        "parsed": result.parsed.model_dump(mode="json"),
    }
    temporary.write_text(
        f"{json.dumps(payload, sort_keys=True, ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path,
    model: type[TopologyCandidate] | type[FunctionalCandidate],
) -> tuple[TopologyCandidate | FunctionalCandidate, str] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "staged-provider-checkpoint-1":
        raise ValueError(f"unsupported staged checkpoint: {path}")
    if payload.get("input_sha256") != path.stem.rsplit("-", 1)[-1]:
        raise ValueError(f"staged checkpoint input hash mismatch: {path}")
    request_hash = payload.get("request_hash")
    if not isinstance(request_hash, str):
        raise ValueError(f"staged checkpoint request hash is invalid: {path}")
    return model.model_validate(payload.get("parsed")), request_hash


def _call_receipt(
    result: LLMCallResult[TopologyCandidate] | LLMCallResult[FunctionalCandidate],
) -> StageCallReceipt:
    usage = result.usage
    return StageCallReceipt(
        request_hash=result.request_hash,
        checkpoint_hit=False,
        provider_cache_hit=result.cache_hit,
        logical_calls=result.logical_calls,
        provider_attempts=result.provider_attempts,
        input_tokens=None if usage is None else usage.input_tokens,
        output_tokens=None if usage is None else usage.output_tokens,
        latency_ms=result.latency_ms,
    )


def _checkpoint_receipt(request_hash: str) -> StageCallReceipt:
    return StageCallReceipt(
        request_hash=request_hash,
        checkpoint_hit=True,
        provider_cache_hit=False,
        logical_calls=0,
        provider_attempts=0,
    )
