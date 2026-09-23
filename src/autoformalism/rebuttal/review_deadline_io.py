"""Frozen public-only experiment matrix for the internal review deadline."""

from __future__ import annotations

import fcntl
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.data import BenchmarkRegistry
from autoformalism.fitting import public_fitting as public
from autoformalism.llm.staged_topology import StagedModelSettings
from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec
from autoformalism.rebuttal.prefit_construction_campaign import load_development
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import public_validation_context
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.staged_topology import ModelingLimits
from autoformalism.search.training_evidence import (
    EvidenceSettings,
    build_training_evidence,
)
from autoformalism.staged_topology import build_scientific_brief, content_hash
from autoformalism.targets import PublicTargetContract

PROTOCOL = "review-deadline-1"
CONTENT_PROTOCOL = "review-deadline-2"
CONTINUATION_PROTOCOL = "review-deadline-3"
PARAMETER_PROTOCOL = "review-deadline-4"
REVISION_PROTOCOL = "review-deadline-5"
MULTI_PROTOCOL = "review-deadline-6"
RESPONSE_PROTOCOL = "review-deadline-7"
INTEGRITY_PROTOCOL = "review-deadline-8"
FRESH_PROTOCOL = "shared-multi-pruning-1"
INTEGRITY_PROTOCOLS = {INTEGRITY_PROTOCOL, FRESH_PROTOCOL}
RESPONSE_PROTOCOLS = {RESPONSE_PROTOCOL, *INTEGRITY_PROTOCOLS}
MULTI_PROTOCOLS = {MULTI_PROTOCOL, *RESPONSE_PROTOCOLS}
SHARED_PROTOCOL = "shared-process-pilot-1"
INTEGRATION_PROTOCOL = "shared-process-integration-1"
SHARED_PROTOCOLS = {SHARED_PROTOCOL, INTEGRATION_PROTOCOL}
PARAMETER_PROTOCOLS = {PARAMETER_PROTOCOL, REVISION_PROTOCOL}
CONTINUATION_PROTOCOLS = {
    CONTINUATION_PROTOCOL,
    *PARAMETER_PROTOCOLS,
    *(MULTI_PROTOCOLS - {FRESH_PROTOCOL}),
}
SCIENTIFIC_PROTOCOLS = {*CONTINUATION_PROTOCOLS, *SHARED_PROTOCOLS, FRESH_PROTOCOL}
CONTENT_PROTOCOLS = {CONTENT_PROTOCOL, *SCIENTIFIC_PROTOCOLS}
ARMS = ("full", "brief_only", "refit_only", "no_latent", "no_spec")
FILES = ("manifest.json", "proposer_prompt.txt", "train.csv", "validation.csv")
REPO = Path(__file__).resolve().parents[3]


class DeadlineConfig(StrictSchema):
    """Bounded matrix; ablations change one declared component each."""

    protocol: Literal[
        "review-deadline-1",
        "review-deadline-2",
        "review-deadline-3",
        "review-deadline-4",
        "review-deadline-5",
        "review-deadline-6",
        "review-deadline-7",
        "review-deadline-8",
        "shared-process-pilot-1",
        "shared-process-integration-1",
        "shared-multi-pruning-1",
    ] = PROTOCOL
    platform: Literal["aces-h100x1"] = "aces-h100x1"
    serving_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_settings: StagedModelSettings
    served_context_tokens: Literal[32768] = 32768
    public_cells: tuple[str, ...] = Field(min_length=1, max_length=8)
    seeds: tuple[int, ...] = (0, 1)
    rounds: int = Field(default=3, ge=1, le=15)
    no_latent_cells: tuple[str, ...] = ()
    no_spec_cells: tuple[str, ...] = ()
    full_only: bool = False
    limits: ModelingLimits = ModelingLimits()
    evidence: EvidenceSettings = EvidenceSettings()
    fit_profile: Literal[
        "collocation-single-target-v2", "collocation-multi-target-v1"
    ] = "collocation-single-target-v2"
    scientific_judge: Literal["off"] = "off"
    revision_attempts: Literal[3] = 3
    wall_seconds: int = Field(default=25200, ge=600, le=43200)
    shutdown_margin_seconds: Literal[300] = 300
    training_replay_seconds: Literal[300] = 300
    fit_worker_seconds: Literal[1800] = 1800

    @model_validator(mode="after")
    def bounded_matrix(self):
        if self.protocol != FRESH_PROTOCOL and self.rounds > 13:
            raise ValueError("historical protocols allow at most thirteen rounds")
        if self.protocol == FRESH_PROTOCOL and (
            not self.full_only or self.fit_profile != "collocation-multi-target-v1"
        ):
            raise ValueError(
                "fresh shared campaign requires full-only joint-output fitting"
            )
        if self.protocol not in MULTI_PROTOCOLS and self.rounds > 6:
            raise ValueError("historical protocols allow at most six phase visits")
        if self.full_only and (
            self.protocol not in {CONTENT_PROTOCOL, *MULTI_PROTOCOLS}
            or self.no_latent_cells
            or self.no_spec_cells
        ):
            raise ValueError(
                "full-only pilots require review-deadline-2 without ablation cells"
            )
        if self.fit_profile == "collocation-multi-target-v1" and self.protocol not in {
            CONTENT_PROTOCOL,
            *MULTI_PROTOCOLS,
        }:
            raise ValueError("multi-target pilots require review-deadline-2")
        if self.protocol in SHARED_PROTOCOLS and (
            self.no_latent_cells or self.no_spec_cells or self.rounds != 2
        ):
            raise ValueError(
                "shared-process pilot requires two rounds and only paired prompt arms"
            )
        if len(set(self.public_cells)) != len(self.public_cells):
            raise ValueError("duplicate public cells")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(s < 0 for s in self.seeds)
        ):
            raise ValueError("distinct nonnegative seeds required")
        if not {*self.no_latent_cells, *self.no_spec_cells} <= set(self.public_cells):
            raise ValueError("no-latent cells must belong to the planned matrix")
        if self.model_settings.timeout_seconds >= self.shutdown_margin_seconds:
            raise ValueError("provider timeout exceeds shutdown margin")
        for cell in self.public_cells:
            spec = BenchmarkRegistry().get(cell)
            context = public_validation_context(cell)
            if spec.one_step_target_history:
                raise ValueError("deadline campaign requires open rollouts")
            if (
                self.fit_profile == "collocation-single-target-v2"
                and len(context.targets) != 1
            ):
                raise ValueError(
                    "deadline campaign requires single-target open rollouts"
                )
        return self


def tasks(config: DeadlineConfig) -> list[dict]:
    """Counterbalance fresh-construction arm order; refit shares full round zero."""
    if config.protocol == INTEGRATION_PROTOCOL:
        from autoformalism.rebuttal.shared_process_integration import (
            tasks as integration_tasks,
        )

        return integration_tasks(config)
    if config.protocol == SHARED_PROTOCOL:
        from autoformalism.rebuttal.shared_process_pilot import tasks as paired_tasks

        return paired_tasks(config)
    result = []
    for ci, cell in enumerate(config.public_cells):
        for seed in config.seeds:
            arms = ["full", "brief_only"]
            if (ci + seed) % 2:
                arms.reverse()
            if cell in config.no_latent_cells:
                arms.append("no_latent")
            if cell in config.no_spec_cells:
                arms.append("no_spec")
            arms.append("refit_only")
            if config.full_only:
                arms = ["full"]
            for arm in arms:
                result.append(
                    {
                        "index": len(result),
                        "cell": cell,
                        "seed": seed,
                        "arm": arm,
                        "task_id": f"cell{ci:02d}_seed{seed}_{arm}",
                        "shared_round_zero": f"cell{ci:02d}_seed{seed}_full"
                        if arm == "refit_only"
                        else None,
                    }
                )
    return result


def launcher_hash(protocol: str = PROTOCOL) -> str:
    paths = (
        "scripts/review_deadline.py",
        "scripts/hpc/run_review_deadline_aces.sh",
        "scripts/hpc/submit_review_deadline_aces.sh",
        "scripts/hpc/run_staged_topology_server.sh",
    )
    if protocol == CONTENT_PROTOCOL:
        paths += ("scripts/hpc/submit_review_deadline_v2_aces.sh",)
    if protocol in CONTINUATION_PROTOCOLS:
        paths += (
            "scripts/submit_review_continuation.py",
            "scripts/hpc/run_review_continuation_aces.sh",
            "scripts/hpc/submit_review_continuation_aces.sh",
            "scripts/audit_review_continuation.py",
            "scripts/smoke_review_continuation.py",
        )
    if protocol in PARAMETER_PROTOCOLS:
        paths += (
            "scripts/audit_review_parameters.py",
            "scripts/smoke_review_parameters.py",
        )
    if protocol == REVISION_PROTOCOL:
        paths += ("scripts/smoke_review_revision.py",)
    if protocol in MULTI_PROTOCOLS:
        paths += (
            "scripts/smoke_review_multi.py",
            "scripts/hpc/submit_review_multi_aces.sh",
        )
    if protocol in RESPONSE_PROTOCOLS:
        paths += (
            "scripts/review_response.py",
            "scripts/smoke_review_response.py",
            "scripts/hpc/submit_review_response_aces.sh",
        )
    if protocol == INTEGRITY_PROTOCOL:
        paths += ("scripts/hpc/submit_review_integrity_aces.sh",)
    if protocol in SHARED_PROTOCOLS:
        paths += (
            "scripts/submit_shared_process_pilot.py",
            "scripts/submit_review_continuation.py",
            "scripts/hpc/run_shared_process_pilot_aces.sh",
            "scripts/hpc/submit_shared_process_pilot_aces.sh",
            "scripts/smoke_shared_process_pilot.py",
            "scripts/recover_shared_process_fixed_models.py",
            "scripts/hpc/run_shared_process_fixed_recovery_aces.sh",
        )
    if protocol == INTEGRATION_PROTOCOL:
        paths += (
            "scripts/hpc/submit_shared_process_integration_aces.sh",
            "scripts/smoke_shared_process_integration.py",
        )
    if protocol == FRESH_PROTOCOL:
        paths += (
            "scripts/submit_fresh_shared.py",
            "scripts/recover_review_continuation_submission.py",
            "scripts/fresh_shared.py",
            "scripts/smoke_fresh_shared.py",
            "scripts/hpc/run_fresh_shared_aces.sh",
            "scripts/hpc/submit_fresh_shared_aces.sh",
            "scripts/submit_shared_process_pilot.py",
            "scripts/submit_review_continuation.py",
        )
    return content_hash(
        {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in paths}
    )


def freeze(config_path: Path, public_root: Path, root: Path) -> dict:
    """Copy only development assets; every task exists before provider work."""
    config = DeadlineConfig.model_validate_json(config_path.read_text())
    if config.protocol in CONTINUATION_PROTOCOLS:
        raise ValueError("continuations require an explicit source checkpoint import")
    if root.resolve().is_relative_to(
        public_root.resolve()
    ) or public_root.resolve().is_relative_to(root.resolve()):
        raise ValueError("campaign output and public release must be disjoint")
    with public._lock(root):
        if (root / "plan.json").exists():
            plan = verify(root)
            if plan["config"] != config.model_dump(mode="json"):
                raise ValueError("configuration differs from frozen campaign")
            return plan
        cells = {}
        for cell in config.public_cells:
            source = public_root / "phase_b_v1" / cell
            hashes = {}
            for name in FILES:
                payload = (source / name).read_bytes()
                destination = root / "public/phase_b_v1" / cell / name
                if destination.exists() and destination.read_bytes() != payload:
                    raise ValueError("partially frozen public file differs")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                hashes[name] = hashlib.sha256(payload).hexdigest()
            context = public_validation_context(cell)
            prompt = (source / "proposer_prompt.txt").read_text()
            target = PublicTargetContract.model_validate_json(
                (
                    REPO / "configs/target_eval/phase_b_v2/specs" / f"{cell}.json"
                ).read_text()
            )
            mechanism = MechanismEvaluationSpec.model_validate_json(
                (
                    REPO / "configs/mechanism_eval/phase_b_v1/specs" / f"{cell}.json"
                ).read_text()
            )
            brief = build_scientific_brief(
                prompt, context, target, mechanism, limits=config.limits
            )
            development = load_development(root / "public", cell)
            evidence = build_training_evidence(
                development.train, context, config.evidence
            )
            cells[cell] = {
                "assets": hashes,
                "context": context.model_dump(mode="json"),
                "brief": brief.model_dump(mode="json"),
                "evidence": evidence.model_dump(mode="json"),
                "target_contract": target.model_dump(mode="json"),
                "mechanism_spec": mechanism.model_dump(mode="json"),
                "training": public.pack_split(development.train).model_dump(
                    mode="json"
                ),
                "validation": public.pack_split(development.validation).model_dump(
                    mode="json"
                ),
            }
        extra = {}
        if config.protocol == FRESH_PROTOCOL:
            from autoformalism.rebuttal.fresh_shared import policy

            extra["fresh_shared_policy"] = policy()
        return sealed_write(
            root / "plan.json",
            {
                "protocol": config.protocol,
                "config": config.model_dump(mode="json"),
                "cells": cells,
                "tasks": tasks(config),
                "source_sha256": public._source_identity(),
                "runtime": public._runtime(),
                "launcher_sha256": launcher_hash(config.protocol),
                **(
                    {
                        "revision_call_policy": {
                            "physical_attempts": 3,
                            "cumulative_token_limit": None,
                            "schema": (
                                "general-shared-revision-1"
                                if config.protocol == INTEGRATION_PROTOCOL
                                else "response-oriented-revision-1"
                                if config.protocol == FRESH_PROTOCOL
                                else "scientific-content-revision-6"
                                if config.protocol == SHARED_PROTOCOL
                                else "model-content-inferred-routing-1"
                            ),
                            "round_zero_reused": False,
                        }
                    }
                    if config.protocol
                    in {CONTENT_PROTOCOL, *SHARED_PROTOCOLS, FRESH_PROTOCOL}
                    else {}
                ),
                "test_data_opened": False,
                "private_reference_opened": False,
                **extra,
                "selection": (
                    "certified-failures-excluded_then-validation-tolerance-then-terms-1"
                    if config.protocol == FRESH_PROTOCOL
                    else "certified-failures-excluded_then-validation-nmse"
                    "_then-additive-terms"
                ),
                "uncertified_semantics": (
                    "reported separately; never counted as a certified pass"
                ),
                **(
                    {
                        "shared_process_experiment": {
                            "guidance_policy": "shared-process-guidance-1",
                            "revision_policy": "scientific-content-revision-6",
                            "matched_fitter": True,
                            "automatic_test_access": False,
                            "automatic_parameter_tying": False,
                            "controls_also_use_whole_model_revision": True,
                        }
                    }
                    if config.protocol == SHARED_PROTOCOL
                    else {}
                ),
                **(
                    {
                        "integration_contract": {
                            "construction": "general-shared-construction-1",
                            "gain_assembly": "declared_or_independent",
                            "revision": "general-shared-revision-1",
                            "benchmark_specific_checks": False,
                            "comparison_claim": "integration_only",
                        }
                    }
                    if config.protocol in {INTEGRATION_PROTOCOL, FRESH_PROTOCOL}
                    else {}
                ),
                "round_zero_control": "no-iteration endpoint; not equal total compute",
                "refit_control": (
                    "no separate refit arm; no-change/failed revision reuses the "
                    "same visit fit allowance"
                    if config.protocol in {INTEGRATION_PROTOCOL, FRESH_PROTOCOL}
                    else "same C+S budget and retained parameter seed; "
                    "no revision calls"
                ),
            },
        )


def verify(root: Path, *, execution: bool = True) -> dict:
    """Check immutable inputs, including code for executing rather than reporting."""
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] not in {PROTOCOL, *CONTENT_PROTOCOLS}:
        raise ValueError("unsupported deadline protocol")
    config = DeadlineConfig.model_validate(plan["config"])
    if config.protocol != plan["protocol"]:
        raise ValueError("plan and config protocol differ")
    if plan["tasks"] != tasks(config):
        raise ValueError("task matrix differs")
    if config.protocol == FRESH_PROTOCOL:
        from autoformalism.rebuttal.fresh_shared import policy

        if plan.get("fresh_shared_policy") != policy():
            raise ValueError("fresh shared policy differs")
    if execution and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
        or plan["launcher_sha256"] != launcher_hash(plan["protocol"])
    ):
        raise ValueError("frozen execution runtime differs")
    for cell, value in plan["cells"].items():
        for name, digest in value["assets"].items():
            if (
                name not in FILES
                or hashlib.sha256(
                    (root / "public/phase_b_v1" / cell / name).read_bytes()
                ).hexdigest()
                != digest
            ):
                raise ValueError("frozen public content differs")
    if plan["protocol"] in CONTINUATION_PROTOCOLS:
        from autoformalism.rebuttal.review_continuation import verify_imports

        verify_imports(root, plan)
    if "parameter_free_recovery" in plan:
        from autoformalism.rebuttal.shared_process_recovery import verify_imports

        verify_imports(root, plan)
    return plan


def round_path(root: Path, task: dict, round_index: int) -> Path:
    return root / "results" / task["task_id"] / f"round_{round_index:02d}"


def read_round(root: Path, task: dict, index: int) -> dict | None:
    path = round_path(root, task, index) / "result.json"
    return sealed_read(path) if path.exists() else None


def require_open(root: Path) -> None:
    """No candidate edits after any post-freeze evaluation is authorized."""
    if (root / "evaluation_freeze.json").exists():
        raise ValueError("campaign is sealed for evaluation; further work is forbidden")


@contextmanager
def execution_lease(root: Path, *, exclusive: bool = False):
    """Prevent freezing while a provider/CPU worker can still publish results."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "execution.lock").open("a") as stream:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(stream, mode | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "active campaign work conflicts with evaluation freeze"
            ) from error
        try:
            if not exclusive:
                require_open(root)
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
