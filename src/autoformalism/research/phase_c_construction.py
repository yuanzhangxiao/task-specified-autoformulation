"""Phase C adapter using the established bounded public-construction repair."""

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.research import construction_contract as contract
from autoformalism.search import review_multi_construction, shared_construction


def propose(cell, task, client, directory):
    """Reuse Phase B admission/repair; preserve construction and repair traces."""
    construction_attempts = []

    def fresh(visible, task, client, directory):
        visible = {
            **visible,
            "brief": contract.visible_brief(visible, task).model_dump(mode="json"),
        }
        value = shared_construction.construct(
            visible,
            task,
            client,
            directory,
            build_bundle=pipeline._bundle,
            certificate_for=pipeline.certificates,
            retain_failed_draft=True,
            explicit_mechanism_bindings=True,
            target_definitions=contract.target_definitions(visible),
            complete_process_context=True,
        )
        construction_attempts.extend(value.get("attempts", []))
        # Keep pre-repair route evidence: the established wrapper owns `attempts`.
        atomic_json(directory / "unrepaired_construction.json", value)
        return value

    value = review_multi_construction.propose(
        {"protocol": "phase-c-construction-baseline-1", "cells": {"current": cell}},
        {**task, "cell": "current"},
        {},
        client,
        directory,
        fresh_builder=fresh,
        strict_parameters=True,
        shared_relationships=task.get("shared_processes", True),
    )
    # No executable draft means no whole-model repair was attempted.
    repairs = [a for a in value.get("attempts", []) if "request_hash" in a]
    value.update(
        attempts=construction_attempts,
        construction_attempts=construction_attempts,
        public_contract_repair_attempts=repairs,
        adapter_policy=contract.POLICY,
    )
    atomic_json(directory / "public_contract_repair.json", value)
    return value
