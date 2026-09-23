#!/usr/bin/env python3
"""Synthetic multi-output import, public repair, real fitting, fallback and resume."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import review_continuation as continuation
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicFitResult
from autoformalism.schemas.staged_topology import ModelingLimits
from autoformalism.search.residual_evidence import build_residual_evidence
from autoformalism.search.training_evidence import build_training_evidence
from autoformalism.staged_topology import content_hash

SPEC = importlib.util.spec_from_file_location(
    "joint_control", Path(__file__).with_name("smoke_multi_target_fitting.py")
)
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)
CELL = "phase_b_dalla_man_t2_canonical_named_easy"


def example(*, draft=False):
    """Three targets with an optional deliberately misrepresented rate output."""
    request, train, val, truth = CONTROL.control()
    if draft:
        raw = request.model_dump(mode="json")
        candidate = raw["base_candidate"]
        candidate["processes"] = []
        candidate["states"].append({"name": "disposal", "kind": "observed"})
        candidate["state_equations"].append({"state": "disposal", "rhs": "c*q*g+Uii"})
        candidate["initial_conditions"].append(
            {"state": "disposal", "scope": "global", "fixed_value": 0}
        )
        request = PublicFitRequest.model_validate(raw)
    model, guesses, audit = public._lower(request)
    brief = {
        "scientific_context": "Synthetic disposal with delayed insulin input response.",
        "public_variables": [
            {"name": n, "data_role": role}
            for role, names in (
                ("target", request.context.targets),
                ("auxiliary", request.context.auxiliaries),
                ("external_input", request.context.external_inputs),
            )
            for n in names
        ],
        "requirements": [],
        "limits": ModelingLimits().model_dump(mode="json"),
    }
    bundle = {
        "source_task": {"arm": "full", "task_id": "synthetic"},
        "brief": brief,
        "context": request.context.model_dump(mode="json"),
        "candidate": model.validated.candidate.model_dump(mode="json"),
        "initialization": {
            "base_candidate": request.base_candidate.model_dump(mode="json"),
            "base_context": request.context.model_dump(mode="json"),
            "plan": request.initialization_plan.model_dump(mode="json"),
            "candidate": model.validated.candidate.model_dump(mode="json"),
            "context": model.validated.context.model_dump(mode="json"),
            "guesses": guesses,
            "audit": audit,
        },
    }
    packet = build_residual_evidence(
        public.unpack_split(train),
        request.context,
        {
            row.trajectory_id: {"time": row.time, "predictions": row.targets}
            for row in train.rows
        },
        candidate_sha256=content_hash(bundle["candidate"]),
        parameters=truth,
        numerical_status={"synthetic_fixture": True},
    ).model_dump(mode="json")
    return bundle, packet, truth, request, train, val


def fixture(root: Path, *, real_fit=False):
    """Seal one synthetic fitted branch and one failed construction."""
    bundle, packet, params, request, train, val = example()
    draft, *_ = example(draft=True)
    config = io.DeadlineConfig(
        protocol=io.CONTENT_PROTOCOL,
        serving_image_sha256="0" * 64,
        model_settings=io.StagedModelSettings(),
        public_cells=(CELL,),
        seeds=(0, 1),
        full_only=True,
        rounds=3,
        fit_profile="collocation-multi-target-v1",
    )
    assets = {}
    for name in io.FILES:
        path = root / "public/phase_b_v1" / CELL / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic development fixture: " + name)
        assets[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    cell = {
        "brief": bundle["brief"],
        "context": bundle["context"],
        "assets": assets,
        "training": train.model_dump(mode="json"),
        "validation": val.model_dump(mode="json"),
        "evidence": build_training_evidence(
            public.unpack_split(train), request.context
        ).model_dump(mode="json"),
        "target_contract": {
            "schema_version": "public-target-contract-2",
            "benchmark_id": CELL,
            "tier": "easy",
            "public_prompt_sha256": assets["proposer_prompt.txt"],
            "targets": [
                {
                    "target_channel": n,
                    "public_requirement": f"Generate {n}",
                    "expected_representation": "instantaneous_process"
                    if n == "U"
                    else "dynamic_state",
                    "representation_requirement": f"Synthetic {n} role",
                    "required_dependencies": [
                        {
                            "dependency_id": "basal",
                            "acceptable_symbols": ["Uii"],
                            "public_requirement": "Total U includes Uii",
                        }
                    ]
                    if n == "U"
                    else [],
                }
                for n in request.context.targets
            ],
        },
        "mechanism_spec": {
            "benchmark_id": CELL,
            "tier": "easy",
            "required_mechanisms": [
                {
                    "id": "memory",
                    "required_drivers": ["insulin"],
                    "required_targets": ["U"],
                    "requires_dynamic_memory": True,
                }
            ],
        },
    }
    plan = sealed_write(
        root / "plan.json",
        {
            "protocol": config.protocol,
            "config": config.model_dump(mode="json"),
            "tasks": io.tasks(config),
            "cells": {CELL: cell},
            "source_sha256": public._source_identity(),
            "runtime": public._runtime(),
            "launcher_sha256": io.launcher_hash(config.protocol),
            "test_data_opened": False,
        },
    )
    if real_fit:
        public.prepare_fit(request, train, val, root / "synthetic-source-fit")
        fit = public.execute_fit(root / "synthetic-source-fit")
        packet = pipeline.replay_packet(
            root, root / "source-replay", request, fit, train
        )
    else:
        fit = PublicFitResult(
            status="complete",
            profile=request.profile,
            identity="0" * 64,
            request_sha256="1" * 64,
            lowered_candidate_sha256="2" * 64,
            initialization_plan_sha256="3" * 64,
            training_content_sha256="4" * 64,
            validation_content_sha256="5" * 64,
            source=request.source,
            parameters=params,
            training={"available": True, "normalized_mse": 0.1},
            validation={"available": True, "normalized_mse": 0.1},
            message="synthetic unit fixture",
        )
    selected = {
        "bundle": bundle,
        "request": request.model_dump(mode="json"),
        "fit": fit.model_dump(mode="json"),
        "packet": packet,
        "certificate": pipeline.certificates(bundle, cell, plan["tasks"][0]),
        "origin_round": 0,
    }
    for task in plan["tasks"]:
        sealed_write(
            io.round_path(root, task, 2) / "result.json",
            {
                "task": task,
                "round": 2,
                "status": "closed_lineage",
                "selected": selected if task["seed"] == 0 else None,
                "closed": True,
                "test_data_opened": False,
            },
        )
        if task["seed"] == 1:
            sealed_write(
                io.round_path(root, task, 0) / "proposal.json",
                {
                    "task": task,
                    "round": 0,
                    "status": "requirement_failed",
                    "bundle": draft,
                    "test_data_opened": False,
                },
            )
    return plan


def run(root: Path) -> dict:
    source = root / "source"
    fixture(source, real_fit=True)
    before = {str(p): p.read_bytes() for p in source.rglob("*.json")}
    output = root / "continuation"
    plan = continuation.prepare(source, output, 2, 2, protocol=io.MULTI_PROTOCOL)
    calls = []
    for index in (1, 2):
        for task in plan["tasks"]:

            def transport(url, body, timeout, visit=index, seed=task["seed"]):
                payload = json.loads(body["messages"][1]["content"])
                calls.append(payload)
                assert "validation" not in payload and "test" not in payload
                assert {
                    t["target_channel"]
                    for t in payload["public_target_contract"]["requirements"]
                } == {"Gp", "I", "U"}
                raw = {
                    "hypothesis": "Prescribed synthetic contract exercise",
                    "evidence_refs": ["unavailable_samples"],
                }
                if visit == 2:
                    raw["equations"] = [
                        {"component": "g", "expression": "missing_symbol*g"}
                    ]
                elif seed == 1:
                    raw["equations"] = [
                        {
                            "component": "disposal",
                            "kind": "algebraic",
                            "expression": "c*q*g+Uii",
                        }
                    ]
                    assert payload["retained_fitted_parameters"] is None
                else:
                    raw.update(
                        output_mappings=[
                            {"channel": "U", "expression": "disposal+offset"}
                        ],
                        new_parameters=[{"name": "offset"}],
                    )
                return {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(raw)},
                        }
                    ],
                    "usage": {"total_tokens": 100},
                }

            client = pipeline._client(
                output, plan, task, index, "http://offline", lambda: True, transport
            )
            proposal = pipeline.propose_one(output, plan, task, index, client)
            result = pipeline.fit_one(output, plan, task, index)
            assert result["selected"] is not None and not result["closed"], proposal
            assert set(
                result["trial"]["fit"]["training"]["per_target_normalized_mse"]
            ) == {"Gp", "I", "U"}
            if index == 2:
                assert (
                    proposal["status"] == "revision_failed"
                    and result["fit_trigger"] == "incumbent_fallback"
                )
            else:
                assert proposal["status"] == (
                    "constructed" if task["seed"] == 1 else "committed"
                ), proposal
                assert (
                    proposal["decision"]["provenance"]["citation_audit"][
                        "valid_evidence_ids"
                    ]
                    == []
                )
            assert pipeline.propose_one(output, plan, task, index, None) == proposal
            assert pipeline.fit_one(output, plan, task, index) == result
    reporting.report(output)
    assert {str(p): p.read_bytes() for p in source.rglob("*.json")} == before
    return {
        "status": "pass",
        "joint_output_revision": True,
        "construction_repair": True,
        "real_multi_target_fits": True,
        "incumbent_fallback": True,
        "exact_resume": True,
        "source_preserved": True,
        "cached_calls": len(calls),
        "live_llm_calls": 0,
        "test_data_opened": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
