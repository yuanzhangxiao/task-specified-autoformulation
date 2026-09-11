"""Frozen, CPU-only comparison of mandatory and optional collocation warm-up."""

from __future__ import annotations

import shutil
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import model_validator

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    _role_start,
    fit_collocation_forward_sensitivity,
)
from autoformalism.rebuttal.fitter_diagnostic import (
    _finite_payload,
    read_json,
    sha256,
    write_json,
)
from autoformalism.rebuttal.fitter_methods import identity_runtime
from autoformalism.rebuttal.fitter_stagnation import checkpoint
from autoformalism.rebuttal.staged_prefit_fitting_campaign import load_public_data
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash

POLICIES = ("rollout_required", "rollout_or_observed")


class NodeCampaignPlan(StrictSchema):
    """Only node-guess policy varies; all numerical budgets remain paired."""

    protocol: Literal["collocation-node-comparison-1"] = "collocation-node-comparison-1"
    fit: CollocationSensitivityConfig = CollocationSensitivityConfig()
    include_synthetic_controls: bool = True

    @model_validator(mode="after")
    def bounded_worker(self):
        if self.worker_seconds > 1740:
            raise ValueError("worker budget must fit the 30-minute scheduler limit")
        return self

    @property
    def worker_seconds(self) -> float:
        # Refinement plus two production replays, each with the existing fit limit.
        return self.fit.initializer_seconds + 3 * self.fit.refinement_seconds + 180


def safe_path(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError("artifact path escapes experiment")
    return path


def verify_bundle(root: Path) -> dict:
    """Verify the portable manifest and every referenced source byte."""
    bundle = read_json(root / "bundle.json")
    if (
        bundle.get("protocol") != "collocation-node-cases-1"
        or bundle.get("identity")
        != content_hash({k: v for k, v in bundle.items() if k != "identity"})
        or bundle.get("test_data_opened") is not False
        or bundle.get("private_reference_opened") is not False
        or bundle.get("llm_calls") != 0
    ):
        raise ValueError("source bundle identity or data separation differs")
    for relative, expected in bundle["files"].items():
        parts = Path(relative).parts
        allowed = (
            len(parts) == 2
            and parts[0] in {"candidates", "provenance"}
            and parts[-1].endswith(".json")
        ) or (
            len(parts) == 4
            and parts[:2] == ("public", "phase_b_v1")
            and parts[-1]
            in {"manifest.json", "proposer_prompt.txt", "train.csv", "validation.csv"}
        )
        if not allowed:
            raise ValueError("bundle contains an unsupported or nonpublic artifact")
        if sha256(safe_path(root, relative)) != expected:
            raise ValueError(f"source bundle asset differs: {relative}")
    for case in bundle["cases"]:
        if case["candidate"] not in bundle["files"]:
            raise ValueError("candidate is not in source ledger")
        for name in (
            "manifest.json",
            "proposer_prompt.txt",
            "train.csv",
            "validation.csv",
        ):
            if (
                f"public/phase_b_v1/{case['benchmark_id']}/{name}"
                not in bundle["files"]
            ):
                raise ValueError(
                    "public development file is missing from source ledger"
                )
    return bundle


def launcher_identity() -> str:
    root = Path(__file__).resolve().parents[3]
    names = (
        "scripts/run_collocation_node_campaign.py",
        "scripts/export_collocation_node_cases.py",
        "scripts/hpc/collocation_node_delta.slurm",
        "scripts/hpc/submit_collocation_node_delta.sh",
    )
    return content_hash({n: sha256(root / n) for n in names})


def synthetic_problem(name: str):
    """Attainable controls with stable or explosive starting parameters."""
    observed = name != "control_latent"
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": name,
            "parent_candidate_id": None,
            "states": [{"name": "x", "kind": "latent"}],
            "state_equations": [{"state": "x", "rhs": "a*x**2"}],
            "observation_mappings": [
                {"channel": "v01", "expression": "x" if observed else "2*x"}
            ],
            "parameters": [{"name": "a", "scope": "global", "role": "rate"}],
            "initial_conditions": [
                {"state": "x", "scope": "global", "fixed_value": 1.0}
            ],
        }
    )
    model = compile_candidate(candidate, ValidationContext(targets=("v01",)))
    splits = []
    for split, end in ((SplitName.TRAIN, 2.0), (SplitName.VALIDATION, 2.3)):
        t = np.linspace(0, end, 41)
        y = (1 if observed else 2) / (1 - 0.1 * t)
        row = Trajectory(split.value, t, {"v01": y}, {}, {}, {}, {})
        splits.append(DatasetSplit(split, (row,), content_hash([name, split.value])))
    return model, *splits


def case_problem(output: Path, case: dict):
    if case["kind"] == "synthetic":
        return synthetic_problem(case["name"])
    data, context = load_public_data(
        output / "source/public", case["benchmark_id"], case["tier"]
    )
    candidate = CandidateModel.model_validate(
        read_json(safe_path(output / "source", case["candidate"]))
    )
    return compile_candidate(candidate, context), data.train, data.validation


def prepare_nodes(plan: NodeCampaignPlan, source: Path, output: Path) -> dict:
    """Freeze source bytes, cases, starts, policies, and runtime before fitting."""
    if source.resolve().is_relative_to(
        output.resolve()
    ) or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("source and experiment must be separate")
    if (output / "freeze.json").exists():
        frozen = verify_nodes(output)
        if frozen["plan"] != plan.model_dump(mode="json"):
            raise ValueError("plan differs on resume")
        return frozen
    bundle = verify_bundle(source)
    for relative in ("bundle.json", *bundle["files"]):
        src, dst = safe_path(source, relative), safe_path(output / "source", relative)
        if dst.exists() and sha256(src) != sha256(dst):
            raise ValueError("partially prepared source differs")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(src, dst)
    verify_bundle(output / "source")
    cases = [{**c, "kind": "public"} for c in bundle["cases"]]
    if plan.include_synthetic_controls:
        cases += [
            {"name": name, "kind": "synthetic", "source_round": None}
            for name in ("control_observed", "control_latent", "control_stable")
        ]
    if len({c["name"] for c in cases}) != len(cases):
        raise ValueError("case names collide")
    for case in cases:
        model, training, _ = case_problem(output, case)
        case["start"] = (
            {"a": 0.15 if case["name"] == "control_stable" else 1.0}
            if case["kind"] == "synthetic"
            else _role_start(model.validated.candidate, training)
        )
    tasks = [
        {
            "index": len(POLICIES) * i + j,
            "case": i,
            "policy": policy,
            "name": f"case_{i:03d}_{policy}",
        }
        for i, _ in enumerate(cases)
        for j, policy in enumerate(POLICIES)
    ]
    frozen = {
        "plan": plan.model_dump(mode="json"),
        "source_identity": bundle["identity"],
        "cases": cases,
        "tasks": tasks,
        "runtime": identity_runtime(),
        "launcher": launcher_identity(),
        "test_data_opened": False,
        "llm_calls": 0,
    }
    frozen["identity"] = content_hash(frozen)
    write_json(output / "freeze.json", frozen, immutable=True)
    return frozen


def verify_nodes(output: Path) -> dict:
    frozen = read_json(output / "freeze.json")
    if (
        frozen.get("identity")
        != content_hash({k: v for k, v in frozen.items() if k != "identity"})
        or frozen["runtime"] != identity_runtime()
        or frozen["launcher"] != launcher_identity()
    ):
        raise ValueError("experiment code, launcher or freeze differs")
    NodeCampaignPlan.model_validate(frozen["plan"])
    if verify_bundle(output / "source")["identity"] != frozen["source_identity"]:
        raise ValueError("source identity differs")
    return frozen


def execute_nodes(output: Path, index: int) -> dict:
    """Run one arm without invoking any proposer or changing a saved candidate."""
    frozen = verify_nodes(output)
    if not 0 <= index < len(frozen["tasks"]):
        raise ValueError("unknown task index")
    task = frozen["tasks"][index]
    identity = content_hash([frozen["identity"], task])
    root = output / "results" / task["name"]
    existing = checkpoint(root / "result.json", identity)
    if existing is not None:
        return existing
    case = frozen["cases"][task["case"]]
    record = {
        "identity": identity,
        "task": task,
        "case": case["name"],
        "case_kind": case["kind"],
        "source_round": case["source_round"],
        "test_data_opened": False,
        "private_reference_opened": False,
        "llm_calls": 0,
    }
    saved = checkpoint(root / "fit.json", identity)
    if saved is not None:
        record.update(
            status=saved["fit"]["status"],
            fit=saved["fit"],
            seconds=saved.get("seconds"),
        )
    elif checkpoint(root / "fit_started.json", identity) is not None:
        record.update(status="interrupted_fit", fit=None)
    else:
        write_json(root / "fit_started.json", record, immutable=True)
        started = monotonic()
        try:
            model, training, validation = case_problem(output, case)
            settings = NodeCampaignPlan.model_validate(frozen["plan"]).fit.model_copy(
                update={"collocation_node_start": task["policy"]}
            )
            fit = fit_collocation_forward_sensitivity(
                model,
                training,
                validation,
                settings,
                root / "numerical",
                initial_parameters=case["start"],
            )
            write_json(
                root / "fit.json",
                {
                    "identity": identity,
                    "fit": _finite_payload(fit),
                    "seconds": monotonic() - started,
                },
            )
            record.update(status=fit["status"], fit=fit)
        except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
            record.update(status="failed", fit=None, error=str(error)[-4000:])
        record["seconds"] = monotonic() - started
    write_json(root / "result.json", _finite_payload(record))
    return record


def summarize_nodes(output: Path) -> dict:
    """Show original parents, accepted revisions, and synthetic controls separately."""
    frozen = verify_nodes(output)
    rows = []
    for task in frozen["tasks"]:
        case = frozen["cases"][task["case"]]
        record = checkpoint(
            output / "results" / task["name"] / "result.json",
            content_hash([frozen["identity"], task]),
        ) or {"status": "missing"}
        fit = record.get("fit") or {}
        init, refinement = fit.get("initializer") or {}, fit.get("refinement") or {}
        rows.append(
            {
                "case": case["name"],
                "group": "synthetic"
                if case["kind"] == "synthetic"
                else "parent"
                if case["source_round"] == 0
                else "accepted_round",
                "policy": task["policy"],
                "status": record["status"],
                "optimizer_started": init.get("collocation_optimizer_started"),
                "fallback_trajectories": sum(
                    r.get("source") == "observed_and_fixed_initials"
                    for r in init.get("node_initialization", [])
                ),
                "initializer_success": init.get("success"),
                "native_success": refinement.get("optimizer_native_success"),
                "accepted_success": refinement.get("optimizer_success"),
                "valid_evaluations": refinement.get("valid_residual_evaluations"),
                "train_nmse": (fit.get("training") or {}).get("normalized_mse"),
                "validation_nmse": (fit.get("validation") or {}).get("normalized_mse"),
                "seconds": record.get("seconds"),
                "error": record.get("error"),
            }
        )
    summary = {
        "identity": frozen["identity"],
        "rows": rows,
        "llm_calls": 0,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    write_json(output / "summary.json", _finite_payload(summary))
    keys = (
        "case",
        "policy",
        "status",
        "optimizer_started",
        "fallback_trajectories",
        "initializer_success",
        "valid_evaluations",
        "train_nmse",
        "validation_nmse",
        "seconds",
    )
    lines = [
        "# Collocation node initialization comparison",
        "",
        "Same candidates, starts and budgets; only node initialization differs.",
        "Completion means finite production replay, not scientific correctness "
        "or good fit.",
        "",
    ]
    for group in ("parent", "accepted_round", "synthetic"):
        lines.extend(
            [
                f"## {group}",
                "",
                "| " + " | ".join(keys) + " |",
                "| " + " | ".join("---" for _ in keys) + " |",
            ]
        )
        lines.extend(
            "| " + " | ".join(str(r[k]) for k in keys) + " |"
            for r in rows
            if r["group"] == group
        )
        lines.append("")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return summary
