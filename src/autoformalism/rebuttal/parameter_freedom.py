"""Isolated fixed-basis versus free-shape reference fits, with paired starts."""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.data import TrainingScaler
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting.collocation_mesh import plan_meshes
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    fit_collocation_forward_sensitivity,
)
from autoformalism.fitting.initialization import LatentInitializationPlan
from autoformalism.fitting.sensitivity_probe import SymbolicODE, symbolic_rollout
from autoformalism.rebuttal.attainability_campaign import (
    _checkpoint,
    _identity,
    checked_replay,
)
from autoformalism.rebuttal.attainability_controls import system_for
from autoformalism.rebuttal.attainability_reference import (
    reference_problem,
    reference_skeleton,
)
from autoformalism.rebuttal.fitter_diagnostic import (
    _finite_payload,
    _write_bytes,
    read_json,
    sha256,
    write_json,
)
from autoformalism.rebuttal.fitter_methods import identity_runtime
from autoformalism.rebuttal.piecewise_campaign import safe_path, unpack_split
from autoformalism.rebuttal.resolution_campaign import profile_point
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash


class FreedomPlan(StrictSchema):
    """One CPU, fixed mesh, equal phase budgets, no selection on validation."""

    protocol: Literal["fitter-parameter-freedom-1"] = "fitter-parameter-freedom-1"
    expected_parameters: int = Field(default=48, ge=2)
    expected_weights: int = Field(default=26, ge=1)
    gate_seconds: float = Field(default=480, gt=0, le=600)
    replay_seconds: float = Field(default=240, gt=0, le=300)
    supervisor_seconds: int = Field(default=6600, ge=60, le=6900)
    fit: CollocationSensitivityConfig

    @model_validator(mode="after")
    def bounded_comparison(self):
        fit = self.fit
        if not (
            fit.budget_profile == "extended_diagnostic"
            and fit.defer_production_replay
            and fit.collocation_assembly == "mapped"
            and fit.collocation_target_variables == 24000
            and fit.collocation_minimum_intervals == 120
            and fit.sensitivity_jacobian_format == "sparse"
            and fit.recovery_policy == "feasible"
            and fit.recovery_screen_seconds is not None
        ):
            raise ValueError("requires the explicit bounded sparse/resolution profile")
        # Two methods per split, two splits. Leave process/compilation grace.
        if self.supervisor_seconds < (
            fit.initializer_seconds
            + fit.refinement_seconds
            + 4 * self.replay_seconds
            + 300
        ):
            raise ValueError("supervisor must include fitting, replay and startup")
        if self.expected_weights >= self.expected_parameters:
            raise ValueError("fixed-basis comparison must remove shape parameters")
        return self


def code_identity() -> str:
    """Hash the runtime and this campaign's executable orchestration."""
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "src/autoformalism").rglob("*.py"))
    paths += [
        root / p
        for p in (
            "scripts/run_parameter_freedom.py",
            "scripts/parameter_freedom_report.py",
            "scripts/hpc/parameter_freedom_delta.slurm",
            "scripts/hpc/submit_parameter_freedom_delta.sh",
        )
    ]
    return content_hash({str(p.relative_to(root)): sha256(p) for p in paths})


def fixed_basis(problem: dict, truth: dict) -> tuple[dict, dict]:
    """Freeze parameters inside function calls, then certify remaining affinity.

    This is a restricted reference diagnostic, not a general parameter-role
    classifier. Inputs are compiled before and after literal AST substitution.
    No Python execution, numerical estimation, or hidden state labels are used.
    """
    system, _ = system_for(problem)
    candidate = problem["candidate"]
    if candidate["processes"]:
        raise ValueError("reference freezing expects expanded state equations")
    if set(truth) != set(system.names) or not all(np.isfinite(list(truth.values()))):
        raise ValueError("reference parameter values differ")
    shape = set()
    expressions = [e["rhs"] for e in candidate["state_equations"]]
    expressions += [e["expression"] for e in candidate["observation_mappings"]]
    for expression in expressions:
        for node in ast.walk(ast.parse(expression, mode="eval")):
            if isinstance(node, ast.Call):
                shape.update(
                    n.id
                    for arg in node.args
                    for n in ast.walk(arg)
                    if isinstance(n, ast.Name) and n.id in truth
                )
    if not shape or shape == set(system.names):
        raise ValueError("expected both nonlinear shapes and outer weights")
    values = {n: float(truth[n]) for n in sorted(shape)}

    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node):
            return (
                ast.copy_location(ast.Constant(values[node.id]), node)
                if node.id in values
                else node
            )

    def substitute(text):
        tree = Substitute().visit(ast.parse(text, mode="eval"))
        return ast.unparse(ast.fix_missing_locations(tree))

    reduced = deepcopy(problem)
    reduced["candidate"]["parameters"] = [
        p for p in candidate["parameters"] if p["name"] not in shape
    ]
    for key, field in (
        ("state_equations", "rhs"),
        ("observation_mappings", "expression"),
    ):
        for item in reduced["candidate"][key]:
            item[field] = substitute(item[field])
    reduced["start"] = {
        n: v for n, v in problem.get("start", {}).items() if n not in shape
    }
    reduced_system, _ = system_for(reduced)
    if not reduced_system.affine or reduced_system.initial_parameter_names:
        raise ValueError(
            "remaining equations/observations are not affine with fixed initials"
        )
    return reduced, values


def task_matrix(problem: dict, truth: dict) -> tuple[list[dict], list[dict], dict]:
    """Pair identical physical starts, plus separately labelled original controls."""
    reduced, fixed = fixed_basis(problem, truth)
    tasks, problems = [], []
    for start_name in ("ordinary", "near"):
        original = profile_point(problem, truth, start_name)
        matched = {**original, **fixed}
        for arm, template, start in (
            (
                "fixed_basis",
                reduced,
                {n: v for n, v in matched.items() if n not in fixed},
            ),
            ("free_shapes_matched", problem, matched),
            ("free_shapes_original", problem, original),
        ):
            p = deepcopy(template)
            p["start"] = start
            tasks.append(
                {
                    "index": len(tasks),
                    "arm": arm,
                    "start": start_name,
                    "free_parameters": len(start),
                    "matched_pair": start_name
                    if arm != "free_shapes_original"
                    else None,
                    "oracle_initials": True,
                    "oracle_shapes": arm != "free_shapes_original"
                    or start_name == "near",
                    "exact_shape_start": arm != "free_shapes_original",
                    "oracle_weight_start": start_name == "near",
                }
            )
            problems.append(p)
    return tasks, problems, fixed


def prepare(source: Path, output: Path, plan: FreedomPlan) -> dict:
    """Freeze new comparisons from the completed v3 generation and preflight."""
    if source.resolve().is_relative_to(
        output.resolve()
    ) or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("source/output must be separate")
    old = read_json(source / "freeze.json")
    _identity(old)
    if old["plan"]["protocol"] != "fitter-attainability-3" or len(old["cases"]) != 1:
        raise ValueError("requires the v3 reference campaign")
    if (
        old.get("test_data_opened") is not False
        or old.get("proposer_access") is not False
    ):
        raise ValueError("source isolation differs")
    for relative, digest in old["assets"].items():
        if sha256(safe_path(source, relative)) != digest:
            raise ValueError("source input changed: " + relative)
    gate = _checkpoint(source / "preflight.json", old["identity"])
    if not gate.get("pass"):
        raise ValueError("source preflight must pass")
    runtime = identity_runtime()
    if (
        any(runtime[k] != old["runtime"][k] for k in ("packages", "casadi"))
        or runtime["python"].split(".")[:2] != old["runtime"]["python"].split(".")[:2]
    ):
        raise ValueError("use the source numerical packages and Python major/minor")
    gen = _checkpoint(
        source / "generated/000/result.json",
        content_hash([old["identity"], "generate", 0]),
    )
    if gen.get("status") != "complete" or not gen.get("scaled_reference_audit", {}).get(
        "pass"
    ):
        raise ValueError("reference generation audit must pass")
    for relative, digest in gen["assets"].items():
        if sha256(safe_path(source / "generated/000", relative)) != digest:
            raise ValueError("generated source changed")
    bundle = read_json(source / "reference_input.json")
    _identity(bundle)
    candidate, context, truth = reference_skeleton(bundle["spec"])
    if any(
        bundle[k] != v
        for k, v in (("candidate", candidate), ("context", context), ("truth", truth))
    ):
        raise ValueError("reference translation changed")
    actual = read_json(source / "problems/000.json")
    if reference_problem(actual, bundle) != actual:
        raise ValueError("reference boundary/input protocol changed")
    problem = read_json(source / "generated/000/synthetic.json")
    if (
        problem["candidate"] != candidate
        or read_json(source / "generated/000/truth.json") != truth
    ):
        raise ValueError("generated model/parameters changed")
    tasks, problems, fixed = task_matrix(problem, truth)
    if (
        len(truth) != plan.expected_parameters
        or len(truth) - len(fixed) != plan.expected_weights
    ):
        raise ValueError("unexpected parameter groups")
    assets = {}
    for relative, destination in (
        ("freeze.json", "provenance/source_freeze.json"),
        ("preflight.json", "provenance/source_preflight.json"),
        ("generated/000/result.json", "provenance/source_generation.json"),
        ("reference_input.json", "provenance/reference_input.json"),
        ("generated/000/synthetic.json", "inputs/synthetic.json"),
        ("generated/000/clean.json", "inputs/clean.json"),
        ("generated/000/truth.json", "inputs/truth.json"),
    ):
        path = output / destination
        _write_bytes(path, safe_path(source, relative).read_bytes(), immutable=True)
        assets[destination] = sha256(path)
    for task, p in zip(tasks, problems, strict=True):
        relative = f"problems/{task['index']:03d}.json"
        write_json(output / relative, p, immutable=True)
        assets[relative] = sha256(output / relative)
    frozen = {
        "plan": plan.model_dump(mode="json"),
        "tasks": tasks,
        "assets": assets,
        "fixed_shapes": fixed,
        "source_identity": old["identity"],
        "code": code_identity(),
        "runtime": runtime,
        "test_data_opened": False,
        "proposer_access": False,
        "llm_calls": 0,
        "known_initial_conditions": True,
        "hidden_trajectory_labels_used": False,
        "data": "unchanged noiseless synthetic reference observations from v3",
    }
    frozen["identity"] = content_hash(frozen)
    write_json(output / "freeze.json", frozen, immutable=True)
    return verify(output)


def verify(output: Path) -> dict:
    """Refuse modified code, runtime, data, plan, or cached parameter starts."""
    frozen = read_json(output / "freeze.json")
    _identity(frozen)
    if frozen["code"] != code_identity() or frozen["runtime"] != identity_runtime():
        raise ValueError("code/runtime differs; use the pinned checkout")
    for relative, expected in frozen["assets"].items():
        if sha256(safe_path(output, relative)) != expected:
            raise ValueError("frozen asset differs: " + relative)
    return frozen


def paired_check(left: dict, right: dict, config, deadline: float) -> dict:
    """Check equal initial models, state meshes and subset trajectory derivatives."""
    systems = [system_for(p)[0] for p in (left, right)]
    systems = [SymbolicODE(s.model, solver_jacobian_format="sparse") for s in systems]
    train = unpack_split(left["splits"]["train"])
    theta = [
        np.array([p["start"][n] for n in s.names])
        for p, s in zip((left, right), systems, strict=True)
    ]
    meshes = [plan_meshes(s, train, 24000, minimum_intervals=120)[0] for s in systems]
    same_mesh = all(
        np.array_equal(a.time, b.time) for a, b in zip(*meshes, strict=True)
    )
    scale = TrainingScaler().fit(train).scales["target:v01"].standard_deviation
    row = train.trajectories[0]
    predictions = [
        symbolic_rollout(s, row, t, config.fit_config(), deadline, sensitivities=True)
        for s, t in zip(systems, theta, strict=True)
    ]
    subset = [systems[1].names.index(n) for n in systems[0].names]
    a, b = predictions
    residual_difference = float(np.max(abs(a[0] - b[0])) / scale)
    ja, jb = a[1], b[1][:, :, subset]
    jac_difference = float(np.linalg.norm(ja - jb) / max(1e-12, np.linalg.norm(jb)))
    return {
        "same_mesh": same_mesh,
        "scaled_prediction_difference": residual_difference,
        "relative_weight_jacobian_difference": jac_difference,
        "fixed_basis_affine_certified": systems[0].affine,
        "pass": same_mesh
        and systems[0].affine
        and residual_difference <= 1e-5
        and jac_difference <= 0.005,
    }


def gate(output: Path) -> dict:
    """Terminal preflight: a killed check cannot silently receive a fresh budget."""
    frozen = verify(output)
    identity = frozen["identity"]
    path = output / "gate/result.json"
    if path.exists():
        return _checkpoint(path, identity)
    started_path = output / "gate/started.json"
    if started_path.exists():
        _checkpoint(started_path, identity)
        result = {"identity": identity, "pass": False, "status": "interrupted"}
    else:
        write_json(started_path, {"identity": identity}, immutable=True)
        deadline = monotonic() + frozen["plan"]["gate_seconds"]
        try:
            checks = [
                paired_check(
                    read_json(output / f"problems/{i:03d}.json"),
                    read_json(output / f"problems/{i + 1:03d}.json"),
                    FreedomPlan.model_validate(frozen["plan"]).fit,
                    deadline,
                )
                for i in (0, 3)
            ]
            result = {
                "identity": identity,
                "pass": all(c["pass"] for c in checks),
                "checks": checks,
                "status": "complete",
            }
        except (ValueError, RuntimeError, TimeoutError) as error:
            result = {
                "identity": identity,
                "pass": False,
                "status": "failed",
                "error": str(error),
            }
    write_json(path, _finite_payload(result), immutable=True)
    return result


def execute(output: Path, index: int) -> dict:
    """Checkpoint parameter estimation before independent bounded replay."""
    frozen = verify(output)
    if not 0 <= index < len(frozen["tasks"]):
        raise ValueError("task index outside the frozen plan")
    task = frozen["tasks"][index]
    identity = content_hash([frozen["identity"], task])
    root = output / f"results/task_{index:03d}"
    result_path, fit_path = root / "result.json", root / "fit.json"
    if result_path.exists():
        return _checkpoint(result_path, identity)
    if not _checkpoint(output / "gate/result.json", frozen["identity"]).get("pass"):
        raise ValueError("paired model preflight did not pass")
    record = {
        "identity": identity,
        "task": task,
        "test_data_opened": False,
        "proposer_access": False,
    }
    if (root / "fit_started.json").exists() and not fit_path.exists():
        _checkpoint(root / "fit_started.json", identity)
        record.update(
            status="interrupted", error="Native fit interrupted; no fresh budget"
        )
        write_json(result_path, record, immutable=True)
        return record
    plan = FreedomPlan.model_validate(frozen["plan"])
    problem = read_json(output / f"problems/{index:03d}.json")
    system, _ = system_for(problem)
    train, val = [unpack_split(problem["splits"][s]) for s in ("train", "val")]
    scale = TrainingScaler().fit(train).scales["target:v01"].standard_deviation
    try:
        if fit_path.exists():
            saved = _checkpoint(fit_path, identity)
        else:
            write_json(
                root / "fit_started.json", {"identity": identity}, immutable=True
            )
            started = monotonic()
            fit = fit_collocation_forward_sensitivity(
                compile_candidate(
                    CandidateModel.model_validate(problem["candidate"]),
                    ValidationContext.model_validate(problem["context"]),
                ),
                train,
                val,
                plan.fit,
                root,
                initial_parameters=problem["start"],
                initialization_plan=LatentInitializationPlan.model_validate(
                    problem["initialization_plan"]
                ),
            )
            saved = {
                "identity": identity,
                "fit": fit,
                "estimation_seconds": monotonic() - started,
            }
            write_json(fit_path, saved, immutable=True)
        record.update(saved)
        fit = saved["fit"]
        if not fit.get("parameters"):
            record["status"] = "fit_failed"
        else:
            clean = read_json(output / "inputs/clean.json")
            checks = {}
            for key, split in (("train", train), ("val", val)):
                path, intent = (
                    root / f"replay_{key}.json",
                    root / f"replay_{key}_started.json",
                )
                if not path.exists():
                    if intent.exists():
                        _checkpoint(intent, identity)
                        check = {
                            "pass": False,
                            "status": "interrupted",
                            "error": "No fresh replay budget",
                        }
                    else:
                        write_json(intent, {"identity": identity}, immutable=True)
                        check = checked_replay(
                            system.model,
                            split,
                            fit["parameters"],
                            scale,
                            clean[key],
                            seconds=plan.replay_seconds,
                        )
                    write_json(
                        path, {"identity": identity, "check": check}, immutable=True
                    )
                checks[key] = _checkpoint(path, identity)["check"]
            verified = all(c.get("pass") for c in checks.values())
            record.update(
                replays=checks,
                verified=verified,
                status="complete" if verified else "replay_unverified",
                recovered=verified
                and all(
                    c["scores"]["Radau"]["clean_nmse"] <= 1e-4 for c in checks.values()
                ),
            )
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    write_json(result_path, _finite_payload(record), immutable=True)
    return record
