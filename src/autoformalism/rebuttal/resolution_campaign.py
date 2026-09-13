"""Gated mesh and sparse-sensitivity diagnostics on an isolated frozen reference."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.data import TrainingScaler
from autoformalism.fitting.sensitivity_probe import (
    SymbolicODE,
    SymbolicOracle,
    symbolic_rollout,
)
from autoformalism.rebuttal.attainability_campaign import (
    AttainabilityPlan,
    _checkpoint,
    _identity,
    code_identity,
    verify,
)
from autoformalism.rebuttal.attainability_controls import ordinary_start, system_for
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
from autoformalism.staged_topology import content_hash


def prepare_resolution(source: Path, output: Path, plan: AttainabilityPlan) -> dict:
    """Reuse audited observations byte-for-byte; authorize new, labelled fits only."""
    if source.resolve().is_relative_to(
        output.resolve()
    ) or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("source/output must be separate")
    old = read_json(source / "freeze.json")
    _identity(old)
    if (
        old["plan"]["protocol"] != "fitter-attainability-2"
        or plan.protocol != "fitter-attainability-3"
    ):
        raise ValueError("requires completed v2 generation and a v3 plan")
    if (
        old.get("test_data_opened") is not False
        or old.get("proposer_access") is not False
    ):
        raise ValueError("source isolation differs")
    for relative, digest in old["assets"].items():
        if sha256(safe_path(source, relative)) != digest:
            raise ValueError("source input changed: " + relative)
    runtime = identity_runtime()
    if any(runtime[k] != old["runtime"][k] for k in ("packages", "casadi")):
        raise ValueError("use original numerical packages")
    if runtime["python"].split(".")[:2] != old["runtime"]["python"].split(".")[:2]:
        raise ValueError("use original Python major/minor")
    cases = [c for c in old["cases"] if c["reference_skeleton"]]
    if len(cases) != 1:
        raise ValueError("expected one reference case")
    index = cases[0]["index"]
    generated = _checkpoint(
        source / f"generated/{index:03d}/result.json",
        content_hash([old["identity"], "generate", index]),
    )
    if generated.get("status") != "complete" or not generated.get(
        "scaled_reference_audit", {}
    ).get("pass"):
        raise ValueError("reference generation must have passed its independent audit")
    bundle = read_json(source / "reference_input.json")
    _identity(bundle)
    if (
        bundle.get("proposer_access") is not False
        or bundle.get("test_data_opened") is not False
    ):
        raise ValueError("reference export isolation differs")
    candidate, context, truth = reference_skeleton(bundle["spec"])
    if any(
        bundle[k] != v
        for k, v in (("candidate", candidate), ("context", context), ("truth", truth))
    ):
        raise ValueError("reference translation changed")
    problem = read_json(source / f"problems/{index:03d}.json")
    if reference_problem(problem, bundle) != problem:
        raise ValueError("reference input/initial protocol changed")
    assets = {}

    def copy(relative, destination):
        path = safe_path(output, destination)
        _write_bytes(path, safe_path(source, relative).read_bytes(), immutable=True)
        assets[destination] = sha256(path)

    copy("freeze.json", "provenance/source_freeze.json")
    copy(f"generated/{index:03d}/result.json", "provenance/source_generation.json")
    copy("reference_input.json", "reference_input.json")
    copy(f"problems/{index:03d}.json", "problems/000.json")
    for relative, digest in generated["assets"].items():
        path = safe_path(source / f"generated/{index:03d}", relative)
        if sha256(path) != digest:
            raise ValueError("generated asset changed")
        copy(f"generated/{index:03d}/{relative}", f"generated/000/{relative}")
    tasks = []
    for dataset, arms in (
        (
            "synthetic",
            (
                "fixed_full",
                "fixed_mesh",
                "blind0_full",
                "blind0_mesh",
                "near_full",
                "near_mesh",
            ),
        ),
        ("actual", ("blind0_mesh", "near_mesh")),
    ):
        for arm in arms:
            tasks.append(
                {"index": len(tasks), "case_index": 0, "dataset": dataset, "arm": arm}
            )
    profiles = [
        {"index": i, "point": point, "format": fmt}
        for i, (point, fmt) in enumerate(
            (p, f) for p in ("truth", "near", "ordinary") for f in ("dense", "sparse")
        )
    ]
    frozen = {
        "plan": plan.model_dump(mode="json"),
        "cases": [
            {"index": 0, "label": "reference_skeleton", "reference_skeleton": True}
        ],
        "tasks": tasks,
        "profiles": profiles,
        "assets": assets,
        "code": code_identity(),
        "runtime": runtime,
        "source_identity": old["identity"],
        "test_data_opened": False,
        "proposer_access": False,
        "private_reference_opened": True,
        "llm_calls": 0,
        "profile_settings": {
            "total_seconds": 360,
            "state_seconds": 60,
            "sensitivity_seconds": 180,
            "directional_relative_tolerance": 0.005,
            "solver_relative_tolerance": 0.001,
            "solver_residual_maximum_difference": 1e-5,
        },
    }
    frozen["identity"] = content_hash(frozen)
    # The re-keyed generation record labels reuse explicitly; its original bytes
    # and identity remain separately hashed provenance. No trajectory is generated.
    record = {
        **deepcopy(generated),
        "identity": content_hash([frozen["identity"], "generate", 0]),
        "case": frozen["cases"][0],
        "reused_from": generated["identity"],
    }
    write_json(output / "generated/000/result.json", record, immutable=True)
    write_json(output / "freeze.json", frozen, immutable=True)
    return verify(output)


def profile_point(problem, truth, name):
    """Reuse the original deterministic ordinary and truth-assisted starts."""
    if name == "ordinary":
        return ordinary_start(problem)
    if name == "truth":
        return truth
    if name != "near":
        raise ValueError("unknown diagnostic point")
    rng = np.random.default_rng(20260914)
    return {
        n: float(v * np.exp(rng.normal(0, 0.2))) if v else 0.1
        for n, v in sorted(truth.items())
    }


def evaluate_profile(problem, point, fmt, directory, limits):
    """Measure a whole-training residual/Jacobian without taking optimizer steps."""
    started = monotonic()
    system, _ = system_for(problem)
    system = SymbolicODE(system.model, allow_piecewise=True, solver_jacobian_format=fmt)
    training = unpack_split(problem["splits"]["train"])
    scale = TrainingScaler().fit(training).scales["target:v01"].standard_deviation
    settings = AttainabilityPlan().fit.fit_config()
    theta = np.array([point[n] for n in system.names])
    result = {
        "format": fmt,
        "audit": system.audit,
        "training_only": True,
        "point": point,
    }
    for label, sens, seconds in (
        ("state", False, limits["state_seconds"]),
        ("sensitivity", True, limits["sensitivity_seconds"]),
    ):
        deadline = min(started + limits["total_seconds"], monotonic() + seconds)
        oracle = SymbolicOracle(
            system,
            training,
            scale,
            settings,
            directory / label,
            deadline,
            sensitivities=sens,
        )
        before = monotonic()
        try:
            residual = oracle(theta)
            if not oracle.valid_calls:
                raise ValueError(
                    oracle.last_evaluation.get("error", "unavailable evaluation")
                )
            record = {
                "status": "complete",
                "seconds": monotonic() - before,
                "nmse": float(np.mean(residual**2)),
                "solver_counts": oracle.solver_counts,
            }
            if sens:
                jac = oracle.jacobian(theta)
                temporary = directory / "arrays.tmp.npz"
                np.savez_compressed(temporary, residual=residual, jacobian=jac)
                temporary.replace(directory / "arrays.npz")
                result["arrays_sha256"] = sha256(directory / "arrays.npz")
        except (TimeoutError, ValueError, RuntimeError) as error:
            record = {
                "status": "timeout" if isinstance(error, TimeoutError) else "failed",
                "error": str(error),
                "seconds": monotonic() - before,
                "failure": oracle.last_evaluation,
            }
        result[label] = record
        write_json(directory / "progress.json", _finite_payload(result))
    checks = []
    if result["sensitivity"]["status"] == "complete":
        data = training.trajectories[0]
        jac = np.load(directory / "arrays.npz")["jacobian"][: len(data.time)]
        rng = np.random.default_rng(20260915)
        for _ in range(2):
            direction = rng.normal(size=len(theta)) * np.maximum(1, abs(theta))
            direction /= np.linalg.norm(direction)
            step = 1e-4
            deadline = started + limits["total_seconds"]
            try:
                high = symbolic_rollout(
                    system, data, theta + step * direction, settings, deadline
                )[0][:, 0]
                low = symbolic_rollout(
                    system, data, theta - step * direction, settings, deadline
                )[0][:, 0]
                finite = (high - low) / (2 * step * scale)
                analytic = jac @ direction
                error = float(
                    np.linalg.norm(analytic - finite)
                    / max(1e-8, np.linalg.norm(finite))
                )
                checks.append(
                    {
                        "relative_error": error,
                        "pass": bool(error <= limits["directional_relative_tolerance"]),
                    }
                )
            except (TimeoutError, ValueError, RuntimeError) as error:
                checks.append({"pass": False, "error": str(error)})
                break
    result.update(
        directional_checks=checks,
        seconds=monotonic() - started,
        status="complete"
        if result["sensitivity"]["status"] == "complete"
        and len(checks) == 2
        and all(c["pass"] for c in checks)
        else "unverified",
    )
    return _finite_payload(result)


def run_profile(output: Path, index: int):
    """Resume completed records; never renew an interrupted profile budget."""
    frozen = verify(output)
    task = frozen["profiles"][index]
    identity = content_hash([frozen["identity"], "profile", task])
    directory = output / f"profiles/{index:03d}"
    result_file = directory / "result.json"
    if result_file.exists():
        return _checkpoint(result_file, identity)
    started = directory / "started.json"
    if started.exists():
        _checkpoint(started, identity)
        result = {
            "status": "interrupted",
            "error": "Profile interrupted; no renewed budget",
        }
    else:
        write_json(started, {"identity": identity}, immutable=True)
        problem = read_json(output / "generated/000/synthetic.json")
        truth = read_json(output / "generated/000/truth.json")
        result = evaluate_profile(
            problem,
            profile_point(problem, truth, task["point"]),
            task["format"],
            directory,
            frozen["profile_settings"],
        )
    result.update(identity=identity, task=task)
    write_json(result_file, result, immutable=True)
    return result


def preflight(output: Path):
    """Gate expensive fits on resolved fixed nodes and verified sparse derivatives."""
    frozen = verify(output)
    if (output / "preflight.json").exists():
        return _checkpoint(output / "preflight.json", frozen["identity"])
    checks = []
    for task in frozen["tasks"][:2]:
        path = output / f"results/task_{task['index']:03d}/result.json"
        row = (
            _checkpoint(path, content_hash([frozen["identity"], task]))
            if path.exists()
            else {}
        )
        fit = row.get("fit") or {}
        checks.append(
            {
                "check": task["arm"],
                "pass": bool(
                    fit.get("success")
                    and fit.get("collocation_training_nmse") is not None
                    and fit["collocation_training_nmse"] <= 1e-4
                    and (fit.get("last_iteration") or {}).get(
                        "constraint_maximum", np.inf
                    )
                    <= 1e-6
                ),
            }
        )
    profiles = {}
    for task in frozen["profiles"]:
        path = output / f"profiles/{task['index']:03d}/result.json"
        if path.exists():
            row = _checkpoint(path, content_hash([frozen["identity"], "profile", task]))
            profiles[(task["point"], task["format"])] = row
            if (
                row.get("arrays_sha256")
                and sha256(path.parent / "arrays.npz") != row["arrays_sha256"]
            ):
                raise ValueError("profile array changed")
    checks.append(
        {
            "check": "truth_sparse_derivative",
            "pass": profiles.get(("truth", "sparse"), {}).get("status") == "complete"
            and profiles[("truth", "sparse")]["sensitivity"].get("nmse", np.inf)
            <= 1e-6,
        }
    )
    comparisons = []
    for name in ("truth", "near", "ordinary"):
        a, b = [profiles.get((name, fmt), {}) for fmt in ("dense", "sparse")]
        if all(x.get("sensitivity", {}).get("status") == "complete" for x in (a, b)):
            arrays = [
                np.load(output / f"profiles/{x['task']['index']:03d}/arrays.npz")
                for x in (a, b)
            ]
            errors = profile_differences(*arrays)
            comparisons.append(
                {
                    "point": name,
                    "errors": errors,
                    "pass": errors["maximum_scaled_prediction_difference"]
                    <= frozen["profile_settings"]["solver_residual_maximum_difference"]
                    and errors["relative_jacobian_difference"]
                    <= frozen["profile_settings"]["solver_relative_tolerance"],
                }
            )
    checks.extend(
        {"check": "dense_sparse:" + r["point"], "pass": r["pass"]} for r in comparisons
    )
    result = {
        "identity": frozen["identity"],
        "pass": all(c["pass"] for c in checks),
        "checks": checks,
        "dense_sparse_comparisons": comparisons,
        "profile_statuses": [
            {"point": p, "format": f, "status": r["status"]}
            for (p, f), r in profiles.items()
        ],
        "training_only": True,
        "test_data_opened": False,
        "proposer_access": False,
    }
    write_json(output / "preflight.json", result, immutable=True)
    return result


def profile_differences(left, right):
    """Compare predictions in training-scale units even at a zero-loss truth."""
    for name in ("residual", "jacobian"):
        if left[name].shape != right[name].shape or not all(
            np.isfinite(a[name]).all() for a in (left, right)
        ):
            raise ValueError("profile arrays must be aligned and finite")
    return {
        "maximum_scaled_prediction_difference": float(
            np.max(abs(left["residual"] - right["residual"]), initial=0)
        ),
        "relative_jacobian_difference": float(
            np.linalg.norm(left["jacobian"] - right["jacobian"])
            / max(1e-8, np.linalg.norm(left["jacobian"]))
        ),
    }
