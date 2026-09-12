"""Explicitly authorized reference-skeleton diagnostics, never proposer inputs.

No benchmark coefficients are stored in source. The private specification is an
external, hashed artifact. All continuous equation constants become parameters;
skew sharing and the declared functional graph are retained.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from autoformalism.benchmarks.phase_b_generation import (
    _scalar_input,
    _simulate_alien,
    phase_b_protocols,
)
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.rebuttal.fitter_diagnostic import read_json, sha256, write_json
from autoformalism.schemas import CandidateModel
from autoformalism.staged_topology import content_hash


def reference_skeleton(spec: dict) -> tuple[dict, dict, dict]:
    """Translate the trusted generator grammar without embedding fitted truth."""
    if spec.get("grammar_version") != "alien_grammar_v1":
        raise ValueError("unsupported reference grammar")
    n = int(spec["n_latent"])
    if not 1 <= n <= 8:
        raise ValueError("unsupported reference dimension")
    names = [f"z{i + 1}" for i in range(n)]
    skew = np.asarray(spec["skew"], dtype=float)
    if skew.shape != (n, n) or not np.allclose(skew, -skew.T, atol=1e-14):
        raise ValueError("reference coupling is not skew symmetric")
    parameters, truth = [], {}

    def parameter(name, value, positive=False):
        value = float(value)
        if not np.isfinite(value) or (positive and value <= 0):
            raise ValueError("nonfinite or invalid reference parameter")
        parameters.append(
            {
                "name": name,
                "scope": "global",
                "role": "rate" if positive else "coefficient",
            }
        )
        truth[name] = value
        return name

    equations = [
        f"-{parameter(f'decay{i}', spec['decay'][i], True)}*{s}"
        for i, s in enumerate(names)
    ]
    for i in range(n):
        for j in range(i + 1, n):
            if skew[i, j] != 0:
                p = parameter(f"skew{i}_{j}", skew[i, j])
                equations[i] += f"+{p}*{names[j]}"
                equations[j] += f"-{p}*{names[i]}"

    def source(term, key):
        index = int(term[key])
        if index not in range(n):
            raise ValueError("reference source outside state range")
        return names[index]

    def nonlinear(term, prefix, product=False, bias=False):
        weight = parameter(prefix + "w", term["coefficient"])
        if product:
            factors = [
                f"tanh({parameter(prefix + f's{k}', term[f'scale_{k}'], True)}"
                f"*{source(term, f'source_{k}')})"
                for k in (1, 2)
            ]
        else:
            scale = parameter(prefix + "s", term["scale"], True)
            offset = "+" + parameter(prefix + "b", term["bias"]) if bias else ""
            factors = [f"tanh({scale}*{source(term, 'source')}{offset})"]
        return "+" + "*".join([weight, *factors])

    for i in range(n):
        for j, term in enumerate(spec["tanh_terms"][i]):
            equations[i] += nonlinear(term, f"t{i}_{j}_", bias=True)
        for j, term in enumerate(spec["product_terms"][i]):
            equations[i] += nonlinear(term, f"p{i}_{j}_", product=True)
    input_scale = parameter("input_scale", spec["input_scale"], True)
    for i, value in enumerate(spec["input_vector"]):
        if value != 0:
            equations[i] += f"+{parameter(f'input{i}', value)}*tanh({input_scale}*u01)"
    output = f"-{parameter('output_decay', spec['output_decay'], True)}*v01"
    for j, term in enumerate(spec["output_terms"]):
        output += nonlinear(term, f"o{j}_")
    for j, term in enumerate(spec["output_product_terms"]):
        output += nonlinear(term, f"op{j}_", product=True)
    names.append("v01")
    equations.append(output)
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "isolated_reference_skeleton",
            "parent_candidate_id": None,
            "states": [
                {"name": s, "kind": "observed" if s == "v01" else "latent"}
                for s in names
            ],
            "state_equations": [
                {"state": s, "rhs": rhs}
                for s, rhs in zip(names, equations, strict=True)
            ],
            "observation_mappings": [{"channel": "v01", "expression": "v01"}],
            "parameters": parameters,
            "initial_conditions": [
                {
                    "state": s,
                    "scope": "global",
                    "expression": s if s == "v01" else f"known_{s}",
                }
                for s in names
            ],
        }
    )
    context = ValidationContext(
        targets=("v01",),
        external_inputs=("u01",),
        fixed_covariates=tuple(f"known_{s}" for s in names[:-1]),
        fitted_initialization=True,
    )
    compile_candidate(candidate, context)
    return candidate.model_dump(mode="json"), context.model_dump(mode="json"), truth


def export_reference(spec_path, output) -> dict:
    """Export only the selected specification; no trajectory/test files are read."""
    spec = read_json(spec_path)
    candidate, context, truth = reference_skeleton(spec)
    result = {
        "protocol": "isolated-fitter-reference-1",
        "spec": spec,
        "candidate": candidate,
        "context": context,
        "truth": truth,
        "source_sha256": sha256(spec_path),
        "test_data_opened": False,
        "proposer_access": False,
        "llm_calls": 0,
    }
    result["identity"] = content_hash(result)
    write_json(output, result, immutable=True)
    return result


def reference_problem(template: dict, bundle: dict) -> dict:
    """Supply known boundaries only in a labelled oracle diagnostic.

    Public trajectory IDs and forcing samples are checked against the generator
    protocol before assigning initial conditions. Hidden trajectories are not
    loaded or placed in fitting residuals.
    """
    problem = deepcopy(template)
    problem.update(
        candidate=bundle["candidate"],
        context=bundle["context"],
        initialization_plan={"rules": {}},
        start={},
    )
    n = bundle["spec"]["n_latent"]
    for key, split in (("train", "train"), ("val", "validation")):
        protocols = [p for p in phase_b_protocols("alien_device") if p.split == split]
        rows = problem["splits"][key]["rows"]
        if len(rows) != len(protocols):
            raise ValueError("reference/public trajectory counts differ")
        by_id = {r["trajectory_id"]: r for r in rows}
        for i, protocol in enumerate(protocols):
            row = by_id[f"{split}_{i:03d}"]
            t = np.asarray(row["time"])
            expected = np.asarray([_scalar_input(x, protocol.specification) for x in t])
            if (
                abs(t[0]) > 1e-12
                or abs(t[-1] - protocol.duration) > 1e-10
                or not np.allclose(
                    expected, row["external_inputs"]["u01"], rtol=1e-10, atol=1e-11
                )
            ):
                raise ValueError("reference/public forcing or horizon differs")
            initial = protocol.specification.get("initial_shift", [0.0] * (n + 1))
            row["fixed_covariates"] = {
                f"known_z{j + 1}": float(initial[j]) for j in range(n)
            }
    return problem


def native_replay_audit(problem: dict, spec: dict) -> dict:
    """Check original generator against public data, independently of the compiler."""
    audit = {}
    for key, split in (("train", "train"), ("val", "validation")):
        rows = {r["trajectory_id"]: r for r in problem["splits"][key]["rows"]}
        checks = []
        for i, protocol in enumerate(
            p for p in phase_b_protocols("alien_device") if p.split == split
        ):
            row = rows[f"{split}_{i:03d}"]
            solved = _simulate_alien(protocol, spec, {})
            if not np.allclose(solved.time, row["time"], rtol=0, atol=1e-10):
                raise ValueError("native/public grids differ")
            difference = solved.states[:, -1] - np.asarray(row["targets"]["v01"])
            checks.append(
                {
                    "trajectory": row["trajectory_id"],
                    "maximum_absolute_difference": float(np.max(abs(difference))),
                }
            )
        audit[key] = checks
    return audit


def shared_boundary_lower_bound(problem: dict) -> dict:
    """Public-only least-squares floor for indistinguishable training experiments.

    Equal full forcing paths, observed initial values and public covariates must
    produce identical outputs under a deterministic shared-initial model. The
    pointwise mean is the unconstrained best common trajectory. No validation
    observations or hidden trajectories enter this bound.
    """
    from autoformalism.data import TrainingScaler
    from autoformalism.rebuttal.piecewise_campaign import unpack_split

    train = unpack_split(problem["splits"]["train"])
    scale = TrainingScaler().fit(train).scales["target:v01"].standard_deviation
    groups = {}
    for row in problem["splits"]["train"]["rows"]:
        signature = {
            "time": row["time"],
            "external_inputs": row["external_inputs"],
            "auxiliaries": row["auxiliaries"],
            "initial_targets": {k: v[0] for k, v in row["targets"].items()},
            "public_covariates": {
                k: v
                for k, v in row["fixed_covariates"].items()
                if not k.startswith("known_z")
            },
        }
        groups.setdefault(content_hash(signature), []).append(row)
    error, conflicts = 0.0, []
    for group in groups.values():
        y = np.asarray([r["targets"]["v01"] for r in group])
        value = float(np.sum((y - np.mean(y, axis=0)) ** 2))
        error += value
        if value > 0:
            conflicts.append(
                {
                    "trajectory_ids": [r["trajectory_id"] for r in group],
                    "unavoidable_squared_error": value,
                }
            )
    count = sum(len(t.time) for t in train.trajectories)
    return {
        "training_nmse_lower_bound": error / (count * scale**2),
        "conflicting_groups": conflicts,
        "training_only": True,
        "applies_to": "deterministic models with shared or public-causal initials",
    }
