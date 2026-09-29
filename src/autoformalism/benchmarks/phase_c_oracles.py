"""Evaluator-only equation witnesses; never include these in public prompts."""

from __future__ import annotations

from time import monotonic

import numpy as np

from autoformalism.benchmarks.phase_b_public import PhaseBPublicSpec
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit


def _product(term: dict, states: list[str]) -> str:
    return (
        f"({term['coefficient']!r})*tanh(({term['scale_1']!r})*"
        f"{states[term['source_1']]})*tanh(({term['scale_2']!r})*"
        f"{states[term['source_2']]})"
    )


def reference_request(spec: PhaseBPublicSpec, truth: dict) -> PublicFitRequest:
    """Transcribe the actual generator with public initial observations and reset.

    Constants are evaluator assistance. This tests public-interface attainability,
    not optimization or identification. Neither a hidden trajectory nor a fitted
    validation initial state is used.
    """
    inputs = {
        c.private_source: c.public_name
        for c in spec.channels
        if c.role == "external_input"
    }
    if spec.family == "alien_device":
        count = truth["n_latent"]
        sources = [*(f"z{i + 1}" for i in range(count)), "y"]
        states = [f"x{i}" for i in range(count + 1)]
        initials = [0.0] * len(states)
        equations = []
        for i in range(count):
            terms = [f"-({truth['decay'][i]!r})*{states[i]}"]
            terms.extend(
                f"({c!r})*{states[j]}" for j, c in enumerate(truth["skew"][i]) if c
            )
            terms.extend(
                f"({t['coefficient']!r})*tanh(({t['scale']!r})*{states[t['source']]}+({t['bias']!r}))"
                for t in truth["tanh_terms"][i]
            )
            terms.extend(_product(t, states) for t in truth["product_terms"][i])
            terms.append(
                f"({truth['input_vector'][i]!r})*tanh(({truth['input_scale']!r})*{inputs['u']})"
            )
            equations.append(" + ".join(terms))
        terms = [f"-({truth['output_decay']!r})*{states[-1]}"]
        terms.extend(
            f"({t['coefficient']!r})*tanh(({t['scale']!r})*{states[t['source']]})"
            for t in truth["output_terms"]
        )
        terms.extend(_product(t, states) for t in truth["output_product_terms"])
        equations.append(" + ".join(terms))
    elif spec.family == "cstr":
        p = truth["parameters"]
        sources, states = ["C", "T", "Tj"], ["x0", "x1", "x2"]
        initials = [truth["equilibrium"][n] for n in sources]
        reaction = f"({p['k0']!r})*exp(-({p['E_over_R']!r})/max(x1,250))*max(x0,0)"
        equations = [
            f"({p['flow_rate']!r})*({inputs['Cf']}-x0)-({reaction})",
            f"({p['flow_rate']!r})*({inputs['Tf']}-x1)+({p['source_gain']!r})*({reaction})-({p['exchange_rate']!r})*(x1-x2)",
            f"({p['secondary_flow_rate']!r})*({inputs['Tjf']}-x2)+({p['secondary_exchange_rate']!r})*(x1-x2)",
        ]
    else:
        raise ValueError("no exact public-interface oracle for this family")
    mapping = dict(zip(sources, states, strict=True))
    observations = [
        {"channel": c.public_name, "expression": mapping[c.private_source]}
        for c in spec.channels
        if c.role == "target"
    ]
    observed_states = {item["expression"] for item in observations}
    available_initials = {
        mapping[c.private_source]: c.public_name
        for c in spec.channels
        if c.role == "auxiliary" and c.private_source in mapping
    }
    initial_rules = {
        state: {
            "initial": {
                "mode": "map",
                "expression": available_initials.get(state, repr(value)),
                "parameters": [],
            }
        }
        for state, value in zip(states, initials, strict=True)
        if state not in observed_states
    }
    candidate = {
        "candidate_id": f"phase_c_reference_{spec.family}_{spec.tier}",
        "parent_candidate_id": None,
        "states": [{"name": s, "kind": "latent"} for s in states],
        "state_equations": [
            {"state": s, "rhs": rhs} for s, rhs in zip(states, equations, strict=True)
        ],
        "observation_mappings": observations,
        "initial_conditions": [
            {"state": s, "scope": "global", "fixed_value": v}
            for s, v in zip(states, initials, strict=True)
        ],
    }
    return PublicFitRequest.model_validate(
        {
            "base_candidate": candidate,
            "context": {
                "targets": [c.public_name for c in spec.channels if c.role == "target"],
                "auxiliaries": [
                    c.public_name for c in spec.channels if c.role == "auxiliary"
                ],
                "external_inputs": list(inputs.values()),
            },
            "initialization_plan": {"rules": initial_rules},
            "profile": "general-rollout-v1",
            "source": {
                "stage": "synthetic_control",
                "task_id": candidate["candidate_id"],
                "artifact_sha256": public.content_sha256(truth),
            },
        }
    )


def replay(
    request: PublicFitRequest,
    splits: tuple[PublicSplit, ...],
    parameters: dict | None = None,
) -> dict:
    """Replay known equations through the production solver without fitting."""
    if any(s.name == "test" for s in splits):
        raise ValueError("public-interface replay is development-only")
    model, _, _ = public._lower(request)
    parameters = parameters or {}
    if set(parameters) != set(model.parameter_names):
        raise ValueError("incomplete reference parameter vector")
    train = next(s for s in splits if s.name == "train")
    channels = tuple(request.context.targets)
    scales = {
        c: max(float(np.std(np.concatenate([r.targets[c] for r in train.rows]))), 1e-12)
        for c in channels
    }
    records = []
    deadline = monotonic() + 600
    for split in splits:
        for row in public.unpack_split(split).trajectories:
            predictions = []
            for method in ("Radau", "DOP853"):
                sim = simulate_trajectory(
                    model,
                    row,
                    parameters,
                    {},
                    FitConfig(
                        integration_method=method,
                        relative_tolerance=1e-9,
                        absolute_tolerance=1e-11,
                    ),
                    reset_observed_states=False,
                    deadline=deadline,
                )
                if not sim.success:
                    raise ValueError(
                        f"reference replay failed: {row.trajectory_id}/{method}: "
                        f"{sim.message}"
                    )
                predictions.append(sim.predictions)
            records.append(
                {
                    "split": split.name,
                    "trajectory_id": row.trajectory_id,
                    "nmse": {
                        c: float(
                            np.mean(
                                ((predictions[0][c] - row.targets[c]) / scales[c]) ** 2
                            )
                        )
                        for c in channels
                    },
                    "solver_difference": max(
                        float(
                            np.max(np.abs(predictions[0][c] - predictions[1][c]))
                            / scales[c]
                        )
                        for c in channels
                    ),
                }
            )
    return {
        "passed": all(
            max(r["nmse"].values()) <= 1e-6 and r["solver_difference"] <= 1e-4
            for r in records
        ),
        "maximum_trajectory_nmse": max(max(r["nmse"].values()) for r in records),
        "maximum_solver_difference": max(r["solver_difference"] for r in records),
        "rows": records,
        "parameter_fitting_performed": False,
        "test_evaluated": False,
    }
