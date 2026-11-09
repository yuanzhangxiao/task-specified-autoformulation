"""Public specifications and executable representation constraints for paired search."""

from __future__ import annotations

import hashlib
import re

from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.rebuttal import mechanisms
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.staged_topology import (
    ModelingLimits,
    PublicScientificBrief,
    PublicVariable,
    ScientificRequirement,
)
from autoformalism.targets import PublicTargetContract, evaluate_public_targets

ARMS = (
    "full",
    "fixed_gamma",
    "no_latent",
    "no_spec",
    "underspecified",
    "incorrect",
    "overrestrictive",
    "observed_only",
)
MUTATIONS = {"underspecified", "incorrect", "overrestrictive"}


def clauses(prompt: str) -> list[str]:
    """Read explicit scientific bullets from section A, never infer a private law."""
    intro = prompt.partition("B. Available data")[0]
    items = re.findall(r"(?m)^-\s+([^\n]+)", intro)
    if not items:
        raise ValueError("public section A has no explicit mechanism clauses")
    return items


def family(context: ValidationContext) -> str:
    roles = (set(context.targets), set(context.external_inputs))
    if roles[0] == {"Gp", "I"} and "insulin_pmol_per_kg_min" in roles[1]:
        return "t2"
    if roles[0] == {"Gp"}:
        return "t1"
    if roles == ({"T"}, {"Tf", "Tjf", "Cf"}):
        return "cstr"
    if roles == ({"v01"}, {"u01"}):
        return "device"
    raise ValueError("unsupported public task and channel contract")


def active_spec(prompt: str, context: ValidationContext, arm: str) -> tuple:
    """Construct separately versioned manipulations of the public scientific brief."""
    if arm not in ARMS:
        raise ValueError("unknown search treatment")
    items = clauses(prompt)
    kind = family(context)
    target = {"t1": "Gp", "t2": "Gp", "device": "v01", "cstr": "T"}[kind]
    if kind in {"t1", "t2"}:
        meal = [v for v in context.external_inputs if "meal" in v]
        if len(meal) != 1 or "meal" not in items[0].lower():
            raise ValueError("unrecognized glucose mechanism clause")
        bindings = [("meal_pathway", (meal[0],), False)]
        if kind == "t2":
            if len(items) != 2 or "delayed insulin" not in items[1].lower():
                raise ValueError("unrecognized insulin obligation")
            bindings.append(("delayed_insulin_action", ("I",), True))
    elif kind == "cstr":
        if "reactor-temperature balance" not in items[0] or "jacket" not in items[0]:
            raise ValueError("unrecognized reactor balance")
        bindings = [("reactor_heat_balance", (), False)]
    else:
        bindings = [("input_driven_memory", ("u01",), True)]
        if "memory" not in items[0].lower():
            raise ValueError("unrecognized procedural device specification")
    if len(items) != len(bindings):
        raise ValueError("public obligations and reviewed operational bindings differ")
    if arm in MUTATIONS and kind not in {"t2", "cstr"}:
        raise ValueError("specification variants are defined on the two hard anchors")
    active = list(items)
    if arm == "underspecified":
        if kind == "t2":
            active.pop()
            bindings.pop()
        else:
            active = [
                "a reactor-temperature balance accounting for feed transport "
                "and reaction heat generation."
            ]
    if arm == "incorrect":
        if kind == "t2":
            active[-1] = (
                "a delayed pathway through which meal ingestion alone regulates the "
                "delayed disposal drive; plasma insulin has no causal pathway to the "
                "modeled glucose balance."
            )
            bindings[-1] = ("delayed_meal_disposal", (meal[0],), True)
        else:
            active = [
                "a reactor-temperature balance driven by feed transport and reaction "
                "heat, with no influence from the jacket-feed temperature."
            ]
            bindings = [("reactor_feed_balance", ("Tf",), False)]
    if arm == "overrestrictive":
        active.append(
            "Use exactly one unobserved differential state for the entire system."
        )
    if arm == "no_spec":
        active = []
        bindings = []
    source = prompt
    if arm in MUTATIONS:
        old_block = "\n".join("- " + text for text in items)
        if prompt.count(old_block) != 1:
            raise ValueError("ambiguous mechanism clause boundaries")
        source = prompt.replace(old_block, "\n".join("- " + text for text in active))
        if kind == "t2" and arm in {"incorrect", "underspecified"}:
            source = source.replace(
                "post-meal glucose regulation and insulin-dependent disposal",
                "post-meal glucose regulation",
            )
    elif arm == "no_spec":
        data = prompt.partition("B. Available data")[2].partition(
            "C. Modeling requirements"
        )[0]
        if not data.strip():
            raise ValueError("public channel/preparation section missing")
        background = {
            "t1": "post-meal glucose regulation",
            "t2": "post-meal glucose regulation",
            "cstr": "a controlled stirred-tank reactor",
            "device": "an unfamiliar device",
        }[kind]
        source = (
            f"A. Task specification\nConstruct a reduced model of {background}.\n"
            + "\nB. Available data"
            + data
            + "\nC. Modeling requirements\nGenerate every target by safe analytic "
            "continuous-time equations. Use globally shared parameters and causal "
            "initialization. Latent ODE states and algebraic processes are allowed. "
            "Future target measurements and free time-series fits are forbidden.\n"
        )
    if arm in {"no_latent", "observed_only"}:
        marker = "C. Modeling requirements"
        if marker not in source:
            raise ValueError("missing modeling-requirements section")
        source = source.replace(
            marker,
            marker + "\nUse differential states only for "
            "directly observed target coordinates. Other generated quantities must be "
            "algebraic. Do not use delays, filters, or extra memory.",
            1,
        )
    requirements = [
        mechanisms.MechanismRequirement(
            id=name,
            public_requirement=text,
            required_drivers=drivers,
            required_targets=(target,),
            requires_dynamic_memory=memory,
        )
        for text, (name, drivers, memory) in zip(active, bindings, strict=False)
    ]
    return source, requirements, active


def treatment(cell: dict, arm: str, anchor: dict | None = None) -> dict:
    """Rebind every search interface to the arm's public information."""
    context = ValidationContext.model_validate(cell["context"])
    prompt, requirements, active = active_spec(cell["prompt"], context, arm)
    cap = cell.get("max_parameters")
    if cap is not None:
        prompt = prompt.replace(
            "C. Modeling requirements",
            "C. Modeling requirements\n"
            f"Fit at most {cap} coefficients, including initializer coefficients.",
            1,
        )
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    target_rows = [
        {
            "target_channel": name,
            "public_requirement": f"Generate the public target {name}.",
        }
        for name in context.targets
    ]
    # Keep the historical role of measured plasma glucose as a dynamic coordinate.
    for row in target_rows:
        if row["target_channel"] == "Gp":
            row.update(
                expected_representation="dynamic_state",
                representation_requirement="plasma glucose mass",
            )
    targets = PublicTargetContract.model_validate(
        {
            "schema_version": "public-target-contract-2",
            "benchmark_id": cell["benchmark_id"],
            "tier": cell["tier"],
            "public_prompt_sha256": digest,
            "targets": target_rows,
        }
    )
    variables = tuple(
        PublicVariable(name=n, data_role=r)
        for r, names in (
            ("target", context.targets),
            ("auxiliary", context.auxiliaries),
            ("external_input", context.external_inputs),
            ("covariate", context.fixed_covariates),
            ("time", (context.time_symbol,)),
        )
        for n in names
    )
    brief = PublicScientificBrief(
        scientific_context=prompt.partition("F. Required response")[0].strip(),
        public_variables=variables,
        requirements=tuple(
            ScientificRequirement(
                id=r.id,
                public_requirement=r.public_requirement,
                targets=r.required_targets,
                drivers=r.required_drivers,
                requires_dynamic_memory=r.requires_dynamic_memory,
                positive_requirements=(
                    "Represent dynamic memory in the required pathway.",
                )
                if r.requires_dynamic_memory
                else (),
            )
            for r in requirements
        ),
        limits=ModelingLimits.model_validate(cell["limits"]),
    )
    spec = None
    if requirements:
        spec = mechanisms.MechanismEvaluationSpec(
            benchmark_id=cell["benchmark_id"],
            tier=cell["tier"],
            source="public_prompt",
            public_prompt_sha256=digest,
            required_mechanisms=tuple(requirements),
        ).model_dump(mode="json")
    return {
        **cell,
        "prompt": prompt,
        "brief": brief.model_dump(mode="json"),
        "active_prompt": prompt,
        "active_clauses": active,
        "target_contract": targets.model_dump(mode="json"),
        "mechanism_spec": spec,
        "controlled_treatment": arm,
        "fixed_gamma": anchor,
    }


def gamma(bundle: dict) -> dict:
    """Identify dynamic state roles and directly observed identities for a branch."""
    candidate = CandidateModel.model_validate(bundle["candidate"])
    model = compile_candidate(
        candidate, ValidationContext.model_validate(bundle["initialization"]["context"])
    )
    inventory = bundle.get("topology", {}).get("inventory", [])
    return {
        "states": sorted(
            [s.model_dump(mode="json") for s in candidate.states],
            key=lambda s: s["name"],
        ),
        "direct": dict(model.direct_state_observation_channels),
        "roles": sorted(
            [v for v in inventory if v.get("definition") == "differential"],
            key=lambda v: v["name"],
        ),
    }


def certificate(bundle: dict, cell: dict, task: dict) -> dict:
    """Check the actual proposal and keep unresolved scientific evidence explicit."""
    model = CandidateModel.model_validate(bundle["candidate"])
    context = ValidationContext.model_validate(bundle["initialization"]["context"])
    compiled = compile_candidate(model, context)
    observed = compiled.direct_state_observation_channels
    outside = sorted(
        s for s in compiled.state_names if observed.get(s) not in context.targets
    )
    hidden = sorted(set(compiled.state_names) - set(observed))
    t = evaluate_public_targets(
        model, PublicTargetContract.model_validate(cell["target_contract"])
    )
    m = (
        mechanisms.evaluate_mechanisms(
            model,
            mechanisms.MechanismEvaluationSpec.model_validate(cell["mechanism_spec"]),
        )
        if cell["mechanism_spec"]
        else None
    )
    arm = cell["controlled_treatment"]
    verifiable = m is None or all(p.status != "failed" for p in m.mechanism_results)
    verified = m is None or (
        m.mechanism_compliance_complete and m.mechanism_compliance == 1
    )
    checks = {}
    if cell.get("max_parameters") is not None:
        checks["fitted_parameter_budget"] = (
            len(model.parameters) <= cell["max_parameters"]
        )
    if arm in {"no_latent", "observed_only"}:
        checks["observed_target_state_space"] = not outside
    if arm == "no_latent" and any(
        r["requires_dynamic_memory"]
        for r in cell["mechanism_spec"]["required_mechanisms"]
    ):
        checks["required_internal_dynamic_mediator"] = bool(hidden)
    if arm == "overrestrictive":
        checks["one_unobserved_state"] = len(hidden) == 1
    if arm == "fixed_gamma" and cell["fixed_gamma"] is not None:
        current = gamma(bundle)
        checks["unchanged_state_definitions"] = all(
            current[k] == cell["fixed_gamma"][k] for k in ("states", "direct")
        )
        if not bundle.get("lowered_pruning_request") or "topology" in bundle:
            checks["unchanged_state_roles"] = (
                current["roles"] == cell["fixed_gamma"]["roles"]
            )
    if arm == "incorrect":
        graph = mechanisms._dependency_graph(model)
        if family(context) == "cstr":
            checks["no_jacket_feed_path"] = not mechanisms._reaches_any(
                graph, "Tjf", {"target:T"}
            )
        else:
            checks["no_insulin_to_balance_path"] = not mechanisms._reaches_any(
                graph, "I", {"target:Gp"}
            )
    ok = all(checks.values())
    return {
        "scientific_verifier_enabled": task["scientific_verifier"],
        "runtime_valid": True,
        "targets": t.model_dump(mode="json"),
        "mechanisms": m.model_dump(mode="json") if m else {"mechanism_results": []},
        "latent_states": hidden,
        "differential_states_outside_targets": outside,
        "ablation_constraint_pass": ok,
        "controlled_checks": checks,
        "eligible_for_development_selection": t.passed
        and ok
        and (verifiable or arm == "observed_only"),
        "all_public_graph_requirements_certified": bool(verified and ok),
        "scientific_correctness_certified": False,
    }
