"""Public basin bindings and evaluation adapters; never a construction repair."""

from __future__ import annotations

import ast
from pathlib import Path

from autoformalism.data import DatasetSplit
from autoformalism.expressions import RestrictedParser, ValidationContext
from autoformalism.rebuttal import basin_equation_checks as equations
from autoformalism.rebuttal import mechanism_functional as functional
from autoformalism.rebuttal.mechanism_checks import effective_trees
from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.staged_topology import ModelingLimits, PublicScientificBrief
from autoformalism.search.requirement_feedback import _ancestors
from autoformalism.targets import PublicTargetContract

POLICY = "phase-c-basin-finite-predicates-1"
BASINS = {
    f"phase_c_detention_{case}_noise0_v1": case for case in ("coupled", "independent")
}
COUPLED_QUOTES = (
    "Represent accumulation in both physical basins and causal upstream memory.",
    "Represent one internal volumetric transfer: what leaves upstream enters "
    "downstream, with the appropriate area conversion if states are depths.",
    "Represent threshold-dependent overflow and the downstream discharge.",
    "Preserve water balance, nonnegative stored water, and no artificial source "
    "or sink between basins under the stated assumptions.",
)
LOCAL_QUOTE = (
    "Represent downstream accumulation, a threshold-dependent outlet, nonnegative "
    "storage and its local water balance."
)
ISOLATION_QUOTE = (
    "Do not introduce a physical transfer from the disconnected upstream basin."
)


def contracts(
    name: str, prompt: str, digest: str
) -> tuple[PublicTargetContract, MechanismEvaluationSpec, list[dict]]:
    """Bind public quotations to existing target/graph schemas and scoped tests.

    Graph admission tests only paths and upstream dynamic memory. Full physics
    remains in the public context and the separate fitted equation assessment.
    """
    case = BASINS[name]
    quotes = COUPLED_QUOTES if case == "coupled" else (LOCAL_QUOTE, ISOLATION_QUOTE)
    normalized = " ".join(prompt.split())
    if any(q not in normalized for q in quotes):
        raise ValueError("basin requirements differ from the public prompt")
    common = {
        "benchmark_id": name,
        "tier": "development",
        "public_prompt_sha256": digest,
        "source": "public_prompt",
    }
    target = PublicTargetContract.model_validate(
        {
            **common,
            "schema_version": "public-target-contract-2",
            "targets": [
                {
                    "target_channel": "h_down",
                    "public_requirement": "Predict downstream basin depth h_down.",
                }
            ],
        }
    )
    requirements = [
        {
            "id": "local_balance",
            "public_requirement": quotes[3] if case == "coupled" else LOCAL_QUOTE,
            "required_targets": ["h_down"],
            "required_drivers": ["inflow_down"],
        }
    ]
    if case == "coupled":
        requirements.append(
            {
                "id": "upstream_memory",
                "public_requirement": COUPLED_QUOTES[0],
                "required_targets": ["h_down"],
                "required_drivers": ["inflow_up"],
                "requires_dynamic_memory": True,
            }
        )
    mechanism = MechanismEvaluationSpec.model_validate(
        {**common, "required_mechanisms": requirements}
    )
    # Fixed denominators: optional diagnostic witnesses never create extra credit.
    rules = [
        {
            "id": "local_inflow_conversion",
            "public_requirement": quotes[-1] if case == "coupled" else LOCAL_QUOTE,
            "kind": "basin_equation",
            "checks": ["inflow_conversion:h_down"],
        },
        {
            "id": "threshold_outlet",
            "public_requirement": quotes[2] if case == "coupled" else LOCAL_QUOTE,
            "kind": "basin_equation",
            "checks": ["storage_dependent_outlet", "free_outlet_threshold_probes"],
        },
    ]
    if case == "coupled":
        rules.extend(
            [
                {
                    "id": "upstream_memory",
                    "public_requirement": COUPLED_QUOTES[0],
                    "kind": "memory",
                    "driver": "inflow_up",
                    "target": "h_down",
                    "readout": "balance_rate",
                    "expected_sign": 0,
                    "interpretation": (
                        "Effective upstream dynamic path to downstream balance "
                        "with downstream storage held fixed."
                    ),
                },
                {
                    "id": "upstream_inflow_conversion",
                    "public_requirement": COUPLED_QUOTES[3],
                    "kind": "basin_equation",
                    "checks": ["inflow_conversion:h_up"],
                },
                {
                    "id": "transfer_cancellation",
                    "public_requirement": COUPLED_QUOTES[1],
                    "kind": "basin_equation",
                    "checks": ["internal_transfer_cancellation"],
                },
            ]
        )
    else:
        rules.append(
            {
                "id": "no_upstream_coupling",
                "public_requirement": ISOLATION_QUOTE,
                "kind": "basin_isolation",
            }
        )
    return target, mechanism, rules


def brief(
    prompt: str,
    context: ValidationContext,
    mechanism: MechanismEvaluationSpec,
    limits: dict,
) -> PublicScientificBrief:
    """Keep the entire basin prompt, which has no legacy response-section boundary."""
    return PublicScientificBrief(
        scientific_context=prompt.rstrip(),
        public_variables=[
            {"name": name, "data_role": role}
            for role, names in (
                ("target", context.targets),
                ("auxiliary", context.auxiliaries),
                ("external_input", context.external_inputs),
                ("covariate", context.fixed_covariates),
                ("time", (context.time_symbol,)),
            )
            for name in names
        ],
        requirements=[
            {
                "id": r.id,
                "public_requirement": r.public_requirement,
                "targets": r.required_targets,
                "drivers": r.required_drivers,
                "requires_dynamic_memory": r.requires_dynamic_memory,
                "positive_requirements": [
                    "Represent dynamic memory in the required pathway."
                ]
                if r.requires_dynamic_memory
                else [],
                "public_pathway_sign": r.required_sign,
            }
            for r in mechanism.required_mechanisms
        ],
        limits=ModelingLimits.model_validate(limits),
    )


def isolation(candidate: CandidateModel, parameters: dict) -> dict:
    """Absence certifies isolation; a syntactic path alone is unresolved.

    Include initial maps: an autonomous downstream law can still inherit the
    disconnected upstream gauge reading through a hidden state's initializer.
    """
    trees = effective_trees(candidate, parameters)
    dependencies = {
        n: {x.id for x in ast.walk(tree) if isinstance(x, ast.Name)}
        for n, tree in trees.items()
    }
    for initial in candidate.initial_conditions:
        if initial.expression is not None:
            symbols = (
                RestrictedParser()
                .parse(initial.expression, location="basin initialization isolation")
                .symbols
            )
            dependencies.setdefault(initial.state, set()).update(symbols)
    upstream = {"inflow_up", "initial_up"}
    reached = upstream & _ancestors("output:h_down", dependencies)
    return {
        "status": "unresolved" if reached else "pass",
        "reason": "An upstream dependency remains; activity/cancellation is unproved."
        if reached
        else "No upstream runoff or initial-gauge path to output/initialization.",
        "upstream_dependencies": sorted(reached),
    }


def assess(
    row: dict,
    cell: dict,
    train: DatasetSplit,
    settings: functional.Settings,
    directory: Path,
) -> list[dict]:
    """Reuse saved-model physics witnesses and the general fitted memory probe."""
    case = BASINS[cell["public_specification"]["benchmark_id"]]
    candidate = CandidateModel.model_validate(row["candidate"])
    context = ValidationContext.model_validate(row["context"])
    surveys = list(
        {
            tuple(sorted(t.fixed_covariates.items())): dict(t.fixed_covariates)
            for t in train.trajectories
        }.values()
    )
    evidence = equations.assess(candidate, context, case, surveys, row["parameters"])
    by_code = {f["code"]: f for f in evidence["checks"]}
    # The witness uses the actual state name; the public predicate uses its channel.
    down = evidence["coordinate_bindings"]["downstream_depth"]
    conversion = by_code.get(f"inflow_conversion:{down}")
    if conversion is not None:
        by_code["inflow_conversion:h_down"] = conversion
    rules = cell["independent_rules"]
    dynamic = [r for r in rules if r["kind"] == "memory"]
    memory = (
        {
            f["id"]: f
            for f in functional.assess(row, dynamic, train, settings, directory)
        }
        if dynamic
        else {}
    )
    findings = []
    for rule in rules:
        kind = rule["kind"]
        if kind == "memory":
            value = memory[rule["id"]]
        elif kind == "basin_isolation":
            value = isolation(candidate, row["parameters"])
        else:
            selected = [
                by_code.get(
                    code,
                    {
                        "code": code,
                        "status": "unverified",
                        "message": "No supported physical coordinate/check available.",
                    },
                )
                for code in rule["checks"]
            ]
            value = {
                "status": functional.conjunctive(
                    [
                        f["status"] if f["status"] in {"pass", "fail"} else "unresolved"
                        for f in selected
                    ]
                ),
                "checks": selected,
                "coordinate_bindings": evidence["coordinate_bindings"],
                "assumptions": evidence["assumptions"],
            }
        findings.append(
            {
                "id": rule["id"],
                "public_requirement": rule["public_requirement"],
                **value,
                "assessment_policy": POLICY,
                "scope": "finite_public_predicate; not full mechanism certification",
            }
        )
    return findings
