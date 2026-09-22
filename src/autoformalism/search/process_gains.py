"""Compile declared shared laws and consumer gains without domain-specific checks."""

from __future__ import annotations

import ast
from copy import deepcopy

from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.initialization import (
    LatentInitializationPlan,
    apply_initialization_plan,
)
from autoformalism.schemas import CandidateModel
from autoformalism.search.signed_processes import SignedProcess, conversion_value
from autoformalism.staged_topology import content_hash

POLICIES = ("explicit_conversion", "independent_gains", "declared_or_independent")


def _replace(expression, replacements):
    """Replace certified symbol leaves only, never substrings or executable text."""
    RestrictedParser().parse(expression, location="gain assembly")

    class Replace(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in replacements:
                return ast.parse(replacements[node.id], mode="eval").body
            return node

    return ast.unparse(Replace().visit(ast.parse(expression, mode="eval")).body)


def _outer_gain(expression, parameters):
    """Identify a direct coefficient magnitude under the fixed consumer signs."""
    root = ast.parse(expression, mode="eval").body

    def factors(n):
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            return factors(n.operand)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            return factors(n.left) + factors(n.right)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            return factors(n.left)
        return [n]

    for f in factors(root):
        if (
            isinstance(f, ast.Name)
            and f.id in parameters
            and (
                parameters[f.id]["domain"] in {"nonnegative", "positive"}
                or parameters[f.id]["role"] == "coefficient"
            )
            and parameters[f.id]["scope"] == "global"
            and parameters[f.id].get("bounds") is None
            and sum(isinstance(n, ast.Name) and n.id == f.id for n in ast.walk(root))
            == 1
        ):
            return f.id
    return None


def compile_bundle(bundle, contract, policy, training):
    """Compile shared shapes and boundaries using a declared gain policy.

    Only a private-to-the-process direct amplitude is normalized to one; its role
    is replaced by one tied transfer gain or one effective gain per consumer.
    Internal parameters and signed nonlinear laws remain untouched.
    """
    if policy not in POLICIES:
        raise ValueError("unknown process gain policy")
    result = deepcopy(bundle)
    base = deepcopy(bundle["initialization"]["base_candidate"])
    context = ValidationContext.model_validate(bundle["context"])
    aliases = {}
    if policy == "declared_or_independent":
        from autoformalism.schemas.staged_topology import (
            EquationDefinition,
            PublicScientificBrief,
            ScientificVariable,
        )
        from autoformalism.staged_topology import lower_topology

        _, aliases = lower_topology(
            PublicScientificBrief.model_validate(bundle["brief"]),
            tuple(
                ScientificVariable.model_validate(v)
                for v in bundle["topology"]["inventory"]
            ),
            tuple(
                EquationDefinition.model_validate(e)
                for e in bundle["topology"]["equations"]
            ),
            context,
        )
    process_by_name = {p["name"]: p for p in base["processes"]}
    # Legacy policies retain their original symbol treatment for historical runs.
    all_names = (
        {p["name"] for p in base["parameters"]}
        | {p["name"] for p in base["processes"]}
        | {p["name"] for p in base["states"]}
        | set(context.fixed_covariates)
        | set(context.external_inputs)
    )
    if policy == "declared_or_independent":
        all_names |= (
            set(context.forcing_channels) | set(context.targets) | {context.time_symbol}
        )
    parameter_by_name = {p["name"]: p for p in base["parameters"]}
    rows, removed, gain_guesses = [], set(), {}
    bindings = (contract or {}).get("bindings", [])
    for index, b in enumerate(bindings):
        p = SignedProcess.model_validate(b["signed_declaration"])
        if aliases:
            p = p.model_copy(
                update={
                    "name": aliases.get(p.name, p.name),
                    "depends_on": tuple(aliases.get(n, n) for n in p.depends_on),
                    "uses": tuple(
                        u.model_copy(update={"target": aliases.get(u.target, u.target)})
                        for u in p.uses
                    ),
                }
            )
        if p.name not in process_by_name:
            raise ValueError("process alias needs explicit mapping")
        law = process_by_name[p.name]
        original = law["expression"]
        amplitude = _outer_gain(original, parameter_by_name)
        other_expressions = (
            [x["expression"] for x in base["processes"] if x["name"] != p.name]
            + [x["rhs"] for x in base["state_equations"]]
            + [x["expression"] for x in base["observation_mappings"]]
            + [
                x["expression"]
                for x in base["initial_conditions"]
                if x.get("expression")
            ]
        )
        if amplitude and (
            base["constraints"]
            or any(
                amplitude
                in RestrictedParser().parse(e, location="gain identity").symbols
                for e in other_expressions
            )
        ):
            amplitude = None
        if amplitude:
            law["expression"] = _replace(original, {amplitude: "1"})
            removed.add(amplitude)
        effective_policy = policy
        if policy == "declared_or_independent":
            # Never invent unit conversions. A missing conversion means effective
            # per-consumer gains, without claiming a conserved physical transfer.
            complete_conversions = all(u.conversion is not None for u in p.uses)
            effective_policy = (
                "explicit_conversion" if complete_conversions else "independent_gains"
            )
        tied = effective_policy == "explicit_conversion" and p.kind == "transfer"
        uses = []
        for use_index, use in enumerate(p.uses):
            conversion = (
                use.conversion if effective_policy == "explicit_conversion" else "1"
            )
            if conversion is None:
                if p.kind == "transfer":
                    raise ValueError(
                        f"conversion unavailable: {p.name} -> {use.target}"
                    )
                conversion = "1"
            for trajectory in training["rows"]:
                conversion_value(conversion, trajectory["fixed_covariates"])
            gain = f"af_process_gain_{index}_{0 if tied else use_index}"
            guess, start_status = 1.0, "explicit_conversion"
            if effective_policy == "independent_gains":
                start_status = "unmatched_conversion_unavailable"
                try:
                    declared = (
                        [
                            conversion_value(use.conversion, r["fixed_covariates"])
                            for r in training["rows"]
                        ]
                        if use.conversion
                        else []
                    )
                    if declared and len(set(declared)) == 1:
                        guess, start_status = declared[0], "matched_constant_conversion"
                    elif declared:
                        start_status = "unmatched_varying_conversion"
                except (ValueError, ArithmeticError):
                    pass
            gain_guesses[gain] = guess
            if not tied or use_index == 0:
                if gain in all_names:
                    raise ValueError("generated process gain name collision")
                all_names.add(gain)
                base["parameters"].append(
                    {"name": gain, "role": "nonnegative_coefficient", "scope": "global"}
                )
            targets = [x for x in base["state_equations"] if x["state"] == use.target]
            field = "rhs"
            if not targets:
                targets = [x for x in base["processes"] if x["name"] == use.target]
                field = "expression"
            if len(targets) != 1:
                raise ValueError("process consumer has no unique defining equation")
            before = targets[0][field]
            if (
                sum(
                    isinstance(n, ast.Name) and n.id == p.name
                    for n in ast.walk(ast.parse(before, mode="eval"))
                )
                != 1
            ):
                raise ValueError(
                    "process consumer must contain exactly one bound identity"
                )
            targets[0][field] = _replace(
                before, {p.name: f"{gain}*({conversion})*{p.name}"}
            )
            uses.append(
                {
                    "target": use.target,
                    "sign": use.sign,
                    "gain": gain,
                    "conversion": conversion,
                    "declared_conversion": use.conversion,
                    "initial_gain": guess,
                    "starting_point": start_status,
                }
            )
        if policy == "declared_or_independent":
            for row in uses:
                row["assembly_policy"] = effective_policy
        rows.append(
            {
                "process": p.name,
                "kind": p.kind,
                "uses": uses,
                "original_law": original,
                "normalized_law": law["expression"],
                "removed_direct_amplitude": amplitude,
                "magnitude_policy": "nonnegative_under_declared_use_signs",
                "global_scale_redundancy_fully_audited": False,
            }
        )
    base["parameters"] = [p for p in base["parameters"] if p["name"] not in removed]
    candidate = CandidateModel.model_validate(base)
    model = compile_candidate(candidate, context)
    plan = LatentInitializationPlan.model_validate(bundle["initialization"]["plan"])
    compiled, guesses, audit = apply_initialization_plan(model, plan)
    initialization = {
        **bundle["initialization"],
        "identity": content_hash(
            [
                bundle["initialization"]["identity"],
                policy,
                candidate.model_dump(mode="json"),
            ]
        ),
        "base_candidate": candidate.model_dump(mode="json"),
        "candidate": compiled.validated.candidate.model_dump(mode="json"),
        "context": compiled.validated.context.model_dump(mode="json"),
        "guesses": guesses,
        "audit": audit,
    }
    result.update(initialization=initialization, candidate=initialization["candidate"])
    result["fit_parameter_guesses"] = gain_guesses
    result["gain_compilation"] = {
        "policy": policy,
        "source_bundle_sha256": content_hash(bundle),
        "source_contract_sha256": content_hash(contract),
        "processes": rows,
        "candidate_sha256": content_hash(result["candidate"]),
        "initialization_plan_unchanged": True,
        "conservation_certified": False,
    }
    return result
