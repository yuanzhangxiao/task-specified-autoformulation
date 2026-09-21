"""Advisory equation facts for the public basin task, independent of prose labels."""

from __future__ import annotations

import ast
import copy
from collections import defaultdict

from autoformalism.expressions import RestrictedParser
from autoformalism.rebuttal.process_gain_comparison import conservation_diagnostic


def _tree(expression: str) -> ast.AST:
    return RestrictedParser().parse(
        expression, location="process equation diagnostic"
    ).tree.body


def _symbols(node: ast.AST) -> set[str]:
    functions = {
        n.func.id
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} - functions


def _expanded(node: ast.AST, processes: dict[str, ast.AST]) -> ast.AST:
    """Expand declared algebraic dependencies with a finite, shared node budget."""
    remaining = 20000

    def visit(value, path):
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            raise ValueError("diagnostic expansion budget exceeded")
        if isinstance(value, ast.Name) and value.id in processes:
            if value.id in path:
                raise ValueError("algebraic cycle in diagnostic")
            return visit(processes[value.id], path | {value.id})
        value = copy.copy(value)
        for field, child in ast.iter_fields(value):
            if isinstance(child, ast.AST):
                setattr(value, field, visit(child, path))
            elif isinstance(child, list):
                setattr(
                    value,
                    field,
                    [visit(c, path) if isinstance(c, ast.AST) else c for c in child],
                )
        return value

    return visit(node, set())


def _degree(node: ast.AST, dynamic: set[str]) -> int | None:
    """Sufficient affine certificate at fixed parameter/covariate values."""
    if not _symbols(node) & dynamic:
        return 0
    if isinstance(node, ast.Name):
        return 1
    if isinstance(node, ast.UnaryOp):
        return _degree(node.operand, dynamic)
    if isinstance(node, ast.BinOp):
        left, right = _degree(node.left, dynamic), _degree(node.right, dynamic)
        if left is None or right is None:
            return None
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return max(left, right)
        if isinstance(node.op, ast.Mult):
            return left + right
        if isinstance(node.op, ast.Div) and right == 0:
            return left
    return None


def _canonical(node: ast.AST) -> str:
    """Remove only identity multiplications; no scientific factor merging."""

    class Identities(ast.NodeTransformer):
        def visit_BinOp(self, value):
            value = self.generic_visit(value)
            if isinstance(value.op, ast.Mult):
                for left, right in (
                    (value.left, value.right),
                    (value.right, value.left),
                ):
                    if isinstance(left, ast.Constant) and left.value == 1:
                        return right
            return value

    return ast.dump(Identities().visit(copy.deepcopy(node)), include_attributes=False)


def diagnose(bundle: dict, parameters: dict | None, cell: dict) -> dict:
    """Expose facts, leaving interpretation and corrective edits to the proposer.

    This is explicitly scoped to the public basin coordinate vocabulary. Other
    latent coordinate meanings are not inferred from their names or descriptions.
    No reference laws, trajectories, fitting or LLM review enter these checks.
    """
    candidate = bundle["candidate"]
    context = bundle["context"]
    states = {s["name"] for s in candidate["states"]}
    dynamic = (
        states
        | set(context.get("external_inputs", []))
        | {context.get("time_symbol", "t")}
    )
    processes = {p["name"]: _tree(p["expression"]) for p in candidate["processes"]}
    equations = {e["state"]: _tree(e["rhs"]) for e in candidate["state_equations"]}
    findings, traits, groups = [], [], defaultdict(list)
    for name, node in processes.items():
        groups[_canonical(node)].append(name)
    duplicates = []
    for names in groups.values():
        if len(names) < 2:
            continue
        consumers = [
            {"state": state, "processes": sorted(_symbols(rhs) & set(names))}
            for state, rhs in equations.items()
            if len(_symbols(rhs) & set(names)) >= 2
        ]
        duplicates.append(
            {
                "processes": names,
                "common_consumers": consumers,
                "scope": "identical_law_syntax_after_identity_factors",
                "parameter_nonidentifiability_proved": False,
            }
        )
    for kind, definitions in (("process", processes), ("state", equations)):
        for name, node in definitions.items():
            try:
                expanded = _expanded(node, processes)
                symbols = _symbols(expanded)
                degree = _degree(expanded, dynamic)
                operators = sorted(
                    {
                        n.func.id
                        for n in ast.walk(expanded)
                        if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name)
                        and n.func.id in {"max", "min", "abs"}
                    }
                )
                trait = {
                    "kind": kind,
                    "name": name,
                    "expression": ast.unparse(node),
                    "resolved_expression": ast.unparse(expanded),
                    "dependencies": sorted(symbols),
                    "threshold_operators": operators,
                    "affine_in_dynamic_channels": degree is not None and degree <= 1,
                    "threshold_behavior_certified": False,
                }
                traits.append(trait)
                depths = symbols & states & {"h_up", "h_down"}
                if depths and trait["affine_in_dynamic_channels"]:
                    findings.append(
                        {
                            "code": "AFFINE_DEPTH_LAW",
                            "component": name,
                            "kind": kind,
                            "depths": sorted(depths),
                            "evidence": ast.unparse(expanded),
                            "meaning": (
                                "Affine in dynamic channels; no nontrivial threshold "
                                "switch in this expression. Relevance to a required "
                                "outlet needs scientific review."
                            ),
                        }
                    )
                if kind == "state" and "initial_up" in symbols:
                    findings.append(
                        {
                            "code": "INITIAL_COVARIATE_IN_ONGOING_RHS",
                            "component": name,
                            "evidence": ast.unparse(expanded),
                            "meaning": (
                                "The public initial gauge reading appears in an "
                                "ongoing rate. This is not solely an initial "
                                "assignment; it may parameterize the dynamics and "
                                "needs review, not automatic rejection."
                            ),
                        }
                    )
            except (ValueError, RecursionError) as exc:
                traits.append(
                    {
                        "kind": kind,
                        "name": name,
                        "status": "unavailable",
                        "error": str(exc),
                    }
                )
    # Boundary covariates are explicitly identified by this public task contract,
    # never inferred by searching arbitrary model descriptions for the word initial.
    cancellation = conservation_diagnostic(bundle, parameters, cell["training"])
    return {
        "protocol": "basin-equation-facts-1",
        "advisory_only": True,
        "hard_rejection": False,
        "scientific_compliance_certified": False,
        "duplicate_process_laws": duplicates,
        "expression_traits": traits,
        "findings": findings,
        "transfer_cancellation": cancellation,
        "initial_conditions": candidate["initial_conditions"],
        "limitations": [
            "Syntax duplicates do not prove full parameter nonidentifiability.",
            "Max/min/abs syntax does not prove the correct threshold or direction.",
            "Cancellation covers declared depth-coordinate terms, not total balance.",
            "Latent coordinate meanings and physical roles are not inferred "
            "from prose.",
        ],
    }
