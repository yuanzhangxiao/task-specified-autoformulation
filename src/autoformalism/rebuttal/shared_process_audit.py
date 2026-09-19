"""Read-only structural inventory of saved public equations, without refitting.

Repeated syntax is a factoring opportunity, not proof of a conserved transfer.
Parameter-renamed similarity is weaker still. Neither is scientific certification.
"""

from __future__ import annotations

import ast
import copy
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from autoformalism.expressions import (
    ModelValidationError,
    RestrictedParser,
    ValidationContext,
    compile_candidate,
)
from autoformalism.rebuttal.mechanism_audit_sources import BUNDLE_PROTOCOL
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.repair_transactions import signed_tree
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas import CandidateModel
from autoformalism.staged_topology import content_hash

PROTOCOL = "shared-process-inventory-1"


def _terms(node: ast.AST, sign: int = 1) -> Iterator[tuple[int, ast.AST]]:
    """Read outer additive signs without removing internal differences."""
    if isinstance(node, ast.UnaryOp):
        yield from _terms(
            node.operand, -sign if isinstance(node.op, ast.USub) else sign
        )
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        yield from _terms(node.left, sign)
        yield from _terms(node.right, -sign if isinstance(node.op, ast.Sub) else sign)
    else:
        yield sign, node


def _key(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _shape(node: ast.AST, parameters: set[str]) -> str:
    """Rename parameters consistently within one term, preserving repeated uses."""
    aliases = {}

    class Names(ast.NodeTransformer):
        def visit_Name(self, item):
            if item.id in parameters:
                alias = aliases.setdefault(item.id, f"__parameter_{len(aliases)}")
                return ast.Name(id=alias, ctx=ast.Load())
            return item

    return _key(Names().visit(copy.deepcopy(node)))


def _groups(items: list[dict], field: str, *, different_parameters=False) -> list[dict]:
    groups = defaultdict(list)
    for item in items:
        groups[item[field]].append(item)
    result = []
    for key, rows in sorted(groups.items()):
        if len({r["component"] for r in rows}) < 2:
            continue
        if different_parameters and len({r["exact_key"] for r in rows}) < 2:
            continue
        witnesses = [
            {k: v for k, v in row.items() if not k.endswith("_key")} for row in rows
        ]
        result.append({"signature_sha256": content_hash(key), "witnesses": witnesses})
    return result


def inventory(candidate: CandidateModel, context: ValidationContext) -> dict:
    """Compile the same public candidate and record conservative syntax witnesses."""
    compile_candidate(candidate, context)
    definitions = {
        **{f"state:{e.state}": e.rhs for e in candidate.state_equations},
        **{f"process:{p.name}": p.expression for p in candidate.processes},
        **{f"output:{m.channel}": m.expression for m in candidate.observation_mappings},
    }
    process_names = {p.name for p in candidate.processes}
    parameters = {p.name for p in candidate.parameters}
    scientific = {s.name for s in candidate.states} | process_names
    scientific |= set(context.forcing_channels) | {context.time_symbol}
    consumers = defaultdict(list)
    terms, calls, sizes = [], [], {}
    for component, expression in sorted(definitions.items()):
        parsed = RestrictedParser().parse(expression, location=component)
        for name in sorted(parsed.symbols & process_names):
            consumers[name].append(component)
        tree = signed_tree(parsed.tree).body
        additive = list(_terms(tree))
        sizes[component] = {
            "additive_terms": len(additive),
            "ast_nodes": sum(1 for _ in ast.walk(tree)),
        }
        for position, (sign, node) in enumerate(additive):
            symbols = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if not symbols & scientific or isinstance(node, (ast.Name, ast.Constant)):
                continue
            item = {
                "component": component,
                "term": position,
                "outer_sign": sign,
                "expression": ast.unparse(node),
                "parameters": sorted(symbols & parameters),
                "named_processes": sorted(symbols & process_names),
                "exact_key": _key(node),
                "shape_key": _shape(node, parameters),
            }
            terms.append(item)
        # Calls can expose a shared response inside otherwise different terms.
        seen = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            symbols = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if not symbols & scientific or _key(node) in seen:
                continue
            seen.add(_key(node))
            calls.append(
                {
                    "component": component,
                    "expression": ast.unparse(node),
                    "exact_key": _key(node),
                }
            )
    shared = [
        {
            "process": p.name,
            "expression": p.expression,
            "consumers": sorted(consumers[p.name]),
        }
        for p in candidate.processes
        if len(consumers[p.name]) >= 2
    ]
    return {
        "candidate_sha256": content_hash(candidate.model_dump(mode="json")),
        "counts": {
            "dynamic_states": len(candidate.states),
            "algebraic_processes": len(candidate.processes),
            "parameters": len(candidate.parameters),
            "directly_reused_named_processes": len(shared),
            "equation_additive_terms": sum(
                v["additive_terms"]
                for k, v in sizes.items()
                if not k.startswith("output:")
            ),
        },
        "definition_sizes": sizes,
        "existing_shared_processes": shared,
        "exact_repeated_terms": _groups(terms, "exact_key"),
        "exact_repeated_nonlinear_calls": _groups(calls, "exact_key"),
        "similar_terms_review_only": _groups(
            terms, "shape_key", different_parameters=True
        ),
        "physical_transfer_status": "not_inferred_from_syntax",
        "missing_shared_mechanisms": None,
        "scientific_correctness": "not_assessed",
    }


def audit_bundle(bundle_path: Path, output: Path) -> dict:
    """Seal inputs and checkpoint each row; identical reruns reuse checked records."""
    bundle = sealed_read(bundle_path)
    if bundle["protocol"] != BUNDLE_PROTOCOL:
        raise ValueError("expected a public-mechanism-model-bundle-1 snapshot")
    if (
        bundle.get("test_data_opened") is not False
        or bundle.get("private_reference_opened") is not False
    ):
        raise ValueError(
            "public audit requires an explicitly test/private-free snapshot"
        )
    source_sha = bundle["artifact_sha256"]
    # Pin implementation as well as input: changed audit code needs a fresh output.
    code_sha = runtime_source_hash()
    sealed_write(
        output / "plan.json",
        {
            "protocol": PROTOCOL,
            "bundle_sha256": source_sha,
            "implementation_sha256": code_sha,
            "rows": len(bundle["rows"]),
        },
    )
    rows = []
    for index, source in enumerate(bundle["rows"]):
        path = output / "rows" / f"{index:04d}.json"
        identity = {
            "index": index,
            "bundle_sha256": source_sha,
            "source_row_sha256": content_hash(source),
            "implementation_sha256": code_sha,
        }
        if path.exists():
            row = sealed_read(path)
            if any(row.get(k) != v for k, v in identity.items()):
                raise ValueError("audit checkpoint identity differs")
        else:
            row = {
                **identity,
                **{
                    k: source.get(k)
                    for k in (
                        "benchmark_id",
                        "tier",
                        "method",
                        "repetition",
                        "round",
                        "source_id",
                    )
                },
                "status": "unavailable",
            }
            if source.get("status") == "ready":
                if source.get("semantics") != "continuous_time":
                    row.update(
                        status="unsupported",
                        error="continuous-time candidates required",
                    )
                else:
                    try:
                        row["inventory"] = inventory(
                            CandidateModel.model_validate(source["candidate"]),
                            ValidationContext.model_validate(source["context"]),
                        )
                        row["status"] = "complete"
                    except (ValueError, ModelValidationError) as exc:
                        row.update(status="invalid", error=str(exc))
            row = sealed_write(path, row)
        rows.append(row)
    summary = sealed_write(
        output / "summary.json",
        {
            "protocol": PROTOCOL,
            "bundle_sha256": source_sha,
            "implementation_sha256": code_sha,
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "rows": rows,
            "test_data_opened": False,
            "private_reference_opened": False,
            "trajectory_tables_opened": False,
            "llm_calls": 0,
            "optimizer_calls": 0,
            "model_changes": 0,
        },
    )
    # Rebuild the presentation only from immutable, checked numerical-free records.
    lines = [
        "# Shared-process inventory",
        "",
        "Syntax witnesses only; neither physical transfers nor missing "
        "mechanisms are certified.",
        "Counts may overlap. Missing/invalid rows are not zeros. "
        "No fitting, LLM calls or trajectory reads.",
        "",
        "| Benchmark | Method | Seed | Status | States | Named shared | "
        "Exact term groups | Repeated calls | Similar term groups |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        inv = row.get("inventory")
        values = [
            row["benchmark_id"],
            row["method"],
            row["repetition"],
            row["status"],
            inv["counts"]["dynamic_states"] if inv else "—",
            inv["counts"]["directly_reused_named_processes"] if inv else "—",
            len(inv["exact_repeated_terms"]) if inv else "—",
            len(inv["exact_repeated_nonlinear_calls"]) if inv else "—",
            len(inv["similar_terms_review_only"]) if inv else "—",
        ]
        lines.append(
            "| "
            + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in values)
            + " |"
        )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return summary
