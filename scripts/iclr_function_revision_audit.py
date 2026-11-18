"""Audit retained equation-family revisions without counting coefficient refits.

Only the frozen, already selected histories are read. Restricted expression ASTs
are converted to bounded sparse polynomials whose atoms are variables or explicit
nonlinear kernels. Scalar coefficients are ignored when comparing families.
This operational signature is not a complete mathematical equivalence prover.
Primary counts keep the state roster fixed. The separate ``any_roster`` fields
also inventory joint state/equation edits; these may move an existing nonlinear
kernel into or out of a memory coordinate, without inventing a new kernel.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
from collections import Counter, defaultdict
from functools import cache
from itertools import pairwise
from pathlib import Path

import sympy as sp

from autoformalism.expressions import RestrictedParser
from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts.collect_iclr_latent_histories import digest
from scripts.iclr_active_history import reachable
from scripts.iclr_latent_revision_audit import arm_name, write_table

Monomial = tuple[tuple[str, int], ...]
Polynomial = dict[Monomial, sp.Expr]
Signature = tuple[Monomial, ...]
MAX_TERMS = 1024


def clean(poly: Polynomial) -> Polynomial:
    """Cancel exact scalar coefficients and bound symbolic work."""
    if len(poly) > MAX_TERMS:
        raise ValueError("polynomial audit term bound exceeded")
    result = {}
    for key, coefficient in poly.items():
        if sp.count_ops(coefficient) > 4096:
            raise ValueError("scalar audit expression bound exceeded")
        value = sp.expand(coefficient)
        if value != 0:
            result[key] = value
    return result


def add(a: Polynomial, b: Polynomial) -> Polynomial:
    """Combine coefficients of identical dynamic kernels."""
    out = dict(a)
    for key, value in b.items():
        out[key] = out.get(key, sp.Integer(0)) + value
    return clean(out)


def multiply(a: Polynomial, b: Polynomial) -> Polynomial:
    """Distribute products with an explicit resource limit."""
    if len(a) * len(b) > MAX_TERMS:
        raise ValueError("polynomial audit product bound exceeded")
    out = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            powers = Counter(dict(ka))
            for atom, power in kb:
                powers[atom] += power
            key = tuple(sorted((n, p) for n, p in powers.items() if p))
            out[key] = out.get(key, sp.Integer(0)) + va * vb
    return clean(out)


def signature(poly: Polynomial) -> Signature:
    """Ignore scalar names, values, signs, and sharing, retaining dynamic kernels."""
    return structural_polynomial(payload(clean(poly)))


def encoded(value: object) -> str:
    """Stable encoding for nested nonlinear atoms, never executable text."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def payload(poly: Polynomial) -> list:
    """Keep exact kernel coefficients during cancellation; serialize as data."""
    return [(key, str(value)) for key, value in sorted(poly.items())]


def structural_polynomial(terms: list) -> Signature:
    """Strip coefficients only after exact cancellation, including nested calls.

    For example, exp(a*x)-exp(b*x) must not cancel merely because its two
    kernels share a functional class. Coefficient strings are never parsed.
    """

    def atom_family(atom: str) -> str:
        if atom.startswith("var:"):
            return atom
        if atom.startswith("inv:"):
            return "inv:" + encoded(structural_polynomial(json.loads(atom[4:])))
        if atom.startswith("call:"):
            _, name, arguments = atom.split(":", 2)
            families = [structural_polynomial(a) for a in json.loads(arguments)]
            if name in {"min", "max"}:
                families.sort()
            return "call:" + name + ":" + encoded(families)
        raise ValueError("unknown dynamic atom")

    normalized = set()
    for term, _coefficient in terms:
        powers = Counter()
        for atom, power in term:
            powers[atom_family(atom)] += power
        normalized.add(tuple(sorted((n, p) for n, p in powers.items() if p)))
    return tuple(sorted(normalized))


def reciprocal(poly: Polynomial) -> Polynomial:
    """Normalize monomial quotients, otherwise retain the denominator family."""
    poly = clean(poly)
    if not poly:
        raise ValueError("zero denominator in archived expression")
    if len(poly) == 1:
        ((key, value),) = poly.items()
        return {tuple((n, -p) for n, p in key): 1 / value}
    return {(("inv:" + encoded(payload(poly)), 1),): sp.Integer(1)}


def call(name: str, arguments: list[Polynomial]) -> Polynomial:
    """Normalize common equivalent spellings without evaluating model code."""
    one = {(): sp.Integer(1)}
    if name == "sigmoid":
        negative = multiply({(): sp.Integer(-1)}, arguments[0])
        return reciprocal(add(one, call("exp", [negative])))
    if name == "softplus":
        return call("log", [add(one, call("exp", arguments))])
    if all(not any(powers for powers in arg) for arg in arguments):
        values = [arg.get((), sp.Integer(0)) for arg in arguments]
        functions = {
            "exp": sp.exp,
            "log": sp.log,
            "sin": sp.sin,
            "tanh": sp.tanh,
            "abs": sp.Abs,
            "sqrt": sp.sqrt,
            "max": sp.Max,
            "min": sp.Min,
        }
        return clean({(): functions[name](*values)})
    shapes = [payload(arg) for arg in arguments]
    if name in {"min", "max"}:
        shapes.sort()
    return {(("call:" + name + ":" + encoded(shapes), 1),): sp.Integer(1)}


class Families:
    """Expand only algebraic processes, keeping dynamic state identities fixed."""

    def __init__(self, model: dict, *, fitted: bool = False):
        self.model = model
        self.candidate = model["candidate"]
        self.parameters = model["parameters"]
        self.fitted = fitted
        self.processes = {
            p["name"]: p["expression"] for p in self.candidate["processes"]
        }
        self.parameter_names = {p["name"] for p in self.candidate["parameters"]}
        if self.parameter_names != set(self.parameters):
            raise ValueError("incomplete fitted coefficient vector")
        ctx = model["context"]
        self.variables = {s["name"] for s in self.candidate["states"]}
        self.variables.update(ctx.get("external_inputs", []))
        self.variables.update(ctx.get("auxiliaries", []))
        self.variables.update(ctx.get("fixed_covariates", []))
        self.variables.add(ctx.get("time_symbol", "t"))
        self.parser = RestrictedParser()
        self.expanded = {}

    def parse(self, source: str, pending: frozenset[str] = frozenset()) -> Polynomial:
        """Parse through the project's whitelist before traversing any AST."""
        node = self.parser.parse(source, location="function-family audit").tree
        return self.convert(node, pending)

    def convert(self, node: ast.AST, pending: frozenset[str]) -> Polynomial:
        """Construct scalar algebra explicitly; do not sympify generated strings."""
        if isinstance(node, ast.Expression):
            return self.convert(node.body, pending)
        if isinstance(node, ast.Constant):
            return clean({(): sp.Rational(str(node.value))})
        if isinstance(node, ast.Name):
            name = node.id
            if name in self.parameter_names:
                value = (
                    sp.Rational(str(self.parameters[name]))
                    if self.fitted
                    else sp.Symbol(name, real=True)
                )
                return clean({(): value})
            if name in self.variables:
                return {(("var:" + name, 1),): sp.Integer(1)}
            if name in self.processes:
                if name in pending:
                    raise ValueError("cyclic algebraic process")
                if name not in self.expanded:
                    self.expanded[name] = self.parse(
                        self.processes[name], pending | {name}
                    )
                return self.expanded[name]
            raise ValueError("unbound archived symbol: " + name)
        if isinstance(node, ast.UnaryOp):
            value = self.convert(node.operand, pending)
            return (
                multiply({(): sp.Integer(-1)}, value)
                if isinstance(node.op, ast.USub)
                else value
            )
        if isinstance(node, ast.Call):
            return call(node.func.id, [self.convert(a, pending) for a in node.args])
        if isinstance(node, ast.BinOp):
            a, b = self.convert(node.left, pending), self.convert(node.right, pending)
            if isinstance(node.op, ast.Add):
                return add(a, b)
            if isinstance(node.op, ast.Sub):
                return add(a, multiply({(): sp.Integer(-1)}, b))
            if isinstance(node.op, ast.Mult):
                return multiply(a, b)
            if isinstance(node.op, ast.Div):
                return multiply(a, reciprocal(b))
            if isinstance(node.op, ast.Pow):
                exponent = int(b.get((), 0))
                result = {(): sp.Integer(1)}
                for _ in range(abs(exponent)):
                    result = multiply(result, a)
                return reciprocal(result) if exponent < 0 else result
        raise ValueError("unsupported validated AST")

    def equations(self) -> dict[str, Signature]:
        """Compare expanded differential equations and public algebraic readouts."""
        expressions = {e["state"]: e["rhs"] for e in self.candidate["state_equations"]}
        expressions.update(
            {
                "output:" + o["channel"]: o["expression"]
                for o in self.candidate["observation_mappings"]
            }
        )
        return {lhs: signature(self.parse(rhs)) for lhs, rhs in expressions.items()}


def nonlinear(term: Monomial) -> bool:
    """Detect nonlinear kernels, interactions, powers, and variable denominators."""
    return bool(term) and (
        len(term) > 1
        or any(not name.startswith("var:") or power != 1 for name, power in term)
    )


def shape(term: Monomial) -> str:
    """Separate nonlinear law classes from swapping which variables interact.

    Variable names are ignored, but powers, products and nonlinear operators
    remain. Thus adding a second bilinear interaction is a dependency revision;
    replacing linear decay with quadratic decay is a functional-class revision.
    """

    def atom_shape(atom: str) -> str:
        if atom.startswith("var:"):
            return "variable"
        if atom.startswith("inv:"):
            return "inv:" + encoded(
                sorted({shape(tuple(map(tuple, t))) for t in json.loads(atom[4:])})
            )
        if atom.startswith("call:"):
            _, name, payload = atom.split(":", 2)
            return (
                "call:"
                + name
                + ":"
                + encoded(
                    [
                        sorted({shape(tuple(map(tuple, t))) for t in arg})
                        for arg in json.loads(payload)
                    ]
                )
            )
        raise ValueError("unknown structural atom")

    return encoded(sorted((atom_shape(atom), power) for atom, power in term))


def roster(model: dict) -> set[tuple[str, str]]:
    """Keep state names and kinds fixed for this separate equation audit."""
    return {(s["name"], s["kind"]) for s in model["candidate"]["states"]}


def audit_campaign(name: str, campaign: dict) -> tuple[list, list, list]:
    """Count model-level revisions and preserve every before/after witness."""
    models = campaign["models"]

    @cache
    def families(uid: str, fitted: bool = False) -> dict:
        return Families(models[uid], fitted=fitted).equations()

    @cache
    def active(uid: str) -> set[str]:
        m = models[uid]
        return set(
            reachable({**m, "lowered_context": m["context"]})["active_states"]
        ) | {"output:" + t for t in m["context"]["targets"]}

    runs, events, errors = [], [], []
    for run in campaign["runs"]:
        task = run["task"]
        base = {
            "run_id": campaign["plan_sha256"] + ":" + task["task_id"],
            "cohort": name,
            "proposer": campaign["model_settings"]["model"],
            "run_class": "guided_alternative"
            if name.startswith("guided")
            else "search",
            "benchmark": task["cell"],
            "seed": task["seed"],
            "arm": arm_name(task),
            "final_status": run["final_status"],
        }
        visits = sorted(run["visits"], key=lambda v: v["round"])
        transitions = []
        history_errors = []
        for i, (before, after) in enumerate(pairwise(visits)):
            p, c = before.get("selected_model"), after.get("selected_model")
            if not p or not c or after["round"] != before["round"] + 1:
                continue
            fixed_state_roster = roster(models[p]) == roster(models[c])
            accepted = [a for a in after.get("attempts", []) if a.get("accepted")]
            if not accepted and "guided_feedback" not in after:
                continue
            try:
                a, b = families(p), families(c)
                for lhs in sorted(a.keys() & b.keys() & active(c)):
                    if not lhs.startswith("output:") and dict(roster(models[p])).get(
                        lhs
                    ) != dict(roster(models[c])).get(lhs):
                        continue
                    if a[lhs] == b[lhs]:
                        continue
                    hypothesis = (
                        (accepted[-1].get("raw") or {}).get("hypothesis")
                        if accepted
                        else after["guided_feedback"].get("hypothesis")
                    )
                    transitions.append(
                        {
                            "round": after["round"],
                            "visit": after["round"] + 1,
                            "fixed_state_roster": fixed_state_roster,
                            "lhs": lhs,
                            "parent_model": p,
                            "child_model": c,
                            "before_family": a[lhs],
                            "after_family": b[lhs],
                            "new_nonlinear_terms": sorted(
                                t for t in set(b[lhs]) - set(a[lhs]) if nonlinear(t)
                            ),
                            "removed_nonlinear_terms": sorted(
                                t for t in set(a[lhs]) - set(b[lhs]) if nonlinear(t)
                            ),
                            "hypothesis": hypothesis,
                            "request_hash": accepted[-1].get("request_hash")
                            if accepted
                            else None,
                            "reply_evidence_refs": (accepted[-1].get("raw") or {}).get(
                                "evidence_refs", []
                            )
                            if accepted
                            else after["guided_feedback"].get(
                                "reply_evidence_refs", []
                            ),
                            "coordinate_survives": all(
                                v.get("selected_model")
                                and lhs in families(v["selected_model"])
                                for v in visits[i + 2 :]
                            ),
                        }
                    )
            except (ValueError, KeyError, TypeError) as exc:
                error = {**base, "round": after["round"], "error": str(exc)}
                errors.append(error)
                history_errors.append(error)
        for snapshot, uid in [
            ("final", run.get("final_model")),
            ("latest_available", run.get("final_model") or run.get("latest_model")),
        ]:
            row = {
                **base,
                "snapshot": snapshot,
                "available": uid is not None,
                "model_id": uid,
                "history_audit_errors": len(history_errors),
            }
            if uid is None:
                runs.append(row)
                continue
            try:
                final, final_active = families(uid, True), active(uid)
                own = []
                for transition in transitions:
                    lhs = transition["lhs"]
                    end = final.get(lhs, ())
                    entire = bool(end) and end == transition["after_family"]
                    newly_nonlinear = set(transition["new_nonlinear_terms"]) & set(end)
                    before_shapes = {
                        shape(t) for t in transition["before_family"] if nonlinear(t)
                    }
                    after_shapes = {
                        shape(t) for t in transition["after_family"] if nonlinear(t)
                    }
                    final_shapes = {shape(t) for t in end if nonlinear(t)}
                    new_shapes_retained = {
                        shape(t)
                        for t in newly_nonlinear
                        if shape(t) not in before_shapes
                    }
                    removed_shapes_retained = (
                        before_shapes - after_shapes - final_shapes if entire else set()
                    )
                    retained = lhs in final_active and transition["coordinate_survives"]
                    event = {
                        **base,
                        "snapshot": snapshot,
                        **transition,
                        "final_model": uid,
                        "final_family": end,
                        "entire_revised_family_retained": bool(retained and entire),
                        "new_nonlinear_terms_retained": sorted(newly_nonlinear)
                        if retained
                        else [],
                        "nonlinear_revision_retained": bool(
                            retained
                            and (
                                newly_nonlinear
                                or (entire and transition["removed_nonlinear_terms"])
                            )
                        ),
                        "new_nonlinear_shapes_retained": sorted(new_shapes_retained)
                        if retained
                        else [],
                        "removed_nonlinear_shapes_retained": sorted(
                            removed_shapes_retained
                        )
                        if retained
                        else [],
                        "nonlinear_shape_revision_retained": bool(
                            retained
                            and (new_shapes_retained or removed_shapes_retained)
                        ),
                    }
                    events.append(event)
                    own.append(event)
                row["selected_equation_revision_any_roster_count"] = len(
                    {e["round"] for e in own}
                )
                row["retained_equation_revision_any_roster_count"] = len(
                    {
                        e["round"]
                        for e in own
                        if e["entire_revised_family_retained"]
                        or e["nonlinear_revision_retained"]
                    }
                )
                row["retained_shape_revision_any_roster_count"] = len(
                    {e["round"] for e in own if e["nonlinear_shape_revision_retained"]}
                )
                own = [e for e in own if e["fixed_state_roster"]]
                row["selected_fixed_state_revision_count"] = len(
                    {e["round"] for e in own}
                )
                row["retained_equation_revision_count"] = len(
                    {
                        e["round"]
                        for e in own
                        if e["entire_revised_family_retained"]
                        or e["nonlinear_revision_retained"]
                    }
                )
                row["retained_nonlinear_revision_count"] = len(
                    {e["round"] for e in own if e["nonlinear_revision_retained"]}
                )
                row["retained_new_nonlinearity_count"] = len(
                    {e["round"] for e in own if e["new_nonlinear_terms_retained"]}
                )
                row["retained_shape_revision_count"] = len(
                    {e["round"] for e in own if e["nonlinear_shape_revision_retained"]}
                )
            except (ValueError, KeyError, TypeError) as exc:
                row["audit_error"] = str(exc)
                errors.append({**base, "snapshot": snapshot, "error": str(exc)})
            runs.append(row)
    return runs, events, errors


def summarize(rows: list[dict]) -> dict:
    """Give both all-assignment and available-model denominators."""
    available = [r for r in rows if r["available"]]
    ready = [
        r
        for r in available
        if not r.get("audit_error") and not r.get("history_audit_errors")
    ]
    result = {
        "assigned": len(rows),
        "available": len(available),
        "assessed": len(ready),
    }
    for field in [
        "selected_fixed_state_revision_count",
        "retained_equation_revision_count",
        "retained_nonlinear_revision_count",
        "retained_new_nonlinearity_count",
        "retained_shape_revision_count",
        "selected_equation_revision_any_roster_count",
        "retained_equation_revision_any_roster_count",
        "retained_shape_revision_any_roster_count",
    ]:
        counts = Counter(r[field] for r in ready)
        any_count = sum(v for k, v in counts.items() if k)
        result[field] = {
            "histogram": dict(sorted(counts.items())),
            "models_with_any": any_count,
            "percent_of_assessed": 100 * any_count / len(ready) if ready else None,
            "percent_of_assignments": 100 * any_count / len(rows) if rows else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs, events, errors, sources = [], [], [], []
    plans = set()
    for path in sorted(args.sources.glob("*.json.gz")):
        d = json.loads(gzip.decompress(path.read_bytes()))
        if d["artifact_sha256"] != digest(
            {k: v for k, v in d.items() if k != "artifact_sha256"}
        ):
            raise ValueError("source hash mismatch")
        if d["plan_sha256"] in plans:
            raise ValueError("duplicate campaign")
        plans.add(d["plan_sha256"])
        name = path.name.removesuffix(".json.gz")
        r, e, failures = audit_campaign(name, d)
        runs.extend(r)
        events.extend(e)
        errors.extend(failures)
        sources.append({"cohort": name, "source_sha256": d["artifact_sha256"]})
        print(name, len(r) // 2, "runs;", len(failures), "audit errors", flush=True)
    summaries = {}
    for keys in [
        ("snapshot", "run_class", "proposer"),
        ("snapshot", "run_class", "proposer", "arm"),
        ("snapshot", "cohort", "benchmark", "arm"),
    ]:
        groups = defaultdict(list)
        for row in runs:
            groups[tuple(row[k] for k in keys)].append(row)
        summaries["/".join(keys)] = [
            {**dict(zip(keys, key, strict=True)), **summarize(rows)}
            for key, rows in sorted(groups.items())
        ]
    sealed_write(
        args.output / "audit.json",
        json.loads(
            json.dumps(
                {"sources": sources, "runs": runs, "events": events, "errors": errors}
            )
        ),
    )
    sealed_write(args.output / "summary.json", json.loads(json.dumps(summaries)))
    write_table(args.output / "runs.csv", runs)
    write_table(args.output / "events.csv", events)


if __name__ == "__main__":
    main()
