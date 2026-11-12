"""Score the mechanisms frozen reactor models declare, under frozen interventions.

Each model's own temperature equation decides which of its quantities is which
mechanism, by the rule frozen in ``configs/cstr_declared_mechanisms_v1.json``:
its right-hand side is split into additive terms, each term is assigned to feed
transport, reaction heat or jacket heat exchange by the inputs and states it
contains, and its latent states stand for the reactant concentration or the
jacket temperature by the inputs that drive them. The quantities are then
compared with the simulator's, response by response, with the study's own
metrics, on a scored intervention root whose reference responses already exist.

Stages:

* ``attribute``: apply the rule to every frozen model of the root and seal the
  assignment. Opens no reference response.
* ``score``: simulate every model on the root's schedules, read its terms and
  states along each trajectory, check that its terms add up to its full
  right-hand side, and score each quantity's response.
* ``report``: per method and quantity, the median response RSE and directions.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import statistics
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.optimize import lsq_linear

from autoformalism.baselines.d3_rollout import predict as native_rollout
from autoformalism.expressions import (
    ValidationContext,
    baseline_validator,
    compile_candidate,
)
from autoformalism.expressions.diagnostics import (
    ModelValidationError,
    RuntimeExpressionError,
)
from autoformalism.expressions.parser import grammar_parser
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting.simulation import trajectory_forcing
from autoformalism.rebuttal.basin_algebra import expanded
from autoformalism.rebuttal.mechanism_probes import _native
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import CandidateModel
from scripts import iclr_anchor_fidelity as anchor

PROTOCOL = "declared-mechanism-scores-1"
RULE = (
    Path(__file__).resolve().parents[1] / "configs" / "cstr_declared_mechanisms_v1.json"
)
TARGET = "T"
MECHANISMS = ("feed_transport", "reaction_heat", "jacket_heat")
STATES = ("C", "Tj")
#: Latent-state kinds, and the quantity each stands for.
KIND_OF_STATE = {"concentration": "C", "jacket": "Tj"}
MAX_TERMS = 64
RTOL, ATOL, SECONDS = 1e-8, 1e-10, 45.0


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- the rule -----------------------------------------------------------------


def split_terms(node: ast.AST, classify=None) -> list[tuple[int, ast.AST, ast.AST]]:
    """Additive terms as (sign, term, own content).

    Sums and differences split. A factor or divisor common to a sum (a heat
    capacity, say) is distributed over the sum's terms only when those terms
    belong to different mechanisms by ``classify``, so a difference such as
    ``(Tf - T)`` inside a term stays whole. A term's own content excludes the
    distributed factor, which therefore never decides its mechanism.
    """

    def spans(parts):
        if classify is None:
            return True
        return len({classify(own) for _, _, own in parts} - {None}) >= 2

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub):
        right = split_terms(node.right, classify)
        if isinstance(node.op, ast.Sub):
            right = [(-sign, term, own) for sign, term, own in right]
        terms = split_terms(node.left, classify) + right
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        terms = [
            (-sign, term, own)
            for sign, term, own in split_terms(node.operand, classify)
        ]
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        terms = split_terms(node.operand, classify)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left, right = (
            split_terms(node.left, classify),
            split_terms(node.right, classify),
        )
        split_left = len(left) > 1 and spans(left)
        split_right = len(right) > 1 and spans(right)
        if not split_left and not split_right:
            return [(1, node, node)]
        left = left if split_left else [(1, node.left, None)]
        right = right if split_right else [(1, node.right, None)]
        terms = []
        for a, x, x_own in left:
            for b, y, y_own in right:
                own = (
                    ast.BinOp(x_own, ast.Mult(), y_own)
                    if x_own is not None and y_own is not None
                    else (x_own if x_own is not None else y_own)
                )
                terms.append((a * b, ast.BinOp(x, ast.Mult(), y), own))
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        numerator = split_terms(node.left, classify)
        if len(numerator) == 1 or not spans(numerator):
            return [(1, node, node)]
        terms = [
            (a, ast.BinOp(x, ast.Div(), node.right), own) for a, x, own in numerator
        ]
    else:
        return [(1, node, node)]
    return terms if len(terms) <= MAX_TERMS else [(1, node, node)]


def names_in(node: ast.AST) -> set[str]:
    """Variable names in an expression, not the names of the functions it calls."""
    calls = {
        id(item.func)
        for item in ast.walk(node)
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
    }
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and id(item) not in calls
    }


def _kind(names: set[str], kinds: dict[str, str | None]) -> str | None:
    if "Tjf" in names or any(kinds.get(n) == "jacket" for n in names):
        return "jacket"
    if "Tf" in names or any(kinds.get(n) == "feed" for n in names):
        return "feed"
    if "Cf" in names or any(kinds.get(n) == "concentration" for n in names):
        return "concentration"
    return None


def attribute(view: dict) -> dict:
    """Apply the frozen rule to one model's temperature equation and states."""
    candidate = CandidateModel.model_validate(view["candidate"])
    grammar = view["grammar"]
    parser = grammar_parser(grammar)
    processes = {p.name: p.expression for p in candidate.processes}
    equations = {e.state: e.rhs for e in candidate.state_equations}
    if TARGET not in equations:
        raise ValueError("the model has no temperature equation")
    context = ValidationContext.model_validate(view["context"])
    observed = {*context.targets, *context.auxiliaries}
    kinds: dict[str, str | None] = {}
    for state in candidate.states:
        if state.name in observed or state.name not in equations:
            continue
        inlined = expanded(equations[state.name], processes, grammar)
        kinds[state.name] = _kind(names_in(inlined) & {"Cf", "Tf", "Tjf"}, {})
    tree = parser.parse(equations[TARGET], location="temperature equation").tree.body

    def classify(node):
        return _kind(names_in(expanded(ast.unparse(node), processes, grammar)), kinds)

    terms = []
    for sign, node, own in split_terms(tree, classify):
        text = ast.unparse(node)
        written = names_in(own)
        inlined = names_in(expanded(ast.unparse(own), processes, grammar))
        kind = _kind(inlined, kinds)
        reaction_named = any(
            name in processes and ("react" in name.lower() or "rxn" in name.lower())
            for name in written
        )
        if kind == "jacket":
            mechanism = "jacket_heat"
        elif kind == "feed":
            mechanism = "feed_transport"
        elif kind == "concentration" or reaction_named:
            mechanism = "reaction_heat"
        else:
            mechanism = "unattributed"
        terms.append({"sign": sign, "expression": text, "mechanism": mechanism})
    hidden = {}
    for kind, quantity in KIND_OF_STATE.items():
        chosen = [s.name for s in candidate.states if kinds.get(s.name) == kind]
        hidden[quantity] = chosen[0] if chosen else None
    return {
        "state_kinds": kinds,
        "terms": terms,
        "represented": {m: any(t["mechanism"] == m for t in terms) for m in MECHANISMS},
        "hidden": hidden,
    }


# --- the models ---------------------------------------------------------------


def view_of(row: dict) -> dict | None:
    """A frozen model as the rule and the rollout read it, or None without equations."""
    if row["status"] != "ready":
        return None
    if "model" in row:  # an external baseline, read with its own grammar
        model = row["model"]
        if model["semantics"] == "latent_neural_ode":
            return None
        return {
            "candidate": model["candidate"],
            "context": model["context"],
            "parameters": model["parameters"],
            "initials": model["initials"],
            "grammar": "baseline",
            "semantics": model["semantics"],
            "increment_step": model.get("increment_step"),
            "native": model if model["semantics"] == "native_increment" else None,
        }
    return {
        "candidate": row["candidate"],
        "context": row["context"],
        "parameters": row["parameters"],
        "initials": row["initials"],
        "grammar": "default",
        "semantics": "continuous_time",
        "increment_step": None,
        "native": None,
    }


def curves(view: dict, assignment: dict, tr) -> dict:
    """The model's temperature, terms and declared states along one trajectory."""
    payload = view["candidate"]
    if view["semantics"] == "native_increment":
        # D3's one-step map uses its fitted values as they are: declared
        # parameter ranges are audited, never imposed, so its terms are too.
        payload = {
            **payload,
            "parameters": [
                # The role only implies a domain; "coefficient" implies none.
                {**p, "role": "coefficient", "domain": "real", "bounds": None}
                for p in payload["parameters"]
            ],
        }
    candidate = CandidateModel.model_validate(payload)
    context = ValidationContext.model_validate(view["context"])
    baseline = view["grammar"] == "baseline"
    compiled = compile_candidate(
        candidate, context, validator=baseline_validator() if baseline else None
    )
    time = np.asarray(tr.time, dtype=float)
    if view["semantics"] == "native_increment":
        native = _native(view["native"])
        values = native_rollout(native, tr, teacher_forced=False, seconds=SECONDS)
        order = [native.states.index(n) for n in compiled.state_names]
        states = values[:, order].T
        per_time = 1.0 / float(view["increment_step"])
    else:
        sim = simulate_trajectory(
            compiled,
            tr,
            view["parameters"],
            view["initials"],
            FitConfig(
                integration_method="Radau",
                relative_tolerance=RTOL,
                absolute_tolerance=ATOL,
                allow_derivative_regression=False,
            ),
            reset_observed_states=False,
            deadline=monotonic() + SECONDS,
        )
        if not sim.success:
            raise ValueError(sim.message)
        states = np.asarray(sim.states)
        per_time = 1.0
    parser = grammar_parser(view["grammar"])
    forcing = trajectory_forcing(compiled, tr)
    rhs = parser.parse(
        {e.state: e.rhs for e in candidate.state_equations}[TARGET], location="rhs"
    )
    parsed = [
        parser.parse(t["expression"], location="term") for t in assignment["terms"]
    ]
    out = {m: np.zeros(len(time)) for m in (*MECHANISMS, "unattributed")}
    total = np.zeros(len(time))
    for k, t in enumerate(time):
        state = states[:, k]
        total[k] = compiled.evaluate_expression(
            rhs, t, state, view["parameters"], forcing
        )
        for term, expression in zip(assignment["terms"], parsed, strict=True):
            value = term["sign"] * compiled.evaluate_expression(
                expression, t, state, view["parameters"], forcing
            )
            out[term["mechanism"]][k] += value
    summed = sum(out.values())
    if np.max(np.abs(summed - total)) > 1e-7 * max(1.0, float(np.max(np.abs(total)))):
        raise ValueError("the assigned terms do not add up to the right-hand side")
    names = list(compiled.state_names)
    result = {
        TARGET: states[names.index(TARGET)].tolist(),
        **{m: (out[m] * per_time).tolist() for m in (*MECHANISMS, "unattributed")},
    }
    for quantity, state in assignment["hidden"].items():
        if state is not None:
            result["state." + quantity] = states[names.index(state)].tolist()
    return result


# --- stages -------------------------------------------------------------------


def subjects(root: Path) -> list[dict]:
    plan = sealed_read(root / "plan.json")
    return [*plan["rows"], *plan.get("study_rows", [])]


def attribute_stage(root: Path, out: Path) -> dict:
    """Seal the rule's assignment for every frozen model of the root."""
    plan = sealed_read(root / "plan.json")
    rule = json.loads(RULE.read_text())
    if rule["status"] != "frozen_before_scoring" or rule["cell"] not in plan["cells"]:
        raise ValueError("the rule is not frozen for this root's cell")
    rows = []
    for row in subjects(root):
        view = view_of(row)
        entry = {
            "source_id": row["source_id"],
            "method": row["method"],
            "repetition": row["repetition"],
        }
        if view is None:
            entry["status"] = (
                "no_explicit_equation" if row["status"] == "ready" else row["status"]
            )
        else:
            entry |= {"status": "declared", **attribute(view)}
        rows.append(entry)
    out.mkdir(parents=True, exist_ok=True)
    return sealed_write(
        out / "attribution.json",
        {
            "protocol": PROTOCOL,
            "rule_sha256": _digest(RULE),
            "root_plan_sha256": plan["artifact_sha256"],
            "rows": rows,
            "reference_opened": False,
        },
    )


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """An affine map with a nonnegative scale, fitted on calibration samples."""
    if np.std(x) <= 1e-12 * max(1.0, float(np.max(np.abs(x)))):
        return None
    design = np.column_stack((x, np.ones_like(x)))
    fitted = lsq_linear(design, y, bounds=([0.0, -np.inf], [np.inf, np.inf]))
    return float(fitted.x[0]), float(fitted.x[1])


def _truth(record: dict) -> dict:
    return {
        "feed_transport": np.asarray(record["Tf"]) - np.asarray(record["T"]),
        "reaction_heat": np.asarray(record["reaction_heat"]),
        "jacket_heat": np.asarray(record["jacket_heat"]),
        "C": np.asarray(record["C"]),
        "Tj": np.asarray(record["Tj"]),
    }


def score_stage(root: Path, out: Path) -> dict:
    """Simulate every declared model and score each mechanism's responses."""
    plan = sealed_read(root / "plan.json")
    assignment = sealed_read(out / "attribution.json")
    if assignment["root_plan_sha256"] != plan["artifact_sha256"] or assignment[
        "rule_sha256"
    ] != _digest(RULE):
        raise ValueError("the assignment belongs to another root or rule")
    cell = plan["cells"][0]
    schedule = plan["schedules"][cell]
    refs = {
        case: sealed_read(root / "references" / cell / f"{case}.json")["reference"]
        for case in schedule
    }
    truth = {case: _truth(record) for case, record in refs.items()}
    calibration = [k for k, v in schedule.items() if v["role"] == "train_calibration"]
    contrasts = {k: v for k, v in schedule.items() if v["role"] == "contrast"}
    by_id = {r["source_id"]: r for r in subjects(root)}
    rows = []
    for entry in assignment["rows"]:
        base = {k: entry[k] for k in ("source_id", "method", "repetition")}
        if entry["status"] != "declared":
            for quantity in (*MECHANISMS, *STATES):
                rows.append(base | {"quantity": quantity, "status": entry["status"]})
            continue
        view = view_of(by_id[entry["source_id"]])
        model = {}
        failed = None
        for case, record in refs.items():
            try:
                model[case] = curves(view, entry, anchor.trajectory(cell, record))
            except (
                ValueError,
                ArithmeticError,
                TimeoutError,
                RuntimeExpressionError,
                ModelValidationError,
            ) as exc:
                failed = failed or {}
                failed[case] = str(exc)[:300]
        for quantity in (*MECHANISMS, *STATES):
            is_state = quantity in STATES
            key = "state." + quantity if is_state else quantity
            present = (
                entry["hidden"][quantity] is not None
                if is_state
                else entry["represented"][quantity]
            )
            if not present:
                rows.append(base | {"quantity": quantity, "status": "not_represented"})
                continue
            readouts = ("affine",) if is_state else ("direct", "affine")
            y = np.concatenate([truth[k][quantity] for k in calibration])
            scale = float(np.std(y))
            mapping = None
            if all(k in model for k in calibration):
                x = np.concatenate([model[k][key] for k in calibration])
                mapping = _fit(np.asarray(x), y)
            for readout in readouts:
                for case, item in contrasts.items():
                    row = base | {
                        "quantity": quantity,
                        "readout": readout,
                        "case": case,
                    }
                    control = item["control"]
                    if case not in model or control not in model:
                        rows.append(row | {"status": "rollout_unavailable"})
                        continue
                    if readout == "affine" and mapping is None:
                        rows.append(row | {"status": "calibration_unavailable"})
                        continue
                    a, b = mapping if readout == "affine" else (1.0, 0.0)
                    pred = a * np.asarray(model[case][key]) + b
                    pred_control = a * np.asarray(model[control][key]) + b
                    try:
                        metrics = anchor.scores(
                            refs[case]["time"],
                            truth[case][quantity],
                            truth[control][quantity],
                            pred,
                            pred_control,
                            item["window"],
                            scale,
                        )
                    except (ValueError, ArithmeticError) as exc:
                        metrics = {"status": "metric_unavailable", "reason": str(exc)}
                    if "peak_time_error_min" in metrics:
                        metrics["peak_time_error"] = metrics.pop("peak_time_error_min")
                    rows.append(
                        row
                        | {"map": mapping if readout == "affine" else None}
                        | metrics
                    )
        if failed:
            rows.append(
                base
                | {"quantity": "*", "status": "rollout_failures", "failures": failed}
            )
    return sealed_write(
        out / "scores.json",
        {
            "protocol": PROTOCOL,
            "attribution_sha256": assignment["artifact_sha256"],
            "rows": rows,
            "private_reference_opened": True,
            "test_data_opened": False,
        },
    )


def report_stage(out: Path) -> dict:
    """Per method and quantity: median RSE over all its responses, and directions."""
    rows = sealed_read(out / "scores.json")["rows"]
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["quantity"] == "*":
            continue
        readout = r.get("readout", "-")
        groups.setdefault((r["method"], r["quantity"], readout), []).append(r)
    summary = []
    for (method, quantity, readout), items in sorted(groups.items()):
        done = [r for r in items if r["status"] == "complete"]
        values = [
            r["effect_rse"] if r["status"] == "complete" else math.inf
            for r in items
            if r["status"] in ("complete", "rollout_unavailable")
        ]
        per_seed = {}
        for r in done:
            per_seed.setdefault(r["repetition"], []).append(r["effect_rse"])
        summary.append(
            {
                "method": method,
                "quantity": quantity,
                "readout": readout,
                "evaluations": len(items),
                "statuses": sorted({r["status"] for r in items}),
                "median_rse": statistics.median(values) if values else None,
                "per_seed_median_rse": {
                    str(k): statistics.median(v) for k, v in sorted(per_seed.items())
                },
                "directions": sum(bool(r.get("direction_correct")) for r in done),
            }
        )
    return sealed_write(out / "report.json", {"rows": summary})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("attribute", "score", "report"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    if a.stage == "attribute":
        result = attribute_stage(a.root, a.out)
        print(result["artifact_sha256"], len(result["rows"]), "models")
    elif a.stage == "score":
        result = score_stage(a.root, a.out)
        print(result["artifact_sha256"], len(result["rows"]), "rows")
    else:
        print(json.dumps(report_stage(a.out)["rows"], indent=1))
