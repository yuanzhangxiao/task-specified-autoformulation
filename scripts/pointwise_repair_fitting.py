"""Match training refits of the three affine parents against their admitted repairs."""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

import numpy as np

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.preproposal_cases import T2
from autoformalism.research.trajectory_function_fitting import (
    _check_grid,
    _rows,
    validate_formula,
)
from autoformalism.staged_topology import content_hash
from scripts import trajectory_boundary_trial as archived
from scripts import trajectory_functions as trial
from scripts.construction_functions import file_hash

SLOTS = (2, 14, 15)
BUDGET = {"starts": 1, "max_evaluations": 500, "wall_seconds": 22.5, "seed": 100}
PRED_TOL = 1e-8
TARGET = "I"


def _tree(node, label):
    """Conservative expression key. Preserve operator order and literal values."""
    if isinstance(node, ast.Name):
        return label(node.id)
    if isinstance(node, ast.Constant):
        return ("literal", float(node.value).hex())
    if isinstance(node, ast.BinOp):
        return (
            type(node.op).__name__,
            _tree(node.left, label),
            _tree(node.right, label),
        )
    if isinstance(node, ast.UnaryOp):
        return (type(node.op).__name__, _tree(node.operand, label))
    return ("call", node.func.id, *(_tree(a, label) for a in node.args))


def _terms(node, sign=1):
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        yield from _terms(node.left, sign)
        yield from _terms(node.right, sign if isinstance(node.op, ast.Add) else -sign)
    else:
        yield sign, node


def _expressions(formula):
    parser = RestrictedParser()
    for group in ("equations", "initializers"):
        for name, expression in sorted(formula[group].items()):
            yield group, name, parser.parse(expression, location=name).tree.body


def _roles(formula):
    params = formula["parameters"]
    roles = defaultdict(list)
    for group, node, ex in _expressions(formula):
        for sign, term in _terms(ex):
            seen = {
                n.id for n in ast.walk(term) if isinstance(n, ast.Name)
            } & params.keys()
            for focal in seen:

                def label(name, focus=focal):
                    return (
                        ("parameter", name == focus)
                        if name in params
                        else ("signal", name)
                    )

                roles[focal].append((group, node, sign, _tree(term, label)))
    return {name: tuple(sorted(seq)) for name, seq in roles.items()}


def family_key(graph, formula):
    """Identify consistent coefficient renaming with the same ordered syntax."""
    labels = {}

    def label(name):
        if name in formula["parameters"]:
            return ("p", labels.setdefault(name, len(labels)))
        return ("signal", name)

    expressions = [
        (group, node, _tree(ex, label)) for group, node, ex in _expressions(formula)
    ]
    key = content_hash({"graph": graph, "expressions": expressions})
    return key, labels


def _additive(formula):
    out = {}
    for group, name, ex in _expressions(formula):
        terms = defaultdict(list)
        for sign, term in _terms(ex):
            terms[sign, _tree(term, lambda s: ("symbol", s))].append(term)
        out[group, name] = terms
    return out


def _total(node):
    """Sufficient finite-real mathematical domains for recognizing zero factors."""
    if isinstance(node, (ast.Constant, ast.Name)):
        return True
    if isinstance(node, ast.UnaryOp):
        return _total(node.operand)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Div):
            return (
                _total(node.left)
                and isinstance(node.right, ast.Constant)
                and node.right.value != 0
            )
        if isinstance(node.op, ast.Pow):
            return (
                _total(node.left)
                and isinstance(node.right, ast.Constant)
                and node.right.value in range(1, 17)
            )
        return _total(node.left) and _total(node.right)
    return (
        isinstance(node, ast.Call)
        and node.func.id
        in {"tanh", "exp", "sin", "abs", "min", "max", "softplus", "sigmoid"}
        and all(map(_total, node.args))
    )


def _zero_product(node, fresh):
    if isinstance(node, ast.Name):
        return node.id in fresh
    if isinstance(node, ast.UnaryOp):
        return _zero_product(node.operand, fresh)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _zero_product(node.left, fresh) or _zero_product(node.right, fresh)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return (
            isinstance(node.right, ast.Constant)
            and node.right.value in range(1, 17)
            and _zero_product(node.left, fresh)
        )
    return False


def parameter_relation(parent, candidate):
    """Map unchanged coefficient roles and recognize a restricted nested subfamily."""
    g, f = parent["graph"], parent["formula"]
    h, z = candidate["graph"], candidate["formula"]
    a, pa = family_key(g, f)
    b, pb = family_key(h, z)
    if a == b:
        reverse = {value: name for name, value in pa.items()}
        return {
            "parameter_map": {n: reverse[k] for n, k in pb.items()},
            "nested": "same_family",
            "neutral_parameters": [],
        }
    new_roles, old_roles = _roles(z), _roles(f)
    byrole = defaultdict(list)
    for n, r in old_roles.items():
        byrole[r].append(n)
    counts = Counter(new_roles.values())
    mapped = {
        n: byrole[v][0]
        for n, v in new_roles.items()
        if len(byrole[v]) == counts[v] == 1
    }
    relation = {
        "parameter_map": mapped,
        "nested": "unverified",
        "neutral_parameters": [],
    }
    if (
        g["variable_types"] != h["variable_types"]
        or g["initializers"] != h["initializers"]
    ):
        return relation
    before, after = _additive(f), _additive(z)
    fresh = z["parameters"].keys() - f["parameters"].keys()
    if (
        set(before) != set(after)
        or not fresh
        or not f["parameters"].keys() <= z["parameters"].keys()
    ):
        return relation
    for key in before:
        old, new = before[key], after[key]
        if any(len(new.get(term, ())) < len(copies) for term, copies in old.items()):
            return relation
        extras = [
            node
            for term, values in new.items()
            for node in values[len(old.get(term, ())) :]
        ]
        if any(not (_total(e) and _zero_product(e, fresh)) for e in extras):
            return relation
    return {
        "parameter_map": {k: k for k in f["parameters"]},
        "nested": "zero_added_terms",
        "neutral_parameters": sorted(fresh),
    }


def starting_points(parent, candidate, slot):
    relation = parameter_relation(parent, candidate)
    mapped = relation["parameter_map"]
    incumbent = parent["incumbent"]
    p = {
        k: incumbent[mapped[k]] if k in mapped else v
        for k, v in candidate["formula"]["parameters"].items()
    }
    first = p | dict.fromkeys(relation["neutral_parameters"], 0.0)
    second = {}
    for key, x in p.items():
        # Independent of the number or lexical ordering of added coefficients.
        token = f"{slot}:100:{mapped.get(key, key)}".encode()
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
        )
        shifted = x + 0.25 * max(abs(x), 0.01) * float(rng.normal())
        second[key] = shifted if np.isfinite(shifted) else x
    return relation, [first, second]


def parents_from_archive(campaign):
    plan, data, generation, freeze = archived._sources(campaign)
    out = {}
    for slot in SLOTS:
        g, arm, _ = trial.slots(data)[slot]
        row = data["graphs"][g]
        candidate = trial._read_candidate(campaign, plan, slot)
        fit = sealed_read(campaign / "fits" / f"{slot:03}" / "result.json")
        graph = trial.fitting_graph(candidate, row)
        result = fit["result"]
        if (
            row["dataset"] != T2
            or arm != "algorithm"
            or candidate["artifact_sha256"] != generation["candidates"][slot]
            or fit["artifact_sha256"] != freeze["fit_receipts"][slot]
            or fit["index"] != slot
            or fit["candidate"] != candidate["artifact_sha256"]
            or fit["generation"] != generation["artifact_sha256"]
            or result["status"] != "fitted"
            or result["graph_sha256"] != content_hash(graph)
            or result["formula_sha256"] != content_hash(candidate["formula"])
            or result["parameters"].keys() != candidate["formula"]["parameters"].keys()
        ):
            raise ValueError("Parent trial identity or incumbent failed verification")
        out[slot] = {
            "slot": slot,
            "graph": graph,
            "formula": candidate["formula"],
            "incumbent": result["parameters"],
            "archive_fit": fit["artifact_sha256"],
            "archive_candidate": candidate["artifact_sha256"],
            "archived_train_nmse": result["training_nmse"],
        }
    return data["train"][T2], out


def panel_data(root):
    from scripts import pointwise_repair_trial as panel

    records = panel.verified_records(root)
    plan, data = panel.load(root)
    return plan, data, records


def prepare(campaign: Path, panel: Path, root: Path):
    if any(root.resolve().is_relative_to(p.resolve()) for p in (campaign, panel)):
        raise ValueError("Use an output root separate from original and generation")
    rows, parents = parents_from_archive(campaign)
    pp, data, records = panel_data(panel)
    if content_hash(data["train"]) != content_hash(rows) or len(records) != 18:
        raise ValueError(
            "Require all 18 terminal decisions and the registered training cohort"
        )
    _rows(rows, "train")
    decisions = []
    cells = Counter()
    models = [
        {
            "key": f"parent-{s:03}",
            "slot": s,
            "graph": p["graph"],
            "formula": p["formula"],
        }
        for s, p in parents.items()
    ]
    for k, p in parents.items():
        src = data["parents"][str(k)]
        if (
            src["archive_candidate_sha256"] != p["archive_candidate"]
            or src["graph"] != p["graph"]
            or any(
                src["formula"][g] != p["formula"][g]
                for g in ("equations", "initializers")
            )
            or src["formula"]["parameters"].keys() != p["formula"]["parameters"].keys()
        ):
            raise ValueError("Panel parent differs from the original training model")
    keys = set()
    for r in sorted(records, key=lambda v: v["index"]):
        k, seed, arm = r["parent_slot"], r["seed"], r["condition"]
        key = (k, seed, arm)
        if k not in parents or arm not in ("control", "witness") or key in keys:
            raise ValueError("Unexpected or duplicate panel cell")
        keys.add(key)
        cells[k, arm] += 1
        item = {
            "index": r["index"],
            "slot": k,
            "seed": seed,
            "condition": arm,
            "admission": r["status"],
            "receipt": r["artifact_sha256"],
            "model": None,
        }
        if r["status"] == "eligible":
            item["model"] = len(models)
            f = validate_formula(r["graph_dict"], r["formula"])
            models.append(
                {
                    "key": f"reply-{r['index']:02}",
                    "slot": k,
                    "graph": r["graph_dict"],
                    "formula": f,
                    "condition": arm,
                    "seed": seed,
                }
            )
        decisions.append(item)
    if set(cells.values()) != {3} or len(cells) != 6:
        raise ValueError("Incomplete matched panel")
    seeds = {seed for _, seed, _ in keys}
    if len(seeds) != 3 or sorted(r["index"] for r in decisions) != list(range(18)):
        raise ValueError("Require three paired seeds and distinct panel indices")
    for m in models:
        src = parents[m["slot"]]
        relation, starts = starting_points(src, m, m["slot"])
        m.update(
            relation=relation,
            starts=starts,
            family_key=family_key(m["graph"], m["formula"])[0],
        )
    design = {
        "protocol": "pointwise-repair-training-fit-1",
        "code": file_hash(Path(__file__)),
        "runtime": archived._code(),
        "source_plan": archived.ARCHIVED_PLAN,
        "generation": pp["artifact_sha256"],
        "training_sha256": content_hash(rows),
        "parent_fits": {str(k): p["archive_fit"] for k, p in parents.items()},
        "archived_parent_nmse": {
            str(k): p["archived_train_nmse"] for k, p in parents.items()
        },
        "decisions": decisions,
        "models": models,
        "budget_per_start": BUDGET,
        "selection": "Lowest verified training NMSE across two paired starts, "
        "retaining each start as well as its optimization. Two starts times 500 "
        "residual calls and 22.5 seconds. Incumbents transferred only through "
        "recognized coefficient roles or a proved neutral embedding. "
        "No score can establish impossibility for all parameter vectors.",
        "tolerance_I": 1.0,
        "equivalent_on_training_scaled_tolerance": PRED_TOL,
        "validation_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        plan = sealed_write(root / "plan.json", design)
        sealed_write(
            root / "train.json", {"plan": plan["artifact_sha256"], "rows": rows}
        )
        for a in (0, 1):
            ms = []
            for m in models:
                ms.append(
                    {
                        "graph": m["graph"],
                        "formula": m["formula"] | {"parameters": m["starts"][a]},
                    }
                )
            sealed_write(
                root / f"start-{a}" / "plan.json",
                {
                    "design": plan["artifact_sha256"],
                    "budget": BUDGET,
                    "models": ms,
                    "start": a,
                },
            )
        return plan, rows


def _predictions(curves):
    return {r["trajectory_id"]: r["predicted"] for r in curves}


def _point_error(curves):
    y = np.concatenate([e["observed"][TARGET] for e in curves])
    p = np.concatenate([e["predicted"][TARGET] for e in curves])
    d = abs(p - y)
    return {
        "max_abs_I_error": float(max(d)),
        "within_1_unit": bool(np.all(d <= 1)),
        "insulin_rmse": float(np.sqrt(np.mean(d**2))),
        "samples": len(y),
        "negative_I": int(np.count_nonzero(p < -1e-10)),
        "near_zero_I_while_observed_above_one": int(
            np.count_nonzero((abs(p) <= 1) & (y > 1))
        ),
    }


def _check(m, rows, parameters, scales):
    names = sorted(scales)
    return _check_grid(
        m["graph"],
        m["formula"],
        _rows(rows, "train"),
        parameters,
        names,
        np.array([scales[n] for n in names]),
        curves=True,
    )


def run_one(root, plan, rows, index):
    if not 0 <= index < len(plan["models"]):
        raise ValueError("Invalid fit index")
    m = plan["models"][index]
    folder = root / "fitted" / f"{index:02}"
    with _lock(folder):
        ident = {"plan": plan["artifact_sha256"], "index": index, "key": m["key"]}
        path = folder / "score.json"
        if path.exists():
            r = sealed_read(path)
            if any(r[k] != v for k, v in ident.items()):
                raise ValueError("Existing fit identity differs")
            return r
        fits, start_errors = [], []
        candidates = []
        for a in (0, 1):
            sub = root / f"start-{a}"
            inner = sealed_read(sub / "plan.json")
            if inner["design"] != plan["artifact_sha256"] or inner["start"] != a:
                raise ValueError("Paired-start plan differs")
            folder_a = sub / "fits" / f"{index:02}"
            error_path = folder_a / "numerical-error.json"
            try:
                rec = archived._fit(sub, inner, index, rows)
            except (ArithmeticError, RuntimeError, np.linalg.LinAlgError) as exc:
                if not (folder_a / "reservation.json").exists():
                    raise
                sealed_write(
                    error_path,
                    {
                        "plan": inner["artifact_sha256"],
                        "index": index,
                        "error": str(exc),
                    },
                )
                # The existing reservation only permits recovery of a saved warm
                # vector. The interrupted optimization receives no new allowance.
                rec = archived._fit(sub, inner, index, rows)
            start_errors.append(
                sealed_read(error_path)["error"] if error_path.exists() else None
            )
            fits.append(rec)
            if (j := rec["selected_option"]) is not None:
                r = rec["options"][j]
                candidates.append((r["training_nmse"], a, j, r))
        base = ident | {
            "fit_receipts": [r["artifact_sha256"] for r in fits],
            "evaluations": sum(r["evaluations"] for r in fits),
            "option_statuses": [[o["status"] for o in f["options"]] for f in fits],
            "start_outcomes": [f["status"] for f in fits],
            "numerical_errors": start_errors,
            "complete_budget_accounting": all(
                r["evaluation_count_complete"] for r in fits
            ),
        }
        if not candidates:
            outcomes = {s for group in base["option_statuses"] for s in group}
            if "resolution_unverified" in outcomes:
                status = "resolution_unverified"
            elif not base["complete_budget_accounting"]:
                status = "interrupted_without_verified_fit"
            elif len(outcomes) == 1:
                status = outcomes.pop()
            else:
                status = "no_verified_fit"
            return sealed_write(path, base | {"status": status})
        _, a, j, r = min(candidates, key=lambda x: x[:3])
        c = _check(m, rows, r["parameters"], r["scales"])
        report = {
            k: v for k, v in c.items() if k not in ("curves", "diagnostic_fine_grid")
        }
        if c["status"] != "scored":
            return sealed_write(path, base | report)
        missing = {
            n: sorted(set(v) - set(r["effective_sources"][n]))
            for n, v in m["graph"]["sources"].items()
        }
        return sealed_write(
            path,
            base
            | report
            | _point_error(c["curves"])
            | {
                "selected_start": a,
                "selected_option": j,
                "parameters": r["parameters"],
                "scales": r["scales"],
                "exactly_inactive_fitted_sources": missing,
                "predictions": _predictions(c["curves"]),
                "stop_reasons": r["stop_reasons"],
            },
        )


def numerical_distance(a, b, scales):
    """Largest normalized difference on the registered sampled training grid."""
    if set(a) != set(b):
        raise ValueError("Different training event IDs")
    maximum = 0.0
    for event, x in a.items():
        if set(x) != set(b[event]) or set(x) != set(scales):
            raise ValueError("Different training target names")
        for target, v in x.items():
            left, right = np.asarray(v), np.asarray(b[event][target])
            if left.shape != right.shape:
                raise ValueError("Different time grid lengths")
            maximum = max(maximum, float(np.max(abs(left - right)) / scales[target]))
    return maximum


def _compact(score):
    return {
        k: v
        for k, v in score.items()
        if k
        not in (
            "predictions",
            "fit_receipts",
            "parameters",
            "scales",
            "artifact_sha256",
            "plan",
        )
    }


def compare(parent, child):
    if parent["status"] != "scored" or child["status"] != "scored":
        return {"status": "no_verified_pair"}
    if child["scales"] != parent["scales"]:
        raise ValueError("Training normalization differs across matched fits")
    equal = numerical_distance(
        parent["predictions"], child["predictions"], parent["scales"]
    )
    report = {
        "status": "scored",
        "max_scaled_prediction_difference": equal,
        "same_training_predictions": equal <= PRED_TOL,
    }
    for key in ("pooled", "insulin"):
        a, b = (
            r["pooled_nmse"] if key == "pooled" else r["per_target_nmse"][TARGET]
            for r in (parent, child)
        )
        report[f"{key}_improved"] = sqrt(b) + PRED_TOL < sqrt(a)
        report[f"{key}_nmse_change"] = b - a
        # The two measured coarse/fine prediction gaps bound the shift in each RMSE.
        report[f"{key}_ordering_separated_across_grids"] = (
            sqrt(b) + child["maximum_scaled_grid_difference"]
            < sqrt(a) - parent["maximum_scaled_grid_difference"]
        )
    report["both_improved"] = report["pooled_improved"] and report["insulin_improved"]
    return report


def report(root: Path):
    plan = sealed_read(root / "plan.json")
    models = plan["models"]
    paths = [root / "fitted" / f"{i:02}" / "score.json" for i in range(len(models))]
    if not all(p.exists() for p in paths):
        return {
            "status": "incomplete",
            "expected": len(paths),
            "recorded": sum(p.exists() for p in paths),
        }
    with (root / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "summary.json"
        scores = [sealed_read(p) for p in paths]
        for i, s in enumerate(scores):
            if s["plan"] != plan["artifact_sha256"] or s["index"] != i:
                raise ValueError("Score identity differs")
            for a, sha in enumerate(s["fit_receipts"]):
                fit = sealed_read(
                    root / f"start-{a}" / "fits" / f"{i:02}" / "result.json"
                )
                if fit["artifact_sha256"] != sha:
                    raise ValueError("Score source receipt differs")
        identity = {
            "plan": plan["artifact_sha256"],
            "scores": [s["artifact_sha256"] for s in scores],
        }
        if path.exists():
            r = sealed_read(path)
            if any(r[k] != v for k, v in identity.items()):
                raise ValueError("Existing summary differs from fit receipts")
            return r
        cohort = sealed_read(root / "train.json")
        if (
            cohort["plan"] != plan["artifact_sha256"]
            or content_hash(cohort["rows"]) != plan["training_sha256"]
        ):
            raise ValueError("Training observations differ from the fit freeze")
        rows = []
        for decision in plan["decisions"]:
            summary = dict(decision)
            idx = decision["model"]
            if idx is not None:
                m = models[idx]
                parent_idx = SLOTS.index(m["slot"])
                p, c = scores[parent_idx], scores[idx]
                comparison = compare(p, c)
                added = set(m["graph"]["sources"][TARGET]) - set(
                    models[parent_idx]["graph"]["sources"][TARGET]
                )
                if c["status"] == "scored":
                    inactive = added & set(c["exactly_inactive_fitted_sources"][TARGET])
                    comparison.update(
                        added_insulin_sources=sorted(added),
                        exactly_inactive_added_sources=sorted(inactive),
                    )
                    comparison["equivalent_predictions_on_grid_to_prior_replies"] = [
                        older["index"]
                        for older in rows
                        if older["slot"] == m["slot"]
                        and older["model"] is not None
                        and scores[older["model"]]["status"] == "scored"
                        and numerical_distance(
                            scores[older["model"]]["predictions"],
                            c["predictions"],
                            c["scales"],
                        )
                        <= PRED_TOL
                    ]
                summary.update(
                    score=_compact(c),
                    comparison=comparison,
                    family_key=m["family_key"],
                    parameter_relation=m["relation"],
                )
                if m["relation"]["nested"] != "unverified" and p["status"] == "scored":
                    mapped = m["relation"]["parameter_map"]
                    pars = {k: p["parameters"][v] for k, v in mapped.items()}
                    pars.update(dict.fromkeys(m["relation"]["neutral_parameters"], 0.0))
                    if pars.keys() != m["formula"]["parameters"].keys():
                        raise ValueError("Incomplete nested parameter embedding")
                    neutral = _check(m, cohort["rows"], pars, p["scales"])
                    replay = {"status": neutral["status"]}
                    if neutral["status"] == "scored":
                        gap = numerical_distance(
                            _predictions(neutral["curves"]),
                            p["predictions"],
                            p["scales"],
                        )
                        replay.update(
                            max_scaled_difference=gap,
                            matches_refitted_parent=gap <= PRED_TOL,
                        )
                    summary["parent_embedded_after_fitting"] = replay
            rows.append(summary)
        counts = {}
        for arm in ("control", "witness"):
            decisions = [d for d in rows if d["condition"] == arm]
            scored = [
                d
                for d in decisions
                if d.get("comparison", {}).get("status") == "scored"
            ]
            counts[arm] = {
                "requested": len(decisions),
                "admitted": sum(d["model"] is not None for d in decisions),
                "admission_statuses": dict(Counter(d["admission"] for d in decisions)),
                "fitting_statuses": dict(
                    Counter(
                        d["score"]["status"]
                        for d in decisions
                        if d["model"] is not None
                    )
                ),
                "verified_pairs": len(scored),
                "distinct_scored_families": len({d["family_key"] for d in scored}),
                "improved_insulin_and_pooled": sum(
                    d["comparison"]["both_improved"] for d in scored
                ),
                "within_one_unit_all_insulin_measurements": sum(
                    d["score"]["within_1_unit"] for d in scored
                ),
                "all_added_insulin_sources_exactly_inactive": sum(
                    bool(d["comparison"]["added_insulin_sources"])
                    and d["comparison"]["exactly_inactive_added_sources"]
                    == d["comparison"]["added_insulin_sources"]
                    for d in scored
                ),
            }
        return sealed_write(
            path,
            identity
            | {
                "status": "complete",
                "parents": [_compact(s) for s in scores[:3]],
                "decisions": rows,
                "conditions": counts,
                "evaluations": sum(s["evaluations"] for s in scores),
                "validation_opened": False,
                "test_data_opened": False,
                "llm_calls": 0,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run")
    run.add_argument("--campaign", required=True, type=Path)
    run.add_argument("--panel", required=True, type=Path)
    run.add_argument("--index", type=int)
    for sub in (run, commands.add_parser("report")):
        sub.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "run":
        plan, rows = prepare(args.campaign, args.panel, args.root)
        indices = [args.index] if args.index is not None else range(len(plan["models"]))
        for i in indices:
            run_one(args.root, plan, rows, i)
    result = report(args.root)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in ("status", "expected", "recorded", "conditions", "evaluations")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
