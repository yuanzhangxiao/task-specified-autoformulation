"""Inspect target-reachable equations and paired historical seed behavior."""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from functools import cache, lru_cache
from pathlib import Path

import numpy as np

from autoformalism.expressions import (
    PiecewiseLinearForcing,
    ValidationContext,
    compile_candidate,
)
from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.rebuttal.mechanism_checks import effective_trees
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.schemas import CandidateModel
from scripts.iclr_search_history import CUTOFF, SUCCESS

PARSER = RestrictedParser()


def graph_for(candidate, parameters):
    """Apply the repository's restricted fitted-expression simplifier."""
    model = CandidateModel.model_validate(candidate)
    if set(parameters) != {p.name for p in model.parameters}:
        raise ValueError("missing or extraneous fitted coefficient")
    return {
        k: {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}
        for k, expr in effective_trees(model, parameters).items()
    }


def reachable(model: dict) -> dict:
    """Inspect upstream states, forcing reachability, and active annotations."""
    candidate, params, ctx = (
        model["candidate"],
        model["parameters"],
        model["lowered_context"],
    )
    states = {r["name"]: r for r in candidate["states"]}
    processes = {r["name"]: r for r in candidate["processes"]}
    graph = graph_for(candidate, params)
    observations = {r["channel"] for r in candidate["observation_mappings"]}
    if observations != set(ctx["targets"]):
        raise ValueError("incomplete output mapping")
    active, public_edges = set(), set()
    upstream = {}
    public_inputs = set(ctx["external_inputs"]) | set(ctx["auxiliaries"])
    for target in sorted(observations):
        seen = set()
        queue = deque(sorted(graph["output:" + target]))
        while queue:
            n = queue.popleft()
            if n not in seen:
                seen.add(n)
                queue.extend(sorted(graph.get(n, ())))
        upstream[target] = sorted(seen & states.keys())
        active |= seen
        public_edges.update(f"{n}→{target}" for n in seen & public_inputs)
    tagged = {
        tag
        for n, item in (states | processes).items()
        if n in active
        for tag in item.get("mechanisms", [])
    }
    return {
        "states": sorted(states),
        "active_states": sorted(active & states.keys()),
        "inactive_states": sorted(states.keys() - active),
        "active_latents": sorted(
            n for n, s in states.items() if n in active and s["kind"] == "latent"
        ),
        "active_roles": sorted(tagged),
        "public_edges": sorted(public_edges),
        "target_upstream_states": upstream,
    }


def endpoint_complexity(row: dict) -> dict:
    """Count executable syntax and target-reachable states of a frozen endpoint."""
    out = dict.fromkeys(
        (
            "state_count",
            "target_reachable_state_count",
            "target_reachable_latent_count",
            "parameter_count",
            "expression_nodes",
        )
    )
    out["complexity_status"] = row.get("status", "model_unavailable")
    if row.get("status") != "ready" or row.get("candidate") is None:
        return out
    try:
        candidate = CandidateModel.model_validate(row["candidate"])
        out.update(
            state_count=len(candidate.states), parameter_count=len(candidate.parameters)
        )
        expressions = [e.rhs for e in candidate.state_equations]
        expressions.extend(p.expression for p in candidate.processes)
        expressions.extend(m.expression for m in candidate.observation_mappings)
        expressions.extend(
            ic.expression for ic in candidate.initial_conditions if ic.expression
        )
        out["expression_nodes"] = sum(
            sum(
                1
                for _ in ast.walk(
                    PARSER.parse(expr, location="endpoint complexity").tree
                )
            )
            for expr in expressions
        )
        graph = reachable({**row, "lowered_context": row["context"]})
    except (ValueError, KeyError, ModelValidationError):
        out["complexity_status"] = "model_or_expression_unavailable"
        return out
    out.update(
        complexity_status="complete",
        target_reachable_state_count=len(graph["active_states"]),
        target_reachable_latent_count=len(graph["active_latents"]),
    )
    return out


def mixed_action(m):
    """Evaluate the unit-square insulin/action mixed difference in the Gp rate."""
    ctx = ValidationContext.model_validate(m["lowered_context"])
    fitted = compile_candidate(CandidateModel.model_validate(m["candidate"]), ctx)
    allowed = frozenset((*ctx.external_inputs, *ctx.auxiliaries))
    forcing = PiecewiseLinearForcing(
        [0, 1], {k: [0, 0] for k in allowed}, allowed_channels=allowed
    )
    names = list(fitted.state_names)
    x = np.zeros(len(names))
    x[names.index("Gp")] = 100.0
    action = "DelayedInsulinAction" if "DelayedInsulinAction" in names else "X"

    def rate(i, d):
        x[names.index("I")] = i
        x[names.index(action)] = d
        return float(fitted.rhs(0.0, x, m["parameters"], forcing)[names.index("Gp")])

    # Mixed finite difference at unit coordinates. For these saved bilinear
    # equations it extracts the coefficient on I times the driving state.
    coefficient = (rate(1, 1) - rate(1, 0)) - (rate(0, 1) - rate(0, 0))
    return action, coefficient


def worked_lineage(inv, traces):
    """Inspect the signed insulin/action term in trial and incumbent."""
    rows = []
    for step in (
        r for r in traces if r["task"] == SUCCESS and r["visit"] <= CUTOFF + 1
    ):
        for role in ("trial", "selected"):
            uid = step[role + "_fit_model_id"]
            if not uid:
                continue
            m = inv["models"][uid]
            action, coefficient = mixed_action(m)
            graph = reachable(m)
            rows.append(
                {
                    "task": SUCCESS,
                    "visit": step["visit"],
                    "role": role,
                    "fit_model_id": uid,
                    "source_result_sha256": step["result_sha256"],
                    "retained": uid == step["selected_fit_model_id"],
                    "active_states": graph["active_states"],
                    "inactive_states": graph["inactive_states"],
                    "driving_state": action,
                    "gp_rate_unit_mixed_difference": coefficient,
                    "utilization_expression": next(
                        (
                            e["expression"]
                            for e in m["candidate"]["processes"]
                            if e["name"] == "InsulinDependentGlucoseUtilization"
                        ),
                        None,
                    ),
                    "candidate_change_summary": m["base_candidate"].get(
                        "change_summary"
                    ),
                    "validation_nmse": step["retained_validation_nmse"]
                    if role == "selected"
                    else step["trial_validation_nmse"],
                }
            )
    return rows


def overlap(a, b):
    """Return Jaccard overlap; two empty sets provide no evidence of correspondence."""
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else None


def phenotype(candidate, parameters):
    """Expose named components for classifying changes within a single lineage."""
    dynamics = {r["state"]: r["rhs"] for r in candidate["state_equations"]}
    processes = {r["name"]: r["expression"] for r in candidate["processes"]}
    readouts = {
        r["channel"]: r["expression"] for r in candidate["observation_mappings"]
    }
    terms = {
        **{f"rate:{k}": v for k, v in dynamics.items()},
        **{f"process:{k}": v for k, v in processes.items()},
        **{f"output:{k}": v for k, v in readouts.items()},
    }
    states = {
        s["name"]: (s["kind"], s["description"], s.get("mechanisms"))
        for s in candidate["states"]
    }
    deps = graph_for(candidate, parameters)
    form = {
        k: ast.dump(PARSER.parse(v, location="archived term").tree)
        for k, v in terms.items()
    }
    return states, deps, form


def changes(old, new):
    """Classify observed changes between successive retained models."""
    a, b = (m["candidate"] for m in (old, new))
    s0, g0, f0 = phenotype(a, old["parameters"])
    s1, g1, f1 = phenotype(b, new["parameters"])
    return {
        "states_added": sorted(s1.keys() - s0.keys()),
        "states_removed": sorted(s0.keys() - s1.keys()),
        "state_declaration_changed": sorted(
            k for k in s1.keys() & s0.keys() if s0[k] != s1[k]
        ),
        "dependency_changes": sorted(
            k for k in g0.keys() | g1.keys() if g0.get(k) != g1.get(k)
        ),
        "expression_changes": sorted(
            k for k in f0.keys() | f1.keys() if f0.get(k) != f1.get(k)
        ),
        "fitted_coefficients_changed": old["parameters"] != new["parameters"],
        "initializer_changed": old["initialization_plan"] != new["initialization_plan"],
    }


def histories(inv, visits, inventory, function_scores):
    """Join trial and selection receipts to the historical search snapshot."""
    source = {(r["task_id"], r["round"]): r for r in visits}
    occurrences = {}
    for o in inv["occurrences"]:
        if o["role"] not in ("selected", "trial"):
            continue
        k = o["task"]["task_id"], o["round"]
        if k not in source or not source[k]["result_sha256"]:
            continue
        if source[k]["result_sha256"] != o["source_sha256"]:
            raise ValueError("incorrect historical fit identity")
        key = (*k, o["role"])
        if key in occurrences:
            raise ValueError("duplicate archived model")
        occurrences[key] = o

    @cache
    def active(uid):
        return reachable(inv["models"][uid])

    @cache
    def transition(prev, current):
        return changes(inv["models"][prev], inv["models"][current])

    seed_rows, traces, by_task = [], [], {}
    for task, step in sorted(source):
        r = source[task, step]
        o = occurrences.get((task, step, "selected"))
        trial = occurrences.get((task, step, "trial"))
        last = occurrences.get((task, step - 1, "selected"))
        uid = o["model_id"] if o else None
        why = (
            "available"
            if o
            else (
                "model_export_missing"
                if r["retained_validation_nmse"] is not None
                else "no_retained_fit"
            )
        )
        entry = {
            "task": task,
            "visit": step + 1,
            "status": r["status"],
            "structure_status": why,
            "result_sha256": r["result_sha256"],
            "selected_fit_model_id": uid,
            "trial_fit_model_id": trial["model_id"] if trial else None,
            "retained_validation_nmse": r["retained_validation_nmse"],
            "trial_validation_nmse": r["trial_validation_nmse"],
            "selected_origin_round": r["selected_origin_round"],
            "critic_text": "archive_unavailable",
            "stage": "pre_pruning",
        }
        if o:
            m = inv["models"][uid]
            entry.update(active(uid))
            entry["source_metadata"] = m["base_candidate"].get("change_summary")
            entry["fitted_receipt"] = o["provenance"].get("fit_result_sha256")
            if not last:
                entry["transition"] = (
                    "initial" if step == 0 else "prior_model_unavailable"
                )
            elif uid == last["model_id"]:
                entry["transition"] = "incumbent_unchanged"
            else:
                entry["transition"] = "retained_model_changed"
                entry.update(transition(last["model_id"], uid))
        traces.append(entry)
        if step == CUTOFF:
            score = function_scores.get((task, step, r["result_sha256"]))
            lookup = inventory[task]
            if lookup["source_result_sha256"] != (r["result_sha256"] or ""):
                raise ValueError("submitted per-seed row differs from source")
            row = {
                **lookup,
                "cohort_id": "historical-six",
                "structure_status": why,
                "fit_model_id": uid,
                "selected_functional_fraction": score,
                "functional_status": "measured" if score is not None else "unavailable",
                "functional_metric": "replayed functional checks on the fitted model",
            }
            if o:
                row.update(
                    {
                        "lowered_model_sha256": inv["models"][uid][
                            "lowered_candidate_sha256"
                        ],
                        "fit_receipt": entry["fitted_receipt"],
                        **active(uid),
                    }
                )
            seed_rows.append(row)
            by_task[task] = row
    pairs = []
    for r in seed_rows:
        if int(r["seed"]):
            continue
        other = by_task[r["task"].replace("_seed0_", "_seed1_")]
        a, b = r, other
        data = {
            "cell": a["cell"],
            "prompt": a["prompt"],
            "arm": a["configuration"],
            "seed0_task": a["task"],
            "seed1_task": b["task"],
            "seed0_status": a["structure_status"],
            "seed1_status": b["structure_status"],
            "comparable": bool(a["fit_model_id"] and b["fit_model_id"]),
        }
        if data["comparable"]:
            for i, m in enumerate((a, b)):
                data[f"seed{i}_declared_states"] = len(m["states"])
                data[f"seed{i}_active_states"] = len(m["active_states"])
                data[f"seed{i}_inactive_states"] = m["inactive_states"]
                data[f"seed{i}_public_edges"] = m["public_edges"]
                data[f"seed{i}_active_role_annotations"] = m["active_roles"]
            data["same_active_dimension"] = len(a["active_states"]) == len(
                b["active_states"]
            )
            data["public_dependency_jaccard"] = overlap(
                a["public_edges"], b["public_edges"]
            )
            data["active_role_annotation_jaccard"] = overlap(
                a["active_roles"], b["active_roles"]
            )
        pairs.append(data)
    return seed_rows, pairs, traces


def fixed_curves(visits, scores, horizon):
    """Use the same tasks at every visit, with a separate panel for every prompt/arm."""
    groups = defaultdict(dict)
    for r in visits:
        if r["round"] < horizon:
            groups[r["arm"], r["task_id"].rsplit("_", 1)[1]].setdefault(
                r["task_id"], []
            ).append(r)
    out = []
    for (prompt, arm), runs in sorted(groups.items()):
        joint = [
            task
            for task, records in runs.items()
            if len(records) == horizon
            and all(
                r["retained_validation_nmse"] is not None
                and scores.get((task, r["round"], r["result_sha256"])) is not None
                for r in records
            )
        ]
        for step in range(horizon):
            points = [r for t in joint for r in runs[t] if r["round"] == step]
            if len(points) != len(joint):
                raise ValueError("duplicate visit")
            out.append(
                {
                    "prompt": prompt,
                    "arm": arm,
                    "horizon": horizon,
                    "visit": step + 1,
                    "planned_runs": len(runs),
                    "fixed_runs": len(joint),
                    "tasks": sorted(joint),
                    "median_validation_nmse": float(
                        np.median([r["retained_validation_nmse"] for r in points])
                    )
                    if points
                    else None,
                    "mean_confirmed": float(
                        np.mean(
                            [
                                scores[r["task_id"], step, r["result_sha256"]]
                                for r in points
                            ]
                        )
                    )
                    if points
                    else None,
                }
            )
    return out


def behavioral_pairs(fidelity: Path) -> list[dict]:
    """Measure paired seed response differences against reference effect energy."""
    manifest = sealed_read(fidelity / "frozen_selection_manifest.json")
    if manifest["mode"] != "development":
        raise ValueError("development assessment required")
    seal = manifest["artifact_sha256"]
    scored = sealed_read(fidelity / "scored_rows.json")
    if scored["plan_sha256"] != seal:
        raise ValueError("wrong scored panel")
    status = {
        (r["subject"], r["case"], r["quantity"]): r["status"]
        for r in scored["rows"]
        if r["readout"] == "target"
    }
    arms = defaultdict(dict)
    for i, r in enumerate(manifest["roster"]):
        if r["method"].startswith("autoformalism:"):
            arms[r["benchmark_id"], r["method"]][r["repetition"]] = i
    out = []

    @lru_cache(maxsize=128)
    def load(folder, schedule):
        m = sealed_read(folder / f"{schedule}.json")
        if m["plan"] != seal:
            raise ValueError("wrong trajectory plan")
        return m

    for (cell, method), seeds in sorted(arms.items()):
        if set(seeds) != {0, 1}:
            raise ValueError("missing paired subject")
        for name, s in manifest["schedules"][cell].items():
            if s["role"] != "contrast":
                continue
            control = s["control"]
            for target in ("Gp", "I"):
                row = {
                    "cell": cell,
                    "method": method,
                    "case": name,
                    "control": control,
                    "target": target,
                    "seed0_subject": seeds[0],
                    "seed1_subject": seeds[1],
                    "seed0_status": status[seeds[0], name, target],
                    "seed1_status": status[seeds[1], name, target],
                }
                if (
                    row["seed0_status"] == "complete"
                    and row["seed1_status"] == "complete"
                ):

                    def effect(folder, key, name=name, control=control, target=target):
                        return (
                            np.array(load(folder, name)[key][target])
                            - load(folder, control)[key][target]
                        )

                    truth = effect(fidelity / "references" / cell, "reference")
                    a, b = (
                        effect(fidelity / "subjects" / f"{seeds[i]:04d}", "curves")
                        for i in (0, 1)
                    )
                    row["pair_effect_rse"] = float(
                        np.mean((a - b) ** 2) / np.mean(truth**2)
                    )
                    row["pair_difference_rmse"] = float(np.sqrt(np.mean((a - b) ** 2)))
                out.append(row)
    return out
