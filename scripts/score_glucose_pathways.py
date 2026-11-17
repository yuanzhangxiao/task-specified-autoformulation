"""Plasma-glucose pathways of frozen T1 models against the simulator's true fluxes.

The true plasma balance is dGp/dt = Ra + X + (EGP - Uii - E), with X the net
tissue-to-plasma exchange; under the perturbed dynamics
X = k2 Gt [1 - 0.25 tanh((Gt - Gtb)/(0.2 Gtb))]
    - k1 Gp [1 + 0.35 tanh((Gp - Gpb)/(0.2 Gpb))]
(published k1, k2; basal masses from the basal training reference). A model's
Gp equation is split into additive terms (``score_declared_mechanisms.split_terms``,
processes inlined) and each term is assigned by the names it uses:

    meal      the meal input, or a latent state driven only by the meal input
    supplied  only the supplied fluxes EGP, Uii and E
    exchange  everything else (tissue glucose, plasma glucose, other states,
              constants), i.e. what the model puts where the true balance has X
    mixed     the meal pathway together with anything else

Terms are evaluated with the compiled model's restricted evaluator along its own
Radau rollout of each saved reference schedule (inputs and supplied channels as
saved, initial Gp as saved), so no text is executed. Per contrast (fed - fasting,
split - fed) and pathway: the integral over the trajectory, RSE, direction
accuracy A, shape error S and best gain against the true pathway. The exchange law
is also read on the true states: when the model's exchange uses only Gp, Gt and
parameters, it is evaluated on the saved true Gp and Gt of every schedule and
compared with X there (RSE and explained variance), beside the best law
-a Gp + b Gt + c fitted to X itself.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.expressions.parser import grammar_parser
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting.simulation import trajectory_forcing
from autoformalism.rebuttal.basin_algebra import expanded
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.schemas import CandidateModel
from scripts import iclr_anchor_fidelity as anchor
from scripts.score_declared_mechanisms import names_in, split_terms

TARGET = "Gp"
MEAL = "meal_rate_g_per_min"
SUPPLIED = {"EGP", "Uii", "E"}
PATHWAYS = ("meal", "supplied", "exchange", "mixed")
K1, K2 = 0.065, 0.079
CASES = (
    "fasting",
    "fed",
    "split",
    "train_basal",
    "train_meal_30g_t0",
    "train_meal_90g_t0",
)
CONTRASTS = (("fed", "fasting"), ("split", "fed"))
RTOL, ATOL, SECONDS = 1e-8, 1e-10, 45.0


def true_exchange(record: dict, basal: dict) -> np.ndarray:
    """X under the perturbed laws, from saved true Gp and Gt."""
    gp, gt = np.asarray(record["Gp"]), np.asarray(record["Gt"])
    gpb, gtb = basal["Gp"][0], basal["Gt"][0]
    jpt = K1 * gp * (1 + 0.35 * np.tanh((gp - gpb) / (0.2 * gpb)))
    jtp = K2 * gt * (1 - 0.25 * np.tanh((gt - gtb) / (0.2 * gtb)))
    return jtp - jpt


def truth(record: dict, basal: dict) -> dict:
    return {
        "meal": np.asarray(record["Ra"]),
        "supplied": np.asarray(record["EGP"])
        - np.asarray(record["Uii"])
        - np.asarray(record["E"]),
        "exchange": true_exchange(record, basal),
    }


def meal_states(candidate: CandidateModel, observed: set[str]) -> set[str]:
    """Latent states whose equations use only the meal input and other such states."""
    equations = {e.state: e.rhs for e in candidate.state_equations}
    processes = {p.name: p.expression for p in candidate.processes}
    latent = {
        s.name
        for s in candidate.states
        if s.name not in observed and s.name in equations
    }
    uses = {s: names_in(expanded(equations[s], processes, "default")) for s in latent}
    driven: set[str] = set()
    changed = True
    while changed:
        changed = False
        for s in latent - driven:
            states_used = uses[s] & (latent | observed)
            if MEAL in uses[s] and states_used <= driven | {s}:
                driven.add(s)
                changed = True
    return driven


def attribute(row: dict) -> dict:
    """Assign every additive term of the Gp equation to a pathway."""
    candidate = CandidateModel.model_validate(row["candidate"])
    context = ValidationContext.model_validate(row["context"])
    observed = {*context.targets, *context.auxiliaries}
    processes = {p.name: p.expression for p in candidate.processes}
    equations = {e.state: e.rhs for e in candidate.state_equations}
    if TARGET not in equations:
        raise ValueError("the model has no plasma-glucose equation")
    driven = meal_states(candidate, observed)
    parameters = set(row["parameters"])
    tree = (
        grammar_parser("default")
        .parse(equations[TARGET], location="Gp equation")
        .tree.body
    )
    terms = []
    for sign, node, _ in split_terms(tree):
        names = names_in(expanded(ast.unparse(node), processes, "default")) - parameters
        meal = bool(names & ({MEAL} | driven))
        others = names - {MEAL} - driven
        if meal and others:
            pathway = "mixed"
        elif meal:
            pathway = "meal"
        elif names and names <= SUPPLIED:
            pathway = "supplied"
        else:
            pathway = "exchange"
        terms.append(
            {
                "sign": sign,
                "expression": ast.unparse(node),
                "pathway": pathway,
                "names": sorted(names),
            }
        )
    exchange_names = set().union(
        *([set(t["names"]) for t in terms if t["pathway"] == "exchange"] or [set()])
    )
    return {
        "meal_states": sorted(driven),
        "terms": terms,
        "represented": {p: any(t["pathway"] == p for t in terms) for p in PATHWAYS},
        "exchange_names": sorted(exchange_names),
        "exchange_is_a_law_of_gp_gt": exchange_names <= {"Gp", "Gt"}
        and bool(exchange_names & {"Gt"}),
    }


def pathway_curves(row: dict, assignment: dict, record: dict) -> dict:
    """The model's Gp and pathway terms along its rollout of one saved schedule."""
    candidate = CandidateModel.model_validate(row["candidate"])
    context = ValidationContext.model_validate(row["context"])
    compiled = compile_candidate(candidate, context)
    tr = anchor.trajectory(anchor.fits.T1P, record)
    sim = simulate_trajectory(
        compiled,
        tr,
        row["parameters"],
        row["initials"],
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
    parser = grammar_parser("default")
    forcing = trajectory_forcing(compiled, tr)
    rhs = parser.parse(
        {e.state: e.rhs for e in candidate.state_equations}[TARGET], location="rhs"
    )
    parsed = [
        parser.parse(t["expression"], location="term") for t in assignment["terms"]
    ]
    time = np.asarray(tr.time, float)
    names = list(compiled.state_names)
    out = {p: np.zeros(len(time)) for p in PATHWAYS}
    total = np.zeros(len(time))
    law = np.zeros(len(time))
    on_truth = assignment["exchange_is_a_law_of_gp_gt"]
    for k, t in enumerate(time):
        state = states[:, k]
        total[k] = compiled.evaluate_expression(
            rhs, t, state, row["parameters"], forcing
        )
        true_state = state.copy()
        if on_truth:
            true_state[names.index(TARGET)] = record["Gp"][k]
        for term, expression in zip(assignment["terms"], parsed, strict=True):
            value = term["sign"] * compiled.evaluate_expression(
                expression, t, state, row["parameters"], forcing
            )
            out[term["pathway"]][k] += value
            if on_truth and term["pathway"] == "exchange":
                law[k] += term["sign"] * compiled.evaluate_expression(
                    expression, t, true_state, row["parameters"], forcing
                )
    if np.max(np.abs(sum(out.values()) - total)) > 1e-7 * max(
        1.0, float(np.max(np.abs(total)))
    ):
        raise ValueError("the assigned terms do not add up to the right-hand side")
    return {
        "time": time,
        "Gp": states[names.index(TARGET)],
        **out,
        "law_on_true_states": law if on_truth else None,
    }


def measures(pred, true, time) -> dict | None:
    """Integral, RSE, direction accuracy A, shape error S and gain of a response."""
    if pred is None:
        return None
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    den = float(np.sum(true**2))
    dot, norm = float(np.sum(pred * true)), float(np.sum(pred**2))
    cos = dot / np.sqrt(norm * den) if norm > 0 and den > 0 else 0.0
    return {
        "integral": float(np.trapezoid(pred, time)),
        "true_integral": float(np.trapezoid(true, time)),
        "rse": float(np.sum((pred - true) ** 2) / den) if den > 0 else None,
        "A": float(
            np.sum(np.abs(true) * (np.sign(pred) == np.sign(true)))
            / np.sum(np.abs(true))
        )
        if den > 0
        else None,
        "S": 1 - max(0.0, cos) ** 2,
        "gain": max(0.0, dot / norm) if norm > 0 else 0.0,
    }


def best_linear_law(records: dict, basal: dict) -> dict:
    """The best -a Gp + b Gt + c fitted to the true exchange over the schedules."""
    rows, target = [], []
    for record in records.values():
        gp, gt = np.asarray(record["Gp"]), np.asarray(record["Gt"])
        rows.append(np.column_stack([-gp, gt, np.ones_like(gp)]))
        target.append(true_exchange(record, basal))
    design, x = np.vstack(rows), np.concatenate(target)
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    resid = x - design @ coef
    return {
        "a": float(coef[0]),
        "b": float(coef[1]),
        "c": float(coef[2]),
        "explained_variance": 1
        - float(resid @ resid) / float(((x - x.mean()) ** 2).sum()),
    }


def score(export: Path, references: Path) -> dict:
    """Every frozen model of an export on the saved perturbed T1-easy schedules."""
    bundle = sealed_read(export)
    records = {
        c: json.loads((references / f"{c}.json").read_text())["reference"]
        for c in CASES
    }
    basal = records["train_basal"]
    truths = {c: truth(r, basal) for c, r in records.items()}
    result = {
        "export_sha256": bundle["artifact_sha256"],
        "best_linear_law": best_linear_law(records, basal),
        "models": [],
    }
    for row in bundle["rows"]:
        entry = {
            "seed": row["repetition"],
            "validation_nmse": row.get("validation_nmse"),
            "status": row["status"],
        }
        if row["status"] != "ready":
            result["models"].append(entry)
            continue
        assignment = attribute(row)
        entry["assignment"] = assignment
        try:
            curves = {c: pathway_curves(row, assignment, records[c]) for c in CASES}
        except (ValueError, ArithmeticError) as exc:
            entry["rollout"] = f"failed: {exc}"
            result["models"].append(entry)
            continue
        contrasts = {}
        for run, control in CONTRASTS:
            time = curves[run]["time"]
            true_gp = np.asarray(records[run]["Gp"]) - np.asarray(
                records[control]["Gp"]
            )
            item = {
                "Gp": measures(curves[run]["Gp"] - curves[control]["Gp"], true_gp, time)
            }
            for p in ("meal", "supplied", "exchange"):
                if not assignment["represented"][p] and p != "exchange":
                    item[p] = None
                    continue
                item[p] = measures(
                    curves[run][p] - curves[control][p],
                    truths[run][p] - truths[control][p],
                    time,
                )
            item["mixed"] = (
                measures(
                    curves[run]["mixed"] - curves[control]["mixed"],
                    np.zeros_like(time),
                    time,
                )
                if assignment["represented"]["mixed"]
                else None
            )
            contrasts[f"{run} - {control}"] = item
        entry["contrasts"] = contrasts
        if assignment["exchange_is_a_law_of_gp_gt"]:
            law = np.concatenate([curves[c]["law_on_true_states"] for c in CASES])
            x = np.concatenate([truths[c]["exchange"] for c in CASES])
            entry["exchange_law_on_true_states"] = {
                "rse": float(np.sum((law - x) ** 2) / np.sum((x - x.mean()) ** 2)),
                "explained_variance": 1
                - float(np.sum((law - x) ** 2)) / float(np.sum((x - x.mean()) ** 2)),
            }
        result["models"].append(entry)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export", type=Path, required=True, help="frozen_models.json of a T1 export"
    )
    parser.add_argument(
        "--references",
        type=Path,
        required=True,
        help="folder of saved perturbed T1-easy reference schedules",
    )
    parser.add_argument("--out", type=Path)
    a = parser.parse_args()
    out = score(a.export, a.references)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:6000])
