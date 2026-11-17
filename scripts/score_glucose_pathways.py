"""Plasma-glucose pathways of frozen T1/T2 models against the simulator's true fluxes.

The true plasma balance is dGp/dt = EGP + Ra - Uii - E + X, with X the net
tissue-to-plasma exchange (under the perturbed T1 dynamics
X = k2 Gt [1 - 0.25 tanh((Gt - Gtb)/(0.2 Gtb))]
    - k1 Gp [1 + 0.35 tanh((Gp - Gpb)/(0.2 Gpb))],
published k1, k2, basal masses from the basal training reference). A model's Gp
equation is split into additive terms (``score_declared_mechanisms.split_terms``,
processes inlined) and each term is assigned by the names it uses:

    meal       the meal input, or a latent state driven only by the meal input
    insulin    (T2) plasma insulin, the insulin input, or a latent state driven
               only by them; such a term may also scale with Gp itself
    supplied   only supplied fluxes (EGP, Uii, E, as the task supplies them)
    remainder  everything else: tissue glucose, plasma glucose, other states,
               constants
    mixed      the meal pathway with anything else, or insulin with another
               state or channel

The true pathways sum to the true dGp/dt: meal = Ra; insulin = -Uid, the
insulin-dependent disposal the T2 task names; supplied = the supplied fluxes with
their balance signs; remainder = the rest of dGp/dt (for T1-easy exactly X).
In the T2 reference, exogenous insulin lowers glucose mostly by suppressing
endogenous production, which T2-hard does not supply, and Uid barely moves; so on
every contrast whose meal schedule equals its control's, the insulin pathway is
also scored against the whole change of the true dGp/dt ("insulin_effect"), all of
which the insulin causes there.
Terms are evaluated with the compiled model's restricted evaluator along its own
Radau rollout of each saved reference schedule (inputs and supplied channels as
saved, initial targets as saved), so no text is executed. Per contrast of the
confirmation panel and pathway: the integral over the trajectory, RSE over the
whole trajectory (as the panel's effect RSE) and over the contrast's own window,
direction accuracy A, shape error S and best gain of the response (run minus
control) against the true response. For perturbed T1 the remainder
is also read on the true states when it is written in Gp and Gt only: its
explained variance of X over every schedule, beside the best law
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
INSULIN = "insulin_pmol_per_kg_min"
SIGNS = {"EGP": 1.0, "Uii": -1.0, "E": -1.0}
PATHWAYS = ("meal", "insulin", "supplied", "remainder", "mixed")
K1, K2 = 0.065, 0.079
TRAINING = ("train_basal", "train_meal_30g_t0", "train_meal_90g_t0")
RTOL, ATOL, SECONDS = 1e-8, 1e-10, 45.0


def perturbed_exchange(record: dict, basal: dict) -> np.ndarray:
    """X under the perturbed T1 laws, from saved true Gp and Gt."""
    gp, gt = np.asarray(record["Gp"]), np.asarray(record["Gt"])
    gpb, gtb = basal["Gp"][0], basal["Gt"][0]
    jpt = K1 * gp * (1 + 0.35 * np.tanh((gp - gpb) / (0.2 * gpb)))
    jtp = K2 * gt * (1 - 0.25 * np.tanh((gt - gtb) / (0.2 * gtb)))
    return jtp - jpt


def truth(record: dict, cell: str, supplied: list[str], basal: dict) -> dict:
    """True pathways of dGp/dt; they sum to the true derivative."""
    time, gp = np.asarray(record["time"], float), np.asarray(record["Gp"], float)
    meal = np.asarray(record["Ra"], float)
    flux = sum(
        (SIGNS[c] * np.asarray(record[c], float) for c in supplied), np.zeros_like(gp)
    )
    insulin = (
        -np.asarray(record["Uid"], float)
        if cell == anchor.fits.T2
        else np.zeros_like(gp)
    )
    if cell == anchor.fits.T1P:
        remainder = perturbed_exchange(record, basal)
    else:
        remainder = np.gradient(gp, time, edge_order=2) - meal - flux - insulin
    return {"meal": meal, "insulin": insulin, "supplied": flux, "remainder": remainder}


def driven_states(
    candidate: CandidateModel, observed: set[str], sources: set[str]
) -> set[str]:
    """Latent states whose equations use only the sources and other such states."""
    equations = {e.state: e.rhs for e in candidate.state_equations}
    processes = {p.name: p.expression for p in candidate.processes}
    latent = {
        s.name
        for s in candidate.states
        if s.name not in observed and s.name in equations
    }
    uses = {s: names_in(expanded(equations[s], processes, "default")) for s in latent}
    channels = observed | latent | {MEAL, INSULIN}
    driven: set[str] = set()
    changed = True
    while changed:
        changed = False
        for s in latent - driven:
            used = uses[s] & channels
            if used & (sources | driven) and used <= sources | driven | {s}:
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
    meal = {MEAL} | driven_states(candidate, observed, {MEAL})
    insulin_sources = {
        n for n in (INSULIN, "I") if n in observed | set(context.external_inputs)
    }
    insulin = insulin_sources | (
        driven_states(candidate, observed, insulin_sources)
        if insulin_sources
        else set()
    )
    supplied = set(context.auxiliaries) & set(SIGNS)
    parameters = set(row["parameters"])
    tree = (
        grammar_parser("default")
        .parse(equations[TARGET], location="Gp equation")
        .tree.body
    )
    terms = []
    for sign, node, _ in split_terms(tree):
        names = names_in(expanded(ast.unparse(node), processes, "default")) - parameters
        m, i = names & meal, names & insulin
        rest = names - meal - insulin
        if (m and (i or rest)) or (i and rest - {TARGET}):
            pathway = "mixed"
        elif m:
            pathway = "meal"
        elif i:
            pathway = "insulin"
        elif names and names <= supplied:
            pathway = "supplied"
        else:
            pathway = "remainder"
        terms.append(
            {
                "sign": sign,
                "expression": ast.unparse(node),
                "pathway": pathway,
                "names": sorted(names),
            }
        )
    remainder_names = set().union(
        *([set(t["names"]) for t in terms if t["pathway"] == "remainder"] or [set()])
    )
    return {
        "meal_states": sorted(meal - {MEAL}),
        "insulin_states": sorted(insulin - insulin_sources),
        "supplied": sorted(supplied),
        "terms": terms,
        "represented": {p: any(t["pathway"] == p for t in terms) for p in PATHWAYS},
        "remainder_names": sorted(remainder_names),
        "remainder_is_a_law_of_gp_gt": remainder_names <= {"Gp", "Gt"}
        and "Gt" in remainder_names,
    }


def pathway_curves(row: dict, assignment: dict, cell: str, record: dict) -> dict:
    """The model's Gp and pathway terms along its rollout of one saved schedule."""
    candidate = CandidateModel.model_validate(row["candidate"])
    context = ValidationContext.model_validate(row["context"])
    compiled = compile_candidate(candidate, context)
    tr = anchor.trajectory(cell, record)
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
    on_truth = assignment["remainder_is_a_law_of_gp_gt"]
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
            if on_truth and term["pathway"] == "remainder":
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


def measures(pred, true, time, window=None) -> dict | None:
    """Integral, RSE (window and whole), direction A, shape error S and gain."""
    if pred is None:
        return None
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    den = float(np.sum(true**2))
    if den == 0:
        # No true response: report only what the model's pathway does.
        return {
            "integral": float(np.trapezoid(pred, time)),
            "true_integral": 0.0,
            "rse": None,
            "rse_window": None,
            "A": None,
            "S": None,
            "gain": None,
        }
    inside = (
        (time >= window[0]) & (time <= window[1])
        if window
        else np.ones(len(time), bool)
    )
    den_window = float(np.sum(true[inside] ** 2))
    dot, norm = float(np.sum(pred * true)), float(np.sum(pred**2))
    cos = dot / np.sqrt(norm * den) if norm > 0 else 0.0
    return {
        "integral": float(np.trapezoid(pred, time)),
        "true_integral": float(np.trapezoid(true, time)),
        "rse": float(np.sum((pred - true) ** 2) / den),
        "rse_window": float(np.sum((pred[inside] - true[inside]) ** 2) / den_window)
        if den_window > 0
        else None,
        "A": float(
            np.sum(np.abs(true) * (np.sign(pred) == np.sign(true)))
            / np.sum(np.abs(true))
        ),
        "S": 1 - max(0.0, cos) ** 2,
        "gain": max(0.0, dot / norm) if norm > 0 else 0.0,
    }


def best_linear_law(records: dict, basal: dict) -> dict:
    """The best -a Gp + b Gt + c fitted to the true perturbed exchange."""
    rows, target = [], []
    for record in records.values():
        gp, gt = np.asarray(record["Gp"]), np.asarray(record["Gt"])
        rows.append(np.column_stack([-gp, gt, np.ones_like(gp)]))
        target.append(perturbed_exchange(record, basal))
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


def contrasts(cell: str) -> list[dict]:
    """The panel's contrasts: control, window, and whether only insulin differs."""
    table = anchor.cases(cell, "confirmation")
    meals = {
        k: v["protocol"]["specification"].get("meals", []) for k, v in table.items()
    }
    return [
        {
            "case": k,
            "control": v["control"],
            "window": v["window"],
            "insulin_only": meals[k] == meals[v["control"]],
        }
        for k, v in table.items()
        if v["role"] == "contrast"
    ]


def score_rows(rows: list[dict], cell: str, references: Path) -> dict:
    """Every ready model on the saved confirmation-panel schedules of one cell."""
    pairs = contrasts(cell)
    cases = sorted(
        {c for p in pairs for c in (p["case"], p["control"])} | set(TRAINING)
    )
    records = {
        c: json.loads((references / f"{c}.json").read_text())["reference"]
        for c in cases
    }
    basal = records["train_basal"]
    result = {"cell": cell, "contrasts": pairs, "models": []}
    if cell == anchor.fits.T1P:
        result["best_linear_law"] = best_linear_law(records, basal)
    for row in rows:
        entry = {
            k: row.get(k)
            for k in ("method", "repetition", "source_id", "validation_nmse", "status")
        }
        if row["status"] != "ready":
            result["models"].append(entry)
            continue
        assignment = attribute(row)
        entry["assignment"] = assignment
        truths = {
            c: truth(r, cell, assignment["supplied"], basal) for c, r in records.items()
        }
        try:
            curves = {
                c: pathway_curves(row, assignment, cell, records[c]) for c in cases
            }
        except (ValueError, ArithmeticError) as exc:
            entry["rollout"] = f"failed: {exc}"
            result["models"].append(entry)
            continue
        scored = {}
        for pair in pairs:
            run, control, window = pair["case"], pair["control"], pair["window"]
            time = curves[run]["time"]
            true_gp = np.asarray(records[run]["Gp"]) - np.asarray(
                records[control]["Gp"]
            )
            item = {
                "Gp": measures(
                    curves[run]["Gp"] - curves[control]["Gp"], true_gp, time, window
                )
            }
            for p in ("meal", "insulin", "supplied", "remainder"):
                true_response = truths[run][p] - truths[control][p]
                if not np.any(true_response) and not assignment["represented"][p]:
                    item[p] = None
                    continue
                item[p] = measures(
                    curves[run][p] - curves[control][p], true_response, time, window
                )
                item[p]["represented"] = assignment["represented"][p]
            if cell == anchor.fits.T2 and pair["insulin_only"]:
                whole = sum(
                    truths[run][q] - truths[control][q]
                    for q in ("meal", "insulin", "supplied", "remainder")
                )
                item["insulin_effect"] = measures(
                    curves[run]["insulin"] - curves[control]["insulin"],
                    whole,
                    time,
                    window,
                )
                item["insulin_effect"]["represented"] = assignment["represented"][
                    "insulin"
                ]
            item["mixed_integral"] = float(
                np.trapezoid(curves[run]["mixed"] - curves[control]["mixed"], time)
            )
            scored[f"{run} - {control}"] = item
        entry["contrasts"] = scored
        if assignment["remainder_is_a_law_of_gp_gt"] and cell == anchor.fits.T1P:
            law = np.concatenate([curves[c]["law_on_true_states"] for c in cases])
            x = np.concatenate([truths[c]["remainder"] for c in cases])
            entry["exchange_law_on_true_states"] = {
                "explained_variance": 1
                - float(np.sum((law - x) ** 2)) / float(np.sum((x - x.mean()) ** 2)),
            }
        result["models"].append(entry)
    return result


def score(
    export: Path, references: Path, cell: str, benchmark_ids: set[str] | None = None
) -> dict:
    """An export's frozen models on one cell (its rows may carry an alias id)."""
    bundle = sealed_read(export)
    rows = [
        r
        for r in bundle["rows"]
        if benchmark_ids is None or r["benchmark_id"] in benchmark_ids
    ]
    return {
        "export_sha256": bundle["artifact_sha256"],
        **score_rows(rows, cell, references),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export", type=Path, required=True, help="a frozen_models.json bundle"
    )
    parser.add_argument(
        "--references",
        type=Path,
        required=True,
        help="saved confirmation-panel schedules",
    )
    parser.add_argument(
        "--cell", required=True, choices=[anchor.fits.T1P, anchor.fits.T2]
    )
    parser.add_argument(
        "--benchmark-id", action="append", help="keep rows with these ids"
    )
    parser.add_argument("--out", type=Path)
    a = parser.parse_args()
    out = score(
        a.export, a.references, a.cell, set(a.benchmark_id) if a.benchmark_id else None
    )
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:4000])
