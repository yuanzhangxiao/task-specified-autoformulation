"""Replay budgeted ODE construction decisions on matched archival candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import cache
from itertools import product
from math import isfinite
from pathlib import Path
from statistics import mean

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.search import evidence_decisions
from autoformalism.search.evidence_decisions import Action, DecisionProblem, Scenario
from autoformalism.staged_topology import content_hash
from scripts import evidence_decision_panels as data
from scripts.construction_functions import file_hash

ROOT = Path(__file__).resolve().parents[1] / "output/evidence-decisions-20261126"
FAMILIES = ("simple", "extended")
BUDGETS = (6, 9)
PRICE = 0.03
ABSTENTION = 1.0
# A finite observation alphabet and decision loss are fixed across both domains.
EDGES = (0.0001, 0.001, 0.01, 0.1, 0.3, 1.0)
CENTERS = (0.00005, 0.00055, 0.0055, 0.055, 0.2, 0.65, 1.5)
POLICIES = ("adaptive", "one_step", "simple_first", "extended_first", "screens_first")
TOLERANCES = {
    "forecast": 0.0001,
    "precision": 0.0001,
    "precision_1mm": 0.001,
    "precision_10mm": 0.01,
}
MODES = tuple(TOLERANCES)
ACTIONS = tuple(
    Action(f"{kind}_{family}", kind, family, cost)
    for family in FAMILIES
    for kind, cost in (("check", 1), ("fit", 4))
)
BY_NAME = {a.name: a for a in ACTIONS}


def loss_bin(value: float | None) -> str:
    """Preserve a failed qualification as an outcome distinct from a large NMSE."""
    if value is None:
        return "failed"
    if not isfinite(value) or value < 0:
        raise ValueError("Invalid selection NMSE")
    return str(sum(value > edge for edge in EDGES))


def observation(candidate: dict, kind: str, mode: str) -> str:
    """A separate check never reads a fit outcome or a selection score."""
    if mode not in TOLERANCES:
        raise ValueError("Unknown objective")
    if kind == "check":
        if candidate["certified_reject"]:
            return "tension" if mode == "forecast" else "reject"
        return "open"
    if kind != "fit":
        raise ValueError("Unknown computation")
    return loss_bin(
        candidate["raw_selection_loss" if mode == "forecast" else "selection_loss"]
    )


def build_scenarios(development: list[dict], mode: str, transfer: float) -> list:
    """Estimate joint action outcomes only from other-domain training episodes.

    Split the empirical mass uniformly among their panels. A symmetric base
    measure supplies support for previously unseen check and fit combinations.
    """
    if not development or not 0 <= transfer < 1:
        raise ValueError("Need development episodes and positive base mass")
    fit_labels = ("failed", *(str(i) for i in range(len(CENTERS))))
    marginal = {}
    for fit in fit_labels:
        marginal[("open", fit)] = 0.75 / len(fit_labels)
        if mode == "forecast":
            marginal[("tension", fit)] = 0.25 / len(fit_labels)
    if mode != "forecast":
        marginal[("reject", "failed")] = 0.25
    weights = Counter(
        {
            (a, b): (1 - transfer) * marginal[a] * marginal[b]
            for a, b in product(marginal, repeat=2)
        }
    )
    for panel in development:
        pattern = tuple(
            tuple(
                observation(panel["candidates"][f], k, mode) for k in ("check", "fit")
            )
            for f in FAMILIES
        )
        weights[pattern] += transfer / len(development)
    worlds = []
    for pattern, weight in weights.items():
        observed, risk = {}, {}
        for family, (check, fit) in zip(FAMILIES, pattern, strict=True):
            observed[f"check_{family}"] = check
            observed[f"fit_{family}"] = fit
            risk[family] = ABSTENTION if fit == "failed" else CENTERS[int(fit)]
        worlds.append(Scenario(weight, observed, risk))
    return worlds


def fixed_action(policy: str, history: dict, remaining: int) -> str | None:
    """Two fit orders and a diagnostic-first workflow share the same action prices."""
    order = {
        "simple_first": ("fit_simple", "fit_extended"),
        "extended_first": ("fit_extended", "fit_simple"),
        "screens_first": (
            "check_simple",
            "check_extended",
            "fit_simple",
            "fit_extended",
        ),
    }[policy]
    for name in order:
        a = BY_NAME[name]
        if (
            name not in history
            and a.cost <= remaining
            and history.get(f"check_{a.family}") != "reject"
        ):
            return name
    return None


def episode(panel, solver, policy: str, mode: str, budget: int) -> dict:
    """Reveal only purchased action outcomes. Record comparisons at each decision."""
    history, trace = {}, []
    used = evaluations = 0
    while True:
        depth = {"adaptive": None, "one_step": 1}.get(policy, 0)
        decision = solver.decide(history, depth=depth)
        name = (
            decision.action
            if policy in {"adaptive", "one_step"}
            else fixed_action(policy, history, budget - used)
        )
        trace.append(
            {
                "spent": used,
                "action": name or "stop",
                "stop_model": decision.model,
                "stop_risk": decision.stop_value,
                "estimated_risk": decision.value,
                "action_values": dict(decision.action_values),
            }
        )
        if name is None:
            break
        a = BY_NAME[name]
        used += a.cost
        if used > budget or name in history:
            raise ValueError("Invalid action purchase")
        candidate = panel["candidates"][a.family]
        outcome = observation(candidate, a.kind, mode)
        history[name] = trace[-1]["observation"] = outcome
        if a.kind == "fit":
            evaluations += candidate["fit_evaluations"]
    winner = decision.model
    return {
        "model": winner,
        "cost_units": used,
        "optimizer_evaluations_replayed": evaluations,
        "checks": sum(BY_NAME[k].kind == "check" for k in history),
        "fits": sum(BY_NAME[k].kind == "fit" for k in history),
        "explicit_rejections": [k for k, v in history.items() if v == "reject"],
        "selection_nmse": panel["candidates"][winner]["raw_selection_loss"]
        if winner
        else None,
        "endpoint": panel["candidates"][winner]["fit_receipt"] if winner else None,
        "history": history,
        "trace": trace,
    }


def design() -> dict:
    """Declare forecast utility and sensitivity analyses independently of validation."""
    return {
        "protocol": "budgeted-ode-decisions-1",
        "modes": list(MODES),
        "family_roles": list(FAMILIES),
        "budgets": list(BUDGETS),
        "prices": {"fit": 4, "check": 1, "loss_per_unit": PRICE},
        "abstention": ABSTENTION,
        "validation_loss_cap": 2.0,
        "selection_nmse_upper_edges": list(EDGES),
        "risk_estimates": list(CENTERS),
        "transfer_mass": [0.75, 0.0],
        "base_check_tension_probability": 0.25,
        "policy": list(POLICIES),
        "primary": ["forecast", 0.75, 6],
        "precision_tolerance_basin_m": 0.0001,
        "additional_basin_tolerances_m": [0.001, 0.01],
    }


def select(root: Path) -> dict:
    """Freeze all policy choices before evaluation on official validation."""
    with _lock(root / "select-lock"):
        by_tolerance = {
            tol: data.load_panels(basin_tolerance=tol)
            for tol in sorted(set(TOLERANCES.values()))
        }
        panels = by_tolerance[TOLERANCES["forecast"]]
        codes = [Path(__file__), Path(data.__file__), Path(evidence_decisions.__file__)]
        plan = sealed_write(
            root / "plan.json",
            {
                "design": design(),
                "code": {p.name: file_hash(p) for p in codes},
                "panels": {
                    str(t): content_hash(rows) for t, rows in by_tolerance.items()
                },
                "basin_screens": {
                    str(t): {
                        f: {
                            "family_rejections": sum(
                                p["candidates"][f]["certified_reject"]
                                for p in rows
                                if p["benchmark"] == "basin"
                            ),
                            "eligible_fits": sum(
                                p["candidates"][f]["selection_loss"] is not None
                                for p in rows
                                if p["benchmark"] == "basin"
                            ),
                        }
                        for f in FAMILIES
                    }
                    for t, rows in by_tolerance.items()
                },
            },
        )
        domains = sorted({p["benchmark"] for p in panels})
        if len(domains) != 2 or len({p["panel_id"] for p in panels}) != len(panels):
            raise ValueError("Expected two distinct domains and unique episodes")
        receipts = []
        for target in domains:
            for mode, transfer, budget in product(MODES, (0.75, 0.0), BUDGETS):
                panels = by_tolerance[TOLERANCES[mode]]
                development = [p for p in panels if p["benchmark"] != target]
                worlds = build_scenarios(development, mode, transfer)
                solver = DecisionProblem(
                    ACTIONS,
                    worlds,
                    budget=budget,
                    cost_per_unit=PRICE,
                    abstention_loss=ABSTENTION,
                )
                for p in (p for p in panels if p["benchmark"] == target):
                    decisions = {
                        policy: episode(p, solver, policy, mode, budget)
                        for policy in POLICIES
                    }
                    key = f"{p['panel_id']}/{mode}-{transfer}-{budget}"
                    rec = sealed_write(
                        root / "decisions" / f"{key}.json",
                        {
                            "plan": plan["artifact_sha256"],
                            "panel": p["panel_id"],
                            "benchmark": target,
                            "fit_ids": p["fit_ids"],
                            "selection_ids": p["selection_ids"],
                            "mode": mode,
                            "transfer": transfer,
                            "budget": budget,
                            "decisions": decisions,
                        },
                    )
                    receipts.append((key, rec["artifact_sha256"]))
        return sealed_write(
            root / "choices.json",
            {"plan": plan["artifact_sha256"], "receipts": list(map(list, receipts))},
        )


def evaluate(root: Path) -> dict:
    """Score locked decisions and charge every failed or abstaining episode."""
    with _lock(root / "evaluation-lock"):
        plan = sealed_read(root / "plan.json")
        manifest = sealed_read(root / "choices.json")
        if plan["artifact_sha256"] != manifest["plan"] or plan["design"] != design():
            raise ValueError("Selection policy or source manifest changed")

        @cache
        def score(panel, model):
            path = root / "validation" / panel / f"{model}.json"
            if path.exists():
                record = sealed_read(path)
                if record["choices"] != manifest["artifact_sha256"]:
                    raise ValueError("Validation receipt belongs to other choices")
            else:
                record = sealed_write(
                    path,
                    {
                        "choices": manifest["artifact_sha256"],
                        "panel": panel,
                        "model": model,
                        **data.validation(panel, model),
                    },
                )
            return record

        rows = []
        for key, digest in manifest["receipts"]:
            r = sealed_read(root / "decisions" / f"{key}.json")
            if (r["plan"], r["artifact_sha256"]) != (manifest["plan"], digest):
                raise ValueError("Decision identity differs")
            for policy, d in r["decisions"].items():
                v = score(r["panel"], d["model"]) if d["model"] else None
                if v is not None and v["fit_receipt"] != d["endpoint"]:
                    raise ValueError("Validation endpoint differs from selected fit")
                raw = v["nmse"] if v and v["status"] == "qualified" else None
                if raw is not None and (not isfinite(raw) or raw < 0):
                    raise ValueError("Invalid validation score")
                penalty = ABSTENTION if raw is None else min(raw, 2.0)
                rows.append(
                    {
                        k: r[k]
                        for k in ("panel", "benchmark", "mode", "transfer", "budget")
                    }
                    | {
                        k: d[k]
                        for k in (
                            "model",
                            "fits",
                            "checks",
                            "cost_units",
                            "selection_nmse",
                            "optimizer_evaluations_replayed",
                        )
                    }
                    | {
                        "policy": policy,
                        "validation_nmse": raw,
                        "terminal_loss": penalty,
                        "loss_with_cost": penalty + PRICE * d["cost_units"],
                        "validation_status": v["status"] if v else "abstained",
                        "validation_receipt": v["artifact_sha256"] if v else None,
                    }
                )
        return sealed_write(
            root / "results.json",
            {"choices": manifest["artifact_sha256"], "rows": rows},
        )


def summarize(rows: list[dict]) -> dict:
    qualified = [r["validation_nmse"] for r in rows if r["validation_nmse"] is not None]
    return {
        "episodes": len(rows),
        "qualified": len(qualified),
        "mean_nmse": mean(qualified) if qualified else None,
        "all_episode_loss": mean(r["terminal_loss"] for r in rows),
        "loss_with_cost": mean(r["loss_with_cost"] for r in rows),
        **{
            k: sum(r[k] for r in rows)
            for k in ("fits", "checks", "cost_units", "optimizer_evaluations_replayed")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "evaluate"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(
        (select if args.stage == "select" else evaluate)(args.root)["artifact_sha256"]
    )
