"""Construct and fit matched T2 disposal laws on registered training events."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import trajectory_function_families as initial
from autoformalism.research import trajectory_function_fitting as rollout
from autoformalism.research import trajectory_nonnegative_fitting as bounded
from autoformalism.staged_topology import content_hash
from scripts import run_hidden_form_forecasts as prior
from scripts import screen_t2_disposal_dependencies as dependencies
from scripts.audit_hidden_shape_t2 import FROZEN
from scripts.construction_functions import file_hash
from scripts.matched_equation_experiment import DEFAULT as SOURCE
from scripts.matched_equation_experiment import _registered

DEFAULT = SOURCE.parent / "t2-balanced-disposal-20261126"
NATIVE = {"graph02": "action", "graph03": "both"}
LAWS = {
    "action": (),
    "plasma": ("Gp",),
    "tissue": ("Gt",),
    "both": ("Gp", "Gt"),
    "interaction": ("Gt",),
}
D_AUDIT_TOLERANCE = 1e-9


def _symbols(form: dict, equation: str) -> set:
    parsed = RestrictedParser().parse(form["equations"][equation], location=equation)
    return parsed.symbols & form["parameters"].keys()


def _path(graph: dict, start: str, target: str) -> bool:
    """Follow dependency edges backwards through algebraic and differential nodes."""
    todo, visited = [target], set()
    while todo:
        here = todo.pop()
        if here == start:
            return True
        if here not in visited:
            visited.add(here)
            todo.extend(graph["sources"].get(here, ()))
    return False


def _starts(
    graph: dict, form: dict, constant_intercept: str, level: float
) -> dict[str, dict]:
    """Flatten the two differential target RHSs and initialize a positive readout."""
    if any(
        graph["initializers"].get(n, {}).get("mode") != "observed_initial"
        for n in ("Gp", "I")
    ):
        raise ValueError("Expected observed differential initials on Gp and I")
    touched = set.union(*(_symbols(form, n) for n in ("Gp", "I", "U")))
    constant = {k: 0.0 if k in touched else v for k, v in form["parameters"].items()}
    if constant_intercept not in touched:
        raise ValueError("Readout intercept missing")
    constant[constant_intercept] = level
    return {"fold_local": form["parameters"], "constant_observed": constant}


def _recorded(rows: list[dict], name: str) -> np.ndarray:
    return np.concatenate([(r["targets"] | r["auxiliaries"])[name] for r in rows])


def candidates(packet: dict, screen: dict, native: str) -> dict:
    """Pair each disposal dependency hypothesis with fixed and freely fitted Uii."""
    rows = packet["train_rows"]
    base = packet["shape"]["candidates"][0]
    base_graph = packet["graph"]
    supplied = _recorded(rows, "Uii")
    mean_input = float(np.mean(supplied))
    disposal = _recorded(rows, "U") - supplied
    if set(base_graph["sources"]["U"]) != {"Uii", "I_act", *LAWS[native]}:
        raise ValueError("Incorrect native disposal graph")
    if base_graph["variable_types"]["U"] != "algebraic":
        raise ValueError("Require an algebraic readout")
    if packet["shape"]["candidate_families"][0]["U"] != "affine":
        raise ValueError("The reference U equation must be affine")
    original_params = _symbols(base, "U")
    native_intercept = base["equations"]["U"].partition("+")[0]
    if native_intercept not in original_params:
        raise ValueError("Expected a leading free intercept")
    original_action_gain = next(
        term.removesuffix("*I_act")
        for term in base["equations"]["U"].split("+")
        if term.endswith("*I_act")
    )
    out = {}
    for law, measured in LAWS.items():
        graph = deepcopy(base_graph)
        graph["sources"]["U"] = sorted(["I_act", "Uii", *measured])
        if not all(
            _path(graph, a, b)
            for a, b in (("meal_rate_g_per_min", "Gp"), ("I", "I_act"), ("I_act", "U"))
        ):
            raise ValueError("Lost a required dependency")
        key = "+".join(("I_act", *measured))
        eligible = screen["hidden_candidates"][key]["eligible"]
        columns = [np.ones_like(disposal)]
        bases = {}
        for name in measured:
            values = _recorded(rows, name)
            center, scale = initial._scale([values])
            columns.append((values - center) / scale)
            bases[name] = f"(({name}-({center:.12g}))/{scale:.12g})"
        # The hidden action is unobserved. Only measured columns enter the
        # readout warm start. All readout and dynamics weights remain fit parameters.
        weights = initial._coefficients(np.column_stack(columns), disposal)
        for arm in ("free", "balanced"):
            fixed = arm == "balanced"
            form = deepcopy(base)
            if not fixed and law == native:
                intercept = native_intercept
            else:
                intercept = "dOffset"
                form["parameters"] = {
                    p: v
                    for p, v in base["parameters"].items()
                    if p not in original_params
                }
                new_params = {
                    intercept: float(weights[0]) + (0 if fixed else mean_input),
                    "dAction": base["parameters"][original_action_gain],
                }
                terms = [intercept, "dAction*I_act"]
                for name, weight in zip(measured, weights[1:], strict=True):
                    terms.append(f"d{name}*{bases[name]}")
                    new_params[f"d{name}"] = float(weight)
                if law == "interaction":
                    terms.append(f"dJoint*I_act*{bases['Gt']}")
                    new_params["dJoint"] = 0.0
                if fixed:
                    terms.insert(0, "Uii")
                else:
                    terms.append(f"dUii*(Uii-({mean_input:.12g}))")
                    new_params["dUii"] = 0.0
                form["equations"]["U"] = "+".join(terms)
                form["parameters"].update(new_params)
            form = rollout.validate_formula(graph, form)
            baseline_level = float(np.mean(disposal)) + (0 if fixed else mean_input)
            out[f"{arm}_{law}"] = {
                "arm": arm,
                "law": law,
                "graph": graph,
                "formula": form,
                "starts": _starts(graph, form, intercept, baseline_level),
                "dependency_key": key,
                "dependency_eligible": eligible,
            }
    return out


def prepare(root: Path, source: Path) -> dict:
    """Seal fold-only screens, candidates, optimization ceilings and choice splits."""
    parent = _registered(source / "plan.json", FROZEN)
    index = {p["panel_id"]: p for p in parent["panels"]}
    panels = []
    with _lock(root / "preparation"):
        for g, native in NATIVE.items():
            for fold in (0, 1):
                pid = f"{g}-fold{fold}"
                other = index[f"{g}-fold{1 - fold}"]
                cfg = index[pid]
                packet = _registered(source / "packets" / f"{pid}.json", cfg["packet"])
                rows = packet["train_rows"]
                ids = [r["trajectory_id"] for r in rows]
                result = dependencies.screen_dependencies(
                    packet["graph"], rows, interpolation=packet["interpolation"]
                )
                screen = sealed_write(
                    root / "screens" / f"{pid}.json",
                    {
                        "packet": cfg["packet"],
                        "fit_ids": ids,
                        "screen": result,
                    },
                )
                families = candidates(packet, result, native)
                panels.append(
                    {
                        "panel_id": pid,
                        "graph_name": g,
                        "packet_id": pid,
                        "packet": cfg["packet"],
                        "other_packet_id": other["panel_id"],
                        "other_packet": other["packet"],
                        "opposite_sha256": cfg["held_sha256"],
                        "fit_ids": ids,
                        "choice_rotations": prior.rotations(ids, cfg["held_ids"]),
                        "candidates": families,
                        "native_family": f"free_{native}",
                        "screen": screen["artifact_sha256"],
                    }
                )
        return sealed_write(
            root / "plan.json",
            {
                "protocol": "matched-balanced-disposal-1",
                "source_plan": FROZEN,
                "dataset": parent["dataset"],
                "panels": panels,
                "nonnegative_targets": parent["nonnegative_targets"],
                "per_start_budget": prior.BUDGET,
                "starts": list(prior.STARTS),
                "D_audit_tolerance": D_AUDIT_TOLERANCE,
                "code": {
                    Path(p).name: file_hash(Path(p))
                    for p in (
                        __file__,
                        prior.__file__,
                        dependencies.__file__,
                        initial.__file__,
                        rollout.__file__,
                        bounded.__file__,
                    )
                },
                "versions": {p: version(p) for p in ("casadi", "scipy", "numpy")},
            },
        )


def load_endpoint(root, plan, panel, family) -> tuple[dict, dict, dict | None, str]:
    """Verify a chosen endpoint and return its original graph, formula and saved fit."""
    c = panel["candidates"][family]
    folder = root / "fits" / panel["panel_id"] / family
    best = sealed_read(folder / "best.json")
    expected = (plan["artifact_sha256"], panel["panel_id"], family)
    if (best["plan"], best["panel_id"], best["family"]) != expected:
        raise ValueError("Different endpoint identity")
    start = best["best_start"]
    if start is None:
        return c["graph"], c["formula"], None, best["artifact_sha256"]
    form = {**c["formula"], "parameters": c["starts"][start]}
    rec = _registered(folder / f"{start}.json", best["starts"][start])
    if (rec["plan"], rec["start"], rec["formula"]) != (
        plan["artifact_sha256"],
        start,
        content_hash(form),
    ):
        raise ValueError("Starting vector or formula differs")
    result = rec["result"]
    if (result["graph_sha256"], result["formula_sha256"]) != (
        content_hash(c["graph"]),
        content_hash(form),
    ):
        raise ValueError("Fitted equations differ")
    return c["graph"], form, result, best["artifact_sha256"]


def _run(job):
    root, source, pid, family = job
    plan = sealed_read(root / "plan.json")
    panel = next(p for p in plan["panels"] if p["panel_id"] == pid)
    c = panel["candidates"][family]
    folder = root / "fits" / pid / family
    marker = {"plan": plan["artifact_sha256"], "panel_id": pid, "family": family}
    rows = _registered(source / "packets" / f"{pid}.json", panel["packet"])[
        "train_rows"
    ]
    with _lock(folder):
        receipts = {}
        # Preserve the frozen original comparator even if its dependency screen fails.
        if c["dependency_eligible"] or family == panel["native_family"]:
            for s, vector in c["starts"].items():
                form = {**c["formula"], "parameters": vector}
                receipts[s] = prior._train_start(
                    folder,
                    plan["artifact_sha256"],
                    s,
                    c["graph"],
                    form,
                    rows,
                    plan["nonnegative_targets"],
                )
        fits = [
            (r["result"]["training_nmse"], i, s)
            for i, (s, r) in enumerate(receipts.items())
            if r["result"]["status"] == "fitted"
        ]
        sealed_write(
            folder / "best.json",
            {
                **marker,
                "best_start": min(fits)[2] if fits else None,
                "training_nmse": min(fits)[0] if fits else None,
                "starts": {s: r["artifact_sha256"] for s, r in receipts.items()},
            },
        )
        graph, form, result, identity = load_endpoint(root, plan, panel, family)
        other = _registered(
            source / "packets" / f"{panel['other_packet_id']}.json",
            panel["other_packet"],
        )
        if content_hash(other["train_rows"]) != panel["opposite_sha256"]:
            raise ValueError("Opposite event identity differs")
        for row in other["train_rows"]:
            eid = row["trajectory_id"]
            path = root / "curves" / pid / family / f"{eid}.json"
            identity_fields = {**marker, "event_id": eid, "fit": identity}
            if path.exists():
                saved = sealed_read(path)
                if any(saved[k] != v for k, v in identity_fields.items()):
                    raise ValueError("Forecast identity differs")
                continue
            out = (
                prior._forecast(graph, form, result, row, plan["nonnegative_targets"])
                if result
                else {"status": "no_fit", "nmse": None}
            )
            if "predicted" in out:
                d = np.subtract(out["predicted"]["U"], row["auxiliaries"]["Uii"])
                out["effective_disposal"] = {
                    "minimum": float(min(d)),
                    "negative_samples": int(sum(d < -D_AUDIT_TOLERANCE)),
                }
            sealed_write(path, {**identity_fields, **out})
        return (pid, family, result["training_nmse"] if result else None)


def run(root: Path, source: Path, workers: int) -> None:
    plan = prepare(root, source)
    jobs = [
        (root, source, p["panel_id"], f)
        for p in plan["panels"]
        for f in p["candidates"]
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for item in pool.map(_run, jobs):
            print(item, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.root, args.source, args.workers)
