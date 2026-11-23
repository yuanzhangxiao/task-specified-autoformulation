"""Transfer one frozen nonnegative insulin production law into two complete models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autoformalism.expressions.compiler import SAFE_DIVISION_EPSILON
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.trajectory_function_fitting import _Rollout, _rows
from autoformalism.research.trajectory_function_revision import (
    _check_graph,
    assess_fixed,
)
from autoformalism.staged_topology import content_hash
from scripts import compare_positive_insulin_rates as production
from scripts import transfer_insulin_order_law as transfer
from scripts.construction_functions import file_hash

SCALAR = "03139ff00271c59e7689bbbc1ca0f6659c9eaf518482645a83732a1a1ec9c9d4"
SCALAR_REPLAY = "80e8b5f490ff4c4b63158eaf0a52672624c5fb1a8bb6f7b8ddde5870a3b98813"
RESTORING = "f31f28a3a99b8d669b054cef21b75a07fbf0aeaaf336d869091c2e3671501253"
LEVELS = (2, 4, 8, 16, 32)
SCALED_GRID = 1e-5
ATOL, RTOL = 1e-5, 1e-7
TARGETS = transfer.TARGETS


def _hat(channel, knots, i):
    def ramp(j, k):
        width = abs(knots[k] - knots[j])
        if width < SAFE_DIVISION_EPSILON:
            raise ValueError("Basis spacing is below portable division precision")
        numerator = (
            f"({channel}-({knots[j]!r}))" if k > j else f"(({knots[j]!r})-{channel})"
        )
        return f"min(1,max(0,{numerator}/{width!r}))"

    if i == 0:
        return ramp(1, 0)
    if i == len(knots) - 1:
        return ramp(i - 1, i)
    return f"min({ramp(i - 1, i)},{ramp(i + 1, i)})"


def export(parent, knots, selected):
    """Write a restricted RHS with fixed fold weights and both saturated end hats."""
    if set(knots) != set(production.SOURCES["EGP"]):
        raise ValueError("Expected all three prespecified production inputs")
    labels = production.coefficient_hats(knots)
    if selected.get("coefficient_hats") != labels:
        raise ValueError("Each coefficient must identify its channel and knot")
    weights = production.checked_weights(knots, selected)
    tau = float(selected["tau"])
    if np.any(weights < 0) or not np.isfinite(tau) or tau < SAFE_DIVISION_EPSILON:
        raise ValueError(
            "Expected nonnegative weights and a positive portable timescale"
        )
    form, graph = parent["formula"], parent["graph"]
    old = form["parameters"]
    parameters = {"pos_tau": tau}
    terms = []
    removed = {}
    for name in production.channel_order(knots):
        points = knots[name]
        if (
            not len(points)
            or not np.all(np.isfinite(points))
            or not np.all(np.diff(points) > 0)
        ):
            raise ValueError("Expected finite increasing fitting-fold knots")
        values = [
            float(w)
            for w, lab in zip(weights, labels, strict=True)
            if lab["channel"] == name
        ]
        if len(points) == 1 or all(w == values[0] for w in values):
            removed[name] = (
                "single fitting knot" if len(points) == 1 else "identical fixed weights"
            )
            if values[0] != 0:
                key = f"pos_{name}_flat"
                terms.append(key)
                parameters[key] = values[0]
            continue
        for i, w in enumerate(values):
            if w == 0:
                continue
            key = f"pos_{name}_{i}"
            terms.append(f"{key}*{_hat(name, points, i)}")
            parameters[key] = w
    if parameters.keys() & old.keys():
        raise ValueError("A new coefficient name collides with a parent parameter")
    rhs = "-I/pos_tau" + "".join("+" + term for term in terms)
    parser = RestrictedParser()
    carry = set()
    for name, text in [(k, v) for k, v in form["equations"].items() if k != "I"] + list(
        form["initializers"].items()
    ):
        carry.update(parser.parse(text, location=name).symbols)
    sources = sorted(parser.parse(rhs, location="I").symbols - parameters.keys())
    parameters.update({k: v for k, v in old.items() if k in carry})
    return {
        "graph": graph | {"sources": graph["sources"] | {"I": sources}},
        "formula": form
        | {"equations": form["equations"] | {"I": rhs}, "parameters": parameters},
        "constant_channels": removed,
        "production_invariant": (
            "Each cardinal hat is nonnegative over the entire real input axis. "
            "All stored weights are nonnegative and tau is positive. With these "
            "fixed coefficients any nonnegative initial insulin stays nonnegative."
        ),
        "original_insulin_rhs": form["equations"]["I"],
        "discarded_insulin_only_parameters": sorted(old.keys() - parameters.keys()),
        "knots": knots,
        "scalar_model": selected,
    }


def prepare(evidence, root):
    """Authenticate both parent vectors, the corrected basis and matched folds."""
    paired = evidence / "paired"
    rows = transfer.scalar.source(paired)
    intake = sealed_read(paired / "generation/intake.json")
    pos = evidence / "positive-rates-final"
    source = sealed_read(pos / "plan.json")
    sc = sealed_read(pos / "summary.json")
    sr = sealed_read(pos / "saved-replay.json")
    base_dir = evidence / "order-transfer"
    base = sealed_read(base_dir / "plan.json")
    bs = sealed_read(base_dir / "summary.json")
    cold = evidence / "aces-cold-folds/results"
    checks = sealed_read(cold / "summary.json")
    if (
        sc["artifact_sha256"] != SCALAR
        or sc["plan"] != source["artifact_sha256"]
        or sr["artifact_sha256"] != SCALAR_REPLAY
        or sr["summary"] != SCALAR
        or source["protocol"] != "positive-insulin-production-2"
        or source["train"] != transfer.scalar.TRAIN
        or bs["artifact_sha256"] != RESTORING
        or bs["plan"] != base["artifact_sha256"]
        or checks["artifact_sha256"] != transfer.COLD
        or base["cold_summary"] != transfer.COLD
        or base["folds"] != source["folds"]
        or source["folds"] != transfer.coupled.folds(rows.values())
    ):
        raise ValueError("Wrong scalar, baseline, cohort or fold identity")
    ids = []
    comparators = {}
    for j, m in enumerate(base["models"]):
        if m["arm"] != "restoring":
            continue
        slot, fold = m["slot"], m["fold"]
        key = f"{slot:03}-{fold}"
        split = source["folds"][fold - 1]
        archived = sealed_read(base_dir / f"{key}-restoring.json")
        cold_i = checks["checks"].index(m["cold_receipt"])
        fit = sealed_read(cold / "fits" / f"{cold_i:02}" / "result.json")
        check = sealed_read(cold / "checks" / f"{cold_i:02}" / "outcome.json")
        fitted = fit["options"][check["selected_option"]]["parameters"]
        if (
            archived["artifact_sha256"] != bs["results"][j]
            or archived["model"] != content_hash(m)
            or archived["plan"] != bs["plan"]
            or archived["status"] != "scored"
            or check["artifact_sha256"] != m["cold_receipt"]
            or check["fit"] != fit["artifact_sha256"]
            or check["scales"] != m["scales"]
            or set(check["parameters"]) != set(fitted)
            or m["formula"]["parameters"] != fitted
            or check["parameters"] != fitted
        ):
            raise ValueError("Restoring model or fitted vector differs from its score")
        s = sealed_read(pos / f"{fold}-EGP.json")
        if (
            s["artifact_sha256"] != sc["checkpoints"][str(fold)]["EGP"]
            or s["plan"] != sc["plan"]
            or s["split"] != str(fold)
            or s["source"] != "EGP"
            or production.calibrate(
                [rows[k] for k in split["fit"]], production.CHANNEL_ORDER
            )
            != s["knots"]
        ):
            raise ValueError("Positive model is bound to different knots or events")
        sel = s["arms"]["nonnegative"]
        scores = sel["scores"]
        if [x["event"] for x in scores] != split["held"] or not all(
            x["verified"] for x in scores
        ):
            raise ValueError("Scalar check missing or in disagreement")
        saved = next(
            r
            for r in sr["verified"]
            if (r["split"], r["sources"], r["weights"])
            == (str(fold), "EGP", "nonnegative")
        )
        if saved["scores_sha256"] != content_hash(scores):
            raise ValueError("Saved scalar forecast hashes disagree")
        record = export(m, s["knots"], sel["selected"])
        path = root / "models" / f"{key}.json"
        sealed_write(
            path,
            {
                "slot": slot,
                "fold": fold,
                "split": split,
                **record,
                "scales": m["scales"],
                "restoring_model_sha256": content_hash(m),
                "restoring_receipt": archived["artifact_sha256"],
                "cold_vector_receipt": m["cold_receipt"],
                "positive_receipt": s["artifact_sha256"],
                "positive_forecasts": scores,
                "validation_opened": False,
                "test_opened": False,
            },
        )
        ids.append({"name": key, "model": sealed_read(path)["artifact_sha256"]})
        comparators[key] = (m, archived)
    if sorted(comparators) != [f"{s:03}-{i}" for s in (2, 15) for i in (1, 2, 3)]:
        raise ValueError("Missing or duplicate parent/fold")
    plan = sealed_write(
        root / "plan.json",
        {
            "protocol": "complete-positive-egp-transfer-1",
            "scalar_summary": SCALAR,
            "scalar_replay": SCALAR_REPLAY,
            "restoring": RESTORING,
            "cold": transfer.COLD,
            "training": transfer.scalar.TRAIN,
            "models": ids,
            "levels": list(LEVELS),
            "max_scaled_grid_difference": SCALED_GRID,
            "scalar_atol": ATOL,
            "scalar_rtol": RTOL,
            "comparison": "Fixed positive EGP law versus its fitted restoring parent, "
            "using frozen fitting-fold scales and event/sample weights.",
            "code": file_hash(Path(__file__)),
            "optimizer_calls": 0,
            "llm_calls": 0,
            "validation_opened": False,
            "test_opened": False,
        },
    )
    return plan, rows, intake, comparators


def _predict(model, row, knots, steps):
    """Use only initial target readings and supplied forcing in the coupled rollout."""
    engine = _Rollout(model["graph"], model["formula"], TARGETS, steps)
    params = [model["formula"]["parameters"][n] for n in engine.parameters]
    r = _rows([row], "train")[0]
    _, (z, u, _) = engine.record(r)
    t = np.asarray(r.time)
    times = production.grid(row, knots)[1] if knots else t
    inputs = np.asarray([np.interp(times, t, values) for values in u])
    dt = np.diff(times).reshape(1, -1)
    [(y, _)] = engine.run([(engine._flow(len(times)), (z, inputs, dt))], params)
    return y[:, np.searchsorted(times, t)]


def _metrics(row, pred, scales):
    y = {k: np.asarray(row["targets"][k]) for k in TARGETS}
    curve = {
        "trajectory_id": row["trajectory_id"],
        "time": row["time"],
        "observed": y,
        "predicted": dict(zip(TARGETS, pred, strict=True)),
    }
    e = transfer.coupled.event_metrics(curve, scales)
    return e | {
        "negative_samples": {
            k: int(np.sum(v < -1e-10)) for k, v in curve["predicted"].items()
        },
        "minima": {k: float(min(v)) for k, v in curve["predicted"].items()},
        "max_abs_error": {
            k: float(max(abs(v - y[k]))) for k, v in curve["predicted"].items()
        },
        "forecast_sha256": content_hash(pred.tolist()),
    }


def compare_event(model, parent, archived, row, score):
    """Reproduce baseline SSE and verify the joined ODE on successively finer grids."""
    scales = model["scales"]
    base = _metrics(row, _predict(parent, row, None, 4), scales)
    if any(
        not np.isclose(base["sse"][k], archived["sse"][k], atol=1e-8, rtol=1e-8)
        for k in TARGETS
    ):
        raise ValueError("The restoring SSE differs from its verified archive")
    knots, selected = model["knots"], model["scalar_model"]
    exact = production.predict(row, knots, selected)
    scalar = production.independent(row, knots, selected)
    if (
        content_hash(exact.tolist()) != score["forecast_sha256"]
        or content_hash(scalar.tolist()) != score["independent_forecast_sha256"]
        or not np.allclose(exact, scalar, atol=production.ATOL, rtol=production.RTOL)
    ):
        raise ValueError("Corrected scalar forecast did not reproduce")
    logs = []
    previous = None
    for steps in LEVELS:
        pred = _predict(model, row, knots, steps)
        grid = (
            None
            if previous is None
            else float(
                np.max(
                    abs(previous - pred)
                    / np.array([scales[k] for k in TARGETS])[:, None]
                )
            )
        )
        difference = float(np.max(abs(pred[1] - exact)))
        good = np.allclose(pred[1], exact, atol=ATOL, rtol=RTOL) and np.allclose(
            pred[1], scalar, atol=ATOL, rtol=RTOL
        )
        logs.append(
            {
                "substeps": steps,
                "max_scaled_grid_difference": grid,
                "scalar_max_difference": difference,
            }
        )
        if grid is not None and grid <= SCALED_GRID and good:
            return base, _metrics(row, pred, scales) | {"levels": logs}
        previous = pred
    return base, {
        "event": row["trajectory_id"],
        "status": "numerical_check_unresolved",
        "levels": logs,
    }


def summarize(events):
    good = [e for e in events if e["status"] == "scored"]
    totals = transfer.coupled.aggregate(events, 15)
    for field, fn in (
        ("negative_samples", sum),
        ("minima", min),
        ("max_abs_error", max),
    ):
        totals[field] = (
            {k: fn(e[field][k] for e in good) for k in TARGETS} if good else {}
        )
    return totals


def run(root, plan, rows, intake, comparators, verify=False):
    ids = []
    arms = {2: {"restoring": [], "positive": []}, 15: {"restoring": [], "positive": []}}
    for label in plan["models"]:
        key = label["name"]
        model = sealed_read(root / "models" / f"{key}.json")
        if model["artifact_sha256"] != label["model"]:
            raise ValueError("Exported model differs from its plan")
        destination = root / "checks" / f"{key}.json"
        if verify or not destination.exists():
            parent, old = comparators[key]
            split = model["split"]
            case = transfer.fold_case(
                intake, model["slot"], [rows[n] for n in split["fit"]]
            )
            pub = _check_graph(
                model["graph"], case, "Check the fixed nonnegative production model."
            )
            form = assess_fixed(model["graph"], case, model["formula"])
            static = (
                pub["status"] == "topology_screen_passed"
                and form["status"] == "eligible"
            )
            checks = {a: [] for a in ("restoring", "positive")}
            refs = {e["event"]: e for e in old["events"]}
            for s in model["positive_forecasts"]:
                n = s["event"]
                try:
                    b, c = compare_event(model, parent, refs[n], rows[n], s)
                except (ArithmeticError, ValueError, RuntimeError) as exc:
                    b = {"event": n, "status": "unverified", "reason": str(exc)}
                    c = b.copy()
                if not static:
                    c["numerical_status"] = c["status"]
                    c["status"] = "public_check_unresolved"
                checks["restoring"].append(b)
                checks["positive"].append(c)
            sealed_write(
                destination,
                {
                    "plan": plan["artifact_sha256"],
                    "model": model["artifact_sha256"],
                    "events": checks,
                    "topology": pub["status"],
                    "topology_reason": pub.get("reason"),
                    "formula": form["status"],
                    "formula_reason": form.get("reason"),
                    "balance": form.get("balance_check"),
                },
            )
        result = sealed_read(destination)
        if (
            result["plan"] != plan["artifact_sha256"]
            or result["model"] != label["model"]
        ):
            raise ValueError("Scoring checkpoint refers to a different complete model")
        ids.append(result["artifact_sha256"])
        for arm, ev in result["events"].items():
            arms[model["slot"]][arm].extend(ev)
    summary = {}
    for slot, e in arms.items():
        summary[str(slot)] = {
            **{a: summarize(rows) for a, rows in e.items()},
            "positive_vs_restoring": transfer.paired_results(
                e["restoring"], e["positive"]
            ),
        }
    return sealed_write(
        root / "summary.json",
        {"plan": plan["artifact_sha256"], "checks": ids, "parents": summary},
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("prepare", "run", "verify"))
    cli.add_argument("--evidence-root", type=Path, required=True)
    cli.add_argument("--root", type=Path, required=True)
    args = cli.parse_args()
    evidence = args.evidence_root.resolve()
    root = args.root.resolve()
    if any(
        root.is_relative_to(evidence / n) or (evidence / n).is_relative_to(root)
        for n in ("paired", "order-transfer", "positive-rates-final", "aces-cold-folds")
    ):
        raise ValueError("Keep the six exports separate from all input receipts")
    with _lock(root):
        plan, rows, intake, comp = prepare(evidence, root)
        result = (
            plan
            if args.action == "prepare"
            else run(root, plan, rows, intake, comp, verify=args.action == "verify")
        )
    print(json.dumps({"action": args.action, "sha256": result["artifact_sha256"]}))


if __name__ == "__main__":
    main()
