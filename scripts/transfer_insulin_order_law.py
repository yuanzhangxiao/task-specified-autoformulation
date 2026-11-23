"""Replay fixed insulin law substitutions in two archived coupled training models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.trajectory_function_fitting import _check_grid, _rows
from autoformalism.research.trajectory_function_revision import (
    _check_graph,
    assess_fixed,
)
from autoformalism.staged_topology import content_hash
from scripts import compare_insulin_order_repairs as scalar
from scripts import t2_challenge_folds as coupled
from scripts.construction_functions import file_hash

COLD = "db576fc54abde9d47624c8695bcc2978daaa95e8182d53c7850128b214226363"
SINGLE = "333339e713c757ef4b04dfb90ff4b8074ee0a913d9f8bc66a662e5733621d14b"
ARMS = ("restoring", "reference", "EGP")
TARGETS = ("Gp", "I", "U")


def transplant(graph, template, fitted, arm, selected):
    """Keep every coefficient used by any untouched equation or initial map."""
    if set(fitted) != set(template["parameters"]):
        raise ValueError("Archived vector does not match the parent template")
    if arm == "restoring":
        return graph, template | {"parameters": fitted}
    if arm not in ARMS or selected["tau"] <= 0 or selected["tau_z"] is not None:
        raise ValueError("Only the two registered scalar forms are transferable")
    rhs = "-I/order_tau+order_offset+order_gt*Gt+order_ins*" + scalar.DRUG
    names = ["order_offset", "order_gt", "order_ins"]
    sources = {"I", "Gt", scalar.DRUG}
    if arm == "EGP":
        names.append("order_egp")
        rhs += "+order_egp*EGP"
        sources.add("EGP")
    pars = dict(zip(names, selected["gains"], strict=True)) | {
        "order_tau": selected["tau"]
    }
    if set(pars) & fitted.keys():
        raise ValueError("A projected parameter would shadow an archived coefficient")
    equations = template["equations"] | {"I": rhs}
    parser = RestrictedParser()
    symbols = set()
    for name, expr in list(template["initializers"].items()) + [
        (k, v) for k, v in equations.items() if k != "I"
    ]:
        symbols.update(parser.parse(expr, location=name).symbols)
    pars.update({k: v for k, v in fitted.items() if k in symbols})
    return (
        graph | {"sources": graph["sources"] | {"I": sorted(sources)}},
        template | {"equations": equations, "parameters": pars},
    )


def fold_case(intake, slot, rows):
    """Repeat public topology checks on only the fitting events for these vectors."""
    case = intake["cases"][intake["parents"][str(slot)]["case_id"]]
    e = {k: v for k, v in case["evidence"].items() if k != "sha256"}
    e["trajectories"] = rows
    return case | {"evidence": e | {"sha256": content_hash(e)}}


def prepare(cold, single, paired, root):
    """Bind both fit receipts for each fold and freeze the exact exported models."""
    data = scalar.source(paired)
    intake = sealed_read(paired / "generation" / "intake.json")
    c, h = sealed_read(cold / "plan.json"), sealed_read(cold / "summary.json")
    s, t = sealed_read(single / "plan.json"), sealed_read(single / "summary.json")
    if (
        h["artifact_sha256"] != COLD
        or t["artifact_sha256"] != SINGLE
        or h["plan"] != c["artifact_sha256"]
        or t["plan"] != s["artifact_sha256"]
        or s["training_sha256"] != scalar.TRAIN
        or s["source_plan"] != scalar.SOURCE
        or c["folds"] != s["folds"]
        or s["folds"] != coupled.folds(data.values())
    ):
        raise ValueError("Unexpected source identity, event assignment or protocol")
    models = []
    outcomes = {}
    for j, m in enumerate(c["models"]):
        if m["arm"] != "restoring" or m["slot"] not in (2, 15):
            continue
        i = m["fold"] + 1
        split = c["folds"][i - 1]
        r = sealed_read(cold / "checks" / f"{j:02}" / "outcome.json")
        fitted = sealed_read(cold / "fits" / f"{j:02}" / "result.json")
        if (
            r["artifact_sha256"] != h["checks"][j]
            or r["fit"] != fitted["artifact_sha256"]
            or r["plan"] != h["plan"]
            or r["index"] != j
            or r["parameters"] != fitted["options"][r["selected_option"]]["parameters"]
            or r["arm"] != "restoring"
            or r["status"] != "fitted"
            or {x["event"] for x in r["events"]} != set(split["held"])
        ):
            raise ValueError("The archived restoring coefficients or scored IDs differ")
        outcomes[i, m["slot"]] = r["events"]
        for arm in ARMS:
            sf, key = None, None
            if arm != "restoring":
                v = sealed_read(single / f"{i}-{arm}.json")
                key = v["artifact_sha256"]
                if (
                    key != t["arms"][arm]["fits"][str(i)]
                    or v["plan"] != t["plan"]
                    or v["split"] != str(i)
                    or v["arm"] != arm
                    or [x["event"] for x in v["fit"]] != split["fit"]
                    or [x["event"] for x in v["held"]] != split["held"]
                ):
                    raise ValueError("Scalar law came from a different fitting fold")
                sf = v["selected"]
            graph, form = transplant(m["graph"], m["formula"], r["parameters"], arm, sf)
            models.append(
                {
                    "fold": i,
                    "slot": m["slot"],
                    "arm": arm,
                    "cold_receipt": r["artifact_sha256"],
                    "scalar_receipt": key,
                    "graph": graph,
                    "formula": form,
                    "scalar": sf,
                    "scales": r["scales"],
                }
            )
    if len(models) != 18 or len(outcomes) != 6:
        raise ValueError("Incomplete parent and fold panel")
    return (
        sealed_write(
            root / "plan.json",
            {
                "protocol": "t2-order-law-transfer-1",
                "cold_summary": COLD,
                "scalar_summary": SINGLE,
                "source_training": scalar.TRAIN,
                "models": models,
                "folds": s["folds"],
                "code": file_hash(Path(__file__)),
                "policy": (
                    "Replay all registered models. Report public topology and formula "
                    "checks using fitting events only. Four-substep held grid supplies "
                    "the outcome "
                    "after the normal two/four verification. Reproduce archived parent "
                    "SSE with atol=1e-8 and rtol=1e-8. For transfers, verify I against "
                    "independent RK4 at each held knot, atol=1e-4 + rtol=1e-7. "
                    "Pool fold-normalized squared errors with equal target weights and "
                    "count strictly improved matched events. No refitting or arm "
                    "selection."
                ),
                "optimizer_calls": 0,
                "llm_calls": 0,
                "validation_opened": False,
                "test_opened": False,
            },
        ),
        data,
        intake,
        outcomes,
    )


def replay(m, case, held, archived):
    """Run standard expression, topology and numerical checks with fixed vectors."""
    g, f = m["graph"], m["formula"]
    public = _check_graph(
        g, case, "Check public mechanisms for a registered insulin transfer."
    )
    check = assess_fixed(g, case, f)
    gates = {
        "topology": public["status"],
        "formula": check["status"],
        "balance": check.get("balance_check"),
        "topology_reason": public.get("reason"),
        "formula_reason": check.get("reason"),
    }
    if check["status"] != "eligible" or public["status"] != "topology_screen_passed":
        return {"status": "public_gate_failed", "gates": gates, "events": []}
    scales = m["scales"]
    scored = _check_grid(
        g,
        f,
        _rows(held, "train"),
        f["parameters"],
        TARGETS,
        np.array([scales[n] for n in TARGETS]),
        curves=True,
    )
    grid = scored.get("maximum_scaled_grid_difference")
    result = {
        "gates": gates,
        "grid_difference_scaled": grid,
        "status": scored["status"],
        "events": [],
    }
    if scored["status"] != "scored":
        return result | {"reason": scored.get("reason")}
    previous = {x["event"]: x for x in archived}
    rows = {r["trajectory_id"]: r for r in held}
    for cr in scored["curves"]:
        e = coupled.event_metrics(cr, scales)
        name = e["event"]
        predicted = {k: np.asarray(v) for k, v in cr["predicted"].items()}
        if m["arm"] == "restoring":
            diff = max(abs(e["sse"][k] - previous[name]["sse"][k]) for k in TARGETS)
            verified = all(
                np.isclose(e["sse"][k], previous[name]["sse"][k], atol=1e-8, rtol=1e-8)
                for k in TARGETS
            )
            e["archive_max_sse_difference"] = diff
        else:
            expected = scalar.independent(rows[name], m["arm"], m["scalar"])
            e["max_scalar_I_difference"] = float(max(abs(predicted["I"] - expected)))
            verified = np.allclose(predicted["I"], expected, atol=1e-4, rtol=1e-7)
        e["negative_samples"] = {
            k: int(np.sum(y < -1e-10)) for k, y in predicted.items()
        }
        e["minima"] = {k: float(min(y)) for k, y in predicted.items()}
        if not verified:
            e["status"] = "reference_replay_differs"
            result["status"] = e["status"]
        result["events"].append(e)
    return result


def paired_results(original, new):
    """Compare only verified events scored in both arms, without pooling parents."""
    base = {e["event"]: e for e in original if e["status"] == "scored"}
    common = [
        (base[e["event"]], e)
        for e in new
        if e["status"] == "scored" and e["event"] in base
    ]
    return {
        "compared": len(common),
        "lower_pooled": sum(b["pooled_nmse"] < a["pooled_nmse"] for a, b in common),
        "lower_by_target": {
            k: sum(b["sse"][k] < a["sse"][k] for a, b in common) for k in TARGETS
        },
    }


def run(root, plan, data, intake, outcomes):
    groups = {}
    shas = []
    for m in plan["models"]:
        i, slot, arm = m["fold"], m["slot"], m["arm"]
        split = plan["folds"][i - 1]
        path = root / f"{slot:03}-{i}-{arm}.json"
        if path.exists():
            r = sealed_read(path)
            if r["plan"] != plan["artifact_sha256"] or r["model"] != content_hash(m):
                raise ValueError("Result checkpoint differs from the frozen equations")
        else:
            case = fold_case(intake, slot, [data[n] for n in split["fit"]])
            o = replay(m, case, [data[n] for n in split["held"]], outcomes[i, slot])
            r = sealed_write(
                path,
                {
                    "plan": plan["artifact_sha256"],
                    "model": content_hash(m),
                    "fold": i,
                    "slot": slot,
                    "arm": arm,
                    **o,
                },
            )
        shas.append(r["artifact_sha256"])
        events = r["events"] or [
            {"event": n, "status": r["status"]} for n in split["held"]
        ]
        groups.setdefault(slot, {}).setdefault(arm, []).extend(events)
    summary = {}
    for slot, arms in groups.items():
        summary[slot] = {}
        for arm, events in arms.items():
            good = [e for e in events if e["status"] == "scored"]
            summary[slot][arm] = {
                **coupled.aggregate(events, 15),
                "against_restoring": paired_results(arms["restoring"], events),
                "against_scalar_reference": paired_results(arms["reference"], events),
                "minima": {k: min(e["minima"][k] for e in good) for k in TARGETS}
                if good
                else {},
                "negative_samples": {
                    k: sum(e["negative_samples"][k] for e in good) for k in TARGETS
                },
            }
    return sealed_write(
        root / "summary.json",
        {
            "plan": plan["artifact_sha256"],
            "groups": {str(n): r for n, r in summary.items()},
            "results": shas,
        },
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("prepare", "run"))
    for name in ("cold", "single", "paired", "root"):
        cli.add_argument("--" + name, type=Path, required=True)
    a = cli.parse_args()
    if any(
        a.root.resolve().is_relative_to(p.resolve())
        for p in (a.cold, a.single, a.paired)
    ):
        raise ValueError("Keep results outside all source roots")
    with _lock(a.root):
        p, d, i, o = prepare(a.cold, a.single, a.paired, a.root)
        answer = (
            {"plan": p["artifact_sha256"]}
            if a.action == "prepare"
            else run(a.root, p, d, i, o)
        )
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
