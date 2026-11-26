"""Select disposal equations on four training events and replay frozen choices."""
# ruff: noqa: E501 -- The report embeds human-readable HTML paragraphs.

from __future__ import annotations

import argparse
import base64
import math
from collections import Counter
from html import escape
from inspect import getsource
from io import BytesIO
from pathlib import Path
from statistics import mean

import numpy as np

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import trajectory_function_fitting as rollout
from autoformalism.staged_topology import content_hash
from scripts.construction_functions import file_hash
from scripts.matched_equation_experiment import DEFAULT as SOURCE
from scripts.matched_equation_experiment import _registered
from scripts.report_hidden_form_forecasts import _curve, _read_plan, _table
from scripts.topology_forecast_fit import INPUTS, TRAIN, VALIDATION

ROOT = SOURCE.parent / "t2-balanced-disposal-20261126"
TARGETS = ("Gp", "I", "U")
# The first eligible law at numerical equality is chosen.
LAWS = ("action", "plasma", "tissue", "both", "interaction")
CONTRASTS = (
    ("free", "balanced"),
    ("free_native", "balanced_native"),
    ("free_native", "balanced"),
)
POLICIES = ("free", "balanced", "free_native", "balanced_native")


def menu(panel: dict) -> dict[str, list[str]]:
    """Preserve the same law comparison for both portfolio arms."""
    candidates = panel["candidates"]
    arms = {}
    for arm in ("free", "balanced"):
        members = [(n, c) for n, c in candidates.items() if c["arm"] == arm]
        options = {c["law"]: n for n, c in members}
        if set(options) != set(LAWS) or len(members) != len(LAWS):
            raise ValueError("Expected a matched pair of five-law portfolios")
        arms[arm] = [
            options[law]
            for law in LAWS
            if candidates[options[law]]["dependency_eligible"]
        ]
    native_law = candidates[panel["native_family"]]["law"]
    return {
        **arms,
        **{
            arm + "_native": [
                next(
                    n
                    for n, c in candidates.items()
                    if c["law"] == native_law and c["arm"] == arm
                )
            ]
            for arm in arms
        },
    }


def event_metrics(record: dict) -> dict:
    """Use the fit-scale post-initial normalization for all targets and policies."""
    if record["status"] != "scored":
        return {"status": record["status"], "nmse": None}
    scores = record["per_target_nmse"]
    if set(scores) != set(TARGETS) or any(
        not math.isfinite(v) or v < 0 for v in scores.values()
    ):
        raise ValueError("Invalid channel scores")
    nmse = mean(scores.values())
    if not math.isclose(nmse, record["nmse"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Wrong three-target event NMSE")
    return {"status": "scored", "nmse": nmse, **scores}


def choose(options: list[str], metrics: dict[str, dict]) -> dict:
    """Use all four selection outcomes; an invalid event disqualifies the slot."""
    if len(metrics) != 4 or any(not eid.startswith("train_") for eid in metrics):
        raise ValueError("Need four training-selection events")
    scores = {}
    for family in options:
        errors = [m[family]["nmse"] for m in metrics.values()]
        if all(v is not None for v in errors):
            scores[family] = mean(errors)
    winner = next(
        (
            name
            for name in options
            if name in scores
            and math.isclose(
                scores[name], min(scores.values()), rel_tol=1e-10, abs_tol=1e-12
            )
        ),
        None,
    )
    return {"family": winner, "selection_nmse": scores}


def _endpoint_receipt(root, plan, panel, family):
    name = panel["panel_id"]
    record = sealed_read(root / "fits" / name / family / "best.json")
    if (record["plan"], record["panel_id"], record["family"]) != (
        plan["artifact_sha256"],
        name,
        family,
    ):
        raise ValueError("Unmatched fit endpoint")
    return record


def selection_plan(root: Path) -> dict:
    plan = _read_plan(root)
    with _lock(root / "selection-plan-lock"):
        return sealed_write(
            root / "selection_plan.json",
            {
                "protocol": "all-four-disposal-selection-1",
                "fit_plan": plan["artifact_sha256"],
                "arms": {p["panel_id"]: menu(p) for p in plan["panels"]},
                "selection": "Minimize mean whole-model NMSE on all four selection events. "
                "A failed event or incomplete fit disqualifies the slot.",
                "metric": "Per-target squared error after t0, divided by fitting-set "
                "target variance, averaged equally across targets, events and rotations.",
                "ties": list(LAWS),
                "comparisons": list(map(list, CONTRASTS)),
                "selector_sha256": content_hash(
                    getsource(choose) + getsource(event_metrics)
                ),
                "eligibility": "Require all scoring events to pass the grid and public "
                "nonnegativity checks. Report all abstentions and per-arm coverage.",
            },
        )


def select(root: Path) -> dict:
    plan = _read_plan(root)
    policy = selection_plan(root)
    receipts = []
    endpoints = {}
    with _lock(root / "choices-lock"):
        for panel in plan["panels"]:
            name = panel["panel_id"]
            arms = policy["arms"][name]
            families = list(panel["candidates"])
            endpoints[name] = {
                f: _endpoint_receipt(root, plan, panel, f)["artifact_sha256"]
                for f in families
            }
            for index, rotation in enumerate(panel["choice_rotations"]):
                curves = {
                    eid: {f: _curve(root, plan, name, f, eid) for f in families}
                    for eid in rotation["selection_ids"]
                }
                metrics = {
                    e: {f: event_metrics(c) for f, c in cs.items()}
                    for e, cs in curves.items()
                }
                choices = {arm: choose(opts, metrics) for arm, opts in arms.items()}
                receipt = sealed_write(
                    root / "choices" / name / f"rotation{index}.json",
                    {
                        "panel": name,
                        "rotation": index,
                        "policy": policy["artifact_sha256"],
                        "ids": rotation["selection_ids"],
                        "choices": choices,
                        "curves": {
                            e: {f: c["artifact_sha256"] for f, c in cs.items()}
                            for e, cs in curves.items()
                        },
                    },
                )
                receipts.append(receipt["artifact_sha256"])
        return sealed_write(
            root / "choices.json",
            {
                "plan": plan["artifact_sha256"],
                "policy": policy["artifact_sha256"],
                "endpoints": endpoints,
                "receipts": receipts,
            },
        )


def summaries(rows: list[dict]) -> dict:
    """Score each planned comparison on both arms' qualified whole rotations."""
    result = {}
    for ref, alt in CONTRASTS:
        common = [r for r in rows if r["scores"][ref] and r["scores"][alt]]
        pairs = []
        for r in rows:
            for e in r["events"]:
                item = {a: r["metrics"][e].get(r["choices"][a]) for a in (ref, alt)}
                if all(v and v["nmse"] is not None for v in item.values()):
                    pairs.append(item)
        values = {
            arm: {
                k: mean(r["scores"][arm][k] for r in common) for k in ("nmse", *TARGETS)
            }
            if common
            else None
            for arm in (ref, alt)
        }
        gains = [r["scores"][ref]["nmse"] - r["scores"][alt]["nmse"] for r in common]
        result[ref + "->" + alt] = {
            "common_rotations": len(common),
            "planned": len(rows),
            "coverage": {
                arm: sum(r["scores"][arm] is not None for r in rows)
                for arm in (ref, alt)
            },
            "means": values,
            "paired_event_comparisons": len(pairs),
            "event_means": {
                arm: mean(v[arm]["nmse"] for v in pairs) if pairs else None
                for arm in (ref, alt)
            },
            "wins": sum(g > 1e-10 for g in gains),
            "losses": sum(g < -1e-10 for g in gains),
            "ties": sum(abs(g) <= 1e-10 for g in gains),
        }
    return result


def _rotation(panel, index, selected, curves):
    choices = {a: d["family"] for a, d in selected["choices"].items()}
    metrics = {
        e: {f: event_metrics(c) for f, c in cs.items()} for e, cs in curves.items()
    }
    scores = {}
    for arm, family in choices.items():
        vals = [m.get(family) for m in metrics.values()]
        scores[arm] = (
            {k: mean(v[k] for v in vals) for k in ("nmse", *TARGETS)}
            if all(v and v["nmse"] is not None for v in vals)
            else None
        )
    return {
        "panel": panel["panel_id"],
        "graph": panel["graph_name"],
        "rotation": index,
        "selection_receipt": selected["artifact_sha256"],
        "choices": choices,
        "events": list(curves),
        "metrics": metrics,
        "scores": scores,
    }


def _manifest(root, plan):
    manifest = sealed_read(root / "choices.json")
    policy = selection_plan(root)
    if (manifest["plan"], manifest["policy"], len(manifest["receipts"])) != (
        plan["artifact_sha256"],
        policy["artifact_sha256"],
        len(plan["panels"]) * 2,
    ):
        raise ValueError("Unmatched choice manifest")
    return manifest


def external_plan(root: Path) -> dict:
    """Register choices and fixed endpoints before loading official validation."""
    fitplan = _read_plan(root)
    manifest = _manifest(root, fitplan)
    with _lock(root / "official-validation" / "plan-lock"):
        return sealed_write(
            root / "official-validation" / "plan.json",
            {
                "protocol": "frozen-balanced-disposal-validation-1",
                "fit_plan": fitplan["artifact_sha256"],
                "choices": manifest["artifact_sha256"],
                "train_release": TRAIN,
                "validation_release": VALIDATION,
                "targets": list(TARGETS),
                "evaluator": content_hash(
                    getsource(_validation_curves)
                    + getsource(_rotation)
                    + getsource(summaries)
                    + getsource(event_metrics)
                ),
                "rollout_source": file_hash(Path(rollout.__file__)),
            },
        )


def _validation_curves(root, plan, panel, rows, families, manifest, experiment_id):
    from scripts.run_t2_balanced_disposal import load_endpoint

    out = {r["trajectory_id"]: {} for r in rows}
    for family in sorted(families):
        graph, formula, fit, endpoint = load_endpoint(root, plan, panel, family)
        if manifest["endpoints"][panel["panel_id"]][family] != endpoint:
            raise ValueError("Selected family's fit changed after selection")
        if fit is None:
            raise ValueError("No fitted endpoint for a selected equation")
        for row in rows:
            eid = row["trajectory_id"]
            path = (
                root
                / "official-validation"
                / "curves"
                / panel["panel_id"]
                / family
                / f"{eid}.json"
            )
            ids = {
                "plan": experiment_id,
                "fit": endpoint,
                "panel": panel["panel_id"],
                "family": family,
                "event": eid,
            }
            if path.exists():
                current = sealed_read(path)
                if any(current[k] != v for k, v in ids.items()):
                    raise ValueError("Unmatched external forecast checkpoint")
            else:
                result = rollout.score_formula(graph, formula, fit, [row])
                detail = {"status": result["status"], "nmse": None}
                if result["status"] == "scored":
                    curve = result["curves"][0]
                    err = {
                        k: float(
                            np.mean(
                                np.square(
                                    (
                                        np.array(curve["observed"][k][1:])
                                        - curve["predicted"][k][1:]
                                    )
                                    / fit["scales"][k]
                                )
                            )
                        )
                        for k in TARGETS
                    }
                    valid = all(
                        result["negative_samples"][k] == 0
                        for k in plan["nonnegative_targets"]
                    )
                    component = np.subtract(
                        curve["predicted"]["U"], row["auxiliaries"]["Uii"]
                    )
                    detail = {
                        **curve,
                        "status": "scored" if valid else "negative_forecast",
                        "nmse": mean(err.values()) if valid else None,
                        "per_target_nmse": err,
                        "scales": fit["scales"],
                        "negative_samples": result["negative_samples"],
                        "effective_disposal_min": float(component.min()),
                        "negative_disposal_samples": int((component < -1e-9).sum()),
                    }
                current = sealed_write(path, {**ids, **detail})
            out[eid][family] = current
    return out


def _write_scores(path: Path, result: dict) -> dict:
    """Keep existing identities when only the order of diagnostic entries differed."""

    def by_event(record):
        return (record["panel"], record["family"], record["event"])

    for key in ("failures", "negative_disposal_forecasts"):
        result[key] = sorted(result[key], key=by_event)
    if path.exists():
        saved = sealed_read(path)
        for key in ("failures", "negative_disposal_forecasts"):
            if sorted(saved[key], key=by_event) == result[key]:
                result[key] = saved[key]
    return sealed_write(path, result)


def score(root: Path, source: Path, *, official: bool) -> dict:
    plan = _read_plan(root)
    manifest = _manifest(root, plan)
    destination = root / ("official-validation" if official else "omitted-training")
    with _lock(destination / "score-lock"):
        if official:
            if not (destination / "plan.json").exists():
                raise ValueError(
                    "Freeze the validation plan before accessing validation"
                )
            ext = external_plan(root)
            release = _registered(INPUTS / "validation.json", ext["validation_release"])
            parent = _registered(source / "plan.json", plan["source_plan"])
            if (
                release["training_intake_sha256"],
                release["split"],
                parent["registered_training"],
            ) != (TRAIN, "val", TRAIN):
                raise ValueError("Validation release does not match training data")
            data = release["rows"][parent["dataset"]]
            ids = [r.trajectory_id for r in rollout._rows(data, "val")]
        records = []
        failed = []
        negatives = []
        for pi, panel in enumerate(plan["panels"]):
            name = panel["panel_id"]
            decs = []
            for i, rot in enumerate(panel["choice_rotations"]):
                dec = _registered(
                    root / "choices" / name / f"rotation{i}.json",
                    manifest["receipts"][2 * pi + i],
                )
                if (dec["panel"], dec["rotation"], dec["ids"], dec["policy"]) != (
                    name,
                    i,
                    rot["selection_ids"],
                    manifest["policy"],
                ):
                    raise ValueError("Rotation decision identity changed")
                decs.append(dec)
            selected_families = {
                v["family"]
                for d in decs
                for v in d["choices"].values()
                if v["family"] is not None
            }
            if official:
                curves = _validation_curves(
                    root,
                    plan,
                    panel,
                    data,
                    selected_families,
                    manifest,
                    ext["artifact_sha256"],
                )
            else:
                curves = {
                    e: {f: _curve(root, plan, name, f, e) for f in selected_families}
                    for rot in panel["choice_rotations"]
                    for e in rot["held_ids"]
                }
            for e, fs in curves.items():
                for f, c in fs.items():
                    if c["status"] != "scored":
                        failed.append(
                            {
                                "panel": name,
                                "family": f,
                                "event": e,
                                "status": c["status"],
                                "negative_samples": c.get("negative_samples"),
                            }
                        )
                    if official and c.get("negative_disposal_samples", 0) > 0:
                        negatives.append(
                            {
                                "panel": name,
                                "family": f,
                                "event": e,
                                "minimum_disposal": c["effective_disposal_min"],
                            }
                        )
            for i, dec in enumerate(decs):
                subset = ids if official else panel["choice_rotations"][i]["held_ids"]
                records.append(_rotation(panel, i, dec, {e: curves[e] for e in subset}))
        return _write_scores(
            destination / "summary.json",
            {
                "choices": manifest["artifact_sha256"],
                "split": "val" if official else "inner",
                "rows": records,
                "comparisons": summaries(records),
                "failures": failed,
                "negative_disposal_forecasts": negatives,
            },
        )


def _comparison_table(report):
    titles = {
        "free->balanced": "Five matched laws per arm",
        "free_native->balanced_native": "Original source set",
        "free_native->balanced": "Five balanced laws vs original free law",
    }

    def fmt(x):
        return "—" if x is None else f"{x:.5f}"

    rows = []
    for key, c in report["comparisons"].items():
        ref, alt = key.split("->")
        a, b = (c["means"][arm] or {} for arm in (ref, alt))
        rows.append(
            [
                titles[key],
                f"{c['coverage'][ref]}/{c['planned']}",
                f"{c['coverage'][alt]}/{c['planned']}",
                c["common_rotations"],
                fmt(a.get("nmse")),
                fmt(b.get("nmse")),
                fmt(a.get("U")),
                fmt(b.get("U")),
                f"{c['wins']}/{c['ties']}/{c['losses']}",
            ]
        )
    return _table(
        [
            "Comparison",
            "Free covered",
            "Balanced covered",
            "Shared rotations",
            "Free NMSE",
            "Balanced NMSE",
            "Free U NMSE",
            "Balanced U NMSE",
            "Balanced lower / tie / higher",
        ],
        rows,
    )


def _figure(root, results):
    import matplotlib as mpl
    from matplotlib.figure import Figure

    plan = sealed_read(root / "plan.json")
    official = results["split"] == "val"
    for row in results["rows"]:
        if not row["scores"]["balanced"] or not row["scores"]["free"]:
            continue
        e = row["events"][0]
        pid = row["panel"]
        fams = [row["choices"][arm] for arm in ("free", "balanced")]
        curves = []
        for f in fams:
            c = (
                sealed_read(
                    root / "official-validation" / "curves" / pid / f / f"{e}.json"
                )
                if official
                else _curve(root, plan, pid, f, e)
            )
            curves.append(c)
        if any(curves[0][k] != curves[1][k] for k in ("time", "observed")):
            raise ValueError("Two policies received different events")
        with mpl.rc_context({"svg.hashsalt": "balanced-disposal", "font.size": 10}):
            fig = Figure(figsize=(7.5, 5.1), layout="constrained")
            axes = fig.subplots(2, 1, sharex=True)
            t = curves[0]["time"]
            for ax, target in zip(axes, ("U", "I"), strict=True):
                ax.plot(
                    t,
                    curves[0]["observed"][target],
                    color="#263743",
                    linewidth=2.4,
                    label="Observed",
                )
                for c, color, ls, label in zip(
                    curves,
                    ("#348464", "#bc5b20"),
                    ("--", ":"),
                    ("Free", "Balanced"),
                    strict=True,
                ):
                    ax.plot(
                        t,
                        c["predicted"][target],
                        color=color,
                        linestyle=ls,
                        linewidth=1.7,
                        label=label,
                    )
                ax.spines[["right", "top"]].set_visible(False)
            axes[0].legend(frameon=False, ncol=3)
            axes[0].set_ylabel("U (mg/kg/min)")
            axes[1].set(xlabel="Time (min)", ylabel="Insulin (pmol/L)")
            buf = BytesIO()
            fig.savefig(buf, format="svg", metadata={"Date": None})
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        caption = (
            f"First common eligible rotation ({pid}, rotation {row['rotation']}) "
            f"and first event by ID ({e}). Free selected {fams[0]}. "
            f"Balanced selected {fams[1]}."
        )
        return (
            f"<figure><img alt='Observed and forecast disposal and insulin on "
            f"{escape(e)}' src='data:image/svg+xml;base64,{data}'>"
            f"<figcaption>{escape(caption)}</figcaption></figure>"
        )
    return "<p>No rotation qualified in both arms for this split.</p>"


def _dependency_table(root, plan):
    """Show how exact equal-argument comparisons treat measured and hidden action."""
    rows = []
    for panel in plan["panels"]:
        pid = panel["panel_id"]
        screen = _registered(root / "screens" / f"{pid}.json", panel["screen"])[
            "screen"
        ]
        for name, source in (
            ("Measured insulin alone", "measured_candidates"),
            ("Latent action alone", "hidden_candidates"),
        ):
            case = screen[source]["I" if source == "measured_candidates" else "I_act"]
            rows.append(
                [
                    pid,
                    name,
                    case["repeated_argument_groups"],
                    case["conflicting_groups"],
                    f"{case['minimum_uniform_error']:.5f}",
                ]
            )
    return _table(
        [
            "Fit panel",
            "D depends on",
            "Repeated argument groups",
            "Conflicting groups",
            "Minimum maximum error (mg/kg/min)",
        ],
        rows,
    )


def _secondary(report):
    stats = report["comparisons"]["free->balanced"]
    if not stats["paired_event_comparisons"]:
        return "There were no common individually qualified event forecasts."
    vals = stats["event_means"]
    n = stats["paired_event_comparisons"]
    total = sum(len(r["events"]) for r in report["rows"])
    return (
        f"Considering individual events, the two portfolios both qualified on "
        f"{n}/{total} event-rotation forecasts. Mean NMSE was {vals['free']:.5f} "
        f"for the free portfolio and {vals['balanced']:.5f} for the balanced portfolio."
    )


def render(root: Path) -> Path:
    plan = _read_plan(root)
    manifest = _manifest(root, plan)
    inner = sealed_read(root / "omitted-training/summary.json")
    outer = sealed_read(root / "official-validation/summary.json")
    if (
        inner["choices"] != manifest["artifact_sha256"]
        or outer["choices"] != inner["choices"]
    ):
        raise ValueError("Reports do not refer to the same frozen choices")
    per_graph = []
    for graph in sorted({r["graph"] for r in outer["rows"]}):
        rows = [r for r in outer["rows"] if r["graph"] == graph]
        c = summaries(rows)["free->balanced"]
        for arm in ("free", "balanced"):
            vals = c["means"][arm] or {}
            per_graph.append(
                [
                    graph,
                    arm,
                    c["common_rotations"],
                    *[
                        f"{vals[k]:.5f}" if k in vals else "—"
                        for k in ("nmse", *TARGETS)
                    ],
                ]
            )
    choices = []
    for r in outer["rows"]:
        p = next(p for p in plan["panels"] if p["panel_id"] == r["panel"])
        for arm in POLICIES[:2]:
            name = r["choices"][arm]
            fit = predictor = None
            receipt = sealed_read(
                root / "choices" / r["panel"] / f"rotation{r['rotation']}.json"
            )
            eligible = receipt["choices"][arm]["selection_nmse"]
            if name:
                candidate = p["candidates"][name]
                fit = sealed_read(root / "fits" / r["panel"] / name / "best.json")
                predictor = candidate["formula"]["equations"]["U"]
            choices.append(
                [
                    r["panel"],
                    r["rotation"],
                    arm,
                    name or "Abstained",
                    "—" if fit is None else f"{fit['training_nmse']:.4f}",
                    "—" if name is None else f"{eligible[name]:.4f}",
                    "—"
                    if r["scores"][arm] is None
                    else f"{r['scores'][arm]['nmse']:.4f}",
                    predictor or "—",
                ]
            )
    count = Counter(x["status"] for x in outer["failures"])
    summary = outer["comparisons"]["free->balanced"]
    mean_free = summary["means"]["free"]
    mean_bal = summary["means"]["balanced"]
    if mean_free is None:
        top = "No rotation yielded qualified forecasts from both portfolios on all validation events."
    else:
        v = mean_bal["nmse"] - mean_free["nmse"]
        top = (
            f"On {summary['common_rotations']} shared rotations, official validation "
            f"NMSE was {mean_free['nmse']:.5f} for the free readout and {mean_bal['nmse']:.5f} "
            f"for U = Uii + D (balanced minus free {v:+.5f})."
        )
    count_text = (
        ", ".join(f"{n} {s.replace('_', ' ')}" for s, n in sorted(count.items())) or "0"
    )
    nval = len(outer["rows"][0]["events"])
    budget = plan["per_start_budget"]
    attempted = sum(
        len(sealed_read(file)["starts"])
        for file in (root / "fits").glob("*/*/best.json")
    )
    w = _registered(
        root / "screens/graph02-fold0.json",
        next(p["screen"] for p in plan["panels"] if p["panel_id"] == "graph02-fold0"),
    )["screen"]["measured_candidates"]["I"]["witness"]
    witness = (
        f"In {w['events'][0]} and {w['events'][1]} at t = 0, both insulin values were "
        f"{w['measured_arguments']['I']:.5f} pmol/L. D was {w['D'][0]:.5f} and "
        f"{w['D'][1]:.5f} mg/kg/min."
    )
    fn = root / "REPORT.html"
    fn.write_text(
        f"""<!doctype html><html lang="en"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>T2 disposal and whole-event forecasts</title><style>body{{font:16px/1.5 system-ui,sans-serif;color:#20343d;max-width:73rem;margin:2rem auto;padding:0 1rem}}p{{max-width:85ch}}h1,h2{{line-height:1.18}}h2{{margin-top:2rem}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;font-size:.91rem;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:.45rem .6rem;border-bottom:1px solid #ccd8dc}}th{{background:#e9f0f1}}img{{max-width:100%;width:760px}}figure{{margin:1rem 0}}figcaption{{color:#4f6370}}a{{color:#215d7a}}</style><main>
    <h1>Does enforcing U = Uii + D improve T2 forecasts?</h1><p><strong>{escape(top)}</strong></p>
    <h2>Official validation</h2>{_comparison_table(outer)}
    <p>The same {nval} official validation events recur across rotations. For each comparison, both means use the rotations on which both arms passed every event. Coverage counts all planned rotations. A blank score means the arm abstained during selection or an event failed qualification. Lower NMSE is better. Three targets receive equal weight. U uses the same fitting-set scale in both arms. Subtracting the supplied Uii gives the same residual for D.</p>
    <p>{escape(_secondary(outer))}</p>
    <p>Across the selected fits and events, the recorded failures were {escape(count_text)}. Forecasts with negative reconstructed D on at least one sampled time numbered {len(outer["negative_disposal_forecasts"])}. D negativity is recorded separately from the qualification checks on Gp, I and U.</p>
    {_table(["Graph", "Arm", "Shared rotations", "All-target NMSE", "Gp", "I", "U"], per_graph)}
    {_figure(root, outer)}
    <h2>Omitted events within the training split</h2>{_comparison_table(inner)}
    <p>Each panel used eight trajectories to fit coefficients, four to choose the lowest average whole-model NMSE and four to evaluate its forecast. The two rotations exchange the latter two sets. The official validation replay used frozen training choices and parameters. Each fitted model propagates its own generated variables.</p>
    <p>{escape(_secondary(inner))}</p>
    <h2>Which dependencies does reconstructed disposal require?</h2>
    <p>For each fitting event, D(t) = U(t) &minus; Uii(t). A shared deterministic function must give the same D whenever all its arguments are equal. For an equal-argument group with observed range R, at least one point must have absolute error of R/2 or greater. The screen uses that lower bound and an absolute rounding slack of 10<sup>&minus;10</sup> mg/kg/min. A latent argument is treated as equal only when the initializer or matched input histories establish equality. The repeated-group count measures how often the screen can actually test a candidate.</p>
    {_dependency_table(root, plan)}
    <p>{escape(witness)} The observed insulin value alone fails on both fitting folds. Each of the tested latent-action source lists survives its available comparisons. The data therefore support retaining multiple hypotheses for fitting. Survival gives no guarantee of adequate forecast accuracy.</p>
    <h2>Selected disposal equations</h2>
    <p>Within each portfolio, the free and balanced slots use matched source laws. The native comparison restricts each portfolio to its original source set. The final comparison permits the balanced arm to search more readouts than the native reference. Uii was constant at 1 on every fitting trajectory. Its freely fitted coefficient is therefore confounded with the intercept, so these data cannot identify that coefficient. The two parameterizations can still reach different fits under a bounded optimizer. Each candidate received {len(plan["starts"])} starting vectors with a ceiling of {budget["max_evaluations"]} evaluations and {budget["wall_seconds"]:.0f} s each. The experiment attempted {attempted} starts.</p>
    {_table(["Panel", "Rotation", "Arm", "Selected law", "Fit NMSE", "Selection NMSE", "Validation NMSE", "U readout"], choices)}
    <p><a href="model-choice/summary.json">Complete ODEs selected across graphs</a> · <a href="plan.json">Generated candidates and fitting budgets</a> · <a href="choices.json">Frozen choices</a> · <a href="official-validation/summary.json">Validation results and failures</a> · <a href="omitted-training/summary.json">Omitted training events</a></p>
    </main></html>""",
        encoding="utf-8",
    )
    return fn


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "stage",
        choices=[
            "plan",
            "select",
            "internal",
            "freeze-validation",
            "validate",
            "report",
        ],
    )
    cli.add_argument("--root", type=Path, default=ROOT)
    cli.add_argument("--source", type=Path, default=SOURCE)
    args = cli.parse_args()
    if args.stage == "report":
        print(render(args.root))
    elif args.stage in {"internal", "validate"}:
        print(
            score(args.root, args.source, official=args.stage == "validate")[
                "artifact_sha256"
            ]
        )
    else:
        action = {
            "plan": selection_plan,
            "select": select,
            "freeze-validation": external_plan,
        }
        print(action[args.stage](args.root)["artifact_sha256"])
