"""Replay frozen Full scaling models and report median [worst, best] diagnostics.

No search, fitting, selection or reference generation occurs here. The original
three-schedule NMSE and two-contrast response RSE must reproduce before a model
is reported. Direction uses the magnitude-weighted sign rule in Claude's
score_term_directions.py. Shape uses the nonnegative gain projection from
score_term_timing.py, applied to the WHOLE matched response, not event windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_stress_evaluation as stress
from scripts.iclr_anchor_fidelity import verify_controlled_bundle

PROTOCOL = "iclr-scaling-response-1"
DEFINITIONS = {
    "test_nmse": "Original mean over all three held-out schedules, training variance.",
    "response_rse": "Original mean of integral squared-error ratios, two contrasts.",
    "direction": "sum(abs(delta)*[sign(pred_delta)==sign(delta)])/sum(abs(delta))",
    "direction_accuracy": "Fraction of active contrasts with direction > 0.5.",
    "shape_rse": "min_{a>=0} sum((a*pred_delta-delta)^2)/sum(delta^2)",
    "shape_window": "Whole response at the saved uniform time grid; no time shift.",
    "shape_floor": (
        "If max(abs(pred_delta)) <= 1e-8*max(1,abs(truth),abs(control)), a=0."
    ),
    "aggregation": "Within-model mean; then retained-model median [worst, best].",
    "cost": "Recorded visit time plus pruning; no queue, startup or evaluation time.",
    "cost_cohort": "Retained models in main table; all attempted runs also exported.",
    "diagnostic_only": "The shape gain changes no model, prediction or other score.",
}


def digest(path: Path) -> str:
    """Hash the exact input bytes for reproducible replay."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> dict:
    """Seal JSON-normalized values, refusing incompatible resumes."""
    return sealed_write(path, json.loads(json.dumps(value, allow_nan=False)))


def response_metrics(truth, control, predicted, predicted_control) -> dict:
    """Score sign and gain-normalized shape on one matched response curve."""
    arrays = [
        np.asarray(v, dtype=float)
        for v in (truth, control, predicted, predicted_control)
    ]
    if (
        any(v.ndim != 1 or len(v) < 2 for v in arrays)
        or len({v.shape for v in arrays}) != 1
        or not all(np.isfinite(v).all() for v in arrays)
    ):
        raise ValueError("expected four equal finite response vectors")
    truth, control, predicted, predicted_control = arrays
    delta, estimate = truth - control, predicted - predicted_control
    # Match the original stress evaluator's reference-effect eligibility.
    if np.max(abs(delta)) <= 1e-8 * max(1.0, np.max(abs(truth))):
        return {"status": "negligible_reference"}
    weights = abs(delta)
    same = np.sign(estimate) == np.sign(delta)
    agreement = float(weights[same].sum() / weights.sum())
    floor = 1e-8 * max(1.0, np.max(abs(truth)), np.max(abs(control)))
    power = float(estimate @ estimate)
    gain = (
        max(0.0, float(estimate @ delta) / power)
        if np.max(abs(estimate)) > floor and power > 0
        else 0.0
    )
    error = gain * estimate - delta
    shape = float((error @ error) / (delta @ delta))
    if not -1e-12 <= shape <= 1 + 1e-12:
        raise ValueError("shape projection escaped its [0, 1] bound")
    return {
        "status": "complete",
        "sign_agreement": agreement,
        "direction_correct": agreement > 0.5,
        "shape_rse": min(1.0, max(0.0, shape)),
        "best_scale": gain,
    }


def cached_predict(
    out: Path, plan: dict, endpoint: dict, cell: dict, pair: dict
) -> tuple[dict, dict]:
    """Checkpoint paired solver-checked curves, bound to the frozen analysis."""
    binding = {
        "plan_sha256": plan["artifact_sha256"],
        "source_id": endpoint["source_id"],
        "reference_sha256": public.content_sha256(pair),
    }
    path = out / "curves" / endpoint["source_id"] / (pair["case"] + ".json")
    if not path.exists():
        p, q = stress.predict(endpoint, cell, pair)
        write(
            path,
            binding
            | {
                "predicted": {k: v.tolist() for k, v in p.items()},
                "control": {k: v.tolist() for k, v in q.items()},
            },
        )
    record = sealed_read(path)
    if any(record[k] != v for k, v in binding.items()):
        raise ValueError("stale rollout checkpoint")
    return tuple(
        {k: np.asarray(v) for k, v in record[field].items()}
        for field in ("predicted", "control")
    )


def assert_parity(old: dict, new: dict) -> None:
    """Require original per-schedule and aggregate errors before adding metrics."""
    if old["evaluation_status"] != "complete" or new["evaluation_status"] != "complete":
        raise ValueError("original evaluation or replay is incomplete")

    def compare(a, b):
        if a is None or b is None:
            ok = a is b
        else:
            ok = math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10)
        if not ok:
            raise ValueError(f"original metric parity failed: {a} vs {b}")

    for field in ("test_nmse", "response_rse"):
        compare(old[field], new[field])
    left = {r["case"]: r for r in old["trajectories"]}
    right = {r["case"]: r for r in new["trajectories"]}
    if left.keys() != right.keys():
        raise ValueError("replay schedule coverage differs")
    for case, a in left.items():
        b = right[case]
        if a["channels"].keys() != b["channels"].keys():
            raise ValueError("replay target coverage differs")
        for channel, scores in a["channels"].items():
            for field in ("nmse", "response_rse"):
                compare(scores[field], b["channels"][channel][field])


def interval(values: list[float], *, higher_is_better: bool = False) -> dict:
    """Keep the user-requested ordering; high accuracy reverses the extrema."""
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("summary requires finite measured values")
    worst, best = (min, max) if higher_is_better else (max, min)
    return {
        "median": statistics.median(values),
        "worst": worst(values),
        "best": best(values),
        "n": len(values),
    }


def cost_values(cost: dict) -> dict:
    """Check recorded timing completeness and keep proposer token scope explicit."""
    if any(
        cost.get(k, 0)
        for k in (
            "unknown_visit_costs",
            "unknown_pruning_costs",
            "unknown_proposal_token_counts",
        )
    ):
        raise ValueError("incomplete recorded costs")
    total = cost["search_and_pruning_wall_seconds"]
    if (
        not math.isclose(total, cost["visit_seconds"] + cost["pruning_seconds"])
        or not 0 <= cost["fit_seconds"] <= cost["visit_seconds"] <= total
    ):
        raise ValueError("inconsistent recorded timing")
    return {
        "fit_minutes": cost["fit_seconds"] / 60,
        "total_minutes": total / 60,
        "proposer_tokens_k": cost["observed_proposal_tokens"] / 1000,
    }


def summarize(rows: list[dict]) -> list[dict]:
    """Retain failed-model counts and show costs for both retained/all-run cohorts."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dimension"], row["variant"])].append(row)
    result = []
    def order(k):
        return k[1] != "dimension", k[0], k[1]
    for (dimension, variant), cohort in sorted(
        grouped.items(), key=lambda kv: order(kv[0])
    ):
        kept = [r for r in cohort if r["status"] == "complete"]
        metrics = (
            "test_nmse",
            "response_rse",
            "direction_accuracy_pct",
            "shape_rse",
            "weighted_sign_agreement_pct",
            "fit_minutes",
            "total_minutes",
            "proposer_tokens_k",
        )
        result.append(
            {
                "dimension": dimension,
                "variant": variant,
                "retained": len(kept),
                "planned": len(cohort),
                "statistics": {
                    k: interval(
                        [r[k] for r in kept],
                        higher_is_better=(
                            k
                            in {"direction_accuracy_pct", "weighted_sign_agreement_pct"}
                        ),
                    )
                    for k in metrics
                },
                "all_attempt_costs": {
                    k: interval([r[k] for r in cohort])
                    for k in ("fit_minutes", "total_minutes", "proposer_tokens_k")
                },
            }
        )
    return result


def report(out: Path, summary: list[dict]) -> None:
    """Render the two compact rebuttal tables with explicit metric denominators."""
    lines = [
        "Full complexity study: median [worst, best] over retained models.",
        "Two systems x three search seeds per row. Failed models remain in counts.",
        "NMSE: all 3 held-out schedules. Response metrics: 2 nonzero contrasts.",
        "Direction accuracy: % contrasts with magnitude-weighted sign agreement >50%.",
        "Shape: whole-response RSE after a nonnegative gain adjustment; no time shift.",
        "Gain is diagnostic only; original predictions and RSE remain unchanged.",
        "Total: recorded search + pruning, including proposal/critic/fitting work.",
        "Excludes queueing, server startup and evaluation; not allocation time.",
        "Both time columns use retained models, as do the accuracy columns.",
        "All-attempt costs and continuous sign agreement are in summary.json.",
        "A broader scalability or orders-of-magnitude multiscale claim is unsupported.",
        "",
    ]

    def label(row):
        return {
            "dimension": f"{row['dimension']} states ({row['dimension'] - 1} hidden)",
            "double_recurrent_coupling": "8 states, 2x linear coupling",
            "wider_decay_spread": "8 states, wider decay timescales",
        }[row["variant"]]

    def fmt(row, metric, decimals):
        s = row["statistics"][metric]
        a, b, c = (f"{s[k]:.{decimals}f}" for k in ("median", "worst", "best"))
        return f"{a} [{b}, {c}]"

    for columns in (
        [
            ("Test NMSE", "test_nmse", 3),
            ("Response RSE", "response_rse", 3),
            ("Direction accuracy (%)", "direction_accuracy_pct", 1),
            ("Shape error", "shape_rse", 3),
        ],
        [
            ("Fitting (min)", "fit_minutes", 1),
            ("Total (min)", "total_minutes", 1),
            ("Proposer tokens (k)", "proposer_tokens_k", 1),
        ],
    ):
        lines.append(
            "| Condition | Retained | " + " | ".join(c[0] for c in columns) + " |"
        )
        lines.append("|" + "---|" * (2 + len(columns)))
        for row in summary:
            cells = [label(row), f"{row['retained']}/{row['planned']}"]
            cells += [fmt(row, k, n) for _, k, n in columns]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    (out / "RESULTS.txt").write_text("\n".join(lines) + "\n")


def run(source: Path, out: Path) -> dict:
    """Authenticate the completed study, replay all retained Full models and report."""
    if out.resolve().is_relative_to(source.resolve()):
        raise ValueError("write the new analysis outside the original frozen package")
    manifest = sealed_read(source / "design/manifest.json")
    model_path = source / "frozen/frozen_models.json"
    models = sealed_read(model_path)
    verify_controlled_bundle(model_path, models)
    old = sealed_read(source / "evaluation/evaluation.json")
    freeze = sealed_read(source / "evaluation/freeze.json")
    if (
        old["manifest_sha256"] != manifest["artifact_sha256"]
        or old["search_bundle_sha256"] != models["artifact_sha256"]
        or old["freeze_sha256"] != freeze["artifact_sha256"]
        or freeze["search_sha256"] != models["artifact_sha256"]
        or freeze["design_sha256"] != manifest["artifact_sha256"]
    ):
        raise ValueError("evaluation is not bound to the frozen models/design")
    selected = [
        (c, r)
        for c, r in stress.select_endpoints(manifest, models)
        if c["study"] == "scaling" and r["method"] == "full"
    ]
    expected = {
        (c["cell"], s)
        for c in manifest["cells"]
        if c["study"] == "scaling"
        for s in manifest["algorithm_seeds"]
    }
    if {(c["cell"], r["repetition"]) for c, r in selected} != expected:
        raise ValueError("missing scaling model denominator")
    previous = {}
    for row in old["rows"]:
        key = (row["benchmark_id"], row["method"], row["repetition"])
        if key in previous:
            raise ValueError("duplicate original evaluation row")
        previous[key] = row
    references, scales, bindings = {}, {}, {}
    for cell, _ in selected:
        name = cell["cell"]
        if name in references:
            continue
        ref = source / "evaluation/references" / (name + ".json")
        train = source / "design/public" / name / "train.json"
        if digest(train) != cell["public_files"]["train.json"]:
            raise ValueError("training scale input changed")
        record = read_seal(ref)
        if record["freeze_sha256"] != freeze["artifact_sha256"]:
            raise ValueError("references belong to another evaluation freeze")
        references[name] = record["records"]
        scales[name] = stress.data.train_scales(read_seal(train)["rows"])
        bindings[name] = {"reference_file": digest(ref), "train_file": digest(train)}
    plan = write(
        out / "plan.json",
        {
            "protocol": PROTOCOL,
            "definitions": DEFINITIONS,
            "source": str(source.resolve()),
            "manifest": manifest["artifact_sha256"],
            "models": models["artifact_sha256"],
            "evaluation": old["artifact_sha256"],
            "inputs": bindings,
            "adapter_sha256": digest(Path(__file__)),
            "evaluator_sha256": digest(Path(stress.__file__)),
            "fitting_source": public._source_identity(),
            "runtime": public._runtime(),
        },
    )
    rows = []
    for cell, endpoint in selected:
        name, seed = cell["cell"], endpoint["repetition"]
        earlier = previous[(name, "full", seed)]
        if (
            earlier["source_id"] != endpoint["source_id"]
            or earlier["model_sha256"] != endpoint.get("model_sha256")
            or earlier["cost"] != endpoint["cost"]
        ):
            raise ValueError("original endpoint identity or costs differ")
        row = {k: cell[k] for k in ("dimension", "variant", "generator_seed")}
        row.update(
            {
                "benchmark_id": name,
                "repetition": seed,
                "source_id": endpoint["source_id"],
                "status": earlier["evaluation_status"],
                **cost_values(earlier["cost"]),
            }
        )
        if endpoint["status"] == "ready":

            def predictor(e, c, pair):
                return cached_predict(out, plan, e, c, pair)

            new = stress.score(
                endpoint, cell, references[name], scales[name], predictor
            )
            assert_parity(earlier, new)
            diagnostics = []
            for pair in references[name]:
                p, q = predictor(endpoint, cell, pair)
                for channel, truth in pair["perturbed"]["targets"].items():
                    diagnostics.append(
                        {"case": pair["case"], "channel": channel}
                        | response_metrics(
                            truth,
                            pair["control"]["targets"][channel],
                            p[channel],
                            q[channel],
                        )
                    )
            active = [r for r in diagnostics if r["status"] == "complete"]
            if len(diagnostics) != 3 or len(active) != 2:
                raise ValueError("expected three schedules and two active contrasts")
            row.update(
                {
                    "status": "complete",
                    "parity": "passed",
                    "contrasts": diagnostics,
                    "test_nmse": earlier["test_nmse"],
                    "response_rse": earlier["response_rse"],
                    "direction_accuracy_pct": 100
                    * statistics.mean(r["direction_correct"] for r in active),
                    "weighted_sign_agreement_pct": 100
                    * statistics.mean(r["sign_agreement"] for r in active),
                    "shape_rse": statistics.mean(r["shape_rse"] for r in active),
                    "parity_abs_difference": {
                        k: abs(new[k] - earlier[k])
                        for k in ("test_nmse", "response_rse")
                    },
                }
            )
        elif earlier["evaluation_status"] != "model_unavailable":
            raise ValueError("missing model has an inconsistent evaluation status")
        write(
            out / "models" / (endpoint["source_id"] + ".json"),
            {"plan_sha256": plan["artifact_sha256"], **row},
        )
        rows.append(row)
        print(f"{name} seed {seed}: {row['status']}", flush=True)
    summary = summarize(rows)
    result = write(
        out / "summary.json",
        {
            "protocol": PROTOCOL,
            "plan_sha256": plan["artifact_sha256"],
            "definitions": DEFINITIONS,
            "rows": rows,
            "groups": summary,
            "no_refitting_or_selection": True,
        },
    )
    report(out, summary)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out)
