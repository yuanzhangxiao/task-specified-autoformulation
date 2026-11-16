"""Replay frozen Full sensor variants on the existing CSTR intervention panel.

Run as a file with Claude's scoring checkout on PYTHONPATH. Its reference
records, term splitter, rollout evaluator and metrics are reused unchanged.
This adapter supplies only each model's declared auxiliaries and extends the
hard-case attribution rule to the measured C/Tj channels. It never fits or
selects a discovery model.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

MASKS = {
    "sensor_00": (),
    "sensor_01": ("C",),
    "sensor_02": ("Tj",),
    "sensor_03": ("C", "Tj"),
}
PROTOCOL = "iclr-cstr-observation-panel-1"


def native():
    """Load the existing scoring implementation, without modifying its checkout."""
    from scripts import iclr_anchor_fidelity as anchor
    from scripts import report_intervention_suite as events
    from scripts import score_declared_mechanisms as declared

    return anchor, declared, events


def digest(path: Path) -> str:
    """Hash exact source or artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    """Authenticate a sealed analysis record."""
    from autoformalism.rebuttal.prefit_replay import sealed_read

    return sealed_read(path)


def write(path: Path, value: dict) -> dict:
    """Atomically seal a new record, refusing incompatible resumes."""
    from autoformalism.rebuttal.prefit_replay import sealed_write

    # Native affine maps are tuples; JSON reloads them as lists. Compare their
    # stored representation on resume, rather than unequal Python containers.
    return sealed_write(path, json.loads(json.dumps(value)))


def source_identity() -> dict:
    """Bind the adapter and the native numerical and attribution implementations."""
    anchor, declared, events = native()
    from autoformalism.fitting import public_fitting as public

    modules = (anchor, declared, events, anchor.probe)
    return {
        "adapter": digest(Path(__file__)),
        "native": {m.__name__: digest(Path(m.__file__)) for m in modules},
        "rule": digest(declared.RULE),
        "fitting": public._source_identity(),
        "runtime": public._runtime(),
    }


def observation_trajectory(context: dict, record: dict):
    """Expose initial target values and permitted auxiliaries, never future T."""
    import numpy as np

    from autoformalism.data import Trajectory

    if (
        context["targets"] != ["T"]
        or set(context["external_inputs"]) != {"Cf", "Tf", "Tjf"}
        or set(context["auxiliaries"]) - {"C", "Tj"}
        or context.get("lagged_targets")
        or context.get("fixed_covariates")
    ):
        raise ValueError("unsupported reactor observation contract")
    t = np.asarray(record["time"], dtype=float)
    return Trajectory(
        trajectory_id="public_forcing",
        time=t,
        targets={"T": np.full(len(t), record["T"][0])},
        auxiliaries={n: np.asarray(record[n]) for n in context["auxiliaries"]},
        external_inputs={n: np.asarray(record[n]) for n in context["external_inputs"]},
        fixed_covariates={},
        derivatives={},
    )


def attribute_observed(view: dict) -> dict:
    """Extend the fixed input-dependency rule to explicitly measured C and Tj.

    Original hard-case assignments are returned verbatim. Measured channels
    can identify a term or drive a latent process; they are never themselves
    reported as discovered latent states.
    """
    _, declared, _ = native()
    from autoformalism.expressions import ValidationContext
    from autoformalism.expressions.parser import grammar_parser
    from autoformalism.schemas import CandidateModel

    if not view["context"]["auxiliaries"]:
        return declared.attribute(view)
    candidate = CandidateModel.model_validate(view["candidate"])
    context = ValidationContext.model_validate(view["context"])
    grammar = view["grammar"]
    parser = grammar_parser(grammar)
    processes = {p.name: p.expression for p in candidate.processes}
    equations = {e.state: e.rhs for e in candidate.state_equations}
    measured = {
        n: {"C": "concentration", "Tj": "jacket"}[n] for n in context.auxiliaries
    }
    kinds = {}
    observed = {*context.targets, *context.auxiliaries}
    for state in candidate.states:
        if state.name in observed or state.name not in equations:
            continue
        expanded = declared.expanded(equations[state.name], processes, grammar)
        names = declared.names_in(expanded) & ({"Cf", "Tf", "Tjf"} | set(measured))
        kinds[state.name] = declared._kind(names, measured)
    roles = measured | kinds

    def classify(node):
        expanded = declared.expanded(ast.unparse(node), processes, grammar)
        return declared._kind(declared.names_in(expanded), roles)

    tree = parser.parse(equations["T"], location="temperature equation").tree.body
    terms = []
    for sign, node, own in declared.split_terms(tree, classify):
        written = declared.names_in(own)
        kind = classify(own)
        named = any(
            n in processes and ("react" in n.lower() or "rxn" in n.lower())
            for n in written
        )
        mechanism = {
            "jacket": "jacket_heat",
            "feed": "feed_transport",
            "concentration": "reaction_heat",
        }.get(kind, "reaction_heat" if named else "unattributed")
        terms.append(
            {"sign": sign, "expression": ast.unparse(node), "mechanism": mechanism}
        )
    hidden = {}
    for kind, quantity in declared.KIND_OF_STATE.items():
        options = [s.name for s in candidate.states if kinds.get(s.name) == kind]
        hidden[quantity] = options[0] if options and quantity not in measured else None
    return {
        "state_kinds": kinds,
        "observed_kinds": measured,
        "terms": terms,
        "represented": {
            m: any(t["mechanism"] == m for t in terms) for m in declared.MECHANISMS
        },
        "hidden": hidden,
    }


def freeze(root: Path, panel: Path, bundles: list[Path]) -> dict:
    """Pin all twelve frozen Full models and the original validity decisions."""
    anchor, declared, _ = native()
    original = read(panel / "plan.json")
    validity = read(panel / "validity.json")
    cell = anchor.fits.CSTR
    if validity["plan"] != original["artifact_sha256"]:
        raise ValueError("unbound panel validity record")
    schedule = declared.kept_schedule(panel, original, cell)
    if sum(v["role"] == "contrast" for v in schedule.values()) != 24:
        raise ValueError("expected the existing 24-contrast panel")
    selected = {}
    sources = []
    for source in bundles:
        bundle = read(source)
        anchor.verify_controlled_bundle(source, bundle)
        sources.append(
            {
                "path": str(source.resolve()),
                "sha256": digest(source),
                "bundle": bundle["artifact_sha256"],
            }
        )
        for row in bundle["rows"]:
            key = (row["benchmark_id"], row["repetition"])
            if row["method"] != "full" or key[0] not in MASKS:
                continue
            if key in selected:
                raise ValueError(f"duplicate source model: {key}")
            if tuple(row["context"]["auxiliaries"]) != MASKS[key[0]]:
                raise ValueError("source model has the wrong observation mask")
            selected[key] = {
                **row,
                "benchmark_id": cell,
                "source_benchmark_id": key[0],
                "native_source_id": row["source_id"],
                "source_id": f"{key[0]}_full_seed{key[1]}",
                "source_bundle_sha256": bundle["artifact_sha256"],
                "source_protocol": bundle["protocol"],
                "direct_readouts": {},
            }
    if set(selected) != {(c, s) for c in MASKS for s in (0, 1, 2)}:
        raise ValueError("expected three Full seeds for each of four sensor masks")
    models = [selected[k] for k in sorted(selected)]
    refs = {
        name: {
            "path": str((panel / "references" / cell / f"{name}.json").resolve()),
            "sha256": digest(panel / "references" / cell / f"{name}.json"),
        }
        for name in schedule
    }
    assignments = []
    for row in models:
        if row["status"] != "ready":
            raise ValueError(
                "this frozen sensor cohort should have twelve ready models"
            )
        assignments.append(attribute_observed(declared.view_of(row)))
    plan = write(
        root / "plan.json",
        {
            "protocol": PROTOCOL,
            "source": source_identity(),
            "sources": sources,
            "panel": str(panel.resolve()),
            "panel_plan": original["artifact_sha256"],
            "validity": validity["artifact_sha256"],
            "dropped": validity["dropped"],
            "models": models,
            "assignments": assignments,
            "references": refs,
            "schedules": schedule,
            "training_sd": original["training_sd"][cell],
            "selection": "All twelve native Full endpoints; no refit or reselection.",
            "observation_policy": (
                "Only T(0), public forcings and declared auxiliary trajectories."
            ),
            "attribution_extension": (
                "Measured C/Tj have concentration/jacket roles; "
                "observed states are not latent readouts."
            ),
            "post_hoc_observation_analysis": True,
        },
    )
    # The original reference seal remains intact; no new physical truth is generated.
    for name, spec in refs.items():
        source = Path(spec["path"])
        record = read(source)
        if record["plan"] != original["artifact_sha256"]:
            raise ValueError("reference does not belong to the original panel")
        destination = root / "references" / f"{name}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and digest(destination) != spec["sha256"]:
            raise ValueError("changed reference copy")
        destination.write_bytes(source.read_bytes())
    return plan


def run_subject(args) -> dict:
    """Checkpoint each model/trajectory and preserve every failed rollout."""
    root, index = args
    import numpy as np

    from autoformalism.expressions.diagnostics import (
        ModelValidationError,
        RuntimeExpressionError,
    )

    anchor, declared, _ = native()
    plan = read(root / "plan.json")
    row, assignment = plan["models"][index], plan["assignments"][index]
    view = declared.view_of(row)
    records = {}
    for case in plan["schedules"]:
        path = root / "subjects" / f"{index:02d}" / f"{case}.json"
        if not path.exists():
            reservation = path.with_suffix(".started.json")
            if reservation.exists():
                write(
                    path,
                    {
                        "plan": plan["artifact_sha256"],
                        "subject": index,
                        "case": case,
                        "status": "rollout_unavailable",
                        "reason": "interrupted_rollout",
                    },
                )
                records[case] = read(path)
                continue
            write(
                reservation,
                {"plan": plan["artifact_sha256"], "subject": index, "case": case},
            )
            ref = read(root / "references" / f"{case}.json")["reference"]
            try:
                tr = observation_trajectory(row["context"], ref)
                curves = declared.curves(view, assignment, tr)
                alternate = anchor.predict(row, tr, "BDF")
                gap = anchor.probe.solver_agreement(
                    {"T": curves["T"]}, {"T": alternate["T"]}, 1e-4
                )
                value = {
                    "status": "complete",
                    "curves": curves,
                    "alternate_temperature": alternate["T"],
                    "solver_gap": gap,
                }
                if not all(
                    np.isfinite(v).all() for v in map(np.asarray, curves.values())
                ):
                    raise ValueError("non-finite term or state curve")
            except (
                ArithmeticError,
                ValueError,
                RuntimeError,
                TimeoutError,
                ModelValidationError,
                RuntimeExpressionError,
            ) as exc:
                value = {"status": "rollout_unavailable", "reason": str(exc)[:500]}
            write(
                path,
                {
                    "plan": plan["artifact_sha256"],
                    "subject": index,
                    "case": case,
                    **value,
                },
            )
        record = read(path)
        if record["plan"] != plan["artifact_sha256"]:
            raise ValueError("stale rollout checkpoint")
        records[case] = record
    return {
        "subject": index,
        "source_id": row["source_id"],
        "complete": sum(r["status"] == "complete" for r in records.values()),
        "planned": len(records),
    }


def target_score(anchor, row, baseline, ref, control, item, scale) -> dict:
    """Apply the original target metrics and paired solver agreement gate."""
    import numpy as np

    if row["status"] != "complete" or baseline["status"] != "complete":
        return {"status": "rollout_unavailable"}
    a, b = np.asarray(row["curves"]["T"]), np.asarray(baseline["curves"]["T"])
    alt = np.subtract(row["alternate_temperature"], baseline["alternate_temperature"])
    amplitude = max(
        1e-8, np.max(abs(np.subtract(ref["T"], control["T"]))), np.max(abs(a - b))
    )
    if np.max(abs(alt - (a - b))) / amplitude > 5e-3:
        return {"status": "effect_solver_disagreement"}
    return metric(
        anchor, ref["time"], ref["T"], control["T"], a, b, item["window"], scale
    )


def metric(anchor, *args) -> dict:
    """Keep numerical scoring failures and use dimensionally neutral time labels."""
    try:
        result = anchor.scores(*args)
    except (ValueError, ArithmeticError) as exc:
        return {"status": "metric_unavailable", "reason": str(exc)}
    if "peak_time_error_min" in result:
        result["peak_time_error"] = result.pop("peak_time_error_min")
    return result


def score(root: Path) -> dict:
    """Score T and all three declared mechanisms, including missing representations."""
    import numpy as np

    anchor, declared, events = native()
    plan = read(root / "plan.json")
    schedule = plan["schedules"]
    refs = {}
    for k in schedule:
        path = root / "references" / f"{k}.json"
        if digest(path) != plan["references"][k]["sha256"]:
            raise ValueError("reference bytes changed")
        refs[k] = read(path)["reference"]
    truths = {k: declared._truth(v) for k, v in refs.items()}
    calib = [k for k, v in schedule.items() if v["role"] == "train_calibration"]
    targets, mechanisms, event_rows = [], [], []
    for index, row in enumerate(plan["models"]):
        data = {
            k: read(root / "subjects" / f"{index:02d}" / f"{k}.json") for k in schedule
        }
        if any(
            v["plan"] != plan["artifact_sha256"]
            or v["subject"] != index
            or v["case"] != k
            for k, v in data.items()
        ):
            raise ValueError("stale or misidentified subject checkpoint")
        base = {
            "source_id": row["source_id"],
            "sensor": row["source_benchmark_id"],
            "seed": row["repetition"],
            "model_sha256": row["model_sha256"],
        }
        assignment = plan["assignments"][index]
        mappings, scales = {}, {}
        for quantity in declared.MECHANISMS:
            y = np.concatenate([truths[k][quantity] for k in calib])
            scales[quantity] = float(np.std(y))
            mappings[quantity] = None
            if all(data[k]["status"] == "complete" for k in calib):
                x = np.concatenate([data[k]["curves"][quantity] for k in calib])
                mappings[quantity] = declared._fit(x, y)
        for case, item in schedule.items():
            if item["role"] != "contrast":
                continue
            control = item["control"]
            bind = {**base, "case": case, "group": item["group"], "control": control}
            a, b = data[case], data[control]
            target = target_score(
                anchor, a, b, refs[case], refs[control], item, plan["training_sd"]["T"]
            )
            targets.append(bind | target)
            if target["status"] == "complete":
                for event in item.get("events", []):
                    measured = events.event_metrics(
                        refs[case]["time"],
                        refs[case]["T"],
                        refs[control]["T"],
                        a["curves"]["T"],
                        b["curves"]["T"],
                        event["window"],
                        event["timing"],
                    )
                    event_rows.append(bind | {"event": event["name"]} | measured)
            for quantity in declared.MECHANISMS:
                for readout in ("direct", "affine"):
                    details = bind | {"quantity": quantity, "readout": readout}
                    if not assignment["represented"][quantity]:
                        measured = {"status": "not_represented"}
                    elif a["status"] != "complete" or b["status"] != "complete":
                        measured = {"status": "rollout_unavailable"}
                    elif readout == "affine" and mappings[quantity] is None:
                        measured = {"status": "calibration_unavailable"}
                    else:
                        gain, offset = (
                            mappings[quantity] if readout == "affine" else (1, 0)
                        )
                        pred = gain * np.asarray(a["curves"][quantity]) + offset
                        pred_control = gain * np.asarray(b["curves"][quantity]) + offset
                        measured = metric(
                            anchor,
                            refs[case]["time"],
                            truths[case][quantity],
                            truths[control][quantity],
                            pred,
                            pred_control,
                            item["window"],
                            scales[quantity],
                        )
                    mechanisms.append(
                        details
                        | {"map": mappings[quantity] if readout == "affine" else None}
                        | measured
                    )
    return write(
        root / "scores.json",
        {
            "protocol": PROTOCOL,
            "plan": plan["artifact_sha256"],
            "targets": targets,
            "mechanisms": mechanisms,
            "events": event_rows,
        },
    )


def csv_table(path: Path, rows: list[dict]) -> None:
    """Export all statuses and identities along with numerical results."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
        )
        writer.writeheader()
        writer.writerows(rows)


def report(root: Path) -> dict:
    """Retain both Claude's pooled medians and means per complete seed."""
    scores = read(root / "scores.json")
    summary = []
    for sensor in MASKS:
        target = [r for r in scores["targets"] if r["sensor"] == sensor]
        assert len(target) == 72
        row = {
            "sensor": sensor,
            "target_planned": 72,
            "target_complete": sum(r["status"] == "complete" for r in target),
        }
        for field in ("absolute_nmse", "effect_rse"):
            values = [
                r[field] if r["status"] == "complete" else math.inf for r in target
            ]
            row["pooled_median_" + field] = statistics.median(values)
        for seed in (0, 1, 2):
            items = [r for r in target if r["seed"] == seed]
            complete = all(r["status"] == "complete" for r in items)
            row[f"seed{seed}_mean_nmse"] = (
                statistics.mean(r["absolute_nmse"] for r in items) if complete else None
            )
            row[f"seed{seed}_median_nmse"] = (
                statistics.median(r["absolute_nmse"] for r in items)
                if complete
                else None
            )
        for quantity in ("feed_transport", "reaction_heat", "jacket_heat"):
            items = [
                r
                for r in scores["mechanisms"]
                if r["sensor"] == sensor
                and r["quantity"] == quantity
                and r["readout"] == "direct"
            ]
            done = [r for r in items if r["status"] == "complete"]
            row[quantity + "_complete"] = len(done)
            row[quantity + "_not_represented"] = sum(
                r["status"] == "not_represented" for r in items
            )
            row[quantity + "_median_rse_complete"] = (
                statistics.median(r["effect_rse"] for r in done) if done else None
            )
        summary.append(row)
    for key in ("targets", "mechanisms", "events"):
        csv_table(root / f"{key}.csv", scores[key])
    csv_table(root / "summary.csv", summary)
    return write(
        root / "summary.json",
        {
            "scores": scores["artifact_sha256"],
            "rows": summary,
            "note": (
                "Mechanism medians condition on complete represented scores; "
                "always accompany coverage. Affine scores are oracle diagnostics. "
                "Target pooled medians match Claude's aggregation; "
                "seed means match the earlier stress aggregation."
            ),
        },
    )


def main() -> None:
    """Freeze, resume, score and report without changing any discovery model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "score", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--models", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.stage == "freeze":
        result = freeze(args.root, args.panel, args.models)
        print(result["artifact_sha256"], len(result["models"]), "models")
        return
    plan = read(args.root / "plan.json")
    if plan["source"] != source_identity():
        raise ValueError("analysis source/runtime changed; use the pinned code")
    if args.stage == "run":
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(run_subject, [(args.root, i) for i in range(12)]):
                print(json.dumps(result), flush=True)
    elif args.stage == "score":
        result = score(args.root)
        print(result["artifact_sha256"], len(result["targets"]), "target rows")
    else:
        print(json.dumps(report(args.root)["rows"], indent=2))


if __name__ == "__main__":
    main()
