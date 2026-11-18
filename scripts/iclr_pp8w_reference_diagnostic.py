"""Fit supplied CSTR structure on the new public conditions, then score frozen fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_reference_fits as fits
from scripts import iclr_stress_data as data

CONDITIONS = ("cstr_trajectories_04", "cstr_trajectories_16", "cstr_noise_10pct")


def fit(source: Path, root: Path) -> dict:
    """Freeze public inputs, fit three starts, and select on validation alone."""
    if root.resolve().is_relative_to(source.resolve()):
        raise ValueError("write diagnostics outside the public design")
    manifest = sealed_read(source / "manifest.json")
    cells = {r["cell"]: r for r in manifest["cells"]}
    plan = sealed_write(
        root / "plan.json",
        {
            "protocol": "iclr-pp8w-supplied-structure-1",
            "design": manifest["artifact_sha256"],
            "implementation": data.digest(Path(__file__)),
            "fit_implementation": data.digest(Path(fits.__file__)),
            "conditions": list(CONDITIONS),
            "starts": 3,
            "max_evaluations": 50,
            "selection": "minimum validation NMSE among numerically qualified fits",
            "scope": (
                "Supplied physical equations, scale priors and parameter bounds; "
                "three optimizer starts, not three independent discovery seeds"
            ),
            "test_data_opened": False,
        },
    )
    models = []
    for condition in CONDITIONS:
        original = source / "public" / condition
        work = root / condition
        public = work / "public"
        destination = public / fits.CSTR
        for name, digest in cells[condition]["public_files"].items():
            if data.digest(original / name) != digest:
                raise ValueError("changed condition inputs")
        spec = read_seal(original / "specification.json")
        seal(destination / "specification.json", {**spec, "benchmark_id": fits.CSTR})
        for name in ("train.json", "val.json"):
            seal(destination / name, read_seal(original / name))
            if data.digest(destination / name) != data.digest(original / name):
                raise ValueError("reference fitting changed the observation arrays")
        prompt = (original / "proposer_prompt.txt").read_bytes()
        target = destination / "proposer_prompt.txt"
        if target.exists() and target.read_bytes() != prompt:
            raise ValueError("reference prompt differs on resume")
        target.write_bytes(prompt)
        output = work / "fitting"
        fits.prepare(public, output, 3, 50, [fits.CSTR])
        for start in range(3):
            result = fits.run_one(public, output, fits.CSTR, start)
            print(
                json.dumps(
                    {"condition": condition, "start": start, "status": result["status"]}
                ),
                flush=True,
            )
        bundle = fits.freeze(output)
        models.append(
            {
                "condition": condition,
                "cell": cells[condition],
                "frozen_bundle": bundle["artifact_sha256"],
                "endpoint": bundle["rows"][0] | {"source_protocol": fits.VERSION},
            }
        )
    return sealed_write(
        root / "frozen_models.json",
        {"plan": plan["artifact_sha256"], "models": models, "test_data_opened": False},
    )


def evaluate(root: Path, panel: Path) -> dict:
    """Open held-out references only after every condition's selected fit is frozen."""
    from scripts import iclr_anytime_response as metrics
    from scripts import iclr_cstr_observation_panel as cstr

    anchor, declared, _ = cstr.native()
    frozen = sealed_read(root / "frozen_models.json")
    fit_plan = sealed_read(root / "plan.json")
    if frozen["plan"] != fit_plan["artifact_sha256"]:
        raise ValueError("fit plan changed")
    original = sealed_read(panel / "plan.json")
    plan = sealed_write(
        root / "evaluation-plan.json",
        {
            "models": frozen["artifact_sha256"],
            "reference_plan": original["artifact_sha256"],
            "source": cstr.source_identity(),
            "metric_source": data.digest(Path(metrics.__file__)),
            "implementation": data.digest(Path(__file__)),
            "selection_frozen_before_test_access": True,
        },
    )
    references = {}
    for case, spec in original["references"].items():
        path = panel / "references" / f"{case}.json"
        if data.digest(path) != spec["sha256"]:
            raise ValueError("held-out reference bytes changed")
        references[case] = sealed_read(path)["reference"]
    rows = []
    for model in frozen["models"]:
        condition, endpoint = model["condition"], model["endpoint"]
        if endpoint["status"] != "ready":
            rows.append({"condition": condition, "status": "fit_failed"})
            continue
        predictions = {}
        for case, reference in references.items():
            path = root / "predictions" / condition / f"{case}.json"
            if not path.exists():
                try:
                    tr = anchor.trajectory(fits.CSTR, reference)
                    a = anchor.predict(endpoint, tr, "Radau")
                    b = anchor.predict(endpoint, tr, "BDF")
                    anchor.probe.solver_agreement({"T": a["T"]}, {"T": b["T"]}, 1e-4)
                    theta = endpoint["fit"]["theta"]
                    flow = fits.coefficients(fits.CSTR)["flow"] * np.exp(theta[0])
                    curves = {
                        "T": a["T"],
                        "feed_transport": (
                            flow * (np.asarray(reference["Tf"]) - np.asarray(a["T"]))
                        ).tolist(),
                        "reaction_heat": a["direct.reaction_heat"],
                        "jacket_heat": a["direct.jacket_heat"],
                    }
                    value = {
                        "status": "complete",
                        "curves": curves,
                        "alternate_temperature": b["T"],
                    }
                except (ValueError, RuntimeError, ArithmeticError) as error:
                    value = {"status": "rollout_unavailable", "reason": str(error)}
                sealed_write(path, {"plan": plan["artifact_sha256"], **value})
            predictions[case] = sealed_read(path)
            if predictions[case]["plan"] != plan["artifact_sha256"]:
                raise ValueError("prediction cache belongs to another diagnostic")
        cases = []
        for case, schedule in original["schedules"].items():
            if schedule["role"] != "contrast":
                continue
            control = schedule["control"]
            p, q = predictions[case], predictions[control]
            r, c = references[case], references[control]
            gate = cstr.target_score(
                anchor, p, q, r, c, schedule, model["cell"]["scales"]["T"]
            )
            if gate["status"] != "complete":
                cases.append({"case": case, "status": gate["status"]})
                continue
            truth = {"T": r["T"], **declared._truth(r)}
            base = {"T": c["T"], **declared._truth(c)}
            for quantity in ("T", *declared.MECHANISMS):
                cases.append(
                    {
                        "case": case,
                        "quantity": quantity,
                        "status": "complete",
                        **metrics.continuous_score(
                            r["time"],
                            truth[quantity],
                            base[quantity],
                            p["curves"][quantity],
                            q["curves"][quantity],
                            model["cell"]["scales"]["T"] if quantity == "T" else None,
                            integral=False,
                        ),
                    }
                )
        result = {
            "plan": plan["artifact_sha256"],
            "condition": condition,
            "cases": cases,
        }
        sealed_write(root / "scores" / f"{condition}.json", result)
        rows.append(
            {
                "condition": condition,
                "selected_optimizer_start": endpoint["fit"]["start"],
                "validation_nmse": endpoint["fit"]["scores"]["validation"][0],
                **metrics.pooled(cases, {"T"}, 24),
                "mechanisms": {
                    q: metrics.pooled(cases, {q}, 24) for q in declared.MECHANISMS
                },
            }
        )
    result = sealed_write(
        root / "summary.json",
        {
            "plan": plan["artifact_sha256"],
            "rows": rows,
            "discovery_baseline": False,
            "test_used_for_selection": False,
        },
    )
    lines = [
        "Supplied-structure diagnostic: one validation-selected fit of three starts.",
        "These are optimizer starts, not three discovery seeds; no seed interval.",
        "Known equations and physical scale bounds are supplied. No LLM calls.",
        "Scored on the unchanged 24 CSTR interventions, with clean evaluation inputs.",
        "All conditions use the common clean 16-trajectory training SD.",
        "",
        "| Condition | Selected start | Test NMSE | RSE | Direction | Shape |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["status"] != "complete":
            lines.append(f"| {row['condition']} | — | {row['status']} | — | — | — |")
            continue
        values = " | ".join(
            f"{row[k]:.6g}"
            for k in (
                "test_nmse",
                "response_rse",
                "directional_accuracy",
                "shape_error",
            )
        )
        lines.append(
            f"| {row['condition']} | {row['selected_optimizer_start']} | {values} |"
        )
    (root / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--panel", type=Path)
    args = parser.parse_args()
    if args.stage == "fit" and args.source:
        result = fit(args.source, args.root)
    elif args.stage == "evaluate" and args.panel:
        result = evaluate(args.root, args.panel)
    else:
        parser.error("fit requires --source; evaluate requires --panel")
    print(json.dumps({"artifact_sha256": result["artifact_sha256"]}))


if __name__ == "__main__":
    main()
