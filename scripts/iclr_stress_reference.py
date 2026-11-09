"""Stage two acquisition endpoints for the privileged B1 fitter."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_stress_data as data

CONDITIONS = {data.T2: "acquisition_03", data.CSTR.format("hard"): "sensor_03"}


def stage(release: Path, output: Path) -> dict:
    """Alias only untouched conditioned public files into the B1 fitter's cell names."""
    if output.resolve().is_relative_to((release / "public").resolve()):
        raise ValueError("write a separate reference-fit input tree")
    roster = sealed_read(release / "manifest.json")
    cells = {c["cell"]: c for c in roster["cells"]}
    rows = []
    for anchor, key in CONDITIONS.items():
        original = release / "public" / key
        destination = output / anchor
        cell = cells[key]
        if cell.get("source") != anchor:
            raise ValueError("expected the specified anchor's acquisition condition")
        for f in (
            "train.json",
            "val.json",
            "specification.json",
            "proposer_prompt.txt",
        ):
            if data.digest(original / f) != cell["public_files"][f]:
                raise ValueError("changed acquisition inputs")
        spec = read_seal(original / "specification.json")
        seal(destination / "specification.json", {**spec, "benchmark_id": anchor})
        prompt = original / "proposer_prompt.txt"
        copy = destination / "proposer_prompt.txt"
        if copy.exists() and copy.read_bytes() != prompt.read_bytes():
            raise ValueError("existing prompt differs")
        copy.write_bytes(prompt.read_bytes())
        for split in ("train", "val"):
            name = split + ".json"
            # Writing the sealed payload also verifies previous artifacts on resume.
            seal(destination / name, read_seal(original / name))
            if data.digest(destination / name) != data.digest(original / name):
                raise ValueError("reference input rows changed during projection")
        rows.append(
            {
                "cell": key,
                "fit_family": anchor,
                "files": {
                    f: data.digest(destination / f)
                    for f in ("train.json", "val.json", "specification.json")
                },
            }
        )
    return sealed_write(
        output / "conditions.json",
        {
            "protocol": "iclr-conditioned-reference-inputs-1",
            "design_sha256": roster["artifact_sha256"],
            "rows": rows,
        },
    )


def development_report(release: Path, fitted_root: Path, output: Path) -> dict:
    """Compare frozen privileged fits with the clean validation targets."""
    from scripts import iclr_reference_fits as fit
    from scripts import iclr_stress_evaluation as evaluation

    fitted = sealed_read(fitted_root / "frozen_models.json")
    plan = sealed_read(fitted_root / "plan.json")
    ledger = sealed_read(fitted_root / "conditions.json")
    suite = sealed_read(release / "manifest.json")
    if (
        plan["artifact_sha256"] != fitted["plan_sha256"]
        or ledger["design_sha256"] != suite["artifact_sha256"]
        or fitted["protocol"] != fit.VERSION
    ):
        raise ValueError("frozen conditioned reference fit has wrong provenance")
    cells = {c["cell"]: c for c in suite["cells"]}
    summaries = []
    for row in fitted["rows"]:
        anchor = row["benchmark_id"]
        condition = CONDITIONS[anchor]
        variant = cells[condition]
        source = next(r for r in ledger["rows"] if r["cell"] == condition)
        if source["files"] != plan["files"][anchor] or any(
            source["files"][f] != variant["public_files"][f]
            for f in ("train.json", "val.json")
        ):
            raise ValueError("conditioned reference source arrays differ")
        summary = {"cell": condition, "fit_family": anchor, "status": row["status"]}
        if row["status"] == "ready":
            clean = "acquisition_00" if anchor == data.T2 else "sensor_03"
            rows = read_seal(release / "public" / clean / "val.json")["rows"]
            train = read_seal(release / "public" / clean / "train.json")["rows"]
            scales = data.train_scales(train)
            errors = []
            for r in rows:
                predictions, _ = evaluation.reference_predict(
                    row, variant, {"perturbed": r, "control": r}
                )
                errors.append(
                    {
                        c: float(np.mean(((yp - r["targets"][c]) / scales[c]) ** 2))
                        for c, yp in predictions.items()
                    }
                )
            per_target = {c: float(np.mean([e[c] for e in errors])) for c in errors[0]}
            summary.update(
                {
                    "selected_start": row["fit"]["start"],
                    "nfev": row["fit"]["nfev"],
                    "optimizer_converged": row["fit"]["optimizer_converged"],
                    "clean_validation_nmse": float(np.mean(list(per_target.values()))),
                    "clean_validation_per_target": per_target,
                    "acquired_validation_per_target": dict(
                        zip(
                            fit.TARGETS[anchor],
                            row["fit"]["scores"]["validation"],
                            strict=True,
                        )
                    ),
                    "validation_trajectories": len(rows),
                }
            )
        summaries.append(summary)
    return sealed_write(
        output,
        {
            "protocol": "iclr-condition-reference-development-1",
            "method": "fitted_reference_structure",
            "fitted_sha256": fitted["artifact_sha256"],
            "conditions_sha256": ledger["artifact_sha256"],
            "test_data_opened": False,
            "used_for_selection": False,
            "rows": summaries,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stress", type=Path, required=True)
    p.add_argument("--public", type=Path)
    p.add_argument("--fit-root", type=Path)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    if args.fit_root and args.report and not args.public:
        development_report(args.stress, args.fit_root, args.report)
    elif args.public and not args.fit_root and not args.report:
        with public._lock(args.public):
            stage(args.stress, args.public)
    else:
        p.error("choose --public or both --fit-root and --report")
