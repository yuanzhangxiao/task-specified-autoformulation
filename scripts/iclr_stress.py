"""Prepare and evaluate the ICLR data quality and procedural scaling experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_stress_data as data

DEFAULT_RELEASE = Path("artifacts/phase-c-development-final-v2")
DEFAULT_CORRECTED = (
    Path("artifacts/reference-integrity-audit-2026-09-26-v3/public-development")
    / data.ORIGINAL
)
SEEDS = (0, 1, 2)
ARMS = ("full", "fixed_gamma")
PROTOCOL = "iclr-acquisition-and-scaling-1"


def excitation_audit(root: Path, cells: list[dict]) -> None:
    """Measure input-driven variation of each retained hidden coordinate on training."""
    path = root / "scaling_excitation.json"
    systems = [
        c for c in cells if c["study"] == "scaling" and c["qualification"]["passed"]
    ]
    ids = {c["cell"]: c["realization_sha256"] for c in systems}
    if path.is_file():
        if sealed_read(path)["realizations"] != ids:
            raise ValueError("excitation audit belongs to different systems")
        return
    quiet, pulse = data.device_protocols("train")[:2]
    measurements = []
    for cell in systems:
        spec = read_seal(root / "evaluator" / cell["cell"] / "system.json")["equations"]
        base = data._simulate_alien(quiet, spec, {}, data.SOLVER)
        forced = data._simulate_alien(pulse, spec, {}, data.SOLVER)
        differences = np.max(abs(forced.states[:, :-1] - base.states[:, :-1]), axis=0)
        measurements.append(
            {
                "cell": cell["cell"],
                "peak_input_driven_excursions": differences.tolist(),
                "all_excited": bool(np.all(differences > 1e-6)),
            }
        )
    sealed_write(
        path,
        {
            "realizations": ids,
            "threshold": 1e-6,
            "training_only": True,
            "changed_draw_selection": False,
            "measurements": measurements,
        },
    )


def prepare(
    root: Path, release: Path, corrected: Path, study: str, extended: bool
) -> dict:
    """Bind source, signal design, and planned discovery roster."""
    source = read_seal(release / "summary.json")
    if (
        source["protocol"] != "phase-c-development-2"
        or not source["whole_phase_c_roster_ready"]
    ):
        raise ValueError("a verified complete Phase C development release is required")
    ids = {data.T2, data.ALIEN, data.CSTR.format("easy"), data.CSTR.format("hard")}
    for row in source["cells"]:
        if row["cell"] in ids:
            for path, sha in row["files"].items():
                if path.startswith("public/") and data.digest(release / path) != sha:
                    raise ValueError(f"public source digest mismatch: {path}")
    identity = sealed_write(
        root / "design.json",
        {
            "protocol": PROTOCOL,
            "source_release_sha256": data.digest(release / "summary.json"),
            "data_code_sha256": data.digest(Path(data.__file__)),
            "runtime": public._runtime(),
            "study": study,
            "extended_scaling": extended,
            "algorithm_seeds": list(SEEDS),
            "arms": list(ARMS),
            "noise_seed": data.NOISE_SEED,
            "generator_seeds": list(data.DRAW_SEEDS),
            "dimensions": [4, 8, 12],
            "test_design": (
                "six reserved Phase C family schedules, or three novel device schedules"
            ),
            "acquisition": (
                "Strides 1/2/4 plus input corners. Gaussian perturbation paired by "
                "trajectory and channel, including initial observations and "
                "auxiliaries. "
                "Constant channels stay exact. All test arms score against clean "
                "targets. No derivative labels are supplied. Training scales are "
                "estimated per condition."
            ),
            "generator_acceptance": (
                "First of 8 registered-grammar draws with training forced/control "
                "peak effect between 0.05 and 8 and maximum finite-difference output "
                "derivative of each internal initial coordinate >=0.005. "
                "Unit-amplitude pulse. Record all rejected draws. Stress variants "
                "reuse the qualified dimension-8 draw."
            ),
            "interpretation": (
                "Observed output and unit input stay fixed. Dynamical dimension "
                "includes generated output; it does not determine minimum "
                "realization size."
            ),
        },
    )
    # Keep these imported numerical dependencies independent of unrelated project work.
    src = Path(public.__file__).resolve().parents[1]
    sealed_write(
        root / "generator_environment.json",
        {
            "design_sha256": identity["artifact_sha256"],
            "source_files": {
                f: data.digest(src / f)
                for f in (
                    "benchmarks/phase_b_generation.py",
                    "benchmarks/phase_c_protocols.py",
                    "benchmarks/phase_b_public.py",
                    "reference_integration.py",
                    "input_schedule.py",
                    "rebuttal/dalla_man.py",
                    "benchmarks/audited_release.py",
                )
            },
        },
    )
    sealed_write(
        root / "source_files.json",
        {
            "design_sha256": identity["artifact_sha256"],
            "cells": {
                cell: {
                    name: data.digest(release / "public" / cell / name)
                    for name in ("train.json", "val.json", "specification.json")
                }
                for cell in sorted(ids)
            },
        },
    )
    cells = []
    if study in {"all", "data"}:
        cells.extend(data.accuracy_conditions(release, root))
        cells.append(data.corrected_cell(corrected, root))
    if study in {"all", "scaling"}:
        for dimension in (4, 8, 12):
            for seed in data.DRAW_SEEDS:
                base = data.scaling_cell(root, release, dimension, seed, "dimension")
                cells.append(base)
                if extended and dimension == 8:
                    eq = read_seal(root / "evaluator" / base["cell"] / "system.json")[
                        "equations"
                    ]
                    for name in ("double_recurrent_coupling", "wider_decay_spread"):
                        cells.append(
                            data.scaling_cell(root, release, dimension, seed, name, eq)
                        )
    excitation_audit(root, cells)
    rows = [
        {"benchmark_id": c["cell"], "arm": arm, "repetition": s}
        for c in cells
        for s in SEEDS
        for arm in ARMS
    ]
    sealed_write(
        root / "data_robustness_manifest.json",
        {
            "design_sha256": identity["artifact_sha256"],
            "cells": [
                c for c in cells if c["study"] in {"data_quality", "corrected_easy"}
            ],
            "public_data": "Only the public/<cell> directories enter discovery.",
        },
    )
    sealed_write(
        root / "scaling_system_manifest.json",
        {
            "design_sha256": identity["artifact_sha256"],
            "projections": (
                "v01 = output y; u01 = commanded input; shared zero hidden preparation."
            ),
            "systems": [
                {**c, **read_seal(root / "evaluator" / c["cell"] / "system.json")}
                for c in cells
                if c["study"] == "scaling"
            ],
        },
    )
    return sealed_write(
        root / "manifest.json",
        {
            "protocol": PROTOCOL,
            "design_sha256": identity["artifact_sha256"],
            "cells": cells,
            "runs": rows,
            "arms": list(ARMS),
            "algorithm_seeds": list(SEEDS),
            "test_data_generated": False,
            "provider_calls": 0,
            "fitting_calls": 0,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    cmd = p.add_subparsers(dest="command", required=True)
    make = cmd.add_parser("prepare")
    make.add_argument("--root", type=Path, required=True)
    make.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    make.add_argument("--corrected", type=Path, default=DEFAULT_CORRECTED)
    make.add_argument("--study", choices=("all", "data", "scaling"), default="all")
    make.add_argument("--extended-scaling", action="store_true")
    ev = cmd.add_parser("evaluate")
    ev.add_argument("--root", type=Path, required=True)
    ev.add_argument("--models", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument(
        "--reference-root", type=Path, help="Optional frozen B1 reference-fit root"
    )
    ev.add_argument("--data-root", type=Path, default=Path("data_raw"))
    rp = cmd.add_parser("report")
    rp.add_argument("--root", type=Path, required=True)
    rp.add_argument("--evaluations", type=Path, nargs="*", default=[])
    rp.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        with public._lock(args.root):
            result = prepare(
                args.root,
                args.release,
                args.corrected,
                args.study,
                args.extended_scaling,
            )
        print(
            json.dumps(
                {
                    "cells": len(result["cells"]),
                    "planned_runs": len(result["runs"]),
                    "manifest_sha256": result["artifact_sha256"],
                },
                indent=2,
            )
        )
    else:
        from scripts import iclr_stress_evaluation as evaluation

        manifest = sealed_read(args.root / "manifest.json")
        with public._lock(args.output):
            if args.command == "evaluate":
                evaluation.evaluate(
                    args.root,
                    manifest,
                    args.models,
                    args.data_root,
                    args.output,
                    args.reference_root,
                )
            else:
                evaluation.report(manifest, args.evaluations, args.output)


if __name__ == "__main__":
    main()
