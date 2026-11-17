"""Separate CPU-only canonical-equation controls on the supplied perturbed CSTR."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from autoformalism.benchmarks.phase_c_release import load_public_cell
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_reference_fits as reference
from scripts import iclr_targeted_perturbations as targeted
from scripts.submit_review_continuation import submit_job

PROTOCOL = "iclr-perturbed-cstr-reference-bounds-1"
PROFILES = ("original_bounds", "wide_same_starts", "wide_distributed_starts")


def identity() -> dict:
    """Bind numerical code, canonical equation implementation and runtime."""
    return {
        "source_sha256": public._source_identity(),
        "scripts": {
            Path(p).name: targeted.search.file_digest(Path(p))
            for p in (__file__, reference.__file__)
        },
        "runtime": public._runtime(),
    }


def bounds(profile: str) -> tuple[float, bool]:
    """Reuse the submitted ±0.5 and expanded ±log(4) multiplier conventions."""
    if profile not in PROFILES:
        raise ValueError("unknown reference-fit bounds profile")
    return (0.5 if profile == "original_bounds" else float(np.log(4))), (
        profile == "wide_distributed_starts"
    )


def prepare(root: Path, inputs: Path) -> dict:
    """Seal nine prescribed CPU starts, separate from construction and test data."""
    targeted.verify_public(inputs)
    if root.resolve().is_relative_to(inputs.resolve()) or (
        inputs.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError("reference fits and public inputs must be disjoint")
    settings = targeted.config()["reference_refits"]
    _, train, val = load_public_cell(inputs / targeted.CSTR)
    with public._lock(root):
        data = sealed_write(
            root / "data.json",
            {
                "training": train.model_dump(mode="json"),
                "validation": val.model_dump(mode="json"),
            },
        )
        return sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                **identity(),
                "cell": targeted.CSTR,
                "equation_family": reference.CSTR,
                "data_sha256": data["artifact_sha256"],
                "public_files": targeted.config()["public_files"][targeted.CSTR],
                "max_evals": settings["maximum_function_evaluations"],
                "tasks": [
                    {"index": i, "profile": p, "start": s}
                    for i, (p, s) in enumerate(
                        (p, s) for p in settings["profiles"] for s in settings["starts"]
                    )
                ],
                "coefficient_scales": reference.coefficients(reference.CSTR),
                "access": (
                    "Privileged canonical equations and scale priors; public data only."
                ),
                "llm_calls": 0,
                "test_data_opened": False,
                "selection": "Lowest validation NMSE within each fixed bounds profile.",
            },
        )


def verify(root: Path) -> dict:
    """Refuse an incompatible continuation or changed training/validation arrays."""
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] != PROTOCOL or any(plan[k] != v for k, v in identity().items()):
        raise ValueError("reference-fit source or runtime changed")
    if sealed_read(root / "data.json")["artifact_sha256"] != plan["data_sha256"]:
        raise ValueError("reference-fit public data changed")
    return plan


def run(root: Path, index: int) -> dict:
    """Fit one prescribed start, caching each exact residual/Jacobian evaluation."""
    from autoformalism.schemas.public_fitting import PublicSplit

    plan = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("task index outside frozen plan")
    task = plan["tasks"][index]
    folder = root / "fits" / f"task-{index:02d}"
    with public._lock(folder):
        path = folder / "result.json"
        if path.exists():
            result = sealed_read(path)
            if (
                result["plan_sha256"] != plan["artifact_sha256"]
                or result["task"] != task
            ):
                raise ValueError("reference-fit checkpoint identity differs")
            return result
        data = sealed_read(root / "data.json")
        train, val = (
            PublicSplit.model_validate(data[k]) for k in ("training", "validation")
        )
        width, distributed = bounds(task["profile"])
        try:
            result = reference.fit_start(
                reference.CSTR,
                train,
                val,
                task["start"],
                plan["max_evals"],
                folder,
                bound_width=width,
                distributed_start=distributed,
            )
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            result = {"status": "fit_failed", "reason": str(exc)}
        return sealed_write(
            path,
            {
                **result,
                "cell": targeted.CSTR,
                "equation_family": reference.CSTR,
                "task": task,
                "plan_sha256": plan["artifact_sha256"],
                "log_bound_width": width,
                "distributed_start": distributed,
            },
        )


def finish(root: Path) -> Path:
    """Retain all starts and failures; export every fit and one pick per profile."""
    plan = verify(root)
    rows, fits, parameters = [], [], []
    for task in plan["tasks"]:
        path = root / "fits" / f"task-{task['index']:02d}" / "result.json"
        if not path.is_file():
            raise ValueError("finish requires all nine reference-fit dispositions")
        fit = sealed_read(path)
        if fit["plan_sha256"] != plan["artifact_sha256"] or fit["task"] != task:
            raise ValueError("reference fit is from a different plan")
        fits.append(fit)
        rows.append(
            {
                **task,
                "status": fit["status"],
                "training_nmse": np.mean(fit["scores"]["training"])
                if "scores" in fit
                else None,
                "validation_nmse": np.mean(fit["scores"]["validation"])
                if "scores" in fit
                else None,
                "optimizer_converged": fit.get("optimizer_converged"),
                "nfev": fit.get("nfev"),
                "max_grid_discrepancy": fit.get("max_grid_discrepancy"),
            }
        )
        for name, theta in zip(
            fit.get("coefficient_names", []), fit.get("theta", []), strict=True
        ):
            scale = plan["coefficient_scales"][name]
            width = fit["log_bound_width"]
            parameters.append(
                {
                    **task,
                    "parameter": name,
                    "value": scale * np.exp(theta),
                    "log_multiplier": theta,
                    "scale": scale,
                    "lower": scale * np.exp(-width),
                    "upper": scale * np.exp(width),
                    "bound_hit": abs(abs(theta) - width) <= 1e-6,
                }
            )
    picks = []
    for profile in PROFILES:
        eligible = [
            f
            for f in fits
            if f["task"]["profile"] == profile and f["status"] == "ready"
        ]
        best = (
            min(
                eligible,
                key=lambda f: (np.mean(f["scores"]["validation"]), f["task"]["start"]),
            )
            if eligible
            else None
        )
        picks.append(
            {
                "profile": profile,
                "status": "ready" if best else "fit_failed",
                "fit": best,
            }
        )
    sealed_write(
        root / "frozen_models.json",
        {
            "protocol": PROTOCOL,
            "plan_sha256": plan["artifact_sha256"],
            "equation_family": reference.CSTR,
            "cell": targeted.CSTR,
            "rows": picks,
            "scientific_recovery_certified": False,
            "test_data_opened": False,
        },
    )
    targeted.search.write_csv(root / "reference_by_start.csv", rows)
    if parameters:
        targeted.search.write_csv(root / "reference_parameters.csv", parameters)
    final = root / "perturbed-cstr-reference-fits.zip"
    with TemporaryDirectory(dir=root) as tmp:
        archive = Path(tmp) / "export.zip"
        members = [
            root / n
            for n in ("plan.json", "frozen_models.json", "reference_by_start.csv")
        ]
        members += sorted((root / "fits").glob("task-*/result.json"))
        if parameters:
            members.append(root / "reference_parameters.csv")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in members:
                z.write(p, str(p.relative_to(root)))
        os.replace(archive, final)
    return final


def submit(root: Path, wave: str, account: str, platform: str) -> str:
    """Submit only unfinished starts, preserving uncertain scheduler replies."""
    if platform not in {"aces", "delta"} or not re.fullmatch(r"wave[1-9][0-9]*", wave):
        raise ValueError("expected aces/delta and wave1, wave2, ...")
    plan = verify(root)
    with public._lock(root / "submissions"):
        folder = root / "submissions" / wave
        if (folder / "array.id").exists():
            return (folder / "array.id").read_text().strip()
        for intent in (root / "submissions").glob("*/array.intent.json"):
            if not intent.with_name("array.id").is_file():
                raise ValueError(f"uncertain Slurm reply; reconcile {intent.parent}")
        prior = [
            p.read_text().strip() for p in (root / "submissions").glob("*/array.id")
        ]
        if prior:
            active = subprocess.check_output(
                ["squeue", "-h", "-u", os.environ["USER"], "-o", "%F"], text=True
            )
            if set(active.split()).intersection(prior):
                raise ValueError("a reference-fit wave is still active")
        indices = [
            t["index"]
            for t in plan["tasks"]
            if not (root / "fits" / f"task-{t['index']:02d}" / "result.json").exists()
        ]
        if not indices:
            return "no pending reference fits"
        folder.mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        os.environ["AF_REFERENCE_ROOT"] = str(root.resolve())
        os.environ["AF_PLATFORM"] = platform
        os.environ.pop("AF_JETSTREAM_API_KEY", None)
        return submit_job(
            folder,
            "array",
            [
                f"--account={account}",
                "--partition=cpu",
                "--job-name=iclr-cstr-refit",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=2",
                "--mem=16G",
                "--time=04:00:00",
                "--export=ALL",
                "--array=" + ",".join(map(str, indices)) + "%9",
                f"--output={root}/logs/{wave}-%A_%a.out",
                f"--error={root}/logs/{wave}-%A_%a.err",
            ],
            targeted.search.io.REPO / "scripts/hpc/iclr_targeted_refits.sh",
            "worker",
            0,
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "run", "finish", "submit"))
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--public", type=Path)
    p.add_argument("--index", type=int)
    p.add_argument("--wave", default="wave1")
    p.add_argument("--platform", choices=("aces", "delta"), default="delta")
    p.add_argument("--account")
    a = p.parse_args()
    if a.action == "prepare":
        if a.public is None:
            p.error("prepare requires --public")
        result = prepare(a.root, a.public)["artifact_sha256"]
    elif a.action == "run":
        if a.index is None:
            p.error("run requires --index")
        r = run(a.root, a.index)
        result = {"status": r["status"], "task": r["task"]}
    elif a.action == "finish":
        result = str(finish(a.root))
    else:
        if a.account is None:
            p.error("submit requires --account")
        result = submit(a.root, a.wave, a.account, a.platform)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
