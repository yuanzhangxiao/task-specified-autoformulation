"""Run the three anchor-configured Full searches for the perturbed T1-easy table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import component_critic as critic
from autoformalism.rebuttal import iclr_controlled_search as search
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts.iclr_anchor_fidelity import verify_controlled_bundle
from scripts.iclr_controlled_receipts import export as export_receipts

CELL = "phase_c_dalla_man_t1_perturbed_named_easy_rates_v1"
ANCHOR_PLAN = "df11f54897898ceeb21eb635938af8eded1ffaea5bed698e02bee46ae0fdb34a"
FILES = {
    "specification.json": (
        "55295fa110b494240454cfef2e5ff13fec77c0ec2075f15b60333f1e9fddf469"
    ),
    "proposer_prompt.txt": (
        "03367bc88a0cda7478961c0209a2ac354fc9a4bed644c8709dcba7c23dd5574b"
    ),
    "train.json": "8c935212a9e84aff703568db7cb06d8ed3793d8aac3c0f8b1243166aff8c95d9",
    "val.json": "6624c51815a579f565c22e99551a45d11186f09617b5dedfe7b2383838168d05",
}
SEEDS = [0, 1, 2]
CONFIG = (
    "model_settings",
    "fit_profile",
    "serving_image_sha256",
    "max_parameters",
    "visits",
    "seeds",
    "critic_enabled",
    "pruning_enabled",
    "runtime",
    "protocol",
    "test_data_opened",
    "private_reference_opened",
)


def prepare(root: Path, public_dir: Path, anchor: Path, gate: Path) -> dict:
    """Prepare Full on new public data and bind the effective failed-start rule."""
    original = sealed_read(anchor / "plan.json")
    if (
        original["artifact_sha256"] != ANCHOR_PLAN
        or original["runtime"] != public._runtime()
    ):
        raise ValueError("expected the original anchor Full plan and fitting runtime")
    if {n: search.file_digest(public_dir / CELL / n) for n in FILES} != FILES:
        raise ValueError("public files differ from the perturbed-T1 anchor panel")
    original_auth = critic.checked_authorization(anchor, original)
    if (
        sealed_read(gate / "plan.json")["artifact_sha256"]
        != original_auth["calibration_plan_sha256"]
        or content_hash(json.loads((gate / "summary.json").read_text()))
        != original_auth["summary_sha256"]
    ):
        raise ValueError("different critic calibration")
    if root.resolve().is_relative_to(
        public_dir.resolve()
    ) or root.resolve().is_relative_to(anchor.resolve()):
        raise ValueError("use a separate campaign directory")
    # Generate cell contracts with the established search preparer. Install the
    # final plan once, with the original paired Full's failed-start rule made explicit.
    with public._lock(root), TemporaryDirectory() as tmp:
        staged = Path(tmp) / "search"
        base = search.prepare(
            staged, public_dir, [CELL], ["full"], SEEDS, 15, True, True
        )
        cell = sealed_read(staged / "cells" / f"{CELL}.json")
        if any(base[k] != original[k] for k in CONFIG) or cell["files"] != FILES:
            raise ValueError("search settings or public inputs differ")
        variants = sealed_read(staged / "spec_variants.json")
        if (
            variants["variants"][CELL]["full"]["prompt"]
            != search.controls.treatment(cell, "full")["active_prompt"]
        ):
            raise ValueError("incorrect active Full prompt")
        anchors = {
            b["seed"]: next(t for t in b["tasks"] if t["treatment"] == "full")
            for b in original["blocks"]
            if b["cell"] == "acquisition_00"
        }
        roles = (
            "arm",
            "treatment",
            "scientific_verifier",
            "critic",
            "shared_processes",
            "secondary",
        )
        for block in base["blocks"]:
            (task,) = block["tasks"]
            if any(task[k] != anchors[block["seed"]][k] for k in roles):
                raise ValueError("a Full treatment switch differs from the anchor")
        switches = sealed_read(staged / "ablation_switch_manifest.json")
        switches["initial_construction"] = (
            "Full constructs its fitted start at visit zero."
        )
        switches.pop("artifact_sha256")
        base.pop("artifact_sha256")
        sealed_write(
            root / "cells" / f"{CELL}.json",
            {k: v for k, v in cell.items() if k != "artifact_sha256"},
        )
        sealed_write(
            root / "spec_variants.json",
            {k: v for k, v in variants.items() if k != "artifact_sha256"},
        )
        switches = sealed_write(root / "ablation_switch_manifest.json", switches)
        plan = sealed_write(
            root / "plan.json",
            {
                **base,
                "switch_manifest_sha256": switches["artifact_sha256"],
                "require_fitted_full_start": True,
                "reference_plan_sha256": ANCHOR_PLAN,
                "study": "Perturbed T1-easy Full, matched to anchor 2337694.",
            },
        )
        critic.bind_calibration(root, plan, gate / "plan.json", gate / "summary.json")
    return plan


def finish(root: Path) -> Path:
    """Freeze every assigned seed and package the native Full endpoints."""
    plan = sealed_read(root / "plan.json")
    if (
        plan.get("reference_plan_sha256") != ANCHOR_PLAN
        or plan["arms"] != ["full"]
        or not plan.get("require_fitted_full_start")
        or plan["seeds"] != SEEDS
        or set(plan["cells"]) != {CELL}
        or plan["runtime"] != public._runtime()
        or plan["source_sha256"] != public._source_identity()
    ):
        raise ValueError("expected the matched perturbed-T1 Full plan and runtime")
    critic.checked_authorization(root, plan)
    bundle = search.report(root, True)
    verify_controlled_bundle(root / "frozen_models.json", bundle)
    if sorted(
        (r["benchmark_id"], r["method"], r["repetition"]) for r in bundle["rows"]
    ) != [(CELL, "full", s) for s in SEEDS]:
        raise ValueError("frozen seed roster differs")
    audit = export_receipts(root, root / "endpoint-audit")
    if audit["counts"]["visits"] != 45:
        raise ValueError("expected fifteen visit dispositions for each seed")
    records = []
    for r in sorted(bundle["rows"], key=lambda r: r["repetition"]):
        if r["public_file_sha256"] != FILES:
            raise ValueError("endpoint public input hashes differ")
        if r["status"] == "ready":
            if (
                public.content_sha256(r["candidate"]) != r["model_sha256"]
                or not math.isfinite(r["validation_nmse"])
                or not all(
                    isinstance(r[k], dict)
                    for k in ("context", "parameters", "initials")
                )
            ):
                raise ValueError("incomplete frozen model")
        elif r["status"] != "no_retained_model" or r.get("validation_nmse") is not None:
            raise ValueError("unexpected terminal status")
        records.append(
            {
                "seed": r["repetition"],
                "status": r["status"],
                "validation_nmse": r.get("validation_nmse"),
                "source_id": r["source_id"],
                "model_sha256": r.get("model_sha256"),
                "endpoint_sha256": r["source_sha256"],
            }
        )
    files = (
        "frozen_models.json",
        "evaluation_freeze.json",
        "plan.json",
        "spec_variants.json",
        "ablation_switch_manifest.json",
        "paired_ablation_results.csv",
        "critic_authorization.json",
    )
    final = root / "t1-perturbed-full.zip"
    with TemporaryDirectory(dir=root) as temp:
        directory = Path(temp)
        with (directory / "validation_by_seed.csv").open("w", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
        (directory / "READ_ME.txt").write_text(
            "Perturbed named T1-easy, Phase C continuous rates. Full matches anchor "
            "array 2337694 with the 20B proposer, calibrated 120B critic, 15 visits, "
            "public validation selection and pruning.\n"
            "frozen_models.json holds each native candidate, context, parameters and "
            "initials. An empty initials dictionary uses candidate.initial_conditions "
            "and fitted initializer parameters. validation_by_seed.csv includes all "
            "three seeds and each available recorded validation NMSE.\n"
            "For Claude, run verify_controlled_bundle with the adjacent plan.json "
            "and evaluation_freeze.json. Use confirmation panel "
            "3f7f86cb442dafa3f6f97fb35984ddd2b8233edaf97bf3d759d1fc62cbc73695 "
            "and the scorer on t1-perturbed-bound-interventions at ba6e36b. Its "
            "reference-only CLI needs a learned-subject adapter. Use a fresh output "
            "directory, set the controlled source protocol and direct_readouts={} "
            "unless independently mapped, and retain the panel references and "
            "training SD. The requested contrasts are fed vs fasting and split "
            "vs fed. Missing models retain the native no_retained_model status.\n"
        )
        members = {name: root / name for name in files}
        members.update(
            {
                name: directory / name
                for name in ("validation_by_seed.csv", "READ_ME.txt")
            }
        )
        (directory / "SHA256SUMS").write_text(
            "".join(
                f"{search.file_digest(path)}  {name}\n"
                for name, path in sorted(members.items())
            )
        )
        members["SHA256SUMS"] = directory / "SHA256SUMS"
        with zipfile.ZipFile(
            directory / "export.zip", "w", compression=zipfile.ZIP_DEFLATED
        ) as z:
            for name, path in members.items():
                z.write(path, name)
        with zipfile.ZipFile(directory / "export.zip") as z:
            if z.testzip() is not None:
                raise ValueError("export ZIP failed integrity check")
        os.replace(directory / "export.zip", final)
    return final


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "finish"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--public", type=Path)
    p.add_argument("--anchor", type=Path)
    p.add_argument("--gate", type=Path)
    args = p.parse_args()
    if args.action == "prepare":
        if not all((args.public, args.anchor, args.gate)):
            p.error("prepare requires --public, --anchor and --gate")
        result = prepare(args.root, args.public, args.anchor, args.gate)
        print(
            json.dumps(
                {
                    "plan_sha256": result["artifact_sha256"],
                    "cell": CELL,
                    "seeds": SEEDS,
                    "model": result["model_settings"]["model"],
                    "visits": result["visits"],
                }
            )
        )
    else:
        print(finish(args.root))
