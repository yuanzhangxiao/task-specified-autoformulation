"""Repeat the six Full lineages of ACES anchor array 2337694 with a 120B proposer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import component_critic as critic
from autoformalism.rebuttal import iclr_controlled_search as search
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts.iclr_anchor_fidelity import verify_controlled_bundle
from scripts.iclr_controlled_receipts import export as export_receipts
from scripts.submit_structural_revision_aces import MODELS, snapshot_identity

REFERENCE_PLAN = "df11f54897898ceeb21eb635938af8eded1ffaea5bed698e02bee46ae0fdb34a"
CELLS = {"acquisition_00": "T2-hard", "sensor_00": "CSTR-hard"}


def checked_input(path: Path, expected: str) -> dict:
    """Read design or public inputs authenticated by the reference plan."""
    receipt = sealed_read(path)
    if receipt.pop("artifact_sha256") != expected:
        raise ValueError(f"Input differs from the 20B ACES anchor: {path}")
    return receipt


def prepare(
    root: Path,
    reference: Path,
    gate: Path,
    *,
    cells: list[str] | None = None,
    reference_identity: str | None = None,
) -> dict:
    """Start a new campaign from the original Full tasks and public inputs."""
    root, reference = root.resolve(), reference.resolve()
    reference_identity = reference_identity or REFERENCE_PLAN
    if root.is_relative_to(reference) or reference.is_relative_to(root):
        raise ValueError("Use disjoint 20B reference and 120B output directories")
    original = checked_input(reference / "plan.json", reference_identity)
    names = list(original["cells"]) if cells is None else cells
    if (
        not names
        or len(set(names)) != len(names)
        or set(names) - original["cells"].keys()
    ):
        raise ValueError("select distinct cells from the authenticated reference")
    runtime = public._runtime()
    if original["runtime"] != runtime:
        raise ValueError(
            f"Anchor fitting runtime required: {original['runtime']}. Found {runtime}."
        )
    original_auth = critic.checked_authorization(
        reference, {"artifact_sha256": reference_identity}
    )
    gate_plan, gate_summary = gate / "plan.json", gate / "summary.json"
    checked_input(gate_plan, original_auth["calibration_plan_sha256"])
    if (
        content_hash(json.loads(gate_summary.read_text()))
        != original_auth["summary_sha256"]
    ):
        raise ValueError("Critic calibration summary differs from the 20B anchor")

    cell_data = {
        name: checked_input(reference / "cells" / f"{name}.json", digest)
        for name, digest in original["cells"].items()
        if name in names
    }
    variants = checked_input(
        reference / "spec_variants.json", original["variants_sha256"]
    )["variants"]
    switches = checked_input(
        reference / "ablation_switch_manifest.json", original["switch_manifest_sha256"]
    )
    settings = {
        **original["model_settings"],
        "model": MODELS["120b"]["model"],
        "model_revision": MODELS["120b"]["revision"],
    }
    blocks = []
    reference_tasks = {}
    for block in original["blocks"]:
        if block["cell"] not in names:
            continue
        (full,) = [t for t in block["tasks"] if t["treatment"] == "full"]
        if cells is not None:
            number = len(blocks)
            task_id = f"block{number:04d}_full"
            reference_tasks[task_id] = full["task_id"]
            full = {**full, "block": number, "index": number, "task_id": task_id}
            block = {**block, "block": number}
        blocks.append({**block, "tasks": [full]})
    variant = {"variants": {c: {"full": variants[c]["full"]} for c in cell_data}}
    for name, cell in cell_data.items():
        if (
            search.controls.treatment(cell, "full")["active_prompt"]
            != variants[name]["full"]["prompt"]
        ):
            raise ValueError(f"Full public prompt differs from the 20B anchor: {name}")
    switch = {
        **switches,
        "arms": ["full"],
        "model_settings": settings,
        "initial_construction": "Full builds its own fitted start at visit zero.",
    }
    plan = {
        **original,
        "cells": {c: original["cells"][c] for c in cell_data},
        "arms": ["full"],
        "blocks": blocks,
        "model_settings": settings,
        "study": (
            "Full with a 120B proposer, matched to ACES anchor 2337694."
            if cells is None and reference_identity == REFERENCE_PLAN
            else "Full with a 120B proposer, matched to its public Full reference."
        ),
        "reference_plan_sha256": reference_identity,
        **({"reference_tasks": reference_tasks} if cells is not None else {}),
        "require_fitted_full_start": True,
        "proposer_tensor_parallel_size": MODELS["120b"]["gpus"],
        "source_sha256": public._source_identity(),
        "variants_sha256": content_hash(variant),
        "switch_manifest_sha256": content_hash(switch),
    }
    with public._lock(root):
        if not (root / "plan.json").exists() and any(
            p.name != ".lock" for p in root.iterdir()
        ):
            raise ValueError(
                "120B output contains files without its plan; choose the campaign root"
            )
        result = sealed_write(root / "plan.json", plan)
        for name, cell in cell_data.items():
            sealed_write(root / "cells" / f"{name}.json", cell)
        sealed_write(root / "spec_variants.json", variant)
        sealed_write(root / "ablation_switch_manifest.json", switch)
        critic.bind_calibration(root, result, gate_plan, gate_summary)
    return result


def paired_validation(root: Path, reference: Path) -> list[dict]:
    """Pair frozen validation endpoints by cell and seed, including failed starts."""
    plan = sealed_read(root / "plan.json")
    checked_input(reference / "plan.json", plan["reference_plan_sha256"])
    expected = {(b["cell"], b["seed"]) for b in plan["blocks"]}
    endpoints = []
    for directory in (reference, root):
        path = directory / "frozen_models.json"
        frozen = sealed_read(path)
        verify_controlled_bundle(path, frozen)
        endpoints.append(
            {
                (r["benchmark_id"], r["repetition"]): r
                for r in frozen["rows"]
                if r["method"] == "full"
                and (r["benchmark_id"], r["repetition"]) in expected
            }
        )
    older, newer = endpoints
    if set(older) != expected or older.keys() != newer.keys():
        raise ValueError("Expected the same planned Full endpoints in both rosters")
    rows = []
    for (cell, seed), first in sorted(older.items()):
        second = newer[cell, seed]
        if first["public_file_sha256"] != second["public_file_sha256"]:
            raise ValueError("Different public data across proposers")
        a, b = first.get("validation_nmse"), second.get("validation_nmse")
        rows.append(
            {
                "benchmark": CELLS.get(cell, cell),
                "cell": cell,
                "seed": seed,
                "status_20b": first["status"],
                "status_120b": second["status"],
                "validation_nmse_20b": a,
                "validation_nmse_120b": b,
                "delta_120b_minus_20b": b - a
                if a is not None and b is not None
                else None,
            }
        )
    search.write_csv(root / "paired_proposer_validation.csv", rows)
    return rows


def finish(
    root: Path,
    reference: Path,
    *,
    reference_identity: str | None = None,
    allow_unmatched_reference: bool = False,
) -> list[dict]:
    """Freeze only this matched search, then export its audit and validation pairs."""
    reference_identity = reference_identity or REFERENCE_PLAN
    checked_input(reference / "plan.json", reference_identity)
    plan = sealed_read(root / "plan.json")
    if (
        plan.get("reference_plan_sha256") != reference_identity
        or plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("Expected the matched 120B Full run and its fitting runtime")
    search.report(root, True)
    export_receipts(root, root / "endpoint-audit")
    if (reference / "frozen_models.json").exists():
        return paired_validation(root, reference)
    if not allow_unmatched_reference:
        raise ValueError("matched 20B frozen export is missing")
    rows = sealed_read(root / "frozen_models.json")["rows"]
    table = [
        {
            k: r.get(k)
            for k in (
                "benchmark_id",
                "repetition",
                "status",
                "validation_nmse",
                "source_id",
            )
        }
        for r in rows
    ]
    search.write_csv(root / "validation_by_seed.csv", table)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    prep = stages.add_parser("prepare")
    prep.add_argument("--gate", type=Path, required=True)
    prep.add_argument("--hf-home", type=Path, required=True)
    prep.add_argument("--cells", nargs="+")
    for cmd in (prep, stages.add_parser("finish")):
        cmd.add_argument("--root", type=Path, required=True)
        cmd.add_argument("--reference", type=Path, required=True)
        cmd.add_argument("--reference-plan-sha256", default=REFERENCE_PLAN)
    args = parser.parse_args()
    if args.stage == "prepare":
        snapshot_identity(args.hf_home, "120b")
        plan = prepare(
            args.root,
            args.reference,
            args.gate,
            cells=args.cells,
            reference_identity=args.reference_plan_sha256,
        )
        result = {
            "plan_sha256": plan["artifact_sha256"],
            "cells": list(plan["cells"]),
            "seeds": plan["seeds"],
            "arms": plan["arms"],
            "visits_per_lineage": plan["visits"],
            "proposer": plan["model_settings"]["model"],
            "lineages": len(plan["blocks"]),
        }
    else:
        result = finish(
            args.root,
            args.reference,
            reference_identity=args.reference_plan_sha256,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
