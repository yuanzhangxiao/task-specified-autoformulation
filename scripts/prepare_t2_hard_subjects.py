#!/usr/bin/env python3
"""Freeze T2-hard incumbents at six visits and external subjects for replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from itertools import product
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.rebuttal.final_evaluation import FrozenEvaluationSubject
from autoformalism.rebuttal.mechanism_audit_sources import BUNDLE_PROTOCOL, d3_rows
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import CandidateModel
from autoformalism.staged_topology import content_hash

CELL = "phase_b_dalla_man_t2_canonical_named_hard"
ROUND = 5  # Six visits including construction, fixed before looking at new outcomes.
PROMPT = "563356816ed4d6f33ed4bbe12538ba077232455588eab46b8fe7a8801ebbf210"
METHODS = {"sindy", "pysr", "raw_data_agent", "d3"}


def verify_seal(record: dict) -> str:
    """Verify an original campaign SHA without deserializing executable code."""
    body = {key: value for key, value in record.items() if key != "artifact_sha256"}
    if content_hash(body) != record.get("artifact_sha256"):
        raise ValueError("input artifact digest differs")
    return record["artifact_sha256"]


def component_rows(inventory: dict) -> list[dict]:
    """Keep the primary six-visit incumbents and all missing outcomes."""
    if (
        inventory["protocol"] != "component-model-inspection-1"
        or inventory["campaign_protocol"] != "final-component-campaign-1"
        or inventory["test_data_opened"] is not False
        or inventory["public_context"][CELL]["target_contract"]["public_prompt_sha256"]
        != PROMPT
    ):
        raise ValueError("wrong component campaign, prompt or split")
    selections = {}
    for entry in inventory["occurrences"]:
        task = entry["task"]
        if (
            task["cell"] == CELL
            and not task["secondary"]
            and entry["round"] == ROUND
            and entry["role"] == "selected"
        ):
            if task["task_id"] in selections:
                raise ValueError("duplicate selected incumbent")
            selections[task["task_id"]] = entry
    rows = []
    for visit in inventory["visits"]:
        t = visit["task"]
        if t["cell"] != CELL or t["secondary"] or visit["round"] != ROUND:
            continue
        if t["shared_processes"] is not True:
            raise ValueError("expected primary shared-process lineage")
        components = f"c{int(t['critic'])}v{int(t['scientific_verifier'])}s1"
        method = f"autoformalism:{t['arm']}:{components}"
        task_id = t["task_id"]
        row = {
            "benchmark_id": CELL,
            "tier": "hard",
            "method": method,
            "repetition": t["seed"],
            "source_id": task_id,
            "semantics": "continuous_time",
            "task": t,
            "source_round": ROUND,
            "public_prompt_sha256": PROMPT,
            "source_plan_sha256": inventory["plan_sha256"],
            "status": "unavailable",
            "source_terminal_status": visit["status"],
            "error": "no saved selected incumbent at the fixed visit",
        }
        rows.append(row)
        entry = selections.get(task_id)
        if entry is None:
            continue
        model = inventory["models"][entry["model_id"]]
        fit, provenance = entry["fit"], entry["provenance"]
        if (
            entry["task"] != t
            or content_hash(model) != entry["model_id"]
            or model["cell"] != CELL
            or model["candidate_representation"] != "lowered"
            or model["parameter_status"] != "saved_fit_vector"
            or content_sha256(model["candidate"]) != fit["lowered_candidate_sha256"]
            or model["parameters"] != fit["parameters"]
            or fit["status"] != "complete"
            or fit["training_only_parameter_estimation"] is not True
            or fit["validation_initials_fitted"] is not False
            or provenance["origin_task"] != task_id
            or not 0 <= provenance["origin_round"] <= ROUND
        ):
            raise ValueError(
                f"fitted model identity or selection differs for {task_id}"
            )
        candidate = CandidateModel.model_validate(model["candidate"])
        context = ValidationContext.model_validate(model["lowered_context"])
        if (
            set(model["parameters"]) != {p.name for p in candidate.parameters}
            or tuple(context.targets) != ("Gp", "I")
            or tuple(context.auxiliaries) != ("Uii",)
        ):
            raise ValueError(f"parameter or channel identity differs for {task_id}")
        row.update(
            status="ready",
            error=None,
            candidate=model["candidate"],
            context=model["lowered_context"],
            parameters=model["parameters"],
            initials={},
            source_model_sha256=entry["model_id"],
            source_fit_sha256=provenance["fit_result_sha256"],
            source_result_sha256=entry["source_sha256"],
            selected_origin_round=provenance["origin_round"],
            training_nmse=fit["training"]["normalized_mse"],
            validation_nmse=fit["validation"]["normalized_mse"],
        )
    expected = {
        (f"autoformalism:{arm}:c{c}v{v}s1", seed)
        for arm, c, v, seed in product(("full", "brief_only"), (0, 1), (0, 1), (0, 1))
    }
    if (
        len(rows) != len(expected)
        or {(r["method"], r["repetition"]) for r in rows} != expected
    ):
        raise ValueError("incomplete or duplicate planned component roster")
    return rows


def baseline_subjects(plan: dict) -> list[dict]:
    """Use development subjects, preserving entries with unavailable native fits."""
    if (
        plan["protocol"] != "saved-baseline-validation-1"
        or plan["test_data_opened"] is not False
    ):
        raise ValueError("wrong baseline campaign or split")
    rows = {}
    for source in plan["rows"]:
        if source["benchmark_id"] != CELL or source["cohort"] != "phase_b":
            continue
        method, seed = source["source_kind"], source["repetition"]
        if (method, seed) in rows or source["tier"] != "hard":
            raise ValueError("duplicate or mismatched external row")
        row = {
            "benchmark_id": CELL,
            "tier": "hard",
            "method": method,
            "repetition": seed,
            "source_plan_sha256": plan["artifact_sha256"],
            "source_id": f"plan#row={source['index']}",
            "source_file_sha256": sorted(source.get("source_files", {}).values()),
            "semantics": "native_increment" if method == "d3" else "continuous_time",
            "status": "unavailable",
            "source_terminal_status": source["status"],
            "error": source.get("reason")
            or "no fitted model in the saved development plan",
        }
        rows[method, seed] = row
        if method == "d3":
            row["error"] = "native fitted selection required separately"
            continue
        if source["status"] != "ready":
            continue
        s = FrozenEvaluationSubject.model_validate(source["subject"])
        if (
            (s.benchmark_id, s.tier, s.repetition) != (CELL, "hard", seed)
            or source["data_identity"]["prompt"] != PROMPT
            or s.parameterization.status not in {"available", "not_required"}
            or s.method
            != {"raw_data_agent": "raw_data_agent:openai:gpt-5.6-sol"}.get(
                method, method
            )
        ):
            raise ValueError(
                "baseline selection, fitted values or public prompt differ"
            )
        row.update(
            status="ready",
            error=None,
            candidate=s.candidate.model_dump(mode="json"),
            parameters=s.parameterization.global_parameters,
            initials=s.parameterization.global_initial_conditions,
            context=s.validation_context.model_dump(mode="json"),
            public_prompt_sha256=PROMPT,
            source_subject_id=s.subject_id,
            data_identity=source["data_identity"],
        )
    if set(rows) != set(product(METHODS, range(3))):
        raise ValueError(
            "expected external roster of four methods and three repetitions"
        )
    return list(rows.values())


def supplement_d3(bundle_path: Path, native_d3: Path, output: Path) -> dict:
    """Add native Delta replay subjects directly to the portable roster."""
    bundle = sealed_read(bundle_path)
    if (
        bundle.get("snapshot_protocol") != "iclr-t2-hard-six-visits-1"
        or bundle["test_data_opened"] is not False
    ):
        raise ValueError("expected frozen six-visit diagnostic bundle")
    native = [r for r in d3_rows(native_d3) if r["benchmark_id"] == CELL]
    old = [r for r in bundle["rows"] if r["method"] == "d3"]
    if any(
        sorted(r["repetition"] for r in group) != [0, 1, 2] for group in (old, native)
    ):
        raise ValueError("expected three native D3 entries per source")
    if any(r["status"] != "unavailable" for r in old):
        raise ValueError("previously frozen D3 selections cannot be replaced")
    if any(
        r["tier"] != "hard" or r["data_identity"]["prompt"] != PROMPT for r in native
    ):
        raise ValueError("native D3 public interface differs")
    parent_sha = bundle.pop("artifact_sha256")
    revised = [r for r in bundle["rows"] if r["method"] != "d3"] + native
    return sealed_write(
        output,
        {
            **bundle,
            "source_subject_bundle_sha256": parent_sha,
            "rows": sorted(revised, key=lambda r: (r["method"], r["repetition"])),
        },
    )


def export(inventory_path: Path, baseline_path: Path, output: Path) -> dict:
    """Verify both archives and persist a deterministic portable subject manifest."""
    with zipfile.ZipFile(inventory_path) as archive:
        inventory = json.loads(archive.read("inventory.json"))
    inventory_seal = verify_seal(inventory)
    with tarfile.open(baseline_path, "r:*") as archive:
        member = archive.getmember("plan.json")
        if not member.isfile():
            raise ValueError("expected a regular baseline plan member")
        with archive.extractfile(member) as stream:
            plan = json.load(stream)
    plan_seal = verify_seal(plan)
    rows = component_rows(inventory) + baseline_subjects(plan)
    return sealed_write(
        output,
        {
            "protocol": BUNDLE_PROTOCOL,
            "snapshot_protocol": "iclr-t2-hard-six-visits-1",
            "benchmark_id": CELL,
            "snapshot_round": ROUND,
            "selection": "Six-visit retained incumbent for each primary arm. "
            "external methods use their native saved selections and different budgets.",
            "source_inventory_sha256": inventory_seal,
            "source_baseline_plan_sha256": plan_seal,
            "inventory_zip_sha256": hashlib.sha256(
                inventory_path.read_bytes()
            ).hexdigest(),
            "baseline_archive_sha256": hashlib.sha256(
                baseline_path.read_bytes()
            ).hexdigest(),
            "rows": sorted(rows, key=lambda r: (r["method"], r["repetition"])),
            "test_data_opened": False,
            "optimizer_calls": 0,
            "llm_calls": 0,
        },
    )


def main() -> None:
    """Build a portable, sealed, fixed-roster replay input without refitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--inventory", type=Path, help="dalla-all-models.zip")
    inputs.add_argument(
        "--retained-bundle", type=Path, help="packaged subjects/models.json"
    )
    parser.add_argument(
        "--baseline-archive",
        type=Path,
        help="baseline-validation-models.tar.gz",
    )
    parser.add_argument(
        "--native-d3-root", type=Path, help="optional native Phase B D3 campaign"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.retained_bundle:
        if not args.native_d3_root or args.baseline_archive:
            parser.error(
                "--retained-bundle requires only --native-d3-root and --output"
            )
        result = supplement_d3(args.retained_bundle, args.native_d3_root, args.output)
    else:
        if not args.baseline_archive or args.native_d3_root:
            parser.error("--inventory requires only --baseline-archive and --output")
        result = export(args.inventory, args.baseline_archive, args.output)
    print(
        json.dumps(
            {"rows": len(result["rows"]), "artifact_sha256": result["artifact_sha256"]}
        )
    )


if __name__ == "__main__":
    main()
