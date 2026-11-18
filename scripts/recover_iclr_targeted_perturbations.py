"""Recover the six proven pre-optimization packaging failures into a fresh root.

The old evaluation freeze is never removed. Proposals, calls, critic reviews,
data and fitting freezes are copied byte-for-byte. Numerical attempts are allowed
only for the specific missing identity-file failure, with no backend work saved.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.fitter_diagnostic import runtime_identity
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_targeted_perturbations as campaign

PROTOCOL = "iclr-targeted-packaging-recovery-1"
MISSING_RUNNER = "scripts/run_phase_b_fitter_diagnostic.py"


def _files(folder: Path) -> list[Path]:
    """Inventory regular files only; never follow a supplied symbolic link."""
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError(f"required real directory missing: {folder}")
    paths = sorted(folder.rglob("*"))
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in paths):
        raise ValueError(f"links or special files in recovery input: {folder}")
    return [p for p in paths if p.is_file() and not p.name.endswith(".lock")]


def inventory(source: Path) -> tuple[dict, dict[str, Path], list[dict]]:
    """Prove each failed attempt stopped at the missing-file identity read."""
    plan = campaign.verify(source)
    frozen = sealed_read(source / "frozen_models.json")
    freeze = sealed_read(source / "evaluation_freeze.json")
    if (
        frozen["plan_sha256"] != plan["artifact_sha256"]
        or frozen["selection_freeze_sha256"] != freeze["artifact_sha256"]
        or len(frozen["rows"]) != 6
        or any(r["status"] != "no_retained_model" for r in frozen["rows"])
    ):
        raise ValueError("expected the frozen six-seed infrastructure failure")
    copies = {
        n: source / n
        for n in (
            "plan.json",
            "spec_variants.json",
            "ablation_switch_manifest.json",
            "critic_authorization.json",
        )
    }
    for name in ("cells", "judge_cache", "runtime"):
        for p in _files(source / name):
            copies[str(p.relative_to(source))] = p
    evidence = []
    for block in plan["blocks"]:
        task = block["tasks"][0]
        initial = campaign.search.io.round_path(source, task, 0)
        result = sealed_read(initial / "result.json")
        proposal = sealed_read(initial / "proposal.json")
        review = sealed_read(initial / "critic.json")
        folder = initial / "fit"
        env = public._read(folder / "result.json")
        fit = env["result"]
        missing = fit.get("message", "")
        frozen_fit, request, _, _ = public._load(folder)
        cell = campaign.search.controls.treatment(
            sealed_read(source / "cells" / f"{task['cell']}.json"), "full"
        )
        expected_request = pipeline.request_for(
            proposal["bundle"], campaign.search.view(plan, cell), task, 0
        )
        if (
            result["status"] != "fit_failed"
            or result["selected"] is not None
            or result["task"] != task
            or result["round"] != 0
            or result["proposal_sha256"] != proposal["artifact_sha256"]
            or proposal["task"] != task
            or proposal["round"] != 0
            or proposal["parent_sha256"] is not None
            or proposal["status"] != "constructed"
            or review["identity"] != plan["artifact_sha256"]
            or review["task"] != task
            or review["round"] != 0
            or review["proposal_sha256"] != proposal["artifact_sha256"]
            or fit != result["trial"]["fit"]
            or public.content_sha256(env) != result["trial"]["fit_result_sha256"]
            or public.content_sha256(fit) != env["sha256"]
            or fit["identity"] != frozen_fit["identity"]
            or public._read(folder / "started.json")["identity"] != fit["identity"]
            or request != expected_request
            or frozen_fit["training"] != cell["training"]
            or frozen_fit["validation"] != cell["validation"]
            or fit.get("backend_result_sha256") is not None
            or fit.get("actual_residual_calls") is not None
            or not missing.startswith("FileNotFoundError: [Errno 2] ")
            or not missing.endswith("/" + MISSING_RUNNER + "'")
            or any(p.is_file() for p in (folder / "backend").rglob("*"))
            or (folder / "backend_result.json").exists()
        ):
            raise ValueError(f"not the proven pre-optimization failure: {initial}")
        for i in range(1, plan["visits"]):
            later = campaign.search.io.round_path(source, task, i)
            skipped = sealed_read(later / "result.json")
            if (
                skipped["status"] != "shared_initial_unavailable"
                or skipped["task"] != task
                or skipped["round"] != i
                or skipped["shared_initial_sha256"] != result["artifact_sha256"]
                or any(p.name != "result.json" for p in _files(later))
            ):
                raise ValueError("later visits contain real work; recovery refused")
        for p in _files(initial):
            relative = p.relative_to(initial)
            if relative.parts[0] == "fit":
                if relative.as_posix() != "fit/freeze.json":
                    continue
            elif relative.as_posix() in {"result.json", "cost.json"}:
                continue
            copies[str(p.relative_to(source))] = p
        evidence.append(
            {
                "task": task["task_id"],
                "failed_visit_sha256": result["artifact_sha256"],
                "failed_fit_envelope_sha256": public.content_sha256(env),
                "fit_identity": fit["identity"],
                "failure_message": missing,
                "proposal_sha256": proposal["artifact_sha256"],
                "critic_sha256": review["artifact_sha256"],
                "previous_proposal_cost": proposal["cost"],
            }
        )
    return plan, copies, evidence


def recover(source: Path, destination: Path) -> dict:
    """Copy only prefit checkpoints, with an immutable, resumable provenance seal."""
    source, destination = source.resolve(), destination.resolve()
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("source and recovery root must be disjoint")
    # Exercise the exact file read that failed, before writing or submitting work.
    identity = runtime_identity()
    with campaign.search.io.execution_lease(source, exclusive=True):
        plan, copies, evidence = inventory(source)
        provenance = {
            "protocol": PROTOCOL,
            "source_root": str(source),
            "source_plan_sha256": plan["artifact_sha256"],
            "source_freeze_sha256": sealed_read(source / "evaluation_freeze.json")[
                "artifact_sha256"
            ],
            "runtime_identity": identity,
            "files": {
                name: campaign.search.file_digest(p) for name, p in copies.items()
            },
            "failed_attempts": evidence,
            "policy": (
                "Reuse the exact six initial proposals and reviews; retry only the "
                "proven pre-optimization failures. Preserve all old results. "
                "Continue the original 15 visits without extra seeds or budgets."
            ),
            "recovery_preparation_llm_calls": 0,
            "recovery_preparation_optimizer_calls": 0,
            "test_data_opened": False,
        }
        with public._lock(destination):
            manifest = destination / "packaging_recovery.json"
            if manifest.exists():
                saved = sealed_read(manifest)
                if {
                    k: v for k, v in saved.items() if k != "artifact_sha256"
                } != provenance:
                    raise ValueError("recovery provenance changed")
                for name, digest in saved["files"].items():
                    if campaign.search.file_digest(destination / name) != digest:
                        raise ValueError(f"recovered immutable file changed: {name}")
                return saved
            allowed = {
                ".lock",
                "packaging_recovery_intent.json",
                *copies,
                *(name + ".recovery-tmp" for name in copies),
            }
            if any(
                str(p.relative_to(destination)) not in allowed
                for p in _files(destination)
            ):
                raise ValueError(
                    "destination has unrelated work; use a fresh recovery root"
                )
            sealed_write(destination / "packaging_recovery_intent.json", provenance)
            for name, path in sorted(copies.items()):
                target = destination / name
                if target.exists():
                    if campaign.search.file_digest(target) != provenance["files"][name]:
                        raise ValueError(f"partial recovery file differs: {target}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                # Atomic publication preserves deterministic resume after an interrupt.
                temp = target.with_name(target.name + ".recovery-tmp")
                shutil.copyfile(path, temp)
                temp.replace(target)
            campaign.verify(destination)
            return sealed_write(manifest, provenance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    result = recover(args.source, args.destination)
    print(
        json.dumps(
            {
                "status": "prepared",
                "root": str(args.destination),
                "plan_sha256": result["source_plan_sha256"],
                "reused_initial_proposals": len(result["failed_attempts"]),
                "recovery_sha256": result["artifact_sha256"],
                "llm_calls": 0,
                "optimizer_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
