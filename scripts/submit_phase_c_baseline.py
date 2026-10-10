#!/usr/bin/env python3
"""Submit one bounded pilot wave with immutable scheduler receipts."""

import argparse
import json
import os
import re
from pathlib import Path

if __package__:
    from .submit_component_campaign import support
else:
    from submit_component_campaign import support

from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_baseline as baseline
from autoformalism.research import (
    construction_comparison,
    topology_confirmation,
    variable_confirmation,
)


def submit(
    root: Path,
    wave: str,
    *,
    variables_only: bool = False,
    topology_only: bool = False,
    compare_construction: bool = False,
) -> dict:
    if sum((variables_only, topology_only, compare_construction)) > 1:
        raise ValueError("choose one stage-only continuation")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", wave):
        raise ValueError("wave must be a simple identifier")
    controller = (
        construction_comparison
        if compare_construction
        else topology_confirmation
        if topology_only
        else variable_confirmation
        if variables_only
        else baseline
    )
    stage_only = variables_only or topology_only or compare_construction
    label = (
        "comparison"
        if compare_construction
        else "topology"
        if topology_only
        else "variables"
    )
    plan = controller.verify(root)
    live_confirmation = compare_construction and plan.get("study") in {
        "live_confirmation",
        "basin_confirmation",
        "shared_law_comparison",
        "prompt_comparison",
        "refinement_confirmation",
        "variable_checklist_confirmation",
        "stage_check_confirmation",
        "deferred_interaction_confirmation",
    }
    if live_confirmation:
        construction_comparison.check_storage(root)
        label = "confirmation"
    for key in (
        "AF_PYTHON",
        "AF_VLLM_IMAGE",
        "AF_HF_HOME",
        "AF_COMPUTE_CACHE_ROOT",
        "AF_IPC_TMP_ROOT",
    ):
        if not os.environ.get(key):
            raise ValueError(f"set {key}")
    source = support("submit_shared_process_pilot")
    commit = source.source_commit(controller.REPO)
    os.environ.update(
        AF_REPO_ROOT=str(controller.REPO), AF_OUTPUT_ROOT=str(root), AF_COMMIT=commit
    )
    identity = {
        "plan_sha256": plan["artifact_sha256"],
        "commit": commit,
        "wave": wave,
        "resources": f"aces-1h100-{label}-only-1"
        if stage_only
        else f"aces-1h100-{len(plan['tasks'])}-fit-assess-tasks-concurrency4-1",
    }
    directory = root / "submissions" / wave
    worker = controller.REPO / "scripts/hpc/run_phase_c_baseline.sh"
    with public._lock(directory):
        path = directory / "identity.json"
        if path.exists() and public._read(path) != identity:
            raise ValueError("submission identity differs")
        public._write(path, identity)
        (root / "logs").mkdir(exist_ok=True)
        jobs = {}
        stages = (
            ("propose", "report") if stage_only else ("propose", "fit-assess", "report")
        )
        for stage in stages:
            opts = [
                "--kill-on-invalid-dep=yes",
                "--account=156264627414",
                "--nodes=1",
                "--ntasks=1",
                "--export=ALL",
                "--exclude=ac042",
                f"--job-name=phasec-{label + '-' if stage_only else ''}{stage}",
                f"--output={root}/logs/{wave}-{stage}-%A_%a.out",
                f"--error={root}/logs/{wave}-{stage}-%A_%a.err",
            ]
            if stage == "propose":
                opts += [
                    "--partition=gpu",
                    "--gres=gpu:h100:1",
                    "--cpus-per-task=8",
                    "--mem=64G",
                    "--time=03:30:00" if live_confirmation else "--time=06:30:00",
                    "--signal=B:TERM@300",
                ]
            else:
                predecessor = (
                    "propose"
                    if stage_only
                    else {"fit-assess": "propose", "report": "fit-assess"}[stage]
                )
                opts += [
                    "--partition=cpu",
                    "--cpus-per-task=1",
                    "--mem=16G",
                    "--time=03:00:00",
                    f"--dependency=afterany:{jobs[predecessor]}",
                ]
                if stage != "report":
                    opts += [f"--array=0-{len(plan['tasks']) - 1}%4"]
            # submit_job deliberately refuses unknown outcomes; never reset its intent.
            intent = directory / f"{stage}.intent.json"
            expected = {
                "argv": ["sbatch", "--parsable", *opts, str(worker), stage, "0"]
            }
            if intent.exists():
                job = directory / f"{stage}.id"
                if public._read(intent) != expected or not job.exists():
                    raise ValueError(
                        f"uncertain/different submission; inspect {intent}"
                    )
                jobs[stage] = job.read_text().strip()
                if not jobs[stage].isdecimal():
                    raise ValueError("invalid saved job ID")
            else:
                jobs[stage] = source.submit_job(
                    directory, stage, opts, worker, stage, 0
                )
        value = {
            **identity,
            "jobs": jobs,
            "automatic_followup": False,
            "test_data_opened": False,
        }
        public._write(directory / "manifest.json", value)
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wave", default="construction-1")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--variables-only", action="store_true")
    mode.add_argument("--topology-only", action="store_true")
    mode.add_argument("--construction-comparison", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            submit(
                args.root.resolve(),
                args.wave,
                variables_only=args.variables_only,
                topology_only=args.topology_only,
                compare_construction=args.construction_comparison,
            ),
            indent=2,
        )
    )
