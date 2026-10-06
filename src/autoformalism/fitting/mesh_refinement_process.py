"""Bounded, journaled M11 subprocesses with explicit failed cleanup outcomes."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from time import monotonic

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public


def invoke(mode: str, payload: dict, folder: Path, seconds: float) -> dict:
    """Resume terminal work exactly; never grant a started operation a new budget."""
    if (
        mode
        not in {
            "nodes",
            "point",
            "native",
            "recovery_rollout",
            "recovery_check",
            "profiled_rollout",
        }
        or seconds <= 0
    ):
        raise ValueError("invalid mesh operation")
    identity = {
        "mode": mode,
        "payload_sha256": public.content_sha256(payload),
        "seconds": seconds,
    }
    saved, started = folder / "process.json", folder / "started.json"
    if saved.exists():
        result = read_seal(saved)
        if result["identity"] != identity:
            raise ValueError("mesh operation identity differs")
        for name, digest in result.get("artifacts_sha256", {}).items():
            if public.content_sha256(public._read(folder / name)) != digest:
                raise ValueError("mesh operation output differs")
        return result
    if started.exists():
        if read_seal(started) != identity:
            raise ValueError("mesh operation start differs")
        result = {
            "identity": identity,
            "status": "interrupted",
            "termination_confirmed": False,
            "budget_restarted": False,
        }
        seal(saved, result)
        return result
    folder.mkdir(parents=True, exist_ok=True)
    public._write(folder / "payload.json", payload)
    seal(started, identity)
    begun = monotonic()
    public._write(folder / "launch.json", {"monotonic": begun})
    result = {
        "identity": identity,
        "status": "complete",
        "cleanup_grace_seconds": 10,
        "termination_confirmed": True,
        "budget_restarted": False,
    }
    with (folder / "worker.log").open("w") as log:
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "autoformalism.fitting.mesh_refinement_worker",
                    mode,
                    str(folder),
                ],
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        except OSError as error:
            result.update(
                status="launch_failed",
                error=str(error),
                elapsed_seconds=monotonic() - begun,
            )
            seal(saved, result)
            return result
        try:
            process.wait(timeout=max(0.001, seconds - (monotonic() - begun)))
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            result["status"] = (
                "timeout"
                if isinstance(error, subprocess.TimeoutExpired)
                else "interrupted"
            )
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                result.update(status="cleanup_unconfirmed", termination_confirmed=False)
        result.update(
            returncode=process.returncode, elapsed_seconds=monotonic() - begun
        )
        if result["status"] == "complete" and process.returncode:
            result["status"] = "failed"
    result["artifacts_sha256"] = {
        name: public.content_sha256(public._read(folder / name))
        for name in (
            "result.json",
            "nodes.json",
            "native.json",
            "final_checkpoint_diagnostics.json",
            "final_checkpoint_status.json",
            "checkpoint_rejections.json",
            "checkpoints.json",
            *(
                ("refinement/best.json",)
                if mode in {"recovery_rollout", "profiled_rollout"}
                else ()
            ),
            *(
                (
                    "refinement/structure.json",
                    "refinement/training_history.json",
                    "refinement/accounting.json",
                )
                if mode == "profiled_rollout"
                else ()
            ),
        )
        if result["termination_confirmed"] and (folder / name).exists()
    }
    seal(saved, result)
    return result
