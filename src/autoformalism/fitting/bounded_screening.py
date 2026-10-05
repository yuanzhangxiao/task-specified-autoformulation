"""Training-only, killable point evaluations with an append-only attempt journal."""

from __future__ import annotations

import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.identifiable_campaign import SETTINGS, scale_for
from autoformalism.fitting.sensitivity_probe import SymbolicODE, SymbolicOracle
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit


class TrainingOnlySplit(PublicSplit):
    """Public observations with validation explicitly excluded."""

    name: Literal["train"]


class ScreeningPolicy(StrictSchema):
    """A point ceiling includes Python startup, integration and output writes."""

    method: Literal["RK45", "Radau"] = "RK45"
    point_seconds: float = Field(default=20, ge=1, le=120)


def evaluate(payload: dict, directory: Path) -> dict:
    """One entire training rollout; a failure penalty never becomes a score."""
    policy = ScreeningPolicy.model_validate(payload["screening"])
    data = TrainingOnlySplit.model_validate(payload["training"])
    request = PublicFitRequest.model_validate(payload["request"])
    train = public.unpack_split(data)
    system = SymbolicODE(public._lower(request)[0], allow_piecewise=True)
    settings = SETTINGS.model_copy(update={"integration_method": policy.method})
    oracle = SymbolicOracle(
        system,
        train,
        scale_for(train),
        settings,
        directory / "oracle",
        monotonic() + payload["seconds"],
        sensitivities=False,
    )
    try:
        values = oracle(oracle.vector(payload["parameters"]))
        if not oracle.valid_calls:
            raise ValueError("no complete unclipped training rollout")
        sizes = [len(r.time) * len(system.channels) for r in train.trajectories]
        result = {
            "status": "complete",
            "parameters": payload["parameters"],
            "training_nmse": float(np.mean(values**2)),
            "maximum_trajectory_nmse": max(
                float(np.mean(v**2)) for v in np.split(values, np.cumsum(sizes)[:-1])
            ),
        }
    except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
        result = {"status": "unavailable", "error": str(error)[-1000:]}
    result["payload_sha256"] = public.content_sha256(payload)
    public._write(directory / "result.json", result)
    return result


def invoke_point(payload: dict, folder: Path, seconds: float) -> dict:
    """Child stays in the enclosing fit's process group; its own deadline is hard."""
    begun = monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    public._write(folder / "payload.json", payload)
    with (folder / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", __name__, str(folder)], stdout=log, stderr=log
        )
        timeout = False
        try:
            process.wait(timeout=max(0.001, seconds - (monotonic() - begun)))
        except subprocess.TimeoutExpired:
            timeout = True
            process.kill()
            process.wait(timeout=2)
        except BaseException:
            with suppress(ProcessLookupError):
                process.kill()
            process.wait(timeout=2)
            raise
    return {
        "wall_timeout": timeout,
        "returncode": process.returncode,
        "elapsed_seconds": monotonic() - begun,
        "cleanup_grace_seconds": 2,
    }


def screen(
    payload: dict,
    folder: Path,
    phase: str,
    points: list[dict],
    *,
    deadline: float,
    maximum: int,
    best: dict | None,
) -> tuple[dict | None, int]:
    """Record starts before work; never retry an interrupted phase or reset its budget.

    The caller serializes access in the campaign worker lock. Each phase owns unique
    attempt directories. The shared journal keeps even killed/in-flight attempts.
    """
    policy = ScreeningPolicy.model_validate(payload["screening"])
    if not phase or not all(c.isalnum() or c in "-_" for c in phase):
        raise ValueError("invalid screening phase")
    identity = public.content_sha256(
        {k: payload[k] for k in ("request", "training", "screening")}
    )
    path = folder / "screening.json"
    journal = (
        public._read(path)
        if path.exists()
        else {
            "identity": identity,
            "phases": {},
            "attempts": [],
            "best": None,
        }
    )
    if journal["identity"] != identity:
        raise ValueError("screening identity differs")
    digest = public.content_sha256(points)
    if phase in journal["phases"]:
        if journal["phases"][phase]["points_sha256"] != digest:
            raise ValueError("screening phase points differ")
        for attempt in journal["attempts"]:
            if attempt["phase"] == phase and attempt["status"] == "started":
                attempt["status"] = "interrupted"
        journal["phases"][phase]["status"] = "closed_without_retry"
        public._write(path, journal)
        saved = journal.get("best")
        return (
            saved
            if saved and (not best or saved["training_nmse"] < best["training_nmse"])
            else best
        ), len(journal["attempts"])
    journal["phases"][phase] = {"points_sha256": digest, "status": "started"}
    public._write(path, journal)
    for index, point in enumerate(points):
        seconds = min(policy.point_seconds, deadline - monotonic())
        if seconds <= 0.05 or len(journal["attempts"]) >= maximum:
            break
        directory = folder / "screens" / phase / f"{index:03d}"
        attempt = {
            "phase": phase,
            "index": index,
            "status": "started",
            "parameters_sha256": public.content_sha256(point["parameters"]),
            "allowance_seconds": seconds,
            "method": policy.method,
            "source": point.get("source"),
        }
        journal["attempts"].append(attempt)
        public._write(path, journal)
        request = {k: payload[k] for k in ("request", "training", "screening")} | {
            "parameters": point["parameters"],
            "seconds": seconds,
        }
        process = invoke_point(request, directory, seconds)
        attempt["process"] = process
        record = (
            public._read(directory / "result.json")
            if (directory / "result.json").exists()
            else {}
        )
        if record and record.get("payload_sha256") != public.content_sha256(request):
            raise ValueError("screen result identity differs")
        if record.get("status") == "complete":
            attempt.update(status="complete", training_nmse=record["training_nmse"])
            if best is None or record["training_nmse"] < best["training_nmse"]:
                best = {
                    k: record[k]
                    for k in ("parameters", "training_nmse", "maximum_trajectory_nmse")
                } | {
                    "source": point.get("source", f"{phase}:{index}"),
                }
                journal["best"] = best
                public._write(folder / "best.json", best)
        else:
            attempt.update(
                status="timeout" if process["wall_timeout"] else "unavailable",
                error=record.get("error"),
            )
        public._write(path, journal)
    journal["phases"][phase]["status"] = "complete"
    public._write(path, journal)
    return best, len(journal["attempts"])


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    evaluate(public._read(directory / "payload.json"), directory)
