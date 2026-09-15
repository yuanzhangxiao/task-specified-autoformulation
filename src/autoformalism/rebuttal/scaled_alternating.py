"""Frozen two-arm diagnostic; no changes to the production fitter policy."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.fitting.conditional_scaling import system_for
from autoformalism.fitting.models import FitConfig
from autoformalism.rebuttal.fitter_diagnostic import (
    _write_bytes,
    read_json,
    sha256,
    write_json,
)
from autoformalism.rebuttal.fitter_methods import identity_runtime
from autoformalism.rebuttal.piecewise_campaign import safe_path, unpack_split
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash


class ScaledAlternatingPlan(StrictSchema):
    """One fixed penalty and exactly two arms; no automatic search expansion."""

    protocol: Literal["scaled-alternating-1"] = "scaled-alternating-1"
    initializer_seconds: float = Field(default=1200, gt=0, le=1200)
    screen_seconds: float = Field(default=240, gt=0, le=240)
    refinement_seconds: float = Field(default=3360, gt=0, le=3360)
    replay_seconds: float = Field(default=240, gt=0, le=240)
    screen_point_seconds: float = Field(default=60, gt=0, le=60)
    sensitivity_point_seconds: float = Field(default=180, gt=0, le=180)
    refinement_calls: int = Field(default=118, ge=1, le=118)
    initializer_iterations: int = Field(default=1000, ge=1, le=1000)
    alternating_cycles: int = Field(default=40, ge=1, le=40)
    block_iterations: int = Field(default=25, ge=1, le=25)
    target_variables: int = Field(default=24000, ge=20, le=24000)
    minimum_intervals: int = Field(default=120, ge=2, le=120)
    penalty: float = Field(default=1e4, gt=0, allow_inf_nan=False)
    warmup_seconds: float = Field(default=5, gt=0, le=5)

    @model_validator(mode="after")
    def matched_iterations(self):
        if (
            self.initializer_iterations
            != self.alternating_cycles * self.block_iterations
        ):
            raise ValueError("joint and alternating iteration ceilings must match")
        return self

    def settings(self, *, tight=False, method="Radau"):
        return FitConfig(
            integration_method=method,
            relative_tolerance=1e-9 if tight else 1e-7,
            absolute_tolerance=1e-11 if tight else 1e-9,
            maximum_function_evaluations=self.refinement_calls,
            allow_derivative_regression=False,
        )


def checked(path: Path, identity: str) -> dict:
    value = read_json(path)
    if value.get("identity") != identity:
        raise ValueError("checkpoint identity differs: " + str(path))
    return value


def code_identity() -> str:
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "src/autoformalism").rglob("*.py")) + [
        root / p
        for p in (
            "scripts/run_scaled_alternating.py",
            "scripts/scaled_alternating_report.py",
            "scripts/hpc/scaled_alternating_delta.slurm",
            "scripts/hpc/submit_scaled_alternating_delta.sh",
        )
    ]
    return content_hash({str(p.relative_to(root)): sha256(p) for p in paths})


def prepare(source: Path, output: Path, plan: ScaledAlternatingPlan) -> dict:
    """Copy only the frozen ordinary free-shape problem, never prior estimates."""
    if source.resolve().is_relative_to(
        output.resolve()
    ) or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("source and output must be separate")
    old = read_json(source / "freeze.json")
    if old.get("identity") != content_hash(
        {k: v for k, v in old.items() if k != "identity"}
    ):
        raise ValueError("source identity differs")
    if (
        old["plan"]["protocol"] != "final-fitter-alternatives-1"
        or old.get("test_data_opened") is not False
        or old.get("proposer_access") is not False
    ):
        raise ValueError("requires isolated final-alternatives source")
    if not checked(source / "gate/result.json", old["identity"]).get("pass"):
        raise ValueError("source gate did not pass")
    runtime = identity_runtime()
    if (
        any(runtime[k] != old["runtime"][k] for k in ("packages", "casadi"))
        or runtime["python"].split(".")[:2] != old["runtime"]["python"].split(".")[:2]
    ):
        raise ValueError("source numerical runtime differs")
    tasks = [
        t
        for t in old["tasks"]
        if t["family"] == "free_shapes" and t["strategy"] == "collocation_exact"
    ]
    if len(tasks) != 1:
        raise ValueError("requires exactly one ordinary free-shape source")
    task = tasks[0]
    name = f"problems/{task['index']:03d}.json"
    if sha256(safe_path(source, name)) != old["assets"][name]:
        raise ValueError("source problem changed")
    problem = read_json(source / name)
    if (
        task.get("oracle_weight_start") is not False
        or task.get("oracle_shapes") is not False
        or task.get("oracle_initials") is not True
        or task["starts"][0] != problem["start"]
    ):
        raise ValueError("source ordinary start or boundary contract differs")
    system = system_for(problem)
    if (
        len(system.names) != 48
        or system.state_count != 6
        or set(problem["splits"]) != {"train", "val"}
    ):
        raise ValueError("requires the 48-parameter six-state reference problem")
    for split in ("train", "val"):
        data = unpack_split(problem["splits"][split])
        if data.name.value != split:
            raise ValueError("split labels differ")
    assets = {}
    for origin, destination in (
        (name, "problem.json"),
        ("freeze.json", "provenance/source_freeze.json"),
        ("gate/result.json", "provenance/source_gate.json"),
    ):
        _write_bytes(
            output / destination, (source / origin).read_bytes(), immutable=True
        )
        assets[destination] = sha256(output / destination)
    frozen = {
        "plan": plan.model_dump(mode="json"),
        "tasks": ["joint", "alternating"],
        "assets": assets,
        "code": code_identity(),
        "runtime": runtime,
        "source_identity": old["identity"],
        "test_data_opened": False,
        "proposer_access": False,
        "llm_calls": 0,
        "historical_estimates_used": False,
        "known_hidden_initials_fixed": True,
        "no_automatic_followup": True,
    }
    frozen["identity"] = content_hash(frozen)
    write_json(output / "freeze.json", frozen, immutable=True)
    return verify(output)


def verify(output: Path) -> dict:
    frozen = read_json(output / "freeze.json")
    if frozen["identity"] != content_hash(
        {k: v for k, v in frozen.items() if k != "identity"}
    ):
        raise ValueError("freeze identity differs")
    if frozen["code"] != code_identity() or frozen["runtime"] != identity_runtime():
        raise ValueError("code/runtime differs; use pinned checkout")
    if frozen["tasks"] != ["joint", "alternating"]:
        raise ValueError("paired matrix differs")
    for name, digest in frozen["assets"].items():
        if sha256(safe_path(output, name)) != digest:
            raise ValueError("frozen asset changed: " + name)
    return frozen


def common_data(output: Path, frozen: dict):
    gate = checked(output / "gate/result.json", frozen["identity"])
    if not gate.get("pass"):
        raise ValueError("preflight did not pass")
    common = read_json(output / "common.json")
    if common["identity"] != content_hash(
        {k: v for k, v in common.items() if k != "identity"}
    ):
        raise ValueError("common identity differs")
    if (
        common["identity"] != gate["common_identity"]
        or sha256(output / "nodes.npy") != gate["nodes_sha256"]
    ):
        raise ValueError("frozen common initialization changed")
    nodes = np.load(output / "nodes.npy", allow_pickle=False)
    if content_hash(nodes.tolist()) != common["nodes_identity"]:
        raise ValueError("node identity differs")
    return common, nodes


def inputs(output: Path, frozen: dict):
    problem = read_json(output / "problem.json")
    system = system_for(problem)
    training = unpack_split(problem["splits"]["train"])
    common, nodes = common_data(output, frozen)
    if list(system.names) != common["parameter_names"]:
        raise ValueError("parameter order differs")
    return problem, system, training, common, nodes
