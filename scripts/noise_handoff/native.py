"""Bridges to the project's Phase C baseline and continuous-model readers."""

from pathlib import Path
from time import monotonic

from .exchange import CELLS, FILES, file_hash, read


def baseline_cell(task: dict, directory: Path):
    """Load a real PhaseCBaselineCell from one checked noise acquisition."""
    from autoformalism.rebuttal.phase_c_baselines import (
        PhaseCBaselineCell,
        _DeclaredChannels,
        with_estimated_derivatives,
    )

    from autoformalism.baselines.core import baseline_validation_context
    from autoformalism.benchmarks.phase_c_release import load_public_cell
    from autoformalism.data.models import DevelopmentDataset, TierRoles
    from autoformalism.fitting.public_fitting import unpack_split

    name = task["cell"]
    manifest = directory.parent.parent / "release.json"
    release = read(manifest)
    if name not in CELLS or directory.name != name or directory.parent.name != "public":
        raise ValueError("wrong acquisition path")
    actual = {key: file_hash(directory / key) for key in FILES}
    if (
        actual != task["public_files"]
        or actual != release["cells"][name]["public_files"]
    ):
        raise ValueError("baseline inputs differ from the frozen task")
    spec, train, validation = load_public_cell(directory)
    prompt = (directory / "proposer_prompt.txt").read_text()
    if spec["test_released"] or prompt != spec["public_prompt"].rstrip("\n") + "\n":
        raise ValueError("wrong public prompt or data scope")
    dataset = DevelopmentDataset(
        benchmark_id=name,
        tier="hard",
        roles=TierRoles(
            targets=tuple(spec["targets"]), auxiliaries=tuple(spec["auxiliaries"])
        ),
        train=with_estimated_derivatives(unpack_split(train)),
        validation=with_estimated_derivatives(unpack_split(validation)),
    )
    context = baseline_validation_context(
        dataset,
        _DeclaredChannels(
            external_inputs=tuple(spec["external_inputs"]),
            fixed_covariates=tuple(spec["fixed_covariates"]),
        ),
    )
    identity = {
        "protocol": release["format"],
        "release_summary_sha256": file_hash(manifest),
        "benchmark_id": name,
        "train": dataset.train.fingerprint,
        "validation": dataset.validation.fingerprint,
        "prompt": actual["proposer_prompt.txt"],
    }
    return PhaseCBaselineCell(dataset, context, prompt, identity)


def blind_trajectory(inputs: dict):
    """Create the same predictor-visible Trajectory that Full receives."""
    import numpy as np

    from autoformalism.data import Trajectory

    time = np.asarray(inputs["time"], dtype=float)
    return Trajectory(
        trajectory_id="blind",
        time=time,
        targets={k: np.full_like(time, v) for k, v in inputs["targets0"].items()},
        auxiliaries={k: np.asarray(v) for k, v in inputs["auxiliaries"].items()},
        external_inputs={
            k: np.asarray(v) for k, v in inputs["external_inputs"].items()
        },
        fixed_covariates={},
        derivatives={},
    )


def compiled_forecast(endpoint: dict, inputs: dict, solver: str) -> dict:
    """Run Full's numerical backend for a frozen model in its restricted ODE grammar."""
    from autoformalism.expressions import ValidationContext, compile_candidate
    from autoformalism.fitting import FitConfig, simulate_trajectory
    from autoformalism.schemas import CandidateModel

    if solver not in ("Radau", "BDF"):
        raise ValueError("expected an independent continuous solver")
    compiled = compile_candidate(
        CandidateModel.model_validate(endpoint["candidate"]),
        ValidationContext.model_validate(endpoint["context"]),
    )
    result = simulate_trajectory(
        compiled,
        blind_trajectory(inputs),
        endpoint["parameters"],
        endpoint["initials"],
        FitConfig(
            integration_method=solver, relative_tolerance=1e-8, absolute_tolerance=1e-10
        ),
        deadline=monotonic() + 120,
        reset_observed_states=False,
    )
    if not result.success:
        raise ValueError(result.message)
    return result.predictions
