"""New preparation contracts; historical Phase-B protocols remain unchanged."""

from __future__ import annotations

from typing import Literal

from autoformalism.benchmarks.phase_b_generation import (
    PhaseBProtocol,
    phase_b_protocols,
)

PREPARATION = {
    "alien_device": (
        "Before every run, the device is returned to the same reproducible internal "
        "preparation. Unobserved initial quantities are shared across runs; their "
        "values and the internal dimension are not supplied. The measured initial "
        "output may vary and must initialize the output. There is no unrecorded "
        "prehistory or trajectory-specific hidden preparation."
    ),
    "cstr": (
        "Before every run, unobserved initial quantities are reset to the same "
        "reproducible preparation. The initial target reading may vary. Use that "
        "reading, any supplied initial auxiliary readings, and globally shared "
        "initialization parameters; do not fit validation-specific hidden initials."
    ),
}


def phase_c_protocols(
    family: Literal["alien_device", "cstr"],
) -> tuple[PhaseBProtocol, ...]:
    """Keep input designs; vary only observed initial coordinates after reset.

    In alien-device the old private z1/z2 perturbations become observed output
    perturbations. In CSTR only the observed temperature perturbation remains;
    concentration and jacket temperature share the same initial preparation.
    This is a new experiment, never a reinterpretation of historical data.
    """
    if family not in PREPARATION:
        raise ValueError("family has no qualified Phase C preparation contract")
    result = []
    for row in phase_b_protocols(family, input_contract="continuous-rates-1"):
        spec = dict(row.specification)
        identifier = row.protocol_id
        if "initial_shift" in spec:
            if family == "alien_device":
                output = {
                    "train_initial_shift_a": 0.5,
                    "train_initial_shift_b": -0.5,
                    "test_initial_extrapolation": 0.8,
                    "test_combined_shift": -0.8,
                }[identifier]
                # The simulator's existing output coordinate is last. Do not
                # change latent preparation or expose it as an input.
                spec["initial_shift"] = [0.0] * (len(spec["initial_shift"]) - 1) + [
                    output
                ]
                identifier = identifier.replace("initial_shift", "initial_output")
            else:
                spec["initial_shift"] = [0.0, spec["initial_shift"][1], 0.0]
                identifier = identifier.replace(
                    "initial_dominant", "initial_temperature"
                )
        result.append(
            row.model_copy(update={"specification": spec, "protocol_id": identifier})
        )
    return tuple(result)
