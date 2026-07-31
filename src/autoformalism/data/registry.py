"""Explicit registry for the Phase 1 benchmark subset."""

from __future__ import annotations

from pathlib import Path

from autoformalism.data.exceptions import BenchmarkNotFoundError
from autoformalism.data.models import BenchmarkSpec, TierRoles


def _b1_roles(target: str) -> dict[str, TierRoles]:
    return {
        "easy": TierRoles(
            targets=(target,), auxiliaries=("EGP", "Uii", "E", "Gt")
        ),
        "medium": TierRoles(targets=(target,), auxiliaries=("EGP", "Uii")),
        "hard": TierRoles(targets=(target,)),
    }


def _opaque_b1_roles(
    target: str,
    easy: tuple[str, ...],
    medium: tuple[str, ...],
) -> dict[str, TierRoles]:
    return {
        "easy": TierRoles(targets=(target,), auxiliaries=easy),
        "medium": TierRoles(targets=(target,), auxiliaries=medium),
        "hard": TierRoles(targets=(target,)),
    }


def _standalone_roles(
    target: str,
    easy: tuple[str, ...],
    medium: tuple[str, ...],
) -> dict[str, TierRoles]:
    return _opaque_b1_roles(target, easy, medium)


_SPECS = (
    BenchmarkSpec(
        benchmark_id="original_b1",
        relative_root=Path(
            "benchmark1_original_dalla_man/benchmarks/B1_meal_appearance"
        ),
        manifest_relative_path=Path("benchmark1_original_dalla_man/manifest.json"),
        tier_roles=_b1_roles("Gp"),
        time_column="time",
        external_inputs=("meal_event_g", "meal_schedule"),
        fixed_covariates=("body_weight_kg",),
        input_filename_template="metadata_{split}.csv",
        sampling_interval=1.0,
    ),
    BenchmarkSpec(
        benchmark_id="perturbed_b1",
        relative_root=Path("benchmark2_perturbed_dalla_man/B1_meal_appearance"),
        manifest_relative_path=Path(
            "benchmark2_perturbed_dalla_man/B1_meal_appearance/manifest.json"
        ),
        tier_roles=_b1_roles("Gp"),
        time_column="time",
        external_inputs=("meal_event_g", "meal_schedule"),
        fixed_covariates=("body_weight_kg",),
        input_filename_template="metadata_{split}.csv",
        sampling_interval=1.0,
    ),
    BenchmarkSpec(
        benchmark_id="obfuscated_original_case01",
        relative_root=Path(
            "benchmark3_obfuscated_dalla_man/public/case_01"
        ),
        manifest_relative_path=Path(
            "benchmark3_obfuscated_dalla_man/public/case_01/manifest.json"
        ),
        tier_roles=_opaque_b1_roles(
            "v009", ("v016", "v012", "v004", "v025"), ("v016", "v012")
        ),
        time_column="t",
        external_inputs=("u01", "input_schedule"),
        fixed_covariates=("c01",),
        input_filename_template="input_{split}.csv",
        sampling_interval=1.0,
    ),
    BenchmarkSpec(
        benchmark_id="obfuscated_perturbed_case01",
        relative_root=Path(
            "benchmark4_obfuscated_perturbed_dalla_man/public/case_01"
        ),
        manifest_relative_path=Path(
            "benchmark4_obfuscated_perturbed_dalla_man/public/case_01/manifest.json"
        ),
        tier_roles=_opaque_b1_roles(
            "v015", ("v008", "v014", "v028", "v026"), ("v008", "v014")
        ),
        time_column="t",
        external_inputs=("u01",),
        fixed_covariates=("c01",),
        input_filename_template="input_{split}.csv",
        sampling_interval=1.0,
    ),
    BenchmarkSpec(
        benchmark_id="benchmark5",
        relative_root=Path("benchmark5_anonymous_nonlinear_process/public"),
        manifest_relative_path=Path(
            "benchmark5_anonymous_nonlinear_process/public/manifest.json"
        ),
        tier_roles=_standalone_roles("v02", ("v01", "v03"), ("v01",)),
        time_column="t",
        trajectory_id_column="trajectory_id",
        external_inputs=("u01", "u02", "u03"),
        input_filename_template="input_{split}.csv",
        sampling_interval=0.1,
    ),
    BenchmarkSpec(
        benchmark_id="benchmark6",
        relative_root=Path("benchmark6_alien_device/public"),
        manifest_relative_path=Path(
            "benchmark6_alien_device/public/manifest.json"
        ),
        tier_roles=_standalone_roles("v02", ("v05", "v01"), ("v05",)),
        time_column="t",
        trajectory_id_column="trajectory_id",
        external_inputs=("u01",),
        input_filename_template="input_{split}.csv",
        sampling_interval=0.1,
        clean_observations_available=False,
    ),
)


class BenchmarkRegistry:
    """Read-only registry for explicitly supported public benchmarks."""

    def __init__(self, specs: tuple[BenchmarkSpec, ...] = _SPECS) -> None:
        self._specs = {spec.benchmark_id: spec for spec in specs}

    def identifiers(self) -> tuple[str, ...]:
        """Return supported identifiers in deterministic order."""
        return tuple(sorted(self._specs))

    def get(self, benchmark_id: str) -> BenchmarkSpec:
        """Return a benchmark specification or a clear exception."""
        try:
            return self._specs[benchmark_id]
        except KeyError as exc:
            supported = ", ".join(self.identifiers())
            raise BenchmarkNotFoundError(
                f"unknown benchmark {benchmark_id!r}; supported: {supported}"
            ) from exc

