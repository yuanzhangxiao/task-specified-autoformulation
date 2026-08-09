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
        external_inputs=("meal_event_g",),
        fixed_covariates=("body_weight_kg",),
        input_filename_template="metadata_{split}.csv",
        sampling_interval=1.0,
        one_step_target_history=True,
    ),
    BenchmarkSpec(
        benchmark_id="perturbed_b1",
        relative_root=Path("benchmark2_perturbed_dalla_man/B1_meal_appearance"),
        manifest_relative_path=Path(
            "benchmark2_perturbed_dalla_man/B1_meal_appearance/manifest.json"
        ),
        tier_roles=_b1_roles("Gp"),
        time_column="time",
        external_inputs=("meal_event_g",),
        fixed_covariates=("body_weight_kg",),
        input_filename_template="metadata_{split}.csv",
        sampling_interval=1.0,
        one_step_target_history=True,
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
        one_step_target_history=True,
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
        one_step_target_history=True,
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


def _phase_b_roles(task: str, tier: str, *, named: bool) -> TierRoles:
    """Return public roles without consulting private benchmark material."""

    if named and task.startswith("T"):
        targets = {
            "T1": ("Gp",),
            "T2": ("Gp", "I", "U") if tier == "easy" else ("Gp", "I"),
            "T3": (
                ("Gp", "I", "EGP", "U")
                if tier == "easy"
                else ("Gp", "I", "EGP")
            ),
            "T4": ("Gp", "I"),
        }[task]
        auxiliaries = {
            "T1": ("EGP", "Uii", "E", "Gt") if tier == "easy" else ("Gt",),
            "T2": ("EGP", "Uii", "E", "Gt") if tier == "easy" else ("Uii",),
            "T3": ("Uii", "E", "Gt", "Ipo") if tier == "easy" else (),
            "T4": ("Uii", "E", "Gt", "Ipo") if tier == "easy" else (),
        }[task]
        return TierRoles(targets=targets, auxiliaries=auxiliaries)
    if named and task == "controlled_reactor_mechanism":
        return TierRoles(
            targets=("T",),
            auxiliaries=("C", "Tj") if tier == "easy" else (),
        )
    if task == "T1":
        return TierRoles(
            targets=("v01",),
            auxiliaries=("v02", "v03", "v04", "v05") if tier == "easy" else ("v02",),
        )
    if task == "T2":
        return TierRoles(
            targets=("v01", "v02", "v03") if tier == "easy" else ("v01", "v02"),
            auxiliaries=("v04", "v05", "v06", "v07") if tier == "easy" else ("v03",),
        )
    if task == "T3":
        return TierRoles(
            targets=(
                ("v01", "v02", "v03", "v04")
                if tier == "easy"
                else ("v01", "v02", "v03")
            ),
            auxiliaries=("v05", "v06", "v07", "v08") if tier == "easy" else (),
        )
    if task == "T4":
        return TierRoles(
            targets=("v01", "v02"),
            auxiliaries=("v03", "v04", "v05", "v06") if tier == "easy" else (),
        )
    if task == "controlled_reactor_mechanism":
        return TierRoles(
            targets=("v01",),
            auxiliaries=("v02", "v03") if tier == "easy" else (),
        )
    if task == "unknown_device_mechanism":
        return TierRoles(
            targets=("v01",),
            auxiliaries=("v02", "v03") if tier == "easy" else (),
        )
    raise ValueError(f"unsupported Phase-B task {task!r}")


def _phase_b_specs() -> tuple[BenchmarkSpec, ...]:
    """Build the 40 public Phase-B registry entries deterministically."""

    definitions: list[
        tuple[str, str, str, str, str, tuple[str, ...], float]
    ] = []
    for task in ("T1", "T2", "T3", "T4"):
        input_count = {"T1": 1, "T2": 2, "T3": 3, "T4": 3}[task]
        for dynamics in ("canonical", "perturbed"):
            for variant in ("named", "obfuscated"):
                public_family = (
                    "dalla_man" if variant == "named" else "anonymous_system"
                )
                inputs = (
                    (
                        "meal_event_g",
                        *(('insulin_pmol_per_kg_min',) if input_count >= 2 else ()),
                        *(('glucose_mg_per_kg_min',) if input_count >= 3 else ()),
                    )
                    if variant == "named"
                    else tuple(
                        f"u{index:02d}" for index in range(1, input_count + 1)
                    )
                )
                definitions.append(
                    (
                        public_family,
                        task.lower(),
                        task,
                        dynamics,
                        variant,
                        inputs,
                        1.0,
                    )
                )
    for variant in ("named", "obfuscated"):
        public_family = "cstr" if variant == "named" else "anonymous_system"
        definitions.append(
            (
                public_family,
                "controlled_reactor_mechanism" if variant == "named" else "task",
                "controlled_reactor_mechanism",
                "canonical",
                variant,
                (
                    ("Cf", "Tf", "Tjf")
                    if variant == "named"
                    else ("u01", "u02", "u03")
                ),
                0.1,
            )
        )
    for variant in ("functional", "opaque"):
        public_family = (
            "alien_device" if variant == "functional" else "anonymous_system"
        )
        definitions.append(
            (
                public_family,
                "unknown_device_mechanism" if variant == "functional" else "task",
                "unknown_device_mechanism",
                "canonical",
                variant,
                ("u01",),
                0.1,
            )
        )

    specs: list[BenchmarkSpec] = []
    for (
        public_family,
        public_task,
        private_task,
        dynamics,
        variant,
        inputs,
        interval,
    ) in definitions:
        for tier in ("easy", "hard"):
            identifier = "_".join(
                ("phase_b", public_family, public_task, dynamics, variant, tier)
            )
            relative_root = Path("phase_b_v1") / identifier
            specs.append(
                BenchmarkSpec(
                    benchmark_id=identifier,
                    relative_root=relative_root,
                    manifest_relative_path=relative_root / "manifest.json",
                    tier_roles={
                        tier: _phase_b_roles(
                            private_task,
                            tier,
                            named=variant in {"named", "functional"},
                        )
                    },
                    time_column="t",
                    trajectory_id_column="trajectory_id",
                    external_inputs=inputs,
                    input_filename_template="unused_{split}.csv",
                    tier_directory_template=".",
                    data_layout="tidy_split_file",
                    split_filename_template="{split}.csv",
                    sampling_interval=interval,
                    clean_observations_available=False,
                    one_step_target_history=False,
                )
            )
    return tuple(specs)


_ALL_SPECS = (*_SPECS, *_phase_b_specs())


class BenchmarkRegistry:
    """Read-only registry for explicitly supported public benchmarks."""

    def __init__(self, specs: tuple[BenchmarkSpec, ...] = _ALL_SPECS) -> None:
        self._specs = {spec.benchmark_id: spec for spec in specs}

    def identifiers(self) -> tuple[str, ...]:
        """Return supported identifiers in deterministic order."""
        return tuple(sorted(self._specs))

    def specs(self) -> tuple[BenchmarkSpec, ...]:
        """Return registered specifications in deterministic identifier order."""
        return tuple(self._specs[name] for name in self.identifiers())

    def get(self, benchmark_id: str) -> BenchmarkSpec:
        """Return a benchmark specification or a clear exception."""
        try:
            return self._specs[benchmark_id]
        except KeyError as exc:
            supported = ", ".join(self.identifiers())
            raise BenchmarkNotFoundError(
                f"unknown benchmark {benchmark_id!r}; supported: {supported}"
            ) from exc
