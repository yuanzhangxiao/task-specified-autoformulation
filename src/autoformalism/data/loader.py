"""Strict public benchmark loader with trajectory grouping."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autoformalism.config import DataConfig
from autoformalism.data.exceptions import (
    ChannelRoleError,
    DataAlignmentError,
    DataFileNotFoundError,
)
from autoformalism.data.io import load_csv, load_json, require_file
from autoformalism.data.models import (
    BenchmarkDataset,
    BenchmarkSpec,
    DatasetSplit,
    DevelopmentDataset,
    FrozenTestAccess,
    SplitName,
    SplitPaths,
    TierRoles,
    Trajectory,
)
from autoformalism.data.registry import BenchmarkRegistry


class BenchmarkLoader:
    """Load a registered benchmark without inferring channel roles."""

    def __init__(self, registry: BenchmarkRegistry | None = None) -> None:
        self._registry = registry or BenchmarkRegistry()

    def load(self, config: DataConfig) -> BenchmarkDataset:
        """Load and validate all three splits for a configured tier."""
        spec = self._registry.get(config.benchmark_id)
        if spec.data_layout == "tidy_split_file":
            raise ChannelRoleError(
                "Phase-B test data cannot be opened by load(); use "
                "load_development() and FrozenTestAccess"
            )
        roles = self._roles(spec, config.tier)
        self._validate_root_paths(config.root, spec)
        self._validate_manifest(config.root, spec, config.tier, roles)

        split_paths = {
            split: self._resolve_split_paths(
                config.root,
                spec,
                config.tier,
                split,
                config.use_clean_observations,
            )
            for split in SplitName
        }
        self._validate_split_paths_are_distinct(split_paths)
        loaded = {
            split: self._load_split(spec, roles, split, paths)
            for split, paths in split_paths.items()
        }
        return BenchmarkDataset(
            benchmark_id=spec.benchmark_id,
            tier=config.tier,
            roles=roles,
            train=loaded[SplitName.TRAIN],
            validation=loaded[SplitName.VALIDATION],
            test=loaded[SplitName.TEST],
        )

    def load_development(self, config: DataConfig) -> DevelopmentDataset:
        """Load only train and validation, leaving test unopened."""
        spec = self._registry.get(config.benchmark_id)
        roles = self._roles(spec, config.tier)
        self._validate_root_paths(config.root, spec)
        self._validate_manifest(config.root, spec, config.tier, roles)
        paths = {
            split: self._resolve_split_paths(
                config.root,
                spec,
                config.tier,
                split,
                config.use_clean_observations,
            )
            for split in (SplitName.TRAIN, SplitName.VALIDATION)
        }
        self._validate_selected_paths_are_distinct(paths)
        return DevelopmentDataset(
            benchmark_id=spec.benchmark_id,
            tier=config.tier,
            roles=roles,
            train=self._load_split(
                spec, roles, SplitName.TRAIN, paths[SplitName.TRAIN]
            ),
            validation=self._load_split(
                spec,
                roles,
                SplitName.VALIDATION,
                paths[SplitName.VALIDATION],
            ),
        )

    def load_test(
        self,
        config: DataConfig,
        *,
        access: FrozenTestAccess | None = None,
    ) -> DatasetSplit:
        """Load the test split for a previously frozen selection."""
        spec = self._registry.get(config.benchmark_id)
        if spec.data_layout == "tidy_split_file":
            if access is None:
                raise ChannelRoleError(
                    "Phase-B test loading requires FrozenTestAccess"
                )
            if (access.benchmark_id, access.tier) != (
                config.benchmark_id,
                config.tier,
            ):
                raise ChannelRoleError(
                    "test access grant does not match the configured benchmark"
                )
        roles = self._roles(spec, config.tier)
        self._validate_root_paths(config.root, spec)
        self._validate_manifest(config.root, spec, config.tier, roles)
        paths = self._resolve_split_paths(
            config.root,
            spec,
            config.tier,
            SplitName.TEST,
            config.use_clean_observations,
        )
        return self._load_split(spec, roles, SplitName.TEST, paths)

    def validate_test_paths(self, config: DataConfig) -> None:
        """Validate test file locations without opening their contents."""
        spec = self._registry.get(config.benchmark_id)
        self._validate_root_paths(config.root, spec)
        self._resolve_split_paths(
            config.root,
            spec,
            config.tier,
            SplitName.TEST,
            config.use_clean_observations,
        )

    @staticmethod
    def _roles(spec: BenchmarkSpec, tier: str) -> TierRoles:
        try:
            return spec.tier_roles[tier]
        except KeyError as exc:
            raise ChannelRoleError(
                f"{spec.benchmark_id} does not define tier {tier!r}"
            ) from exc

    @staticmethod
    def _resolve_under(root: Path, relative: Path) -> Path:
        if any(
            part.lower() == "private" or part.lower().startswith("hidden")
            for part in relative.parts
        ):
            raise DataFileNotFoundError(f"non-public path is forbidden: {relative}")
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise DataFileNotFoundError(
                f"resolved path escapes configured data root: {resolved}"
            )
        return resolved

    def _validate_root_paths(self, root: Path, spec: BenchmarkSpec) -> None:
        benchmark_root = self._resolve_under(root, spec.relative_root)
        if not benchmark_root.is_dir():
            raise DataFileNotFoundError(
                f"benchmark directory does not exist: {benchmark_root}"
            )
        require_file(self._resolve_under(root, spec.manifest_relative_path))

    def _resolve_split_paths(
        self,
        root: Path,
        spec: BenchmarkSpec,
        tier: str,
        split: SplitName,
        use_clean: bool,
    ) -> SplitPaths:
        if use_clean and not spec.clean_observations_available:
            raise DataFileNotFoundError(
                f"{spec.benchmark_id} has no clean observation files"
            )
        benchmark_root = self._resolve_under(root, spec.relative_root)
        if spec.data_layout == "tidy_split_file":
            split_name = "validation" if split is SplitName.VALIDATION else split.value
            assert spec.split_filename_template is not None
            path = benchmark_root / spec.split_filename_template.format(
                split=split_name
            )
            require_file(path)
            return SplitPaths(path, path, path)
        tier_directory = benchmark_root / spec.tier_directory_template.format(tier=tier)
        clean_suffix = "_clean" if use_clean else ""
        observations = tier_directory / f"X_{split.value}{clean_suffix}.csv"
        derivatives = tier_directory / f"Y_{split.value}.csv"
        inputs = benchmark_root / spec.input_filename_template.format(
            split=split.value
        )
        for path in (observations, derivatives, inputs):
            require_file(path)
        return SplitPaths(observations, derivatives, inputs)

    @staticmethod
    def _validate_split_paths_are_distinct(
        paths: dict[SplitName, SplitPaths],
    ) -> None:
        for field in ("observations", "derivatives", "inputs"):
            values = [getattr(paths[split], field).resolve() for split in SplitName]
            if len(set(values)) != len(values):
                raise DataAlignmentError(f"{field} files overlap across splits")

    @staticmethod
    def _validate_selected_paths_are_distinct(
        paths: dict[SplitName, SplitPaths],
    ) -> None:
        for field in ("observations", "derivatives", "inputs"):
            values = [getattr(item, field).resolve() for item in paths.values()]
            if len(set(values)) != len(values):
                raise DataAlignmentError(f"{field} files overlap across splits")

    def _validate_manifest(
        self,
        root: Path,
        spec: BenchmarkSpec,
        tier: str,
        roles: TierRoles,
    ) -> None:
        manifest = load_json(self._resolve_under(root, spec.manifest_relative_path))
        if spec.data_layout == "tidy_split_file":
            if manifest.get("schema_version") != "phase_b_public_release_v1":
                raise ChannelRoleError("Phase-B manifest is not a frozen release")
            if manifest.get("status") != "production_registered":
                raise ChannelRoleError("Phase-B package is not production registered")
            if manifest.get("test_sealed") is not True:
                raise ChannelRoleError("Phase-B production test split is not sealed")
            if manifest.get("benchmark_id") != spec.benchmark_id:
                raise ChannelRoleError(
                    f"manifest benchmark ID does not match {spec.benchmark_id}"
                )
            if manifest.get("tier") != tier:
                raise ChannelRoleError(
                    f"manifest tier does not match {spec.benchmark_id}/{tier}"
                )
            channels = manifest.get("channels")
            if not isinstance(channels, list):
                raise ChannelRoleError("tidy manifest has no channel declarations")
            declared_roles = {
                str(item.get("public_name")): str(item.get("role"))
                for item in channels
                if isinstance(item, dict)
            }
            expected_roles = {
                **dict.fromkeys(roles.targets, "target"),
                **dict.fromkeys(roles.auxiliaries, "auxiliary"),
                **dict.fromkeys(spec.external_inputs, "external_input"),
            }
            if declared_roles != expected_roles:
                raise ChannelRoleError(
                    f"manifest roles for {spec.benchmark_id}/{tier} do not match "
                    "the registry"
                )
            split_hashes = manifest.get("splits")
            if not isinstance(split_hashes, dict) or set(split_hashes) != {
                "train",
                "validation",
                "test",
            }:
                raise ChannelRoleError(
                    "Phase-B release must declare all three split fingerprints"
                )
            benchmark_root = self._resolve_under(root, spec.relative_root)
            for split_name, expected_hash in split_hashes.items():
                split_path = benchmark_root / f"{split_name}.csv"
                require_file(split_path)
                actual_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
                if actual_hash != expected_hash:
                    raise DataAlignmentError(
                        f"Phase-B split fingerprint mismatch: {split_path}"
                    )
            return
        declared: list[str] | None = None
        tiers = manifest.get("tiers")
        if isinstance(tiers, dict) and tier in tiers:
            tier_entry = tiers[tier]
            if isinstance(tier_entry, list):
                declared = tier_entry
            elif isinstance(tier_entry, dict):
                declared = tier_entry.get("observed_columns")
        masks = manifest.get("benchmark_masks")
        if declared is None and isinstance(masks, dict):
            matching_keys = [
                key
                for key in masks
                if isinstance(key, str) and key in spec.relative_root.parts
            ]
            benchmark_masks = (
                masks[matching_keys[0]]
                if len(matching_keys) == 1
                else masks.get("B1_meal_appearance", masks)
            )
            if isinstance(benchmark_masks, dict):
                declared = benchmark_masks.get(tier)
        expected = list(roles.targets + roles.auxiliaries)
        if declared is not None and set(declared) != set(expected):
            raise ChannelRoleError(
                f"manifest channels for {spec.benchmark_id}/{tier} "
                f"are {declared}, registry roles require {expected}"
            )

    def _load_split(
        self,
        spec: BenchmarkSpec,
        roles: TierRoles,
        split: SplitName,
        paths: SplitPaths,
    ) -> DatasetSplit:
        if spec.data_layout == "tidy_split_file":
            return self._load_tidy_split(spec, roles, split, paths.observations)
        observed_columns = roles.targets + roles.auxiliaries
        observations = load_csv(paths.observations, observed_columns)
        derivative_columns = tuple(f"d{name}_dt" for name in observed_columns)
        derivatives = load_csv(paths.derivatives, derivative_columns)
        input_columns = (
            (spec.trajectory_id_column,) if spec.trajectory_id_column else ()
        ) + (
            spec.time_column,
            *spec.external_inputs,
            *spec.fixed_covariates,
        )
        inputs = load_csv(paths.inputs, input_columns)

        if not (
            len(observations) == len(derivatives) == len(inputs)
            and len(inputs) > 0
        ):
            raise DataAlignmentError(
                f"row counts do not align for {spec.benchmark_id}/{split.value}: "
                f"X={len(observations)}, Y={len(derivatives)}, input={len(inputs)}"
            )
        self._validate_exact_observation_columns(
            paths.observations, observations, observed_columns
        )
        self._validate_numeric(observations, paths.observations)
        self._validate_numeric(derivatives, paths.derivatives)

        groups = self._group_indices(inputs, spec.trajectory_id_column, split)
        trajectories = tuple(
            self._make_trajectory(
                spec,
                roles,
                trajectory_id,
                index,
                observations,
                derivatives,
                inputs,
                paths,
            )
            for trajectory_id, index in groups
        )
        fingerprint = self._fingerprint(paths)
        return DatasetSplit(split, trajectories, fingerprint)

    def _load_tidy_split(
        self,
        spec: BenchmarkSpec,
        roles: TierRoles,
        split: SplitName,
        path: Path,
    ) -> DatasetSplit:
        """Load one canonical Phase-B table and derive numerical derivatives."""

        expected = (
            spec.trajectory_id_column,
            spec.time_column,
            *roles.targets,
            *roles.auxiliaries,
            *spec.external_inputs,
        )
        required = tuple(name for name in expected if name is not None)
        frame = load_csv(path, required)
        extras = sorted(set(frame.columns) - set(required))
        if extras:
            raise ChannelRoleError(
                f"{path} contains columns without declared public roles: {extras}"
            )
        numeric_columns = (
            spec.time_column,
            *roles.targets,
            *roles.auxiliaries,
            *spec.external_inputs,
        )
        self._validate_numeric(frame[list(numeric_columns)], path)
        groups = self._group_indices(frame, spec.trajectory_id_column, split)
        trajectories = tuple(
            self._make_tidy_trajectory(
                spec, roles, trajectory_id, index, frame, path
            )
            for trajectory_id, index in groups
        )
        return DatasetSplit(split, trajectories, self._fingerprint_file(path))

    def _make_tidy_trajectory(
        self,
        spec: BenchmarkSpec,
        roles: TierRoles,
        trajectory_id: str,
        index: np.ndarray,
        frame: pd.DataFrame,
        path: Path,
    ) -> Trajectory:
        selected = frame.iloc[index]
        time = self._numeric_array(selected[spec.time_column], path, spec.time_column)
        if len(time) < 2:
            raise DataAlignmentError(
                f"trajectory {trajectory_id} requires at least two samples"
            )
        differences = np.diff(time)
        if np.any(differences <= 0):
            raise DataAlignmentError(
                f"time is not strictly increasing for {trajectory_id}"
            )
        if not np.allclose(
            differences,
            spec.sampling_interval,
            rtol=1e-7,
            atol=max(1e-10, spec.sampling_interval * 1e-9),
        ):
            raise DataAlignmentError(
                f"sampling interval differs from {spec.sampling_interval} "
                f"for {trajectory_id}"
            )
        observed = roles.targets + roles.auxiliaries
        observed_values = {
            name: selected[name].to_numpy(dtype=float, copy=True)
            for name in observed
        }
        derivatives = {
            name: np.gradient(values, time, edge_order=2 if len(time) >= 3 else 1)
            for name, values in observed_values.items()
        }
        return Trajectory(
            trajectory_id=trajectory_id,
            time=time,
            targets={name: observed_values[name] for name in roles.targets},
            auxiliaries={name: observed_values[name] for name in roles.auxiliaries},
            external_inputs={
                name: selected[name].to_numpy(dtype=float, copy=True)
                for name in spec.external_inputs
            },
            fixed_covariates={},
            derivatives=derivatives,
        )

    @staticmethod
    def _validate_exact_observation_columns(
        path: Path,
        frame: pd.DataFrame,
        expected: tuple[str, ...],
    ) -> None:
        extras = sorted(set(frame.columns) - set(expected))
        if extras:
            raise ChannelRoleError(
                f"{path} contains columns without target/auxiliary roles: {extras}"
            )

    @staticmethod
    def _validate_numeric(frame: pd.DataFrame, path: Path) -> None:
        nonnumeric = [
            name
            for name in frame
            if not pd.api.types.is_numeric_dtype(frame[name])
        ]
        if nonnumeric:
            raise ChannelRoleError(f"{path} has nonnumeric channels: {nonnumeric}")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise DataAlignmentError(f"{path} contains NaN or infinite values")

    @staticmethod
    def _group_indices(
        inputs: pd.DataFrame,
        trajectory_id_column: str | None,
        split: SplitName,
    ) -> list[tuple[str, np.ndarray]]:
        if trajectory_id_column is None:
            return [(split.value, np.arange(len(inputs)))]
        identifiers = inputs[trajectory_id_column]
        if identifiers.isna().any():
            raise DataAlignmentError("trajectory identifier contains missing values")
        groups: list[tuple[str, np.ndarray]] = []
        seen: set[str] = set()
        for raw_identifier in identifiers.drop_duplicates():
            trajectory_id = str(raw_identifier)
            index = np.flatnonzero(identifiers.to_numpy() == raw_identifier)
            if trajectory_id in seen:
                raise DataAlignmentError(f"duplicate trajectory ID: {trajectory_id}")
            if len(index) > 1 and not np.all(np.diff(index) == 1):
                raise DataAlignmentError(
                    f"trajectory rows are not contiguous: {trajectory_id}"
                )
            seen.add(trajectory_id)
            groups.append((trajectory_id, index))
        return groups

    def _make_trajectory(
        self,
        spec: BenchmarkSpec,
        roles: TierRoles,
        trajectory_id: str,
        index: np.ndarray,
        observations: pd.DataFrame,
        derivatives: pd.DataFrame,
        inputs: pd.DataFrame,
        paths: SplitPaths,
    ) -> Trajectory:
        selected_inputs = inputs.iloc[index]
        time = self._numeric_array(
            selected_inputs[spec.time_column], paths.inputs, spec.time_column
        )
        if len(time) > 1:
            differences = np.diff(time)
            if np.any(differences <= 0):
                raise DataAlignmentError(
                    f"time is not strictly increasing for {trajectory_id}"
                )
            if not np.allclose(
                differences,
                spec.sampling_interval,
                rtol=1e-7,
                atol=max(1e-10, spec.sampling_interval * 1e-9),
            ):
                raise DataAlignmentError(
                    f"sampling interval differs from {spec.sampling_interval} "
                    f"for {trajectory_id}"
                )

        target_values = {
            name: observations.iloc[index][name].to_numpy(dtype=float, copy=True)
            for name in roles.targets
        }
        auxiliary_values = {
            name: observations.iloc[index][name].to_numpy(dtype=float, copy=True)
            for name in roles.auxiliaries
        }
        external_values = {
            name: self._input_array(selected_inputs[name], paths.inputs, name)
            for name in spec.external_inputs
        }
        fixed_values = {
            name: self._constant_value(selected_inputs[name], paths.inputs, name)
            for name in spec.fixed_covariates
        }
        derivative_values = {
            name: derivatives.iloc[index][f"d{name}_dt"].to_numpy(
                dtype=float, copy=True
            )
            for name in roles.targets + roles.auxiliaries
        }
        return Trajectory(
            trajectory_id=trajectory_id,
            time=time,
            targets=target_values,
            auxiliaries=auxiliary_values,
            external_inputs=external_values,
            fixed_covariates=fixed_values,
            derivatives=derivative_values,
        )

    @staticmethod
    def _numeric_array(series: pd.Series, path: Path, name: str) -> np.ndarray:
        try:
            values = pd.to_numeric(series, errors="raise").to_numpy(
                dtype=float, copy=True
            )
        except (TypeError, ValueError) as exc:
            raise ChannelRoleError(f"{path}:{name} must be numeric") from exc
        if not np.isfinite(values).all():
            raise DataAlignmentError(f"{path}:{name} contains NaN or infinite values")
        return values

    def _input_array(self, series: pd.Series, path: Path, name: str) -> np.ndarray:
        if pd.api.types.is_numeric_dtype(series):
            return self._numeric_array(series, path, name)
        values = series.astype(str).to_numpy(copy=True)
        values.setflags(write=False)
        return values

    @staticmethod
    def _constant_value(series: pd.Series, path: Path, name: str) -> Any:
        unique = series.drop_duplicates()
        if len(unique) != 1:
            raise DataAlignmentError(
                f"{path}:{name} must be constant within each trajectory"
            )
        return unique.iloc[0]

    @staticmethod
    def _fingerprint(paths: SplitPaths) -> str:
        digest = hashlib.sha256()
        for path in (paths.observations, paths.derivatives, paths.inputs):
            digest.update(path.name.encode())
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _fingerprint_file(path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
