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

    def load_test(self, config: DataConfig) -> DatasetSplit:
        """Load the test split for a previously frozen selection."""
        spec = self._registry.get(config.benchmark_id)
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
            benchmark_masks = masks.get("B1_meal_appearance", masks)
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
