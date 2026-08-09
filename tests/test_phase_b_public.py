"""Tests for Phase-B public projections, prompts, and leakage controls."""

import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest

from autoformalism.benchmarks import (
    audit_public_bundle,
    phase_b_protocols,
    phase_b_public_spec,
    render_phase_b_prompts,
    simulate_phase_b,
    write_public_production_bundle,
    write_public_staging_bundle,
)
from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, FrozenTestAccess, SplitName
from autoformalism.data.exceptions import ChannelRoleError, DataAlignmentError
from autoformalism.schemas import JudgeResult

DATA_ROOT = Path("data_raw")


@pytest.mark.parametrize(
    ("family", "variant", "task"),
    [
        ("dalla_man", "named", "T1"),
        ("dalla_man", "obfuscated", "T4"),
        ("cstr", "named", None),
        ("cstr", "obfuscated", None),
        ("alien_device", "functional", None),
        ("alien_device", "opaque", None),
    ],
)
def test_public_specs_are_bijective_and_prompt_symbols_are_public(
    family: str, variant: str, task: str | None
) -> None:
    spec = phase_b_public_spec(family, "easy", variant, task=task, data_root=DATA_ROOT)
    proposer, judge = render_phase_b_prompts(spec)

    assert len({item.public_name for item in spec.channels}) == len(spec.channels)
    assert len({item.private_source for item in spec.channels}) == len(spec.channels)
    assert all(item.public_name in proposer for item in spec.channels)
    assert "private_source" not in proposer + judge
    assert all(f"{section}." in proposer for section in ("A", "B", "C", "D", "E", "F"))


def test_all_cells_share_one_judge_prompt() -> None:
    prompts = set()
    for family, variants, task in (
        ("dalla_man", ("named", "obfuscated"), "T2"),
        ("cstr", ("named", "obfuscated"), None),
        ("alien_device", ("functional", "opaque"), None),
    ):
        for variant in variants:
            spec = phase_b_public_spec(
                family, "hard", variant, task=task, data_root=DATA_ROOT
            )
            prompts.add(render_phase_b_prompts(spec)[1])
    assert len(prompts) == 1
    prompt = prompts.pop()
    match = re.search(
        r"Return strict JSON with exactly this shape:\n(\{.*?\n\})", prompt, re.S
    )
    assert match is not None
    JudgeResult.model_validate_json(match.group(1))


def test_named_t1_uses_natural_task_language_and_units() -> None:
    named = phase_b_public_spec(
        "dalla_man", "easy", "named", task="T1", data_root=DATA_ROOT
    )
    obfuscated = phase_b_public_spec(
        "dalla_man", "easy", "obfuscated", task="T1", data_root=DATA_ROOT
    )
    named_prompt = render_phase_b_prompts(named)[0]
    obfuscated_prompt = render_phase_b_prompts(obfuscated)[0]

    assert "model of the post-meal glucose response" in named_prompt
    assert "meal timing and meal amount contribute" in named_prompt
    assert "held-out meal timings and amounts" not in named_prompt
    assert "[unit: mg kg^-1]" in named_prompt
    assert "[unit:" not in obfuscated_prompt
    assert "u01(t)" in obfuscated_prompt
    assert "u06" not in obfuscated_prompt
    assert "schedule object" not in named_prompt
    assert "one-step-ahead" not in named_prompt
    assert "free-rollout" not in named_prompt


@pytest.mark.parametrize("task", ["T2", "T3", "T4"])
def test_multi_channel_obfuscated_dalla_preserves_role_level_obligations(
    task: str,
) -> None:
    spec = phase_b_public_spec(
        "dalla_man", "easy", "obfuscated", task=task, data_root=DATA_ROOT
    )
    prompt = render_phase_b_prompts(spec)[0]

    assert "u01(t)" in prompt
    assert "primary target v01(t)" in prompt
    assert "v02(t)" in prompt
    assert "supplied public channel" not in prompt
    assert "enters the v01(t) balance" not in prompt
    assert "contributes directly to v01(t)" not in prompt
    assert "u02(t) contributes to v02(t)" not in prompt
    assert "u03(t) contributes" not in prompt


@pytest.mark.parametrize("tier", ["easy", "hard"])
def test_obfuscated_cstr_avoids_endpoint_level_input_mappings(tier: str) -> None:
    spec = phase_b_public_spec("cstr", tier, "obfuscated", data_root=DATA_ROOT)
    prompt = render_phase_b_prompts(spec)[0]

    assert all(f"u0{index}(t)" in prompt for index in range(1, 4))
    assert "external transport" in prompt
    assert "state-dependent source" in prompt
    assert "exchange with a coupled quantity" in prompt
    assert "contributes directly to v01(t)" not in prompt
    assert "u01(t) influences" not in prompt
    assert "u03(t) influences" not in prompt


@pytest.mark.parametrize("task", ["T1", "T2", "T3", "T4"])
@pytest.mark.parametrize("tier", ["easy", "hard"])
@pytest.mark.parametrize("dynamics", ["canonical", "perturbed"])
def test_dalla_semantic_pairs_have_matching_mechanism_counts(
    task: str, tier: str, dynamics: str
) -> None:
    prompts = []
    for variant in ("named", "obfuscated"):
        spec = phase_b_public_spec(
            "dalla_man",
            tier,
            variant,
            task=task,
            dynamics=dynamics,
            data_root=DATA_ROOT,
        )
        prompts.append(render_phase_b_prompts(spec)[0])

    assert _mechanism_bullets(prompts[0]) == _mechanism_bullets(prompts[1])


@pytest.mark.parametrize("tier", ["easy", "hard"])
def test_cstr_semantic_pairs_have_matching_mechanism_counts(tier: str) -> None:
    prompts = [
        render_phase_b_prompts(
            phase_b_public_spec("cstr", tier, variant, data_root=DATA_ROOT)
        )[0]
        for variant in ("named", "obfuscated")
    ]

    assert _mechanism_bullets(prompts[0]) == _mechanism_bullets(prompts[1]) == 1


def _mechanism_bullets(prompt: str) -> int:
    section = prompt.split("The dimensionality", maxsplit=1)[0]
    return sum(line.startswith("- ") for line in section.splitlines())


def test_semantic_control_pairs_have_identical_numeric_rows(tmp_path: Path) -> None:
    protocol = phase_b_protocols("cstr")[0]
    trajectory = simulate_phase_b(protocol, data_root=DATA_ROOT)
    named = phase_b_public_spec("cstr", "easy", "named")
    obfuscated = phase_b_public_spec("cstr", "easy", "obfuscated")
    named_root = tmp_path / "named"
    opaque_root = tmp_path / "obfuscated"

    write_public_staging_bundle(named_root, named, (trajectory,))
    write_public_staging_bundle(opaque_root, obfuscated, (trajectory,))

    with (named_root / "train.csv").open(encoding="utf-8") as handle:
        named_table = list(csv.reader(handle))
    with (opaque_root / "train.csv").open(encoding="utf-8") as handle:
        opaque_table = list(csv.reader(handle))
    named_rows = named_table[1:]
    opaque_rows = opaque_table[1:]
    named_manifest = json.loads((named_root / "manifest.json").read_text())
    opaque_manifest = json.loads((opaque_root / "manifest.json").read_text())
    assert named_rows == opaque_rows
    assert named_rows[0][0] == "train_000"
    assert not any("schedule" in item.lower() for item in named_table[0])
    assert (
        named_manifest["numeric_payload_sha256"]
        == opaque_manifest["numeric_payload_sha256"]
    )
    assert (named_root / "proposer_prompt.txt").read_text() != (
        opaque_root / "proposer_prompt.txt"
    ).read_text()


def test_public_bundle_omits_test_by_default_and_hides_mapping(tmp_path: Path) -> None:
    protocols = phase_b_protocols("alien_device")
    selected = tuple(
        simulate_phase_b(item, data_root=DATA_ROOT)
        for item in protocols
        if item.split in {"train", "validation"}
    )
    spec = phase_b_public_spec("alien_device", "hard", "opaque")

    write_public_staging_bundle(tmp_path, spec, selected)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest["splits"]) == {"train", "validation"}
    assert not manifest["test_sealed"]
    assert "private_source" not in json.dumps(manifest)
    assert audit_public_bundle(tmp_path, spec).passed


@pytest.mark.parametrize(
    ("tier", "functional_phrases", "opaque_phrases"),
    [
        (
            "easy",
            ("input-driven memory", "causal contribution"),
            ("dynamic-memory mechanism", "declared input"),
        ),
        (
            "hard",
            ("input memory", "persistent coupling", "nonlinear feedback"),
            ("input memory", "persistent coupling", "nonlinear feedback"),
        ),
    ],
)
def test_alien_semantic_pairs_preserve_tier_mechanism_burden(
    tier: str,
    functional_phrases: tuple[str, ...],
    opaque_phrases: tuple[str, ...],
) -> None:
    prompts = [
        render_phase_b_prompts(
            phase_b_public_spec("alien_device", tier, variant, data_root=DATA_ROOT)
        )[0]
        for variant in ("functional", "opaque")
    ]

    assert _mechanism_bullets(prompts[0]) == _mechanism_bullets(prompts[1]) == 1
    assert all(phrase in prompts[0] for phrase in functional_phrases)
    assert all(phrase in prompts[1] for phrase in opaque_phrases)
    if tier == "easy":
        assert all(phrase not in prompts[1] for phrase in functional_phrases)


@pytest.mark.parametrize("tier", ["easy", "hard"])
def test_opaque_alien_identifies_target_as_primary_output(tier: str) -> None:
    prompt = render_phase_b_prompts(
        phase_b_public_spec("alien_device", tier, "opaque", data_root=DATA_ROOT)
    )[0]

    assert "- v01(t): primary output" in prompt


def test_leakage_audit_catches_injected_private_truth(tmp_path: Path) -> None:
    protocol = phase_b_protocols("dalla_man", task="T1")[0]
    trajectory = simulate_phase_b(protocol, data_root=DATA_ROOT)
    spec = phase_b_public_spec("dalla_man", "hard", "obfuscated", task="T1")
    write_public_staging_bundle(tmp_path, spec, (trajectory,))
    prompt = tmp_path / "proposer_prompt.txt"
    prompt.write_text(prompt.read_text() + "private Qgut glucose equation\n")

    report = audit_public_bundle(tmp_path, spec)

    assert not report.passed
    assert any("forbidden private token" in item for item in report.violations)


def test_phase_b_production_bundle_loads_tidy_development_and_test(
    tmp_path: Path,
) -> None:
    spec = phase_b_public_spec("cstr", "easy", "named", data_root=DATA_ROOT)
    trajectories = tuple(
        simulate_phase_b(protocol, data_root=DATA_ROOT)
        for protocol in phase_b_protocols("cstr")
    )
    cell_root = tmp_path / "phase_b_v1" / spec.benchmark_id
    write_public_production_bundle(cell_root, spec, trajectories)
    config = DataConfig(
        root=tmp_path,
        benchmark_id=spec.benchmark_id,
        tier="easy",
        use_clean_observations=False,
    )

    loader = BenchmarkLoader()
    development = loader.load_development(config)
    with pytest.raises(ChannelRoleError, match="FrozenTestAccess"):
        loader.load_test(config)
    with pytest.raises(ChannelRoleError, match=r"cannot be opened by load\(\)"):
        loader.load(config)
    with pytest.raises(ChannelRoleError, match="does not match"):
        loader.load_test(
            config,
            access=FrozenTestAccess(
                benchmark_id="wrong-cell",
                tier="easy",
                selection_hash="frozen-selection-hash",
            ),
        )
    test = loader.load_test(
        config,
        access=FrozenTestAccess(
            benchmark_id=spec.benchmark_id,
            tier="easy",
            selection_hash="frozen-selection-hash",
        ),
    )

    assert development.train.name is SplitName.TRAIN
    assert development.validation.name is SplitName.VALIDATION
    assert test.name is SplitName.TEST
    trajectory = development.train.trajectories[0]
    assert set(trajectory.targets) == {"T"}
    assert set(trajectory.auxiliaries) == {"C", "Tj"}
    assert set(trajectory.external_inputs) == {"Cf", "Tf", "Tjf"}
    assert set(trajectory.derivatives) == {"T", "C", "Tj"}
    assert all(np.isfinite(values).all() for values in trajectory.derivatives.values())
    manifest = json.loads((cell_root / "manifest.json").read_text())
    assert manifest["status"] == "production_registered"
    assert manifest["test_sealed"]

    with (cell_root / "train.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(DataAlignmentError, match="fingerprint mismatch"):
        loader.load_development(config)


def test_phase_b_production_bundle_requires_all_three_splits(tmp_path: Path) -> None:
    spec = phase_b_public_spec("cstr", "hard", "obfuscated", data_root=DATA_ROOT)
    trajectories = tuple(
        simulate_phase_b(protocol, data_root=DATA_ROOT)
        for protocol in phase_b_protocols("cstr")
        if protocol.split != "test"
    )

    with pytest.raises(ValueError, match="requires train, validation, and sealed test"):
        write_public_production_bundle(tmp_path, spec, trajectories)


def test_phase_b_staging_package_cannot_enter_production_loader(
    tmp_path: Path,
) -> None:
    spec = phase_b_public_spec("cstr", "hard", "named", data_root=DATA_ROOT)
    trajectories = tuple(
        simulate_phase_b(protocol, data_root=DATA_ROOT)
        for protocol in phase_b_protocols("cstr")
        if protocol.split != "test"
    )
    cell_root = tmp_path / "phase_b_v1" / spec.benchmark_id
    write_public_staging_bundle(cell_root, spec, trajectories)
    config = DataConfig(
        root=tmp_path,
        benchmark_id=spec.benchmark_id,
        tier="hard",
        use_clean_observations=False,
    )

    with pytest.raises(ChannelRoleError, match="not a frozen release"):
        BenchmarkLoader().load_development(config)
