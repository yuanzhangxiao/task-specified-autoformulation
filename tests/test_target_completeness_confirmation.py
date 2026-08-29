"""Tests for the fresh-structure target-completeness confirmation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.adversarial import AdversarialPair
from autoformalism.schemas import CandidateModel
from scripts.build_target_completeness_confirmation_pairs import (
    build_confirmation_pairs,
    select_fresh_baselines,
    source_structure_fingerprint,
    validate_development_prerequisite,
)
from scripts.verify_target_completeness_confirmation import verify_inputs

CONFIG = Path("configs/target_completeness_fresh_confirmation_v1.json")
SLURM = Path(
    "scripts/hpc/phase_b_target_completeness_fresh_confirmation_120b.slurm"
)


def _candidate(identifier: str, parameter: str = "k") -> CandidateModel:
    utilization = (
        f"{parameter} * X * Gp"
        if parameter == "k"
        else f"{parameter} * X * Gp**2"
    )
    return CandidateModel.model_validate(
        {
            "candidate_id": identifier,
            "parent_candidate_id": None,
            "states": [
                {"name": "Gp", "kind": "observed"},
                {"name": "I", "kind": "observed"},
                {"name": "X", "kind": "latent"},
            ],
            "processes": [
                {
                    "name": "U",
                    "expression": utilization,
                    "mechanisms": ["insulin_dependent_disposal"],
                }
            ],
            "state_equations": [
                {"state": "Gp", "rhs": "EGP - Uii - U"},
                {"state": "I", "rhs": "insulin_input - I"},
                {"state": "X", "rhs": "I - X"},
            ],
            "observation_mappings": [
                {"channel": "Gp", "expression": "Gp"},
                {"channel": "I", "expression": "I"},
                {"channel": "U", "expression": "U"},
            ],
            "parameters": [
                {
                    "name": parameter,
                    "scope": "global",
                    "bounds": {"lower": 0.0, "upper": 2.0},
                    "initialization_range": {"lower": 0.1, "upper": 1.0},
                }
            ],
            "initial_conditions": [
                {"state": "Gp", "scope": "global", "expression": "Gp"},
                {"state": "I", "scope": "global", "expression": "I"},
                {"state": "X", "scope": "global", "fixed_value": 0.0},
            ],
        }
    )


def _pair(pair_id: str, candidate: CandidateModel) -> AdversarialPair:
    return AdversarialPair(
        pair_id=pair_id,
        benchmark_id="benchmark",
        tier="easy",
        mutation_type="omitted_target_component",
        valid_candidate=candidate,
        adversarial_candidate=candidate,
    )


@pytest.fixture
def context() -> ValidationContext:
    return ValidationContext(
        targets=("Gp", "I", "U"),
        auxiliaries=("EGP", "Uii"),
        external_inputs=("insulin_input",),
        lagged_targets=("Gp", "I", "U"),
    )


def test_clean_wrapper_has_same_source_fingerprint(
    context: ValidationContext,
) -> None:
    baseline = _candidate("raw")
    pairs, _ = build_confirmation_pairs(
        (("fingerprint", baseline, "benchmark", "easy"),),
        contexts={("benchmark", "easy"): context},
        target_channel="U",
        total_process="U",
        dependent_process="Uid",
        supplied_component="Uii",
    )
    clean = pairs[0].valid_candidate

    assert source_structure_fingerprint(clean, context) == (
        source_structure_fingerprint(baseline, context)
    )
    processes = {item.name: item.expression for item in clean.processes}
    assert processes == {"Uid": "k * X * Gp", "U": "Uii + Uid"}


def test_selection_rejects_opened_clean_wrapper(
    context: ValidationContext,
) -> None:
    opened = _candidate("opened", "k")
    clean_pairs, _ = build_confirmation_pairs(
        (("openedfingerprint", opened, "benchmark", "easy"),),
        contexts={("benchmark", "easy"): context},
        target_channel="U",
        total_process="U",
        dependent_process="Uid",
        supplied_component="Uii",
    )
    fresh = _pair("fresh", _candidate("fresh", "q"))
    selected, excluded = select_fresh_baselines(
        (_pair("opened-source", opened), fresh),
        clean_pairs,
        baseline_count=1,
        benchmark_id="benchmark",
        tier="easy",
        context=context,
        target_channel="U",
        target_component="U",
        supplied_component="Uii",
    )

    assert selected[0][1].candidate_id == "fresh"
    assert source_structure_fingerprint(opened, context) in excluded


def test_development_prerequisite_fails_closed(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    pairs = tmp_path / "pairs.jsonl"
    analysis = tmp_path / "analysis.json"
    run = tmp_path / "run.json"
    config.write_text(
        json.dumps({"protocol": {"protocol_version": "protocol"}}),
        encoding="utf-8",
    )
    pairs.write_text("pair bytes\n", encoding="utf-8")
    analysis.write_text(
        json.dumps(
            {"schema_version": "target-completeness-judge-analysis-1", "passed": False}
        ),
        encoding="utf-8",
    )
    run.write_text(
        json.dumps(
            {
                "schema_version": "target-completeness-judge-run-1",
                "protocol_version": "protocol",
                "pairs_sha256": hashlib.sha256(pairs.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="did not pass"):
        validate_development_prerequisite(
            analysis_path=analysis,
            run_manifest_path=run,
            pairs_path=pairs,
            config_path=config,
            expected_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        )


def test_verifier_checks_frozen_hashes(tmp_path: Path) -> None:
    pair = _pair("pair", _candidate("candidate"))
    pairs = tmp_path / "pairs.jsonl"
    config = tmp_path / "config.json"
    analysis = tmp_path / "analysis.json"
    manifest = tmp_path / "manifest.json"
    pairs.write_text(pair.model_dump_json() + "\n", encoding="utf-8")
    config.write_text(json.dumps({"planned": {"pairs": 1}}), encoding="utf-8")
    analysis.write_text(json.dumps({"passed": True}), encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "target-completeness-fresh-confirmation-pairs-1",
                "status": "frozen_before_fresh_structure_calls",
                "pairs_sha256": hashlib.sha256(pairs.read_bytes()).hexdigest(),
                "protocol_config_sha256": hashlib.sha256(
                    config.read_bytes()
                ).hexdigest(),
                "development_prerequisite": {
                    "analysis_sha256": hashlib.sha256(
                        analysis.read_bytes()
                    ).hexdigest()
                },
                "selected_fingerprints_overlap_exclusions": False,
                "pair_count": 1,
                "selected_pair_ids": ["pair"],
            }
        ),
        encoding="utf-8",
    )

    result = verify_inputs(
        pairs_path=pairs,
        manifest_path=manifest,
        config_path=config,
        development_analysis_path=analysis,
    )

    assert result["status"] == "verified"


def test_config_and_launcher_freeze_confirmation() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    launcher = SLURM.read_text(encoding="utf-8")

    assert config["status"] == "frozen_before_fresh_structure_calls"
    assert config["pair_construction"]["holdout_unit"] == (
        "canonical_unwrapped_proposer_structure"
    )
    assert config["planned"]["logical_llm_calls"] == 20
    assert "verify_target_completeness_confirmation.py" in launcher
    assert "AF_JUDGE_ENTRYPOINT:=target_completeness" in launcher
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
