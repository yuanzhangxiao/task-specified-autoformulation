"""Tests for representation-invariant Phase-B hidden-subspace evaluation."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from autoformalism.baselines.raw_data_agent import raw_agent_validation_context
from autoformalism.benchmarks import (
    phase_b_protocols,
    phase_b_public_spec,
    simulate_phase_b,
    write_public_production_bundle,
)
from autoformalism.benchmarks.phase_b_generation import PrivateTrajectory
from autoformalism.config import DataConfig
from autoformalism.data import (
    BenchmarkLoader,
    BenchmarkRegistry,
    DatasetSplit,
    SplitName,
    Trajectory,
)
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.final_evaluation import (
    FrozenEvaluationSubject,
    FrozenParameterization,
    SourceArtifactProvenance,
    TargetPredictionEndpoint,
)
from autoformalism.rebuttal.hidden import hidden_subspace_nmse
from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec
from autoformalism.rebuttal.phase_b_hidden_subspace import (
    PhaseBHiddenSubspaceContract,
    _private_directions,
    _validate_private_matches_public,
    _validate_private_release_identity,
    evaluate_phase_b_hidden_subspace,
    phase_b_hidden_subspace_contract,
)
from autoformalism.rebuttal.phase_b_mechanism_specs import (
    phase_b_public_mechanism_spec,
)
from autoformalism.schemas import CandidateModel


def test_hidden_subspace_is_invariant_to_linear_coordinate_mixing() -> None:
    rng = np.random.default_rng(7)
    train_latent = rng.normal(size=(200, 2))
    test_latent = rng.normal(size=(80, 2))
    reference_mixing = np.asarray([[1.0, 0.2, -0.4], [0.3, 1.4, 0.8]])
    candidate_mixing = np.asarray([[2.0, -0.5], [0.7, -1.5]])

    metric = hidden_subspace_nmse(
        train_latent @ candidate_mixing,
        train_latent @ reference_mixing,
        test_latent @ candidate_mixing,
        test_latent @ reference_mixing,
        claimed_dimension=2,
        structurally_recovered=True,
    )

    assert metric.recovered is True
    assert metric.candidate_rank == 2
    assert metric.candidate_rank_coverage == 1.0
    assert metric.train_relative_residual == pytest.approx(0.0, abs=1e-24)
    assert metric.aligned_test_nmse == pytest.approx(0.0, abs=1e-24)


def test_hidden_subspace_withholds_score_when_candidate_rank_is_too_small() -> None:
    coordinate = np.linspace(-1.0, 1.0, 40)[:, None]
    reference = np.column_stack((coordinate[:, 0], coordinate[:, 0] ** 2))

    metric = hidden_subspace_nmse(
        coordinate,
        reference,
        coordinate[:20],
        reference[:20],
        claimed_dimension=2,
        structurally_recovered=True,
    )

    assert metric.recovered is False
    assert metric.candidate_rank == 1
    assert metric.candidate_rank_coverage == 0.5
    assert metric.aligned_test_nmse is None


def test_hidden_subspace_withholds_score_without_structural_recovery() -> None:
    values = np.column_stack((np.arange(20.0), np.arange(20.0) ** 2))

    metric = hidden_subspace_nmse(
        values,
        values,
        values[:10],
        values[:10],
        claimed_dimension=2,
        structurally_recovered=False,
    )

    assert metric.candidate_rank == 2
    assert metric.recovered is False
    assert metric.aligned_test_nmse is None


def test_nominal_pairing_accepts_small_solver_reproducibility_difference() -> None:
    public, private, contract = _nominal_pairing_case(
        np.asarray([25.0, 200.0]),
        np.asarray([25.0 + 3e-5, 200.0 - 5e-4]),
    )

    _validate_private_matches_public(public, private, contract)


def test_nominal_pairing_rejects_scientifically_material_difference() -> None:
    public, private, contract = _nominal_pairing_case(
        np.asarray([100.0, 200.0]),
        np.asarray([100.1, 200.0]),
    )

    with pytest.raises(ValueError, match=r"max_abs=0\.1"):
        _validate_private_matches_public(public, private, contract)


@pytest.mark.parametrize("family", ["cstr", "alien_device"])
def test_private_release_identity_accepts_frozen_sources(family: str) -> None:
    contract = _private_release_contract(family)

    _validate_private_release_identity(contract, Path("data_raw"))


def test_private_release_identity_rejects_modified_source(tmp_path: Path) -> None:
    source = Path(
        "data_raw/benchmark6_alien_device/private/selected_system_spec.json"
    )
    destination = (
        tmp_path
        / "benchmark6_alien_device/private/selected_system_spec.json"
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="private system source SHA-256 differs"):
        _validate_private_release_identity(
            _private_release_contract("alien_device"), tmp_path
        )


def test_alien_directions_do_not_use_cross_solver_nominal_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = next(
        item for item in phase_b_protocols("alien_device") if item.split == "train"
    )
    time = np.asarray([0.0, 1.0])
    split = DatasetSplit(
        SplitName.TRAIN,
        (
            Trajectory(
                trajectory_id="train_000",
                time=time,
                targets={"v01": np.zeros(2)},
                auxiliaries={},
                external_inputs={"u01": np.zeros(2)},
                fixed_covariates={},
                derivatives={},
            ),
        ),
        "public-fingerprint",
    )
    contract = _private_release_contract("alien_device").model_copy(
        update={
            "private_mechanism_directions": ("output_core",),
            "target_sources": {"v01": "y"},
        }
    )

    def simulate(*_args: object, **kwargs: object) -> PrivateTrajectory:
        shifted = bool(kwargs.get("private_mechanism_scales"))
        values = np.asarray([100.1, 101.2]) if shifted else np.asarray([100.0, 101.0])
        return PrivateTrajectory(
            protocol_id=protocol.protocol_id,
            family="alien_device",
            time=time,
            state_names=("y",),
            states=values[:, None],
            input_names=("u",),
            inputs=np.zeros((2, 1)),
        )

    monkeypatch.setattr(
        "autoformalism.rebuttal.phase_b_hidden_subspace."
        "_validate_private_release_identity",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "autoformalism.rebuttal.phase_b_hidden_subspace.phase_b_protocols",
        lambda *_args, **_kwargs: (protocol,),
    )
    monkeypatch.setattr(
        "autoformalism.rebuttal.phase_b_hidden_subspace.simulate_phase_b",
        simulate,
    )
    monkeypatch.setattr(
        "autoformalism.rebuttal.phase_b_hidden_subspace."
        "_validate_private_matches_public",
        lambda *_args: pytest.fail("Alien must not compare fresh nominal values"),
    )

    directions = _private_directions(
        split,
        contract,
        {"v01": 1.0},
        private_data_root=Path("data_raw"),
    )

    assert directions[:, 0] == pytest.approx([100.0, 200.0])


def test_phase_b_contracts_preserve_frozen_private_claims() -> None:
    data_root = Path("data_raw")
    t1 = phase_b_hidden_subspace_contract(
        "phase_b_dalla_man_t1_canonical_named_easy",
        data_root=data_root,
    )
    t4 = phase_b_hidden_subspace_contract(
        "phase_b_dalla_man_t4_canonical_named_easy",
        data_root=data_root,
    )
    cstr = phase_b_hidden_subspace_contract(
        "phase_b_cstr_controlled_reactor_mechanism_canonical_named_hard",
        data_root=data_root,
    )
    alien = phase_b_hidden_subspace_contract(
        "phase_b_anonymous_system_task_canonical_opaque_hard",
        data_root=data_root,
    )

    assert t1.mode == "mechanism_response_equivalence"
    assert t1.claimed_dimension == 2
    assert t4.mode == "not_applicable"
    assert t4.claimed_dimension == 0
    assert cstr.mode == "mechanism_sensitivity_subspace"
    assert cstr.claimed_dimension == 2
    assert alien.claimed_dimension == 3
    assert len({item.sha256 for item in (t1, t4, cstr, alien)}) == 4


def test_hidden_evaluator_populates_recovered_endpoint_from_frozen_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "candidate",
            "parent_candidate_id": None,
            "states": [
                {"name": "memory", "kind": "latent", "mechanisms": ["memory"]},
                {"name": "x", "kind": "observed"},
            ],
            "state_equations": [
                {"state": "memory", "rhs": "-memory + u"},
                {"state": "x", "rhs": "memory"},
            ],
            "observation_mappings": [{"channel": "x", "expression": "x"}],
            "initial_conditions": [
                {"state": "memory", "scope": "global", "fixed_value": 0.0},
                {"state": "x", "scope": "global", "expression": "x"},
            ],
        }
    )
    candidate_hash = hashlib.sha256(
        candidate.model_dump_json().encode("utf-8")
    ).hexdigest()
    context = ValidationContext(targets=("x",), external_inputs=("u",))
    subject = FrozenEvaluationSubject(
        subject_id="subject",
        method="method",
        benchmark_id="synthetic",
        tier="easy",
        repetition=0,
        private_metrics_opened_after_freeze=True,
        source_provenance=SourceArtifactProvenance(
            adapter="direct_candidate",
            request_id="subject",
            source_path="candidate.json",
            source_sha256="a" * 64,
            candidate_sha256=candidate_hash,
        ),
        candidate=candidate,
        parameterization=FrozenParameterization(status="not_required"),
        validation_context=context,
        target_prediction=TargetPredictionEndpoint(
            status="available",
            evaluation_protocol="unseen_condition_free_rollout",
            normalized_mse=0.0,
            per_target_normalized_mse={"x": 0.0},
            normalization_scales={"x": 1.0},
            trajectory_count=1,
            successful_trajectory_count=1,
        ),
    )
    train = _split(SplitName.TRAIN, "train", 4)
    test = _split(SplitName.TEST, "test", 3)
    mechanism_spec = MechanismEvaluationSpec.model_validate(
        {
            "benchmark_id": "synthetic",
            "tier": "easy",
            "required_mechanisms": [
                {
                    "id": "memory",
                    "required_drivers": ["u"],
                    "required_targets": ["x"],
                    "requires_dynamic_memory": True,
                }
            ],
        }
    )
    contract = PhaseBHiddenSubspaceContract(
        benchmark_id="synthetic",
        family="cstr",
        task="synthetic",
        tier="easy",
        dynamics="canonical",
        mode="mechanism_sensitivity_subspace",
        private_mechanism_directions=("private_memory",),
        claimed_dimension=1,
        target_sources={"x": "private_x"},
        public_prompt_sha256="b" * 64,
    )
    train_direction = np.arange(1.0, 5.0)[:, None]
    test_direction = np.arange(5.0, 8.0)[:, None]

    def directions(
        _subject: FrozenEvaluationSubject,
        split: DatasetSplit,
        *_args: object,
        **_kwargs: object,
    ) -> np.ndarray:
        return train_direction if split.name is SplitName.TRAIN else test_direction

    monkeypatch.setattr(
        "autoformalism.rebuttal.phase_b_hidden_subspace._candidate_directions",
        directions,
    )

    updated, outcome = evaluate_phase_b_hidden_subspace(
        subject,
        training_split=train,
        test_split=test,
        public_mechanism_spec=mechanism_spec,
        contract=contract,
        reference_directions=(train_direction, test_direction),
    )

    assert outcome.status == "available"
    assert outcome.metric is not None
    assert outcome.metric.aligned_test_nmse == pytest.approx(0.0)
    assert updated.hidden_mechanisms[0].status == "available"
    assert updated.hidden_mechanisms[0].recovered is True


def _split(name: SplitName, identifier: str, count: int) -> DatasetSplit:
    time = np.arange(float(count))
    return DatasetSplit(
        name,
        (
            Trajectory(
                trajectory_id=identifier,
                time=time,
                targets={"x": np.zeros(count)},
                auxiliaries={},
                external_inputs={"u": np.zeros(count)},
                fixed_covariates={},
                derivatives={},
            ),
        ),
        f"{identifier}-fingerprint",
    )


def _nominal_pairing_case(
    public_values: np.ndarray,
    private_values: np.ndarray,
) -> tuple[
    DatasetSplit,
    tuple[PrivateTrajectory, ...],
    PhaseBHiddenSubspaceContract,
]:
    time = np.arange(float(len(public_values)))
    public = DatasetSplit(
        SplitName.TRAIN,
        (
            Trajectory(
                trajectory_id="train_000",
                time=time,
                targets={"x": public_values},
                auxiliaries={},
                external_inputs={"u": np.zeros(len(time))},
                fixed_covariates={},
                derivatives={},
            ),
        ),
        "public-fingerprint",
    )
    private = (
        PrivateTrajectory(
            protocol_id="train_case",
            family="cstr",
            time=time,
            state_names=("private_x",),
            states=private_values[:, None],
            input_names=("u",),
            inputs=np.zeros((len(time), 1)),
        ),
    )
    contract = PhaseBHiddenSubspaceContract(
        benchmark_id="synthetic",
        family="cstr",
        task="synthetic",
        tier="easy",
        dynamics="canonical",
        mode="mechanism_sensitivity_subspace",
        private_mechanism_directions=("mechanism",),
        claimed_dimension=1,
        target_sources={"x": "private_x"},
        public_prompt_sha256="a" * 64,
    )
    return public, private, contract


def _private_release_contract(family: str) -> PhaseBHiddenSubspaceContract:
    return PhaseBHiddenSubspaceContract(
        benchmark_id=f"synthetic_{family}",
        family=family,
        task="synthetic",
        tier="easy",
        dynamics="canonical",
        mode="mechanism_sensitivity_subspace",
        private_mechanism_directions=("mechanism",),
        claimed_dimension=1,
        target_sources={"x": "private_x"},
        public_prompt_sha256="a" * 64,
    )


def test_hidden_cli_and_merge_preserve_not_applicable_t4_contract(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    private_root = repository / "data_raw"
    public_spec = phase_b_public_spec(
        "dalla_man",
        "easy",
        "named",
        task="T4",
        data_root=private_root,
    )
    protocols = phase_b_protocols("dalla_man", task="T4")
    selected = tuple(
        next(item for item in protocols if item.split == split)
        for split in ("train", "validation", "test")
    )
    trajectories = tuple(
        simulate_phase_b(item, data_root=private_root) for item in selected
    )
    public_root = tmp_path / "public"
    write_public_production_bundle(
        public_root / "phase_b_v1" / public_spec.benchmark_id,
        public_spec,
        trajectories,
    )
    registry = BenchmarkRegistry()
    development = BenchmarkLoader(registry).load_development(
        DataConfig(
            root=public_root,
            benchmark_id=public_spec.benchmark_id,
            tier="easy",
        )
    )
    context = raw_agent_validation_context(
        development,
        registry.get(public_spec.benchmark_id),
    )
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "constant_targets",
            "parent_candidate_id": None,
            "states": [
                {"name": "Gp", "kind": "observed"},
                {"name": "I", "kind": "observed"},
            ],
            "state_equations": [
                {"state": "Gp", "rhs": "0.0"},
                {"state": "I", "rhs": "0.0"},
            ],
            "observation_mappings": [
                {"channel": "Gp", "expression": "Gp"},
                {"channel": "I", "expression": "I"},
            ],
            "initial_conditions": [
                {"state": "Gp", "scope": "global", "expression": "Gp"},
                {"state": "I", "scope": "global", "expression": "I"},
            ],
        }
    )
    candidate_hash = hashlib.sha256(
        candidate.model_dump_json().encode("utf-8")
    ).hexdigest()
    subjects_list = []
    for repetition in range(2):
        subjects_list.append(
            FrozenEvaluationSubject(
                subject_id=f"subject-{repetition}",
                method="test",
                benchmark_id=public_spec.benchmark_id,
                tier="easy",
                repetition=repetition,
                private_metrics_opened_after_freeze=True,
                source_provenance=SourceArtifactProvenance(
                    adapter="direct_candidate",
                    request_id=f"subject-{repetition}",
                    source_path="candidate.json",
                    source_sha256="a" * 64,
                    candidate_sha256=candidate_hash,
                ),
                candidate=candidate,
                parameterization=FrozenParameterization(status="not_required"),
                validation_context=context,
                target_prediction=TargetPredictionEndpoint(
                    status="available",
                    evaluation_protocol="unseen_condition_free_rollout",
                    normalized_mse=1.0,
                    per_target_normalized_mse={"Gp": 1.0, "I": 1.0},
                    normalization_scales={"Gp": 1.0, "I": 1.0},
                    trajectory_count=1,
                    successful_trajectory_count=1,
                ),
            )
        )
    subjects = tmp_path / "subjects.jsonl"
    subjects.write_text(
        "".join(item.model_dump_json() + "\n" for item in subjects_list),
        encoding="utf-8",
    )
    mechanism_root = tmp_path / "mechanisms"
    mechanism_root.mkdir()
    mechanism_spec = phase_b_public_mechanism_spec(public_spec)
    (mechanism_root / f"{public_spec.benchmark_id}.json").write_text(
        mechanism_spec.model_dump_json(),
        encoding="utf-8",
    )
    output = tmp_path / "hidden"
    for shard in range(2):
        subprocess.run(
            [
                sys.executable,
                str(repository / "scripts/evaluate_phase_b_hidden_subspace.py"),
                "--subjects",
                str(subjects),
                "--public-data-root",
                str(public_root),
                "--private-data-root",
                str(private_root),
                "--mechanism-config-root",
                str(mechanism_root),
                "--output-root",
                str(output),
                "--shard-index",
                str(shard),
                "--shard-count",
                "2",
            ],
            cwd=repository,
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/merge_phase_b_hidden_subspace.py"),
            "--subjects",
            str(subjects),
            "--input-root",
            str(output),
            "--output-root",
            str(output),
        ],
        cwd=repository,
        check=True,
    )
    merged = tuple(
        FrozenEvaluationSubject.model_validate_json(line)
        for line in (output / "hidden_subjects.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    manifest = json.loads(
        (output / "hidden_merge_manifest.json").read_text(encoding="utf-8")
    )
    assert [item.subject_id for item in merged] == ["subject-0", "subject-1"]
    assert all(item.hidden_mechanisms[0].status == "not_applicable" for item in merged)
    assert manifest["not_applicable_count"] == 2
    assert manifest["failed_count"] == 0


def test_hidden_contract_audit_confirms_rank_and_semantic_identity(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    private_root = repository / "data_raw"
    public_root = tmp_path / "public"
    benchmark_ids = []
    protocols = phase_b_protocols("cstr")
    trajectories = tuple(
        simulate_phase_b(item, data_root=private_root) for item in protocols
    )
    for variant in ("named", "obfuscated"):
        public_spec = phase_b_public_spec(
            "cstr",
            "easy",
            variant,
            data_root=private_root,
        )
        benchmark_ids.append(public_spec.benchmark_id)
        write_public_production_bundle(
            public_root / "phase_b_v1" / public_spec.benchmark_id,
            public_spec,
            trajectories,
        )
    output = tmp_path / "contract-audit.json"
    command = [
        sys.executable,
        str(repository / "scripts/audit_phase_b_hidden_subspace_contracts.py"),
    ]
    for benchmark_id in benchmark_ids:
        command.extend(("--benchmark-id", benchmark_id))
    command.extend(
        (
            "--public-data-root",
            str(public_root),
            "--private-data-root",
            str(private_root),
            "--output",
            str(output),
        )
    )
    subprocess.run(command, cwd=repository, check=True)

    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["status"] == "pass"
    assert audit["claimed_rank_pass"] is True
    assert audit["semantic_pair_identity_pass"] is True
    assert len(audit["semantic_pair_checks"]) == 1
    assert audit["semantic_pair_checks"][0]["identical"] is True

    failed_output = tmp_path / "failed-contract-audit.json"
    failed_command = [
        *command[:-4],
        "--private-data-root",
        str(tmp_path / "missing-private-root"),
        "--output",
        str(failed_output),
    ]
    failed = subprocess.run(failed_command, cwd=repository, check=False)
    failed_audit = json.loads(failed_output.read_text(encoding="utf-8"))
    assert failed.returncode == 1
    assert failed_audit["status"] == "fail"
    assert failed_audit["failed_benchmark_count"] == 2
    assert len(failed_audit["rows"]) == 2
    assert all(item["audit_status"] == "failed" for item in failed_audit["rows"])
