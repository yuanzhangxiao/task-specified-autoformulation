"""Portable SVD verification keeps exact scientific decisions and executed points."""

from copy import deepcopy
from shutil import copytree
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import fitting_feedback as legacy
from autoformalism.fitting import piecewise_assessment as piecewise
from autoformalism.fitting import portable_feedback as portable
from autoformalism.fitting import staged_feedback as staged
from autoformalism.fitting import staged_feedback_fixtures as fixtures


@pytest.fixture
def saved(tmp_path, monkeypatch):
    """A real tiny fit supplies sealed integrations without any archived model use."""
    monkeypatch.setattr(staged.public, "_source_identity", lambda: "portable-test")
    result = fixtures.fit_fixture("piecewise_inactive_guard", tmp_path / "fit")
    folder = tmp_path / "feedback"
    staged.run(
        staged.training_problem(result["problem"]),
        staged.adapter.FitResult.model_validate(result["fit"]),
        result["backend"],
        folder,
    )
    return folder


def _replace(path, value):
    """Intentionally reseal mutations to test semantics beyond checksum validation."""
    path.unlink()
    seal(path, value)


def test_actual_plan_and_environment_are_sealed_before_first_perturbation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(staged.public, "_source_identity", lambda: "portable-test")
    result = fixtures.fit_fixture("piecewise_inactive_guard", tmp_path / "fit")
    folder = tmp_path / "feedback"
    evaluate = staged.evaluate_stage
    saw_plan = []

    def checking(*args, **kwargs):
        if kwargs.get("metadata_policy") is None and not args[5]:
            plan = read_seal(folder / "perturbation_plan.json")
            environment = read_seal(folder / "numerical_environment.json")
            assert plan["environment_sha256"] == staged.public.content_sha256(
                environment
            )
            assert environment["numpy_build"]["Build Dependencies"]["blas"]
            saw_plan.append(plan)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(staged, "evaluate_stage", checking)
    staged.run(
        staged.training_problem(result["problem"]),
        staged.adapter.FitResult.model_validate(result["fit"]),
        result["backend"],
        folder,
    )
    assert saw_plan and all(plan == saw_plan[0] for plan in saw_plan)
    snapshot = portable.load_snapshot(folder)
    assert snapshot.provenance["plan_origin"] == "sealed_plan"
    assert len(snapshot.plan) == 8
    assert len(snapshot.provenance["stage_environment_hashes"]) == len(
        snapshot.operations
    )


def test_completed_and_legacy_recovery_never_regenerate_svd_directions(
    saved, monkeypatch
):
    original = portable.load_snapshot(saved)
    plan_function = piecewise.perturbation_plan

    def only_frozen(*args, **kwargs):
        assert kwargs.get("weak_direction") is not None, "fresh experimental direction"
        return plan_function(*args, **kwargs)

    monkeypatch.setattr(piecewise, "perturbation_plan", only_frozen)
    monkeypatch.setattr(
        staged, "evaluate_stage", lambda *a, **kw: pytest.fail("integration")
    )
    assert portable.load_snapshot(saved).packet == original.packet
    (saved / "perturbation_plan.json").unlink()
    (saved / "numerical_environment.json").unlink()
    legacy_snapshot = portable.load_snapshot(saved)
    assert legacy_snapshot.plan == original.plan
    assert legacy_snapshot.packet == original.packet
    assert legacy_snapshot.provenance["legacy_environment_unrecorded"]
    assert not (saved / "perturbation_plan.json").exists()


def test_small_svd_variation_is_accepted_without_rewriting_source(saved):
    packet = read_seal(saved / "result.json")
    sensitivity = packet["diagnostics"]["active_sensitivity"]
    sensitivity["singular_values"][0] *= 1 + 1e-13
    sensitivity["column_norms"][0] *= 1 + 1e-13
    sensitivity["weak_direction"]["a"] *= -1
    _replace(saved / "result.json", packet)
    digest = (saved / "result.json").read_bytes()
    assert portable.load_snapshot(saved).packet.model_dump(mode="json") == packet
    assert (saved / "result.json").read_bytes() == digest


@pytest.mark.parametrize(
    "mutation",
    [
        "rank",
        "status",
        "singular",
        "condition",
        "norm",
        "weak",
        "derivative",
        "decision",
    ],
)
def test_resealed_numeric_and_discrete_fabrications_fail(saved, mutation):
    packet = read_seal(saved / "result.json")
    sensitivity = packet["diagnostics"]["active_sensitivity"]
    direction = packet["diagnostics"]["piecewise_assessment"]["directions"][0]
    if mutation == "rank":
        sensitivity["rank"] = 0
    elif mutation == "status":
        direction["status"] = "numerical_noise"
    elif mutation == "singular":
        sensitivity["singular_values"][0] *= 1.01
    elif mutation == "condition":
        sensitivity["condition_number"] = 2
    elif mutation == "norm":
        sensitivity["column_norms"][0] *= 1.01
    elif mutation == "weak":
        sensitivity["weak_direction"]["a"] = 0.5
    elif mutation == "derivative":
        direction["steps"][0]["central_error"] += 1e-5
    else:
        direction["steps"][0]["central_agrees"] = False
    _replace(saved / "result.json", packet)
    with pytest.raises(ValueError):
        portable.load_snapshot(saved)


def test_resealed_frozen_point_cannot_substitute_another_experiment(saved):
    plan = read_seal(saved / "perturbation_plan.json")
    plan["points"][0]["vector"][0] += 1e-10
    _replace(saved / "perturbation_plan.json", plan)
    with pytest.raises(ValueError, match="exact prescribed points"):
        portable.load_snapshot(saved)


def test_legacy_weak_direction_must_match_exact_stage_point_hashes(saved):
    (saved / "perturbation_plan.json").unlink()
    packet = read_seal(saved / "result.json")
    direction = packet["diagnostics"]["piecewise_assessment"]["directions"][-1]
    direction["direction_vector"][0] *= -1
    _replace(saved / "result.json", packet)
    with pytest.raises(ValueError, match="frozen plan"):
        portable.load_snapshot(saved)


def _nullspace_evidence(matrix):
    names = ["a", "b", "c"]
    summary = legacy.LocalSensitivity.model_validate(
        checks.sensitivity(matrix, np.ones(3), names)
    )
    operations = {
        "primary": {
            "status": "complete",
            "value": {
                "metadata": {
                    "names": names,
                    "units": [1.0, 1.0, 1.0],
                    "activity": {"active_parameters": names},
                }
            },
        },
        "sensitivity": {"status": "complete", "value": {"jacobian": matrix.tolist()}},
    }
    return summary, operations


def test_rank_deficient_nullspace_rotation_and_condition_variation():
    matrix = np.array(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]
    )
    expected, operations = _nullspace_evidence(matrix)
    raw = expected.model_dump(mode="json")
    raw["weak_direction"] = {"a": 1 / np.sqrt(2), "b": -1 / np.sqrt(2), "c": 0.0}
    raw["singular_values"][-1] = 2e-17
    raw["condition_number"] = raw["singular_values"][0] / 2e-17
    saved = legacy.LocalSensitivity.model_validate(raw)
    portable._sensitivity(saved, expected, operations)
    bad = saved.model_copy(update={"weak_direction": {"a": 1.0, "b": 0.0, "c": 0.0}})
    with pytest.raises(ValueError, match="weakest subspace"):
        portable._sensitivity(bad, expected, operations)
    bad = saved.model_copy(update={"condition_number": 5.0})
    with pytest.raises(ValueError, match="condition_number"):
        portable._sensitivity(bad, expected, operations)


def test_rotated_nullspace_plan_keeps_exact_executed_vectors():
    oracle = SimpleNamespace(
        names=("a", "b", "c"),
        units=np.ones(3),
        lower=np.full(3, -10.0),
        upper=np.full(3, 10.0),
        system=SimpleNamespace(initial_parameter_names=()),
    )
    matrix = np.tile([1.0, 1.0, 1.0], (4, 1))
    policy = piecewise.PiecewiseAssessmentPolicy(include_weak_direction=True)
    direction = [1 / np.sqrt(2), -1 / np.sqrt(2), 0.0]
    plan = piecewise.perturbation_plan(
        oracle,
        np.ones(3),
        oracle.names,
        policy,
        jacobian=matrix,
        weak_direction=direction,
    )
    assert (
        portable._validate_plan(oracle, np.ones(3), oracle.names, policy, matrix, plan)
        == plan
    )
    assert plan[-1].direction_vector == tuple(direction)
    assert plan[-1].vector == tuple(np.ones(3) + policy.steps[-1] * np.array(direction))


def test_source_snapshot_hash_changes_if_original_provenance_changes(saved, tmp_path):
    """Linked supplements can detect later resealing even if semantics remain valid."""
    before = portable.load_snapshot(saved)
    copied = tmp_path / "copy"
    copytree(saved, copied)
    environment = deepcopy(read_seal(copied / "numerical_environment.json"))
    environment["processor"] += " changed"
    _replace(copied / "numerical_environment.json", environment)
    plan = read_seal(copied / "perturbation_plan.json")
    plan["environment_sha256"] = staged.public.content_sha256(environment)
    _replace(copied / "perturbation_plan.json", plan)
    assert portable.load_snapshot(copied).source_sha256 != before.source_sha256


def test_non_svd_summary_changes_are_not_tolerated(saved):
    packet = read_seal(saved / "result.json")
    packet["diagnostics"]["replay_training_nmse"] += 1e-16
    _replace(saved / "result.json", packet)
    with pytest.raises(ValueError, match="evidence differs"):
        portable.load_snapshot(saved)


def test_resealed_operation_cannot_call_complete_evidence_unavailable(saved):
    manifest = read_seal(saved / "stages.json")
    operations = {
        key: read_seal(saved / "stages" / key / "record.json") for key, _ in manifest
    }
    operation = operations["sensitivity"]
    operation["status"] = "unavailable"
    operation["error"] = "forged failure"
    path = saved / "stages/sensitivity"
    _replace(path / "record.json", operation)
    _replace(
        path / "result.json", {k: v for k, v in operation.items() if k != "progress"}
    )
    _replace(
        saved / "stages.json",
        [[k, staged.public.content_sha256(v)] for k, v in operations.items()],
    )
    context = read_seal(saved / "context.json")
    packet = staged._assemble(
        staged.training_problem(context["problem"]),
        staged.adapter.FitResult.model_validate(context["fit"]),
        context["backend"],
        staged.Policy.model_validate(context["policy"]),
        legacy.Binding.model_validate(read_seal(saved / "identity.json")),
        operations,
    )
    _replace(saved / "result.json", packet.model_dump(mode="json"))
    (saved / "perturbation_plan.json").unlink()
    with pytest.raises(ValueError, match="unavailable diagnostic stage"):
        portable.load_snapshot(saved)


def test_interrupted_run_resumes_only_from_the_presealed_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(staged.public, "_source_identity", lambda: "portable-test")
    result = fixtures.fit_fixture("piecewise_inactive_guard", tmp_path / "fit")
    folder = tmp_path / "feedback"
    inputs = (
        staged.training_problem(result["problem"]),
        staged.adapter.FitResult.model_validate(result["fit"]),
        result["backend"],
    )
    evaluate = staged.evaluate_stage

    def interrupt(*args, **kwargs):
        if kwargs.get("metadata_policy") is None and not args[5]:
            raise KeyboardInterrupt
        return evaluate(*args, **kwargs)

    policy = staged.Policy(perturbation_seconds=1, perturbation_total_seconds=10)
    monkeypatch.setattr(staged, "evaluate_stage", interrupt)
    with pytest.raises(KeyboardInterrupt):
        staged.run(*inputs, folder, policy)
    frozen_bytes = (folder / "perturbation_plan.json").read_bytes()
    plan_function = piecewise.perturbation_plan

    def frozen_only(*args, **kwargs):
        assert kwargs.get("weak_direction") is not None
        return plan_function(*args, **kwargs)

    monkeypatch.setattr(piecewise, "perturbation_plan", frozen_only)
    monkeypatch.setattr(staged, "evaluate_stage", evaluate)
    packet = staged.run(*inputs, folder, policy)
    assert packet.diagnostics.stages[3].status == "interrupted"
    assert (folder / "perturbation_plan.json").read_bytes() == frozen_bytes
    assert portable.load_snapshot(folder).packet == packet


def test_large_column_cancellation_uses_saved_jacobian_roundoff(saved):
    snapshot = portable.load_snapshot(saved)
    matrix = np.array([[1e8, 1e-8, -1e8]])
    direction = np.ones(3) / np.sqrt(3)
    summary, operations = _nullspace_evidence(matrix)
    direct = float(abs((matrix @ direction)[0]))
    reordered = float(abs(np.sum((matrix[0] * direction)[[0, 2, 1]])))
    packet = snapshot.packet.model_dump(mode="json")
    packet["diagnostics"]["active_sensitivity"] = summary.model_dump(mode="json")
    report = packet["diagnostics"]["piecewise_assessment"]["directions"][0]
    report["direction_vector"] = direction.tolist()
    report["ad_derivative_rms"] = direct
    packet["diagnostics"]["piecewise_assessment"]["directions"] = [report]
    recomputed = staged.FeedbackPacket.model_validate(packet)
    report["ad_derivative_rms"] = reordered
    report["steps"][0]["central_error"] += abs(direct - reordered)
    historical = staged.FeedbackPacket.model_validate(packet)
    portable.verify_packet(historical, recomputed, operations)
    report["ad_derivative_rms"] += 100 * portable._direction_roundoff(
        operations, direction.tolist()
    )
    with pytest.raises(ValueError, match="ad_derivative_rms"):
        portable.verify_packet(
            staged.FeedbackPacket.model_validate(packet), recomputed, operations
        )
