"""Adapter quality contract, shared fallback ceilings and incumbent preservation."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import rollout_adapter as adapter
from autoformalism.fitting import scaling_recovery as recovery
from tests.test_nonlinear_comparison import checked
from tests.test_scaling_recovery import response
from tests.test_scaling_recovery import small as small_fixture

small = small_fixture


def test_fixed_alternative_uses_reserved_blocks_not_an_extra_budget(
    small, monkeypatch, tmp_path
):
    problem, starts = small
    seen = []

    def search(o, start, *args, **kw):
        seen.append((start, kw["optimizer_scaling"], kw["calls"]))
        return response(start, 0.5, progress=False)

    monkeypatch.setattr(recovery.numerical, "search", search)
    result = adapter.run(problem, starts, adapter.AdapterPolicy(), tmp_path)
    backend = read_seal(tmp_path / "backend.json")
    assert [m for _, m, _ in seen] == [
        "guarded_relative",
        "fixed_coordinates",
        "guarded_relative",
        "fixed_coordinates",
        "guarded_relative",
        "guarded_relative",
    ]
    assert [s["a"] for s, _, _ in seen] == [1, 0.5, 3, 0.5, 4, 5]
    assert result.cost.blocks_used == 6 and result.cost.maximum_search_calls == 240
    assert all(c == 40 for _, _, c in seen)
    assert result.cost.search_calls_started == 60
    assert result.parameters["a"] == 0.5
    assert not result.verification.target_reached
    assert result.assessment.search_status == "further_search_needed"
    assert [d["reason"] for d in backend["decisions"]].count(
        "bounded_fixed_alternative"
    ) == 2


def test_good_progress_keeps_guard_and_guard_only_never_switches(
    small, monkeypatch, tmp_path
):
    p, starts = small
    seen = []

    def improving(o, s, *args, **kw):
        seen.append(kw["optimizer_scaling"])
        return response(s, 0.1)

    monkeypatch.setattr(recovery.numerical, "search", improving)
    result = adapter.run(p, starts, adapter.AdapterPolicy(), tmp_path / "progress")
    assert result.cost.blocks_used == 6
    assert set(seen) == {"guarded_relative"}
    seen.clear()

    def flat(o, s, *args, **kw):
        seen.append(kw["optimizer_scaling"])
        return response(s, 0.1, progress=False)

    monkeypatch.setattr(recovery.numerical, "search", flat)
    guarded = adapter.run(
        p, starts, adapter.AdapterPolicy(strategy="guarded_only"), tmp_path / "guarded"
    )
    assert guarded.cost.blocks_used == 4 and set(seen) == {"guarded_relative"}


def test_distinct_incumbent_survives_failed_search_and_resume(
    small, monkeypatch, tmp_path
):
    p, starts = small
    p = p.model_copy(update={"incumbent": {"a": 0.01, "initial": 0.0}})
    monkeypatch.setattr(
        recovery.numerical,
        "search",
        lambda o, s, *a, **k: response(s, 0.5, progress=False),
    )
    result = adapter.run(p, starts, adapter.AdapterPolicy(), tmp_path)
    assert result.parameters == p.incumbent
    assert result.selected_origin == "incumbent/check"
    monkeypatch.setattr(recovery.rollout, "build", lambda _: pytest.fail("resume fit"))
    assert adapter.run(p, starts, adapter.AdapterPolicy(), tmp_path) == result
    with pytest.raises(ValueError, match="identity"):
        adapter.run(p, starts, adapter.AdapterPolicy(strategy="guarded_only"), tmp_path)


def test_unverified_branch_diversifies_without_fixed_retry(
    small, monkeypatch, tmp_path
):
    p, starts = small
    seen = []

    def verify(o, s, *a, **kw):
        if s == p.start:
            return checked(s, 1.0)
        raise ValueError("independent integrators failed")

    def search(o, s, *a, **kw):
        seen.append(kw["optimizer_scaling"])
        return response(s, 0.01, progress=False)

    monkeypatch.setattr(recovery.rollout, "verify", verify)
    monkeypatch.setattr(recovery.numerical, "search", search)
    result = adapter.run(p, starts, adapter.AdapterPolicy(), tmp_path)
    assert seen == ["guarded_relative"] * 4
    assert result.parameters == p.start
    assert not result.verification.target_reached


def test_target_checks_worst_trajectory_and_keeps_uncertainty_separate(
    small, monkeypatch, tmp_path
):
    p, starts = small
    monkeypatch.setattr(
        recovery.rollout, "verify", lambda o, s, *a, **kw: checked(s, 1e-14, worst=1)
    )
    monkeypatch.setattr(
        recovery.numerical,
        "search",
        lambda o, s, *a, **kw: response(s, 0.1, progress=False),
    )
    result = adapter.run(p, starts, adapter.AdapterPolicy(), tmp_path)
    assert not result.verification.target_reached
    assert result.assessment.coefficient_accuracy_available_to_fitter is False
    assert result.assessment.global_identifiability_certified is False
    assert result.assessment.probability is None
    assert result.verification.worst_trajectory_nmse == 1


def test_interrupted_portfolio_preserves_charge_and_never_reexecutes_operation(
    small, monkeypatch, tmp_path
):
    p, starts = small
    monkeypatch.setattr(
        recovery.numerical,
        "search",
        lambda o, s, *a, **kw: response(s, 0.1, progress=False),
    )
    policy = adapter.AdapterPolicy()
    adapter.run(p, starts, policy, tmp_path / "source")
    relative = "search/start-0/block-0/rollout/started.json"
    seal(tmp_path / "interrupted" / relative, read_seal(tmp_path / "source" / relative))
    seen = []

    def search(o, s, *a, **kw):
        seen.append(s)
        return response(s, 0.1, progress=False)

    monkeypatch.setattr(recovery.numerical, "search", search)
    result = adapter.run(p, starts, policy, tmp_path / "interrupted")
    assert p.start not in seen
    assert result.cost.search_budget_charge_seconds >= policy.search.block_seconds
    assert not result.cost.accounting_complete
    assert result.cost.search_calls_started is None


def test_typed_training_boundary_rejects_validation_and_reference_fields(small):
    p, starts = small
    for forbidden in ("validation", "test", "reference_parameters"):
        payload = p.model_dump(mode="json") | {forbidden: {"leak": 1}}
        with pytest.raises(ValueError):
            type(p).model_validate(payload)
    with pytest.raises(TypeError, match="typed training"):
        adapter.run(p.model_dump(), starts, adapter.AdapterPolicy(), None)


def test_partial_adapter_cannot_resume_with_changed_source(
    small, monkeypatch, tmp_path
):
    p, starts = small
    monkeypatch.setattr(
        recovery.rollout, "verify", lambda o, s, *a, **kw: checked(s, 1e-15)
    )
    first = adapter.run(p, starts, adapter.AdapterPolicy(), tmp_path / "complete")
    partial = tmp_path / "partial"
    seal(partial / "identity.json", first.identity)
    monkeypatch.setattr(adapter.public, "_source_identity", lambda: "changed-code")
    monkeypatch.setattr(
        recovery.rollout, "build", lambda _: pytest.fail("source reused")
    )
    with pytest.raises(ValueError, match="identity"):
        adapter.run(p, starts, adapter.AdapterPolicy(), partial)


def test_legacy_guarded_protocol_and_new_adapter_paths_are_identical_without_fallback(
    small, monkeypatch, tmp_path
):
    p, starts = small
    seen = []

    def search(o, s, *a, **kw):
        seen.append(deepcopy((s, kw)))
        return response(s, 0.1)

    monkeypatch.setattr(recovery.numerical, "search", search)
    old = recovery.run(
        p, starts, recovery.Policy(), "guarded_relative", tmp_path / "old"
    )
    new = adapter.run(
        p, starts, adapter.AdapterPolicy(strategy="guarded_only"), tmp_path / "new"
    )
    assert [s for s, _ in seen[:6]] == [s for s, _ in seen[6:]]
    assert new.parameters == old["retained_parameters"]
    assert old["identity"]["protocol"] == "generic-scaling-recovery-1"
