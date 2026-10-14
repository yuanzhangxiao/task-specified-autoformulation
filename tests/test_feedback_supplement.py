"""Immutable source reuse, incremental accounting and fixed-point continuation."""

from itertools import count
from time import monotonic

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import feedback_supplement as supplement
from autoformalism.fitting import staged_feedback as feedback
from autoformalism.fitting import staged_feedback_fixtures as fixtures


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    """Parallel implementation work must not alter a tiny fixture's identity."""
    monkeypatch.setattr(feedback.public, "_source_identity", lambda: "test-source")


@pytest.fixture
def partial(tmp_path, monkeypatch):
    """A real fitted control with one successful and one failed perturbed call."""
    saved = fixtures.fit_fixture("piecewise_inactive_guard", tmp_path / "fit")
    problem = feedback.training_problem(saved["problem"])
    fit = feedback.adapter.FitResult.model_validate(saved["fit"])
    source = tmp_path / "original"
    evaluate = feedback.evaluate_stage
    ticks = count(monotonic(), 0.5)

    def fail(oracle, vector, deadline, checkpoint, method, sensitivities, **kw):
        if method == "Radau" and kw.get("metadata_policy") is None:
            checkpoint({"calls": 1, "calls_completed": 0, "failure_kind": "timeout"})
            raise TimeoutError("original finite allowance exhausted")
        return evaluate(
            oracle, vector, deadline, checkpoint, method, sensitivities, **kw
        )

    with monkeypatch.context() as patch:
        patch.setattr(feedback.checks, "monotonic", lambda: next(ticks))
        patch.setattr(feedback, "evaluate_stage", fail)
        packet = feedback.run(
            problem,
            fit,
            saved["backend"],
            source,
            feedback.Policy(perturbation_seconds=1, perturbation_total_seconds=1),
        )
    assert packet.cost.calls == 5 and packet.cost.calls_completed == 4
    assert packet.diagnostics.piecewise_assessment.status == "unavailable"
    return source, packet


def _bytes(folder):
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in folder.rglob("*")
        if path.is_file()
    }


def _replace(path, value):
    path.unlink()
    seal(path, value)


def test_partial_source_reuses_exact_calls_and_preserves_failed_attempt(
    partial, tmp_path, monkeypatch
):
    source, original = partial
    before = _bytes(source)
    output = tmp_path / "supplement"
    plan = supplement.prepare(source, output)
    assert len(plan.reused_operations) == 4
    assert len(plan.missing_keys) == 15
    result = supplement.run(source, output)
    assert result.packet.diagnosis == "accurate_local_evidence"
    assert result.original_packet == original
    assert result.original_cost == original.cost
    assert result.new_calls_started == result.new_calls_completed == 15
    assert result.new_cost.optimization_calls == 0
    assert len(result.ledger) == 20
    assert result.combined_cost.calls == 20
    assert result.combined_cost.calls_completed == 19
    assert result.combined_cost.budget_charge_seconds == pytest.approx(
        original.cost.budget_charge_seconds + result.new_cost.budget_charge_seconds
    )
    assert result.inherited_cost == original.cost
    historical_failure = [r for r in result.ledger if r.stage.status != "complete"]
    assert len(historical_failure) == 1
    failure = historical_failure[0]
    assert failure.stage.status == "unavailable"
    assert (
        failure.identity
        == read_seal(source / "stages" / failure.key / "record.json")["identity"]
    )
    retries = [r for r in result.attempted_operations if r.key == failure.key]
    assert len(retries) == 1
    assert retries[0].identity != failure.identity
    assert retries[0].identity["point_sha256"] == failure.identity["point_sha256"]
    for row in result.reused_operations:
        operation = read_seal(source / "stages" / row.key / "record.json")
        assert feedback.public.content_sha256(operation) == row.operation_sha256
        assert operation["identity"] == row.identity
    assert not result.remaining_keys
    assert result.packet.identity == original.identity
    assert _bytes(source) == before
    monkeypatch.setattr(
        feedback, "evaluate_stage", lambda *a, **kw: pytest.fail("repeated solve")
    )
    assert supplement.run(source, output) == result
    assert supplement.completed(output) == result
    assert _bytes(source) == before


def test_completed_source_is_noop_and_sequential_ledger_stays_complete(
    partial, tmp_path, monkeypatch
):
    source, original = partial
    first = tmp_path / "first"
    result = supplement.run(source, first)
    before = _bytes(first)
    monkeypatch.setattr(
        feedback, "evaluate_stage", lambda *a, **kw: pytest.fail("unneeded solve")
    )
    second = supplement.run(first, tmp_path / "second")
    assert second.required_calls == 0
    assert second.new_cost.calls == second.new_cost.calls_completed == 0
    assert second.new_cost.budget_charge_seconds == 0
    assert second.ledger == result.ledger
    assert second.original_packet == original
    assert second.predecessor_packet == result.packet
    assert second.combined_cost == result.combined_cost
    assert supplement.completed(tmp_path / "second") == second
    assert _bytes(first) == before


def test_interrupted_new_attempt_is_charged_once_and_budget_not_reset(
    partial, tmp_path, monkeypatch
):
    source, original = partial
    output = tmp_path / "interrupted"
    policy = supplement.Policy(extra_seconds=1, per_call_seconds=1)

    def interrupt(*args, **kwargs):
        args[3]({"calls": 1, "calls_completed": 0})
        raise KeyboardInterrupt

    monkeypatch.setattr(feedback, "evaluate_stage", interrupt)
    with pytest.raises(KeyboardInterrupt):
        supplement.run(source, output, policy)
    monkeypatch.setattr(
        feedback, "evaluate_stage", lambda *a, **kw: pytest.fail("restarted attempt")
    )
    result = supplement.run(source, output, policy)
    assert len(result.attempted_operations) == 1
    assert result.attempted_operations[0].stage.status == "interrupted"
    assert result.new_cost.budget_charge_seconds == 1
    assert result.new_cost.calls is None
    assert not result.new_cost.accounting_complete
    assert result.combined_cost.budget_charge_seconds == (
        original.cost.budget_charge_seconds + 1
    )
    assert result.additional_budget_exhausted
    assert result.remaining_keys
    before = _bytes(output)
    assert supplement.run(source, output, policy) == result
    assert supplement.completed(output) == result
    assert _bytes(output) == before
    with pytest.raises(ValueError, match="sealed artifact differs"):
        supplement.run(source, output, supplement.Policy(extra_seconds=2))


def test_sequential_campaign_can_explicitly_retry_an_interrupted_new_attempt(
    partial, tmp_path, monkeypatch
):
    source, _ = partial
    first = tmp_path / "first"
    evaluate = feedback.evaluate_stage
    with monkeypatch.context() as patch:
        patch.setattr(
            feedback,
            "evaluate_stage",
            lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            supplement.run(
                source, first, supplement.Policy(extra_seconds=1, per_call_seconds=1)
            )
    previous = supplement.run(
        source, first, supplement.Policy(extra_seconds=1, per_call_seconds=1)
    )
    monkeypatch.setattr(feedback, "evaluate_stage", evaluate)
    result = supplement.run(first, tmp_path / "next")
    assert result.packet.diagnosis == "accurate_local_evidence"
    assert result.inherited_cost == previous.combined_cost
    assert result.ledger[: len(previous.ledger)] == previous.ledger
    assert result.new_calls_completed == previous.required_calls
    assert result.combined_cost.calls is None  # The interrupted cost stays unknown.
    assert not result.combined_cost.accounting_complete


def test_source_mutation_is_rejected(partial, tmp_path):
    source, _ = partial
    output = tmp_path / "supplement"
    supplement.prepare(source, output)
    path = source / "stages/primary/record.json"
    operation = read_seal(path)
    operation["value"]["residual"][0] += 1
    _replace(path, operation)
    with pytest.raises(ValueError):
        supplement.run(source, output)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("point_sha256", "another-vector"),
        ("solver", "DOP853"),
        ("allowance_seconds", 999),
    ],
)
def test_resealed_new_stage_identity_is_rejected(partial, tmp_path, field, value):
    source, _ = partial
    output = tmp_path / "supplement"
    result = supplement.run(source, output)
    key = result.attempted_operations[0].key
    path = output / "stages" / key
    operation = read_seal(path / "record.json")
    operation["identity"][field] = value
    _replace(path / "record.json", operation)
    _replace(
        path / "result.json", {k: v for k, v in operation.items() if k != "progress"}
    )
    _replace(path / "started.json", operation["identity"])
    manifest = read_seal(output / "stages.json")
    manifest[0][1] = feedback.public.content_sha256(operation)
    _replace(output / "stages.json", manifest)
    with pytest.raises(ValueError, match="identity"):
        supplement.completed(output)


def test_frozen_plan_and_source_cannot_be_changed_in_place(partial, tmp_path):
    source, _ = partial
    output = tmp_path / "supplement"
    supplement.prepare(source, output)
    path = output / "supplement.json"
    raw = read_seal(path)
    raw["perturbation_plan"][0]["vector"][0] += 0.1
    _replace(path, raw)
    with pytest.raises(ValueError, match="sealed artifact differs"):
        supplement.run(source, output)


def test_different_candidate_cannot_resume_same_supplement(partial, tmp_path):
    source, _ = partial
    output = tmp_path / "supplement"
    supplement.prepare(source, output)
    saved = fixtures.fit_fixture("piecewise_kink", tmp_path / "other_fit")
    other = tmp_path / "other"
    feedback.run(
        feedback.training_problem(saved["problem"]),
        feedback.adapter.FitResult.model_validate(saved["fit"]),
        saved["backend"],
        other,
    )
    with pytest.raises(ValueError, match="sealed artifact differs"):
        supplement.run(other, output)


def test_source_directory_cannot_be_used_as_output(partial):
    source, _ = partial
    before = _bytes(source)
    with pytest.raises(ValueError, match="separate directories"):
        supplement.prepare(source, source / "supplement")
    assert _bytes(source) == before


@pytest.mark.parametrize("kwargs", [{"extra_seconds": 0}, {"per_call_seconds": 1801}])
def test_incremental_allowance_is_bounded(kwargs):
    with pytest.raises(ValueError):
        supplement.Policy(**kwargs)


def test_archived_supplement_can_be_relocated_with_source(partial, tmp_path):
    """Relative source references survive copying the complete review bundle."""
    from shutil import copytree

    source, _ = partial
    bundle = tmp_path / "bundle"
    copied_source = bundle / "source_snapshot"
    copytree(source, copied_source)
    result = supplement.run(copied_source, bundle / "results" / "cstr")
    plan = read_seal(bundle / "results" / "cstr" / "supplement.json")
    assert plan["source_path"] == "../../source_snapshot"
    relocated = tmp_path / "relocated"
    bundle.rename(relocated)
    assert supplement.completed(relocated / "results" / "cstr") == result


def test_omitted_policy_resumes_frozen_custom_allowance(partial, tmp_path):
    source, _ = partial
    output = tmp_path / "supplement"
    plan = supplement.prepare(
        source, output, supplement.Policy(extra_seconds=17, per_call_seconds=2)
    )
    result = supplement.run(source, output)
    assert result.identity == plan.identity
    assert supplement.completed(output) == result


def test_resume_keeps_completed_new_stage_before_an_interruption(
    partial, tmp_path, monkeypatch
):
    source, _ = partial
    output = tmp_path / "supplement"
    policy = supplement.Policy(extra_seconds=1, per_call_seconds=1)
    evaluate = feedback.evaluate_stage
    ticks = count(monotonic(), 0.5)
    calls = []

    def interrupt_second(*args, **kwargs):
        calls.append(args[4])
        if len(calls) == 2:
            args[3]({"calls": 1, "calls_completed": 0})
            raise KeyboardInterrupt
        return evaluate(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(feedback.checks, "monotonic", lambda: next(ticks))
        patch.setattr(feedback, "evaluate_stage", interrupt_second)
        with pytest.raises(KeyboardInterrupt):
            supplement.run(source, output, policy)
    completed_paths = list((output / "stages").rglob("record.json"))
    assert len(completed_paths) == 1
    saved = completed_paths[0].read_bytes()
    monkeypatch.setattr(
        feedback, "evaluate_stage", lambda *a, **kw: pytest.fail("repeated attempt")
    )
    result = supplement.run(source, output)
    assert [r.stage.status for r in result.attempted_operations] == [
        "complete",
        "interrupted",
    ]
    assert result.new_cost.budget_charge_seconds == 1
    assert result.attempted_operations[1].stage.allowance_seconds == 0.5
    assert completed_paths[0].read_bytes() == saved
    assert supplement.completed(output) == result


def test_unreported_started_attempt_cannot_disappear_from_ledger(partial, tmp_path):
    source, _ = partial
    output = tmp_path / "supplement"
    supplement.run(source, output)
    seal(output / "stages/perturb/undeclared/primary/started.json", {"attempt": True})
    with pytest.raises(ValueError, match="omits a started attempt"):
        supplement.completed(output)


@pytest.mark.parametrize(
    "change",
    [
        {"status": "unavailable", "error": "fake failure"},
        {"calls": 0},
        {"value": None},
        {"budget_restarted": True},
        {"wall_seconds": 0},
    ],
)
def test_resealed_completion_or_cost_contradiction_is_rejected(
    partial, tmp_path, change
):
    source, _ = partial
    output = tmp_path / "supplement"
    result = supplement.run(source, output)
    path = output / "stages" / result.attempted_operations[0].key
    operation = read_seal(path / "record.json")
    operation.update(change)
    _replace(path / "record.json", operation)
    _replace(
        path / "result.json", {k: v for k, v in operation.items() if k != "progress"}
    )
    manifest = read_seal(output / "stages.json")
    manifest[0][1] = feedback.public.content_sha256(operation)
    _replace(output / "stages.json", manifest)
    with pytest.raises(ValueError, match=r"evidence|cost|budget"):
        supplement.completed(output)
