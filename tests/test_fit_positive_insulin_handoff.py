"""Check stage order, training-only retention and deterministic resume."""

import pytest

from scripts import fit_positive_insulin_handoff as run


def test_displaced_start_preserves_fixed_law_and_moves_shared_downstream_terms():
    model = {"formula": {"equations": {"I": "-I/tau+gain*Gt"}}}
    params = {"tau": 8, "gain": 2, "downstream": 3}
    fixed = run.optimizer_start(model, params, "fixed_insulin", "displaced")
    other = run.optimizer_start(
        {"formula": {"equations": {"I": "-rho*I"}}},
        {"rho": 0.5, "downstream": 3},
        "fixed_insulin",
        "displaced",
    )
    assert (fixed["tau"], fixed["gain"]) == (8, 2)
    assert fixed["downstream"] == other["downstream"] != params["downstream"]
    moving = run.optimizer_start(model, params, "joint", "displaced")
    assert 0 < moving["tau"] != params["tau"]


def test_incomplete_event_cannot_win():
    with pytest.raises(ArithmeticError):
        run.qualified_score({"summary": {"pooled_nmse": 0.0, "failed_events": 1}})


def test_initial_verification_failure_blocks_fitting_and_development(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(run, "design", lambda *args: {})
    calls = []

    def fail(tag, **kwargs):
        assert kwargs["starts_only"]
        calls.append(tag)
        raise ValueError("changed supplied start")

    monkeypatch.setattr(run, "train_one", fail)
    monkeypatch.setattr(run.intake, "load_validation", lambda *a: pytest.fail("dev"))
    with pytest.raises(ValueError, match="changed supplied start"):
        run.run(tmp_path, tmp_path, tmp_path, tmp_path, 2, 1)
    assert calls
    assert not (tmp_path / "training-freeze.json").exists()


def test_stage_qualification_and_resume_preserve_incumbent(tmp_path, monkeypatch):
    tag = "002-restoring"
    model = {"formula": {"parameters": {"p": 1.0}, "equations": {"I": "-I"}}}
    monkeypatch.setattr(run.intake, "load_models", lambda _: {tag: model})
    monkeypatch.setattr(run.intake, "load_training", lambda _: [])
    monkeypatch.setattr(run.intake, "verify_start", lambda *a: {"verified": True})
    calls = []

    class Problem:
        def __init__(self, *a):
            pass

        def fit_stage(self, stage, p, *, max_nfev):
            calls.append((stage, max_nfev))
            return {
                "parameters": {"p": 4 if stage == "fixed_insulin" else 2},
                "training_nmse": 0.01,
                "diagnostics": {},
            }

    monkeypatch.setattr(run.kernel, "FitProblem", Problem)

    def replay(model, rows, **kw):
        p = model["formula"]["parameters"]["p"]
        return {"summary": {"pooled_nmse": 1 / p, "failed_events": p == 4}}

    monkeypatch.setattr(run, "cache_replay", replay)
    args = {
        "archive": tmp_path,
        "training": tmp_path,
        "root": tmp_path,
        "plan": {"artifact_sha256": "plan", "max_nfev_per_start": 2},
    }
    start = run.train_one(tag, starts_only=True, **args)
    assert calls == []
    first = run.train_one(tag, starts_only=False, **args)
    assert first["fixed_insulin"]["parameters"] == start["start"]["parameters"]
    assert first["joint"]["parameters"] == {"p": 2}
    assert first["joint"]["diagnostics"]["selected"] == "warm"
    assert run.train_one(tag, starts_only=False, **args) == first
    assert calls == [(s, 2) for s in run.STAGES for _ in run.STARTS]
    args["plan"]["artifact_sha256"] = "other"
    with pytest.raises(ValueError, match="different fit"):
        run.train_one(tag, starts_only=False, **args)
