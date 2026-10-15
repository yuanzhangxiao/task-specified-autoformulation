"""Standalone evidence-bound advice export without hidden numerical execution."""

import json
from pathlib import Path

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import interaction_feedback as feedback
from autoformalism.fitting import staged_feedback as staged
from autoformalism.fitting import staged_feedback_fixtures as fixtures
from autoformalism.fitting.rollout_adapter import FitResult
from scripts import phase_c_interaction_feedback as cli


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """Generate a small real fitted control, verified diagnostics and M37 packet."""
    root = tmp_path_factory.mktemp("interaction-advice-cli")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fixtures.public, "_source_identity", lambda: "advice-cli-source")
        values = fixtures.fit_fixture("stationary_poor_retained", root / "fixture")
        staged.run(
            staged.training_problem(values["problem"]),
            FitResult.model_validate(values["fit"]),
            values["backend"],
            root / "diagnostics",
            staged.Policy(
                primary_seconds=5,
                alternate_seconds=5,
                sensitivity_seconds=5,
                perturbation_seconds=2,
                perturbation_total_seconds=10,
            ),
        )
        feedback.export_packet(root / "diagnostics", root / "packet")
    return root


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    """Concurrent unrelated edits must not change identities inside one test."""
    monkeypatch.setattr(
        fixtures.public, "_source_identity", lambda: "advice-cli-source"
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _run(monkeypatch, capsys, command: str, source: Path, output: Path) -> dict:
    monkeypatch.setattr(
        "sys.argv",
        [
            "phase_c_interaction_feedback.py",
            command,
            "--diagnostics",
            str(source / "diagnostics"),
            "--packet",
            str(source / "packet"),
            "--output",
            str(output),
        ],
    )
    cli.main()
    return json.loads(capsys.readouterr().out)


def test_draft_and_verify_use_explicit_paths_without_new_numerical_work(
    source, tmp_path, monkeypatch, capsys
):
    before = _snapshot(source)

    def unexpected(*args, **kwargs):
        pytest.fail("advice export attempted new fitting or integration")

    monkeypatch.setattr(fixtures, "fit_fixture", unexpected)
    monkeypatch.setattr(staged, "run", unexpected)
    monkeypatch.setattr(staged, "evaluate_stage", unexpected)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "feedback draft"
    draft = _run(monkeypatch, capsys, "draft", source, out)
    saved = _snapshot(out)
    assert {"result.json", "request.json", "feedback.md"} <= set(saved)
    assert set(saved) <= {"result.json", "request.json", "feedback.md", ".lock"}
    assert read_seal(out / "result.json") == draft
    assert (out / "feedback.md").read_text().strip()
    assert _run(monkeypatch, capsys, "verify-draft", source, out) == draft
    assert _run(monkeypatch, capsys, "draft", source, out) == draft
    assert _snapshot(out) == saved
    assert _snapshot(source) == before


def test_verify_draft_does_not_create_missing_output(
    source, tmp_path, monkeypatch, capsys
):
    out = tmp_path / "absent"
    with pytest.raises((FileNotFoundError, ValueError)):
        _run(monkeypatch, capsys, "verify-draft", source, out)
    assert not out.exists()


def test_missing_source_is_rejected_without_partial_advice(
    tmp_path, monkeypatch, capsys
):
    out = tmp_path / "out"
    with pytest.raises((FileNotFoundError, ValueError)):
        _run(monkeypatch, capsys, "draft", tmp_path / "missing", out)
    assert not out.exists()


def test_resealed_m37_packet_is_rejected_before_advice(
    source, tmp_path, monkeypatch, capsys
):
    import shutil

    copied = tmp_path / "copy"
    shutil.copytree(source, copied)
    path = copied / "packet/result.json"
    packet = read_seal(path)
    packet["feedback"]["diagnosis"] = "accurate_local_evidence"
    path.unlink()
    seal(path, packet)
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        _run(monkeypatch, capsys, "draft", copied, out)
    assert not out.exists()


@pytest.mark.parametrize("missing", ["--diagnostics", "--packet", "--output"])
def test_draft_requires_each_explicit_source_path(monkeypatch, missing):
    args = ["phase_c_interaction_feedback.py", "draft"]
    for option in ("--diagnostics", "--packet", "--output"):
        if option != missing:
            args.extend((option, "unused"))
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2
