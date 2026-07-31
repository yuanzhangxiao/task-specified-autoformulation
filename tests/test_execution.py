"""User-facing dry-run, mock execution, resume, and summary tests."""

from __future__ import annotations

from pathlib import Path

from autoformalism.execution import ExecutionArguments, execute


def _arguments(tmp_path: Path, *, dry_run: bool, resume: bool = False):
    return ExecutionArguments(
        data_root=tmp_path,
        benchmark_id="synthetic",
        tier="easy",
        seed=31,
        proposer_model=None,
        judge_model=None,
        iteration_budget=1,
        beam_size=1,
        output_root=tmp_path / "runs",
        resume=resume,
        dry_run=dry_run,
        mock_llm=True,
        use_clean_observations=False,
    )


def test_dry_run_creates_no_output_and_exposes_no_test_fingerprint(
    tmp_path: Path,
) -> None:
    result = execute(_arguments(tmp_path, dry_run=True))

    assert result["status"] == "dry_run"
    assert set(result["split_fingerprints"]) == {"train", "validation"}
    assert not (tmp_path / "runs").exists()


def test_mock_execution_and_resume_are_idempotent(tmp_path: Path) -> None:
    first = execute(_arguments(tmp_path, dry_run=False))
    resumed = execute(_arguments(tmp_path, dry_run=False, resume=True))

    assert first == resumed
    assert first["status"] == "complete"
    assert first["test_failed_trajectories"] == []
