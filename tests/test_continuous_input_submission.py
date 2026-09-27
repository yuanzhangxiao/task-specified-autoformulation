"""CPU launcher receipts must survive ambiguous Slurm replies without duplication."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("confirmed", [True, False])
@pytest.mark.parametrize("mode", ["audit", "qualify", "refine"])
def test_launcher_records_reply_and_never_blindly_resubmits(tmp_path, confirmed, mode):
    repo = tmp_path / "repo"
    script = repo / "scripts/hpc/submit_continuous_input_audit.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(Path("scripts/hpc") / script.name, script)
    (repo / "SOURCE_COMMIT").write_text("a" * 40)
    for name in (
        "benchmark5_anonymous_nonlinear_process/private/system_specification.json",
        "benchmark6_alien_device/private/selected_system_spec.json",
    ):
        path = repo / "data_raw" / name
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    binary = tmp_path / "bin"
    binary.mkdir()
    counter = tmp_path / "calls"
    scheduler = binary / "sbatch"
    scheduler.write_text(
        '#!/bin/bash\nprintf "called\\n" >> "$AF_TEST_CALLS"\n'
        + ('echo "12345"\n' if confirmed else 'echo "Socket timed out" >&2\n')
    )
    scheduler.chmod(0o755)
    output = tmp_path / "results"
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "summary.json").write_text("{}")
    (audit / "tasks.json").write_text("{}")
    env = dict(
        os.environ,
        AF_PYTHON=sys.executable,
        AF_OUTPUT_ROOT=str(output),
        AF_TEST_CALLS=str(counter),
        AF_REFERENCE_MODE=mode,
        AF_AUDIT_ROOT=str(audit),
        AF_QUALIFICATION_ROOT=str(audit),
        PATH=f"{binary}:{os.environ['PATH']}",
    )
    first = subprocess.run(["bash", str(script), "delta"], env=env, capture_output=True)
    second = subprocess.run(
        ["bash", str(script), "delta"], env=env, capture_output=True
    )
    assert first.returncode == second.returncode == (0 if confirmed else 2)
    assert counter.read_text().splitlines() == ["called"]
    assert (output / "submission/stdout.txt").exists()
    assert (output / "submission/stderr.txt").exists()
    assert (output / "submission/job.id").exists() is confirmed
    command = (output / "submission/command.txt").read_text()
    assert "--cpus-per-task=1" in command and "--partition=cpu" in command
    assert "--gres" not in command
    assert ("--array=0-13%2" in command) is (mode == "refine")
