"""Exercise the Mac staging helper with real rsync and local SSH stand-ins."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/hpc/stage_review_deadline_public.sh"


@pytest.fixture
def staging(tmp_path):
    """Map the two remote roots to local fixtures without a network connection."""
    rsync = shutil.which("rsync")
    if not rsync or not shutil.which("sha256sum"):
        pytest.skip("rsync and sha256sum required for transfer integration test")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    config = json.loads((ROOT / "configs/review_deadline_v1.json").read_text())
    names = [
        f"phase_b_v1/{cell}/{name}"
        for cell in config["public_cells"]
        for name in (
            "manifest.json",
            "proposer_prompt.txt",
            "train.csv",
            "validation.csv",
        )
    ]
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + "\n")
    (source / "phase_b_v1" / config["public_cells"][0] / "test.csv").write_text(
        "must not be copied"
    )
    programs = {
        "ssh": """import subprocess, sys
raise SystemExit(subprocess.run(['bash', '-c', sys.argv[-1]]).returncode)
""",
        "rsync": """import os, re, subprocess, sys
from pathlib import Path
args = [a.split(':', 1)[1] if a.startswith(('aces:', 'delta:')) else a
        for a in sys.argv[1:]]
result = subprocess.run([os.environ['TEST_REAL_RSYNC'], *args],
                        capture_output=True, text=True,
                        env={**os.environ, 'PATH': os.environ['TEST_REAL_PATH']})
# A local rsync copy reports receiving; the actual ACES copy reports sending.
output = re.sub(r'^[<>]f', '<f', result.stdout, flags=re.MULTILINE)
output = re.sub(r'<f[+]+', '<f' + '+' * int(os.environ['TEST_PLUS_COUNT']), output)
print(output, end='')
print(result.stderr, end='', file=sys.stderr)
if '--ignore-existing' in args and os.environ.get('TEST_CORRUPT_COPY'):
    next(Path(args[-1]).rglob('manifest.json')).write_text('corrupt')
raise SystemExit(result.returncode)
""",
    }
    for name, program in programs.items():
        path = binaries / name
        path.write_text(f"#!{sys.executable}\n" + program)
        path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "AF_REPO_ROOT": str(ROOT),
        "AF_LOCAL_PYTHON": sys.executable,
        "AF_DELTA_PUBLIC_ROOT": str(source),
        "AF_PUBLIC_ROOT": str(destination),
        "TEST_REAL_RSYNC": rsync,
        "TEST_REAL_PATH": os.environ["PATH"],
        "TEST_PLUS_COUNT": "7",
    }
    # The production helper deliberately restricts remote paths to simple names.
    assert " " not in str(tmp_path)
    return env, source, destination, names


def run(env):
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30
    )


@pytest.mark.parametrize("plus_count", [7, 9])
def test_new_release_and_resume_with_both_rsync_formats(staging, plus_count):
    env, source, destination, names = staging
    env["TEST_PLUS_COUNT"] = str(plus_count)
    for _ in range(2):
        result = run(env)
        assert result.returncode == 0, result.stderr
        assert "Verified 24 development files on ACES." in result.stdout
        assert sorted(
            str(p.relative_to(destination))
            for p in destination.rglob("*")
            if p.is_file()
        ) == sorted(names)
        assert all(
            (destination / n).read_bytes() == (source / n).read_bytes() for n in names
        )


def test_existing_different_content_is_not_overwritten(staging):
    env, _, destination, names = staging
    path = destination / names[0]
    path.parent.mkdir(parents=True)
    path.write_text("existing different bytes")
    result = run(env)
    assert result.returncode != 0
    assert "Existing ACES development files differ" in result.stderr
    assert path.read_text() == "existing different bytes"


def test_corrupted_copy_cannot_report_success(staging):
    env, _, _, _ = staging
    env["TEST_CORRUPT_COPY"] = "1"
    result = run(env)
    assert result.returncode != 0
    assert "Verified 24" not in result.stdout


def test_missing_source_file_cannot_report_success(staging):
    env, source, destination, names = staging
    (source / names[0]).unlink()
    result = run(env)
    assert result.returncode != 0
    assert "Verified 24" not in result.stdout
    assert not destination.exists()
