"""Exercise the portable archive writer and pilot submission preflight."""

import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def copy_launcher(tmp_path: Path, name: str) -> Path:
    path = tmp_path / "scripts" / "hpc" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / "scripts" / "hpc" / name, path)
    return path


@pytest.mark.parametrize("override", [None, "custom-gate"])
def test_gate_defaults_to_known_cases_and_preserves_overrides(tmp_path, override):
    launcher = copy_launcher(tmp_path, "iclr_aces.sh")
    (tmp_path / "SOURCE_COMMIT").write_text("1" * 40)
    for name, body in (("module", "exit 0"), ("squeue", 'echo "$AF_JUDGE_GATE_ROOT"')):
        stub = tmp_path / name
        stub.write_text(f"#!/bin/sh\n{body}\n")
        stub.chmod(0o755)
    env = {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "AF_GROUP": str(tmp_path),
        "USER": "tester",
    }
    if override:
        env["AF_JUDGE_GATE_ROOT"] = str(tmp_path / override)
    result = subprocess.check_output(
        ["bash", str(launcher), "status"], env=env, text=True
    )
    assert result.strip() == str(
        tmp_path / (override or "judge-jetstream-known-cases-v1")
    )


def test_pilot_missing_authorization_stops_before_credential_prompt(tmp_path):
    launcher = copy_launcher(tmp_path, "iclr_aces.sh")
    (tmp_path / "SOURCE_COMMIT").write_text("1" * 40)
    module = tmp_path / "module"
    module.write_text("#!/bin/sh\nexit 0\n")
    module.chmod(0o755)
    result = subprocess.run(
        ["bash", str(launcher), "submit", "pilot", "wave1"],
        env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "AF_GROUP": str(tmp_path)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        start_new_session=True,
    )
    assert result.returncode != 0
    assert "Missing critic authorization for pilot" in result.stderr
    assert "/dev/tty" not in result.stderr
    assert not list(tmp_path.rglob("array.intent.json"))


def test_public_delivery_has_portable_metadata_and_valid_checksums(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    for name in ("iclr-nonbaseline-public.zip", "iclr-nonbaseline-evidence.zip"):
        source = REPO / "output" / name
        if not source.exists():
            pytest.skip("needs the pinned release archives")
        (output / name).symlink_to(source)
    vocab = (
        REPO
        / "output/insulin-revision-delivery-20261105/bundle-smoke"
        / "phase-c-insulin-revision-v1/assets/harmony"
    )
    if not (vocab / "o200k_base.tiktoken").is_file() or not (REPO / ".git").exists():
        pytest.skip("needs the pinned offline vocabulary and source checkout")

    launcher = copy_launcher(tmp_path, "package_iclr_aces.sh")
    git_dir = subprocess.check_output(
        ["git", "rev-parse", "--absolute-git-dir"], cwd=REPO, text=True
    ).strip()
    subprocess.run(
        ["bash", str(launcher)],
        env=dict(os.environ, GIT_DIR=git_dir, AF_VOCAB_DIR=str(vocab)),
        check=True,
        capture_output=True,
    )

    hashes = {}
    declared = {}
    with tarfile.open(output / "iclr-nonbaseline-aces.tar.gz", "r|gz") as archive:
        for member in archive:
            assert not member.pax_headers
            if not member.isfile():
                continue
            name = member.name.removeprefix("iclr-nonbaseline-aces/")
            raw = archive.extractfile(member).read()
            if name == "SHA256SUMS":
                for line in raw.decode().splitlines():
                    digest, filename = line.split("  ", 1)
                    declared[filename] = digest
            else:
                hashes[name] = hashlib.sha256(raw).hexdigest()
    assert hashes == declared
    assert "code/scripts/hpc/iclr_aces.sh" in hashes
