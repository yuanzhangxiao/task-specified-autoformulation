"""Offline OCI pinning, validated publication and interrupted build recovery."""

import hashlib
import io
import json
import subprocess
from types import SimpleNamespace

import pytest

from autoformalism.rebuttal import vllm_image_bootstrap as bootstrap
from autoformalism.rebuttal.prefit_replay import sealed_read


def mock_registry(monkeypatch, *, bad_digest=False):
    """Return a public token and a small OCI index; no network or real credentials."""
    calls = []
    manifest = json.dumps({"mediaType": bootstrap.MEDIA_TYPES[0]}).encode()

    def request(url, **kwargs):
        calls.append(url)
        if isinstance(url, str):
            response = io.BytesIO(b'{"token":"synthetic-public-token"}')
        else:
            assert url.headers["Authorization"] == "Bearer synthetic-public-token"
            response = io.BytesIO(manifest)
        response.headers = {
            "Docker-Content-Digest": "sha256:"
            + ("0" * 64 if bad_digest else hashlib.sha256(manifest).hexdigest())
        }
        return response

    monkeypatch.setattr(bootstrap, "urlopen", request)
    return calls


def mock_runtime(monkeypatch, version=bootstrap.VERSION):
    """Exercise real filesystem publication with a fake Apptainer subprocess."""
    calls = []
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _: "/mock/apptainer")
    monkeypatch.setattr(
        bootstrap.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3)
    )

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "pull":
            from pathlib import Path

            Path(argv[-2]).write_bytes(b"synthetic-sif-bytes")
            assert "@sha256:" in argv[-1]
            assert "--disable-cache" in argv
            assert kwargs["env"]["APPTAINER_TMPDIR"]
        else:
            assert argv[1] == "exec" and "--nv" not in argv
        return subprocess.CompletedProcess(argv, 0, stdout=version + "\n")

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    return calls


def test_resolves_once_and_does_not_save_registry_token(tmp_path, monkeypatch):
    calls = mock_registry(monkeypatch)
    value = bootstrap.pin_image(tmp_path)
    assert bootstrap.pin_image(tmp_path) == value
    assert len(calls) == 2
    assert "token" not in (tmp_path / "oci_source.json").read_text()
    assert "@sha256:" in value["uri"]


def test_registry_digest_mismatch(tmp_path, monkeypatch):
    mock_registry(monkeypatch, bad_digest=True)
    with pytest.raises(ValueError, match="manifest digest"):
        bootstrap.pin_image(tmp_path)
    assert not (tmp_path / "oci_source.json").exists()


def test_build_and_resume(tmp_path, monkeypatch):
    network = mock_registry(monkeypatch)
    calls = mock_runtime(monkeypatch)
    image, scratch = tmp_path / "vllm.sif", tmp_path / "scratch"
    result = bootstrap.prepare(image, scratch)
    assert image.read_bytes() == b"synthetic-sif-bytes"
    assert result["gpu_calls"] == 0
    assert bootstrap.prepare(image, scratch) == result
    assert len(calls) == 2 and len(network) == 2
    directory = image.with_name(image.name + ".build")
    assert not (directory / "candidate.sif").exists()
    assert sealed_read(directory / "result.json") == result
    assert not list(scratch.iterdir())


def test_wrong_version_not_published(tmp_path, monkeypatch):
    mock_registry(monkeypatch)
    mock_runtime(monkeypatch, version="0.26.0")
    image = tmp_path / "vllm.sif"
    with pytest.raises(ValueError, match=r"requires vLLM 0\.27\.1"):
        bootstrap.prepare(image, tmp_path / "scratch")
    assert not image.exists()
    assert not (tmp_path / "vllm.sif.build/validated.json").exists()


@pytest.mark.parametrize("stage", ["publish", "receipt"])
def test_resume_publication_without_new_download(tmp_path, monkeypatch, stage):
    network = mock_registry(monkeypatch)
    calls = mock_runtime(monkeypatch)
    image, scratch = tmp_path / "vllm.sif", tmp_path / "scratch"
    link, write = bootstrap.os.link, bootstrap.sealed_write

    def crash(*args, **kwargs):
        if stage == "publish" or args[0].name == "result.json":
            raise OSError("simulated interruption")
        return write(*args, **kwargs)

    if stage == "publish":
        monkeypatch.setattr(bootstrap.os, "link", crash)
    else:
        monkeypatch.setattr(bootstrap, "sealed_write", crash)
    with pytest.raises(OSError, match="interruption"):
        bootstrap.prepare(image, scratch)
    monkeypatch.setattr(bootstrap.os, "link", link)
    monkeypatch.setattr(bootstrap, "sealed_write", write)
    result = bootstrap.prepare(image, scratch)
    assert result["image_sha256"] == bootstrap.image_hash(image)
    assert len(network) == len(calls) == 2


def test_existing_foreign_image_is_preserved(tmp_path, monkeypatch):
    calls = mock_runtime(monkeypatch)
    image = tmp_path / "ollama.sif"
    image.write_bytes(b"existing image")
    with pytest.raises(ValueError, match="already exists"):
        bootstrap.prepare(image, tmp_path / "scratch")
    assert image.read_bytes() == b"existing image" and not calls


def test_published_image_corruption_not_overwritten(tmp_path, monkeypatch):
    mock_registry(monkeypatch)
    calls = mock_runtime(monkeypatch)
    image, scratch = tmp_path / "vllm.sif", tmp_path / "scratch"
    bootstrap.prepare(image, scratch)
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="published SIF differs"):
        bootstrap.prepare(image, scratch)
    assert image.read_bytes() == b"changed" and len(calls) == 2
