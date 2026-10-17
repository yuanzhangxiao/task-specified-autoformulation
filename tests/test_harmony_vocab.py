"""Offline serving must render Harmony before queueing or spending model calls."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import harmony_vocab as vocab

REPO = Path(__file__).resolve().parents[1]
ASSET = b"verified test vocabulary\n"


@pytest.fixture
def asset(tmp_path, monkeypatch):
    monkeypatch.setattr(vocab, "VOCAB_SHA256", hashlib.sha256(ASSET).hexdigest())
    directory = tmp_path / "assets"
    directory.mkdir()
    (directory / vocab.VOCAB_FILENAME).write_bytes(ASSET)
    return directory


def test_verify_rejects_missing_and_corrupt_assets_without_network(asset, monkeypatch):
    monkeypatch.setattr(vocab, "urlopen", lambda *a, **k: pytest.fail("network"))
    assert vocab.verify(asset)["sha256"] == vocab.VOCAB_SHA256
    path = asset / vocab.VOCAB_FILENAME
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        vocab.verify(asset)
    path.unlink()
    with pytest.raises(ValueError, match="Missing bundled Harmony vocabulary"):
        vocab.verify(asset)


def test_download_verifies_before_publishing_and_reuses_existing_asset(
    asset, tmp_path, monkeypatch
):
    calls = []

    def fetch(url, *, timeout):
        calls.append((url, timeout))
        return io.BytesIO(ASSET)

    monkeypatch.setattr(vocab, "urlopen", fetch)
    destination = tmp_path / "download"
    assert vocab.download(destination) == vocab.verify(asset)
    assert vocab.download(destination) == vocab.verify(asset)
    assert calls == [(vocab.VOCAB_URL, 60)]
    assert (destination / vocab.VOCAB_FILENAME).read_bytes() == ASSET


@pytest.mark.parametrize("failure", ["hash", "size", "network"])
def test_failed_download_never_leaves_an_asset_or_temporary_file(
    asset, tmp_path, monkeypatch, failure
):
    destination = tmp_path / "download"
    if failure == "size":
        monkeypatch.setattr(vocab, "MAX_DOWNLOAD_BYTES", 1)

    def fetch(*args, **kwargs):
        if failure == "network":
            raise OSError("offline")
        return io.BytesIO(b"unverified bytes")

    monkeypatch.setattr(vocab, "urlopen", fetch)
    with pytest.raises((ValueError, OSError)):
        vocab.download(destination)
    assert list(destination.iterdir()) == []


def test_explicit_download_does_not_silently_replace_corrupt_existing_asset(
    asset, monkeypatch
):
    path = asset / vocab.VOCAB_FILENAME
    path.write_bytes(b"existing corrupt asset")
    monkeypatch.setattr(vocab, "urlopen", lambda *a, **k: pytest.fail("network"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        vocab.download(asset)
    assert path.read_bytes() == b"existing corrupt asset"


def fake_harmony(monkeypatch, tokens):
    calls = []

    def load(name):
        calls.append(("load", name))

        def render(conversation, role):
            calls.append(("render", conversation, role))
            return tokens

        return SimpleNamespace(render_conversation_for_completion=render)

    harmony = SimpleNamespace(
        HarmonyEncodingName=SimpleNamespace(HARMONY_GPT_OSS="gpt-oss"),
        Role=SimpleNamespace(USER="user", ASSISTANT="assistant"),
        Message=SimpleNamespace(from_role_and_content=lambda *args: args),
        Conversation=SimpleNamespace(from_messages=lambda messages: messages),
        load_harmony_encoding=load,
    )
    monkeypatch.setattr(vocab, "import_module", lambda name: harmony)
    monkeypatch.setattr(vocab, "version", lambda name: "test-harmony")
    monkeypatch.setattr(vocab, "urlopen", lambda *a, **k: pytest.fail("network"))
    return calls


def test_cpu_probe_loads_and_renders_without_generation(asset, monkeypatch):
    monkeypatch.setenv("TIKTOKEN_ENCODINGS_BASE", str(asset))
    calls = fake_harmony(monkeypatch, [1, 2, 3])
    result = vocab.probe(asset)
    assert result == {
        **vocab.verify(asset),
        "openai_harmony": "test-harmony",
        "rendered_tokens": 3,
        "harmony_render_passed": True,
        "generation_requests": 0,
    }
    assert calls == [
        ("load", "gpt-oss"),
        ("render", [("user", "Hello.")], "assistant"),
    ]


@pytest.mark.parametrize("configured", [None, "/incorrect-directory"])
def test_probe_rejects_implicit_or_different_cache_before_import(
    asset, monkeypatch, configured
):
    monkeypatch.delenv("TIKTOKEN_ENCODINGS_BASE", raising=False)
    if configured is not None:
        monkeypatch.setenv("TIKTOKEN_ENCODINGS_BASE", configured)
    monkeypatch.setattr(vocab, "import_module", lambda name: pytest.fail("import"))
    with pytest.raises(ValueError, match="TIKTOKEN_ENCODINGS_BASE"):
        vocab.probe(asset)


@pytest.mark.parametrize("tokens", [[], [True], ["not a token"]])
def test_probe_rejects_failed_render(asset, monkeypatch, tokens):
    monkeypatch.setenv("TIKTOKEN_ENCODINGS_BASE", str(asset))
    fake_harmony(monkeypatch, tokens)
    with pytest.raises(RuntimeError, match="CPU render preflight failed"):
        vocab.probe(asset)


def test_standalone_cli_has_clear_missing_asset_error(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(REPO / "scripts/harmony_vocab.py"),
            "verify",
            "--directory",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Missing bundled Harmony vocabulary" in result.stderr
    assert "Traceback" not in result.stderr


def launcher_fixture(tmp_path, failure):
    """Stub infrastructure while executing the launcher's real asset verification."""
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    hpc = scripts / "hpc"
    hpc.mkdir(parents=True)
    for name in ("start", "run"):
        filename = f"{name}_saved_initialization_aces.sh"
        shutil.copyfile(REPO / "scripts/hpc" / filename, hpc / filename)
    source = (REPO / "scripts/harmony_vocab.py").read_text()
    (scripts / "harmony_vocab.py").write_text(
        source.replace(vocab.VOCAB_SHA256, hashlib.sha256(ASSET).hexdigest())
    )
    assets = repo / "assets/harmony"
    assets.mkdir(parents=True)
    if failure != "missing":
        (assets / vocab.VOCAB_FILENAME).write_bytes(
            b"corrupt" if failure == "corrupt" else ASSET
        )
    worker = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "with Path(os.environ['AF_TEST_CALLS']).open('a') as stream:\n"
        "    stream.write(Path(sys.argv[0]).stem + ' ' + sys.argv[1] + '\\n')\n"
    )
    for filename in (
        "saved_initialization_aces.py",
        "submit_saved_initialization_aces.py",
    ):
        (scripts / filename).write_text(worker)
    commands = tmp_path / "bin"
    commands.mkdir()
    container = commands / "apptainer"
    container.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['AF_TEST_CONTAINER_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "if 'probe' in args:\n"
        "    if os.environ['AF_TEST_FAILURE'] == 'probe':\n"
        "        print('Harmony render failed', file=sys.stderr)\n"
        "        sys.exit(37)\n"
        "    print(json.dumps({'harmony_render_passed': True}))\n"
        "elif 'python3' in args:\n"
        "    print(json.dumps({'cuda_kernel_passed': True}))\n"
    )
    container.chmod(0o755)
    curl = commands / "curl"
    curl.write_text(
        f"#!{sys.executable}\n"
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "for _ in range(100):\n"
        "    path = Path(os.environ['AF_TEST_CONTAINER_CALLS'])\n"
        "    if path.exists() and 'vllm' in path.read_text():\n"
        "        sys.exit(0)\n"
        "    time.sleep(0.01)\n"
        "sys.exit(1)\n"
    )
    curl.chmod(0o755)
    python = commands / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "if sys.argv[1] == '-c' and sys.argv[2].startswith('import socket;'):\n"
        "    print('8000')\n"
        "else:\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    python.chmod(0o755)
    image = tmp_path / "image.sif"
    image.write_bytes(b"pinned image fixture")
    hf = tmp_path / "hf"
    snapshot = hf / (
        "hub/models--openai--gpt-oss-20b/snapshots/"
        "6cee5e81ee83917806bbde320786a8fb61efebee/config.json"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("{}")
    output = tmp_path / "output"
    output.mkdir()
    (output / "execution.json").write_text(
        json.dumps({"image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
    )
    env = {
        **os.environ,
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "AF_REPO_ROOT": str(repo),
        "AF_GROUP": str(tmp_path),
        "AF_PYTHON": str(python),
        "AF_VLLM_IMAGE": str(image),
        "AF_HF_HOME": str(hf),
        "AF_OUTPUT_ROOT": str(output),
        "AF_TEST_CALLS": str(tmp_path / "calls.log"),
        "AF_TEST_CONTAINER_CALLS": str(tmp_path / "containers.jsonl"),
        "AF_TEST_FAILURE": failure,
        "SLURM_JOB_ID": "offline-harmony-test",
    }
    env.pop("AF_HARMONY_ASSETS", None)
    return repo, env


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("failure", ["missing", "corrupt", "probe", "none"])
def test_launchers_gate_submission_and_server_on_offline_render(
    tmp_path, worker, failure
):
    repo, env = launcher_fixture(tmp_path, failure)
    name = "run" if worker else "start"
    result = subprocess.run(
        [
            "bash",
            str(repo / f"scripts/hpc/{name}_saved_initialization_aces.sh"),
            "propose" if worker else "run",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    calls = Path(env["AF_TEST_CALLS"]).read_text()
    container_log = Path(env["AF_TEST_CONTAINER_CALLS"])
    containers = (
        [json.loads(line) for line in container_log.read_text().splitlines()]
        if container_log.exists()
        else []
    )
    if failure in {"missing", "corrupt"}:
        assert result.returncode == 2, result.stderr
        assert "Harmony vocabulary" in result.stderr
        assert not containers
    else:
        assert "probe" in containers[0]
        assert "--nv" not in containers[0]
        for args in containers:
            assert f"TIKTOKEN_ENCODINGS_BASE={repo}/assets/harmony" in args
            assert f"{repo}/assets/harmony:{repo}/assets/harmony:ro" in args[
                args.index("--bind") + 1
            ]
        assert result.returncode == (37 if failure == "probe" else 0), result.stderr
    if failure != "none":
        assert " prepare\n" not in calls
        assert " run\n" not in calls
        assert "submit_saved_initialization_aces" not in calls
        assert len(containers) <= 1
    elif worker:
        assert "saved_initialization_aces run\n" in calls
        assert len(containers) == 3
        assert all("--nv" in args for args in containers[1:])
        runtime = json.loads(
            (Path(env["AF_OUTPUT_ROOT"]) / "runtime/probe-offline-harmony-test.json")
            .read_text()
        )
        assert runtime["harmony"]["harmony_render_passed"] is True
    else:
        assert "saved_initialization_aces prepare\n" in calls
        assert "submit_saved_initialization_aces --root\n" in calls
        assert len(containers) == 1
