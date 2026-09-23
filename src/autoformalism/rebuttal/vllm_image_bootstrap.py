"""Build one persistent vLLM SIF from a recorded OCI digest, without GPU work."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write

VERSION = "0.27.1"
REPOSITORY = "vllm/vllm-openai"
TAG = f"v{VERSION}"
MEDIA_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
)


def image_hash(path: Path) -> str:
    """Hash SIF bytes without loading the entire image into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin_image(directory: Path) -> dict:
    """Resolve the public version tag once; retries always use the saved digest."""
    path = directory / "oci_source.json"
    if path.exists():
        value = sealed_read(path)
        if value["repository"] != REPOSITORY or value["tag"] != TAG:
            raise ValueError("saved OCI source differs")
        return value
    token_url = (
        "https://auth.docker.io/token?service=registry.docker.io"
        f"&scope=repository:{REPOSITORY}:pull"
    )
    with urlopen(token_url, timeout=60) as response:
        token = json.load(response)["token"]
    request = Request(
        f"https://registry-1.docker.io/v2/{REPOSITORY}/manifests/{TAG}",
        headers={"Authorization": f"Bearer {token}", "Accept": ", ".join(MEDIA_TYPES)},
    )
    with urlopen(request, timeout=60) as response:
        data = response.read()
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        if response.headers.get("Docker-Content-Digest", digest) != digest:
            raise ValueError("OCI manifest digest differs")
    if json.loads(data).get("mediaType") not in MEDIA_TYPES:
        raise ValueError("unsupported OCI manifest format")
    return sealed_write(
        path,
        {
            "repository": REPOSITORY,
            "tag": TAG,
            "digest": digest,
            "uri": f"docker://{REPOSITORY}@{digest}",
            "architecture": "amd64",
        },
    )


def check_version(runtime: str, image: Path) -> None:
    """Read installed package metadata without importing the GPU serving stack."""
    value = subprocess.run(
        [
            runtime,
            "exec",
            str(image),
            "python3",
            "-c",
            "from importlib.metadata import version; print(version('vllm'))",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.strip()
    if value != VERSION:
        raise ValueError(f"requires vLLM {VERSION}, image reports {value!r}")


def prepare(image: Path, scratch: Path) -> dict:
    """Validate before atomic publication; reuse completed or publication-ready work."""
    image = image.absolute()
    if image.is_symlink():
        raise ValueError("bootstrap output must not be a symlink")
    directory = image.with_name(image.name + ".build")
    runtime = shutil.which("apptainer") or shutil.which("singularity")
    if not runtime:
        raise ValueError("apptainer or singularity is required on the CPU build node")
    with public._lock(directory):
        candidate = directory / "candidate.sif"
        ready_path, result_path = (
            directory / "validated.json",
            directory / "result.json",
        )
        if result_path.exists() or (image.exists() and ready_path.exists()):
            value = sealed_read(result_path if result_path.exists() else ready_path)
            if not image.is_file() or image_hash(image) != value["image_sha256"]:
                raise ValueError("published SIF differs from its bootstrap receipt")
        else:
            if image.exists():
                raise ValueError("image already exists; submit without --prepare-image")
            source = pin_image(directory)
            if ready_path.exists():
                value = sealed_read(ready_path)
                if (
                    value["oci_source_sha256"] != source["artifact_sha256"]
                    or not candidate.is_file()
                    or image_hash(candidate) != value["image_sha256"]
                ):
                    raise ValueError("validated SIF staging artifact differs")
            else:
                scratch.mkdir(parents=True, exist_ok=True)
                if shutil.disk_usage(scratch).free < 40 * 1024**3:
                    raise ValueError(
                        "image build scratch requires at least 40 GiB free"
                    )
                with tempfile.TemporaryDirectory(
                    prefix="vllm-image-", dir=scratch
                ) as tmp:
                    environment = {
                        **os.environ,
                        "APPTAINER_TMPDIR": tmp,
                        "APPTAINER_CACHEDIR": str(Path(tmp) / "cache"),
                        "SINGULARITY_TMPDIR": tmp,
                        "SINGULARITY_CACHEDIR": str(Path(tmp) / "cache"),
                    }
                    # Only this helper's unpublished candidate can be replaced.
                    subprocess.run(
                        [
                            runtime,
                            "pull",
                            "--force",
                            "--disable-cache",
                            "--arch",
                            "amd64",
                            str(candidate),
                            source["uri"],
                        ],
                        env=environment,
                        check=True,
                        timeout=6600,
                    )
                check_version(runtime, candidate)
                value = sealed_write(
                    ready_path,
                    {
                        "protocol": "vllm-sif-bootstrap-1",
                        "vllm_version": VERSION,
                        "image_sha256": image_hash(candidate),
                        "oci_source_sha256": source["artifact_sha256"],
                        "oci_uri": source["uri"],
                        "gpu_calls": 0,
                    },
                )
            # A same-filesystem hard link publishes atomically without overwriting.
            os.link(candidate, image)
        result = sealed_write(
            result_path, {k: v for k, v in value.items() if k != "artifact_sha256"}
        )
        if candidate.exists() and os.path.samefile(candidate, image):
            candidate.unlink()
        return result
