#!/usr/bin/env python3
"""Validate only the owned Docker server configuration and record its provenance."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts.jetstream_construction_probe import IMAGE, MODEL, REVISION, validate_server


def server_record(inspect: dict, work: Path) -> dict:
    """Reject a reused container with different settings, storage, or exposure."""
    config = inspect["Config"]
    expected = [
        "--model",
        MODEL,
        "--revision",
        REVISION,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        "1",
    ]
    if config["Image"] != IMAGE or config["Cmd"] != expected:
        raise ValueError("existing container differs from the frozen serving command")
    if inspect["HostConfig"]["PortBindings"] != {
        "8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8000"}]
    }:
        raise ValueError("server must be exposed on loopback only")
    mounts = {m["Destination"]: m for m in inspect["Mounts"]}
    if (
        mounts["/af/hf"]["Source"] != str(work / "hf")
        or mounts["/af/hf"]["RW"]
        or mounts["/af/cache"]["Source"] != str(work / "cache")
    ):
        raise ValueError("server mounts differ")
    if not {"HF_HOME=/af/hf", "HF_HUB_OFFLINE=1"}.issubset(config["Env"]):
        raise ValueError("server must use verified offline weights")
    weights = json.loads((work / "weights.json").read_text())
    if weights["repo"] != MODEL or weights["revision"] != REVISION:
        raise ValueError("weights revision differs")
    value = {
        "platform": "jetstream2-h100x1-docker",
        "image": IMAGE,
        "image_id": inspect["Image"],
        "model": MODEL,
        "revision": REVISION,
        "context_tokens": 32768,
        "max_num_seqs": 1,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "cuda": json.loads((work / "cuda.json").read_text()),
        "weights_manifest_sha256": hashlib.sha256(
            (work / "weights.json").read_bytes()
        ).hexdigest(),
    }
    validate_server(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    if args.image != IMAGE:
        parser.error("image differs")
    value = server_record(json.load(sys.stdin)[0], args.work.resolve())
    value["gpu_driver"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    (args.work / "server.json").write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
