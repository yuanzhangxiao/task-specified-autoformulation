"""Run a frozen nine-request stage on one offline 20B endpoint."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

MODEL = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
CASES = ("basin", "reactor", "insulin")
SEEDS = (100, 101, 102)


def digest(value: object) -> str:
    """Hash JSON consistently on the CPU and provider hosts."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def save(path: Path, obj: dict) -> None:
    """Write an atomic receipt on the same filesystem."""
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_stage(path: Path) -> dict:
    """Check the request manifest before reserving provider calls."""
    plan = json.loads(path.read_text())
    requests = plan["requests"]
    if (
        plan["stage"] not in ("generate", "rank")
        or plan["sha256"] != digest(requests)
        or set(requests) != {f"{case}-{seed}" for case in CASES for seed in SEEDS}
    ):
        raise ValueError("Invalid stage manifest")
    for label, body in requests.items():
        if (
            body["model"] != MODEL
            or body["seed"] != int(label.split("-")[1])
            or body["max_tokens"] != 1536
            or body.get("n", 1) != 1
        ):
            raise ValueError("Unexpected serving or budget settings")
    return plan


def run(root: Path, stage: str, endpoint: str) -> None:
    """Reserve each call once, persisting ambiguous interruptions across resumes."""
    if not endpoint.startswith("http://127.0.0.1:"):
        raise ValueError("Expected the local offline endpoint")
    plan = load_stage(root / f"{stage}.json")
    if stage != plan["stage"]:
        raise ValueError("Stage differs")
    directory = root / f"{stage}-responses"
    directory.mkdir(exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for label, body in plan["requests"].items():
            identity = digest(body)
            path = directory / f"{label}.json"
            if path.exists():
                if json.loads(path.read_text())["request_sha256"] != identity:
                    raise ValueError("Stored request differs")
                continue
            receipt = {"request_sha256": identity, "status": "reserved"}
            save(path, receipt)
            start = time.monotonic()
            req = Request(
                endpoint + "/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(req, timeout=240) as stream:
                    response = json.load(stream)
                receipt.update(status="responded", response=response)
            except (OSError, ValueError) as exc:
                receipt.update(status="provider_error", error=str(exc))
                if hasattr(exc, "read"):
                    receipt["details"] = exc.read(2048).decode(errors="replace")
            receipt["seconds"] = time.monotonic() - start
            save(path, receipt)
            print(stage, label, receipt["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", choices=("generate", "rank"))
    parser.add_argument("endpoint")
    args = parser.parse_args()
    run(args.root, args.stage, args.endpoint)
