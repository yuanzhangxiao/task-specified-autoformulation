"""Standalone, resumable twelve-call GPT-OSS-20B worker for ACES."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

if __package__:
    from scripts.revision_round_provider import digest, save
else:
    from revision_round_io import digest, save

MODEL = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
EXPECTED = 12
MAX_TOKENS = 2304
SEED = 101
PROTOCOL = "matched-equation-hybrid-20b-1"
LABEL = re.compile(r"[\w-]{1,96}", re.ASCII)


def load_manifest(root: Path) -> dict:
    """Bind every reserved call to the immutable GPU plan and serving contract."""
    value = json.loads((root / "generate.json").read_text())
    requests = value["requests"]
    if (
        value["protocol"] != PROTOCOL
        or value["sha256"] != digest(requests)
        or value["revision"] != REVISION
        or len(requests) != EXPECTED
        or any(not LABEL.fullmatch(label) for label in requests)
    ):
        raise ValueError("Invalid twelve-panel hybrid manifest")
    for body in requests.values():
        if (
            body.get("model") != MODEL
            or body.get("max_tokens") != MAX_TOKENS
            or body.get("seed") != SEED
            or body.get("n", 1) != 1
            or body["response_format"]["type"] != "json_schema"
            or body["response_format"]["json_schema"]["strict"] is not True
        ):
            raise ValueError("Unexpected model, sample count, budget or schema")
    return value


def perform(endpoint: str, body: dict) -> dict:
    """Submit one structured request to a pinned localhost server."""
    request = Request(
        endpoint + "/v1/chat/completions",
        data=json.dumps(body, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=240) as response:
        return json.load(response)


def run(root: Path, endpoint: str, transport=perform) -> None:
    """Never repeat an acknowledged or previously reserved generation request."""
    if not re.fullmatch(r"http://127\.0\.0\.1:[0-9]{1,5}", endpoint):
        raise ValueError("Expected a loopback serving endpoint")
    manifest = load_manifest(root)
    folder = root / "generate-responses"
    folder.mkdir(exist_ok=True)
    with (folder / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for label, body in manifest["requests"].items():
            path = folder / f"{label}.json"
            identity = {
                "request_sha256": digest(body),
                "manifest_sha256": manifest["sha256"],
                "revision": REVISION,
            }
            if path.exists():
                previous = json.loads(path.read_text())
                if not all(previous[k] == v for k, v in identity.items()):
                    raise ValueError(f"Wrong receipt for {label}")
                continue
            save(path, {**identity, "status": "reserved"})
            start = time.monotonic()
            try:
                response = transport(endpoint, body)
                outcome = {
                    "status": "responded",
                    "response": response,
                    "response_sha256": digest(response),
                }
            except (OSError, ValueError) as exc:
                outcome = {"status": "provider_error", "error": str(exc)}
                if hasattr(exc, "read"):
                    outcome["details"] = exc.read(1500).decode(errors="replace")
            save(path, {**identity, **outcome, "seconds": time.monotonic() - start})
            print(label, outcome["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", choices=("generate",))
    parser.add_argument("endpoint")
    args = parser.parse_args()
    for filename, expected in load_manifest(args.root)["worker_sha256"].items():
        if hashlib.sha256((args.root / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Uploaded worker differs: {filename}")
    run(args.root, args.endpoint)
