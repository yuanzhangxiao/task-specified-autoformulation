#!/usr/bin/env python3
"""Run one command and write portable child-process resource timing."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import tempfile
import time
from pathlib import Path


def run_with_timing(
    command: list[str],
    output: Path,
    *,
    output_format: str = "key_value",
) -> int:
    """Execute ``command``, atomically record usage, and return its exit code."""
    if not command:
        raise ValueError("timed command is empty")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    completed = subprocess.run(command, check=False)
    elapsed = time.monotonic() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    payload = {
        "schema_version": "portable-child-process-timing-1",
        "elapsed_seconds": elapsed,
        "user_cpu_seconds": max(0.0, after.ru_utime - before.ru_utime),
        "system_cpu_seconds": max(0.0, after.ru_stime - before.ru_stime),
        "max_rss_kib": int(after.ru_maxrss),
        "exit_code": completed.returncode,
    }
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    elif output_format == "key_value":
        text = "".join(f"{key}={value}\n" for key, value in payload.items())
    else:  # pragma: no cover - argparse enforces this for CLI use.
        raise ValueError(f"unsupported timing output format: {output_format}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(destination)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return completed.returncode


def main() -> None:
    """Parse a command after ``--`` and preserve its return code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--output-format",
        choices=("key_value", "json"),
        default="key_value",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    raise SystemExit(
        run_with_timing(command, args.output, output_format=args.output_format)
    )


if __name__ == "__main__":
    main()
