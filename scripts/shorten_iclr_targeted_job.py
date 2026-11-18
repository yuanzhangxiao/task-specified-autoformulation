"""Shorten a verified pending targeted array without cancellation or new calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

PROTOCOL = "iclr-targeted-short-allocation-1"
PATCHES = (
    "scripts/iclr_controlled_search.py",
    "scripts/iclr_allocation_window.py",
)
LIMIT = "04:00:00"


def digest(value: bytes) -> str:
    """Hash deployment bytes without loading any candidate code."""
    return hashlib.sha256(value).hexdigest()


def atomic(path: Path, value: bytes) -> None:
    """Install a complete file in one filesystem rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".short-allocation-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def command(argv: list[str]) -> str:
    """Run only the explicitly assembled scheduler command."""
    result = subprocess.run(
        argv, check=True, capture_output=True, text=True, timeout=45
    )
    return result.stdout


def manifest(folder: Path, originals: dict[str, bytes] | None = None) -> dict[str, str]:
    """Verify every manifest entry, allowing only receipted original CLI bytes."""
    entries = {}
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or name in entries:
            raise ValueError("unsafe or duplicate bundle manifest entry")
        target = folder / path
        if target.is_symlink() or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("invalid bundle manifest entry")
        value = (originals or {}).get(name)
        if value is None:
            value = target.read_bytes()
        if digest(value) != expected:
            raise ValueError(f"bundle hash mismatch: {target}")
        entries[name] = expected
    commit = (folder / "SOURCE_COMMIT").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("invalid source commit")
    return entries


def states(job: str, run) -> dict[str, str]:
    """Expand the array so a running member cannot hide behind pending members."""
    output = run(["squeue", "--array", "-h", "-j", job, "-o", "%i|%T"])
    rows = [line.strip().split("|") for line in output.splitlines() if line.strip()]
    if any(len(row) != 2 for row in rows):
        raise ValueError("unrecognized scheduler state reply")
    answer = dict(rows)
    if len(rows) != 6 or set(answer) != {f"{job}_{i}" for i in range(6)}:
        raise ValueError("expected exactly six live array elements 0-5")
    return answer


def details(job: str, root: Path, code: Path, run, *, held=False, shortened=False):
    """Authenticate scheduler identity without inspecting exported credentials."""
    output = run(["scontrol", "-o", "show", "job", job])
    rows = [
        dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", line))
        for line in output.splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("no scheduler job identity returned")
    for row in rows:
        user = row.get("UserId", "")
        uid = re.fullmatch(r"[^()]+\((\d+)\)", user)
        excluded = row.get("ExcNodeList", "").replace("ac[042,096]", "ac042,ac096")
        if (
            not uid
            or int(uid[1]) != os.getuid()
            or row.get("ArrayJobId", row.get("JobId")) != job
            or row.get("JobName") != "iclr-targeted-20b"
            or row.get("Account") != "156264627414"
            or row.get("Partition") != "gpu"
            or row.get("Command")
            != str(code / "scripts/hpc/run_iclr_controlled_search_aces.sh")
            or row.get("WorkDir") != str(code)
            or not {"ac042", "ac096"}.issubset(set(excluded.split(",")))
            or any(
                not row.get(key, "").startswith(str(root / "logs/wave2-"))
                for key in ("StdOut", "StdErr")
            )
        ):
            raise ValueError("scheduler identity differs from the targeted wave")
        if held and (
            row.get("JobState") != "PENDING" or row.get("Reason") != "JobHeldUser"
        ):
            raise ValueError("every element must remain pending under our user hold")
        if shortened and row.get("TimeLimit") != LIMIT:
            raise ValueError("scheduler did not confirm the four-hour limit")
    return rows


def update(
    root: Path, job: str, code: Path, replacement: Path, *, apply=False, run=command
):
    """Deploy the deadline adapter, then shorten and release this exact array."""
    root, code, replacement = (p.resolve() for p in (root, code, replacement))
    if not job.isdecimal() or code == replacement:
        raise ValueError("use a numeric job and a distinct replacement bundle")
    wave = root / "submissions/wave2"
    if (wave / "array.id").read_text().strip() != job:
        raise ValueError("job differs from the recorded wave2 submission")
    argv = json.loads((wave / "array.intent.json").read_text())["argv"]
    required = [
        "--account=156264627414",
        "--partition=gpu",
        "--job-name=iclr-targeted-20b",
        "--exclude=ac042,ac096",
        "--array=0,1,2,3,4,5%6",
        f"--output={root}/logs/wave2-%A_%a.out",
        f"--error={root}/logs/wave2-%A_%a.err",
        str(code / "scripts/hpc/run_iclr_controlled_search_aces.sh"),
    ]
    if not all(value in argv for value in required):
        raise ValueError("submission intent differs from the expected targeted wave")
    folder = wave / f"short-allocation-{job}"
    receipt_path = folder / "deployment.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    originals = {}
    if receipt:
        if any(
            receipt.get(k) != v
            for k, v in {
                "protocol": PROTOCOL,
                "job": job,
                "root": str(root),
                "code": str(code),
                "replacement": str(replacement),
            }.items()
        ):
            raise ValueError("existing deployment belongs to different inputs")
        original = (folder / "original-cli.py").read_bytes()
        if digest(original) != receipt["original_cli_sha256"]:
            raise ValueError("saved original CLI changed")
        originals[PATCHES[0]] = original
    old, new = manifest(code, originals), manifest(replacement)

    def scientific(entries):
        return {
            k: v
            for k, v in entries.items()
            if k.startswith(("src/", "configs/", "public/"))
        }

    if scientific(old) != scientific(new):
        raise ValueError(
            "replacement changes scientific source, configuration or inputs"
        )
    payloads = {name: (replacement / name).read_bytes() for name in PATCHES}
    hashes = {name: digest(value) for name, value in payloads.items()}
    if not set(PATCHES).issubset(new):
        raise ValueError("replacement manifest lacks deadline adapter")
    if receipt:
        if receipt["replacement_sha256"] != hashes:
            raise ValueError("replacement deployment changed")
        for name in PATCHES:
            target = code / name
            allowed = {hashes[name], old.get(name)}
            if target.exists() and digest(target.read_bytes()) not in allowed:
                raise ValueError("unrecognized modification in deployed adapter")
        if receipt["phase"] == "complete":
            if any(digest((code / n).read_bytes()) != h for n, h in hashes.items()):
                raise ValueError("completed deployment files changed")
            return {
                "status": "already_applied",
                "job": job,
                "receipt": str(receipt_path),
            }
    elif (code / PATCHES[1]).exists():
        raise ValueError("unreceipted deadline helper already exists")
    before = details(job, root, code, run)

    def same_submission(rows):
        original = receipt["before"] if receipt else before
        expected = {r["SubmitTime"] for r in original if "SubmitTime" in r}
        actual = {r["SubmitTime"] for r in rows if "SubmitTime" in r}
        if expected and expected != actual:
            raise ValueError("scheduler submission time changed")

    same_submission(before)
    if not receipt and any(row.get("Reason") == "JobHeldUser" for row in before):
        raise ValueError("job already has an unrelated user hold; no change made")
    if (
        receipt
        and receipt["phase"] in {"verified", "release_requested"}
        and not any(row.get("Reason") == "JobHeldUser" for row in before)
    ):
        details(job, root, code, run, shortened=True)
        if any(digest((code / n).read_bytes()) != h for n, h in hashes.items()):
            raise ValueError("released deployment files changed")
        if apply:
            receipt["phase"] = "complete"
            atomic(receipt_path, (json.dumps(receipt, indent=2) + "\n").encode())
        return {
            "status": "already_applied",
            "job": job,
            "receipt": str(receipt_path),
            "release_reconciled": True,
        }
    if set(states(job, run).values()) != {"PENDING"}:
        raise ValueError("array has started; leave its running allocation unchanged")
    if not apply:
        return {
            "status": "ready",
            "job": job,
            "time_limit": LIMIT,
            "receipt": str(receipt_path),
            "apply": False,
        }
    if receipt is None:
        atomic(folder / "original-cli.py", (code / PATCHES[0]).read_bytes())
        receipt = {
            "protocol": PROTOCOL,
            "job": job,
            "root": str(root),
            "code": str(code),
            "replacement": str(replacement),
            "phase": "prepared",
            "before": before,
            "original_cli_sha256": old[PATCHES[0]],
            "replacement_sha256": hashes,
            "original_source_commit": (code / "SOURCE_COMMIT").read_text().strip(),
            "replacement_source_commit": (replacement / "SOURCE_COMMIT")
            .read_text()
            .strip(),
        }

    def save(phase):
        receipt["phase"] = phase
        atomic(
            receipt_path,
            (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
        )

    save(receipt["phase"])
    introduced_hold = False
    patched = receipt["phase"] not in {"prepared", "holding"}
    try:
        if not patched:
            save("holding")
        try:
            run(["scontrol", "hold", job])
        except Exception as error:
            raise RuntimeError(
                f"Hold acknowledgment for job {job} is unconfirmed; it may be held. "
                f"Rerun this updater to reconcile {receipt_path}."
            ) from error
        introduced_hold = True
        if set(states(job, run).values()) != {"PENDING"}:
            raise ValueError(
                "array started during hold; no files or time limit changed"
            )
        same_submission(details(job, root, code, run, held=True))
        # Mark before mutation so an interrupted copy is never released unsafely.
        patched = True
        save("installing")
        for name in reversed(PATCHES):
            atomic(code / name, payloads[name])
        save("installed")
        run(["scontrol", "update", f"JobId={job}", f"TimeLimit={LIMIT}"])
        same_submission(details(job, root, code, run, held=True, shortened=True))
        if set(states(job, run).values()) != {"PENDING"}:
            raise ValueError("array is no longer entirely pending")
        save("release_requested")
        run(["scontrol", "release", job])
        introduced_hold = False
        after = details(job, root, code, run, shortened=True)
        same_submission(after)
        if any(row.get("Reason") == "JobHeldUser" for row in after):
            raise RuntimeError("release was not confirmed; rerun this updater")
        save("complete")
        return {
            "status": "updated",
            "job": job,
            "time_limit": LIMIT,
            "receipt": str(receipt_path),
            "same_submission": True,
            "note": "Old manifest is preserved; use the new bundle for later launches.",
        }
    except Exception as error:
        if introduced_hold and not patched:
            try:
                run(["scontrol", "release", job])
            except Exception as release_error:
                raise RuntimeError(
                    f"Release acknowledgment for job {job} is unconfirmed. "
                    f"No files were changed; reconcile {receipt_path} before retrying."
                ) from release_error
        if patched:
            raise RuntimeError(
                f"Job {job} may remain held: deployment state is unconfirmed. "
                f"Rerun this updater to reconcile {receipt_path}; "
                "do not release blindly."
            ) from error
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("root", "code", "replacement"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            update(args.root, args.job, args.code, args.replacement, apply=args.apply),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
