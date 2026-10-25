#!/usr/bin/env python3
"""Submit bounded smoke/full search-policy phases without resetting call budgets."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from scripts import harmony_vocab

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "construction-search-policy-1"
EXECUTION_PROTOCOL = "construction-search-aces-execution-1"
POLICIES = {"sequential_restart", "independent", "options_shallow"}
PHASES = {"smoke", "full"}
MODELS = {
    "20b": {
        "model": "openai/gpt-oss-20b",
        "revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
        "gpus": 1,
        "cpus": 8,
        "memory_gib": 64,
    },
    "120b": {
        "model": "openai/gpt-oss-120b",
        "revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
        "gpus": 2,
        "cpus": 16,
        "memory_gib": 128,
    },
}
MATCHED_SETTINGS = {
    "maximum_requests": 3,
    "max_output_tokens": 8192,
    "maximum_total_tokens": 98304,
    "temperature": 0.2,
    "reasoning_effort": "low",
}
ENTRYPOINTS = (
    "scripts/construction_search_campaign.py",
    "scripts/submit_construction_search_aces.py",
    "scripts/hpc/run_construction_search_aces.sh",
    "scripts/hpc/start_construction_search_aces.sh",
    "scripts/harmony_vocab.py",
)


def file_hash(path: Path) -> str:
    """Hash source/container bytes without importing the scientific stack."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("artifact_sha256") != _digest(
        {k: v for k, v in value.items() if k != "artifact_sha256"}
    ):
        raise ValueError(f"artifact digest differs: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _seal(path: Path, value: dict) -> dict:
    result = {**value, "artifact_sha256": _digest(value)}
    if path.exists():
        if _read(path) != result:
            raise ValueError(f"frozen artifact differs: {path}")
    else:
        _write(path, result)
    return result


def _campaign_verify(root: Path) -> dict:
    from scripts import construction_search_campaign

    return construction_search_campaign.verify(root)


def _campaign_report(root: Path) -> dict:
    from scripts import construction_search_campaign

    return construction_search_campaign.report(root)


def smoke_case_ids(plan: dict) -> list[str]:
    """Select one fixed parent/seed across all policies from the frozen roster."""
    first_parent = min(case["parent_id"] for case in plan["cases"])
    return sorted(
        case["case_id"]
        for case in plan["cases"]
        if case["parent_id"] == first_parent and case["seed"] == 0
    )


def verify_inputs(root: Path) -> dict:
    """Reject drift in the shared saved-case roster and generation allowance."""
    plan = _campaign_verify(root)
    size = plan.get("model_size")
    if size not in MODELS:
        raise ValueError("unknown structural revision model size")
    model = MODELS[size]
    expected = {
        **MATCHED_SETTINGS,
        "model": model["model"],
        "model_revision": model["revision"],
    }
    cases = plan.get("cases", [])
    if (
        plan.get("protocol") != PROTOCOL
        or len(cases) != 54
        or len({case["name"] for case in cases}) != 54
        or any(case.get("case_id") != case["name"] for case in cases)
        or len({case.get("parent_id") for case in cases}) != 6
        or any(
            {
                (case.get("seed"), case.get("policy"))
                for case in cases
                if case.get("parent_id") == parent
            }
            != {(seed, policy) for seed in (0, 1, 2) for policy in POLICIES}
            for parent in {case.get("parent_id") for case in cases}
        )
        or any(plan.get("settings", {}).get(k) != v for k, v in expected.items())
        or plan.get("automatic_followup") is not False
        or plan.get("test_data_opened") is not False
        or type(plan.get("optimizer_calls")) is not int
        or plan["optimizer_calls"] != 0
    ):
        raise ValueError(
            "requires six saved cases, three seeds, three policies, three calls "
            "and no fitting"
        )
    return plan


def snapshot_identity(cache: Path, size: str) -> dict:
    """Check the pinned local snapshot, tokenizer and every indexed weight shard.

    Large weight files are identified by immutable cache blob name and size, not
    by a claim that all weight bytes were independently SHA-256 checked here.
    """
    model = MODELS[size]
    snapshot = (
        cache / f"hub/models--openai--gpt-oss-{size}/snapshots" / model["revision"]
    )
    metadata = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        metadata.append(index.name)
        weights = sorted(
            set(json.loads(index.read_text()).get("weight_map", {}).values())
        )
        if not weights:
            raise ValueError("offline model shard index is empty")
    else:
        weights = ["model.safetensors"]
    if any(not isinstance(name, str) or Path(name).name != name for name in weights):
        raise ValueError("offline model shard index contains an invalid path")
    for name in [*metadata, *weights]:
        if not (snapshot / name).is_file() or (snapshot / name).stat().st_size <= 0:
            raise ValueError(
                f"pinned offline model asset is missing or empty: {snapshot / name}"
            )
    return {
        "model_revision": model["revision"],
        "metadata_sha256": {name: file_hash(snapshot / name) for name in metadata},
        "weight_files": {
            name: {
                "bytes": (snapshot / name).stat().st_size,
                "resolved_basename": (snapshot / name).resolve().name,
            }
            for name in weights
        },
        "weight_verification": (
            "pinned-revision-index-and-cache-blob-size; not full weight-byte hashing"
        ),
    }


def _assets(size: str) -> tuple[Path, dict]:
    for name in ("AF_PYTHON", "AF_VLLM_IMAGE", "AF_HF_HOME"):
        if not os.environ.get(name):
            raise ValueError(f"set {name} before submission")
    image = Path(os.environ["AF_VLLM_IMAGE"])
    if not image.is_file() or not Path(os.environ["AF_PYTHON"]).is_file():
        raise ValueError("Python or serving image is missing")
    return image, snapshot_identity(Path(os.environ["AF_HF_HOME"]), size)


def execution_identity(plan: dict, image_sha256: str, snapshot: dict) -> dict:
    """Freeze asset/source identity and explicitly record unequal GPU resources."""
    size = plan["model_size"]
    model = MODELS[size]
    return {
        "protocol": EXECUTION_PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "model_size": size,
        "model": model["model"],
        "model_revision": model["revision"],
        "case_ids": [case["name"] for case in plan["cases"]],
        "smoke_case_ids": smoke_case_ids(plan),
        "platform": f"aces-h100x{model['gpus']}",
        "image_sha256": image_sha256,
        "snapshot": snapshot,
        "harmony_vocabulary": harmony_vocab.verify(
            Path(os.environ.get("AF_HARMONY_ASSETS", REPO / "assets/harmony"))
        ),
        "entrypoint_sha256": {name: file_hash(REPO / name) for name in ENTRYPOINTS},
        "served_context_tokens": 32768,
        "maximum_output_tokens": 8192,
        "tensor_parallel_size": model["gpus"],
        "maximum_sequences": 1,
        "gpu_memory_utilization": 0.90,
        "generation_settings": MATCHED_SETTINGS,
        "maximum_cases": 54,
        "maximum_generation_requests": 162,
        "maximum_charged_tokens": 54 * 98304,
        "smoke_maximum_cases": 3,
        "smoke_maximum_generation_requests": 9,
        "worker_seconds": 12600,
        "scheduler_walltime_seconds": 14400,
        "cpus_per_task": model["cpus"],
        "memory_gib": model["memory_gib"],
        "compiler_cache_policy": "node-local-job-temporary-1",
        "server_binding": "127.0.0.1",
        "optimizer_calls": 0,
        "test_data_opened": False,
        "automatic_followup": False,
        "model_promotion": False,
    }


def verify(root: Path, *, check_assets: bool = False) -> tuple[dict, dict]:
    """Verify the arm and, on the worker, actual image/cache assets before serving."""
    plan = verify_inputs(root)
    execution = _read(root / "execution.json")
    image_hash, snapshot = execution["image_sha256"], execution["snapshot"]
    if check_assets:
        image, snapshot = _assets(plan["model_size"])
        image_hash = file_hash(image)
    expected = execution_identity(plan, image_hash, snapshot)
    if execution != {**expected, "artifact_sha256": _digest(expected)}:
        raise ValueError("structural revision execution identity differs")
    return plan, execution


def bind_server(root: Path, runtime_record: Path, endpoint: str) -> dict:
    """Bind the verified arm to an actual local server without ephemeral port drift.

    Model, image, CUDA/torch/vLLM, GPU hardware, driver and Harmony identities stay
    frozen across resume. Job ids, timestamps and endpoint ports are deliberately
    absent from the worker's serving record, so another allocation can resume
    only when its substantive serving configuration matches.
    """
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("server must use an unauthenticated loopback HTTP endpoint")
    _, execution = verify(root)
    runtime = _read(runtime_record)
    serving_keys = (
        "artifact_sha256",
        "model_size",
        "model",
        "model_revision",
        "image_sha256",
        "served_context_tokens",
        "tensor_parallel_size",
        "maximum_sequences",
    )
    if runtime.get("serving_identity") != {k: execution[k] for k in serving_keys}:
        raise ValueError("actual serving identity differs from the sealed arm")
    count = execution["tensor_parallel_size"]
    names = runtime.get("gpu_names", [])
    memory = runtime.get("gpu_memory_bytes", [])
    drivers = runtime.get("gpu_driver", [])
    if (
        type(runtime.get("visible_gpus")) is not int
        or runtime["visible_gpus"] != count
        or runtime.get("cuda_kernel_passed") is not True
        or not isinstance(names, list)
        or len(names) != count
        or any(not isinstance(name, str) or "H100" not in name for name in names)
        or not isinstance(memory, list)
        or len(memory) != count
        or any(type(size) is not int or size < 75 * 1024**3 for size in memory)
        or not isinstance(drivers, list)
        or not drivers
        or any(not isinstance(driver, str) or not driver for driver in drivers)
        or runtime.get("vllm") != "0.27.1"
        or not isinstance(runtime.get("torch"), str)
        or not runtime["torch"]
        or not isinstance(runtime.get("cuda"), str)
        or not runtime["cuda"]
    ):
        raise ValueError(
            "actual runtime requires the arm's working H100 80GB GPUs and pinned vLLM"
        )
    harmony = runtime.get("harmony", {})
    if (
        {k: harmony.get(k) for k in ("protocol", "filename", "sha256")}
        != execution["harmony_vocabulary"]
        or harmony.get("harmony_render_passed") is not True
        or not isinstance(harmony.get("openai_harmony"), str)
        or not harmony["openai_harmony"]
        or type(harmony.get("rendered_tokens")) is not int
        or harmony["rendered_tokens"] <= 0
        or harmony.get("generation_requests") != 0
    ):
        raise ValueError(
            "actual runtime requires the pinned offline Harmony rendering probe"
        )
    with urlopen(endpoint.rstrip("/") + "/v1/models", timeout=15) as response:
        served = json.load(response)
    if [row.get("id") for row in served.get("data", [])] != [execution["model"]]:
        raise ValueError("actual local model alias differs from the sealed arm")
    return _seal(
        root / "serving_runtime.json",
        {k: v for k, v in runtime.items() if k != "artifact_sha256"},
    )


def _submit_job(directory: Path, argv: list[str]) -> str:
    """Preserve an exclusive intent even when ACES gives an ambiguous response."""
    with (directory / "propose.intent.json").open("x") as stream:
        json.dump({"argv": argv}, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        reply = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        _write(directory / "propose.reply.json", {"status": "uncertain_timeout"})
        raise ValueError(
            "uncertain scheduler timeout; inspect existing jobs before any recovery"
        ) from None
    _write(
        directory / "propose.reply.json",
        {
            "returncode": reply.returncode,
            "stdout": reply.stdout,
            "stderr": reply.stderr,
        },
    )
    job = reply.stdout.strip().split(";")[0]
    if reply.returncode or not job.isdecimal():
        raise ValueError(
            f"scheduler reply unconfirmed; do not resubmit; inspect {directory}"
        )
    _write(directory / "propose.receipt.json", {"job": job})
    return job


def _check_prior_submissions(root: Path) -> None:
    """Never overlap allocations or bypass an unresolved scheduler intent."""
    jobs = set()
    for intent in sorted((root / "submission").glob("*/*/propose.intent.json")):
        manifest = intent.parent / "manifest.json"
        if not manifest.is_file():
            raise ValueError(
                "existing submission intent; inspect scheduler/receipt before "
                f"recovery: {intent.parent}"
            )
        job = json.loads(manifest.read_text()).get("job")
        if not isinstance(job, str) or not job.isdecimal():
            raise ValueError("invalid prior scheduler job receipt")
        jobs.add(job)
    if not jobs:
        return
    user = os.environ.get("USER")
    if not user:
        raise ValueError("USER is required to verify no prior allocation is active")
    try:
        reply = subprocess.run(
            ["squeue", "--noheader", "--user", user, "--format=%i|%T"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise ValueError("could not verify prior job state; no submission") from None
    if reply.returncode:
        raise ValueError("could not verify prior job state; no submission")
    active = []
    for line in reply.stdout.splitlines():
        if not line.strip():
            continue
        columns = line.strip().split("|")
        if len(columns) != 2 or not columns[0].strip():
            raise ValueError("unrecognized scheduler state reply; no submission")
        if columns[0].strip() in jobs:
            active.append(line.strip())
    if active:
        raise ValueError("prior allocation still active: " + ", ".join(active))


def submit(root: Path, phase: str = "smoke", wave: str = "wave1") -> dict:
    """Explicit phase/wave submission; every wave retains the same durable budget."""
    if phase not in PHASES or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", wave):
        raise ValueError("invalid phase or wave")
    root = root.resolve()
    plan = verify_inputs(root)
    image, snapshot = _assets(plan["model_size"])
    identity = execution_identity(plan, file_hash(image), snapshot)
    os.environ.update(
        AF_REPO_ROOT=str(REPO), AF_OUTPUT_ROOT=str(root), AF_SEARCH_PHASE=phase
    )
    root.mkdir(parents=True, exist_ok=True)
    with (root / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        execution = _seal(root / "execution.json", identity)
        directory = root / "submission" / phase / wave
        manifest = directory / "manifest.json"
        if manifest.exists():
            value = json.loads(manifest.read_text())
            if (
                value.get("execution_sha256") != execution["artifact_sha256"]
                or value.get("phase") != phase
                or value.get("wave") != wave
            ):
                raise ValueError("submission identity differs")
            return value
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(
                "existing submission intent; inspect scheduler/receipt before "
                f"recovery: {directory}"
            )
        _check_prior_submissions(root)
        summary = _campaign_report(root)
        if phase == "full" and summary.get("smoke_complete") is not True:
            raise ValueError(
                "smoke phase is not complete; inspect results before full submission"
            )
        complete = (
            summary.get("smoke_complete") is True
            if phase == "smoke"
            else summary.get("status") == "complete"
        )
        if complete:
            return {
                "status": "already_complete",
                "phase": phase,
                "execution_sha256": execution["artifact_sha256"],
                "automatic_followup": False,
            }
        directory.mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        model = MODELS[plan["model_size"]]
        argv = [
            "sbatch",
            "--parsable",
            "--nodes=1",
            "--ntasks=1",
            "--export=ALL",
            "--account=" + os.environ.get("AF_ACCOUNT", "156264627414"),
            "--partition=gpu",
            f"--gres=gpu:h100:{model['gpus']}",
            f"--cpus-per-task={model['cpus']}",
            f"--mem={model['memory_gib']}G",
            "--time=04:00:00",
            "--signal=B:TERM@300",
            "--exclude=ac042",
            f"--job-name=phasec-search-{plan['model_size']}-{phase}",
            f"--output={root}/logs/{phase}-{wave}-%j.out",
            f"--error={root}/logs/{phase}-{wave}-%j.err",
            str(REPO / "scripts/hpc/run_construction_search_aces.sh"),
        ]
        job = _submit_job(directory, argv)
        result = {
            "execution_sha256": execution["artifact_sha256"],
            "model_size": plan["model_size"],
            "phase": phase,
            "wave": wave,
            "job": job,
            "maximum_cases": 3 if phase == "smoke" else 54,
            "maximum_generation_requests": 9 if phase == "smoke" else 162,
            "campaign_maximum_generation_requests": 162,
            "budget_reset": False,
            "automatic_followup": False,
        }
        _write(manifest, result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--check-assets", action="store_true")
    parser.add_argument("--phase", choices=sorted(PHASES), default="smoke")
    parser.add_argument("--wave", default="wave1")
    args = parser.parse_args()
    if args.check_assets and not args.verify_only:
        parser.error("--check-assets requires --verify-only")
    result = (
        verify(args.root.resolve(), check_assets=args.check_assets)[1]
        if args.verify_only
        else submit(args.root, args.phase, args.wave)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
