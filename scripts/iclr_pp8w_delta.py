"""Prepare and run the PP8W follow-up using dedicated Delta-hosted LLMs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.llm import local_judge
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal import iclr_controlled_search as campaign
from autoformalism.rebuttal import judge_jetstream_calibration as original_calibration
from autoformalism.rebuttal import local_judge_calibration as calibration
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import harmony_vocab
from scripts import iclr_pp8w_followup as design
from scripts import iclr_stress_data as data

REPO = Path(__file__).resolve().parents[1]
# Delta's independently built SIF, verified by inspect and installed metadata.
# The historical ACES image remains unchanged in the default campaign config.
IMAGE_SHA = "5449f371cabd3178e31797f6f377c9667462544362f5adc271b2e707172ea907"
IMAGE_ORIGIN = (
    "docker.io/vllm/vllm-openai@sha256:"
    "0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967"
)
PROPOSER_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
CRITIC_SECONDS = 24 * 3600


def paths() -> dict[str, Path]:
    """Reuse installed Delta environments and offline model assets."""
    project = Path(os.environ.get("AF_PROJECT", "/projects/bibo/yxiao2"))
    return {
        "python": Path(
            os.environ.get(
                "AF_PYTHON", str(project / "venvs/autoformalism-v21/bin/python")
            )
        ),
        "image": Path(
            os.environ.get(
                "AF_VLLM_IMAGE", str(project / "containers/vllm-openai-v0.27.1.sif")
            )
        ),
        "cache": Path(os.environ.get("AF_HF_HOME", str(project / "huggingface-cache"))),
        "calibration": Path(
            os.environ.get(
                "AF_CALIBRATION_INPUTS",
                "/work/hdd/bibo/yxiao2/phase_b/judge-jetstream-calibration-inputs-v1.json",
            )
        ),
    }


def assets() -> dict:
    """Check local serving assets; never silently download or change a model."""
    files = paths()
    for name in ("python", "image"):
        if not files[name].is_file():
            raise ValueError(f"missing {name}: {files[name]}")
    with files["image"].open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    image = digest.hexdigest()
    if image != IMAGE_SHA:
        raise ValueError("serving image differs from the pinned Delta image")
    models = {}
    for size, revision in (("20b", PROPOSER_REVISION), ("120b", local_judge.REVISION)):
        path = (
            files["cache"]
            / "hub"
            / f"models--openai--gpt-oss-{size}"
            / "snapshots"
            / revision
        )
        if not (path / "config.json").is_file():
            raise ValueError(f"missing offline model revision: {path}")
        index = path / "model.safetensors.index.json"
        if not index.is_file() or not (path / "tokenizer.json").is_file():
            raise ValueError(f"incomplete offline tokenizer or weight index: {path}")
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards or any(
            not (path / shard).is_file() or (path / shard).stat().st_size == 0
            for shard in shards
        ):
            raise ValueError(f"missing offline model weight shards: {path}")
        models[size] = {
            "revision": revision,
            "config_sha256": data.digest(path / "config.json"),
            "weight_index_sha256": data.digest(index),
            "weight_shards": len(shards),
        }
    harmony = harmony_vocab.verify(
        Path(os.environ.get("AF_HARMONY_ASSETS", str(REPO / "assets/harmony")))
    )
    return {
        "image_sha256": image,
        "image_origin": IMAGE_ORIGIN,
        "vllm_version": "0.27.1",
        "models": models,
        "harmony": harmony,
    }


def prepare(root: Path, source: Path) -> dict:
    """Bind the supplied public conditions and Full method on Delta's runtime."""
    root, source = root.resolve(), source.resolve()
    manifest = sealed_read(source / "manifest.json")
    if manifest["protocol"] != design.PROTOCOL or len(manifest["runs"]) != 18:
        raise ValueError("expected the frozen eighteen-run follow-up package")
    for cell in manifest["cells"]:
        for name, digest in cell["public_files"].items():
            if data.digest(source / "public" / cell["cell"] / name) != digest:
                raise ValueError("follow-up public files changed")
    identity = assets()
    plan = campaign.prepare(
        root / "search",
        source / "public",
        [c["cell"] for c in manifest["cells"]],
        ["full"],
        [0, 1, 2],
        15,
        True,
        True,
        local_critic=True,
        require_fitted_full_start=True,
        serving_image_sha256=IMAGE_SHA,
    )
    bundle = paths()["calibration"]
    if not bundle.is_file():
        bundle = root / "calibration-inputs.json"
        if not bundle.exists():
            original_calibration.export(
                Path(
                    "/work/hdd/bibo/yxiao2/phase_b/judge-hybrid-consensus-validation-v1"
                ),
                Path("/projects/bibo/yxiao2/phase_b/inputs/public"),
                bundle,
            )
    gate = calibration.freeze(bundle, root / "calibration", IMAGE_SHA)
    return sealed_write(
        root / "deployment.json",
        {
            "protocol": "iclr-pp8w-delta-1",
            "design_sha256": manifest["artifact_sha256"],
            "search_plan_sha256": plan["artifact_sha256"],
            "calibration_plan_sha256": gate["artifact_sha256"],
            "assets": identity,
            "source": public._source_identity(),
            "runtime": public._runtime(),
            "code_root": str(REPO),
            "llm_service": (
                "Dedicated local 20B proposer and 120B critic; no hosted API"
            ),
        },
    )


def env() -> dict:
    """Keep hosted credentials out of the local serving and worker environments."""
    value = dict(os.environ)
    value.pop("AF_JETSTREAM_API_KEY", None)
    value.pop("OPENAI_API_KEY", None)
    casadi = os.environ.get(
        "AF_CASADI_ROOT", "/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps"
    )
    value["PYTHONPATH"] = f"{REPO / 'src'}:{REPO}:{casadi}"
    value.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="0",
    )
    return value


def ready(url: str, model: str, key: str) -> bool:
    """Check the model identity using a direct connection with proxies disabled."""
    request = urllib.request.Request(
        url + "/v1/models", headers={"Authorization": "Bearer " + key} if key else {}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=5) as response:
            return any(r["id"] == model for r in json.load(response)["data"])
    except (OSError, ValueError, KeyError):
        return False


@contextmanager
def serve(root: Path, *, critic: bool, key: str = ""):
    """Start one owned server, redact its logs, and stop only that process group."""
    files = paths()
    assets()
    runtime = root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    job = os.environ["SLURM_JOB_ID"]
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    model = "openai/gpt-oss-120b" if critic else "openai/gpt-oss-20b"
    revision = local_judge.REVISION if critic else PROPOSER_REVISION
    role = "critic" if critic else "proposer"
    cache = Path(os.environ.get("TMPDIR", "/tmp")) / f"pp8w-{job}-{role}"
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    process_env = env()
    process_env.update(
        HF_HOME=str(files["cache"]),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_CACHE_ROOT=str(cache / "vllm"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "torch"),
        XDG_CACHE_HOME=str(cache / "xdg"),
    )
    if key:
        process_env["VLLM_API_KEY"] = key
    else:
        process_env.pop("VLLM_API_KEY", None)
    command = [
        shutil.which("apptainer") or shutil.which("singularity") or "apptainer",
        "exec",
        "--nv",
        "--bind",
        f"{files['cache']}:{files['cache']},{cache}:{cache}",
    ]
    harmony = os.environ.get("AF_HARMONY_ASSETS", str(REPO / "assets/harmony"))
    if harmony:
        subprocess.run(
            [
                str(files["python"]),
                str(REPO / "scripts/harmony_vocab.py"),
                "verify",
                "--directory",
                harmony,
            ],
            check=True,
            env=env(),
        )
        command.extend(["--bind", f"{harmony}:{harmony}:ro"])
        process_env["TIKTOKEN_ENCODINGS_BASE"] = harmony
    command += [
        str(files["image"]),
        "vllm",
        "serve",
        model,
        "--revision",
        revision,
        "--host",
        "0.0.0.0" if critic else "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "4" if critic else "1",
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        "4" if critic else "1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=process_env,
        start_new_session=True,
    )

    def log():
        with (runtime / f"{role}-{job}.log").open("w") as stream:
            for line in iter(process.stdout.readline, b""):
                text = line.decode("utf-8", errors="replace")
                stream.write(text.replace(key, "[REDACTED]") if key else text)
                stream.flush()

    thread = threading.Thread(target=log, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 2400
        while not ready(url, model, key):
            if process.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError(f"{role} failed readiness; inspect its redacted log")
            time.sleep(5)
        receipt = {
            "job": job,
            "host": socket.gethostname(),
            "port": port,
            "model": model,
            "revision": revision,
            "image_sha256": IMAGE_SHA,
        }
        atomic_json(runtime / f"{role}-{job}.json", receipt)
        yield url, receipt
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        thread.join(timeout=5)


def pending(root: Path) -> list[int]:
    """Only blocks missing a durable terminal pruning receipt need more work."""
    plan = sealed_read(root / "search/plan.json")
    result = []
    for block in plan["blocks"]:
        task = block["tasks"][0]
        directory = root / "search/pruning" / task["task_id"]
        if not (directory / "result.json").exists():
            result.append(block["block"])
            continue
        receipt = sealed_read(directory / "result.json")
        if receipt["identity"] != plan["artifact_sha256"]:
            raise ValueError("pruning belongs to another campaign")
        if not (directory / "cost.json").exists() or (
            receipt["status"] != "skipped_no_finite_parent"
            and not (directory / "critic.json").exists()
        ):
            result.append(block["block"])
    return result


def submit_job(root: Path, *, critic: bool, indices: list[int] | None = None) -> str:
    """Submit only jobs owned by this campaign, recording exact Slurm arguments."""
    tag = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:8]
    name = f"pp8w-{'critic' if critic else 'full'}-{tag}"
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={name}",
        f"--account={os.environ.get('AF_ACCOUNT', 'bibo-delta-gpu')}",
        "--partition=gpuA40x4",
        "--nodes=1",
        "--ntasks=1",
        f"--gpus-per-node={4 if critic else 1}",
        f"--cpus-per-task={16 if critic else 8}",
        f"--mem={'128G' if critic else '64G'}",
        f"--time={'24:00:00' if critic else '06:00:00'}",
        f"--output={root}/logs/%x-%A_%a.out",
        f"--error={root}/logs/%x-%A_%a.err",
        "--export=ALL",
    ]
    if not critic:
        if not indices:
            raise ValueError("worker submission requires unfinished blocks")
        command.append(
            "--array="
            + ",".join(map(str, indices))
            + "%"
            + os.environ.get("AF_CONCURRENCY", "4")
        )
    command += [
        str(REPO / "scripts/hpc/iclr_pp8w_delta_worker.sh"),
        "critic-worker" if critic else "search-worker",
        str(root),
    ]
    (root / "logs").mkdir(exist_ok=True, parents=True)
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env() | {"AF_REPO_ROOT": str(REPO), "AF_PYTHON": str(paths()["python"])},
    )
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise ValueError("unrecognized Slurm submission response")
    atomic_json(
        root / "submissions" / f"{job}.json",
        {"job": job, "command": command, "indices": indices, "critic": critic},
    )
    return job


def critic_worker(root: Path) -> None:
    """Calibrate a dedicated critic, then keep it alive for this campaign only."""
    expires_at = min(
        time.time() + CRITIC_SECONDS - 900,
        float(os.environ.get("SLURM_JOB_END_TIME", "inf")) - 300,
    )
    with public._lock(root / "critic-owner"):
        credential = root / "runtime/local-critic-key"
        credential.parent.mkdir(parents=True, exist_ok=True)
        if not credential.exists():
            fd = os.open(credential, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(secrets.token_urlsafe(40))
        key = credential.read_text().strip()
        with serve(root, critic=True, key=key) as (url, receipt):
            # Delta nodes can resolve to public IPs. Keep the explicitly allowed
            # cluster hostname, and check it before spending calibration calls.
            endpoint = local_judge.checked_url(
                f"http://{receipt['host']}:{receipt['port']}"
            )
            calibration.run(root / "calibration", url, lambda: key)
            calibration.bind(
                root / "search",
                sealed_read(root / "search/plan.json"),
                root / "calibration",
            )
            atomic_json(
                root / "runtime/critic-service.json",
                {
                    **receipt,
                    "endpoint": endpoint,
                    "expires_at": expires_at,
                    "search_plan": sealed_read(root / "search/plan.json")[
                        "artifact_sha256"
                    ],
                },
            )
            for wave in range(3):
                remaining = pending(root)
                if not remaining:
                    break
                if expires_at - time.time() < 1200:
                    break
                job = submit_job(root, critic=False, indices=remaining)
                print(
                    json.dumps(
                        {"worker_array": job, "wave": wave + 1, "blocks": remaining}
                    ),
                    flush=True,
                )
                while subprocess.run(
                    ["squeue", "-h", "-j", job, "-o", "%i"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip():
                    if expires_at - time.time() < 180:
                        # Only this controller's own array can outlive its critic.
                        subprocess.run(["scancel", job], check=True)
                        atomic_json(
                            root / "resume-needed.json",
                            {"reason": "critic_allocation_ending", "array": job},
                        )
                        raise RuntimeError("allocation ending; resubmit to resume")
                    time.sleep(30)
            if pending(root):
                raise RuntimeError(
                    "unfinished blocks remain after three resumable waves"
                )
            finish_campaign(root)


def finish_campaign(root: Path) -> None:
    """Recover an interrupted final export without allocating another LLM service."""
    campaign.report(root / "search", True)
    atomic_json(
        root / "completed.json",
        {
            "search_plan": sealed_read(root / "search/plan.json")["artifact_sha256"],
            "status": "exported",
            "test_data_opened": False,
        },
    )


def search_worker(root: Path, index: int) -> None:
    """Run one exact Full block with a local proposer and this campaign's critic."""
    service = json.loads((root / "runtime/critic-service.json").read_text())
    plan = sealed_read(root / "search/plan.json")
    if (
        service["search_plan"] != plan["artifact_sha256"]
        or service["revision"] != local_judge.REVISION
        or service["image_sha256"] != IMAGE_SHA
    ):
        raise ValueError("critic service does not belong to this frozen campaign")
    key = (root / "runtime/local-critic-key").read_text().strip()
    if not ready(service["endpoint"], local_judge.MODEL, key):
        raise RuntimeError("campaign-owned critic is unavailable")
    if service["expires_at"] - time.time() < 1200:
        raise RuntimeError("critic allocation is ending; resume in its next allocation")
    with serve(root, critic=False) as (url, _):
        allowance = min(16200, int(service["expires_at"] - time.time() - 600))
        if allowance < 600:
            raise RuntimeError("critic allocation is ending; no search calls started")
        worker_env = env() | {
            "AF_LOCAL_CRITIC_URL": service["endpoint"],
            "AF_LOCAL_CRITIC_KEY": key,
        }
        subprocess.run(
            [
                str(paths()["python"]),
                "-m",
                "scripts.iclr_controlled_search",
                "run",
                "--root",
                str(root / "search"),
                "--block",
                str(index),
                "--base-url",
                url,
                "--seconds",
                str(allowance),
            ],
            cwd=REPO,
            env=worker_env,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "submit", "status", "critic-worker", "search-worker"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    args.root = args.root.resolve()
    os.umask(0o077)
    if args.stage == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        result = prepare(args.root, args.source)
        print(json.dumps({"deployment": result["artifact_sha256"]}))
    elif args.stage == "submit":
        sealed_read(args.root / "deployment.json")
        with public._lock(args.root / "submit-owner"):
            tag = hashlib.sha256(str(args.root.resolve()).encode()).hexdigest()[:8]
            existing = subprocess.run(
                [
                    "squeue",
                    "-h",
                    "-u",
                    os.environ["USER"],
                    "-n",
                    f"pp8w-critic-{tag}",
                    "-o",
                    "%A",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.split()
            if not existing and not pending(args.root):
                finish_campaign(args.root)
                print(json.dumps({"status": "already_complete", "new_jobs": 0}))
                return
            print(
                json.dumps(
                    {
                        "critic_job": existing[0]
                        if existing
                        else submit_job(args.root, critic=True),
                        "reused_active_job": bool(existing),
                    }
                )
            )
    elif args.stage == "critic-worker":
        critic_worker(args.root)
    elif args.stage == "search-worker":
        search_worker(args.root, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        print(
            json.dumps(
                {
                    "pending_blocks": pending(args.root),
                    "completed": (args.root / "completed.json").exists(),
                }
            )
        )


if __name__ == "__main__":
    main()
