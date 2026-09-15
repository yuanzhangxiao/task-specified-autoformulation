#!/usr/bin/env python3
"""Recover physical screening/refinement from saved J/A points, without collocation."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep, time

from scaled_recovery_io import checked, prepare, publish, read, report, verify, write

REPO = Path(__file__).resolve().parents[1]


def event(root: Path, name: str, **details) -> None:
    root.mkdir(parents=True, exist_ok=True)
    value = {"event": name, "utc_seconds": time(), "monotonic": monotonic(), **details}
    with (root / "supervisor-events.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")
        stream.flush()
    print(json.dumps(value), flush=True)


def stop(process: subprocess.Popen) -> None:
    """Never follow SIGKILL with an unbounded wait on a filesystem-stalled child."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "killed worker remains uninterruptible; stop this task"
        ) from error


class Publisher:
    """A stalled shared-filesystem copy must not block numerical supervision."""

    def __init__(self, local: Path, persistent: Path, seconds: float):
        self.local, self.persistent, self.seconds = local, persistent, seconds
        self.process = None
        self.started = 0.0
        self.last_success = -float("inf")
        self.error = None

    def tick(self, *, force=False):
        if self.process is not None:
            code = self.process.poll()
            if code is None and monotonic() - self.started >= self.seconds:
                stop(self.process)
                code = 124
            if code is not None:
                if code:
                    self.error = f"publication failed/timeout: {code}"
                else:
                    self.last_success = monotonic()
                event(
                    self.local,
                    "publication_end",
                    exit_code=code,
                    seconds=monotonic() - self.started,
                )
                self.process = None
        if (
            self.process is None
            and self.error is None
            and (force or monotonic() - self.last_success >= 15)
        ):
            event(self.local, "publication_begin")
            self.started = monotonic()
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-S" if sys.flags.no_site else "-u",
                    __file__,
                    "publish",
                    "--output",
                    str(self.local),
                    "--persistent",
                    str(self.persistent),
                ],
                start_new_session=True,
            )

    def flush(self):
        # Complete an in-flight snapshot, then publish all changes made since it began.
        if self.process is not None:
            while self.process is not None:
                self.tick()
                if self.process is not None:
                    sleep(0.05)
        if self.error:
            raise RuntimeError(self.error)
        self.tick(force=True)
        while self.process is not None:
            self.tick()
            if self.process is not None:
                sleep(0.05)
        if self.error:
            raise RuntimeError(self.error)


def failure(root: Path, identity: str, status: str, message: str) -> dict:
    """A terminated refinement still retains any full-training incumbent."""
    best = root / "calls/best_evaluated.json"
    calls = list((root / "calls").glob("[0-9]*.json"))
    return {
        "identity": identity,
        "status": status,
        "message": message,
        "best": read(best) if best.exists() else None,
        "calls": len(calls),
    }


def stage(
    output: Path,
    task: Path,
    index: int,
    name: str,
    seconds: float,
    frozen: dict,
    publisher: Publisher,
) -> dict:
    root = task / name
    root.mkdir(parents=True, exist_ok=True)
    identity = frozen["identity"]
    result_path, intent = root / "result.json", root / "started.json"
    if result_path.exists():
        return checked(result_path, identity)
    if intent.exists():
        previous = checked(intent, identity)
        if previous["task_index"] != index or previous["stage"] != name:
            raise ValueError("resumed stage differs")
        result = failure(
            root,
            identity,
            "interrupted",
            "Prior recovery attempt consumed; no fresh stage budget on resume",
        )
        write(result_path, result, immutable=True)
        publisher.flush()
        return result
    write(
        intent,
        {
            "identity": identity,
            "stage": name,
            "task_index": index,
            "numerical_budget_seconds": seconds,
            "startup_budget_seconds": frozen["startup_seconds"],
        },
        immutable=True,
    )
    publisher.flush()  # Commit the budget claim before any numerical side effect.
    event(task, "worker_launch", stage=name)
    launched = monotonic()
    killed_status = None
    ready = None
    with (root / "worker.log").open("a", buffering=1) as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-S" if sys.flags.no_site else "-u",
                __file__,
                "worker",
                "--output",
                str(output),
                "--task-index",
                str(index),
                "--stage",
                name,
                "--seconds",
                str(seconds),
            ],
            stdout=stream,
            stderr=stream,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                publisher.tick()
                if ready is None and (root / "ready.json").exists():
                    ready = checked(root / "ready.json", identity)
                    event(
                        task,
                        "worker_ready",
                        stage=name,
                        startup_seconds=ready["monotonic"] - launched,
                    )
                deadline = (
                    (ready["monotonic"] + seconds)
                    if ready
                    else (launched + frozen["startup_seconds"])
                )
                # Publication is outside the compute budget, but locally bounded too.
                if (root / "worker-result.json").exists():
                    deadline += 5
                if monotonic() >= deadline:
                    killed_status = "numerical_timeout" if ready else "startup_timeout"
                    stop(process)
                    break
                sleep(0.05)
        finally:
            if process.poll() is None:
                stop(process)
    elapsed = monotonic() - launched
    if ready is None and (root / "ready.json").exists():
        ready = checked(root / "ready.json", identity)
    worker_result = root / "worker-result.json"
    if worker_result.exists():
        result = checked(worker_result, identity)
    else:
        result = failure(
            root,
            identity,
            killed_status or "worker_failed",
            f"worker exit code {process.returncode}",
        )
    result.update(
        startup_seconds=(ready["monotonic"] - launched) if ready else elapsed,
        worker_wall_seconds=elapsed,
        exit_code=process.returncode,
    )
    if ready and "numerical_seconds" not in result:
        result["numerical_seconds"] = max(0, monotonic() - ready["monotonic"])
    write(result_path, result, immutable=True)
    event(task, "worker_finished", stage=name, status=result["status"], seconds=elapsed)
    publisher.flush()
    return result


def select(stages: dict) -> tuple[dict | None, str | None]:
    points = [
        (value["best"], name) for name, value in stages.items() if value.get("best")
    ]
    return min(points, key=lambda pair: pair[0]["cost"]) if points else (None, None)


def finish(output: Path, index: int, frozen: dict, stages: dict) -> dict:
    common = read(output / "common.json")
    best, selected = select(stages)
    replay = {}
    for split in ("train", "val"):
        a, b = [
            stages.get(f"replay-{split}-{method}", {}) for method in ("Radau", "BDF")
        ]
        passed = a.get("status") == b.get("status") == "finished"
        if passed:
            pa, pb = a["predictions"], b["predictions"]
            delta = (
                max(
                    (abs(x - y) for x, y in zip(pa, pb, strict=False)),
                    default=float("inf"),
                )
                / common["output_scale"]
            )
            passed = len(pa) == len(pb) and delta <= 1e-4
            replay[split] = {
                "pass": passed,
                "maximum_scaled_prediction_difference": delta,
                "nmse": a["nmse"],
                "BDF_nmse": b["nmse"],
            }
        else:
            replay[split] = {
                "pass": False,
                "errors": [a.get("status"), b.get("status")],
            }
    agreement = all(v["pass"] for v in replay.values())
    execution_failed = any(
        s["status"]
        in {"startup_timeout", "startup_failed", "worker_failed", "interrupted"}
        for s in stages.values()
    )
    result = {
        "identity": frozen["identity"],
        "arm": frozen["tasks"][index]["arm"],
        "status": ("complete" if agreement else "replay_unverified")
        if best
        else ("execution_failed" if execution_failed else "no_feasible_screen"),
        "selected_source": selected,
        "best": best,
        "replay_agreement": agreement,
        "train_nmse": replay["train"].get("nmse"),
        "validation_nmse": replay["val"].get("nmse"),
        "replay": replay,
        "stages": stages,
        "collocation_reruns": 0,
        "test_data_opened": False,
        "llm_calls": 0,
        "hidden_trajectory_labels_used": False,
        "known_hidden_initials_fixed": True,
    }
    for label, threshold in (("strict", 1e-4), ("good", 0.01), ("practical", 0.1)):
        result[label] = bool(
            agreement and all(v["nmse"] <= threshold for v in replay.values())
        )
    return result


def run(output: Path, persistent: Path, index: int):
    event(output / "runtime", "verification_begin")
    frozen = verify(output, REPO)
    event(output / "runtime", "verification_end")
    if index not in (0, 1):
        raise ValueError("task index must be 0 or 1")
    task = output / f"results/task_{index:03d}"
    destination = persistent / f"results/task_{index:03d}"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Existing durable stage intents/results are authoritative on a fresh node.
        if destination.exists():
            publish(destination, task)
        if (task / "result.json").exists():
            return checked(task / "result.json", frozen["identity"])
        publisher = Publisher(task, destination, frozen["publish_seconds"])
        plan, stages = frozen["plan"], {}
        for name in ("screen-ordinary", "screen-checkpoint"):
            stages[name] = stage(
                output,
                task,
                index,
                name,
                plan["screen_point_seconds"],
                frozen,
                publisher,
            )
        best, source = select(stages)
        write(
            task / "selected.json",
            {"identity": frozen["identity"], "best": best, "source": source},
        )
        if best is None:
            stages["refinement"] = {"status": "no_feasible_screen", "calls": 0}
            for split in ("train", "val"):
                for method in ("Radau", "BDF"):
                    stages[f"replay-{split}-{method}"] = {"status": "no_parameters"}
        else:
            stages["refinement"] = stage(
                output,
                task,
                index,
                "refinement",
                plan["refinement_seconds"],
                frozen,
                publisher,
            )
            best, source = select(stages)
            write(
                task / "selected.json",
                {"identity": frozen["identity"], "best": best, "source": source},
            )
            for split in ("train", "val"):
                for method in ("Radau", "BDF"):
                    name = f"replay-{split}-{method}"
                    stages[name] = stage(
                        output,
                        task,
                        index,
                        name,
                        plan["replay_seconds"],
                        frozen,
                        publisher,
                    )
        result = finish(output, index, frozen, stages)
        write(task / "result.json", result, immutable=True)
        publisher.flush()
        return result


def worker(output: Path, index: int, name: str, seconds: float):
    root = output / f"results/task_{index:03d}" / name
    event(root, "python_entered_before_numerical_imports")
    identity = read(output / "freeze.json")["identity"]
    sys.path.insert(0, str(REPO / "src"))
    try:
        from autoformalism.rebuttal.scaled_recovery_worker import execute

        result = execute(output, root, index, name, seconds)
    except Exception as error:
        result = failure(
            root,
            identity,
            "numerical_failed" if (root / "ready.json").exists() else "startup_failed",
            f"{type(error).__name__}: {error}",
        )
    event(root, "result_publication_begin")
    write(root / "worker-result.json", result, immutable=True)
    event(root, "result_publication_end")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "action", choices=("prepare", "run", "worker", "publish", "summarize")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--persistent", type=Path)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--stage")
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        print(
            json.dumps(
                {
                    "identity": prepare(args.source, args.output, REPO)["identity"],
                    "tasks": 2,
                }
            )
        )
    elif args.action == "run":
        if args.persistent is None:
            parser.error("run requires --persistent")
        print(
            json.dumps(
                {"status": run(args.output, args.persistent, args.task_index)["status"]}
            )
        )
    elif args.action == "worker":
        worker(args.output, args.task_index, args.stage, args.seconds)
    elif args.action == "publish":
        publish(args.output, args.persistent)
    else:
        result = report(args.output)
        print(
            json.dumps(
                {
                    "statuses": [r["status"] for r in result["rows"]],
                    "summary": str(args.output / "SUMMARY.md"),
                }
            )
        )


if __name__ == "__main__":
    main()
