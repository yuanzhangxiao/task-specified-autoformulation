#!/usr/bin/env python3
"""Recover unavailable Delta work using each campaign's unchanged pinned code.

This supervisor uses only the standard library and supports the login node's
Python >= 3.6. Numerical imports happen in a separately supervised interpreter.
"""

import argparse
import collections
import contextlib
import fcntl
import hashlib
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PINS = {
    "stopping": "78fab68cbe83149218367e95d5d2b2ce62778fd0",
    "piecewise": "4d5f73639e9c0f57bed641f3f510b2e396701672",
}
MODULES = {
    "stopping": "autoformalism.rebuttal.fitter_rate_refinement",
    "piecewise": "autoformalism.rebuttal.piecewise_campaign",
}
ARMS = {
    "stopping": ("rates_default", "rates_no_ftol"),
    "piecewise": tuple(
        f"mesh{mesh}_{method}"
        for mesh in (1, 2)
        for method in ("branch_sensitivity", "directional_poll")
    ),
}
INFRASTRUCTURE = {"timeout", "worker_failed"}
SCRIPT = Path(__file__).resolve()


def digest(path):
    """Hash without loading large replay arrays into memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def content_hash(value):
    """Match the two original campaigns' canonical JSON identities."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    with path.open() as stream:
        return json.load(stream)


def saved(path):
    return read(path) if path.exists() else {}


def write(path, value):
    """Atomically replace a small record; never leave half-written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(str(temp), str(path))


def safe_path(root, relative):
    """Reject traversal and symlinks, including symlinked intermediate directories."""
    item = Path(relative)
    if item.is_absolute() or ".." in item.parts:
        raise ValueError("artifact path escapes experiment: " + str(relative))
    path = root
    for part in item.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("symlinked artifact: " + str(path))
    return path


@contextlib.contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def event(root, stage, **fields):
    """Flush stage markers before entering potentially blocking native work."""
    row = dict(fields, stage=stage, unix_seconds=time.time(), pid=os.getpid())
    root.mkdir(parents=True, exist_ok=True)
    with (root / "stages.jsonl").open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
    print(json.dumps(row, sort_keys=True), flush=True)


def units(campaign, frozen):
    if campaign == "stopping":
        if frozen["plan"]["protocol"] != "fitter-rate-stopping-1":
            raise ValueError("expected the original stopping comparison")
        for index, task in enumerate(frozen["tasks"]):
            if task["kind"] != "pair":
                raise ValueError("recovery expects paired stopping tasks")
            yield index, task["source_task"], "results/" + task["name"]
    else:
        for index, case in enumerate(frozen["cases"]):
            if case["index"] != index:
                raise ValueError("noncanonical case indices")
            yield index, case["label"], f"results/case_{index:03d}"


def actions_for(campaign, source, relative):
    """Select by artifact availability only, never by a numerical score."""
    root = safe_path(source, relative)
    parent = saved(root / "result.json") if campaign == "stopping" else {}
    terminal_parent = parent and parent.get("status") not in INFRASTRUCTURE
    actions = []
    for arm in ARMS[campaign]:
        path = root / arm
        result = saved(path / "result.json") if campaign == "piecewise" else {}
        if terminal_parent or result:
            mode = "preserve_terminal"
        elif (path / "fit.json").exists():
            mode = "reuse_fit"
        elif (path / "poll_checkpoint.json").exists():
            if not (path / "initializer.json").exists():
                raise ValueError(
                    "poll checkpoint has no saved initializer: " + str(path)
                )
            mode = "resume_poll"
        elif (path / "fit_started.json").exists() or (
            path / "initializer_started.json"
        ).exists():
            mode = "restart_interrupted_native"
        else:
            mode = "start_unavailable"
        actions.append({"arm": arm, "mode": mode})
    return actions


def timeout_replay_files(source, relative):
    """Renew only interrupted replay records; retain successful and failed solves."""
    root = safe_path(source, relative)
    omissions = set()
    for path in root.glob("replays/*/*/trajectory-*.json"):
        safe_path(source, str(path.relative_to(source)))
        row = read(path)
        message = str(row.get("message", row.get("error", ""))).lower()
        if row.get("status") != "complete" and (
            "wall-clock limit" in message or "timeout" in message
        ):
            omissions.add(str(path.relative_to(source)))
            aggregate = path.parent / "result.json"
            if aggregate.exists():
                omissions.add(str(aggregate.relative_to(source)))
    return omissions


def inventory(source):
    """Inventory numerical checkpoints only; worker logs stay in the original run."""
    files = {}
    root = safe_path(source, "results")
    if root.exists():
        for parent, dirs, names in os.walk(str(root), followlinks=False):
            for name in dirs + names:
                path = safe_path(source, str((Path(parent) / name).relative_to(source)))
                if path.is_file() and path.suffix in {
                    ".json",
                    ".jsonl",
                    ".npz",
                    ".npy",
                }:
                    files[str(path.relative_to(source))] = digest(path)
    return files


def make_plan(campaign, source, repo, python, dependencies):
    frozen = read(safe_path(source, "freeze.json"))
    key = "freeze_sha256" if campaign == "stopping" else "identity"
    if frozen[key] != content_hash({k: v for k, v in frozen.items() if k != key}):
        raise ValueError("original freeze hash differs")
    static = {"freeze.json": digest(source / "freeze.json")}
    static.update(
        {"source/" + name: value for name, value in frozen["source_manifest"].items()}
        if campaign == "stopping"
        else frozen["assets"]
    )
    for name, expected in static.items():
        if digest(safe_path(source, name)) != expected:
            raise ValueError("original frozen input differs: " + name)
    originals = inventory(source)
    tasks, excluded = [], {}
    for index, label, relative in units(campaign, frozen):
        actions = actions_for(campaign, source, relative)
        selected = any(a["mode"] != "preserve_terminal" for a in actions)
        tasks.append(
            {
                "index": index,
                "label": label,
                "path": relative,
                "actions": actions,
                "selected": selected,
            }
        )
        if not selected:
            continue
        if campaign == "stopping" and relative + "/result.json" in originals:
            excluded[relative + "/result.json"] = "infrastructure supervisor result"
        for action in actions:
            prefix = relative + "/" + action["arm"] + "/"
            if action["mode"] == "restart_interrupted_native":
                for name in originals:
                    if name.startswith(prefix) and name != prefix + "initializer.json":
                        excluded[name] = (
                            "explicit fresh native attempt; old run retained"
                        )
            elif campaign == "stopping" and action["mode"] == "reuse_fit":
                for name in timeout_replay_files(source, prefix):
                    excluded[name] = (
                        "explicit replay retry with saved fitted parameters"
                    )
    plan = {
        "protocol": "fitter-outage-recovery-1",
        "campaign": campaign,
        "source": str(source),
        "legacy_repo": str(repo),
        "python": str(python),
        "dependencies": str(dependencies),
        "legacy_commit": PINS[campaign],
        "driver_sha256": digest(SCRIPT),
        "original_identity": frozen[key],
        "static_files": static,
        "original_results": originals,
        "excluded_from_attempt": excluded,
        "tasks": tasks,
        "numerical_settings_changed": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }
    plan["identity"] = content_hash(plan)
    return plan


def load_plan(output):
    plan = read(output / "recovery.json")
    if plan["identity"] != content_hash(
        {k: v for k, v in plan.items() if k != "identity"}
    ):
        raise ValueError("recovery plan changed")
    if plan["driver_sha256"] != digest(SCRIPT):
        raise ValueError("recovery driver changed; use the pinned checkout")
    if saved(output / "prepared.json").get("identity") != plan["identity"]:
        raise ValueError("preparation incomplete; repeat prepare before execution")
    if (
        digest(safe_path(output / "snapshot", "freeze.json"))
        != plan["static_files"]["freeze.json"]
    ):
        raise ValueError("copied original freeze changed")
    return plan


def prepare(args):
    output, source = args.output.resolve(), args.source.resolve()
    if os.path.commonpath([str(output), str(source)]) in {str(output), str(source)}:
        raise ValueError("original and recovery directories must not overlap")
    with locked(output / "prepare.lock"):
        if (output / "prepared.json").exists():
            plan = load_plan(output)
            for key, value in (
                ("campaign", args.campaign),
                ("source", str(source)),
                ("legacy_repo", str(args.legacy_repo.resolve())),
                ("python", str(args.python.absolute())),
                ("dependencies", str(args.dependencies.resolve())),
            ):
                if plan[key] != value:
                    raise ValueError("prepared recovery arguments differ: " + key)
            return plan
        plan = make_plan(
            args.campaign,
            source,
            args.legacy_repo.resolve(),
            args.python.absolute(),
            args.dependencies.resolve(),
        )
        path = output / "recovery.json"
        if path.exists() and read(path) != plan:
            raise ValueError(
                "source changed during partial preparation; use a new directory"
            )
        write(path, plan)
        files = dict(plan["static_files"])
        files.update(
            {
                n: h
                for n, h in plan["original_results"].items()
                if n not in plan["excluded_from_attempt"]
            }
        )
        for name, expected in files.items():
            destination = safe_path(output / "snapshot", name)
            if destination.exists():
                if digest(destination) != expected:
                    raise ValueError("partial recovery copy changed: " + name)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_name(destination.name + ".copying")
            shutil.copyfile(str(safe_path(source, name)), str(temp))
            if digest(temp) != expected:
                raise ValueError("source changed during copy: " + name)
            os.replace(str(temp), str(destination))
        write(output / "prepared.json", {"identity": plan["identity"]})
        return plan


def supervise(command, root, seconds, env=None, grace=5):
    """Bound child startup and teardown; never use an unbounded wait after kill."""
    event(root, "spawn", deadline_seconds=seconds)
    with (root / "worker.log").open("a") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=log, env=env, start_new_session=True
        )
        event(root, "spawned", child_pid=child.pid)
        try:
            code = child.wait(timeout=seconds)
            result = {
                "status": "complete" if code == 0 else "worker_failed",
                "exit_code": code,
            }
        except (subprocess.TimeoutExpired, KeyboardInterrupt, SystemExit) as error:
            result = {
                "status": "timeout"
                if isinstance(error, subprocess.TimeoutExpired)
                else "interrupted",
                "child_pid": child.pid,
            }
            # Record the timeout before any potentially problematic process cleanup.
            write(root / "supervisor.json", result)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, sig)
                try:
                    child.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    continue
            result["child_reaped"] = child.poll() is not None
        write(root / "supervisor.json", result)
        event(root, "supervisor_finished", **result)
        return result


def instrument(module, names, root):
    """Log function boundaries without changing arguments, results or budgets."""
    for name in names:
        original = getattr(module, name)

        def wrapped(*args, _function=original, _name=name, **kwargs):
            event(root, _name + "_begin")
            try:
                return _function(*args, **kwargs)
            finally:
                event(root, _name + "_end")

        setattr(module, name, wrapped)


def legacy_worker(output, smoke, index):
    plan = load_plan(output)
    root = output / ("smoke" if smoke else f"attempts/task_{index:03d}")
    repo = Path(plan["legacy_repo"])
    event(root, "legacy_python_started", python=sys.version)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), universal_newlines=True, timeout=30
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=str(repo),
        universal_newlines=True,
        timeout=30,
    ).strip()
    if commit != plan["legacy_commit"] or dirty:
        raise ValueError("legacy checkout differs from the clean pinned version")
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(1, plan["dependencies"])
    event(root, "numerical_import_begin")
    module = importlib.import_module(MODULES[plan["campaign"]])
    event(root, "numerical_import_end")
    snapshot = output / "snapshot"
    verify = module.verify_rate if plan["campaign"] == "stopping" else module.verify
    event(root, "freeze_verification_begin")
    verify(snapshot)
    event(root, "freeze_verification_end")
    if smoke:
        import casadi as ca
        from scipy.integrate import solve_ivp

        event(root, "native_solver_smoke_begin")
        opti = ca.Opti()
        x = opti.variable()
        opti.minimize((x - 2) ** 2)
        opti.solver("ipopt", {"print_time": False}, {"print_level": 0})
        value = float(opti.solve().value(x))
        ode = solve_ivp(
            lambda t, y: -y, (0, 1), [1.0], method="Radau", rtol=1e-9, atol=1e-11
        )
        if (
            abs(value - 2) > 1e-6
            or not ode.success
            or abs(ode.y[0, -1] - 0.3678794411714423) > 1e-8
        ):
            raise ValueError("native runtime smoke failed")
        write(
            root / "passed.json",
            {"identity": plan["identity"], "casadi": ca.__version__},
        )
        event(root, "native_solver_smoke_passed")
        return
    if plan["campaign"] == "stopping":
        instrument(module, ("start_guard", "refine", "verify_replays"), root)
        result = module.execute_rate(snapshot, index)
    else:
        instrument(module, ("fit_collocation_forward_sensitivity", "replay"), root)
        result = module.execute(snapshot, index)
    event(root, "numerical_task_finished", numerical_status=result["status"])


def run(output, smoke=False, index=None):
    plan = load_plan(output)
    if not smoke and (
        index is None
        or index not in [t["index"] for t in plan["tasks"] if t["selected"]]
    ):
        raise ValueError("task is not selected by the recovery plan")
    root = output / ("smoke" if smoke else f"attempts/task_{index:03d}")
    with locked(root / "run.lock"):
        if saved(root / "supervisor.json").get("status") == "complete":
            return
        if (
            not smoke
            and saved(output / "smoke/passed.json").get("identity") != plan["identity"]
        ):
            raise ValueError("run the native smoke successfully before recovery tasks")
        # Interrupted fresh native attempts require another explicitly named recovery,
        # not repeated budget resets in this directory. Poll-only work can resume.
        if not smoke and (root / "started.json").exists():
            task = plan["tasks"][index]
            current = actions_for(plan["campaign"], output / "snapshot", task["path"])
            if any(a["mode"] == "restart_interrupted_native" for a in current):
                raise ValueError(
                    "this recovery's native fit was interrupted; "
                    "preserve it and prepare a new attempt"
                )
        write(root / "started.json", {"identity": plan["identity"]})
        frozen = read(output / "snapshot/freeze.json")
        if smoke:
            seconds = 300
        elif plan["campaign"] == "stopping":
            config = frozen["plan"]
            seconds = (
                60 + config["guard_seconds"] + 2 * (600 + 6 * config["replay_seconds"])
            )
        else:
            config = frozen["plan"]["fit"]
            seconds = (
                4
                * (
                    config["initializer_seconds"]
                    + 3 * config["refinement_seconds"]
                    + 240
                )
                + 120
            )
        env = dict(os.environ)
        env["PYTHONPATH"] = (
            str(Path(plan["legacy_repo"]) / "src") + os.pathsep + plan["dependencies"]
        )
        env["PYTHONUNBUFFERED"] = "1"
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[name] = "1"
        command = [
            plan["python"],
            str(SCRIPT),
            "legacy-smoke" if smoke else "legacy-worker",
            "--output",
            str(output),
        ]
        if not smoke:
            command += ["--task-index", str(index)]
        result = supervise(command, root, seconds, env=env)
        if result["status"] != "complete":
            raise RuntimeError(
                "supervised worker " + result["status"] + "; see " + str(root)
            )


def summarize(output):
    """Report all planned arms with provenance, without importing numerical code."""
    plan = load_plan(output)
    frozen = read(output / "snapshot/freeze.json")
    rows = []
    for task in plan["tasks"]:
        root = safe_path(output / "snapshot", task["path"])
        parent = saved(root / "result.json") if plan["campaign"] == "stopping" else {}
        for action in task["actions"]:
            arm = action["arm"]
            if plan["campaign"] == "stopping":
                record = parent.get("arms", {}).get(arm, parent)
                expected = content_hash(
                    [frozen["freeze_sha256"], frozen["tasks"][task["index"]]]
                )
                if parent and parent.get("identity") != expected:
                    raise ValueError("stopping result identity differs")
                for case, replay in record.get("replays", {}).items():
                    for trajectory in replay.get("trajectories", []):
                        if trajectory.get("status") == "complete":
                            array = safe_path(
                                root / arm / "replays", case + "/" + trajectory["array"]
                            )
                            if digest(array) != trajectory["array_sha256"]:
                                raise ValueError("saved stopping replay array differs")
                scores = record.get("clean_signal_nmse", {})
                metrics = {
                    "clean_train_nmse": scores.get("train"),
                    "clean_validation_nmse": scores.get("validation"),
                }
                recovered = record.get("status") == "complete" and all(
                    scores.get(s) is not None and scores[s] <= 1e-4
                    for s in ("train", "validation")
                )
                metrics["recovered"] = recovered
            else:
                record = saved(root / arm / "result.json")
                mesh, method = arm.split("_", 1)
                expected = content_hash(
                    [frozen["identity"], task["index"], int(mesh[-1]), method]
                )
                if record and record.get("identity") != expected:
                    raise ValueError("piecewise result identity differs")
                fit = record.get("fit", {})
                metrics = {
                    "train_nmse": (fit.get("training") or {}).get("normalized_mse"),
                    "validation_nmse": (fit.get("validation") or {}).get(
                        "normalized_mse"
                    ),
                    "clean_validation_nmse": record.get("replays", {})
                    .get("val", {})
                    .get("scores", {})
                    .get("Radau", {})
                    .get("clean_nmse"),
                }
            supervisor = saved(
                output / "attempts/task_{:03d}/supervisor.json".format(task["index"])
            )
            rows.append(
                dict(
                    metrics,
                    case=task["label"],
                    index=task["index"],
                    arm=arm,
                    recovery_action=action["mode"],
                    status=record.get("status", supervisor.get("status", "missing"))
                    if record
                    else "unavailable",
                    supervisor_status=supervisor.get("status"),
                    verified=record.get("status") == "complete",
                )
            )
    # Retained checkpoints are immutable even when another arm in the pair runs.
    for name, expected in plan["original_results"].items():
        if name in plan["excluded_from_attempt"]:
            continue
        # In-progress aggregate/checkpoint files are intentionally mutable on resume.
        terminal = any(
            t["path"] + "/" in name
            and a["mode"] == "preserve_terminal"
            and (
                name == t["path"] + "/result.json"
                or (t["path"] + "/" + a["arm"] + "/") in name
            )
            for t in plan["tasks"]
            for a in t["actions"]
        )
        if (terminal or name.endswith("/fit.json")) and digest(
            safe_path(output / "snapshot", name)
        ) != expected:
            raise ValueError("retained result or fit changed: " + name)
    counts = collections.Counter(r["status"] for r in rows)
    buckets = {}
    for row in rows:
        if plan["campaign"] == "stopping":
            kind = next(
                (
                    label
                    for suffix, label in (
                        ("alternating_collocation_sensitivity", "A+S"),
                        ("joint_collocation_sensitivity", "J+S"),
                        ("collocation_sensitivity", "C+S"),
                    )
                    if row["case"].endswith("_" + suffix)
                ),
                "other",
            )
            population = "stress" if "stress" in row["case"] else "ordinary"
        else:
            kind = "piecewise"
            population = (
                "synthetic"
                if frozen["cases"][row["index"]].get("synthetic")
                else "public"
            )
        buckets.setdefault((kind, population, row["arm"]), []).append(row)
    groups = []
    for (kind, population, arm), group in sorted(buckets.items()):
        groups.append(
            {
                "initializer": kind,
                "population": population,
                "arm": arm,
                "total": len(group),
                "verified": sum(r["verified"] for r in group),
                "recovered": sum(r.get("recovered", False) for r in group)
                if plan["campaign"] == "stopping"
                else None,
                "statuses": dict(collections.Counter(r["status"] for r in group)),
            }
        )
    report = {
        "identity": plan["identity"],
        "campaign": plan["campaign"],
        "rows": rows,
        "counts": dict(counts),
        "planned_arms": len(rows),
        "by_arm": groups,
        "timings_comparable": False,
        "numerical_settings_changed": False,
    }
    write(output / "summary.json", report)
    lines = [
        "# {} outage recovery".format(plan["campaign"].title()),
        "",
        "Original numerical code, starts and budgets. "
        "Terminal numerical failures are retained.",
        "Fresh interrupted attempts are explicit; "
        "polling retains its checkpoint budget.",
        "Outage/recovery timings are not a controlled speed comparison.",
        "",
        f"Planned arms: {len(rows)}. Status counts: {dict(counts)}.",
        "",
        "| Initializer | Cases | Arm | Total | Verified | Recovered | Status counts |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for group in groups:
        lines.append(
            "| "
            + " | ".join(
                str(group[k])
                for k in (
                    "initializer",
                    "population",
                    "arm",
                    "total",
                    "verified",
                    "recovered",
                    "statuses",
                )
            )
            + " |"
        )
    (output / "summary-short.md").write_text("\n".join(lines) + "\n")
    lines.extend(
        [
            "",
            "| Case | Arm | Recovery action | Status | "
            "Clean train NMSE | Clean validation NMSE |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row.get(k, "—"))
                for k in (
                    "case",
                    "arm",
                    "recovery_action",
                    "status",
                    "clean_train_nmse",
                    "clean_validation_nmse",
                )
            )
            + " |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "smoke",
            "run",
            "legacy-worker",
            "legacy-smoke",
            "summarize",
            "indices",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign", choices=tuple(PINS))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--legacy-repo", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "prepare":
        if any(
            getattr(args, n) is None
            for n in ("campaign", "source", "legacy_repo", "python", "dependencies")
        ):
            parser.error(
                "prepare requires campaign, source, legacy-repo, "
                "python and dependencies"
            )
        plan = prepare(args)
        print(
            json.dumps(
                {
                    "identity": plan["identity"],
                    "selected_tasks": sum(t["selected"] for t in plan["tasks"]),
                }
            )
        )
    elif args.stage == "indices":
        print(
            ",".join(
                str(t["index"]) for t in load_plan(output)["tasks"] if t["selected"]
            )
        )
    elif args.stage in ("legacy-smoke", "legacy-worker"):
        legacy_worker(output, args.stage == "legacy-smoke", args.task_index)
    elif args.stage == "summarize":
        print(json.dumps(summarize(output)["counts"]))
    else:
        run(output, args.stage == "smoke", args.task_index)


if __name__ == "__main__":

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    main()
