"""Public-only baseline fitting, endpoint freeze, and blind forecast exchange."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

CELLS = tuple(f"acquisition_{i:02d}" for i in range(4))
SEEDS = (0, 1, 2)
FILES = ("proposer_prompt.txt", "specification.json", "train.json", "val.json")
SEMANTICS = {"continuous_ode", "discrete_map"}
TARGETS = ("Gp", "I")
INPUTS = {"meal_rate_g_per_min", "insulin_pmol_per_kg_min"}


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def fingerprint(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    """Read a canonical content-addressed interchange record."""
    data = json.loads(path.read_bytes())
    if set(data) != {"value", "sha256"} or fingerprint(data["value"]) != data["sha256"]:
        raise ValueError(f"record digest differs at {path}")
    return data["value"]


def publish(path: Path, value: dict) -> dict:
    """Publish a complete record once, allowing concurrent identical retries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"sha256": fingerprint(value), "value": value}
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        name = Path(tmp.name)
        try:
            tmp.write(canonical(data) + b"\n")
            tmp.flush()
            os.fsync(tmp.fileno())
            try:
                os.link(name, path)
            except FileExistsError:
                if read(path) != value:
                    raise ValueError(
                        f"different record already exists at {path}"
                    ) from None
        finally:
            name.unlink(missing_ok=True)
    return value


@contextmanager
def task_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def verify(root: Path) -> dict:
    """Check the packaged file bytes against Full's four execution cells."""
    release = read(root / "release.json")
    if tuple(sorted(release["cells"])) != CELLS or release["seeds"] != list(SEEDS):
        raise ValueError("unexpected acquisition or seed roster")
    for cell in CELLS:
        directory = root / "public" / cell
        expected = release["cells"][cell]["public_files"]
        names = {p.name for p in directory.iterdir()}
        if set(expected) != set(FILES) or names != set(FILES):
            raise ValueError(f"unexpected public files at {cell}")
        if {f: file_hash(directory / f) for f in FILES} != expected:
            raise ValueError(f"public data differ from Full at {cell}")
    return release


def task_key(cell: str, seed: int) -> str:
    return f"{cell}-s{seed}"


def make_plan(root: Path, work: Path, config_path: Path, module: str) -> dict:
    """Bind one method's 12 fits, configuration, and trusted adapter before training."""
    release = verify(root)
    cfg = json.loads(config_path.read_bytes())
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", cfg["method"]):
        raise ValueError("invalid method identifier")
    if cfg["semantics"] not in SEMANTICS or any(
        not cfg.get(k) for k in ("implementation_revision", "budget", "selection_rule")
    ):
        raise ValueError(
            "declare native semantics, revision, budget and selection rule"
        )
    source = Path(importlib.import_module(module).__file__)
    tasks = [
        {
            "key": task_key(cell, s),
            "cell": cell,
            "seed": s,
            "public_files": release["cells"][cell]["public_files"],
        }
        for cell in CELLS
        for s in SEEDS
    ]
    return publish(
        work / "plan.json",
        {
            "format": "t2-noise-plan-1",
            "release_sha256": fingerprint(release),
            "config": cfg,
            "adapter_module": module,
            "adapter_sha256": file_hash(source),
            "tasks": tasks,
        },
    )


def load_adapter(plan: dict):
    adapter = importlib.import_module(plan["adapter_module"])
    if file_hash(Path(adapter.__file__)) != plan["adapter_sha256"]:
        raise ValueError("adapter changed after plan")
    return adapter


def check_terminal(directory: Path, plan: dict, task: dict) -> dict:
    r = read(directory / "terminal.json")
    if (
        r["task"] != task
        or r["plan_sha256"] != fingerprint(plan)
        or r["receipt_sha256"] != file_hash(directory / "fit_receipt.json")
    ):
        raise ValueError("terminal result belongs to different data, plan, or receipt")
    if r["status"] == "ready":
        if r["model_sha256"] != file_hash(directory / "model.bin"):
            raise ValueError("selected model changed")
        if not math.isfinite(r["validation_nmse"]) or r["validation_nmse"] < 0:
            raise ValueError("expected a finite validation NMSE")
    elif (
        r["status"] != "no_model"
        or not r.get("reason")
        or r["model_sha256"] is not None
        or r["validation_nmse"] is not None
    ):
        raise ValueError("invalid terminal failure")
    return r


def fit_task(root: Path, work: Path, plan: dict, task: dict, adapter) -> None:
    directory = work / "tasks" / task["key"]
    with task_lock(directory):
        if (directory / "terminal.json").is_file():
            check_terminal(directory, plan, task)
            return
        if (work / "freeze.json").exists():
            raise ValueError("fit requested after freeze")
        result = adapter.fit(
            task, root / "public" / task["cell"], directory, plan["config"]
        )
        receipt = json.loads((directory / "fit_receipt.json").read_bytes())
        if any(receipt[k] != task[k] for k in ("cell", "seed", "public_files")):
            raise ValueError("native fitting receipt has different inputs or seed")
        ready = result["status"] == "ready"
        if ready:
            val = result["validation_nmse"]
            if type(val) not in (int, float) or not math.isfinite(val) or val < 0:
                raise ValueError(
                    "ready model lacks a finite nonnegative validation NMSE"
                )
            if (directory / "model.bin").stat().st_size <= 0:
                raise ValueError("empty selected model")
        elif (
            result["status"] != "no_model"
            or not str(result.get("reason") or "").strip()
        ):
            raise ValueError("unrecognized fitting disposition")
        publish(
            directory / "terminal.json",
            {
                "task": task,
                "plan_sha256": fingerprint(plan),
                "status": result["status"],
                "reason": None if ready else result["reason"],
                "validation_nmse": result["validation_nmse"] if ready else None,
                "model_sha256": file_hash(directory / "model.bin") if ready else None,
                "receipt_sha256": file_hash(directory / "fit_receipt.json"),
            },
        )


def freeze(work: Path) -> dict:
    plan = read(work / "plan.json")
    rows = [check_terminal(work / "tasks" / t["key"], plan, t) for t in plan["tasks"]]
    frozen = {"format": "t2-noise-freeze-1", "plan": plan, "rows": rows}
    validate_freeze(frozen)
    return publish(work / "freeze.json", frozen)


def validate_freeze(frozen: dict) -> None:
    """Require all 12 planned rows, including terminal failures, in fixed order."""
    plan = frozen["plan"]
    keys = [(c, s) for c in CELLS for s in SEEDS]
    if (
        frozen["format"] != "t2-noise-freeze-1"
        or plan["format"] != "t2-noise-plan-1"
        or plan["config"]["semantics"] not in SEMANTICS
        or len(frozen["rows"]) != len(keys)
        or len(plan["tasks"]) != len(keys)
    ):
        raise ValueError("incomplete or invalid frozen plan")
    for (c, s), task, row in zip(keys, plan["tasks"], frozen["rows"], strict=True):
        if (task["cell"], task["seed"], task["key"]) != (c, s, task_key(c, s)):
            raise ValueError("changed seed, cell, or row order")
        if row["task"] != task or row["plan_sha256"] != fingerprint(plan):
            raise ValueError("endpoint from a different plan")
        if not re.fullmatch(r"[a-f0-9]{64}", row["receipt_sha256"] or ""):
            raise ValueError("missing fit receipt identity")
        if row["status"] == "ready":
            if not re.fullmatch(r"[a-f0-9]{64}", row["model_sha256"] or ""):
                raise ValueError("missing model identity")
            val = row["validation_nmse"]
            if type(val) not in (int, float) or not math.isfinite(val) or val < 0:
                raise ValueError("invalid validation NMSE")
        elif (
            row["status"] != "no_model"
            or not row["reason"]
            or row["model_sha256"] is not None
            or row["validation_nmse"] is not None
        ):
            raise ValueError("nonterminal or undeclared result")


def checked_curve(value: dict, n: int) -> dict:
    if set(value) != set(TARGETS):
        raise ValueError("expected both Gp and I")
    out = {}
    for channel in TARGETS:
        values = list(value[channel])
        if len(values) != n or any(not math.isfinite(float(x)) for x in values):
            raise ValueError("forecast must be finite on the entire time grid")
        out[channel] = [float(x) for x in values]
    return out


def forecast(
    work: Path, queries: dict, frozen: dict, row: dict, out: Path, adapter
) -> None:
    """Run a frozen adapter on permitted observations only and checkpoint each pair."""
    plan, task = frozen["plan"], row["task"]
    directory = work / "tasks" / task["key"]
    if check_terminal(directory, plan, task) != row:
        raise ValueError("terminal result changed after freeze")
    if row["status"] != "ready":
        return
    bound = {
        "freeze_sha256": fingerprint(frozen),
        "queries_sha256": fingerprint(queries),
        "key": task["key"],
        "model_sha256": row["model_sha256"],
    }
    methods = (
        ("Radau", "BDF")
        if plan["config"]["semantics"] == "continuous_ode"
        else ("native",)
    )
    for pair in queries["cells"][task["cell"]]:
        batch = out / task["key"]
        path = batch / (pair["id"] + ".json")
        with task_lock(batch):
            if path.is_file():
                saved = read(path)
                if saved["binding"] != bound or saved["id"] != pair["id"]:
                    raise ValueError("forecast checkpoint belongs to a different task")
                continue
            try:
                curves = {}
                for part in ("perturbed", "control"):
                    inputs = pair[part]
                    curves[part] = {
                        m: checked_curve(
                            adapter.predict(
                                directory / "model.bin", inputs, m, plan["config"]
                            ),
                            len(inputs["time"]),
                        )
                        for m in methods
                    }
                result = {"status": "complete", "curves": curves}
            except (ArithmeticError, ValueError, TimeoutError) as exc:
                result = {
                    "status": "unresolved",
                    "reason": str(exc)[:512] or type(exc).__name__,
                }
            publish(path, {"binding": bound, "id": pair["id"], **result})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in ("verify", "plan", "fit", "freeze", "predict"):
        p = sub.add_parser(cmd)
        if cmd in ("verify", "plan", "fit"):
            p.add_argument("--data", type=Path, default=Path("."))
        if cmd != "verify":
            p.add_argument("--work", type=Path, required=True)
        if cmd == "plan":
            p.add_argument("--config", type=Path, required=True)
            p.add_argument("--adapter", required=True)
        if cmd in ("fit", "predict"):
            p.add_argument("--index", type=int, choices=range(12))
        if cmd == "predict":
            p.add_argument("--queries", type=Path, required=True)
            p.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    if a.command == "verify":
        print(
            "Verified 16 public files in four ACES cells.", fingerprint(verify(a.data))
        )
    elif a.command == "plan":
        plan = make_plan(a.data, a.work, a.config, a.adapter)
        print("Prepared 12 tasks.", fingerprint(plan))
    elif a.command == "freeze":
        print("Frozen 12 terminal results.", fingerprint(freeze(a.work)))
    else:
        plan = read(a.work / "plan.json")
        adapter = load_adapter(plan)
        indices = range(12) if a.index is None else [a.index]
        if a.command == "fit":
            if fingerprint(verify(a.data)) != plan["release_sha256"]:
                raise ValueError("fit release differs from plan")
            for i in indices:
                fit_task(a.data, a.work, plan, plan["tasks"][i], adapter)
        else:
            frozen, queries = read(a.work / "freeze.json"), read(a.queries)
            validate_freeze(frozen)
            if (
                frozen["plan"] != plan
                or queries["release_sha256"] != plan["release_sha256"]
                or queries["format"] != "t2-noise-blind-1"
            ):
                raise ValueError("forecast inputs differ from frozen release")
            a.out.mkdir(parents=True, exist_ok=True)
            for i in indices:
                forecast(a.work, queries, frozen, frozen["rows"][i], a.out, adapter)


if __name__ == "__main__":
    main()
