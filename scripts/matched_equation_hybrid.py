"""Freeze public training packets for one live 20B alternative per graph/fold."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.trajectory_function_fitting import validate_formula
from scripts import matched_equation_hybrid_provider as provider
from scripts import trajectory_function_prompts as prompts
from scripts.revision_round_requests import body

SCRIPTS = Path(__file__).resolve().parent
EXTRA = (
    " Slot A has already been retained. For the second slot, either select "
    "shortlist_id B or write one different complete model. The graph, variables "
    "and initializer dependencies stay fixed. Use scientific knowledge as well "
    "as the training observations to choose your function forms. The parameter "
    "values you give are starting values for fitting."
)


def make_body(packet: dict) -> dict:
    """Bind the allowed prompt fields and shape scores to the fitting fold."""
    graph, menu = packet["graph"], packet["shape"]
    rows = packet["train_rows"]
    ids = [r["trajectory_id"] for r in rows]
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(not name.startswith("train_") for name in ids)
        or menu["evidence"]["split"] != "train"
        or sorted(menu["evidence"]["events"]) != sorted(ids)
    ):
        raise ValueError("Hybrid evidence must refer to exactly the fitting events")
    for candidate in menu["candidates"]:
        validate_formula(graph, candidate)
    task = packet["public_task"]
    if not isinstance(task, str):
        raise ValueError("Provide only public task prose")
    instruction, user, schema = prompts.request(
        graph,
        {"public_brief": {"scientific_context": task}},
        prompts.observations(rows),
        menu,
        hybrid=True,
    )
    schema_json = schema.model_json_schema()
    schema_json["$defs"]["SelectFamily"]["properties"]["shortlist_id"]["enum"] = ["B"]
    request = body(instruction + EXTRA, user, schema_json, provider.SEED)
    request["max_tokens"] = provider.MAX_TOKENS
    return request


def _files() -> dict:
    # This deployed worker has the existing H100 script's CLI and filename.
    return {
        "run_revision_round_20b.sh": SCRIPTS / "hpc/run_revision_round_20b.sh",
        "revision_round_provider.py": SCRIPTS / "matched_equation_hybrid_provider.py",
        "revision_round_io.py": SCRIPTS / "revision_round_provider.py",
    }


def prepare(root: Path) -> dict:
    """Write a portable, train-only job bundle; refuse to replace a different plan."""
    paths = sorted((root / "packets").glob("*.json"))
    if len(paths) != provider.EXPECTED:
        raise ValueError(f"Expected {provider.EXPECTED} sealed graph/fold packets")
    folder = root / "hybrid"
    with _lock(folder):
        calls = {}
        for path in paths:
            packet = sealed_read(path)
            label = packet["panel_id"]
            if not provider.LABEL.fullmatch(label) or label != path.stem:
                raise ValueError("Panel name differs from the sealed file")
            calls[label] = make_body(packet)
        files = _files()
        hashes = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in files.items()
        }
        value = sealed_write(
            folder / "generate.json",
            {
                "protocol": provider.PROTOCOL,
                "requests": calls,
                "sha256": provider.digest(calls),
                "revision": provider.REVISION,
                "worker_sha256": hashes,
            },
        )
        for name, path in files.items():
            destination = folder / name
            if not destination.exists():
                shutil.copyfile(path, destination)
            if hashlib.sha256(destination.read_bytes()).hexdigest() != hashes[name]:
                raise ValueError(f"Staged worker differs: {name}")
        provider.load_manifest(folder)
        return value


def _expressions(formula: dict) -> dict:
    """Ignore whitespace and parameter starts when checking for a literal repetition."""
    return {
        field: {
            key: ast.dump(ast.parse(expr, mode="eval"))
            for key, expr in formula[field].items()
        }
        for field in ("equations", "initializers")
    }


def result_for(packet: dict, root: Path) -> dict:
    """Return the second hybrid slot's actual status and its verified formula."""
    label = packet["panel_id"]
    if not provider.LABEL.fullmatch(label):
        raise ValueError("Unsafe panel label")
    folder = root / "hybrid"
    path = folder / "generate-responses" / f"{label}.json"
    if not path.is_file():
        return {"status": "pending", "formula": None}
    manifest = sealed_read(folder / "generate.json")
    request = make_body(packet)
    if request != manifest["requests"].get(label):
        raise ValueError("Scientific packet differs from the actual provider request")
    receipt = json.loads(path.read_text())
    if (
        receipt["manifest_sha256"] != manifest["sha256"]
        or receipt["request_sha256"] != provider.digest(request)
        or receipt["revision"] != provider.REVISION
    ):
        raise ValueError("Provider receipt differs from the frozen request")
    status = receipt["status"]
    if status in {"reserved", "provider_error"}:
        return {"status": status, "formula": None, "detail": receipt.get("error")}
    if status != "responded" or receipt["response_sha256"] != provider.digest(
        receipt["response"]
    ):
        raise ValueError("Unrecognized or modified provider receipt")
    response = receipt["response"]
    try:
        if response["model"] != provider.MODEL or len(response["choices"]) != 1:
            raise ValueError("Different model or response count")
        choice = response["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError(f"Generation stopped with {choice['finish_reason']!r}")
        reply = (
            prompts.response_schema(packet["graph"], True)
            .model_validate_json(choice["message"]["content"])
            .model_dump()
        )
        if reply["candidate"].get("shortlist_id") == "A":
            raise ValueError("Slot A is already retained")
        shortlist = packet["shape"]["candidates"]
        formula, action = prompts.to_formula(reply, shortlist)
        formula = validate_formula(packet["graph"], formula)
        if _expressions(shortlist[0]) == _expressions(formula):
            raise ValueError("Repeats slot A's equations and initializers")
        return {"status": "eligible", "formula": formula, "action": action}
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return {"status": "invalid", "formula": None, "reason": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "collect"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        value = prepare(args.root)
        print(json.dumps({"calls": len(value["requests"]), "sha256": value["sha256"]}))
        return
    packets = sorted((args.root / "packets").glob("*.json"))
    if len(packets) != provider.EXPECTED:
        raise ValueError("Incomplete packet cohort")
    print(
        json.dumps(
            {p.stem: result_for(sealed_read(p), args.root)["status"] for p in packets}
        )
    )


if __name__ == "__main__":
    main()
