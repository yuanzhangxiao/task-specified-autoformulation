"""Reassess the three archived basin replies using their exact submitted factors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import basin_process_proposals as pilot
from scripts import trajectory_functions as trial
from scripts.construction_functions import file_hash

SOURCE_PLAN = "5e17a0ccc057f93ab5dd83579bcf14eb28efb3a3870d51c54f1d998d58d867f0"
SOURCE_GENERATION = "3084e8f7ca0db39bd84b88f529e9f53fb85c4eaec62463c29ed0d384968dad0a"


def replay_plan(data: dict) -> dict:
    """Bind the current interpreter to the original generation."""
    pilot._parent(data)
    return {
        "protocol": pilot.REPLAY_PROTOCOL,
        "source_plan": SOURCE_PLAN,
        "source_generation": SOURCE_GENERATION,
        "intake_sha256": pilot.INTAKE,
        "parent": pilot.PARENT,
        "seeds": list(pilot.SEEDS),
        "maximum_fit_slots": len(pilot.SEEDS),
        "maximum_generation_requests": 0,
        "runtime_source": pilot.runtime_source_hash(),
        "entrypoints": {
            name: file_hash(Path(module.__file__))
            for name, module in (("assembly", pilot), ("receipts", trial))
        },
        "replay_script": file_hash(Path(__file__)),
        "retrospective": True,
        "test_data_opened": False,
    }


def _copy_record(src: Path, dst: Path) -> None:
    """Install exact verified record bytes or check an existing checkpoint."""
    if dst.exists():
        if file_hash(src) != file_hash(dst):
            raise ValueError(f"Imported receipt changed: {dst.name}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_suffix(".replay-tmp")
    temporary.write_bytes(src.read_bytes())
    temporary.replace(dst)


def verify_provenance(root: Path, candidates: list[dict]) -> None:
    """Tie every replay candidate to the pinned generation and its original reply."""
    old_plan = sealed_read(root / "original" / "plan.json")
    old_gen = sealed_read(root / "original" / "generation.json")
    if (
        old_plan["artifact_sha256"] != SOURCE_PLAN
        or old_plan["intake_sha256"] != pilot.INTAKE
        or old_gen["artifact_sha256"] != SOURCE_GENERATION
        or old_gen["plan"] != SOURCE_PLAN
        or len(candidates) != len(old_gen["candidates"])
    ):
        raise ValueError("Replay provenance differs from the expected live campaign")
    for i, candidate in enumerate(candidates):
        old = sealed_read(root / "original" / "candidates" / f"{i:03}.json")
        if (
            candidate["source_candidate"] != old_gen["candidates"][i]
            or candidate["source_candidate"] != old["artifact_sha256"]
            or candidate["reply_sha256"] != old["reply_sha256"]
            or candidate["seed"] != old["seed"]
            or candidate["graph"] != old["graph"]
            or old["plan"] != SOURCE_PLAN
            or old["index"] != i
        ):
            raise ValueError("Replay candidate differs from original evidence")


def replay(archive: Path, root: Path) -> dict:
    """Verify the live archive before writing the separate replay cohort."""
    source, dest = archive.resolve(), root.resolve()
    if dest.is_relative_to(source) or source.is_relative_to(dest):
        raise ValueError("Use separate source and replay directories")
    plan = sealed_read(source / "plan.json")
    data = sealed_read(source / "intake.json")
    generation = trial._frozen_candidates(source, plan)
    prompt = sealed_read(source / "prompt.json")
    if (
        plan["artifact_sha256"] != SOURCE_PLAN
        or generation["artifact_sha256"] != SOURCE_GENERATION
        or plan["intake_sha256"] != data["artifact_sha256"]
        or data["artifact_sha256"] != pilot.INTAKE
        or plan["request_sha256"]
        != content_hash([prompt["system"], prompt["user"], prompt["schema"]])
        or plan["seeds"] != list(pilot.SEEDS)
    ):
        raise ValueError(
            "Expected the registered live basin plan, intake, prompt and generation"
        )
    records = []
    for i, seed in enumerate(pilot.SEEDS):
        original = trial._read_candidate(source, plan, i)
        folder = Path("steps") / f"slot-{i:03}"
        reply = sealed_read(source / folder / "reply.json")
        call = folder / "calls" / (reply["request_hash"] + ".json")
        if (
            original["artifact_sha256"] != generation["candidates"][i]
            or original["graph"] != pilot.PARENT
            or original["seed"] != seed
            or original.get("reply_sha256") != reply["artifact_sha256"]
            or reply["status"] != "visible"
            or file_hash(source / call) != reply["physical_receipt_sha256"]
            or list((source / folder / "calls").glob("*.json")) != [source / call]
        ):
            raise ValueError("Live slot, reply or physical receipt differs")
        records.append((original, reply, folder, call))

    with _lock(dest / "generation-lock"):
        new_plan = sealed_write(dest / "plan.json", replay_plan(data))
        _copy_record(source / "intake.json", dest / "intake.json")
        for name in ("plan.json", "generation.json", "prompt.json"):
            _copy_record(source / name, dest / "original" / name)
        candidates = []
        for i, (old, reply, folder, call) in enumerate(records):
            path = Path("candidates") / f"{i:03}.json"
            _copy_record(source / path, dest / "original" / path)
            _copy_record(source / folder / "reply.json", dest / folder / "reply.json")
            _copy_record(source / call, dest / call)
            try:
                outcome = pilot.assemble(data, reply["value"])
            except (
                ValueError,
                SyntaxError,
                ModelValidationError,
                RecursionError,
            ) as exc:
                outcome = {"status": "invalid_formula", "reason": str(exc)}
            candidates.append(
                sealed_write(
                    dest / path,
                    {
                        "plan": new_plan["artifact_sha256"],
                        "index": i,
                        "seed": old["seed"],
                        "graph": old["graph"],
                        "source_candidate": old["artifact_sha256"],
                        "original_status": old["status"],
                        "reply_sha256": old["reply_sha256"],
                        **outcome,
                    },
                )
            )
        sealed_write(
            dest / "generation.json",
            {
                "plan": new_plan["artifact_sha256"],
                "candidates": [c["artifact_sha256"] for c in candidates],
            },
        )
    return pilot.status(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.archive, args.root), indent=2))


if __name__ == "__main__":
    main()
