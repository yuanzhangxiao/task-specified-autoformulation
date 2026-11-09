"""Authenticate Astra's five T2 training packets without executing archive contents."""

from __future__ import annotations

import hashlib
import json
import tarfile
from io import BytesIO
from pathlib import Path, PurePosixPath

from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.rebuttal.prefit_replay import content_hash
from autoformalism.schemas.prefix_initialization import PrefixFitProblem, PrefixPolicy
from autoformalism.schemas.public_fitting import PublicSplit

ARCHIVE = "6446f438dd3a391c4f4915e7c630179c3dd5bc9477b507ed9bf2e6c9ed2f0653"
SOURCE_PLAN = "cefdfbe73bbcc57d1606f3d8e8833fe637aa17c1a447639f5b912ad878649a30"
TRAINING = "169e2b72a1ba4145ac4ec5911875e9d7e064a525f7a1ace92cc2b151efe70795"
BRANCHES = ("c15-4", "c15-0", "c15-2", "c15-3", "c21-2")
CONTROL = "c15-4-zero-basal"
TASKS = (BRANCHES[0], CONTROL, *BRANCHES[1:])


def _sealed(value: dict) -> dict:
    if value.get("artifact_sha256") != content_hash(
        {k: v for k, v in value.items() if k != "artifact_sha256"}
    ):
        raise ValueError("construction receipt digest differs")
    return value


def load_archive(path: Path) -> dict:
    """Read the pinned package, authenticate every member, and select training data."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ARCHIVE:
        raise ValueError("unexpected Astra fitting archive digest")
    files = {}
    with tarfile.open(fileobj=BytesIO(raw), mode="r:gz") as archive:
        for item in archive:
            name = PurePosixPath(item.name)
            if (
                not item.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or name.parts[0] != "t2-easy-orion"
            ):
                raise ValueError("invalid breadth archive member")
            relative = str(name.relative_to("t2-easy-orion"))
            if relative in files:
                raise ValueError("duplicate archive member")
            files[relative] = archive.extractfile(item).read()
    inventory = {}
    for line in files.pop("SHA256SUMS").decode().splitlines():
        digest, name = line.split("  ", 1)
        inventory[name] = digest
    if inventory != {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}:
        raise ValueError("package member inventory differs")

    def read(name):
        return _sealed(json.loads(files["returned/" + name + ".json"]))

    source, plan = read("inputs"), read("plan")
    if (
        plan["artifact_sha256"] != SOURCE_PLAN
        or plan["intake_sha256"] != source["artifact_sha256"]
        or source["test_data_opened"] is not False
        or plan["test_data_opened"] is not False
    ):
        raise ValueError("source plan or training boundary differs")
    training = PublicSplit.model_validate(source["training"])
    prefix = PrefixPolicy.model_validate(source["fitting_contract"]["prefix_policy"])
    if (
        training.name != "train"
        or content_sha256(training) != TRAINING
        or source["fitting_contract"]["training_sha256"] != TRAINING
        or prefix.duration != 8
    ):
        raise ValueError("public training or prefix contract differs")
    controls = json.loads(files["fixed-training.json"])
    if (
        controls["training_sha256"] != TRAINING
        or not controls["fixed_parameters"]
        or controls["optimizer_calls"] != 0
    ):
        raise ValueError("source references must be fixed-parameter training replays")
    models = {}
    for branch in BRANCHES:
        packet, receipt = read("handoff/" + branch), read("results/" + branch)
        problem = PrefixFitProblem.model_validate(packet["problem"])
        origin = packet["origin"]
        if (
            packet["branch"] != branch
            or origin != branch.split("-")[0]
            or packet["plan_sha256"] != SOURCE_PLAN
            or packet["training_sha256"] != TRAINING
            or receipt["artifact_sha256"] != packet["source_receipt"]
            or packet["problem"] != receipt["problem"]
            or content_hash(packet["problem"]) != packet["model_view"]["problem_sha256"]
            or packet["prefix_policy"] != source["fitting_contract"]["prefix_policy"]
        ):
            raise ValueError("fitting packet identity differs")
        parent = (
            source["c15_training_parameters"]
            if origin == "c15"
            else source["c21"]["parameters_including_training_roots"]
        )
        if any(
            value != parent[n]
            for n, value in problem.parameter_guesses.items()
            if n in parent
        ):
            raise ValueError("child changed an inherited warm coefficient")
        warm = dict(problem.parameter_guesses) | {
            n: v for n, v in parent.items() if n.startswith("trajectory:")
        }
        models[branch] = {
            "problem": packet["problem"],
            "warm": warm,
            "origin": origin,
            "receipt": packet["artifact_sha256"],
            "model_sha256": content_sha256(problem),
            "construction_problem_sha256": packet["model_view"]["problem_sha256"],
        }
    if len({m["model_sha256"] for m in models.values()}) != len(BRANCHES):
        raise ValueError("five distinct child problems required")
    return {
        "archive_sha256": ARCHIVE,
        "source_plan": SOURCE_PLAN,
        "training": source["training"],
        "models": models,
        "parent_c15": source["c15"]["problem"],
        "parent_c21": source["c21"]["problem"],
        "prefix": prefix.model_dump(mode="json"),
        "boundary": source["c15"]["boundary_witness"],
        "glucose_windows": source["glucose_windows"],
        "fixed": controls["summary"],
        "fixed_witness": controls["common_witness"],
        "fixed_extension": controls["zero_input_continuation_150g"],
        "training_sha256": TRAINING,
        "validation_or_test_access": False,
    }
