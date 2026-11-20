"""Pinned public-training parents for the graph critic comparison.

The development parent and three original scaled-readout constructions share the
T2-easy training data. The comparison cohort contains every non-15 parent in
the Nov 3 targeted fitting roster. Restricted saturation trials are labeled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.expressions import CompiledModel, compile_candidate
from autoformalism.fitting.initialization_graph import apply_initialization_graph
from autoformalism.fitting.public_fitting import content_sha256, unpack_split
from autoformalism.fitting.simulation import trajectory_initial_state
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit
from autoformalism.staged_topology import content_hash

ROSTER = ("c15-2", "candidate-10", "candidate-12", "candidate-14")
TRANSFER_CONTROL = "c15-3"
SIBLING = Path("output/t2-breadth-independent-starts-20261109")
SIBLING_MODELS = "d97bf553c753d4725b66b3ae43aeb20395ef82f9f0010d9d1166561d5dbfb4b4"
PACKET = Path("output/t2-fitted-breadth-20b-results-20261119/orion-fitting-handoff.zip")
DEVELOPMENT = Path("output/t2-20b-revision-refits-20261120")
TARGETED = Path("output/construction-portfolio-targeted-review-20261103")
PROFILE = Path("output/construction-portfolio-profile-review-20261102")
TRAINING = "169e2b72a1ba4145ac4ec5911875e9d7e064a525f7a1ace92cc2b151efe70795"
PARENT_SOURCE = "419a13809e7d82abb0fb90bed11b4eaff04ceee4bd9bcad482258d6fd2e974a8"
PROBLEMS = "65217d6e461ea0e767b113f7a13b729c3ef500c5ce10f7baf8c1bc439a9bf03a"
# Seals of the two complete fit terminals per parent, in prespecified order.
TERMINALS = {
    TRANSFER_CONTROL: {
        "short-memory": (
            "d6f76268d399f2e6a9a3af230260dbb9a062588dc48c09549f0a0e24a0fcd990"
        ),
        "long-memory": (
            "e1d4ad9b4b18cdac20f9aee76d729654dcfcb1709fc4ffc14bb05363e9b4f83d"
        ),
    },
    "c15-2": {
        "supplied": "7b1bb8c16053ee1cab7eaa6b097daf7e7c0dc4a1ec09088001d0a1ca1a9a0fc7",
        "activated": "30ec651b046203f230d0fb82bf49bd8e3b3e098c72e2919cdfe4e8c3f4e4e9ec",
    },
    "candidate-10": {
        "saturation-0p1": (
            "ffe1e231faa185c6b045b99acdd5194905bf74ef533b001b5c2ce84f69e3938e"
        ),
        "saturation-10": (
            "2d5482aa3bada7eabbbde331060b56f4ef9cbeb6ad9e747d68109d2355428d58"
        ),
    },
    "candidate-12": {
        "incumbent": "2c9d36512d718a2c83ea34bca719ebfd1f387888fdafbd5866ae94b45a26521b",
        "action_tau_10": (
            "8e04ef3f7a82eafc79f6018c9959f2c68b04e6d0195b490a1a420e576664f47c"
        ),
    },
    "candidate-14": {
        "saturation-0p1": (
            "8342bf0955364be1766333d7f3238dd7c3afb61beda96f2ebb2294c6f1ed06ea"
        ),
        "saturation-10": (
            "b54a2d92d5e9c2b34820315a8865dd2d15a5c193376136476178e642e2c20372"
        ),
    },
}


@dataclass(frozen=True)
class FitPoint:
    """Physical RHS parameters and realized t0 states in model state order."""

    parameters: dict[str, float]
    initial_states: dict[str, list[float]]
    training_nmse: float
    search_status: dict


@dataclass(frozen=True)
class ParentData:
    """Only the frozen parent, public evidence and numerical provenance."""

    name: str
    problem: PrefixFitProblem
    model: CompiledModel
    training: PublicSplit
    starts: dict[str, FitPoint]
    public_prompt: str
    provenance: dict


def _terminal_path(name: str, label: str) -> Path:
    if name == "c15-2":
        return DEVELOPMENT / name / label / "fit.json"
    if name == TRANSFER_CONTROL:
        return SIBLING / label / "result.json"
    base = PROFILE if name == "candidate-12" else TARGETED
    return base / "results" / f"{name}-{label}" / "result.json"


def source_paths(name: str, repo: Path = Path(".")) -> list[Path]:
    """Files needed to reproduce intake for the named frozen parent."""
    if name not in (*ROSTER, TRANSFER_CONTROL):
        raise ValueError("unknown frozen parent")
    paths = [PACKET]
    if name == TRANSFER_CONTROL:
        paths.append(SIBLING / "fitted-models.json")
    elif name != "c15-2":
        paths.append(TARGETED / "inputs.json")
    paths.extend(_terminal_path(name, label) for label in TERMINALS[name])
    return [repo / path for path in paths]


def _pinned(path: Path, expected: str) -> dict:
    value = read_seal(path)
    if content_sha256(value) != expected:
        raise ValueError(f"frozen source changed: {path.name}")
    return value


def _materialize_point(
    model: CompiledModel,
    lowered: CompiledModel,
    root_names: list[str],
    training: PublicSplit,
    point: dict,
    status: dict,
) -> FitPoint:
    """Apply the fitted initializer graph once using only each row's first sample."""
    if training.name != "train":
        raise ValueError("graph critic requires public training")
    rows = unpack_split(training).trajectories
    nonroots = set(lowered.parameter_names).difference(root_names)
    expected = nonroots | {
        f"trajectory:{row.trajectory_id}:{k}" for row in rows for k in root_names
    }
    values = point["parameters"]
    if set(values) != expected or not np.isfinite(list(values.values())).all():
        raise ValueError("incomplete or nonfinite original fitted parameter vector")
    shared = {k: values[k] for k in nonroots}
    initials = {}
    for row in rows:
        local = {k: values[f"trajectory:{row.trajectory_id}:{k}"] for k in root_names}
        initials[row.trajectory_id] = trajectory_initial_state(
            lowered, row, {}, parameters=shared | local
        ).tolist()
    return FitPoint(
        {k: shared[k] for k in model.parameter_names},
        initials,
        float(point["training_nmse"]),
        status,
    )


def _endpoint(name: str, label: str, terminal: dict, problem_sha: str) -> tuple:
    if terminal["status"] != "complete":
        raise ValueError("missing completed fit terminal")
    if name in ("c15-2", TRANSFER_CONTROL):
        if name == TRANSFER_CONTROL:
            if terminal["identity"]["task"] != label or terminal["branch"] != name:
                raise ValueError("sibling fit task differs")
            point = terminal["best_admissible"]
            if not point["two_original_solvers_verified"]:
                raise ValueError("unverified sibling fit")
            last = terminal["history"][-1]
            cap = last["call_cap"]
        else:
            if terminal["identity"]["task"] != f"{name}/{label}":
                raise ValueError("development fit task differs")
            point = terminal["best"]["endpoint"]
            last = terminal["blocks"][-1]
            cap = last["cap"]
        if content_sha256(point["parameters"]) != point["parameter_sha256"]:
            raise ValueError("fitted parameters differ")
        return point, {
            "search_scope": "local_free_fit",
            "stop_reason": last["solver"]["message"],
            "budget_exhausted": last["calls"] >= cap,
            "fixed_parameters": {},
        }

    task = terminal["identity"]["task"]
    if (
        task["candidate_id"] != name
        or task["seed"] != label
        or task["problem_sha256"] != problem_sha
        or terminal["test_data_opened"] is not False
    ):
        raise ValueError("historical fit and original equation identity differ")
    fixed = task.get("experiment", {}).get("fixed", {})
    if fixed:
        # A row's incumbent can be inherited from another fit. Use this
        # fixed-coordinate trial's own verified point for the comparison.
        point = terminal["trial_endpoint"]
    else:
        if terminal["retained_origin"] != "endpoint_verification":
            raise ValueError("free-fit endpoint was inherited from another trial")
        point = terminal["retained"]
    if not point["original_equations_verified"] or any(
        point["parameters"][k] != v for k, v in fixed.items()
    ):
        raise ValueError("unverified or different restricted fit")
    return point, {
        "search_scope": "fixed_coordinate_fit" if fixed else "local_free_fit",
        "stop_reason": terminal.get("search_termination")
        or terminal["search_stop_reason"],
        "budget_exhausted": terminal["search_budget_exhausted"],
        "fixed_parameters": fixed,
    }


def load_parent(name: str, repo: Path = Path(".")) -> ParentData:
    """Restore a sealed parent and its original fitted training initials."""
    paths = source_paths(name, repo)
    with ZipFile(paths[0]) as archive:
        # Read just the pinned parent member. The ZIP also holds revised children.
        member = archive.getinfo("parent-and-training.json")
        if member.file_size > 8_000_000:
            raise ValueError("unexpected public parent member size")
        parent = json.loads(archive.read(member))
    if (
        parent.pop("artifact_sha256") != PARENT_SOURCE
        or content_hash(parent) != PARENT_SOURCE
    ):
        raise ValueError("public parent member differs")
    raw_training = parent["training"]
    if raw_training["name"] != "train" or content_sha256(raw_training) != TRAINING:
        raise ValueError("public training identity differs")
    training = PublicSplit.model_validate(raw_training)
    if name == "c15-2":
        raw_problem = parent["parent"]["problem"]
    elif name == TRANSFER_CONTROL:
        siblings = _pinned(repo / SIBLING / "fitted-models.json", SIBLING_MODELS)
        rows = {r["task"]: r for r in siblings["models"] if r["branch"] == name}
        if siblings["training_sha256"] != TRAINING or set(rows) != set(TERMINALS[name]):
            raise ValueError("sibling control has different training or fit tasks")
        raw_problem = rows["short-memory"]["problem"]
        if raw_problem != rows["long-memory"]["problem"]:
            raise ValueError("sibling control must use the same equation for both fits")
    else:
        data = _pinned(repo / TARGETED / "inputs.json", PROBLEMS)
        if (
            data["test_data_opened"] is not False
            or content_sha256(data["training"]) != TRAINING
        ):
            raise ValueError("fitting roster split differs")
        records = [
            r for r in data["records"] if r["source_task"]["candidate_id"] == name
        ]
        if len(records) != 1:
            raise ValueError(
                "frozen roster must contain one original problem per parent"
            )
        raw_problem = records[0]["problem"]

    problem_sha = content_sha256(raw_problem)
    problem = PrefixFitProblem.model_validate(raw_problem)
    model = compile_candidate(problem.base_candidate, problem.context)
    lowered, _, info = apply_initialization_graph(model, problem.graph)
    starts = {}
    for label, expected in TERMINALS[name].items():
        terminal = _pinned(repo / _terminal_path(name, label), expected)
        point, status = _endpoint(name, label, terminal, problem_sha)
        if name == TRANSFER_CONTROL and (
            rows[label]["model_sha256"] != problem_sha
            or rows[label]["retained"]["parameter_sha256"] != point["parameter_sha256"]
        ):
            raise ValueError("sibling control fit does not match its original model")
        starts[label] = _materialize_point(
            model, lowered, info["trajectory_parameter_names"], training, point, status
        )
    provenance = {
        "development_exposure": name in ("c15-2", TRANSFER_CONTROL),
        "problem_sha256": problem_sha,
        "training_sha256": TRAINING,
        "parent_source_sha256": PARENT_SOURCE,
        "fit_terminal_sha256": dict(TERMINALS[name]),
        "initials": "training roots evaluated with the original initializer graph",
    }
    return ParentData(
        name,
        problem,
        model,
        training,
        starts,
        parent["public_brief"]["scientific_context"],
        provenance,
    )
