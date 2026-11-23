"""Freeze a small, structurally diverse train-only cohort from the topology replay."""

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.preproposal_cases import BASIN, T2
from scripts.construction_functions import file_hash
from scripts.preproposal_graph_portfolio import with_reviewed_initial_bindings

REPO = Path(__file__).resolve().parents[1]
REPLAY = REPO / "output/topology-admission-replay-20261123/replay.json"
INSPECTION = REPO / "output/complete-graphs-results-20261123/returned/intake.json"
PUBLIC = REPO / "artifacts/phase-c-development-final-v2/public"
REPLAY_FILE = "179c05d6aec78fbb1772e56b85f9f7363b8627a9a158dc2426e2000af219ffa9"
ORIGINAL_INTAKE = "47c22bc3ca998e0b28bf79bf527bbe06230c82cf86b5e01beaf418722fb3ba1c"
VALIDATION_FILES = {
    BASIN: "a43d23ff1e341904f3db49316de74f943e01b9723e79d4bc483903f92646a0d8",
    T2: "464618cba3bad9f86c72448a128cde7ecbddeb1e781c8a2df23ee7d9abfef8af",
}


def features(graph: dict) -> set[tuple]:
    """Compare equation edges, variable types and initialization permissions."""
    result = {("type", k, v) for k, v in graph["variable_types"].items()}
    result.update(("rhs", k, v) for k, vs in graph["sources"].items() for v in vs)
    for state, rule in graph["initializers"].items():
        result.add(("init", state, rule["mode"], rule.get("channel", "")))
        for kind in ("public_initials", "state_initials"):
            result.update((kind, state, v) for v in rule.get(kind, []))
    return result


def select_graphs(replay: dict) -> list[dict]:
    """Within each origin choose the sparsest then most different admitted graph."""
    options = {}

    def record(case, digest, graph, origin, parent):
        key = (case, digest)
        row = options.setdefault(
            key,
            {
                "case": case,
                "structure_sha256": digest,
                "graph": graph,
                "origin": origin,
                "parents": [],
                "dataset": BASIN if case.startswith("basin_") else T2,
            },
        )
        if row["graph"] != graph or row["origin"] != origin:
            raise ValueError("Ambiguous canonical graph in replay")
        if parent:
            row["parents"].append(parent)

    for row in replay["original_pool_recheck"]:
        if row["selected"] and row["status"] == "topology_screen_passed":
            record(row["case_id"], row["structure_sha256"], row["graph"], "pool", None)
    for row in replay["saved_replies"]:
        for child in row.get("repairs", []):
            if child["status"] != "topology_screen_passed":
                raise ValueError("Expected a passing repair")
            record(
                row["case_id"],
                child["structure_sha256"],
                child["canonical_graph"],
                "repaired",
                row["slot"],
            )

    chosen = []
    for origin, ds in [("pool", T2), ("repaired", T2), ("pool", BASIN)]:
        group = [
            v for v in options.values() if (v["origin"], v["dataset"]) == (origin, ds)
        ]
        first = min(
            group, key=lambda v: (len(features(v["graph"])), v["structure_sha256"])
        )
        a = features(first["graph"])

        def distance_key(other, first=first, a=a):
            b = features(other["graph"])
            return (
                other["case"] == first["case"],
                len(a & b) / len(a | b),
                other["structure_sha256"],
            )

        second = min((v for v in group if v is not first), key=distance_key)
        chosen.extend((first, second))
    return chosen


def public_rows(path: Path) -> list[dict]:
    """Project an authenticated train or validation release into public row roles."""
    split = read_seal(path)
    if split["name"] != path.stem or split["name"] not in {"train", "val"}:
        raise ValueError("Expected the named public development split")
    return [
        {
            "trajectory_id": r["trajectory_id"],
            "time": r["time"],
            "targets": r["targets"],
            "inputs": r["external_inputs"],
            "auxiliaries": r["auxiliaries"],
            "covariates": r["fixed_covariates"],
        }
        for r in split["rows"]
    ]


def make_inputs(out: Path) -> dict:
    """Select using training topology alone; package validation independently."""
    if file_hash(REPLAY) != REPLAY_FILE:
        raise ValueError("Expected the corrected job 2403807 replay")
    archived = sealed_read(INSPECTION)
    if archived["artifact_sha256"] != ORIGINAL_INTAKE:
        raise ValueError("Unexpected topology intake")
    graphs = select_graphs(json.loads(REPLAY.read_text()))
    cases = {}
    training = {}
    for row in graphs:
        case, ds = row["case"], row["dataset"]
        frozen = with_reviewed_initial_bindings(archived["frozen_cases"][case])
        actual = frozen["evidence"]["trajectories"]
        if ds not in training:
            training[ds] = public_rows(PUBLIC / ds / "train.json")
        if actual != training[ds]:
            raise ValueError(f"Graph {case} and fitting release training rows differ")
        cases[case] = {k: v for k, v in frozen.items() if k != "evidence"} | {
            "evidence": {
                k: v for k, v in frozen["evidence"].items() if k != "trajectories"
            }
        }
    intake = sealed_write(
        out / "train.json",
        {
            "graphs": graphs,
            "cases": cases,
            "train": training,
            "source_replay": REPLAY_FILE,
            "validation_files": VALIDATION_FILES,
        },
    )
    heldout = {}
    for dataset, digest in VALIDATION_FILES.items():
        path = PUBLIC / dataset / "val.json"
        if file_hash(path) != digest:
            raise ValueError(f"Validation release identity differs for {dataset}")
        heldout[dataset] = public_rows(path)
        train_ids = {r["trajectory_id"] for r in training[dataset]}
        if any(r["trajectory_id"] in train_ids for r in heldout[dataset]):
            raise ValueError("Training and validation trajectory IDs overlap")
    sealed_write(
        out / "validation.json",
        {
            "split": "val",
            "source_files": VALIDATION_FILES,
            "rows": heldout,
            "training_intake_sha256": intake["artifact_sha256"],
        },
    )
    return {
        "graphs": len(graphs),
        "train": str(out / "train.json"),
        "validation": str(out / "validation.json"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(make_inputs(parser.parse_args().out), indent=2))
