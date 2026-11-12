"""Build separate public-fit, post-freeze blind-input and private-grader ZIPs."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import iclr_stress_data as data
from scripts import iclr_stress_evaluation as full

from .exchange import (
    CELLS,
    FILES,
    INPUTS,
    SEEDS,
    TARGETS,
    file_hash,
    fingerprint,
    publish,
    verify,
)

HERE = Path(__file__).resolve().parent


def official_module(expected: str) -> str:
    """Copy the pinned scoring function and paired numerical check verbatim."""
    if file_hash(Path(full.__file__)) != expected:
        raise ValueError("scorer differs from Full's completed anchor evaluation")
    split = "    primary, control = results\n"
    prefix, sep, check = inspect.getsource(full.predict).partition(split)
    if (
        not sep
        or 'roles = pair["perturbed"]' not in prefix
        or not check.startswith("    for c ")
    ):
        raise ValueError("cannot isolate Full's solver agreement check")
    return (
        '"""Generated from the Full evaluator identified below."""\n'
        "from time import monotonic\nfrom types import SimpleNamespace\n"
        "import numpy as np\n"
        f"SOURCE_SHA256 = {expected!r}\n"
        "ModelValidationError = RuntimeExpressionError = ValueError\n"
        "predict = None  # The grader always supplies a frozen forecast callback.\n\n"
        + inspect.getsource(full.SolverFailure)
        + "\n"
        + inspect.getsource(full.SolverDisagreement)
        + "\n\ndef checked(pair, primary, control):\n"
        + '    roles = pair["perturbed"]\n'
        + '    context = SimpleNamespace(targets=roles["targets"])\n'
        + check
        + "\n\n"
        + inspect.getsource(full.score)
    )


def fit_inputs(
    root: Path, fit: Path, manifest: dict, aces: dict, delivery: Path
) -> dict:
    cells = {}
    with zipfile.ZipFile(delivery) as shipped:
        for idx, cell_id in enumerate(CELLS):
            cell = next(c for c in manifest["cells"] if c["cell"] == cell_id)
            if (cell["noise"], cell["stride"], cell["source"]) != (
                (0, 0.01, 0.05, 0.10)[idx],
                1,
                data.T2,
            ) or aces[cell_id]["public_files"] != cell["public_files"]:
                raise ValueError("ACEs or local stress design differs")
            for f in FILES:
                rel = Path("public") / cell_id / f
                raw = (root / rel).read_bytes()
                if (
                    hashlib.sha256(raw).hexdigest() != cell["public_files"][f]
                    or shipped.read(rel.as_posix()) != raw
                ):
                    raise ValueError(
                        "public input differs from Full's source or delivery"
                    )
                (fit / rel).parent.mkdir(parents=True, exist_ok=True)
                (fit / rel).write_bytes(raw)
            cells[cell_id] = {**aces[cell_id], "noise_percent": (0, 1, 5, 10)[idx]}
    publish(
        fit / "release.json",
        {
            "format": "t2-noise-public-1",
            "source_benchmark": data.T2,
            "seeds": list(SEEDS),
            "cells": cells,
            "targets": list(TARGETS),
            "auxiliaries": ["Uii"],
            "external_inputs": sorted(INPUTS),
            "train_trajectories": 16,
            "validation_trajectories": 4,
        },
    )
    return verify(fit)


def blind_row(row: dict, cell: dict) -> dict:
    """Expose Full's forecast inputs, including one initial reading per target."""
    tr = full.public_input(row, cell)
    return {
        "time": tr.time.tolist(),
        "targets0": {k: float(v[0]) for k, v in tr.targets.items()},
        "auxiliaries": {k: v.tolist() for k, v in tr.auxiliaries.items()},
        "external_inputs": {k: v.tolist() for k, v in tr.external_inputs.items()},
    }


def build(
    root: Path, anchors: Path, aces_source: Path, delivery: Path, out: Path
) -> None:
    manifest = sealed_read(root / "manifest.json")
    evaluation = sealed_read(anchors / "freeze.json")
    saved = read_seal(anchors / "references/acquisition_00.json")
    if (
        saved["freeze_sha256"] != evaluation["artifact_sha256"]
        or manifest["artifact_sha256"] != evaluation["design_sha256"]
        or len(saved["records"]) != 6
    ):
        raise ValueError("reference file does not belong to Full's six-pair test panel")
    with tempfile.TemporaryDirectory() as directory:
        fit, blind, grade = [
            Path(directory) / ("t2-noise-" + s) for s in ("fit", "blind", "grade")
        ]
        for target in (fit, grade):
            mod = target / "noise_handoff"
            mod.mkdir(parents=True)
            for name in ("__init__.py", "exchange.py"):
                shutil.copyfile(HERE / name, mod / name)
        shutil.copyfile(HERE / "native.py", fit / "noise_handoff/native.py")
        shutil.copyfile(HERE / "grade.py", grade / "noise_handoff/grade.py")
        shutil.copyfile(HERE / "CLAUDE.md", fit / "CLAUDE.md")
        shutil.copyfile(HERE / "OPERATOR.md", grade / "OPERATOR.md")
        shutil.copyfile(HERE / "example_adapter.py", fit / "example_adapter.py")
        example = {
            "method": "smoke_native",
            "semantics": "discrete_map",
            "implementation_revision": file_hash(HERE / "example_adapter.py"),
            "budget": "no fit; handoff smoke only",
            "selection_rule": "fixed persistence",
        }
        (fit / "example_config.json").write_text(json.dumps(example, indent=2) + "\n")
        (grade / "requirements.txt").write_text("numpy==2.3.5\n")
        (grade / "noise_handoff/official.py").write_text(
            official_module(evaluation["evaluator_sha256"])
        )
        release = fit_inputs(
            root, fit, manifest, json.loads(aces_source.read_bytes()), delivery
        )
        refs = saved["records"]
        cells, inputs = {}, {}
        for cell_id in CELLS:
            cell = next(c for c in manifest["cells"] if c["cell"] == cell_id)
            cells[cell_id] = {
                "public_files": cell["public_files"],
                "scales": cell["scales"],
                "noise_percent": release["cells"][cell_id]["noise_percent"],
                "condition_sd": data.train_scales(
                    read_seal(root / "public" / cell_id / "train.json")["rows"]
                ),
            }
            inputs[cell_id] = [
                {
                    "id": f"q{i:02d}",
                    **{k: blind_row(pair[k], cell) for k in ("perturbed", "control")},
                }
                for i, pair in enumerate(refs)
            ]
        queries = publish(
            blind / "queries.json",
            {
                "format": "t2-noise-blind-1",
                "release_sha256": fingerprint(release),
                "cells": inputs,
            },
        )
        truth = [
            {
                "blind_id": f"q{i:02d}",
                "case": pair["case"],
                **{
                    k: {f: pair[k][f] for f in ("time", "targets")}
                    for k in ("perturbed", "control")
                },
            }
            for i, pair in enumerate(refs)
        ]
        publish(
            grade / "private/answer-key.json",
            {
                "format": "t2-noise-answer-key-1",
                "release_sha256": fingerprint(release),
                "queries_sha256": fingerprint(queries),
                "full_evaluator_sha256": evaluation["evaluator_sha256"],
                "anchor_evaluation_freeze": evaluation["artifact_sha256"],
                "full_reference_file_sha256": file_hash(
                    anchors / "references/acquisition_00.json"
                ),
                "cells": cells,
                "references": truth,
            },
        )
        out.mkdir(exist_ok=True, parents=True)
        for tree in (fit, blind, grade):
            files = sorted(f for f in tree.rglob("*") if f.is_file())
            (tree / "SHA256SUMS").write_text(
                "".join(
                    f"{file_hash(f)}  {f.relative_to(tree).as_posix()}\n" for f in files
                )
            )
            with zipfile.ZipFile(out / (tree.name + ".zip"), "w") as z:
                for path in [*files, tree / "SHA256SUMS"]:
                    entry = zipfile.ZipInfo(
                        path.relative_to(tree.parent).as_posix(),
                        (2026, 11, 11, 0, 0, 0),
                    )
                    z.writestr(
                        entry, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED
                    )
        shutil.copyfile(HERE / "CLAUDE.md", out / "SEND_TO_CLAUDE.md")
        shutil.copyfile(HERE / "OPERATOR.md", out / "OPERATOR.md")
        archives = [out / (t.name + ".zip") for t in (fit, blind, grade)]
        (out / "SHA256SUMS").write_text(
            "".join(f"{file_hash(p)}  {p.name}\n" for p in archives)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for param in ("root", "anchors", "aces-source", "delivery", "out"):
        parser.add_argument("--" + param, type=Path, required=True)
    build(**vars(parser.parse_args()))
