#!/usr/bin/env python3
"""Fit an unchanged-parent control with the completed child's pinned executor.

Run this new driver with PYTHONPATH pointing at the original replay checkout.
The historical verifier checks the entire imported source tree and runtime;
this driver is separately hashed so no historical source guard is weakened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.rebuttal import prefit_parameter_replay as replay
from autoformalism.rebuttal.prefit_fit_handoff import _disjoint
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write

PROTOCOL = "prefit-matched-refit-1"


def _inputs(source: Path, root: Path) -> tuple[dict, dict, dict, dict, dict]:
    """Reconstruct both branches from verified history before reserving compute."""
    _disjoint(root, source, Path(__file__).resolve().parents[1])
    plan, old, residual = replay.verify(source)
    _disjoint(
        root,
        Path(plan["source_root"]),
        Path(old["paths"]["parent"]).parent,
        Path(old["paths"]["continuation"]).parent,
        Path(old["paths"]["source"]),
        Path(old["paths"]["construction"]),
        Path(public.__file__).resolve().parents[3],
    )
    if not plan["outcome"]["accepted"]:
        raise ValueError("matched refit requires an accepted replay")
    child_contract = replay._contract(plan, old, residual)
    frozen = sibling_fit._freeze(**child_contract)
    if public._read(source / "child_fit/freeze.json") != frozen:
        raise ValueError("child freeze differs from the saved proposal")
    child = sibling_fit.inspect_child_fit(source / "child_fit")
    if not child["result"] or child["result"]["status"] != "complete":
        raise ValueError("matched refit requires a completed child fit")
    contract = {
        **child_contract,
        "child": child_contract["parent"],
        "lineage": {
            "protocol": PROTOCOL,
            "source_plan_sha256": plan["artifact_sha256"],
            "matched_child_identity": child["identity"],
            "matched_child_result_sha256": public.content_sha256(child["result"]),
        },
    }
    control = sibling_fit._freeze(**contract)
    if control["public_fit"]["settings"] != frozen["public_fit"]["settings"]:
        raise ValueError("parent and child fitting settings differ")
    seed = control["seed"]
    if seed["parameters"] != contract["parameters"] or seed["fresh_parameters"]:
        raise ValueError("control must start from every original parent parameter")
    if any(
        frozen["seed"]["parameters"].get(name) != value
        for name, value in seed["parameters"].items()
    ):
        raise ValueError("child did not use the same shared starting parameters")
    body = {
        "protocol": PROTOCOL,
        "source_root": str(source),
        **contract["lineage"],
        "control_identity": control["identity"],
        "executor_source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return body, contract, control, child, residual["seed"]["state"]


def _reservation(source: Path, body: dict) -> Path:
    return source.parent / ".prefit-matched-refits" / body["matched_child_identity"]


def prepare(source: Path, root: Path) -> dict:
    """Reserve one unchanged-parent allocation without running the fitter."""
    source, root = source.resolve(), root.resolve()
    body, *_ = _inputs(source, root)
    with public._lock(root), public._lock(_reservation(source, body)):
        sealed_write(
            _reservation(source, body) / "reservation.json",
            {"output": str(root), "plan_sha256": public.content_sha256(body)},
        )
        sealed_write(root / "plan.json", body)
        return _report(root, *_verify(root))


def _verify(root: Path) -> tuple[dict, dict, dict, dict, dict]:
    plan = sealed_read(root / "plan.json")
    source = Path(plan["source_root"])
    body, contract, frozen, child, parent = _inputs(source, root)
    if body != {k: v for k, v in plan.items() if k != "artifact_sha256"}:
        raise ValueError("matched refit source, child, driver or runtime changed")
    reservation = sealed_read(_reservation(source, body) / "reservation.json")
    if reservation != {
        "output": str(root.resolve()),
        "plan_sha256": public.content_sha256(body),
        "artifact_sha256": reservation["artifact_sha256"],
    }:
        raise ValueError("matched refit reservation differs")
    return plan, contract, frozen, child, parent


def _comparison(parent: dict, child: dict, control: dict | None) -> dict:
    """Describe scores; do not select a branch or interpret numerical failure."""
    result = {}
    for split in ("training", "validation"):
        scores = {
            label: (value.get(split) or {}).get("normalized_mse") if value else None
            for label, value in (
                ("parent", parent),
                ("child", child),
                ("refit", control),
            )
        }
        scores["child_minus_refit"] = (
            scores["child"] - scores["refit"]
            if scores["child"] is not None and scores["refit"] is not None
            else None
        )
        result[split] = scores
    return result


def _report(
    root: Path, plan: dict, contract: dict, frozen: dict, child: dict, parent: dict
) -> dict:
    control = None
    if (root / "refit/freeze.json").exists():
        if public._read(root / "refit/freeze.json") != frozen:
            raise ValueError("control belongs to a different matched comparison")
        control = sibling_fit.inspect_child_fit(root / "refit")
    fitted = control["result"] if control else None
    result = {
        "protocol": PROTOCOL,
        "status": ("complete" if fitted["status"] == "complete" else "refit_failed")
        if fitted
        else "ready_for_fit",
        "plan_sha256": plan["artifact_sha256"],
        "parent": parent,
        "child": child["result"],
        "refit": fitted,
        "comparison": _comparison(parent, child["result"], fitted),
        "matching": {
            "same_shared_starting_parameters": True,
            "same_initializer_starting_parameters": True,
            "same_training_and_validation": True,
            "same_fitter_source_and_runtime": True,
            "same_fitting_settings": True,
            "control_equations_unchanged": True,
            "profile": contract["parent"].profile,
            "settings": frozen["public_fit"]["settings"],
        },
        "live_llm_calls": 0,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_branch_selection": False,
        "limitation": "One retrospective paired diagnostic with equal configured "
        "fitting allocations, not equal optimizer work or a fresh prospective "
        "replication. Wall-clock-limited results can depend on machine load. "
        "Numerical convergence and scientific validity are not certified.",
    }
    public._write(root / "summary.json", result)
    return result


def report(root: Path) -> dict:
    """Verify historical and matched results without allocating numerical work."""
    root = root.resolve()
    with public._lock(root):
        return _report(root, *_verify(root))


def fit(root: Path) -> dict:
    """Execute one frozen control; completed or interrupted allocations cannot renew."""
    root = root.resolve()
    with public._lock(root):
        checked = _verify(root)
        sibling_fit.prepare_child_fit(directory=root / "refit", **checked[1])
        sibling_fit.execute_child_fit(root / "refit")
        return _report(root, *checked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "prepare", "report"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("run", "prepare"):
        if args.source is None:
            parser.error("--source is required for run/prepare")
        result = prepare(args.source, args.root)
        if args.command == "run":
            result = fit(args.root)
    else:
        result = report(args.root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
