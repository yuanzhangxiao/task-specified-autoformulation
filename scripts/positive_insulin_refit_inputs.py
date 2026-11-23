"""Authenticated starts, public split readers and original-equation replay for T2."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from autoformalism.expressions.compiler import SAFE_DIVISION_EPSILON
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research.trajectory_function_fitting import _rows
from autoformalism.staged_topology import content_hash
from scripts import validate_positive_insulin_transfer as original
from scripts.construction_functions import file_hash

TAGS = tuple(tag for tag, _, _ in original.MODELS)
TARGETS = original.TARGETS
PLAN_DIGEST = "7c7c3b95240e2eda5d83ee41939f20c82fe1e0e59a52ad8e1b228a5649ab798c"
# Digest of all four models after removing just the fitted coordinate values
# and the scalar subfit's parameter-dependent diagnostics.
STRUCTURES = "a9fd290572affd0a0fb8e29194f1553f03675cb0f1e4873b21c2264cbd15b192"
SOURCE_MODELS = {
    "002-restoring": "ab60e22f23a625fa180ffb43ce1dc630075eb2481287407a83b8271c5a6d1c3b",
    "002-positive_EGP": (
        "6587f7a66df409b7f4811b601cccfd10" "f4ea86c5206e57b6510da2efd0480020"
    ),
    "015-restoring": "8a677e9870b07325b4cc5fded0cb2d9b8107df1baf53ae626292c12f739e621e",
    "015-positive_EGP": (
        "3ab4bd9eeba990f10ae4f3ba6f25a5e0" "10f31a0f9c306dcbf19c8bd40241bafc"
    ),
}
FULL_SCALAR_FIT = "c61568b5dc574c244631900c11f8afbb4e1172f7639229bebf1802ed0f2c37b6"
TRAIN_IDS = tuple(f"train_{i:03}" for i in range(16))
# Selected training options of the two source fits identified by parent_fit_sha256.
PARENT_TRAINING_NMSE = {
    "002-restoring": {
        "Gp": 0.32364272152941487,
        "I": 0.23018505762310024,
        "U": 0.5338695514251282,
    },
    "015-restoring": {
        "Gp": 0.1487197622930685,
        "I": 0.22939430979052505,
        "U": 0.21779859576325583,
    },
}


def _read_member(archive: ZipFile, name: str) -> dict:
    info = archive.getinfo(name)
    if info.file_size > 2_000_000:
        raise ValueError(f"Oversized handoff member: {name}")
    value = json.loads(archive.read(info))
    if value.get("artifact_sha256") != content_hash(
        {k: v for k, v in value.items() if k != "artifact_sha256"}
    ):
        raise ValueError(f"Archive member digest differs: {name}")
    return value


def load_models(archive: Path) -> dict[str, dict]:
    """Read only the four model exports after verifying their pinned source plan."""
    with ZipFile(archive) as z:
        expected = ["ORION.md", "plan.json", *(f"models/{tag}.json" for tag in TAGS)]
        if sorted(z.namelist()) != sorted(expected):
            raise ValueError("Missing, extra or duplicate member in fit-start bundle")
        plan = _read_member(z, "plan.json")
        if plan["artifact_sha256"] != PLAN_DIGEST:
            raise ValueError("Incorrect development handoff plan")
        if plan["script"] != file_hash(Path(original.__file__)) or plan[
            "export_code"
        ] != file_hash(Path(original.portable.__file__)):
            raise ValueError("Original forecast or export implementation differs")
        for tag in TAGS:
            rec = _read_member(z, f"models/{tag}.json")
            if (
                rec["source_plan"] != PLAN_DIGEST
                or rec["full_training_model"] != plan["models"][tag]
                or rec["training_archived_parent"] != plan["parent_ranking"]
                or rec["selected_positive_full_training_fit"]
                != (
                    plan["selected_positive_training_fit"]
                    if "positive" in tag
                    else None
                )
            ):
                raise ValueError(f"Model export differs from pinned sources: {tag}")
        return plan["models"]


def load_training(train_path: Path) -> list[dict]:
    """Load the exact 16 training events from the registered public intake."""
    if train_path.name != "train.json":
        raise ValueError("Expected the registered train.json")
    source = sealed_read(train_path)
    if source["artifact_sha256"] != original.campaign.TRAIN:
        raise ValueError("Training intake digest differs")
    rows = source["train"][original.T2]
    if content_hash(rows) != original.scalar.TRAIN:
        raise ValueError("T2 training rows differ from the handoff")
    if tuple(r.trajectory_id for r in _rows(rows, "train")) != TRAIN_IDS:
        raise ValueError("Incomplete T2 training cohort")
    return rows


def _structure(model: dict) -> dict:
    f = model["formula"]
    skeleton = model | {"formula": f | {"parameters": sorted(f["parameters"])}}
    if "scalar_model" in model:
        skeleton["scalar_model"] = {
            "coefficient_hats": model["scalar_model"]["coefficient_hats"]
        }
    return skeleton


def load_validation(path: Path, frozen_models: dict[str, dict]) -> list[dict]:
    """Read development rows after the caller authenticates its frozen training fits."""
    if (
        set(frozen_models) != set(TAGS)
        or content_hash({k: _structure(v) for k, v in frozen_models.items()})
        != STRUCTURES
    ):
        raise ValueError(
            "All four frozen model structures are required before validation"
        )
    for model in frozen_models.values():
        with_parameters(model, model["formula"]["parameters"])
    plan = {
        "training_intake": original.campaign.TRAIN,
        "validation_intake": original.campaign.VALIDATION,
    }
    return [row.model_dump(mode="json") for row in original.read_validation(path, plan)]


def with_parameters(model: dict, parameters: dict[str, float]) -> dict:
    """Apply the fitted vector to both the coupled graph and the scalar verifier."""
    former = model["formula"]["parameters"]
    if parameters.keys() != former.keys() or not all(
        np.isfinite(float(v)) for v in parameters.values()
    ):
        raise ValueError("Expected a finite vector with the exported parameter names")
    p = {k: float(v) for k, v in parameters.items()}
    model = model | {"formula": model["formula"] | {"parameters": p}}
    if "scalar_model" not in model:
        return model
    old = model["scalar_model"]
    hats = original.portable.production.coefficient_hats(model["knots"])
    original.portable.production.checked_weights(model["knots"], old)
    if (
        old.get("coefficient_hats") != hats
        or p.get("pos_tau", 0.0) <= SAFE_DIVISION_EPSILON
    ):
        raise ValueError("Invalid basis or positive-insulin timescale")
    weights = [
        p.get(
            f"pos_{h['channel']}_flat",
            p.get(f"pos_{h['channel']}_{h['hat_index']}", 0.0),
        )
        for h in hats
    ]
    if any(w < 0 for w in weights):
        raise ValueError("Production weights must be nonnegative")
    # Unrepresented zero-weight hats stay absent from the equation. Old KKT/SSE
    # diagnostics apply only to the original independently fitted scalar vector.
    if p["pos_tau"] != old["tau"] or weights != old["weights"]:
        model["scalar_model"] = {
            "coefficient_hats": hats,
            "tau": p["pos_tau"],
            "weights": weights,
        }
    return model


def _event(row, split: str) -> original.Inputs:
    if split == "val":
        return original.forecast_inputs(row)
    if set(row.targets) != set(TARGETS) or row.trajectory_id not in TRAIN_IDS:
        raise ValueError("Wrong training targets or trajectory ID")
    times = np.asarray(row.time)
    supplied = (
        row.inputs
        | row.auxiliaries
        | {k: np.full(len(times), v) for k, v in row.covariates.items()}
    )
    if not np.all(np.diff(times) > 0):
        raise ValueError("Time knots must increase")
    initial = {k: v[0] for k, v in (supplied | row.targets).items()}
    return original.Inputs(row.trajectory_id, times, supplied, initial)


def replay(model: dict, rows: list[dict]) -> dict:
    """Score free rollouts with the source RK4 grid and independent insulin test."""
    if not rows or (split := rows[0]["trajectory_id"].split("_", 1)[0]) not in {
        "train",
        "val",
    }:
        raise ValueError("Expected public training or development events")
    model = with_parameters(model, model["formula"]["parameters"])
    events = []
    for row, raw in zip(_rows(rows, split), rows, strict=True):
        event = _event(row, split)
        try:
            pred, log = original.forecast(model, event)
        except ArithmeticError as exc:
            events.append(
                {
                    "event": event.name,
                    "status": "integration_failed",
                    "reason": str(exc),
                }
            )
            continue
        if pred is None:
            events.append(
                {"event": event.name, "status": "grid_unresolved", "levels": log}
            )
            continue
        metrics = original.portable._metrics(raw, pred, model["scales"])
        events.append(
            metrics
            | {
                "time": list(raw["time"]),
                "observed": {k: list(raw["targets"][k]) for k in TARGETS},
                "predicted": {
                    k: y.tolist() for k, y in zip(TARGETS, pred, strict=True)
                },
                "levels": log,
            }
        )
    return {
        "model_sha256": content_hash(model),
        "events": events,
        "summary": original.portable.transfer.coupled.aggregate(events, len(rows)),
    }


def verify_start(tag: str, model: dict, training_replay: dict) -> dict:
    """Compare parent training scores or child insulin SSE to the frozen source."""
    events = training_replay["events"]
    summary = training_replay["summary"]
    if (
        SOURCE_MODELS.get(tag) != content_hash(model)
        or training_replay["model_sha256"]
        != content_hash(with_parameters(model, model["formula"]["parameters"]))
        or tuple(e["event"] for e in events) != TRAIN_IDS
        or (
            summary["scored_events"] != len(TRAIN_IDS) or summary["pooled_nmse"] is None
        )
    ):
        raise ValueError("Incomplete training-start replay")
    if tag in PARENT_TRAINING_NMSE:
        expected = PARENT_TRAINING_NMSE[tag]
        actual = {k: summary["available_per_target_nmse"][k] for k in expected}
        digest = model["parent_fit_sha256"]
        atol, rtol = 1e-8, 1e-8
    else:
        # The archived full-training scalar solver provides a reference only
        # for insulin. Full-system Gp and U are scored as new coupled replays.
        expected = {"insulin_sse": model["scalar_model"]["sse"]}
        actual = {"insulin_sse": sum(e["sse"]["I"] for e in events)}
        digest = FULL_SCALAR_FIT
        atol, rtol = 1e-5, 1e-7
    if not all(
        np.isclose(actual[k], expected[k], atol=atol, rtol=rtol) for k in expected
    ):
        raise ValueError(f"{tag} training-start replay differs from archived scores")
    return {
        "source": digest,
        "expected": expected,
        "replayed": actual,
        "atol": atol,
        "rtol": rtol,
    }
