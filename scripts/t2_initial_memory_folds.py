"""Training-only intake and shared insulin subfit for the two T2 initial-state folds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.optimize import minimize_scalar

from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research.trajectory_function_fitting import _rows
from autoformalism.staged_topology import content_hash
from scripts import compare_positive_insulin_rates as production
from scripts import positive_insulin_refit_inputs as intake
from scripts import transfer_positive_insulin_law as portable

HOLDOUTS = ("train_011", "train_012")
MANIFEST_SHA256 = "0f586db17dd4b3a0b40dcc68dbbfaf96b227a24c871b8f2a60fc328d82827309"
MEMBER = "output/t2-initial-order-20261123/initializer-probe/selected-model.json"
SELECTED_MODEL = "bb4f6c78d058fc26d08829f36999c3cb889c5c108b98def5489409734608a476"
TARGETS = ("Gp", "I", "U")
# Include the original insulin study's coarse timescales and explore below/above
# its end points. All profile coefficients and the final timescale are fold-specific.
TAU_GRID = (1.25, *production.prior.GRID, 540.0)


def authenticate_handoff(archive: Path) -> dict:
    """Verify the designated model and expose only its structural graph."""
    with ZipFile(archive) as bundle:
        for required in ("manifest.json", MEMBER):
            if (
                bundle.namelist().count(required) != 1
                or bundle.getinfo(required).file_size > 2_000_000
            ):
                raise ValueError(
                    f"Missing, duplicate or oversized handoff member {required}"
                )
        manifest_bytes = bundle.read("manifest.json")
        if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
            raise ValueError("Handoff identity differs")
        manifest = json.loads(manifest_bytes)
        data = bundle.read(MEMBER)
        if hashlib.sha256(data).hexdigest() != manifest["sha256"][MEMBER]:
            raise ValueError("Selected model archive member differs")
    record = json.loads(data)
    if (
        record["artifact_sha256"]
        != content_hash({k: v for k, v in record.items() if k != "artifact_sha256"})
        or record["model_sha256"] != SELECTED_MODEL
        or content_hash(record["model"]) != SELECTED_MODEL
    ):
        raise ValueError("Selected model digest differs")
    return {
        "manifest_sha256": MANIFEST_SHA256,
        "model_sha256": SELECTED_MODEL,
        "member": MEMBER,
        "graph": record["model"]["graph"],
    }


def fit_ids(held_id: str) -> tuple[str, ...]:
    """Return the fixed, canonical 15-event training order."""
    if held_id not in HOLDOUTS:
        raise ValueError("Expected train_011 or train_012 as the omitted event")
    return tuple(event for event in intake.TRAIN_IDS if event != held_id)


def load_fit_rows(training_path: Path, held_id: str) -> list[dict]:
    """Authenticate the training artifact and release only the fitting rows."""
    ids = fit_ids(held_id)
    if training_path.name != "train.json":
        raise ValueError("Expected the registered train.json")
    source = sealed_read(training_path)
    if source["artifact_sha256"] != intake.original.campaign.TRAIN:
        raise ValueError("Training intake digest differs")
    rows = source["train"][intake.original.T2]
    if sorted(row["trajectory_id"] for row in rows) != list(intake.TRAIN_IDS):
        raise ValueError("Training IDs differ from the registered cohort")
    subset = {r["trajectory_id"]: r for r in rows if r["trajectory_id"] != held_id}
    return [subset[k] for k in ids]


def _insulin(rows: list[dict]) -> dict:
    """Profile nonnegative production gains conditional on the decay timescale."""
    knots, candidates = production.fit(rows, production.SOURCES["EGP"], TAU_GRID)
    grid = candidates["nonnegative"]
    j = min(range(len(grid)), key=lambda i: (grid[i]["sse"], grid[i]["tau"]))
    fitting_interval = (TAU_GRID[max(0, j - 1)], TAU_GRID[min(len(grid) - 1, j + 1)])

    # A small scalar polish avoids imposing the coarse time grid on both arms.
    def profile(log_tau):
        tau = float(np.exp(log_tau))
        systems = [production.design(row, knots, tau) for row in rows]
        matrix = np.vstack([response for _, response in systems])
        target = np.concatenate(
            [
                row["targets"]["I"] - initial
                for row, (initial, _) in zip(rows, systems, strict=True)
            ]
        )
        return {
            "tau": tau,
            "coefficient_hats": production.coefficient_hats(knots),
            **production.solve(matrix, target, "nonnegative"),
        }

    result = minimize_scalar(
        lambda log_tau: profile(log_tau)["sse"],
        bounds=np.log(fitting_interval),
        method="bounded",
        options={"xatol": 1e-7, "maxiter": 80},
    )
    if not result.success:
        raise ValueError("Insulin timescale profile failed to converge")
    selected = min(
        [grid[j], profile(result.x)], key=lambda rec: (rec["sse"], rec["tau"])
    )
    checks = [production.score(row, knots, selected) for row in rows]
    if not all(r["verified"] and r["negative_samples"] == 0 for r in checks):
        raise ValueError("Fold insulin integration or positivity check failed")
    if not np.isclose(
        sum(r["sse"] for r in checks), selected["sse"], rtol=1e-8, atol=1e-8
    ):
        raise ValueError(
            "Training insulin replay differs from the profiled fitting objective"
        )
    bare = {
        "graph": {"sources": {}},
        "formula": {"initializers": {}, "equations": {"I": "0"}, "parameters": {}},
    }
    exported = portable.export(bare, knots, selected)
    return {
        "knots": knots,
        "scalar_model": selected,
        "formula": exported["formula"]["equations"]["I"],
        "parameters": exported["formula"]["parameters"],
        "sources": exported["graph"]["sources"]["I"],
        "coarse_profile": [{"tau": r["tau"], "sse": r["sse"]} for r in grid],
        "polish_interval_min": list(fitting_interval),
        "replay_checks": checks,
    }


def insulin_formula(fold: dict) -> tuple[str, dict[str, float]]:
    """Return the fold's portable insulin balance and its locked coefficients."""
    insulin = fold["insulin"]
    return insulin["formula"], dict(insulin["parameters"])


def fit_inputs(rows: list[dict], held_id: str) -> dict:
    """Fit insulin and freeze scaling exclusively from the 15 identified events."""
    ids = fit_ids(held_id)
    if sorted(row["trajectory_id"] for row in rows) != list(ids):
        raise ValueError("Expected exactly the fifteen designated fitting events")
    ordered = sorted(rows, key=lambda r: r["trajectory_id"])
    _rows(ordered, "train")
    values = {
        name: np.concatenate([r["targets"][name] for r in ordered]) for name in TARGETS
    }
    centers = {name: float(v.mean()) for name, v in values.items()}
    if ordered[0]["time"][0] != 0:
        raise ValueError("The reference event must start at minute zero")
    centers["Gt_ref"] = float(ordered[0]["auxiliaries"]["Gt"][0])
    scales = {name: max(1e-8, float(v.std())) for name, v in values.items()}
    insulin = _insulin(ordered)
    return {
        "held_id": held_id,
        "fit_ids": list(ids),
        "training_rows_sha256": content_hash(ordered),
        "centers": centers,
        "scales": scales,
        "insulin": insulin,
    }
