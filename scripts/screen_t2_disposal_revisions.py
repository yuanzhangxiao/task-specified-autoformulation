"""Enumerate disposal laws for the frozen T2 slot 015 and fit on training data."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.trajectory_function_fitting import validate_formula
from autoformalism.staged_topology import content_hash
from scripts import positive_insulin_refit_inputs as intake
from scripts import positive_insulin_refit_kernel as kernel
from scripts.construction_functions import file_hash

MODELS = Path("output/t2-orion-positive-refits-20261123-v1/fitted-models.json")
TRAIN = Path("output/trajectory-functions-20261123/inputs/train.json")
OUT = Path("output/t2-disposal-screen-20261123-v1")
MODEL_ARCHIVE = "05d0d6ab101a34334df8f1626154d739a55c40d53d8ccccd8db307a077927e0b"
TARGETS = ("Gp", "I", "U")
EPS = 1e-8


@dataclass(frozen=True)
class Family:
    """A closed algebraic law; each adjustable coefficient enters linearly."""

    name: str
    rhs: str
    coefficients: tuple[str, ...]
    nonnegative: tuple[str, ...]
    action_gains: tuple[str, ...]


def menu(parent: dict) -> list[Family]:
    """Prespecify matched comparators and isolated glucose/action alternatives."""
    old = parent["formula"]["equations"]["U"]
    control = Family("refit_existing", old, ("afcoef11", "afcoef13"), (), ("afcoef13",))
    families = [Family("incumbent", old, (), (), ("afcoef13",)), control]
    action = "max(0,I_act)"
    tissue = "max(0,Gt)"
    for tag, substrate in (
        ("constant", ""),
        ("tissue", f"({tissue}/100)"),
        ("saturating_tissue", f"({tissue}/(100+{tissue}))"),
    ):
        terms = f"disp_action*{action}"
        gains = ("disp_action",)
        if substrate:
            terms = f"(disp_basal+{terms})*{substrate}"
            gains = ("disp_basal", "disp_action")
        families.append(Family(tag, "Uii+" + terms, gains, gains, ("disp_action",)))
        if substrate:
            # Preserve the fitted Gp feedback slope, allowing a signed offset.
            # Nonnegativity of total and insulin-dependent disposal is checked
            # on each training sample for this family, including held-out folds.
            g = "afcoef12*((Gp-(205.410711764))/49.9160934007)"
            families.append(
                Family(
                    tag + "_gp",
                    f"Uii+disp_offset+{g}+{terms}",
                    ("disp_offset", *gains),
                    gains,
                    ("disp_action",),
                )
            )
    return families


def assemble(parent: dict, family: Family) -> dict:
    """Replace only U and its local coordinates; infer exact RHS dependencies."""
    result = deepcopy(parent)
    f = result["formula"]
    old = dict(f["parameters"])
    f["equations"]["U"] = family.rhs
    p = old | dict.fromkeys(family.coefficients, 0.0)
    parser = RestrictedParser()
    exprs = f["equations"] | {f"initial_{k}": v for k, v in f["initializers"].items()}
    symbols = {k: parser.parse(v, location=k).symbols for k, v in exprs.items()}
    used = set().union(*symbols.values())
    f["parameters"] = {k: old.get(k, 0.0) for k in p if k in used}
    src = sorted(symbols["U"] - set(p))
    if set(src) - {"Gp", "Uii", "I_act", "Gt"}:
        raise ValueError("Unexpected or unsupplied disposal source")
    result["graph"]["sources"]["U"] = src
    if family.name != "incumbent":
        # Old topology/formula check labels do not certify a new equation.
        for key in ("training_formula", "training_topology", "balance"):
            result.pop(key, None)
    validate_formula(result["graph"], f)
    return result


def solve(matrix: np.ndarray, offset: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Minimize the sum of squared scaled free-rollout residuals."""
    if matrix.ndim != 2 or matrix.shape[0] != len(offset):
        raise ValueError("Response matrix and residual lengths differ")
    if matrix.shape[1] == 0:
        return np.empty(0)
    result = lsq_linear(matrix, -offset, bounds=(bounds, np.inf), method="bvls")
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ArithmeticError("Bounded disposal fit failed")
    return result.x


def summary(residual: np.ndarray, rows: list[dict], scales: dict) -> dict:
    """Score sample-weighted, scale-normalized predictions and positivity."""
    arr = residual.reshape(-1, len(TARGETS))
    scores = dict(zip(TARGETS, np.mean(arr**2, axis=0).tolist(), strict=True))
    event_sse = {}
    min_dependent = np.inf
    min_target = dict.fromkeys(TARGETS, np.inf)
    offset = 0
    for row in rows:
        ix = slice(offset, offset + len(row["time"]))
        offset = ix.stop
        err = arr[ix] * np.array([scales[k] for k in TARGETS])
        event_sse[row["trajectory_id"]] = float(err[:, 2] @ err[:, 2])
        y = np.asarray([row["targets"][k] for k in TARGETS]).T + err
        for j, k in enumerate(TARGETS):
            min_target[k] = min(min_target[k], float(min(y[:, j])))
        min_dependent = min(
            min_dependent, float(min(y[:, 2] - row["auxiliaries"]["Uii"]))
        )
    if len(arr) != offset or not np.all(np.isfinite(arr)):
        raise ValueError("Invalid predicted trajectory dimensions or values")
    return {
        "per_target_nmse": scores,
        "pooled_nmse": float(np.mean(list(scores.values()))),
        "event_U_sse": event_sse,
        "min_U_minus_Uii": min_dependent,
        "minima": min_target,
    }


def assess(parent: dict, family: Family, rows: list[dict]) -> dict:
    """Fit all leave-one-event-out folds with true affine coupled responses."""
    model = assemble(parent, family)
    problem = kernel.FitProblem(model, rows, substeps=4)
    base = dict(model["formula"]["parameters"])
    base.update(dict.fromkeys(family.coefficients, 0.0))
    p = np.array([base[n] for n in problem.names])
    offset = problem.residual(p)
    columns = []
    indexes = [problem.names.index(name) for name in family.coefficients]
    for i in indexes:
        unit = p.copy()
        unit[i] = 1
        columns.append(problem.residual(unit) - offset)
    matrix = np.column_stack(columns) if columns else np.empty((len(offset), 0))
    lo = np.array(
        [0.0 if k in family.nonnegative else -np.inf for k in family.coefficients]
    )
    best = solve(matrix, offset, lo)
    fitted = p.copy()
    fitted[indexes] = best
    replay = problem.residual(fitted)
    if not np.allclose(replay, offset + matrix @ best, atol=1e-8, rtol=1e-8):
        raise ArithmeticError("The full free rollout disagrees with the affine fit")
    model["formula"]["parameters"].update(
        zip(family.coefficients, best.tolist(), strict=True)
    )
    masks = np.repeat(
        np.arange(len(rows)), [len(r["time"]) * len(TARGETS) for r in rows]
    )
    held = np.empty_like(offset)
    folds = {}
    for index, row in enumerate(rows):
        train = masks != index
        w = solve(matrix[train], offset[train], lo)
        held[~train] = offset[~train] + matrix[~train] @ w
        folds[row["trajectory_id"]] = dict(
            zip(family.coefficients, w.tolist(), strict=True)
        )
    scales = parent["scales"]
    active = [model["formula"]["parameters"][name] for name in family.action_gains]
    return {
        "model": model,
        "training": summary(replay, rows, scales),
        "held_training": summary(held, rows, scales),
        "fold_coefficients": folds,
        "active_action_gain": max(active) > EPS,
        "affine_replay_max_scaled_difference": float(
            np.max(abs(replay - offset - matrix @ best))
        ),
    }


def select(candidates: dict) -> tuple[str, dict]:
    """Require active action, sane concentrations and lower U and pooled error."""
    control = candidates["incumbent"]
    eligible = []
    reasons = {}
    for name, cand in candidates.items():
        if name in ("incumbent", "refit_existing"):
            continue
        fails = []
        if not cand["active_action_gain"]:
            fails.append("inactive_full_training_action")
        for split in ("training", "held_training"):
            s, ref = cand[split], control[split]
            if s["min_U_minus_Uii"] < -EPS or min(s["minima"].values()) < -EPS:
                fails.append(split + "_negative_contribution_or_target")
            if s["pooled_nmse"] >= ref["pooled_nmse"] - EPS:
                fails.append(split + "_pooled")
            if s["per_target_nmse"]["U"] >= ref["per_target_nmse"]["U"] - EPS:
                fails.append(split + "_U")
        reasons[name] = fails
        if not fails:
            eligible.append(name)
    winner = min(
        eligible,
        key=lambda n: candidates[n]["held_training"]["pooled_nmse"],
        default="incumbent",
    )
    return winner, reasons


def run(models: Path, training: Path, root: Path) -> dict:
    """Freeze the menu and the registered training-only endpoints, with resume."""
    exports = sealed_read(models)
    if exports["artifact_sha256"] != MODEL_ARCHIVE:
        raise ValueError("The Orion fitted model archive differs")
    parent = exports["models"]["015-positive_EGP"]["joint"]
    parent_id = content_hash(parent)
    rows = intake.load_training(training)
    families = menu(parent)
    with _lock(root):
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": "t2-015-coupled-disposal-training-1",
                "source": file_hash(Path(__file__)),
                "rollout_kernel": file_hash(Path(kernel.__file__)),
                "parent_model_sha256": parent_id,
                "training_sha256": intake.original.campaign.TRAIN,
                "training_events": [r["trajectory_id"] for r in rows],
                "families": [
                    asdict(f)
                    | {
                        "coefficients": list(f.coefficients),
                        "nonnegative": list(f.nonnegative),
                        "action_gains": list(f.action_gains),
                    }
                    for f in families
                ],
                "selection": (
                    "Choose lowest held-event pooled NMSE among eligible revisions. "
                    "Require active action, nonnegative target and dependent disposal "
                    "on the grids, lower U and pooled NMSE on full and held training. "
                    "Otherwise retain the incumbent."
                ),
                "fitting": (
                    "Coupled RK4 with four substeps per insulin knot interval. "
                    "Fit disposal response columns by bounded least squares using "
                    "frozen Gp/I/U scales. Each fold refits only disposal. "
                    "The upstream model was fitted on all 16 training events."
                ),
                "llm_calls": 0,
                "validation_opened": False,
                "test_opened": False,
            },
        )
        candidates = {}
        for family in families:
            path = root / "families" / f"{family.name}.json"
            if not path.exists():
                sealed_write(
                    path,
                    {
                        "plan": plan["artifact_sha256"],
                        "result": assess(parent, family, rows),
                    },
                )
            record = sealed_read(path)
            if record["plan"] != plan["artifact_sha256"]:
                raise ValueError("Family checkpoint belongs to a different plan")
            candidates[family.name] = record["result"]
        chosen, reasons = select(candidates)
        return sealed_write(
            root / "training-freeze.json",
            {
                "plan": plan["artifact_sha256"],
                "parent_model_sha256": parent_id,
                "training_sha256": intake.original.campaign.TRAIN,
                "incumbent": "incumbent",
                "control": "refit_existing",
                "selected": chosen,
                "candidates": candidates,
                "exclusions": reasons,
                "validation_opened": False,
                "test_opened": False,
            },
        )


def main() -> None:
    """Run a bounded train-only conditional disposal reconstruction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument("--training", type=Path, default=TRAIN)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    freeze = run(args.models, args.training, args.out)
    print(f"Frozen {len(freeze['candidates'])} families; selected {freeze['selected']}")


if __name__ == "__main__":
    main()
