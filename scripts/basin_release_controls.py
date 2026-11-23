"""Apply the registered basin balance method to three related public controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.trajectory_function_fitting import _rows
from scripts import basin_balance_inference as balance
from scripts.construction_functions import file_hash
from scripts.trajectory_function_intake import public_rows

# Byte digests of the three public files in the Phase C development-v2 release
# manifest. Validation identities come from the manifest alone.
FILES = {
    "phase_c_detention_coupled_noise1_v1": {
        "specification": (
            "91cc1191c7ef532bf3dff3e5b9e2f53f7cc44ccdc82d88fd075f6d7b7fc93ede"
        ),
        "train": "5ae322af9ba36d22726c22d7e8cb0182e0dbbbfea0494a7ba59d88cb3c3fe666",
        "val": "1574410d9e73aa7093a598ea820912d4be5dfcafeebca5844333c031e68d1e96",
    },
    "phase_c_detention_independent_noise0_v1": {
        "specification": (
            "064c91cf708b1c9e740ea9e835b5131168d634a89f151ea2a08218d40bed6d85"
        ),
        "train": "60e1f060aed12853e6d841945e0f2e4b6f0db0b77244dd6cf42f2dc5f7f08926",
        "val": "8c0ae6b4376daebf779ca0f1013a03a734441d2a0115df1d15d37653bfc4209f",
    },
    "phase_c_detention_independent_noise1_v1": {
        "specification": (
            "0cbc77edbe672da06b971512e7fe2032bae1c8bd21a748c953f1bf31734b8c0a"
        ),
        "train": "93fa87dc98b2ea1a94661ae8f599e81a479c485daadda0327c72ae5e7b5bb49e",
        "val": "1e5dc7c49b0be3fe9004eedacd3fb82b20ce8c27d4c8bd5efd8f77cd50e59c72",
    },
}


def _path(public: Path, cell: str, kind: str) -> Path:
    if cell not in FILES or kind not in FILES[cell]:
        raise ValueError("Expected one of the registered public cells and split roles")
    path = public / cell / f"{kind}.json"
    if (
        not path.resolve().is_relative_to(public.resolve())
        or file_hash(path) != FILES[cell][kind]
    ):
        raise ValueError(f"Public file differs from release manifest: {cell}/{kind}")
    return path


def _code() -> dict:
    return {
        "script": file_hash(Path(__file__)),
        "balance_runtime": balance._code(),
        "public_reader": file_hash(
            Path(__file__).with_name("trajectory_function_intake.py")
        ),
    }


def independent_estimate(rows, stride: int) -> dict:
    """Fit downstream discharge directly from signed interval volume changes."""
    if stride not in balance.STRIDES:
        raise ValueError("Unexpected sampling stride")
    samples, events = [], {}
    for row in rows:
        if not row.trajectory_id.startswith("train_"):
            raise ValueError("Downstream discharge fitting requires training events")
        t = np.asarray(row.time)
        keep = np.unique(np.r_[np.arange(0, len(t), stride), len(t) - 1])
        inlet = np.asarray(row.inputs["inflow_down"])
        inflow = np.r_[0.0, np.cumsum(np.diff(t) * (inlet[1:] + inlet[:-1]) / 2)]
        h = np.asarray(row.targets["h_down"])[keep]
        head = h - row.covariates["crest_down"]
        area = row.covariates["area_down"]
        if area <= 0 or len(t) < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("Expected a positive surveyed area and increasing times")
        volume = np.diff(inflow[keep]) - area * np.diff(h)
        samples.append(
            balance.BalanceSample(head[:-1], head[1:], np.diff(t[keep]), volume)
        )
        events[row.trajectory_id] = {
            "intervals": len(volume),
            "negative_outlet_intervals": int(np.sum(volume < -1e-9)),
        }
    return {
        "stride": stride,
        "downstream": balance.fit_power(samples),
        "events": events,
    }


def _independent_prediction(t, initial, inputs, c, law, refinement):
    """Roll out only the downstream depth from its own runoff and measured initial."""

    def rhs(when, state):
        discharge = law["gain"] * max(state[0] - c["crest_down"], 0.0) ** law["power"]
        return [
            (np.interp(when, t, inputs["inflow_down"]) - discharge) / c["area_down"]
        ]

    sol = solve_ivp(
        rhs,
        (t[0], t[-1]),
        [initial],
        t_eval=t,
        method="DOP853",
        rtol=1e-8 / refinement**2,
        atol=1e-10 / refinement**2,
        max_step=float(np.min(np.diff(t))) / refinement,
    )
    if not sol.success or not np.all(np.isfinite(sol.y)):
        raise ArithmeticError("Independent downstream integration failed")
    return sol.y[0]


def independent_score(rows, frozen):
    if frozen["laws"] is None:
        return {"status": "fit_unavailable", "pooled_nmse": None}
    scale, law = frozen["training_depth_std_m"], frozen["laws"]["downstream"]
    events, errors = [], []
    grid, response = 0.0, 0.0
    for row in rows:
        t = np.asarray(row.time)
        y = np.asarray(row.targets["h_down"])
        inputs = row.inputs
        shifted = {**inputs, "inflow_up": 10.0 + 2.0 * np.asarray(inputs["inflow_up"])}
        args = (t, y[0], inputs, row.covariates, law)
        try:
            coarse, fine = (_independent_prediction(*args, n) for n in (2, 4))
            counter = _independent_prediction(t, y[0], shifted, row.covariates, law, 4)
        except (ArithmeticError, ValueError) as exc:
            return {
                "status": "rollout_failed",
                "reason": str(exc),
                "event": row.trajectory_id,
                "pooled_nmse": None,
            }
        error = (fine - y) / scale
        change = float(np.max(abs(fine - counter)))
        errors.extend(error.tolist())
        response = max(response, change)
        grid = max(grid, float(np.max(abs(fine - coarse))) / scale)
        events.append(
            {
                "event": row.trajectory_id,
                "time": t.tolist(),
                "observed": y.tolist(),
                "predicted": fine.tolist(),
                "nmse": float(np.mean(error**2)),
                "minimum_downstream_m": float(min(fine)),
                "upstream_intervention_max_change_m": change,
            }
        )
    usable = grid <= 0.01 and response == 0.0
    return {
        "status": "scored" if usable else "prediction_check_failed",
        "pooled_nmse": float(np.mean(np.square(errors))) if usable else None,
        "events": events,
        "max_scaled_grid_difference": grid,
        "upstream_intervention_max_change_m": response,
    }


def train(public: Path, root: Path) -> dict:
    """Freeze all laws from training strata, with stride one selected in advance."""
    if root.resolve().is_relative_to(public.resolve()):
        raise ValueError("Use an output directory outside the public releases")
    with _lock(root):
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": "basin-release-controls-1",
                "code": _code(),
                "files": FILES,
                "power_grid": list(balance.PROFILE),
                "strides": list(balance.STRIDES),
                "selection": "Use stride 1. Check sensitivity at 2 and 4.",
                "noise": "Retain signed volumes; noise may create negative estimates.",
                "independent_upstream_intervention": (
                    "Replace u_up(t) by 10+2*u_up(t). Downstream inputs and "
                    "initials stay fixed."
                ),
                "test_data_opened": False,
            },
        )
        cases = {}
        for cell in FILES:
            spec = read_seal(_path(public, cell, "specification"))
            if spec["benchmark_id"] != cell or spec["targets"] != ["h_down"]:
                raise ValueError("Unexpected public specification")
            independent = spec["negative_control"]
            rows = _rows(public_rows(_path(public, cell, "train")), "train")
            scale = max(
                float(np.std(np.concatenate([x.targets["h_down"] for x in rows]))), 1e-8
            )
            estimates = []
            for stride in balance.STRIDES:
                path = root / cell / f"stride-{stride}.json"
                identity = {
                    "plan": plan["artifact_sha256"],
                    "cell": cell,
                    "stride": stride,
                }
                if path.exists():
                    receipt = sealed_read(path)
                    if any(receipt[k] != v for k, v in identity.items()):
                        raise ValueError(
                            "Training receipt differs from the registered cell"
                        )
                else:
                    value = (
                        independent_estimate(rows, stride)
                        if independent
                        else balance.estimate(rows, stride)
                    )
                    receipt = sealed_write(path, {**identity, **value})
                estimates.append(receipt)
            best = estimates[0]
            sides = ["downstream"] if independent else ["downstream", "upstream"]
            laws = (
                {s: best[s]["best"] for s in sides}
                if all(best[s]["status"] == "estimated" for s in sides)
                else None
            )
            cases[cell] = {
                "independent": independent,
                "noise_sd_fraction": spec["noise_sd_fraction"],
                "training_depth_std_m": scale,
                "laws": laws,
                "estimates": [r["artifact_sha256"] for r in estimates],
            }
        return sealed_write(
            root / "training.json",
            {
                "plan": plan["artifact_sha256"],
                "cases": cases,
                "validation_opened": False,
            },
        )


def evaluate(public: Path, root: Path, split: str) -> dict:
    """Predict training or validation events after all three laws are frozen."""
    if split not in {"train", "val"}:
        raise ValueError("Expected train or val")
    with _lock(root):
        plan, frozen = (sealed_read(root / n) for n in ("plan.json", "training.json"))
        if (
            plan["files"] != FILES
            or plan["code"] != _code()
            or frozen["plan"] != plan["artifact_sha256"]
            or set(frozen["cases"]) != set(FILES)
        ):
            raise ValueError("Release control code or frozen training identity differs")
        for cell, design in frozen["cases"].items():
            receipts = [
                sealed_read(root / cell / f"stride-{i}.json") for i in balance.STRIDES
            ]
            if [r["artifact_sha256"] for r in receipts] != design["estimates"]:
                raise ValueError("Training estimates differ from selection")
        cases = {}
        for cell, design in frozen["cases"].items():
            path = root / cell / f"{split}-rollout.json"
            identity = {
                "cell": cell,
                "split": split,
                "training": frozen["artifact_sha256"],
            }
            if path.exists():
                receipt = sealed_read(path)
                if any(receipt[k] != v for k, v in identity.items()):
                    raise ValueError("Rollout differs from frozen training laws")
            else:
                rows = _rows(public_rows(_path(public, cell, split)), split)
                value = (independent_score if design["independent"] else balance.score)(
                    rows, design
                )
                receipt = sealed_write(path, {**identity, **value})
            cases[cell] = {
                "laws": design["laws"],
                "status": receipt["status"],
                "pooled_nmse": receipt.get("pooled_nmse"),
                "events": len(receipt.get("events", [])),
                "upstream_intervention_max_change_m": receipt.get(
                    "upstream_intervention_max_change_m"
                ),
                "max_scaled_grid_difference": receipt.get("max_scaled_grid_difference"),
                "receipt": receipt["artifact_sha256"],
            }
        return sealed_write(
            root / f"{split}-summary.json",
            {
                "training": frozen["artifact_sha256"],
                "split": split,
                "cases": cases,
                "test_data_opened": False,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "validate"])
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "train":
        train(args.public, args.root)
    result = evaluate(
        args.public, args.root, "train" if args.stage == "train" else "val"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
