#!/usr/bin/env python3
"""Real two-stage polishing on a small independent system; never the hard case."""

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import polishing_campaign as campaign
from autoformalism.fitting import polishing_fit
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import screening_replay as replay
from tests.profiled_fixture import TRUTH, make_base


def run(root: Path):
    base, validation = make_base()
    policy = campaign.PolishingPolicy(
        seconds=90,
        certificate_seconds=15,
        replay_seconds=30,
        training_nmse=1e-5,
        trajectory_nmse=1e-4,
        polish_training_nmse=1e-10,
        polish_trajectory_nmse=1e-9,
    )
    rows = []
    for arm in campaign.ARMS:
        folder = root / arm
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "summary.json").exists():
            raise ValueError("choose a fresh smoke directory")
        fitted = polishing_fit.fit(
            base, arm, policy.model_dump(mode="json"), folder / "fit"
        )
        assert fitted["selected"], fitted
        assert fitted["first_prediction"] and fitted["polishing_attempted"], fitted
        assert fitted["actual_residual_calls"] <= policy.maximum_rollout_calls
        assert fitted["fit_seconds"] < policy.seconds + 1
        selected = fitted["selected"]["parameters"]
        assert fitted["stop_reason"] == "training_prediction_certified", fitted
        errors = {k: abs(selected[k] - v) for k, v in TRUTH.items()}
        assert max(errors.values()) < 0.001, errors
        processes = list((folder / "fit").rglob("process.json"))
        assert all(read_seal(p)["termination_confirmed"] for p in processes)
        (folder / "evaluation").mkdir(exist_ok=True)
        evaluation = replay._evaluation(
            folder / "evaluation",
            {
                "case": {
                    "training": base["training"],
                    "validation": validation,
                    "reference_parameters": TRUTH,
                },
                "case_name": "linear",
                "config": {"parameter_relative": 0.01, "initial_absolute": 0.01},
            },
            {"request": base["request"]},
            selected,
            30,
        )
        assert evaluation["accuracy_passed"] and evaluation["coefficients_recovered"], (
            evaluation
        )
        # Resuming the coordinator never spends a fresh fitting budget.
        stamps = {p: p.stat().st_mtime_ns for p in processes}
        resumed = polishing_fit.fit(
            base, arm, policy.model_dump(mode="json"), folder / "fit"
        )
        assert resumed["selected"] == fitted["selected"]
        assert stamps == {p: p.stat().st_mtime_ns for p in processes}
        row = {
            "arm": arm,
            "seconds": fitted["fit_seconds"],
            "first_training_nmse": fitted["first_prediction"]["certificate"][
                "training_nmse"
            ],
            "polishing_seconds": fitted["polishing_seconds"],
            "residual_calls": fitted["actual_residual_calls"],
            "evaluation": evaluation,
            "max_parameter_absolute_error": max(errors.values()),
            "resume_spent_no_budget": True,
        }
        public._write(folder / "summary.json", row)
        rows.append(row)
    result = {"status": "pass", "rows": rows, "hard_case_fitted": False}
    public._write(root / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().root), indent=2))
