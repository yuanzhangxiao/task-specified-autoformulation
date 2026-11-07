"""Project the reviewed constrained continuation into one training-only request."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.expressions import compile_candidate
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import insulin_boundary_revision as boundary
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PRIMARY = "nested-zero-positive"
FREE = "nested-zero-unconstrained"


def _windows(training: PublicSplit, terminal: dict) -> list[dict]:
    """Recover measured and predicted changes at the three 90 g meal windows."""
    replay = terminal["metrics"]
    channels = replay["channels"]
    insulin = channels.index("I")
    residual = np.array(replay["residual"]).reshape(-1, len(channels))
    shifted = {
        row["trajectory"]: row
        for row in replay["metrics"]["rows"]
        if row["matches_shifted_90g_schedule"] and row["external_insulin_zero"]
    }
    windows = []
    offset = 0
    for row in training.rows:
        length = len(row.time)
        if row.trajectory_id in shifted:
            record = shifted[row.trajectory_id]
            (meal,) = record["meal_events"]
            start, end = (row.time.index(meal[k]) for k in ("start_min", "end_min"))
            observed = row.targets["I"][end] - row.targets["I"][start]
            difference = (
                residual[offset + end, insulin] - residual[offset + start, insulin]
            )
            windows.append(
                {
                    "trajectory": row.trajectory_id,
                    "meal": meal,
                    "observed_rise_pmol_L": observed,
                    "fitted_rise_pmol_L": observed + replay["scales"]["I"] * difference,
                    "direct_ingestion_contribution_pmol_L": (
                        terminal["retained"]["parameters"]["c_I"] * meal["dose_g"]
                    ),
                    "observed_peak_min": record["observed_insulin_peak_min"],
                    "fitted_peak_min": record["predicted_insulin_peak_min"],
                }
            )
        offset += length
    if offset != len(residual) or len(windows) != 3:
        raise ValueError("expected three shifted meals in the complete training replay")
    return windows


def project(review: Path, construction: Path) -> dict:
    """Verify fit/model identities and select the primary matched pair."""
    root = review / "returned"
    plan, data = (read_seal(root / f"{name}.json") for name in ("plan", "inputs"))
    plan_hash = content_sha256(plan)
    pos, free = (
        read_seal(root / "results" / arm / "result.json") for arm in (PRIMARY, FREE)
    )
    documents = data["source_inputs"]["documents"]
    parent = sealed_read(construction / "result.json")
    old_plan = sealed_read(construction / "plan.json")
    construction_id = old_plan["artifact_sha256"]
    context = sealed_read(construction / "inputs.json")
    training = PublicSplit.model_validate(documents["training"])
    train_hash = content_sha256(training)
    model_hash = content_sha256(documents["problem"])
    if (
        plan["protocol"] != "insulin-positive-continuation-1"
        or plan["validation_or_test_access"] is not False
        or content_sha256(data) != plan["inputs_sha256"]
        or training.name != "train"
        or len(training.rows) != 16
        or documents["fitting_contract"]["training_sha256"] != train_hash
        or context["fitting_contract"]["training_sha256"] != train_hash
        or construction_id != documents["provenance"]["proposal_plan_sha256"]
        or parent["status"] != "eligible_for_refit"
        or parent["plan_sha256"] != construction_id
        or old_plan["inputs_sha256"] != context["artifact_sha256"]
        or parent["refit_handoff"]["problem_sha256"] != model_hash
        or content_sha256(parent["problem"]) != model_hash
        or not pos["domain_admissible"]
        or free["domain_admissible"]
    ):
        raise ValueError("expected the reviewed public training model and matched fits")
    for name, result in ((PRIMARY, pos), (FREE, free)):
        if (
            result["identity"]["task"]["task_id"] != name
            or result["identity"]["plan_sha256"] != plan_hash
            or result["status"] != "complete"
            or result["validation_or_test_access"]
            or not result["retained"]["original_equations_verified"]
            or result["metrics"]["problem_sha256"] != model_hash
            or result["metrics"]["metrics"]["training_sha256"] != train_hash
        ):
            raise ValueError("continuation terminal differs from the admitted fit")
    model = PrefixFitProblem.model_validate(parent["problem"])
    model = model.model_copy(
        update={
            "parameter_guesses": {
                name: pos["retained"]["parameters"][name]
                for name in model.parameter_guesses
            }
        }
    )
    compiled = compile_candidate(model.base_candidate, model.context)
    analysis = json.loads((review / "analysis.json").read_text())
    if analysis["plan_sha256"] != plan_hash:
        raise ValueError("review does not describe the sealed continuation")
    case = next(row for row in analysis["tasks"] if row["task"] == PRIMARY)
    point = case["observed_150g_boundary"]
    row = next(r for r in training.rows if r.trajectory_id == point["trajectory"])
    at_zero = {**point["state"], "I": 0.0}
    inputs = point["input_values"]
    time = point["t"]
    if (
        row.trajectory_id != "train_005"
        or time != row.time[-1]
        or abs(point["state"]["I"])
        > pos["concentration_audit"]["outputs"]["I"]["physical_tolerance"]
        or inputs != {boundary.MEAL: 0, boundary.INFUSION: 0}
        or any(row.external_inputs[key][-1] != value for key, value in inputs.items())
    ):
        raise ValueError("expected the terminal 150 g training depletion witness")

    class ZeroInputs:
        def value(self, channel, time):
            return inputs[channel]

    rate = compiled.rhs(
        time,
        [at_zero[n] for n in compiled.state_names],
        model.parameter_guesses,
        ZeroInputs(),
    )[compiled.state_names.index("I")]
    if rate >= 0 or not math.isclose(
        rate, point["rhs_with_I_set_to_zero"], abs_tol=1e-9
    ):
        raise ValueError("compiled boundary rate differs from the reviewed fit")
    extension = next(
        x
        for x in json.loads((review / "boundary-probe.json").read_text())["rows"]
        if x["fit"] == PRIMARY
    )
    if (
        extension["trajectory"] != row.trajectory_id
        or extension["fixed_inputs"] != inputs
        or extension["times"][-1] != time + 1
        or extension["observation_at_301_available"]
        or not math.isclose(
            extension["predicted_I"][-1],
            extension["matrix_exponential_final_I"],
            rel_tol=1e-9,
            abs_tol=1e-10,
        )
    ):
        raise ValueError("unexpected one-minute diagnostic")
    reference = pos["metrics"]["metrics"]
    witness = {
        "trajectory": row.trajectory_id,
        "meal_dose_g": 150,
        "time_min": time,
        "observed_I_pmol_L": row.targets["I"][-1],
        "free_fit_I_pmol_L": next(t for t in analysis["tasks"] if t["task"] == FREE)[
            "observed_150g_boundary"
        ]["predicted_I"],
        "constrained_fit_I_pmol_L": point["predicted_I"],
        "state_at_boundary": at_zero,
        "inputs": inputs,
        "reference_rate_pmol_L_min": float(rate),
        "glucose_source_pmol_L_min": point["rhs_state_terms"]["Gp"],
        "delayed_action_term_pmol_L_min": point["rhs_state_terms"]["X_insulin_action"],
    }
    feedback = {
        "scope": "Sixteen public training trajectories. Start from the primary "
        "concentration-constrained fit. Both matched searches used their "
        "time allowance.",
        "free_training_nmse": free["training_reconstruction_nmse"],
        "constrained_training_nmse": pos["training_reconstruction_nmse"],
        "constrained_dense_grid": pos["concentration_audit"]["outputs"]["I"],
        "free_dense_grid": free["concentration_audit"]["outputs"]["I"],
        "depletion": witness,
        "zero_input_extension": {
            "scope": "Post-hoc model extrapolation from the fitted terminal state.",
            "time_min": time + 1,
            "predicted_I_pmol_L": extension["predicted_I"][-1],
            "observed_I_available": False,
        },
        "ingestion_windows": _windows(training, pos),
    }
    return {
        "protocol": boundary.INTAKE_PROTOCOL,
        "arguments": {
            "fitted_problem": model.model_dump(mode="json"),
            "public_brief": context["brief"],
            "scientific_aliases": context["aliases"],
            "consistency_evidence": context["evidence"],
            "comparison": {"child_metrics": reference},
            "boundary_feedback": feedback,
            "fitting_contract": {
                **documents["fitting_contract"],
                "matched_policy": plan["policy"],
            },
            "result_identity": {
                "continuation_plan_sha256": plan_hash,
                "reference_arm": PRIMARY,
                "endpoint_sha256": content_sha256(pos["retained"]),
                "reference_model_sha256": model_hash,
                "training_sha256": train_hash,
                "fit_terminal_sha256": content_sha256(pos),
                "archive_sha256": analysis["archive"]["archive_sha256"],
                "boundary_probe_sha256": hashlib.sha256(
                    (review / "boundary-probe.json").read_bytes()
                ).hexdigest(),
            },
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--construction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    intake = project(args.review, args.construction)
    seal(args.output, intake)
    print(json.dumps({"reviewed_intake_sha256": content_sha256(intake)}))
