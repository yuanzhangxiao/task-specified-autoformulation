"""Project the recorded Candidate 15 attempt and reviewed training replay for ACES."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import insulin_boundary_revision as boundary
from autoformalism.research.construction_repair_identity import family_identity
from autoformalism.research.revision_spelling import normalize
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from scripts import describe_insulin_boundary as diagnostics
from scripts import propose_meal_insulin_revision as proposer

REVIEW_SHA256 = "b32e6d5e042d486de16c1ad29352490eaedae92f11fbfa5c78c99bcb91486df4"


def project(source: Path, previous: Path, audit_path: Path, fit_terminal: Path) -> dict:
    """Join the immutable prior attempt to the parent and two-solver diagnostic."""
    base = read_seal(source)
    plan = sealed_read(previous / "plan.json")
    inputs = sealed_read(previous / "inputs.json")
    result = proposer._terminal(previous, plan)
    audit = json.loads(audit_path.read_text())
    fit = read_seal(fit_terminal)
    if (
        content_sha256(base) != boundary.INTAKE_SHA256
        or content_sha256(audit) != REVIEW_SHA256
        or plan["protocol"] != boundary.PROTOCOL
        or plan["artifact_sha256"] != audit["plan_sha256"]
        or inputs["artifact_sha256"] != plan["inputs_sha256"]
        or content_sha256(fit) != inputs["result_identity"]["fit_terminal_sha256"]
        or result["artifact_sha256"] != audit["result_sha256"]
        or result["status"] != "rejected"
        or audit["test_data_opened"]
    ):
        raise ValueError("previous plan, terminal or reviewed training replay differs")
    args = base["arguments"]
    if (
        args["fitted_problem"] != inputs["problem"]
        or args["consistency_evidence"] != inputs["evidence"]
        or args["result_identity"] != inputs["result_identity"]
    ):
        raise ValueError("previous proposal used different fitted or training evidence")
    (call_path,) = (previous / "calls").glob("*.json")
    call = json.loads(call_path.read_text())
    if call["request"]["namespace"] != plan["artifact_sha256"]:
        raise ValueError("previous generation belongs to a different plan")
    raw = visible_response(call)
    parent = PrefixFitProblem.model_validate(inputs["problem"])
    patch, spellings = normalize(parent, raw, proposer.BoundaryRevision)
    assessed = proposer.assess(inputs, patch)
    fitted = diagnostics.feedback(inputs)["fitted_feedback"]
    messages = diagnostics.revision_messages(
        assessed,
        inputs["boundary_witness"],
        args["boundary_feedback"]["ingestion_windows"],
    )
    family = family_identity(parent, assessed["problem"])
    if family["status"] != "identified":
        raise ValueError("could not identify the previously reviewed equation family")
    example = audit["fixed_diagnostic_examples"]["domain_witness"]
    row = next(
        row
        for row in inputs["evidence"]["trajectories"]
        if row["trajectory_id"] == example["trajectory"]
    )
    parameters = fit["retained"]["parameters"]
    roots = {
        state: parameters[f"trajectory:{row['trajectory_id']}:init_{state}_value"]
        for state in ("A", "X_insulin_action")
    }
    history = {
        "source_plan_sha256": plan["artifact_sha256"],
        "source_result_sha256": result["artifact_sha256"],
        "raw_reply_sha256": content_sha256(raw),
        "review_sha256": REVIEW_SHA256,
        "fitted_problem_sha256": content_sha256(parent),
        "previous_family_sha256": family["sha256"],
        "fitted_feedback": fitted,
        "previous_attempt": {
            "status": "Unfitted. The original reply was rejected for two spellings. "
            "The equations below are the diagnostic copy with those names resolved.",
            "spelling_resolutions": spellings,
            "patch": {
                k: patch[k]
                for k in ("edit", "new_states", "new_processes", "new_parameters")
            },
            "previous_claim_about_depletion": patch["insulin_rate_at_zero"],
            "deterministic_feedback": messages,
            "fixed_parameter_replay": {
                "trajectory": example["trajectory"],
                "schedule": "150 g meal at minutes 0-10, zero external insulin",
                "values": "Inherited primary constrained-fit parameters and training "
                "initials, with the following values for the previous additions.",
                "new_parameters": {
                    "ingest_start": example["s"],
                    "ingest_end": example["e"],
                    "tau_rise": example["tau_rise"],
                    "tau_fall": example["tau_fall"],
                },
                "initial_states": {
                    "Gp": row["targets"]["Gp"][0],
                    "I": row["targets"]["I"][0],
                    "af_internal_aux_1": row["auxiliaries"]["Uii"][0],
                    **roots,
                    "direct_ingestion": example["D0"],
                },
                "preceding_minute": example["first_negative_minute"] - 1,
                "preceding_I_pmol_L": example["I_at_preceding_sample"],
                "first_negative_sample_min": example["first_negative_minute"],
                "first_negative_I_pmol_L": example["I_at_first_negative_sample"],
                "observed_I_at_that_sample_pmol_L": row["targets"]["I"][
                    row["time"].index(example["first_negative_minute"])
                ],
                "max_solver_difference_I_pmol_L": example[
                    "max_abs_Radau_DOP853_I_difference"
                ],
                "interpretation": "These admissible values contradict the previous "
                "claim of guaranteed nonnegativity. "
                "The revision has never been fitted.",
            },
        },
    }
    return {
        "protocol": boundary.REPAIR_INTAKE_PROTOCOL,
        "arguments": {**args, "repair_history": history},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--fit-terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    value = project(
        options.source, options.previous, options.audit, options.fit_terminal
    )
    seal(options.output, value)
    print(content_sha256(value))
