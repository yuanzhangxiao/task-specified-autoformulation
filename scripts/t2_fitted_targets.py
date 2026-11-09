"""Prepare the two fitted T2 parents and their public meal timing/recovery witnesses."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.expressions import ModelValidationError
from autoformalism.expressions.parser import APPROVED_FUNCTION_ARITY
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.rebuttal import basin_algebra
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import insulin_boundary_revision as boundary
from autoformalism.research import t2_breadth as science
from autoformalism.research.construction_revision_edits import (
    RevisionValidationError,
    model_view,
)
from autoformalism.research.construction_structural_revision import (
    editable_initial_states,
)
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from scripts.construction_functions import file_hash

PROTOCOL = "t2-fitted-timing-recovery-1"
INPUT_PROTOCOL = "t2-fitted-timing-recovery-training-1"
GROUPS = ("c15-3", "c15-2")
OBJECTIVES = {"c15-3": "shifted_meal_peak_timing", "c15-2": "late_150g_recovery"}
INTAKE = "9a68a3e7c8231fbfc721783e663af94ed5b3937b46a6a4812e4ab48449c26a72"
MATCHED_PLAN = "bbd73e98558b9800e8d1e61b983649e2f516f6cc952dc2b5d992ffa190b12f0b"
INDEPENDENT_PLAN = "31a962352ba61c98299fe7b589f8dfb7a9f800f9f55491c6e66b04070b640b14"
# Reviewed public training sources for the portable intake.
PUBLIC = "t2-breadth-results-20261108/returned/inputs.json"
MODELS = "t2-breadth-fitting-20261108/verified-campaign/fitted-models.json"
ENDPOINTS = "t2-breadth-fitting-20261108/convergence-check/convergence-evidence.json"
INDEPENDENT = "t2-breadth-independent-starts-20261109/fitted-models.json"
FEEDBACK = "t2-breadth-fitted-review-20261109/construction-feedback.json"
TERMS = "t2-equation-attribution-20261109/results/equation-evidence.json"
SOURCES = {
    PUBLIC: "db0b5e78406147a8482c1b05f9d2463dacddbb2937f4f0f542ba9d74700a064c",
    MODELS: "514db25a87e62d535a3d83eb0d0384927b8e4f339a1f20aa43914f4035dac490",
    ENDPOINTS: "ee5499759b8664bcbbcff42bfd840ab1a550ea5402bef924f4fc62a8e15bdbc0",
    INDEPENDENT: "d97bf553c753d4725b66b3ae43aeb20395ef82f9f0010d9d1166561d5dbfb4b4",
    FEEDBACK: "8c8dee0bed3c2e994ebdd5443d37599df99da05ec768c6809d5df0c7d29ce745",
    TERMS: "c54bf1bf1e823d1bcf2e46ebfbec750ad329db88f793193c13c9a50bae05053b",
}


class FiveFittedAlternatives(science.FiveGlucoseRepairs):
    """Five independent patches against the same fitted equations."""


def _source(directory: Path, filename: str) -> dict:
    path = directory / filename
    if filename == FEEDBACK:
        digest, value = file_hash(path), json.loads(path.read_text())
    elif filename == PUBLIC:
        value = sealed_read(path)
        digest = value["artifact_sha256"]
    else:
        value = read_seal(path)
        digest = content_sha256(value)
    if digest != SOURCES[filename]:
        raise ValueError(f"Reviewed fitting or training source differs: {filename}")
    return value


def _rate_witness(group: str, signed: dict) -> dict:
    """Present measured changes and fitted rate terms without selecting a repair."""
    if group == "c15-3":
        row = signed["short-memory"]["meal_balance"]["train_006"]
        return {
            "fit": "independent short-memory start",
            "trajectory": "train_006",
            "meal_end_min": row["meal_end"],
            "meal_memory_drive_peak_min": row["memory_drive_peak_min"],
            "window_min": [row["predicted_peak"], row["observed_peak"]],
            "fitted_I_change_pmol_L": row["predicted_I_change"],
            "observed_I_change_pmol_L": row["observed_I_change"],
            "signed_insulin_rates_pmol_L_min": {
                minute: {
                    k: row["snapshots"][minute][k]
                    for k in ("memory", "glucose", "action", "net_self", "I_dot")
                }
                for minute in ("70", "128", "161")
            },
        }
    row = signed["shallow-sigmoid"]["recovery"]
    return {
        "fit": "independent shallow-sigmoid start",
        "trajectory": "train_005",
        "window_min": [240, 300],
        "reservoir_decay_time_min": row["appearance_timescale"],
        "fitted_mean_supply_from_k_abs_A_mg_kg_min": row["mean_appearance"],
        "implied_mean_net_supply_mg_kg_min": row[
            "mean_supply_implied_by_recorded_Gp_and_U"
        ],
        "fitted_mean_disposal_mg_kg_min": row["mean_fitted_disposal"],
        "observed_mean_disposal_mg_kg_min": row["mean_observed_disposal"],
        "interpretation": (
            "The recorded change in Gp plus integrated U identifies mean net supply. "
            "Its physiological components remain unresolved."
        ),
        "I300_with_recorded_Gp_from_fitted_I240_pmol_L": row[
            "recorded_glucose_drive_I300_from_fitted_I240"
        ],
        "substitution_scope": "Diagnostic replay with measured glucose as a drive.",
    }


def build_intake(directory: Path, destination: Path) -> dict:
    """Join fitted equations, signed coefficients, roots and training witnesses."""
    source = _source(directory, PUBLIC)
    exchange = _source(directory, MODELS)
    convergence = _source(directory, ENDPOINTS)
    independent = _source(directory, INDEPENDENT)
    feedback = _source(directory, FEEDBACK)
    terms = _source(directory, TERMS)
    training = source["training"]
    training_hash = content_sha256(training)
    if (
        training["name"] != "train"
        or source["test_data_opened"]
        or source["evidence"]["split"] != "train"
        or exchange["validation_or_test_access"]
        or exchange["prefix_forecast_evaluated"]
        or independent["prefix_forecasts"]
        or convergence["held_out_data_accessed"]
        or not convergence["training_only"]
        or feedback["split"] != "public training"
        or exchange["training_sha256"] != training_hash
        or independent["training_sha256"] != training_hash
        or source["fitting_contract"]["training_sha256"] != training_hash
        or terms["plan"]["training_sha256"] != training_hash
        or terms["plan"]["matched_plan"] != MATCHED_PLAN
        or terms["plan"]["independent_plan"] != INDEPENDENT_PLAN
        or exchange["prefix_policy"] != source["fitting_contract"]["prefix_policy"]
        or {
            exchange["plan_sha256"],
            convergence["matched_plan_sha256"],
            feedback["matched_plan"],
        }
        != {MATCHED_PLAN}
        or {independent["plan_sha256"], feedback["independent_plan_sha256"]}
        != {INDEPENDENT_PLAN}
    ):
        raise ValueError("Fitted sources, forecast policy or training splits disagree")
    values = {
        "protocol": INPUT_PROTOCOL,
        "sources": SOURCES,
        "matched_plan_sha256": MATCHED_PLAN,
        "independent_plan_sha256": INDEPENDENT_PLAN,
        "training": training,
        "evidence": source["evidence"],
        "brief": source["brief"],
        "aliases": source["aliases"],
        "fitting_contract": {
            "training_sha256": training_hash,
            "prefix_policy": exchange["prefix_policy"],
        },
        "test_data_opened": False,
    }
    exported = {row["task"]: row for row in exchange["models"]}
    fitted = {row["branch"]: row for row in convergence["models"]}
    training_rows = {r["trajectory_id"]: r for r in training["rows"]}
    high = training_rows["train_005"]
    early, late = [high["time"].index(t) for t in (240, 300)]
    for name in GROUPS:
        record = exported[name]
        source_problem = record["problem"]
        endpoint = fitted[name]["best_verified_endpoint"]
        point = endpoint["parameters"]
        witness = feedback["records"][name]
        shape_starts = witness["independent_starts"]
        verified = {r["task"]: r for r in independent["models"] if r["branch"] == name}
        if (
            content_sha256(source_problem) != record["model_sha256"]
            or fitted[name]["model_sha256"] != record["model_sha256"]
            or content_sha256(point) != witness["fitted_parameter_sha256"]
            or endpoint["parameter_sha256"] != witness["fitted_parameter_sha256"]
            or not endpoint["sampled_domain_passed"]
            or not endpoint["two_original_solvers_verified"]
            or endpoint["validation_or_test_access"]
            or len(verified) != 2
            or set(verified) != set(shape_starts)
            or any(
                r["model_sha256"] != record["model_sha256"]
                or r["retained"]["parameter_sha256"]
                != shape_starts[key]["parameter_sha256"]
                or terms["signed"][key]["parameter_sha256"]
                != r["retained"]["parameter_sha256"]
                for key, r in verified.items()
            )
        ):
            raise ValueError(
                f"Fitted parent or replicated training witness differs for {name}"
            )
        problem = {
            **source_problem,
            "parameter_guesses": {
                k: point[k] for k in source_problem["parameter_guesses"]
            },
        }
        peaks = [
            {
                "trajectory": r["trajectory"],
                "meal_start_min": r["meal_events"][0]["start_min"],
                "observed_peak_min": r["observed_insulin_peak_min"],
                "fitted_peak_min": r["predicted_insulin_peak_min"],
            }
            for r in endpoint["shifted_90g"]
        ]
        values[name] = {
            "problem": problem,
            "parent_reference": {
                "branch": name,
                "original_model_sha256": record["model_sha256"],
                "fitted_problem_sha256": content_sha256(problem),
                "fitted_parameter_sha256": content_sha256(point),
                "construction_handoff_sha256": record["source_handoff_sha256"],
                "fit_origin": "supplementary verified signed endpoint",
            },
            "parameters_with_training_roots": point,
            "fit": {
                key: endpoint[key]
                for key in (
                    "training_nmse",
                    "target_nmse",
                    "trajectory_median_nmse",
                    "trajectory_unscaled_mad",
                )
            },
            "witnesses": {
                "training_rate_terms": _rate_witness(name, terms["signed"]),
                "shifted_meal_peak_timing": peaks,
                "late_150g_recovery": {
                    "trajectory": "train_005",
                    "time_window_min": [240, 300],
                    "observed_I_rise_pmol_L": high["targets"]["I"][late]
                    - high["targets"]["I"][early],
                    "minute_300": witness["150g_minute_300"],
                },
                "independent_starts": {
                    k: {
                        f: v[f]
                        for f in (
                            "pooled",
                            "I300",
                            "I_rise_240_to_300",
                            "shifted_90g_peaks",
                        )
                    }
                    for k, v in shape_starts.items()
                },
            },
        }
        if name == "c15-2":
            values[name]["witnesses"]["local_rhs_check"] = witness[
                "instantaneous_RHS_check"
            ]
        else:
            values[name]["witnesses"]["fitted_memory"] = witness["memory"]
    return sealed_write(destination, values)


INSTRUCTIONS = {
    "c15-3": (
        "Revise the fitted meal-memory model to move the plasma-insulin peaks later "
        "after each shifted 90 g meal. Compare elapsed time from ingestion across "
        "schedules. Preserve the useful 150 g late recovery and the glucose and "
        "disposal behavior. Explain how each patch changes the time course."
    ),
    "c15-2": (
        "Revise the fitted glucose-sigmoid model to reduce the spurious late "
        "insulin rise on the 150 g meal, especially minutes 240 to 300. Preserve its "
        "shifted 90 g peak timing and the glucose and disposal behavior. You may "
        "revise upstream glucose dynamics, secretion, clearance or their coupling. "
        "The terminal RHS check substitutes recorded glucose at fixed fitted insulin. "
        "The late-window replay instead evolves insulin from its fitted minute-240 "
        "value using recorded glucose as a diagnostic drive."
    ),
}
FIVE = (
    "Return exactly five independently applicable alternatives. Give each a distinct "
    "mechanism or functional change for this parent's target. Write one complete "
    "parent-relative patch per option. Each option supplies its own definitions."
)
EDITING = (
    "Start each alternative from the displayed equations. Choose the mechanism "
    "and functional form. Replace complete right-hand sides where needed. You may "
    "add or remove private quantities and edit the listed latent initializers. "
    "Supply an equation and initializer for each new state and declare every new "
    "coefficient. Put coefficients in new_parameters. Keep existing public outputs, "
    "initial readings and modeled auxiliary identities. After the eight-minute "
    "forecast prefix, use generated quantities and declared external inputs. "
    "Tie each response to its modeled input history or state as meals shift in time. "
    "The target meals have zero insulin infusion. The infusion gain is a separate "
    "fitting control. Cite the training evidence for your change and "
    "describe the observable improvement and behavior to preserve."
)


def packet(data: dict, group: str) -> tuple[str, dict]:
    """Show a compact, exact equation view and witnesses specific to this parent."""
    entry = data[group]
    problem = PrefixFitProblem.model_validate(entry["problem"])
    aliases = data["aliases"]
    view = model_view(problem, scientific_aliases=aliases)
    return INSTRUCTIONS[group] + "\n" + EDITING, {
        "public_task": data["brief"]["scientific_context"],
        "parent": group,
        "primary_evidence_id": OBJECTIVES[group],
        "evidence_ids": list(entry["witnesses"]),
        "training_witnesses": entry["witnesses"],
        "fitted_training_scores": entry["fit"],
        "equations": [r["display"] for r in view["equations"]],
        "readouts": view["observation_mappings"],
        "initialization": view["initialization_graph"],
        "editable_initial_states": editable_initial_states(
            problem, scientific_aliases=aliases
        ),
        "scientific_aliases": aliases,
        "future_external_inputs": data["fitting_contract"]["prefix_policy"][
            "future_inputs"
        ],
        "allowed_function_names": sorted(APPROVED_FUNCTION_ARITY),
        "fitted_coefficients": [
            {
                "name": p["name"],
                "value": problem.parameter_guesses[p["name"]],
                "domain": p["domain"],
                "role": p["role"],
            }
            for p in view["parameters"]
        ],
        "parameter_source": entry["parent_reference"].get(
            "fit_origin", "signed training fit"
        ),
    }


def _zero_infusion_key(node: ast.AST) -> tuple:
    """Specialize total polynomials and retain other operations literally."""
    polynomial = boundary._polynomial(node)
    if polynomial is not None:
        reduced = boundary._at_zero(polynomial, {boundary.INFUSION})
        return ("polynomial", basin_algebra.render(reduced))
    if isinstance(node, ast.BinOp):
        left, right = _zero_infusion_key(node.left), _zero_infusion_key(node.right)
        if isinstance(node.op, (ast.Add, ast.Sub)) and right == ("polynomial", "0"):
            return left
        if isinstance(node.op, ast.Add) and left == ("polynomial", "0"):
            return right
        return type(node.op).__name__, left, right
    if isinstance(node, ast.Call):
        return "call", node.func.id, tuple(_zero_infusion_key(a) for a in node.args)
    if isinstance(node, ast.UnaryOp):
        return type(node.op).__name__, _zero_infusion_key(node.operand)
    return ("literal", ast.dump(node))


def _meal_system(problem: PrefixFitProblem) -> dict:
    c = problem.base_candidate
    laws = {p.name: p.expression for p in c.processes}

    def key(rhs):
        return _zero_infusion_key(basin_algebra.expanded(rhs, laws))

    return {
        "initials": {i.state: i.model_dump(mode="json") for i in c.initial_conditions},
        "graph": problem.graph.model_dump(mode="json"),
        "dynamics": {e.state: key(e.rhs) for e in c.state_equations},
        "outputs": {m.channel: key(m.expression) for m in c.observation_mappings},
    }


def check_meal_edit(parent: PrefixFitProblem, child: PrefixFitProblem) -> None:
    """Reject a proven unchanged complete system on the target no-infusion meals."""
    try:
        same = _meal_system(parent) == _meal_system(child)
    except (ModelValidationError, ValueError, ArithmeticError, RecursionError):
        # The normal compiler and public checks still apply outside this proof.
        return
    if same:
        raise RevisionValidationError(
            "UNCHANGED_ON_TARGET_MEALS",
            "At zero infusion all state equations, readouts and initializers still "
            "give the parent system. Change a relation that can affect the targeted "
            "meal trajectories. Retain useful parts of your previous patch.",
        )


def claim_alternative(name: str, outcome: dict, admitted: dict) -> None:
    """Give an equivalent sibling a chance to supply a genuinely different patch."""
    c = outcome["problem"]["base_candidate"]
    exact = content_sha256(
        {
            "laws": c["state_equations"],
            "processes": {p["name"]: p["expression"] for p in c["processes"]},
            "readouts": c["observation_mappings"],
            "initials": c["initial_conditions"],
            "graph": outcome["problem"]["graph"],
            "constraints": c["constraints"],
            "parameters": c["parameters"],
        }
    )
    keys = [
        k
        for k in (exact, outcome["family"], outcome["balances"]["balance_sha256"])
        if k
    ]
    for key in keys:
        if key in admitted:
            sibling, hypothesis = admitted[key]
            raise RevisionValidationError(
                "DUPLICATE_ALTERNATIVE",
                f"This patch reproduces {sibling}. Choose a different mechanism and "
                "return a complete patch against the fitted parent.",
                sibling=sibling,
                sibling_hypothesis=hypothesis,
            )
    for key in keys:
        admitted[key] = (name, c["change_summary"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    intake = build_intake(args.reviews, args.out)
    print(
        json.dumps(
            {"source": str(args.out), "intake_sha256": intake["artifact_sha256"]}
        )
    )
