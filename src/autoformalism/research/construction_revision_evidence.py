"""Project sealed construction receipts into bounded, training-only feedback.

Historical campaign verification may inspect held-out receipts for provenance.
Only explicitly named training fields cross this boundary. This module performs
arithmetic on saved predictions, never integration, fitting, rank tests or calls
to a provider. All eight original parents remain in their frozen roster order.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_inputs as inputs
from autoformalism.research import construction_fit_qualification as original
from autoformalism.research import construction_verification_campaign as campaign
from autoformalism.schemas.prefix_initialization import PrefixFitProblem, PrefixPolicy
from autoformalism.schemas.public_fitting import PublicSplit
from autoformalism.schemas.staged_topology import PublicScientificBrief

PROTOCOL = "construction-revision-training-evidence-1"
_WITNESS_RULE = (
    "Three largest absolute normalized training residuals per target; ties use "
    "original training trajectory order, then sample index."
)
_SEARCH_INTERPRETATION = (
    "One bounded training search endpoint. Budget exhaustion does not "
    "establish convergence or the best attainable model fit. Full "
    "training reconstruction uses fitted training initials."
)


def _path(root: Path, name: str) -> Path:
    """Refuse path escapes and symlinks before reading an archived member."""
    path = root / name
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root)
        or path.resolve() != path.absolute()
    ):
        raise ValueError("evidence source path escapes root or uses a symlink")
    return path


def _finite(value: object, label: str) -> float:
    """Reject booleans, nonnumeric values and nonfinite recorded statistics."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid recorded {label}")
    if not math.isfinite(value):
        raise ValueError(f"nonfinite recorded {label}")
    return float(value)


def _close(recorded: object, rebuilt: float, label: str) -> None:
    """Permit summation roundoff, never changed scores or normalization."""
    if not math.isclose(
        _finite(recorded, label), rebuilt, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"recorded {label} differs from training reconstruction")


def _stats(values: np.ndarray) -> dict:
    """Compact observed or predicted channel statistics, all from training."""
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
    }


def _training_summary(training: PublicSplit) -> dict:
    """Preserve every observed initial boundary without copying full histories."""
    targets = tuple(training.rows[0].targets)
    return {
        "split": "train",
        "trajectory_count": len(training.rows),
        "sample_count": sum(len(row.time) for row in training.rows),
        "targets": {
            name: _stats(np.concatenate([row.targets[name] for row in training.rows]))
            for name in targets
        },
        "initial_observations": [
            {
                "trajectory_id": row.trajectory_id,
                "time": row.time[0],
                "targets": {name: values[0] for name, values in row.targets.items()},
                "auxiliaries": {
                    name: values[0] for name, values in row.auxiliaries.items()
                },
                "external_inputs": {
                    name: values[0] for name, values in row.external_inputs.items()
                },
                "fixed_covariates": dict(row.fixed_covariates),
            }
            for row in training.rows
        ],
    }


def _shape(value: object, fields: str, label: str) -> dict:
    """Require an exact positive field set at every projected object boundary."""
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        raise ValueError(f"unexpected or missing fields in {label}")
    return value


def _digest(value: object, label: str, *, optional: bool = False) -> None:
    """Check digest syntax without pretending to reopen its external source."""
    if optional and value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"invalid {label} digest")


def _validate_aliases(value: object, problem: PrefixFitProblem) -> dict:
    """Restrict scientific identities to public modeled targets and auxiliaries."""
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(target, str)
        for key, target in value.items()
    ):
        raise ValueError("invalid scientific aliases")
    public_names = set(problem.context.targets) | set(problem.context.auxiliaries)
    modeled_names = {
        item.name
        for item in (*problem.base_candidate.states, *problem.base_candidate.processes)
    }
    if set(value) - public_names or set(value.values()) - modeled_names:
        raise ValueError("scientific aliases are outside public modeled variables")
    return value


def _stat_shape(value: dict, label: str) -> None:
    """Bound all scalar summaries and refuse undeclared auxiliary payloads."""
    _shape(value, "minimum maximum mean standard_deviation", label)
    for name, scalar in value.items():
        _finite(scalar, f"{label}.{name}")
    if (
        value["minimum"] > value["maximum"]
        or value["standard_deviation"] < 0
        or value["mean"] < value["minimum"] - 1e-12
        or value["mean"] > value["maximum"] + 1e-12
    ):
        raise ValueError(f"inconsistent {label} statistics")


def _validate_witnesses(
    witnesses: object, target: str, scale: float, training: PublicSplit
) -> None:
    """Bind displayed observations to training and check saved scalar arithmetic."""
    if not isinstance(witnesses, list) or len(witnesses) != 3:
        raise ValueError("exactly three bounded training witnesses required")
    rows = {row.trajectory_id: (order, row) for order, row in enumerate(training.rows)}
    coordinates, order_keys = set(), []
    for witness in witnesses:
        _shape(
            witness,
            "trajectory_id sample_index time observed predicted signed_error "
            "normalized_error split",
            "training witness",
        )
        if witness["split"] != "train" or witness["trajectory_id"] not in rows:
            raise ValueError("witness is not a training trajectory")
        order, row = rows[witness["trajectory_id"]]
        index = witness["sample_index"]
        if type(index) is not int or not 0 <= index < len(row.time):
            raise ValueError("invalid training witness sample index")
        for field in (
            "time",
            "observed",
            "predicted",
            "signed_error",
            "normalized_error",
        ):
            _finite(witness[field], f"witness {field}")
        if (
            witness["time"] != row.time[index]
            or witness["observed"] != row.targets[target][index]
        ):
            raise ValueError("witness time/observation differs from training data")
        _close(
            witness["signed_error"],
            witness["predicted"] - witness["observed"],
            "witness signed error",
        )
        _close(
            witness["normalized_error"],
            witness["signed_error"] / scale,
            "witness normalized error",
        )
        coordinate = (row.trajectory_id, index)
        if coordinate in coordinates:
            raise ValueError("duplicate training witness")
        coordinates.add(coordinate)
        order_keys.append((-abs(witness["normalized_error"]), order, index))
    if order_keys != sorted(order_keys):
        raise ValueError("training witnesses violate the deterministic ordering")


def _validate_feedback(feedback: dict, training: PublicSplit, summary: dict) -> None:
    """Accept only the producer's bounded training score and search vocabulary."""
    _shape(
        feedback,
        "normalized_mse per_target witness_selection status unavailable_reason "
        "solver_difference rank_assessed search data_scope",
        "prompt_evidence",
    )
    if (
        feedback["data_scope"] != "train_only"
        or feedback["witness_selection"] != _WITNESS_RULE
        or feedback["rank_assessed"] is not False
        or feedback["status"] not in {"verified", "unavailable"}
    ):
        raise ValueError("invalid training feedback scope or availability")
    search = _shape(
        feedback["search"],
        "stop_reason budget_exhausted best_attainable_fit_assessed interpretation",
        "search evidence",
    )
    stops = {
        "budget_limited",
        "wall_clock_limit",
        "step_or_gradient_converged",
        "numerical_target_reached",
        "output_problem_solved",
        "stationary_poor_fit",
        "numerical_failure",
        "no_free_coordinates",
    }
    if (
        search["stop_reason"] not in stops
        or search["budget_exhausted"]
        is not (search["stop_reason"] in {"budget_limited", "wall_clock_limit"})
        or search["best_attainable_fit_assessed"] is not False
        or search["interpretation"] != _SEARCH_INTERPRETATION
    ):
        raise ValueError("unsupported training search claim")
    difference = feedback["solver_difference"]
    if difference is not None and _finite(difference, "solver difference") < 0:
        raise ValueError("negative solver difference")
    if feedback["status"] == "unavailable":
        if (
            feedback["normalized_mse"] is not None
            or feedback["per_target"] != {}
            or feedback["unavailable_reason"]
            not in {
                "numerical_active_wall_limit",
                "overall_wall_limit",
                "integration_timeout",
                "independent_verification_incomplete",
            }
        ):
            raise ValueError("unavailable feedback cannot contain qualified scores")
        return
    if (
        feedback["unavailable_reason"] is not None
        or difference is None
        or difference > 1e-6
    ):
        raise ValueError("verified feedback lacks independent solver agreement")
    scores = _shape(
        feedback["per_target"], " ".join(summary["targets"]), "per-target feedback"
    )
    for name, score in scores.items():
        _shape(
            score,
            "normalized_mse sample_count training_scale observed predicted "
            "mean_signed_error mean_absolute_error maximum_absolute_error witnesses",
            "target feedback",
        )
        if (
            type(score["sample_count"]) is not int
            or score["sample_count"] != summary["sample_count"]
        ):
            raise ValueError("target sample count differs from training")
        for field in (
            "normalized_mse",
            "training_scale",
            "mean_signed_error",
            "mean_absolute_error",
            "maximum_absolute_error",
        ):
            _finite(score[field], f"target {field}")
        if any(
            score[field] < 0
            for field in (
                "normalized_mse",
                "mean_absolute_error",
                "maximum_absolute_error",
            )
        ):
            raise ValueError("negative squared or absolute training error")
        scale = max(summary["targets"][name]["standard_deviation"], 1e-12)
        if score["training_scale"] <= 0:
            raise ValueError("training scale must be positive")
        _close(score["training_scale"], scale, "training scale")
        for field in ("observed", "predicted"):
            _stat_shape(score[field], field)
        for field, scalar in summary["targets"][name].items():
            _close(score["observed"][field], scalar, "observed training summary")
        _validate_witnesses(score["witnesses"], name, scale, training)
    if _finite(feedback["normalized_mse"], "aggregate training NMSE") < 0:
        raise ValueError("negative aggregate training NMSE")
    _close(
        feedback["normalized_mse"],
        float(np.mean([s["normalized_mse"] for s in scores.values()])),
        "aggregate training NMSE",
    )


def validate_training_projection(value: dict) -> None:
    """Validate exact public projection shape and internal training bindings.

    This portable check opens no files and reruns no models. It checks embedded
    model/data/parameter hashes and displayed witness arithmetic. External source
    digests and the omitted full predictions are authenticated by the path-based
    loader and the caller's sealed-bundle provenance, not reconstructed here.
    """
    _shape(
        value,
        "protocol selection_data parent_selection source_plan_sha256 "
        "source_inputs_sha256 supplemental_plan_sha256 fitting_prefix_policy "
        "construction_archive_sha256 public_task training_sha256 training "
        "training_summary parents optimizer_calls numerical_rollouts llm_calls "
        "test_data_opened mechanism_compliance_assessed",
        "training projection",
    )
    if (
        value["protocol"] != PROTOCOL
        or value["selection_data"] != "train_only"
        or value["parent_selection"] != "complete_frozen_roster_no_score_selection"
        or value["test_data_opened"] is not False
        or value["mechanism_compliance_assessed"] is not False
        or any(
            type(value[k]) is not int or value[k] != 0
            for k in ("optimizer_calls", "numerical_rollouts", "llm_calls")
        )
    ):
        raise ValueError("invalid training projection policy or scope")
    for key in (
        "source_plan_sha256",
        "source_inputs_sha256",
        "supplemental_plan_sha256",
        "training_sha256",
    ):
        _digest(value[key], key)
    _digest(value["construction_archive_sha256"], "construction archive", optional=True)
    training = PublicSplit.model_validate(value["training"])
    if (
        training.name != "train"
        or len(training.rows) != 16
        or public.content_sha256(training) != value["training_sha256"]
        or public.content_sha256(value["training"]) != value["training_sha256"]
    ):
        raise ValueError("all sixteen canonical training rows and their hash required")
    prefix = PrefixPolicy.model_validate(value["fitting_prefix_policy"])
    if public.content_sha256(prefix) != public.content_sha256(
        value["fitting_prefix_policy"]
    ):
        raise ValueError("noncanonical fitting prefix policy")
    summary = _shape(
        value["training_summary"],
        "split trajectory_count sample_count targets initial_observations",
        "training summary",
    )
    rebuilt = _training_summary(training)
    for field in ("split", "trajectory_count", "sample_count", "initial_observations"):
        if public.content_sha256(summary[field]) != public.content_sha256(
            rebuilt[field]
        ):
            raise ValueError("training summary or initial observations differ")
    _shape(summary["targets"], " ".join(rebuilt["targets"]), "training summary targets")
    for name, stats in summary["targets"].items():
        _stat_shape(stats, "training summary target")
        for field, scalar in rebuilt["targets"][name].items():
            _close(stats[field], scalar, "training summary target")
    task = _shape(value["public_task"], "prompt_sha256 prompt_text", "public task")
    _digest(task["prompt_sha256"], "public prompt", optional=True)
    if task["prompt_text"] is not None and (
        not isinstance(task["prompt_text"], str)
        or hashlib.sha256(task["prompt_text"].encode("utf-8")).hexdigest()
        != task["prompt_sha256"]
    ):
        raise ValueError("public prompt text does not match its digest")
    if not isinstance(value["parents"], dict) or len(value["parents"]) != 8:
        raise ValueError("all eight parent records are required")
    for name, parent in value["parents"].items():
        _shape(
            parent,
            "task_id source_model_id proposal_seed problem problem_sha256 "
            "source_record_sha256 source_packet_sha256 source_training_receipt_sha256 "
            "supplemental_receipt_sha256 recorded_predictions_sha256 public_brief "
            "public_brief_sha256 scientific_aliases scientific_aliases_sha256 "
            "endpoint prompt_evidence",
            "parent",
        )
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
            or parent["task_id"] != name
        ):
            raise ValueError("invalid parent task identity")
        if (
            not isinstance(parent["source_model_id"], str)
            or not parent["source_model_id"]
            or type(parent["proposal_seed"]) is not int
            or parent["proposal_seed"] < 0
        ):
            raise ValueError("invalid original parent identity")
        for key in (
            "problem_sha256",
            "source_record_sha256",
            "source_packet_sha256",
            "source_training_receipt_sha256",
            "supplemental_receipt_sha256",
            "recorded_predictions_sha256",
        ):
            _digest(parent[key], key)
        problem = PrefixFitProblem.model_validate(parent["problem"])
        if (
            public.content_sha256(problem) != parent["problem_sha256"]
            or public.content_sha256(parent["problem"]) != parent["problem_sha256"]
            or set(problem.context.targets) != set(summary["targets"])
        ):
            raise ValueError("parent model hash or target identity differs")
        if set(prefix.future_inputs) - set(problem.context.external_inputs) or set(
            prefix.future_auxiliaries
        ) - set(problem.context.auxiliaries):
            raise ValueError("prefix forcing is outside parent public context")
        brief = parent["public_brief"]
        if brief is None:
            if parent["public_brief_sha256"] is not None:
                raise ValueError("absent public brief cannot have a digest")
        else:
            parsed = PublicScientificBrief.model_validate(brief)
            _digest(parent["public_brief_sha256"], "public brief")
            if (
                public.content_sha256(parsed) != parent["public_brief_sha256"]
                or public.content_sha256(brief) != parent["public_brief_sha256"]
                or {v.name for v in parsed.public_variables if v.data_role == "target"}
                != set(summary["targets"])
            ):
                raise ValueError("public brief hash or targets differ")
        aliases = _validate_aliases(parent["scientific_aliases"], problem)
        _digest(parent["scientific_aliases_sha256"], "scientific aliases")
        if (
            public.content_sha256(aliases) != parent["scientific_aliases_sha256"]
            or (value["construction_archive_sha256"] is None and aliases)
        ):
            raise ValueError("scientific alias hash or archive provenance differs")
        endpoint = _shape(
            parent["endpoint"],
            "parameters parameters_sha256 verification_status",
            "endpoint",
        )
        if not isinstance(endpoint["parameters"], dict) or any(
            not isinstance(k, str) or not k for k in endpoint["parameters"]
        ):
            raise ValueError("invalid endpoint parameter dictionary")
        for scalar in endpoint["parameters"].values():
            _finite(scalar, "endpoint parameter")
        _digest(endpoint["parameters_sha256"], "endpoint parameters")
        if (
            public.content_sha256(endpoint["parameters"])
            != endpoint["parameters_sha256"]
        ):
            raise ValueError("endpoint parameter hash differs")
        _validate_feedback(parent["prompt_evidence"], training, rebuilt)
        if endpoint["verification_status"] != parent["prompt_evidence"]["status"]:
            raise ValueError("endpoint and feedback availability differ")


def _briefs(archive: Path | None, data: dict, training: PublicSplit) -> dict:
    """Recover public briefs and modeled identities from authenticated handoffs."""
    if archive is None:
        return {}
    if not archive.is_file() or archive.is_symlink():
        raise ValueError("regular construction archive required")
    reader = inputs._Reader(archive)
    if reader.blob_sha256 != data["source"].get("archive_sha256"):
        raise ValueError("construction archive identity differs")
    briefs = {}
    with tempfile.TemporaryDirectory(prefix="af-revision-evidence-") as temporary:
        root = Path(temporary)
        inputs._unpack(reader, root)
        plan, cases = inputs._returned_plan(root)
        if plan["artifact_sha256"] != data["source"].get("plan_sha256"):
            raise ValueError("construction plan identity differs")
        names = [entry["case_id"] for entry in plan["cases"]]
        if len(names) != len(set(names)) or set(names) != set(data["records"]):
            raise ValueError("construction and training parent rosters differ")
        for entry in plan["cases"]:
            name = entry["case_id"]
            record, value = inputs._record(root, plan, entry, cases[name])
            if any(
                record[key] != data["records"][name][key]
                for key in (
                    "problem",
                    "source_packet_sha256",
                    "source_model_id",
                    "proposal_seed",
                )
            ):
                raise ValueError("construction record differs from fitted parent")
            inputs._training_matches(value, training)
            contract = value["source"]["public_cell"]["target_contract"]
            if contract["benchmark_id"] != data["source"].get(
                "benchmark_id"
            ) or contract["public_prompt_sha256"] != data["source"].get(
                "public_files", {}
            ).get("proposer_prompt.txt"):
                raise ValueError("construction public task identity differs")
            brief = PublicScientificBrief.model_validate(value["source"]["brief"])
            targets = {
                v.name for v in brief.public_variables if v.data_role == "target"
            }
            if targets != set(training.rows[0].targets):
                raise ValueError("public brief target identities differ")
            handoff = inputs.sealed_read(root / "results" / name / "fit_handoff.json")
            if (
                handoff["artifact_sha256"] != record["source_packet_sha256"]
                or handoff["prefix_fit_problem"] != record["problem"]
            ):
                raise ValueError("scientific alias handoff identity differs")
            maps = [
                handoff[key]["state_aliases"]
                for key in ("gain_lowering", "initialization_dependency_contract")
                if "state_aliases" in handoff.get(key, {})
            ]
            if not maps or any(
                not isinstance(mapping, dict)
                or any(
                    not isinstance(key, str) or not isinstance(target, str)
                    for key, target in mapping.items()
                )
                for mapping in maps
            ):
                raise ValueError("authenticated scientific alias mapping required")
            if any(mapping != maps[0] for mapping in maps[1:]):
                raise ValueError("scientific alias handoff contracts disagree")
            problem = PrefixFitProblem.model_validate(record["problem"])
            public_names = set(problem.context.targets) | set(
                problem.context.auxiliaries
            )
            aliases = _validate_aliases(
                {key: target for key, target in maps[0].items() if key in public_names},
                problem,
            )
            briefs[name] = {
                "brief": brief.model_dump(mode="json"),
                "scientific_aliases": aliases,
            }
    return briefs


def _scores(value: dict, training: PublicSplit) -> dict:
    """Audit saved scores and derive bounded training witnesses without a rollout."""
    targets = tuple(training.rows[0].targets)
    predictions = value.get("predictions")
    row_ids = {row.trajectory_id for row in training.rows}
    if not isinstance(predictions, dict) or set(predictions) != row_ids:
        raise ValueError(
            "saved training predictions have incomplete trajectory coverage"
        )
    if set(value["training_scales"]) != set(targets):
        raise ValueError("recorded training scale channels differ")
    if set(value["per_target_normalized_mse"]) != set(targets):
        raise ValueError("recorded target score channels differ")
    per_target = {}
    row_errors = {row.trajectory_id: {} for row in training.rows}
    for name in targets:
        observed_parts, predicted_parts, coordinates = [], [], []
        for row in training.rows:
            predicted = predictions[row.trajectory_id]
            if not isinstance(predicted, dict) or set(predicted) != set(targets):
                raise ValueError("saved training prediction channels differ")
            values = predicted[name]
            if not isinstance(values, list) or len(values) != len(row.time):
                raise ValueError("saved training prediction length differs")
            array = np.asarray([_finite(v, "prediction") for v in values])
            observed_parts.append(np.asarray(row.targets[name]))
            predicted_parts.append(array)
            coordinates.extend(
                (row.trajectory_id, index, time) for index, time in enumerate(row.time)
            )
        observed, predicted = (
            np.concatenate(observed_parts),
            np.concatenate(predicted_parts),
        )
        scale = max(float(np.std(observed)), 1e-12)
        _close(value["training_scales"][name], scale, "training scale")
        error = predicted - observed
        normalized = error / scale
        squared = normalized**2
        if not np.isfinite(squared).all():
            raise ValueError("nonfinite reconstructed training error")
        nmse = float(np.mean(squared))
        _close(value["per_target_normalized_mse"][name], nmse, "target NMSE")
        offset = 0
        for row in training.rows:
            end = offset + len(row.time)
            row_errors[row.trajectory_id][name] = float(np.mean(squared[offset:end]))
            offset = end
        indices = np.argsort(-np.abs(normalized), kind="stable")[:3]
        per_target[name] = {
            "normalized_mse": nmse,
            "sample_count": len(observed),
            "training_scale": scale,
            "observed": _stats(observed),
            "predicted": _stats(predicted),
            "mean_signed_error": float(np.mean(error)),
            "mean_absolute_error": float(np.mean(np.abs(error))),
            "maximum_absolute_error": float(np.max(np.abs(error))),
            "witnesses": [
                {
                    "trajectory_id": coordinates[index][0],
                    "sample_index": coordinates[index][1],
                    "time": coordinates[index][2],
                    "observed": float(observed[index]),
                    "predicted": float(predicted[index]),
                    "signed_error": float(error[index]),
                    "normalized_error": float(normalized[index]),
                    "split": "train",
                }
                for index in indices
            ],
        }
    nmse = float(np.mean([v["normalized_mse"] for v in per_target.values()]))
    _close(value["normalized_mse"], nmse, "aggregate NMSE")
    if set(value.get("per_row_scores", {})) != row_ids:
        raise ValueError("recorded row scores have incomplete coverage")
    for row in training.rows:
        recorded = value["per_row_scores"][row.trajectory_id]
        rebuilt = row_errors[row.trajectory_id]
        if recorded["sample_count"] != len(row.time) or set(
            recorded["per_target_normalized_mse"]
        ) != set(targets):
            raise ValueError("recorded row score sample count or channels differ")
        _close(
            recorded["normalized_mse"],
            float(np.mean(list(rebuilt.values()))),
            "row NMSE",
        )
        for name, score in rebuilt.items():
            _close(
                recorded["per_target_normalized_mse"][name], score, "row target NMSE"
            )
    return {
        "normalized_mse": nmse,
        "per_target": per_target,
        "witness_selection": _WITNESS_RULE,
    }


def _verification(
    root: Path, plan: dict, candidate: dict, training: PublicSplit
) -> tuple[dict, dict, str]:
    """Read one complete bound receipt, refusing unsupported success claims."""
    task = candidate["task"]
    folder = _path(root, f"results/{task['task_id']}/operations/output_verification")
    for name in ("started.json", "result.json", "progress.json"):
        _path(folder, name)
    identity = {
        "campaign": campaign._identity(plan, candidate),
        "stage": "output_verification",
        "allowance_seconds": plan["policy"]["execution"]["overall_seconds"],
    }
    operation = original._read_operation(folder, identity)
    value = operation.get("value")
    if not isinstance(value, dict):
        raise ValueError("terminal supplemental numerical receipt required")
    expected = {
        "protocol": "construction-saved-parameter-verification-1",
        "problem_sha256": task["problem_sha256"],
        "training_sha256": public.content_sha256(training),
        "parameters_sha256": candidate["parameters_sha256"],
        "policy": plan["numerical_policy"],
        "execution_policy": plan["policy"]["execution"],
        "selection_data": "train_only",
        "sensitivities": False,
        "no_search_performed": True,
        "optimizer_calls": 0,
        "live_llm_calls": 0,
        "test_data_opened": False,
        "private_data_opened": False,
    }
    if any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ValueError("supplemental training receipt identity or boundary differs")
    if value.get("success") is not value.get("fit_success") or not isinstance(
        value.get("success"), bool
    ):
        raise ValueError("inconsistent supplemental verification success")
    if value["success"]:
        if (
            operation["status"] != "complete"
            or value.get("parameters") != candidate["parameters"]
            or value.get("verification_calls_completed") != 2
            or value.get("rollout_calls_completed") != 2 * len(training.rows)
            or value.get("verification_calls_failed") != 0
            or value.get("rollout_calls_failed") != 0
            or not 0
            <= _finite(value.get("solver_difference"), "solver difference")
            <= plan["numerical_policy"]["solver_tolerance"]
        ):
            raise ValueError("unsupported supplemental verification success")
        evidence = _scores(value, training)
        status = "verified"
    else:
        if (
            value.get("parameters") is not None
            or value.get("normalized_mse") is not None
            or value.get("predictions")
            or value.get("per_target_normalized_mse")
        ):
            raise ValueError("unavailable verification cannot supply qualified scores")
        evidence = {
            "normalized_mse": None,
            "per_target": {},
            "witness_selection": _WITNESS_RULE,
        }
        status = "unavailable"
    evidence.update(
        status=status,
        unavailable_reason=(value.get("timing") or {}).get("timeout_reason")
        or ("independent_verification_incomplete" if status == "unavailable" else None),
        solver_difference=value.get("solver_difference"),
        rank_assessed=False,
    )
    return evidence, value, public.content_sha256(read_seal(folder / "result.json"))


def load_training_evidence(
    source: Path,
    supplemental: Path,
    *,
    public_prompt: Path | None = None,
    construction_archive: Path | None = None,
) -> dict:
    """Load all eight authenticated parents and explicitly project training data.

    The common ``training`` split is retained for independent audits. Prompts
    should use ``training_summary`` and each parent's ``prompt_evidence`` instead
    of full arrays. Optional public prompt and construction archive must match
    their exact historical source hashes; no unverified dictionary is accepted.
    """
    if source.is_symlink() or supplemental.is_symlink():
        raise ValueError("evidence roots cannot be symlinks")
    source, supplemental = source.resolve(), supplemental.resolve()
    plan, snapshot_data = campaign.verify(supplemental, runtime=False)
    source_plan, data, candidates = campaign._source(source)
    if (
        public.content_sha256(source_plan) != plan["source_plan_sha256"]
        or public.content_sha256(data) != public.content_sha256(snapshot_data)
        or candidates != plan["candidates"]
    ):
        raise ValueError("source and supplemental lineage differ")
    for name, digest in plan["source_files"].items():
        if hashlib.sha256(_path(source, name).read_bytes()).hexdigest() != digest:
            raise ValueError("original source bytes differ from preserved snapshot")
    if source_plan["policy"]["population"] != "returned-eight" or len(candidates) != 8:
        raise ValueError("the complete returned eight-parent population is required")
    training = PublicSplit.model_validate(data["training"])
    if training.name != "train" or len(training.rows) != 16:
        raise ValueError("all sixteen training trajectories are required")
    briefs = _briefs(construction_archive, data, training)
    prompt_hash = data["source"].get("public_files", {}).get("proposer_prompt.txt")
    prompt = None
    if public_prompt is not None:
        blob = public_prompt.read_bytes()
        if not prompt_hash or hashlib.sha256(blob).hexdigest() != prompt_hash:
            raise ValueError("public prompt identity differs")
        prompt = blob.decode("utf-8")
    parents = {}
    for candidate in candidates:
        task = candidate["task"]
        name = task["task_id"]
        record = data["records"][name]
        problem = PrefixFitProblem.model_validate(record["problem"])
        if public.content_sha256(problem) != task["problem_sha256"]:
            raise ValueError("parent problem identity differs")
        fit_path = _path(source, f"results/{name}/operations/training/result.json")
        fit_operation = read_seal(fit_path)
        fit = fit_operation["value"]
        evidence, verified, receipt_hash = _verification(
            supplemental, plan, candidate, training
        )
        stop = fit["search"]["stop_reason"]
        evidence["search"] = {
            "stop_reason": stop,
            "budget_exhausted": stop in {"budget_limited", "wall_clock_limit"},
            "best_attainable_fit_assessed": False,
            "interpretation": _SEARCH_INTERPRETATION,
        }
        evidence["data_scope"] = "train_only"
        public_source = briefs.get(name, {})
        brief = public_source.get("brief")
        aliases = public_source.get("scientific_aliases", {})
        parents[name] = {
            "task_id": name,
            "source_model_id": record["source_model_id"],
            "proposal_seed": record["proposal_seed"],
            "problem": problem.model_dump(mode="json"),
            "problem_sha256": task["problem_sha256"],
            "source_record_sha256": public.content_sha256(record),
            "source_packet_sha256": record["source_packet_sha256"],
            "source_training_receipt_sha256": public.content_sha256(fit_operation),
            "supplemental_receipt_sha256": receipt_hash,
            "recorded_predictions_sha256": public.content_sha256(
                verified["predictions"]
            ),
            "public_brief": brief,
            "public_brief_sha256": public.content_sha256(brief) if brief else None,
            "scientific_aliases": aliases,
            "scientific_aliases_sha256": public.content_sha256(aliases),
            "endpoint": {
                "parameters": deepcopy(candidate["parameters"]),
                "parameters_sha256": candidate["parameters_sha256"],
                "verification_status": evidence["status"],
            },
            "prompt_evidence": evidence,
        }
    return {
        "protocol": PROTOCOL,
        "selection_data": "train_only",
        "parent_selection": "complete_frozen_roster_no_score_selection",
        "source_plan_sha256": public.content_sha256(source_plan),
        "source_inputs_sha256": public.content_sha256(data),
        "supplemental_plan_sha256": public.content_sha256(plan),
        "fitting_prefix_policy": deepcopy(source_plan["policy"]["prefix"]),
        "construction_archive_sha256": data["source"].get("archive_sha256")
        if construction_archive is not None
        else None,
        "public_task": {"prompt_sha256": prompt_hash, "prompt_text": prompt},
        "training_sha256": public.content_sha256(training),
        "training": training.model_dump(mode="json"),
        "training_summary": _training_summary(training),
        "parents": parents,
        "optimizer_calls": 0,
        "numerical_rollouts": 0,
        "llm_calls": 0,
        "test_data_opened": False,
        "mechanism_compliance_assessed": False,
    }
