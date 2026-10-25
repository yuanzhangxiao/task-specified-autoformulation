"""Training-only requalification of two saved endpoints and one missing start.

Historical receipts remain immutable. A fresh, explicit budget permits polishing
only after endpoint diagnosis; missing-task recovery does not renew an old lease.
"""

from __future__ import annotations

import hashlib
import sys
from copy import deepcopy
from pathlib import Path
from time import monotonic
from typing import Literal

import casadi as ca
import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_continuation as numerical
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_multioutput_profile as profiling
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_timescale_controls as source
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

pilot = source.pilot
PROTOCOL = "construction-endpoint-requalification-1"
INPUT_PROTOCOL = "construction-endpoint-requalification-inputs-1"
SOURCE_INDICES = (7, 8, 9)


class Policy(StrictSchema):
    """Finite allowances for diagnosis, conditional polishing and certification."""

    search_seconds: pilot.Nonnegative = Field(default=600, gt=0, le=600)
    diagnostic_seconds: pilot.Nonnegative = Field(default=600, gt=0, le=600)
    verification_seconds: pilot.Nonnegative = Field(default=180, gt=0, le=180)
    maximum_calls: int = Field(default=1000, gt=0, le=1000)


class Inputs(StrictSchema):
    """Bind every preceding outcome, including the absent numerical task."""

    protocol: Literal["construction-endpoint-requalification-inputs-1"]
    source_plan: dict
    source_plan_sha256: pilot.Digest
    source_inputs: dict
    source_inputs_sha256: pilot.Digest
    source_report: dict
    source_report_sha256: pilot.Digest
    source_records: list[dict] = Field(min_length=10, max_length=10)
    source_log_hashes: dict[str, pilot.Digest]
    old_allowances_renewed: Literal[False]
    validation_exported: Literal[False]
    test_data_opened: Literal[False]


def _source_operations(plan, record):
    """Restore source journal order after canonical JSON sorts object keys.

    Historical cost sums used declared stage order. Replaying the same receipts
    in JSON lexical order can change their floating sums and terminal digest.
    """
    order = source._allowances(source.Policy.model_validate(plan["policy"]))
    operations = record["operations"]
    if set(operations) - set(order):
        raise ValueError("unknown source operation")
    return {name: operations[name] for name in order if name in operations}


def _source_report(plan, data, records):
    """Rebuild v2's complete report without writing into the source directory."""
    p, rows = source.Policy.model_validate(plan["policy"]), []
    for task, record in zip(plan["tasks"], records, strict=True):
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        value = source._assemble(
            identity,
            data,
            _source_operations(plan, record),
            terminal=record["terminal_present"],
        )
        if record["terminal_present"] and record[
            "terminal_sha256"
        ] != public.content_sha256(value):
            raise ValueError("source terminal differs from authenticated receipts")
        if not record["terminal_present"] and record["terminal_sha256"] is not None:
            raise ValueError("missing source terminal cannot have a terminal digest")
        rows.append(
            task
            | {
                k: v
                for k, v in value.items()
                if k
                not in {
                    "identity",
                    "operations",
                    "source_incumbent",
                    "retained",
                    "selected_start",
                }
            }
            | {
                "training_nmse": (value.get("retained") or {}).get("training_nmse"),
                "source_training_nmse": data["endpoint"]["training_nmse"],
                "currently_verified": value.get("retained") is not None,
            }
        )
    count = sum(row["status"] == "complete" for row in rows)
    return {
        "protocol": source.RECOVERY_PROTOCOL,
        "status": "complete" if count == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": count,
        "rows": rows,
        "time_grid": plan["time_grid"],
        "source_kappa": -data["endpoint"]["effective_parameters"][source.RATE],
        "historical_costs": {
            "profile_pilot_by_task": data["parent_tasks"],
            "prior_campaign_inputs": data["historical_inputs"],
        },
        "matched_optimization_seconds": p.search_seconds,
        "fixed_inner_seconds": source._projection_seconds(p),
        "outer_qualification_and_search_seconds": p.search_seconds
        - source._projection_seconds(p),
        "prior_timescale_costs": [
            {
                "task_index": row["task_index"],
                "cost": pilot.journal._cost(row["operations"]),
            }
            for row in data["prior_controls"]["records"]
        ],
        "old_allowances_renewed": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
        "limitation": (
            "Finite negative-decay slices are conditional fits, "
            "not global lower bounds. "
            "Distinct starts preserve the signed family. Full-coordinate adequacy "
            "needs its own endpoint evidence; no rate, model, or scientific "
            "confidence is automatically promoted."
        ),
    }


def _validate(data):
    parsed = Inputs.model_validate(data)
    plan, old = parsed.source_plan, parsed.source_inputs
    source._validate(old)
    p = source.Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != source.RECOVERY_PROTOCOL
        or p.recovery_mode != "project_first_local"
        or public.content_sha256(plan) != parsed.source_plan_sha256
        or public.content_sha256(old) != parsed.source_inputs_sha256
        or plan["inputs_sha256"] != parsed.source_inputs_sha256
        or plan["tasks"] != source._tasks(old, p)
        or plan["numerical_contract"] != source._contract(p)
        or plan["time_grid"]
        != source.time_grid(source._training_inputs(old)["training"])
        or plan["llm_calls"] != 0
        or any(
            plan[k] is not False
            for k in (
                "old_allowances_renewed",
                "validation_data_opened",
                "test_data_opened",
                "automatic_revision",
                "automatic_promotion",
            )
        )
    ):
        raise ValueError("source campaign identity differs")
    allowed = source._allowances(p)
    for task, record in zip(plan["tasks"], parsed.source_records, strict=True):
        if (
            set(record)
            != {"task_index", "terminal_present", "terminal_sha256", "operations"}
            or record["task_index"] != task["index"]
        ):
            raise ValueError("source roster differs")
        if type(record["terminal_present"]) is not bool or set(
            record["operations"]
        ) - set(allowed):
            raise ValueError("source operation roster differs")
        for name, operation in record["operations"].items():
            if operation.get("identity") != {
                "campaign": {"plan_sha256": parsed.source_plan_sha256, "task": task},
                "stage": name,
                "allowance_seconds": allowed[name],
            } or operation.get("progress_sha256") != public.content_sha256(
                operation.get("observed_progress", {})
            ):
                raise ValueError("source operation binding differs")
    rebuilt = _source_report(plan, old, parsed.source_records)
    if (
        rebuilt != parsed.source_report
        or public.content_sha256(rebuilt) != parsed.source_report_sha256
    ):
        raise ValueError("source report differs from operation receipts")
    for i in (7, 8):
        terminal = _parent(data, i)
        if not parsed.source_records[i]["terminal_present"] or not terminal.get(
            "retained", {}
        ).get("original_equations_verified"):
            raise ValueError("both source endpoints require independent verification")
    missing = parsed.source_records[9]
    if missing["terminal_present"] or missing["operations"]:
        raise ValueError("missing-start recovery requires no old numerical reservation")
    if any(
        Path(name).name != name or not name.endswith((".err", ".out"))
        for name in parsed.source_log_hashes
    ):
        raise ValueError("invalid source log name")


def _parent(data, index):
    task = data["source_plan"]["tasks"][index]
    record = data["source_records"][index]
    return source._assemble(
        {"plan_sha256": data["source_plan_sha256"], "task": task},
        data["source_inputs"],
        _source_operations(data["source_plan"], record),
        terminal=record["terminal_present"],
    )


def export_inputs(root: Path, destination: Path) -> dict:
    """Authenticate all ten source outcomes without rerunning or modifying them."""
    root = root.resolve()
    if destination.resolve().is_relative_to(root):
        raise ValueError("export must not write into source campaign")
    plan, old = source.verify(root, runtime=False)
    p = source.Policy.model_validate(plan["policy"])
    records = []
    for task in plan["tasks"]:
        folder = source._folder(root, task)
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        ops = source._operations(folder, identity, p)
        if any(op.get("status") == "running_or_interrupted" for op in ops.values()):
            raise ValueError(
                "unfinished source reservation must be reconciled before export"
            )
        terminal = (
            read_seal(folder / "result.json")
            if (folder / "result.json").exists()
            else None
        )
        records.append(
            {
                "task_index": task["index"],
                "terminal_present": terminal is not None,
                "terminal_sha256": public.content_sha256(terminal)
                if terminal is not None
                else None,
                "operations": ops,
            }
        )
    report = public._read(root / "report/summary.json")
    data = {
        "protocol": INPUT_PROTOCOL,
        "source_plan": plan,
        "source_plan_sha256": public.content_sha256(plan),
        "source_inputs": old,
        "source_inputs_sha256": public.content_sha256(old),
        "source_records": records,
        "source_log_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "logs").glob("*"))
            if path.is_file()
            and not path.is_symlink()
            and path.name.endswith((".err", ".out"))
        },
        "source_report": report,
        "source_report_sha256": public.content_sha256(report),
        "old_allowances_renewed": False,
        "validation_exported": False,
        "test_data_opened": False,
    }
    _validate(data)
    seal(destination, data)
    return {
        "inputs_sha256": public.content_sha256(data),
        "authenticated_source_tasks": 10,
        "source_report_reproduced": True,
        "missing_source_tasks": [9],
    }


def scale_floors(data):
    """Training-only dimensional floors for the explicitly certified model-six RHS."""
    old = data["source_inputs"]
    training = source._training_inputs(old)["training"]
    grid = source.time_grid(training)
    span = grid["span"]

    def rms(field, name):
        values = np.concatenate(
            [np.asarray(row[field][name], float) for row in training["rows"]]
        )
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("finite training channel required for chart scale")
        return max(float(np.sqrt(np.mean(values * values))), 1e-12)

    scales = {name: rms("targets", name) for name in ("Gp", "I", "U")}
    scales |= {
        name: rms("external_inputs", name)
        for name in ("meal_rate_g_per_min", "insulin_pmol_per_kg_min")
    }
    scales |= {name: rms("auxiliaries", name) for name in ("EGP", "Uii")}
    scales["I_mem"] = scales["U"] / span
    floor = {
        source.RATE: 1 / span,
        source.OUTER_ROLES["feedback"]: scales["I"] / (span * scales["I_mem"]),
        source.OUTER_ROLES["activation"]: scales["I_mem"] / (span * scales["I"]),
        # The declared reciprocal multiplies U in Gp', so its scale follows
        # those observed channels, not the parameter's historical `tau` name.
        source.OUTER_ROLES["glucose_delay"]: span * scales["U"] / scales["Gp"],
        source.OUTER_ROLES["utilization_decay"]: 1 / span,
        source.OUTER_ROLES["memory_decay"]: 1 / span,
        "coefficient:I:insulin_pmol_per_kg_min": scales["I"]
        / (span * scales["insulin_pmol_per_kg_min"]),
        "coefficient:I:meal_rate_g_per_min": scales["I"]
        / (span * scales["meal_rate_g_per_min"]),
        "coefficient:U:Uii": scales["U"] / (span * scales["Uii"]),
        "k_EGP_term_11_1_db1d47d8a7": scales["Gp"] / (span * scales["EGP"]),
        "k_prod_term_11_0_e3a835f757": scales["Gp"]
        / (span * scales["meal_rate_g_per_min"]),
    }
    floor |= {
        f"trajectory:{row['trajectory_id']}:init_I_mem_value": scales["I_mem"]
        for row in training["rows"]
    }
    if set(floor) != set(old["profile_outer_names"] + old["profile_inner_names"]):
        raise ValueError("model-six coordinate floor roster differs")
    return {
        "protocol": "model-six-training-dimensional-floors-1",
        "floors": floor,
        "training_channel_rms": scales,
        "time_grid": grid,
        "rms_floor": 1e-12,
        "hidden_scale_basis": "certified fixed unit I_mem contribution to U derivative",
        "not_physical_parameter_bounds": True,
        "reference_coefficients_used": False,
    }


def _certify_scale_structure(original, oracle):
    """Refuse model-specific dimensions unless the aggregate RHS matches exactly."""
    states = original.system.model.state_names
    inputs = original.system.inputs
    if set(states) != {"Gp", "I", "U", "I_mem"}:
        raise ValueError("dimensional chart requires the model-six state roster")
    p = ca.SX.sym("chart_parameters", len(oracle.names))
    x = {name: ca.SX.sym("chart_state_" + name) for name in states}
    u = {name: ca.SX.sym("chart_input_" + name) for name in inputs}
    names = dict(zip(oracle.names, ca.vertsplit(p), strict=True))
    rhs = oracle.reduction.symbolic_rhs(p, x, u)
    expected = {
        "Gp": names["k_prod_term_11_0_e3a835f757"] * u["meal_rate_g_per_min"]
        + names["k_EGP_term_11_1_db1d47d8a7"] * u["EGP"]
        - x["U"] / names[source.OUTER_ROLES["glucose_delay"]],
        "I": names[source.RATE] * x["I"]
        + names[source.OUTER_ROLES["feedback"]] * x["I_mem"]
        + names["coefficient:I:insulin_pmol_per_kg_min"] * u["insulin_pmol_per_kg_min"]
        + names["coefficient:I:meal_rate_g_per_min"] * u["meal_rate_g_per_min"],
        "U": names["coefficient:U:Uii"] * u["Uii"]
        + x["I_mem"]
        - names[source.OUTER_ROLES["utilization_decay"]] * x["U"],
        "I_mem": names[source.OUTER_ROLES["activation"]] * x["I"]
        - names[source.OUTER_ROLES["memory_decay"]] * x["I_mem"],
    }
    # Different associative ordering of additive terms need not simplify as a
    # whole in CasADi. Prove each state/forcing coefficient and constant exactly;
    # no sampled numerical equality is used as the certificate.
    variables = ca.vertcat(*x.values(), *u.values())
    difference = rhs - ca.vertcat(*(expected[name] for name in states))
    coefficients = ca.jacobian(difference, variables)
    constants = ca.substitute(difference, variables, ca.SX.zeros(variables.shape))
    if any(
        not ca.simplify(v).is_zero() for v in ca.vertsplit(ca.vec(coefficients))
    ) or any(not ca.simplify(v).is_zero() for v in ca.vertsplit(constants)):
        raise ValueError("model-six dimensional chart RHS certificate failed")


def _tasks(data):
    return [
        {
            "index": i,
            "model": 6,
            "task_id": data["source_plan"]["tasks"][source_i]["task_id"],
            "source_task_index": source_i,
            "kind": "endpoint_requalification"
            if source_i in (7, 8)
            else "missing_start_recovery",
            "label": data["source_plan"]["tasks"][source_i]["label"],
        }
        for i, source_i in enumerate(SOURCE_INDICES)
    ]


def _contract():
    return {
        "coordinate_policy": "frozen max(abs(anchor), training dimensional floor)",
        "full_coordinate_raw_evidence": True,
        "history": "exact retained-best prefix",
        "polishing_rule": "only if current endpoint local adequacy unresolved",
        "search_eligibility": (
            "finite two-solver relative agreement and qualified local derivatives"
        ),
        "retention": "unchanged strict independent original-equation verification",
        "verification_tolerances": [[1e-11, 1e-13]],
        "reference_coefficients_used": False,
        "global_minimum_certified": False,
    }


def prepare(root: Path, data: dict, policy: Policy) -> dict:
    """Freeze only three new tasks with separate, nonrenewable allowances."""
    _validate(data)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "policy": policy.model_dump(mode="json"),
        "tasks": _tasks(data),
        "scale_policy": scale_floors(data),
        "numerical_contract": _contract(),
        "source_sha256": public._source_identity(),
        "runtime": pilot._runtime(),
        "old_allowances_renewed": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": 3}


def verify(root: Path, *, runtime=True):
    """Verify immutable identities before reading checkpoints or executing numerics."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    _validate(data)
    Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != _tasks(data)
        or plan["scale_policy"] != scale_floors(data)
        or plan["numerical_contract"] != _contract()
        or plan["llm_calls"] != 0
        or any(
            plan[k] is not False
            for k in (
                "old_allowances_renewed",
                "validation_data_opened",
                "test_data_opened",
                "automatic_revision",
                "automatic_promotion",
            )
        )
    ):
        raise ValueError("endpoint requalification plan identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != pilot._runtime()
    ):
        raise ValueError("endpoint requalification source/runtime differs")
    return plan, data


def _projection_seconds(p):
    return min(60.0, p.search_seconds / 4)


def _allowances(p, task):
    result = {
        "incumbent_verification": p.verification_seconds + 90,
        "continuation_search": p.search_seconds + 150,
        "continuation_verification": p.verification_seconds + 90,
        "retained_diagnosis": p.diagnostic_seconds + 90,
    }
    if task["kind"] == "endpoint_requalification":
        result["initial_diagnosis"] = p.diagnostic_seconds + 90
    else:
        result["fixed_inner"] = _projection_seconds(p) + 90
        result["continuation_search"] -= _projection_seconds(p)
    return result


_folder = pilot._folder


def _operations(folder, identity, policy):
    allowed = _allowances(policy, identity["task"])
    if any(
        p.is_dir() and p.name not in allowed for p in (folder / "operations").glob("*")
    ):
        raise ValueError("unknown endpoint requalification operation")
    result = {}
    for name, seconds in allowed.items():
        path = folder / "operations" / name
        bound = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
        if (path / "result.json").exists():
            result[name] = pilot.journal._read_operation(path, bound)
        elif (path / "started.json").exists():
            if read_seal(path / "started.json") != bound:
                raise ValueError("operation reservation differs")
            result[name] = {
                "status": "running_or_interrupted",
                "value": None,
                "wall_seconds": None,
                "cpu_seconds": None,
                "budget_charge_seconds": seconds,
                "accounting_complete": False,
            }
    return result


def _value(operation):
    return operation.get("value") or operation.get("observed_progress") or {}


def search_eligibility(operation):
    """Separate a usable numerical search start from strict retained certification."""
    value = _value(operation)
    verified = bool(pilot._verified(operation))
    score, alternate = value.get("training_nmse"), value.get("alternate_training_nmse")
    gap = value.get("solver_prediction_discrepancy")
    scalars = (score, alternate, gap)
    finite = all(
        isinstance(v, (float, int)) and np.isfinite(v) and v >= 0 for v in scalars
    )
    relative_gap = gap / max(1, np.sqrt(score), np.sqrt(alternate)) if finite else None
    relative_loss = (
        abs(score - alternate) / max(1, score, alternate) if finite else None
    )
    complete_rollouts = (
        value.get("methods") == ["certified_affine", "original_Radau"]
        and value.get("verification_calls_completed", 0) >= 2
        and value.get("verification_calls_started")
        == value.get("verification_calls_completed")
        and value.get("verification_calls_failed", 0) == 0
        and value.get("status", "complete") == "complete"
    )
    eligible = verified or bool(
        operation.get("status") == "complete"
        and complete_rollouts
        and finite
        and relative_gap <= 1e-7
        and relative_loss <= 1e-7
    )
    return {
        "search_eligible": eligible,
        "strict_endpoint_verified": verified,
        "finite_independent_rollouts": finite,
        "complete_independent_rollouts": complete_rollouts,
        "relative_prediction_gap": relative_gap,
        "relative_loss_gap": relative_loss,
        "relative_tolerance": 1e-7,
        "not_retention_certification": True,
        "local_derivative_qualification_still_required": True,
    }


def _assemble(identity, data, operations, *, terminal=False):
    imported = _parent(data, identity["task"]["source_task_index"])
    historical = imported.get("retained")
    selected = pilot._verified(operations.get("incumbent_verification", {}))
    retained, origin = selected, "incumbent_verification" if selected else None
    fresh = pilot._verified(operations.get("continuation_verification", {}))
    if fresh and (
        retained is None or pilot.recovery_numerics.improves(fresh, retained)
    ):
        retained, origin = fresh, "continuation_verification"
    final_op = operations.get("retained_diagnosis", {})
    initial_op = operations.get("initial_diagnosis", {})
    diagnosis = _value(final_op) or (
        _value(initial_op) if origin == "incumbent_verification" else {}
    )
    if diagnosis and (
        retained is None
        or diagnosis.get("diagnosed_parameters_sha256")
        != public.content_sha256(retained["parameters"])
        or diagnosis.get("verification_sha256") != public.content_sha256(retained)
    ):
        raise ValueError("diagnosis differs from exact retained endpoint")
    fit = pilot.previous.first._search_checkpoint(
        operations.get("continuation_search", {})
    )
    diagnosis_op = final_op if _value(final_op) else initial_op
    assessment = (
        diagnosis.get("assessment")
        if diagnosis_op.get("status") == "complete"
        and diagnosis.get("status", "complete") == "complete"
        else None
    )
    return {
        "identity": identity,
        "status": "complete" if terminal else "incomplete",
        "operations": operations,
        "cost": pilot.journal._cost(operations),
        "historical_endpoint": historical,
        "selected_start": selected,
        "retained": retained,
        "retained_origin": origin,
        "search_eligibility": search_eligibility(
            operations.get("incumbent_verification", {})
        ),
        "search_started": bool(fit.get("search_calls_started")),
        "search_stop_reason": fit.get("stop_reason", "not_started"),
        "fitting_diagnosis": assessment,
        "optimization_status": (assessment or {}).get(
            "classification", "optimization_unresolved"
        ),
        "old_allowances_renewed": False,
        "test_data_opened": False,
        "validation_data_opened": False,
        "llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
    }


def run_task(root: Path, index: int):
    """Requalify, conditionally polish, and certify; resume never resets an attempt."""
    print(
        f"endpoint requalification: verify task={index} root={root}",
        file=sys.stderr,
        flush=True,
    )
    plan, data = verify(root)
    if not 0 <= index < 3:
        raise ValueError("unknown endpoint requalification task")
    task, p = plan["tasks"][index], Policy.model_validate(plan["policy"])
    old, parent = data["source_inputs"], _parent(data, task["source_task_index"])
    train = source._training_inputs(old)
    problem = PrefixFitProblem.model_validate(train["models"][6]["problem"])
    training = PublicSplit.model_validate(train["training"])
    settings = kernel.NumericalPolicy.model_validate(train["numerical_policy"])
    anchor = old["endpoint"]["parameters"]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder, allowed = _folder(root, task), _allowances(p, task)
    endpoint = parent.get("retained")
    start = (
        endpoint["effective_parameters"]
        if endpoint
        else old["endpoint"]["effective_parameters"]
        | data["source_plan"]["tasks"][9]["start_overrides"]
    )
    history = pilot.previous.first._search_checkpoint(
        parent["operations"].get("continuation_search", {})
    )

    def assemble(terminal=False):
        return _assemble(
            identity, data, _operations(folder, identity, p), terminal=terminal
        )

    def operate(name, work, *, chart_anchor=None):
        print(
            f"endpoint requalification: stage={name} reservation",
            file=sys.stderr,
            flush=True,
        )

        def call(save):
            deadline = monotonic() + allowed[name]
            print(
                f"endpoint requalification: stage={name} oracle setup",
                file=sys.stderr,
                flush=True,
            )
            original, base, _ = profiling.make_oracle(
                problem, training, settings, anchor
            )
            _certify_scale_structure(original, base)
            base.configure_local_probes(source._local_probe_floors(old))
            oracle = profiling.EndpointChartOracle(
                base, chart_anchor or start, plan["scale_policy"]["floors"]
            )
            profile = profiling.MultiOutputProfile(oracle)
            print(
                f"endpoint requalification: stage={name} numerical work",
                file=sys.stderr,
                flush=True,
            )
            if [oracle.names[i] for i in profile.outer] != old[
                "profile_outer_names"
            ] or [oracle.names[i] for i in profile.inner] != old["profile_inner_names"]:
                raise ValueError("profile partition differs")

            def checked_save(value):
                save(
                    value
                    | {
                        "affine_execution": deepcopy(original.affine_audit),
                        "recovery_audit": deepcopy(oracle.recovery_audit),
                    }
                )
                if monotonic() >= deadline:
                    raise TimeoutError(
                        "inclusive endpoint operation allowance exhausted"
                    )

            if monotonic() >= deadline:
                raise TimeoutError("endpoint oracle setup allowance exhausted")
            result = work(original, oracle, profile, checked_save)
            checked_save(result)
            return result | {
                "affine_execution": deepcopy(original.affine_audit),
                "recovery_audit": deepcopy(oracle.recovery_audit),
            }

        return pilot.journal._operation(folder, identity, name, allowed[name], call)

    def diagnose(name, retained, context, chart_anchor):
        return operate(
            name,
            lambda o, a, f, save: pilot.diagnose_profile(
                o,
                a,
                pilot.previous._verified_coordinates(o, a, retained),
                retained,
                context,
                p.diagnostic_seconds,
                save,
                include_raw_evidence=True,
            ),
            chart_anchor=chart_anchor,
        )

    print(f"endpoint requalification: task lock={folder}", file=sys.stderr, flush=True)
    with public._lock(folder):
        if (folder / "result.json").exists():
            result = assemble(True)
            if result != read_seal(folder / "result.json"):
                raise ValueError("terminal differs from authenticated operations")
            return result
        if task["kind"] == "missing_start_recovery":
            projection = operate(
                "fixed_inner",
                lambda o, a, f, save: pilot._fixed_inner(
                    a, f, start, _projection_seconds(p), save
                ),
            )
            checkpoint = pilot.previous.first._search_checkpoint(projection)
            search_start = (checkpoint.get("best") or {}).get("parameters")
        else:
            search_start = start
        if search_start:
            operate(
                "incumbent_verification",
                lambda o, a, f, save: numerical.verify_endpoint_ladder(
                    o,
                    a,
                    search_start,
                    p.verification_seconds,
                    save,
                    tolerances=((1e-11, 1e-13),),
                ),
                chart_anchor=search_start,
            )
        current = assemble()
        selected = current["selected_start"]
        initial_diag = {}
        if selected and task["kind"] == "endpoint_requalification":
            initial_diag = diagnose("initial_diagnosis", selected, history, start)
        assessment = (
            _value(initial_diag).get("assessment") or {}
            if initial_diag.get("status") == "complete"
            and _value(initial_diag).get("status", "complete") == "complete"
            else {}
        )
        polishing_needed = (
            assessment.get("classification", "optimization_unresolved")
            == "optimization_unresolved"
        )
        if (
            search_start
            and current["search_eligibility"]["search_eligible"]
            and polishing_needed
        ):
            attempt = operate(
                "continuation_search",
                lambda o, a, f, save: pilot._profile_search(
                    a,
                    f,
                    search_start,
                    p.search_seconds
                    - (
                        _projection_seconds(p)
                        if task["kind"] == "missing_start_recovery"
                        else 0
                    ),
                    p.maximum_calls,
                    save,
                    preserve_full_history=True,
                ),
                chart_anchor=search_start,
            )
            fit = pilot.previous.first._search_checkpoint(attempt)
            if fit.get("best"):
                operate(
                    "continuation_verification",
                    lambda o, a, f, save: numerical.verify_endpoint_ladder(
                        o,
                        a,
                        fit["best"]["parameters"],
                        p.verification_seconds,
                        save,
                        tolerances=((1e-11, 1e-13),),
                    ),
                    chart_anchor=search_start,
                )
        current = assemble()
        retained = current["retained"]
        if retained and (
            task["kind"] == "missing_start_recovery"
            or "continuation_search" in current["operations"]
        ):
            fit = pilot.previous.first._search_checkpoint(
                current["operations"].get("continuation_search", {})
            )
            if (fit.get("best") or {}).get("parameters") != retained.get(
                "effective_parameters"
            ):
                fit = (
                    history
                    if current["retained_origin"] == "incumbent_verification"
                    else {}
                )
            diagnose("retained_diagnosis", retained, fit, search_start)
        result = assemble(True)
        seal(folder / "result.json", result)
        return result


def report(root: Path):
    """Report numerical evidence separately from optimizer and scientific claims."""
    plan, data = verify(root, runtime=False)
    p, rows = Policy.model_validate(plan["policy"]), []
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = _folder(root, task)
        done = (folder / "result.json").exists()
        value = _assemble(
            identity, data, _operations(folder, identity, p), terminal=done
        )
        if done and value != read_seal(folder / "result.json"):
            raise ValueError("terminal differs from authenticated operations")
        rows.append(
            task
            | {
                k: v
                for k, v in value.items()
                if k
                not in {
                    "identity",
                    "operations",
                    "historical_endpoint",
                    "selected_start",
                    "retained",
                }
            }
            | {
                "training_nmse": (value["retained"] or {}).get("training_nmse"),
                "source_training_nmse": (value["historical_endpoint"] or {}).get(
                    "training_nmse"
                ),
                "currently_verified": value["retained"] is not None,
            }
        )
    count = sum(row["status"] == "complete" for row in rows)
    value = {
        "protocol": PROTOCOL,
        "status": "complete" if count == 3 else "incomplete",
        "expected": 3,
        "recorded": count,
        "rows": rows,
        "historical_costs": [
            {
                "source_task_index": record["task_index"],
                "cost": pilot.journal._cost(
                    _source_operations(data["source_plan"], record)
                ),
                "accounting_scope": "recorded numerical operations only",
                "no_numerical_reservation": not bool(record["operations"]),
                "scheduler_preflight_wall_time": "unknown",
            }
            for record in data["source_records"]
        ],
        "missing_task_prior_preflight_cost": (
            "unknown; no numerical operation reservation exists"
        ),
        "source_report_reproduced": True,
        "source_log_hashes": data["source_log_hashes"],
        "old_allowances_renewed": False,
        "test_data_opened": False,
        "validation_data_opened": False,
        "llm_calls": 0,
        "limitation": (
            "Local numerical evidence only; no global optimum, unique physical "
            "coefficients, equation defect, or automatic model promotion is certified."
        ),
    }
    with public._lock(root / "report"):
        public._write(root / "report/summary.json", value)
    return value


def inspect(root: Path):
    """Package this campaign's receipts only, without running numerical stages."""
    import gzip
    import io
    import tarfile

    summary = report(root)
    path, temporary = root / "inspection.tar.gz", root / "inspection.tmp"
    paths = [root / "plan.json", root / "inputs.json"]
    if (root / "launcher_execution.json").exists():
        paths.append(root / "launcher_execution.json")
    for name in ("results", "report", "submissions", "logs"):
        paths.extend(p for p in (root / name).rglob("*") if p.is_file())
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="w", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w") as archive,
    ):
        for item in sorted(paths):
            if item.is_symlink() or not item.resolve().is_relative_to(root.resolve()):
                raise ValueError("archive path escapes campaign")
            if item.name.startswith("."):
                continue
            blob = item.read_bytes()
            entry = tarfile.TarInfo(item.relative_to(root).as_posix())
            entry.mode, entry.size = 0o644, len(blob)
            archive.addfile(entry, io.BytesIO(blob))
    temporary.replace(path)
    return {"status": summary["status"], "archive": str(path.resolve())}
