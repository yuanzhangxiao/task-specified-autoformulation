"""Bounded training-only timescale slices and distinct starts of the frozen model.

A frozen negative I self coefficient is a conditional decay-branch diagnostic,
not a restriction of the signed model family or a global lower bound.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_continuation as numerical
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_multioutput_profile as profiling
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_profile_pilot as pilot
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-timescale-controls-1"
RECOVERY_PROTOCOL = "construction-timescale-controls-2"
RECOVERY_INPUT_PROTOCOL = "construction-timescale-controls-inputs-2"
INPUT_PROTOCOL = "construction-timescale-controls-inputs-1"
RATE = "coefficient:I:I"
OUTER_ROLES = {
    "activation": "k_act_term_14_0_1f55a261f4",
    "self": RATE,
    "glucose_delay": "tau_term_11_2_a8895835eb",
    "utilization_decay": "effective_decay:U:tau_term_13_2_4c01e0f40a",
    "memory_decay": "effective_decay:I_mem:tau_term_14_1_3291bc29fb",
    "feedback": "coefficient:I:I_mem",
}


class Policy(pilot.previous.Policy):
    """Frozen roster and finite per-operation CPU allowances; no new model calls."""

    population: Literal["returned-eight"] = "returned-eight"
    recovery_mode: Literal["legacy", "project_first_local"] = "legacy"
    search_seconds: pilot.Nonnegative = Field(default=600, gt=0, le=600)


class ParentTask(pilot.previous.previous.ParentTask):
    """Authenticated profile-pilot terminal, including precision-only tasks."""

    arm: Literal["joint", "profiled", "fixed_precision"]


class Inputs(StrictSchema):
    """Preserve the complete preceding input/cost chain and all eight receipts."""

    protocol: Literal[
        "construction-timescale-controls-inputs-1",
        "construction-timescale-controls-inputs-2",
    ]
    prior_controls: dict | None = None
    source_plan: dict
    source_plan_sha256: pilot.Digest
    source_inputs_sha256: pilot.Digest
    historical_inputs: dict
    parent_tasks: list[ParentTask] = Field(min_length=8, max_length=8)
    selected_task_index: Literal[1]
    selected_origin: str
    selected_verification_receipt: dict
    endpoint: dict
    endpoint_sha256: pilot.Digest
    profile_outer_names: list[str] = Field(min_length=6, max_length=6)
    profile_inner_names: list[str] = Field(min_length=21, max_length=21)
    old_allowances_renewed: Literal[False]
    validation_exported: Literal[False]
    test_data_opened: Literal[False]


def export_inputs(source: Path, destination: Path) -> dict:
    """Authenticate all eight pilot terminals read-only, retaining their costs."""
    source = source.resolve()
    if destination.resolve().is_relative_to(source):
        raise ValueError("export must not write into source campaign")
    plan, old = pilot.verify(source, runtime=False)
    policy = pilot.Policy.model_validate(plan["policy"])
    if policy.population != "returned-eight" or len(plan["tasks"]) != 8:
        raise ValueError("all eight returned pilot tasks required")
    parents, selected, profile = [], None, None
    selected_receipt, selected_origin = None, None
    for task in plan["tasks"]:
        folder = pilot._folder(source, task)
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        operations = pilot._operations(folder, identity, policy)
        value = pilot._assemble(
            identity, old["models"][task["model"]], operations, terminal=True
        )
        if read_seal(folder / "result.json") != value:
            raise ValueError("source terminal differs from authenticated receipts")
        parents.append(
            {
                **{k: task[k] for k in ("index", "model", "task_id", "arm")},
                "kind": "verification_only"
                if task["arm"] == "fixed_precision"
                else "continuation",
                "task_sha256": public.content_sha256(task),
                "terminal_sha256": public.content_sha256(value),
                "operations": pilot.history._receipts(folder, operations),
                "cost": {
                    k: v
                    for k, v in value["cost"].items()
                    if k in pilot.previous.COST_KEYS
                },
            }
        )
        if task["index"] == 1:
            selected = value["retained"]
            selected_origin = value["retained_origin"]
            selected_receipt = read_seal(
                folder / "operations" / selected_origin / "result.json"
            )
            search = pilot.previous.first._search_checkpoint(
                operations.get("continuation_search", {})
            )
            profile = search.get("profile_audit")
    if not selected or not selected.get("original_equations_verified") or not profile:
        raise ValueError("latest profile endpoint and certified partition required")
    value = {
        "protocol": INPUT_PROTOCOL,
        "source_plan": plan,
        "source_plan_sha256": public.content_sha256(plan),
        "source_inputs_sha256": public.content_sha256(old),
        "historical_inputs": old,
        "parent_tasks": parents,
        "selected_task_index": 1,
        "selected_origin": selected_origin,
        "selected_verification_receipt": selected_receipt,
        "endpoint": selected,
        "endpoint_sha256": public.content_sha256(selected),
        "profile_outer_names": profile["outer_names"],
        "profile_inner_names": profile["inner_names"],
        "old_allowances_renewed": False,
        "validation_exported": False,
        "test_data_opened": False,
    }
    _validate(value)
    seal(destination, value)
    return {
        "inputs_sha256": public.content_sha256(value),
        "authenticated_parent_tasks": len(parents),
        "selected_model": 6,
        "source_kappa": -selected["effective_parameters"][RATE],
        "old_allowances_renewed": False,
    }


def export_recovery_inputs(source: Path, destination: Path) -> dict:
    """Carry all ten immutable v1 outcomes into a separately budgeted v2 run."""
    source = source.resolve()
    if destination.resolve().is_relative_to(source):
        raise ValueError("export must not write into source campaign")
    plan, data = verify(source, runtime=False)
    policy = Policy.model_validate(plan["policy"])
    if plan["protocol"] != PROTOCOL or policy.recovery_mode != "legacy":
        raise ValueError("recovery source must be the returned v1 controls")
    records = []
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = _folder(source, task)
        operations = _operations(folder, identity, policy)
        terminal = _assemble(identity, data, operations, terminal=True)
        if read_seal(folder / "result.json") != terminal:
            raise ValueError("recovery source terminal differs from receipts")
        records.append(
            {
                "task_index": task["index"],
                "terminal_sha256": public.content_sha256(terminal),
                "operations": operations,
            }
        )
    value = deepcopy(data) | {
        "protocol": RECOVERY_INPUT_PROTOCOL,
        "prior_controls": {
            "plan": plan,
            "plan_sha256": public.content_sha256(plan),
            "records": records,
        },
    }
    _validate(value)
    seal(destination, value)
    return {
        "inputs_sha256": public.content_sha256(value),
        "authenticated_parent_tasks": len(records),
        "selected_model": 6,
        "matched_original_anchor_preserved": True,
        "old_allowances_renewed": False,
    }


def _validate_recovery_source(data: dict) -> None:
    """Reassemble every parent from bound receipts before using new allowances."""
    source = data.get("prior_controls")
    if not isinstance(source, dict) or set(source) != {
        "plan",
        "plan_sha256",
        "records",
    }:
        raise ValueError("recovery source evidence required")
    base = {k: deepcopy(v) for k, v in data.items() if k != "prior_controls"}
    base["protocol"] = INPUT_PROTOCOL
    plan = source["plan"]
    policy = Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or policy.recovery_mode != "legacy"
        or public.content_sha256(plan) != source["plan_sha256"]
        or plan["inputs_sha256"] != public.content_sha256(base)
        or plan["tasks"] != _tasks(base, policy)
        or plan["numerical_contract"] != _contract(policy)
        or plan["time_grid"] != time_grid(_training_inputs(base)["training"])
        or len(source["records"]) != len(plan["tasks"])
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
        raise ValueError("recovery source plan differs")
    allowed = _allowances(policy)
    for task, record in zip(plan["tasks"], source["records"], strict=True):
        if (
            set(record) != {"task_index", "terminal_sha256", "operations"}
            or record["task_index"] != task["index"]
        ):
            raise ValueError("recovery source roster differs")
        identity = {"plan_sha256": source["plan_sha256"], "task": task}
        operations = record["operations"]
        if set(operations) - set(allowed):
            raise ValueError("unknown recovery source operation")
        for name, operation in operations.items():
            bound = {
                "campaign": identity,
                "stage": name,
                "allowance_seconds": allowed[name],
            }
            if operation.get("identity") != bound or operation.get(
                "progress_sha256"
            ) != public.content_sha256(operation.get("observed_progress", {})):
                raise ValueError("recovery source operation differs")
        terminal = _assemble(identity, base, operations, terminal=True)
        if public.content_sha256(terminal) != record["terminal_sha256"]:
            raise ValueError("recovery source terminal differs")


def _training_inputs(data: dict) -> dict:
    return data["historical_inputs"]["historical_inputs"]["training_inputs"]


def _validate(data: dict) -> None:
    parsed = Inputs.model_validate(data)
    old, plan = parsed.historical_inputs, parsed.source_plan
    policy = pilot.Policy.model_validate(plan["policy"])
    pilot._validate(old, policy)
    expected = pilot._tasks(old, policy)
    if (
        policy.population != "returned-eight"
        or plan["tasks"] != expected
        or public.content_sha256(plan) != parsed.source_plan_sha256
        or public.content_sha256(old) != parsed.source_inputs_sha256
        or plan["inputs_sha256"] != parsed.source_inputs_sha256
    ):
        raise ValueError("source plan/input identity differs")
    for parent, task in zip(parsed.parent_tasks, expected, strict=True):
        if (
            parent.index != task["index"]
            or parent.model != task["model"]
            or parent.task_id != task["task_id"]
            or parent.arm != task["arm"]
            or parent.task_sha256 != public.content_sha256(task)
        ):
            raise ValueError("parent task identity differs")
        if set(parent.operations) - set(pilot._allowances(policy, task)) or any(
            not o.receipt_sha256 for o in parent.operations.values()
        ):
            raise ValueError("all parent operation receipts required")
        pilot.previous.previous._validate_cost(parent.cost)
    endpoint = parsed.endpoint
    receipt = parsed.selected_verification_receipt
    parent = parsed.parent_tasks[1]
    evidence = parent.operations.get(parsed.selected_origin)
    if (
        evidence is None
        or evidence.receipt_sha256 != public.content_sha256(receipt)
        or pilot._verified(receipt) != endpoint
        or receipt["identity"]
        != {
            "campaign": {"plan_sha256": parsed.source_plan_sha256, "task": expected[1]},
            "stage": parsed.selected_origin,
            "allowance_seconds": pilot._allowances(policy, expected[1])[
                parsed.selected_origin
            ],
        }
    ):
        raise ValueError("selected endpoint differs from parent verification receipt")
    if public.content_sha256(endpoint) != parsed.endpoint_sha256:
        raise ValueError("selected endpoint digest differs")
    original = _training_inputs(data)["models"][6]
    if not endpoint.get("original_equations_verified"):
        raise ValueError("selected endpoint must be independently verified")
    compact = pilot.history._endpoint(endpoint, parsed.selected_origin)
    pilot.history._validate_endpoint(
        pilot.history.Endpoint.model_validate(compact), original
    )
    names = endpoint["recovery_audit"]["coordinate_reduction"]["effective_names"]
    effective = endpoint["effective_parameters"]
    if (
        len(names) != 27
        or len(set(names)) != 27
        or set(names) != set(effective)
        or [effective[n] for n in names] != endpoint["reduced_vector"]
        or not np.isfinite(list(effective.values())).all()
        or effective[RATE] >= 0
        or set(parsed.profile_outer_names) != set(OUTER_ROLES.values())
        or set(parsed.profile_outer_names) & set(parsed.profile_inner_names)
        or set(names) != set(parsed.profile_outer_names + parsed.profile_inner_names)
    ):
        raise ValueError("certified model-six coordinate partition differs")
    if _training_inputs(data)["training"]["name"] != "train":
        raise ValueError("timescale controls require training only")
    if parsed.protocol == RECOVERY_INPUT_PROTOCOL:
        _validate_recovery_source(data)
    elif parsed.prior_controls is not None:
        raise ValueError("legacy inputs cannot carry a recovery source")


def time_grid(training: dict) -> dict:
    """Read only observation times: maximum span and minimum positive interval."""
    times = [np.asarray(row["time"], float) for row in training["rows"]]
    if not times or any(
        len(t) < 2 or not np.isfinite(t).all() or not np.all(np.diff(t) > 0)
        for t in times
    ):
        raise ValueError("finite increasing observation grids required")
    return {
        "span": float(max(t[-1] - t[0] for t in times)),
        "minimum_interval": float(min(np.min(np.diff(t)) for t in times)),
        "time_grid_sha256": public.content_sha256([t.tolist() for t in times]),
    }


def _tasks(data: dict, policy: Policy) -> list[dict]:
    """Freeze six rate slices, three timescale starts and one saved continuation."""
    grid = time_grid(_training_inputs(data)["training"])
    span, step = grid["span"], grid["minimum_interval"]
    middle = float(np.sqrt(span * step))
    source = data["endpoint"]["effective_parameters"]
    roster = [("saved_continuation", "saved_incumbent", {}, None)]
    rates = [
        ("inverse_span", 1 / span),
        ("inverse_geometric_interval", 1 / middle),
        ("inverse_interval", 1 / step),
        ("ten_per_interval", 10 / step),
        ("thousand_per_interval", 1000 / step),
        ("source_rate", -source[RATE]),
    ]
    roster.extend(("frozen_rate", label, {RATE: -rate}, rate) for label, rate in rates)
    for label, scale, ratio in (
        ("slow", span, 0.05),
        ("intermediate", middle, 0.1),
        ("resolved", step, 0.2),
    ):
        overrides = {
            OUTER_ROLES["self"]: -1 / scale,
            OUTER_ROLES["glucose_delay"]: scale,
            OUTER_ROLES["utilization_decay"]: 1 / scale,
            OUTER_ROLES["memory_decay"]: 2 / scale,
            OUTER_ROLES["activation"]: ratio / scale,
            OUTER_ROLES["feedback"]: ratio / scale,
        }
        roster.append(("distinct_start", label, overrides, None))
    task_id = data["historical_inputs"]["models"][6]["task_id"]
    return [
        {
            "index": i,
            "model": 6,
            "task_id": task_id,
            "arm": arm,
            "label": label,
            "start_overrides": overrides,
            "frozen_kappa": rate,
            "optimization_seconds": policy.search_seconds,
            "rate_branch": "negative_I_self_decay" if arm == "frozen_rate" else None,
            "free_outer_coordinates": 5 if arm == "frozen_rate" else 6,
            "inner_coordinates": 21,
            **(
                {"recovery_mode": policy.recovery_mode}
                if policy.recovery_mode != "legacy"
                else {}
            ),
        }
        for i, (arm, label, overrides, rate) in enumerate(roster)
    ]


def _contract(policy: Policy | None = None) -> dict:
    contract = {
        "profile_policy": public._jsonable(profiling.policy_receipt()),
        "qualification": "guarded then all-coordinate full-residual multiprecision",
        "full_accepted_history": True,
        "verification_tolerances": [[1e-11, 1e-13]],
        "start_policy": "observation-time-grid-only; same retained inner anchor",
        "signed_family_bounds_preserved": True,
        "rate_branch": "negative_I_self_decay_only_for_six_conditional_slices",
        "global_lower_bound": False,
        "reference_coefficients_used": False,
        "diagnosis_scope": "all 27 coordinates at each independently verified endpoint",
        "constrained_adequacy": "not certified by outer optimizer termination",
    }

    if policy is not None and policy.recovery_mode == "project_first_local":
        contract.update(
            qualification=(
                "bounded inner projection, independent rollout, "
                "local profiled outer check"
            ),
            start_policy=(
                "same deterministic outer starts; "
                "bounded 21-coordinate projection first"
            ),
            local_probe_policy=(
                "current-physical-coordinate floors from observation time grid; "
                "optimizer units unchanged"
            ),
            search_allowance=(
                "fixed-inner projection and outer qualification/search "
                "share declared optimization allowance"
            ),
            full_presearch_derivative_gate=False,
        )
    return contract


def _protocol(policy: Policy) -> str:
    return (
        RECOVERY_PROTOCOL if policy.recovery_mode == "project_first_local" else PROTOCOL
    )


def _projection_seconds(policy: Policy) -> float:
    """Reserve part of the declared optimization budget for one bounded solve."""
    return min(60.0, policy.search_seconds / 4.0)


def _local_probe_floors(data: dict) -> dict[str, float]:
    """Set physical rate/delay probe floors from training observation times only."""
    grid = time_grid(_training_inputs(data)["training"])
    return {
        name: grid["minimum_interval"] if role == "glucose_delay" else 1 / grid["span"]
        for role, name in OUTER_ROLES.items()
    }


def prepare(root: Path, data: dict, policy: Policy) -> dict:
    """Seal the fixed train-only roster without constructing a numerical oracle."""
    _validate(data)
    if (policy.recovery_mode == "project_first_local") != (
        data["protocol"] == RECOVERY_INPUT_PROTOCOL
    ):
        raise ValueError("recovery policy and source protocol differ")
    plan = {
        "protocol": _protocol(policy),
        "inputs_sha256": public.content_sha256(data),
        "policy": policy.model_dump(mode="json"),
        "numerical_contract": _contract(policy),
        "source_sha256": public._source_identity(),
        "runtime": pilot._runtime(),
        "tasks": _tasks(data, policy),
        "time_grid": time_grid(_training_inputs(data)["training"]),
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
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Authenticate data, roster, source and policy before a numerical operation."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    _validate(data)
    p = Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != _protocol(p)
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != _tasks(data, p)
        or plan["numerical_contract"] != _contract(p)
        or plan["time_grid"] != time_grid(_training_inputs(data)["training"])
        or (
            (p.recovery_mode == "project_first_local")
            != (data["protocol"] == RECOVERY_INPUT_PROTOCOL)
        )
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
        raise ValueError("timescale controls plan differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != pilot._runtime()
    ):
        raise ValueError("timescale controls source/runtime differs")
    return plan, data


_folder = pilot._folder


def _allowances(policy: Policy) -> dict[str, float]:
    if policy.recovery_mode == "project_first_local":
        inner = _projection_seconds(policy)
        return {
            "fixed_inner": inner + 90,
            "incumbent_verification": policy.verification_seconds + 90,
            "continuation_search": policy.search_seconds - inner + 150,
            "continuation_verification": policy.verification_seconds + 90,
            "retained_diagnosis": policy.diagnostic_seconds + 90,
        }
    return {
        "qualification": policy.diagnostic_seconds + 90,
        "incumbent_verification": policy.verification_seconds + 90,
        "continuation_search": policy.search_seconds + 150,
        "continuation_verification": policy.verification_seconds + 90,
        "retained_diagnosis": policy.diagnostic_seconds + 90,
    }


def _operations(folder: Path, identity: dict, policy: Policy) -> dict:
    """Reuse the nonrenewable journal with the new explicit stage ceilings."""
    allowed = _allowances(policy)
    if any(
        p.is_dir() and p.name not in allowed for p in (folder / "operations").glob("*")
    ):
        raise ValueError("unknown timescale operation")
    result = {}
    for name, seconds in allowed.items():
        path = folder / "operations" / name
        bound = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
        if (path / "result.json").exists():
            result[name] = pilot.journal._read_operation(path, bound)
        elif (path / "started.json").exists():
            if read_seal(path / "started.json") != bound:
                raise ValueError("timescale operation reservation differs")
            result[name] = {
                "status": "running_or_interrupted",
                "value": None,
                "wall_seconds": None,
                "cpu_seconds": None,
                "budget_charge_seconds": seconds,
                "accounting_complete": False,
            }
    return result


class FrozenRateProfile:
    """Drop one outer direction while retaining its exact signed coefficient."""

    def __init__(self, profile, oracle, kappa: float):
        self.base, self.fixed_index, self.kappa = (
            profile,
            oracle.names.index(RATE),
            kappa,
        )
        if (
            not np.isfinite(kappa)
            or kappa <= 0
            or self.fixed_index not in profile.outer
        ):
            raise ValueError(
                "positive frozen decay rate and outer self coefficient required"
            )
        self.outer = tuple(i for i in profile.outer if i != self.fixed_index)
        self.inner = profile.inner
        self.columns = [profile.outer.index(i) for i in self.outer]

    @property
    def audit(self):
        return self.base.audit | {
            "frozen_coordinate": RATE,
            "frozen_value": -self.kappa,
            "free_outer_indices": list(self.outer),
            "conditional_rate_slice": True,
            "global_lower_bound": False,
        }

    def evaluate(self, vector, deadline, *, jacobian=True):
        point = np.asarray(vector, float).copy()
        if point[self.fixed_index] != -self.kappa:
            raise ValueError("frozen rate changed")
        full, residual, derivative, audit = self.base.evaluate(
            point, deadline, jacobian=jacobian
        )
        if full[self.fixed_index] != -self.kappa:
            raise ValueError("projection changed frozen rate")
        return full, residual, derivative[:, self.columns] if jacobian else None, audit


def _start(oracle, source: dict, task: dict) -> dict:
    """Validate deterministic aggregate starts against the unchanged original lift."""
    values = source["effective_parameters"] | task["start_overrides"]
    vector = oracle.vector(values)
    return oracle.parameters(vector)


def _assemble(identity: dict, data: dict, operations: dict, *, terminal=False) -> dict:
    value = pilot._assemble(
        identity, {"endpoint": data["endpoint"]}, operations, terminal=terminal
    )
    fit = pilot.previous.first._search_checkpoint(
        operations.get("continuation_search", {})
    )
    if identity["task"].get("recovery_mode") == "project_first_local":
        check = fit.get("profile_derivative_check") or {}
        projection = pilot.previous.first._search_checkpoint(
            operations.get("fixed_inner", {})
        )
        value.update(
            qualification_passed=bool(check.get("search_qualified")),
            qualification_scope="only free profiled outer coordinates before search",
            projection_checkpoint_available=bool(projection.get("best")),
            projection_verified=value.get("selected_start") is not None,
            full_presearch_derivative_gate=False,
        )
    rate = identity["task"]["frozen_kappa"]
    for key in ("selected_start", "retained"):
        endpoint = value.get(key)
        if (
            endpoint
            and rate is not None
            and endpoint["effective_parameters"][RATE] != -rate
        ):
            raise ValueError("verified endpoint escaped frozen rate")
    return value | {
        "conditional_rate_slice": rate is not None,
        "rate_profile_is_global_lower_bound": False,
        "original_signed_family_restricted": False,
        "constrained_fitting_adequacy": "not_assessed",
        "full_coordinate_diagnosis_scope": "all 27 original aggregate coordinates",
        "search_termination": fit.get("stop_reason", "not_started"),
        "accepted_history_protocol": fit.get("accepted_history_protocol"),
    }


def run_task(root: Path, index: int) -> dict:
    """Run/resume one isolated start, independently verifying its own endpoint."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown timescale task")
    task, p = plan["tasks"][index], Policy.model_validate(plan["policy"])
    training_inputs = _training_inputs(data)
    problem = PrefixFitProblem.model_validate(training_inputs["models"][6]["problem"])
    training = PublicSplit.model_validate(training_inputs["training"])
    settings = kernel.NumericalPolicy.model_validate(
        training_inputs["numerical_policy"]
    )
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder, allowed = _folder(root, task), _allowances(p)
    anchor = data["endpoint"]["parameters"]

    def assemble(terminal=False):
        return _assemble(
            identity, data, _operations(folder, identity, p), terminal=terminal
        )

    def operate(name, work):
        def call(save):
            deadline = monotonic() + allowed[name]
            original, oracle, profile = profiling.make_oracle(
                problem, training, settings, anchor
            )
            if p.recovery_mode == "project_first_local":
                oracle.configure_local_probes(_local_probe_floors(data))
            if [oracle.names[i] for i in profile.outer] != data["profile_outer_names"]:
                raise ValueError("profile outer identity differs")
            if [oracle.names[i] for i in profile.inner] != data["profile_inner_names"]:
                raise ValueError("profile inner identity differs")

            def checked_save(value):
                save(
                    value
                    | {
                        "affine_execution": deepcopy(original.affine_audit),
                        "recovery_audit": deepcopy(oracle.recovery_audit),
                    }
                )
                if monotonic() >= deadline:
                    raise TimeoutError("inclusive timescale allowance exhausted")

            if monotonic() >= deadline:
                raise TimeoutError("timescale oracle setup allowance exhausted")
            value = work(original, oracle, profile, checked_save)
            checked_save(value)
            return value | {
                "affine_execution": deepcopy(original.affine_audit),
                "recovery_audit": deepcopy(oracle.recovery_audit),
            }

        return pilot.journal._operation(folder, identity, name, allowed[name], call)

    with public._lock(folder):
        if (folder / "result.json").exists():
            value = assemble(terminal=True)
            if read_seal(folder / "result.json") != value:
                raise ValueError("timescale terminal differs from receipts")
            return value
        project_first = p.recovery_mode == "project_first_local"
        if project_first:
            projected = operate(
                "fixed_inner",
                lambda o, a, f, s: pilot._fixed_inner(
                    a,
                    FrozenRateProfile(f, a, task["frozen_kappa"])
                    if task["frozen_kappa"] is not None
                    else f,
                    _start(a, data["endpoint"], task),
                    _projection_seconds(p),
                    s,
                ),
            )
            checkpoint = pilot.previous.first._search_checkpoint(projected)
            if checkpoint.get("best"):
                operate(
                    "incumbent_verification",
                    lambda o, a, f, s: numerical.verify_endpoint_ladder(
                        o,
                        a,
                        checkpoint["best"]["parameters"],
                        p.verification_seconds,
                        s,
                        tolerances=((1e-11, 1e-13),),
                    ),
                )
        else:
            operate(
                "qualification",
                lambda o, a, f, s: pilot.qualify_profile(
                    problem,
                    training,
                    settings,
                    numerical.to_original(o, a, _start(a, data["endpoint"], task)),
                    p.diagnostic_seconds,
                    s,
                    coordinate_oracle=a,
                    coordinate_parameters=_start(a, data["endpoint"], task),
                ),
            )
            operate(
                "incumbent_verification",
                lambda o, a, f, s: numerical.verify_endpoint_ladder(
                    o,
                    a,
                    _start(a, data["endpoint"], task),
                    p.verification_seconds,
                    s,
                    tolerances=((1e-11, 1e-13),),
                ),
            )
        current = assemble()
        selected = current["selected_start"]
        if selected is not None and (project_first or current["qualification_passed"]):
            attempt = operate(
                "continuation_search",
                lambda o, a, f, s: pilot._profile_search(
                    a,
                    FrozenRateProfile(f, a, task["frozen_kappa"])
                    if task["frozen_kappa"] is not None
                    else f,
                    pilot.previous._verified_coordinates(o, a, selected),
                    p.search_seconds - _projection_seconds(p)
                    if project_first
                    else p.search_seconds,
                    p.maximum_calls,
                    s,
                    preserve_full_history=True,
                ),
            )
            fit = pilot.previous.first._search_checkpoint(attempt)
            if fit.get("best"):
                operate(
                    "continuation_verification",
                    lambda o, a, f, s: numerical.verify_endpoint_ladder(
                        o,
                        a,
                        fit["best"]["parameters"],
                        p.verification_seconds,
                        s,
                        tolerances=((1e-11, 1e-13),),
                    ),
                )
        retained = assemble()["retained"]
        if retained:
            ops = _operations(folder, identity, p)
            fit = pilot.previous.first._search_checkpoint(
                ops.get("continuation_search", {})
            )
            candidate = pilot._verified(ops.get("continuation_verification", {}))
            matches = (
                candidate is not None
                and candidate["parameters"] == retained["parameters"]
            )
            context = (
                fit
                if matches
                else {
                    "stop_reason": "retained_endpoint_has_no_matching_search_history",
                    "accepted_steps": [],
                }
            )
            binding = {
                "diagnosed_parameters_sha256": public.content_sha256(
                    retained["parameters"]
                ),
                "verification_sha256": public.content_sha256(retained),
                "continuation_endpoint_matches_retained": matches,
            }

            def diagnosis(o, a, f, save):
                value = pilot.diagnose_profile(
                    o,
                    a,
                    pilot.previous._verified_coordinates(o, a, retained),
                    retained,
                    context,
                    p.diagnostic_seconds,
                    lambda v: save(v | binding),
                )
                return value | binding

            operate("retained_diagnosis", diagnosis)
        value = assemble(terminal=True)
        seal(folder / "result.json", value)
        return value


def report(root: Path) -> dict:
    """Preserve every fixed slice/start and separate historical from new costs."""
    plan, data = verify(root, runtime=False)
    p, rows = Policy.model_validate(plan["policy"]), []
    for task in plan["tasks"]:
        folder = _folder(root, task)
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        done = (folder / "result.json").exists()
        value = _assemble(
            identity, data, _operations(folder, identity, p), terminal=done
        )
        if done and read_seal(folder / "result.json") != value:
            raise ValueError("timescale terminal differs from receipts")
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
    value = {
        "protocol": _protocol(p),
        "status": "complete" if count == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": count,
        "rows": rows,
        "time_grid": plan["time_grid"],
        "source_kappa": -data["endpoint"]["effective_parameters"][RATE],
        "historical_costs": {
            "profile_pilot_by_task": data["parent_tasks"],
            "prior_campaign_inputs": data["historical_inputs"],
        },
        "matched_optimization_seconds": p.search_seconds,
        **(
            {
                "fixed_inner_seconds": _projection_seconds(p),
                "outer_qualification_and_search_seconds": p.search_seconds
                - _projection_seconds(p),
                "prior_timescale_costs": [
                    {
                        "task_index": row["task_index"],
                        "cost": pilot.journal._cost(row["operations"]),
                    }
                    for row in data["prior_controls"]["records"]
                ],
            }
            if p.recovery_mode == "project_first_local"
            else {}
        ),
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
    with public._lock(root / "report"):
        public._write(root / "report/summary.json", value)
    return value


def inspect(root: Path) -> dict:
    """Package authenticated receipts using the existing deterministic archive path."""
    # The existing packager performs no science; invoke it with this report function
    # by assembling the same narrow archive here, without mutating module globals.
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
        for source in sorted(paths):
            if source.is_symlink() or not source.resolve().is_relative_to(
                root.resolve()
            ):
                raise ValueError("archive path escapes campaign")
            if source.name.startswith("."):
                continue
            blob = source.read_bytes()
            entry = tarfile.TarInfo(source.relative_to(root).as_posix())
            entry.mode, entry.size = 0o644, len(blob)
            archive.addfile(entry, io.BytesIO(blob))
    temporary.replace(path)
    return {"status": summary["status"], "archive": str(path.resolve())}
