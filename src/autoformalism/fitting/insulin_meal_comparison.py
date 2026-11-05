"""Matched parent and child fits for the direct ingestion term in candidate 15."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_meal_coordinates as meal
from autoformalism.fitting import prefix_meal_joint_profile as joint
from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting import prefix_portfolio_timescale as timescale
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.compact_search_checkpoint import SearchCheckpointPolicy
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import construction_portfolio_fitting as first
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "candidate15-direct-insulin-meal-comparison-1"
GAIN = "c_I"
MEAL = "meal_rate_g_per_min"
ARMS = ("parent", "child")
Arm = Literal["parent", "child"]


class Policy(StrictSchema):
    """Identical, nonrenewable allowances for the two scientific models."""

    search_seconds: FiniteFloat = Field(default=1800, gt=0, le=1800)
    maximum_calls: int = Field(default=2000, ge=1, le=2000)
    verification_seconds: FiniteFloat = Field(default=300, gt=0, le=300)
    qualification_seconds: FiniteFloat = Field(default=300, gt=0, le=600)
    checkpoint_interval_seconds: FiniteFloat = Field(default=30, gt=0, le=300)
    persistence_headroom_seconds: FiniteFloat = Field(default=900, ge=0, le=900)


def child_problem(parent: PrefixFitProblem) -> PrefixFitProblem:
    """Replace just the leading unit ingestion term in the saved insulin ODE."""
    payload = parent.model_dump(mode="json")
    model = payload["base_candidate"]
    rows = [row for row in model["state_equations"] if row["state"] == "I"]
    if (
        len(rows) != 1
        or MEAL not in parent.context.external_inputs
        or not rows[0]["rhs"].startswith(f"{MEAL} - I + ")
        or rows[0]["rhs"].count(MEAL) != 1
    ):
        raise ValueError("expected the candidate-15 direct unit insulin meal term")
    names = {x["name"] for k in ("parameters", "states", "processes") for x in model[k]}
    if GAIN in names or GAIN in payload["parameter_guesses"]:
        raise ValueError("insulin meal gain name already in use")
    rows[0]["rhs"] = f"{GAIN} * " + rows[0]["rhs"]
    model["parameters"].append(
        {"name": GAIN, "scope": "global", "role": "nonnegative_coefficient"}
    )
    payload["parameter_guesses"][GAIN] = 1.0
    return PrefixFitProblem.model_validate(payload)


def allowances(policy: Policy) -> dict[str, float]:
    """Stage reservation seconds, including persistence where applicable."""
    return {
        "nesting": policy.verification_seconds + 60,
        "qualification": policy.qualification_seconds + 120,
        "profiled_start_verification": policy.verification_seconds + 60,
        "search": policy.search_seconds + policy.persistence_headroom_seconds + 60,
        "endpoint_verification": policy.verification_seconds + 60,
        "metrics": policy.verification_seconds + 60,
    }


class ComparisonFit:
    """Compile the scoped original and positive meal-coordinate model for one arm."""

    def __init__(self, parent, training, numerical, retained_parameters, arm: Arm):
        if arm not in ARMS or training.name != "train":
            raise ValueError("requires a known arm and public training trajectories")
        self.parent, self.training, self.numerical = parent, training, numerical
        self.arm = arm
        self.problem = parent if arm == "parent" else child_problem(parent)
        self.parent_parameters = dict(retained_parameters)
        self.anchor = self.parent_parameters | ({GAIN: 1.0} if arm == "child" else {})
        self.original, self.oracle = timescale.make_oracle(
            self.problem,
            training,
            numerical,
            self.anchor,
            physical_forcing=(GAIN,) if arm == "child" else (),
        )
        self.start = dict(self.oracle.start)
        self.chart = meal.certify(self.oracle, self.start)
        self.profiler_factory = partial(
            joint.JointMealProfile, additional_forcing=GAIN if arm == "child" else None
        )

    def verify_start(self, expected_nmse, seconds):
        """Reproduce the saved parent loss and the pointwise nested initial rollout."""
        deadline = monotonic() + seconds
        parent_original = (
            self.original
            if self.arm == "parent"
            else kernel.TrainingOracle(self.parent, self.training, self.numerical)
        )
        parent_resid, _ = parent_original.evaluate(
            parent_original.vector(self.parent_parameters),
            deadline,
            jacobian=False,
            method="Radau",
        )
        arm_resid = parent_resid
        if self.arm == "child":
            arm_resid, _ = self.original.evaluate(
                self.original.vector(self.anchor),
                deadline,
                jacobian=False,
                method="Radau",
            )
        chart_resid, _ = self.oracle.evaluate(
            self.oracle.vector(self.start), deadline, jacobian=False
        )
        loss = float(np.mean(arm_resid**2))
        chart_loss = float(np.mean(chart_resid**2))
        chart_loss_gap = abs(loss - chart_loss)
        parent_loss = float(np.mean(parent_resid**2))
        gap = float(np.max(np.abs(arm_resid - parent_resid)))
        chart_gap = float(np.max(np.abs(arm_resid - chart_resid)))
        passed = (
            abs(parent_loss - expected_nmse) <= max(1e-10, 1e-8 * expected_nmse)
            and gap <= 1e-7
            and chart_gap <= 1e-7
            and chart_loss_gap <= max(1e-12, 1e-8 * max(loss, chart_loss))
        )
        return {
            "passed": bool(passed),
            "training_nmse": loss,
            "original_equations_verified": bool(passed),
            "parent_training_nmse": parent_loss,
            "expected_parent_training_nmse": expected_nmse,
            "parent_arm_maximum_scaled_discrepancy": gap,
            "chart_maximum_scaled_discrepancy": chart_gap,
            "solver_loss_discrepancy": chart_loss_gap,
            "solver_prediction_discrepancy": chart_gap,
            "parameters": dict(self.anchor),
            "effective_parameters": dict(self.start),
            "parameter_count": len(self.anchor),
            "direct_insulin_meal_gain": 1.0,
            "chart": self.chart.audit,
            "time_chart_sha256": self.oracle.reduction.receipt["sha256"],
            "model_sha256": public.content_sha256(self.problem),
        }

    def qualify(self, seconds, save):
        """Check the shared/trajectory design and original-coordinate pairing."""
        return roots.root_profile_qualification(
            self.oracle,
            self.start,
            seconds,
            save,
            profiler_factory=self.profiler_factory,
        )

    def search(self, policy, save):
        """Apply the same six-coordinate Powell budget to each arm's profiled loss."""
        return roots.root_profile_search(
            self.oracle,
            self.start,
            policy.search_seconds,
            policy.maximum_calls,
            save,
            profiler_factory=self.profiler_factory,
            coordinate_chart=self.chart,
            failure_diagnostics=True,
            checkpoint_policy=SearchCheckpointPolicy(
                interval_seconds=policy.checkpoint_interval_seconds
            ),
            overall_wall_seconds=policy.search_seconds
            + policy.persistence_headroom_seconds,
        )

    def verify(self, parameters, expected_nmse, seconds, save):
        """Verify the profiled trial in the original equations using Radau."""
        result = portfolio.verify_endpoint(
            self.original, self.oracle, parameters, seconds, save
        )
        if result.get("original_equations_verified") and not np.isclose(
            result["training_nmse"], expected_nmse, rtol=1e-8, atol=1e-10
        ):
            result.update(
                original_equations_verified=False,
                error="replay differs from trial loss",
            )
        return result

    def metrics(self, endpoint, seconds):
        """Summarize insulin timing and controls from the retained original ODE."""
        from autoformalism.research import meal_revision_metrics

        p = endpoint["parameters"]
        residual, _ = self.original.evaluate(
            self.original.vector(p),
            monotonic() + seconds,
            jacobian=False,
            method="Radau",
        )
        score = float(np.mean(residual**2))
        if not np.isclose(score, endpoint["training_nmse"], atol=1e-10, rtol=1e-8):
            raise ValueError("retained endpoint differs from diagnostic replay")
        model = self.original.system.model
        names = model.state_names
        if any(
            model.direct_state_observation_channels.get(n) != n for n in ("Gp", "I")
        ):
            raise ValueError(
                "conditional replay needs observed glucose and insulin states"
            )
        local = self.original.local_vector(
            self.original.vector(p), self.original.rows[0]
        )
        matrix = np.asarray(self.original._propagation.generator(local), dtype=float)
        insulin = matrix[names.index("I")].copy()
        glucose_gain, clearance = insulin[names.index("Gp")], -insulin[names.index("I")]
        accepted = [names.index("I"), names.index("Gp")]
        accepted.extend(
            len(names) + i
            for i, u in enumerate(self.original.system.inputs)
            if u in (MEAL, "insulin_pmol_per_kg_min")
        )
        insulin[accepted] = 0
        if insulin.any():
            raise ValueError("conditional insulin balance has additional drivers")
        return meal_revision_metrics.summarize(
            self.training,
            self.original.system.channels,
            self.original.scales,
            residual,
            glucose_gain=float(glucose_gain),
            clearance=float(clearance),
        )


def _retained(ops):
    """Retain a verified improvement only beyond Orion's solver-error guard."""
    from autoformalism.research import construction_portfolio_residual as retention

    selected, retained = None, None
    for stage in ("nesting", "profiled_start_verification", "endpoint_verification"):
        if first._verified(ops.get(stage)):
            trial = first._value(ops[stage])
            if (
                retained is None
                or retention.compare_endpoints(retained, trial, retention.Policy())[
                    "classification"
                ]
                == "meaningful_improvement"
            ):
                selected, retained = stage, trial
    return selected, retained


def _assemble(binding, ops):
    selected, retained = _retained(ops)
    search = first._value(ops.get("search"), partial=True)
    metrics = first._value(ops.get("metrics"))
    qualified_search = (
        ops.get("search", {}).get("status") == "complete"
        and search.get("search_calls_completed", 0) > 0
        and search.get("stop_reason") == "optimizer_stopped"
        and search.get("optimizer_success") is True
        and search.get("budget_exhausted") is False
        and (search.get("optimizer_terminal") or {}).get("feasible_evaluation_verified")
        is True
    )
    ready = bool(
        first._verified(ops.get("nesting"))
        and first._value(ops.get("qualification")).get("passed")
        and first._verified(ops.get("profiled_start_verification"))
        and qualified_search
        and first._verified(ops.get("endpoint_verification"))
        and metrics
    )
    return {
        "protocol": PROTOCOL,
        "identity": binding["identity"],
        "arm": binding["arm"],
        "binding_sha256": public.content_sha256(binding),
        "ready_for_comparison": ready,
        "operation_sha256": {k: public.content_sha256(v) for k, v in ops.items()},
        "operation_statuses": {k: v["status"] for k, v in ops.items()},
        "operation_errors": {k: v["error"] for k, v in ops.items() if v.get("error")},
        "search_termination": search.get("stop_reason", "not_started"),
        "search_calls_completed": search.get("search_calls_completed", 0),
        "search_calls_started": search.get("search_calls_started", 0),
        "search_budget_exhausted": search.get("budget_exhausted"),
        "search_allowance": search.get("execution_allowance"),
        "qualified_search": qualified_search,
        "retained_stage": selected,
        "retained_endpoint": retained,
        "metrics": metrics or None,
        "profile_certificate": first._value(ops.get("qualification")).get(
            "root_profile_certificate"
        ),
        "public_training_reconstruction": True,
        "stationarity_certified": False,
        "test_data_opened": False,
    }


def read_arm(folder, identity, arm: Arm, policy: Policy):
    """Reconcile the sealed terminal with every nonrenewable stage receipt."""
    folder = Path(folder)
    binding = read_seal(folder / "binding.json")
    if (
        arm not in ARMS
        or binding["protocol"] != PROTOCOL
        or binding["arm"] != arm
        or binding["identity"] != identity
        or binding["policy"] != policy.model_dump(mode="json")
    ):
        raise ValueError("arm source or policy differs from its reservation")
    ops = {}
    bounds = allowances(policy)
    for path in sorted((folder / "operations").iterdir()):
        if path.name not in bounds:
            raise ValueError("unknown arm operation")
        ops[path.name] = journal._read_operation(
            path,
            {
                "campaign": binding,
                "stage": path.name,
                "allowance_seconds": bounds[path.name],
            },
        )
    terminal = read_seal(folder / "result.json")
    if terminal != _assemble(binding, ops):
        raise ValueError("arm report differs from operation receipts")
    return terminal


def run_arm(
    folder: Path,
    identity: dict,
    parent: PrefixFitProblem,
    training: PublicSplit,
    numerical: kernel.NumericalPolicy,
    retained: dict,
    arm: Arm,
    policy: Policy,
) -> dict:
    """Resume one matched, budgeted original-equation fitting comparison arm."""
    policy = Policy.model_validate(policy.model_dump(mode="json"))
    if arm not in ARMS or training.name != "train":
        raise ValueError("requires a known arm and public training split")
    if (
        retained.get("original_equations_verified") is not True
        or not np.isfinite(retained["training_nmse"])
        or retained["training_nmse"] < 0
    ):
        raise ValueError("verified parent endpoint required")
    folder = Path(folder)
    binding = {
        "protocol": PROTOCOL,
        "identity": identity,
        "arm": arm,
        "parent_problem_sha256": public.content_sha256(parent),
        "training_sha256": public.content_sha256(training),
        "numerical_sha256": public.content_sha256(numerical),
        "retained_sha256": public.content_sha256(retained),
        "policy": policy.model_dump(mode="json"),
    }
    with public._lock(folder):
        seal(folder / "binding.json", binding)
        if (folder / "result.json").exists():
            return read_arm(folder, identity, arm, policy)
        ops, compiled = {}, None
        limits = allowances(policy)

        def fit():
            nonlocal compiled
            if compiled is None:
                compiled = ComparisonFit(
                    parent, training, numerical, retained["parameters"], arm
                )
            return compiled

        def perform(stage, fn):
            ops[stage] = journal._operation(folder, binding, stage, limits[stage], fn)
            return first._value(ops[stage])

        def finish():
            terminal = _assemble(binding, ops)
            seal(folder / "result.json", terminal)
            return terminal

        nesting = perform(
            "nesting",
            lambda save: fit().verify_start(
                retained["training_nmse"], policy.verification_seconds
            ),
        )
        if not nesting.get("passed"):
            return finish()
        qualification = perform(
            "qualification",
            lambda save: fit().qualify(policy.qualification_seconds, save),
        )
        if not qualification.get("passed"):
            return finish()
        perform(
            "profiled_start_verification",
            lambda save: fit().verify(
                qualification["parameters"],
                qualification["training_nmse"],
                policy.verification_seconds,
                save,
            ),
        )
        if not first._verified(ops["profiled_start_verification"]):
            return finish()
        perform("search", lambda save: fit().search(policy, save))
        best = first._value(ops["search"], partial=True).get("best")
        if best:
            perform(
                "endpoint_verification",
                lambda save: fit().verify(
                    best["parameters"],
                    best["training_nmse"],
                    policy.verification_seconds,
                    save,
                ),
            )
        _, endpoint = _retained(ops)
        perform(
            "metrics", lambda save: fit().metrics(endpoint, policy.verification_seconds)
        )
        return finish()
