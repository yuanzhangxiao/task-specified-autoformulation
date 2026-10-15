"""Small, sealed, training-only controls for the M37 interaction-feedback view.

These controls qualify packet translation, not scientific discovery. Evaluator
expectations are stored separately and never enter fitting or packet assembly.
"""

from pathlib import Path
from time import monotonic, process_time
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import interaction_feedback as feedback
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import public_rollout as bridge
from autoformalism.fitting import staged_feedback as staged
from autoformalism.fitting import staged_feedback_fixtures as fixtures
from autoformalism.fitting.rollout_adapter import FitResult
from autoformalism.schemas.base import StrictSchema

PROTOCOL = "phase-c-interaction-feedback-qualification-1"
CASES = (
    "initializer_path",
    "stationary_poor_retained",
    "piecewise_kink",
    "piecewise_bound_limited",
)
Case = Literal[
    "initializer_path",
    "stationary_poor_retained",
    "piecewise_kink",
    "piecewise_bound_limited",
]


class Control(StrictSchema):
    """Only ordinary public training data and a frozen fitting policy."""

    case_id: Case
    inputs: bridge.Inputs
    policy: bridge.Policy


class Inputs(StrictSchema):
    """A complete, ordered synthetic roster with no evaluator channel."""

    controls: list[Control]

    @model_validator(mode="after")
    def exact_roster(self):
        if tuple(row.case_id for row in self.controls) != CASES:
            raise ValueError("interaction qualification control roster differs")
        return self


class Expectation(StrictSchema):
    """Evaluator-only observations kept outside every computational input."""

    case_id: Case
    expected_accurate: bool
    expected_diagnosis: str
    expected_derivative_support: str


class Expectations(StrictSchema):
    controls: list[Expectation]

    @model_validator(mode="after")
    def exact_roster(self):
        if tuple(row.case_id for row in self.controls) != CASES:
            raise ValueError("interaction qualification expectation roster differs")
        return self


class Policy(StrictSchema):
    """Short diagnostic allowances; the existing fixture fitter stays fixed."""

    diagnostics: staged.Policy = Field(
        default_factory=lambda: staged.Policy(
            primary_seconds=5,
            alternate_seconds=5,
            sensitivity_seconds=5,
            perturbation_seconds=2,
            perturbation_total_seconds=10,
        )
    )
    interaction: feedback.Policy = Field(default_factory=feedback.Policy)


class Plan(StrictSchema):
    """Bind every task to exact inputs, expectations, code and numerical runtime."""

    protocol: Literal["phase-c-interaction-feedback-qualification-1"] = PROTOCOL
    tasks: list[Case]
    inputs_sha256: str
    expectations_sha256: str
    source_sha256: str
    runtime: dict
    policy: Policy
    live_llm_calls: Literal[0] = 0
    test_data_opened: Literal[False] = False
    validation_used: Literal[False] = False
    benchmark_claim: Literal[False] = False

    @model_validator(mode="after")
    def exact_roster(self):
        if tuple(self.tasks) != CASES:
            raise ValueError("interaction qualification plan roster differs")
        return self


class Identity(StrictSchema):
    plan_sha256: str
    case_id: Case


class Failure(StrictSchema):
    stage: Literal["fixture", "diagnostics", "export"]
    error_type: str
    message: str


class AdapterCost(StrictSchema):
    """Measured local adapter overhead, separate from fitting and diagnostics."""

    export_calls: Literal[0, 1] = 0
    export_wall_seconds: float = Field(default=0, ge=0)
    export_cpu_seconds: float = Field(default=0, ge=0)
    verification_calls: Literal[0, 1] = 0
    verification_wall_seconds: float = Field(default=0, ge=0)
    verification_cpu_seconds: float = Field(default=0, ge=0)
    optimization_calls: Literal[0] = 0
    rollout_calls: Literal[0] = 0
    llm_calls: Literal[0] = 0


class Result(StrictSchema):
    """Terminal per-control checkpoint; failed work remains in the denominator."""

    identity: Identity
    status: Literal["complete", "failed"]
    diagnostic_inputs_sha256: str | None = None
    packet_sha256: str | None = None
    failure: Failure | None = None
    adapter_cost: AdapterCost

    @model_validator(mode="after")
    def consistent(self):
        if self.status == "complete":
            if (
                self.failure
                or not self.diagnostic_inputs_sha256
                or not self.packet_sha256
            ):
                raise ValueError(
                    "complete interaction control lacks artifact identities"
                )
        elif (
            self.failure is None
            or self.diagnostic_inputs_sha256 is not None
            or self.packet_sha256 is not None
        ):
            raise ValueError("failed interaction control has inconsistent evidence")
        return self


def _controls() -> tuple[Inputs, Expectations]:
    """Construct independent fixture inputs and evaluator-only expectations."""
    controls, expected = [], []
    for case_id in CASES:
        inputs, policy, labels = fixtures.make(case_id)
        controls.append(Control(case_id=case_id, inputs=inputs, policy=policy))
        expected.append(
            Expectation.model_validate(
                {name: labels[name] for name in Expectation.model_fields}
            )
        )
    return Inputs(controls=controls), Expectations(controls=expected)


def prepare(root: Path, policy: Policy | None = None) -> dict:
    """Freeze the four synthetic tasks without performing fitting or diagnosis."""
    inputs, expectations = _controls()
    plan = Plan(
        tasks=list(CASES),
        inputs_sha256=public.content_sha256(inputs.model_dump(mode="json")),
        expectations_sha256=public.content_sha256(expectations.model_dump(mode="json")),
        source_sha256=public._source_identity(),
        runtime=public._runtime(),
        policy=policy or Policy(),
    )
    with public._lock(root):
        seal(root / "inputs.json", inputs.model_dump(mode="json"))
        seal(root / "evaluator_expectations.json", expectations.model_dump(mode="json"))
        seal(root / "plan.json", plan.model_dump(mode="json"))
    return {
        "plan_sha256": public.content_sha256(plan.model_dump(mode="json")),
        "tasks": 4,
    }


def verify(root: Path, *, runtime: bool = True) -> tuple[Plan, Inputs, Expectations]:
    """Validate frozen schemas and digests, requiring current code on execution."""
    plan = Plan.model_validate(read_seal(root / "plan.json"))
    inputs = Inputs.model_validate(read_seal(root / "inputs.json"))
    expectations = Expectations.model_validate(
        read_seal(root / "evaluator_expectations.json")
    )
    if plan.inputs_sha256 != public.content_sha256(
        inputs.model_dump(mode="json")
    ) or plan.expectations_sha256 != public.content_sha256(
        expectations.model_dump(mode="json")
    ):
        raise ValueError("interaction qualification input/expectation identity differs")
    if runtime:
        if (
            plan.source_sha256 != public._source_identity()
            or plan.runtime != public._runtime()
        ):
            raise ValueError("interaction qualification source/runtime differs")
        current, labels = _controls()
        if current != inputs or labels != expectations:
            raise ValueError("interaction qualification fixture definition differs")
    return plan, inputs, expectations


def _identity(plan: Plan, case_id: Case) -> Identity:
    return Identity(
        plan_sha256=public.content_sha256(plan.model_dump(mode="json")), case_id=case_id
    )


def _fit_identity(plan: Plan, control: Control) -> dict:
    return {
        "inputs_sha256": public.content_sha256(control.inputs.model_dump(mode="json")),
        "policy": control.policy.model_dump(mode="json"),
        "source_sha256": plan.source_sha256,
        "runtime": plan.runtime,
    }


def completed(folder: Path, plan: Plan, control: Control) -> Result:
    """Verify the actual bridge, diagnostics and exported packet before reuse."""
    result = Result.model_validate(read_seal(folder / "result.json"))
    if result.identity != _identity(plan, control.case_id):
        raise ValueError("interaction qualification task identity differs")
    if result.status == "failed":
        return result
    fitted = bridge.completed(folder / "fixture", _fit_identity(plan, control))
    values = read_seal(folder / "diagnostic_inputs.json")
    if fitted.fit is None or values != {
        "problem": fitted.preparation["value"]["problem"],
        "fit": fitted.fit.model_dump(mode="json"),
        "backend": read_seal(folder / "fixture/fit/backend.json"),
    }:
        raise ValueError("interaction qualification fitted inputs differ")
    if result.diagnostic_inputs_sha256 != public.content_sha256(values):
        raise ValueError("interaction qualification diagnostic input identity differs")
    context = read_seal(folder / "diagnostics/context.json")
    if context != values | {"policy": plan.policy.diagnostics.model_dump(mode="json")}:
        raise ValueError("interaction qualification diagnostic context differs")
    packet = feedback.completed(folder / "diagnostics", folder / "interaction")
    if (
        packet.policy != plan.policy.interaction
        or packet.identity.adapter_source_sha256 != plan.source_sha256
    ):
        raise ValueError("interaction qualification adapter policy/source differs")
    if result.packet_sha256 != public.content_sha256(packet.model_dump(mode="json")):
        raise ValueError("interaction qualification packet identity differs")
    return result


def run_task(root: Path, index: int) -> dict:
    """Run one small control, preserving terminal failures and interrupted stages."""
    plan, inputs, _ = verify(root)
    if not 0 <= index < len(CASES):
        raise ValueError("unknown interaction qualification task index")
    control = inputs.controls[index]
    folder = root / "results" / control.case_id
    identity = _identity(plan, control.case_id)
    with public._lock(folder):
        if (folder / "result.json").exists():
            return completed(folder, plan, control).model_dump(mode="json")
        stage = "fixture"
        cost = AdapterCost().model_dump(mode="json")
        try:
            values = fixtures.fit_fixture(control.case_id, folder / "fixture")
            bridge.completed(folder / "fixture", _fit_identity(plan, control))
            seal(folder / "diagnostic_inputs.json", values)
            stage = "diagnostics"
            staged.run(
                staged.training_problem(values["problem"]),
                FitResult.model_validate(values["fit"]),
                values["backend"],
                folder / "diagnostics",
                plan.policy.diagnostics,
            )
            stage = "export"
            start, cpu_start = monotonic(), process_time()
            cost["export_calls"] = 1
            try:
                packet = feedback.export_packet(
                    folder / "diagnostics",
                    folder / "interaction",
                    plan.policy.interaction,
                )
            finally:
                cost["export_wall_seconds"] = monotonic() - start
                cost["export_cpu_seconds"] = process_time() - cpu_start
            start, cpu_start = monotonic(), process_time()
            cost["verification_calls"] = 1
            try:
                if packet != feedback.completed(
                    folder / "diagnostics", folder / "interaction"
                ):
                    raise ValueError(
                        "interaction qualification exported packet differs"
                    )
            finally:
                cost["verification_wall_seconds"] = monotonic() - start
                cost["verification_cpu_seconds"] = process_time() - cpu_start
            result = Result(
                identity=identity,
                status="complete",
                diagnostic_inputs_sha256=public.content_sha256(values),
                packet_sha256=public.content_sha256(packet.model_dump(mode="json")),
                adapter_cost=AdapterCost.model_validate(cost),
            )
        except (ValueError, RuntimeError, TimeoutError, ArithmeticError) as error:
            result = Result(
                identity=identity,
                status="failed",
                failure=Failure(
                    stage=stage, error_type=type(error).__name__, message=str(error)
                ),
                adapter_cost=AdapterCost.model_validate(cost),
            )
        seal(folder / "result.json", result.model_dump(mode="json"))
        return completed(folder, plan, control).model_dump(mode="json")


def _observation(folder: Path, expectation: Expectation) -> dict:
    """Check declared controls only after both numerical and packet work ends."""
    source = staged.completed(folder / "diagnostics")
    packet = feedback.completed(folder / "diagnostics", folder / "interaction")
    piecewise = source.diagnostics.piecewise_assessment
    derivative = piecewise.status if piecewise else source.diagnostics.derivative_scope
    checks = {
        "expected_diagnosis": source.diagnosis == expectation.expected_diagnosis,
        "expected_accuracy": (source.fit_quality == "numerical_target_reached")
        == expectation.expected_accurate,
        "expected_derivative_support": derivative
        == expectation.expected_derivative_support,
        "verified_training_output": source.diagnostics.output_status == "verified",
        "packet_exported": packet.status == "ready_for_advisory_review"
        and packet.response.status == "available",
        "numerical_evidence_preserved": packet.feedback.diagnosis == source.diagnosis
        and packet.feedback.parameter_information == source.parameter_information
        and packet.feedback.derivative_scope == source.diagnostics.derivative_scope
        and packet.feedback.output_status == source.diagnostics.output_status,
        "advisory_only": packet.edit_contract_status == "requires_constructor_binding"
        and not packet.parameters_changed
        and packet.new_optimization_calls
        == packet.new_rollout_calls
        == packet.live_llm_calls
        == 0,
        "scientific_boundary": source.specification_verdict == "not_assessed"
        and source.automatic_specification_revision_allowed is False,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "diagnosis": source.diagnosis,
        "derivative_support": derivative,
        "packet_status": packet.status,
    }


def _cost_evidence(folder: Path, plan: Plan, control: Control) -> dict:
    """Retain independently verified costs even when a later task stage fails.

    Partial stage seals identify the available evidence, but do not establish a
    complete cost ledger. Missing totals remain unavailable rather than zero.
    """
    fit_cost, diagnostic_cost, fitted = None, None, None
    fit_artifacts = [
        folder / "fixture" / name
        for name in (
            "identity.json",
            "preparation/result.json",
            "result.json",
            "fit/result.json",
            "fit/backend.json",
        )
    ]
    diagnostic_artifacts = [
        folder / "diagnostics" / name
        for name in ("identity.json", "context.json", "stages.json", "result.json")
    ]
    for name in ("started.json", "record.json", "result.json"):
        diagnostic_artifacts.extend((folder / "diagnostics/stages").rglob(name))

    def hashes(paths: list[Path]) -> dict[str, str]:
        return {
            str(path.relative_to(folder)): public.content_sha256(read_seal(path))
            for path in sorted(paths)
            if path.is_file()
        }

    provenance = {
        "fit": hashes(fit_artifacts),
        "diagnostic": hashes(diagnostic_artifacts),
    }
    if (folder / "fixture/result.json").exists():
        fitted = bridge.completed(folder / "fixture", _fit_identity(plan, control))
        if fitted.fit is not None:
            fit_cost = fitted.fit.cost.model_dump(mode="json")
    if (folder / "diagnostics/result.json").exists():
        if fitted is None or fitted.fit is None:
            raise ValueError("diagnostic cost lacks a verified fixture fit")
        context = read_seal(folder / "diagnostics/context.json")
        expected = {
            "problem": fitted.preparation["value"]["problem"],
            "fit": fitted.fit.model_dump(mode="json"),
            "backend": read_seal(folder / "fixture/fit/backend.json"),
            "policy": plan.policy.diagnostics.model_dump(mode="json"),
        }
        if context != expected:
            raise ValueError(
                "interaction qualification diagnostic cost context differs"
            )
        source = staged.completed(folder / "diagnostics")
        diagnostic_cost = source.cost.model_dump(mode="json")
    result = {}
    for name, cost in (("fit", fit_cost), ("diagnostic", diagnostic_cost)):
        result[name + "_cost"] = cost
        result[name + "_cost_accounting"] = {
            "status": "verified" if cost is not None else "unavailable",
            "accounting_complete": bool(cost and cost["accounting_complete"]),
            "reason": "verified_completed_artifact"
            if cost is not None
            else "complete_cost_ledger_unavailable",
            "artifact_sha256": provenance[name],
            "artifact_verification": "completed_source_verifier"
            if cost is not None
            else "seals_only_not_complete_accounting",
            "unrecorded_work_assumed_zero": False,
        }
    return result


def report(root: Path) -> dict:
    """Inspect every frozen task without fitting, diagnostics or export work."""
    plan, inputs, expectations = verify(root, runtime=False)
    rows = []
    for control, expected in zip(inputs.controls, expectations.controls, strict=True):
        folder = root / "results" / control.case_id
        costs = _cost_evidence(folder, plan, control)
        if not (folder / "result.json").exists():
            rows.append({"case_id": control.case_id, "status": "pending", **costs})
            continue
        result = completed(folder, plan, control)
        observation = (
            {"status": "failed", "failure": result.failure.model_dump(mode="json")}
            if result.failure
            else _observation(folder, expected)
        )
        rows.append(
            {
                "case_id": control.case_id,
                **observation,
                **costs,
                "adapter_cost": result.adapter_cost.model_dump(mode="json"),
            }
        )
    counts = {
        name: sum(row["status"] == name for row in rows)
        for name in ("passed", "failed", "pending")
    }
    return {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan.model_dump(mode="json")),
        "qualification_status": "failed"
        if counts["failed"]
        else "pending"
        if counts["pending"]
        else "passed",
        "expected": len(CASES),
        "recorded": len(CASES) - counts["pending"],
        **counts,
        "rows": rows,
        "live_llm_calls": 0,
        "test_data_opened": False,
        "validation_used": False,
        "benchmark_claim": False,
    }


def qualify(root: Path, index: int | None = None) -> dict:
    """Prepare or resume the fixed local control roster, then report all tasks."""
    if index is not None and not 0 <= index < len(CASES):
        raise ValueError("unknown interaction qualification task index")
    if not (root / "plan.json").exists():
        prepare(root)
    for task_index in range(len(CASES)) if index is None else (index,):
        run_task(root, task_index)
    return report(root)
