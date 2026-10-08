"""M21 matched affine recovery with retained fits and actionable assessments."""

from pathlib import Path
from time import monotonic, process_time
from typing import Literal

from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerical
from autoformalism.fitting.affine_profile import AffineProfile
from autoformalism.fitting.affine_propagation import (
    AffineRollouts,
    numerically_verified,
)
from autoformalism.fitting.fitting_assessment import assess
from autoformalism.schemas.base import StrictSchema

METHODS = ("joint", "profiled")


class RecoveryPolicy(StrictSchema):
    """Same per-stage ceilings for both methods; unused allowances are not spent."""

    warm_seconds: float = Field(default=180, ge=1, le=300)
    warm_calls: int = Field(default=400, ge=2, le=1000)
    trial_seconds: float = Field(default=60, ge=1, le=120)
    trial_calls: int = Field(default=150, ge=2, le=500)
    portfolio_size: int = Field(default=3, ge=0, le=5)
    continuation_seconds: float = Field(default=180, ge=1, le=300)
    continuation_calls: int = Field(default=400, ge=2, le=1000)
    check_seconds: float = Field(default=30, ge=1, le=60)
    replay_seconds: float = Field(default=180, ge=1, le=240)
    target_nmse: float = Field(default=1e-20, gt=0, le=1e-14)
    protocol: Literal["assessed-recovery-policy-1"] = "assessed-recovery-policy-1"


def run(
    problem: checks.TrainingProblem, policy: RecoveryPolicy, method: str, folder: Path
) -> dict:
    """Training-only decisions; separately sealed post-fit scoring lives outside."""
    if method not in METHODS:
        raise ValueError("unknown recovery method")
    problem_sha = public.content_sha256(problem.model_dump(mode="json"))
    identity = {
        "problem_sha256": problem_sha,
        "policy": policy.model_dump(mode="json"),
        "method": method,
    }
    finished = folder / "finished.json"
    if finished.exists():
        saved = read_seal(finished)
        if saved["identity"] != identity:
            raise ValueError("recovery identity differs")
        return saved
    begun, cpu = monotonic(), process_time()
    oracle = AffineRollouts(problem, checks.ConfidencePolicy())
    profile = AffineProfile(oracle) if method == "profiled" else None
    setup = folder / "setup"
    setup.mkdir(parents=True, exist_ok=True)
    seal(
        setup / f"{len(list(setup.glob('*.json'))):04d}.json",
        {"wall_seconds": monotonic() - begun, "cpu_seconds": process_time() - cpu},
    )
    operations, candidates, stops = [], [], []
    selected = None

    def operation(name, point, seconds, work):
        record = checks.operation(
            folder / name, identity | {"point": point}, seconds, work
        )
        operations.append({"operation": name, **record})
        return record.get("value")

    def inspect(name, parameters):
        nonlocal selected
        point = operation(
            name,
            parameters,
            policy.check_seconds,
            lambda end, save: numerical.check_point(oracle, parameters, end, save),
        )
        if point and not numerically_verified(point, 1e-12):
            tighter = operation(
                name + "-tight",
                parameters,
                policy.check_seconds,
                lambda end, save: numerical.check_point(
                    oracle, parameters, end, save, tight=True
                ),
            )
            if numerical.usable(tighter):
                point = tighter
        if point:
            point = point | {"origin": name}
            candidates.append(point)
            if numerical.improves(point, selected):
                selected = point
        return point

    def optimize(name, start, seconds, calls):
        result = operation(
            name,
            {"start": start, "calls": calls},
            seconds,
            lambda end, save: numerical.search(
                oracle,
                start,
                end,
                save,
                calls=calls,
                target=policy.target_nmse,
                profile=profile,
            ),
        )
        stops.append(result["stop_reason"] if result else "operation_unavailable")
        if result and result["best"]:
            return inspect(name + "-check", result["best"]["parameters"])
        return None

    def needs_search():
        return (
            not numerical.usable(selected)
            or max(selected["training_nmse"], selected["alternate_training_nmse"])
            > policy.target_nmse
        )

    before = inspect("incumbent", problem.incumbent)
    initial_assessment = assess(selected, None, candidates, stops, policy.target_nmse)
    seal(
        folder / "warm-decision.json",
        {"run": needs_search(), "assessment": initial_assessment},
    )
    if needs_search():
        optimize("warm", problem.incumbent, policy.warm_seconds, policy.warm_calls)
    after_warm = selected
    # Both methods receive exactly the same proposed portfolio. It is used only
    # after their warm search fails to reach the declared numerical target.
    starts = numerical.diverse_starts(
        oracle, problem, policy.portfolio_size, int(problem_sha[:8], 16)
    )
    portfolio = {
        "starts": starts,
        "run": needs_search(),
        "basis": "training_loss_above_target_after_warm",
        "random_seed": int(problem_sha[:8], 16),
        "source": "declared_domains_and_training_start_only",
    }
    seal(folder / "portfolio.json", portfolio)
    trials = []
    if portfolio["run"]:
        for i, start in enumerate(starts):
            # A warm failure freezes the complete short-trial portfolio. Trial
            # order never suppresses later starts after an early successful trial.
            point = optimize(
                f"trials/{i:02d}", start, policy.trial_seconds, policy.trial_calls
            )
            if numerical.usable(point):
                trials.append(point)
    winner = (
        min(trials, key=lambda p: max(p["training_nmse"], p["alternate_training_nmse"]))
        if trials
        else None
    )
    continue_trial = needs_search() and winner is not None
    seal(
        folder / "continuation-decision.json",
        {
            "run": continue_trial,
            "winner": winner,
            "basis": "best_verified_trial_even_if_incumbent_is_better",
        },
    )
    if continue_trial:
        optimize(
            "continuation",
            winner["parameters"],
            policy.continuation_seconds,
            policy.continuation_calls,
        )
    sensitivity = None
    if selected:

        def derivative(end, save):
            save({"calls": 1})
            _, j = oracle.evaluate(oracle.vector(selected["parameters"]), end)
            return checks.sensitivity(j, oracle.units, oracle.names)

        sensitivity = operation(
            "final-sensitivity",
            selected["parameters"],
            policy.check_seconds,
            derivative,
        )
    assessment = assess(selected, sensitivity, candidates, stops, policy.target_nmse)
    stage_costs = {}
    for record in operations:
        stage = record["operation"].split("/")[0]
        cost = stage_costs.setdefault(
            stage,
            {
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
                "budget_charge_seconds": 0.0,
                "calls": 0,
                "accounting_complete": True,
            },
        )
        for k in ("wall_seconds", "cpu_seconds", "budget_charge_seconds", "calls"):
            cost[k] += record[k] or 0
        cost["accounting_complete"] &= record["accounting_complete"]
    setups = [read_seal(p) for p in sorted(setup.glob("*.json"))]
    output = {
        "identity": identity,
        "status": "complete" if selected else "retained_unverified",
        "before": before,
        "after_warm": after_warm,
        "selected": selected,
        "retained_parameters": selected["parameters"]
        if selected
        else problem.incumbent,
        "original_incumbent": problem.incumbent,
        "assessment": assessment,
        "initial_assessment": initial_assessment,
        "portfolio_triggered": portfolio["run"],
        "trial_count": len(starts) if portfolio["run"] else 0,
        "continuation_run": continue_trial,
        "structure": oracle.audit | {"profile": profile.audit if profile else None},
        "operations": operations,
        "setup": setups,
        "stage_costs": stage_costs,
        "additional_wall_seconds": sum(s["wall_seconds"] for s in setups)
        + sum(o["wall_seconds"] or 0 for o in operations),
        "additional_cpu_seconds": sum(s["cpu_seconds"] for s in setups)
        + sum(o["cpu_seconds"] or 0 for o in operations),
        "additional_budget_charge_seconds": sum(s["wall_seconds"] for s in setups)
        + sum(o["budget_charge_seconds"] for o in operations),
        "rollout_calls_observed": sum(o["calls"] or 0 for o in operations),
        "cost_complete": all(o["accounting_complete"] for o in operations),
        "reference_values_used": False,
        "validation_used_for_selection": False,
    }
    seal(finished, output)
    return output
