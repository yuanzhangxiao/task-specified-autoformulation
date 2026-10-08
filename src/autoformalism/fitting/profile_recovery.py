"""M20: verify saved profiles, then spend bounded, training-only recovery budgets."""

from pathlib import Path
from time import monotonic, process_time

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.affine_propagation import (
    AffineRollouts,
    numerically_verified,
    verify_outputs,
)
from autoformalism.schemas.base import StrictSchema


class SavedProfile(StrictSchema):
    """Exact frozen M19 location and optional surviving nuisance-fit candidate."""

    parameter: str
    value: float
    scale: float = Field(gt=0)
    offset: float
    in_domain: bool
    candidate: dict[str, float] | None = None
    progress_sha256: str | None = None


class RecoveryInput(checks.TrainingProblem):
    """Only request/training and saved vectors cross the numerical boundary."""

    anchor: dict[str, float]
    profiles: list[SavedProfile]
    alternatives: list[dict[str, float]]


class RecoveryPolicy(StrictSchema):
    """A new allowance, never a reset of historical profile budgets."""

    check_seconds: float = Field(default=30, ge=1, le=60)
    nuisance_seconds: float = Field(default=60, ge=1, le=180)
    nuisance_calls: int = Field(default=200, ge=2, le=600)
    maximum_profile_refits: int = Field(default=4, ge=0, le=8)
    rescue_seconds: float = Field(default=120, ge=1, le=300)
    rescue_calls: int = Field(default=300, ge=2, le=1000)
    replay_seconds: float = Field(default=180, ge=1, le=240)
    numerical: checks.ConfidencePolicy = Field(default_factory=checks.ConfidencePolicy)


def loss(point):
    """Selection uses the worse of two independent training evaluations."""
    return max(point["training_nmse"], point["alternate_training_nmse"])


def shortlist(profiles, selected, tolerance, maximum):
    """One point per parameter, closest unresolved losses first; no truth ranking."""
    ranked = []
    for i, item in enumerate(profiles):
        if not item["in_domain"]:
            continue
        check = item.get("checked")
        verified = numerically_verified(check, tolerance)
        difference = abs(item["value"] - selected["parameters"][item["parameter"]])
        witness = (
            verified
            and loss(check) <= loss(selected) + tolerance
            and difference >= 0.009 * item["scale"]
        )
        if witness:
            continue
        ranked.append((loss(check) - loss(selected) if verified else float("inf"), i))
    chosen, seen = [], set()
    for _, i in sorted(ranked):
        name = profiles[i]["parameter"]
        if name not in seen and len(chosen) < maximum:
            chosen.append(i)
            seen.add(name)
    return chosen


def run(problem: RecoveryInput, policy: RecoveryPolicy, folder: Path) -> dict:
    """Checkpoint every check/search, retain verified incumbents and charge all work."""
    identity = {
        "problem_sha256": public.content_sha256(problem.model_dump(mode="json")),
        "policy": policy.model_dump(mode="json"),
    }
    finished = folder / "finished.json"
    if finished.exists():
        saved = read_seal(finished)
        if saved["identity"] != identity:
            raise ValueError("profile recovery identity differs")
        return saved
    begun, cpu = monotonic(), process_time()
    oracle = AffineRollouts(problem, policy.numerical)
    anchor = oracle.vector(problem.anchor)
    for point in problem.profiles:
        if (
            point.parameter not in oracle.names
            or not np.isfinite([point.value, point.scale, point.offset]).all()
        ):
            raise ValueError("invalid frozen profile point")
        j = oracle.names.index(point.parameter)
        if point.in_domain != bool(oracle.lower[j] <= point.value <= oracle.upper[j]):
            raise ValueError("frozen profile domain differs")
        if point.candidate is not None:
            oracle.vector(point.candidate)
            if point.candidate[point.parameter] != point.value:
                raise ValueError("saved profile candidate changed fixed coordinate")
        if not np.isclose(
            point.value, anchor[j] + point.offset * point.scale, rtol=0, atol=1e-14
        ):
            raise ValueError("profile grid differs from its frozen anchor")
    setups = folder / "setup"
    setups.mkdir(parents=True, exist_ok=True)
    seal(
        setups / f"{len(list(setups.glob('*.json'))):04d}.json",
        {"wall_seconds": monotonic() - begun, "cpu_seconds": process_time() - cpu},
    )
    operations, candidates = [], []
    tolerance = policy.numerical.loss_tolerances[0]

    def operate(name, point, seconds, function):
        record = checks.operation(
            folder / name, identity | {"point": point}, seconds, function
        )
        operations.append({"operation": name, **record})
        return record.get("value")

    def inspect(name, parameters):
        checked = operate(
            name,
            parameters,
            policy.check_seconds,
            lambda end, save: verify_outputs(oracle, parameters, end, save),
        )
        if numerically_verified(checked, tolerance):
            candidates.append({"origin": name, **checked})
        return checked

    def best():
        return min(candidates, key=loss) if candidates else None

    before = inspect("incumbent", problem.incumbent)
    for i, parameters in enumerate(problem.alternatives):
        inspect(f"alternatives/{i:03d}", parameters)
    profiles = []
    for i, point in enumerate(problem.profiles):
        item = point.model_dump(mode="json") | {"optimizer_converged": False}
        if not point.in_domain:
            item.update(status="domain_limited", checked=None)
        elif point.candidate is None:
            item.update(status="no_saved_candidate", checked=None)
        else:
            checked = inspect(f"saved-profiles/{i:03d}", point.candidate)
            item.update(
                status="verified"
                if numerically_verified(checked, tolerance)
                else "unresolved",
                checked=checked,
            )
        profiles.append(item)
        public._write(folder / "profiles.json", profiles)
    after_saved = best()
    chosen = (
        shortlist(profiles, after_saved, tolerance, policy.maximum_profile_refits)
        if after_saved
        else []
    )
    seal(
        folder / "refit-selection.json",
        {
            "indices": chosen,
            "basis": "training-only closest unresolved; one per parameter",
        },
    )
    for i in chosen:
        item = profiles[i]
        fixed = {item["parameter"]: item["value"]}
        starts = [item["candidate"] or after_saved["parameters"], problem.start]
        optimized = operate(
            f"profile-refits/{i:03d}",
            {"fixed": fixed, "starts": starts},
            policy.nuisance_seconds,
            lambda end, save, fixed=fixed, starts=starts: checks.optimize(
                oracle,
                starts,
                policy.numerical,
                end,
                save,
                fixed=fixed,
                maximum=policy.nuisance_calls,
            ),
        )
        item["refit"] = optimized
        item["optimizer_converged"] = bool(
            optimized and optimized["all_attempts_converged"]
        )
        if optimized and optimized["best"]:
            checked = inspect(
                f"profile-refit-checks/{i:03d}", optimized["best"]["parameters"]
            )
            if numerically_verified(checked, tolerance) and (
                not numerically_verified(item.get("checked"), tolerance)
                or loss(checked) < loss(item["checked"])
            ):
                item.update(checked=checked, status="verified")
        public._write(folder / "profiles.json", profiles)
    after_refits = best()
    # The trigger is observable training loss, never retrospective coefficient error.
    rescue_starts = (
        [after_refits["parameters"], problem.start]
        if (after_refits and loss(after_refits) > policy.numerical.stop_nmse)
        else []
    )
    seal(
        folder / "rescue-selection.json",
        {"starts": rescue_starts, "basis": "training_loss_above_numerical_target"},
    )
    for i, start in enumerate(rescue_starts):
        optimized = operate(
            f"rescue/{i}",
            start,
            policy.rescue_seconds,
            lambda end, save, start=start: checks.optimize(
                oracle,
                [start],
                policy.numerical,
                end,
                save,
                maximum=policy.rescue_calls,
            ),
        )
        if optimized and optimized["best"]:
            inspect(f"rescue-checks/{i}", optimized["best"]["parameters"])
    selected = best()
    if selected:

        def derivative(end, save):
            save({"calls": 1})
            _, j = oracle.evaluate(oracle.vector(selected["parameters"]), end)
            return checks.sensitivity(j, oracle.units, oracle.names)

        sensitivity = operate(
            "final-sensitivity",
            selected["parameters"],
            policy.check_seconds,
            derivative,
        )
        if sensitivity:
            selected = selected | {"sensitivity": sensitivity}
    alternatives = [
        {
            "parameter": n,
            "value": point["parameters"][n],
            "scale": float(unit),
            "checked": point,
            "optimizer_converged": False,
        }
        for point in candidates
        if point["origin"] == "incumbent"
        or point["origin"].startswith(("alternatives/", "rescue-checks/"))
        for n, unit in zip(oracle.names, oracle.units, strict=True)
    ]
    confidence = checks.confidence_report(
        selected["parameters"] if selected else problem.incumbent,
        selected,
        profiles + alternatives,
        problem.anchor,
        policy.numerical,
        False,
    )
    confidence["local_exclusion_certified"] = False
    confidence["profile_search_scope"] = (
        "bounded subset; no completed full-grid minimization"
    )
    stage_costs = {}
    for o in operations:
        stage = o["operation"].split("/")[0]
        cost = stage_costs.setdefault(
            stage,
            {
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
                "calls": 0,
                "accounting_complete": True,
            },
        )
        for k in ("wall_seconds", "cpu_seconds", "calls"):
            cost[k] += o[k] or 0
        cost["accounting_complete"] &= o["accounting_complete"]
    setup = [read_seal(p) for p in sorted(setups.glob("*.json"))]
    result = {
        "identity": identity,
        "status": "complete" if selected else "no_verified_training_rollout",
        "before": before,
        "after_saved": after_saved,
        "after_refits": after_refits,
        "selected": selected,
        "profiles": profiles,
        "confidence": confidence,
        "profile_refit_indices": chosen,
        "rescue_starts": len(rescue_starts),
        "operations": operations,
        "stage_costs": stage_costs,
        "setup": setup,
        "additional_wall_seconds": sum(s["wall_seconds"] for s in setup)
        + sum(o["wall_seconds"] or 0 for o in operations),
        "additional_cpu_seconds": sum(s["cpu_seconds"] for s in setup)
        + sum(o["cpu_seconds"] or 0 for o in operations),
        "cost_complete": all(o["accounting_complete"] for o in operations),
        "rollout_calls_observed": sum(o["calls"] or 0 for o in operations),
        "reference_values_used": False,
        "validation_used_for_selection": False,
        "affine_certificate": oracle.audit,
    }
    seal(finished, result)
    return result
