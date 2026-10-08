"""M19 bounded reliability search and joint coefficient/initial uncertainty checks."""

from pathlib import Path
from time import monotonic, process_time

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import public_fitting as public


def run(problem: checks.TrainingProblem, policy: checks.ConfidencePolicy, folder: Path):
    """Freeze training-only adaptive choices; evaluate truth elsewhere."""
    identity = {
        "problem_sha256": public.content_sha256(problem.model_dump(mode="json")),
        "policy": policy.model_dump(mode="json"),
    }
    if (folder / "finished.json").exists():
        saved = read_seal(folder / "finished.json")
        if saved["identity"] != identity:
            raise ValueError("confidence fitting identity differs")
        return saved
    setup_wall, setup_cpu = monotonic(), process_time()
    oracle = checks.Rollouts(problem, policy)
    operations = []
    candidates = []

    def operate(name, parameters, seconds, function):
        record = checks.operation(
            folder / name, identity | {"parameters": parameters}, seconds, function
        )
        operations.append({"operation": name, **record})
        return record.get("value")

    setup = {
        "wall_seconds": monotonic() - setup_wall,
        "cpu_seconds": process_time() - setup_cpu,
    }
    # Setup repeats on task resume; retain every observed setup charge explicitly.
    setup_folder = folder / "setup"
    setup_folder.mkdir(parents=True, exist_ok=True)
    seal(setup_folder / f"{len(list(setup_folder.glob('*.json'))):04d}.json", setup)
    setups = [read_seal(path) for path in sorted(setup_folder.glob("*.json"))]
    setup = {key: sum(s[key] for s in setups) for key in setup}

    def inspect(name, parameters):
        return operate(
            name,
            parameters,
            policy.check_seconds,
            lambda deadline, progress: checks.inspect_point(
                oracle, parameters, deadline, progress
            ),
        )

    before = inspect("before", problem.incumbent)
    if before:
        candidates.append({"origin": "incumbent", **before})
    # The third start follows the weakest local joint direction, with a fallback
    # perturbation fixed by numerical parameter order if the check was unavailable.
    endpoint = oracle.vector(problem.incumbent)
    weak = (
        np.array([before["sensitivity"]["weak_direction"][n] for n in oracle.names])
        if before
        else np.ones(len(endpoint)) / np.sqrt(len(endpoint))
    )
    perturb = np.clip(endpoint + oracle.units * weak, oracle.lower, oracle.upper)
    starts = [problem.incumbent, problem.start, oracle.parameters(perturb)]
    reliability = []
    for i, start in enumerate(starts):
        result = operate(
            f"reliability-{i}",
            start,
            policy.restart_seconds,
            lambda deadline, progress, start=start: checks.optimize(
                oracle, [start], policy, deadline, progress
            ),
        )
        reliability.append(result)
        if result and result["best"]:
            checked = inspect(f"reliability-check-{i}", result["best"]["parameters"])
            if checked:
                candidates.append({"origin": f"reliability-{i}", **checked})
    if not candidates:
        final = {
            "identity": identity,
            "status": "no_complete_training_rollout",
            "operations": operations,
        }
        seal(folder / "finished.json", final)
        return final

    def select():
        return min(
            candidates,
            key=lambda p: max(p["training_nmse"], p["alternate_training_nmse"]),
        )

    reliability_best = select()
    anchor = reliability_best["parameters"]
    # Frozen grid is centered on the best independently checked reliability vector.
    # Profile minima are feasible upper bounds, never certified global minima.
    grid = list(checks.profile_grid(oracle, anchor, policy.offsets))
    seal(folder / "profile-grid.json", {"anchor": anchor, "points": grid})
    profiles = []
    for i, point in enumerate(grid):
        record = dict(point)
        if not point["in_domain"]:
            profiles.append(
                record | {"status": "domain_limited", "optimizer_converged": False}
            )
            continue
        fixed = {point["parameter"]: point["value"]}

        def profile(deadline, progress, fixed=fixed):
            # Reserve 40% for independent verification. Two nuisance starts share
            # the remainder and the call ceiling; neither uses reference values.
            allowance = max(0, deadline - monotonic())
            optimized = checks.optimize(
                oracle,
                [anchor, problem.start],
                policy,
                monotonic() + allowance * 0.6,
                progress,
                fixed=fixed,
                maximum=policy.profile_calls,
            )
            check = None
            if optimized["best"]:
                prior = optimized["calls"]
                check = checks.inspect_point(
                    oracle,
                    optimized["best"]["parameters"],
                    deadline,
                    lambda p: progress(p | {"calls": prior + p.get("calls", 0)}),
                )
            return {"optimizer": optimized, "checked": check}

        value = operate(
            f"profiles/{i:03d}",
            {"anchor": anchor, "point": point},
            policy.profile_seconds,
            profile,
        )
        if value:
            record.update(
                status="complete",
                checked=value["checked"],
                optimizer_converged=value["optimizer"]["all_attempts_converged"],
                optimizer=value["optimizer"],
            )
            if value["checked"]:
                candidates.append({"origin": f"profile-{i}", **value["checked"]})
        else:
            record.update(status="unavailable", optimizer_converged=False)
        profiles.append(record)
        public._write(folder / "profiles.json", profiles)
    profile_best = select()
    # Profile exploration is allowed to improve the model, transparently charged
    # as diagnostic-assisted recovery, never a zero-cost uncertainty assessment.
    if profile_best["training_nmse"] < reliability_best["training_nmse"]:
        point = profile_best["parameters"]
        result = operate(
            "profile-rescue",
            point,
            policy.restart_seconds,
            lambda deadline, progress: checks.optimize(
                oracle, [point], policy, deadline, progress
            ),
        )
        if result and result["best"]:
            checked = inspect("profile-rescue-check", result["best"]["parameters"])
            if checked:
                candidates.append({"origin": "profile-rescue", **checked})
    selected = select()
    reliable = all(r and r["all_attempts_converged"] for r in reliability)
    # Also count disagreements between freely optimized endpoints, not just the grid.
    alternatives = [
        {
            "parameter": n,
            "value": p["parameters"][n],
            "scale": scale,
            "checked": p,
            "optimizer_converged": reliable,
        }
        for p in candidates
        if p["origin"].startswith("reliability-")
        for n, scale in zip(oracle.names, oracle.units, strict=True)
    ]
    confidence = checks.confidence_report(
        selected["parameters"],
        selected,
        profiles + alternatives,
        anchor,
        policy,
        reliable,
    )
    stage_costs = {}
    for stage in ("before", "reliability", "profiles", "profile-rescue"):
        selected_operations = [
            o for o in operations if o["operation"].startswith(stage)
        ]
        stage_costs[stage] = {
            "wall_seconds_observed": sum(
                o["wall_seconds"] or 0 for o in selected_operations
            ),
            "cpu_seconds_observed": sum(
                o["cpu_seconds"] or 0 for o in selected_operations
            ),
            "rollout_calls_observed": sum(o["calls"] or 0 for o in selected_operations),
            "accounting_complete": all(
                o["accounting_complete"] for o in selected_operations
            ),
        }
    final = {
        "identity": identity,
        "status": "complete",
        "stage_costs": stage_costs,
        "before": before,
        "after_reliability": reliability_best,
        "after_profiles": profile_best,
        "selected": selected,
        "reliability": reliability,
        "profiles": profiles,
        "confidence": confidence,
        "operations": operations,
        "setup": setup,
        "cost_complete": all(o["accounting_complete"] for o in operations),
        "additional_wall_seconds": setup["wall_seconds"]
        + sum(o["wall_seconds"] or 0 for o in operations),
        "additional_cpu_seconds": setup["cpu_seconds"]
        + sum(o["cpu_seconds"] or 0 for o in operations),
        "rollout_calls_observed": sum(o["calls"] or 0 for o in operations),
        "reference_values_used": False,
        "validation_used_for_selection": False,
    }
    seal(folder / "finished.json", final)
    return final
