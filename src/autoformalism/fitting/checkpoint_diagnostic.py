"""M8 matched checkpoint policies and interruption metadata; no new optimizer."""

from pathlib import Path

from autoformalism.fitting import public_fitting as public

PROTOCOL = "phase-c-fitting-checkpoint-diagnostic-1"
BASE_ARMS = (
    "fixed_dense_collocation",
    "fixed_reduced_collocation",
    "fixed_shooting_lbfgs",
)
ARMS = {
    f"{arm}_{mode}": (arm, mode) for arm in BASE_ARMS for mode in ("legacy", "compact")
}


def recover_metadata(directory: Path) -> dict:
    """Keep decisions and completed histories without inventing in-flight counts."""
    result = {}
    for name in ("diagnostic", "stages", "mesh_audit"):
        if (directory / f"{name}.json").exists():
            result[name] = public._read(directory / f"{name}.json")
    diagnostic = result.get("diagnostic", {})
    if diagnostic:
        result["decision"] = diagnostic.get("decision")
        result["stages"] = diagnostic.get("stages", [])
    else:
        stages = result.setdefault("stages", [])
        recorded = {stage["level"] for stage in stages}
        for folder in sorted(directory.glob("mesh-*")):
            if not folder.is_dir() or not folder.name[5:].isdecimal():
                continue
            level = int(folder.name[5:])
            if level in recorded:
                continue
            stages.append(
                {
                    "level": level,
                    "partial_metadata": True,
                    "diagnostics": {
                        name: public._read(folder / f"{name}.json")
                        if (folder / f"{name}.json").exists()
                        else None
                        for name in (
                            "layout",
                            "derivative_profile",
                            "solver_progress",
                            "native",
                        )
                    },
                }
            )
    histories = {}
    for phase in (0, 1):
        path = directory / f"refinement-{phase}/training_history.json"
        if path.exists():
            histories[str(phase)] = len(public._read(path))
    result["completed_training_calls_by_phase"] = histories
    result["actual_residual_calls"] = None
    return result
