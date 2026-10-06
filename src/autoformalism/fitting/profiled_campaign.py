"""M14 paired joint versus conditionally linear output fitting, all original starts."""

from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import generic_recovery as common
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_fit

PROTOCOL = "phase-c-profiled-output-1"
ARMS = ("rollout_only", "profiled_rollout")
ProfiledPolicy = common.RecoveryPolicy


def fit(base: dict, arm: str, policy: dict, folder: Path) -> dict:
    """Change only the rollout optimizer; preserve the budget/check/replay envelope."""
    if arm not in ARMS:
        raise ValueError("unknown profiled fitting arm")
    return recovery_fit.fit(
        base,
        "rollout_only",
        policy,
        folder,
        rollout_mode="recovery_rollout"
        if arm == "rollout_only"
        else "profiled_rollout",
    )


def prepare(root: Path, inputs: Path, policy: ProfiledPolicy) -> dict:
    return common.prepare(root, inputs, policy, protocol=PROTOCOL, arms=ARMS)


def verify(root: Path, *, runtime=True):
    return common.verify(root, runtime=runtime, protocol=PROTOCOL)


def run_task(root: Path, index: int) -> dict:
    return common.run_task(root, index, protocol=PROTOCOL, fitter=fit)


def report(root: Path) -> dict:
    result = common.report(root, protocol=PROTOCOL, arms=ARMS)
    for row in result["rows"]:
        path = root / "results" / row["task_id"] / "backend.json"
        if path.exists() and row["status"] != "missing":
            backend = read_seal(path)
            rollout = backend.get("rollout") or {}
            value = rollout.get("value") or {}
            row["optimizer"] = {
                "process_status": (rollout.get("process") or {}).get("status"),
                **{
                    k: value.get(k)
                    for k in (
                        "stop_reason",
                        "actual_residual_calls",
                        "profiled_output",
                        "selected_inner",
                        "accounting",
                    )
                },
            }
    result["limitation"] += (
        " Terminal identity-observed linear output only. Rank-deficient gain "
        "designs are unavailable; bound changes can make the residual Jacobian "
        "nonsmooth. Inner linear optimality does not certify outer/global recovery."
    )
    with public._lock(root / "report-lock"):
        public._write(root / "summary.json", result)
    return result
