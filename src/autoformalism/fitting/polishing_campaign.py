"""M15: paired staged polishing with frozen before/after post-fit evaluation."""

from pathlib import Path

from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import generic_recovery as common
from autoformalism.fitting import polishing_fit
from autoformalism.fitting import public_fitting as public

PROTOCOL = "phase-c-fitting-polishing-1"
ARMS = ("rollout_only", "profiled_rollout")


class PolishingPolicy(common.RecoveryPolicy):
    """Both phases share inherited budgets; precision depends only on training."""

    polish_training_nmse: float = Field(default=1e-8, gt=0, lt=1)
    polish_trajectory_nmse: float = Field(default=1e-7, gt=0, lt=1)

    @model_validator(mode="after")
    def tighter(self):
        if (
            self.polish_training_nmse >= self.training_nmse
            or self.polish_trajectory_nmse >= self.trajectory_nmse
        ):
            raise ValueError(
                "polishing targets must be stricter than prediction targets"
            )
        return self


def prepare(root: Path, inputs: Path, policy: PolishingPolicy) -> dict:
    return common.prepare(root, inputs, policy, protocol=PROTOCOL, arms=ARMS)


def verify(root: Path, *, runtime=True):
    return common.verify(root, runtime=runtime, protocol=PROTOCOL)


def run_task(root: Path, index: int) -> dict:
    """Evaluate the first checkpoint only after the final selection is sealed."""
    row = common.run_task(root, index, protocol=PROTOCOL, fitter=polishing_fit.fit)
    plan, data = verify(root)
    folder = root / "results" / row["task_id"]
    identity = {"identity": row["identity"], "backend_sha256": row["backend_sha256"]}
    with public._lock(folder):
        path = folder / "comparison.json"
        if path.exists():
            saved = read_seal(path)
            if saved["identity"] != identity:
                raise ValueError("polishing comparison identity differs")
            return row | {"comparison": saved}
        backend = read_seal(folder / "backend.json")
        first = backend.get("first_prediction")
        comparison = {
            "identity": identity,
            "status": "no_certified_prediction",
            "first_evaluation": None,
        }
        if row["evaluation_blocked_by_cleanup"]:
            comparison["status"] = "cleanup_unconfirmed"
        elif first:
            parameters = first["selected"]["parameters"]
            if parameters == row["selected"]["parameters"]:
                evaluation = row["evaluation"]
                comparison["reused_final_evaluation"] = True
            else:
                destination = folder / "first-evaluation"
                seal(
                    destination / "identity.json",
                    {**identity, "parameters": parameters},
                )
                evaluation = common.replay._evaluation(
                    destination,
                    data,
                    data["commons"][row["common"]],
                    parameters,
                    plan["policy"]["replay_seconds"],
                )
                comparison["reused_final_evaluation"] = False
            comparison.update(first_evaluation=evaluation, status=evaluation["status"])
        seal(path, comparison)
    return row | {"comparison": comparison}


def report(root: Path) -> dict:
    """Keep every original start and expose prediction versus polishing outcomes."""
    result = common.report(root, protocol=PROTOCOL, arms=ARMS)
    for row in result["rows"]:
        folder = root / "results" / row["task_id"]
        if row["status"] == "missing":
            continue
        backend = read_seal(folder / "backend.json")
        row["polishing"] = {
            k: backend.get(k)
            for k in (
                "first_prediction",
                "polishing_attempted",
                "polishing_seconds",
                "strict_prediction_certified",
                "retained_stage",
                "actual_residual_calls",
                "accounting_complete",
            )
        }
        row["optimizer_stages"] = [
            {
                "name": s["name"],
                "policy": s["policy"],
                "fit_seconds": s["backend"].get("fit_seconds"),
                "optimizer": (s["backend"].get("rollout") or {}).get("value"),
            }
            for s in backend.get("stages", [])
        ]
        comparison_path = folder / "comparison.json"
        if comparison_path.exists():
            comparison = read_seal(comparison_path)
            if comparison["identity"] != {
                "identity": row["identity"],
                "backend_sha256": row["backend_sha256"],
            }:
                raise ValueError("reported polishing comparison identity differs")
            row["comparison"] = comparison
        else:
            row["comparison"] = {"status": "pending"}
            result["status"] = "incomplete"
    result["polishing_groups"] = {}
    for arm in ARMS:
        rows = [r for r in result["rows"] if r["arm"] == arm]
        result["polishing_groups"][arm] = {
            "expected": len(rows),
            "first_prediction_certified": sum(
                bool(r.get("polishing", {}).get("first_prediction")) for r in rows
            ),
            **{
                k: sum(bool(r.get("polishing", {}).get(k)) for r in rows)
                for k in ("polishing_attempted", "strict_prediction_certified")
            },
            "first_coefficients_recovered": sum(
                bool(
                    (r.get("comparison", {}).get("first_evaluation") or {}).get(
                        "coefficients_recovered"
                    )
                )
                for r in rows
            ),
            "final_coefficients_recovered": result["groups"][arm][
                "coefficients_recovered"
            ],
        }
    result["limitation"] += (
        " One parameter warm-start after independent training certification; "
        "optimizer state restarts, total fitting seconds/calls do not. First and "
        "final vectors are evaluated only after final selection. Before/after "
        "replays have separate allowances, outside fitting. No coupled-state "
        "coefficient profiling or production default changes."
    )
    with public._lock(root / "report-lock"):
        public._write(root / "summary.json", result)
    return result
