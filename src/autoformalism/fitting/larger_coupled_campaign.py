"""M18: matched joint/profiled fitting on three- and six-state linear controls."""

from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import coupled_polishing, polishing_campaign, polishing_fit
from autoformalism.fitting import generic_recovery as common
from autoformalism.fitting import larger_coupled_inputs as controls
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.identifiable_campaign import SETTINGS
from autoformalism.fitting.profiled_coupled import ProfiledCoupled
from autoformalism.fitting.qualification import replay
from autoformalism.fitting.sensitivity_probe import SymbolicODE, symbolic_rollout
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

PROTOCOL = "phase-c-larger-coupled-1"
ARMS = ("rollout_only", "coupled_profiled_rollout")
PolishingPolicy = polishing_campaign.PolishingPolicy
bases = controls.bases


def prepare(root: Path, inputs: Path, policy: PolishingPolicy) -> dict:
    """Freeze before qualification/fitting; both methods share every start/budget."""
    return common.prepare(
        root, inputs, policy, protocol=PROTOCOL, arms=ARMS, base_factory=bases
    )


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    plan, data = common.verify(root, runtime=runtime, protocol=PROTOCOL)
    expected = [
        {"task_id": f"{key}_{arm}", "common": key, "arm": arm}
        for key in sorted(bases(data))
        for arm in ARMS
    ]
    if plan["tasks"] != expected:
        raise ValueError("larger coupled roster differs")
    return plan, data


def qualify(root: Path) -> dict:
    """Repeat reference replay and sampled sensitivity gates on the executing host.

    Reference values are evaluator-only and do not choose starts or alter budgets.
    A failed case blocks the entire array; completed gate records resume verbatim.
    """
    plan, data = verify(root)
    identity = public.content_sha256(plan)
    folder = root / "qualification"
    with public._lock(folder):
        rows = {}
        for name, case in data["cases"].items():
            path = folder / f"{name}.json"
            if path.exists():
                record = read_seal(path)
                if record["plan_sha256"] != identity:
                    raise ValueError("qualification identity differs")
                rows[name] = record
                continue
            request = PublicFitRequest.model_validate(
                data["commons"][f"{name}_s0"]["request"]
            )
            model, _, _ = public._lower(request)
            system = SymbolicODE(model)
            profile = ProfiledCoupled(system)
            train = PublicSplit.model_validate(case["training"])
            parameters = case["reference_parameters"]
            checked = replay(
                request,
                parameters,
                train,
                PublicSplit.model_validate(case["validation"]),
                120,
                journal=folder / f"{name}-replay.json",
            )
            jacobians = []
            for trajectory in public.unpack_split(train).trajectories:
                _, jac, _, _ = symbolic_rollout(
                    system,
                    trajectory,
                    np.array([parameters[p] for p in system.names]),
                    SETTINGS,
                    monotonic() + 30,
                    sensitivities=True,
                )
                jacobians.append(jac[:, 0, :])
            rank = controls.rank_audit(np.vstack(jacobians), list(system.names))
            record = {
                "plan_sha256": identity,
                "passed": bool(
                    checked["complete"]
                    and max(checked["metrics"].values()) <= 1e-10
                    and checked["maximum_solver_difference"] <= 1e-5
                    and rank["passed"]
                ),
                "reference_replay": checked,
                "identifiability": rank,
                "profile_structure": profile.audit,
                "direct_augmented_states": system.state_count * (1 + len(system.names)),
            }
            seal(path, record)
            rows[name] = record
        result = {
            "plan_sha256": identity,
            "passed": all(r["passed"] for r in rows.values()),
            "cases": rows,
        }
        seal(folder / "summary.json", result)
        if not result["passed"]:
            raise ValueError("larger coupled qualification failed; fitting blocked")
        return result


def fit(base: dict, arm: str, policy: dict, folder: Path) -> dict:
    return polishing_fit.fit(
        base,
        arm,
        policy,
        folder,
        stage_fitter=coupled_polishing.stage_fit,
        allowed_arms=ARMS,
    )


def evaluation_data(data: dict, task: dict) -> dict:
    name = data["commons"][task["common"]]["case"]
    return {"case": data["cases"][name], "case_name": name, "config": data["config"]}


def run_task(root: Path, index: int) -> dict:
    plan, data = verify(root)
    gate = read_seal(root / "qualification/summary.json")
    if not gate["passed"] or gate["plan_sha256"] != public.content_sha256(plan):
        raise ValueError("missing or failed qualification for this plan")
    row = common.run_task(
        root,
        index,
        protocol=PROTOCOL,
        fitter=fit,
        base_factory=bases,
        evaluation_data=evaluation_data,
    )
    return polishing_campaign.compare_first(
        root, row, plan, data, evaluation_data=evaluation_data
    )


def report(root: Path) -> dict:
    """Retain every start, first/final accuracy and overlapping worker timing."""
    plan, data = verify(root, runtime=False)
    result = common.report(root, protocol=PROTOCOL, arms=ARMS)
    qualification = root / "qualification/summary.json"
    result["qualification"] = (
        read_seal(qualification) if qualification.exists() else None
    )
    if result["qualification"] and result["qualification"][
        "plan_sha256"
    ] != public.content_sha256(plan):
        raise ValueError("qualification/report identity differs")
    for row in result["rows"]:
        entry = data["commons"][row["common"]]
        row.update(
            case=entry["case"],
            seed=entry["seed"],
            states=data["cases"][entry["case"]]["states"],
        )
        if row["status"] == "missing":
            continue
        folder = root / "results" / row["task_id"]
        backend = read_seal(folder / "backend.json")
        row["timing_operations"] = coupled_polishing.timing_records(folder)
        row["polishing"] = {
            key: backend.get(key)
            for key in (
                "polishing_attempted",
                "strict_prediction_certified",
                "actual_residual_calls",
                "retained_stage",
                "accounting_complete",
                "first_prediction",
            )
        }
        row["optimizer_stages"] = [
            {
                "name": s["name"],
                "optimizer": (s["backend"].get("rollout") or {}).get("value"),
            }
            for s in backend.get("stages", [])
        ]
        comparison = folder / "comparison.json"
        if comparison.exists():
            value = read_seal(comparison)
            if value["identity"] != {
                "identity": row["identity"],
                "backend_sha256": row["backend_sha256"],
            }:
                raise ValueError("larger coupled comparison identity differs")
            row["comparison"] = value
        else:
            row["comparison"] = {"status": "pending"}
            result["status"] = "incomplete"
    result["limitation"] = (
        "Noiseless correct-skeleton 3/6-state stable skew-coupled controls, separate "
        "from benchmarks. Known couplings anchor latent coordinates. Sampled rank "
        "is local, not global uniqueness or noise robustness. First/final vectors "
        "are scored only after training selection freezes; all starts are reported. "
        "Inclusive timing spans overlap. No nonlinear profiling or production change."
    )
    with public._lock(root / "report-lock"):
        public._write(root / "summary.json", result)
    return result
