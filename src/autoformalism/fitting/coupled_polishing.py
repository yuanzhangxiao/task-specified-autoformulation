"""M17: separate failed-start repeats and matched training-only polishing."""

from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import coupled_campaign as coupled
from autoformalism.fitting import generic_recovery as common
from autoformalism.fitting import polishing_campaign, polishing_fit, recovery_fit
from autoformalism.fitting import public_fitting as public

PROTOCOL = "phase-c-coupled-polishing-1"
ARMS = ("repeat_joint", "repeat_profiled", "polish_joint", "polish_profiled")
PolishingPolicy = polishing_campaign.PolishingPolicy
SOURCE_INPUT_SHA256 = "425186a426cc3ea90dd7fd1246a71851c58bdc13887669d3aa7daf8bdf423e22"
SOURCE = {
    "commit": "b0b1b129ab4f619edb723ece5404072d8b5405f4",
    "plan_sha256": "de1ea2b3de2d8e80a6361f48a9f1d97fe9fe276377f018371bc7159dc5279e78",
    "original_expected": 12,
    "original_completed": 10,
    "original_unavailable": 2,
    "accuracy_passed": {"rollout_only": 5, "coupled_profiled_rollout": 5},
    "denominator_per_method": 6,
    "repeats_replace_original_results": False,
}


def eligible(task: dict) -> bool:
    """Repeat only the two failed starts; polish all twelve original starts."""
    return task["arm"].startswith("polish_") or task["common"] == "coupled_fast_slow_s2"


def prepare(root: Path, inputs: Path, policy: PolishingPolicy) -> dict:
    data = read_seal(inputs)
    if public.content_sha256(data) != SOURCE_INPUT_SHA256:
        raise ValueError("M17 requires the exact M16 frozen inputs; do not regenerate")
    result = common.prepare(
        root,
        inputs,
        policy,
        protocol=PROTOCOL,
        arms=ARMS,
        base_factory=coupled.bases,
        task_filter=eligible,
    )
    seal(root / "original-results.json", SOURCE)
    return result


def verify(root: Path, *, runtime=True):
    plan, data = common.verify(root, runtime=runtime, protocol=PROTOCOL)
    if (
        plan["inputs_sha256"] != SOURCE_INPUT_SHA256
        or read_seal(root / "original-results.json") != SOURCE
    ):
        raise ValueError("original M16 provenance differs")
    expected = [
        {"task_id": f"{key}_{arm}", "common": key, "arm": arm}
        for key in sorted(coupled.bases(data))
        for arm in ARMS
        if eligible({"common": key, "arm": arm})
    ]
    if plan["tasks"] != expected:
        raise ValueError("M17 roster differs")
    return plan, data


def stage_fit(base: dict, arm: str, policy: dict, folder: Path) -> dict:
    """Same numerical algorithm and point caps, with diagnostic clocks enabled."""
    if arm not in coupled.ARMS:
        raise ValueError("unknown coupled method")
    return recovery_fit.fit(
        base,
        "rollout_only",
        policy,
        folder,
        rollout_mode="recovery_rollout" if arm == "rollout_only" else arm,
        diagnostic_timing=True,
    )


def fit(base: dict, arm: str, policy: dict, folder: Path) -> dict:
    if arm not in ARMS:
        raise ValueError("unknown M17 cohort/method")
    method = "rollout_only" if arm.endswith("_joint") else "coupled_profiled_rollout"
    if arm.startswith("repeat_"):
        settings = {key: policy[key] for key in common.RecoveryPolicy.model_fields}
        return stage_fit(base, method, settings, folder)
    return polishing_fit.fit(
        base,
        method,
        policy,
        folder,
        stage_fitter=stage_fit,
        allowed_arms=coupled.ARMS,
    )


def run_task(root: Path, index: int) -> dict:
    plan, data = verify(root)
    row = common.run_task(
        root,
        index,
        protocol=PROTOCOL,
        fitter=fit,
        base_factory=coupled.bases,
        evaluation_data=coupled.evaluation_data,
    )
    if row["arm"].startswith("polish_"):
        return polishing_campaign.compare_first(
            root,
            row,
            plan,
            data,
            evaluation_data=coupled.evaluation_data,
        )
    return row


def timing_records(folder: Path) -> list[dict]:
    """Keep missing timing explicit; verify terminal diagnostic artifact hashes."""
    records = []
    for path in sorted((folder / "fit").rglob("started.json")):
        receipt = path.with_name("process.json")
        entry = {"operation": str(path.parent.relative_to(folder)), "timing": None}
        if not receipt.exists():
            records.append(entry | {"status": "no_process_receipt"})
            continue
        process = read_seal(receipt)
        entry.update(
            status=process["status"], elapsed_seconds=process.get("elapsed_seconds")
        )
        digest = process.get("artifacts_sha256", {}).get("timing.json")
        if digest and process.get("termination_confirmed"):
            timing = public._read(path.with_name("timing.json"))
            if public.content_sha256(timing) != digest:
                raise ValueError("timing artifact differs")
            launch = public._read(path.with_name("launch.json"))["monotonic"]
            entry.update(
                timing=timing,
                launch_to_entry_seconds=timing["entry_monotonic"] - launch,
            )
        records.append(entry)
    return records


def report(root: Path) -> dict:
    _, data = verify(root, runtime=False)
    result = common.report(root, protocol=PROTOCOL, arms=ARMS)
    result["original_m16"] = SOURCE
    for row in result["rows"]:
        entry = data["commons"][row["common"]]
        row.update(
            case=entry["case"], seed=entry["seed"], cohort=row["arm"].split("_")[0]
        )
        if row["status"] == "missing":
            continue
        folder = root / "results" / row["task_id"]
        backend = read_seal(folder / "backend.json")
        row["timing_operations"] = timing_records(folder)
        row["polishing"] = {
            key: backend.get(key)
            for key in (
                "first_prediction",
                "polishing_attempted",
                "polishing_seconds",
                "strict_prediction_certified",
                "retained_stage",
                "actual_residual_calls",
            )
        }
        stages = backend.get("stages", [{"name": "repeat", "backend": backend}])
        row["optimizer_stages"] = [
            {
                "name": s["name"],
                "optimizer": (s["backend"].get("rollout") or {}).get("value"),
            }
            for s in stages
        ]
        if row["cohort"] == "polish":
            comparison = folder / "comparison.json"
            if not comparison.exists():
                row["comparison"] = {"status": "pending"}
                result["status"] = "incomplete"
            else:
                value = read_seal(comparison)
                if value["identity"] != {
                    "identity": row["identity"],
                    "backend_sha256": row["backend_sha256"],
                }:
                    raise ValueError("M17 comparison identity differs")
                row["comparison"] = value
    result["limitation"] = (
        "Two noiseless two-state correct-skeleton controls. Repeats are separate "
        "from the original 5/6 successes per method and cannot replace failures. "
        "Polishing shares 300 seconds/300 calls with initial fitting under the "
        "default policy; reference/validation are scored after final selection. "
        "Timing spans overlap; CPU/wall gaps alone cannot identify I/O or scheduling. "
        "Per-point caps retain their M16 method-specific behavior. "
        "No larger nonlinear-block qualification or production promotion."
    )
    with public._lock(root / "report-lock"):
        public._write(root / "summary.json", result)
    return result
