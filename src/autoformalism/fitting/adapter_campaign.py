"""M28: paired common-adapter qualification plus one fixed-endpoint diagnostic."""

from collections import Counter
from pathlib import Path
from statistics import median
from typing import Literal

from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import broad_scaling_campaign as previous
from autoformalism.fitting import broad_scaling_inputs as source
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import rollout_adapter as adapter
from autoformalism.fitting import verified_replay
from autoformalism.fitting.transcription_campaign import coefficient_metrics
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

PROTOCOL = "phase-c-rollout-adapter-1"
INPUT_PROTOCOL = "phase-c-rollout-adapter-inputs-1"
STRATEGIES = ("guarded_only", "guarded_portfolio")


class Policy(StrictSchema):
    """Common search and evaluator ceilings are frozen for both adapter arms."""

    protocol: Literal["phase-c-rollout-adapter-1"] = PROTOCOL
    search: previous.Policy = Field(default_factory=previous.Policy)
    replay: verified_replay.ReplayPolicy = Field(
        default_factory=verified_replay.ReplayPolicy
    )


def export_inputs(m27: Path, output: Path) -> dict:
    """Copy immutable arrays/starts; the old endpoint goes only to the evaluator."""
    plan, development = previous.verify(m27, runtime=False)
    task = next(
        t for t in plan["tasks"] if t["task_id"] == "basin_coupled_s0__guarded_relative"
    )
    previous.incumbent_campaign._completed(
        m27 / "results" / task["task_id"],
        {"plan_sha256": public.content_sha256(plan), "task": task},
    )
    backend = read_seal(m27 / "results/basin_coupled_s0__guarded_relative/backend.json")
    if backend["identity"]["starts"] != plan["starts"]["basin_coupled_s0"]:
        raise ValueError("diagnostic endpoint source identity differs")
    value = {
        "protocol": INPUT_PROTOCOL,
        "source_plan_sha256": public.content_sha256(plan),
        "development": development,
        "starts": plan["starts"],
        "fixed_endpoint": {
            "common": "basin_coupled_s0",
            "parameters": backend["retained_parameters"],
            "source_backend_sha256": public.content_sha256(backend),
        },
        "test_data_opened": False,
    }
    seal(output, value)
    return {"inputs_sha256": public.content_sha256(value), "tasks": 31}


def roster(data: dict) -> list[dict]:
    development = data["development"]
    if data["protocol"] != INPUT_PROTOCOL or data["test_data_opened"] is not False:
        raise ValueError("adapter requires sealed development inputs")
    bases = source.bases(development)
    if set(data["starts"]) != set(bases):
        raise ValueError("adapter restart roster differs")
    for name, problem in bases.items():
        if len(data["starts"][name]) != 4 or data["starts"][name][0] != problem.start:
            raise ValueError("adapter original/restart identity differs")
    tasks = [
        {
            "task_id": f"{name}__{strategy}",
            "kind": "fit",
            "common": name,
            "strategy": strategy,
            "case": development["commons"][name]["case"],
            "seed": development["commons"][name]["seed"],
        }
        for name in sorted(bases)
        for strategy in STRATEGIES
    ]
    if data["fixed_endpoint"]["common"] != "basin_coupled_s0":
        raise ValueError("expected the saved basin verification endpoint")
    return [
        *tasks,
        {
            "task_id": "basin_fixed_endpoint",
            "kind": "fixed_endpoint",
            "common": "basin_coupled_s0",
            "case": "basin_coupled",
            "strategy": "fixed_endpoint",
            "seed": 0,
        },
    ]


def prepare(root: Path, inputs: Path, policy: Policy) -> dict:
    data = read_seal(inputs)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": roster(data),
        "test_data_opened": False,
        "live_llm_calls": 0,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["tasks"] != roster(data)
        or plan["inputs_sha256"] != public.content_sha256(data)
    ):
        raise ValueError("adapter plan/input identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("adapter source/runtime differs")
    return plan, data


def completed(folder: Path, identity: dict) -> dict:
    """Verify the exact stored handoff and scoring record before terminal reuse."""
    row = read_seal(folder / "result.json")
    if row["identity"] != identity:
        raise ValueError("adapter task identity differs")
    if row["fit"] is not None:
        fit = adapter.FitResult.model_validate(read_seal(folder / "fit/result.json"))
        if fit.model_dump(mode="json") != row[
            "fit"
        ] or fit.backend_sha256 != public.content_sha256(
            read_seal(folder / "fit/backend.json")
        ):
            raise ValueError("adapter fitted handoff differs")
    if row["evaluation"] != read_seal(folder / "evaluation.json"):
        raise ValueError("adapter evaluation differs")
    return row


def evaluate(data: dict, task: dict, parameters: dict | None, policy, folder) -> dict:
    """Only this post-fit evaluator receives validation arrays and reference values."""
    if parameters is None:
        return {"status": "no_verified_parameters", "solver_agreement_passed": False}
    common = data["development"]["commons"][task["common"]]
    case = data["development"]["cases"][task["case"]]
    verification = verified_replay.run(
        PublicFitRequest.model_validate(common["request"]),
        parameters,
        PublicSplit.model_validate(case["training"]),
        PublicSplit.model_validate(case["validation"]),
        policy,
        folder,
    )
    errors = coefficient_metrics(case, common["request"], parameters)
    threshold = data["development"]["evaluation_thresholds"]
    error = errors.get("maximum_coefficient_relative_error")
    return {
        **verification,
        "coefficient_recovery": errors,
        # Deliberately separate the arithmetic error from integrator certification.
        "coefficient_error_within_threshold": error is not None
        and error <= threshold["parameter_relative"],
        "prediction_accuracy_passed": verification["solver_agreement_passed"]
        and max(verification["metrics"].values()) <= threshold["cstr_nmse"],
    }


def run_task(root: Path, index: int) -> dict:
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown adapter task index")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = root / "results" / task["task_id"]
    with public._lock(folder):
        if (folder / "result.json").exists():
            return completed(folder, identity)
        policy = Policy.model_validate(plan["policy"])
        fit = None
        if task["kind"] == "fit":
            fit = adapter.run(
                source.bases(data["development"])[task["common"]],
                data["starts"][task["common"]],
                adapter.AdapterPolicy(strategy=task["strategy"], search=policy.search),
                folder / "fit",
            )
            parameters = fit.parameters
        else:
            parameters = data["fixed_endpoint"]["parameters"]
        # FitResult and its backend are already sealed before evaluator access.
        evaluation_path = folder / "evaluation.json"
        if not evaluation_path.exists():
            seal(
                evaluation_path,
                evaluate(data, task, parameters, policy.replay, folder / "evaluation"),
            )
        row = task | {
            "identity": identity,
            "status": "complete",
            "fit": fit.model_dump(mode="json") if fit else None,
            "parameters_sha256": public.content_sha256(parameters),
            "evaluation": read_seal(evaluation_path),
            "parameter_fitting_performed": fit is not None,
        }
        seal(folder / "result.json", row)
        return row


def report(root: Path) -> dict:
    plan, _ = verify(root, runtime=False)
    rows = [
        completed(
            root / "results" / t["task_id"],
            {"plan_sha256": public.content_sha256(plan), "task": t},
        )
        if (root / "results" / t["task_id"] / "result.json").exists()
        else t | {"status": "missing"}
        for t in plan["tasks"]
    ]
    groups = []
    for case in source.CASES:
        for strategy in STRATEGIES:
            selected = [
                r
                for r in rows
                if r["case"] == case
                and r["strategy"] == strategy
                and r["status"] != "missing"
            ]
            times = [r["fit"]["cost"]["wall_seconds"] for r in selected]
            groups.append(
                {
                    "case": case,
                    "strategy": strategy,
                    "expected": source.STARTS,
                    "recorded": len(selected),
                    "training_target_reached": sum(
                        r["fit"]["verification"]["target_reached"] for r in selected
                    ),
                    "solver_agreement_passed": sum(
                        r["evaluation"]["solver_agreement_passed"] for r in selected
                    ),
                    "coefficient_error_within_threshold": sum(
                        r["evaluation"].get("coefficient_error_within_threshold", False)
                        for r in selected
                    ),
                    "median_fit_seconds": median(times) if times else None,
                    "fit_seconds": sum(times),
                    "evaluation_seconds": sum(
                        r["evaluation"].get("wall_seconds", 0) for r in selected
                    ),
                }
            )
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete"
        if all(r["status"] != "missing" for r in rows)
        else "incomplete",
        "expected": len(rows),
        "recorded": sum(r["status"] != "missing" for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "rows": rows,
        "groups": groups,
        "cost_complete": all(
            r["status"] != "missing"
            and (r["fit"] is None or r["fit"]["cost"]["accounting_complete"])
            and r["evaluation"].get(
                "accounting_complete",
                r["evaluation"]["status"] == "no_verified_parameters",
            )
            for r in rows
        ),
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": "Known structures, repeated starts and noiseless data. "
        "Fit completion is not accuracy; local sensitivity is not global "
        "identifiability. Validation/reference diagnostics are evaluator-only. "
        "Fixed endpoint is not a fit.",
    }
    public._write(root / "summary.json", result)
    return result
