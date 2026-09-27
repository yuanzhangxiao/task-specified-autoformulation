"""Matched CSTR numerical diagnostics imported from a sealed qualification.

This is evaluator-assisted fitting, not discovery. Original optimizer guesses,
equations and public development arrays are reused; fitted winners are not starts.
"""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path

from autoformalism.benchmarks import reference_qualification as qualification
from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

PROTOCOL = "phase-c-cstr-refinement-1"
PROFILES = {
    "wide_standard": "collocation-single-target-v2",
    "local_standard": "collocation-local-poll-v1",
    "wide_long": "collocation-budget-control-v1",
    "local_physical_initials": "collocation-local-poll-v1",
}


@contextmanager
def _campaign_lock(root: Path):
    """Serialize short preparation/report writes across array workers."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".followup.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def physical_initializers(
    request: PublicFitRequest, truth: dict, training: PublicSplit
) -> tuple[PublicFitRequest, dict, dict]:
    """Positive concentration anchors and centered jacket deviations.

    The concentration map is affine between training initial-temperature extrema
    and constant beyond them. This guarantees nonnegative C(0), not a general
    state-domain certificate. The jacket map is an equivalent affine map in
    deviation coordinates; no new constraint is imposed on its output.
    """
    temperatures = [r.targets["T"][0] for r in training.rows]
    low = min(temperatures)
    span = max(max(temperatures) - low, 1.0)
    weight = f"min(max((T-({low!r}))/({span!r}),0),1)"
    payload = request.model_dump(mode="json")
    rules = payload["initialization_plan"]["rules"]
    mapped_truth = {k: v for k, v in truth.items() if not k.startswith("init_")}
    for state in ("C", "Tj"):
        initial = rules[state]["initial"]
        if initial["expression"] != "a+b*(T-350)":
            raise ValueError("follow-up requires the original affine initializer")
        guesses = {p["name"]: p["guess"] for p in initial["parameters"]}
        if state == "C":
            expression = f"lo*(1-({weight}))+hi*({weight})"
            parameters = []
            for name, temperature in (("lo", low), ("hi", low + span)):
                guess = guesses["a"] + guesses["b"] * (temperature - 350)
                value = truth["init_C_a"] + truth["init_C_b"] * (temperature - 350)
                if min(guess, value) < 0:
                    raise ValueError("source concentration anchors must be nonnegative")
                parameters.append(
                    {
                        "name": name,
                        "role": "nonnegative_coefficient",
                        "guess": guess,
                    }
                )
                mapped_truth[f"init_C_{name}"] = value
        else:
            expression = f"T+offset+change*((T-({low!r}))/({span!r}))"
            parameters = [
                {
                    "name": "offset",
                    "role": "coefficient",
                    "guess": guesses["a"] + guesses["b"] * (low - 350) - low,
                },
                {
                    "name": "change",
                    "role": "coefficient",
                    "guess": (guesses["b"] - 1) * span,
                },
            ]
            mapped_truth["init_Tj_offset"] = (
                truth["init_Tj_a"] + truth["init_Tj_b"] * (low - 350) - low
            )
            mapped_truth["init_Tj_change"] = (truth["init_Tj_b"] - 1) * span
        rules[state] = {
            "initial": {
                "mode": "map",
                "expression": expression,
                "parameters": parameters,
            }
        }
    return (
        PublicFitRequest.model_validate(payload),
        mapped_truth,
        {
            "policy": "cstr-public-initial-coordinates-1",
            "training_initial_temperature_low": low,
            "training_initial_temperature_span": span,
            "concentration_nonnegative_by_construction": True,
            "jacket_initial_domain_constrained": False,
            "extrapolation": "C initializer saturates outside training anchors",
            "selection_uses_validation": False,
        },
    )


def _source_records(source: Path) -> tuple[dict, list[dict]]:
    plan = read_seal(source / "plan.json")
    if plan["protocol"] != "phase-c-cstr-qualification-1":
        raise ValueError("expected original CSTR qualification")
    roster = read_seal(source / "tasks.json")["tasks"]
    expected = {
        f"cstr_{tier}_{start}"
        for tier in ("easy", "hard")
        for start in qualification.STARTS
    }
    if len(roster) != 4 or {r["task"] for r in roster} != expected:
        raise ValueError("source roster must contain the four predetermined starts")
    records = []
    for task in roster:
        directory = source / task["task"]
        frozen = public._read(directory / "fit/freeze.json")
        body = {k: v for k, v in frozen.items() if k != "identity"}
        result = read_seal(directory / "qualification.json")
        truth = read_seal(directory / "reference_parameters.json")
        if (
            public.content_sha256(body) != frozen["identity"]
            or frozen["source_sha256"] != plan["source_sha256"]
            or result["status"] != "complete"
            or result["fit"]["identity"] != frozen["identity"]
        ):
            raise ValueError("source qualification identity differs or is incomplete")
        request = PublicFitRequest.model_validate(frozen["request"])
        if (
            request.profile != "collocation-single-target-v2"
            or request.base_candidate.candidate_id
            != f"cstr_reference_{task['tier']}_coupled"
            or task["task"] != f"cstr_{task['tier']}_{task['start']}"
        ):
            raise ValueError("source request is not the declared CSTR diagnostic")
        records.append(
            {
                "task": task,
                "frozen": frozen,
                "truth": truth,
                "result_sha256": public.content_sha256(result),
            }
        )
    return plan, records


def prepare(root: Path, source: Path) -> dict:
    """Freeze all fourteen comparisons before any optimization; resume identically."""
    if root.resolve().is_relative_to(source.resolve()):
        raise ValueError("follow-up must be outside the historical qualification")
    parent, records = _source_records(source)
    plan = {
        "protocol": PROTOCOL,
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "source_plan_sha256": public.content_sha256(parent),
        "source_records": [
            {
                "task": r["task"],
                "identity": r["frozen"]["identity"],
                "result_sha256": r["result_sha256"],
                "truth_sha256": public.content_sha256(r["truth"]),
            }
            for r in records
        ],
        "assistance": parent["assistance"],
        "profiles": PROFILES,
        "other_families": parent["other_families"],
        "original_guesses_used": True,
        "previous_fit_parameters_used": False,
        "test_data_opened": False,
        "live_llm_calls": 0,
    }
    tasks = []
    with _campaign_lock(root), public._lock(root):
        seal(root / "plan.json", plan)
        for record in records:
            origin, frozen = record["task"], record["frozen"]
            request = PublicFitRequest.model_validate(frozen["request"])
            train = PublicSplit.model_validate(frozen["training"])
            val = PublicSplit.model_validate(frozen["validation"])
            for arm, profile in PROFILES.items():
                if arm == "local_physical_initials" and origin["tier"] != "hard":
                    continue
                task = {
                    **origin,
                    "source_task": origin["task"],
                    "arm": arm,
                    "task": f"{origin['task']}_{arm}",
                }
                candidate, truth = request, record["truth"]
                initial_audit = {"policy": "unchanged"}
                if arm == "local_physical_initials":
                    candidate, truth, initial_audit = physical_initializers(
                        request, truth, train
                    )
                payload = candidate.model_dump(mode="json")
                payload["profile"] = profile
                payload["source"] = {
                    "stage": "synthetic_control",
                    "task_id": task["task"],
                    "artifact_sha256": frozen["identity"],
                }
                candidate = PublicFitRequest.model_validate(payload)
                directory = root / task["task"]
                seal(directory / "reference_parameters.json", truth)
                seal(directory / "initial_coordinate_audit.json", initial_audit)
                freeze_path = directory / "fit/freeze.json"
                if freeze_path.exists():
                    # Frozen handoffs are immutable. A different array worker
                    # may be optimizing here while this worker prepares itself.
                    # Validate bytes/identity without acquiring its execution lock.
                    if public._read(freeze_path) != public._bundle(
                        candidate, train, val
                    ):
                        raise ValueError("existing follow-up freeze differs")
                else:
                    public.prepare_fit(candidate, train, val, directory / "fit")
                tasks.append(task)
        seal(root / "tasks.json", {"tasks": tasks})
    return {
        "protocol": PROTOCOL,
        "tasks": len(tasks),
        "identity": public.content_sha256(plan),
        "test_data_opened": False,
        "live_llm_calls": 0,
    }


def report(root: Path) -> dict:
    """Retain every arm; accuracy, feasibility and optimizer progress stay separate."""
    with _campaign_lock(root):
        return _report(root)


def _report(root: Path) -> dict:
    result = qualification.report(root)
    result["limitation"] = (
        "Assisted CSTR numerical diagnosis, not discovery or unique recovery. "
        "Local refinement and larger budget each compare against wide_standard; "
        "hard initializer constraints/coordinates compare against local_standard. "
        "Initializer coordinates and positivity are a combined intervention. "
        "All endpoints reported; no selection on validation or test evaluation."
    )
    for row in result["rows"]:
        directory = root / row["task"]
        path = directory / "fit/backend_result.json"
        if not path.exists():
            continue
        backend = public._read(path)
        completed = directory / "qualification.json"
        if completed.exists():
            fit = read_seal(completed).get("fit", {})
            if fit.get("backend_result_sha256") != public.content_sha256(backend):
                raise ValueError("reported backend result digest differs")
        initializer = backend.get("initializer") or {}
        refinement = backend.get("refinement") or {}
        row["optimizer_diagnostics"] = {
            "initializer_success": initializer.get("success"),
            "initializer_message": initializer.get("message"),
            "initializer_seconds": initializer.get("seconds"),
            "refinement_seconds": refinement.get("fit_seconds"),
            "residual_calls": refinement.get("actual_residual_calls"),
            "polls": [
                {
                    k: stage.get("result", {}).get(k)
                    for k in ("policy", "poll_radius", "cost", "actual_residual_calls")
                }
                for stage in refinement.get("stages", [])
                if stage.get("mode") == "directional_poll"
            ],
        }
        initials = (backend.get("training") or {}).get(
            "trajectory_initial_conditions", {}
        )
        row["training_initial_concentration_minimum"] = min(
            (values["C"] for values in initials.values() if "C" in values), default=None
        )
    with public._lock(root):
        public._write(root / "summary.json", result)
    return result
