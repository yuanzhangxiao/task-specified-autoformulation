"""Export M9 saved pools and training-qualified starts without new optimization."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import transcription_campaign as source

PROTOCOL = "phase-c-screening-replay-inputs-1"


def export(source_root: Path, output: Path, *, case_name: str = "alien_hard") -> dict:
    """Verify saved worker/point identities; read neither test data nor val scores.

    Validation arrays and reference coefficients are copied only for the separate
    post-selection evaluator. All roster decisions use case/arm and training-only
    source eligibility. No new start is generated and no missing fit is rerun.
    """
    plan, inputs = source.verify(source_root, runtime=False)
    if plan["protocol"] != source.screening.PROTOCOL or plan["test_data_opened"]:
        raise ValueError("requires an M9 development source")
    qualification = read_seal(source_root / "qualification/result.json")
    if not qualification["passed"] or qualification[
        "plan_sha256"
    ] != public.content_sha256(plan):
        raise ValueError("source qualification missing or mismatched")
    if case_name not in inputs["cases"]:
        raise ValueError("unknown source case")
    bundle = {
        "protocol": PROTOCOL,
        "source_plan_sha256": public.content_sha256(plan),
        "source_code_sha256": plan["source_sha256"],
        "source_inputs_sha256": plan["inputs_sha256"],
        "source_qualification_sha256": public.content_sha256(qualification),
        "config": plan["config"],
        "case": deepcopy(inputs["cases"][case_name]),
        "case_name": case_name,
        "commons": {},
        "pools": [],
        "assisted": [],
        "calibration_points": [],
        "excluded_assisted": [],
        "test_data_opened": False,
    }
    for key, common in plan["commons"].items():
        if common["case"] != case_name:
            continue
        bundle["commons"][key] = deepcopy(common)
        _, ordinary, _ = public._lower(
            source.PublicFitRequest.model_validate(common["request"])
        )
        bundle["calibration_points"].append(
            {
                "id": f"{key}_ordinary",
                "common": key,
                "parameters": ordinary,
                "required_good_start": False,
            }
        )
        assisted = common["assisted_start"]
        if assisted["eligible"]:
            bundle["calibration_points"].append(
                {
                    "id": f"{key}_assisted",
                    "common": key,
                    "parameters": assisted["parameters"],
                    "required_good_start": True,
                }
            )
    for task in plan["tasks"]:
        if task["common"] not in bundle["commons"]:
            continue
        folder = source_root / "results" / task["task_id"]
        worker = source.worker_payload(plan, inputs, task)
        if "assisted_start" in worker:
            if worker["assisted_start"]["eligible"]:
                bundle["assisted"].append(
                    {
                        **task,
                        "payload": {
                            k: v
                            for k, v in worker.items()
                            if k not in {"request", "coordinates", "nodes", "training"}
                        },
                        "source_worker_payload_sha256": public.content_sha256(worker),
                    }
                )
            else:
                bundle["excluded_assisted"].append(task)
            continue
        backend = read_seal(folder / "backend.json")
        if backend["worker_payload_sha256"] != public.content_sha256(worker):
            raise ValueError("source backend payload differs")
        journal = backend.get("screening") or {}
        points: dict[str, dict] = {}

        def include(parameters: dict, origin: str, points=points) -> dict:
            digest = public.content_sha256(parameters)
            point = points.setdefault(
                digest,
                {
                    "parameters": parameters,
                    "parameters_sha256": digest,
                    "origins": [],
                    "cached": None,
                },
            )
            point["origins"].append(origin)
            return point

        for attempt in journal.get("attempts", []):
            directory = (
                folder / "fit/screens" / attempt["phase"] / f"{attempt['index']:03d}"
            )
            payload = public._read(directory / "payload.json")
            if any(
                payload[k] != worker[k] for k in ("training", "request", "screening")
            ):
                raise ValueError("saved screening contract differs")
            if (
                public.content_sha256(payload["parameters"])
                != attempt["parameters_sha256"]
            ):
                raise ValueError("saved screening vector differs")
            point = include(
                payload["parameters"], f"{attempt['phase']}:{attempt['index']}"
            )
            if attempt["status"] == "complete":
                result = public._read(directory / "result.json")
                if (
                    result.get("status") != "complete"
                    or result["payload_sha256"] != public.content_sha256(payload)
                    or result["training_nmse"] != attempt["training_nmse"]
                    or result["parameters"] != point["parameters"]
                ):
                    raise ValueError("saved complete training score differs")
                point["cached"] = {
                    **{
                        k: result[k]
                        for k in (
                            "parameters",
                            "training_nmse",
                            "maximum_trajectory_nmse",
                        )
                    },
                    "method": worker["screening"]["method"],
                    "source_payload_sha256": result["payload_sha256"],
                    "source_result_sha256": public.content_sha256(result),
                    "source_process_seconds": attempt["process"]["elapsed_seconds"],
                }
        checkpoint_path = folder / "fit/mesh-0/checkpoints.json"
        if checkpoint_path.exists():
            checkpoints = public._read(checkpoint_path)
            for point in checkpoints.get("pool", []):
                include(point["parameters"], f"native_iteration:{point['iteration']}")
        if not points:
            raise ValueError("source pool is empty")
        bundle["pools"].append(
            {
                **task,
                "method": worker["screening"]["method"],
                "source_worker_payload_sha256": public.content_sha256(worker),
                "points": list(points.values()),
                "source_backend_sha256": public.content_sha256(backend),
                "source_fit_seconds": backend["total_seconds"],
                "source_selected_parameters": backend.get("parameters"),
            }
        )
    if not bundle["pools"] or not bundle["assisted"]:
        raise ValueError("requires saved generic pools and eligible assisted starts")
    seal(output, bundle)
    return {
        "sha256": public.content_sha256(bundle),
        "pools": len(bundle["pools"]),
        "assisted": len(bundle["assisted"]),
        "calibration_points": len(bundle["calibration_points"]),
    }
