"""Matched-budget generic-start recovery; training evidence owns every decision."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import mesh_refinement_process as process
from autoformalism.fitting import mesh_refinement_worker as mesh
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_mesh


def checkpoint_points(folder: Path) -> list[dict]:
    """Preserve a final iterate and an earlier low-objective candidate for rollout.

    A collocation objective ranks candidates for bounded screening only. It never
    makes a candidate a deployable incumbent, even with tiny discrete defects.
    """
    points = []
    detail = folder / "final_checkpoint_diagnostics.json"
    if detail.exists():
        points.append({"source": "final", **public._read(detail)})
    pool = folder / "checkpoints.json"
    if pool.exists():
        saved = public._read(pool)
        points.extend(
            sorted(saved.get("pool", []), key=lambda p: p["collocation_nmse"])
        )
        if saved.get("latest"):
            points.append(saved["latest"])
    unique = {}
    for p in points:
        unique.setdefault(public.content_sha256(p["parameters"]), p)
    return list(unique.values())[:2]


def fit(
    base: dict,
    arm: str,
    policy: dict,
    folder: Path,
    *,
    rollout_mode: str = "recovery_rollout",
) -> dict:
    """One 20-minute envelope, including setup/screens and early certificates.

    A coordinator interrupted before sealing its backend is terminal on resume;
    no unused-looking stage refreshes a fitting allowance. Safe retained evidence
    remains available for the separate evaluator.
    """
    if arm not in {"rollout_only", "medium_rollout", "mesh_rollout"}:
        raise ValueError("unknown recovery arm")
    if rollout_mode not in {
        "recovery_rollout",
        "profiled_rollout",
        "coupled_profiled_rollout",
    } or (rollout_mode != "recovery_rollout" and arm != "rollout_only"):
        raise ValueError("unsupported recovery rollout mode")
    identity_fields = {"base": base, "arm": arm, "policy": policy}
    if rollout_mode != "recovery_rollout":
        identity_fields["rollout_mode"] = rollout_mode
    identity = public.content_sha256(identity_fields)
    started = folder / "fit-started.json"
    if started.exists():
        if read_seal(started)["identity"] != identity:
            raise ValueError("generic recovery resume identity differs")
        progress = folder / "progress.json"
        result = (
            public._read(progress)
            if progress.exists()
            else {"selected": None, "levels": [], "certificates": []}
        )
        return {
            **result,
            "stop_reason": "interrupted_no_fit_restart",
            "budget_restarted": False,
        }
    begun = monotonic()
    seal(started, {"identity": identity})
    deadline = begun + policy["seconds"]
    collocation_end = (
        begun
        + (policy["seconds"] - policy["certificate_seconds"])
        * policy["collocation_fraction"]
    )
    result = {
        "selected": None,
        "levels": [],
        "certificates": [],
        "budget_restarted": False,
        "reference_values_used": False,
        "validation_used_for_fitting": False,
    }
    halted = False
    calls = 0

    def save():
        public._write(folder / "progress.json", result)

    def operate(mode, payload, path, cap, until=deadline):
        nonlocal halted
        seconds = min(cap, until - monotonic())
        if halted or seconds < 1:
            return None
        payload = {**payload, "seconds": seconds}
        outcome = process.invoke(mode, payload, path, seconds)
        if not outcome["termination_confirmed"]:
            halted = True
        target = path / "result.json"
        value = public._read(target) if target.exists() else None
        if value and value.get("payload_sha256") != public.content_sha256(payload):
            raise ValueError("generic recovery worker output differs")
        return {"process": outcome, "value": value}

    def screen(parameters, origin, path, until):
        nonlocal calls
        out = operate(
            "point",
            {k: base[k] for k in ("request", "training")}
            | {
                "parameters": parameters,
                "screening": {
                    "method": "Radau",
                    "point_seconds": policy["point_seconds"],
                },
            },
            path,
            policy["point_seconds"],
            until,
        )
        if not out:
            return
        calls += 1
        v = out["value"]
        if out["process"]["status"] == "complete" and v and v["status"] == "complete":
            if v["parameters"] != parameters:
                raise ValueError("screened parameter identity differs")
            if (
                result["selected"] is None
                or v["training_nmse"] < result["selected"]["training_nmse"]
            ):
                result["selected"] = {
                    "parameters": parameters,
                    "training_nmse": v["training_nmse"],
                    "maximum_trajectory_nmse": v["maximum_trajectory_nmse"],
                    "origin": origin,
                }
        save()

    def certificate(only_promising):
        selected = result["selected"]
        if not selected or halted:
            return False
        if only_promising and selected["training_nmse"] > policy["training_nmse"]:
            return False
        digest = public.content_sha256(selected["parameters"])
        for prior in result["certificates"]:
            if prior["parameters_sha256"] == digest:
                return bool(prior.get("passed"))
        out = operate(
            "recovery_check",
            {k: base[k] for k in ("request", "training")}
            | {
                "parameters": selected["parameters"],
                **{
                    k: policy[k]
                    for k in ("training_nmse", "trajectory_nmse", "solver_agreement")
                },
            },
            folder / "checks" / digest,
            policy["certificate_seconds"],
        )
        if not out:
            return False
        value = out["value"] or {}
        if value and value.get("parameters") != selected["parameters"]:
            raise ValueError("certificate vector differs")
        passed = out["process"]["status"] == "complete" and value.get("passed", False)
        result["certificates"].append(
            {"parameters_sha256": digest, **out, "passed": passed}
        )
        save()
        return passed

    # Preserve the generic pair before any structural discretization moves it.
    if arm != "rollout_only":
        screen(
            base["start"], "generic_start", folder / "initial-screen", collocation_end
        )
    certified = certificate(True)
    previous = None
    targets = (
        [policy["medium_target"]] if arm == "medium_rollout" else policy["targets"]
    )
    if arm != "rollout_only" and not certified:
        for level, target in enumerate(targets):
            if halted or monotonic() >= collocation_end - 1:
                break
            path = folder / f"level-{level}"
            grids, audit = recovery_mesh.grids(
                base, target, policy["minimum_intervals"], policy["observation_anchors"]
            )
            guesses = (
                mesh.transfer(previous, grids)
                if previous
                else recovery_mesh.initial_guesses(base, grids)
            )
            parameters = previous["parameters"] if previous else base["start"]
            payload = mesh.native_payload(base, parameters, grids, guesses)
            # Leave one ordinary rollout screen inside the collocation allocation.
            seconds = min(
                policy["native_seconds"],
                collocation_end - monotonic() - policy["point_seconds"],
            )
            if seconds < 1:
                break
            payload["seconds"] = seconds
            outcome = process.invoke("native", payload, path / "native", seconds)
            halted |= not outcome["termination_confirmed"]
            record = {
                "level": level,
                "mesh": audit,
                "process": outcome,
                "transfer": "radau_polynomial" if previous else "frozen_generic_nodes",
            }
            result["levels"].append(record)
            save()
            if halted:
                break
            points = checkpoint_points(path / "native")
            for i, p in enumerate(points):
                screen(
                    p["parameters"],
                    f"level-{level}-checkpoint-{i}",
                    path / f"screen-{i}",
                    collocation_end,
                )
            certified = certificate(True)
            if certified:
                break
            native_path = path / "native/native.json"
            detail_path = path / "native/final_checkpoint_diagnostics.json"
            native = public._read(native_path) if native_path.exists() else {}
            detail = public._read(detail_path) if detail_path.exists() else None
            qualified = bool(
                outcome["status"] == "complete"
                and native.get("native_success")
                and detail
                and detail["maximum_scaled_defect"] <= 1e-6
            )
            record["qualified_for_transfer"] = qualified
            save()
            if not qualified:
                break
            previous = detail

    if not certified and not halted:
        seconds = min(1200, deadline - monotonic() - policy["certificate_seconds"])
        if seconds >= 5:
            selected = result["selected"]
            points = [
                {
                    "parameters": selected["parameters"] if selected else base["start"],
                    "source": selected["origin"] if selected else "generic_start",
                }
            ]
            out = operate(
                rollout_mode,
                {k: base[k] for k in ("request", "training", "coordinates")}
                | {
                    "points": points,
                    "maximum_calls": policy["maximum_rollout_calls"],
                    "training_nmse": policy["training_nmse"],
                    "trajectory_nmse": policy["trajectory_nmse"],
                },
                folder / "rollout",
                seconds,
            )
            result["rollout"] = out
            if out and not halted:
                v = out["value"]
                fallback = folder / "rollout/refinement/best.json"
                if not v and fallback.exists():
                    v = public._read(fallback)
                if (
                    v
                    and v.get("parameters")
                    and (
                        result["selected"] is None
                        or v["training_nmse"] < result["selected"]["training_nmse"]
                    )
                ):
                    result["selected"] = {
                        k: v[k] for k in ("parameters", "training_nmse")
                    } | {"origin": "rollout_refinement"}
            save()
        certified = certificate(False)
    result.update(
        stop_reason="training_prediction_certified"
        if certified
        else "cleanup_unconfirmed"
        if halted
        else "budget_or_optimizer_stop",
        fit_seconds=monotonic() - begun,
        fitting_allowance_seconds=policy["seconds"],
        screening_calls=calls,
    )
    save()
    return result
