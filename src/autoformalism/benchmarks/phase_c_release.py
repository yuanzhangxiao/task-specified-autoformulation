"""Checkpointed Phase C development release: public data and private witnesses.

No fitting, LLM calls or test generation. Dalla Man permits reduced ODE models;
T3/T4 are outside the Phase C roster. Historical data stay
unchanged. Only public/<cell> is an input to discovery or external baselines.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from autoformalism.benchmarks import detention
from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.benchmarks.phase_b_public import (
    PhaseBPublicSpec,
    _private_array,
    phase_b_public_spec,
    render_phase_b_prompts,
)
from autoformalism.benchmarks.phase_c_oracles import reference_request, replay
from autoformalism.benchmarks.phase_c_protocols import PREPARATION, phase_c_protocols
from autoformalism.benchmarks.reference_audit import _digest, audit_one
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.public_fitting import PublicSplit

REPO = Path(__file__).resolve().parents[3]
PROTOCOL = "phase-c-development-2"
SPEC_PATHS = {
    "cstr": "benchmark5_anonymous_nonlinear_process/private/system_specification.json",
    "alien_device": "benchmark6_alien_device/private/selected_system_spec.json",
}
FAMILIES = ("cstr", "alien_device", "detention", "dalla_man")
CELL_COUNTS = {"cstr": 4, "alien_device": 4, "detention": 4, "dalla_man": 16}
DALLA_SCOPE = {
    "policy": "dalla-reduced-ode-development-1",
    "reference_dynamics_changed": False,
    "public_channels_changed": False,
    "exact_reference_transcription_required": False,
    "exact_causal_initializer_certified": False,
    "interpretation": (
        "Continuous ODE candidates may approximate the private reference's gastric "
        "bookkeeping and unobserved preparation. Evaluate the public task mechanisms "
        "and free rollouts with shared training-fitted parameters and causal "
        "initialization. No gastric compartments, normalization variable, or reset "
        "law are added to the public requirements. Numerical qualification does "
        "not establish exact representability, an irreducible prediction error, "
        "or recovery of the private physiological system."
    ),
}


def public_information_audit(split: PublicSplit, *, atol: float = 1e-8) -> dict:
    """Detect conflicting deterministic labels under identical allowed information.

    Full declared input/auxiliary schedules and initial target readings are used.
    IDs and future target observations cannot distinguish an initial preparation.
    This necessary check is not a general identifiability certificate. Apply to
    noiseless reference projections; observation noise is audited separately.
    """
    groups: dict[str, list] = {}
    for row in split.rows:
        info = {
            "time": row.time,
            "external_inputs": row.external_inputs,
            "auxiliaries": row.auxiliaries,
            "initial_targets": {c: y[0] for c, y in row.targets.items()},
            "fixed_covariates": row.fixed_covariates,
        }
        groups.setdefault(public.content_sha256(info), []).append(row)
    collisions = []
    for rows in groups.values():
        if len(rows) < 2:
            continue
        difference = max(
            float(np.max(np.ptp(np.array([r.targets[c] for r in rows]), axis=0)))
            for c in rows[0].targets
        )
        collisions.append(
            {
                "trajectories": [r.trajectory_id for r in rows],
                "maximum_target_difference": difference,
            }
        )
    return {
        "passed": all(g["maximum_target_difference"] <= atol for g in collisions),
        "duplicate_groups": collisions,
        "absolute_tolerance": atol,
        "scope": "necessary public-information check, not full observability",
    }


def project_split(
    spec: PhaseBPublicSpec, trajectories: tuple, name: str
) -> PublicSplit:
    """Project authorized channels without serializing private names or states."""
    selected = [
        t
        for t in trajectories
        if t.protocol_id.startswith("train_" if name == "train" else "validation_")
    ]
    rows = []
    for i, t in enumerate(selected):
        row = {"trajectory_id": f"{name}_{i:03d}", "time": t.time.tolist()}
        for role, field in [
            ("target", "targets"),
            ("auxiliary", "auxiliaries"),
            ("external_input", "external_inputs"),
        ]:
            row[field] = {
                c.public_name: _private_array(t, c.private_source).tolist()
                for c in spec.channels
                if c.role == role
            }
        rows.append(row)
    return PublicSplit.model_validate(
        {"name": name, "fingerprint": public.content_sha256(rows), "rows": rows}
    )


def _numeric_identity(splits: tuple[PublicSplit, ...]) -> str:
    """Compare semantic pairs by arrays in channel order, ignoring channel names."""
    return public.content_sha256(
        [
            [
                {
                    "time": r.time,
                    **{
                        k: list(getattr(r, k).values())
                        for k in ("targets", "auxiliaries", "external_inputs")
                    },
                }
                for r in s.rows
            ]
            for s in splits
        ]
    )


def _write_cell(
    root: Path,
    cell: str,
    splits: tuple[PublicSplit, ...],
    specification: dict,
    audit: dict,
) -> dict:
    directory = root / "public" / cell
    for split in splits:
        seal(directory / f"{split.name}.json", split.model_dump(mode="json"))
    seal(directory / "specification.json", specification)
    prompt_path = directory / "proposer_prompt.txt"
    prompt = specification["public_prompt"] + "\n"
    if prompt_path.exists() and prompt_path.read_text() != prompt:
        raise ValueError(f"published prompt changed: {cell}")
    if not prompt_path.exists():
        prompt_path.write_text(prompt)
    seal(root / "diagnostic" / cell / "audit.json", audit)
    return {
        "cell": cell,
        "ready_for_development": audit["passed"],
        "files": {
            str(p.relative_to(root)): _digest(p)
            for p in [*directory.iterdir(), root / "diagnostic" / cell / "audit.json"]
        },
        "numeric_identity": _numeric_identity(splits),
    }


def _phase_b_family(
    root: Path, family: str, data_root: Path, identity: str
) -> list[dict]:
    records, trajectories = [], []
    for protocol in phase_c_protocols(family):
        if protocol.split == "test":
            continue
        record, trajectory = audit_one(
            root / "diagnostic" / "references" / family,
            protocol,
            "canonical",
            data_root,
            identity,
        )
        if not record["passed"]:
            raise ValueError(
                f"reference numerical audit failed: {protocol.protocol_id}"
            )
        records.append(record)
        trajectories.append(trajectory)
    truth = json.loads((data_root / SPEC_PATHS[family]).read_text())
    hidden_indices = (
        list(range(truth["n_latent"])) if family == "alien_device" else [0, 2]
    )
    initial_reference = trajectories[0].states[0, hidden_indices]
    reset_difference = max(
        float(np.max(np.abs(t.states[0, hidden_indices] - initial_reference)))
        for t in trajectories
    )
    if reset_difference != 0:
        raise ValueError("reference violated shared hidden preparation")
    cells = []
    variants = ("named", "obfuscated") if family == "cstr" else ("functional", "opaque")
    for tier in ("easy", "hard"):
        paired = []
        for variant in variants:
            spec = phase_b_public_spec(
                family,
                tier,
                variant,
                input_contract="continuous-rates-1",
                data_root=data_root,
            )
            cell = spec.benchmark_id.replace("phase_b_", "phase_c_", 1).replace(
                "_rates_v1", "_reset_v1"
            )
            splits = tuple(
                project_split(spec, tuple(trajectories), name)
                for name in ("train", "val")
            )
            request = reference_request(spec, truth)
            seal(
                root / "diagnostic" / cell / "reference_request.json",
                request.model_dump(mode="json"),
            )
            replay_path = root / "diagnostic" / cell / "reference_replay.json"
            if replay_path.exists():
                replay_result = read_seal(replay_path)
            else:
                replay_result = replay(request, splits)
                seal(replay_path, replay_result)
            combined = PublicSplit.model_validate(
                {
                    "name": "train",
                    "fingerprint": "combined-development",
                    "rows": [r.model_dump(mode="json") for s in splits for r in s.rows],
                }
            )
            information = public_information_audit(combined)
            prompt, judge = render_phase_b_prompts(spec)
            # Insert the preparation rule in the available-information section.
            prompt = prompt.replace(
                "C. Modeling requirements",
                "Preparation and initialization:\n"
                + PREPARATION[family]
                + "\n\nC. Modeling requirements",
            )
            specification = {
                "protocol": PROTOCOL,
                "benchmark_id": cell,
                "public_prompt": prompt,
                "judge_prompt": judge,
                "input_contract": "continuous-rates-1",
                "preparation_contract": PREPARATION[family],
                "targets": list(splits[0].rows[0].targets),
                "auxiliaries": list(splits[0].rows[0].auxiliaries),
                "external_inputs": list(splits[0].rows[0].external_inputs),
                "fixed_covariates": [],
                "test_released": False,
                "noise_sd_fraction": 0,
                "initial_observation_noise": 0,
            }
            audit = {
                "passed": information["passed"] and replay_result["passed"],
                "numerical_protocols": len(records),
                "reference_numerics_passed": all(r["passed"] for r in records),
                "shared_hidden_preparation_verified": reset_difference == 0,
                "public_information": information,
                "public_interface_replay": replay_result,
                "mechanism_recovery_or_identifiability_certified": False,
            }
            row = _write_cell(root, cell, splits, specification, audit)
            row["files"][str(replay_path.relative_to(root))] = _digest(replay_path)
            request_path = root / "diagnostic" / cell / "reference_request.json"
            row["files"][str(request_path.relative_to(root))] = _digest(request_path)
            cells.append(row)
            paired.append(row["numeric_identity"])
        if len(set(paired)) != 1:
            raise ValueError("semantic variants changed numerical data")
    return cells


def _dalla_family(root: Path, data_root: Path, identity: str) -> list[dict]:
    """Qualify T1/T2 reference numerics without claiming exact ODE attainability.

    The private generator and existing public prompts remain unchanged. In
    particular, its gastric normalization is not exposed as an extra input.
    """
    cells = []
    for task in ("T1", "T2"):
        for dynamics in ("canonical", "perturbed"):
            records, trajectories = [], []
            for protocol in phase_c_protocols("dalla_man", task=task):
                if protocol.split == "test":
                    continue
                record, trajectory = audit_one(
                    root / "diagnostic" / "references" / "dalla_man" / task / dynamics,
                    protocol,
                    dynamics,
                    data_root,
                    identity,
                )
                if not record["passed"]:
                    raise ValueError(
                        f"Dalla reference numerical audit failed: "
                        f"{task}/{dynamics}/{protocol.protocol_id}"
                    )
                records.append(record)
                trajectories.append(trajectory)
            initials = np.array([t.states[0] for t in trajectories])
            variation = {
                name: float(np.ptp(initials[:, i]))
                for i, name in enumerate(trajectories[0].state_names)
                if np.ptp(initials[:, i]) != 0
            }
            # The t=0 meal convention assumes an initially empty gut. Verify
            # that assumption instead of silently extending it to other states.
            gut_indices = [
                trajectories[0].state_names.index(n) for n in ("Qsto1", "Qsto2", "Qgut")
            ]
            empty_gut = bool(np.all(initials[:, gut_indices] == 0))
            for tier in ("easy", "hard"):
                paired = []
                for variant in ("named", "obfuscated"):
                    spec = phase_b_public_spec(
                        "dalla_man",
                        tier,
                        variant,
                        task=task,
                        dynamics=dynamics,
                        input_contract="continuous-rates-1",
                        data_root=data_root,
                    )
                    cell = spec.benchmark_id.replace("phase_b_", "phase_c_", 1)
                    splits = tuple(
                        project_split(spec, tuple(trajectories), name)
                        for name in ("train", "val")
                    )
                    combined = PublicSplit.model_validate(
                        {
                            "name": "train",
                            "fingerprint": "combined-development",
                            "rows": [
                                r.model_dump(mode="json")
                                for s in splits
                                for r in s.rows
                            ],
                        }
                    )
                    information = public_information_audit(combined)
                    prompt, judge = render_phase_b_prompts(spec)
                    specification = {
                        "protocol": PROTOCOL,
                        "benchmark_id": cell,
                        "public_prompt": prompt,
                        "judge_prompt": judge,
                        "input_contract": "continuous-rates-1",
                        "preparation_contract": (
                            "Use initial target readings, supplied auxiliary readings "
                            "and causal initialization with globally shared parameters "
                            "estimated on training. No validation-specific latent "
                            "initial fitting or unlisted observations are available."
                        ),
                        "targets": list(splits[0].rows[0].targets),
                        "auxiliaries": list(splits[0].rows[0].auxiliaries),
                        "external_inputs": list(splits[0].rows[0].external_inputs),
                        "fixed_covariates": [],
                        "test_released": False,
                        "noise_sd_fraction": 0,
                        "initial_observation_noise": 0,
                    }
                    audit = {
                        "passed": information["passed"] and empty_gut,
                        "numerical_protocols": len(records),
                        "reference_numerics_passed": all(r["passed"] for r in records),
                        "public_information": information,
                        "initially_empty_gut_verified": empty_gut,
                        "private_initial_coordinate_ranges": variation,
                        "shared_hidden_preparation_asserted": False,
                        "model_class_scope": DALLA_SCOPE,
                        "public_interface_replay": {
                            "status": "not_required_under_accepted_reduced_model_scope",
                            "performed": False,
                        },
                        "mechanism_recovery_or_identifiability_certified": False,
                    }
                    row = _write_cell(root, cell, splits, specification, audit)
                    cells.append(row)
                    paired.append(row["numeric_identity"])
                if len(set(paired)) != 1:
                    raise ValueError("semantic variants changed numerical data")
    return cells


def _detention_family(root: Path) -> list[dict]:
    config = detention.DetentionConfig()
    cells = []
    for case in ("coupled", "independent"):
        # Sealed generation checkpoint separates a partial release from a rerun.
        path = root / "diagnostic" / "references" / f"detention_{case}.json"
        if path.exists():
            generated = read_seal(path)
            data, private = generated["public"], generated["private"]
        else:
            data, private = detention.generate_case(config, case)
            seal(path, {"public": data, "private": private})
        clean = tuple(
            PublicSplit.model_validate(data["noise0"][s]) for s in ("train", "val")
        )
        reference = detention.reference_request(case, config.starts[0])
        parameters = {k: detention.TRUTH[k] for k in reference.parameter_guesses}
        replay_path = (
            root / "diagnostic" / "references" / f"detention_{case}_replay.json"
        )
        if replay_path.exists():
            result = read_seal(replay_path)
        else:
            result = replay(reference, clean, parameters)
            seal(replay_path, result)
        prompt = (REPO / "configs/detention_prompts" / f"{case}.md").read_text()
        info = [public_information_audit(s) for s in clean]
        for n, noise in enumerate(config.noise_fractions):
            cell = f"phase_c_detention_{case}_noise{n}_v1"
            splits = tuple(
                PublicSplit.model_validate(data[f"noise{n}"][s])
                for s in ("train", "val")
            )
            row = splits[0].rows[0]
            calibration = all(
                a.targets["h_down"][0] == b.targets["h_down"][0]
                for a, b in zip(clean[0].rows, splits[0].rows, strict=True)
            ) and all(
                a.targets["h_down"][0] == b.targets["h_down"][0]
                for a, b in zip(clean[1].rows, splits[1].rows, strict=True)
            )
            specification = {
                "protocol": PROTOCOL,
                "benchmark_id": cell,
                "public_prompt": prompt,
                "input_contract": "continuous-rates-1",
                "preparation_contract": (
                    "Measured initial_up and initial target reading; "
                    "no hidden per-trajectory initializer fitting."
                ),
                "targets": ["h_down"],
                "auxiliaries": [],
                "external_inputs": list(row.external_inputs),
                "fixed_covariates": list(row.fixed_covariates),
                "time_unit": "minute",
                "noise_sd_fraction": noise,
                "initial_observation_noise": 0,
                "test_released": False,
                "negative_control": case == "independent",
            }
            audit = {
                "passed": result["passed"]
                and calibration
                and all(i["passed"] for i in info),
                "public_information_clean": info,
                "public_interface_clean_replay": result,
                "initial_measurements_unchanged_by_noise": calibration,
                "maximum_balance_error": max(
                    v["maximum_relative_balance_error"]
                    for v in private["rows"].values()
                ),
                "maximum_reference_solver_difference": max(
                    v["solver_depth_difference_m"] for v in private["rows"].values()
                ),
                "rating_law": "piecewise-linear rating table, exact analytic hinges",
                "noise_checked_against_clean_reference": True,
                "mechanism_recovery_or_identifiability_certified": False,
            }
            cells.append(_write_cell(root, cell, splits, specification, audit))
    return cells


def _plan(data_root: Path, families: tuple[str, ...]) -> dict:
    return {
        "protocol": PROTOCOL,
        "families": list(families),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "private_spec_sha256": {
            n: _digest(data_root / p) for n, p in SPEC_PATHS.items() if n in families
        },
        "basin_prompt_sha256": {
            c: _digest(REPO / "configs/detention_prompts" / f"{c}.md")
            for c in ("coupled", "independent")
        }
        if "detention" in families
        else {},
        "test_generation_enabled": False,
        "parameter_fitting_performed": False,
        "llm_calls": 0,
        "dalla_model_class_scope": DALLA_SCOPE if "dalla_man" in families else None,
        "expected_cells": sum(CELL_COUNTS[f] for f in families),
        "pending_families": {},
        "excluded_tasks": ["T3", "T4"],
    }


def build(root: Path, data_root: Path, families: tuple[str, ...] = FAMILIES) -> dict:
    """Build/resume the new release; a changed source/configuration needs a new root."""
    if (
        not families
        or len(set(families)) != len(families)
        or set(families) - set(FAMILIES)
    ):
        raise ValueError("choose unique supported Phase C families")
    with public._lock(root):
        plan = _plan(data_root, families)
        seal(root / "plan.json", plan)
        identity = public.content_sha256(plan)
        if (root / "summary.json").exists():
            return verify(root)
        cells = []
        public._write(
            root / "progress.json",
            {
                "status": "incomplete",
                "completed_families": [],
                "planned_families": list(families),
                "completed_cells": 0,
            },
        )
        for family in families:
            receipt = root / "diagnostic" / f"{family}_complete.json"
            if receipt.exists():
                checkpoint = read_seal(receipt)
                if checkpoint["identity"] != identity:
                    raise ValueError("family checkpoint differs from frozen plan")
                rows = checkpoint["cells"]
                _verify_files(root, rows)
                _verify_files(root, [{"files": checkpoint["reference_files"]}])
            else:
                if family == "detention":
                    rows = _detention_family(root)
                elif family == "dalla_man":
                    rows = _dalla_family(root, data_root, identity)
                else:
                    rows = _phase_b_family(root, family, data_root, identity)
                references = root / "diagnostic" / "references"
                reference_paths = (
                    references.glob("detention_*.json")
                    if family == "detention"
                    else (references / family).rglob("*")
                )
                seal(
                    receipt,
                    {
                        "identity": identity,
                        "cells": rows,
                        "reference_files": {
                            str(p.relative_to(root)): _digest(p)
                            for p in sorted(reference_paths)
                            if p.is_file()
                        },
                    },
                )
            cells.extend(rows)
            public._write(
                root / "progress.json",
                {
                    "status": "incomplete",
                    "completed_families": list(families[: families.index(family) + 1]),
                    "planned_families": list(families),
                    "completed_cells": len(cells),
                },
            )
        complete_roster = len(cells) == plan["expected_cells"] and len(
            {c["cell"] for c in cells}
        ) == len(cells)
        ready = complete_roster and all(c["ready_for_development"] for c in cells)
        report = {
            "protocol": PROTOCOL,
            "status": "complete",
            "plan_sha256": identity,
            "development_cells": len(cells),
            "expected_cells": plan["expected_cells"],
            "ready_cells": sum(c["ready_for_development"] for c in cells),
            "ready_for_development": ready,
            "whole_phase_c_roster_ready": ready and set(families) == set(FAMILIES),
            "pending_families": {},
            "families_not_requested": sorted(set(FAMILIES) - set(families)),
            "dalla_model_class_scope": plan["dalla_model_class_scope"],
            "test_generated": False,
            "parameter_fitting_performed": False,
            "llm_calls": 0,
            "historical_releases_modified": False,
            "cells": cells,
            "limitation": (
                "Development reference/input/interface qualification, not fitting "
                "success, mechanism identifiability, or final held-out readiness."
            ),
        }
        report["files"] = {
            str(p.relative_to(root)): _digest(p)
            for directory in (root / "public", root / "diagnostic")
            for p in sorted(directory.rglob("*"))
            if p.is_file()
        }
        seal(root / "summary.json", report)
        public._write(
            root / "progress.json",
            {
                "status": "complete",
                "completed_families": list(families),
                "planned_families": list(families),
                "completed_cells": len(cells),
            },
        )
        return report


def _verify_files(root: Path, cells: list[dict]) -> None:
    for row in cells:
        for name, expected in row["files"].items():
            path = root / name
            if (
                not path.resolve().is_relative_to(root.resolve())
                or _digest(path) != expected
            ):
                raise ValueError(f"published release file changed: {name}")


def verify(root: Path) -> dict:
    """Read-only release verification; no generation, fitting or hidden test reads."""
    plan = read_seal(root / "plan.json")
    result = read_seal(root / "summary.json")
    if result["plan_sha256"] != public.content_sha256(plan):
        raise ValueError("summary differs from frozen plan")
    _verify_files(root, result["cells"])
    _verify_files(root, [{"files": result["files"]}])
    return result


def load_public_cell(directory: Path) -> tuple[dict, PublicSplit, PublicSplit]:
    """Load only proposer-visible data after release verification by the operator.

    This does not traverse the parent release or open evaluator diagnostics.
    The ordinary PublicSplit interface serves fitting and construction adapters;
    the legacy Phase-B registry is deliberately unchanged.
    """
    spec = read_seal(directory / "specification.json")
    if (
        spec["protocol"] not in {"phase-c-development-1", PROTOCOL}
        or spec["benchmark_id"] != directory.name
    ):
        raise ValueError("public cell identity differs")
    splits = tuple(
        PublicSplit.model_validate(read_seal(directory / f"{name}.json"))
        for name in ("train", "val")
    )
    for split, name in zip(splits, ("train", "val"), strict=True):
        if split.name != name:
            raise ValueError("public split name differs")
        for row in split.rows:
            for role in (
                "targets",
                "auxiliaries",
                "external_inputs",
                "fixed_covariates",
            ):
                if set(getattr(row, role)) != set(spec[role]):
                    raise ValueError(f"public channel contract differs: {role}")
    return spec, *splits
