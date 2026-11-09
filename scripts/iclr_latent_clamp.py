"""Measure the effect of holding learned latent coordinates on their control path."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.expressions.diagnostics import RuntimeExpressionError
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.simulation import (
    forcing_segment_indices,
    trajectory_forcing,
    trajectory_initial_state,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import CandidateModel
from scripts import iclr_anchor_fidelity as assay
from scripts.iclr_submission_evidence import digest, table

PANEL = "007208388d9c58cbb02b82451eda131388559fc89d802f0a9866e8f6eb33d1fa"
CONTRASTS = ("fed", "short_high")
FULL = "autoformalism:full:c1v1s1"
TARGETS = ("Gp", "I")
SOLVERS = ("Radau", "BDF")
EFFECT_TOL = 0.005
REPLAY_TOL = 1e-4


def simulate_branches(model, pars, initials, control, treated, groups, solver, seconds):
    """Solve the control, intact treatment and clamped treatments simultaneously."""
    t = control.time
    if not np.array_equal(t, treated.time) or model.validated.context.lagged_targets:
        raise ValueError(
            "paired interventions require identical grids and free rollouts"
        )
    n = len(model.state_names)
    init = trajectory_initial_state(model, control, initials, parameters=pars)
    if not np.array_equal(
        init, trajectory_initial_state(model, treated, initials, parameters=pars)
    ):
        raise ValueError("the pair has different initial states")
    forcings = [trajectory_forcing(model, tr) for tr in (control, treated)]
    # Only unclamped states are integrated in each treatment branch. Clamped
    # coordinates are read from the simultaneous control at every solver stage.
    free_indices = {
        label: np.asarray([i for i in range(n) if i not in indices], dtype=int)
        for label, indices in groups.items()
    }
    y0 = np.concatenate([init, *(init[idx] for idx in free_indices.values())])
    boundaries = sorted(
        set(forcing_segment_indices(model, control))
        | set(forcing_segment_indices(model, treated))
    )
    deadline = monotonic() + seconds

    def states(y):
        ctrl = y[:n]
        offset = n
        yield ctrl
        for idx in free_indices.values():
            x = ctrl.copy()
            x[idx] = y[offset : offset + len(idx)]
            offset += len(idx)
            yield x

    def rhs(time, y):
        if monotonic() > deadline:
            raise TimeoutError("clamp solver budget exhausted")
        ctrl, *hybrids = states(y)
        pieces = [model.rhs(time, ctrl, pars, forcings[0])]
        for idx, x in zip(free_indices.values(), hybrids, strict=True):
            pieces.append(model.rhs(time, x, pars, forcings[1])[idx])
        return np.concatenate(pieces)

    values = np.empty((len(y0), len(t)))
    values[:, 0] = y0
    for start, end in pairwise(boundaries):
        res = solve_ivp(
            rhs,
            (t[start], t[end]),
            values[:, start],
            t_eval=t[start : end + 1],
            method=solver,
            rtol=1e-8,
            atol=1e-10,
        )
        if not res.success or not np.all(np.isfinite(res.y)):
            raise RuntimeError(res.message)
        values[:, start : end + 1] = res.y
    targets = model.validated.context.targets
    preds = {label: {k: [] for k in targets} for label in ("control", *groups)}
    for i, time in enumerate(t):
        for j, (label, x) in enumerate(zip(preds, states(values[:, i]), strict=True)):
            model.validate_state_constraints(x)
            obs = model.observe(time, x, pars, forcings[int(j > 0)])
            for k in targets:
                preds[label][k].append(float(obs[k]))
    return preds


def summarize_channel(curves, saved, reference, time, target, label):
    """Compare reference errors and changes within the same fitted model."""
    primary, check = curves
    control = np.asarray(primary["control"][target])
    intact = np.asarray(primary["intact"][target])
    clamped = np.asarray(primary[label][target])
    effect = intact - control
    remaining = clamped - control
    ref_control, ref_treat = (np.asarray(r[target]) for r in reference)
    truth = ref_treat - ref_control
    saved_effect = np.asarray(saved[1]["curves"][target]) - saved[0]["curves"][target]
    verify = np.asarray(check[label][target]) - check["control"][target]
    verify_full = np.asarray(check["intact"][target]) - check["control"][target]
    floor = 1e-8 * max(1.0, np.max(abs(ref_control)), np.max(abs(control)))
    magnitude = max(floor, np.max(abs(truth)), np.max(abs(effect)))
    scale = max(magnitude, np.max(abs(remaining)))
    gap = max(
        np.max(abs(verify - remaining)) / scale,
        np.max(abs(verify_full - effect)) / magnitude,
        np.max(abs(saved_effect - effect)) / magnitude,
    )
    abs_gap = max(
        np.max(abs(np.asarray(saved[i]["curves"][target]) - primary[k][target]))
        / max(1.0, np.max(abs(np.asarray(saved[i]["curves"][target]))))
        for i, k in enumerate(("control", "intact"))
    )
    record = {
        "target": target,
        "condition": label,
        "effect_solver_or_replay_gap": gap,
        "absolute_replay_gap": abs_gap,
    }
    if gap > EFFECT_TOL or abs_gap > REPLAY_TOL:
        return {**record, "status": "numerical_disagreement"}

    def rms(a):
        return float(np.sqrt(np.mean(np.square(a))))

    truth_rms = rms(truth)
    effect_rms = rms(effect)
    return {
        **record,
        "status": "complete",
        "reference_signed_peak": float(truth[np.argmax(abs(truth))]),
        "intact_signed_peak": float(effect[np.argmax(abs(effect))]),
        "clamped_signed_peak": float(remaining[np.argmax(abs(remaining))]),
        "intact_effect_rse": (rms(effect - truth) / truth_rms) ** 2
        if truth_rms > floor
        else None,
        "clamped_effect_rse": (rms(remaining - truth) / truth_rms) ** 2
        if truth_rms > floor
        else None,
        "intact_effect_rms": effect_rms,
        "clamped_effect_rms": rms(remaining),
        "clamp_change_rms": rms(remaining - effect),
        "clamped_to_intact_rms": rms(remaining) / effect_rms
        if effect_rms > floor
        else None,
        "clamped_auc": float(np.trapezoid(remaining, time)),
        "intact_auc": float(np.trapezoid(effect, time)),
    }


def produce(job):
    """Save each preassigned intervention result atomically and retain failures."""
    root, out, index, contrast, row, plan = job
    path = out / "jobs" / f"{index:04d}-{contrast}.json"
    with public._lock(path.with_suffix("")):
        identity = {
            "plan": plan["artifact_sha256"],
            "subject": index,
            "contrast": contrast,
        }
        if path.exists():
            result = sealed_read(path)
            if any(result[k] != v for k, v in identity.items()):
                raise ValueError("result belongs to another assay")
            return result
        cell = row["benchmark_id"]
        target_names = assay.fits.TARGETS[assay.base_cell(cell)]
        model_hash = row.get("source_model_sha256") or row.get("model_sha256")
        meta = dict(
            identity,
            model=model_hash,
            source_id=row["source_id"],
            method=row["method"],
            seed=row["repetition"],
            cell=cell,
        )

        def unavailable(status):
            return sealed_write(
                path,
                {
                    **meta,
                    "latents": None,
                    "solver_target_curves": [],
                    "rows": [
                        {"condition": "all_latents", "target": k, "status": status}
                        for k in target_names
                    ],
                },
            )

        if row["status"] != "ready":
            return unavailable(row["status"])
        refs, saved = [], []
        for case in (plan["controls"][cell][contrast], contrast):
            ref = sealed_read(root / "references" / cell / (case + ".json"))
            curve = sealed_read(root / "subjects" / f"{index:04d}" / (case + ".json"))
            if any(r["plan"] != plan["source"] for r in (ref, curve)):
                raise ValueError("rollouts belong to another panel")
            if (
                ref["artifact_sha256"] != plan["references"][f"{cell}/{case}"]
                or curve["artifact_sha256"] != plan["curves"][f"{index}/{case}"]
            ):
                raise ValueError("saved numerical input differs from the frozen plan")
            refs.append(ref["reference"])
            saved.append(curve)
        if any("error" in curve for curve in saved):
            return unavailable("intact_replay_unavailable")
        candidate = CandidateModel.model_validate(row["candidate"])
        model = compile_candidate(
            candidate, ValidationContext.model_validate(row["context"])
        )
        latents = {
            i: state.name
            for i, state in enumerate(candidate.states)
            if state.kind.value == "latent"
            and state.name not in model.direct_state_observation_channels
        }
        groups = {"intact": [], "all_latents": list(latents)}
        if row["method"] == FULL:
            groups.update({n: [i] for i, n in latents.items()})
        tx = [assay.trajectory(cell, ref) for ref in refs]
        rows, traces = [], []
        try:
            traces = [
                simulate_branches(
                    model, row["parameters"], row["initials"], *tx, groups, solver, 150
                )
                for solver in SOLVERS
            ]
            for label in groups:
                if label != "intact":
                    for target in target_names:
                        rows.append(
                            summarize_channel(
                                traces, saved, refs, tx[0].time, target, label
                            )
                        )
        except (
            RuntimeExpressionError,
            RuntimeError,
            ArithmeticError,
            ValueError,
            TimeoutError,
        ) as e:
            rows = [
                {
                    "target": k,
                    "condition": label,
                    "status": "simulation_failed",
                    "detail": str(e),
                }
                for label in groups
                if label != "intact"
                for k in target_names
            ]
        return sealed_write(
            path,
            {
                **meta,
                "latents": list(latents.values()),
                "rows": rows,
                "solver_target_curves": traces,
            },
        )


def run(root: Path, out: Path, workers: int, search: Path | None = None) -> None:
    """Use the archived development panel or the final paired confirmation subjects."""
    if workers < 1:
        raise ValueError("positive worker count required")
    panel = sealed_read(root / "frozen_selection_manifest.json")
    if search is None:
        if panel["artifact_sha256"] != PANEL or panel["mode"] != "development":
            raise ValueError("requires the original retrospective intervention panel")
        subjects = {
            i: r
            for i, r in enumerate(panel["roster"])
            if r["benchmark_id"] == assay.LEGACY
        }
        if len(subjects) != 16:
            raise ValueError("historical cohort incomplete")
        contrasts = {assay.LEGACY: CONTRASTS}
        bundle_hash = None
    else:
        from scripts.iclr_specification_results import confirmed_scores

        source = search / "frozen_models.json"
        bundle = sealed_read(source)
        if bundle["protocol"] != "iclr-controlled-search-models-1":
            raise ValueError("requires a frozen controlled-search bundle")
        assay.verify_controlled_bundle(source, bundle)
        selected = {r["source_id"]: r for r in bundle["rows"]}
        bundle_hash = bundle["artifact_sha256"]
        verified, _ = confirmed_scores(root, [bundle_hash], selected)
        subjects = {
            i: r
            for i, r in enumerate(panel["roster"])
            if r["source_bundle_sha256"] == bundle_hash
            and r["method"] in ("full", "fixed_gamma")
        }
        contrasts = {
            assay.fits.T2: CONTRASTS,
            assay.fits.CSTR: ("feed_high", "jacket_warm"),
        }
        expected = {
            (c, seed, arm)
            for c in contrasts
            for seed in range(3)
            for arm in ("full", "fixed_gamma")
        }
        keys = {
            (r["benchmark_id"], r["repetition"], r["method"]) for r in subjects.values()
        }
        if (
            len(subjects) != 12
            or keys != expected
            or any(r["source_id"] not in verified for r in subjects.values())
        ):
            raise ValueError("incomplete confirmatory cohort")
    controls = {
        cell: {case: panel["schedules"][cell][case]["control"] for case in cases}
        for cell, cases in contrasts.items()
    }
    ref_ids, curve_ids = {}, {}
    for i, r in subjects.items():
        cell = r["benchmark_id"]
        for k in set(controls[cell].values()) | set(contrasts[cell]):
            ref_ids[f"{cell}/{k}"] = sealed_read(
                root / "references" / cell / (k + ".json")
            )["artifact_sha256"]
            if r["status"] == "ready":
                curve_ids[f"{i}/{k}"] = sealed_read(
                    root / "subjects" / f"{i:04d}" / (k + ".json")
                )["artifact_sha256"]
    plan = sealed_write(
        out / "plan.json",
        {
            "protocol": "iclr-learned-latent-path-clamp-1",
            "source": panel["artifact_sha256"],
            "mode": panel["mode"],
            "search_bundle": bundle_hash,
            "code_sha256": digest(Path(__file__)),
            "solver_code": assay.source_hash(),
            "runtime": public._runtime(),
            "solvers": list(SOLVERS),
            "effect_tolerance": EFFECT_TOL,
            "replay_tolerance": REPLAY_TOL,
            "policy": "Joint clamps for all; single clamps for historical Full seeds.",
            "contrasts": {c: list(items) for c, items in contrasts.items()},
            "controls": controls,
            "references": ref_ids,
            "curves": curve_ids,
        },
    )
    jobs = [
        (root, out, i, k, r, plan)
        for i, r in subjects.items()
        for k in contrasts[r["benchmark_id"]]
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(produce, jobs))
    rows = [
        {
            k: v
            for k, v in r.items()
            if k
            in ("subject", "contrast", "source_id", "method", "seed", "model", "cell")
        }
        | x
        for r in results
        for x in r["rows"]
    ]
    sealed_write(
        out / "summary.json",
        {
            "plan": plan["artifact_sha256"],
            "rows": rows,
            "job_seals": [r["artifact_sha256"] for r in results],
        },
    )
    table(out / "latent_clamp.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fidelity", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--search", type=Path, help="production anchor root for confirmation"
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.fidelity, args.output, args.workers, args.search)
