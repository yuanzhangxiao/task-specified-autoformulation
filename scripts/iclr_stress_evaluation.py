"""Post-freeze rollout and failure-aware reporting for acquisition/scaling controls."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.benchmarks.phase_b_generation import (
    _simulate_alien,
    phase_b_protocols,
    simulate_phase_b,
)
from autoformalism.benchmarks.phase_b_public import _private_array
from autoformalism.benchmarks.phase_c_protocols import phase_c_protocols
from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.expressions.diagnostics import (
    ModelValidationError,
    RuntimeExpressionError,
)
from autoformalism.fitting import (
    FitConfig,
    simulate_trajectory,
)
from autoformalism.fitting import (
    public_fitting as public,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.reference_integration import ReferenceSolver
from autoformalism.schemas import CandidateModel
from scripts import iclr_active_history as complexity
from scripts import iclr_stress_data as data


class SolverFailure(ValueError):
    """A frozen forecast integration explicitly failed."""


class SolverDisagreement(ValueError):
    """Independent trajectories differed beyond the declared numerical tolerance."""


def build_references(root: Path, cell: dict, data_root: Path) -> list[dict]:
    """Generate paired private truths after verifying the endpoint freeze."""
    family = cell["family"]
    spec = read_seal(root / "public" / cell["cell"] / "specification.json")
    if family == "alien":
        eq = read_seal(root / "evaluator" / cell["cell"] / "system.json")["equations"]
        if public.content_sha256(eq) != cell["realization_sha256"]:
            raise ValueError("private system no longer matches prepared cell")
        protocols = data.device_protocols("test")
    elif cell["cell"] == data.CORRECTED:
        protocols = phase_b_protocols("cstr")
    else:
        protocols = (
            phase_c_protocols("dalla_man", task="T2")
            if family == "t2"
            else phase_c_protocols("cstr")
        )
    records = []
    for p in (p for p in protocols if p.split == "test"):
        quiet = {
            "t2": {"meals": []},
            "cstr": {"kind": "constant", "value": [0, 0, 0]},
            "alien": {"kind": "constant", "value": 0.0},
        }[family]
        if "initial_shift" in p.specification:
            quiet["initial_shift"] = p.specification["initial_shift"]
        pair = []
        for condition in (p, p.model_copy(update={"specification": quiet})):

            def run(solver, condition=condition):
                if family == "alien":
                    return _simulate_alien(condition, eq, {}, solver)
                return simulate_phase_b(condition, data_root=data_root, solver=solver)

            real = run(data.SOLVER)
            check = run(
                ReferenceSolver(method="Radau", max_step=0.4, rtol=1e-9, atol=1e-11)
            )
            if not np.allclose(real.states, check.states, rtol=2e-6, atol=2e-6):
                raise ValueError(
                    f"reference solvers disagree on {condition.protocol_id}"
                )
            row = {"trajectory_id": p.protocol_id, "time": real.time.tolist()}
            for role in data.ROLES:
                row[role] = {
                    c: _private_array(real, {"u01": "u", "v01": "y"}.get(c, c)).tolist()
                    for c in spec[role]
                }
            pair.append(row)
        records.append(
            {"case": p.protocol_id, "perturbed": pair[0], "control": pair[1]}
        )
    return records


def public_input(row: dict, cell: dict) -> Trajectory:
    """Keep noisy initial targets and interpolate acquired auxiliaries."""
    sampled = data.acquire(
        row,
        scales=cell.get("scales", {}),
        fraction=cell.get("noise", 0),
        stride=cell.get("stride", 1),
        mask=cell.get("mask"),
    )
    t = np.asarray(row["time"])
    return Trajectory(
        trajectory_id=row["trajectory_id"],
        time=t,
        targets={c: np.full_like(t, v[0]) for c, v in sampled["targets"].items()},
        auxiliaries={
            c: np.interp(t, sampled["time"], v)
            for c, v in sampled["auxiliaries"].items()
        },
        external_inputs={c: np.asarray(v) for c, v in row["external_inputs"].items()},
        fixed_covariates={},
        derivatives={},
    )


def predict(endpoint: dict, cell: dict, pair: dict) -> tuple[dict, dict]:
    """Run the frozen model on both preparations with two solvers."""
    context = ValidationContext.model_validate(endpoint["context"])
    roles = pair["perturbed"]
    if (
        set(context.targets) != set(roles["targets"])
        or set(context.auxiliaries) != set(roles["auxiliaries"])
        or set(context.external_inputs) != set(roles["external_inputs"])
        or context.lagged_targets
        or context.fixed_covariates
    ):
        raise ValueError("frozen model and public input contracts differ")
    model = compile_candidate(
        CandidateModel.model_validate(endpoint["candidate"]), context
    )
    results = []
    for key in ("perturbed", "control"):
        trajectory = public_input(pair[key], cell)
        curves = []
        for solver in ("Radau", "BDF"):
            value = simulate_trajectory(
                model,
                trajectory,
                endpoint["parameters"],
                endpoint["initials"],
                FitConfig(
                    integration_method=solver,
                    relative_tolerance=1e-8,
                    absolute_tolerance=1e-10,
                ),
                deadline=monotonic() + 120,
                reset_observed_states=False,
            )
            if not value.success:
                raise SolverFailure(f"{solver} {key}: {value.message}")
            curves.append(value.predictions)
        results.append(curves)
    primary, control = results
    for c in context.targets:
        obs = np.asarray(roles["targets"][c])
        abs_gap = np.max(np.abs(primary[0][c] - primary[1][c]))
        effect = primary[0][c] - control[0][c]
        effect_gap = np.max(np.abs(effect - (primary[1][c] - control[1][c])))
        truth_effect = obs - pair["control"]["targets"][c]
        if abs_gap > 1e-4 * max(
            1.0, max(abs(obs)), max(abs(primary[0][c]))
        ) or effect_gap > 5e-3 * max(1e-8, max(abs(effect)), max(abs(truth_effect))):
            raise SolverDisagreement(f"paired solver disagreement on {c}")
    return primary[0], control[0]


def score(
    endpoint: dict, cell: dict, refs: list, scales: dict, predictor=predict
) -> dict:
    """Measure clean-target and paired response errors per trajectory."""
    if not refs:
        raise ValueError("empty frozen intervention design")
    records = []
    start = monotonic()
    for pair in refs:
        try:
            p, q = predictor(endpoint, cell, pair)
            channels = {}
            for c, vals in pair["perturbed"]["targets"].items():
                truth = np.asarray(vals)
                effect = truth - pair["control"]["targets"][c]
                t = np.asarray(pair["perturbed"]["time"])
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    denom = float(np.trapezoid(effect**2, t))
                    sq = float(np.mean((p[c] - truth) ** 2))
                    diff = p[c] - q[c]
                    peak = int(np.argmax(abs(effect)))
                    pred_peak = int(np.argmax(abs(diff)))
                    magnitude = max(abs(truth).max(), 1.0) * 1e-8
                    active = abs(effect[peak]) > magnitude
                    pred_active = abs(diff[pred_peak]) > magnitude
                    channels[c] = {
                        "clean_mse": sq,
                        "nmse": sq / scales[c] ** 2,
                        "clean_scale_nmse": sq / cell.get("scales", scales)[c] ** 2,
                        "response_rse": float(
                            np.trapezoid((diff - effect) ** 2, t) / denom
                        )
                        if active
                        else None,
                        "effect_status": "complete"
                        if active
                        else "negligible_reference",
                        "reference_area": float(np.trapezoid(effect, t)),
                        "predicted_area": float(np.trapezoid(diff, t)),
                        "signed_peak_direction_correct": (
                            bool(
                                pred_active
                                and np.sign(diff[pred_peak]) == np.sign(effect[peak])
                            )
                            if active
                            else None
                        ),
                        "reference_signed_peak": float(effect[peak]),
                        "predicted_signed_peak": float(diff[pred_peak]),
                        "peak_timing_error": float(t[pred_peak] - t[peak])
                        if active and pred_active
                        else None,
                    }
            records.append(
                {"case": pair["case"], "status": "complete", "channels": channels}
            )
        except (
            ArithmeticError,
            ValueError,
            RuntimeError,
            TimeoutError,
            ModelValidationError,
            RuntimeExpressionError,
        ) as exc:
            kind = "unclassified_evaluation_error"
            if isinstance(exc, SolverFailure):
                kind = "solver_failure"
            elif isinstance(exc, SolverDisagreement):
                kind = "solver_disagreement"
            records.append(
                {
                    "case": pair["case"],
                    "status": "unresolved",
                    "reason": str(exc),
                    "failure_kind": kind,
                }
            )
    all_ok = all(r["status"] == "complete" for r in records)
    effects = [
        c["response_rse"]
        for r in records
        for c in r.get("channels", {}).values()
        if c["response_rse"] is not None
    ]
    return {
        "evaluation_status": "complete" if all_ok else "unresolved",
        "test_nmse": float(
            np.mean([c["nmse"] for r in records for c in r["channels"].values()])
        )
        if all_ok
        else None,
        "common_scale_test_nmse": float(
            np.mean(
                [c["clean_scale_nmse"] for r in records for c in r["channels"].values()]
            )
        )
        if all_ok
        else None,
        "response_rse": float(np.mean(effects)) if all_ok and effects else None,
        "evaluation_seconds": monotonic() - start,
        "trajectories": records,
    }


def reference_predict(endpoint: dict, cell: dict, pair: dict) -> tuple[dict, dict]:
    """Apply the frozen clean-anchor reference fit to the acquired measurements."""
    from scripts import iclr_reference_fits as model

    anchor = endpoint["benchmark_id"]
    curves = []
    for case in ("perturbed", "control"):
        tr = public_input(pair[case], cell)
        names = model.TARGETS[anchor]
        initial = [tr.targets[k][0] for k in names]
        knots = model.public_inputs(anchor, tr)
        predictions = []
        for substeps in (4, 8):
            fn = model.family(anchor, knots.shape[1], substeps, jacobian=False)
            predictions.append(
                np.asarray(fn(initial, knots, endpoint["fit"]["theta"]))[: len(names)]
            )
        if not np.isfinite(predictions).all() or not np.allclose(
            *predictions, rtol=1e-4, atol=1e-5
        ):
            raise SolverDisagreement(
                "reference integration failed its refinement check"
            )
        curves.append(dict(zip(names, predictions[0], strict=True)))
    return tuple(curves)


def select_endpoints(manifest: dict, bundle: dict) -> list[tuple[dict, dict]]:
    """Require three paired seeds per evaluated cell in a terminal freeze."""
    if (
        bundle.get("protocol") != "iclr-controlled-search-models-1"
        or not bundle.get("plan_sha256")
        or not bundle.get("selection_freeze_sha256")
        or bundle.get("test_data_opened") is not False
        or bundle.get("private_reference_opened") is not False
    ):
        raise ValueError("expected a pre-test sealed controlled search model export")
    cells = {c["cell"]: c for c in manifest["cells"]}
    seen = set()
    result = []
    for r in bundle["rows"]:
        key = (r["benchmark_id"], r["method"], r["repetition"])
        if key[0] not in cells or key[1] not in manifest["arms"]:
            continue
        if key in seen or key[2] not in manifest["algorithm_seeds"]:
            raise ValueError("duplicate or unplanned model")
        cell = cells[key[0]]
        if r.get("public_file_sha256") != cell["public_files"]:
            raise ValueError("endpoint did not train on the prepared public files")
        if r.get("public_spec_sha256") != cell.get("public_spec_sha256"):
            raise ValueError("endpoint public specification differs")
        if r["status"] in {"pending", "running", "planned"}:
            raise ValueError("search still in progress")
        seen.add(key)
        result.append((cell, r))
    required = {
        (k, a, s)
        for k in {k for k, _, _ in seen}
        for a in manifest["arms"]
        for s in manifest["algorithm_seeds"]
    }
    if not result or seen != required:
        raise ValueError("missing paired arm or seed in frozen cell")
    return result


def evaluate(
    root: Path,
    manifest: dict,
    models: Path,
    data_root: Path,
    out: Path,
    reference_root: Path | None = None,
) -> None:
    """Checkpoint models and paired physical curves after freezing."""
    public_root = (root / "public").resolve()
    if out.resolve().is_relative_to(public_root):
        raise ValueError("private evaluation cannot write under development/public")
    from scripts.iclr_anchor_fidelity import verify_controlled_bundle

    bundle = sealed_read(models)
    verify_controlled_bundle(models, bundle)
    selected = select_endpoints(manifest, bundle)
    plan = sealed_write(
        out / "freeze.json",
        {
            "design_sha256": manifest["artifact_sha256"],
            "search_sha256": bundle["artifact_sha256"],
            "evaluator_sha256": data.digest(Path(__file__)),
            "complexity_source_sha256": data.digest(Path(complexity.__file__)),
            "source_sha256": public._source_identity(),
            "runtime": public._runtime(),
            "reference_cstr_sha256": data.digest(
                data_root
                / "benchmark5_anonymous_nonlinear_process/private"
                / "system_specification.json"
            )
            if any(c["family"] == "cstr" for c, _ in selected)
            else None,
        },
    )
    ref_models = {}
    if reference_root is not None:
        from scripts import iclr_reference_fits as fit

        fitted = sealed_read(reference_root / "frozen_models.json")
        origin = sealed_read(reference_root / "plan.json")
        if (
            fitted["protocol"] != fit.VERSION
            or fitted["plan_sha256"] != origin["artifact_sha256"]
        ):
            raise ValueError("invalid reference-structure development seal")
        expected_sources = sealed_read(root / "source_files.json")
        if expected_sources["design_sha256"] != manifest["design_sha256"]:
            raise ValueError("source data ledger belongs to another design")
        condition_ledger = reference_root / "conditions.json"
        condition_sources = {}
        if condition_ledger.exists():
            projected = sealed_read(condition_ledger)
            if projected["design_sha256"] != manifest["artifact_sha256"]:
                raise ValueError("conditioned reference roster differs")
            condition_sources = {r["fit_family"]: r for r in projected["rows"]}
        for row in fitted["rows"]:
            anchor = row["benchmark_id"]
            if anchor not in (data.T2, data.CSTR.format("hard")):
                continue
            if origin["files"][anchor] == expected_sources["cells"][anchor]:
                ref_models[anchor] = row
            elif anchor in condition_sources:
                projection = condition_sources[anchor]
                if origin["files"][anchor] != projection["files"]:
                    raise ValueError("conditioned reference fit has wrong inputs")
                candidate_cell = next(
                    c for c in manifest["cells"] if c["cell"] == projection["cell"]
                )
                if any(
                    projection["files"][f] != candidate_cell["public_files"][f]
                    for f in ("train.json", "val.json")
                ):
                    raise ValueError("conditioned reference public splits differ")
                ref_models[candidate_cell["cell"]] = row
            else:
                raise ValueError("reference fit used a different development release")
        sealed_write(
            out / "reference_fit_binding.json",
            {
                "fit_bundle": fitted["artifact_sha256"],
                "freeze_sha256": plan["artifact_sha256"],
            },
        )
    results = []
    for cell, r in selected:
        key = cell["cell"]
        path = out / "results" / f"{key}-{r['method']}-{r['repetition']}.json"
        if not path.exists():
            folder = root / "public" / key
            for name, sha in cell["public_files"].items():
                if data.digest(folder / name) != sha:
                    raise ValueError("development input identity changed")
            record = {
                "benchmark_id": key,
                "method": r["method"],
                "repetition": r["repetition"],
                "freeze_sha256": plan["artifact_sha256"],
                "search_status": r["status"],
                "validation_nmse": r.get("validation_nmse"),
                "cost": r.get("cost", {}),
                "evaluation_status": "model_unavailable",
                "source_id": r.get("source_id"),
                "model_sha256": r.get("model_sha256"),
                "last_visit_status": r.get("last_visit_status"),
                "optimizer_converged": r.get("optimizer_converged"),
                **complexity.endpoint_complexity(r),
            }
            if r["status"] == "ready":
                ref = out / "references" / f"{key}.json"
                if not ref.exists():
                    seal(
                        ref,
                        {
                            "freeze_sha256": plan["artifact_sha256"],
                            "records": build_references(root, cell, data_root),
                        },
                    )
                saved = read_seal(ref)
                if saved["freeze_sha256"] != plan["artifact_sha256"]:
                    raise ValueError("reference freeze identity differs")
                scales = data.train_scales(read_seal(folder / "train.json")["rows"])
                if any(
                    scales[c] <= 0 for c in saved["records"][0]["perturbed"]["targets"]
                ):
                    raise ValueError("target has no training variation")
                begin = monotonic()
                record.update(score(r, cell, saved["records"], scales))
                record["evaluation_seconds"] = monotonic() - begin
            sealed_write(path, record)
        record = sealed_read(path)
        if record["freeze_sha256"] != plan["artifact_sha256"]:
            raise ValueError("result belongs to another freeze")
        results.append(record)
    # A shared reference fit is a diagnostic across altered measurements; it is
    # not an independent search replicate. Require the same forecast gate first.
    for key in sorted({c["cell"] for c, _ in selected}):
        c = next(c for c, _ in selected if c["cell"] == key)
        anchor = c.get("source")
        ref_path = out / "references" / f"{key}.json"
        source_key = key if key in ref_models else anchor
        if source_key not in ref_models:
            continue
        if not ref_path.exists():
            seal(
                ref_path,
                {
                    "freeze_sha256": plan["artifact_sha256"],
                    "records": build_references(root, c, data_root),
                },
            )
        path = out / "privileged-references" / (key + ".json")
        if not path.exists():
            row = ref_models[source_key]
            val = {"evaluation_status": "model_unavailable"}
            if row["status"] == "ready":
                scales = data.train_scales(
                    read_seal(root / "public" / key / "train.json")["rows"]
                )
                val = score(
                    row, c, read_seal(ref_path)["records"], scales, reference_predict
                )
            sealed_write(
                path,
                {
                    "cell": key,
                    "method": (
                        "condition_fitted_reference"
                        if source_key == key
                        else "clean_anchor_fitted_reference"
                    ),
                    "fit_bundle": fitted["artifact_sha256"],
                    **val,
                },
            )
    sealed_write(
        out / "evaluation.json",
        {
            "protocol": "iclr-stress-evaluation-1",
            "manifest_sha256": manifest["artifact_sha256"],
            "freeze_sha256": plan["artifact_sha256"],
            "search_bundle_sha256": bundle["artifact_sha256"],
            "rows": results,
        },
    )


def write_csv(path: Path, rows: list, columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def numerical_coverage(record: dict) -> dict:
    """Separate unknown outcomes from the measured failure denominators."""
    status = record.get("search_status")
    optimizer = record.get("optimizer_converged")
    if type(optimizer) is bool:
        fit_status = "converged" if optimizer else "reported_nonconvergence"
    elif status == "ready":
        fit_status = "not_reported_in_endpoint"
    elif status is None:
        fit_status = "awaiting_frozen_model"
    else:
        fit_status = "no_selected_fit"
    pairs = record.get("trajectories", [])
    errors = Counter(
        r.get("failure_kind", "unclassified_evaluation_error")
        for r in pairs
        if r["status"] != "complete"
    )
    ok = sum(r["status"] == "complete" for r in pairs)
    attempted = ok + errors["solver_failure"] + errors["solver_disagreement"]
    return {
        "optimizer_status": fit_status,
        "issued_test_pairs": len(pairs) if "trajectories" in record else None,
        "solver_classified_pairs": attempted if pairs else None,
        "solver_failure_pairs": errors["solver_failure"] if pairs else None,
        "solver_disagreement_pairs": errors["solver_disagreement"] if pairs else None,
        "unclassified_error_pairs": errors["unclassified_evaluation_error"]
        if pairs
        else None,
    }


def report(manifest: dict, evaluations: list[Path], out: Path) -> None:
    """Tabulate the complete frozen denominator; plot endpoints only when measured."""
    out.mkdir(parents=True, exist_ok=True)
    found = {}
    for path in evaluations:
        result = sealed_read(path)
        if (
            result["protocol"] != "iclr-stress-evaluation-1"
            or result["manifest_sha256"] != manifest["artifact_sha256"]
        ):
            raise ValueError("evaluation belongs to a different experiment")
        for r in result["rows"]:
            key = (r["benchmark_id"], r["method"], r["repetition"])
            if key in found:
                raise ValueError("duplicate evaluated subject")
            found[key] = r
    cells = {c["cell"]: c for c in manifest["cells"]}
    rows = []
    costs = []
    effects = []
    for run in manifest["runs"]:
        id = run["benchmark_id"]
        arm = run["arm"]
        seed = run["repetition"]
        c = cells[id]
        record = found.pop((id, arm, seed), {})
        row = {
            **run,
            **{
                k: c.get(k)
                for k in (
                    "study",
                    "dimension",
                    "generator_seed",
                    "variant",
                    "noise",
                    "stride",
                )
            },
            "mask": "+".join(c.get("mask", [])),
            "search_status": record.get("search_status", "unavailable"),
            **numerical_coverage(record),
            "complexity_status": record.get(
                "complexity_status",
                ("not_recorded_on_endpoint" if record else "awaiting_frozen_model"),
            ),
            "evaluation_status": record.get(
                "evaluation_status", c.get("status", "awaiting_frozen_model")
            ),
            **{
                k: record.get(k)
                for k in (
                    "test_nmse",
                    "common_scale_test_nmse",
                    "response_rse",
                    "validation_nmse",
                    "state_count",
                    "parameter_count",
                    "target_reachable_state_count",
                    "target_reachable_latent_count",
                    "expression_nodes",
                    "source_id",
                    "model_sha256",
                    "last_visit_status",
                )
            },
        }
        rows.append(row)
        costs.append(
            {
                **run,
                "evaluation_seconds": record.get("evaluation_seconds"),
                **{
                    k: json.dumps(v) if isinstance(v, (list, dict)) else v
                    for k, v in record.get("cost", {}).items()
                },
            }
        )
        for sample in record.get("trajectories", []):
            if sample["status"] != "complete":
                effects.append(
                    {
                        **run,
                        "case": sample["case"],
                        "channel": None,
                        "trajectory_status": sample["status"],
                        "failure_kind": sample.get(
                            "failure_kind", "unclassified_evaluation_error"
                        ),
                        "reason": sample.get("reason"),
                    }
                )
            for channel, score in sample.get("channels", {}).items():
                effects.append(
                    {
                        **run,
                        "case": sample["case"],
                        "channel": channel,
                        "trajectory_status": sample["status"],
                        **score,
                    }
                )
    if found:
        raise ValueError("evaluation contains unplanned identity")
    complete = any(r["evaluation_status"] == "complete" for r in rows)
    have_effect = any(r["response_rse"] is not None for r in rows)
    have_scaling = any(
        r["study"] == "scaling" and r["evaluation_status"] == "complete" for r in rows
    )
    measured_variants = any(
        r["variant"] in {"double_recurrent_coupling", "wider_decay_spread"}
        and r["evaluation_status"] == "complete"
        for r in rows
    )
    files = {
        "stress-curves.pdf": complete,
        "stress-effect-curves.pdf": have_effect,
        "scaling-cost-curves.pdf": have_scaling,
        "fixed-dimension-sensitivity.pdf": measured_variants,
    }
    if any(not measured and (out / name).exists() for name, measured in files.items()):
        raise ValueError("use a fresh report directory for an unevaluated roster")
    write_csv(out / "stress_metrics.csv", rows, list(rows[0]))
    write_csv(
        out / "stress_costs.csv",
        costs,
        list(dict.fromkeys(k for r in costs for k in r)),
    )
    write_csv(
        out / "stress_effects.csv",
        effects,
        [
            "benchmark_id",
            "arm",
            "repetition",
            "case",
            "channel",
            "clean_mse",
            "nmse",
            "response_rse",
            "reference_signed_peak",
            "predicted_signed_peak",
            "peak_timing_error",
            "clean_scale_nmse",
            "effect_status",
            "reference_area",
            "predicted_area",
            "signed_peak_direction_correct",
            "trajectory_status",
            "failure_kind",
            "reason",
        ],
    )
    for study, filename in (
        ("data_quality", "data_robustness_metrics"),
        ("scaling", "scaling_metrics"),
        ("corrected_easy", "corrected_easy_metrics"),
    ):
        write_csv(
            out / (filename + ".csv"),
            [r for r in rows if r["study"] == study],
            list(rows[0]),
        )
    scaling_ids = {c["cell"] for c in cells.values() if c["study"] == "scaling"}
    write_csv(
        out / "scaling_costs.csv",
        [r for r in costs if r["benchmark_id"] in scaling_ids],
        list(dict.fromkeys(k for r in costs for k in r)),
    )
    by_pair = defaultdict(dict)
    for row in rows:
        by_pair[(row["benchmark_id"], row["repetition"])][row["arm"]] = row
    paired_rows = []
    for (cell, seed), treatments in by_pair.items():
        a, b = (treatments.get(arm, {}) for arm in ("full", "fixed_gamma"))
        valid = a.get("evaluation_status") == b.get("evaluation_status") == "complete"
        paired_rows.append(
            {
                "benchmark_id": cell,
                "repetition": seed,
                "scored_pair": valid,
                "full_status": a.get("evaluation_status", "absent"),
                "fixed_status": b.get("evaluation_status", "absent"),
                "full_minus_fixed_nmse": a["test_nmse"] - b["test_nmse"]
                if valid
                else None,
                "full_minus_fixed_response_rse": a["response_rse"] - b["response_rse"]
                if valid and all(r.get("response_rse") is not None for r in (a, b))
                else None,
            }
        )
    write_csv(out / "paired_stress_results.csv", paired_rows, list(paired_rows[0]))
    groups = defaultdict(list)
    for r in rows:
        groups[r["benchmark_id"], r["arm"]].append(r)
    per_cell = []
    for (cell, arm), subject_rows in groups.items():
        ok = [r for r in subject_rows if r["evaluation_status"] == "complete"]
        enrolled = sum(r["solver_classified_pairs"] or 0 for r in subject_rows)
        failed = sum(r["solver_failure_pairs"] or 0 for r in subject_rows)
        disagreements = sum(r["solver_disagreement_pairs"] or 0 for r in subject_rows)
        opt = sum(
            r["optimizer_status"] in {"converged", "reported_nonconvergence"}
            for r in subject_rows
        )
        nonconverged = sum(
            r["optimizer_status"] == "reported_nonconvergence" for r in subject_rows
        )
        per_cell.append(
            {
                "benchmark_id": cell,
                "classified_solver_pairs": enrolled,
                "solver_failure_count": failed,
                "solver_disagreement_count": disagreements,
                "solver_failure_fraction_among_classified": failed / enrolled
                if enrolled
                else None,
                "solver_disagreement_fraction_among_classified": disagreements
                / enrolled
                if enrolled
                else None,
                "selected_fits_with_reported_convergence": opt,
                "reported_optimizer_nonconvergence_count": nonconverged,
                "nonconvergence_fraction_among_reported": nonconverged / opt
                if opt
                else None,
                "arm": arm,
                "complete": len(ok),
                "planned": len(subject_rows),
                "completion_fraction": len(ok) / len(subject_rows),
                "median_nmse_among_complete": float(
                    np.median([r["test_nmse"] for r in ok])
                )
                if ok
                else None,
            }
        )
    write_csv(out / "completion.csv", per_cell, list(per_cell[0]))
    counts = Counter(r["evaluation_status"] for r in rows)
    summary = {
        "planned": len(rows),
        "status": dict(counts),
        "evaluated_bundles": len(evaluations),
        "complete": counts["complete"],
        "cells": len(cells),
    }
    public._write(out / "summary.json", summary)
    if complete:
        plot(rows, out)
    if have_effect:
        plot(rows, out, "response_rse")
    if have_scaling:
        cost_plot(rows, costs, out)
    if measured_variants:
        fixed_dimension_plot(rows, costs, out)


def plot(rows: list[dict], out: Path, metric: str = "test_nmse") -> None:
    """Retain each seed and procedural draw on the reported response curves."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
    for ax, (label, key) in zip(
        axes.flat,
        (
            ("Noise (training SD fraction)", "noise"),
            ("Sampling stride", "stride"),
            ("Supplied CSTR sensors", "mask"),
            ("Total simulator states", "dimension"),
        ),
        strict=True,
    ):
        for arm, marker in (("full", "o"), ("fixed_gamma", "x")):
            subset = [
                r
                for r in rows
                if r["arm"] == arm
                and r["evaluation_status"] == "complete"
                and r[metric] is not None
            ]
            if key == "noise":
                subset = [
                    r
                    for r in subset
                    if r["benchmark_id"].startswith("acquisition_") and r["stride"] == 1
                ]
            elif key == "stride":
                subset = [
                    r
                    for r in subset
                    if r["benchmark_id"].startswith("acquisition_") and r["noise"] == 0
                ]
            elif key == "mask":
                subset = [r for r in subset if r["benchmark_id"].startswith("sensor_")]
            else:
                subset = [r for r in subset if r["variant"] == "dimension"]
            points = defaultdict(list)
            for r in subset:
                points[(r.get("generator_seed"), r["repetition"])].append(
                    (
                        r[key]
                        if key != "mask"
                        else {"": 0, "C": 1, "Tj": 2, "C+Tj": 3}[r[key]],
                        r[metric],
                    )
                )
            for i, series in enumerate(points.values()):
                xx, yy = zip(*sorted(series), strict=True)
                ax.plot(xx, yy, marker=marker, alpha=0.7, label=arm if i == 0 else None)
        if key == "mask":
            ax.set_xticks(range(4), ["none", "C", "Tj", "C+Tj"])
        ax.set_xlabel(label)
        ax.set_ylabel(
            "Clean-target NMSE"
            if metric == "test_nmse"
            else "Intervention response RSE"
        )
        if ax.lines:
            ax.legend(fontsize=8)
    fig.savefig(
        out
        / ("stress-curves.pdf" if metric == "test_nmse" else "stress-effect-curves.pdf")
    )
    plt.close(fig)


def cost_plot(rows: list[dict], costs: list[dict], root: Path) -> None:
    """Show measured work and error per generator draw and algorithm seed."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figures, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for ax, (field, title) in zip(
        axes.flat,
        (
            ("visit_seconds", "Search wall time (s)"),
            ("observed_proposal_tokens", "Recorded proposal tokens"),
            ("state_count", "Selected state count"),
            ("target_reachable_state_count", "Target-reachable state count"),
            ("expression_nodes", "Executable expression nodes"),
            ("response_rse", "Intervention response RSE"),
        ),
        strict=True,
    ):
        groups = defaultdict(list)
        for row, cost in zip(rows, costs, strict=True):
            if row["variant"] != "dimension":
                continue
            value = row.get(field, cost.get(field))
            if value is not None:
                groups[row["arm"], row["repetition"], row["generator_seed"]].append(
                    (row["dimension"], value)
                )
        shown = set()
        for (arm, _, _), points in groups.items():
            xx, yy = zip(*sorted(points), strict=True)
            ax.plot(
                xx,
                yy,
                marker="o" if arm == "full" else "x",
                alpha=0.55,
                label=arm if arm not in shown else None,
            )
            shown.add(arm)
        ax.set_xlabel("Total simulator states")
        ax.set_ylabel(title)
        if ax.lines:
            ax.legend(fontsize=8)
    figures.savefig(root / "scaling-cost-curves.pdf")
    plt.close(figures)


def fixed_dimension_plot(rows: list[dict], costs: list[dict], out: Path) -> None:
    """Show the paired draw and seed for both one-factor, eight-state interventions."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    dims = [
        (r | {"visit_seconds": c.get("visit_seconds")})
        for r, c in zip(rows, costs, strict=True)
        if r["dimension"] == 8
    ]
    generators = sorted({r["generator_seed"] for r in dims})
    colors = {seed: f"C{i % 10}" for i, seed in enumerate(generators)}
    figure = Figure(figsize=(10, 10), layout="constrained")
    axes = figure.subplots(4, 2)
    metrics = (
        ("common_scale_test_nmse", "Common-scale test NMSE"),
        ("response_rse", "Intervention response RSE"),
        ("target_reachable_state_count", "Target-reachable state count"),
        ("visit_seconds", "Search visit time (s)"),
    )
    for col, (variant, label) in enumerate(
        (
            ("double_recurrent_coupling", "Double recurrent coupling"),
            ("wider_decay_spread", "Wider decay spread"),
        )
    ):
        axes[0, col].set_title(label)
        for i, (metric, title) in enumerate(metrics):
            ax = axes[i, col]
            series = defaultdict(list)
            for r in dims:
                if r["variant"] not in {"dimension", variant} or r.get(metric) is None:
                    continue
                series[r["generator_seed"], r["repetition"], r["arm"]].append(
                    (int(r["variant"] == variant), r[metric])
                )
            for (draw, seed, arm), points in series.items():
                x, y = zip(*sorted(points), strict=True)
                ax.plot(
                    x,
                    y,
                    color=colors[draw],
                    linestyle="-" if arm == "full" else "--",
                    marker=("o", "s", "^")[seed],
                    alpha=0.72,
                )
            ax.set_xticks([0, 1], ["Original", "Modified"])
            ax.set_ylabel(title)
    handles = [
        Line2D([], [], color=color, label=f"Draw {seed}")
        for seed, color in colors.items()
    ]
    handles.extend(
        Line2D(
            [], [], color="0.3", marker=marker, linestyle="none", label=f"Seed {seed}"
        )
        for seed, marker in enumerate(("o", "s", "^"))
    )
    handles.extend(
        Line2D([], [], color="0.3", linestyle=style, label=arm)
        for arm, style in (("Full", "-"), ("Fixed state", "--"))
    )
    axes[-1, 1].legend(handles=handles, loc="best", fontsize=7, ncols=2)
    figure.savefig(
        out / "fixed-dimension-sensitivity.pdf", metadata={"CreationDate": None}
    )
