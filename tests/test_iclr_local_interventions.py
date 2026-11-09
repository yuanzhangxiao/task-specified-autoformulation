"""Free and clamped branches, confirmation pairing, and corrected endpoint intake."""

import hashlib
import shutil

import numpy as np
import pytest

from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.rebuttal.final_evaluation import FrozenEvaluationSubject
from autoformalism.schemas import CandidateModel
from scripts import iclr_anchor_comparison as comparison
from scripts import iclr_corrected_cstr_compliance as compliance
from scripts import iclr_latent_clamp as clamp
from scripts.iclr_corrected_cstr_test import CSTR
from scripts.iclr_submission_evidence import REPO


def candidate(states, derivatives, targets):
    return CandidateModel.model_validate(
        {
            "candidate_id": "fixture",
            "parent_candidate_id": None,
            "states": [
                {"name": n, "kind": "observed" if n in targets else "latent"}
                for n in states
            ],
            "state_equations": [{"state": n, "rhs": derivatives[n]} for n in states],
            "observation_mappings": [{"channel": n, "expression": n} for n in targets],
            "initial_conditions": [
                {
                    "state": n,
                    "scope": "global",
                    "expression": n if n in targets else "0",
                }
                for n in states
            ],
        }
    )


@pytest.mark.parametrize("solver", clamp.SOLVERS)
def test_only_the_latent_mediated_path_disappears(solver):
    names = ["Gp", "I", "z"]
    model = compile_candidate(
        candidate(names, {"Gp": "z", "I": "u", "z": "u"}, clamp.TARGETS),
        ValidationContext(targets=list(clamp.TARGETS), external_inputs=["u"]),
    )
    time = np.arange(4.0)

    def trajectory(driver, initial=0.0):
        return Trajectory(
            "example",
            time,
            targets={n: np.full(4, initial) for n in clamp.TARGETS},
            external_inputs={"u": np.asarray(driver)},
            auxiliaries={},
            derivatives={},
            fixed_covariates={},
        )

    ctrl = trajectory([0.0, 0.0, 0.0, 0.0])
    tx = trajectory([0.0, 1.0, 1.0, 1.0])
    result = clamp.simulate_branches(
        model, {}, {}, ctrl, tx, {"intact": [], "latent": [2]}, solver, 15
    )
    assert result["intact"]["Gp"][-1] == pytest.approx(19 / 6)
    assert result["latent"]["Gp"] == pytest.approx([0.0] * 4, abs=1e-8)
    assert result["latent"]["I"] == pytest.approx(result["intact"]["I"])
    with pytest.raises(ValueError, match="different initial states"):
        clamp.simulate_branches(
            model, {}, {}, ctrl, trajectory([0.0] * 4, 1), {}, solver, 10
        )


def test_numerically_unresolved_channel_preserves_denominator():
    z = [0.0, 0.0, 0.0]
    a = {"control": {"Gp": z}, "intact": {"Gp": [0, 1, 2]}, "clamped": {"Gp": z}}
    b = {**a, "clamped": {"Gp": [0, 0.5, 0]}}
    saved = [{"curves": a[k]} for k in ("control", "intact")]
    r = clamp.summarize_channel(
        [a, b], saved, [a["control"], a["intact"]], np.arange(3), "Gp", "clamped"
    )
    assert r["status"] == "numerical_disagreement" and "clamped_effect_rse" not in r


def test_paired_outcomes_retains_failures_and_rejects_missing_rows():
    subjects = {arm: {"source_id": arm} for arm in comparison.ARMS}
    r = {
        "case": "short_high",
        "readout": "target",
        "quantity": "Gp",
        "status": "complete",
        "effect_rse": 0.25,
    }
    s = {
        a: [
            r
            | (
                {"status": "model_unavailable", "effect_rse": None}
                if a == "fixed_gamma"
                else {}
            )
        ]
        for a in subjects
    }
    (p,) = comparison.pair_outcomes(subjects, s)
    assert (
        p["fixed_gamma_status"] == "model_unavailable"
        and p["full_minus_fixed_effect_rse"] is None
    )
    with pytest.raises(ValueError, match="incomplete paired"):
        comparison.pair_outcomes(subjects, s | {"full": []})


def test_corrected_compliance_frozen_rows_and_resume(tmp_path, monkeypatch):
    from tests.test_iclr_corrected_cstr_test import cohort

    src = (
        REPO
        / "artifacts/reference-integrity-audit-2026-09-26-v3/public-development"
        / CSTR
    )
    if not src.is_dir():
        pytest.skip("original corrected CSTR training release unavailable")
    root = tmp_path / "campaign"
    shutil.copytree(src, root / "public" / "phase_b_v1" / CSTR)
    plan, freeze = cohort()
    for entry in freeze["roster"]:
        entry["source_sha256"] = "a" * 64
    e = freeze["roster"][0]
    e["model_available"] = True
    model = candidate(["T"], {"T": "0"}, ["T"])
    s = FrozenEvaluationSubject.model_validate(
        {
            "subject_id": e["subject_id"],
            "method": compliance.component_campaign.method_name(e["task"], "final"),
            "benchmark_id": CSTR,
            "tier": "easy",
            "repetition": e["task"]["seed"],
            "private_metrics_opened_after_freeze": False,
            "source_provenance": {
                "adapter": "direct_candidate",
                "request_id": e["subject_id"],
                "source_path": "fixture",
                "source_sha256": e["source_sha256"],
                "candidate_sha256": hashlib.sha256(
                    model.model_dump_json().encode()
                ).hexdigest(),
            },
            "candidate": model,
            "parameterization": {"status": "not_required"},
            "validation_context": {
                "targets": ["T"],
                "external_inputs": ["Cf", "Tf", "Tjf"],
                "auxiliaries": ["C", "Tj"],
            },
            "target_prediction": {"status": "missing"},
        }
    )
    monkeypatch.setattr(
        compliance, "intake", lambda *args: (plan, freeze, {s.subject_id: s}, "a" * 64)
    )
    args = (root, tmp_path / "stage", tmp_path / "scores")
    result = compliance.run(*args)
    assert result["subjects"] == 16 and not result["test_data_used"]
    assert len({r["source_id"] for r in result["requirements"]}) == 16
    assert {
        r["status"]
        for r in result["requirements"]
        if r["availability"] == "unavailable"
    } == {"unresolved"}
    assert compliance.run(*args) == result
    with pytest.raises(ValueError, match="another final endpoint"):
        compliance.score_row(e | {"source_sha256": "0" * 64}, s)


def test_corrected_development_never_authorizes_test_or_changed_manifest(tmp_path):
    from autoformalism.data import BenchmarkLoader, BenchmarkRegistry, SplitName
    from autoformalism.data.exceptions import ChannelRoleError

    src = (
        REPO
        / "artifacts/reference-integrity-audit-2026-09-26-v3/public-development"
        / CSTR
    )
    if not src.is_dir():
        pytest.skip("corrected development fixture unavailable")
    dst = tmp_path / "phase_b_v1" / CSTR
    shutil.copytree(src, dst)
    loader = BenchmarkLoader()
    spec = BenchmarkRegistry().get(CSTR)
    args = (tmp_path, spec, "easy", loader._roles(spec, "easy"))
    loader._validate_manifest(
        *args, selected_splits=(SplitName.TRAIN, SplitName.VALIDATION)
    )
    with pytest.raises(ChannelRoleError, match="not a frozen release"):
        loader._validate_manifest(*args, selected_splits=(SplitName.TEST,))
    p = dst / "manifest.json"
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(ChannelRoleError, match="not a frozen release"):
        loader._validate_manifest(*args, selected_splits=(SplitName.TRAIN,))


def test_prospective_confirmation_preserves_ready_and_unavailable_models(tmp_path):
    from autoformalism.fitting.public_fitting import content_sha256
    from autoformalism.rebuttal.prefit_replay import sealed_write
    from scripts import iclr_anchor_fidelity as assay

    source, fidelity, out = (tmp_path / n for n in ("search", "fidelity", "results"))
    blocks, models, roster, effects = [], [], [], []
    for cell in comparison.PRIMARY:
        for seed in range(3):
            tasks = []
            for arm in comparison.TREATMENTS:
                ident = f"{cell}_{seed}_{arm}"
                t = {"task_id": ident, "cell": cell, "seed": seed, "treatment": arm}
                tasks.append(t)
                r = {
                    "source_id": ident,
                    "benchmark_id": cell,
                    "method": arm,
                    "repetition": seed,
                    "source_sha256": content_sha256(t),
                    "status": "no_retained_model",
                    "shared_initial_result_sha256": content_sha256((cell, seed)),
                }
                if (cell, seed, arm) == ("sensor_00", 0, "full"):
                    c = candidate(["T"], {"T": "0"}, ["T"]).model_dump(mode="json")
                    r.update(
                        status="ready",
                        candidate=c,
                        model_sha256=content_sha256(c),
                        parameters={},
                        initials={},
                        context={
                            "targets": ["T"],
                            "external_inputs": ["Cf", "Tf", "Tjf"],
                            "auxiliaries": [],
                        },
                    )
                models.append(r)
            blocks.append({"cell": cell, "seed": seed, "tasks": tasks})
    plan = sealed_write(
        source / "plan.json", {"blocks": blocks, "max_parameters": None}
    )
    freeze = sealed_write(
        source / "evaluation_freeze.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "roster_sha256": content_sha256(models),
        },
    )
    bundle = sealed_write(
        source / "frozen_models.json",
        {
            "protocol": "iclr-controlled-search-models-1",
            "plan_sha256": plan["artifact_sha256"],
            "selection_freeze_sha256": freeze["artifact_sha256"],
            "rows": models,
        },
    )
    cases = {
        assay.fits.T2: {"fed": "fasting", "short_high": "fasting"},
        assay.fits.CSTR: {"feed_high": "basal", "jacket_warm": "basal"},
    }
    for i, r in enumerate(models):
        alias = r["benchmark_id"]
        c = assay.ALIASES[alias]
        row = {
            **r,
            "benchmark_id": c,
            "source_benchmark_id": alias,
            "source_bundle_sha256": bundle["artifact_sha256"],
        }
        roster.append(row)
        for case in cases[c]:
            for k in assay.fits.TARGETS[c]:
                effects.append(
                    {
                        **row,
                        "subject": i,
                        "case": case,
                        "quantity": k,
                        "readout": "target",
                        "status": "model_unavailable",
                    }
                )
    panel = sealed_write(
        fidelity / "frozen_selection_manifest.json",
        {
            "protocol": assay.VERSION,
            "mode": "confirmation",
            "roster": roster,
            "schedules": {
                c: {k: {"control": b} for k, b in vals.items()}
                for c, vals in cases.items()
            },
        },
    )
    scored = sealed_write(
        fidelity / "scored_rows.json",
        {
            "protocol": assay.VERSION,
            "rows": effects,
            "plan_sha256": panel["artifact_sha256"],
            "metrics_sha256": content_sha256(effects),
            "mode": "confirmation",
            "source_bundle_hashes": [bundle["artifact_sha256"]],
        },
    )
    sealed_write(
        fidelity / "summary.json",
        {
            "protocol": assay.VERSION,
            "source_mode": "confirmation",
            "plan_sha256": panel["artifact_sha256"],
            "scored_rows_sha256": scored["artifact_sha256"],
            "metrics_sha256": scored["metrics_sha256"],
            "row_count": len(effects),
        },
    )
    # Engineering fixture only; no simulator confirmations are generated here.
    for c, vals in cases.items():
        for name in set(vals) | set(vals.values()):
            sealed_write(
                fidelity / "references" / c / (name + ".json"),
                {
                    "plan": panel["artifact_sha256"],
                    "reference": {
                        "time": [0.0, 1.0, 2.0],
                        "T": [350.0] * 3,
                        "Cf": [1.0] * 3,
                        "Tf": [340.0] * 3,
                        "Tjf": [330.0] * 3,
                    },
                },
            )
    for i, r in enumerate(roster):
        if r["status"] == "ready":
            for name in ("feed_high", "basal", "jacket_warm"):
                sealed_write(
                    fidelity / "subjects" / f"{i:04d}" / (name + ".json"),
                    {
                        "plan": panel["artifact_sha256"],
                        "curves": {"T": [350.0] * 3},
                        "verification_curves": {"T": [350.0] * 3},
                    },
                )
    summary = comparison.run(source, fidelity, out)
    assert len(summary["endpoints"]) == 12 and len(summary["outcomes"]) == 18
    assert comparison.run(source, fidelity, out) == summary
    clamp.run(fidelity, tmp_path / "clamps", 1, source)
    from autoformalism.rebuttal.prefit_replay import sealed_read

    result = sealed_read(tmp_path / "clamps/summary.json")
    assert len(result["rows"]) == 36
    assert sum(r["status"] == "no_retained_model" for r in result["rows"]) == 34
    ready = [r for r in result["rows"] if r["status"] == "complete"]
    assert len(ready) == 2 and all(r["clamp_change_rms"] == 0 for r in ready)
