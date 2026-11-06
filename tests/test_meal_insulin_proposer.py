"""The optional post-fit proposer consumes only authenticated training evidence."""

import json
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.search.topology_compatibility import TrainingCompatibilityEvidence
from autoformalism.staged_topology import content_hash
from scripts import propose_meal_insulin_revision as cli
from tests.test_construction_initialization_graph_cli import reply, tokenizer
from tests.test_construction_revision_edits import brief, problem


def source():
    model = problem()
    ids = ["train_003", "train_006", "train_007"]
    evidence = TrainingCompatibilityEvidence.model_validate(
        {
            "context": model.context,
            "contract": {"provenance": "synthetic public training"},
            "sha256": "0" * 64,
            "trajectories": [
                {
                    "trajectory_id": name,
                    "time": [0, 1, 2],
                    "targets": {"I": [1, 1, 1], "Gp": [1, 1, 1]},
                    "auxiliaries": {"aux": [1, 1, 1]},
                    "covariates": {},
                    "inputs": {"u": [0, 0, 0]},
                }
                for name in ids
            ],
        }
    ).model_dump(mode="json")
    evidence["sha256"] = content_hash(
        {k: v for k, v in evidence.items() if k != "sha256"}
    )
    witnesses = [
        {
            "trajectory": n,
            "external_insulin_zero": True,
            "observed_insulin_peak_min": i + 101,
            "predicted_insulin_peak_min": i + 10,
            "observed_insulin_max": 301,
            "predicted_insulin_max": 114,
            "meal_events": [{"start_min": i, "end_min": i + 10, "dose_g": 90}],
            "grid_tolerance_min": 1,
        }
        for n, i in zip(ids, [0, 60, 120], strict=True)
    ]
    return {
        "fitted_problem": model,
        "public_brief": brief(),
        "scientific_aliases": {},
        "consistency_evidence": evidence,
        "comparison": {
            "routing": {
                "eligible_for_proposer": True,
                "status": "persistent_early_peak",
                "witnesses": witnesses,
            },
            "child_metrics": {
                "training_sha256": "f" * 64,
                "training_nmse": 0.4,
                "target_training_nmse": {"I": 0.7, "Gp": 0.2, "U": 0.3},
            },
            "validation": "heldout sentinel",
        },
        "result_identity": {"frozen": "fixture"},
        "fitting_contract": {
            "training_sha256": "f" * 64,
            "prefix_policy": {"duration": 8},
            "matched_policy": {},
        },
    }


def patch(**changes):
    return {
        "hypothesis": "A slower release may sustain insulin production.",
        "mechanism_preserved": "Delayed insulin regulates glucose through mem.",
        "expected_peak_timing": "Later",
        "expected_peak_height": "Higher",
        "edit": {
            "component": "I",
            "kind": "dynamic",
            "expression": "-q+u+Gp",
            "sources": ["q", "u", "Gp"],
        },
        "new_parameters": [],
        "new_states": [],
        "new_processes": [],
        **changes,
    }


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_source", lambda: {"fixture": "frozen"})
    data = source()
    cli.prepare(tmp_path, **data)
    return tmp_path


def forbidden(*args, **kwargs):
    pytest.fail("unexpected provider request")


def test_ineligible_fit_cannot_prepare_any_request(tmp_path):
    inputs = source()
    inputs["comparison"]["routing"]["eligible_for_proposer"] = False
    with pytest.raises(ValueError, match="trigger"):
        cli.prepare(tmp_path, **inputs)
    assert not (tmp_path / "plan.json").exists()


def test_refuses_non_training_row_and_unsealed_evidence(tmp_path):
    inputs = source()
    inputs["comparison"]["routing"]["witnesses"][0]["trajectory"] = "val_001"
    with pytest.raises(ValueError, match="training row"):
        cli.prepare(tmp_path, **inputs)
    inputs = source()
    inputs["consistency_evidence"]["trajectories"][0]["targets"]["I"][0] = 900
    with pytest.raises(ValueError, match="digest"):
        cli.prepare(tmp_path, **inputs)


def test_one_cached_generation_validates_then_replays_without_server(prepared):
    root = prepared
    plan, inputs = cli._load(root)
    before = deepcopy(inputs["problem"])
    prompt = json.loads(inputs["user"])
    assert "heldout sentinel" not in inputs["user"]
    assert "root" not in prompt["fitted_shared_coefficients"]
    assert (
        len(prompt["training_evidence"]["shifted_meals_with_zero_external_insulin"])
        == 3
    )
    assert plan["settings"]["maximum_requests"] == 1
    seen = []

    def respond(url, body, timeout):
        seen.append(body)
        return reply(patch())

    args = (root, "http://127.0.0.1:8000")
    first = cli.run(*args, transport=respond, token_transport=tokenizer)
    assert first["status"] == "eligible_for_refit", first
    assert len(seen) == len(first["call_files"]) == 1
    assert seen[0]["messages"] == [
        {"role": "system", "content": inputs["system"]},
        {"role": "user", "content": inputs["user"]},
    ]
    assert first["audit"]["static_eligibility"] == "eligible"
    assert (
        first["training_consistency"]["affine_family_consistency"][
            "scaled_readout_initial_policy"
        ]
        == "shared-scaled-initial-readout-1"
    )
    assert not first["test_data_opened"] and not first["automatic_fit"]
    assert sealed_read(root / "inputs.json")["problem"] == before
    # Completion can be lost after the durable provider response was written.
    (root / "result.json").unlink()
    assert cli.run(*args, transport=forbidden, token_transport=forbidden) == first
    assert cli.run(*args, transport=forbidden, token_transport=forbidden) == first


def test_scope_rejects_noninsulin_edit_and_unused_new_state(prepared):
    _, inputs = cli._load(prepared)
    with pytest.raises(ValueError, match="one existing edit"):
        cli.assess(
            inputs,
            patch(
                edit={
                    "component": "Gp",
                    "kind": "dynamic",
                    "expression": "I",
                    "sources": ["I"],
                }
            ),
        )
    with pytest.raises(ValueError, match="must contribute"):
        cli.assess(
            inputs,
            patch(
                new_states=[
                    {
                        "name": "unused",
                        "description": "spare",
                        "rhs": "-unused",
                        "sources": ["unused"],
                        "initial": {"mode": "value", "scope": "trajectory"},
                    }
                ]
            ),
        )


def test_unknown_outcome_and_preflight_failure_never_resend(prepared, tmp_path):
    plan = sealed_read(prepared / "plan.json")
    sealed_write(
        prepared / "attempt.json",
        {"plan_sha256": plan["artifact_sha256"], "serving": {"test_transport": True}},
    )
    result = cli.run(
        prepared, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
    )
    assert result["status"] == "interrupted_after_reservation"
    other = tmp_path / "other"
    cli.prepare(other, **source())
    result = cli.run(
        other,
        "http://127.0.0.1",
        transport=forbidden,
        token_transport=lambda *a: {"count": 33000, "max_model_len": 32768},
    )
    assert result["status"] == "preflight_blocked" and result["call_files"] == {}
    assert (
        cli.run(
            other, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )
        == result
    )
    with pytest.raises(ValueError, match="loopback"):
        cli.run(other, "https://some-server", transport=forbidden)


def test_bad_reply_stays_rejected_on_resume(prepared):
    r = cli.run(
        prepared,
        "http://127.0.0.1",
        token_transport=tokenizer,
        transport=lambda *a: reply(
            patch(
                edit={
                    "component": "I",
                    "kind": "dynamic",
                    "sources": [],
                    "expression": "__import__('os')",
                }
            )
        ),
    )
    assert r["status"] == "rejected" and r["problem"] is None
    assert (
        cli.run(
            prepared, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )
        == r
    )


def test_nonlinear_state_revision_uses_existing_gate(prepared):
    _, inputs = cli._load(prepared)
    payload = patch(
        edit={
            "component": "I",
            "kind": "dynamic",
            "sources": ["q", "u", "R"],
            "expression": "-q+u+R",
        },
        new_states=[
            {
                "name": "R",
                "description": "pool",
                "rhs": "Gp/(1+Gp**2)-R",
                "sources": ["Gp", "R"],
                "initial": {"mode": "value", "scope": "shared_training"},
            }
        ],
    )
    outcome = cli.assess(inputs, payload)
    assert outcome["status"] == "eligible_for_refit"
    assert (
        outcome["problem"]["graph"]["rules"]["R"]["initial"]["scope"]
        == "shared_training"
    )


def test_refit_helper_compares_original_equation_residual_and_requires_identity(
    tmp_path,
):
    import numpy as np

    from autoformalism.fitting.public_fitting import content_sha256
    from autoformalism.research import meal_revision_metrics as metrics
    from tests.test_meal_revision_metrics import evaluate, training

    data = training()
    baseline = evaluate(data)
    source = sealed_write(
        tmp_path / "inputs.json",
        {
            "fitted_child_metrics": baseline,
            "fitting_contract": {"training_sha256": content_sha256(data)},
            "result_identity": {"endpoint_sha256": "fixture"},
        },
    )
    plan = sealed_write(
        tmp_path / "plan.json",
        {"protocol": cli.PROTOCOL, "inputs_sha256": source["artifact_sha256"]},
    )
    model = problem().model_dump(mode="json")
    sealed_write(
        tmp_path / "result.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "problem": model,
            "status": "eligible_for_refit",
        },
    )
    args = {
        "training": data,
        "problem_sha256": content_sha256(model),
        "scales": baseline["training_scales"],
        "channels": metrics.CHANNELS,
        "residual": np.zeros(3 * sum(len(r.time) for r in data.rows)),
        "verified_training_nmse": 0.0,
    }
    result = cli.compare_refit_replay(tmp_path, **args)
    assert result["metrics"]["training_nmse"] == 0
    assert result["comparison"]["routing"]["status"] == "early_peak_criterion_cleared"
    with pytest.raises(ValueError, match="admitted training problem"):
        cli.compare_refit_replay(tmp_path, **(args | {"problem_sha256": "wrong"}))
    with pytest.raises(ValueError, match="verified training NMSE"):
        cli.compare_refit_replay(tmp_path, **(args | {"verified_training_nmse": 0.5}))


def test_normalized_existing_process_is_not_a_disconnected_addition(tmp_path):
    from autoformalism.schemas.prefix_initialization import PrefixFitProblem

    data = source()
    payload = data["fitted_problem"].model_dump(mode="json")
    payload["base_candidate"]["processes"].append(
        {"name": "unused_law", "expression": "a*Gp"}
    )
    data["fitted_problem"] = PrefixFitProblem.model_validate(payload)
    cli.prepare(tmp_path, **data)
    _, inputs = cli._load(tmp_path)
    ordinary = cli.assess(inputs, patch())
    repeated = cli.assess(
        inputs,
        patch(
            new_processes=[
                {
                    "name": "unused_law",
                    "description": "unchanged declaration",
                    "expression": "a*Gp",
                    "sources": ["Gp"],
                }
            ]
        ),
    )
    assert ordinary["status"] == repeated["status"] == "eligible_for_refit"
    assert repeated["audit"]["new_processes"] == []
    assert (
        cli.family_identity(data["fitted_problem"], ordinary["problem"])["sha256"]
        == cli.family_identity(data["fitted_problem"], repeated["problem"])["sha256"]
    )


def test_portable_intake_is_pinned_and_resume_does_not_rebind(tmp_path, monkeypatch):
    from autoformalism.benchmarks.audited_release import seal

    data = source()
    data["fitted_problem"] = data["fitted_problem"].model_dump(mode="json")
    data["public_brief"] = data["public_brief"].model_dump(mode="json")
    intake = {"protocol": cli.INTAKE_PROTOCOL, "arguments": data}
    monkeypatch.setattr(cli, "REVIEWED_INTAKE_SHA256", cli.content_sha256(intake))
    path, root = tmp_path / "intake.json", tmp_path / "campaign"
    seal(path, intake)
    first = cli.prepare_source(root, path)
    assert cli.prepare_source(root, path) == first
    assert cli.status(root)["status"] == "pending"
    # Even a resealed edit to the intake is rejected by the reviewed content pin.
    changed = deepcopy(intake)
    changed["arguments"]["result_identity"]["frozen"] = "wrong endpoint"
    seal(tmp_path / "wrong.json", changed)
    with pytest.raises(ValueError, match="reviewed"):
        cli.prepare_source(tmp_path / "wrong-root", tmp_path / "wrong.json")


def test_report_and_inspect_are_provider_free_and_preserve_cost(prepared, monkeypatch):
    import tarfile

    first = cli.run(
        prepared,
        "http://127.0.0.1",
        token_transport=tokenizer,
        transport=lambda *a: reply(patch()),
    )
    monkeypatch.setattr(cli, "_server", forbidden)
    monkeypatch.setattr(cli, "run", forbidden)
    summary = cli.status(prepared)
    assert summary["terminal_available"]
    assert summary["physical_requests_started"] == 1
    assert summary["observed_total_tokens"] == 31
    assert summary["usage_complete"]
    assert summary["training_consistency"] == first["training_consistency"]["status"]
    inspected = cli.inspect(prepared)
    archive = prepared / "inspection.tar.gz"
    before = archive.read_bytes()
    assert cli.inspect(prepared) == inspected
    assert archive.read_bytes() == before
    with tarfile.open(archive) as bundle:
        assert {
            "result.json",
            "plan.json",
            "inputs.json",
            "REPORT.html",
            "PROMPT.html",
        } <= set(bundle.getnames())
        assert ".lock" not in bundle.getnames()
    assert "Full user message" in (prepared / "PROMPT.html").read_text()
    assert "Model after revision" in (prepared / "REPORT.html").read_text()


def test_status_retains_uncertain_reservation_and_rejects_call_drift(prepared):
    plan, _ = cli._load(prepared)
    sealed_write(
        prepared / "attempt.json",
        {"plan_sha256": plan["artifact_sha256"], "serving": {}},
    )
    assert cli.status(prepared)["unrecorded_request_outcome_possible"]
    cli.run(
        prepared, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
    )
    summary = cli.status(prepared)
    assert summary["status"] == "interrupted_after_reservation"
    assert summary["unrecorded_request_outcome_possible"]
    assert not summary["usage_complete"]
    (prepared / "calls").mkdir(exist_ok=True)
    (prepared / "calls/extra.json").write_text("{}")
    with pytest.raises(ValueError, match="calls differ"):
        cli.status(prepared)


def test_inspection_rejects_external_symlink(prepared, tmp_path):
    external = tmp_path / "external.txt"
    external.write_text("not campaign evidence")
    (prepared / "logs").mkdir(exist_ok=True)
    (prepared / "logs/out.log").symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        cli.inspect(prepared)


def test_prompt_explains_complete_replacement_and_initializer_scope(prepared):
    _, inputs = cli._load(prepared)
    prompt = json.loads(inputs["user"])
    assert "replaces the whole named RHS" in inputs["system"]
    assert "initializer-local" in inputs["system"]
    assert prompt["initialization_contract"]["fixed_prefix_policy"]["duration"] == 8
    assert (
        "never the full target history"
        in prompt["initialization_contract"]["trajectory"]
    )
