"""Regression contracts for matched repair arms, not mocked scientific recovery."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoformalism.expressions import (
    ModelValidationError,
    ValidationContext,
    compile_candidate,
)
from autoformalism.fitting.initialization import apply_initialization_plan
from autoformalism.rebuttal import repair_comparison as campaign
from autoformalism.rebuttal.repair_evidence import (
    Finding,
    decision_report,
    domain_findings,
    model_hash,
    numerical_findings,
)
from autoformalism.rebuttal.repair_scientific_judge import (
    judge_protocol,
    review_request,
)
from autoformalism.rebuttal.repair_transactions import (
    RepairAction,
    commit_action,
    default_initialization,
    request_repair,
)
from autoformalism.schemas import CandidateModel

CONTEXT = ValidationContext(targets=("v01",), external_inputs=("u01",))


def candidate():
    return CandidateModel.model_validate(
        {
            "candidate_id": "parent",
            "parent_candidate_id": None,
            "states": [
                {"name": "m", "kind": "latent"},
                {"name": "y", "kind": "observed"},
            ],
            "processes": [{"name": "f", "expression": "sigmoid(m)"}],
            "state_equations": [
                {"state": "m", "rhs": "-rate*m+gain*u01"},
                {"state": "y", "rhs": "-decay*y+f"},
            ],
            "parameters": [
                {"name": n, "scope": "global", "role": "nonnegative_coefficient"}
                for n in ("rate", "gain", "decay")
            ],
            "observation_mappings": [{"channel": "v01", "expression": "y"}],
            "initial_conditions": [
                {"state": "m", "scope": "global", "fixed_value": 0},
                {"state": "y", "scope": "global", "expression": "v01"},
            ],
        }
    )


def apply(raw):
    model = candidate()
    return commit_action(
        model,
        default_initialization(model, CONTEXT),
        RepairAction.model_validate(raw),
        CONTEXT,
        ("v01",),
    )


def test_harmless_global_keep_does_not_reject_valid_edits():
    revised, _, audit = apply(
        {
            "scope": "function",
            "keep": ["y", "u01", "m"],
            "equations": [{"component": "f", "expression": "tanh(m)"}],
        }
    )
    assert revised.processes[0].expression == "tanh(m)"
    assert audit["status"] == "committed"
    assert audit["ignored_known_keep"] == ["y", "u01", "m"]


def test_function_sign_change_needs_explicit_model_scope():
    raw = {
        "scope": "function",
        "equations": [{"component": "m", "expression": "rate*m+gain*u01"}],
    }
    with pytest.raises(ValueError, match="scope=model"):
        apply(raw)
    raw["scope"] = "model"
    revised, _, _ = apply(raw)
    assert {p.name: p.role.value for p in revised.parameters}[
        "rate"
    ] == "nonnegative_coefficient"


def test_cross_level_variable_and_mapping_edit_commits_coherently():
    revised, initials, audit = apply(
        {
            "scope": "model",
            "equations": [
                {"component": "z", "kind": "dynamic", "expression": "m-z"},
                {"component": "f", "expression": "sigmoid(z)"},
            ],
            "mappings": [{"channel": "v01", "expression": "y+z"}],
        }
    )
    assert audit["added_variables"] == ["z"]
    assert set(initials.rules) == {"m", "y", "z"}
    compiled, guesses, _ = apply_initialization_plan(
        compile_candidate(revised, CONTEXT), initials
    )
    assert "init_z_value" in guesses
    assert "init_z_value" in compiled.parameter_names


def test_new_direct_gain_offset_roles_and_parent_roles_are_runtime_owned():
    revised, _, _ = apply(
        {
            "scope": "model",
            "equations": [
                {
                    "component": "m",
                    "expression": "-rate*m+new_gain*u01+offset",
                    "parameters": [
                        {"name": "rate", "role": "positive_shape"},
                        {"name": "new_gain"},
                        {"name": "offset"},
                    ],
                }
            ],
        }
    )
    roles = {p.name: p.role.value for p in revised.parameters}
    assert roles["rate"] == roles["new_gain"] == "nonnegative_coefficient"
    assert roles["offset"] == "offset"


@pytest.mark.parametrize("expression", ["v01", "unknown", "__import__('os')", "u01[1]"])
def test_rhs_cannot_read_target_measurement_or_unknown_code(expression):
    with pytest.raises((ValueError, ModelValidationError)):
        apply(
            {
                "scope": "model",
                "equations": [{"component": "m", "expression": expression}],
            }
        )


def test_algebraic_cycle_and_dangling_removal_are_rejected():
    with pytest.raises((ValueError, ModelValidationError)):
        apply(
            {
                "scope": "model",
                "equations": [
                    {"component": "z", "kind": "algebraic", "expression": "f"},
                    {"component": "f", "expression": "sigmoid(z)"},
                ],
            }
        )
    with pytest.raises((ValueError, ModelValidationError)):
        apply({"scope": "model", "remove": ["m"]})


def test_initial_map_is_trained_and_limited_to_first_public_channels():
    _, plan, _ = apply(
        {
            "scope": "model",
            "initializers": [
                {
                    "state": "m",
                    "causal_map": {
                        "mode": "map",
                        "expression": "a+b*v01",
                        "parameters": [{"name": "a"}, {"name": "b"}],
                    },
                }
            ],
        }
    )
    compiled, _, _ = apply_initialization_plan(
        compile_candidate(candidate(), CONTEXT), plan
    )
    assert {"init_m_a", "init_m_b"} <= set(compiled.parameter_names)
    with pytest.raises(ValueError):
        apply({"scope": "model", "initializers": [{"state": "y"}]})
    with pytest.raises(ValueError):
        apply(
            {
                "scope": "model",
                "initializers": [
                    {"state": "m", "causal_map": {"mode": "map", "expression": "y"}}
                ],
            }
        )


@pytest.mark.parametrize(
    "law,code,blocking",
    [
        ("m/(1+m)", "DOMAIN_UNRESOLVED", False),
        ("log(m)", "DOMAIN_UNRESOLVED", False),
        ("sqrt(m)", "DOMAIN_UNRESOLVED", False),
        ("log(-1)", "DOMAIN_INVALID", True),
        ("sqrt(-2)", "DOMAIN_INVALID", True),
        ("m/0", "DOMAIN_INVALID", True),
        ("m**-1", "DOMAIN_UNRESOLVED", False),
    ],
)
def test_general_domain_audit_has_explicit_uncertainty(law, code, blocking):
    raw = candidate().model_dump(mode="json")
    raw["processes"][0]["expression"] = law
    facts = domain_findings(CandidateModel.model_validate(raw), CONTEXT)
    assert facts[0].code == code and facts[0].blocking == blocking


def test_legacy_proposer_ranges_do_not_create_domain_certificates():
    raw = candidate().model_dump(mode="json")
    raw["parameters"][1].update(
        role="coefficient", domain="real", bounds={"lower": 1, "upper": 2}
    )
    raw["processes"][0]["expression"] = "log(gain)"
    findings = domain_findings(CandidateModel.model_validate(raw), CONTEXT)
    assert findings[0].certainty == "unresolved"


@pytest.mark.parametrize(
    "law", ["m/(1+m*m)", "log(1+m**2)", "sqrt(m**2)", "m/(1+abs(m))"]
)
def test_total_domain_laws_are_not_flagged(law):
    raw = candidate().model_dump(mode="json")
    raw["processes"][0]["expression"] = law
    # m*m interval correlation is deliberately not inferred; m**2 is certified.
    if "m*m" in law:
        assert domain_findings(CandidateModel.model_validate(raw), CONTEXT)
    else:
        assert domain_findings(CandidateModel.model_validate(raw), CONTEXT) == []


def test_no_change_and_reordering_do_not_consume_fit_budget():
    original = candidate()
    revised, _, audit = apply({"scope": "no_change", "keep": ["u01", "m"]})
    assert revised == original and audit["status"] == "no_change"
    raw = original.model_dump(mode="json")
    raw["states"].reverse()
    assert model_hash(CandidateModel.model_validate(raw)) == model_hash(original)


class Client:
    def __init__(self, replies):
        self.replies, self.requests = replies, []
        self.settings = SimpleNamespace(attempts_per_step=len(replies))

    def call(self, **kwargs):
        self.requests.append(json.loads(kwargs["user"]))
        return {
            "request_hash": str(kwargs["attempt"]),
            "status": "responded",
            "raw_response": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(self.replies[kwargs["attempt"]])
                        },
                    }
                ]
            },
        }


def test_partial_valid_revisions_survive_retry_and_resume(tmp_path):
    client = Client(
        [
            {
                "scope": "function",
                "equations": [
                    {"component": "f", "expression": "tanh(m)"},
                    {"component": "y", "expression": "unknown"},
                ],
                "keep": ["m", "u01"],
            },
            {
                "scope": "function",
                "equations": [{"component": "y", "expression": "-decay*y+2*f"}],
            },
        ]
    )
    parent = candidate()
    kwargs = {
        "parent": parent,
        "plan": default_initialization(parent, CONTEXT),
        "context": CONTEXT,
        "report": {},
        "public_prompt": "Public task",
        "client": client,
        "directory": tmp_path,
        "round_index": 1,
        "nonlinear_targets": ("v01",),
    }
    revised, _, result = request_repair(**kwargs)
    assert result["status"] == "committed"
    assert client.requests[1]["pending_components"] == ["y"]
    assert set(client.requests[1]["provisional_edits"]) == {"f"}
    assert revised.processes[0].expression == "tanh(m)"
    request_repair(**kwargs)
    assert len(client.requests) == 2


def test_pending_omission_cannot_silently_commit_a_partial_model(tmp_path):
    client = Client(
        [
            {
                "scope": "function",
                "equations": [
                    {"component": "f", "expression": "tanh(m)"},
                    {"component": "y", "expression": "unknown"},
                ],
            },
            {"scope": "function", "equations": []},
        ]
    )
    parent = candidate()
    revised, _, result = request_repair(
        parent,
        default_initialization(parent, CONTEXT),
        CONTEXT,
        {},
        "Public task",
        client,
        tmp_path,
        1,
        ("v01",),
    )
    assert revised == parent and result["status"] == "exhausted"
    assert result["attempts"][-1]["diagnostics"][0]["code"] == "PENDING_ACTION_OMITTED"


def test_structured_report_does_not_turn_timeout_into_scientific_cause():
    model = candidate()
    numerical = numerical_findings(
        model, {"status": "fit_failed", "initializer": {"message": "timeout"}}
    )
    report = decision_report(model, [], [], numerical, [{"outcome": "no_change"}])
    assert report["objective_category"] == "numerical_feasibility"
    assert report["scientific_findings"] == []
    assert report["cause_is_not_determined_by_runtime"]
    assert report["last_numerical_evidence"]
    concern = Finding(
        source="scientific_judge",
        stage="prefit",
        candidate_sha256=model_hash(model),
        category="scientific_requirement",
        code="test",
        certainty="advisory",
        observation="possible unsupported mechanism",
        recheck="judge",
    )
    report = decision_report(model, [], [concern], numerical, [])
    assert report["objective_category"] == "scientific_requirement"
    assert not concern.blocking


def test_scientific_request_has_no_runtime_or_numerical_feedback():
    model = candidate()
    request = review_request(model, model, CONTEXT, "Public prompt", 0, "a" * 40)
    assert set(request) == {
        "parent",
        "candidate",
        "context",
        "public_prompt",
        "seed",
        "model_revision",
        "protocol",
    }
    assert request["protocol"] == judge_protocol()
    assert "gpt-oss-120b" in request["protocol"]["model"]


def test_matched_config_uses_verified_feasibility_not_legacy_or_new_mesh():
    config = campaign.RepairComparisonConfig(judge_revision="a" * 40)
    assert config.fit.recovery_policy == "feasible"
    assert config.fit.collocation_node_start == "rollout_or_observed"
    assert config.fit.collocation_mesh_substeps == 1
    assert config.fit.maximum_function_evaluations == 240
    assert config.fit.least_squares_ftol is None


def test_prefit_judge_pauses_before_fit_but_nojudge_does_not(tmp_path, monkeypatch):
    model = candidate()
    initial = default_initialization(model, CONTEXT)
    calls = []
    monkeypatch.setattr(
        campaign,
        "fit_collocation_forward_sensitivity",
        lambda *args, **kw: calls.append(kw["initialization_plan"])
        or {"status": "complete", "training": {}, "validation": {}},
    )
    kwargs = {
        "root": tmp_path,
        "directory": tmp_path / "without",
        "parent": model,
        "candidate": model,
        "initial": initial,
        "dataset": SimpleNamespace(train=object(), validation=object()),
        "context": CONTEXT,
        "prompt": "Public scientific task",
        "config": campaign.RepairComparisonConfig(judge_revision="a" * 40),
    }
    a = campaign._assess(task={"arm": campaign.ARMS[0], "seed": 0}, **kwargs)
    assert a["scientific_review"] is None and len(calls) == 1
    kwargs["directory"] = tmp_path / "with"
    with pytest.raises(campaign.JudgePending):
        campaign._assess(task={"arm": campaign.ARMS[1], "seed": 0}, **kwargs)
    assert len(calls) == 1
    request_path = next((tmp_path / "reviews").glob("*/request.json"))
    from autoformalism.llm.staged_topology import atomic_json
    from autoformalism.staged_topology import content_hash

    request = json.loads(request_path.read_text())
    atomic_json(
        request_path.parent / "review.json",
        {"request_sha256": content_hash(request), "findings": [], "status": "reviewed"},
    )
    campaign._assess(task={"arm": campaign.ARMS[1], "seed": 0}, **kwargs)
    assert len(calls) == 2 and calls[0] == calls[1]


def test_interrupted_fit_is_not_silently_restarted(tmp_path, monkeypatch):
    from autoformalism.llm.staged_topology import atomic_json

    model = candidate()
    atomic_json(tmp_path / "fit_started.json", {})
    monkeypatch.setattr(
        campaign,
        "fit_collocation_forward_sensitivity",
        lambda *a, **k: pytest.fail("refit"),
    )
    result = campaign._assess(
        tmp_path,
        {"arm": campaign.ARMS[0], "seed": 0},
        tmp_path,
        model,
        model,
        default_initialization(model, CONTEXT),
        SimpleNamespace(train=None, validation=None),
        CONTEXT,
        "Public",
        campaign.RepairComparisonConfig(judge_revision="a" * 40),
    )
    assert result["fit"]["status"] == "interrupted"


def test_new_shared_parameter_declared_once_is_available_in_both_equations():
    revised, _, _ = apply(
        {
            "scope": "model",
            "equations": [
                {
                    "component": "m",
                    "expression": "-shared*m+gain*u01",
                    "parameters": [{"name": "shared"}],
                },
                {"component": "y", "expression": "-shared*y+f"},
            ],
        }
    )
    assert sum(p.name == "shared" for p in revised.parameters) == 1
    assert "shared" in revised.state_equations[-1].rhs


def test_certified_memory_absence_is_not_full_scientific_judgment():
    from autoformalism.rebuttal.repair_evidence import memory_path_findings

    model = candidate()
    assert memory_path_findings(model, CONTEXT, ("v01",)) == []
    raw = model.model_dump(mode="json")
    raw["state_equations"][0]["rhs"] = "-rate*m"
    raw["state_equations"][1]["rhs"] = "-decay*y+f+gain*u01"
    finding = memory_path_findings(
        CandidateModel.model_validate(raw), CONTEXT, ("v01",)
    )[0]
    assert finding.code == "INTERNAL_MEMORY_PATH_ABSENT" and finding.blocking
    assert "not scientific correctness" in finding.recheck


def test_related_mapping_edit_is_retained_across_local_retry(tmp_path):
    parent = candidate()
    client = Client(
        [
            {
                "scope": "model",
                "equations": [
                    {"component": "f", "expression": "tanh(m)"},
                    {"component": "y", "expression": "unknown"},
                ],
                "mappings": [{"channel": "v01", "expression": "y+m"}],
            },
            {
                "scope": "model",
                "equations": [{"component": "y", "expression": "-decay*y+f"}],
            },
        ]
    )
    revised, _, result = request_repair(
        parent,
        default_initialization(parent, CONTEXT),
        CONTEXT,
        {},
        "Public",
        client,
        tmp_path,
        1,
        ("v01",),
    )
    assert result["status"] == "committed"
    assert revised.observation_mappings[0].expression == "y+m"
    assert client.requests[1]["provisional_related_edits"]["mappings"]


def test_cached_transport_keeps_physical_budget_across_restart(tmp_path):
    from autoformalism.llm.staged_topology import StagedModelSettings

    kwargs = {
        "settings": StagedModelSettings(maximum_requests=1),
        "base_url": "http://unused",
        "directory": tmp_path,
        "namespace": "test",
        "seed": 0,
        "transport": lambda *a: {"choices": [{"message": {"content": "{}"}}]},
    }
    call = {
        "system": "s",
        "user": "u",
        "response_model": RepairAction,
        "step": "one",
        "attempt": 0,
    }
    first = campaign.BudgetedRepairClient(**kwargs)
    first.call(**call)
    second = campaign.BudgetedRepairClient(**kwargs)
    second.call(**call)
    assert len(second.records) == 1
    with pytest.raises(campaign.RepairBudgetExceeded):
        second.call(**{**call, "step": "two"})


def test_judge_cost_counts_failure_plus_final_success_without_double_count(tmp_path):
    from autoformalism.rebuttal.repair_scientific_judge import review_cost

    events = [
        {
            "event": "llm_failure",
            "request_hash": "a",
            "attempt": 1,
            "provider_attempts": 1,
            "usage": {"total_tokens": 5},
        },
        {
            "event": "llm_response",
            "request_hash": "a",
            "provider_attempts": 2,
            "usage": {"total_tokens": 7},
        },
        {
            "event": "llm_response",
            "request_hash": "a",
            "cache_hit": True,
            "provider_attempts": 0,
            "usage": {"total_tokens": 7},
        },
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    cost = review_cost(tmp_path)
    assert cost["physical_requests"] == 2 and cost["observed_total_tokens"] == 12
    assert cost["cache_hit_events"] == 1 and cost["usage_complete"]


def test_real_paired_judge_adapter_is_numerically_blind_and_cached(
    tmp_path, monkeypatch
):
    from autoformalism.judging import (
        build_atomic_evidence_plan,
        extract_public_requirements,
        semantic_absolute_units,
    )
    from autoformalism.llm import MockLLMClient
    from autoformalism.rebuttal import repair_scientific_judge as adapter
    from autoformalism.schemas import (
        AtomicJudgeResult,
        HybridJudgeResult,
        RelativeCriterion,
    )

    model = candidate()
    prompt = "Public task: predict v01 from u01 using internal dynamic memory."
    units = semantic_absolute_units(
        extract_public_requirements(prompt), include_role_consistency=False
    )
    atomic_plan = build_atomic_evidence_plan(model, model)
    atomic = AtomicJudgeResult.model_validate(
        {
            "signed_occurrence_assessments": [
                {
                    "occurrence_id": o.occurrence_id,
                    "expected_direction": "context_dependent",
                    "evidence": "Unspecified public direction.",
                }
                for o in atomic_plan.occurrences
            ],
            "repeated_contribution_assessments": [
                {
                    "repeat_pair_id": p.repeat_pair_id,
                    "relation": "insufficient_public_information",
                    "evidence": "Unknown identity.",
                }
                for p in atomic_plan.repeat_candidates
            ],
        }
    )
    hybrid = HybridJudgeResult.model_validate(
        {
            "absolute_assessments": [
                {
                    "criterion": c.value,
                    "subject_id": s,
                    "candidate_a": {
                        "verdict": "fail",
                        "evidence": "Synthetic concern.",
                    },
                    "candidate_b": {
                        "verdict": "fail",
                        "evidence": "Synthetic concern.",
                    },
                }
                for c, s in units
            ],
            "comparative_assessments": [
                {"criterion": c.value, "verdict": "tie", "evidence": "Identical."}
                for c in RelativeCriterion
            ],
        }
    )
    clients = []

    def client(config):
        clients.append(config)
        return MockLLMClient(
            atomic_responses=[atomic, atomic], hybrid_responses=[hybrid, hybrid]
        )

    monkeypatch.setattr(adapter, "create_llm_client", client)
    request = review_request(model, model, CONTEXT, prompt, 0, "a" * 40)
    result = adapter.perform_review(request, tmp_path, "http://unused")
    assert result["status"] == "reviewed"
    assert all(
        f["certainty"] == "advisory" and not f["blocking"] for f in result["findings"]
    )
    assert not result["numerical_results_visible"]
    assert all(c.model == "openai/gpt-oss-120b" for c in clients)
    assert adapter.perform_review(request, tmp_path, "http://unused") == result
    assert len(clients) == 2


def test_whole_loop_keeps_numerical_evidence_and_best_model_after_rejection(
    tmp_path, monkeypatch
):
    from autoformalism.llm.staged_topology import atomic_json

    model = candidate()
    task = {
        "task_id": "runtime0",
        "candidate_path": "parent.json",
        "benchmark_id": "control",
        "tier": "hard",
        "seed": 0,
        "arm": campaign.ARMS[0],
    }
    plan = {
        "config": campaign.RepairComparisonConfig(judge_revision="a" * 40).model_dump(
            mode="json"
        )
    }
    atomic_json(tmp_path / "inputs/parent.json", model.model_dump(mode="json"))
    prompt = tmp_path / "inputs/frozen/public/phase_b_v1/control/proposer_prompt.txt"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("Public task\nF. Required response\nOLD SCHEMA")
    monkeypatch.setattr(campaign, "verify", lambda root: plan)
    monkeypatch.setattr(
        campaign,
        "load_public_data",
        lambda *a: (SimpleNamespace(train=None, validation=None), CONTEXT),
    )
    monkeypatch.setattr(campaign, "_public_fit_context", lambda d: {"train_only": True})
    fitted = []

    def fit(*args, **kwargs):
        fitted.append(args[0])
        return {
            "status": "complete",
            "training": {"normalized_mse": 1.0},
            "validation": {"normalized_mse": 1 / len(fitted)},
        }

    monkeypatch.setattr(campaign, "fit_collocation_forward_sensitivity", fit)

    class RoundsClient(Client):
        def call(self, **kwargs):
            step = kwargs["step"]
            self.replies = (
                [
                    {
                        "scope": "function",
                        "equations": [{"component": "f", "expression": "tanh(m)"}],
                    }
                ]
                if step.endswith("1")
                else [
                    {
                        "scope": "model",
                        "equations": [{"component": "m", "expression": "missing"}],
                    }
                ]
                if step.endswith("2")
                else [
                    {
                        "scope": "no_change",
                        "hypothesis": "No justified repair; consider other scope.",
                    }
                ]
            )
            return super().call(**kwargs)

    client = RoundsClient([{}])
    result = campaign.run_task(tmp_path, task, client)
    assert len(fitted) == 2
    assert result["status"] == "stopped_no_progress"
    assert [r["outcome"] for r in result["rounds"]] == [
        "committed",
        "exhausted",
        "no_change",
        "no_change",
    ]
    assert result["best"]["round"] == 1
    assert result["last_numerical"] == result["rounds"][0]["assessment"]["numerical"]
    assert "OLD SCHEMA" not in json.dumps(client.requests)
    assert not (tmp_path / "reviews").exists()
    assert campaign.run_task(tmp_path, task, client) == result
    assert len(fitted) == 2 and len(client.requests) == 4


def test_freeze_matches_both_arms_and_verifies_launchers(tmp_path, monkeypatch):
    from autoformalism.llm.staged_topology import atomic_json

    inputs = {
        "tasks": [{"task_id": f"seed{s}", "seed": s} for s in range(2)],
        "plan_sha256": "public",
        "config": {
            "model_settings": {"model": "same20b"},
            "serving_image_sha256": "sif",
            "public_obligation_quote": "memory",
            "public_obligation_prompt_sha256": "prompt",
        },
    }
    monkeypatch.setattr(campaign, "freeze_inputs", lambda *a: inputs)
    monkeypatch.setattr(campaign, "_verified_plan", lambda p: inputs)
    plan = campaign.freeze(tmp_path / "source", tmp_path / "frozen", "a" * 40)
    assert len(plan["tasks"]) == 4
    assert [t["arm"] for t in plan["tasks"]] == [
        *campaign.ARMS,
        *reversed(campaign.ARMS),
    ]
    assert campaign.verify(tmp_path / "frozen") == plan
    plan["config"]["fit"]["recovery_policy"] = "legacy"
    atomic_json(tmp_path / "frozen/plan.json", plan)
    with pytest.raises(ValueError, match="digest"):
        campaign.verify(tmp_path / "frozen")


@pytest.mark.parametrize("timeout", [False, True])
def test_aces_submission_is_bounded_and_refuses_ambiguous_retries(tmp_path, timeout):
    if not all(shutil.which(c) for c in ("bash", "jq", "sha256sum")):
        pytest.skip("shell launcher tools unavailable")
    repo = Path(__file__).resolve().parents[1]
    bins = tmp_path / "bin"
    bins.mkdir()
    output, source = tmp_path / "output", tmp_path / "source"
    (source / "summary").mkdir(parents=True)
    (source / "summary/summary.json").write_text("{}")
    image = tmp_path / "image.sif"
    image.write_bytes(b"synthetic image fixture")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    log = tmp_path / "scheduler.json"
    scripts = {
        "module": "#!/bin/sh\nexit 0\n",
        "git": '#!/bin/sh\ncase "$*" in *status*) exit 0 ;; *) echo abc123 ;; esac\n',
        "squeue": "#!/bin/sh\nexit 0\n",
        "sacct": "#!/bin/sh\necho COMPLETED\n",
        "fakepython": f"""#!{sys.executable}
import json, pathlib
p = pathlib.Path({str(output)!r}); p.mkdir(exist_ok=True)
plan = {{"config": {{"judge_revision": "a"*40}}, "plan_sha256": "frozen",
        "serving_image_sha256": {digest!r}}}
(p / "plan.json").write_text(json.dumps(plan))
""",
        "sbatch": f"""#!{sys.executable}
import json, pathlib, sys
p = pathlib.Path({str(log)!r})
calls = json.loads(p.read_text()) if p.exists() else []
calls.append(sys.argv[1:]); p.write_text(json.dumps(calls))
if {timeout!r}: sys.exit(1)
print(100 + len(calls))
""",
    }
    for name, text in scripts.items():
        path = bins / name
        path.write_text(text)
        path.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(bins) + os.pathsep + os.environ["PATH"],
        AF_REPO_ROOT=str(repo),
        AF_PYTHON=str(bins / "fakepython"),
        AF_OUTPUT_ROOT=str(output),
        AF_SOURCE_RESCUE_ROOT=str(source),
        AF_VLLM_IMAGE=str(image),
        AF_HF_HOME=str(tmp_path / "hf"),
        AF_JUDGE_REVISION="a" * 40,
        AF_RESUME="0",
    )
    command = ["bash", str(repo / "scripts/hpc/submit_repair_comparison_aces.sh")]
    first = subprocess.run(command, env=env, capture_output=True, text=True)
    calls = json.loads(log.read_text())
    assert len(calls) == (1 if timeout else 2)
    assert first.returncode == (1 if timeout else 0), first.stderr
    repeated = subprocess.run(command, env=env, capture_output=True, text=True)
    assert json.loads(log.read_text()) == calls
    assert repeated.returncode != 0 if timeout else repeated.returncode == 0
    if not timeout:
        assert "--dependency=afterok:101" in calls[1]
        assert "--gres=gpu:h100:2" in calls[1]
        env["AF_RESUME"] = "1"
        resumed = subprocess.run(command, env=env, capture_output=True, text=True)
        assert resumed.returncode == 0, resumed.stderr
        assert len(json.loads(log.read_text())) == 3
        manifest = json.loads((output / "submission_manifest.json").read_text())
        assert manifest["prior_job"] == "102" and manifest["job_id"] == "103"
