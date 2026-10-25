"""Endpoint requalification preserves source evidence and bounded continuation."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.research import construction_endpoint_requalification as c
from tests.test_construction_fit_reliability import endpoint
from tests.test_construction_timescale_controls import source_data


def _receipt(value, identity, stage, seconds):
    return {
        "identity": {
            "campaign": identity,
            "stage": stage,
            "allowance_seconds": seconds,
        },
        "status": "complete",
        "value": value,
        "observed_progress": value,
        "progress_sha256": c.public.content_sha256(value),
        "wall_seconds": 1.0,
        "cpu_seconds": 1.0,
        "budget_charge_seconds": 1.0,
        "accounting_complete": True,
    }


def provenance(monkeypatch):
    """Tiny source with real provenance reconstruction and nine terminal records."""
    old = source_data()
    old["protocol"] = c.source.RECOVERY_INPUT_PROTOCOL
    old["prior_controls"] = {"records": []}
    p = c.source.Policy(recovery_mode="project_first_local")
    plan = {
        "protocol": c.source.RECOVERY_PROTOCOL,
        "policy": p.model_dump(mode="json"),
        "tasks": c.source._tasks(old, p),
        "inputs_sha256": c.public.content_sha256(old),
        "numerical_contract": c.source._contract(p),
        "time_grid": c.source.time_grid(c.source._training_inputs(old)["training"]),
        "llm_calls": 0,
        "old_allowances_renewed": False,
        "validation_data_opened": False,
        "test_data_opened": False,
        "automatic_revision": False,
        "automatic_promotion": False,
    }
    plan_hash, records = c.public.content_sha256(plan), []
    for task in plan["tasks"]:
        identity = {"plan_sha256": plan_hash, "task": task}
        ops = (
            {}
            if task["index"] == 9
            else {
                "incumbent_verification": _receipt(
                    deepcopy(old["endpoint"]),
                    identity,
                    "incumbent_verification",
                    c.source._allowances(p)["incumbent_verification"],
                )
            }
        )
        # Frozen source slices must retain their declared rate.
        if task["frozen_kappa"] is not None:
            params = old["endpoint"]["effective_parameters"] | {
                c.source.RATE: -task["frozen_kappa"]
            }
            ep = deepcopy(old["endpoint"])
            ep.update(
                parameters=params,
                effective_parameters=params,
                reduced_vector=list(params.values()),
            )
            ops["incumbent_verification"] = _receipt(
                ep,
                identity,
                "incumbent_verification",
                c.source._allowances(p)["incumbent_verification"],
            )
        terminal = c.source._assemble(identity, old, ops, terminal=task["index"] != 9)
        records.append(
            {
                "task_index": task["index"],
                "terminal_present": task["index"] != 9,
                "terminal_sha256": c.public.content_sha256(terminal)
                if task["index"] != 9
                else None,
                "operations": ops,
            }
        )
    report = c._source_report(plan, old, records)
    data = {
        "protocol": c.INPUT_PROTOCOL,
        "source_plan": plan,
        "source_plan_sha256": plan_hash,
        "source_inputs": old,
        "source_inputs_sha256": c.public.content_sha256(old),
        "source_report": report,
        "source_report_sha256": c.public.content_sha256(report),
        "source_records": records,
        "source_log_hashes": {},
        "old_allowances_renewed": False,
        "validation_exported": False,
        "test_data_opened": False,
    }
    monkeypatch.setattr(c.source, "_validate", lambda data: None)
    return data


def test_authenticate_all_source_receipts_and_report(monkeypatch):
    data = provenance(monkeypatch)
    c._validate(data)
    assert data["source_report"]["recorded"] == 9
    changed = deepcopy(data)
    changed["source_records"][7]["operations"]["incumbent_verification"]["value"][
        "training_nmse"
    ] = 0
    with pytest.raises(ValueError, match="binding differs"):
        c._validate(changed)
    changed = deepcopy(data)
    changed["source_report"]["recorded"] = 10
    with pytest.raises(ValueError, match="report differs"):
        c._validate(changed)


def test_reserved_missing_task_is_not_silently_renewed(monkeypatch):
    data = provenance(monkeypatch)
    changed = deepcopy(data)
    changed["source_records"][9]["operations"] = deepcopy(
        changed["source_records"][8]["operations"]
    )
    with pytest.raises(ValueError, match="binding differs"):
        c._validate(changed)


def verification(value, score=0.3, valid=True):
    out = endpoint(value, score, valid)
    out.update(
        effective_parameters=value,
        reduced_vector=list(value.values()),
        alternate_training_nmse=score,
        solver_prediction_discrepancy=0,
        methods=["certified_affine", "original_Radau"],
        verification_calls_started=2,
        verification_calls_completed=2,
        verification_calls_failed=0,
        status="complete",
    )
    return out


def test_search_eligibility_is_separate_from_strict_certification():
    value = verification({"x": 1}, 51069064457.80239, False)
    value.update(
        alternate_training_nmse=51069064457.782486,
        solver_prediction_discrepancy=1.923181116580963e-6,
    )
    op = {"status": "complete", "value": value}
    result = c.search_eligibility(op)
    assert result["search_eligible"]
    assert result["strict_endpoint_verified"] is False
    assert c.pilot._verified(op) is None
    for changes in (
        {"solver_prediction_discrepancy": 1e6},
        {"alternate_training_nmse": float("inf")},
        {"verification_calls_failed": 1},
        {"verification_calls_completed": 1},
    ):
        assert not c.search_eligibility(
            {"status": "complete", "value": value | changes}
        )["search_eligible"]
    assert not c.search_eligibility({"status": "interrupted", "value": value})[
        "search_eligible"
    ]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    data = provenance(monkeypatch)
    names = list(data["source_inputs"]["endpoint"]["effective_parameters"])
    monkeypatch.setattr(
        c, "scale_floors", lambda data: {"floors": dict.fromkeys(names, 1.0)}
    )
    monkeypatch.setattr(c.public, "_source_identity", lambda: "a" * 64)
    monkeypatch.setattr(c.pilot, "_runtime", lambda: {"runtime": "fixed"})
    root = tmp_path / "campaign"
    c.prepare(
        root,
        data,
        c.Policy(
            search_seconds=8,
            diagnostic_seconds=4,
            verification_seconds=3,
            maximum_calls=10,
        ),
    )
    return root, data


def install_fake(
    monkeypatch, *, converged=False, initial_valid=True, interrupt=False, equal=False
):
    events = []
    monkeypatch.setattr(
        c, "PrefixFitProblem", SimpleNamespace(model_validate=lambda p: p)
    )
    monkeypatch.setattr(c, "PublicSplit", SimpleNamespace(model_validate=lambda p: p))
    monkeypatch.setattr(
        c.kernel, "NumericalPolicy", SimpleNamespace(model_validate=lambda p: p)
    )
    monkeypatch.setattr(c, "_certify_scale_structure", lambda o, a: None)
    monkeypatch.setattr(
        c.pilot.previous,
        "_verified_coordinates",
        lambda o, a, ep: ep["effective_parameters"],
    )

    def make(problem, training, settings, anchor):
        names = tuple(c.source.OUTER_ROLES.values()) + tuple(
            f"inner_{i}" for i in range(21)
        )
        a = SimpleNamespace(
            names=names,
            units=np.ones(27),
            vector=lambda p: np.array([p[n] for n in names]),
            parameters=lambda v: dict(zip(names, v.tolist(), strict=True)),
            affine_audit={},
            recovery_audit={},
            configure_local_probes=lambda p: None,
        )
        a.original = a
        return a, a, None

    monkeypatch.setattr(c.profiling, "make_oracle", make)
    monkeypatch.setattr(c.profiling, "EndpointChartOracle", lambda a, p, f: a)
    monkeypatch.setattr(
        c.profiling,
        "MultiOutputProfile",
        lambda a: SimpleNamespace(outer=tuple(range(6)), inner=tuple(range(6, 27))),
    )

    def verify(o, a, params, seconds, save, **kwargs):
        events.append("verify")
        result = verification(
            params, params["inner_0"], initial_valid or params["inner_0"] == 0.1
        )
        save(result)
        return result

    monkeypatch.setattr(c.numerical, "verify_endpoint_ladder", verify)

    def diagnose(
        o, a, params, certificate, fit, seconds, save, *, include_raw_evidence
    ):
        events.append("diagnose")
        assert include_raw_evidence
        result = {
            "status": "complete",
            "assessment": {
                "classification": "locally_converged_poor_fit"
                if converged
                else "optimization_unresolved"
            },
            "diagnosed_parameters_sha256": c.public.content_sha256(
                certificate["parameters"]
            ),
            "verification_sha256": c.public.content_sha256(certificate),
        }
        save(result)
        return result

    monkeypatch.setattr(c.pilot, "diagnose_profile", diagnose)

    def fixed(a, f, params, seconds, save):
        events.append("project")
        result = {"best": {"parameters": params, "training_nmse": params["inner_0"]}}
        save(result)
        return result

    monkeypatch.setattr(c.pilot, "_fixed_inner", fixed)

    def search(a, f, params, seconds, calls, save, *, preserve_full_history):
        events.append("search")
        assert preserve_full_history
        result = {
            "best": {
                "parameters": params if equal else params | {"inner_0": 0.1},
                "training_nmse": params["inner_0"] if equal else 0.1,
            },
            "stop_reason": "xtol",
            "search_calls_started": 1,
        }
        save(result)
        if interrupt:
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(c.pilot, "_profile_search", search)
    return events


def test_stationary_endpoint_skips_polish_and_resume_has_no_new_work(
    prepared, monkeypatch
):
    root, _ = prepared
    events = install_fake(monkeypatch, converged=True)
    result = c.run_task(root, 0)
    assert events == ["verify", "diagnose"]
    assert result["retained"]["training_nmse"] == 1.0
    assert result["optimization_status"] == "locally_converged_poor_fit"
    before = events[:]
    assert c.run_task(root, 0) == result
    assert events == before
    assert c.report(root)["recorded"] == 1
    assert c.inspect(root)["status"] == "incomplete"


def test_unresolved_endpoint_polishes_verifies_and_rediagnoses(prepared, monkeypatch):
    root, _ = prepared
    events = install_fake(monkeypatch)
    result = c.run_task(root, 1)
    assert events == ["verify", "diagnose", "search", "verify", "diagnose"]
    assert result["retained"]["training_nmse"] == 0.1
    assert result["historical_endpoint"]["training_nmse"] == 1.0
    assert result["retained_origin"] == "continuation_verification"
    assert result["cost"]["operations"] == 5


def test_equal_polishing_point_still_gets_matching_final_diagnosis(
    prepared, monkeypatch
):
    root, _ = prepared
    events = install_fake(monkeypatch, equal=True)
    result = c.run_task(root, 0)
    assert events.count("diagnose") == 2
    assert result["retained_origin"] == "incumbent_verification"


def test_missing_start_recovery_projects_without_initial_diagnosis(
    prepared, monkeypatch
):
    root, _ = prepared
    events = install_fake(monkeypatch)
    result = c.run_task(root, 2)
    assert events == ["project", "verify", "search", "verify", "diagnose"]
    assert result["historical_endpoint"] is None
    assert result["old_allowances_renewed"] is False
    allowances = c._allowances(
        c.Policy(search_seconds=8), {"kind": "missing_start_recovery"}
    )
    assert allowances["fixed_inner"] - 90 + allowances["continuation_search"] - 150 == 8


def test_search_can_use_inconclusive_start_without_retaining_it(prepared, monkeypatch):
    root, _ = prepared
    events = install_fake(monkeypatch, initial_valid=False)
    result = c.run_task(root, 2)
    assert "search" in events
    assert result["selected_start"] is None
    assert result["search_eligibility"]["search_eligible"]
    assert result["retained"]["training_nmse"] == 0.1


def test_interrupted_search_reuses_saved_checkpoint_without_budget_reset(
    prepared, monkeypatch
):
    root, _ = prepared
    events = install_fake(monkeypatch, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 0)
    result = c.run_task(root, 0)
    assert events.count("search") == 1
    assert result["operations"]["continuation_search"]["status"] == "interrupted"
    assert result["retained"]["training_nmse"] == 0.1


def test_stale_diagnosis_and_plan_are_rejected(prepared, monkeypatch):
    root, _ = prepared
    install_fake(monkeypatch)
    c.run_task(root, 0)
    folder = root / "results/task_00/operations/retained_diagnosis"
    receipt = read_seal(folder / "result.json")
    receipt["value"]["diagnosed_parameters_sha256"] = "b" * 64
    (folder / "result.json").unlink()
    seal(folder / "result.json", receipt)
    with pytest.raises(ValueError, match="diagnosis differs"):
        c.report(root)
    plan = read_seal(root / "plan.json")
    plan["tasks"][0]["source_task_index"] = 9
    (root / "plan.json").unlink()
    seal(root / "plan.json", plan)
    with pytest.raises(ValueError, match="plan identity"):
        c.verify(root)


def test_policy_and_invalid_task_are_bounded(prepared):
    root, _ = prepared
    with pytest.raises(ValueError):
        c.Policy(search_seconds=601)
    with pytest.raises(ValueError, match="unknown"):
        c.run_task(root, 3)


def test_exact_scale_structure_certificate_handles_addition_order_and_rejects_changes():
    import casadi as ca

    names = [
        *c.source.OUTER_ROLES.values(),
        "coefficient:I:insulin_pmol_per_kg_min",
        "coefficient:I:meal_rate_g_per_min",
        "coefficient:U:Uii",
        "k_EGP_term_11_1_db1d47d8a7",
        "k_prod_term_11_0_e3a835f757",
    ]
    states = ("Gp", "I", "U", "I_mem")
    original = SimpleNamespace(
        system=SimpleNamespace(
            model=SimpleNamespace(state_names=states),
            inputs=("meal_rate_g_per_min", "insulin_pmol_per_kg_min", "EGP", "Uii"),
        )
    )

    def rhs(vector, x, u, *, wrong=False):
        p = dict(zip(names, ca.vertsplit(vector), strict=True))
        # Associative ordering deliberately differs from the expected template.
        return ca.vertcat(
            p["k_EGP_term_11_1_db1d47d8a7"] * u["EGP"]
            - x["U"] / p[c.source.OUTER_ROLES["glucose_delay"]]
            + p["k_prod_term_11_0_e3a835f757"] * u["meal_rate_g_per_min"],
            p["coefficient:I:meal_rate_g_per_min"] * u["meal_rate_g_per_min"]
            + p[c.source.OUTER_ROLES["feedback"]] * x["I_mem"]
            + p[c.source.RATE] * x["I"]
            + p["coefficient:I:insulin_pmol_per_kg_min"] * u["insulin_pmol_per_kg_min"],
            x["I_mem"] * (-1 if wrong else 1)
            - p[c.source.OUTER_ROLES["utilization_decay"]] * x["U"]
            + p["coefficient:U:Uii"] * u["Uii"],
            p[c.source.OUTER_ROLES["activation"]] * x["I"]
            - p[c.source.OUTER_ROLES["memory_decay"]] * x["I_mem"],
        )

    oracle = SimpleNamespace(names=names, reduction=SimpleNamespace(symbolic_rhs=rhs))
    c._certify_scale_structure(original, oracle)
    oracle.reduction.symbolic_rhs = lambda *args: rhs(*args, wrong=True)
    with pytest.raises(ValueError, match="certificate failed"):
        c._certify_scale_structure(original, oracle)


def test_dimensional_floors_transform_with_observed_channel_units():
    row = {
        "trajectory_id": "a",
        "time": [0, 10, 100],
        "targets": {"Gp": [100, 200, 150], "I": [5, 10, 15], "U": [1, 2, 3]},
        "external_inputs": {
            "meal_rate_g_per_min": [0, 1, 0],
            "insulin_pmol_per_kg_min": [0, 2, 0],
        },
        "auxiliaries": {"EGP": [2, 2, 2], "Uii": [1, 1, 1]},
    }
    names = [
        *c.source.OUTER_ROLES.values(),
        "coefficient:I:insulin_pmol_per_kg_min",
        "coefficient:I:meal_rate_g_per_min",
        "coefficient:U:Uii",
        "k_EGP_term_11_1_db1d47d8a7",
        "k_prod_term_11_0_e3a835f757",
        "trajectory:a:init_I_mem_value",
    ]
    data = {
        "source_inputs": {
            "profile_outer_names": names[:6],
            "profile_inner_names": names[6:],
            "historical_inputs": {
                "historical_inputs": {"training_inputs": {"training": {"rows": [row]}}}
            },
        }
    }
    first = c.scale_floors(data)
    changed = deepcopy(data)
    changed["source_inputs"]["historical_inputs"]["historical_inputs"][
        "training_inputs"
    ]["training"]["rows"][0]["targets"]["U"] = [10, 20, 30]
    second = c.scale_floors(changed)
    f, g = first["floors"], second["floors"]
    for name in (
        c.source.OUTER_ROLES["glucose_delay"],
        c.source.OUTER_ROLES["activation"],
        "trajectory:a:init_I_mem_value",
        "coefficient:U:Uii",
    ):
        assert g[name] == pytest.approx(10 * f[name])
    assert g[c.source.OUTER_ROLES["feedback"]] == pytest.approx(
        f[c.source.OUTER_ROLES["feedback"]] / 10
    )
    assert g[c.source.RATE] == f[c.source.RATE]


def test_sealed_export_roundtrip_restores_historical_operation_order(
    tmp_path, monkeypatch
):
    data = provenance(monkeypatch)
    plan, old = data["source_plan"], data["source_inputs"]
    task = plan["tasks"][7]
    identity = {"plan_sha256": data["source_plan_sha256"], "task": task}
    allowed = c.source._allowances(c.source.Policy.model_validate(plan["policy"]))
    incumbent = deepcopy(
        data["source_records"][7]["operations"]["incumbent_verification"]
    )
    incumbent.update(wall_seconds=0.3, cpu_seconds=0.3, budget_charge_seconds=0.3)
    operations = {
        "fixed_inner": _receipt({}, identity, "fixed_inner", allowed["fixed_inner"]),
        "incumbent_verification": incumbent,
        "continuation_search": _receipt(
            {}, identity, "continuation_search", allowed["continuation_search"]
        ),
    }
    for name, amount in (("fixed_inner", 0.2), ("continuation_search", 0.1)):
        operations[name].update(
            wall_seconds=amount, cpu_seconds=amount, budget_charge_seconds=amount
        )
    for operation in operations.values():
        operation["value"]["timing"] = {
            "callback_cpu_seconds": operation["cpu_seconds"]
        }
        operation["observed_progress"] = deepcopy(operation["value"])
        operation["progress_sha256"] = c.public.content_sha256(
            operation["observed_progress"]
        )
    terminal = c.source._assemble(identity, old, operations, terminal=True)
    lexical = c.source._assemble(
        identity, old, dict(sorted(operations.items())), terminal=True
    )
    assert (
        terminal["cost"]["callback_cpu_seconds_total"]
        != lexical["cost"]["callback_cpu_seconds_total"]
    )
    record = data["source_records"][7]
    record.update(
        operations=operations, terminal_sha256=c.public.content_sha256(terminal)
    )
    data["source_report"] = c._source_report(plan, old, data["source_records"])
    data["source_report_sha256"] = c.public.content_sha256(data["source_report"])
    c._validate(data)
    path = tmp_path / "exported-inputs.json"
    seal(path, data)
    returned = read_seal(path)
    assert list(returned["source_records"][7]["operations"]) == sorted(operations)
    c._validate(returned)
    assert c._parent(returned, 7) == terminal
