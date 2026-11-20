"""Scientific and operational boundaries of the three-origin breadth pilot."""

import json
from copy import deepcopy

import pytest

from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.llm.staged_topology import strict_provider_schema
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import t2_breadth as science
from autoformalism.schemas.staged_topology import PublicScientificBrief
from scripts import submit_t2_breadth as aces
from scripts import t2_breadth as pilot
from tests.test_construction_initialization_graph_cli import reply, tokenizer

MEAL = "meal_rate_g_per_min"
INFUSION = "insulin_pmol_per_kg_min"


def proposal():
    equations = [
        ("Gp", "dynamic", [MEAL, "EGP", "U"], f"{MEAL}+EGP-U"),
        ("I", "dynamic", ["Gp", "I", INFUSION], f"a*Gp-a*I+{INFUSION}"),
        ("U", "algebraic", ["Uii", "X"], "Uii+X"),
        ("X", "dynamic", ["I", "X"], "a*(I-X)"),
    ]
    topo = {
        "hypothesis": "Delayed insulin action with basal production.",
        "components": [
            {"name": n, "kind": k, "role": n, "sources": s} for n, k, s, _ in equations
        ],
    }
    laws = {
        "formulas": [{"component": n, "expression": ex} for n, _, _, ex in equations],
        "parameters": [{"name": "a", "role": "rate"}],
        "latent_initials": [
            {
                "state": "X",
                "initial": {
                    "mode": "value",
                    "scope": "trajectory",
                    "role": "coefficient",
                },
            }
        ],
    }
    return topo, laws


def public():
    ctx = ValidationContext(
        targets=["Gp", "I", "U"],
        auxiliaries=["Uii", "EGP"],
        external_inputs=[MEAL, INFUSION],
    )
    brief = PublicScientificBrief.model_validate(
        {
            "scientific_context": "A new public T2 model.",
            "public_variables": [
                {"name": x, "data_role": r}
                for r, ns in [
                    ("target", ctx.targets),
                    ("external_input", ctx.external_inputs),
                    ("auxiliary", ctx.auxiliaries),
                    ("time", ["t"]),
                ]
                for x in ns
            ],
            "requirements": [
                {
                    "id": "meal",
                    "drivers": [MEAL],
                    "targets": ["Gp"],
                    "public_requirement": "Meal appearance.",
                },
                {
                    "id": "memory",
                    "drivers": ["I"],
                    "targets": ["U"],
                    "public_requirement": "Delayed action.",
                    "requires_dynamic_memory": True,
                },
            ],
            "target_dependencies": [
                {
                    "target": "U",
                    "acceptable_sources": ["Uii"],
                    "public_requirement": "Independent uptake.",
                }
            ],
        }
    )
    return ctx, brief


def test_fresh_public_model_compiles_and_keeps_prefix_root():
    ctx, brief = public()
    topo, laws = proposal()
    graph = science.Topology.model_validate(topo)
    science.check_topology(graph, ctx, brief)
    model = science.assemble(graph, science.Laws.model_validate(laws), ctx)
    assert model.base_candidate.parent_candidate_id is None
    assert model.graph.rules["X"].initial.scope == "trajectory"
    assert model.parameter_guesses.keys() == {"a"}
    assert len(science.check_paths(model, brief, {})) == 3
    assert science.glucose_balance(model)["status"] == "nonzero_expression"
    assert science.balance_view(model)["balance_sha256"]
    # A polynomial zero at ingestion=0 is an identity for all declared parameters.
    topo["components"][0]["sources"].remove("EGP")
    laws["formulas"][0]["expression"] = f"3*{MEAL}-U"
    model = science.assemble(
        science.Topology.model_validate(topo), science.Laws.model_validate(laws), ctx
    )
    assert science.glucose_balance(model)["status"] == "identically_zero"
    topo["components"][0]["sources"].append("EGP")
    laws["formulas"][0]["expression"] = f"EGP/a+{MEAL}-U"
    mixed = science.assemble(
        science.Topology.model_validate(topo), science.Laws.model_validate(laws), ctx
    )
    assert science.glucose_balance(mixed)["status"] == "deferred"


def test_undeclared_sources_topology_path_and_unbound_symbol():
    ctx, brief = public()
    topo, laws = proposal()
    bad = deepcopy(topo)
    bad["components"][0]["sources"].append("D")
    with pytest.raises(ValueError, match=r"undeclared.*D"):
        science.check_topology(science.Topology.model_validate(bad), ctx, brief)
    bad = deepcopy(topo)
    bad["components"][-1]["sources"] = ["X"]
    with pytest.raises(ValueError, match="Missing required path"):
        science.check_topology(science.Topology.model_validate(bad), ctx, brief)
    laws["formulas"][1]["expression"] += "+direct_ingestion"
    with pytest.raises(ValueError, match="UNKNOWN_FORMULA_SYMBOL"):
        science.assemble(
            science.Topology.model_validate(topo),
            science.Laws.model_validate(laws),
            ctx,
        )


def test_current_equation_notation_and_unambiguous_fresh_metadata():
    ctx, _ = public()
    topo, laws = proposal()
    checks = []
    topo["components"][1]["sources"].remove("I")
    laws["parameters"][0]["name"] = "tau"
    laws["formulas"][0]["expression"] = f"dGp/dt = {MEAL}+EGP-U"
    laws["formulas"][1]["expression"] = f"dI/dt = τ*Gp(t)-τ*I(t)+{INFUSION}+t"
    laws["formulas"][2]["expression"] = "U(t) = Uii(t)+X(t)"
    laws["formulas"][3]["expression"] = "d(X)/dt = tau*(I-X)"
    model = science.assemble(
        science.Topology.model_validate(topo),
        science.Laws.model_validate(laws),
        ctx,
        normalizations=checks,
    )
    rhs = next(
        row.rhs for row in model.base_candidate.state_equations if row.state == "I"
    )
    assert science._symbols(rhs, "I") == {"tau", "Gp", "I", INFUSION, "t"}
    assert {"component": "I", "identifier_spellings": {"τ": "tau"}} in checks
    assert any(
        r.get("sources_after") == sorted(["Gp", "I", INFUSION, "t"]) for r in checks
    )
    assert (
        "τ" in laws["formulas"][1]["expression"]
        and "I" not in topo["components"][1]["sources"]
    )


@pytest.mark.parametrize(
    ("row", "expression", "error"),
    [
        (0, "dGpx/dt = 1", "RHS_LABEL"),
        (0, "dGp/ds = 1", "RHS_LABEL"),
        (0, "Gp(t) = 1", "RHS_LABEL"),
        (0, "dGp/dt = dI/dt = 1", "RHS_LABEL"),
        (2, "dU/dt = Uii+X", "RHS_LABEL"),
        (2, "U(t-1) = Uii+X", "RHS_LABEL"),
        (2, "U(t) = Uii(t-1)+X", "UNSUPPORTED_FUNCTION"),
        (1, f"dI/dt = a*Gp-I+{INFUSION}+τ_unknown", "UNKNOWN_FORMULA_SYMBOL"),
        (0, f"dGp/dt = {MEAL}+EGP-X", "GRAPH_FORMULA_DISAGREEMENT"),
        (1, f"dI/dt = Gp+{INFUSION}", "GRAPH_FORMULA_DISAGREEMENT"),
    ],
)
def test_current_equation_reader_preserves_scientific_boundaries(
    row, expression, error
):
    ctx, _ = public()
    topo, laws = proposal()
    laws["formulas"][row]["expression"] = expression
    with pytest.raises((ValueError, ModelValidationError), match=error):
        science.assemble(
            science.Topology.model_validate(topo),
            science.Laws.model_validate(laws),
            ctx,
        )


def test_measured_initial_value_cannot_become_a_private_root():
    ctx, _ = public()
    topology, laws = proposal()
    laws["latent_initials"].append({**laws["latent_initials"][0], "state": "I"})
    with pytest.raises(science.RevisionValidationError) as error:
        science.assemble(
            science.Topology.model_validate(topology),
            science.Laws.model_validate(laws),
            ctx,
        )
    assert error.value.code == "LATENT_INITIALIZERS"
    assert error.value.details["extra"] == ["I"]


def test_auxiliary_can_be_supplied_or_explicitly_modeled():
    ctx, brief = public()
    topo, laws = proposal()
    topo["components"].append(
        {
            "name": "Q",
            "kind": "dynamic",
            "role": "basal disposal",
            "represents_auxiliary": "Uii",
            "sources": [],
        }
    )
    topo["components"][2]["sources"] = ["Q", "X"]
    laws["formulas"][2]["expression"] = "Q+X"
    laws["formulas"].append({"component": "Q", "expression": "0"})
    laws["latent_initials"].append(
        {
            "state": "Q",
            "initial": {
                "mode": "map",
                "expression": "u0",
                "parameters": [],
                "public_initials": [{"alias": "u0", "channel": "Uii"}],
                "state_initials": [],
            },
        }
    )
    graph = science.Topology.model_validate(topo)
    science.check_topology(graph, ctx, brief)
    model = science.assemble(graph, science.Laws.model_validate(laws), ctx)
    assert all(
        r["status"] == "pass"
        for r in science.check_paths(model, brief, graph.aliases())
    )


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    ctx, brief = public()
    topo, laws = proposal()
    problem = science.assemble(
        science.Topology.model_validate(topo), science.Laws.model_validate(laws), ctx
    )
    row = {
        "trajectory_id": "train_000",
        "external_inputs": {MEAL: [0.0]},
        "targets": {"Gp": [100.0], "U": [2.0]},
    }
    windows = [
        {
            "trajectory": "train_001",
            "meal": {"dose": 90},
            "observed_peak_min": 101,
            "observed_rise_pmol_L": 2,
        }
    ]
    inputs = {
        "brief": brief.model_dump(mode="json"),
        "aliases": {},
        "training": {"rows": [row]},
        "evidence": {"context": ctx.model_dump(mode="json")},
        "glucose_windows": [],
        "fitting_contract": {
            "training_sha256": "1" * 64,
            "prefix_policy": {
                "duration": 8,
                "future_inputs": list(ctx.external_inputs),
                "future_auxiliaries": list(ctx.auxiliaries),
            },
        },
        "c15": {
            "problem": problem.model_dump(mode="json"),
            "boundary_feedback": {"ingestion_windows": windows},
            "boundary_witness": {},
            "repair_history": {"previous_attempt": {"fixed_parameter_replay": {}}},
            "result_identity": {},
        },
        "c21": {
            "problem": problem.model_dump(mode="json"),
            "training_nmse": 1.5,
            "anchor_sha256": "a" * 64,
        },
        "last_failure": {"error": "UNKNOWN_FORMULA_SYMBOL"},
    }
    input_file = tmp_path / "intake.json"
    intake = sealed_write(input_file, inputs)
    monkeypatch.setattr(pilot, "INTAKE", intake["artifact_sha256"])
    monkeypatch.setattr(
        pilot.structural, "_assessment", lambda *a, **k: {"status": "deferred"}
    )
    root = tmp_path / "campaign"
    pilot.prepare(root, input_file)
    return root


def test_budget_full_roster_fresh_independence_and_resume(experiment, monkeypatch):
    root = experiment
    topo, laws = proposal()
    calls = []

    def transport(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"])
        name = body["response_format"]["json_schema"]["name"]
        calls.append((name, body["seed"]))
        if name.startswith("Five"):
            return reply({"alternatives": [{"incomplete_option": i} for i in range(5)]})
        if name in {"InScopeRevisionReply", "BoundaryRevision"}:
            assert "rejection" in payload and "previous_reply" in payload
            return reply({})
        assert "current_model" not in payload
        assert (
            "prior_failure" not in payload
            and "current_model" not in body["messages"][0]["content"]
        )
        if name == "Topology":
            return reply(topo)
        if name == "Laws":
            return reply({**laws, "latent_initials": []})
        if name == "FreshCorrection":
            assert science.Topology.model_validate(
                payload["previous_topology"]
            ) == science.Topology.model_validate(topo)
            # The last round succeeds. This exercises all 57 physical slots.
            if payload.get("correction_number", 1) < 3:
                return reply(
                    {"topology": topo, "laws": {**laws, "latent_initials": []}}
                )
            return reply({"topology": topo, "laws": laws})
        raise AssertionError(name)

    summary = pilot.run(
        root, "http://127.0.0.1", transport=transport, token_transport=tokenizer
    )
    assert [name for name, _ in calls[:7]] == [
        "FiveInsulinRepairs",
        "FiveGlucoseRepairs",
    ] + ["Topology"] * 5
    assert len({seed for _, seed in calls[2:7]}) == 5
    assert len(calls) == summary["physical_requests"] == pilot.MAX_CALLS == 57
    assert summary["groups"] == {
        "c15": {"rejected": 5},
        "c21": {"rejected": 5},
        "fresh": {"eligible": 5},
    }
    assert (
        summary["eligible_after_correction"] == 5
        and summary["corrections_attempted"] == 45
    )
    assert [d["physical_requests"] for d in summary["depth_comparison"]] == [27, 42, 57]
    assert [d["eligible"] for d in summary["depth_comparison"]] == [0, 0, 5]
    assert summary["first_27_physical_calls"]["eligible"] == 0
    assert summary["representatives"] == ["fresh-100"]
    handoff = sealed_read(root / "handoff/fresh-100.json")
    assert handoff["problem"]["base_candidate"]["parent_candidate_id"] is None
    assert (
        pilot.run(
            root,
            "http://127.0.0.1",
            transport=lambda *a: pytest.fail("generation"),
            token_transport=tokenizer,
        )
        == summary
    )
    assert pilot.inspect(root)["recorded"] == 15


def test_bad_topology_can_be_corrected_once_and_then_receive_formulas(experiment):
    topo, laws = proposal()

    def transport(url, body, timeout):
        name = body["response_format"]["json_schema"]["name"]
        data = json.loads(body["messages"][1]["content"])
        if name.startswith("Five"):
            return reply({"alternatives": []})
        if name == "Topology":
            return reply(
                topo
                if "rejection" in data
                else {"components": [], "hypothesis": "Incomplete"}
            )
        if name == "Laws":
            return reply({**laws, "parameters": []})
        assert name == "FreshCorrection" and data["correction_number"] == 2
        return reply({"topology": topo, "laws": laws})

    r = pilot.run(
        experiment, "http://127.0.0.1", transport=transport, token_transport=tokenizer
    )
    assert r["groups"] == {
        "c15": {"unavailable": 5},
        "c21": {"unavailable": 5},
        "fresh": {"eligible": 5},
    }
    assert r["physical_requests"] == 22
    assert [d["physical_requests"] for d in r["depth_comparison"]] == [17, 22, 22]
    assert r["first_27_physical_calls"]["eligible"] == 5
    assert [
        h["stage"]
        for h in sealed_read(experiment / "results/fresh-104.json")["history"]
    ] == ["topology", "topology_correction", "laws", "complete_correction"]


def test_scheduler_idempotency_and_resume(experiment, monkeypatch, tmp_path):
    root = experiment
    image = tmp_path / "image"
    image.write_bytes(b"test")
    monkeypatch.setattr(aces.aces, "_assets", lambda: (image, {}))
    monkeypatch.setattr(aces.aces.harmony_vocab, "verify", lambda d: {"sha256": "test"})
    jobs = []

    def submit(directory, argv):
        assert argv[-1] == "t2-breadth" and "--gres=gpu:h100:2" in argv
        jobs.append(str(100 + len(jobs)))
        return jobs[-1]

    monkeypatch.setattr(aces, "_submit_job", submit)
    first = aces.submit(root)
    assert (
        aces.submit(root) == first
        and aces.verify(root, True)["maximum_generation_requests"] == 57
    )
    monkeypatch.setattr(aces, "_finished", lambda job: False)
    assert aces.submit(root, resume=True) == first
    monkeypatch.setattr(aces, "_finished", lambda job: True)
    assert aces.submit(root, resume=True)["job"] != "100"
    assert len(jobs) == 2


def test_duplicate_sibling_and_provider_failure_remain_distinguishable(experiment):
    def transport(url, body, timeout):
        schema = body["response_format"]["json_schema"]["name"]
        if schema == "FiveInsulinRepairs":
            return reply({"alternatives": [{"broken": True}] * 5})
        if schema == "FiveGlucoseRepairs":
            broken = reply({})
            broken["choices"][0]["finish_reason"] = "length"
            return broken
        if schema == "Topology":
            raise TimeoutError("simulated failure")
        assert schema == "BoundaryRevision"
        return reply({})

    result = pilot.run(
        experiment, "http://127.0.0.1", transport=transport, token_transport=tokenizer
    )
    assert result["physical_requests"] == 10
    assert result["unavailable_branches"] == 10
    assert result["corrections_attempted"] == 3
    assert (
        sealed_read(experiment / "results/c15-3.json")["correction_duplicate_of"]
        == "c15-0"
    )


def test_interrupted_generation_spends_its_slot_and_other_branches_resume(experiment):
    calls = []

    def interrupted(url, body, timeout):
        calls.append(body["response_format"]["json_schema"]["name"])
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        pilot.run(
            experiment,
            "http://127.0.0.1",
            transport=interrupted,
            token_transport=tokenizer,
        )

    def remaining(url, body, timeout):
        name = body["response_format"]["json_schema"]["name"]
        assert name != "FiveInsulinRepairs"
        calls.append(name)
        raise TimeoutError("simulated timeout")

    summary = pilot.run(
        experiment, "http://127.0.0.1", transport=remaining, token_transport=tokenizer
    )
    assert summary["physical_requests"] == len(calls) == 7
    assert summary["provider_outcomes"]["uncertain"] == 1
    assert summary["usage_missing"] == 7
    assert summary["unavailable_branches"] == 15


def test_laws_can_follow_third_topology_correction(experiment):
    topo, laws = proposal()

    def transport(url, body, timeout):
        schema = body["response_format"]["json_schema"]["name"]
        packet = json.loads(body["messages"][1]["content"])
        if schema.startswith("Five"):
            return reply({"alternatives": []})
        if schema == "Topology":
            return reply(topo if packet.get("correction_number") == 3 else {})
        assert schema == "Laws"
        return reply(laws)

    result = pilot.run(
        experiment, "http://127.0.0.1", transport=transport, token_transport=tokenizer
    )
    assert [r["eligible"] for r in result["depth_comparison"]] == [0, 0, 5]
    assert [r["physical_requests"] for r in result["depth_comparison"]] == [12, 17, 27]
    assert result["first_27_physical_calls"]["eligible"] == 5
    row = sealed_read(experiment / "results/fresh-104.json")
    assert row["correction_count"] == 3 and row["history"][-1]["stage"] == "laws"


def named_proposal():
    topo, laws = proposal()
    targets = {}
    additional = []
    for component in topo["components"]:
        if component["name"] in ("Gp", "I", "U"):
            targets[component["name"]] = {
                k: v for k, v in component.items() if k != "name"
            }
        else:
            additional.append(component)
    return {
        "hypothesis": topo["hypothesis"],
        "targets": targets,
        "additional_components": additional,
    }, laws


def test_public_names_are_schema_keys_and_preserve_independent_choices():
    ctx, brief = public()
    keyed, _ = named_proposal()
    form, _ = science.named_topology_forms(tuple(ctx.targets))
    graph = form.model_validate(keyed).as_topology()
    assert graph == science.Topology.model_validate(proposal()[0])
    assert set(
        strict_provider_schema(form.model_json_schema())["$defs"]["PublicOutputs"][
            "required"
        ]
    ) == set(ctx.targets)
    other, _ = science.named_topology_forms(("heat", "mass"))
    assert set(other.model_json_schema()["$defs"]["PublicOutputs"]["required"]) == {
        "heat",
        "mass",
    }
    # A distinct stored quantity and its measured readout can both be intentional.
    keyed["additional_components"].append(
        {"name": "state_Gp", **keyed["targets"]["Gp"]}
    )
    keyed["targets"]["Gp"] = {
        "kind": "algebraic",
        "sources": ["state_Gp"],
        "role": "readout",
    }
    graph = form.model_validate(keyed).as_topology()
    science.check_topology(graph, ctx, brief)
    keyed["additional_components"].append({"name": "Gp", **keyed["targets"]["Gp"]})
    with pytest.raises(ValueError, match="Gp already has a declaration"):
        form.model_validate(keyed).as_topology()
    del keyed["targets"]["Gp"]
    with pytest.raises(ValueError):
        form.model_validate(keyed)


def test_fresh_20b_complete_corrections_keep_target_contract_and_resume(experiment):
    root = experiment.parent / "fresh"
    plan = pilot.prepare(
        root, experiment / "inputs.json", expected_protocol=pilot.FRESH_20B
    )
    assert plan["groups"] == ["fresh"] and plan["maximum_generation_requests"] == 25
    assert plan["settings"]["model_revision"] == pilot.MODELS["20b"]["revision"]
    keyed, laws = named_proposal()
    calls = []

    def transport(url, body, timeout):
        name = body["response_format"]["json_schema"]["name"]
        packet = json.loads(body["messages"][1]["content"])
        assert body["model"] == "openai/gpt-oss-20b" and "current_model" not in packet
        assert packet["required_target_keys"] == ["Gp", "I", "U"]
        calls.append((name, body["seed"]))
        if name == "NamedTargetTopology":
            if "rejection" not in packet:
                return reply(
                    {"components": [{"name": "state_Gp"}, {"name": "target_Gp"}]}
                )
            errors = packet["rejection"]
            assert errors["code"] == "PUBLIC_TARGET_KEYS"
            assert errors["details"]["missing"] == ["Gp", "I", "U"]
            assert errors["details"]["other_submitted_names"] == [
                "state_Gp",
                "target_Gp",
            ]
            return reply(keyed)
        if name == "Laws":
            assert packet["topology"] == keyed
            return reply({**laws, "latent_initials": []})
        assert name == "NamedTargetCorrection"
        if packet["correction_number"] == 2:
            bad = deepcopy(keyed)
            bad["targets"].pop("I")
            return reply({"topology": bad, "laws": laws})
        assert packet["rejection"]["details"]["missing"] == ["I"]
        return reply({"topology": keyed, "laws": laws})

    report = pilot.run(
        root, "http://127.0.0.1", transport=transport, token_transport=tokenizer
    )
    assert report["physical_requests"] == len(calls) == 25
    assert len({s for _, s in calls[:5]}) == 5
    assert [s["initial_sampling_seed"] for s in pilot._rows(root, plan)] == list(
        range(100, 105)
    )
    assert report["fresh_topology"] == {
        "public_targets": ["Gp", "I", "U"],
        "initial_attempts_recorded": 5,
        "initial_replies_with_all_target_declarations": 0,
        "initial_topology_passes": 0,
        "branches_sent_to_functions": 5,
        "eligible_branches": 5,
    }
    assert report["representatives"] == ["fresh-100"]
    assert (
        pilot.run(
            root, "http://127.0.0.1", transport=lambda *a: pytest.fail("duplicate call")
        )
        == report
    )


def test_fresh_20b_first_pass_and_actual_graph_errors(experiment):
    root = experiment.parent / "fresh"
    pilot.prepare(root, experiment / "inputs.json", expected_protocol=pilot.FRESH_20B)
    keyed, laws = named_proposal()
    _, data = pilot.load(root)
    bad = deepcopy(keyed)
    bad["additional_components"][0]["sources"] = ["X"]
    held = pilot._assess(data, "fresh", bad, "topology", named_targets=True)
    assert (
        held["status"] == "rejected"
        and "Missing required path" in held["diagnostic"]["message"]
    )

    def success(url, body, timeout):
        return reply(
            laws if body["response_format"]["json_schema"]["name"] == "Laws" else keyed
        )

    report = pilot.run(
        root, "http://127.0.0.1", transport=success, token_transport=tokenizer
    )
    assert report["physical_requests"] == 10
    assert report["fresh_topology"]["initial_replies_with_all_target_declarations"] == 5
    assert report["fresh_topology"]["initial_topology_passes"] == 5
    assert report["fresh_topology"]["eligible_branches"] == 5


def test_fresh_aces_pins_one_20b_gpu_and_resubmission_is_idempotent(
    experiment, monkeypatch
):
    root = experiment.parent / "fresh"
    pilot.prepare(root, experiment / "inputs.json", expected_protocol=pilot.FRESH_20B)
    with pytest.raises(ValueError, match="own generation slots"):
        pilot.prepare(
            root,
            experiment / "inputs.json",
            previous=experiment,
            expected_protocol=pilot.FRESH_20B,
        )
    with pytest.raises(ValueError, match="frozen artifact differs"):
        pilot.prepare(
            experiment, experiment / "inputs.json", expected_protocol=pilot.FRESH_20B
        )
    calls = []

    def assets(size):
        assert size == "20b"
        return experiment / "inputs.json", {
            "model_revision": pilot.MODELS[size]["revision"]
        }

    def submit(wave, argv):
        calls.append(argv)
        assert "--gres=gpu:h100:1" in argv and "--time=01:00:00" in argv
        assert "--job-name=phasec-t2-fresh-20b" in argv
        return "12345"

    monkeypatch.setattr(aces.aces, "_assets", assets)
    monkeypatch.setattr(aces.aces.harmony_vocab, "verify", lambda _: {"sha256": "test"})
    monkeypatch.setattr(aces, "_submit_job", submit)
    assert aces.submit(root)["job"] == aces.submit(root)["job"] == "12345"
    record = aces.verify(root, check_assets=True)
    assert len(calls) == 1 and record["maximum_generation_requests"] == 25
    assert (
        record["model"]
        == record["generation_settings"]["model"]
        == "openai/gpt-oss-20b"
    )
    assert record["model_revision"] == record["generation_settings"]["model_revision"]
    assert record["tensor_parallel_size"] == 1
