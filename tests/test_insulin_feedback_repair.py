"""A recorded failure informs one fresh 120B repair from the fitted parent."""

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from autoformalism.benchmarks.audited_release import seal
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import insulin_boundary_revision as boundary
from autoformalism.research import revision_spelling
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from scripts import propose_meal_insulin_revision as cli
from tests.test_construction_initialization_graph_cli import reply, tokenizer
from tests.test_insulin_boundary_feedback import point
from tests.test_insulin_boundary_revision import with_rhs
from tests.test_meal_insulin_proposer import forbidden, patch, source


def repair(expr="a*Gp - q*I*τ_decay"):
    return patch(
        hypothesis="Check τ_decay in the net rate.",
        insulin_rate_at_zero="At I=0, dI/dt=a*Gp.",
        expected_ingestion_window_change="The added term vanishes at zero.",
        edit={
            "component": "I",
            "kind": "dynamic",
            "sources": ["Gp", "q", "I"],
            "expression": expr,
        },
        new_parameters=[{"name": "tau_decay", "role": "rate"}],
    )


def test_whole_symbols_in_dynamic_and_initializer_scopes():
    parent = with_rhs("-q")
    raw = repair("a*Gp - τ_decay*I + τ_decay_extra")
    raw["new_states"] = [
        {
            "name": "θ",
            "description": "τ_decay",
            "rhs": "τ_decay*q",
            "sources": ["q"],
            "initial": {
                "mode": "map",
                "expression": "β*β0 + τ_decay",
                "parameters": [{"name": "β", "role": "coefficient"}],
                "public_initials": [{"alias": "beta0", "channel": "aux"}],
                "state_initials": [],
            },
        }
    ]
    raw["new_processes"] = [
        {"name": "pulse", "description": "θ", "expression": "θ", "sources": ["θ"]}
    ]
    original = deepcopy(raw)
    canonical, changes = revision_spelling.normalize(parent, raw, cli.BoundaryRevision)
    assert raw == original and changes
    assert canonical["hypothesis"] == raw["hypothesis"]
    assert canonical["new_states"][0]["description"] == "τ_decay"
    assert canonical["edit"]["expression"] == "a*Gp - tau_decay*I + τ_decay_extra"
    state = canonical["new_states"][0]
    assert state["name"] == "theta" and state["rhs"] == "tau_decay*q"
    assert state["initial"]["expression"] == "beta*beta0 + τ_decay"
    assert state["initial"]["parameters"][0]["name"] == "beta"
    assert canonical["new_processes"][0]["sources"] == ["theta"]
    assert canonical["new_processes"][0]["expression"] == "theta"
    assert (
        canonical
        == revision_spelling.normalize(parent, canonical, cli.BoundaryRevision)[0]
    )


def test_spelling_does_not_guess_or_weaken_the_schema():
    parent = with_rhs("-q")
    raw = repair()
    raw["new_parameters"].append({"name": "τ_decay", "role": "coefficient"})
    with pytest.raises(cli.RevisionValidationError, match="AMBIGUOUS_IDENTIFIER"):
        revision_spelling.normalize(parent, raw, cli.BoundaryRevision)
    raw = repair()
    raw["edit"]["sources"] = ["τ_missing"]
    with pytest.raises(cli.RevisionValidationError, match="UNKNOWN_IDENTIFIER"):
        revision_spelling.normalize(parent, raw, cli.BoundaryRevision)
    raw["unexpected_field"] = True
    with pytest.raises(ValidationError):
        revision_spelling.normalize(parent, raw, cli.BoundaryRevision)


def test_role_only_change_reuses_the_witness_only_if_the_old_values_are_legal():
    parent = with_rhs("-q")

    def child(role, rhs="Gp + rate1*I"):
        data = with_rhs(rhs).model_dump(mode="json")
        data["base_candidate"]["parameters"].append(
            {"name": "rate1", "scope": "global", "role": role}
        )
        data["parameter_guesses"]["rate1"] = 1.0
        return PrefixFitProblem.model_validate(data)

    old = child("nonnegative_coefficient")
    previous = cli.family_identity(parent, old)["sha256"]
    assert previous
    witness_values = {"rate1": 2.0}
    history = {
        "previous_family_sha256": previous,
        "previous_attempt": {
            "fixed_parameter_replay": {"new_parameters": witness_values},
            "patch": {
                "new_parameters": [{"name": "rate1", "role": "nonnegative_coefficient"}]
            },
        },
    }
    assert boundary.previous_counterexample_applies(
        parent, child("time_constant"), history
    )
    assert not boundary.previous_counterexample_applies(
        parent, child("time_constant", "Gp - rate1*I"), history
    )
    witness_values["rate1"] = 0
    assert not boundary.previous_counterexample_applies(
        parent, child("time_constant"), history
    )


def arguments():
    args = source()
    parent = with_rhs("-q")
    w = {**point(), "constrained_fit_I_pmol_L": 0, "observed_I_pmol_L": 1}
    args.update(
        fitted_problem=parent.model_dump(mode="json"),
        public_brief=args["public_brief"].model_dump(mode="json"),
        boundary_feedback={
            "depletion": w,
            "ingestion_windows": [],
            "zero_input_extension": {"predicted_I_pmol_L": -1},
        },
        repair_history={
            "source_plan_sha256": "1" * 64,
            "previous_family_sha256": "2" * 64,
            "fitted_problem_sha256": cli.content_sha256(parent),
            "fitted_feedback": ["The fitted boundary points outward."],
            "previous_attempt": {
                "patch": {"new_states": []},
                "deterministic_feedback": ["A clock rate ignores the input."],
                "fixed_parameter_replay": {"trajectory": "train_003"},
            },
        },
    )
    return args


def test_new_request_replays_once_and_assesses_the_repaired_equations(
    tmp_path, monkeypatch
):
    root = tmp_path / "repair"
    data = {"protocol": boundary.REPAIR_INTAKE_PROTOCOL, "arguments": arguments()}
    monkeypatch.setattr(boundary, "REPAIR_INTAKE_SHA256", cli.content_sha256(data))
    seal(tmp_path / "source.json", data)
    plan = cli.prepare_source(root, tmp_path / "source.json")
    assert plan["protocol"] == boundary.REPAIR_PROTOCOL
    assert plan["model_size"] == "120b" and plan["settings"]["maximum_requests"] == 1
    with pytest.raises(ValueError, match="120B"):
        cli.prepare_source(tmp_path / "20b", tmp_path / "source.json", "20b")
    inputs = sealed_read(root / "inputs.json")
    request = json.loads(inputs["user"])
    assert (
        request["previous_attempt"]
        == data["arguments"]["repair_history"]["previous_attempt"]
    )
    assert (
        request["training_evidence"]["deterministic_feedback"]
        == data["arguments"]["repair_history"]["fitted_feedback"]
    )
    assert "current_model is the fitted" in inputs["system"]
    assert "heldout sentinel" not in inputs["user"]
    first = cli.run(
        root,
        "http://127.0.0.1",
        transport=lambda *args: reply(repair()),
        token_transport=tokenizer,
    )
    assert first["status"] == "eligible_for_refit", first
    assert first["proposal"]["edit"]["expression"].endswith("tau_decay")
    assert first["audit"]["identifier_spellings"] == [
        {"field": ["edit", "expression"], "from": "τ_decay", "to": "tau_decay"}
    ]
    assert (
        first["insulin_boundary_check"]["boundary_sign"]
        == "nonnegative_under_stated_assumptions"
    )
    assert (
        first["refit_handoff"]["fitted_parent_reference"] == inputs["result_identity"]
    )
    call = json.loads(next((root / "calls").glob("*.json")).read_text())
    assert cli.visible_response(call)["edit"]["expression"].endswith("τ_decay")
    assert call["request"]["body"]["messages"][1]["content"] == inputs["user"]
    (root / "result.json").unlink()
    assert (
        cli.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )
        == first
    )
    assert cli.inspect(root)["physical_requests_started"] == 1
    assert "A clock rate ignores the input." in (root / "PROMPT.html").read_text()
    family = cli.family_identity(inputs["problem"], first["problem"])["sha256"]
    inputs["repair_history"]["previous_family_sha256"] = family
    with pytest.raises(
        cli.RevisionValidationError, match="PREVIOUS_REPAIR_REPEATED"
    ) as info:
        cli.assess(inputs, repair())
    assert (
        info.value.details["identifier_spellings"]
        == first["audit"]["identifier_spellings"]
    )
    with pytest.raises(cli.RevisionValidationError, match="UNKNOWN_FORMULA_SYMBOL"):
        cli.assess(inputs, repair("a*Gp + τ_typo"))
    with pytest.raises(cli.RevisionValidationError):
        cli.assess(inputs, repair("__import__('os')"))
