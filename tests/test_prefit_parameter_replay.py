"""Exact inherited declarations, historical replay, isolation and fit resume."""

import copy

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import prefit_parameter_replay as replay
from autoformalism.schemas.staged_functions import InteractionFunctionReply
from autoformalism.search import numerical_sibling as review
from autoformalism.search.parameter_reuse import inherit_parameters
from scripts.smoke_prefit_parameter_replay import saved_history


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    return saved_history(tmp_path_factory.mktemp("parameter-replay"))


def test_opt_in_recovers_canonical_parameter_identity_and_role(history):
    _, plan, residual, slot, raw = history
    original = copy.deepcopy((plan, raw))
    with pytest.raises(ValueError, match="source mismatch"):
        review.apply_revision(
            plan["bundle"],
            plan["bindings"],
            residual["packet"],
            raw,
            policy=review.ROUTED_POLICY,
        )
    result = review.apply_revision(
        plan["bundle"],
        plan["bindings"],
        residual["packet"],
        raw,
        policy=review.ROUTED_POLICY,
        inherit_existing_parameters=True,
    )
    assert result["outcome"] == "committed"
    record = result["provenance"]["parameter_inheritance"][0]
    old = slot["canonical_function"]["parameters"][0]
    assert record["canonical_name"] == old["name"]
    assert record["role"] == old["role"] and record["declaration_inherited"]
    assert record["local_name"] == slot["accepted_reply"]["parameters"][0]["name"]
    assert (
        old in result["final"]["selected_function"]["canonical_function"]["parameters"]
    )
    assert original == (plan, raw)


@pytest.mark.parametrize("kind", ["local", "canonical"])
def test_both_exact_parent_names_inherit_without_fuzzy_matching(history, kind):
    _, _, _, slot, _ = history
    source = slot["accepted_reply"] if kind == "local" else slot["canonical_function"]
    reply = InteractionFunctionReply(expression=source["expression"], parameters=())
    normalized, records = inherit_parameters(slot, reply)
    assert (
        normalized.parameters[0].name == slot["accepted_reply"]["parameters"][0]["name"]
    )
    assert records[0]["declaration_inherited"]
    unknown = reply.model_copy(update={"expression": reply.expression + "+unknown*v01"})
    normalized, _ = inherit_parameters(slot, unknown)
    assert "unknown" not in {p.name for p in normalized.parameters}


@pytest.mark.parametrize("kind", ["unknown", "foreign", "role", "source", "linear"])
def test_inheritance_does_not_waive_other_contracts(history, kind):
    _, plan, residual, slot, original = history
    raw = copy.deepcopy(original)
    revision = raw["revision"]
    if kind == "role":
        revision["parameters"].append(
            {
                "name": slot["canonical_function"]["parameters"][0]["name"],
                "role": "scale",
            }
        )
    elif kind == "foreign":
        other = next(
            s
            for s in plan["bundle"]["slots"]
            if s != slot and s["canonical_function"]["parameters"]
        )
        revision["expression"] += (
            "+" + other["canonical_function"]["parameters"][0]["name"]
        )
    elif kind == "linear":
        revision["expression"] = "h*v01"
    else:
        revision["expression"] += "+" + ("unknown" if kind == "unknown" else "t")
    with pytest.raises(ValueError):
        review.apply_revision(
            plan["bundle"],
            plan["bindings"],
            residual["packet"],
            raw,
            policy=review.ROUTED_POLICY,
            inherit_existing_parameters=True,
        )


def test_replay_fit_resumes_and_preserves_history(tmp_path, monkeypatch):
    source, _, _, slot, _ = saved_history(tmp_path)
    historical = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    root = tmp_path / "replay"
    result = replay.prepare(source, root)
    assert result["status"] == "ready_for_fit" and result["live_llm_calls"] == 0
    assert replay.prepare(source, root) == replay.report(root) == result
    with pytest.raises(ValueError, match="frozen artifact differs"):
        replay.prepare(source, tmp_path / "duplicate")
    with pytest.raises(ValueError, match="frozen artifact differs"):
        replay.prepare(source, root, attempt=1)
    # A real frozen backend call below also verifies the preserved warm start.
    fitted = replay.fit(root)
    assert fitted["status"] == "complete"
    assert fitted["child"]["result"]["status"] == "complete"
    name = slot["canonical_function"]["parameters"][0]["name"]
    assert name in fitted["child"]["seed"]["retained_parameters"]
    assert fitted["child"]["seed"]["retained_initializer_parameters"]
    monkeypatch.setattr(public, "_run_backend", lambda *a: pytest.fail("fit repeated"))
    assert replay.fit(root) == replay.report(root) == fitted
    assert historical == {p: p.read_bytes() for p in historical}


def test_mutated_saved_reply_fails_before_fitting(tmp_path):
    source, _, _, _, _ = saved_history(tmp_path)
    root = tmp_path / "replay"
    replay.prepare(source, root)
    path = source / "results/state.json"
    path.write_text(path.read_text().replace("linear component", "changed component"))
    with pytest.raises(ValueError, match="digest"):
        replay.fit(root)
    assert not (root / "child_fit").exists()


def test_unrelated_error_still_blocks_child_allocation(tmp_path, monkeypatch):
    source, _, _, _, _ = saved_history(tmp_path, extra_expression="+unknown_symbol")
    root = tmp_path / "replay"
    result = replay.prepare(source, root)
    assert result["status"] == "rejected" and "source mismatch" in result["error"]
    monkeypatch.setattr(public, "_run_backend", lambda *a: pytest.fail("invalid child"))
    assert replay.fit(root) == result
    assert not (root / "child_fit").exists()


def test_inherited_aliases_do_not_turn_unchanged_rhs_into_a_child(history):
    _, plan, residual, slot, original = history
    raw = copy.deepcopy(original)
    raw["revision"].update(
        expression=slot["canonical_function"]["expression"], parameters=[]
    )
    result = review.apply_revision(
        plan["bundle"],
        plan["bindings"],
        residual["packet"],
        raw,
        policy=review.ROUTED_POLICY,
        inherit_existing_parameters=True,
    )
    assert result["outcome"] == "unchanged_canonical_function"
    assert result["final"] is None
