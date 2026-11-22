"""Test scientific intake boundaries and replay the pinned real training cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.research import preproposal_cases as cases
from autoformalism.search.preproposal_screen import assess_options

ROOT = Path(__file__).resolve().parents[1]


def test_joint_inventory_discards_source_lists_and_scientific_prose():
    value = {
        "targets": {
            "I": {"kind": "dynamic", "sources": ["held_back"], "role": "not public"}
        },
        "additional_components": [
            {"name": "A", "kind": "algebraic", "sources": ["another_leak"]}
        ],
    }
    assert cases._joint_inventory(value) == [
        {"name": "A", "kind": "algebraic"},
        {"name": "I", "kind": "differential"},
    ]
    value["additional_components"][0]["name"] = "I"
    with pytest.raises(ValueError, match="Duplicate"):
        cases._joint_inventory(value)


@pytest.mark.parametrize(
    "stage,accepted,complete",
    [("equations", True, True), ("variables", False, True), ("variables", True, False)],
)
def test_ineligible_event_is_not_an_inventory(stage, accepted, complete):
    with pytest.raises(ValueError, match="variable-stage exit"):
        cases._variable_exit(
            {"stage": stage, "accepted": accepted, "stage_complete": complete}
        )


@pytest.fixture(scope="module")
def archives():
    if (
        not (ROOT / cases.FRESH_RECEIPT).exists()
        or not (ROOT / cases.STAGED_DIR).exists()
    ):
        pytest.skip("pinned experiment archives not installed in this checkout")
    return cases.load_inputs(ROOT)


def test_training_catalogue_and_real_preproposal_results(archives):
    fresh, basin = archives[cases.T2_CASE], archives[cases.BASIN_CASE]
    assert fresh["provenance"]["inventory_origin"] == "projected_from_joint_declaration"
    assert basin["provenance"]["inventory_origin"] == "accepted_variable_stage_exit"
    for row in (fresh, basin):
        assert row["evidence"]["split"] == "train"
        assert set(row["histories"]).isdisjoint(v["name"] for v in row["inventory"])
        assert all(set(v) == {"name", "kind"} for v in row["inventory"])
    sf, sb = map(cases.screen_case, (fresh, basin))
    assert (sf.subsets_examined, sf.pair_count) == (4096, 120)
    assert sf.subset_counts == {"screened": 3072, "contradicted": 1024, "unresolved": 0}
    assert assess_options(sf, ()).conflicting_pairs == 91
    assert {x.options for x in sf.candidates} == {("history:EGP",), ("history:Gt",)}
    assert (sb.pair_count, assess_options(sb, ()).conflicting_pairs) == (15, 0)
    assert [c.options for c in sb.candidates] == [()]
    assert set(sb.required_options) == {
        "initial:h_down",
        "history:inflow_up",
        "history:inflow_down",
    }
    strict = cases.load_strict_t2(ROOT)
    assert strict["provenance"]["inventory_origin"] == "accepted_variable_stage_exit"
    assert strict["inventory"] != fresh["inventory"]
    assert fresh["evidence"]["trajectories"] == strict["evidence"]["trajectories"]
    assert cases.screen_case(strict).candidates == sf.candidates
    projected = cases.selection_case(fresh)
    assert "trajectories" not in projected
    assert projected["evidence_sha256"] == sf.evidence_sha256
    assert "endogenous glucose production" in projected["public_task"]
    assert "supplied auxiliary" in projected["evidence_summary"]
    assert "sources" not in json.dumps(projected["inventory"])


def test_paired_insulin_arm_changes_only_training_gap(archives):
    exact = archives[cases.T2_CASE]
    relaxed = archives[cases.T2_GAP_CASE]
    assert relaxed == {**exact, "case_id": cases.T2_GAP_CASE, "minimum_target_gap": 1.0}
    assert exact["minimum_target_gap"] == 0.0
    menu = cases.selection_case(relaxed)
    assert menu["evidence_sha256"] == exact["evidence"]["sha256"]
    assert "experimental sensitivity" in menu["evidence_summary"]
    assert "gap of 1 " in menu["evidence_summary"]
    screen = cases.screen_case(relaxed)
    assert screen.minimum_target_gap == 1.0
    assert screen.subset_counts == {
        "screened": 3312,
        "contradicted": 784,
        "unresolved": 0,
    }
    additional = ("history:insulin_pmol_per_kg_min", "history:meal_rate_g_per_min")
    assert {c.options for c in screen.candidates} == {
        ("history:Gt",),
        ("history:EGP",),
        *((*additional, f"initial:{n}") for n in ("EGP", "Gp", "Gt", "U")),
    }


def test_declared_auxiliary_loses_its_future_history(archives):
    fresh = archives[cases.T2_CASE]
    row = cases._case(
        "alternative",
        fresh["target"],
        [*fresh["inventory"], {"name": "Gt", "kind": "algebraic"}],
        fresh["public_brief"],
        fresh["evidence"],
        set(fresh["histories"]),
        set(fresh["initials"]) | {"I"},
        {},
    )
    assert "Gt" not in row["histories"]
    assert "Gt" in row["initials"]
    assert [c.options for c in cases.screen_case(row).candidates] == [("history:EGP",)]


def test_bundle_verifier_rejects_resealed_unapproved_content(archives, tmp_path):
    rows = list(archives.values())
    payload = {"protocol": cases.PROTOCOL, "cases": rows}
    path = tmp_path / "frozen.json"
    sealed_write(path, payload)
    assert cases.load_frozen_cases(path) == archives
    path = tmp_path / "new.json"
    altered = {**rows[0], "previous_topology": {"sources": ["held_back"]}}
    sealed_write(path, {**payload, "cases": [altered, *rows[1:]]})
    with pytest.raises(ValueError, match="projection differs"):
        cases.load_frozen_cases(path)


def test_live_variable_exit_supports_typed_inputs_and_explicit_permissions():
    from autoformalism.schemas.staged_topology import PublicScientificBrief
    from tests.test_topology_repair_generator import evidence, row

    event = {
        "stage": "variables",
        "accepted": True,
        "stage_complete": True,
        "after": {
            "variables": [
                {"name": n, "definition": "differential"} for n in ("level", "upstream")
            ]
        },
    }
    public = PublicScientificBrief.model_validate(
        {
            "scientific_context": "Model two reservoirs.",
            "public_variables": [
                {"name": n, "data_role": role}
                for n, role in (
                    ("level", "target"),
                    ("upstream", "target"),
                    ("rain", "external_input"),
                    ("spill", "external_input"),
                )
            ],
            "requirements": [],
        }
    )
    packet = evidence(
        [
            row("a", (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)),
            row("b", (0, 1, 2), (0, 0, 0), (1, 1, 1), (1, 1, 1)),
        ]
    )
    kw = {
        "case_id": "live",
        "target": "level",
        "event": event,
        "public_brief": public,
        "evidence": packet,
        "histories": ("rain",),
        "initials": ("level",),
    }
    case = cases.from_variable_exit(**kw)
    assert case["histories"] == ["rain"] and case["initials"] == []
    assert cases.selection_case(case)["candidates"][0]["sources"] == ["history:rain"]
    assert (
        cases.from_variable_exit(**kw, minimum_target_gap=1.0)["minimum_target_gap"]
        == 1.0
    )
    assert cases.from_variable_exit(**kw, required_histories=["rain"])[
        "required_histories"
    ] == ["rain"]
    with pytest.raises(ValueError, match="Required histories"):
        cases.from_variable_exit(**kw, required_histories=["spill"])
    with pytest.raises(ValueError, match="time zero"):
        cases.from_variable_exit(**(kw | {"initials": ("upstream",)}))
    with pytest.raises(ValueError, match="variable-only"):
        cases.from_variable_exit(
            **(
                kw
                | {
                    "event": {
                        **event,
                        "after": {**event["after"], "equations": ["old equation"]},
                    }
                }
            )
        )
