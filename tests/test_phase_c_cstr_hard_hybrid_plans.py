"""HDTwinGen and spec-informed SINDy on CSTR-hard: only the cell and its source move."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoformalism.rebuttal import phase_b_nine_case as nine
from autoformalism.rebuttal import phase_b_nine_case_d3 as d3_family
from autoformalism.rebuttal import phase_b_nine_case_spec_sindy as spec_sindy

REPO = Path(__file__).resolve().parents[1]
HARD = "phase_c_cstr_controlled_reactor_mechanism_canonical_named_hard_reset_v1"
RELEASE = "200deb42d60614915e2226070f405a98295c439626832e40191d2a0207b19665"
PAIRS = (
    (
        "phase_c_cstr_hard_hdtwingen_hosted_120b_v1.json",
        "phase_b_nine_case_hdtwingen_hosted_120b_v1.json",
        d3_family.load_plan,
    ),
    (
        "phase_c_cstr_hard_spec_sindy_hosted_120b_v1.json",
        "phase_b_nine_case_spec_sindy_hosted_120b_v1.json",
        spec_sindy.load_plan,
    ),
)


@pytest.mark.parametrize(("hard", "frozen", "load"), PAIRS)
def test_a_cstr_hard_plan_changes_only_its_cell_and_source(hard, frozen, load):
    plan = load(REPO / "configs" / hard).model_dump(mode="json")
    base = load(REPO / "configs" / frozen).model_dump(mode="json")
    changed = {key for key in plan if plan[key] != base[key]}
    # The nine-case plans already ran seeds 0, 1 and 2.
    assert changed == {"inputs_protocol", "release_summary_sha256", "cells", "purpose"}
    assert plan["inputs_protocol"] == nine.PHASE_C
    assert plan["release_summary_sha256"] == RELEASE
    assert [cell["benchmark_id"] for cell in plan["cells"]] == [HARD]
    assert plan["repetitions"] == [0, 1, 2]
    assert plan["status"] == base["status"] == "frozen_before_calls"
    assert frozen.removesuffix(".json") in plan["purpose"]


def _payload(name: str) -> dict:
    return json.loads((REPO / "configs" / name).read_text())


def test_a_phase_c_plan_must_name_its_release_and_a_baseline_cell():
    payload = _payload(PAIRS[0][0])
    d3_family.NineCaseD3Plan.model_validate(payload)
    with pytest.raises(ValidationError, match="names its release"):
        d3_family.NineCaseD3Plan.model_validate(
            {**payload, "release_summary_sha256": None}
        )
    other = {**payload["cells"][0], "benchmark_id": "phase_c_detention_coupled_v9"}
    with pytest.raises(ValidationError, match="not a Phase C baseline cell"):
        d3_family.NineCaseD3Plan.model_validate({**payload, "cells": [other]})
    easy = {**payload["cells"][0], "tier": "easy"}
    with pytest.raises(ValidationError, match="wrong tier"):
        spec_sindy.SpecSindyPlan.model_validate(
            {**_payload(PAIRS[1][0]), "cells": [easy]}
        )


def test_a_nine_case_plan_still_reads_only_the_staged_cases():
    payload = _payload(PAIRS[1][1])
    spec_sindy.SpecSindyPlan.model_validate(payload)
    with pytest.raises(ValidationError, match="names its release"):
        spec_sindy.SpecSindyPlan.model_validate(
            {**payload, "release_summary_sha256": RELEASE}
        )
    assert nine.scope_of(payload["inputs_protocol"]) == (
        "the nine Phase B cases of the main comparison"
    )
