"""Initializer edits survive legacy handle validation without changing controls."""

import pytest

from autoformalism.search import construction_actionable_repair as previous
from autoformalism.search import construction_initialized_repair as initialized
from tests.test_construction_actionable_repair import (
    boundary,
    normalize,
    public,
    raw_reply,
    select,
    with_outlet,
)
from tests.test_construction_evidence import apply
from tests.test_construction_ledger import equation, variable


def retained_outlet_and_new_initializer():
    """A real initializer-only edit may restate an unchanged boundary assignment."""
    draft = apply(
        with_outlet(), variables=[variable("z")], equations=[equation("z", "u")]
    )
    selection = select(draft)
    patch, _ = normalize(draft, raw_reply(external_exchanges=[selection]))
    draft = previous.apply_patch(public(), draft, patch)
    raw = initialized.response_model("repair")(stage_complete=False).model_dump(
        mode="json"
    )
    raw.update(
        external_exchanges=[selection],
        initialization={
            "records": [{"state": "z", "rule": {"mode": "shared_parameter"}}]
        },
    )
    return draft, raw


def test_initializer_only_edit_can_restate_an_existing_handle_without_finishing():
    draft, raw = retained_outlet_and_new_initializer()
    value, log = initialized.normalize_reply(
        public(), raw, draft=draft, boundary_contract=boundary()
    )
    patch = initialized.DraftPatch.model_validate(value)
    assert not patch.stage_complete
    assert not raw["stage_complete"]
    assert patch.initialization.records[0].state == "z"
    assert patch.initialization.records[0].rule.mode == "shared_parameter"
    assert patch.external_exchanges[0].shared_process == "outlet"
    assert log[-1]["code"] == "external_contribution_handle_resolved"
    assert initialized.apply_patch(public(), draft, patch) == draft


def test_null_initializer_retains_legacy_noop_rejection():
    draft, raw = retained_outlet_and_new_initializer()
    raw["initialization"] = None
    with pytest.raises(ValueError, match="no draft change"):
        initialized.normalize_reply(
            public(), raw, draft=draft, boundary_contract=boundary()
        )


def test_initializer_edit_does_not_bypass_stale_handle_validation():
    draft, raw = retained_outlet_and_new_initializer()
    raw["external_exchanges"][0]["contribution_handle"] = "contribution-" + "0" * 64
    with pytest.raises(ValueError, match="absent or stale"):
        initialized.normalize_reply(
            public(), raw, draft=draft, boundary_contract=boundary()
        )


def test_initializer_edit_does_not_bypass_changed_contribution_validation():
    draft, raw = retained_outlet_and_new_initializer()
    changed = draft.processes[0].model_dump(mode="json")
    changed["scientific_meaning"] = "A different intended physical contribution."
    raw["processes"] = [changed]
    with pytest.raises(ValueError, match="selected contribution changed"):
        initialized.normalize_reply(
            public(), raw, draft=draft, boundary_contract=boundary()
        )
