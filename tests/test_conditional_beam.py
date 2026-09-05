"""Topology-conditioned functional beam tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoformalism.construction import finalize_topology_draft
from autoformalism.expressions import ValidationContext
from autoformalism.schemas import FunctionalDraft, TopologyDraft
from autoformalism.search.conditional_beam import ConditionalConstructionBeam
from autoformalism.staging import topology_commitment_sha256


def _context() -> ValidationContext:
    return ValidationContext(targets=("target",), external_inputs=("input_u",))


def _delayed_topology() -> TopologyDraft:
    return TopologyDraft.model_validate(
        {
            "states": [{"name": "x"}, {"name": "z"}],
            "interactions": [
                {
                    "interaction_id": "z_drive",
                    "target": "z",
                    "sources": ["input_u"],
                },
                {
                    "interaction_id": "z_decay",
                    "target": "z",
                    "sources": ["z"],
                    "polarity": "subtractive",
                },
                {
                    "interaction_id": "x_drive",
                    "target": "x",
                    "sources": ["z"],
                },
            ],
            "target_mappings": [{"channel": "target", "source": "x"}],
        }
    )


def _direct_topology() -> TopologyDraft:
    return TopologyDraft.model_validate(
        {
            "states": [{"name": "x"}],
            "interactions": [
                {
                    "interaction_id": "x_drive",
                    "target": "x",
                    "sources": ["input_u"],
                }
            ],
            "target_mappings": [{"channel": "target", "source": "x"}],
        }
    )


def _delayed_functions(
    topology_draft: TopologyDraft,
    *,
    incompatible: bool = False,
) -> FunctionalDraft:
    topology = finalize_topology_draft(topology_draft, _context())
    return FunctionalDraft.model_validate(
        {
            "topology_commitment_sha256": topology_commitment_sha256(topology),
            "interaction_functions": [
                {"interaction_id": "z_drive", "expression": "input_u"},
                {
                    "interaction_id": "z_decay",
                    "expression": "k_z * z",
                    "parameters": [{"name": "k_z", "role": "rate"}],
                },
                {
                    "interaction_id": "x_drive",
                    "expression": (
                        "gain * input_u" if incompatible else "gain * z"
                    ),
                    "parameters": [{"name": "gain", "role": "scale"}],
                },
            ],
            "latent_initials": [
                {"state": "z", "initial": {"fixed_value": 0.0}}
            ],
        }
    )


def _direct_functions(topology_draft: TopologyDraft) -> FunctionalDraft:
    topology = finalize_topology_draft(topology_draft, _context())
    return FunctionalDraft.model_validate(
        {
            "topology_commitment_sha256": topology_commitment_sha256(topology),
            "interaction_functions": [
                {"interaction_id": "x_drive", "expression": "input_u"}
            ],
        }
    )


def test_conditional_beam_retains_topology_after_incompatible_function(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "conditional-beam.json"
    beam = ConditionalConstructionBeam(
        checkpoint_path=checkpoint,
        maximum_functions_per_topology=2,
    )
    delayed, delayed_new = beam.register_topology(
        _delayed_topology(), _context()
    )
    direct, direct_new = beam.register_topology(_direct_topology(), _context())
    scorer_calls: list[str] = []

    with pytest.raises(ValueError, match="before one functional attempt"):
        beam.select(beam_size=2)

    rejected, rejected_new = beam.evaluate_function_child(
        topology_sha256=delayed.topology_sha256,
        draft=_delayed_functions(_delayed_topology(), incompatible=True),
        context=_context(),
        score_expansion=lambda expansion: scorer_calls.append(
            expansion.candidate.candidate_id
        ),
    )
    direct_child, _ = beam.evaluate_function_child(
        topology_sha256=direct.topology_sha256,
        draft=_direct_functions(_direct_topology()),
        context=_context(),
        score_expansion=lambda _expansion: 0.2,
    )
    with pytest.raises(ValueError, match="while function allowance remains"):
        beam.select(beam_size=2)
    delayed_child, _ = beam.evaluate_function_child(
        topology_sha256=delayed.topology_sha256,
        draft=_delayed_functions(_delayed_topology()),
        context=_context(),
        score_expansion=lambda _expansion: 0.1,
    )
    selected = beam.select(beam_size=2)

    assert delayed_new is True
    assert direct_new is True
    assert rejected_new is True
    assert rejected.status == "rejected_incompatible"
    assert scorer_calls == []
    assert delayed_child.status == "scored"
    assert direct_child.status == "scored"
    assert {item.topology_sha256 for item in selected} == {
        delayed.topology_sha256,
        direct.topology_sha256,
    }
    assert beam.remaining_function_allowance(delayed.topology_sha256) == 0
    assert len(beam.state.topology_branches) == 2
    assert checkpoint.exists()


def test_conditional_beam_resume_does_not_rescore_duplicate_child(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "conditional-beam.json"
    first = ConditionalConstructionBeam(
        checkpoint_path=checkpoint,
        maximum_functions_per_topology=2,
    )
    branch, _ = first.register_topology(_direct_topology(), _context())
    draft = _direct_functions(_direct_topology())
    original, original_new = first.evaluate_function_child(
        topology_sha256=branch.topology_sha256,
        draft=draft,
        context=_context(),
        score_expansion=lambda _expansion: 0.25,
    )

    resumed = ConditionalConstructionBeam(
        checkpoint_path=checkpoint,
        maximum_functions_per_topology=2,
    )

    def must_not_score(_expansion):
        raise AssertionError("duplicate child was rescored")

    duplicate, duplicate_new = resumed.evaluate_function_child(
        topology_sha256=branch.topology_sha256,
        draft=draft,
        context=_context(),
        score_expansion=must_not_score,
    )

    assert original_new is True
    assert duplicate_new is False
    assert duplicate == original
    assert resumed.state.revision == first.state.revision
