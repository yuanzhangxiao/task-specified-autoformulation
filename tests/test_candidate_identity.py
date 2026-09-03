"""Tests for the versioned candidate-identity projections."""

from __future__ import annotations

from autoformalism.schemas import CandidateModel
from autoformalism.search.identity import candidate_identity


def _candidate(
    *,
    latent: str = "X",
    rate: str = "k_x",
    latent_rhs: str | None = None,
    upper: float = 5.0,
    observation_channel: str = "target",
) -> CandidateModel:
    rhs = latent_rhs or f"{rate} * (input_signal - {latent})"
    return CandidateModel.model_validate(
        {
            "candidate_id": f"candidate_{latent}",
            "parent_candidate_id": None,
            "change_summary": "Identity test candidate.",
            "states": [
                {"name": "observed", "kind": "observed"},
                {"name": latent, "kind": "latent"},
            ],
            "processes": [
                {"name": "generated", "expression": f"{latent} * observed"}
            ],
            "state_equations": [
                {"state": "observed", "rhs": "-generated"},
                {"state": latent, "rhs": rhs},
            ],
            "observation_mappings": [
                {"channel": observation_channel, "expression": "observed"}
            ],
            "parameters": [
                {
                    "name": rate,
                    "scope": "global",
                    "bounds": {"lower": 0.0, "upper": upper},
                    "initialization_range": {"lower": 0.1, "upper": 1.0},
                }
            ],
            "initial_conditions": [
                {"state": "observed", "scope": "global", "fixed_value": 1.0},
                {"state": latent, "scope": "global", "fixed_value": 0.0},
            ],
        }
    )


def test_identity_is_invariant_to_proposer_owned_symbol_renaming() -> None:
    first = candidate_identity(_candidate())
    renamed = candidate_identity(_candidate(latent="delay", rate="relaxation"))

    assert renamed == first


def test_function_change_preserves_topology_but_changes_functional_identity() -> None:
    linear = candidate_identity(_candidate())
    saturating = candidate_identity(
        _candidate(latent_rhs="k_x * (tanh(input_signal) - X)")
    )

    assert saturating.topology_sha256 == linear.topology_sha256
    assert saturating.functional_sha256 != linear.functional_sha256
    assert saturating.executable_sha256 != linear.executable_sha256


def test_bound_change_only_changes_executable_identity() -> None:
    first = candidate_identity(_candidate(upper=5.0))
    wider = candidate_identity(_candidate(upper=10.0))

    assert wider.topology_sha256 == first.topology_sha256
    assert wider.functional_sha256 == first.functional_sha256
    assert wider.executable_sha256 != first.executable_sha256


def test_public_observation_channel_anchors_identity() -> None:
    first = candidate_identity(_candidate(observation_channel="target"))
    other = candidate_identity(_candidate(observation_channel="other_target"))

    assert other.topology_sha256 != first.topology_sha256
    assert other.functional_sha256 != first.functional_sha256
    assert other.executable_sha256 != first.executable_sha256


def test_commutative_reordering_is_not_a_new_functional_candidate() -> None:
    first = _candidate()
    payload = first.model_dump(mode="json")
    payload["processes"][0]["expression"] = "observed * X"
    reordered = CandidateModel.model_validate(payload)

    assert candidate_identity(reordered) == candidate_identity(first)
