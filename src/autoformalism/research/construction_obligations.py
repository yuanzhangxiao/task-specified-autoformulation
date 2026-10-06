"""Reviewed public interpretations for the eight-case topology-only study.

This catalog is authored policy. It does not extract science from prose at runtime,
load reference equations, or change the finalized benchmark prompts.
"""

from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search.public_graph_obligations import PublicGraphContract
from autoformalism.staged_topology import content_hash

CSTR = "phase_c_cstr_controlled_reactor_mechanism_canonical_named_easy_reset_v1"
ALIEN = (
    "phase_c_alien_device_unknown_device_mechanism_canonical_functional_easy_reset_v1"
)
COUPLED = "phase_c_detention_coupled_noise0_v1"
INDEPENDENT = "phase_c_detention_independent_noise0_v1"
DALLA = tuple(
    f"phase_c_dalla_man_{task}_canonical_named_{difficulty}_rates_v1"
    for task in ("t1", "t2")
    for difficulty in ("easy", "hard")
)


def reviewed_contract(case: str, brief: PublicScientificBrief) -> PublicGraphContract:
    """Fail closed on unknown cases or changed evidence; do not guess obligations."""
    rules = []
    if case == CSTR:
        rules.append(
            {
                "id": "temperature_balance_feedback",
                "kind": "target_feedback",
                "target": "T",
                "public_quote": (
                    "a reactor-temperature balance that distinguishes feed "
                    "transport, reaction heat generation, and heat exchange with "
                    "the jacket"
                ),
                "interpretation": (
                    "The temperature balance needs current thermal-state feedback. "
                    "Declare a feedback cycle through T when T is a state. "
                    "For an algebraic T readout, explicitly bind its thermal/energy "
                    "coordinates in feedback_bindings; each needs feedback and an "
                    "algebraic path to T. A feed-driven "
                    "accumulator with no such cycle cannot express this balance. "
                    "This is a reviewed operationalization, not an exact kinetic law."
                ),
            }
        )
        deferred = (
            (
                "controlled_balance remains partly unresolved: graph connectivity"
                " alone cannot distinguish feed transport, reaction heat "
                "generation and jacket exchange."
            ),
            (
                "No direct C or Tj dependence is mandatory: the public task "
                "permits omitting supplied auxiliaries or modeling alternative "
                "coordinates."
            ),
            (
                "Heat-flow signs, restoring behavior, reaction-temperature "
                "dependence, scaling and conservation require function/behavior "
                "inspection. No Arrhenius law is imposed."
            ),
        )
    elif case in (COUPLED, INDEPENDENT):
        rules.append(
            {
                "id": "storage_release_feedback",
                "kind": "target_feedback",
                "target": "h_down",
                "public_quote": (
                    (
                        "Represent threshold-dependent overflow and the "
                        "downstream discharge."
                    )
                    if case == COUPLED
                    else (
                        "Represent downstream accumulation, a threshold-dependent"
                        " outlet, nonnegative storage and its local water "
                        "balance."
                    )
                ),
                "interpretation": (
                    "Downstream storage-dependent discharge requires feedback through "
                    "h_down when it is a state. For an algebraic h_down readout, "
                    "declare its downstream storage coordinates in feedback_bindings; "
                    "each needs feedback and an algebraic path to h_down. An upstream "
                    "loop alone is insufficient. Coordinate meaning is your scientific "
                    "assignment, not inferred from names by the runtime."
                ),
            }
        )
        if case == INDEPENDENT:
            rules.append(
                {
                    "id": "disconnected_upstream_input",
                    "kind": "forbidden_path",
                    "source": "inflow_up",
                    "target": "h_down",
                    "public_quote": (
                        "There is no pipe, spillway, overflow route, or common "
                        "unmeasured inflow linking these two basins."
                    ),
                    "interpretation": (
                        "The disconnected upstream forcing must have no declared "
                        "causal path to downstream depth, including through "
                        "generated intermediates."
                    ),
                }
            )
        deferred = (
            (
                "Threshold behavior, physical crest versus warning level, area "
                "conversions and nonnegative water balance require "
                "function/behavior inspection."
            ),
            (
                "No fixed latent names, discharge formula, named shared process "
                "or obligatory direct crest covariate is imposed; equivalent "
                "coordinates/laws remain allowed."
            ),
            (
                "Coupled transfer cancellation and upstream storage depletion "
                "need scientific assignment and assembled-law checks, beyond "
                "existence of any feedback cycle."
            ),
        )
    elif case == ALIEN:
        deferred = (
            (
                "Existing input-memory-output paths and proposer-selected memory "
                "bindings remain enforced."
            ),
            (
                "Input-driven memory does not explicitly require fading memory. "
                "An accumulator is not rejected merely for lacking "
                "self-dependence."
            ),
            (
                "Initialization, function shape and response behavior remain "
                "unassessed at topology."
            ),
        )
    elif case in DALLA:
        deferred = (
            (
                "Existing meal paths, delayed insulin-memory paths where "
                "specified, target generation and T2-easy Uii composition remain "
                "enforced."
            ),
            (
                "No textbook equation skeleton, gastric compartment count or "
                "universal memory self-loop is required by the public task."
            ),
            (
                "Role-consistent signs, duplicated physical contributions, "
                "disposal/source identification and initialization require "
                "scientific/function inspection; a source set or explanation does"
                " not prove them."
            ),
        )
    else:
        raise ValueError(f"no reviewed public graph profile for {case}")
    contract = PublicGraphContract(
        policy="reviewed-public-graph-2",
        source_brief_sha256=content_hash(brief.model_dump(mode="json")),
        obligations=rules,
        deferred_scientific_checks=deferred,
    )
    contract.validate_public(brief)
    return contract
