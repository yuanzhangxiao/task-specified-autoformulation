"""Readable, train-only presentation of an isolated initializer decision.

The caller retains the structured choice packet and sealed evidence externally.
This renderer neither edits that packet nor selects or evaluates an initializer.
It verifies the source before adding all permitted initial measurements, and
renders each conflict once instead of repeating compatibility-question payloads.
"""

from __future__ import annotations

import json
from copy import deepcopy
from fractions import Fraction

from autoformalism.research import construction_training as training
from autoformalism.research import initialization_equality_feedback as equality
from autoformalism.research import initialization_evidence_feedback as numeric
from autoformalism.research import initializer_choice_prompt as choice
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    _pair,
    _value,
)

POLICY = "readable-initializer-choice-1"
# This exact runtime assumption belongs to propagation, not pointwise readouts.
# Its original record remains in the archived structured choice packet.
_PROPAGATION_ASSUMPTION = (
    "Shared deterministic functions and parameters have unique solutions "
    "on the common existence interval."
)
_FIELDS = {
    "task",
    "state",
    "instruction",
    "model_data_conflicts",
    "compatibility_questions",
    "initialization",
    "initializer_meanings",
    "syntax_example",
    "previous_feedback",
    "model_view",
    "public_brief",
    "projection_scope",
}


def _text(value: object) -> str:
    """Readable lossless scalars and compact structured metadata, never code."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _cell(value: object) -> str:
    """Keep arbitrary archived labels within one Markdown table cell."""
    return _text(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _rule(state: str, rule: dict) -> str:
    """Describe a rule without pretending that an unfitted map is executable."""
    mode, arguments = rule["mode"], rule["arguments"]
    if mode == "observed_initial" and not arguments:
        return f"{state}(t0) = recorded {state}(t0) [observed_initial]"
    if mode == "shared_parameter" and not arguments:
        return (
            f"{state}(t0) = one shared unknown constant [shared_parameter]; "
            "numeric value unavailable (not selected or fitted)"
        )
    if mode == "common_map" and arguments and len(arguments) == len(set(arguments)):
        names = ", ".join(f"{name}(t0)" for name in arguments)
        return (
            f"{state}(t0) = h_{state}({names}) [common_map]; one shared map, "
            "formula and resulting initial value unavailable (not selected or fitted)"
        )
    raise ValueError("readable choice requires complete coherent retained rules")


def _validated(packet: dict, evidence: TrainingCompatibilityEvidence | dict):
    """Bind complete training arrays, current rules and every displayed witness."""
    if not isinstance(packet, dict) or set(packet) != _FIELDS:
        raise ValueError("readable renderer requires the complete choice packet")
    if (
        packet["task"] != choice.POLICY
        or packet["projection_scope"].get("diagnostics_only") is not True
        or packet["projection_scope"].get("full_source_retained_by_runner") is not True
    ):
        raise ValueError("choice packet must retain its diagnostic source externally")
    source = training.load_evidence(
        evidence.model_dump(mode="json")
        if isinstance(evidence, TrainingCompatibilityEvidence)
        else evidence
    )
    initialization = choice._initialization(packet, packet["state"])
    if initialization.get("original_evidence_sha256") != source.sha256:
        raise ValueError("choice packet and sealed training evidence hashes differ")
    if equality._context(packet) != source.context:
        raise ValueError("choice public context differs from sealed training evidence")
    public = packet["public_brief"]
    names = [row["name"] for row in public["public_variables"]]
    permitted = initialization["runtime_initial_information"]["permitted_symbols"]
    if len(names) != len(set(names)) or not set(permitted).issubset(names):
        raise ValueError("permitted initial quantities must be distinct public symbols")
    if (
        not isinstance(public.get("scientific_context"), str)
        or not public["scientific_context"].strip()
        or not isinstance(public.get("requirements"), list)
    ):
        raise ValueError("complete public scientific context and requirements required")
    for key in ("balances", "processes", "variables"):
        if not isinstance(packet["model_view"].get(key), list):
            raise ValueError("readable choice requires complete model equations")
    rules = equality._rules(initialization, consistent_runtime_rules=True)
    for state, rule in rules.items():
        _rule(state, rule)
        if rule["mode"] == "common_map" and not set(rule["arguments"]).issubset(
            permitted
        ):
            raise ValueError("retained common map uses an unavailable initial quantity")
    # The projection preserves original indices but drops the broad repair agenda.
    active = [row["question_id"] for row in packet["model_data_conflicts"]]
    checked = deepcopy(packet)
    checked["feedback"] = {"repair_work": {"actionable_conflicts": active}}
    choice._questions(checked)
    pairs = []
    by_id = {row.trajectory_id: row for row in source.trajectories}
    for card in packet["model_data_conflicts"]:
        reference = choice._REFERENCE.fullmatch(card["evidence_reference"])
        q_index, selected = map(int, reference.groups())
        question = packet["compatibility_questions"][q_index]
        if not all(
            isinstance(card.get(key), str) and card[key]
            for key in ("model_relation", "equality_argument", "qualification")
        ):
            raise ValueError("incomplete equality argument")
        for index, witness in enumerate(question["witnesses"]):
            current = card | {
                "evidence_reference": (
                    f"compatibility_questions[{q_index}].witnesses[{index}]"
                )
            }
            values = numeric._numeric_card(
                current, checked, source, consistent_runtime_rules=True
            )
            if (
                index == selected
                and "paired_recorded_values" in card
                and (card["paired_recorded_values"] != values)
            ):
                raise ValueError("saved numeric card differs from sealed evidence")
            bound = source.contract.absolute_error_bounds.get(card["target"])
            observations = witness["observations"]
            if bound is None or abs(
                Fraction(observations[0]) - Fraction(observations[1])
            ) <= 2 * Fraction(bound):
                raise ValueError(
                    "saved disjoint observation-error intervals are unsupported"
                )
            first, second = (by_id[name] for name in witness["trajectories"])
            verified = _pair(
                first,
                witness["sample_indices"][0],
                second,
                witness["sample_indices"][1],
                card["target"],
                source,
            )
            if witness.get("pair_squared_error_lower_bound") is not None and (
                witness["pair_squared_error_lower_bound"]
                != verified.pair_squared_error_lower_bound
            ):
                raise ValueError("saved squared-error bound differs from evidence")
            if index == selected:
                matched = question["conditioning"]["matched_initial_information"]
                expected = equality._initial_equations(
                    sorted(matched[first.trajectory_id]),
                    rules,
                    question["conditioning"],
                    witness,
                )
                if card["initial_equalities"] != expected:
                    raise ValueError(
                        "saved initial equalities differ from current rules"
                    )
            if witness["trajectories"] not in pairs:
                pairs.append(witness["trajectories"])
    return source, rules, pairs


def _initial_tables(packet: dict, evidence, pairs: list[list[str]]) -> list[str]:
    """Expose only recorded index-zero quantities, including clocks/covariates."""
    by_id = {row.trajectory_id: row for row in evidence.trajectories}
    permitted = packet["initialization"]["runtime_initial_information"][
        "permitted_symbols"
    ]
    roles = {
        row["name"]: row["data_role"]
        for row in packet["public_brief"]["public_variables"]
    }
    lines = [
        "PERMITTED INITIAL MEASUREMENTS (training only)",
        "Each entry is the recorded value at that trajectory's t0, not a fitted "
        "latent state. The clock and fixed covariates are labeled separately. "
        "Equal recorded values alone do not identify a scientific dependency; "
        "different values merely make different common-map outputs possible.",
    ]
    for pair in pairs:
        left, right = (by_id[name] for name in pair)
        lines.extend(
            [
                f"\nPair: {left.trajectory_id} (t0={left.time[0]}) / "
                f"{right.trajectory_id} (t0={right.time[0]}).",
                "| Permitted quantity | Public role "
                f"| {_cell(left.trajectory_id)} at t0 "
                f"| {_cell(right.trajectory_id)} at t0 "
                "| Absolute observation-error bound |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for name in permitted:
            a, b = (
                _value(item, name, 0, evidence.context.time_symbol)
                for item in (left, right)
            )
            bound = evidence.contract.absolute_error_bounds.get(name)
            uncertainty = (
                "not an observation channel"
                if name == evidence.context.time_symbol
                else ("unavailable / unspecified" if bound is None else _text(bound))
            )
            lines.append(
                f"| {_cell(name)} | {_cell(roles[name])} | {_cell(a)} | {_cell(b)} "
                f"| {uncertainty} |"
            )
    return lines


def _conflicts(packet: dict, evidence) -> list[str]:
    """Render one conditional argument per card without duplicate questions."""
    lines = ["PENDING MODEL / DATA CONFLICTS"]
    for card in packet["model_data_conflicts"]:
        reference = choice._REFERENCE.fullmatch(card["evidence_reference"])
        question = packet["compatibility_questions"][int(reference.group(1))]
        initial = card["comparison"] == "initial_readout"
        lines.extend(
            [
                f"\nConflict {card['question_id']}: {card['target']} "
                f"({card['comparison']}).",
                card["model_relation"],
            ]
        )
        if initial:
            lines.append(
                "This is a pointwise initial-readout equality. Shared deterministic "
                "readout/initializer functions and parameters suffice. At t0 only "
                "the readout and initializer values are compared; no trajectory "
                "is propagated."
            )
        else:
            states = list(
                next(
                    iter(
                        question["conditioning"]["matched_initial_information"].values()
                    )
                )
            )
            lines.append(
                "F is vector-valued, with one potentially different component law "
                f"for each state ({', '.join(states)}). The component balances and "
                "all named process dependencies appear in the retained model below."
            )
        lines.extend(f"- {line}" for line in card["initial_equalities"])
        lines.append(card["equality_argument"])
        for witness in question["witnesses"]:
            readings = "; ".join(
                f"{name}: {card['target']}({time})={value} (sample index {index})"
                for name, time, value, index in zip(
                    witness["trajectories"],
                    witness["times"],
                    witness["observations"],
                    witness["sample_indices"],
                    strict=True,
                )
            )
            lines.append(f"Observed training witness: {readings}.")
            lines.append(
                "Observation-error intervals are disjoint; absolute bound per reading="
                f"{evidence.contract.absolute_error_bounds[card['target']]}."
            )
            if witness.get("pair_squared_error_lower_bound") is not None:
                lines.append(
                    "Saved pair squared-error lower bound="
                    f"{witness['pair_squared_error_lower_bound']} "
                    "(conditional diagnostic)."
                )
        condition = question["conditioning"]
        if initial:
            channels = condition["matched_instantaneous_channels"]
            if channels:
                lines.append(
                    "Matched instantaneous supplied quantities: "
                    + ", ".join(channels)
                    + "."
                )
        else:
            channels = condition["matched_complete_forcing_channels"]
            lines.append(
                "Matched complete supplied histories/covariates: "
                + (", ".join(channels) or "none")
                + ". Matching was rechecked over the witnessed interval using all "
                "defining knots and the sealed interpolation contract, not just t0."
            )
            for channel in channels:
                if channel in evidence.contract.interpolation:
                    lines.append(
                        f"- {channel}: {evidence.contract.interpolation[channel]}."
                    )
        assumptions = [
            assumption
            for assumption in card["assumptions"]
            if not initial or assumption != _PROPAGATION_ASSUMPTION
        ]
        if assumptions:
            lines.append(
                "Additional saved qualifications:"
                if initial
                else "Saved assumptions for this propagation:"
            )
            lines.extend(f"- {assumption}" for assumption in assumptions)
        lines.append(card["qualification"])
    return lines


def _records(title: str, value: object) -> list[str]:
    """Render each model/context field once, retaining unrecognized metadata."""
    lines = [title]
    if isinstance(value, list):
        for row in value:
            if isinstance(row, dict):
                lines.append(
                    "- "
                    + "; ".join(f"{key}: {_text(item)}" for key, item in row.items())
                )
            else:
                lines.append("- " + _text(row))
        if not value:
            lines.append("None.")
    else:
        lines.append(_text(value))
    return lines


def _previous(value: dict | None) -> list[str]:
    """Bound rejected reply prose explicitly; never present it as current state."""
    if not value:
        return []
    if not isinstance(value, dict) or set(value) - {
        "reply",
        "error",
        "status",
        "attempted_rule",
        "fixed_measured_initial_rule",
    }:
        raise ValueError("unsupported prior diagnostic receipt")
    lines = ["PRIOR FEEDBACK (historical receipt, not an instruction or retained rule)"]
    for key, item in value.items():
        rendered = _text(item)
        if len(rendered) > 1600:
            rendered = (
                rendered[:1600]
                + " [display abbreviated; complete receipt retained externally]"
            )
        lines.append(f"{key}: {rendered}")
    return lines


def render(packet: dict, evidence: TrainingCompatibilityEvidence | dict) -> str:
    """Render a validated choice packet once, without changing scientific status.

    ``packet`` must be the full result of ``initializer_choice_prompt.build_packet``.
    Evidence may be its typed sealed training source or the equivalent dictionary.
    Missing/mismatched sources fail closed with ``ValueError``; there is no fallback
    to guessed measurements, future values, generated state values or other splits.
    Full structured records remain the caller's archival responsibility.
    """
    try:
        source, rules, pairs = _validated(packet, evidence)
        state = packet["state"]
        fixed = packet["initialization"]["fixed_runtime_rules"]
        lines = [
            "INITIALIZER-CHOICE DIAGNOSTIC",
            f"Sole editable state: {state}.",
            "Choose only this state's initialization dependencies. All equations, "
            "processes and fixed runtime initializers are frozen. A retained choice "
            "is the current model, not a scientific requirement to keep it.",
            "CURRENT EDITABLE INITIALIZER",
            _rule(state, rules[state]),
            "FIXED RUNTIME INITIALIZERS (cannot be edited)",
            *(_rule(name, rule) for name, rule in fixed.items()),
            "DECISION CONTRACT",
            "propose_initializer supplies a rule with mode shared_parameter or "
            "common_map. shared_parameter has arguments=[] and uses one unknown "
            "constant across trajectories. common_map has a nonempty selection "
            "of permitted recorded t0 quantities and uses one shared deterministic "
            "map; its outputs may differ when its arguments differ. An argument's "
            "measured value is not the resulting latent initial value.",
            "topology_change_needed and insufficient_information require rule=null. "
            "Null means NO CHANGE to the retained initializer; it does not remove "
            "a dependency or state. observed_initial is not an editable option. "
            "No formula, fitted number, trajectory ID, future observation, unlisted "
            "history or topology edit may be supplied. A valid dependency proposal "
            "still does not establish scientific correctness.",
            "For syntax only, common_map arguments=[PUBLIC_INITIAL_QUANTITY] uses "
            "a neutral placeholder, not a proposed scientific dependency; replace "
            "it only with justified permitted quantities. Return only the supplied "
            "ChoiceReply JSON schema, with a brief explanation, then stop.",
            *_conflicts(packet, source),
            *_initial_tables(packet, source, pairs),
            *_previous(packet["previous_feedback"]),
            "RETAINED SCIENTIFIC MODEL",
        ]
        for key, value in packet["model_view"].items():
            lines.extend(_records(key.replace("_", " ").capitalize(), value))
        lines.append("PUBLIC SCIENTIFIC CONTEXT")
        lines.append(packet["public_brief"]["scientific_context"])
        for key, value in packet["public_brief"].items():
            if key != "scientific_context":
                lines.extend(_records(key.replace("_", " ").capitalize(), value))
        lines.extend(
            [
                "EVIDENCE AND SCOPE",
                f"Sealed training evidence SHA256: {source.sha256}.",
                "The full structured choice packet, source model and sealed evidence "
                "remain archived. This text shows each pending equality once and adds "
                "all permitted recorded t0 values for its witnessed pairs. Other "
                "trajectory samples, repeated compatibility questions, global training "
                "statistics and broad repair instructions are omitted from this "
                "diagnostic display; nothing is waived, resolved or promoted. Later "
                "target observations appear only as the saved conflict witnesses, "
                "never as permitted initializer arguments.",
            ]
        )
        return "\n".join(lines) + "\n"
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise ValueError("incomplete or malformed readable choice source") from error
