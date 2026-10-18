"""Opt-in presentation of sealed training values and rejected initializer edits.

This adapter only annotates existing equality cards and structured rejected
transactions. It does not create diagnostics, choose dependencies, or evaluate
an initializer. Earlier packet fields and diagnostic statuses remain unchanged.
"""

from __future__ import annotations

import re
from copy import deepcopy

from autoformalism.research import initialization_equality_feedback as equality
from autoformalism.search.construction_initialization_focus import (
    expand_compatibility_details,
)
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_compatibility import (
    CompleteTrajectory,
    TrainingCompatibilityEvidence,
    _forcing_equal,
    _value,
    validate_evidence,
)

POLICY = "recorded-initialization-evidence-feedback-1"
_REFERENCE = re.compile(r"compatibility_questions\[(\d+)\]\.witnesses\[(\d+)\]")
_MAX_ROWS = 8
_MAX_FORCING = 4
_MAX_KNOTS = 4


def _readings(
    pair: list[CompleteTrajectory], indices: list[int], channel: str, time_symbol: str
) -> list[dict]:
    """Label each exact recorded value with its channel, trajectory and time."""
    return [
        {
            "trajectory": item.trajectory_id,
            "channel": channel,
            "time": item.time[index],
            "value": _value(item, channel, index, time_symbol),
        }
        for item, index in zip(pair, indices, strict=True)
    ]


def _forcing_rows(pair: list, channels: list, evidence, end: float) -> list[dict]:
    """Bound the display only after checking the complete recorded histories."""
    equal, missing = _forcing_equal(*pair, set(channels), evidence, end)
    if not equal or missing:
        raise ValueError("saved forcing match is not supported by complete evidence")
    result = []
    for channel in channels[:_MAX_FORCING]:
        if channel in {
            *evidence.context.fixed_covariates,
            evidence.context.time_symbol,
        }:
            result.append(
                {
                    "role": "fixed_covariate"
                    if channel in pair[0].covariates
                    else "clock",
                    "readings": _readings(
                        pair, [0, 0], channel, evidence.context.time_symbol
                    ),
                }
            )
            continue
        trajectories = []
        for item in pair:
            indices = [i for i, time in enumerate(item.time) if time <= end]
            # Retain endpoints and early knots without implying unseen knots are zero.
            shown = (
                indices if len(indices) <= _MAX_KNOTS else indices[:3] + indices[-1:]
            )
            trajectories.append(
                {
                    "trajectory": item.trajectory_id,
                    "knots": [
                        {
                            "time": item.time[index],
                            "value": _value(
                                item, channel, index, evidence.context.time_symbol
                            ),
                        }
                        for index in shown
                    ],
                    "omitted_knot_count": len(indices) - len(shown),
                }
            )
        result.append(
            {
                "role": "complete_supplied_history",
                "channel": channel,
                "interval": [pair[0].time[0], end],
                "interpolation": evidence.contract.interpolation[channel],
                "complete_histories_match": True,
                "trajectories": trajectories,
            }
        )
    return result


def _numeric_card(card: dict, packet: dict, evidence) -> dict:
    """Resolve an existing witness against its sealed source, failing closed."""
    reference = _REFERENCE.fullmatch(card.get("evidence_reference", ""))
    if reference is None:
        raise ValueError("unsupported witness reference")
    q_index, w_index = map(int, reference.groups())
    question = packet["compatibility_questions"][q_index]
    witness = question["witnesses"][w_index]
    active = packet["feedback"]["repair_work"]["actionable_conflicts"]
    if (
        not equality._qualified(question, active)
        or not equality._pair(witness)
        or card["question_id"] != question["question_id"]
        or card["target"] != question["target"]
    ):
        raise ValueError("unqualified or stale equality card")
    by_id = {item.trajectory_id: item for item in evidence.trajectories}
    pair = [by_id[name] for name in witness["trajectories"]]
    indices = witness["sample_indices"]
    if pair[0].time[0] != pair[1].time[0]:
        raise ValueError("different recorded time origins")
    outputs = _readings(pair, indices, card["target"], evidence.context.time_symbol)
    if any(
        row["time"] != time or row["value"] != value
        for row, time, value in zip(
            outputs, witness["times"], witness["observations"], strict=True
        )
    ):
        raise ValueError("witness does not identify these recorded outputs")
    conditioning = question["conditioning"]
    matched = conditioning["matched_initial_information"]
    rules = equality._rules(packet["initialization"])
    if set(matched[pair[0].trajectory_id]) != set(matched[pair[1].trajectory_id]):
        raise ValueError("initializer conditioning differs")
    states = sorted(matched[pair[0].trajectory_id])
    if equality._initial_equations(states, rules, conditioning, witness) is None:
        raise ValueError("saved initializer conditioning is stale")
    rows = []
    for state in states:
        entry = matched[pair[0].trajectory_id][state]
        for channel in sorted(entry["recorded_arguments"]):
            readings = _readings(pair, [0, 0], channel, evidence.context.time_symbol)
            if any(
                row["value"]
                != matched[row["trajectory"]][state]["recorded_arguments"][channel]
                for row in readings
            ):
                raise ValueError(
                    "initializer argument is not the recorded initial value"
                )
            rows.append(
                {
                    "role": "recorded_initializer_argument",
                    "state": state,
                    "retained_rule": rules[state],
                    "readings": readings,
                }
            )
    forcing = []
    if card["comparison"] == "initial_readout":
        if indices != [0, 0]:
            raise ValueError("initial comparison is not at recorded t0")
        channels = conditioning["matched_instantaneous_channels"]
        for channel in channels:
            readings = _readings(pair, indices, channel, evidence.context.time_symbol)
            if readings[0]["value"] != readings[1]["value"]:
                raise ValueError("instantaneous conditioning differs")
            rows.append(
                {"role": "instantaneous_supplied_argument", "readings": readings}
            )
    elif card["comparison"] == "later_subsystem" and all(
        index > 0 for index in indices
    ):
        channels = conditioning["matched_complete_forcing_channels"]
        forcing = _forcing_rows(pair, channels, evidence, witness["times"][0])
    else:
        raise ValueError("unsupported equality comparison")
    result = {
        "source": "sealed training evidence",
        "evidence_sha256": evidence.sha256,
        "observed_output": outputs,
        "conditioning_values": rows[:_MAX_ROWS],
        "omitted_conditioning_row_count": max(0, len(rows) - _MAX_ROWS),
        "interpretation": "These are recorded public values, not fitted initials. "
        "A common_map argument is a measurement supplied to the shared map; its "
        "value is not the resulting state value. Shared parameters have no recorded "
        "numeric value here. The equality card's assumptions and uncertainty "
        "qualification still apply.",
    }
    if forcing:
        result["supplied_histories"] = forcing
        result["omitted_forcing_channel_count"] = max(0, len(channels) - _MAX_FORCING)
        result["history_display"] = (
            "Only bounded knots are displayed; matching was "
            "checked on complete defining histories over the stated interval."
        )
    return result


def _rejected_reply(packet: dict) -> dict | None:
    """Read the structured rejected payload, never infer an edit from error prose."""
    if packet.get("previous_edit", {}).get("status") != "rejected":
        return None
    feedback = packet.get("feedback", {})
    repeated = feedback.get("repeated_repair_feedback", {})
    candidates = [
        packet.get("rejected_reply"),
        packet.get("previous_edit", {}).get("rejected_reply"),
        feedback.get("rejected_reply"),
        repeated.get("observations", {}).get("previous_reply"),
        repeated.get("previous_reply"),
    ]
    return next((reply for reply in candidates if isinstance(reply, dict)), None)


def _unsupported_observed(packet: dict, draft: Draft) -> list[dict]:
    """Explain illegal observed bindings using retained rules and public symbols."""
    reply = _rejected_reply(packet)
    if reply is None or not isinstance(reply.get("initialization"), dict):
        return []
    initialization = packet.get("initialization", {})
    runtime = initialization.get("runtime_initial_information", {})
    if runtime.get("scope") != "recorded_t0":
        return []
    public = {
        row["name"]
        for row in packet.get("public_brief", {}).get("public_variables", [])
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    quantities = sorted(set(runtime.get("permitted_symbols", [])) & public)
    states = {v.name for v in draft.variables if v.definition == "differential"}
    editable = set(initialization.get("editable_states", []))
    fixed = initialization.get("fixed_runtime_rules", {})
    retained = {
        row["state"]: row["rule"]
        for row in initialization.get("retained_handoff", {}).get("records", [])
    }
    result, seen = [], set()
    for record in reply["initialization"].get("records", []):
        if not isinstance(record, dict) or not isinstance(record.get("rule"), dict):
            continue
        state = record.get("state")
        if (
            not isinstance(state, str)
            or state in seen
            or state not in states & editable
            or record["rule"].get("mode") != "observed_initial"
            or fixed.get(state, {}).get("mode", "unknown") != "unknown"
        ):
            continue
        seen.add(state)
        result.append(
            {
                "state": state,
                "attempted_rule": deepcopy(record["rule"]),
                "retained_rule": deepcopy(retained.get(state, fixed.get(state))),
                "fixed_measured_initial_rule": None,
                "available_initial_public_quantities": quantities,
                "legal_editable_choices": {
                    "shared_parameter": "One unknown initial value shared across "
                    "trajectories; arguments=[].",
                    "common_map": "One shared deterministic map of a scientifically "
                    "justified selection of the listed recorded t0 quantities.",
                },
                "instruction": f"{state} has no fixed measured-initial rule. "
                "Declaring observed_initial does not create a measurement. Choose "
                "a justified editable rule and its dependencies; no value or "
                "dependency is selected here. The rejected transaction did not "
                "replace the retained rule. Preserve other intended valid records "
                "when resubmitting. This schema correction alone need not resolve "
                "the conditional model/data conflict.",
            }
        )
    return result


def project(
    packet: dict, draft: Draft, evidence: TrainingCompatibilityEvidence | dict
) -> dict:
    """Annotate V5 packets after equality projection; omit unverifiable additions."""
    result = deepcopy(packet)
    try:
        guidance = _unsupported_observed(packet, draft)
    except (KeyError, TypeError, ValueError, AttributeError):
        guidance = []
    if guidance:
        result.setdefault("feedback", {})["unsupported_observed_initial"] = guidance
    try:
        source = (
            evidence
            if isinstance(evidence, TrainingCompatibilityEvidence)
            else TrainingCompatibilityEvidence.model_validate(evidence)
        )
        validate_evidence(source)
        if packet["initialization"]["original_evidence_sha256"] != source.sha256:
            return result
        expanded = expand_compatibility_details(packet)
    except (KeyError, TypeError, ValueError, AttributeError):
        return result
    for card in result.get("model_data_conflicts", [])[:2]:
        try:
            card["paired_recorded_values"] = _numeric_card(card, expanded, source)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            continue
    return result
