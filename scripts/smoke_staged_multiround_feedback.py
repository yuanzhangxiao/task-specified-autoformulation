#!/usr/bin/env python3
"""CPU-only structural smoke for function-first multi-round routing."""

from __future__ import annotations

import json

from autoformalism.rebuttal.fitter_recovery import CONTEXT, recovery_candidate
from autoformalism.rebuttal.staged_multiround_feedback_campaign import (
    ComponentRevisionReply,
    _route,
    apply_component_revision,
    select_revision_components,
)
from autoformalism.search.identity import candidate_identity


def main() -> None:
    """Verify selection, function isolation, and bounded topology escalation."""
    source = recovery_candidate()
    selected = select_revision_components(source)
    reply = ComponentRevisionReply.model_validate(
        {
            "revisions": [
                {
                    "component": "f",
                    "expression": "k_feedback*sigmoid(v01) - f/tau_feedback",
                    "parameters": [
                        {
                            "name": "k_feedback",
                            "role": "nonnegative_coefficient",
                        },
                        {"name": "tau_feedback", "role": "time_constant"},
                    ],
                },
                {
                    "component": "v01",
                    "expression": (
                        "offset + k_readout*sigmoid(f) + m + k_p*p + k_u*u01"
                    ),
                    "parameters": [
                        {"name": "offset", "role": "offset"},
                        {
                            "name": "k_readout",
                            "role": "nonnegative_coefficient",
                        },
                        {"name": "k_p", "role": "nonnegative_coefficient"},
                        {"name": "k_u", "role": "nonnegative_coefficient"},
                    ],
                },
            ]
        }
    )
    revised, audit = apply_component_revision(
        source,
        reply,
        CONTEXT,
        selected=selected,
        route="function_revision",
    )
    before = candidate_identity(source)
    after = candidate_identity(revised)
    result = {
        "schema_version": "staged-multiround-feedback-smoke-1",
        "status": "pass"
        if selected == ("f", "v01")
        and before.topology_sha256 == after.topology_sha256
        and before.functional_sha256 != after.functional_sha256
        and not audit["topology_changed"]
        and _route(1, []) == "function_revision"
        and _route(2, [{"numerically_stable": False}]) == "topology_revision"
        else "fail",
        "selected_components": list(selected),
        "round_one_route": _route(1, []),
        "persistent_instability_route": _route(2, [{"numerically_stable": False}]),
        "topology_preserved_by_function_revision": (
            before.topology_sha256 == after.topology_sha256
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
