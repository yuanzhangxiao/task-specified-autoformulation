"""Resolve explicit observed-target bounds from the public variable declarations."""

from __future__ import annotations

import re

# These two physical meanings were reviewed for the matched T2 experiment.
# Only the full description matches. Deviations or net changes need separate review.
REVIEWED_NONNEGATIVE = {
    "insulin concentration": "Absolute plasma insulin concentration.",
    "total glucose utilization/disposal rate, including insulin-independent "
    "and insulin-dependent contributions": "One-way total glucose disposal.",
}

DECLARATION = re.compile(r"-\s+([A-Za-z_]\w*)(?:\(t\))?\s*:\s*(.+)")
EXPLICIT = re.compile(
    r"(?:primary\s+)?non[- ]?negative\s+(?:stored-quantity\s+)?target"
    r"(?:\s+representing\s+[\w -]+)?",
    re.IGNORECASE,
)


def observed_target_bounds(brief: dict) -> dict:
    """Use target descriptions from the frozen public task, without observed values."""
    targets = {
        v["name"] for v in brief["public_variables"] if v["data_role"] == "target"
    }
    records = {
        name: {"lower": None, "basis": "unresolved", "declaration": None}
        for name in sorted(targets)
    }
    section = False
    for raw in brief["scientific_context"].splitlines():
        line = raw.strip()
        if line.casefold() == "target channels:":
            section = True
            continue
        if not section or not line:
            continue
        if not line.startswith("-"):
            section = False
            continue
        match = DECLARATION.fullmatch(line)
        if not match or match[1] not in targets:
            continue
        name, description = match.groups()
        if records[name]["declaration"] is not None:
            raise ValueError(f"Duplicate public target declaration for {name}")
        desc = re.sub(r"\s*\[unit:[^\]]+\]\s*$", "", description).strip()
        record = records[name]
        record["declaration"] = line
        if EXPLICIT.fullmatch(desc):
            record.update(lower=0.0, basis="explicit_public")
        elif desc.casefold() in REVIEWED_NONNEGATIVE:
            record.update(
                lower=0.0,
                basis="reviewed_quantity",
                reason=REVIEWED_NONNEGATIVE[desc.casefold()],
            )
    return records


def matched_bound_contract(cohort: dict) -> dict:
    """Bind the same observed-output bounds across the inventories being compared."""
    inventories = sorted({r["case"] for r in cohort["rows"]})
    if not inventories:
        raise ValueError("A matched bound contract requires a nonempty cohort")
    first = observed_target_bounds(cohort["cases"][inventories[0]]["public_brief"])
    for case in inventories[1:]:
        current = observed_target_bounds(cohort["cases"][case]["public_brief"])
        if current != first:
            raise ValueError("Matched inventories disagree on public target bounds")
    return {
        "targets": [name for name, rec in first.items() if rec["lower"] == 0],
        "observed_target_bounds": first,
    }
