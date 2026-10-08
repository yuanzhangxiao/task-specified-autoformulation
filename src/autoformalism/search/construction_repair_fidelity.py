"""Concrete repair receipts; preserve proposer ownership of every scientific edit."""

from collections import Counter

from autoformalism.search import construction_bookkeeping as bookkeeping
from autoformalism.search import construction_handoff as handoff
from autoformalism.search import construction_ledger as ledger
from autoformalism.staged_topology import content_hash

INSTRUCTION = """Repair the stated failures; preserve other intended contributions.
The displayed balances include all ordinary terms and inserted process uses.
For a coordinate error choose the scientific representation first: keep a
differential target and repair its actual dependency paths, or explicitly choose
an algebraic readout with its real dynamic coordinates and coordinate all affected
types, topologies, uses and bindings. Repeating a binding adds no edges.
When replacing a joint contribution, account for every physical role you intend
to retain. Removing the whole term also removes its other drivers; runtime cannot
split it for you. Do not flip a type or sign merely to clear a structural error.
The next receipt shows the exact before/after balances and removals. Correct any
unintended loss using explicit edits; prose never deletes or restores entries."""


def removal_options(draft: ledger.Draft) -> list[dict]:
    """Show exact legal keys without recommending that any entry be removed."""
    return [
        {
            "process": p.name,
            "if_you_choose_to_delete": {"remove_processes": [p.name]},
            "affected_consumers": [u.target for u in p.uses],
        }
        for p in draft.processes
    ]


def _removed(before: list[dict], after: list[dict]) -> list[dict]:
    remaining = Counter(content_hash(x) for x in after)
    result = []
    for item in before:
        key = content_hash(item)
        if remaining[key]:
            remaining[key] -= 1
        else:
            result.append(item)
    return result


def receipt(before: ledger.Draft, after: ledger.Draft) -> dict:
    """Audit real committed changes, including propagated effects and multiplicity."""
    names = sorted(
        {v.name for d in (before, after) for v in d.variables}
        | {u.target for d in (before, after) for p in d.processes for u in p.uses}
    )
    balances = []
    for name in names:
        old, new = (
            bookkeeping.balance_view(before, name),
            bookkeeping.balance_view(after, name),
        )
        if old == new:
            continue
        balances.append(
            {
                "target": name,
                "before": old,
                "after": new,
                "removed_ordinary_records": _removed(
                    old["ordinary_terms"], new["ordinary_terms"]
                ),
                "removed_process_uses": _removed(
                    old["runtime_inserted_process_uses"],
                    new["runtime_inserted_process_uses"],
                ),
                "type_changed": old["definition"] != new["definition"],
            }
        )
    return {
        "committed_edits": handoff.edit_effects(before, after),
        "balances": balances,
        "scientific_correctness": "unassessed",
        "note": "Record differences only; the physical effect may remain equivalent. "
        "The runtime does not infer that equivalence or repair the model from prose.",
    }
