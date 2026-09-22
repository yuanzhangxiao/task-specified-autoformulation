"""General shared-law revision through the established whole-model compiler."""

from autoformalism.expressions import RestrictedParser
from autoformalism.search import review_revision_v6 as previous

POLICY = "general-shared-revision-1"
ScientificRevision = previous.ScientificRevision
SYSTEM_PROMPT = (
    previous.SYSTEM_PROMPT
    + """
Shared processes are ordinary named algebraic definitions with multiple consumers.
Edit a named law ONCE to change every use; do not duplicate its function or parameters
in each consumer. The runtime compiles all affected equations together. If you want
to split, remove or replace a shared law, supply the coordinated equation changes.
Consumer signs and conversions are visible in the COMPLETE assembled model below;
they have already been applied. Do not insert them into the intrinsic law again.
A shared parameter name denotes one fitted value everywhere it occurs. Use a new
identity only if you intend an independent parameter. No physical conservation is
inferred from naming, repeated syntax, or opposite signs. There is no separate
confirmation call: a valid explicit patch is eligible for fitting. The next prompt
always shows the current complete model. No change remains a valid choice.
"""
)


def relationships(candidate: dict) -> list[dict]:
    """Derive actual law consumers from expressions, never historical annotations."""
    definitions = {e["state"]: e["rhs"] for e in candidate["state_equations"]}
    definitions.update({p["name"]: p["expression"] for p in candidate["processes"]})
    symbols = {
        n: RestrictedParser().parse(e, location=n).symbols
        for n, e in definitions.items()
    }
    result = []
    for process in candidate["processes"]:
        name = process["name"]
        direct = {n for n, deps in symbols.items() if name in deps}
        affected = set(direct)
        while True:
            more = {n for n, deps in symbols.items() if deps & affected} - affected
            if not more:
                break
            affected |= more
        result.append(
            {
                "process": name,
                "law": process["expression"],
                "direct_consumers": sorted(direct),
                "affected_equations": sorted(affected),
            }
        )
    return result


def payload(bundle, packet, parameters, retry=None):
    """Use complete alias-consistent equations and training-only residual evidence."""
    value = previous.payload(bundle, packet, parameters, retry)
    value["protocol"] = POLICY
    model = value["model"]
    candidate = {
        "state_equations": [
            {"state": e["component"], "rhs": e["expression"]}
            for e in model["equations"]
            if e["kind"] == "dynamic"
        ],
        "processes": [
            {"name": e["component"], "expression": e["expression"]}
            for e in model["equations"]
            if e["kind"] == "algebraic"
        ],
    }
    value["shared_law_relationships"] = relationships(candidate)
    return value


def apply_edits(bundle, packet, raw):
    """Propagate shared-law edits and allow coordinated topology changes."""
    result = previous.apply_edits(bundle, packet, raw)
    child = result["bundle"]
    before = relationships(bundle["candidate"])
    after = relationships((child or bundle)["candidate"])
    result["provenance"]["shared_law_revision"] = {
        "policy": POLICY,
        "before": before,
        "after": after,
        "conservation_certified": False,
    }
    if child:
        child["revision_provenance"] = result["provenance"]
    return result


def feedback(bundle, packet, parameters, raw, error):
    """Retain actionable syntax/role/closure diagnostics from the shared compiler."""
    return previous.feedback(bundle, packet, parameters, raw, error)
