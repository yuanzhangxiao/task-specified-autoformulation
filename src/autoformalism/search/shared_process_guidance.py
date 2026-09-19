"""Opt-in scientific scaffolding using existing algebraic variables and edges.

Guidance is not a new scientific requirement or a transfer certification. It
never changes accepted expressions, parameter identities or numerical settings.
"""

from typing import Literal

Stage = Literal["variables", "topology", "functions", "revision"]
POLICY = "shared-process-guidance-1"

COMMON = """
Shared-process modeling guidance (optional, not an extra task requirement):
Distinguish a state that stores memory from an algebraic process evaluated from
current states and supplied channels. An algebraic process has no independent
initial condition. Use one when the same scientifically meaningful law is needed
in several equations; do not add states or processes just to increase reuse.
Keep the model task-sufficient. Supplied auxiliaries need not be modeled merely
to create a second consumer. A reduced model may need no shared process.
A common source does not imply a common contribution or equal coefficients.
A transfer, a converted contribution and a shared response are different claims.
For a same-unit transfer, one law may enter one balance negatively and another
positively. Different units/volumes may require explicit conversion factors.
For a shared response, independent consumer gains can be scientifically necessary.
Do not infer conservation, positivity, or parameter equality from similar syntax
or opposite signs. A signed process and internal differences remain permitted.
"""

STAGES = {
    "variables": """
During variable identification, a new algebraic variable may organize a reusable
scientific law even if the same law could be written inline. Use the existing
algebraic definition and scientific_role fields. Describe its meaning and intended
reuse concisely. Do not emit its equation or parameters at this stage. Distinct
memory states still need distinct scientific roles; avoid redundant copies.
""",
    "topology": """
When an accepted algebraic process represents the intended shared contribution,
use its name as a source in each appropriate consumer rather than separately
reconstructing its upstream source set. Define its own drivers in its equation.
Describe the balance, response, or conversion in each term's scientific_role.
Place nonlinear/saturation shape descriptions in the defining law's terms. A
consumer that simply assembles that law does not apply a second nonlinear
response to it; describe that consumer as a balance or conversion contribution.
Keep the sign decision in outer_weight_sign. Do not duplicate a process law or
silently impose an independent consumer gain. A necessary conversion is allowed.
Existing inventory, source, cycle, and public-requirement checks still apply.
""",
    "functions": """
Define each named algebraic law once in its own equation. Refer to that source
by name at consumers. If it already is the complete unit-compatible contribution,
use the source itself with no new fitted gain; the runtime supplies the selected
outer sign. Introduce a conversion/gain only when scientifically needed.
In interaction_local mode, repeated parameter spellings do not share parameters.
Reuse the algebraic source to reuse the law and its coefficients. Do not inline
its expression into a consumer, invent a callable, or change a frozen source set.
Unknown physical conversions remain unknown; do not fix them to one for reuse.
""",
    "revision": """
Review all affected equations together. You may introduce an algebraic law and
revise all its consumers in one coherent patch using the existing equation schema.
Preserve an existing shared parameter when its scientific meaning is unchanged.
Factoring an identical expression preserves predictions and parameter count;
tying previously independent coefficients changes the model and needs a scientific
hypothesis and fitting. Similarity alone is insufficient. Do not automatically
factor, tie, or remove terms. Keep causal boundaries for every new dynamic state.
""",
}


def system_prompt(base: str, stage: Stage, enabled: bool = False) -> str:
    """Return legacy bytes when off; guide only the current scientific stage."""
    if not enabled:
        return base
    if stage == "variables":
        old = (
            "Introduce a new internal\n"
            "variable only when the displayed variables cannot represent a necessary\n"
            "finite-dimensional mechanism. Distinct internal variables "
            "must have distinct\n"
            "scientific roles."
        )
        if old not in base:
            raise ValueError("variable prompt changed; review shared-process guidance")
        base = base.replace(
            old, "Use scientifically meaningful states and algebraic processes."
        )
    return base + "\n" + COMMON + STAGES[stage]
