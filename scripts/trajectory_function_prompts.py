"""Matched 20B prompts for function construction on fixed public topologies."""

from typing import Annotated, Literal

from pydantic import Field, create_model

from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema


class Parameter(StrictSchema):
    """Shared fitted coefficient and its suggested starting value."""

    name: Identifier
    initial: FiniteFloat


class SelectFamily(StrictSchema):
    """Use one complete shortlisted formula bundle exactly as displayed."""

    shortlist_id: Literal["A", "B"]


def response_schema(graph: dict, hybrid: bool):
    """Require the exact equation and initializer keys in authored expressions."""

    def expressions(name, keys):
        return create_model(
            name,
            __base__=StrictSchema,
            **{
                k: (Annotated[str, Field(min_length=1, max_length=4096)], ...)
                for k in sorted(keys)
            },
        )

    free = [
        k for k, r in graph["initializers"].items() if r["mode"] != "observed_initial"
    ]
    authored = create_model(
        "WriteFunctions",
        __base__=StrictSchema,
        equations=(expressions("Equations", graph["sources"]), ...),
        initializers=(expressions("Initializers", free), ...),
        parameters=(Annotated[list[Parameter], Field(max_length=96)], ...),
    )
    return create_model(
        "FunctionReply",
        __base__=StrictSchema,
        candidate=((SelectFamily | authored) if hybrid else authored, ...),
    )


def to_formula(reply: dict, shortlist: list[dict]) -> tuple[dict, str]:
    """Convert the provider representation to the common fitter interface."""
    v = reply["candidate"]
    if "shortlist_id" in v:
        i = ("A", "B").index(v["shortlist_id"])
        return shortlist[i], f"selected_{v['shortlist_id']}"
    params = {r["name"]: r["initial"] for r in v["parameters"]}
    if len(params) != len(v["parameters"]):
        raise ValueError("Return each shared parameter name exactly once")
    formula = {**v, "parameters": params}
    for i, alternative in enumerate(shortlist):
        if formula == alternative:
            return formula, "authored_exact_" + ("A", "B")[i]
    return formula, "authored"


def observations(rows: list[dict]) -> list[dict]:
    """Compact public summaries shared by the proposer and hybrid arms."""
    results = []
    for row in rows:
        t = row["time"]
        results.append(
            {
                "trajectory": row["trajectory_id"],
                "targets": {
                    name: {
                        "start": round(v[0], 5),
                        "end": round(v[-1], 5),
                        "min": round(min(v), 5),
                        "max": round(max(v), 5),
                        "time_of_sampled_max": t[v.index(max(v))],
                    }
                    for name, v in row["targets"].items()
                },
                "input_ranges": {
                    name: [round(min(v), 5), round(max(v), 5)]
                    for name, v in row["inputs"].items()
                },
            }
        )
    return results


def request(graph: dict, frozen: dict, summary: list, menu: dict, hybrid: bool):
    """Give both proposer arms the same task, graph, observations and grammar."""
    instruction = (
        "Write one complete continuous-time model for the fixed topology. For each "
        "differential variable return its derivative expression, and for each "
        "algebraic variable return its value expression. Return only the expression "
        "on the RHS. Use every listed source for that equation with a meaningful "
        "dependence. Use a single shared parameter name for the same coefficient in "
        "different equations. Respect physical units, zero-input behavior and the "
        "public mechanisms insofar as this graph permits. Use supplied inputs at "
        "their current times. Give an initialization expression for every unmeasured "
        "differential initial. For maps, use the listed public channel names at time "
        "zero and state0_NAME for each listed state initial. A shared initial "
        "constant may be one parameter. List each estimated coefficient and its "
        "finite starting value once. Use numbers, variable and parameter names, "
        "+, -, *, /, ** with integer literal exponents from -16 to 16, and the "
        "functions abs, exp, log, sqrt, min, max, "
        "sin, tanh, sigmoid or softplus. Use parentheses for grouping."
    )
    packet = {
        "scientific_task": frozen["public_brief"]["scientific_context"],
        "variable_roles": frozen.get("inventory_roles", {}),
        "public_requirements": frozen["public_brief"].get("requirements", []),
        "assigned_memory": frozen.get("mechanism_bindings", []),
        "graph": graph,
        "training_observations": summary,
    }
    if hybrid:
        instruction += (
            " You may select a complete suggested model by its shortlist_id or write "
            "your own complete model. Use the evidence to make that choice."
        )
        packet["evidence_for_functions"] = menu["evidence"]
        packet["shortlist"] = dict(zip(("A", "B"), menu["candidates"], strict=True))
    return instruction, packet, response_schema(graph, hybrid)
