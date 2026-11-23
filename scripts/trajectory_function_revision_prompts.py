"""Offer a fixed-graph formula or a scientifically motivated change of sources."""

from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints, create_model

from autoformalism.schemas.base import StrictSchema
from scripts import trajectory_function_prompts as fixed


class SelectOnGraph(fixed.SelectFamily):
    """Select the entire shortlisted model including its parameter starts."""

    route: Literal["fixed_graph"]


def response_schema(graph: dict, hybrid: bool = True) -> type[BaseModel]:
    """Require the complete existing inventory on either authored route."""
    authored = fixed.response_schema(graph, False).model_fields["candidate"].annotation
    keep = create_model(
        "WriteOnGraph", __base__=authored, route=(Literal["fixed_graph"], ...)
    )
    change = create_model(
        "ChangeEquationSources",
        __base__=authored,
        route=(Literal["request_topology_change"], ...),
        scientific_reason=(
            Annotated[
                str,
                StringConstraints(strip_whitespace=True, min_length=1, max_length=800),
            ],
            ...,
        ),
    )
    choices = (SelectOnGraph | keep | change) if hybrid else (keep | change)
    return create_model(
        "RoutedFunctionReply", __base__=StrictSchema, candidate=(choices, ...)
    )


def request(
    graph: dict, frozen: dict, summary: list, menu: dict, hybrid: bool = True
) -> tuple[str, dict, type[BaseModel]]:
    """Present only public context and training summaries to the proposer."""
    from autoformalism.research.trajectory_function_revision import permitted_sources

    _, packet, _ = fixed.request(graph, frozen, summary, menu, hybrid)
    packet["sources_available_to_add"] = permitted_sources(graph, frozen)
    packet["measured_initial_bindings"] = frozen.get("measured_initial_bindings", {})
    instruction = (
        "Choose one route. On fixed_graph, "
        + ("select A or B by shortlist_id, or " if hybrid else "")
        + "write a complete model using exactly the listed sources in each equation. "
        "On request_topology_change, write a complete model whose equations add or "
        "remove at least one source, and give the scientific reason. The runtime reads "
        "the new sources from your equations and checks the revised graph against "
        "the public requirements and training observations. Both routes use the given "
        "variables, variable types and initialization rules.\n"
        "For an authored model, give only the RHS for each differential or algebraic "
        "equation. Use its existing sources, names from sources_available_to_add, "
        "and declared coefficients. Use "
        "the model's generated values throughout the rollout. A measured initial "
        "binding gives one reading at time zero. Add an initial channel as a source "
        "only when it appears in sources_available_to_add. A fixed covariate "
        "keeps its value throughout the trajectory. In each initializer use "
        "the specified public initial values and state0_NAME for any listed state "
        "initials. Each shared parameter needs one name and one finite starting "
        "value. For expressions use +, -, *, /, integer powers ** from -16 to 16, "
        "and abs, exp, log, sqrt, min, max, sin, tanh, sigmoid or softplus."
    )
    return instruction, packet, response_schema(graph, hybrid)


def decode_reply(
    graph: dict, reply: dict, shortlist: list[dict], hybrid: bool = True
) -> tuple[str, dict, str, str | None]:
    """Validate the chosen route and lower the formula for its admission check."""
    candidate = response_schema(graph, hybrid).model_validate(reply).candidate
    fields = candidate.model_dump()
    route = fields.pop("route")
    reason = fields.pop("scientific_reason", None)
    formula, origin = fixed.to_formula({"candidate": fields}, shortlist)
    return route, formula, origin, reason
