"""Deterministic semantic and domain validation for candidate models."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated

from pydantic import Field, model_validator

from autoformalism.expressions.diagnostics import (
    ModelValidationError,
    ValidationDiagnostic,
)
from autoformalism.expressions.intervals import (
    UNKNOWN_INTERVAL,
    Interval,
    analyze_interval,
)
from autoformalism.expressions.parser import (
    APPROVED_FUNCTION_ARITY,
    ParsedExpression,
    RestrictedParser,
)
from autoformalism.schemas import CandidateModel, ConstraintKind, ParameterScope
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema

IdentifierTuple = Annotated[tuple[Identifier, ...], Field(default=())]


class ValidationContext(StrictSchema):
    """Benchmark channels available to candidate expressions."""

    targets: IdentifierTuple
    lagged_targets: IdentifierTuple = ()
    auxiliaries: IdentifierTuple = ()
    external_inputs: IdentifierTuple = ()
    fixed_covariates: IdentifierTuple = ()
    unavailable_observed_channels: IdentifierTuple = ()
    time_symbol: Identifier = "t"
    forcing_bounds: Mapping[Identifier, tuple[FiniteFloat, FiniteFloat]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def channel_sets_are_disjoint(self) -> ValidationContext:
        """Reject ambiguous channel availability declarations."""
        named_sets = {
            "targets": set(self.targets),
            "auxiliaries": set(self.auxiliaries),
            "external_inputs": set(self.external_inputs),
            "fixed_covariates": set(self.fixed_covariates),
            "unavailable_observed_channels": set(
                self.unavailable_observed_channels
            ),
        }
        for name, values in named_sets.items():
            source = getattr(self, name)
            if len(values) != len(source):
                raise ValueError(f"duplicate channel in {name}")
        names = tuple(named_sets)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                overlap = named_sets[left_name] & named_sets[right_name]
                if overlap:
                    raise ValueError(
                        f"{left_name}/{right_name} overlap: {sorted(overlap)}"
                    )
        if not self.targets:
            raise ValueError("at least one target is required")
        unknown_lagged = set(self.lagged_targets) - set(self.targets)
        if unknown_lagged:
            raise ValueError(
                f"lagged targets are not target channels: {sorted(unknown_lagged)}"
            )
        unknown_bounds = set(self.forcing_bounds) - self.forcing_channels
        if unknown_bounds:
            raise ValueError(
                "forcing bounds reference unavailable channels: "
                f"{sorted(unknown_bounds)}"
            )
        for name, (lower, upper) in self.forcing_bounds.items():
            if lower > upper:
                raise ValueError(f"forcing bounds are reversed for {name}")
        return self

    @property
    def forcing_channels(self) -> frozenset[str]:
        """Return channels that may be supplied over the prediction horizon."""
        return frozenset(
            self.auxiliaries
            + self.external_inputs
            + self.fixed_covariates
            + self.lagged_targets
        )


@dataclass(frozen=True)
class ValidatedCandidate:
    """Candidate plus parsed expressions and a process evaluation order."""

    candidate: CandidateModel
    context: ValidationContext
    process_expressions: MappingProxyType[str, ParsedExpression]
    equation_expressions: MappingProxyType[str, ParsedExpression]
    observation_expressions: MappingProxyType[str, ParsedExpression]
    initial_condition_expressions: MappingProxyType[str, ParsedExpression]
    causal_derivative_initials: MappingProxyType[str, str]
    process_order: tuple[str, ...]
    forcing_symbols: frozenset[str]
    warnings: tuple[ValidationDiagnostic, ...] = ()


def repair_protected_declarations(
    candidate: CandidateModel,
    context: ValidationContext,
) -> tuple[CandidateModel, tuple[str, ...]]:
    """Apply lossless repairs at the benchmark/runtime boundary."""
    identifier_pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    declared_names = {
        *(item.name for item in candidate.states),
        *(item.name for item in candidate.processes),
        *(item.name for item in candidate.parameters),
    }
    referenced_names = {
        symbol
        for expression in (
            *(item.rhs for item in candidate.state_equations),
            *(item.expression for item in candidate.processes),
            *(item.expression for item in candidate.observation_mappings),
        )
        for symbol in identifier_pattern.findall(expression)
    }
    lag_aliases = {
        f"{target}_prev": target
        for target in context.lagged_targets
        if f"{target}_prev" in referenced_names
        and f"{target}_prev" not in declared_names
    }
    initial_repairs: list[str] = []
    if lag_aliases:
        payload = candidate.model_dump(mode="json")
        payload["processes"].extend(
            {"name": alias, "expression": target}
            for alias, target in sorted(lag_aliases.items())
        )
        candidate = CandidateModel.model_validate(payload)
        initial_repairs.extend(
            f"mapped causal lag alias {alias} to interval-boundary {target}"
            for alias, target in sorted(lag_aliases.items())
        )
    protected = set(context.external_inputs) | set(context.fixed_covariates)
    equation_states = {item.state for item in candidate.state_equations}
    undifferentiated_auxiliaries = {
        item.name
        for item in candidate.states
        if item.name in context.auxiliaries and item.name not in equation_states
    }
    removed_states = {
        item.name for item in candidate.states if item.name in protected
    } | undifferentiated_auxiliaries
    removed_processes = {
        item.name for item in candidate.processes if item.name in protected
    }
    process_by_name = {item.name: item for item in candidate.processes}
    reachable_processes: set[str] = set()
    pending = {
        symbol
        for expression in (
            *(item.rhs for item in candidate.state_equations),
            *(item.expression for item in candidate.observation_mappings),
        )
        for symbol in identifier_pattern.findall(expression)
        if symbol in process_by_name
    }
    while pending:
        name = pending.pop()
        if name in reachable_processes:
            continue
        reachable_processes.add(name)
        pending.update(
            symbol
            for symbol in identifier_pattern.findall(
                process_by_name[name].expression
            )
            if symbol in process_by_name and symbol not in reachable_processes
        )
    unused_processes = set(process_by_name) - reachable_processes
    removed_processes |= unused_processes
    removed_parameters = {
        item.name for item in candidate.parameters if item.name in protected
    }
    removed = removed_states | removed_processes | removed_parameters
    modeled_auxiliaries = set(context.auxiliaries) & {
        item.name for item in candidate.states if item.name not in removed_states
    }
    redundant_auxiliary_mappings = {
        item.channel
        for item in candidate.observation_mappings
        if item.channel in context.auxiliaries
        and item.channel not in modeled_auxiliaries
    }
    numeric_qualitative_constraints = {
        item.subject
        for item in candidate.constraints
        if item.kind is not ConstraintKind.BOUNDED and item.bounds is not None
    }
    retained_declarations = (
        declared_names - removed
    ) | set(context.forcing_channels) | {
        item.channel
        for item in candidate.observation_mappings
        if item.channel not in protected
        and item.channel not in redundant_auxiliary_mappings
    }
    unknown_constraint_subjects = {
        item.subject
        for item in candidate.constraints
        if item.subject not in retained_declarations
    }
    if (
        not removed
        and not redundant_auxiliary_mappings
        and not numeric_qualitative_constraints
        and not unknown_constraint_subjects
    ):
        return candidate, tuple(initial_repairs)
    payload = candidate.model_dump(mode="json")
    payload["states"] = [
        item for item in payload["states"] if item["name"] not in removed_states
    ]
    payload["processes"] = [
        item for item in payload["processes"] if item["name"] not in removed_processes
    ]
    payload["parameters"] = [
        item for item in payload["parameters"] if item["name"] not in removed_parameters
    ]
    payload["state_equations"] = [
        item
        for item in payload["state_equations"]
        if item["state"] not in removed_states
    ]
    payload["initial_conditions"] = [
        item
        for item in payload["initial_conditions"]
        if item["state"] not in removed_states
    ]
    payload["observation_mappings"] = [
        item
        for item in payload["observation_mappings"]
        if item["channel"] not in protected
        and item["channel"] not in redundant_auxiliary_mappings
    ]
    payload["constraints"] = [
        item
        for item in payload["constraints"]
        if item["subject"] not in unknown_constraint_subjects
    ]
    for constraint in payload["constraints"]:
        if (
            constraint["kind"] != ConstraintKind.BOUNDED.value
            and constraint["bounds"] is not None
        ):
            constraint["bounds"] = None
    repairs = list(initial_repairs)
    repairs.extend(
        f"used supplied forcing instead of modeled declaration: {name}"
        for name in sorted(
            removed - undifferentiated_auxiliaries - unused_processes
        )
    )
    repairs.extend(
        f"used supplied auxiliary because no derivative was declared: {name}"
        for name in sorted(undifferentiated_auxiliaries)
    )
    repairs.extend(
        f"removed unreferenced algebraic process: {name}"
        for name in sorted(unused_processes - protected)
    )
    repairs.extend(
        f"removed redundant supplied-auxiliary observation mapping: {name}"
        for name in sorted(redundant_auxiliary_mappings)
    )
    repairs.extend(
        f"removed invented numeric bounds from qualitative constraint: {name}"
        for name in sorted(numeric_qualitative_constraints)
    )
    repairs.extend(
        f"removed constraint on undeclared subject: {name}"
        for name in sorted(unknown_constraint_subjects)
    )
    return CandidateModel.model_validate(payload), tuple(repairs)


class CandidateValidator:
    """Validate expression syntax, semantics, dependencies, and domains."""

    def __init__(self, parser: RestrictedParser | None = None) -> None:
        self._parser = parser or RestrictedParser()

    def validate(
        self,
        candidate: CandidateModel,
        context: ValidationContext,
    ) -> ValidatedCandidate:
        """Return a compiled-ready candidate or all stable diagnostics."""
        diagnostics: list[ValidationDiagnostic] = []
        state_names = {item.name for item in candidate.states}
        process_names = {item.name for item in candidate.processes}
        parameter_names = {item.name for item in candidate.parameters}
        forcing_names = set(context.forcing_channels)
        auxiliary_names = set(context.auxiliaries)
        protected_forcing_names = set(context.external_inputs) | set(
            context.fixed_covariates
        )
        reserved = set(APPROVED_FUNCTION_ARITY) | {context.time_symbol}

        self._validate_declaration_collisions(
            state_names,
            process_names,
            parameter_names,
            auxiliary_names,
            protected_forcing_names,
            reserved,
            diagnostics,
        )
        self._validate_parameter_scopes(candidate, diagnostics)
        self._validate_state_equation_closure(candidate, diagnostics)
        self._validate_constraint_subjects(
            candidate,
            state_names | process_names | parameter_names | forcing_names,
            diagnostics,
        )
        self._validate_constraint_consistency(candidate, diagnostics)
        self._validate_observation_channels(candidate, context, diagnostics)

        process_expressions: dict[str, ParsedExpression] = {}
        for process in candidate.processes:
            parsed = self._parse(
                process.expression,
                f"process:{process.name}",
                diagnostics,
            )
            if parsed is not None:
                process_expressions[process.name] = parsed

        equation_expressions: dict[str, ParsedExpression] = {}
        for equation in candidate.state_equations:
            parsed = self._parse(
                equation.rhs,
                f"equation:{equation.state}",
                diagnostics,
            )
            if parsed is not None:
                equation_expressions[equation.state] = parsed

        observation_expressions: dict[str, ParsedExpression] = {}
        for observation in candidate.observation_mappings:
            parsed = self._parse(
                observation.expression,
                f"observation:{observation.channel}",
                diagnostics,
            )
            if parsed is not None:
                observation_expressions[observation.channel] = parsed

        initial_condition_expressions: dict[str, ParsedExpression] = {}
        observed_states = self._identity_observed_states(candidate, context)
        initial_states = {item.state for item in candidate.initial_conditions}
        causal_derivative_initials: dict[str, str] = {}
        for state in sorted(state_names - initial_states):
            base = state.removesuffix("_rate") if state.endswith("_rate") else ""
            if context.lagged_targets and base in observed_states:
                causal_derivative_initials[state] = base
            else:
                diagnostics.append(
                    ValidationDiagnostic(
                        "MISSING_INITIAL_CONDITION",
                        f"initial_condition:{state}",
                        "dynamic state requires an initialization rule",
                    )
                )
        for state in sorted(initial_states - state_names):
            diagnostics.append(
                ValidationDiagnostic(
                    "UNKNOWN_INITIAL_CONDITION_STATE",
                    f"initial_condition:{state}",
                    "initialization target is not a declared state",
                )
            )
        for initial in candidate.initial_conditions:
            if (
                initial.fixed_value is None
                and initial.expression is None
                and initial.initialization_range is None
            ):
                diagnostics.append(
                    ValidationDiagnostic(
                        "MISSING_INITIALIZATION_MODE",
                        f"initial_condition:{initial.state}",
                        "dynamic state initialization requires a fixed value, "
                        "causal expression, or supported fitted range",
                    )
                )
            if initial.expression is None:
                continue
            parsed = self._parse(
                initial.expression,
                f"initial_condition:{initial.state}",
                diagnostics,
            )
            if parsed is None:
                continue
            initial_condition_expressions[initial.state] = parsed
            allowed_initial_symbols = forcing_names | {context.time_symbol}
            if initial.state in observed_states:
                allowed_initial_symbols.add(initial.state)
            unknown = parsed.symbols - allowed_initial_symbols
            for symbol in sorted(unknown):
                diagnostics.append(
                    ValidationDiagnostic(
                        "INVALID_INITIALIZATION_SYMBOL",
                        f"initial_condition:{initial.state}",
                        "initialization expressions may use only known initial "
                        f"observations, inputs, covariates, and time: {symbol}",
                    )
                )

        if context.lagged_targets:
            for initial in candidate.initial_conditions:
                if (
                    initial.state not in observed_states
                    and initial.initialization_range is not None
                ):
                    diagnostics.append(
                        ValidationDiagnostic(
                            "LATENT_INITIALIZATION_NOT_CAUSAL",
                            f"initial_condition:{initial.state}",
                            "latent state requires fixed_value or an analytic "
                            "expression of known initial variables",
                        )
                    )

        all_expressions = (
            tuple(
                (f"process:{name}", expression)
                for name, expression in process_expressions.items()
            )
            + tuple(
                (f"equation:{name}", expression)
                for name, expression in equation_expressions.items()
            )
            + tuple(
                (f"observation:{name}", expression)
                for name, expression in observation_expressions.items()
            )
        )
        allowed_symbols = (
            state_names
            | process_names
            | parameter_names
            | forcing_names
            | {context.time_symbol}
        )
        for location, expression in all_expressions:
            self._validate_symbols(
                expression,
                location,
                allowed_symbols,
                state_names,
                process_names,
                parameter_names,
                context,
                diagnostics,
            )

        process_order = self._process_order(
            candidate,
            process_expressions,
            diagnostics,
        )
        self._validate_parameter_usage(
            candidate,
            all_expressions,
            diagnostics,
        )
        if len(process_order) == len(process_expressions):
            self._validate_domains(
                candidate,
                context,
                process_expressions,
                equation_expressions,
                observation_expressions,
                initial_condition_expressions,
                process_order,
                diagnostics,
            )

        warnings = tuple(
            item for item in diagnostics if item.code == "DOMAIN_DIVISION_ZERO"
        )
        diagnostics[:] = [
            item for item in diagnostics if item.code != "DOMAIN_DIVISION_ZERO"
        ]
        if diagnostics:
            raise ModelValidationError(tuple(diagnostics))
        forcing_symbols = frozenset(
            symbol
            for _, expression in all_expressions
            for symbol in expression.symbols
            if symbol in forcing_names and symbol not in state_names
        )
        return ValidatedCandidate(
            candidate=candidate,
            context=context,
            process_expressions=MappingProxyType(process_expressions),
            equation_expressions=MappingProxyType(equation_expressions),
            observation_expressions=MappingProxyType(observation_expressions),
            initial_condition_expressions=MappingProxyType(
                initial_condition_expressions
            ),
            causal_derivative_initials=MappingProxyType(
                causal_derivative_initials
            ),
            process_order=process_order,
            forcing_symbols=forcing_symbols,
            warnings=tuple(
                ValidationDiagnostic(
                    "DOMAIN_DIVISION_GUARDED",
                    item.location,
                    f"{item.message}; runtime uses a sign-preserving epsilon guard",
                )
                for item in warnings
            ),
        )

    def _parse(
        self,
        source: str,
        location: str,
        diagnostics: list[ValidationDiagnostic],
    ) -> ParsedExpression | None:
        try:
            return self._parser.parse(source, location=location)
        except ModelValidationError as exc:
            diagnostics.extend(exc.diagnostics)
            return None

    @staticmethod
    def _validate_constraint_subjects(
        candidate: CandidateModel,
        declared_names: set[str],
        diagnostics: list[ValidationDiagnostic],
    ) -> None:
        """Allow context-declared forcing and reject genuinely unknown subjects."""
        allowed = declared_names | {
            mapping.channel for mapping in candidate.observation_mappings
        }
        for constraint in candidate.constraints:
            if constraint.subject not in allowed:
                diagnostics.append(
                    ValidationDiagnostic(
                        "UNKNOWN_CONSTRAINT_SUBJECT",
                        f"constraint:{constraint.subject}",
                        "constraint subject is not a declared model or forcing symbol",
                    )
                )

    @staticmethod
    def _validate_declaration_collisions(
        state_names: set[str],
        process_names: set[str],
        parameter_names: set[str],
        auxiliary_names: set[str],
        protected_forcing_names: set[str],
        reserved: set[str],
        diagnostics: list[ValidationDiagnostic],
    ) -> None:
        namespace_pairs = (
            ("state/process", state_names & process_names),
            ("state/parameter", state_names & parameter_names),
            ("process/parameter", process_names & parameter_names),
        )
        for namespaces, overlap in namespace_pairs:
            for name in sorted(overlap):
                diagnostics.append(
                    ValidationDiagnostic(
                        "DECLARATION_NAME_COLLISION",
                        f"declaration:{name}",
                        f"symbol is declared in both {namespaces} namespaces",
                    )
                )
        model_names = state_names | process_names | parameter_names
        forbidden_shadowing = (
            (process_names | parameter_names) & auxiliary_names
        ) | (model_names & protected_forcing_names)
        for name in sorted(forbidden_shadowing):
            diagnostics.append(
                ValidationDiagnostic(
                    "CHANNEL_NAME_COLLISION",
                    f"declaration:{name}",
                    "model symbol shadows an available forcing channel",
                )
            )
        for name in sorted(model_names & reserved):
            diagnostics.append(
                ValidationDiagnostic(
                    "RESERVED_NAME",
                    f"declaration:{name}",
                    "model symbol shadows time or an approved function",
                )
            )

    @staticmethod
    def _validate_parameter_scopes(
        candidate: CandidateModel,
        diagnostics: list[ValidationDiagnostic],
    ) -> None:
        for parameter in candidate.parameters:
            if parameter.scope not in (
                ParameterScope.GLOBAL,
                ParameterScope.TRAJECTORY_SPECIFIC,
            ):
                diagnostics.append(
                    ValidationDiagnostic(
                        "INVALID_PARAMETER_SCOPE",
                        f"parameter:{parameter.name}",
                        f"unsupported parameter scope {parameter.scope}",
                    )
                )

    @staticmethod
    def _validate_state_equation_closure(
        candidate: CandidateModel,
        diagnostics: list[ValidationDiagnostic],
    ) -> None:
        states = {item.name for item in candidate.states}
        equations = [item.state for item in candidate.state_equations]
        equation_set = set(equations)
        for state in sorted(states - equation_set):
            diagnostics.append(
                ValidationDiagnostic(
                    "MISSING_STATE_EQUATION",
                    f"equation:{state}",
                    "declared state has no governing equation",
                )
            )
        for state in sorted(equation_set - states):
            diagnostics.append(
                ValidationDiagnostic(
                    "UNKNOWN_EQUATION_STATE",
                    f"equation:{state}",
                    "equation target is not a declared state",
                )
            )
        duplicates = sorted(
            {state for state in equations if equations.count(state) > 1}
        )
        for state in duplicates:
            diagnostics.append(
                ValidationDiagnostic(
                    "DUPLICATE_STATE_EQUATION",
                    f"equation:{state}",
                    "state has more than one governing equation",
                )
            )

    @staticmethod
    def _validate_constraint_consistency(
        candidate: CandidateModel,
        diagnostics: list[ValidationDiagnostic],
    ) -> None:
        state_names = {item.name for item in candidate.states}
        parameter_by_name = {
            item.name: item for item in candidate.parameters
        }
        initial_by_state = {
            item.state: item.initialization_range
            for item in candidate.initial_conditions
        }
        fixed_initial_by_state = {
            item.state: item.fixed_value
            for item in candidate.initial_conditions
            if item.fixed_value is not None
        }
        for state_name in sorted(state_names):
            interval = CandidateValidator._state_interval(candidate, state_name)
            if interval.lower > interval.upper:
                diagnostics.append(
                    ValidationDiagnostic(
                        "CONSTRAINT_CONFLICT",
                        f"constraint:{state_name}",
                        "state constraints define an empty interval",
                    )
                )
                continue
            initialization = initial_by_state.get(state_name)
            if initialization and (
                initialization.lower < interval.lower
                or initialization.upper > interval.upper
            ):
                diagnostics.append(
                    ValidationDiagnostic(
                        "INITIAL_CONSTRAINT_CONFLICT",
                        f"initial_condition:{state_name}",
                        "initialization range violates state constraints",
                    )
                )
            fixed_initial = fixed_initial_by_state.get(state_name)
            if fixed_initial is not None and not (
                interval.lower <= fixed_initial <= interval.upper
            ):
                diagnostics.append(
                    ValidationDiagnostic(
                        "INITIAL_CONSTRAINT_CONFLICT",
                        f"initial_condition:{state_name}",
                        "fixed initial value violates state constraints",
                    )
                )

        for constraint in candidate.constraints:
            parameter = parameter_by_name.get(constraint.subject)
            if parameter is None:
                continue
            lower = parameter.bounds.lower
            upper = parameter.bounds.upper
            conflict = False
            if constraint.kind is ConstraintKind.NONNEGATIVE:
                conflict = lower < 0.0
            elif constraint.kind is ConstraintKind.POSITIVE:
                conflict = lower <= 0.0
            if constraint.bounds:
                conflict = conflict or (
                    lower < constraint.bounds.lower
                    or upper > constraint.bounds.upper
                )
            if conflict:
                diagnostics.append(
                    ValidationDiagnostic(
                        "PARAMETER_CONSTRAINT_CONFLICT",
                        f"constraint:{parameter.name}",
                        "parameter bounds violate the declared constraint",
                    )
                )

    @staticmethod
    def _validate_observation_channels(
        candidate: CandidateModel,
        context: ValidationContext,
        diagnostics: list[ValidationDiagnostic],
    ) -> None:
        mapped = {item.channel for item in candidate.observation_mappings}
        expected = set(context.targets)
        modeled_auxiliaries = set(context.auxiliaries) & {
            state.name for state in candidate.states
        }
        for channel in sorted(expected - mapped):
            diagnostics.append(
                ValidationDiagnostic(
                    "MISSING_OBSERVATION_MAPPING",
                    f"observation:{channel}",
                    "target has no observation mapping",
                )
            )
        for channel in sorted(mapped - expected - modeled_auxiliaries):
            diagnostics.append(
                ValidationDiagnostic(
                    "UNEXPECTED_OBSERVATION_MAPPING",
                    f"observation:{channel}",
                    "mapping channel is not a target",
                )
            )

    @staticmethod
    def _identity_observed_states(
        candidate: CandidateModel,
        context: ValidationContext,
    ) -> set[str]:
        """Return states directly tied to available measured channels."""
        available = set(context.targets) | set(context.auxiliaries)
        observed = {
            state.name for state in candidate.states if state.name in available
        }
        for mapping in candidate.observation_mappings:
            try:
                parsed = RestrictedParser().parse(
                    mapping.expression,
                    location=f"observation:{mapping.channel}",
                )
            except ModelValidationError:
                continue
            if (
                mapping.channel in available
                and isinstance(parsed.tree.body, ast.Name)
                and parsed.tree.body.id in {state.name for state in candidate.states}
            ):
                observed.add(parsed.tree.body.id)
        return observed

    @staticmethod
    def _validate_symbols(
        expression: ParsedExpression,
        location: str,
        allowed_symbols: set[str],
        state_names: set[str],
        process_names: set[str],
        parameter_names: set[str],
        context: ValidationContext,
        diagnostics: list[ValidationDiagnostic],
    ) -> None:
        generated_names = state_names | process_names | parameter_names
        for symbol in sorted(expression.symbols - allowed_symbols):
            if symbol in context.targets and symbol not in generated_names:
                code = "TARGET_LEAKAGE"
                message = "target channel is used as an exogenous symbol"
            elif symbol in context.unavailable_observed_channels:
                code = "UNAVAILABLE_OBSERVED_CHANNEL"
                message = "observed channel is unavailable in this tier"
            else:
                code = "UNDEFINED_SYMBOL"
                message = "symbol is not declared or supplied"
            diagnostics.append(
                ValidationDiagnostic(code, location, f"{message}: {symbol}")
            )

    @staticmethod
    def _process_order(
        candidate: CandidateModel,
        expressions: dict[str, ParsedExpression],
        diagnostics: list[ValidationDiagnostic],
    ) -> tuple[str, ...]:
        process_names = set(expressions)
        dependencies = {
            name: expressions[name].symbols & process_names for name in expressions
        }
        order: list[str] = []
        states: dict[str, int] = {}

        def visit(name: str, path: tuple[str, ...]) -> None:
            state = states.get(name, 0)
            if state == 2:
                return
            if state == 1:
                cycle_start = path.index(name) if name in path else 0
                cycle = (*path[cycle_start:], name)
                diagnostics.append(
                    ValidationDiagnostic(
                        "ALGEBRAIC_CYCLE",
                        f"process:{name}",
                        f"cyclic process dependency: {' -> '.join(cycle)}",
                    )
                )
                return
            states[name] = 1
            for dependency in sorted(dependencies[name]):
                visit(dependency, (*path, name))
            states[name] = 2
            order.append(name)

        declared_order = [
            item.name for item in candidate.processes if item.name in expressions
        ]
        for name in declared_order:
            visit(name, ())
        if any(item.code == "ALGEBRAIC_CYCLE" for item in diagnostics):
            return ()
        return tuple(order)

    @staticmethod
    def _validate_parameter_usage(
        candidate: CandidateModel,
        expressions: tuple[tuple[str, ParsedExpression], ...],
        diagnostics: list[ValidationDiagnostic],
    ) -> None:
        referenced = {
            symbol for _, expression in expressions for symbol in expression.symbols
        }
        for parameter in candidate.parameters:
            if parameter.name not in referenced:
                diagnostics.append(
                    ValidationDiagnostic(
                        "UNUSED_PARAMETER",
                        f"parameter:{parameter.name}",
                        "declared parameter is not referenced",
                    )
                )

    @staticmethod
    def _validate_domains(
        candidate: CandidateModel,
        context: ValidationContext,
        processes: dict[str, ParsedExpression],
        equations: dict[str, ParsedExpression],
        observations: dict[str, ParsedExpression],
        initial_conditions: dict[str, ParsedExpression],
        process_order: tuple[str, ...],
        diagnostics: list[ValidationDiagnostic],
    ) -> None:
        intervals: dict[str, Interval] = {
            context.time_symbol: Interval(-float("inf"), float("inf")),
            **{
                name: (
                    Interval(*context.forcing_bounds[name])
                    if name in context.forcing_bounds
                    else UNKNOWN_INTERVAL
                )
                for name in context.forcing_channels
            },
            **{
                parameter.name: Interval(
                    parameter.bounds.lower, parameter.bounds.upper
                )
                for parameter in candidate.parameters
            },
            **{
                state.name: CandidateValidator._state_interval(candidate, state.name)
                for state in candidate.states
            },
        }
        for name in process_order:
            intervals[name] = analyze_interval(
                processes[name].tree,
                intervals,
                location=f"process:{name}",
                diagnostics=diagnostics,
            )
        for name, expression in equations.items():
            analyze_interval(
                expression.tree,
                intervals,
                location=f"equation:{name}",
                diagnostics=diagnostics,
            )
        for name, expression in observations.items():
            analyze_interval(
                expression.tree,
                intervals,
                location=f"observation:{name}",
                diagnostics=diagnostics,
            )
        for name, expression in initial_conditions.items():
            analyze_interval(
                expression.tree,
                intervals,
                location=f"initial_condition:{name}",
                diagnostics=diagnostics,
            )

    @staticmethod
    def _state_interval(candidate: CandidateModel, state_name: str) -> Interval:
        interval = UNKNOWN_INTERVAL
        for constraint in candidate.constraints:
            if constraint.subject != state_name:
                continue
            if constraint.kind is ConstraintKind.BOUNDED and constraint.bounds:
                interval = Interval(
                    constraint.bounds.lower,
                    constraint.bounds.upper,
                )
            elif constraint.kind is ConstraintKind.NONNEGATIVE:
                interval = Interval(max(0.0, interval.lower), interval.upper)
            elif constraint.kind is ConstraintKind.POSITIVE:
                interval = Interval(
                    max(float.fromhex("0x0.0000000000001p-1022"), interval.lower),
                    interval.upper,
                )
            if constraint.bounds:
                interval = Interval(
                    max(constraint.bounds.lower, interval.lower),
                    min(constraint.bounds.upper, interval.upper),
                )
        return interval
