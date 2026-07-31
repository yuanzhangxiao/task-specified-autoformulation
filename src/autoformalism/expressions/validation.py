"""Deterministic semantic and domain validation for candidate models."""

from __future__ import annotations

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
from autoformalism.schemas.base import Identifier, StrictSchema

IdentifierTuple = Annotated[tuple[Identifier, ...], Field(default=())]


class ValidationContext(StrictSchema):
    """Benchmark channels available to candidate expressions."""

    targets: IdentifierTuple
    auxiliaries: IdentifierTuple = ()
    external_inputs: IdentifierTuple = ()
    fixed_covariates: IdentifierTuple = ()
    unavailable_observed_channels: IdentifierTuple = ()
    time_symbol: Identifier = "t"

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
        return self

    @property
    def forcing_channels(self) -> frozenset[str]:
        """Return channels that may be supplied over the prediction horizon."""
        return frozenset(
            self.auxiliaries + self.external_inputs + self.fixed_covariates
        )


@dataclass(frozen=True)
class ValidatedCandidate:
    """Candidate plus parsed expressions and a process evaluation order."""

    candidate: CandidateModel
    context: ValidationContext
    process_expressions: MappingProxyType[str, ParsedExpression]
    equation_expressions: MappingProxyType[str, ParsedExpression]
    observation_expressions: MappingProxyType[str, ParsedExpression]
    process_order: tuple[str, ...]
    forcing_symbols: frozenset[str]


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
        reserved = set(APPROVED_FUNCTION_ARITY) | {context.time_symbol}

        self._validate_declaration_collisions(
            state_names,
            process_names,
            parameter_names,
            forcing_names,
            reserved,
            diagnostics,
        )
        self._validate_parameter_scopes(candidate, diagnostics)
        self._validate_state_equation_closure(candidate, diagnostics)
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
                process_order,
                diagnostics,
            )

        if diagnostics:
            raise ModelValidationError(tuple(diagnostics))
        forcing_symbols = frozenset(
            symbol
            for _, expression in all_expressions
            for symbol in expression.symbols
            if symbol in forcing_names
        )
        return ValidatedCandidate(
            candidate=candidate,
            context=context,
            process_expressions=MappingProxyType(process_expressions),
            equation_expressions=MappingProxyType(equation_expressions),
            observation_expressions=MappingProxyType(observation_expressions),
            process_order=process_order,
            forcing_symbols=forcing_symbols,
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
    def _validate_declaration_collisions(
        state_names: set[str],
        process_names: set[str],
        parameter_names: set[str],
        forcing_names: set[str],
        reserved: set[str],
        diagnostics: list[ValidationDiagnostic],
    ) -> None:
        model_names = state_names | process_names | parameter_names
        for name in sorted(model_names & forcing_names):
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
            elif constraint.kind is ConstraintKind.BOUNDED and constraint.bounds:
                conflict = (
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
        for channel in sorted(expected - mapped):
            diagnostics.append(
                ValidationDiagnostic(
                    "MISSING_OBSERVATION_MAPPING",
                    f"observation:{channel}",
                    "target has no observation mapping",
                )
            )
        for channel in sorted(mapped - expected):
            diagnostics.append(
                ValidationDiagnostic(
                    "UNEXPECTED_OBSERVATION_MAPPING",
                    f"observation:{channel}",
                    "mapping channel is not a target",
                )
            )

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
        process_order: tuple[str, ...],
        diagnostics: list[ValidationDiagnostic],
    ) -> None:
        intervals: dict[str, Interval] = {
            context.time_symbol: Interval(-float("inf"), float("inf")),
            **dict.fromkeys(context.forcing_channels, UNKNOWN_INTERVAL),
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
        return interval
