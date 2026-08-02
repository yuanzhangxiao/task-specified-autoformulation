"""Selection, final fitting, and exactly-once testing for baselines."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from autoformalism.baselines.core import (
    evaluate_equations,
    feature_names,
    persistence_metrics,
    regression_table,
    target_scales,
)
from autoformalism.baselines.models import BaselineConfig, BaselineResult
from autoformalism.baselines.pysr import fit_pysr
from autoformalism.baselines.sindy import fit_sindy
from autoformalism.data import DatasetSplit, DevelopmentDataset, SplitName, Trajectory
from autoformalism.expressions import (
    ModelValidationError,
    RestrictedParser,
    RuntimeExpressionError,
    ValidationContext,
)
from autoformalism.llm import LLMClient


def run_baseline(
    config: BaselineConfig,
    dataset: DevelopmentDataset,
    test_loader: Callable[[], DatasetSplit],
    context: ValidationContext,
    *,
    llm_client: LLMClient | None = None,
    proposer_prompt: str = "",
) -> BaselineResult:
    """Run one baseline without opening test until selection is frozen."""
    scales = target_scales(dataset.train, context.targets)
    if config.method == "persistence":
        train_metrics = persistence_metrics(dataset.train, scales)
        validation_metrics = persistence_metrics(dataset.validation, scales)
        test_metrics = persistence_metrics(test_loader(), scales)
        return _result(
            config,
            dataset,
            {target: target for target in context.targets},
            {},
            train_metrics.normalized_mse,
            validation_metrics.normalized_mse,
            test_metrics,
        )

    names = feature_names(context)
    extra_train: Mapping[str, Callable[[Trajectory, int], float]] = {}
    extra_validation: Mapping[str, Callable[[Trajectory, int], float]] = {}
    proposed_expressions: dict[str, str] = {}
    if config.method == "llm_feature_sindy":
        if llm_client is None:
            raise ValueError("llm_feature_sindy requires one configured LLM client")
        proposed_expressions = _propose_features(
            llm_client, proposer_prompt, context
        )
        extra_train = _feature_functions(proposed_expressions, context)
        extra_validation = extra_train

    x_train, y_train, expanded_names = regression_table(
        dataset.train, names, context.targets, extra_train
    )
    x_validation, _, _ = regression_table(
        dataset.validation, names, context.targets, extra_validation
    )
    del x_validation  # rollout MSE, not derivative loss, selects structures.
    equation_feature_names = tuple(
        (
            f"({proposed_expressions[name]})"
            if name in proposed_expressions
            else name
        )
        for name in expanded_names
    )

    if config.method in {"sindy", "llm_feature_sindy"}:
        selected_threshold, equations, train_mse, validation_mse = _select_sindy(
            config,
            dataset,
            context,
            scales,
            x_train,
            y_train,
            equation_feature_names,
        )
        combined = _combine(dataset.train, dataset.validation)
        combined_extras = _feature_functions(proposed_expressions, context)
        x_combined, y_combined, combined_names = regression_table(
            combined, names, context.targets, combined_extras
        )
        combined_names = tuple(
            (
                f"({proposed_expressions[name]})"
                if name in proposed_expressions
                else name
            )
            for name in combined_names
        )
        equations = fit_sindy(
            x_combined,
            y_combined,
            combined_names,
            context.targets,
            threshold=selected_threshold,
        ).equations
        test_metrics = evaluate_equations(
            equations,
            context,
            test_loader(),
            scales,
            identifier=f"{config.method}_test",
        )
        return _result(
            config,
            dataset,
            equations,
            {
                "threshold": selected_threshold,
                "proposed_feature_count": len(proposed_expressions),
            },
            train_mse,
            validation_mse,
            test_metrics,
        )

    if config.method == "pysr":
        candidates, metadata = fit_pysr(
            x_train,
            y_train,
            expanded_names,
            context.targets,
            iterations=config.pysr_iterations,
            seed=config.seed,
            maximum_expression_size=config.maximum_expression_size,
            timeout_seconds=config.wall_timeout_seconds / len(context.targets),
        )
        equations = _select_pysr(candidates, dataset, context, scales)
        train_metrics = evaluate_equations(
            equations, context, dataset.train, scales, identifier="pysr_train"
        )
        validation_metrics = evaluate_equations(
            equations,
            context,
            dataset.validation,
            scales,
            identifier="pysr_validation",
        )
        test_metrics = evaluate_equations(
            equations, context, test_loader(), scales, identifier="pysr_test"
        )
        return _result(
            config,
            dataset,
            equations,
            {"iterations": config.pysr_iterations, "metadata": str(metadata)},
            train_metrics.normalized_mse,
            validation_metrics.normalized_mse,
            test_metrics,
        )
    raise ValueError(f"method {config.method} requires its dedicated adapter")


def _select_sindy(config, dataset, context, scales, x, y, names):
    outcomes = []
    failures = []
    for threshold in config.sindy_thresholds:
        fit = fit_sindy(x, y, names, context.targets, threshold=threshold)
        try:
            train = evaluate_equations(
                fit.equations, context, dataset.train, scales,
                identifier=f"sindy_train_{threshold}",
            )
            validation = evaluate_equations(
                fit.equations, context, dataset.validation, scales,
                identifier=f"sindy_validation_{threshold}",
            )
        except (
            ArithmeticError,
            ModelValidationError,
            RuntimeError,
            RuntimeExpressionError,
            TypeError,
            ValueError,
        ) as exc:
            failures.append(f"threshold {threshold}: {type(exc).__name__}: {exc}")
            continue
        outcomes.append((validation.normalized_mse, threshold, fit.equations,
                         train.normalized_mse))
    if not outcomes:
        raise ValueError(
            "no SINDy support passed safe validation rollout: "
            + "; ".join(failures)[-4000:]
        )
    validation_mse, threshold, equations, train_mse = min(
        outcomes, key=lambda item: (item[0], -item[1])
    )
    return threshold, equations, train_mse, validation_mse


def _select_pysr(candidates, dataset, context, scales):
    """Select PySR's Pareto candidate by validation one-step rollout MSE."""
    if len(context.targets) != 1:
        raise ValueError(
            "PySR validation selection currently requires one target output"
        )
    target = context.targets[0]
    outcomes = []
    for index, expression in enumerate(candidates[target]):
        equations = {target: expression}
        try:
            metrics = evaluate_equations(
                equations,
                context,
                dataset.validation,
                scales,
                identifier=f"pysr_validation_{index}",
            )
        except (
            ArithmeticError,
            ModelValidationError,
            RuntimeError,
            RuntimeExpressionError,
            TypeError,
            ValueError,
        ):
            continue
        if not metrics.failed_trajectories:
            outcomes.append((metrics.normalized_mse, expression))
    if not outcomes:
        raise ValueError("no PySR expression passed safe validation rollout")
    return {target: min(outcomes, key=lambda item: item[0])[1]}


def _combine(train: DatasetSplit, validation: DatasetSplit) -> DatasetSplit:
    return DatasetSplit(
        SplitName.TRAIN,
        (*train.trajectories, *validation.trajectories),
        f"{train.fingerprint}+{validation.fingerprint}",
    )


def _propose_features(
    client: LLMClient, task_prompt: str, context: ValidationContext
) -> dict[str, str]:
    allowed = (*context.targets, *context.auxiliaries, *context.external_inputs,
               *context.fixed_covariates, context.time_symbol)
    prompt = (
        f"{task_prompt}\n\nPropose one ProposerCandidateV2 using exactly the "
        f"observed target states {context.targets}, each with rhs `0`. Put 1 to 4 "
        "useful causal algebraic feature transformations in `algebraics`. Every "
        "feature must be independently computable directly from the allowed symbols; "
        "do not reference another proposed algebraic and do not copy a single input "
        "symbol unchanged. Declare "
        "no latent states and no parameters. Algebraic expressions may use only: "
        f"{', '.join(allowed)}. This is a single non-iterative feature-design call."
    )
    candidate = client.propose(
        system_prompt=prompt, user_prompt="Design features."
    ).parsed
    parser = RestrictedParser()
    allowed_set = set(allowed)
    features: dict[str, str] = {}
    seen_expressions: set[str] = set()
    for process in candidate.processes:
        parsed = parser.parse(process.expression, location=f"feature:{process.name}")
        canonical = "".join(process.expression.split())
        if (
            set(parsed.symbols) <= allowed_set
            and canonical not in allowed_set
            and canonical not in seen_expressions
        ):
            features[process.name] = process.expression
            seen_expressions.add(canonical)
    if not features:
        raise ValueError("LLM did not propose any independently computable features")
    return features


def _feature_functions(expressions, context):
    from autoformalism.baselines.core import _channel_value
    from autoformalism.expressions.compiler import _evaluate

    parser = RestrictedParser()
    parsed = {name: parser.parse(expr, location=f"feature:{name}")
              for name, expr in expressions.items()}
    def make_function(expression):
        def evaluate(trajectory, index):
            environment = {
                name: _channel_value(trajectory, name, index)
                for name in feature_names(context)
            }
            environment[context.time_symbol] = float(trajectory.time[index])
            return float(_evaluate(expression, environment))
        return evaluate
    return {name: make_function(expression) for name, expression in parsed.items()}


def _result(config, dataset, equations, hyperparameters, train_mse,
            validation_mse, test_metrics):
    return BaselineResult(
        method=config.method,
        benchmark_id=dataset.benchmark_id,
        tier=dataset.tier,
        seed=config.seed,
        equations=dict(equations),
        selected_hyperparameters=dict(hyperparameters),
        training_normalized_mse=train_mse,
        validation_normalized_mse=validation_mse,
        test_normalized_mse=test_metrics.normalized_mse,
        test_per_target_normalized_mse=dict(test_metrics.per_target_normalized_mse),
    )
