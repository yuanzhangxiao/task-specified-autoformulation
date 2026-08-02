"""Optional PySR derivative-regression adapter."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


class PySRUnavailableError(RuntimeError):
    """Raised when the optional Julia/PySR runtime is unavailable."""


def fit_pysr(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    names: Sequence[str],
    targets: Sequence[str],
    *,
    iterations: int,
    seed: int,
    maximum_expression_size: int,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    """Run the published operator set once per supplied derivative output."""
    try:
        from pysr import PySRRegressor
    except (ImportError, OSError) as exc:  # pragma: no cover - optional runtime
        raise PySRUnavailableError(
            "PySR is not installed. Install the baseline extra with "
            "`pip install -e '.[pysr]'`; Julia packages are initialized by PySR."
        ) from exc
    equations: dict[str, tuple[str, ...]] = {}
    metadata: dict[str, Any] = {}
    pysr_names = [f"af_x{index}" for index in range(len(names))]
    for index, target in enumerate(targets):
        model = PySRRegressor(
            niterations=iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["tanh", "exp", "neg"],
            maxsize=maximum_expression_size,
            random_state=seed + index,
            deterministic=True,
            parallelism="serial",
            model_selection="best",
            verbosity=0,
        )
        model.fit(x, y[:, index], variable_names=pysr_names)
        expressions = tuple(
            _restore_feature_names(
                _restricted_expression(str(expression)), pysr_names, names
            )
            for expression in model.equations_["sympy_format"]
        )
        if not expressions:
            raise ValueError(f"PySR returned no equation for {target}")
        equations[target] = expressions
        metadata[target] = {"equation_count": len(expressions)}
    return equations, metadata


def _restricted_expression(expression: str) -> str:
    """Normalize common SymPy printing into the restricted runtime grammar."""
    return expression.replace("-1*", "-")


def _restore_feature_names(
    expression: str,
    pysr_names: Sequence[str],
    original_names: Sequence[str],
) -> str:
    """Restore exact dataset symbols after fitting with SymPy-safe aliases."""
    replacements = dict(zip(pysr_names, original_names, strict=True))
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(name) for name in pysr_names) + r")\b"
    )
    return pattern.sub(lambda match: replacements[match.group(0)], expression)
