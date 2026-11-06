"""Bounded fitting routes for the remaining unchanged construction parents.

The campaign authenticates inputs and journals each qualification, search and
independent replay separately. These helpers preserve the certified coordinate
maps and declared domains. Deterministic starts are numerical probes, never new
scientific constraints, and a failed finite set cannot prove model infeasibility.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Literal

import numpy as np

from autoformalism.fitting import prefix_continuation as coordinates
from autoformalism.fitting import prefix_initialization as initialization
from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import prefix_portfolio_nonlinear_reduction as nonlinear
from autoformalism.fitting import prefix_portfolio_polish as affine
from autoformalism.fitting import prefix_portfolio_residual_search as residual
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting.compact_search_checkpoint import SearchCheckpointPolicy
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "remaining-portfolio-numerical-routes-1"
Route = Literal["affine", "nonlinear"]
Checkpoint = Callable[[dict], None]


def _route(route: Route, oracle) -> None:
    """Reject a route inconsistent with the actual executable certificate."""
    expected = {"affine": "certified_affine", "nonlinear": "original_ODE"}
    if route not in expected or oracle.portfolio_audit.get("route") != expected[route]:
        raise ValueError("requested fitting route differs from equation certificate")
    if route == "nonlinear" and not isinstance(oracle.oracle, nonlinear.ReducedOracle):
        raise ValueError("nonlinear route requires the certified nonlinear chart")


def setup(
    problem: PrefixFitProblem,
    training: PublicSplit,
    numerical: initialization.NumericalPolicy,
    anchor: Mapping[str, float] | None,
    route: Route,
):
    """Compile one unchanged parent and expose its exact reduced starting point.

    An absent anchor means the supplied constructor guesses, not a fitted parent.
    Compiling a chart performs no integration and does not qualify its start.
    """
    if training.name != "train":
        raise ValueError("remaining portfolio accepts training only")
    if route not in ("affine", "nonlinear"):
        raise ValueError("unknown remaining-portfolio route")
    if anchor is None:
        anchor = initialization.TrainingOracle(problem, training, numerical).start
    anchor = deepcopy(dict(anchor))
    if route == "affine":
        original, oracle = affine.make_oracle(problem, training, numerical, anchor)
        roots.RootProfile(oracle)  # Executable affine-in-initial-roots certificate.
    else:
        original, oracle = nonlinear.make_oracle(problem, training, anchor, numerical)
    _route(route, oracle)
    return original, oracle, coordinates.to_effective(original, oracle, anchor)


def _legal(oracle, parameters: Mapping[str, float]) -> np.ndarray:
    """Check the declared reduced domain and its exact original-model lift."""
    point = oracle.vector(parameters)
    lifted = oracle.reduction.lift(point)
    oracle.original.vector(oracle.original.parameters(lifted))
    if not np.allclose(oracle.reduction.reduce(lifted), point, rtol=1e-13, atol=1e-14):
        raise ValueError("restart does not round-trip through the original domain")
    return point


def _variant(value: float, lower: float, upper: float, unit: float, expand: bool):
    """Scale the anchor and clip only to actual declared bounds, never their span."""
    factor = 4.0 if expand else 0.25
    with np.errstate(over="ignore", invalid="ignore"):
        if lower <= 0 <= upper:
            proposed = factor * value if value else (unit if expand else -unit) * factor
            # A zero on a one-sided boundary needs an inward, nonzero probe.
            if value == 0 and lower == 0:
                proposed = factor * unit
            elif value == 0 and upper == 0:
                proposed = -factor * unit
        elif np.isfinite(lower):
            distance = value - lower
            proposed = lower + factor * (distance if distance > 0 else unit)
        elif np.isfinite(upper):
            distance = upper - value
            proposed = upper - factor * (distance if distance > 0 else unit)
        else:
            raise ValueError("invalid declared restart domain")
        proposed = np.clip(proposed, lower, upper)
    if not np.isfinite(proposed):
        raise ValueError("predetermined restart is not representable")
    return float(proposed)


def deterministic_starts(
    oracle,
    start: Mapping[str, float],
    route: Route,
    *,
    recovery: bool = False,
) -> list[dict]:
    """Freeze warm plus two shared restarts, or warm plus four recovery probes.

    Recovery probes also vary trajectory roots and include two alternating
    shrink/expand patterns. All patterns depend only on the frozen parameter
    vector, declared bounds and coordinate order. They use no trajectory errors,
    held-out observations, scientific name heuristics or private coefficients.
    Duplicate legal vectors are omitted and explicitly named in the warm record.
    """
    _route(route, oracle)
    if type(recovery) is not bool or (recovery and route != "nonlinear"):
        raise ValueError("recovery probes require the nonlinear route")
    point = _legal(oracle, start)
    units = np.asarray(oracle.units, float)
    if units.shape != point.shape or not np.isfinite(units).all() or np.any(units <= 0):
        raise ValueError("finite positive declared coordinate scales required")
    active = [
        i
        for i, name in enumerate(oracle.names)
        if oracle.lower[i] < oracle.upper[i]
        and (recovery or not name.startswith("trajectory:"))
    ]
    labels = ["warm", "shared_shrink", "shared_expand"]
    if recovery:
        labels = ["warm", "all_shrink", "all_expand", "alternating_a", "alternating_b"]
    result, seen, duplicates = [], set(), []
    for label in labels:
        trial = point.copy()
        if label != "warm":
            for ordinal, i in enumerate(active):
                expand = label.endswith("expand") or (
                    label.startswith("alternating")
                    and ordinal % 2 == (0 if label.endswith("a") else 1)
                )
                trial[i] = _variant(
                    point[i], oracle.lower[i], oracle.upper[i], units[i], expand
                )
        parameters = oracle.parameters(trial)
        _legal(oracle, parameters)
        digest = content_sha256(parameters)
        if digest in seen:
            duplicates.append(label)
            continue
        seen.add(digest)
        result.append(
            {
                "label": label,
                "parameters": parameters,
                "parameters_sha256": digest,
                "changed_coordinates": [
                    name for i, name in enumerate(oracle.names) if trial[i] != point[i]
                ],
                "source_parameters_sha256": content_sha256(dict(start)),
                "original_domain_checked": True,
                "artificial_bounds_added": False,
                "feasibility_established": False,
            }
        )
    result[0]["duplicate_patterns_omitted"] = duplicates
    return result


def shared_blocks(oracle, size: int = 3) -> list[list[str]]:
    """Partition active shared coordinates in their certified order."""
    if type(size) is not int or not 1 <= size <= 4:
        raise ValueError("shared block size must be an integer between one and four")
    names = [
        name
        for i, name in enumerate(oracle.names)
        if not name.startswith("trajectory:") and oracle.lower[i] < oracle.upper[i]
    ]
    return [names[i : i + size] for i in range(0, len(names), size)]


def qualify(
    original,
    oracle,
    parameters,
    seconds: float,
    checkpoint: Checkpoint,
    route: Route,
) -> dict:
    """Qualify a fixed seed for derivative-free fitting without gradient claims."""
    if original is not oracle.original:
        raise ValueError("qualification original oracle differs from certified chart")
    _route(route, oracle)
    _legal(oracle, parameters)
    if route == "affine":
        return roots.root_profile_qualification(oracle, parameters, seconds, checkpoint)
    result = residual.qualify_residual(
        original, oracle, parameters, seconds, checkpoint
    )
    return result | {"parameters": deepcopy(dict(parameters))}


def search(
    oracle,
    parameters,
    seconds: float,
    maximum_calls: int,
    checkpoint: Checkpoint,
    route: Route,
    *,
    block: Sequence[str] | None = None,
    persistence_headroom_seconds: float = 900,
) -> dict:
    """Run one reserved block or full search; caller verifies and retains its best.

    Held coordinates are excluded from Powell directions and seed probes. Releasing
    all coordinates restores the full fitting objective, including training roots.
    A block stop cannot certify stationarity of that full problem.
    """
    _route(route, oracle)
    _legal(oracle, parameters)
    if (
        isinstance(persistence_headroom_seconds, (bool, np.bool_))
        or not isinstance(persistence_headroom_seconds, (int, float))
        or not np.isfinite(persistence_headroom_seconds)
        or not 0 <= persistence_headroom_seconds <= 900
    ):
        raise ValueError("persistence headroom must be between zero and 900 seconds")
    fixed = None
    if block is not None:
        if (
            route != "nonlinear"
            or isinstance(block, (str, bytes))
            or not block
            or len(set(block)) != len(block)
            or len(block) > 4
            or set(block) - set(oracle.names)
            or any(name.startswith("trajectory:") for name in block)
        ):
            raise ValueError(
                "nonlinear blocks need one to four unique shared coordinates"
            )
        if any(
            oracle.lower[oracle.names.index(n)] >= oracle.upper[oracle.names.index(n)]
            for n in block
        ):
            raise ValueError("fixed-domain coordinates cannot form an active block")
        fixed = {name: value for name, value in parameters.items() if name not in block}
    policy = SearchCheckpointPolicy(interval_seconds=30)
    if route == "affine":
        return roots.root_profile_search(
            oracle,
            parameters,
            seconds,
            maximum_calls,
            checkpoint,
            failure_diagnostics=True,
            checkpoint_policy=policy,
            overall_wall_seconds=min(seconds + persistence_headroom_seconds, 3600),
        )
    return residual.search(
        oracle,
        parameters,
        seconds,
        maximum_calls,
        checkpoint,
        fixed=fixed,
        checkpoint_policy=policy,
    )


def verify(
    original, oracle, parameters, seconds: float, checkpoint: Checkpoint
) -> dict:
    """Replay the lifted original ODE with independent Radau integration."""
    if original is not oracle.original:
        raise ValueError("verification original oracle differs from certified chart")
    _legal(oracle, parameters)
    return portfolio.verify_endpoint(original, oracle, parameters, seconds, checkpoint)
