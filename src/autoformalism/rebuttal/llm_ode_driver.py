"""Drive the pinned LLM-ODE search over one Phase-B cell.

Upstream's own ``LlmOde`` wrapper cannot be used: it builds its trajectories by
integrating ``problem['equation']``, the true system, and then differentiates
them. We have observed data and no true system to hand it. ``LlmOde`` is only a
loop over ``LlmOdeEquation`` -- one per state variable -- so this module drives
those directly and reproduces the wrapper's iteration order, refinement
schedule and Pareto-product selection against our own data.

Nothing about the method is reimplemented. The island evolution, the prompt
instructions, the BFGS coefficient fitting and the complexity-versus-error
Pareto frontier all come from the checkout. This module supplies data, appends
the declared task specification, and scores the resulting systems on
development data only.

Two departures from upstream are deliberate and declared in the campaign plan:

* Upstream searches every state variable. Here the auxiliaries, measured
  inputs and fixed covariates are supplied over the horizon and only the
  targets are predicted, so only the targets are searched. The supplied
  channels remain variables every equation may use; searching their own
  equations would spend calls on equations the evaluation never reads.
* Upstream selects by rolling out its own ``System`` and sorting on
  ``mse_train_val``. Its ``System`` integrates every variable as a state, which
  would reinterpret our supplied auxiliaries as free states. Selection here
  uses the same quantity -- development rollout error -- computed by our
  evaluator, which holds the auxiliaries to their observed values. Upstream's
  ``evaluate_system`` also reports test error; it is never called.

Each request is upstream's own: the same call, prompt and output limit. Two
details of how it reaches a shared gateway are not upstream's, and neither
changes the method. Upstream names whichever model the server lists first,
which on its own one-model vLLM is the model it served; the Jetstream2 gateway
lists several, so a campaign names its plan's model. That gateway also replays
a stored answer to a repeated request, where upstream's server samples afresh,
so a campaign against it asks for a fresh answer, and an answer the gateway
marks as replayed is logged as a cache hit.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import openai

from autoformalism.data import DatasetSplit, SplitName, TrainingScaler
from autoformalism.expressions import (
    ModelValidationError,
    ValidationContext,
    baseline_validator,
    compile_candidate,
)
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate
from autoformalism.rebuttal.llm_call_log import CallLog
from autoformalism.rebuttal.llm_ode_campaign import (
    CellArrays,
    select_system,
    specification_block,
)
from autoformalism.rebuttal.llm_ode_upstream import (
    InexpressibleEquation,
    load_upstream,
    specification_prompt,
    to_state_equations,
)
from autoformalism.rebuttal.llm_sr_shim import (
    LITELLM_CACHE_HIT_HEADER,
    LITELLM_NO_CACHE,
)
from autoformalism.rebuttal.phase_b_d3 import accounting as d3_accounting

LOGGER = logging.getLogger(__name__)

#: Iterations after which a total absence of candidates is pathological
#: rather than unlucky. Upstream logs a construction failure and continues,
#: so an environment fault otherwise costs an entire wall-clock allocation.
BARREN_ITERATIONS = 5


class BarrenSearch(RuntimeError):
    """The search produced no candidate at all, well past the point it should."""


#: How long a request may go on failing, from its first failure, before the
#: search is stopped as an infrastructure failure. Upstream's client retries
#: twice within seconds and then raises, which ends the search with whatever
#: it had found; a campaign against a service that comes back passes longer.
DEFAULT_PATIENCE_SECONDS = 15 * 60.0

#: First and longest pause between attempts at a failed request; the pause
#: doubles per attempt, so an outage is retried about once a minute.
BACKOFF_SECONDS = (1.0, 60.0)

#: Refusals that describe the request rather than the server. Retrying cannot
#: change them, so they end the search at once, as upstream would.
REQUEST_ERRORS = frozenset({400, 404, 413, 422})


class EndpointUnavailable(RuntimeError):
    """The endpoint kept failing for longer than the campaign's patience."""


def _pause(seconds: float) -> None:
    """Wait before the next attempt; a seam so tests need not sleep."""
    time.sleep(seconds)


def _clock() -> float:
    """Seconds on a monotonic clock; a seam so tests can move time on."""
    return monotonic()


def _server_failure(exc: Exception) -> bool:
    """Whether a failed call is the server's doing rather than the request's.

    The Jetstream2 service answered 401 and then 503 while it was offline, so
    only the statuses that describe the request itself are excluded.
    """
    if isinstance(exc, openai.APIConnectionError):  # timeouts included
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code not in REQUEST_ERRORS
    return False


#: Upstream assembles systems from the full Cartesian product of the
#: per-variable Pareto frontiers, so the frontiers are used whole. When the
#: product exceeds the campaign's cap the task records that refusal rather
#: than ranking a subset and reporting it as a completed search.


def development_rollout_error(
    equations: dict[str, str],
    context: ValidationContext,
    train: DatasetSplit,
    evaluate: DatasetSplit,
    *,
    seconds: float = 60.0,
) -> float | None:
    """Normalized rollout error on ``evaluate``; ``None`` if any rollout fails.

    The same quantity the frozen evaluator reports, computed here for model
    selection and for the development scores the sealed selection records.
    ``train`` always supplies the normalizing scales. Only TRAIN and VALIDATION
    are accepted, so a selection metric can never be computed against held-out
    data. A system the evaluator's grammar refuses raises
    ``InexpressibleEquation`` naming the refusal, for the caller to count.
    """
    if train.name is not SplitName.TRAIN or evaluate.name not in {
        SplitName.TRAIN,
        SplitName.VALIDATION,
    }:
        raise ValueError("selection requires TRAIN and VALIDATION, never TEST")
    try:
        compiled = compile_candidate(
            equation_candidate("llm_ode", equations, context),
            context,
            validator=baseline_validator(),
        )
    except ModelValidationError as exc:
        # The baseline grammar admits real exponents and the method's whole
        # vocabulary; what it still refuses (a comparison, an undefined name)
        # is counted rather than allowed to end a search already paid for.
        raise InexpressibleEquation(
            "; ".join(f"{name} = {value}" for name, value in equations.items()),
            tuple(sorted({item.code for item in exc.diagnostics})),
        ) from exc
    scaling = TrainingScaler().fit(train).scales
    scales = {
        target: float(scaling[f"target:{target}"].standard_deviation)
        for target in context.targets
    }
    squared: list[float] = []
    for trajectory in evaluate.trajectories:
        simulation = simulate_trajectory(
            compiled,
            trajectory,
            {},
            {},
            FitConfig(),
            deadline=monotonic() + seconds,
            reset_observed_states=False,
        )
        if not simulation.success:
            return None
        errors = [
            np.square(
                (simulation.predictions[target] - trajectory.targets[target])
                / scales[target]
            )
            for target in context.targets
        ]
        if not all(np.isfinite(value).all() for value in errors):
            return None
        squared.append(float(np.mean([np.mean(value) for value in errors])))
    return float(np.mean(squared)) if squared else None


def _stop_reason(response: Any) -> str | None:
    """Why a reply ended: what cut it short if it is incomplete, else its status."""
    details = getattr(response, "incomplete_details", None)
    return getattr(details, "reason", None) or getattr(response, "status", None)


@contextmanager
def _counted_requests(
    llm: Any,
    log: CallLog,
    *,
    patience: float = DEFAULT_PATIENCE_SECONDS,
    bypass_cache: bool = False,
) -> Iterator[None]:
    """Record every provider call, with the usage upstream discards.

    Their client returns `response.output_text` and drops the usage object, so
    a campaign otherwise reports request counts that cannot be compared with
    the token accounting every other method reports. Why each reply ended is
    kept too, so how often the output limit cut one off can be counted. A
    request the server fails is retried until `patience` runs out, and
    `bypass_cache` asks a caching gateway for a fresh answer; the prompt and
    the reply are upstream's own either way.
    """
    original = llm.make_request
    first, longest = BACKOFF_SECONDS
    extra = {"extra_body": LITELLM_NO_CACHE} if bypass_cache else {}

    def make_request(prompt):
        failing_since: float | None = None
        pause = first
        while True:
            try:
                raw = llm.client.responses.with_raw_response.create(
                    model=llm.model_name, input=prompt, **llm.request_kwargs, **extra
                )
                response = raw.parse()
            except Exception as exc:
                log.failure(f"{type(exc).__name__}: {exc}")
                if not _server_failure(exc):
                    raise
                now = _clock()
                failing_since = now if failing_since is None else failing_since
                if now - failing_since >= patience:
                    raise EndpointUnavailable(
                        f"the endpoint failed for {(now - failing_since) / 60:.0f} "
                        f"minutes; last: {type(exc).__name__}: {exc}"
                    ) from exc
                _pause(pause)
                pause = min(longest, 2 * pause)
                continue
            reason = _stop_reason(response)
            log.response(
                response,
                finish_reasons=None if reason is None else [reason],
                cache_hit=raw.headers.get(LITELLM_CACHE_HIT_HEADER) is not None,
            )
            return response.output_text

    llm.make_request = make_request
    try:
        yield
    finally:
        llm.make_request = original


@contextmanager
def _specification_appended(module: Any, specification: str) -> Iterator[None]:
    """Append the declared specification to every prompt upstream builds.

    ``_evolve_islands`` calls the module-level ``generate_prompt`` it imported,
    so the extension is installed there and removed afterwards. Upstream's own
    instructions, in-context examples and output format are untouched.
    """
    original = module.generate_prompt
    if not specification:
        yield
        return

    def extended(examples_str: list[str], n_new: int, n_variables: int) -> Any:
        return specification_prompt(
            original(examples_str, n_new, n_variables), specification
        )

    module.generate_prompt = extended
    try:
        yield
    finally:
        module.generate_prompt = original


def _frontier_equations(searcher: Any) -> tuple[str, ...]:
    """One target's complete complexity/error Pareto frontier, best first."""
    frontier = searcher.get_pareto_frontier()
    if frontier.empty:
        return ()
    ordered = frontier.sort_values(by="mse_val")
    return tuple(str(program.equation) for program in ordered["program"])


def build_searcher(
    *,
    upstream_root: Path,
    base_url: str,
    api_key: str = "EMPTY",
    iterations: int,
    islands: int,
    config: dict | None = None,
    seconds_per_rollout: float = 60.0,
    patience_seconds: float = DEFAULT_PATIENCE_SECONDS,
    model: str | None = None,
    bypass_cache: bool = False,
):
    """Bind the pinned checkout to our data; returns a campaign searcher.

    ``model``, when given, is named in every request in place of the first
    model the server lists; ``bypass_cache`` asks a caching gateway for fresh
    answers. Without them the requests are exactly upstream's.
    """
    modules = load_upstream(upstream_root)
    import llmode.llmode as upstream_module  # imported after load_upstream

    search_config = {"n_islands": islands, **(config or {})}

    def search(
        *,
        train: CellArrays,
        validation: CellArrays,
        targets: tuple[str, ...],
        prompt: str,
        directory: Path,
        development: tuple[DatasetSplit, DatasetSplit],
        context: ValidationContext,
    ) -> dict:
        llm = modules.llm(api_key, base_url)
        if model is not None:
            llm.model_name = model
        log = CallLog(directory / "llm_calls.jsonl")
        indices = [train.channels.index(name) for name in targets]
        searchers = [
            modules.equation_searcher(
                llm=llm,
                t_train=train.time,
                X_train=train.states,
                y_train=train.derivatives[:, index],
                t_val=validation.time,
                X_val=validation.states,
                y_val=validation.derivatives[:, index],
                pareto_level_file=directory / f"{name}.complexity_pf.jsonl",
                config=search_config,
            )
            for name, index in zip(targets, indices, strict=True)
        ]
        # Each searcher is proposing a different derivative, so each gets its
        # own specification; one shared string would tell every island it was
        # working on the same target.
        specifications = [
            specification_block(prompt, train.channels, index) if prompt else ""
            for index in indices
        ]
        error: str | None = None
        started = monotonic()
        try:
            for iteration in range(1, iterations + 1):
                for searcher, specification in zip(
                    searchers, specifications, strict=True
                ):
                    with (
                        _counted_requests(
                            llm,
                            log,
                            patience=patience_seconds,
                            bypass_cache=bypass_cache,
                        ),
                        _specification_appended(upstream_module, specification),
                    ):
                        searcher.step()
                # About ten progress lines whatever the budget, so a short
                # probe reports timing instead of finishing silently.
                # Upstream swallows a failed program construction as a
                # warning, so a systematically broken environment produces a
                # full-length run with permanently empty islands. Stop once
                # that is unambiguous instead of spending the whole wall.
                if iteration == BARREN_ITERATIONS and not any(
                    _frontier_equations(item) for item in searchers
                ):
                    raise BarrenSearch(
                        f"no target produced a single candidate in "
                        f"{BARREN_ITERATIONS} iterations; the search is "
                        "not running, check the task log for repeated "
                        "upstream warnings"
                    )
                if iteration % max(1, iterations // 10) == 0:
                    LOGGER.info(
                        "iteration %d of %d after %.1fs",
                        iteration,
                        iterations,
                        monotonic() - started,
                    )
        except EndpointUnavailable as exc:
            # The search did not run to its budget, so whatever it had found is
            # not this method's result; the task is an infrastructure failure.
            return {
                "status": "endpoint_unavailable",
                "error": str(exc),
                "accounting": {
                    "llm_queries": int(getattr(llm, "n_queries", 0)),
                    **d3_accounting(directory / "llm_calls.jsonl"),
                    "search_seconds": round(monotonic() - started, 1),
                },
            }
        except Exception as exc:  # upstream raised; keep whatever it found
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("search stopped early: %s", error)

        frontiers = [_frontier_equations(searcher) for searcher in searchers]
        calls = int(getattr(llm, "n_queries", 0))
        if any(not frontier for frontier in frontiers):
            return {
                "status": "no_candidates",
                "error": error or "a target produced an empty Pareto frontier",
                "accounting": {"llm_queries": calls, "scored_systems": 0},
            }

        inexpressible = 0
        operators: set[str] = set()
        expressible: dict[tuple[str, ...], tuple[dict[str, str], float]] = {}
        scored = 0

        def score(combination: tuple[str, ...]) -> float | None:
            """Convert, roll out on development data, and count what we cannot read."""
            nonlocal inexpressible, scored
            try:
                equations = to_state_equations(combination, train.channels, targets)
                value = development_rollout_error(
                    equations, context, *development, seconds=seconds_per_rollout
                )
            except InexpressibleEquation as exc:
                inexpressible += 1
                operators.update(exc.functions)
                return None
            scored += 1
            if value is not None:
                expressible[combination] = (equations, value)
            return value

        accounting = {
            "llm_queries": calls,
            # The shared rule, so this campaign is comparable with D3's.
            **d3_accounting(directory / "llm_calls.jsonl"),
            "search_seconds": round(monotonic() - started, 1),
            "frontier_sizes": [len(frontier) for frontier in frontiers],
            "search_error": error,
        }
        try:
            best = select_system(tuple(frontiers), score)
        except ValueError as exc:
            return {
                "status": "product_too_large",
                "error": str(exc),
                "accounting": accounting,
            }
        accounting |= {
            "scored_systems": scored,
            "inexpressible_systems": inexpressible,
            "inexpressible_operators": sorted(operators),
        }
        if best is None:
            return {
                "status": "inexpressible" if scored == 0 else "rollout_failed",
                "error": (
                    f"every candidate used {', '.join(sorted(operators))}, which "
                    "the restricted grammar does not approve"
                    if scored == 0
                    else "no candidate completed a development rollout"
                ),
                "accounting": accounting,
            }
        equations, validation_error = expressible[best]
        return {
            "status": "complete",
            "error": error,
            "equations": equations,
            "training_rollout_error": development_rollout_error(
                equations, context, development[0], development[0],
                seconds=seconds_per_rollout,
            ),
            # Reuse the recorded score; re-scoring would repeat the rollout
            # and count the same system twice in the accounting.
            "development_rollout_error": validation_error,
            "accounting": accounting,
        }

    return search
