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
* Upstream selects by rolling out its own ``System`` from the training initial
  state over the whole training trajectory, the part its equations were
  fitted on and the validation part after it, and sorting on that error in
  the data's own units, ``mse_train_val``. Its ``System`` integrates every
  variable as a state, which would reinterpret our supplied auxiliaries as
  free states. Selection here ranks the same product of frontiers by the same
  quantity over every development trajectory, training and validation, rolled
  out by our evaluator, which holds the supplied channels to their observed
  values. Upstream's ``evaluate_system`` also reports test error; it is never
  called.

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

import functools
import logging
import math
import multiprocessing
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, fields
from itertools import product
from pathlib import Path
from time import monotonic
from typing import Any, Literal

import numpy as np
import openai

from autoformalism.data import DatasetSplit, SplitName, TrainingScaler, Trajectory
from autoformalism.expressions import (
    ModelValidationError,
    ValidationContext,
    baseline_validator,
    compile_candidate,
)
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate
from autoformalism.rebuttal.llm_call_log import CallLog
from autoformalism.rebuttal.llm_ode_campaign import (
    SELECTION_METRIC,
    CellArrays,
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


@dataclass(frozen=True)
class SystemScore:
    """What rolling one assembled system out over the development data found.

    A ``scored`` system carries the error selection ranks by and, for the
    record, its normalized error on each split, as ``development_rollout_error``
    computes it. A ``pruned`` one had already exceeded the error of a system
    that finished, so it could not have been chosen.
    """

    status: Literal["scored", "pruned", "rollout_failed", "inexpressible"]
    equations: dict[str, str] | None = None
    train_val_rollout_mse: float | None = None
    training_rollout_error: float | None = None
    validation_rollout_error: float | None = None
    functions: tuple[str, ...] = ()
    message: str | None = None

    def record(self, system: tuple[str, ...]) -> dict:
        """One line of the task's record of every system it scored."""
        return {
            "system": list(system),
            "status": self.status,
            "equations": self.equations,
            SELECTION_METRIC: self.train_val_rollout_mse,
            "training_rollout_error": self.training_rollout_error,
            "validation_rollout_error": self.validation_rollout_error,
            "functions": list(self.functions),
            "message": self.message,
        }


def _target_scales(train: DatasetSplit, targets: tuple[str, ...]) -> dict[str, float]:
    """Each target's training standard deviation, the evaluator's scale."""
    scaling = TrainingScaler().fit(train).scales
    return {
        target: float(scaling[f"target:{target}"].standard_deviation)
        for target in targets
    }


def score_system(
    system: tuple[str, ...],
    development: tuple[DatasetSplit, DatasetSplit],
    context: ValidationContext,
    channels: tuple[str, ...],
    *,
    seconds: float,
    bound: Any = None,
) -> SystemScore:
    """Roll one assembled system out over every development trajectory.

    Its ``train_val_rollout_mse`` is what LLM-ODE's released code ranks by:
    each trajectory, training and validation, is rolled out from its own
    first values, and the targets' squared errors are averaged over every
    trajectory and time point in the data's own units. A rollout that fails,
    or takes longer than ``seconds``, leaves the system unscored. ``bound``,
    a shared value, holds the lowest error a system has finished with; a
    system whose squared errors so far already exceed it is stopped, since it
    could not be chosen. The bound only ever holds a finished system's error,
    so stopping early changes no choice.
    """
    train, validation = development
    if (
        train.name is not SplitName.TRAIN
        or validation.name is not SplitName.VALIDATION
    ):
        raise ValueError("selection requires TRAIN and VALIDATION, never TEST")
    targets = tuple(context.targets)
    try:
        equations = to_state_equations(system, channels, targets)
    except InexpressibleEquation as exc:
        return SystemScore("inexpressible", functions=exc.functions, message=str(exc))
    except ValueError as exc:  # a variable beyond the channels
        return SystemScore(
            "inexpressible", functions=("unknown_variable",), message=str(exc)
        )
    try:
        compiled = compile_candidate(
            equation_candidate("llm_ode", equations, context),
            context,
            validator=baseline_validator(),
        )
    except ModelValidationError as exc:
        # What the baseline grammar still refuses (a comparison, an undefined
        # name) is counted, not allowed to end a search already paid for.
        return SystemScore(
            "inexpressible",
            equations=equations,
            functions=tuple(sorted({item.code for item in exc.diagnostics})),
            message=str(exc),
        )
    except Exception as exc:  # one odd system must not lose the search
        return SystemScore(
            "inexpressible",
            equations=equations,
            functions=(type(exc).__name__,),
            message=str(exc),
        )
    scales = _target_scales(train, targets)
    points = len(targets) * sum(
        len(trajectory.time)
        for split in development
        for trajectory in split.trajectories
    )
    total = 0.0
    normalized: dict[SplitName, list[float]] = {
        split.name: [] for split in development
    }
    for split in development:
        for trajectory in split.trajectories:
            try:
                simulation = simulate_trajectory(
                    compiled,
                    trajectory,
                    {},
                    {},
                    FitConfig(),
                    deadline=monotonic() + seconds,
                    reset_observed_states=False,
                )
            except TimeoutError:
                # The simulator raises past its deadline rather than failing.
                return SystemScore(
                    "rollout_failed",
                    equations=equations,
                    message=f"{trajectory.trajectory_id}: over {seconds:g} seconds",
                )
            except Exception as exc:  # one odd system must not lose the search
                return SystemScore(
                    "rollout_failed",
                    equations=equations,
                    message=f"{trajectory.trajectory_id}: {type(exc).__name__}: {exc}",
                )
            if not simulation.success:
                return SystemScore(
                    "rollout_failed",
                    equations=equations,
                    message=f"{trajectory.trajectory_id}: {simulation.message}",
                )
            residuals = [
                np.asarray(simulation.predictions[target], dtype=float)
                - np.asarray(trajectory.targets[target], dtype=float)
                for target in targets
            ]
            squared = sum(float(np.sum(np.square(value))) for value in residuals)
            if not math.isfinite(squared):
                return SystemScore(
                    "rollout_failed",
                    equations=equations,
                    message=f"{trajectory.trajectory_id}: a nonfinite error",
                )
            total += squared
            normalized[split.name].append(
                float(
                    np.mean(
                        [
                            np.mean(np.square(value / scales[target]))
                            for target, value in zip(targets, residuals, strict=True)
                        ]
                    )
                )
            )
            # Squared errors only add, so the error so far never exceeds the
            # final one: past the bound, the system cannot be the lowest.
            if bound is not None and total / points > bound.value:
                return SystemScore("pruned", equations=equations)
    value = total / points
    if bound is not None:
        with bound.get_lock():
            bound.value = min(bound.value, value)
    return SystemScore(
        "scored",
        equations=equations,
        train_val_rollout_mse=value,
        training_rollout_error=float(np.mean(normalized[SplitName.TRAIN])),
        validation_rollout_error=float(np.mean(normalized[SplitName.VALIDATION])),
    )


#: One scoring process's development data, set once when the process starts.
_SCORING: dict[str, Any] = {}


def _portable(development: tuple[DatasetSplit, ...]) -> tuple:
    """The development splits as plain values a new process can receive.

    A trajectory keeps its channels in read-only mappings, which do not
    pickle, so each is sent as a dict and frozen again on arrival.
    """
    return tuple(
        (
            split.name,
            split.fingerprint,
            tuple(
                {
                    field.name: (
                        dict(value) if isinstance(value, Mapping) else value
                    )
                    for field in fields(trajectory)
                    for value in (getattr(trajectory, field.name),)
                }
                for trajectory in split.trajectories
            ),
        )
        for split in development
    )


def _rebuilt(portable: tuple) -> tuple[DatasetSplit, ...]:
    """The splits `_portable` sent, as the same frozen objects."""
    return tuple(
        DatasetSplit(name, tuple(Trajectory(**item) for item in rows), fingerprint)
        for name, fingerprint, rows in portable
    )


def _start_scoring(
    development: tuple,
    context: ValidationContext,
    channels: tuple[str, ...],
    seconds: float,
    bound: Any,
) -> None:
    """Hold one task's development data in a scoring process."""
    _SCORING.update(
        development=_rebuilt(development),
        context=context,
        channels=channels,
        seconds=seconds,
        bound=bound,
    )


def _score_in_pool(system: tuple[str, ...]) -> SystemScore:
    """Score one system in a process `_start_scoring` prepared."""
    return score_system(
        system,
        _SCORING["development"],
        _SCORING["context"],
        _SCORING["channels"],
        seconds=_SCORING["seconds"],
        bound=_SCORING["bound"],
    )


def rank_systems(
    frontiers: tuple[tuple[str, ...], ...],
    development: tuple[DatasetSplit, DatasetSplit],
    context: ValidationContext,
    channels: tuple[str, ...],
    *,
    seconds: float,
    workers: int = 1,
) -> tuple[list[tuple[str, ...]], list[SystemScore]]:
    """Score every system in the product of the frontiers, in product order.

    As upstream's driver does, the whole product is scored: there is no cap,
    so a long search is never lost to the size of what it found. ``workers``
    processes share the work, as upstream's 30 do; they are started afresh,
    so none inherits the search's state.
    """
    systems = list(product(*frontiers))
    spawn = multiprocessing.get_context("spawn")
    bound = spawn.Value("d", math.inf)
    started = monotonic()

    def progress(results) -> list[SystemScore]:
        """Collect the scores in order, with about ten progress lines."""
        scores: list[SystemScore] = []
        for score in results:
            scores.append(score)
            if len(scores) % max(1, len(systems) // 10) == 0:
                LOGGER.info(
                    "scored %d of %d systems after %.1fs",
                    len(scores),
                    len(systems),
                    monotonic() - started,
                )
        return scores

    if workers == 1 or len(systems) == 1:
        return systems, progress(
            score_system(
                system, development, context, channels, seconds=seconds, bound=bound
            )
            for system in systems
        )
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=spawn,
        initializer=_start_scoring,
        initargs=(_portable(development), context, channels, seconds, bound),
    ) as pool:
        return systems, progress(pool.map(_score_in_pool, systems))


def best_system(scores: list[SystemScore]) -> int | None:
    """The system with the lowest error, the first of any tie.

    Upstream sorts on the error alone and leaves ties in no set order; taking
    the first in product order, best derivative fits first, makes it fixed.
    """
    best: int | None = None
    lowest = math.inf
    for index, score in enumerate(scores):
        value = score.train_val_rollout_mse
        if score.status == "scored" and value is not None and value < lowest:
            best, lowest = index, value
    return best


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
    workers: int = 1,
    patience_seconds: float = DEFAULT_PATIENCE_SECONDS,
    model: str | None = None,
    bypass_cache: bool = False,
):
    """Bind the pinned checkout to our data; returns a campaign searcher.

    ``model``, when given, is named in every request in place of the first
    model the server lists; ``bypass_cache`` asks a caching gateway for fresh
    answers. Without them the requests are exactly upstream's. Selection
    gives each development rollout ``seconds_per_rollout`` and scores the
    systems in ``workers`` processes.
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
        accounting = {
            "llm_queries": calls,
            # The shared rule, so this campaign is comparable with D3's.
            **d3_accounting(directory / "llm_calls.jsonl"),
            "search_seconds": round(monotonic() - started, 1),
            "frontier_sizes": [len(frontier) for frontier in frontiers],
            "search_error": error,
        }
        return _select(
            frontiers,
            development,
            context,
            train.channels,
            directory,
            accounting,
            error,
            seconds=seconds_per_rollout,
            workers=workers,
        )

    @functools.wraps(search)
    def logged(**kwargs) -> dict:
        with _upstream_log(kwargs["directory"] / "llm_ode.log"):
            return search(**kwargs)

    return logged


@contextmanager
def _upstream_log(path: Path) -> Iterator[None]:
    """Log as upstream's ``main.py`` does: everything at INFO, to ``llm_ode.log``.

    Upstream's client logs every prompt it sends and every reply it gets
    there. Without it a campaign keeps each call's usage and why it ended,
    but not what was asked or answered.
    """
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        yield
    finally:
        root.removeHandler(handler)
        root.setLevel(level)
        handler.close()


def _select(
    frontiers: list[tuple[str, ...]],
    development: tuple[DatasetSplit, DatasetSplit],
    context: ValidationContext,
    channels: tuple[str, ...],
    directory: Path,
    accounting: dict,
    error: str | None,
    *,
    seconds: float,
    workers: int,
) -> dict:
    """Choose the system as upstream's driver does, and record every one scored.

    ``systems.json`` takes the place of the table upstream writes after its
    final iteration: each system, how it fared, and its error.
    """
    started = monotonic()
    systems, scores = rank_systems(
        tuple(frontiers),
        development,
        context,
        channels,
        seconds=seconds,
        workers=workers,
    )
    atomic_json(
        directory / "systems.json",
        {
            "selection_metric": SELECTION_METRIC,
            "systems": [
                score.record(system)
                for system, score in zip(systems, scores, strict=True)
            ],
        },
    )
    counts = Counter(score.status for score in scores)
    operators = sorted(
        {
            name
            for score in scores
            if score.status == "inexpressible"
            for name in score.functions
        }
    )
    accounting = {
        **accounting,
        "systems": len(scores),
        "scored_systems": counts["scored"],
        "pruned_systems": counts["pruned"],
        "failed_systems": counts["rollout_failed"],
        "inexpressible_systems": counts["inexpressible"],
        "inexpressible_operators": operators,
        "selection_seconds": round(monotonic() - started, 1),
    }
    best = best_system(scores)
    if best is None:
        refused = counts["inexpressible"] == len(scores)
        return {
            "status": "inexpressible" if refused else "rollout_failed",
            "error": (
                f"every candidate used {', '.join(operators)}, which the "
                "restricted grammar does not approve"
                if refused
                else "no candidate completed a development rollout"
            ),
            "accounting": accounting,
        }
    chosen = scores[best]
    return {
        "status": "complete",
        "error": error,
        "equations": chosen.equations,
        SELECTION_METRIC: chosen.train_val_rollout_mse,
        # Recorded beside the selection, from the same rollouts; neither
        # chooses anything.
        "training_rollout_error": chosen.training_rollout_error,
        "development_rollout_error": chosen.validation_rollout_error,
        "accounting": accounting,
    }
