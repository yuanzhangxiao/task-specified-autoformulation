"""Drive the pinned LLM-SR checkout over one Phase-B target.

Their `pipeline.main` runs unmodified. What this supplies is the data, the
specification file their runner is always pointed at, and a transport: their
sampler posts a bespoke payload to a hardcoded local URL, and this answers it
from the vLLM endpoint the other methods use.

A plan may declare an adaptation to a reasoning model: a longer generation
limit than their engine's 512 tokens, and a function header the model split
over several lines read as one (`llm_sr_shim.join_split_header`). Both act in
the transport; without a declaration the transport behaves as their engine.
Against an endpoint that replays stored answers, every request asks it to
generate afresh, as their engine always does. A plan may also declare the
settings the LLM-SR paper states where their released code differs: the
sampling temperature, which their client leaves unset, and two fields of their
own configuration, the cluster-sampling period and the number of evaluators.

Their evaluator executes each synthesized program to score it. That is what
program synthesis is, and it cannot be removed without removing the method, so
the job confines it rather than preventing it. Nothing in the recovery path
here executes anything: the selected program is read without being run, its
coefficients are refitted as their evaluator fits them, and the result is
scored by our own rollout.

Each target is its own LLM-SR run, so a cell's targets can be searched by
separate processes at once. A finished search leaves a record beside its
samples and is never run again; one that stopped part-way is set aside and
starts over, because their search cannot resume. The cell's model is assembled
once every target has finished.

The one behaviour that must be guarded is `_draw_samples_local`, which wraps
its request loop in `while True: except Exception: continue`. An endpoint fault
would otherwise spin until the job's walltime with nothing recorded, so the
search is stopped once the endpoint has failed for longer than the campaign's
patience.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from autoformalism.data import DatasetSplit
from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.llm_call_log import CallLog
from autoformalism.rebuttal.llm_ode_upstream import InexpressibleEquation
from autoformalism.rebuttal.llm_sr_shim import (
    DEFAULT_MAX_TOKENS,
    ShimAccounting,
    complete,
)
from autoformalism.rebuttal.llm_sr_upstream import (
    MAX_NPARAMS,
    InexpressibleProgram,
    best_sample,
    build_specification,
    convert_program,
    equation_body,
    refit_program,
    sample_yield,
    training_error,
)
from autoformalism.rebuttal.phase_b_d3 import accounting as d3_accounting

LOGGER = logging.getLogger(__name__)

#: How long the endpoint may go on failing, from the first failure in a row,
#: before the search is stopped. A vLLM started inside the job does not come
#: back once it has died; a campaign against a service that does come back
#: passes a longer patience. Their retry loop catches Exception, so escaping it
#: needs BaseException.
DEFAULT_PATIENCE_SECONDS = 15 * 60.0

#: First and longest pause before their sampler retries a failed request. The
#: pause doubles per consecutive failure: a blip costs seconds, and a long
#: outage is retried about once a minute until it ends or patience runs out.
BACKOFF_SECONDS = (1.0, 60.0)

#: What a finished target search leaves beside its samples: what it cost and
#: when it ran. A target with one is never searched again.
SEARCH_RECORD = "search.json"

#: The shared accounting rule's counts, summed over a task's target searches.
CALL_COUNTS = ("physical_requests", "observed_tokens", "unknown_usage_requests",
               "cache_hits")


class SamplerStalled(BaseException):
    """Raised through upstream's `except Exception` to end a wedged search."""


def _pause(seconds: float) -> None:
    """Wait before the next attempt; a seam so tests need not sleep."""
    time.sleep(seconds)


def _clock() -> float:
    """Seconds on a monotonic clock; a seam so tests can move time on."""
    return monotonic()


@contextmanager
def _transport(
    module: Any,
    base_url: str,
    model: str,
    accounting: ShimAccounting,
    *,
    patience: float = DEFAULT_PATIENCE_SECONDS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    join_headers: bool = False,
    bypass_cache: bool = False,
    temperature: float | None = None,
):
    """Answer their sampler's requests from the OpenAI-compatible endpoint.

    Replaces the request method rather than the URL, which avoids running a
    second server inside the job; the payload and reply are theirs unchanged.
    `max_tokens` and `join_headers` are what a plan declares for a reasoning
    model; their defaults are their engine's behaviour. `temperature` is the
    paper's, when a plan declares it; their client sends none. `bypass_cache`
    follows the endpoint kind.
    """
    original = module.LocalLLM._do_request
    first, longest = BACKOFF_SECONDS
    state = {"consecutive": 0, "since": 0.0, "pause": first}

    def _do_request(self, content: str) -> list[str]:
        payload = {
            "prompt": content.strip("\n").strip(),
            "repeat_prompt": self._samples_per_prompt if self._batch_inference else 1,
            "params": {"do_sample": True, "temperature": temperature, "top_k": None,
                       "top_p": None, "add_special_tokens": False,
                       "skip_special_tokens": True},
        }
        try:
            answer = complete(
                payload,
                base_url=base_url,
                model=model,
                max_tokens=max_tokens,
                join_headers=join_headers,
                bypass_cache=bypass_cache,
                accounting=accounting,
            )
        except Exception:
            now = _clock()
            if state["consecutive"] == 0:
                state["since"], state["pause"] = now, first
            state["consecutive"] += 1
            if now - state["since"] >= patience:
                raise SamplerStalled(
                    f"the endpoint failed {state['consecutive']} times in a row "
                    f"over {(now - state['since']) / 60:.0f} minutes; their "
                    "sampler retries forever, so the search is stopped here"
                ) from None
            # Their loop retries at once; without a pause an outage would be
            # met with a stream of requests.
            _pause(state["pause"])
            state["pause"] = min(longest, 2 * state["pause"])
            raise
        state["consecutive"] = 0
        return answer["content"]

    module.LocalLLM._do_request = _do_request
    try:
        yield
    finally:
        module.LocalLLM._do_request = original


def build_searcher(
    *,
    upstream_root: Path,
    base_url: str,
    model: str,
    samples: int,
    seconds_per_rollout: float = 60.0,
    patience_seconds: float = DEFAULT_PATIENCE_SECONDS,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    join_split_headers: bool = False,
    bypass_cache: bool = False,
    before_search: Callable[[], None] | None = None,
    temperature: float | None = None,
    cluster_sampling_temperature_period: int | None = None,
    num_evaluators: int | None = None,
):
    """Bind the pinned checkout to our data; returns a campaign searcher.

    `max_new_tokens` and `join_split_headers` come from a plan's declared
    reasoning-model adaptation; the defaults are upstream's own behaviour.
    `temperature`, `cluster_sampling_temperature_period` and `num_evaluators`
    are the paper's values where a plan declares them; left out, their code's
    own apply. `bypass_cache` is for an endpoint that replays stored answers.
    `before_search` runs before each target search that actually starts, so a
    run that only assembles finished searches contacts no endpoint.
    """
    import sys

    resolved = upstream_root.expanduser().resolve()
    if not (resolved / "llmsr" / "pipeline.py").is_file():
        raise ValueError(f"not an LLM-SR checkout: {resolved}")
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    from llmsr import config as config_lib
    from llmsr import evaluator, pipeline, sampler

    settings: dict[str, Any] = {}
    if cluster_sampling_temperature_period is not None:
        settings["experience_buffer"] = config_lib.ExperienceBufferConfig(
            cluster_sampling_temperature_period=cluster_sampling_temperature_period
        )
    if num_evaluators is not None:
        settings["num_evaluators"] = num_evaluators

    def search(
        *,
        channels: tuple[str, ...],
        targets: tuple[str, ...],
        values: np.ndarray,
        labels: np.ndarray,
        description: str,
        directory: Path,
        development: tuple[DatasetSplit, DatasetSplit],
        context: ValidationContext,
        score_rollout,
        only: str | None = None,
    ) -> dict:
        """One LLM-SR run per target, as their specification format requires.

        `values` holds one row per training point over `channels`, and
        `labels` the derivative of each of `targets` at that row. A target
        whose search has finished is not searched again. `only` searches that
        one target and returns without assembling a model.
        """
        if only is not None and only not in targets:
            raise ValueError(f"{only!r} is not among the searched targets {targets}")
        if labels.shape != (values.shape[0], len(targets)):
            raise ValueError("one derivative label per row and target is required")
        equations: dict[str, str] = {}
        selected: dict[str, dict] = {}
        inexpressible: list[str] = []
        # The search this process is running, counted if it stalls.
        running: dict[str, Any] = {"accounting": ShimAccounting(), "started": None}

        def spent() -> dict:
            return _accounting(directory, targets, inexpressible, running)

        for target in targets if only is None else (only,):
            log_dir = directory / f"llmsr-{target}"
            if (log_dir / SEARCH_RECORD).is_file():
                continue
            if before_search is not None:
                before_search()
            set_aside_unfinished_search(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            specification, _ = build_specification(description, channels, target)
            (log_dir / "specification.txt").write_text(specification, encoding="utf-8")

            dataset = {
                "data": {
                    "inputs": values,
                    "outputs": labels[:, targets.index(target)].reshape(-1),
                }
            }
            class_config = config_lib.ClassConfig(
                llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox
            )
            accounting = ShimAccounting(log=CallLog(log_dir / "llm_calls.jsonl"))
            started_utc = _utc_now()
            running.update(accounting=accounting, started=monotonic())
            # Their sampler counts samples on its class, which only a new
            # process resets; their runner gives each problem a new process.
            # Without this, a second target searched here would stop at once,
            # its budget spent by the first.
            sampler.Sampler._global_samples_nums = 1
            try:
                with _transport(
                    sampler,
                    base_url,
                    model,
                    accounting,
                    patience=patience_seconds,
                    max_tokens=max_new_tokens,
                    join_headers=join_split_headers,
                    bypass_cache=bypass_cache,
                    temperature=temperature,
                ):
                    pipeline.main(
                        specification=specification,
                        inputs=dataset,
                        config=config_lib.Config(**settings),
                        max_sample_nums=samples,
                        class_config=class_config,
                        log_dir=str(log_dir),
                    )
            except SamplerStalled as exc:
                return {
                    "status": "endpoint_unavailable",
                    "error": str(exc),
                    "accounting": spent(),
                }
            atomic_json(
                log_dir / SEARCH_RECORD,
                {
                    "target": target,
                    "llm_requests": accounting.requests,
                    "llm_samples": accounting.samples,
                    "transport_failures": accounting.failures,
                    "transport_failure_reasons": dict(accounting.reasons),
                    "search_seconds": round(monotonic() - running["started"], 1),
                    "started_utc": started_utc,
                    "finished_utc": _utc_now(),
                },
            )
            running.update(accounting=ShimAccounting(), started=None)

        if only is not None:
            return {"status": "searched", "target": only, "accounting": spent()}

        columns = {name: values[:, index] for index, name in enumerate(channels)}
        for target in targets:
            log_dir = directory / f"llmsr-{target}"
            _, mapping = build_specification(description, channels, target)
            label = labels[:, targets.index(target)].reshape(-1)
            best = best_sample(log_dir)
            if best is None:
                return {
                    "status": "no_candidates",
                    "error": f"no sample scored for target {target}",
                    "accounting": spent(),
                }
            try:
                body = equation_body(best["function"])
                # A body with no equation is refused for its own reason, not
                # for the fit that could not then be made.
                convert_program(body, mapping, dict.fromkeys(range(MAX_NPARAMS), 1.0))
                fitted = refit_program(body, mapping, columns, label)
                if fitted is None:
                    raise InexpressibleProgram("coefficient refit did not converge")
                equations[target] = convert_program(body, mapping, fitted).expression
                selected[target] = {
                    "sample_order": best.get("sample_order"),
                    "llm_sr_score": best["score"],
                    "refitted_training_mse": training_error(
                        equations[target], columns, label
                    ),
                }
            except (InexpressibleProgram, RecursionError) as exc:
                inexpressible.append(f"{target}: {exc}")

        if len(equations) != len(targets):
            return {
                "status": "inexpressible",
                "error": "; ".join(inexpressible),
                "selected_samples": selected,
                "accounting": spent(),
            }
        try:
            error = score_rollout(equations, context, *development,
                                  seconds=seconds_per_rollout)
        except InexpressibleEquation as exc:
            # What converts can still name something the evaluator's grammar
            # refuses; record it, do not crash.
            inexpressible.append(str(exc))
            return {
                "status": "inexpressible",
                "error": "; ".join(inexpressible),
                "selected_samples": selected,
                "accounting": spent(),
            }
        if error is None:
            return {
                "status": "rollout_failed",
                "error": "the selected system did not complete a development rollout",
                "equations": equations,
                "selected_samples": selected,
                "accounting": spent(),
            }
        return {
            "status": "complete",
            "error": None,
            "equations": equations,
            "selected_samples": selected,
            "development_rollout_error": error,
            "training_rollout_error": score_rollout(
                equations, context, development[0], development[0],
                seconds=seconds_per_rollout,
            ),
            "accounting": spent(),
        }

    return search


def _utc_now() -> str:
    """The wall-clock time a record states, in UTC."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def set_aside_unfinished_search(log_dir: Path) -> Path | None:
    """Move a target search that stopped part-way out of the next one's way.

    It is kept beside the task for its cost and its logs, under a name the
    task's accounting does not read. Restarting into the same directory would
    let the new search's model be chosen from samples the stopped one wrote.
    """
    if not log_dir.is_dir() or (log_dir / SEARCH_RECORD).exists():
        return None
    if not any(log_dir.iterdir()):
        return None
    number = 1
    while (
        moved := log_dir.with_name(f"{log_dir.name}.interrupted-{number}")
    ).exists():
        number += 1
    log_dir.rename(moved)
    return moved


def _accounting(
    directory: Path,
    targets: tuple[str, ...],
    inexpressible: list[str],
    running: dict[str, Any],
) -> dict:
    """What the searches cost, how many samples scored, what our grammar refused.

    A finished search is counted from its record, so one that another process
    or an earlier run finished is counted too; a search that stopped in this
    process is counted from `running`. `search_seconds` adds the searches'
    durations; `wall_seconds` spans the first start to the last finish, which
    is shorter when targets were searched at once.
    """
    calls = dict.fromkeys(CALL_COUNTS, 0)
    written = scored = 0
    accounting: ShimAccounting = running["accounting"]
    requests, samples, failures = (
        accounting.requests, accounting.samples, accounting.failures
    )
    reasons = dict(accounting.reasons)
    seconds = 0.0 if running["started"] is None else monotonic() - running["started"]
    starts: list[datetime] = []
    finishes: list[datetime] = []
    for target in targets:
        log_dir = directory / f"llmsr-{target}"
        for key, value in d3_accounting(log_dir / "llm_calls.jsonl").items():
            calls[key] += value
        counts = sample_yield(log_dir)
        written += counts["model_samples"]
        scored += counts["model_samples_scored"]
        path = log_dir / SEARCH_RECORD
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        requests += record["llm_requests"]
        samples += record["llm_samples"]
        failures += record["transport_failures"]
        for reason, count in record["transport_failure_reasons"].items():
            reasons[reason] = reasons.get(reason, 0) + count
        seconds += record["search_seconds"]
        starts.append(datetime.fromisoformat(record["started_utc"]))
        finishes.append(datetime.fromisoformat(record["finished_utc"]))
    value = {
        **calls,
        "llm_requests": requests,
        "llm_samples": samples,
        # Samples the model wrote that LLM-SR's evaluator could score; one
        # that is empty or does not parse never scores.
        "model_samples": written,
        "model_samples_scored": scored,
        "transport_failures": failures,
        "transport_failure_reasons": reasons,
        "search_seconds": round(seconds, 1),
        "inexpressible_targets": inexpressible,
    }
    if running["started"] is None and len(starts) == len(targets):
        value["wall_seconds"] = round(
            (max(finishes) - min(starts)).total_seconds(), 1
        )
    return value
