"""Drive the pinned LLM-SR checkout over one Phase-B target.

Their `pipeline.main` runs unmodified. What this supplies is the data, the
specification file their runner is always pointed at, and a transport: their
sampler posts a bespoke payload to a hardcoded local URL, and this answers it
from the vLLM endpoint the other methods use.

Their evaluator executes each synthesized program to score it. That is what
program synthesis is, and it cannot be removed without removing the method, so
the job confines it rather than preventing it. Nothing in the recovery path
here executes anything: the selected program is parsed, its coefficients are
refitted over a numeric evaluator, and the result is scored by our own rollout.

The one behaviour that must be guarded is `_draw_samples_local`, which wraps
its request loop in `while True: except Exception: continue`. An endpoint fault
would otherwise spin until the job's walltime with nothing recorded, so the
search is stopped once the endpoint has failed for longer than the campaign's
patience.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from autoformalism.data import DatasetSplit
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.llm_call_log import CallLog
from autoformalism.rebuttal.llm_ode_upstream import InexpressibleEquation
from autoformalism.rebuttal.llm_sr_shim import ShimAccounting, complete
from autoformalism.rebuttal.llm_sr_upstream import (
    InexpressibleProgram,
    best_sample,
    build_specification,
    convert_program,
    equation_body,
    refit_parameters,
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
):
    """Answer their sampler's requests from the OpenAI-compatible endpoint.

    Replaces the request method rather than the URL, which avoids running a
    second server inside the job; the payload and reply are theirs unchanged.
    """
    original = module.LocalLLM._do_request
    first, longest = BACKOFF_SECONDS
    state = {"consecutive": 0, "since": 0.0, "pause": first}

    def _do_request(self, content: str) -> list[str]:
        payload = {
            "prompt": content.strip("\n").strip(),
            "repeat_prompt": self._samples_per_prompt if self._batch_inference else 1,
            "params": {"do_sample": True, "temperature": None, "top_k": None,
                       "top_p": None, "add_special_tokens": False,
                       "skip_special_tokens": True},
        }
        try:
            answer = complete(
                payload, base_url=base_url, model=model, accounting=accounting
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
):
    """Bind the pinned checkout to our data; returns a campaign searcher."""
    import sys

    resolved = upstream_root.expanduser().resolve()
    if not (resolved / "llmsr" / "pipeline.py").is_file():
        raise ValueError(f"not an LLM-SR checkout: {resolved}")
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    from llmsr import config as config_lib
    from llmsr import evaluator, pipeline, sampler

    def search(
        *,
        channels: tuple[str, ...],
        targets: tuple[str, ...],
        values: np.ndarray,
        derivatives: np.ndarray,
        description: str,
        directory: Path,
        development: tuple[DatasetSplit, DatasetSplit],
        context: ValidationContext,
        score_rollout,
    ) -> dict:
        """One LLM-SR run per target, as their specification format requires."""
        accounting = ShimAccounting(log=CallLog(directory / "llm_calls.jsonl"))
        equations: dict[str, str] = {}
        inexpressible: list[str] = []
        started = monotonic()
        for target in targets:
            specification, mapping = build_specification(
                description, channels, target
            )
            log_dir = directory / f"llmsr-{target}"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "specification.txt").write_text(specification, encoding="utf-8")

            column = channels.index(target)
            dataset = {
                "data": {
                    "inputs": values,
                    "outputs": derivatives[:, column].reshape(-1),
                }
            }
            class_config = config_lib.ClassConfig(
                llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox
            )
            try:
                with _transport(
                    sampler, base_url, model, accounting, patience=patience_seconds
                ):
                    pipeline.main(
                        specification=specification,
                        inputs=dataset,
                        config=config_lib.Config(),
                        max_sample_nums=samples,
                        class_config=class_config,
                        log_dir=str(log_dir),
                    )
            except SamplerStalled as exc:
                return {
                    "status": "endpoint_unavailable",
                    "error": str(exc),
                    "accounting": _accounting(accounting, started, inexpressible,
                                  directory / 'llm_calls.jsonl'),
                }

            best = best_sample(log_dir)
            if best is None:
                return {
                    "status": "no_candidates",
                    "error": f"no sample scored for target {target}",
                    "accounting": _accounting(accounting, started, inexpressible,
                                  directory / 'llm_calls.jsonl'),
                }
            try:
                body = equation_body(best["function"])
                symbolic = convert_program(body, mapping)
                fitted = refit_parameters(
                    symbolic.expression,
                    {name: values[:, index] for index, name in enumerate(channels)},
                    derivatives[:, column].reshape(-1),
                    symbolic.used_parameters,
                )
                if fitted is None:
                    raise InexpressibleProgram("coefficient refit did not converge")
                equations[target] = convert_program(body, mapping, fitted).expression
            except InexpressibleProgram as exc:
                inexpressible.append(f"{target}: {exc}")

        if len(equations) != len(targets):
            return {
                "status": "inexpressible",
                "error": "; ".join(inexpressible),
                "accounting": _accounting(accounting, started, inexpressible,
                                  directory / 'llm_calls.jsonl'),
            }
        try:
            error = score_rollout(equations, context, *development,
                                  seconds=seconds_per_rollout)
        except InexpressibleEquation as exc:
            # A refitted exponent such as `G ** 0.73` converts cleanly but the
            # grammar accepts only integer powers; record it, do not crash.
            inexpressible.append(str(exc))
            return {
                "status": "inexpressible",
                "error": "; ".join(inexpressible),
                "accounting": _accounting(accounting, started, inexpressible,
                                  directory / 'llm_calls.jsonl'),
            }
        if error is None:
            return {
                "status": "rollout_failed",
                "error": "the selected system did not complete a development rollout",
                "equations": equations,
                "accounting": _accounting(accounting, started, inexpressible,
                                  directory / 'llm_calls.jsonl'),
            }
        return {
            "status": "complete",
            "error": None,
            "equations": equations,
            "development_rollout_error": error,
            "training_rollout_error": score_rollout(
                equations, context, development[0], development[0],
                seconds=seconds_per_rollout,
            ),
            "accounting": _accounting(accounting, started, inexpressible,
                                  directory / 'llm_calls.jsonl'),
        }

    return search


def _accounting(
    accounting: ShimAccounting,
    started: float,
    inexpressible: list[str],
    log_path: Path,
) -> dict:
    """What the run cost, and what our grammar could not read."""
    return {
        **d3_accounting(log_path),
        "llm_requests": accounting.requests,
        "llm_samples": accounting.samples,
        "transport_failures": accounting.failures,
        "transport_failure_reasons": accounting.reasons,
        "search_seconds": round(monotonic() - started, 1),
        "inexpressible_targets": inexpressible,
    }
