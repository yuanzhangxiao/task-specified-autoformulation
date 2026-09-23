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
would otherwise spin until the job's walltime with nothing recorded.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from autoformalism.data import DatasetSplit
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.llm_sr_shim import ShimAccounting, complete
from autoformalism.rebuttal.llm_sr_upstream import (
    InexpressibleProgram,
    best_sample,
    build_specification,
    convert_program,
    equation_body,
    refit_parameters,
)

LOGGER = logging.getLogger(__name__)

#: Consecutive transport failures after which the sampler is stopped. Their
#: retry loop catches Exception, so escaping it needs BaseException.
STALL_LIMIT = 20


class SamplerStalled(BaseException):
    """Raised through upstream's `except Exception` to end a wedged search."""


@contextmanager
def _transport(module: Any, base_url: str, model: str, accounting: ShimAccounting):
    """Answer their sampler's requests from the OpenAI-compatible endpoint.

    Replaces the request method rather than the URL, which avoids running a
    second server inside the job; the payload and reply are theirs unchanged.
    """
    original = module.LocalLLM._do_request
    state = {"consecutive": 0}

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
            state["consecutive"] += 1
            if state["consecutive"] >= STALL_LIMIT:
                raise SamplerStalled(
                    f"{STALL_LIMIT} consecutive endpoint failures; their sampler "
                    "retries forever, so the search is stopped here"
                ) from None
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
        accounting = ShimAccounting()
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
                with _transport(sampler, base_url, model, accounting):
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
                    "accounting": _accounting(accounting, started, inexpressible),
                }

            best = best_sample(log_dir)
            if best is None:
                return {
                    "status": "no_candidates",
                    "error": f"no sample scored for target {target}",
                    "accounting": _accounting(accounting, started, inexpressible),
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
                "accounting": _accounting(accounting, started, inexpressible),
            }
        error = score_rollout(equations, context, *development,
                              seconds=seconds_per_rollout)
        if error is None:
            return {
                "status": "rollout_failed",
                "error": "the selected system did not complete a development rollout",
                "equations": equations,
                "accounting": _accounting(accounting, started, inexpressible),
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
            "accounting": _accounting(accounting, started, inexpressible),
        }

    return search


def _accounting(
    accounting: ShimAccounting, started: float, inexpressible: list[str]
) -> dict:
    """What the run cost, and what our grammar could not read."""
    return {
        "llm_requests": accounting.requests,
        "llm_samples": accounting.samples,
        "transport_failures": accounting.failures,
        "transport_failure_reasons": accounting.reasons,
        "search_seconds": round(monotonic() - started, 1),
        "inexpressible_targets": inexpressible,
    }
