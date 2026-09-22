"""Serve LLM-SR's own completion protocol from an OpenAI-compatible endpoint.

LLM-SR's sampler posts a bespoke payload to a hardcoded local URL and expects
a bespoke reply. Its published deployment answers that with a Flask server
wrapping HuggingFace transformers. Pointing it at vLLM instead is a transport
change only: the prompts, the sampling loop, the evaluator and the search are
untouched, and it puts LLM-SR on the same inference stack as every other
method in the comparison rather than a second one.

Two behaviours are copied from upstream's engine deliberately:

* ``max_new_tokens`` defaults to 512. Their sampler never sends it, so their
  engine's own default applies.
* ``temperature``, ``top_k`` and ``top_p`` arrive explicitly as ``None``.
  Their engine reads them with ``params.get(name, default)``, and because the
  keys are present the argparse defaults never apply -- the values reach
  HuggingFace as ``None`` and its own defaults take over. There is no exact
  equivalent across inference stacks, so a ``None`` here means "the served
  model's default", which is what this records and what must be declared.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

#: Their engine's argparse default, which applies because the sampler omits it.
DEFAULT_MAX_TOKENS = 512


class UpstreamEndpointError(RuntimeError):
    """The served endpoint could not answer, with the reason preserved.

    LLM-SR's sampler retries forever on any exception, so a fault that is not
    surfaced here becomes a job that spins to its walltime in silence.
    """


@dataclass
class ShimAccounting:
    """What the shim was asked for, so a silent stall is visible afterwards."""

    requests: int = 0
    samples: int = 0
    failures: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def record_failure(self, reason: str) -> None:
        """Count a failure by kind rather than only in total."""
        self.failures += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def translate_request(payload: dict, model: str) -> dict:
    """Turn one LLM-SR request into an OpenAI completions request."""
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise UpstreamEndpointError("request carried no prompt")
    repeat = payload.get("repeat_prompt", 1)
    if not isinstance(repeat, int) or repeat < 1:
        raise UpstreamEndpointError(f"repeat_prompt is not a positive int: {repeat!r}")
    params = payload.get("params") or {}
    request = {
        "model": model,
        "prompt": prompt,
        "n": repeat,
        "max_tokens": int(params.get("max_new_tokens") or DEFAULT_MAX_TOKENS),
    }
    # Only forward sampling controls the caller actually set. Their sampler
    # sends explicit Nones, which mean "the model's default", not a value.
    for name in ("temperature", "top_p"):
        value = params.get(name)
        if value is not None:
            request[name] = float(value)
    if params.get("top_k") is not None:
        request["top_k"] = int(params["top_k"])
    return request


def translate_response(payload: dict) -> list[str]:
    """Turn an OpenAI completions reply into LLM-SR's ``content`` list."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise UpstreamEndpointError("endpoint returned no choices")
    texts = [str(choice.get("text", "")) for choice in choices]
    if not any(texts):
        raise UpstreamEndpointError("endpoint returned only empty completions")
    return texts


def complete(
    payload: dict,
    *,
    base_url: str,
    model: str,
    timeout: float = 300.0,
    accounting: ShimAccounting | None = None,
) -> dict:
    """Answer one LLM-SR request from the OpenAI-compatible endpoint."""
    request = translate_request(payload, model)
    url = base_url.rstrip("/") + "/v1/completions"
    body = json.dumps(request).encode("utf-8")
    http = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    if accounting is not None:
        accounting.requests += 1
    try:
        with urllib.request.urlopen(http, timeout=timeout) as response:
            answer = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        if accounting is not None:
            accounting.record_failure(type(exc).__name__)
        raise UpstreamEndpointError(f"{type(exc).__name__}: {exc}") from exc
    texts = translate_response(answer)
    if accounting is not None:
        accounting.samples += len(texts)
    return {"content": texts}
