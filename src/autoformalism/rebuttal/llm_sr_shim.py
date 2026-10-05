"""Serve LLM-SR's own completion protocol from an OpenAI-compatible endpoint.

LLM-SR's sampler posts a bespoke payload to a hardcoded local URL and expects
a bespoke reply. Its published deployment answers that with a Flask server
wrapping HuggingFace transformers, which puts the prompt into the model's chat
template as one user message (``apply_chat_template`` with a generation
prompt) and returns the generated text. The OpenAI-compatible equivalent is a
chat completion with that one user message, answered by the message content,
and that is what this sends. Their sampling loop, evaluator and search are
untouched, and LLM-SR runs on the same inference stack as every other method
in the comparison rather than a second one.

Until 2026-10-05 this sent the prompt to ``/v1/completions`` as raw text,
outside any chat template. The Phase C budget pilot found it: gpt-oss-120b
continued the raw text with prose, and almost no sample held a program.

Two behaviours are copied from upstream's engine deliberately:

* ``max_new_tokens`` defaults to 512. Their sampler never sends it, so their
  engine's own default applies. A plan may declare a longer limit for a
  reasoning model (``phase_c_vendored_campaign.ReasoningModelAdaptation``).
* ``temperature``, ``top_k`` and ``top_p`` arrive explicitly as ``None``.
  Their engine reads them with ``params.get(name, default)``, and because the
  keys are present the argparse defaults never apply -- the values reach
  HuggingFace as ``None`` and its own defaults take over. There is no exact
  equivalent across inference stacks, so a ``None`` here means "the served
  model's default", which is what this records and what must be declared.
  Reasoning effort is not sent either, so it is the served default too.

One reading of the reply is an adaptation, applied only when a plan declares
it: ``join_split_header``.

Their engine generates afresh for every request. The Jetstream2 hosted
service is a LiteLLM gateway that replays a stored answer to a repeated
request, and LLM-SR repeats requests: its islands all begin from the same
prompt. Against that service every request carries LiteLLM's switch to
generate anew (``bypass_cache``), and an answer the gateway marks as replayed
is logged as a cache hit, so a switch that stopped working would show.
"""

from __future__ import annotations

import ast
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

#: Their engine's argparse default, which applies because the sampler omits it.
DEFAULT_MAX_TOKENS = 512

#: How many lines after a ``def`` line are searched for the end of its header.
HEADER_LOOKAHEAD_LINES = 40

#: LiteLLM's per-request switch to generate a new answer rather than replay a
#: stored one. On 2026-10-05 (LiteLLM 1.98.0) it worked on the hosted service;
#: a ``Cache-Control: no-cache`` header did not.
LITELLM_NO_CACHE = {"cache": {"no-cache": True}}

#: The header LiteLLM adds only to an answer it replayed from its cache.
LITELLM_CACHE_HIT_HEADER = "X-Litellm-Cache-Key"


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
    #: Written so the shared accounting rule can read this campaign too.
    log: object | None = None

    def record_failure(self, reason: str) -> None:
        """Count a failure by kind rather than only in total."""
        self.failures += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def translate_request(
    payload: dict,
    model: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    bypass_cache: bool = False,
) -> dict:
    """Turn one LLM-SR request into an OpenAI chat completions request.

    ``max_tokens`` is the engine's default length, used when the payload sets
    none, as their sampler never does. ``bypass_cache`` asks a LiteLLM gateway
    to generate afresh, as their engine always does.
    """
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise UpstreamEndpointError("request carried no prompt")
    repeat = payload.get("repeat_prompt", 1)
    if not isinstance(repeat, int) or repeat < 1:
        raise UpstreamEndpointError(f"repeat_prompt is not a positive int: {repeat!r}")
    params = payload.get("params") or {}
    request = {
        "model": model,
        # Their engine applies the chat template to the prompt as one user turn.
        "messages": [{"role": "user", "content": prompt}],
        "n": repeat,
        "max_tokens": int(params.get("max_new_tokens") or max_tokens),
    }
    # Only forward sampling controls the caller actually set. Their sampler
    # sends explicit Nones, which mean "the model's default", not a value.
    for name in ("temperature", "top_p"):
        value = params.get(name)
        if value is not None:
            request[name] = float(value)
    if params.get("top_k") is not None:
        request["top_k"] = int(params["top_k"])
    if bypass_cache:
        request.update(LITELLM_NO_CACHE)
    return request


def join_split_header(text: str) -> str:
    """Put a ``def`` header that the model split over several lines on one line.

    Upstream's ``sampler._extract_body`` finds the first line that starts with
    ``def`` and takes every line after it as the function body. A model that
    writes its header over several lines, one parameter per line as gpt-oss
    does, leaves the rest of the header at the top of that body; upstream's
    trimming then cuts the body at its first syntax error and keeps nothing.

    This finds the same first ``def`` line and, when the header continues past
    it, replaces the header with a one-line equivalent, so upstream reads the
    body the model wrote. The text is only parsed, never run. A header already
    on one line, a header that never closes and text without one are returned
    unchanged, so upstream sees exactly what it would have seen.
    """
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines) if line[:3] == "def"), None)
    if start is None:
        return text
    stop = min(len(lines), start + HEADER_LOOKAHEAD_LINES)
    for end in range(start, stop):
        try:
            module = ast.parse("\n".join(lines[start : end + 1]) + "\n    pass\n")
        except SyntaxError:
            continue
        node = module.body[0] if len(module.body) == 1 else None
        if end == start or not isinstance(node, ast.FunctionDef):
            return text
        node.body = [ast.Pass()]
        header = ast.unparse(node).splitlines()
        if len(header) != 2:
            return text
        joined = "\n".join([*lines[:start], header[0], *lines[end + 1 :]])
        return joined + "\n" if text.endswith("\n") else joined
    return text


def finish_reasons(payload: dict) -> list[str]:
    """Why each choice stopped; ``length`` means the token limit cut it off."""
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return []
    return [
        str(choice.get("finish_reason"))
        for choice in choices
        if isinstance(choice, dict)
    ]


def translate_response(payload: dict, *, join_headers: bool = False) -> list[str]:
    """Turn an OpenAI chat completions reply into LLM-SR's ``content`` list.

    Each sample is a choice's message content, the model's answer without its
    reasoning; a choice with no content is an empty sample.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise UpstreamEndpointError("endpoint returned no choices")
    texts = []
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        texts.append(content if isinstance(content, str) else "")
    if not any(texts):
        reasons = ", ".join(finish_reasons(payload)) or "none given"
        raise UpstreamEndpointError(
            f"endpoint returned only empty completions (finish reasons: {reasons})"
        )
    if join_headers:
        texts = [join_split_header(text) for text in texts]
    return texts


def http_error_detail(exc: urllib.error.HTTPError) -> str:
    """The server's own explanation of a refusal, on one short line.

    The status line alone can mislead. On 2026-10-04 the Jetstream2 service
    refused with 401 Unauthorized, then with a 503 whose body alone said that
    its authentication database was unreachable.
    """
    try:
        body = exc.read(4096).decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""
    message = body
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error", payload.get("detail"))
        if isinstance(error, dict):
            error = error.get("message")
        if isinstance(error, str):
            message = error
    return " ".join(message.split())[:300]


def complete(
    payload: dict,
    *,
    base_url: str,
    model: str,
    timeout: float = 600.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    join_headers: bool = False,
    bypass_cache: bool = False,
    accounting: ShimAccounting | None = None,
) -> dict:
    """Answer one LLM-SR request from the OpenAI-compatible endpoint.

    The timeout is per request; a reasoning model writing four samples of a few
    thousand tokens each has taken about a minute and a half on the hosted
    service.
    """
    request = translate_request(
        payload, model, max_tokens=max_tokens, bypass_cache=bypass_cache
    )
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(request).encode("utf-8")
    http = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    if accounting is not None:
        accounting.requests += 1
    try:
        with urllib.request.urlopen(http, timeout=timeout) as response:
            answer = json.loads(response.read().decode("utf-8"))
            replayed = response.headers.get(LITELLM_CACHE_HIT_HEADER) is not None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        kind, message = type(exc).__name__, f"{type(exc).__name__}: {exc}"
        if isinstance(exc, urllib.error.HTTPError):
            kind = f"HTTPError {exc.code}"
            if detail := http_error_detail(exc):
                message = f"{message} ({detail})"
        if accounting is not None:
            accounting.record_failure(kind)
            if accounting.log is not None:
                accounting.log.failure(message)
        raise UpstreamEndpointError(message) from exc
    # An answered request cost tokens whatever it holds, so it is logged first.
    if accounting is not None and accounting.log is not None:
        accounting.log.response(
            answer, finish_reasons=finish_reasons(answer), cache_hit=replayed
        )
    try:
        texts = translate_response(answer, join_headers=join_headers)
    except UpstreamEndpointError:
        if accounting is not None:
            accounting.record_failure("empty reply")
        raise
    if accounting is not None:
        accounting.samples += len(texts)
    return {"content": texts}
