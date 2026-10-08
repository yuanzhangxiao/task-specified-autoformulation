"""Run D3 as published on Phase C public cells, against a declared endpoint.

D3's own loop runs, with its prompts and its search settings
(``baselines/d3_upstream.py``): a first model, then generations that each
reflect on the best models so far and write a new one, native fitting, and the
model with the lowest validation loss, which is D3's own selection rule. The
model writes equations in our restricted format instead of code, so this is
D3's white-box mode. The recursive rollout check and the sealed result are
Phase B's (``phase_b_d3.discover_and_seal``). As for LLM-SR and LLM-ODE, the
data come from a verified Phase C release, and a plan declares where its model
is served and names it.

Each request carries the plan's sampling settings. The plans use the paper's
temperature, 0.7; upstream's code overrides its own configured value with 0.
They use upstream's configured top-p of 0.95. They send no reasoning effort,
so a gpt-oss server applies its default, as it does for LLM-SR and LLM-ODE.
Upstream sets no output limit, and its GPT-4 model stopped at 4,096 tokens.
The plans' limit is higher because a reasoning model's thinking counts
against it.

The transport adds two things, and neither changes the method. The Jetstream2
hosted service replays a stored answer to a repeated request, so a request to
it asks for a fresh answer, and an answer it marks as replayed is counted as a
cache hit. A request the server fails is retried for the endpoint's patience,
so a service that is down costs no generation. An outage that outlasts that
patience stops the task without a result, and running the task again resumes
it after its last finished generation. Every reply is kept in the task's
directory, so a resumed task re-reads the replies it already has.
"""

from __future__ import annotations

import hashlib
import http.client
import importlib
import importlib.metadata
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from autoformalism.baselines.d3 import D3_UPSTREAM_REVISION
from autoformalism.baselines.d3_upstream import (
    SELECTION,
    D3ModelReply,
    D3Settings,
    run_d3_upstream,
)
from autoformalism.llm.exceptions import LLMProviderError, LLMResponseError
from autoformalism.llm.ollama import _ollama_compatible_schema
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.llm_call_log import CallLog
from autoformalism.rebuttal.llm_sr_shim import (
    LITELLM_CACHE_HIT_HEADER,
    LITELLM_NO_CACHE,
    http_error_detail,
)
from autoformalism.rebuttal.phase_b_d3 import discover_and_seal, summarize
from autoformalism.rebuttal.phase_c_baseline_plan import PhaseCBaselineCell
from autoformalism.rebuttal.phase_c_baselines import load_cell, tier_of
from autoformalism.rebuttal.phase_c_vendored_campaign import (
    JETSTREAM2_HOSTED_MODEL,
    EndpointKind,
    check_served_model,
    endpoint_environment,
    served_model_ids,
    verify_plan_release,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.phase_c_inputs import ROSTER
from autoformalism.schemas import normalize_proposer_candidate_v2_payload

PROTOCOL = "phase-c-d3-upstream-1"

#: Where this D3 cannot be upstream's, as every sealed plan records it.
DEPARTURES = (
    "models are equations in a restricted grammar instead of executed PyTorch "
    "code, so neural-network components cannot be expressed: D3's white-box "
    "mode",
    "prompts change only where they describe code or black-box components, "
    "and they state the update the fitting uses: x_next = x + rhs, no dt "
    "multiplier",
    "a generation without a usable model is recorded as failed; while no model "
    "exists, the next generation asks for a first model again",
    "feature acquisition is off: a Phase C cell has no features to acquire",
)

#: What a Phase C D3 summary states about every number in it.
LIMITATION = (
    "Phase C D3 as published, in its white-box mode: upstream's propose-reflect "
    "loop, prompts and search settings, with models written as equations in a "
    "restricted grammar instead of executed code, on development data only; "
    "the model with the lowest one-step validation loss, D3's own rule, is "
    "kept. Native one-step uses measured target histories; the rollout resets "
    "only supplied auxiliaries. No dt multiplier or ODE reinterpretation. "
    "Declared parameter bounds are audited, not imposed retroactively. No "
    "latent states or scientific/continuous-time certification. Validation is "
    "not an independent test estimate; failures remain visible."
)

#: How long one request may take. Phase B's job-local server answered within
#: the client's two minutes. The hosted service is shared, and on 2026-10-06 a
#: reply it cut off at the 8192-token limit took 46 s.
REQUEST_TIMEOUT_SECONDS = 900.0

#: First and longest pause between attempts at a failed request; the pause
#: doubles per attempt, so an outage is retried about once a minute.
BACKOFF_SECONDS = (1.0, 60.0)

#: Refusals that describe the request rather than the server. Retrying cannot
#: change them, so D3 sees them at once and records the generation as failed,
#: as it did in Phase B.
REQUEST_ERRORS = frozenset({400, 404, 413, 422})


class EndpointUnavailable(Exception):
    """The endpoint kept failing for longer than the campaign's patience.

    Deliberately not a RuntimeError or ValueError: D3 records those as a
    failed generation, and a service that is down is not something the method
    did. This stops the task instead, without a result.
    """


class KeptReplyMismatch(Exception):
    """A kept reply answers a different request than the one being made.

    Not a generation failure: it means the task's directory belongs to other
    code or another plan, so the task stops instead.
    """


class PhaseCD3Plan(BaseModel):
    """Everything that fixes a Phase C D3 campaign before its first call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-c-d3-plan-2"]
    status: Literal["proposed_pending_review", "frozen_before_calls"]
    development_only: Literal[True]
    release_protocol: Literal["phase-c-development-2"]
    release_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    purpose: str = Field(min_length=1)
    endpoint: EndpointKind
    model: str = Field(min_length=1)
    cells: tuple[PhaseCBaselineCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    #: Upstream config.yaml: generations, d3_patience, keep_top_samples.
    generations: int = Field(default=20, ge=1, le=50)
    patience: int = Field(default=20, ge=1, le=50)
    keep_top_samples: int = Field(default=16, ge=1, le=64)
    #: The paper's temperature; upstream's code sends 0 in place of its
    #: configured value.
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, allow_inf_nan=False)
    #: Upstream config.yaml: top_p.
    top_p: float = Field(default=0.95, gt=0.0, le=1.0, allow_inf_nan=False)
    #: None sends no reasoning effort, so the server applies its default.
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    max_output_tokens: int = Field(default=8192, ge=128, le=32768)
    trajectory_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    test_data_opened: Literal[False]
    private_reference_opened: Literal[False]

    @model_validator(mode="after")
    def roster_cells_and_a_servable_model(self) -> PhaseCD3Plan:
        """Require roster cells with their own tiers, and a model the endpoint has."""
        names = [cell.benchmark_id for cell in self.cells]
        if len(names) != len(set(names)):
            raise ValueError("Phase C campaign cells must be unique")
        for cell in self.cells:
            if cell.benchmark_id not in ROSTER:
                raise ValueError(f"{cell.benchmark_id} is outside the Phase C roster")
            if cell.tier != tier_of(cell.benchmark_id):
                raise ValueError(f"{cell.benchmark_id} declares the wrong tier")
        if len(self.repetitions) != len(set(self.repetitions)) or any(
            seed < 0 for seed in self.repetitions
        ):
            raise ValueError("repetitions must be unique and nonnegative")
        if self.patience < self.generations:
            raise ValueError("patience must cover the frozen generation budget")
        if ":" in self.model:
            raise ValueError("supply a model id without a provider prefix")
        if (
            self.endpoint == "jetstream2_hosted"
            and self.model != JETSTREAM2_HOSTED_MODEL
        ):
            raise ValueError(
                f"the Jetstream2 hosted endpoint serves only {JETSTREAM2_HOSTED_MODEL}"
            )
        return self

    def settings(self) -> D3Settings:
        """The loop's own budget."""
        return D3Settings(
            generations=self.generations,
            patience=self.patience,
            keep_top_samples=self.keep_top_samples,
        )

    def reporting_qualifications(self) -> tuple[str, ...]:
        """What a report must state beyond the D3 limitation itself."""
        if self.endpoint == "jetstream2_hosted":
            return (
                "served by the Jetstream2 hosted endpoint, whose model revision "
                "and serving software are not verifiable",
            )
        return ()


def load_plan(path: Path) -> PhaseCD3Plan:
    """Load one strict Phase C D3 plan."""
    return PhaseCD3Plan.model_validate_json(path.read_text(encoding="utf-8"))


def environment(endpoint: str) -> dict:
    """Bind resume to the code, the endpoint kind and D3's numerical libraries.

    D3 fits with torch, so a machine without it is refused before any call.
    """
    importlib.import_module("torch")
    return {
        **endpoint_environment(endpoint),
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pydantic", "openai")
        },
    }


def prepare_phase_c(config_path: Path, release: Path, root: Path) -> dict:
    """Freeze a Phase C D3 campaign against one verified release, before any call."""
    plan = load_plan(config_path)
    release = release.expanduser().resolve()
    receipt = verify_plan_release(plan, release)  # type: ignore[arg-type]
    rows: list[dict] = []
    for cell in plan.cells:
        loaded = load_cell(release, cell.benchmark_id)
        for repetition in plan.repetitions:
            rows.append(
                {
                    "index": len(rows),
                    "benchmark_id": cell.benchmark_id,
                    "tier": cell.tier,
                    "repetition": repetition,
                    "public_identity": loaded.identity,
                    "validation_context": loaded.context.model_dump(mode="json"),
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    return sealed_write(
        root / "plan.json",
        {
            "protocol": PROTOCOL,
            "plan": plan.model_dump(mode="json"),
            "model": plan.model,
            "release": str(release),
            "release_summary_sha256": receipt,
            "environment": environment(plan.endpoint),
            "upstream_revision": D3_UPSTREAM_REVISION,
            "loop": "upstream_reflection",
            "departures": list(DEPARTURES),
            "max_attempts": 1,
            # A first model, then a reflection and a model per generation.
            "maximum_logical_calls": len(rows) * (2 * plan.generations - 1),
            "reporting_qualifications": list(plan.reporting_qualifications()),
            "rows": rows,
            "test_data_opened": False,
            "private_reference_opened": False,
            "selection": SELECTION,
        },
    )


def _pause(seconds: float) -> None:
    """Wait before the next attempt; a seam so tests need not sleep."""
    time.sleep(seconds)


def _clock() -> float:
    """Seconds on a monotonic clock; a seam so tests can move time on."""
    return monotonic()


def _post(url: str, body: dict, timeout: float) -> tuple[dict, bool]:
    """Send one request; return the reply and whether a gateway replayed it."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
        replayed = response.headers.get(LITELLM_CACHE_HIT_HEADER) is not None
    if not isinstance(payload, dict):
        raise LLMResponseError("vLLM response must be a JSON object")
    return payload, replayed


def patient_transport(
    log: CallLog, *, patience_seconds: float, bypass_cache: bool
) -> Callable[[str, dict, float], dict]:
    """A chat-completions transport made to outlast a service that goes down.

    A request the server fails is logged and sent again, the pause doubling to
    a minute, until `patience_seconds` have passed since the first failure;
    then the task stops with ``EndpointUnavailable``. A refusal of the request
    itself, or a reply that is not JSON, reaches D3 at once, as in Phase B.
    `bypass_cache` asks a caching gateway for a fresh answer. Each reply
    carries, under ``_autoformalism_gateway``, whether it was replayed, which
    the accounting counts as a cache hit.
    """
    first, longest = BACKOFF_SECONDS

    def transport(url: str, body: dict, timeout: float) -> dict:
        sent = {**body, **LITELLM_NO_CACHE} if bypass_cache else body
        failing_since: float | None = None
        pause = first
        while True:
            try:
                payload, replayed = _post(url, sent, timeout)
            except urllib.error.HTTPError as exc:
                detail = http_error_detail(exc)
                failure = f"HTTP {exc.code}: {exc.reason}" + (
                    f" ({detail})" if detail else ""
                )
                if exc.code in REQUEST_ERRORS:
                    raise LLMProviderError(
                        f"vLLM {failure}", retryable=False
                    ) from exc
            except (OSError, http.client.HTTPException) as exc:
                # Refused or dropped connections and timeouts included.
                failure = f"{type(exc).__name__}: {exc}"
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise LLMResponseError(f"vLLM returned invalid JSON: {exc}") from exc
            else:
                payload["_autoformalism_gateway"] = {
                    "cache_bypass": bypass_cache,
                    "cache_hit": replayed,
                }
                return payload
            log.failure(failure)
            now = _clock()
            failing_since = now if failing_since is None else failing_since
            if now - failing_since >= patience_seconds:
                raise EndpointUnavailable(
                    f"the endpoint failed for {(now - failing_since) / 60:.0f} "
                    f"minutes; last: {failure}"
                )
            _pause(pause)
            pause = min(longest, 2 * pause)

    return transport


def _choice(payload: dict) -> dict:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _content(payload: dict) -> str:
    """The reply's text; a reply with none is the model's failure."""
    choice = _choice(payload)
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError(
            f"the reply has no text (finish_reason={choice.get('finish_reason')!r})"
        )
    return content


class EndpointChat:
    """D3's two calls to a declared endpoint, each kept for resume and logged.

    A reflection is plain text. A model is requested against
    ``D3ModelReply``'s strict JSON schema and read as our proposer contract
    is: lossless normalization of its parameter list, then strict validation.
    Every reply is kept under ``conversation/`` by generation and call, with a
    digest of the request it answers; a resumed task re-reads it instead of
    asking again, and the log records that as a cache hit.
    """

    def __init__(
        self,
        plan: PhaseCD3Plan,
        directory: Path,
        *,
        base_url: str,
        patience_seconds: float,
        bypass_cache: bool,
    ) -> None:
        self._plan = plan
        self._url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self._kept = directory / "conversation"
        self._log = CallLog(directory / "llm_calls.jsonl")
        self._transport = patient_transport(
            self._log, patience_seconds=patience_seconds, bypass_cache=bypass_cache
        )
        self._schema = _ollama_compatible_schema(
            D3ModelReply.model_json_schema(mode="validation")
        )

    def _body(self, messages: list[dict[str, str]], *, structured: bool) -> dict:
        body: dict[str, Any] = {
            "model": self._plan.model,
            "messages": messages,
            "stream": False,
            "temperature": self._plan.temperature,
            "top_p": self._plan.top_p,
            "max_tokens": self._plan.max_output_tokens,
        }
        if self._plan.reasoning_effort is not None:
            body["reasoning_effort"] = self._plan.reasoning_effort
        if structured:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": D3ModelReply.__name__,
                    "strict": True,
                    "schema": self._schema,
                },
            }
        return body

    def _call(self, role: str, generation: int, body: dict) -> dict:
        path = self._kept / f"g{generation:02d}-{role}.json"
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if path.exists():
            kept = json.loads(path.read_text(encoding="utf-8"))
            if kept["request_sha256"] != digest:
                raise KeptReplyMismatch(f"{path} answers a different request")
            payload = kept["response"]
            self._log.response(
                payload, finish_reasons=_finish_reasons(payload), cache_hit=True
            )
            return payload
        payload = self._transport(self._url, body, REQUEST_TIMEOUT_SECONDS)
        atomic_json(
            path,
            {
                "generation": generation,
                "role": role,
                "request_sha256": digest,
                "request": body,
                "response": payload,
            },
        )
        gateway = payload.get("_autoformalism_gateway") or {}
        self._log.response(
            payload,
            finish_reasons=_finish_reasons(payload),
            cache_hit=bool(gateway.get("cache_hit")),
        )
        return payload

    def reflect(self, messages: list[dict[str, str]], *, generation: int) -> str:
        """D3's reflection call: free text, no function to call."""
        body = self._body(messages, structured=False)
        return _content(self._call("reflection", generation, body))

    def write_model(
        self, messages: list[dict[str, str]], *, generation: int
    ) -> D3ModelReply:
        """D3's model call, read against the restricted format."""
        body = self._body(messages, structured=True)
        content = _content(self._call("model", generation, body))
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"the model reply is not JSON: {exc}") from exc
        if isinstance(value, dict) and isinstance(value.get("model"), dict):
            model, _ = normalize_proposer_candidate_v2_payload(value["model"])
            value = {**value, "model": model}
        try:
            return D3ModelReply.model_validate(value)
        except ValidationError as exc:
            raise LLMResponseError(f"the model reply failed its format: {exc}") from exc


def _finish_reasons(payload: dict) -> list[str]:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return []
    return [
        str(choice.get("finish_reason"))
        for choice in choices
        if isinstance(choice, dict)
    ]


def endpoint_chat(
    plan: PhaseCD3Plan,
    directory: Path,
    *,
    base_url: str,
    patience_seconds: float,
    bypass_cache: bool,
) -> EndpointChat:
    """D3's calls to the plan's endpoint for one task."""
    return EndpointChat(
        plan,
        directory,
        base_url=base_url,
        patience_seconds=patience_seconds,
        bypass_cache=bypass_cache,
    )


def run_phase_c(
    root: Path,
    index: int,
    *,
    endpoint: str,
    base_url: str,
    patience_seconds: float,
    bypass_cache: bool = False,
    chat: Any = None,
) -> dict:
    """Resume one Phase C D3 task, only against the endpoint kind it was frozen for.

    A sealed result returns without a call. Before the first call of a task
    with generations left, the endpoint must be serving the plan's model.
    """
    sealed = sealed_read(root / "plan.json")
    if sealed["protocol"] != PROTOCOL:
        raise ValueError(f"{root / 'plan.json'} is not a Phase C D3 plan")
    frozen = sealed["plan"]["endpoint"]
    if endpoint != frozen:
        raise ValueError(
            f"endpoint kind differs from the frozen plan: frozen for {frozen}, "
            f"running against {endpoint}. Run against the declared endpoint, or "
            "freeze a new plan in a new root."
        )
    if sealed["environment"] != environment(endpoint):
        raise ValueError(
            "protocol, code or dependencies changed since this plan was frozen. "
            f"Run from the checkout that froze it, or delete {root / 'plan.json'} "
            "and its results to re-freeze at the current code."
        )
    if not 0 <= index < len(sealed["rows"]):
        raise ValueError("task index out of range")
    row = sealed["rows"][index]
    cell = load_cell(Path(sealed["release"]), row["benchmark_id"])
    if cell.identity["release_summary_sha256"] != sealed["release_summary_sha256"]:
        raise ValueError("release receipt differs from the frozen plan")
    if (
        cell.identity != row["public_identity"]
        or cell.dataset.tier != row["tier"]
        or cell.context.model_dump(mode="json") != row["validation_context"]
    ):
        raise ValueError("public development input drift")
    plan = PhaseCD3Plan.model_validate(sealed["plan"])
    directory = root / "results" / str(index)
    searching = not any(
        (directory / name).exists()
        for name in ("result.json", "native-selection.json")
    )
    if chat is None and searching:
        # Outside the search, so a server without the model is not recorded
        # as a failed discovery.
        check_served_model(served_model_ids(base_url), plan.model)

    def discover():
        talker = chat
        if talker is None:
            talker = endpoint_chat(
                plan,
                directory,
                base_url=base_url,
                patience_seconds=patience_seconds,
                bypass_cache=bypass_cache,
            )
        return run_d3_upstream(
            cell.dataset,
            cell.context,
            task_prompt=cell.prompt,
            chat=talker,
            settings=plan.settings(),
            seed=row["repetition"],
            work_directory=directory,
            identity={"plan_sha256": sealed["artifact_sha256"], "task_index": index},
        )

    return discover_and_seal(
        sealed,
        index,
        cell.dataset,
        cell.context,
        directory=directory,
        trajectory_seconds=plan.trajectory_seconds,
        protocol=PROTOCOL,
        discover=discover,
        selection_metric=SELECTION,
    )


def report(root: Path) -> dict:
    """Keep missing and failed cells visible; state what the endpoint cannot prove."""
    sealed = sealed_read(root / "plan.json")
    if sealed["protocol"] != PROTOCOL:
        raise ValueError(f"{root / 'plan.json'} is not a Phase C D3 plan")
    statements = [LIMITATION, *sealed["reporting_qualifications"]]
    return summarize(
        root,
        protocol=PROTOCOL,
        limitation=" ".join(
            item if item.endswith(".") else f"{item[0].upper()}{item[1:]}."
            for item in statements
        ),
    )
