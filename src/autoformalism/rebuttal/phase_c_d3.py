"""Run the D3-native baseline on Phase C public cells, against a declared endpoint.

The method is Phase B's D3 campaign, unchanged: the native prompt with its
declared runtime clarification, the generation and patience budget, native
fitting, validation selection, the recursive rollout check and one sealed
result, all through ``phase_b_d3.discover_and_seal``. As for LLM-SR and
LLM-ODE, two things change. The data come from a verified Phase C release, and
a plan declares where its model is served and names it.

D3 reaches the model through our strict-JSON vLLM client at the settings Phase
B ran it with: low reasoning effort, temperature 0, one attempt, the plan's
output limit. Its transport adds two things, and neither changes the method.
The Jetstream2 hosted service replays a stored answer to a repeated request, so
a request to it asks for a fresh answer, and an answer it marks as replayed is
counted as a cache hit. And D3 records a failed request as a failed generation,
which is right for a reply the model got wrong and wrong for a service that is
down, so a request the server fails is retried for the endpoint's patience
before D3 sees it. An outage that outlasts it stops the task without a result,
and running the task again resumes it from its last finished generation.
"""

from __future__ import annotations

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

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.baselines.d3 import D3_UPSTREAM_REVISION
from autoformalism.expressions import ValidationContext
from autoformalism.llm.config import VLLMReasoningEffort
from autoformalism.llm.exceptions import LLMProviderError, LLMResponseError
from autoformalism.llm.vllm import VLLMClient
from autoformalism.rebuttal.llm_call_log import CallLog
from autoformalism.rebuttal.llm_sr_shim import (
    LITELLM_CACHE_HIT_HEADER,
    LITELLM_NO_CACHE,
    http_error_detail,
)
from autoformalism.rebuttal.phase_b_d3 import (
    ADAPTATION,
    discover_and_seal,
    summarize,
)
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

PROTOCOL = "phase-c-d3-native-validation-1"

#: What a Phase C D3 summary states about every number in it; Phase B's
#: statement, for the Phase C cells.
LIMITATION = (
    "Phase C D3-native-no-tools on development data; validation-selected "
    "discrete models. Native one-step uses measured target histories; the "
    "rollout resets only supplied auxiliaries. No dt multiplier or ODE "
    "reinterpretation. Declared parameter bounds are audited, not imposed "
    "retroactively. No latent states or scientific/continuous-time "
    "certification. Validation is not an independent test estimate; failures "
    "remain visible."
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


class PhaseCD3Plan(BaseModel):
    """Everything that fixes a Phase C D3 campaign before its first call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-c-d3-plan-1"]
    status: Literal["proposed_pending_review", "frozen_before_calls"]
    development_only: Literal[True]
    release_protocol: Literal["phase-c-development-2"]
    release_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    purpose: str = Field(min_length=1)
    endpoint: EndpointKind
    model: str = Field(min_length=1)
    #: Phase B ran D3 at our vLLM client's low effort.
    reasoning_effort: Literal["low", "medium", "high"] = "low"
    cells: tuple[PhaseCBaselineCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    generations: int = Field(default=5, ge=1, le=20)
    patience: int = Field(default=5, ge=1, le=20)
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
            "adaptation": ADAPTATION,
            "max_attempts": 1,
            "maximum_logical_calls": len(rows) * plan.generations,
            "reporting_qualifications": list(plan.reporting_qualifications()),
            "rows": rows,
            "test_data_opened": False,
            "private_reference_opened": False,
            "selection": "native_one_step_validation",
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
    """Our vLLM client's transport, made to outlast a service that goes down.

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


def endpoint_client(
    plan: PhaseCD3Plan,
    context: ValidationContext,
    directory: Path,
    *,
    base_url: str,
    patience_seconds: float,
    bypass_cache: bool,
) -> VLLMClient:
    """Our vLLM client at Phase B's D3 settings, over the patient transport."""
    log_path = directory / "llm_calls.jsonl"
    return VLLMClient(
        model=plan.model,
        cache_directory=directory / "llm_cache",
        log_path=log_path,
        base_url=base_url,
        reasoning_effort=VLLMReasoningEffort(plan.reasoning_effort),
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        max_output_tokens=plan.max_output_tokens,
        max_attempts=1,
        proposal_target_channels=context.targets,
        transport=patient_transport(
            CallLog(log_path),
            patience_seconds=patience_seconds,
            bypass_cache=bypass_cache,
        ),
    )


def run_phase_c(
    root: Path,
    index: int,
    *,
    endpoint: str,
    base_url: str,
    patience_seconds: float,
    bypass_cache: bool = False,
    client: Any = None,
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
    if client is None and searching:
        # Outside the search, so a server without the model is not recorded
        # as a failed discovery.
        check_served_model(served_model_ids(base_url), plan.model)

    def native_client() -> Any:
        if client is not None:
            return client
        return endpoint_client(
            plan,
            cell.context,
            directory,
            base_url=base_url,
            patience_seconds=patience_seconds,
            bypass_cache=bypass_cache,
        )

    return discover_and_seal(
        sealed,
        index,
        cell.dataset,
        cell.context,
        directory=directory,
        generations=plan.generations,
        patience=plan.patience,
        trajectory_seconds=plan.trajectory_seconds,
        llm_model=f"vllm:{plan.model}",
        task_prompt=lambda: cell.prompt,
        native_client=native_client,
        protocol=PROTOCOL,
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
