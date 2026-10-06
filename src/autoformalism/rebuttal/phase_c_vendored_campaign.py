"""Run the vendored LLM baselines on Phase C public cells, against a named endpoint.

LLM-SR and LLM-ODE keep the Phase B campaign machinery: the same sealed plan,
rows, search, selection and sealed result. Two things change. The data come
from a verified Phase C release instead of the legacy registry, and a plan
declares where its model is served, because the same campaign can run against
a vLLM started inside a cluster job, a vLLM on a Jetstream2 VM, or the
Jetstream2 hosted service. A task resumes only against the kind it was frozen
for. The plan names the model; nothing here chooses one. An LLM-SR plan may
also declare how LLM-SR is adapted to a reasoning model, which a report must
then state.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, get_args
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.data.models import DevelopmentDataset
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.llm_sr_shim import DEFAULT_MAX_TOKENS, http_error_detail
from autoformalism.rebuttal.phase_c_baseline_plan import PhaseCBaselineCell
from autoformalism.rebuttal.phase_c_baselines import load_cell, tier_of, verify_release
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.rebuttal.vendored_campaign import (
    DeclaredBudget,
    PromptPolicy,
    UpstreamRevision,
    VendoredMethod,
    campaign_qualifications,
)
from autoformalism.research.phase_c_inputs import ROSTER

#: Where a campaign's model answers. Resume binds to the kind, never to an
#: address: a job-local port changes from job to job.
EndpointKind = Literal["job_local_vllm", "vm_local_vllm", "jetstream2_hosted"]

#: The environment each kind contributes to a frozen plan. The job-local entry
#: is the value Phase B plans were frozen with.
ENDPOINT_IDENTITIES: dict[str, dict[str, str]] = {
    "job_local_vllm": {"provider": "vllm", "endpoint_kind": "job-local vllm endpoint"},
    "vm_local_vllm": {"provider": "vllm", "endpoint_kind": "vm-local vllm endpoint"},
    "jetstream2_hosted": {
        "provider": "jetstream2",
        "endpoint_kind": "jetstream2 hosted endpoint",
    },
}

#: The hosted route answers only from Jetstream2 networks and needs no key.
JETSTREAM2_HOSTED_BASE_URL = "https://llm.jetstream-cloud.org/gpt-oss-120b"
#: The one model that route serves, under the alias the service lists.
JETSTREAM2_HOSTED_MODEL = "gpt-oss-120b"

#: How long a search waits out an endpoint that keeps failing before the task
#: is recorded as an infrastructure failure. A vLLM we started does not come
#: back once it has died. The hosted service does: it went offline for a while
#: on 2026-10-04 (its authentication database was unreachable), and LLM-SR
#: cannot resume a stopped search, so the hosted kind is given hours.
OUTAGE_PATIENCE_SECONDS: dict[str, float] = {
    "job_local_vllm": 15 * 60.0,
    "vm_local_vllm": 15 * 60.0,
    "jetstream2_hosted": 6 * 60 * 60.0,
}

#: Endpoint kinds that replay a stored answer to a repeated request unless a
#: request says otherwise. The hosted service is a LiteLLM gateway (1.98.0 on
#: 2026-10-05): an identical request came back in under a millisecond with
#: the first answer's id, and on 2026-10-06 the responses route LLM-ODE calls
#: did the same. Both upstreams sample afresh for every request; LLM-SR's
#: islands begin from the same prompt, and two LLM-ODE islands can ask the
#: same thing, so both methods' requests ask the gateway to generate anew.
CACHING_ENDPOINTS = frozenset({"jetstream2_hosted"})

_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


def endpoint_environment(kind: str) -> dict[str, str]:
    """Bind resume to the code and the endpoint kind, never to a port."""
    return {"runtime_source_sha256": runtime_source_hash(), **ENDPOINT_IDENTITIES[kind]}


def resolve_endpoint(kind: str, base_url: str | None) -> tuple[str, str]:
    """Check that an address is the kind of endpoint it is declared to be.

    The hosted service has one fixed route, which is also the default. A local
    kind must be on this machine, so a plan frozen for a model we serve can
    never quietly run against the hosted service, or the reverse.
    """
    if kind not in ENDPOINT_IDENTITIES:
        raise ValueError(
            f"unknown endpoint kind {kind!r}; declare one of "
            f"{', '.join(get_args(EndpointKind))}"
        )
    if kind == "jetstream2_hosted":
        url = (base_url or JETSTREAM2_HOSTED_BASE_URL).rstrip("/")
        if url != JETSTREAM2_HOSTED_BASE_URL:
            raise ValueError(
                f"the Jetstream2 hosted endpoint is {JETSTREAM2_HOSTED_BASE_URL}, "
                f"not {url}"
            )
        return kind, url
    if not base_url:
        raise ValueError(f"a {kind} endpoint needs its base URL")
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or parts.hostname not in _LOOPBACK:
        raise ValueError(
            f"a {kind} endpoint is served on this machine; {base_url} is not "
            "a loopback address"
        )
    return kind, base_url.rstrip("/")


def served_model_ids(base_url: str, *, timeout: float = 30.0) -> tuple[str, ...]:
    """List the models an OpenAI-compatible endpoint serves, in its order."""
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/models", headers={"Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = http_error_detail(exc)
        raise ValueError(
            f"cannot list the models served at {base_url}: {exc}"
            + (f" ({detail})" if detail else "")
        ) from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise ValueError(f"cannot list the models served at {base_url}: {exc}") from exc
    data = value.get("data") if isinstance(value, dict) else None
    if not isinstance(data, list):
        raise ValueError(f"{base_url} did not return a model list")
    return tuple(str(item.get("id")) for item in data if isinstance(item, dict))


def check_served_model(served: tuple[str, ...], model: str) -> None:
    """Refuse an endpoint that does not serve the plan's model.

    Both methods name the plan's model in every request: LLM-SR as upstream
    does, LLM-ODE in place of upstream's first listed model. So the model must
    be served, and a gateway may serve others beside it.
    """
    if model not in served:
        raise ValueError(
            f"the plan's model {model!r} is not among the served models {served}"
        )


class ReasoningModelAdaptation(BaseModel):
    """What LLM-SR needs to read a reasoning model's replies, declared.

    LLM-SR was published with models that answer at once. A reasoning model
    such as gpt-oss reasons before it answers, within the same token limit,
    and writes its function header over several lines. Upstream's 512-token
    default and its one-line reading of a header then leave almost every
    sample empty, so the comparison would measure the mismatch rather than
    the method.

    Both changes act in the transport, before upstream reads a reply. The
    search is upstream's: prompts, islands, sampling controls, evaluator,
    scoring and selection, which is why ``upstream.search_modified`` stays
    false while this block is reported beside it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Generation limit per sample, in place of their engine's 512.
    max_new_tokens: int = Field(ge=1, le=32768)
    #: Read a header the model split over several lines as one line
    #: (``llm_sr_shim.join_split_header``).
    join_split_headers: bool
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def departs_from_upstream(self) -> ReasoningModelAdaptation:
        """An adaptation that changes nothing is not declared at all."""
        if self.max_new_tokens == DEFAULT_MAX_TOKENS and not self.join_split_headers:
            raise ValueError(
                "this adaptation keeps upstream's behaviour; omit it instead"
            )
        return self

    def qualification(self) -> str:
        """The statement a report of an adapted campaign carries."""
        changes = []
        if self.max_new_tokens != DEFAULT_MAX_TOKENS:
            changes.append(
                f"up to {self.max_new_tokens} new tokens per sample against "
                f"upstream's {DEFAULT_MAX_TOKENS}"
            )
        if self.join_split_headers:
            changes.append(
                "a function header the model split over several lines is read "
                "as one line"
            )
        return (
            "adapted to a reasoning model: "
            + "; ".join(changes)
            + "; the search itself is upstream's"
        )


def llm_sr_transport_settings(plan: dict) -> dict[str, int | bool]:
    """The generation limit and header reading a sealed LLM-SR plan declares.

    A plan without an adaptation runs as upstream's engine would.
    """
    declared = plan.get("reasoning_model_adaptation")
    if declared is None:
        return {"max_new_tokens": DEFAULT_MAX_TOKENS, "join_split_headers": False}
    adaptation = ReasoningModelAdaptation.model_validate(declared)
    return {
        "max_new_tokens": adaptation.max_new_tokens,
        "join_split_headers": adaptation.join_split_headers,
    }


class PhaseCVendoredCampaignPlan(BaseModel):
    """Everything that fixes a Phase C vendored campaign before its first call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-c-vendored-campaign-plan-1"]
    status: Literal["proposed_pending_review", "frozen_before_calls"]
    development_only: Literal[True]
    release_protocol: Literal["phase-c-development-2"]
    release_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    method: VendoredMethod
    upstream: UpstreamRevision
    budget: DeclaredBudget
    prompt_policy: PromptPolicy
    #: As in Phase B: part of the sealed identity, never read from the
    #: environment at run time.
    islands: int = Field(default=4, ge=1, le=64)
    scope_note: str = ""
    #: The Phase C roster, so partial coverage is stated against it.
    full_design_conditions: int = Field(default=len(ROSTER), gt=0)
    endpoint: EndpointKind = "job_local_vllm"
    model: str = Field(min_length=1)
    #: LLM-SR only; absent means upstream's own generation limit and reading.
    reasoning_model_adaptation: ReasoningModelAdaptation | None = None
    cells: tuple[PhaseCBaselineCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    execution_semantics: Literal["continuous_ode_free_rollout"]
    #: Both campaigns differentiate with the fourth-order stencil in
    #: ``cell_arrays``, so no other provenance can be declared truthfully.
    derivative_provenance: Literal["upstream_findiff_fourth_order"]
    test_data_opened: Literal[False]
    private_reference_opened: Literal[False]

    @model_validator(mode="after")
    def cells_are_the_roster(self) -> PhaseCVendoredCampaignPlan:
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
        if ":" in self.model:
            raise ValueError("supply a model id without a provider prefix")
        if (
            self.endpoint == "jetstream2_hosted"
            and self.model != JETSTREAM2_HOSTED_MODEL
        ):
            raise ValueError(
                f"the Jetstream2 hosted endpoint serves only {JETSTREAM2_HOSTED_MODEL}"
            )
        if self.reasoning_model_adaptation is not None and self.method != "llm_sr":
            raise ValueError("a reasoning-model adaptation is declared for LLM-SR only")
        return self

    @property
    def expected_task_count(self) -> int:
        """One task per planned cell and repetition."""
        return len(self.cells) * len(self.repetitions)

    def reporting_qualifications(self) -> tuple[str, ...]:
        """The Phase B statements, plus what a hosted endpoint cannot prove."""
        notes = list(
            campaign_qualifications(
                budget=self.budget,
                prompt_policy=self.prompt_policy,
                cells=len(self.cells),
                full_design_conditions=self.full_design_conditions,
                scope_note=self.scope_note,
            )
        )
        if self.endpoint == "jetstream2_hosted":
            notes.append(
                "served by the Jetstream2 hosted endpoint, whose model revision "
                "and serving software are not verifiable"
            )
        if self.reasoning_model_adaptation is not None:
            notes.append(self.reasoning_model_adaptation.qualification())
        return tuple(notes)


def load_phase_c_vendored_plan(path: Path) -> PhaseCVendoredCampaignPlan:
    """Load one strict Phase C vendored campaign plan."""
    return PhaseCVendoredCampaignPlan.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def verify_plan_release(plan: PhaseCVendoredCampaignPlan, release: Path) -> str:
    """Return the release receipt, once it and every cell's prompt match the plan."""
    receipt = verify_release(release, tuple(cell.benchmark_id for cell in plan.cells))
    if receipt != plan.release_summary_sha256:
        raise ValueError("release receipt differs from the campaign plan")
    for cell in plan.cells:
        prompt = release / "public" / cell.benchmark_id / "proposer_prompt.txt"
        if hashlib.sha256(prompt.read_bytes()).hexdigest() != cell.public_prompt_sha256:
            raise ValueError(f"public prompt differs: {cell.benchmark_id}")
    return receipt


def require_endpoint(sealed: dict, root: Path, endpoint: str) -> None:
    """Resume only against the endpoint kind and code the plan was frozen with."""
    frozen = sealed["plan"]["endpoint"]
    if endpoint != frozen:
        raise ValueError(
            f"endpoint kind differs from the frozen plan: frozen for {frozen}, "
            f"running against {endpoint}. Run against the declared endpoint, or "
            "freeze a new plan in a new root."
        )
    if sealed["environment"] != endpoint_environment(endpoint):
        raise ValueError(
            "protocol or code changed since this plan was frozen. Run from the "
            f"checkout that froze it, or delete {root / 'plan.json'} and its "
            "results to re-freeze at the current code."
        )


def load_development(
    sealed: dict, row: dict
) -> tuple[DevelopmentDataset, ValidationContext]:
    """Reload one frozen cell, refusing any release or input but the frozen one."""
    cell = load_cell(Path(sealed["release"]), row["benchmark_id"])
    if cell.identity["release_summary_sha256"] != sealed["release_summary_sha256"]:
        raise ValueError("release receipt differs from the frozen plan")
    if cell.identity != row["public_identity"] or cell.dataset.tier != row["tier"]:
        raise ValueError("public development input drift")
    return cell.dataset, cell.context


def set_aside_unfinished(directory: Path) -> Path | None:
    """Move an attempt that stopped before sealing a result out of its task's way.

    The attempt is kept beside the task, for its cost and its logs. Restarting
    into the same directory would let the new search's model be chosen from
    samples, or its usage counted from calls, that the stopped attempt wrote.
    """
    if not directory.is_dir() or (directory / "result.json").exists():
        return None
    if not any(directory.iterdir()):
        return None
    number = 1
    while (
        target := directory.with_name(f"{directory.name}.interrupted-{number}")
    ).exists():
        number += 1
    directory.rename(target)
    return target


@contextmanager
def exclusive_attempt(root: Path, index: int) -> Iterator[None]:
    """Hold one task for this process, and start it clean after an interruption.

    Neither upstream search can resume part-way, and on a VM no scheduler stops
    the same task from being started twice. A second process for a task is
    refused while the first holds its lock, which also makes it safe to set an
    unfinished attempt aside: only a stopped attempt can be left unlocked.
    """
    locks = root / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{index}.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                f"task {index} under {root} is already running in another process"
            ) from None
        set_aside_unfinished(root / "results" / str(index))
        yield
