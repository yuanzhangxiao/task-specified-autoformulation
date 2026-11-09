"""SINDy with library terms an LLM designs from the task's public specification.

One request shows the model a cell's public prompt and asks it for up to
``feature_count`` algebraic features of the observed channels, inputs and
time, which it writes in the restricted grammar (``llm_feature_sindy`` in
``baselines/runner.py``). SINDy then fits its usual library -- constant,
linear, pairwise products and tanh of every channel -- plus those features,
over the same twelve thresholds as the compared SINDy, and keeps the threshold
whose validation rollout error is lowest, the compared SINDy's own rule. Every
step after the request is the compared SINDy's path, run on the nine staged
cells of ``phase_b_nine_case``.

The request goes through Phase C D3's patient transport: it asks the hosted
gateway for a fresh answer, waits out an outage, and its reply is kept for
resume. A reply that does not fit the format leaves the library without
designed features; that is recorded, and SINDy still runs. The SINDy fit has
the compared SINDy's wall-clock limit, in a child process that is stopped
when the limit passes. Features are algebraic in current values, so the model
cannot introduce latent states or memory.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import multiprocessing
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from autoformalism.baselines.core import numeric_feature_names
from autoformalism.baselines.models import BaselineConfig, BaselineDevelopmentResult
from autoformalism.baselines.runner import _propose_features, run_baseline_development
from autoformalism.llm.exceptions import LLMResponseError
from autoformalism.llm.ollama import _ollama_compatible_schema
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal import phase_b_nine_case as nine
from autoformalism.rebuttal.llm_call_log import CallLog
from autoformalism.rebuttal.phase_b_d3 import accounting
from autoformalism.rebuttal.phase_b_nine_case_d3 import NineCaseCell
from autoformalism.rebuttal.phase_c_d3 import (
    REQUEST_TIMEOUT_SECONDS,
    KeptReplyMismatch,
    _content,
    _finish_reasons,
    patient_transport,
)
from autoformalism.rebuttal.phase_c_vendored_campaign import (
    JETSTREAM2_HOSTED_MODEL,
    EndpointKind,
    check_served_model,
    endpoint_environment,
    served_model_ids,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import (
    CandidateModel,
    ProposerCandidateV2,
    enrich_proposal_v2,
    normalize_proposer_candidate_v2_payload,
)

PROTOCOL = "phase-b-nine-case-spec-sindy-1"

#: The compared SINDy's threshold grid (Phase B full plan, method sindy).
COMPARED_THRESHOLDS = (
    1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 30.0, 100.0, 300.0, 1_000.0, 3_000.0, 10_000.0
)

LIMITATION = (
    "SINDy with up to the plan's number of features designed by an LLM from "
    "the cell's public prompt, in one request, added to the compared SINDy's "
    "library; the compared SINDy's thresholds, validation-rollout selection "
    "and wall-clock limit. Features are algebraic in current values, so no "
    "latent state or memory can be introduced. Development data only; "
    "validation is not an independent test estimate; failures and timeouts "
    "remain visible."
)


class SpecSindyPlan(BaseModel):
    """Everything that fixes a spec-informed SINDy campaign before its calls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-nine-case-spec-sindy-plan-1"]
    status: Literal["proposed_pending_review", "frozen_before_calls"]
    development_only: Literal[True]
    inputs_protocol: Literal["phase-b-nine-case-inputs-1"]
    purpose: str = Field(min_length=1)
    endpoint: EndpointKind
    model: str = Field(min_length=1)
    cells: tuple[NineCaseCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    feature_count: int = Field(default=8, ge=1, le=16)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, allow_inf_nan=False)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0, allow_inf_nan=False)
    #: None sends no reasoning effort, so the server applies its default.
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    max_output_tokens: int = Field(default=8192, ge=128, le=32768)
    sindy_thresholds: tuple[float, ...] = COMPARED_THRESHOLDS
    wall_timeout_seconds: float = Field(default=1_800.0, gt=0, allow_inf_nan=False)
    test_data_opened: Literal[False]
    private_reference_opened: Literal[False]

    @model_validator(mode="after")
    def nine_case_cells_and_a_servable_model(self) -> SpecSindyPlan:
        """Require the comparison's cells at their tiers, and a servable model."""
        inputs = nine.load_inputs()
        names = [cell.benchmark_id for cell in self.cells]
        if len(names) != len(set(names)):
            raise ValueError("campaign cells must be unique")
        for cell in self.cells:
            recorded = inputs.cells.get(cell.benchmark_id)
            if recorded is None:
                raise ValueError(f"{cell.benchmark_id} is not one of the nine cases")
            if cell.tier != recorded.tier:
                raise ValueError(f"{cell.benchmark_id} declares the wrong tier")
            if cell.public_prompt_sha256 != recorded.files["proposer_prompt.txt"]:
                raise ValueError(f"{cell.benchmark_id} declares another prompt")
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
        return self

    def baseline_config(self, seed: int) -> BaselineConfig:
        """The compared SINDy's configuration, with the designed features."""
        return BaselineConfig(
            method="llm_feature_sindy",
            seed=seed,
            llm_model=f"{self.endpoint}:{self.model}",
            sindy_thresholds=self.sindy_thresholds,
            wall_timeout_seconds=self.wall_timeout_seconds,
            llm_feature_count=self.feature_count,
        )

    def reporting_qualifications(self) -> tuple[str, ...]:
        """What a report must state beyond the method's limitation."""
        if self.endpoint == "jetstream2_hosted":
            return (
                "served by the Jetstream2 hosted endpoint, whose model revision "
                "and serving software are not verifiable",
            )
        return ()


def load_plan(path: Path) -> SpecSindyPlan:
    """Load one strict spec-informed SINDy plan."""
    return SpecSindyPlan.model_validate_json(path.read_text(encoding="utf-8"))


def prepare(config_path: Path, inputs_root: Path, root: Path) -> dict:
    """Freeze a campaign against the verified staged inputs, before any call."""
    plan = load_plan(config_path)
    inputs_root = inputs_root.expanduser().resolve()
    receipt = nine.verify(inputs_root)
    rows: list[dict] = []
    for cell in plan.cells:
        loaded = nine.load_cell(inputs_root, cell.benchmark_id, context="baseline")
        if loaded.identity["prompt"] != cell.public_prompt_sha256:
            raise ValueError(f"public prompt differs: {cell.benchmark_id}")
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
            "inputs": str(inputs_root),
            "inputs_receipt_sha256": receipt,
            "environment": endpoint_environment(plan.endpoint),
            "max_attempts": 1,
            "maximum_logical_calls": len(rows),
            "reporting_qualifications": list(plan.reporting_qualifications()),
            "rows": rows,
            "test_data_opened": False,
            "private_reference_opened": False,
            "selection": "lowest_validation_rollout_nmse_over_thresholds",
        },
    )


class _KeptFeatures:
    """Answers SINDy's one design call with the reply already kept."""

    def __init__(self, candidate) -> None:
        self._candidate = candidate

    def propose(self, *, system_prompt: str, user_prompt: str, cache_only=False):
        del system_prompt, user_prompt, cache_only
        return SimpleNamespace(parsed=self._candidate)


class _Recorded(Exception):
    """Raised by the recorder once the request has been read."""


def design_messages(plan: SpecSindyPlan, cell) -> list[dict[str, str]]:
    """The design request, exactly as ``_propose_features`` words it.

    The request is read off the method itself, by letting it call a recorder,
    so the wording cannot drift from what the method sends.
    """
    seen: dict[str, str] = {}

    class Recorder:
        def propose(self, *, system_prompt, user_prompt, cache_only=False):
            seen.update(system=system_prompt, user=user_prompt)
            raise _Recorded

    with contextlib.suppress(_Recorded):
        _propose_features(
            Recorder(),
            cell.prompt,
            cell.context,
            numeric_feature_names(cell.dataset.train, cell.context),
            feature_count=plan.feature_count,
        )
    return [
        {"role": "system", "content": seen["system"]},
        {"role": "user", "content": seen["user"]},
    ]


def _no_features(targets: tuple[str, ...]):
    """The library's own terms only: a candidate with no designed features."""
    return enrich_proposal_v2(
        ProposerCandidateV2.model_validate(
            {
                "candidate_id": "no_designed_features",
                "states": [
                    {"name": name, "kind": "observed", "observed_channel": name,
                     "rhs": "0"}
                    for name in targets
                ],
            }
        ),
        targets,
    )


def design_features(
    plan: SpecSindyPlan,
    cell,
    directory: Path,
    *,
    base_url: str,
    patience_seconds: float,
    bypass_cache: bool,
    transport=None,
) -> dict:
    """Make or re-read the one design request; return what it yielded.

    The reply is kept in ``conversation/design.json`` with a digest of the
    request it answers, so a resumed task re-reads it and the log counts that
    as a cache hit.
    """
    messages = design_messages(plan, cell)
    schema = _ollama_compatible_schema(
        ProposerCandidateV2.model_json_schema(mode="validation")
    )
    body: dict[str, Any] = {
        "model": plan.model,
        "messages": messages,
        "stream": False,
        "temperature": plan.temperature,
        "top_p": plan.top_p,
        "max_tokens": plan.max_output_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": ProposerCandidateV2.__name__,
                "strict": True,
                "schema": schema,
            },
        },
    }
    if plan.reasoning_effort is not None:
        body["reasoning_effort"] = plan.reasoning_effort
    log = CallLog(directory / "llm_calls.jsonl")
    path = directory / "conversation" / "design.json"
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if path.exists():
        kept = json.loads(path.read_text(encoding="utf-8"))
        if kept["request_sha256"] != digest:
            raise KeptReplyMismatch(f"{path} answers a different request")
        payload = kept["response"]
        log.response(payload, finish_reasons=_finish_reasons(payload), cache_hit=True)
    else:
        send = transport or patient_transport(
            log, patience_seconds=patience_seconds, bypass_cache=bypass_cache
        )
        payload = send(
            f"{base_url.rstrip('/')}/v1/chat/completions", body, REQUEST_TIMEOUT_SECONDS
        )
        atomic_json(
            path,
            {"request_sha256": digest, "request": body, "response": payload},
        )
        gateway = payload.get("_autoformalism_gateway") or {}
        log.response(
            payload,
            finish_reasons=_finish_reasons(payload),
            cache_hit=bool(gateway.get("cache_hit")),
        )
    try:
        value = json.loads(_content(payload))
        normalized, _ = normalize_proposer_candidate_v2_payload(value)
        candidate = enrich_proposal_v2(
            ProposerCandidateV2.model_validate(normalized), cell.context.targets
        )
        return {"candidate": candidate.model_dump(mode="json"), "reply_error": None}
    except (LLMResponseError, ValidationError, ValueError, TypeError) as exc:
        return {
            "candidate": _no_features(cell.context.targets).model_dump(mode="json"),
            "reply_error": f"{type(exc).__name__}: {str(exc)[:2000]}",
        }


def _fit_worker(root: str, index: int, config: str) -> None:
    """Fit SINDy for one task from its kept design; write development.json.

    Runs in a fresh process of its own, so the parent passes the inputs
    record it verified against, and this process reads everything with it.
    """
    nine.CONFIG = Path(config)
    root_path = Path(root)
    sealed = sealed_read(root_path / "plan.json")
    row = sealed["rows"][index]
    plan = SpecSindyPlan.model_validate(sealed["plan"])
    cell = nine.load_cell(
        Path(sealed["inputs"]),
        row["benchmark_id"],
        context="baseline",
        config=Path(config),
    )
    if cell.identity != row["public_identity"]:
        raise ValueError("public development input drift")
    directory = root_path / "results" / str(index)
    design = json.loads((directory / "design.json").read_text(encoding="utf-8"))
    candidate = CandidateModel.model_validate(design["candidate"])
    result = run_baseline_development(
        plan.baseline_config(row["repetition"]),
        cell.dataset,
        cell.context,
        llm_client=_KeptFeatures(candidate),
        proposer_prompt=cell.prompt,
    )
    atomic_json(directory / "development.json", result.model_dump(mode="json"))


def _fit_with_limit(root: Path, index: int, seconds: float) -> tuple[str, float, str]:
    """Run the fit in a child process; stop it when the limit passes."""
    context = multiprocessing.get_context("spawn")
    started = time.monotonic()
    process = context.Process(
        target=_fit_worker, args=(str(root), index, str(nine.CONFIG))
    )
    process.start()
    process.join(seconds)
    elapsed = time.monotonic() - started
    if process.is_alive():
        process.kill()
        process.join()
        return "timed_out", elapsed, "SINDy wall-clock limit reached"
    if process.exitcode != 0:
        return "failed", elapsed, f"the fit exited with status {process.exitcode}"
    return "complete", elapsed, ""


def run(
    root: Path,
    index: int,
    *,
    endpoint: str,
    base_url: str,
    patience_seconds: float,
    bypass_cache: bool = False,
    transport=None,
) -> dict:
    """Resume one task: design the features once, then fit SINDy under its limit."""
    sealed = sealed_read(root / "plan.json")
    if sealed["protocol"] != PROTOCOL:
        raise ValueError(f"{root / 'plan.json'} is not a spec-informed SINDy plan")
    frozen = sealed["plan"]["endpoint"]
    if endpoint != frozen:
        raise ValueError(
            f"endpoint kind differs from the frozen plan: frozen for {frozen}, "
            f"running against {endpoint}"
        )
    if sealed["environment"] != endpoint_environment(endpoint):
        raise ValueError(
            "protocol, code or dependencies changed since this plan was frozen. "
            f"Run from the checkout that froze it, or delete {root / 'plan.json'} "
            "and its results to re-freeze at the current code."
        )
    if not 0 <= index < len(sealed["rows"]):
        raise ValueError("task index out of range")
    row = sealed["rows"][index]
    plan = SpecSindyPlan.model_validate(sealed["plan"])
    cell = nine.load_cell(
        Path(sealed["inputs"]), row["benchmark_id"], context="baseline"
    )
    if cell.identity["inputs_receipt_sha256"] != sealed["inputs_receipt_sha256"]:
        raise ValueError("staged inputs differ from the frozen plan")
    if (
        cell.identity != row["public_identity"]
        or cell.context.model_dump(mode="json") != row["validation_context"]
    ):
        raise ValueError("public development input drift")
    directory = root / "results" / str(index)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    with (directory / "task.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if result_path.exists():
            result = sealed_read(result_path)
            if result["plan_sha256"] != sealed["artifact_sha256"]:
                raise ValueError("result belongs to a different plan")
            return result
        design_path = directory / "design.json"
        if not design_path.exists():
            if transport is None and not (directory / "conversation").exists():
                check_served_model(served_model_ids(base_url), plan.model)
            design = design_features(
                plan,
                cell,
                directory,
                base_url=base_url,
                patience_seconds=patience_seconds,
                bypass_cache=bypass_cache,
                transport=transport,
            )
            atomic_json(design_path, design)
        design = json.loads(design_path.read_text(encoding="utf-8"))
        (directory / "development.json").unlink(missing_ok=True)
        status, elapsed, error = _fit_with_limit(
            root, index, plan.wall_timeout_seconds
        )
        outcome: dict[str, Any] = {"status": status, "elapsed_wall_seconds": elapsed}
        if status == "complete":
            development = BaselineDevelopmentResult.model_validate_json(
                (directory / "development.json").read_text(encoding="utf-8")
            )
            outcome.update(
                validation_normalized_mse=development.validation_normalized_mse,
                training_normalized_mse=development.training_normalized_mse,
                equations=development.equations,
                selected_hyperparameters=development.selected_hyperparameters,
            )
        else:
            outcome["error"] = error
        return sealed_write(
            result_path,
            {
                **row,
                **outcome,
                "protocol": PROTOCOL,
                "plan_sha256": sealed["artifact_sha256"],
                "designed_features": [
                    {"name": item["name"], "expression": item["expression"]}
                    for item in design["candidate"].get("processes", [])
                ],
                "design_reply_error": design["reply_error"],
                "accounting": accounting(directory / "llm_calls.jsonl"),
                "wall_timeout_seconds": plan.wall_timeout_seconds,
                "test_data_opened": False,
                "private_reference_opened": False,
                "selection_metric": sealed["selection"],
            },
        )


def report(root: Path) -> dict:
    """Keep pending, failed and timed-out tasks visible; median of finite scores."""
    sealed = sealed_read(root / "plan.json")
    if sealed["protocol"] != PROTOCOL:
        raise ValueError(f"{root / 'plan.json'} is not a spec-informed SINDy plan")
    rows = []
    for task in sealed["rows"]:
        path = root / "results" / str(task["index"]) / "result.json"
        if path.exists():
            row = sealed_read(path)
            if row["plan_sha256"] != sealed["artifact_sha256"]:
                raise ValueError("result belongs to a different plan")
        else:
            row = {**task, "status": "pending"}
        rows.append(row)
    counts = dict(Counter(row["status"] for row in rows))
    values = [
        row["validation_normalized_mse"]
        for row in rows
        if row["status"] == "complete"
    ]
    totals: Counter = Counter()
    for row in rows:
        totals.update(row.get("accounting", {}))
    statements = [LIMITATION, *sealed["reporting_qualifications"]]
    value = {
        "protocol": PROTOCOL,
        "plan_sha256": sealed["artifact_sha256"],
        "status": "pending" if counts.get("pending") else "complete",
        "model": sealed["model"],
        "expected": len(rows),
        "counts": counts,
        "metrics": {
            "validation_rollout": {
                "finite": len(values),
                "median_normalized_mse": float(np.median(values)) if values else None,
                "finite_only": True,
            }
        },
        "accounting": dict(totals),
        "rows": rows,
        "test_data_opened": False,
        "private_reference_opened": False,
        "limitation": " ".join(
            item if item.endswith(".") else f"{item[0].upper()}{item[1:]}."
            for item in statements
        ),
    }
    atomic_json(root / "summary.json", value)
    return value
