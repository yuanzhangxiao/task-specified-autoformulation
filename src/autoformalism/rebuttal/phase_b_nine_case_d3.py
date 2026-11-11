"""D3 and HDTwinGen on the nine Phase B cases of the main comparison.

Both run the loop ``d3_upstream`` ports from D3's release: a first model, then
generations that each reflect on the best models so far and write a new one,
native one-step fitting, and the model with the lowest one-step validation
loss, which is the loop's own selection rule. D3 runs as it does on Phase C
(``phase_c_d3``), in its white-box mode. HDTwinGen runs the same loop with its
own prompts, which allow black-box multilayer perceptrons beside white-box
terms (``baselines/hdtwingen.py``). The endpoint transport, the kept replies
and the sealed result with its recursive rollout check are Phase C D3's.

The inputs are the nine staged cells of ``phase_b_nine_case``: the exact
training and validation tables and prompts the compared methods saw, checked
against the digests the comparison's campaign recorded. Test data are never
staged or read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.baselines.d3 import D3_UPSTREAM_REVISION
from autoformalism.baselines.d3_upstream import (
    D3_LOOP,
    SELECTION,
    D3Settings,
    LoopVariant,
    run_d3_upstream,
)
from autoformalism.baselines.hdtwingen import (
    HDTWINGEN,
    HDTWINGEN_UPSTREAM_REVISION,
    HybridModelReply,
)
from autoformalism.rebuttal import phase_b_nine_case as nine
from autoformalism.rebuttal.phase_b_d3 import discover_and_seal, summarize
from autoformalism.rebuttal.phase_c_d3 import DEPARTURES as D3_DEPARTURES
from autoformalism.rebuttal.phase_c_d3 import EndpointChat, environment
from autoformalism.rebuttal.phase_c_vendored_campaign import (
    JETSTREAM2_HOSTED_MODEL,
    EndpointKind,
    check_served_model,
    served_model_ids,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write

PROTOCOL = "phase-b-nine-case-d3-family-1"

Method = Literal["d3", "hdtwingen"]

#: Where each method cannot be upstream's, as every sealed plan records it.
DEPARTURES: dict[str, tuple[str, ...]] = {
    "d3": (
        *(item for item in D3_DEPARTURES if not item.startswith("feature")),
        "feature acquisition is off: the comparison's cells have no features "
        "to acquire",
    ),
    "hdtwingen": (
        "models are equations in a restricted grammar instead of executed "
        "PyTorch code; each black-box multilayer perceptron is declared in a "
        "fixed form (input expressions in the grammar, 1 to 3 hidden layers of "
        "2 to 64 units, one of six activations, one output, at most four "
        "networks), and the harness builds and fits it",
        "prompts change only where they describe code, and they state the "
        "update the fitting uses: x_next = x + rhs, no dt multiplier",
        "a generation without a usable model is recorded as failed; while no "
        "model exists, the next generation asks for a first model again",
        "the fit runs in float64 on a CPU over all of a cell's trajectories at "
        "once; upstream uses float32 batches of up to 1,000 trajectories, more "
        "than a cell has",
    ),
}

#: The loop each method runs and the release it follows.
VARIANTS: dict[str, tuple[LoopVariant, str]] = {
    "d3": (D3_LOOP, D3_UPSTREAM_REVISION),
    "hdtwingen": (HDTWINGEN, HDTWINGEN_UPSTREAM_REVISION),
}

#: What a summary states about every number in it.
LIMITATIONS: dict[str, str] = {
    "d3": (
        "D3 as published, in its white-box mode, on the nine Phase B cases of "
        "the main comparison: upstream's propose-reflect loop, prompts and "
        "search settings, with models written as equations in a restricted "
        "grammar instead of executed code, on development data only; the model "
        "with the lowest one-step validation loss, D3's own rule, is kept."
    ),
    "hdtwingen": (
        "HDTwinGen on the nine Phase B cases of the main comparison: D3's "
        "propose-reflect loop and search settings with HDTwinGen's prompts, "
        "which allow black-box multilayer perceptrons beside white-box terms, "
        "declared in a fixed form and fitted by the harness; on development "
        "data only; the model with the lowest one-step validation loss, the "
        "loop's own rule, is kept."
    ),
}

ROLLOUT_LIMITATION = (
    "Native one-step uses measured target histories; the rollout resets only "
    "supplied auxiliaries. No dt multiplier or ODE reinterpretation. Declared "
    "parameter bounds are audited, not imposed retroactively. No latent states "
    "or scientific/continuous-time certification. Validation is not an "
    "independent test estimate; failures remain visible."
)


class NineCaseCell(BaseModel):
    """One of the nine cells, at its compared tier, with its prompt's digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str
    tier: Literal["easy", "hard"]
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class NineCaseD3Plan(BaseModel):
    """Everything that fixes a nine-case D3 or HDTwinGen campaign before a call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-nine-case-d3-plan-1"]
    status: Literal["proposed_pending_review", "frozen_before_calls"]
    development_only: Literal[True]
    inputs_protocol: Literal["phase-b-nine-case-inputs-1", "phase-c-development-2"]
    #: Required, and only allowed, when the plan reads the Phase C release.
    release_summary_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    purpose: str = Field(min_length=1)
    method: Method
    endpoint: EndpointKind
    model: str = Field(min_length=1)
    cells: tuple[NineCaseCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    #: Upstream config.yaml (D3's and HDTwinGen's agree): generations,
    #: patience and keep_top_samples.
    generations: int = Field(default=20, ge=1, le=50)
    patience: int = Field(default=20, ge=1, le=50)
    keep_top_samples: int = Field(default=16, ge=1, le=64)
    #: config.yaml's temperature and top-p; the code sends 0 in place of 0.7.
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, allow_inf_nan=False)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0, allow_inf_nan=False)
    #: None sends no reasoning effort, so the server applies its default.
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    max_output_tokens: int = Field(default=8192, ge=128, le=32768)
    trajectory_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    test_data_opened: Literal[False]
    private_reference_opened: Literal[False]

    @model_validator(mode="after")
    def nine_case_cells_and_a_servable_model(self) -> NineCaseD3Plan:
        """Require the plan's cells at their tiers, and a servable model."""
        names = [cell.benchmark_id for cell in self.cells]
        if len(names) != len(set(names)):
            raise ValueError("campaign cells must be unique")
        for cell in self.cells:
            nine.check_plan_cell(
                self.inputs_protocol,
                cell.benchmark_id,
                cell.tier,
                cell.public_prompt_sha256,
            )
        if (self.inputs_protocol == nine.PHASE_C) != (
            self.release_summary_sha256 is not None
        ):
            raise ValueError("a Phase C plan, and only one, names its release")
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
        """What a report must state beyond the method's limitation."""
        if self.endpoint == "jetstream2_hosted":
            return (
                "served by the Jetstream2 hosted endpoint, whose model revision "
                "and serving software are not verifiable",
            )
        return ()


class HybridEndpointChat(EndpointChat):
    """The endpoint chat, requesting and reading HDTwinGen's hybrid replies."""

    reply_type = HybridModelReply


def load_plan(path: Path) -> NineCaseD3Plan:
    """Load one strict nine-case D3-family plan."""
    return NineCaseD3Plan.model_validate_json(path.read_text(encoding="utf-8"))


def prepare(config_path: Path, inputs_root: Path, root: Path) -> dict:
    """Freeze a campaign against the verified staged inputs, before any call."""
    plan = load_plan(config_path)
    inputs_root = inputs_root.expanduser().resolve()
    protocol = plan.inputs_protocol
    receipt = nine.verify_inputs(
        protocol, inputs_root, [cell.benchmark_id for cell in plan.cells]
    )
    if protocol == nine.PHASE_C and receipt != plan.release_summary_sha256:
        raise ValueError("the release differs from the plan's")
    rows: list[dict] = []
    for cell in plan.cells:
        loaded = nine.load_plan_cell(protocol, inputs_root, cell.benchmark_id)
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
    variant, revision = VARIANTS[plan.method]
    root.mkdir(parents=True, exist_ok=True)
    return sealed_write(
        root / "plan.json",
        {
            "protocol": PROTOCOL,
            "plan": plan.model_dump(mode="json"),
            "method": plan.method,
            "model": plan.model,
            "inputs": str(inputs_root),
            "inputs_receipt_sha256": receipt,
            "environment": environment(plan.endpoint),
            "upstream_revision": revision,
            "loop": variant.name,
            "departures": list(DEPARTURES[plan.method]),
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


def run(
    root: Path,
    index: int,
    *,
    endpoint: str,
    base_url: str,
    patience_seconds: float,
    bypass_cache: bool = False,
    chat: Any = None,
) -> dict:
    """Resume one task, only against the endpoint kind it was frozen for.

    A sealed result returns without a call. Before the first call of a task
    with generations left, the endpoint must be serving the plan's model.
    """
    sealed = sealed_read(root / "plan.json")
    if sealed["protocol"] != PROTOCOL:
        raise ValueError(f"{root / 'plan.json'} is not a nine-case D3-family plan")
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
    cell = nine.load_plan_cell(
        sealed["plan"]["inputs_protocol"], Path(sealed["inputs"]), row["benchmark_id"]
    )
    if nine.receipt_of(cell) != sealed["inputs_receipt_sha256"]:
        raise ValueError("staged inputs differ from the frozen plan")
    if (
        cell.identity != row["public_identity"]
        or cell.dataset.tier != row["tier"]
        or cell.context.model_dump(mode="json") != row["validation_context"]
    ):
        raise ValueError("public development input drift")
    plan = NineCaseD3Plan.model_validate(sealed["plan"])
    variant, _ = VARIANTS[plan.method]
    directory = root / "results" / str(index)
    searching = not any(
        (directory / name).exists() for name in ("result.json", "native-selection.json")
    )
    if chat is None and searching:
        # Outside the search, so a server without the model is not recorded
        # as a failed discovery.
        check_served_model(served_model_ids(base_url), plan.model)

    def discover():
        talker = chat
        if talker is None:
            kind = HybridEndpointChat if plan.method == "hdtwingen" else EndpointChat
            talker = kind(
                plan,  # type: ignore[arg-type]
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
            variant=variant,
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
        raise ValueError(f"{root / 'plan.json'} is not a nine-case D3-family plan")
    scope = nine.scope_of(sealed["plan"]["inputs_protocol"])
    statements = [
        LIMITATIONS[sealed["method"]].replace(
            "the nine Phase B cases of the main comparison", scope
        ),
        ROLLOUT_LIMITATION,
        *sealed["reporting_qualifications"],
    ]
    return summarize(
        root,
        protocol=PROTOCOL,
        limitation=" ".join(
            item if item.endswith(".") else f"{item[0].upper()}{item[1:]}."
            for item in statements
        ),
    )
