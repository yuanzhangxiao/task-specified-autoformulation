"""User-facing experiment assembly shared by command-line entry scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autoformalism.config import DataConfig
from autoformalism.data import (
    BenchmarkLoader,
    BenchmarkRegistry,
    DatasetSplit,
    DevelopmentDataset,
    FrozenTestAccess,
    SplitName,
    Trajectory,
)
from autoformalism.expressions import ValidationContext
from autoformalism.fitting import FitConfig
from autoformalism.judging import HybridScoringConfig, extract_public_requirements
from autoformalism.judging.prompts import (
    ATOMIC_EVIDENCE_PROMPT,
    ATOMIC_STAGE_TWO_NOTE,
    HYBRID_JUDGE_PROMPT,
)
from autoformalism.llm import (
    LLMClient,
    LLMConfig,
    LLMProvider,
    MockLLMClient,
    OllamaThinking,
    VLLMReasoningEffort,
    create_llm_client,
)
from autoformalism.pruning import PruningConfig
from autoformalism.schemas import CandidateModel, ScientificJudgeResult
from autoformalism.search import FinalEvaluation, SearchConfig, SearchController
from autoformalism.search.hybrid_pair import PairedHybridJudge
from autoformalism.targets import PublicTargetContract

_CONTROLLER_PROMPT = """
Return exactly one complete ProposerCandidateV2. Treat every current target value as a
generated output. When the runtime contract lists lagged targets, they may appear
on right-hand sides and mean the measured value from the immediately preceding
sample; current and future target values remain prohibited. Use only declared
auxiliaries, external inputs, fixed covariates, states, processes, and parameters.
Expressions must follow the restricted grammar. Use the beam feedback to make a
structurally meaningful exploratory proposal; do not merely tune numeric values.
Changing only component or parameter names is a structural duplicate and will be
rejected. Change at least one dependency, operator, state, or algebraic mechanism
relative to the supplied beam equations.
Each item in `states` contains its derivative expression in `rhs`; therefore every
declared state must have dynamics. An observed state must set `observed_channel`
and omit `initial`; a latent state must omit `observed_channel` and provide
`initial` with exactly one `fixed_value` or analytic `expression`. Put non-dynamic
named formulas in `algebraics`. Target outputs are inferred by matching each
benchmark target to an `observed_channel` or same-named state/algebraic; do not
emit `target_channel`. Put any qualitative or bounded constraint inside the
declaration it governs. Use `mechanisms` to tag the task-required scientific
mechanisms implemented by each state or algebraic.
Every constraint must remain attached to the state or algebraic it governs. Do not
create constraints for prose mechanism labels or other undeclared concepts.
Every declared parameter must appear in at least one state RHS, algebraic, or
initialization expression; omit declarations that are not used. Refer to modeled
states and parameters as bare symbols such as `I`, not as calls such as `I(t)`.
Named function calls must have explicit safe mathematical runtime support; never
use a function call as an undeclared mechanism or as a substitute for a state.
The only function names are `abs`, `exp`, `log`, `max`, `min`, `sigmoid`,
`softplus`, `sqrt`, and `tanh`. These names and the runtime time symbol are
reserved and cannot be state, process, or parameter names. Use Python `**` for
powers; `^` is forbidden. Do not use undeclared aliases such as `sigma`.
Do not redeclare supplied auxiliary, external-input, or fixed-covariate channel
names as states, processes, or parameters. Map each observed target exactly once:
never create both an observed state and a same-named algebraic for one target.
Do not invent numeric bounds for states or other variables. Parameter bounds are
required; `initialization_range` is optional and defaults to the parameter bounds.
For states, use qualitative `nonnegative` or `positive` constraints
without `bounds`; use a `bounded` constraint only when the benchmark supplies the
numeric range explicitly.
Division by a quantity whose range includes zero is permitted and guarded with a
small sign-preserving epsilon at runtime. Unsafe log and sqrt domains are rejected.
For balance equations, give production and input terms source signs and
utilization, elimination, and outflow terms sink signs. Check the RHS behavior at
any declared state boundary; do not rely on parameter fitting to reverse a fixed
structural sign.
Natural-language concepts in the benchmark prompt are descriptions, not expression
identifiers. Every expression symbol must exactly match a name in the runtime
symbol contract below or a state, process, or parameter declared in this proposal.
""".strip()

_JUDGE_CONTROLLER_PROMPT = """
This runtime uses the prospective scientific judge rubric, schema version 2.
Any earlier version-1 category names, weights, validity caps, or response example
in the benchmark judge prompt are historical and superseded by the response
schema and instructions below.

The candidate is the canonical runtime model, not the smaller proposer schema.
Read dynamics from `state_equations`, target outputs from `observation_mappings`,
instantaneous definitions from `processes`, and initialization rules from
`initial_conditions`. Supplied auxiliaries, external inputs, fixed covariates,
lagged targets, and time are defined by the exact runtime symbol contract and do
not require governing equations.

The request contains a list of deterministic certifications. Treat those facts as
authoritative and do not rescore syntax, closure, symbol availability, mappings,
causal channel access, parameter bounds, or runtime executability. Evaluate only:
mechanistic coherence; source/sink and balance semantics; dynamic plausibility;
mechanism coupling and task sufficiency; nonredundancy and accounting; and the
scientific justification of latent states and complexity.

Look specifically for fixed sign mistakes, duplicated fluxes, disconnected
task-critical mechanisms, one-signed accumulators without scientific
justification, missing relaxation or feedback, conflicting representations of
one mechanism, and latent states without a necessary interpretable role. Assess
dimensional plausibility only when the supplied units are informative; otherwise
state that unit evidence is insufficient. Do not infer fit quality, hidden
trajectories, reference equations, or private benchmark facts. Red flags are
advisory and must cite exact candidate equations or dependencies. The runtime
computes the weighted aggregate deterministically; return only category scores,
red flags, missing scientific requirements, and actionable scientific edits.
""".strip()


@dataclass(frozen=True)
class ExecutionArguments:
    """Normalized command-line values for one experiment."""

    data_root: Path
    benchmark_id: str
    tier: str
    seed: int
    proposer_model: str | None
    judge_model: str | None
    iteration_budget: int
    beam_size: int
    output_root: Path
    resume: bool
    dry_run: bool
    mock_llm: bool
    use_clean_observations: bool
    llm_timeout_seconds: float = 900.0
    llm_max_output_tokens: int = 2048
    fit_starts: int = 1
    fit_max_nfev: int = 50
    fit_timeout_seconds: float = 300.0
    final_fit_max_nfev: int = 150
    final_fit_timeout_seconds: float = 300.0
    use_judge: bool = True
    forbid_latent_states: bool = False
    use_derivative_fit_fast_path: bool = True
    llm_cache_only: bool = False
    llm_cache_root: Path | None = None
    require_initial_proposer_cache_hit: bool = False
    development_only: bool = False
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_thinking: OllamaThinking = OllamaThinking.AUTO
    ollama_temperature: float = 0.0
    ollama_seed: int | None = None
    stagnation_iterations: int | None = None
    selection_policy: str = "validation_only"
    judge_weight: float = 0.25
    judge_score_epsilon: float = 0.05
    hybrid_science_weight: float = 0.5
    hybrid_judge_seed_base: int | None = None
    public_target_contract: Path | None = None
    vllm_base_url: str = "http://127.0.0.1:8000"
    vllm_reasoning_effort: VLLMReasoningEffort = VLLMReasoningEffort.LOW
    vllm_proposer_reasoning_effort: VLLMReasoningEffort | None = None
    vllm_judge_reasoning_effort: VLLMReasoningEffort | None = None
    vllm_temperature: float = 0.0
    vllm_seed: int | None = None


class _RoleClient:
    """Delegate proposer and judge calls to independently configured clients."""

    def __init__(self, proposer: LLMClient, judge: LLMClient) -> None:
        self._proposer = proposer
        self._judge = judge

    def propose(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        cache_only: bool = False,
    ):
        return self._proposer.propose(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cache_only=cache_only,
        )

    def judge(self, *, system_prompt: str, user_prompt: str):
        return self._judge.judge(system_prompt=system_prompt, user_prompt=user_prompt)


def build_experiment_parser(
    *,
    description: str,
    default_resume: bool = False,
) -> argparse.ArgumentParser:
    """Build the common run/resume command-line parser."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("AUTOFORMALISM_DATA_ROOT", "data_raw")),
    )
    parser.add_argument("--benchmark-id", default="original_b1")
    parser.add_argument("--tier", choices=("easy", "medium", "hard"), default="easy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--proposer-model",
        help="provider:model, for example openai:gpt-5.2 or ollama:gpt-oss:20b",
    )
    parser.add_argument(
        "--judge-model",
        help="provider:model; defaults to --proposer-model",
    )
    parser.add_argument("--iteration-budget", type=int, default=5)
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument(
        "--selection-policy",
        choices=(
            "validation_only",
            "normalized_weighted_sum",
            "incumbent_relative_hybrid",
        ),
        default="validation_only",
        help=(
            "candidate ranking policy; normalized weighted selection uses "
            "standalone judge scores, while incumbent-relative hybrid uses the "
            "frozen paired-question-consensus development protocol"
        ),
    )
    parser.add_argument(
        "--judge-weight",
        type=float,
        default=0.25,
        help="nonnegative judge-penalty weight for normalized weighted selection",
    )
    parser.add_argument(
        "--judge-score-epsilon",
        type=float,
        default=0.05,
        help="positive guard added inside the negative-log judge penalty",
    )
    parser.add_argument(
        "--hybrid-science-weight",
        type=float,
        default=0.5,
        help=(
            "single fit/science tradeoff in [0, 1] for the development-only "
            "incumbent-relative hybrid policy"
        ),
    )
    parser.add_argument(
        "--hybrid-judge-seed-base",
        type=int,
        help=(
            "first of two distinct paired-orientation vLLM judge seeds; "
            "defaults deterministically from --seed"
        ),
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=float,
        default=900.0,
        help="provider request timeout; especially useful for local Ollama models",
    )
    parser.add_argument(
        "--llm-max-output-tokens",
        type=int,
        default=2048,
        help="maximum tokens generated by each proposer or judge call",
    )
    parser.add_argument(
        "--fit-starts",
        type=int,
        default=1,
        help="number of deterministic bounded least-squares starts",
    )
    parser.add_argument(
        "--fit-max-nfev",
        type=int,
        default=50,
        help="maximum residual evaluations per fitting start",
    )
    parser.add_argument(
        "--fit-timeout-seconds",
        type=float,
        default=300.0,
        help=(
            "wall-clock limit for one candidate fit; timeout rejects only "
            "that candidate"
        ),
    )
    parser.add_argument(
        "--final-fit-max-nfev",
        type=int,
        default=150,
        help="maximum residual evaluations for the warm-started final refit",
    )
    parser.add_argument(
        "--final-fit-timeout-seconds",
        type=float,
        default=300.0,
        help="wall-clock limit for the adaptive final refit",
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=default_resume,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help=(
            "stop after validation selection and train-plus-validation refit; "
            "never open or evaluate the test split"
        ),
    )
    parser.add_argument(
        "--llm-cache-only",
        action="store_true",
        help="fail closed on any LLM cache miss; never contact a provider",
    )
    parser.add_argument(
        "--llm-cache-root",
        type=Path,
        help="shared flat cache directory used by proposer and judge",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        help="Ollama HTTP endpoint; allows collision-free local GPU workers",
    )
    parser.add_argument(
        "--ollama-thinking",
        choices=tuple(item.value for item in OllamaThinking),
        default=OllamaThinking.AUTO.value,
        help=(
            "Ollama reasoning control; auto uses low for GPT-OSS and off for "
            "other models"
        ),
    )
    parser.add_argument(
        "--ollama-temperature",
        type=float,
        default=0.0,
        help="Ollama sampling temperature in [0, 2]",
    )
    parser.add_argument(
        "--ollama-seed",
        type=int,
        help="nonnegative Ollama sampling seed; omitted by default",
    )
    parser.add_argument(
        "--vllm-base-url",
        default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000"),
        help="vLLM OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--vllm-reasoning-effort",
        choices=tuple(item.value for item in VLLMReasoningEffort),
        default=VLLMReasoningEffort.LOW.value,
        help=(
            "legacy shared vLLM reasoning effort; role-specific options "
            "override it"
        ),
    )
    parser.add_argument(
        "--vllm-proposer-reasoning-effort",
        choices=tuple(item.value for item in VLLMReasoningEffort),
        help="vLLM reasoning effort used only for proposer calls",
    )
    parser.add_argument(
        "--vllm-judge-reasoning-effort",
        choices=tuple(item.value for item in VLLMReasoningEffort),
        help="vLLM reasoning effort used only for judge calls",
    )
    parser.add_argument("--vllm-temperature", type=float, default=0.0)
    parser.add_argument("--vllm-seed", type=int)
    parser.add_argument(
        "--stagnation-iterations",
        type=int,
        help=(
            "consecutive fitted rounds without improvement before stopping; "
            "defaults to the existing budget-dependent policy"
        ),
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="disable judge calls and feedback for the no-judge ablation",
    )
    parser.add_argument(
        "--require-initial-proposer-cache-hit",
        action="store_true",
        help=(
            "require round zero to restore the proposer response from the shared "
            "cache without making a provider request"
        ),
    )
    parser.add_argument(
        "--public-target-contract",
        type=Path,
        help=(
            "prompt-committed deterministic target contract; candidates that "
            "violate it are rejected before fitting"
        ),
    )
    parser.add_argument(
        "--forbid-latent-states",
        action="store_true",
        help=(
            "forbid unobserved dynamic states while retaining algebraic "
            "processes and the ordinary iterative search pipeline"
        ),
    )
    parser.add_argument(
        "--disable-derivative-fit-fast-path",
        action="store_true",
        help="force generic rollout fitting even when derivative regression applies",
    )
    return parser


def arguments_from_namespace(namespace: argparse.Namespace) -> ExecutionArguments:
    """Validate and normalize parsed command-line arguments."""
    if namespace.iteration_budget < 1:
        raise SystemExit("--iteration-budget must be at least 1")
    if namespace.beam_size < 1:
        raise SystemExit("--beam-size must be at least 1")
    if namespace.judge_weight < 0.0:
        raise SystemExit("--judge-weight must be nonnegative")
    if namespace.judge_score_epsilon <= 0.0:
        raise SystemExit("--judge-score-epsilon must be positive")
    if not 0.0 <= namespace.hybrid_science_weight <= 1.0:
        raise SystemExit("--hybrid-science-weight must be in [0, 1]")
    if namespace.selection_policy == "normalized_weighted_sum" and namespace.no_judge:
        raise SystemExit("--selection-policy normalized_weighted_sum requires judge")
    if namespace.require_initial_proposer_cache_hit and not namespace.no_judge:
        raise SystemExit(
            "--require-initial-proposer-cache-hit is only valid with --no-judge"
        )
    if (
        namespace.require_initial_proposer_cache_hit
        and namespace.llm_cache_root is None
    ):
        raise SystemExit(
            "--require-initial-proposer-cache-hit requires --llm-cache-root"
        )
    if namespace.selection_policy == "incumbent_relative_hybrid":
        if namespace.no_judge:
            raise SystemExit(
                "--selection-policy incumbent_relative_hybrid requires judge"
            )
        if not namespace.development_only:
            raise SystemExit(
                "--selection-policy incumbent_relative_hybrid requires "
                "--development-only"
            )
        if namespace.beam_size != 1:
            raise SystemExit(
                "--selection-policy incumbent_relative_hybrid requires --beam-size 1"
            )
        if namespace.mock_llm:
            raise SystemExit(
                "--selection-policy incumbent_relative_hybrid does not support "
                "--mock-llm"
            )
        judge_model = namespace.judge_model or namespace.proposer_model
        if judge_model is not None and not judge_model.startswith("vllm:"):
            raise SystemExit(
                "--selection-policy incumbent_relative_hybrid requires a "
                "vllm: judge model"
            )
    if namespace.llm_timeout_seconds <= 0:
        raise SystemExit("--llm-timeout-seconds must be positive")
    if namespace.llm_max_output_tokens < 128:
        raise SystemExit("--llm-max-output-tokens must be at least 128")
    if not 0.0 <= namespace.ollama_temperature <= 2.0:
        raise SystemExit("--ollama-temperature must be between 0 and 2")
    if not 0.0 <= namespace.vllm_temperature <= 2.0:
        raise SystemExit("--vllm-temperature must be between 0 and 2")
    if namespace.vllm_seed is not None and namespace.vllm_seed < 0:
        raise SystemExit("--vllm-seed must be nonnegative")
    if (
        namespace.hybrid_judge_seed_base is not None
        and namespace.hybrid_judge_seed_base < 0
    ):
        raise SystemExit("--hybrid-judge-seed-base must be nonnegative")
    if namespace.ollama_seed is not None and namespace.ollama_seed < 0:
        raise SystemExit("--ollama-seed must be nonnegative")
    if (
        namespace.stagnation_iterations is not None
        and namespace.stagnation_iterations < 1
    ):
        raise SystemExit("--stagnation-iterations must be at least 1")
    if namespace.fit_starts < 1:
        raise SystemExit("--fit-starts must be at least 1")
    if namespace.fit_max_nfev < 1:
        raise SystemExit("--fit-max-nfev must be at least 1")
    if namespace.fit_timeout_seconds <= 0:
        raise SystemExit("--fit-timeout-seconds must be positive")
    if namespace.final_fit_max_nfev < 1:
        raise SystemExit("--final-fit-max-nfev must be at least 1")
    if namespace.final_fit_timeout_seconds <= 0:
        raise SystemExit("--final-fit-timeout-seconds must be positive")
    if (
        not namespace.mock_llm
        and not namespace.dry_run
        and not namespace.proposer_model
    ):
        raise SystemExit("--proposer-model is required unless --mock-llm is used")
    return ExecutionArguments(
        data_root=namespace.data_root.expanduser().resolve(),
        benchmark_id=namespace.benchmark_id,
        tier=namespace.tier,
        seed=namespace.seed,
        proposer_model=namespace.proposer_model,
        judge_model=namespace.judge_model or namespace.proposer_model,
        iteration_budget=namespace.iteration_budget,
        beam_size=namespace.beam_size,
        output_root=namespace.output_root.expanduser().resolve(),
        resume=namespace.resume,
        dry_run=namespace.dry_run,
        mock_llm=namespace.mock_llm,
        use_clean_observations=namespace.clean,
        llm_timeout_seconds=namespace.llm_timeout_seconds,
        llm_max_output_tokens=namespace.llm_max_output_tokens,
        fit_starts=namespace.fit_starts,
        fit_max_nfev=namespace.fit_max_nfev,
        fit_timeout_seconds=namespace.fit_timeout_seconds,
        final_fit_max_nfev=namespace.final_fit_max_nfev,
        final_fit_timeout_seconds=namespace.final_fit_timeout_seconds,
        use_judge=not namespace.no_judge,
        forbid_latent_states=namespace.forbid_latent_states,
        use_derivative_fit_fast_path=(not namespace.disable_derivative_fit_fast_path),
        llm_cache_only=namespace.llm_cache_only,
        llm_cache_root=(
            None
            if namespace.llm_cache_root is None
            else namespace.llm_cache_root.expanduser().resolve()
        ),
        require_initial_proposer_cache_hit=(
            namespace.require_initial_proposer_cache_hit
        ),
        development_only=namespace.development_only,
        ollama_base_url=namespace.ollama_base_url,
        ollama_thinking=OllamaThinking(namespace.ollama_thinking),
        ollama_temperature=namespace.ollama_temperature,
        ollama_seed=namespace.ollama_seed,
        stagnation_iterations=namespace.stagnation_iterations,
        selection_policy=namespace.selection_policy,
        judge_weight=namespace.judge_weight,
        judge_score_epsilon=namespace.judge_score_epsilon,
        hybrid_science_weight=namespace.hybrid_science_weight,
        hybrid_judge_seed_base=namespace.hybrid_judge_seed_base,
        public_target_contract=(
            None
            if namespace.public_target_contract is None
            else namespace.public_target_contract.expanduser().resolve()
        ),
        vllm_base_url=namespace.vllm_base_url,
        vllm_reasoning_effort=VLLMReasoningEffort(
            namespace.vllm_reasoning_effort
        ),
        vllm_proposer_reasoning_effort=(
            None
            if namespace.vllm_proposer_reasoning_effort is None
            else VLLMReasoningEffort(namespace.vllm_proposer_reasoning_effort)
        ),
        vllm_judge_reasoning_effort=(
            None
            if namespace.vllm_judge_reasoning_effort is None
            else VLLMReasoningEffort(namespace.vllm_judge_reasoning_effort)
        ),
        vllm_temperature=namespace.vllm_temperature,
        vllm_seed=namespace.vllm_seed,
    )


def execute(arguments: ExecutionArguments) -> dict[str, Any]:
    """Validate inputs, optionally dry-run, then execute or resume one run."""
    dataset, test_loader, proposer_prompt, judge_prompt = _load_inputs(arguments)
    public_target_contract = _load_public_target_contract(
        arguments,
        proposer_prompt=proposer_prompt,
        targets=dataset.roles.targets,
    )
    benchmark_spec = (
        None
        if arguments.benchmark_id == "synthetic"
        else BenchmarkRegistry().get(arguments.benchmark_id)
    )
    use_derivative_fit_fast_path = arguments.use_derivative_fit_fast_path and (
        benchmark_spec is None
        or benchmark_spec.data_layout == "legacy_split_files"
    )
    experiment_directory = _experiment_directory(arguments)
    checkpoint_directory = experiment_directory / "checkpoints"
    plan = {
        "benchmark_id": dataset.benchmark_id,
        "tier": dataset.tier,
        "seed": arguments.seed,
        "targets": list(dataset.roles.targets),
        "auxiliaries": list(dataset.roles.auxiliaries),
        "prediction_protocol": (
            "one_step_ahead"
            if _context(arguments, dataset).lagged_targets
            else "open_loop"
        ),
        "iteration_budget": arguments.iteration_budget,
        "beam_size": arguments.beam_size,
        "selection_policy": arguments.selection_policy,
        "judge_weight": arguments.judge_weight,
        "judge_score_epsilon": arguments.judge_score_epsilon,
        "hybrid_science_weight": arguments.hybrid_science_weight,
        "hybrid_judge_seed_base": (
            arguments.hybrid_judge_seed_base
            if arguments.hybrid_judge_seed_base is not None
            else 12000 + 2 * arguments.seed
        ),
        "hybrid_judge_contract": (
            {
                "model": arguments.judge_model,
                "reasoning_effort": _judge_reasoning_effort(arguments).value,
                "temperature": 0.2,
                "max_output_tokens": 6144,
                "max_provider_attempts_per_stage": 10,
                "paired_seed_attempts": 2,
                "orientations_per_attempt": 2,
                "logical_stages_per_orientation": 2,
                "aggregation": "paired_question_consensus",
            }
            if arguments.selection_policy == "incumbent_relative_hybrid"
            else None
        ),
        "public_target_contract": (
            None
            if public_target_contract is None
            else {
                "path": str(arguments.public_target_contract),
                "schema_version": public_target_contract.schema_version,
                "public_prompt_sha256": (
                    public_target_contract.public_prompt_sha256
                ),
                "contract_sha256": hashlib.sha256(
                    arguments.public_target_contract.read_bytes()
                ).hexdigest(),
            }
        ),
        "llm_timeout_seconds": arguments.llm_timeout_seconds,
        "llm_max_output_tokens": arguments.llm_max_output_tokens,
        "fit_starts": arguments.fit_starts,
        "fit_max_nfev": arguments.fit_max_nfev,
        "fit_timeout_seconds": arguments.fit_timeout_seconds,
        "final_fit_max_nfev": arguments.final_fit_max_nfev,
        "final_fit_timeout_seconds": arguments.final_fit_timeout_seconds,
        "screening_integrator": "fixed_rk4",
        "final_integrator": "solve_ivp",
        "proposer_model": arguments.proposer_model,
        "judge_model": arguments.judge_model,
        "mock_llm": arguments.mock_llm,
        "llm_cache_only": arguments.llm_cache_only,
        "llm_cache_root": (
            None if arguments.llm_cache_root is None else str(arguments.llm_cache_root)
        ),
        "require_initial_proposer_cache_hit": (
            arguments.require_initial_proposer_cache_hit
        ),
        "development_only": arguments.development_only,
        "ollama_base_url": arguments.ollama_base_url,
        "ollama_thinking": arguments.ollama_thinking.value,
        "ollama_temperature": arguments.ollama_temperature,
        "ollama_seed": arguments.ollama_seed,
        "vllm_base_url": arguments.vllm_base_url,
        "vllm_reasoning_effort": arguments.vllm_reasoning_effort.value,
        "vllm_proposer_reasoning_effort": (
            _proposer_reasoning_effort(arguments).value
        ),
        "vllm_judge_reasoning_effort": _judge_reasoning_effort(arguments).value,
        "vllm_temperature": arguments.vllm_temperature,
        "vllm_seed": arguments.vllm_seed,
        "stagnation_iterations": (
            arguments.stagnation_iterations
            if arguments.stagnation_iterations is not None
            else max(2, min(5, arguments.iteration_budget))
        ),
        "use_judge": arguments.use_judge,
        "forbid_latent_states": arguments.forbid_latent_states,
        "use_derivative_fit_fast_path": use_derivative_fit_fast_path,
        "experiment_directory": str(experiment_directory),
        "split_fingerprints": {
            "train": dataset.train.fingerprint,
            "validation": dataset.validation.fingerprint,
        },
    }
    if arguments.dry_run:
        return {"status": "dry_run", **plan}

    run_metadata = checkpoint_directory / "run.json"
    if arguments.resume and not run_metadata.exists():
        raise SystemExit(
            f"cannot resume; checkpoint does not exist: {checkpoint_directory}"
        )
    if not arguments.resume and run_metadata.exists():
        raise SystemExit(f"run already exists at {experiment_directory}; pass --resume")
    experiment_directory.mkdir(parents=True, exist_ok=True)
    context = _context(arguments, dataset)
    client = _make_client(arguments, dataset, experiment_directory)
    pairwise_judge = _make_pairwise_judge(
        arguments,
        experiment_directory,
        context,
        proposer_prompt,
    )
    symbol_contract = _symbol_contract(context)
    protocol_prompt = _prediction_protocol_prompt(context)
    search_config = SearchConfig(
        checkpoint_directory=checkpoint_directory,
        maximum_iterations=arguments.iteration_budget,
        beam_size=arguments.beam_size,
        stagnation_iterations=(
            arguments.stagnation_iterations
            if arguments.stagnation_iterations is not None
            else max(2, min(5, arguments.iteration_budget))
        ),
        validation_mse_target=0.0,
        cheap_prefit_judge=False,
        use_judge=arguments.use_judge,
        require_initial_proposer_cache_hit=(
            arguments.require_initial_proposer_cache_hit
        ),
        selection_policy=arguments.selection_policy,
        judge_weight=arguments.judge_weight,
        judge_score_epsilon=arguments.judge_score_epsilon,
        hybrid_science_weight=arguments.hybrid_science_weight,
        evaluate_test=not arguments.development_only,
        proposer_system_prompt=(
            f"Configured proposer model: {arguments.proposer_model or 'mock'}\n\n"
            f"{proposer_prompt}\n\n{protocol_prompt}\n\n{symbol_contract}\n\n"
            f"Controller requirements:\n{_CONTROLLER_PROMPT}"
            f"{_latent_ablation_prompt(arguments)}"
        ),
        judge_system_prompt=(
            f"Configured judge model: {arguments.judge_model or 'mock'}\n\n"
            f"Proposer task:\n{proposer_prompt}\n\n{protocol_prompt}\n\n"
            f"{symbol_contract}\n\n"
            f"Judge task:\n{judge_prompt}\n\n"
            f"Runtime judge amendment:\n{_JUDGE_CONTROLLER_PROMPT}"
        ),
        fit_config=FitConfig(
            number_of_starts=arguments.fit_starts,
            random_seed=arguments.seed,
            integration_backend="fixed_rk4",
            maximum_function_evaluations=arguments.fit_max_nfev,
            maximum_wall_time_seconds=arguments.fit_timeout_seconds,
            allow_derivative_regression=use_derivative_fit_fast_path,
        ),
        final_fit_config=FitConfig(
            number_of_starts=arguments.fit_starts,
            random_seed=arguments.seed,
            integration_backend="solve_ivp",
            maximum_function_evaluations=arguments.final_fit_max_nfev,
            maximum_wall_time_seconds=arguments.final_fit_timeout_seconds,
            allow_derivative_regression=use_derivative_fit_fast_path,
        ),
        pruning_config=PruningConfig(),
    )
    result = SearchController(
        llm_client=client,
        context=context,
        training=dataset.train,
        validation=dataset.validation,
        test_loader=test_loader,
        config=search_config,
        pairwise_judge=pairwise_judge,
        public_target_contract=public_target_contract,
    ).run()
    summary = _result_summary(arguments, result)
    (experiment_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (experiment_directory / "run_config.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_public_target_contract(
    arguments: ExecutionArguments,
    *,
    proposer_prompt: str,
    targets: tuple[str, ...],
) -> PublicTargetContract | None:
    """Load and bind an optional target contract to the exact public task."""
    path = arguments.public_target_contract
    if path is None:
        return None
    if not path.is_file():
        raise SystemExit(f"public target contract is missing: {path}")
    try:
        contract = PublicTargetContract.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise SystemExit(f"invalid public target contract: {path}: {exc}") from exc
    if (contract.benchmark_id, contract.tier) != (
        arguments.benchmark_id,
        arguments.tier,
    ):
        raise SystemExit("public target contract does not match benchmark/tier")
    observed_targets = tuple(item.target_channel for item in contract.targets)
    if set(observed_targets) != set(targets):
        raise SystemExit("public target contract does not match benchmark targets")
    prompt_sha256 = hashlib.sha256(proposer_prompt.encode("utf-8")).hexdigest()
    if contract.public_prompt_sha256 != prompt_sha256:
        raise SystemExit("public target contract does not match proposer prompt")
    return contract


def _load_inputs(
    arguments: ExecutionArguments,
) -> tuple[
    DevelopmentDataset,
    Callable[[Any], DatasetSplit],
    str,
    str,
]:
    if arguments.benchmark_id == "synthetic":
        if not arguments.data_root.is_dir():
            raise SystemExit(f"data root is not a directory: {arguments.data_root}")
        return (
            _synthetic_development_dataset(),
            lambda _frozen: _synthetic_test_split(),
            "Generate a one-state continuous-time decay model for target.",
            "Require a closed causal state equation and target observation mapping.",
        )
    data_config = DataConfig(
        root=arguments.data_root,
        benchmark_id=arguments.benchmark_id,
        tier=arguments.tier,
        use_clean_observations=arguments.use_clean_observations,
    )
    registry = BenchmarkRegistry()
    loader = BenchmarkLoader(registry)
    dataset = loader.load_development(data_config)
    loader.validate_test_paths(data_config)
    spec = registry.get(arguments.benchmark_id)
    prompt_root = (arguments.data_root / spec.relative_root).resolve()
    if spec.data_layout == "legacy_split_files":
        prompt_root = (
            prompt_root / spec.tier_directory_template.format(tier=arguments.tier)
        ).resolve()
    proposer_path = prompt_root / "proposer_prompt.txt"
    judge_path = prompt_root / "judge_prompt.txt"
    for path in (proposer_path, judge_path):
        if not path.is_file():
            raise SystemExit(f"benchmark prompt is missing: {path}")
        if not path.is_relative_to(arguments.data_root):
            raise SystemExit(f"benchmark prompt escapes data root: {path}")
    return (
        dataset,
        lambda frozen: loader.load_test(
            data_config,
            access=FrozenTestAccess(
                benchmark_id=data_config.benchmark_id,
                tier=data_config.tier,
                selection_hash=frozen.selection_hash,
            ),
        ),
        proposer_path.read_text(encoding="utf-8"),
        judge_path.read_text(encoding="utf-8"),
    )


def _context(
    arguments: ExecutionArguments,
    dataset: DevelopmentDataset,
) -> ValidationContext:
    if arguments.benchmark_id == "synthetic":
        return ValidationContext(targets=("target",))
    spec = BenchmarkRegistry().get(arguments.benchmark_id)
    forcing_bounds = _development_forcing_bounds(
        dataset, include_targets=spec.one_step_target_history
    )
    return ValidationContext(
        targets=dataset.roles.targets,
        auxiliaries=dataset.roles.auxiliaries,
        external_inputs=_numeric_declared_channels(
            spec.external_inputs, forcing_bounds
        ),
        fixed_covariates=_numeric_declared_channels(
            spec.fixed_covariates, forcing_bounds
        ),
        lagged_targets=(dataset.roles.targets if spec.one_step_target_history else ()),
        forcing_bounds=forcing_bounds,
        forbid_latent_states=arguments.forbid_latent_states,
    )


def _latent_ablation_prompt(arguments: ExecutionArguments) -> str:
    """Render the opt-in no-persistent-latent-dynamics restriction."""
    if not arguments.forbid_latent_states:
        return ""
    return (
        "\n\nAblation restriction: do not declare latent dynamic states. Every "
        "dynamic state must map to a benchmark target or a supplied auxiliary "
        "channel. Algebraic generated processes and permitted lagged targets "
        "remain allowed. External inputs and fixed covariates remain supplied "
        "forcing and must not be redeclared."
    )


def _numeric_declared_channels(
    names: tuple[str, ...], bounds: dict[str, tuple[float, float]]
) -> tuple[str, ...]:
    """Keep only declared channels with numeric development-data envelopes."""
    return tuple(name for name in names if name in bounds)


def _development_forcing_bounds(
    dataset: DevelopmentDataset,
    *,
    include_targets: bool = False,
) -> dict[str, tuple[float, float]]:
    """Compute finite forcing envelopes without opening the test split."""
    collected: dict[str, list[np.ndarray]] = {}
    for split in (dataset.train, dataset.validation):
        for trajectory in split.trajectories:
            channels: dict[str, Any] = {
                **trajectory.auxiliaries,
                **trajectory.external_inputs,
                **trajectory.fixed_covariates,
            }
            if include_targets:
                channels.update(trajectory.targets)
            for name, raw_values in channels.items():
                try:
                    values = np.asarray(raw_values, dtype=float).reshape(-1)
                except (TypeError, ValueError):
                    continue
                if len(values) and np.isfinite(values).all():
                    collected.setdefault(name, []).append(values)
    return {
        name: (
            float(min(np.min(values) for values in arrays)),
            float(max(np.max(values) for values in arrays)),
        )
        for name, arrays in collected.items()
    }


def _symbol_contract(context: ValidationContext) -> str:
    """Render exact expression symbols so semantic labels are not guessed."""
    lines = (
        ("Target channels (generate at each predicted slot)", context.targets),
        ("Causally available one-slot-lagged targets", context.lagged_targets),
        ("Supplied auxiliary trajectories", context.auxiliaries),
        ("External input trajectories", context.external_inputs),
        ("Fixed numeric covariates", context.fixed_covariates),
    )
    rendered = ["Exact runtime symbol contract:"]
    rendered.extend(
        f"- {label}: {', '.join(values) if values else '(none)'}"
        for label, values in lines
    )
    rendered.append(f"- Time symbol in expressions: {context.time_symbol}")
    rendered.append(
        "Use only these exact identifiers for supplied data. Natural-language "
        "descriptions must not be converted into new identifier-like aliases. If "
        "no matching identifier is listed, the concept is not a separate supplied "
        "variable; represent it only through the listed channels."
    )
    rendered.append(
        "Supplied auxiliaries may be referenced directly or deliberately promoted "
        "to modeled states with their own equations and initial conditions. A "
        "promoted auxiliary is generated by the model rather than injected as a "
        "forcing trajectory. Never redeclare external inputs or fixed covariates "
        "as states, processes, parameters, or initial conditions. Observation "
        "mappings are required for targets and optional for promoted auxiliaries."
    )
    return "\n".join(rendered)


def _prediction_protocol_prompt(context: ValidationContext) -> str:
    """Render the authoritative target-history protocol for both LLM roles."""
    if context.lagged_targets:
        channels = ", ".join(context.lagged_targets)
        return (
            "Authoritative prediction-protocol amendment: this task is "
            "one-step-ahead. When predicting sample i (i >= 1), measured target "
            f"history through i-1 is available for these channels: {channels}. "
            "A bare target symbol on a right-hand side denotes the most recent "
            "strictly prior measured sample. The current sample i and all future "
            "target values are unavailable. At each interval start, model states "
            "directly identified with target or supplied-auxiliary channels are "
            "reset from the observations then available; latent states are never "
            "revealed and instead propagate causally from their prior estimates. "
            "This amendment supersedes any older open-loop wording in the "
            "benchmark prompt."
        )
    return (
        "Authoritative prediction protocol: open-loop. Target trajectories are "
        "not available as forcing during the prediction horizon."
    )


def _proposer_reasoning_effort(
    arguments: ExecutionArguments,
) -> VLLMReasoningEffort:
    """Resolve the proposer effort while preserving the legacy shared option."""
    return (
        arguments.vllm_proposer_reasoning_effort
        or arguments.vllm_reasoning_effort
    )


def _judge_reasoning_effort(
    arguments: ExecutionArguments,
) -> VLLMReasoningEffort:
    """Resolve the judge effort while preserving the legacy shared option."""
    return arguments.vllm_judge_reasoning_effort or arguments.vllm_reasoning_effort


def _make_client(
    arguments: ExecutionArguments,
    dataset: DevelopmentDataset,
    experiment_directory: Path,
) -> LLMClient:
    if arguments.mock_llm:
        candidates = _mock_candidates(dataset, arguments.iteration_budget)
        judges = [_mock_judge()] * (2 * arguments.iteration_budget)
        return MockLLMClient(
            proposer_responses=candidates,
            judge_responses=judges,
        )
    assert arguments.proposer_model is not None
    assert arguments.judge_model is not None
    proposer_provider, proposer_model = _parse_model(arguments.proposer_model)
    judge_provider, judge_model = _parse_model(arguments.judge_model)
    cache_root = arguments.llm_cache_root or experiment_directory / "llm_cache"
    proposer_cache = (
        cache_root if arguments.llm_cache_root is not None else cache_root / "proposer"
    )
    judge_cache = (
        cache_root if arguments.llm_cache_root is not None else cache_root / "judge"
    )
    proposer = create_llm_client(
        LLMConfig(
            provider=proposer_provider,
            model=proposer_model,
            cache_directory=proposer_cache,
            log_path=experiment_directory / "proposer_events.jsonl",
            timeout_seconds=arguments.llm_timeout_seconds,
            max_output_tokens=arguments.llm_max_output_tokens,
            proposal_target_channels=dataset.roles.targets,
            cache_only=arguments.llm_cache_only,
            ollama_base_url=arguments.ollama_base_url,
            ollama_thinking=arguments.ollama_thinking,
            ollama_temperature=arguments.ollama_temperature,
            ollama_seed=arguments.ollama_seed,
            vllm_base_url=arguments.vllm_base_url,
            vllm_reasoning_effort=_proposer_reasoning_effort(arguments),
            vllm_temperature=arguments.vllm_temperature,
            vllm_seed=arguments.vllm_seed,
        )
    )
    if arguments.selection_policy == "incumbent_relative_hybrid":
        return _RoleClient(proposer, proposer)
    judge = create_llm_client(
        LLMConfig(
            provider=judge_provider,
            model=judge_model,
            cache_directory=judge_cache,
            log_path=experiment_directory / "judge_events.jsonl",
            timeout_seconds=arguments.llm_timeout_seconds,
            max_output_tokens=arguments.llm_max_output_tokens,
            cache_only=arguments.llm_cache_only,
            ollama_base_url=arguments.ollama_base_url,
            ollama_thinking=arguments.ollama_thinking,
            ollama_temperature=arguments.ollama_temperature,
            ollama_seed=arguments.ollama_seed,
            vllm_base_url=arguments.vllm_base_url,
            vllm_reasoning_effort=_judge_reasoning_effort(arguments),
            vllm_temperature=arguments.vllm_temperature,
            vllm_seed=arguments.vllm_seed,
        )
    )
    return _RoleClient(proposer, judge)


def _make_pairwise_judge(
    arguments: ExecutionArguments,
    experiment_directory: Path,
    context: ValidationContext,
    public_prompt: str,
) -> PairedHybridJudge | None:
    """Construct the frozen development-only incumbent-relative judge."""
    if arguments.selection_policy != "incumbent_relative_hybrid":
        return None
    assert arguments.judge_model is not None
    provider, model = _parse_model(arguments.judge_model)
    if provider is not LLMProvider.VLLM:
        raise SystemExit("incumbent_relative_hybrid requires a vllm: judge model")
    seed_base = (
        arguments.hybrid_judge_seed_base
        if arguments.hybrid_judge_seed_base is not None
        else 12000 + 2 * arguments.seed
    )
    cache_root = arguments.llm_cache_root or experiment_directory / "llm_cache"
    pair_cache = (
        cache_root
        if arguments.llm_cache_root is not None
        else cache_root / "hybrid_pair"
    )
    seeded_clients = tuple(
        (
            seed,
            create_llm_client(
                LLMConfig(
                    provider=provider,
                    model=model,
                    cache_directory=pair_cache / f"seed_{seed}",
                    log_path=experiment_directory / "hybrid_pair_events.jsonl",
                    max_attempts=10,
                    initial_backoff_seconds=1.0,
                    max_backoff_seconds=30.0,
                    jitter_fraction=0.0,
                    cache_only=arguments.llm_cache_only,
                    vllm_base_url=arguments.vllm_base_url,
                    vllm_reasoning_effort=_judge_reasoning_effort(arguments),
                    vllm_temperature=0.2,
                    vllm_seed=seed,
                    timeout_seconds=900.0,
                    max_output_tokens=6144,
                )
            ),
        )
        for seed in (seed_base, seed_base + 1)
    )
    symbol_contract = _symbol_contract(context)
    protocol_prompt = _prediction_protocol_prompt(context)
    hybrid_system_prompt = (
        f"Configured judge model: {arguments.judge_model}\n\n"
        f"Public scientific task:\n{public_prompt}\n\n"
        f"{protocol_prompt}\n\n{symbol_contract}\n\n"
        f"Hybrid judge protocol:\n{HYBRID_JUDGE_PROMPT}"
        f"{ATOMIC_STAGE_TWO_NOTE}"
    )
    atomic_system_prompt = (
        f"Configured judge model: {arguments.judge_model}\n\n"
        f"Public scientific task:\n{public_prompt}\n\n"
        f"{symbol_contract}\n\n"
        f"Atomic evidence protocol:\n{ATOMIC_EVIDENCE_PROMPT}"
    )
    identity = json.dumps(
        {
            "judge_model": arguments.judge_model,
            "provider": provider.value,
            "reasoning_effort": _judge_reasoning_effort(arguments).value,
            "temperature": 0.2,
            "max_provider_attempts": 10,
            "max_output_tokens": 6144,
            "seed_base": seed_base,
        },
        sort_keys=True,
    )
    return PairedHybridJudge(
        seeded_clients=seeded_clients,
        requirements=extract_public_requirements(public_prompt),
        task_inputs=tuple(context.external_inputs),
        system_prompt=hybrid_system_prompt,
        atomic_system_prompt=atomic_system_prompt,
        scoring=HybridScoringConfig(
            comparative_indeterminate_policy="neutral_fixed_denominator"
        ),
        identity=identity,
    )


def _parse_model(value: str) -> tuple[LLMProvider, str]:
    if ":" not in value:
        return LLMProvider.OPENAI, value
    provider_name, model = value.split(":", 1)
    try:
        provider = LLMProvider(provider_name)
    except ValueError as exc:
        raise SystemExit(
            f"unsupported model provider {provider_name!r}; "
            "use openai, gemini, or ollama"
        ) from exc
    if not model:
        raise SystemExit("model identifier cannot be empty")
    return provider, model


def _experiment_directory(arguments: ExecutionArguments) -> Path:
    name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"{arguments.benchmark_id}_{arguments.tier}_seed{arguments.seed}",
    )
    return arguments.output_root / name


def _synthetic_split(
    name: SplitName,
    identifier: str,
    duration: float,
) -> DatasetSplit:
    time = np.linspace(0.0, duration, int(duration * 10) + 1)
    target = 1.25 * np.exp(-0.55 * time)
    trajectory = Trajectory(
        identifier,
        time,
        {"target": target},
        {},
        {},
        {},
        {},
    )
    return DatasetSplit(name, (trajectory,), f"synthetic-{name.value}")


def _synthetic_development_dataset() -> DevelopmentDataset:
    from autoformalism.data.models import TierRoles

    return DevelopmentDataset(
        benchmark_id="synthetic",
        tier="easy",
        roles=TierRoles(targets=("target",)),
        train=_synthetic_split(SplitName.TRAIN, "train", 2.0),
        validation=_synthetic_split(SplitName.VALIDATION, "validation", 2.5),
    )


def _synthetic_test_split() -> DatasetSplit:
    return _synthetic_split(SplitName.TEST, "test", 3.0)


def _mock_candidates(
    dataset: DevelopmentDataset,
    count: int,
) -> list[CandidateModel]:
    target_names = dataset.roles.targets
    combined = {
        target: np.concatenate(
            [trajectory.targets[target] for trajectory in dataset.train.trajectories]
        )
        for target in target_names
    }
    candidates: list[CandidateModel] = []
    for round_index in range(count):
        identifier = f"mock_candidate_{round_index}"
        states = [f"state_{index}" for index in range(len(target_names))]
        parameters: list[dict[str, Any]] = []
        equations: list[dict[str, str]] = []
        for index, state in enumerate(states):
            decay = f"decay_{index}"
            parameters.append(_parameter(decay, 0.0, 5.0))
            rhs = f"-{decay} * {state}"
            if round_index:
                coefficient = f"nonlinear_{round_index}_{index}"
                parameters.append(_parameter(coefficient, -2.0, 2.0))
                rhs += f" + {coefficient} * {state} ** {round_index + 1}"
            equations.append({"state": state, "rhs": rhs})
        payload = {
            "candidate_id": identifier,
            "parent_candidate_id": None,
            "change_summary": "Deterministic offline mock proposal.",
            "states": [
                {
                    "name": state,
                    "kind": "observed",
                    "unit": "data_unit",
                    "description": f"Generated state for {target}.",
                }
                for state, target in zip(states, target_names, strict=True)
            ],
            "state_equations": equations,
            "observation_mappings": [
                {
                    "channel": target,
                    "expression": state,
                    "unit": "data_unit",
                }
                for state, target in zip(states, target_names, strict=True)
            ],
            "parameters": parameters,
            "initial_conditions": [
                {
                    "state": state,
                    "scope": "global",
                    "initialization_range": _initial_range(combined[target]),
                }
                for state, target in zip(states, target_names, strict=True)
            ],
        }
        candidates.append(CandidateModel.model_validate(payload))
    return candidates


def _parameter(name: str, lower: float, upper: float) -> dict[str, Any]:
    return {
        "name": name,
        "scope": "global",
        "bounds": {"lower": lower, "upper": upper},
        "initialization_range": {"lower": lower, "upper": upper},
        "unit": "1/time",
        "description": f"Mock parameter {name}.",
    }


def _initial_range(values: np.ndarray) -> dict[str, float]:
    lower = float(np.min(values))
    upper = float(np.max(values))
    padding = max(1e-3, (upper - lower) * 0.25)
    return {"lower": lower - padding, "upper": upper + padding}


def _mock_judge() -> ScientificJudgeResult:
    return ScientificJudgeResult.model_validate(
        {
            "hard_red_flags": [],
            "category_scores": {
                "mechanistic_coherence": 1.0,
                "source_sink_balance_semantics": 1.0,
                "dynamic_plausibility": 1.0,
                "mechanism_coupling_task_sufficiency": 1.0,
                "nonredundancy_accounting": 1.0,
                "latent_state_complexity_justification": 1.0,
            },
            "missing_requirements": [],
            "actionable_edits": [],
        }
    )


def _result_summary(
    arguments: ExecutionArguments,
    result: FinalEvaluation,
) -> dict[str, Any]:
    summary = {
        "status": "complete",
        "evaluation_stage": (
            "development_selection_frozen"
            if arguments.development_only
            else "test_evaluated"
        ),
        "benchmark_id": arguments.benchmark_id,
        "tier": arguments.tier,
        "seed": arguments.seed,
        "stopping_reason": result.stopping_reason,
        "completed_iterations": result.completed_iterations,
        "selection_hash": result.frozen_selection.selection_hash,
        "selected_candidate": result.frozen_selection.candidate.model_dump(mode="json"),
        "selection_validation_normalized_mse": (result.frozen_selection.validation_mse),
        "selection_policy": result.frozen_selection.selection_policy,
        "selection_objective": result.frozen_selection.selection_objective,
        "selection_normalized_log_validation": (
            result.frozen_selection.normalized_log_validation
        ),
        "selection_normalized_judge_penalty": (
            result.frozen_selection.normalized_judge_penalty
        ),
        "selection_judge_score": result.frozen_selection.judge_score,
        "selection_incumbent_path_score": (
            result.frozen_selection.incumbent_path_score
        ),
        "selection_hybrid_science_weight": (
            result.frozen_selection.hybrid_science_weight
        ),
        "final_global_parameters": dict(result.final_fit.global_parameters),
        "final_global_initial_conditions": dict(
            result.final_fit.global_initial_conditions
        ),
        "final_training_normalized_mse": (
            result.final_fit.training_metrics.normalized_mse
        ),
    }
    if result.test_metrics is not None:
        summary.update(
            {
                "test_normalized_mse": result.test_metrics.normalized_mse,
                "test_per_target_normalized_mse": dict(
                    result.test_metrics.per_target_normalized_mse
                ),
                "test_failed_trajectories": list(
                    result.test_metrics.failed_trajectories
                ),
            }
        )
    return summary
