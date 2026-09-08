"""Small cached function-stage probe for explicit outer-weight sign contracts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import signal
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    StagedTopologyClient,
    atomic_json,
)
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.candidate import ParameterRole
from autoformalism.schemas.staged import InteractionPolarity
from autoformalism.schemas.staged_functions import InteractionFunctionReply
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    PublicScientificBrief,
    PublicVariable,
    ScientificVariable,
)
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.staged_functions import normalize_outer_weight_reply
from autoformalism.staged_topology import content_hash, lower_topology


class SignContractCampaignConfig(StrictSchema):
    """A bounded one-model scientific-contract transport probe."""

    protocol: Literal["scientific-staged-sign-contract-1"]
    purpose: str = Field(min_length=1)
    platform: Literal["aces-h100x1"]
    serving_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_settings: StagedModelSettings
    served_context_tokens: int = Field(ge=16384)
    seeds: tuple[int, ...] = Field(min_length=1, max_length=8)
    wall_seconds: int = Field(ge=60)
    shutdown_margin_seconds: int = Field(ge=30)

    @model_validator(mode="after")
    def bounded_unique_tasks(self) -> SignContractCampaignConfig:
        """Reject duplicate seeds and an impossible shutdown schedule."""
        if len(self.seeds) != len(set(self.seeds)) or any(
            seed < 0 for seed in self.seeds
        ):
            raise ValueError("seeds must be unique and nonnegative")
        if (
            not self.model_settings.timeout_seconds
            < self.shutdown_margin_seconds
            < self.wall_seconds
        ):
            raise ValueError("require timeout < shutdown margin < worker wall time")
        return self


def sign_contract_launcher_hash() -> str:
    """Bind the CLI and ACES worker used by the frozen campaign."""
    repository = Path(__file__).resolve().parents[3]
    paths = (
        "scripts/staged_sign_contract_campaign.py",
        "scripts/hpc/run_staged_topology_server.sh",
        "scripts/hpc/staged_sign_contract_aces.slurm",
        "scripts/hpc/submit_staged_sign_contract_aces.sh",
    )
    return content_hash(
        {
            path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
            for path in paths
        }
    )


def sign_contract_source() -> tuple[
    PublicScientificBrief, ValidationContext, dict[str, Any]
]:
    """Return one reviewed toy topology with four distinct sign obligations."""
    brief = PublicScientificBrief(
        scientific_context=(
            "Construct one driven continuous-time response. Its equation contains "
            "an unknown positive input gain, a negative relaxation contribution, "
            "a signed internal contrast under a positive outer gain, and an unknown "
            "baseline offset whose sign is scientifically unspecified."
        ),
        public_variables=(
            PublicVariable(name="x", data_role="target"),
            PublicVariable(name="u", data_role="external_input"),
        ),
        requirements=(),
    )
    context = ValidationContext(targets=("x",), external_inputs=("u",))
    inventory = (
        ScientificVariable(
            name="x", definition="differential", scientific_role="driven response"
        ),
        ScientificVariable(
            name="u", definition="supplied", scientific_role="external forcing"
        ),
    )
    equations = (
        EquationDefinition(
            name="x",
            definition="differential",
            terms=(
                {
                    "sources": ["u"],
                    "outer_weight_sign": "positive",
                    "scientific_role": (
                        "direct input response with an unknown outer gain magnitude"
                    ),
                },
                {
                    "sources": ["x"],
                    "outer_weight_sign": "negative",
                    "scientific_role": "self-relaxation toward baseline",
                },
                {
                    "sources": ["u", "x"],
                    "outer_weight_sign": "positive",
                    "scientific_role": (
                        "unknown outer magnitude multiplying a signed internal "
                        "contrast between forcing and a fitted threshold times x"
                    ),
                },
                {
                    "sources": [],
                    "outer_weight_sign": "unrestricted",
                    "scientific_role": (
                        "unknown signed baseline offset; do not assume it is zero"
                    ),
                },
            ),
        ),
    )
    topology, _ = lower_topology(brief, inventory, equations, context)
    return (
        brief,
        context,
        {
            "complete_topology": True,
            "inventory": [item.model_dump(mode="json") for item in inventory],
            "equations": [item.model_dump(mode="json") for item in equations],
            "topology": topology.model_dump(mode="json"),
        },
    )


def freeze_sign_contract_campaign(config_path: Path, output: Path) -> dict[str, Any]:
    """Freeze the compact source and all seeds before provider calls."""
    config = SignContractCampaignConfig.model_validate_json(config_path.read_text())
    brief, context, source = sign_contract_source()
    tasks = [
        {
            "task_id": f"sign_contract_seed{seed}",
            "seed": seed,
            "brief": brief.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "source": source,
        }
        for seed in config.seeds
    ]
    plan = {
        "config": config.model_dump(mode="json"),
        "runtime_source_sha256": runtime_source_hash(),
        "launcher_sha256": sign_contract_launcher_hash(),
        "tasks": tasks,
        "paired_control": (
            "Replay each unrestricted signed-offset response offline under a fixed "
            "positive sign; only the topology sign differs"
        ),
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    plan["plan_sha256"] = content_hash(plan)
    if output.exists():
        if json.loads(output.read_text()) != plan:
            raise ValueError("existing frozen sign-contract campaign differs")
    else:
        atomic_json(output, plan)
    return plan


def run_sign_contract_campaign(
    plan_path: Path,
    output: Path,
    base_url: str,
    *,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Run and checkpoint every seed, replaying cached physical requests."""
    plan = json.loads(plan_path.read_text())
    expected = content_hash(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if expected != plan.get("plan_sha256"):
        raise ValueError("frozen sign-contract plan digest mismatch")
    if runtime_source_hash() != plan["runtime_source_sha256"]:
        raise ValueError("runtime source differs from frozen sign-contract campaign")
    if sign_contract_launcher_hash() != plan["launcher_sha256"]:
        raise ValueError("launcher differs from frozen sign-contract campaign")
    config = SignContractCampaignConfig.model_validate(plan["config"])
    deadline = time.monotonic() + (wall_seconds or config.wall_seconds)
    stop = False

    def drain(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    previous = {
        sig: signal.signal(sig, drain) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    records: list[dict[str, Any]] = []
    try:
        for task in plan["tasks"]:
            root = output / task["task_id"]
            root.mkdir(parents=True, exist_ok=True)
            with (root / "worker.lock").open("w") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                identity = content_hash([plan["plan_sha256"], task])
                terminal = root / "terminal.json"
                if terminal.exists():
                    record = json.loads(terminal.read_text())
                    if record["identity"] != identity:
                        raise ValueError("terminal result belongs to another task")
                else:
                    client = StagedTopologyClient(
                        settings=config.model_settings,
                        base_url=base_url,
                        directory=root / "calls",
                        namespace=identity,
                        seed=task["seed"],
                        can_start=lambda: not stop
                        and time.monotonic()
                        < deadline - config.shutdown_margin_seconds,
                    )
                    try:
                        result = run_staged_functions(
                            PublicScientificBrief.model_validate(task["brief"]),
                            ValidationContext.model_validate(task["context"]),
                            task["source"],
                            client,
                            root,
                        )
                    except DeferredCall:
                        break
                    result["sign_contract_audit"] = _sign_contract_audit(result)
                    atomic_json(root / "result.json", result)
                    record = {
                        "identity": identity,
                        "task_id": task["task_id"],
                        "seed": task["seed"],
                        "result": result,
                    }
                    atomic_json(terminal, record)
                records.append(record)
                atomic_json(output / "summary.json", summarize(plan, records))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    summary = summarize(plan, records)
    atomic_json(output / "summary.json", summary)
    return summary


def _sign_contract_audit(result: dict[str, Any]) -> dict[str, Any]:
    """Compare requested/effective roles and replay the offset with one sign change."""
    accepted = result.get("accepted_functions", [])
    effective = {
        item["interaction_id"]: item
        for item in result.get("draft", {}).get("interaction_functions", [])
    }
    rows = []
    for index, item in enumerate(accepted):
        selected = item["selected_term"]
        interaction_id = f"term_0_{index}"
        requested_roles = {value["name"]: value["role"] for value in item["parameters"]}
        effective_roles = {
            value["name"]: value["role"]
            for value in effective.get(interaction_id, {}).get("parameters", [])
        }
        rows.append(
            {
                "interaction_id": interaction_id,
                "outer_weight_sign": selected["outer_weight_sign"],
                "requested_roles": requested_roles,
                "effective_roles": effective_roles,
                "derived_parameters": sorted(
                    name
                    for name in requested_roles
                    if requested_roles[name] != effective_roles.get(name)
                ),
            }
        )
    baseline = next(
        (
            item
            for item in accepted
            if item["selected_term"]["scientific_role"]
            == "unknown signed baseline offset; do not assume it is zero"
        ),
        None,
    )
    if baseline is None:
        return {
            "rows": rows,
            "unrestricted_offset_preserved_signed": None,
            "fixed_positive_offset_replay": None,
            "matched_offset_domain_control_pass": None,
        }
    baseline_reply = InteractionFunctionReply.model_validate(
        {
            "expression": baseline["expression"],
            "parameters": baseline["parameters"],
        }
    )
    try:
        fixed, derivations = normalize_outer_weight_reply(
            baseline_reply, InteractionPolarity.POSITIVE
        )
        fixed_replay = {
            "success": True,
            "effective_roles": {
                item.name: item.role.value for item in fixed.parameters
            },
            "derived_parameters": [item.parameter for item in derivations],
            "error": None,
        }
    except ValueError as exc:
        fixed_replay = {
            "success": False,
            "effective_roles": {},
            "derived_parameters": [],
            "error": str(exc)[:6000],
        }
    baseline_row = next(
        item for item in rows if item["outer_weight_sign"] == "unrestricted"
    )
    signed_roles = {ParameterRole.COEFFICIENT.value, ParameterRole.OFFSET.value}
    unrestricted_signed = bool(baseline_row["effective_roles"]) and all(
        role in signed_roles for role in baseline_row["effective_roles"].values()
    )
    return {
        "rows": rows,
        "unrestricted_offset_preserved_signed": unrestricted_signed,
        "fixed_positive_offset_replay": fixed_replay,
        "matched_offset_domain_control_pass": (
            unrestricted_signed
            and fixed_replay["success"]
            and bool(fixed_replay["derived_parameters"])
            and all(
                role == ParameterRole.NONNEGATIVE_COEFFICIENT.value
                for role in fixed_replay["effective_roles"].values()
            )
        ),
    }


def summarize(plan: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report contract behavior without defining a model-quality winner."""
    indexed = {item["task_id"]: item for item in records}
    rows = []
    for task in plan["tasks"]:
        record = indexed.get(task["task_id"])
        result = record["result"] if record else None
        audit = result.get("sign_contract_audit", {}) if result else {}
        rows.append(
            {
                "task_id": task["task_id"],
                "seed": task["seed"],
                "result_present": result is not None,
                "complete_model": (
                    result.get("complete_model", False) if result else False
                ),
                "matched_offset_domain_control_pass": audit.get(
                    "matched_offset_domain_control_pass"
                ),
                "physical_requests": (
                    result.get("physical_requests", 0) if result else 0
                ),
                "observed_total_tokens": result.get("observed_total_tokens", 0)
                if result
                else 0,
                "provider_seconds": result.get("provider_seconds", 0.0)
                if result
                else 0.0,
                "error": result.get("error") if result else None,
            }
        )
    terminal = [item for item in rows if item["result_present"]]
    complete = [item for item in terminal if item["complete_model"]]
    controls = [
        item["matched_offset_domain_control_pass"]
        for item in terminal
        if item["matched_offset_domain_control_pass"] is not None
    ]
    return {
        "schema_version": "scientific-staged-sign-contract-summary-1",
        "status": "complete" if len(terminal) == len(rows) else "incomplete",
        "plan_sha256": plan["plan_sha256"],
        "planned_tasks": len(rows),
        "terminal_results": len(terminal),
        "complete_models": len(complete),
        "complete_model_rate": len(complete) / len(terminal) if terminal else None,
        "matched_offset_domain_control_pass_rate": (
            sum(bool(value) for value in controls) / len(controls) if controls else None
        ),
        "physical_requests": sum(item["physical_requests"] for item in terminal),
        "observed_total_tokens": sum(
            item["observed_total_tokens"] for item in terminal
        ),
        "provider_seconds": sum(item["provider_seconds"] for item in terminal),
        "rows": rows,
        "automatic_winner_defined": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
