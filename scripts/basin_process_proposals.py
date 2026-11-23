"""Ask 20B for two discharge laws and assemble the public basin balances."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import Field, StringConstraints

from autoformalism.expressions.compiler import SAFE_DIVISION_EPSILON, _evaluate
from autoformalism.expressions.diagnostics import (
    ModelValidationError,
    RuntimeExpressionError,
)
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research.preproposal_cases import BASIN
from autoformalism.research.trajectory_function_revision import assess_revision
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash
from scripts import t2_breadth as pilot
from scripts import trajectory_functions as trial
from scripts.construction_functions import file_hash
from scripts.submit_trajectory_functions import ENTRYPOINTS
from scripts.trajectory_function_prompts import Parameter

PROTOCOL = "basin-process-proposals-1"
INTAKE = "4c28072b2deb510a7712e62d08a7afdc4b959f0d1464d32105006f313849411d"
PARENT = "a6b51e36260aa5714827d66a0055143a8746a7888a2ba90eb3b3aefc6d0a391c"
SEEDS = (100, 101, 102)
EXPRESSION = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)
]


class DischargeLaws(StrictSchema):
    """Two local per-head factors and all their fitted coefficient starts."""

    upstream_factor: EXPRESSION
    downstream_factor: EXPRESSION
    parameters: Annotated[list[Parameter], Field(max_length=8)]


INSTRUCTION = """Propose the two unknown discharge laws. The runtime assembles
area_up * dh_up/dt = inflow_up - Q and
area_down * dh_down/dt = inflow_down + Q - D.
For each outlet, head is max(local depth minus its surveyed crest, 0), in metres.
The volumetric discharge is head * abs(your factor), in cubic metres per minute.
Give the upstream factor for Q and the downstream factor for D. Each factor must
be finite at head = 0. You can use head and fitted coefficients in either
factor. Each coefficient is shared across events. Reusing its name in both
factors also ties the two coefficients together. Declare one finite starting
value per coefficient. Use the training observations to choose the forms.
Depth h_down is observed in training. Only the initial upstream depth is
measured; subsequent h_up values and the internal flow Q are unknown.
Write each factor as a single expression using numbers, +, -, *, /, integer
powers ** from -16 to 16, and abs, exp, log, sqrt, min, max, sin, tanh,
sigmoid or softplus. A nonnegative half-integer power of head up to 15.5
is also accepted. The runtime supplies the balances, area conversion,
initial conditions, head and the outer head * abs(...) in both laws."""


def _outlet_observations(rows: list[dict]) -> list[dict]:
    """Pair downstream head with estimated discharge on isolated training events."""
    samples = []
    for row in rows:
        if not row["trajectory_id"].startswith("train_"):
            raise ValueError("Expected training observations only")
        c = row["covariates"]
        if c["initial_up"] > c["crest_up"] or any(row["inputs"]["inflow_up"]):
            continue
        t = np.asarray(row["time"], dtype=float)
        h = np.asarray(row["targets"]["h_down"], dtype=float)
        u = np.asarray(row["inputs"]["inflow_down"], dtype=float)
        head = np.maximum(h - c["crest_down"], 0)
        means = (head[:-1] + head[1:]) / 2
        flow = (u[:-1] + u[1:]) / 2 - c["area_down"] * np.diff(h) / np.diff(t)
        ordered = np.argsort(means)
        for indexes in np.array_split(ordered, min(5, len(ordered))):
            samples.append(
                {
                    "event": row["trajectory_id"],
                    "intervals": len(indexes),
                    "mean_head_m": round(float(np.mean(means[indexes])), 5),
                    "mean_outlet_m3_per_min": round(float(np.mean(flow[indexes])), 5),
                }
            )
    return samples


def _request(data: dict) -> dict:
    row = _parent(data)
    case = data["cases"][row["case"]]
    return {
        "scientific_task": case["public_brief"]["scientific_context"],
        "public_requirements": case["public_brief"]["requirements"],
        "training_observations": data["observations"][BASIN],
        "outlet_evidence": {
            "basis": "With zero upstream inflow and initial upstream depth at or "
            "below its crest, Q=0. D averaged over an interval equals mean downstream "
            "inflow minus area_down times the interval depth change / duration. "
            "Head uses the mean of its two endpoints. Bins average these intervals.",
            "rows": _outlet_observations(data["train"][BASIN]),
        },
    }


def _parent(data: dict) -> dict:
    if data.get("artifact_sha256") != INTAKE:
        raise ValueError("Expected the frozen basin-only training intake")
    return next(row for row in data["graphs"] if row["structure_sha256"] == PARENT)


def _plan(data: dict) -> dict:
    request = _request(data)
    return {
        "protocol": PROTOCOL,
        "intake_sha256": INTAKE,
        "parent": PARENT,
        "seeds": list(SEEDS),
        "maximum_generation_requests": len(SEEDS),
        "maximum_fit_slots": len(SEEDS),
        "fit_budget": trial.BUDGET,
        "settings": trial.SETTINGS.model_dump(mode="json"),
        "runtime_source": runtime_source_hash(),
        "entrypoints": {
            path: file_hash(pilot.REPO / path)
            for path in (*ENTRYPOINTS, "scripts/basin_process_proposals.py")
        },
        "request_sha256": content_hash(
            [INSTRUCTION, request, DischargeLaws.model_json_schema()]
        ),
        "test_data_opened": False,
    }


def prepare(root: Path, campaign: Path) -> dict:
    """Freeze the training intake, public prompt, three settings and budgets."""
    data = sealed_read(campaign / "intake.json")
    plan = _plan(data)
    with _lock(root / "prepare-lock"):
        sealed_write(
            root / "intake.json",
            {k: v for k, v in data.items() if k != "artifact_sha256"},
        )
        sealed_write(
            root / "prompt.json",
            {
                "system": INSTRUCTION,
                "user": _request(data),
                "schema": DischargeLaws.model_json_schema(),
            },
        )
        return sealed_write(root / "plan.json", plan)


def load(root: Path) -> tuple[dict, dict]:
    """Verify the immutable experiment against the installed worker."""
    data = sealed_read(root / "intake.json")
    plan = sealed_read(root / "plan.json")
    if plan["artifact_sha256"] != content_hash(_plan(data)):
        raise ValueError("Basin process plan or training inputs changed")
    return plan, data


def _factor(text: str, basin: str, parameters: dict) -> str:
    """Check the local factor, then substitute the known nonnegative head."""
    location = f"{basin}_factor"
    # Only a literal half-integer power of the known nonnegative head has
    # this rewrite. All other syntax still passes through the restricted parser.
    tree = ast.parse(text, mode="eval")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Pow)
            and isinstance(node.left, ast.Name)
            and node.left.id == "head"
            and isinstance(node.right, ast.Constant)
            and type(node.right.value) is float
            and 0 < node.right.value <= 15.5
            and node.right.value % 1 == 0.5
        ):
            power = int(node.right.value)
            root = ast.Call(ast.Name("sqrt", ast.Load()), [node.left], [])
            replacement = (
                root
                if power == 0
                else ast.BinOp(
                    ast.BinOp(node.left, ast.Pow(), ast.Constant(power)),
                    ast.Mult(),
                    root,
                )
            )
            # Replace x**p with 1 * normalized, preserving Python's parsed precedence.
            node.left, node.op, node.right = ast.Constant(1), ast.Mult(), replacement
    parsed = RestrictedParser().parse(ast.unparse(tree), location=location)
    unknown = parsed.symbols - parameters.keys() - {"head"}
    if unknown:
        raise ValueError(f"{location} has undefined names {sorted(unknown)}")
    try:
        initial = {**parameters, "head": 0.0}
        # The rollout protects tiny denominators numerically. A submitted factor
        # must already be defined at the crest without that numerical fallback.
        for node in ast.walk(parsed.tree):
            if not isinstance(node, ast.BinOp):
                continue
            if isinstance(node.op, ast.Div):
                divisor = node.right
            elif isinstance(node.op, ast.Pow) and ast.literal_eval(node.right) < 0:
                divisor = node.left
            else:
                continue
            subexpression = replace(parsed, tree=ast.Expression(divisor))
            if abs(_evaluate(subexpression, initial)) < SAFE_DIVISION_EPSILON:
                raise RuntimeExpressionError("singular factor at zero head")
        _evaluate(parsed, initial)
    except RuntimeExpressionError as exc:
        raise ValueError(
            f"{location} is undefined at zero head with its submitted starts"
        ) from exc
    head = f"max(h_{basin} - crest_{basin}, 0)"
    head_value = ast.parse(head, mode="eval").body

    class BindHead(ast.NodeTransformer):
        def visit_Name(self, node):
            return head_value if node.id == "head" else node

    bound = ast.unparse(BindHead().visit(parsed.tree))
    return f"({head} * abs({bound}))"


def assemble(data: dict, raw: dict) -> dict:
    """Lower authored factors into outlet flows with one conserved transfer."""
    parent = _parent(data)
    laws = DischargeLaws.model_validate(raw)
    parameters = {p.name: p.initial for p in laws.parameters}
    if len(parameters) != len(laws.parameters) or "head" in parameters:
        raise ValueError("Declare each parameter once, using names different from head")
    q = _factor(laws.upstream_factor, "up", parameters)
    d = _factor(laws.downstream_factor, "down", parameters)
    formula = {
        "equations": {
            "h_up": f"(inflow_up - {q}) / area_up",
            "h_down": f"(inflow_down + {q} - {d}) / area_down",
        },
        "initializers": {},
        "parameters": parameters,
    }
    outcome = assess_revision(
        parent["graph"],
        trial.training_case(data, parent),
        formula,
        "Include one internal transfer and one outlet in both measured-area balances.",
    )
    if outcome["status"] == "eligible":
        graph = outcome.pop("graph")
        outcome.update(revised_graph=graph, revised_graph_sha256=content_hash(graph))
    return {**outcome, "authored_factors": raw, "route": "request_topology_change"}


def run(root, endpoint, *, runtime_record=None, transport=None, token_transport=None):
    """Generate three independent candidates with durable per-call receipts."""
    from scripts.submit_t2_breadth import bind

    plan, data = load(root)
    if transport is None:
        if runtime_record is None:
            raise ValueError("Live generation requires the pinned 20B runtime")
        bind(root, Path(runtime_record), endpoint)
    with _lock(root / "generation-lock"):
        ids = []
        for i, seed in enumerate(SEEDS):
            slot = f"slot-{i:03}"
            reply = pilot._ask(
                root,
                plan,
                slot,
                seed,
                INSTRUCTION,
                _request(data),
                DischargeLaws,
                endpoint,
                transport,
                token_transport,
                {"physical_requests": 0, "slots": {}},
                len(SEEDS),
                sampling_step="basin-discharge-laws",
            )
            record = {
                "plan": plan["artifact_sha256"],
                "graph": PARENT,
                "seed": seed,
                "index": i,
                "reply_sha256": reply["artifact_sha256"],
                "status": reply["status"],
            }
            if reply["status"] == "visible":
                try:
                    record.update(assemble(data, reply["value"]))
                except (
                    ValueError,
                    SyntaxError,
                    ModelValidationError,
                    RecursionError,
                ) as exc:
                    record.update(status="invalid_formula", reason=str(exc))
            elif reply.get("error"):
                record["reason"] = reply["error"]
            candidate = sealed_write(root / "candidates" / f"{i:03}.json", record)
            ids.append(candidate["artifact_sha256"])
        sealed_write(
            root / "generation.json",
            {"plan": plan["artifact_sha256"], "candidates": ids},
        )
    return status(root)


def status(root: Path) -> dict:
    """Verify all recorded candidates and physical receipts before reporting."""
    plan, _ = load(root)
    observed = [
        trial._read_candidate(root, plan, i)
        for i in range(len(SEEDS))
        if (root / "candidates" / f"{i:03}.json").is_file()
    ]
    finished = (root / "generation.json").is_file()
    if finished:
        freeze = trial._frozen_candidates(root, plan)
        if freeze["candidates"] != [r["artifact_sha256"] for r in observed]:
            raise ValueError("Candidate identities differ from generation freeze")
    return {
        "status": "complete" if finished else "incomplete",
        "recorded": len(observed),
        "expected": len(SEEDS),
        "candidate_statuses": dict(Counter(r["status"] for r in observed)),
        "physical_requests": len(list((root / "steps").glob("*/calls/*.json"))),
        "test_data_opened": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "status", "inspect", "run"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--runtime-record", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.campaign is None:
            parser.error("prepare requires --campaign")
        result = prepare(args.root, args.campaign)
    elif args.action == "run":
        if args.endpoint is None or args.runtime_record is None:
            parser.error("run requires --endpoint and --runtime-record")
        result = run(args.root, args.endpoint, runtime_record=args.runtime_record)
    else:
        result = status(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
