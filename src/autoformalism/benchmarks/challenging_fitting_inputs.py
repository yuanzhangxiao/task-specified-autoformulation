"""Evaluator-only M6 inputs from unchanged, audited Phase C development data.

The alien witness supplies fixed internal couplings and nonlinear shapes. These
anchor its latent coordinates: this is conditional parameter recovery with known
equations, not identification of an unrestricted hidden-state realization.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np

from autoformalism.benchmarks import detention
from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.public_fitting import PublicFitRequest

PROTOCOL = "phase-c-challenging-fitting-inputs-1"
CELLS = {
    "basin_coupled": "phase_c_detention_coupled_noise0_v1",
    "alien_hard": (
        "phase_c_alien_device_unknown_device_mechanism_canonical_functional_hard_reset_v1"
    ),
}


def _terms(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [*_terms(node.left), *_terms(node.right)]
    return [node]


def _literal(node: ast.expr) -> float:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal(node.operand)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = float(node.value)
        if np.isfinite(value):
            return value
    raise ValueError("alien witness requires a finite numeric coefficient")


def alien_request(witness: dict) -> tuple[PublicFitRequest, dict]:
    """Lift selected numeric factors, retaining their signs and all other laws.

    Six decays, two input gains, one input saturation scale and four output
    gains are fitted. Five hidden initials are shared across trajectories.
    Numeric values are returned separately for evaluator use only.
    """
    payload = PublicFitRequest.model_validate(witness).model_dump(mode="json")
    candidate = payload["base_candidate"]
    equations = candidate["state_equations"]
    if (
        [e["state"] for e in equations] != [f"x{i}" for i in range(6)]
        or payload["context"]["targets"] != ["v01"]
        or payload["context"]["external_inputs"] != ["u01"]
        or payload["context"]["auxiliaries"]
        or candidate["parameters"]
    ):
        raise ValueError("unexpected alien hard witness layout")
    truth, guesses = {}, {}

    def lift(node: ast.expr, name: str, guess: float) -> ast.expr:
        value = _literal(node)
        if not value or (name in truth and truth[name] != abs(value)):
            raise ValueError("inconsistent or zero free factor")
        truth[name], guesses[name] = abs(value), guess
        symbol = ast.Name(id=name, ctx=ast.Load())
        return ast.UnaryOp(op=ast.USub(), operand=symbol) if value < 0 else symbol

    for i, equation in enumerate(equations):
        tree = ast.parse(equation["rhs"], mode="eval")
        terms = _terms(tree.body)
        decay = terms[0]
        if not (
            isinstance(decay, ast.BinOp)
            and isinstance(decay.op, ast.Mult)
            and isinstance(decay.right, ast.Name)
            and decay.right.id == f"x{i}"
            and _literal(decay.left) < 0
        ):
            raise ValueError("missing leading decay in witness")
        decay.left = lift(decay.left, f"decay_x{i}", 0.1)
        if i < 5:
            forcing = terms[-1]
            if not (
                isinstance(forcing, ast.BinOp)
                and isinstance(forcing.op, ast.Mult)
                and isinstance(forcing.right, ast.Call)
                and isinstance(forcing.right.func, ast.Name)
                and forcing.right.func.id == "tanh"
                and len(forcing.right.args) == 1
            ):
                raise ValueError("missing input saturation in witness")
            argument = forcing.right.args[0]
            if not (
                isinstance(argument, ast.BinOp)
                and isinstance(argument.op, ast.Mult)
                and isinstance(argument.right, ast.Name)
                and argument.right.id == "u01"
            ):
                raise ValueError("unexpected input saturation argument")
            if _literal(forcing.left):
                forcing.left = lift(forcing.left, f"input_x{i}", 0.5)
            argument.left = lift(argument.left, "input_scale", 1.0)
        else:
            for j, term in enumerate(terms[1:]):
                if not isinstance(term, ast.BinOp) or not isinstance(term.op, ast.Mult):
                    raise ValueError("unexpected output factorization")
                factor = term
                while isinstance(factor.left, ast.BinOp) and isinstance(
                    factor.left.op, ast.Mult
                ):
                    factor = factor.left
                factor.left = lift(factor.left, f"output_gain_{j}", 0.5)
        equation["rhs"] = ast.unparse(tree)
    if len(truth) != 13:
        raise ValueError("expected thirteen dynamic factors in alien witness")
    candidate["parameters"] = [
        {"name": name, "scope": "global", "role": "nonnegative_coefficient"}
        for name in guesses
    ]
    candidate["initial_conditions"] = [
        {"state": f"x{i}", "scope": "global", "fixed_value": 0.2} for i in range(6)
    ]
    payload["parameter_guesses"] = guesses
    payload["initialization_plan"]["rules"] = {
        f"x{i}": {"initial": {"mode": "value", "guess": 0.2}} for i in range(5)
    }
    # The reference preparation is private evaluator information; only the
    # shared-unknown-initial convention enters the fitted request.
    for i in range(5):
        initial = witness["initialization_plan"]["rules"][f"x{i}"]["initial"]
        if initial["mode"] != "map" or initial.get("parameters"):
            raise ValueError("reference initial is not a numeric preparation")
        truth[f"init_x{i}_value"] = _literal(
            ast.parse(initial["expression"], mode="eval").body
        )
    request = PublicFitRequest.model_validate(payload)
    if set(public._lower(request)[0].parameter_names) != set(truth):
        raise ValueError("free factor identities differ from lowered model")
    return request, truth


def request(case: dict, seed: int) -> PublicFitRequest:
    """Generic deterministic starts, independent of reference/fitted parameters."""
    payload = PublicFitRequest.model_validate(case["request"]).model_dump(mode="json")
    rng = np.random.default_rng(20261002 + seed)
    width = (0.5, 1.0, 1.5)[seed % 3]
    payload["parameter_guesses"] = {
        n: float(v * np.exp(rng.uniform(-width, width)))
        for n, v in payload["parameter_guesses"].items()
    }
    for rule in payload["initialization_plan"]["rules"].values():
        initial = rule["initial"]
        if initial["mode"] == "value":
            initial["guess"] = float(rng.uniform(-width, width))
    return PublicFitRequest.model_validate(payload)


def export_inputs(release: Path, output: Path) -> dict:
    """Copy verified public arrays and parameterize their sealed equation witnesses."""
    if output.resolve().is_relative_to(release.resolve()):
        raise ValueError("inputs must be outside the dataset release")
    summary, plan = (
        read_seal(release / "summary.json"),
        read_seal(release / "plan.json"),
    )
    if (
        summary["protocol"] != "phase-c-development-2"
        or not summary["whole_phase_c_roster_ready"]
        or summary["plan_sha256"] != public.content_sha256(plan)
        or summary["test_generated"]
    ):
        raise ValueError("requires completed development-2 release without test")
    records = {r["cell"]: r for r in summary["cells"]}
    cases = {}
    for name, cell in CELLS.items():
        record = records[cell]
        if not record["ready_for_development"]:
            raise ValueError("unqualified development cell")
        assets, hashes = {}, {}
        paths = [
            f"public/{cell}/{f}"
            for f in (
                "train.json",
                "val.json",
                "specification.json",
                "proposer_prompt.txt",
            )
        ]
        if name == "alien_hard":
            paths.append(f"diagnostic/{cell}/reference_request.json")
        for relative in paths:
            raw = (release / relative).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != record["files"][relative]:
                raise ValueError(f"release asset differs: {relative}")
            hashes[relative] = digest
            assets[Path(relative).name] = (
                raw.decode()
                if relative.endswith(".txt")
                else read_seal(release / relative)
            )
        if name == "alien_hard":
            req, truth = alien_request(assets["reference_request.json"])
            conditioning = (
                "Fixed internal couplings and internal/output tanh shapes anchor "
                "latent coordinates. Free decays, input/output gains, input tanh "
                "scale and shared hidden initials. Conditional local rank only; "
                "global uniqueness and practical identifiability unproven."
            )
        else:
            req = detention.reference_request(
                "coupled", detention.Start(k_transfer=30, k_outlet=30)
            )
            truth = dict(detention.TRUTH)
            conditioning = (
                "Known reservoir geometry, laws and public upstream initial; "
                "two free hydraulic coefficients. Regression control."
            )
        cases[name] = {
            "kind": "challenging_fixed_equations",
            "cell": cell,
            "request": req.model_dump(mode="json"),
            "reference_parameters": truth,
            "training": assets["train.json"],
            "validation": assets["val.json"],
            "public_prompt": assets["proposer_prompt.txt"],
            "source_files": hashes,
            "identifiability": {"scope": conditioning, "global_uniqueness": "unproven"},
        }
    result = {
        "protocol": PROTOCOL,
        "release_plan_sha256": summary["plan_sha256"],
        "release_summary_sha256": public.content_sha256(summary),
        "cases": cases,
        "test_data_opened": False,
        "assistance": (
            "Known equations and fixed parameter blocks; not model discovery."
        ),
    }
    seal(output, result)
    return {"cases": len(cases), "sha256": public.content_sha256(result)}
