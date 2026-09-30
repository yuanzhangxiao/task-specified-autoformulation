"""Evaluator-only fixed-equation inputs for Phase C fitting qualification.

This exports existing development arrays without generating or modifying data.
Known equations and diagnostic fixed parameters are assistance, not discovery.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np

from autoformalism.benchmarks import detention
from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.benchmarks.reference_qualification import SPEC_PATH, cstr_request
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.public_fitting import PublicFitRequest

CELLS = {
    "cstr_easy": (
        "phase_c_cstr_controlled_reactor_mechanism_canonical_named_easy_reset_v1"
    ),
    "cstr_hard": (
        "phase_c_cstr_controlled_reactor_mechanism_canonical_named_hard_reset_v1"
    ),
    "basin_coupled": "phase_c_detention_coupled_noise0_v1",
}


def export_inputs(release: Path, data_root: Path, output: Path) -> dict:
    """Seal three verified cases for portable CPU fitting; no private trajectories."""
    if output.resolve().is_relative_to(release.resolve()):
        raise ValueError("qualification inputs must be outside the dataset release")
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
        raise ValueError("requires the completed Phase C development-2 release")
    spec_path = data_root / SPEC_PATH
    if (
        hashlib.sha256(spec_path.read_bytes()).hexdigest()
        != plan["private_spec_sha256"]["cstr"]
    ):
        raise ValueError("CSTR equation witness differs from release")
    specification = public._read(spec_path)
    records = {r["cell"]: r for r in summary["cells"]}
    cases = {}
    for case, cell in CELLS.items():
        record = records[cell]
        if not record["ready_for_development"]:
            raise ValueError("unqualified development cell")
        assets = {}
        for name in (
            "train.json",
            "val.json",
            "specification.json",
            "proposer_prompt.txt",
        ):
            relative = f"public/{cell}/{name}"
            raw = (release / relative).read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["files"][relative]:
                raise ValueError(f"release asset differs: {relative}")
            assets[name] = (
                raw.decode() if name.endswith(".txt") else read_seal(release / relative)
            )
        if case.startswith("cstr"):
            tier = case.split("_")[1]
            request, truth = cstr_request(tier, "generic", specification)
            payload = request.model_dump(mode="json")
            # The new release has a common reset, not the old affine hidden shift.
            truth = {n: v for n, v in truth.items() if not n.startswith("init_")}
            if tier == "hard":
                payload["initialization_plan"]["rules"] = {
                    s: {"initial": {"mode": "value", "guess": guess}}
                    for s, guess in (("C", 0.25), ("Tj", 345.0))
                }
                truth.update(
                    {
                        f"init_{s}_value": specification["equilibrium"][s]
                        for s in ("C", "Tj")
                    }
                )
            request = PublicFitRequest.model_validate(payload)
            state_proxies = {
                "T": "T",
                "C" if tier == "hard" else "c_internal": "Cf",
                "Tj" if tier == "hard" else "j_internal": "Tjf",
            }
        else:
            request = detention.reference_request(
                "coupled", detention.Start(k_transfer=30, k_outlet=30)
            )
            truth = dict(detention.TRUTH)
            state_proxies = {}
        cases[case] = {
            "cell": cell,
            "request": request.model_dump(mode="json"),
            "reference_parameters": truth,
            "training": assets["train.json"],
            "validation": assets["val.json"],
            "public_prompt": assets["proposer_prompt.txt"],
            "state_channel_proxies": state_proxies,
            "source_files": record["files"],
        }
    result = {
        "protocol": "phase-c-fitting-inputs-1",
        "release_plan_sha256": summary["plan_sha256"],
        "release_summary_sha256": public.content_sha256(summary),
        "cases": cases,
        "assistance": (
            "known equation skeletons; reference parameters only for "
            "evaluator controls and scoring"
        ),
        "test_data_opened": False,
    }
    seal(output, result)
    return {"cases": len(cases), "sha256": public.content_sha256(result)}


class _Constants(ast.NodeTransformer):
    def __init__(self, values):
        self.values = values

    def visit_Name(self, node):
        return (
            ast.copy_location(ast.Constant(self.values[node.id]), node)
            if node.id in self.values
            else node
        )


def experiment_request(
    case: dict, scope: str, seed: int, domains: bool
) -> tuple[PublicFitRequest, dict]:
    """Pair identical physical guesses; freeze known blocks in assisted controls."""
    payload = PublicFitRequest.model_validate(case["request"]).model_dump(mode="json")
    if scope not in {"parameters_only", "initials_only", "joint"}:
        raise ValueError("unknown fitting scope")
    truth = dict(case["reference_parameters"])
    rng = np.random.default_rng(20260930 + seed)
    payload["parameter_guesses"] = {
        n: float(v * np.exp(rng.uniform(-0.7, 0.7)))
        for n, v in payload["parameter_guesses"].items()
    }
    rules = payload["initialization_plan"]["rules"]
    for state, rule in rules.items():
        initial = rule["initial"]
        if initial["mode"] != "value":
            continue
        name = f"init_{state}_value"
        if scope == "parameters_only":
            rule["initial"] = {
                "mode": "map",
                "expression": repr(truth[name]),
                "parameters": [],
            }
        else:
            initial["guess"] = (
                float(initial["guess"] + rng.uniform(-3, 3))
                if state == "Tj"
                else float(initial["guess"] * np.exp(rng.uniform(-0.7, 0.7)))
            )
            if domains and state == "C":
                initial["role"] = "nonnegative_coefficient"
    if scope == "initials_only":
        candidate = payload["base_candidate"]
        constants = {p["name"]: truth[p["name"]] for p in candidate["parameters"]}
        for eq in candidate["state_equations"]:
            eq["rhs"] = ast.unparse(
                _Constants(constants).visit(ast.parse(eq["rhs"], mode="eval"))
            )
        candidate["parameters"] = []
        payload["parameter_guesses"] = {}
    request = PublicFitRequest.model_validate(payload)
    model, _, _ = public._lower(request)
    return request, {n: truth[n] for n in model.parameter_names}
