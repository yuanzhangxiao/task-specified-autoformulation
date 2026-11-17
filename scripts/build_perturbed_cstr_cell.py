"""Build the perturbed CSTR-hard development cell for the targeted rebuttal experiment.

Same schedules, preparation, channels, prompt and critic text as the released
``phase_c_cstr_controlled_reactor_mechanism_canonical_named_hard_reset_v1``; only the
reaction-rate law differs (``phase_b_generation.CSTR_PERTURBATION``). Trajectories
come from the release's own reference audit (primary solver, independent solver and
grid refinement) and are projected and sealed by the release's own writer, so the
cell loads with ``load_public_cell``. No test data are generated here. Checks:

- the unchanged canonical path reproduces the released canonical cell exactly;
- time grids, inputs and the initial target reading equal the canonical cell's;
- an independent re-simulation from the public inputs reproduces the new targets;
- the reference equilibrium is unchanged and stable under the perturbed law.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.benchmarks.phase_b_generation import (
    CSTR_PERTURBATION,
    cstr_rate_multiplier,
    simulate_phase_b,
)
from autoformalism.benchmarks.phase_b_public import (
    phase_b_public_spec,
    render_phase_b_prompts,
)
from autoformalism.benchmarks.phase_c_protocols import PREPARATION, phase_c_protocols
from autoformalism.benchmarks.phase_c_release import (
    _write_cell,
    load_public_cell,
    project_split,
    public_information_audit,
)
from autoformalism.benchmarks.reference_audit import PRIMARY, _digest, audit_one
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.public_fitting import PublicSplit

CANONICAL = "phase_c_cstr_controlled_reactor_mechanism_canonical_named_hard_reset_v1"
CELL = CANONICAL.replace("_canonical_", "_perturbed_")
SPEC_PATH = "benchmark5_anonymous_nonlinear_process/private/system_specification.json"
VERSION = "phase-c-cstr-perturbed-1"


def plan(release: Path, data_root: Path) -> dict:
    """Frozen inputs of this build; its hash keys every audit checkpoint."""
    return {
        "protocol": VERSION,
        "cell": CELL,
        "canonical_cell": CANONICAL,
        "perturbation": {
            "law": "rate = k0 exp(-E/T) C"
            " [1 - amplitude tanh((T - T_eq) / width_kelvin)]",
            **CSTR_PERTURBATION,
        },
        "canonical_files": {
            name: _digest(release / "public" / CANONICAL / name)
            for name in (
                "specification.json",
                "train.json",
                "val.json",
                "proposer_prompt.txt",
            )
        },
        "private_spec_sha256": _digest(data_root / SPEC_PATH),
        "source_sha256": public._source_identity(),
        "test_generated": False,
    }


def resimulate(row, truth: dict) -> np.ndarray:
    """Independent check: integrate the perturbed law from the public inputs."""
    p, eq = truth["parameters"], truth["equilibrium"]
    time = np.asarray(row.time, float)
    u = {k: np.asarray(v, float) for k, v in row.external_inputs.items()}

    def rhs(s, x):
        c, t, tj = x
        cf, tf, tjf = (np.interp(s, time, u[k]) for k in ("Cf", "Tf", "Tjf"))
        protected = max(t, 250.0)
        r = (
            p["k0"]
            * np.exp(-p["E_over_R"] / protected)
            * max(c, 0.0)
            * cstr_rate_multiplier(protected, eq["T"], "perturbed")
        )
        return [
            p["flow_rate"] * (cf - c) - r,
            p["flow_rate"] * (tf - t)
            + p["source_gain"] * r
            - p["exchange_rate"] * (t - tj),
            p["secondary_flow_rate"] * (tjf - tj)
            + p["secondary_exchange_rate"] * (t - tj),
        ]

    sol = solve_ivp(
        rhs,
        (time[0], time[-1]),
        [eq["C"], row.targets["T"][0], eq["Tj"]],
        method="LSODA",
        t_eval=time,
        rtol=1e-10,
        atol=1e-12,
        max_step=0.05,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[1]


def stability(truth: dict) -> dict:
    """Balance residual and Jacobian eigenvalues at the reference equilibrium."""
    p, eq = truth["parameters"], truth["equilibrium"]
    x0 = np.array([eq["C"], eq["T"], eq["Tj"]])

    def rhs(x):
        c, t, tj = x
        r = (
            p["k0"]
            * np.exp(-p["E_over_R"] / t)
            * c
            * cstr_rate_multiplier(t, eq["T"], "perturbed")
        )
        return np.array(
            [
                p["flow_rate"] * (p["C_feed_base"] - c) - r,
                p["flow_rate"] * (p["T_feed_base"] - t)
                + p["source_gain"] * r
                - p["exchange_rate"] * (t - tj),
                p["secondary_flow_rate"] * (p["T_secondary_feed_base"] - tj)
                + p["secondary_exchange_rate"] * (t - tj),
            ]
        )

    jac = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1e-6 * max(1.0, abs(x0[j]))
        jac[:, j] = (rhs(x0 + e) - rhs(x0 - e)) / (2 * e[j])
    eig = np.linalg.eigvals(jac)
    return {
        "equilibrium_residual": float(np.max(np.abs(rhs(x0)))),
        "eigenvalues_real": sorted(float(v) for v in eig.real),
        "max_real_part": float(eig.real.max()),
        "stable": bool(eig.real.max() < 0),
    }


def build(release: Path, data_root: Path, out: Path) -> dict:
    frozen = plan(release, data_root)
    seal(out / "plan.json", frozen)
    identity = public.content_sha256(frozen)
    truth = json.loads((data_root / SPEC_PATH).read_text())
    protocols = [p for p in phase_c_protocols("cstr") if p.split != "test"]
    records, trajectories = [], []
    for protocol in protocols:
        record, trajectory = audit_one(
            out / "diagnostic" / "references" / "cstr",
            protocol,
            "perturbed",
            data_root,
            identity,
        )
        if not record["passed"]:
            raise ValueError(
                f"reference numerical audit failed: {protocol.protocol_id}"
            )
        records.append(record)
        trajectories.append(trajectory)
    hidden = [t.states[0, [0, 2]] for t in trajectories]
    reset_difference = max(float(np.max(np.abs(h - hidden[0]))) for h in hidden)
    if reset_difference != 0:
        raise ValueError("perturbed reference violated the shared hidden preparation")
    spec = phase_b_public_spec(
        "cstr",
        "hard",
        "named",
        input_contract="continuous-rates-1",
        data_root=data_root,
    )
    released, canon_train, canon_val = load_public_cell(release / "public" / CANONICAL)
    if (
        spec.benchmark_id.replace("phase_b_", "phase_c_", 1).replace(
            "_rates_v1", "_reset_v1"
        )
        != CANONICAL
    ):
        raise ValueError("public specification does not render the canonical cell")
    splits = tuple(
        project_split(spec, tuple(trajectories), name) for name in ("train", "val")
    )
    prompt, judge = render_phase_b_prompts(spec)
    prompt = prompt.replace(
        "C. Modeling requirements",
        "Preparation and initialization:\n"
        + PREPARATION["cstr"]
        + "\n\nC. Modeling requirements",
    )
    if prompt != released["public_prompt"] or judge != released["judge_prompt"]:
        raise ValueError(
            "the perturbed cell must keep the canonical prompt and critic text"
        )
    specification = {**released, "benchmark_id": CELL}
    combined = PublicSplit.model_validate(
        {
            "name": "train",
            "fingerprint": "combined-development",
            "rows": [r.model_dump(mode="json") for s in splits for r in s.rows],
        }
    )
    information = public_information_audit(combined)

    # The unchanged canonical path must reproduce the released canonical cell exactly.
    canonical_rows = tuple(
        project_split(
            spec,
            tuple(
                simulate_phase_b(p, data_root=data_root, solver=PRIMARY)
                for p in protocols
            ),
            n,
        )
        for n in ("train", "val")
    )
    regression = max(
        float(np.max(np.abs(np.asarray(a.targets["T"]) - np.asarray(b.targets["T"]))))
        for mine, theirs in zip(canonical_rows, (canon_train, canon_val), strict=True)
        for a, b in zip(mine.rows, theirs.rows, strict=True)
    )
    same_interface = all(
        a.time == b.time
        and a.external_inputs == b.external_inputs
        and a.targets["T"][0] == b.targets["T"][0]
        for mine, theirs in zip(splits, (canon_train, canon_val), strict=True)
        for a, b in zip(mine.rows, theirs.rows, strict=True)
    )
    variance = float(np.var(np.concatenate([r.targets["T"] for r in canon_train.rows])))
    change = {
        s.name: {
            "nmse_vs_canonical": float(
                np.mean(
                    np.concatenate(
                        [
                            np.asarray(a.targets["T"]) - np.asarray(b.targets["T"])
                            for a, b in zip(s.rows, c.rows, strict=True)
                        ]
                    )
                    ** 2
                )
                / variance
            ),
            "max_abs_kelvin": float(
                max(
                    np.max(
                        np.abs(np.asarray(a.targets["T"]) - np.asarray(b.targets["T"]))
                    )
                    for a, b in zip(s.rows, c.rows, strict=True)
                )
            ),
        }
        for s, c in zip(splits, (canon_train, canon_val), strict=True)
    }
    independent = max(
        float(np.max(np.abs(resimulate(r, truth) - np.asarray(r.targets["T"]))))
        for s in splits
        for r in s.rows
    )
    stable = stability(truth)
    audit = {
        "passed": bool(
            information["passed"]
            and all(r["passed"] for r in records)
            and regression <= 1e-9
            and same_interface
            and independent <= 1e-5
            and stable["stable"]
            and stable["equilibrium_residual"] <= 1e-9
        ),
        "numerical_protocols": len(records),
        "reference_numerics_passed": all(r["passed"] for r in records),
        "shared_hidden_preparation_verified": reset_difference == 0,
        "public_information": information,
        "canonical_path_regression_max_abs_kelvin": regression,
        "same_times_inputs_and_initial_reading_as_canonical": same_interface,
        "target_change_vs_canonical": change,
        "independent_resimulation_max_abs_kelvin": independent,
        "equilibrium": stable,
        "prompt_and_critic_identical_to_canonical": True,
        "mechanism_recovery_or_identifiability_certified": False,
    }
    row = _write_cell(out, CELL, splits, specification, audit)
    load_public_cell(out / "public" / CELL)
    summary = {
        "protocol": VERSION,
        "plan_sha256": identity,
        "cell": row,
        "audit": audit,
        "test_generated": False,
        "llm_calls": 0,
        "parameter_fitting_performed": False,
    }
    seal(out / "summary.json", summary)
    return read_seal(out / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release", type=Path, default=Path("artifacts/phase-c-development-final-v2")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_raw"))
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/phase-c-cstr-perturbed-v1")
    )
    args = parser.parse_args()
    result = build(args.release, args.data_root, args.out)
    print(
        json.dumps(
            {
                k: result["audit"][k]
                for k in (
                    "passed",
                    "canonical_path_regression_max_abs_kelvin",
                    "same_times_inputs_and_initial_reading_as_canonical",
                    "target_change_vs_canonical",
                    "independent_resimulation_max_abs_kelvin",
                    "equilibrium",
                )
            },
            indent=1,
        )
    )
