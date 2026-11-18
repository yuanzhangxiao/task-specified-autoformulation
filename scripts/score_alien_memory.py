"""Memory scores of procedural-device models against each device's private equations.

The stress study's procedural devices (``alien_grammar_v1``) drive internal
coordinates z with the command and generate the output only from them:
dy/dt = -d_y y + G(z). Their coordinates are not identifiable from (u, y), so a
model is scored on what the measured output fixes, under the frozen suite
``configs/alien_device_memory_suite_v1.json`` (its eight interventions plus the
two saved held-out pulses, each against the shared no-input control):

- memory: the run-minus-control output response after the command ends, which
  only stored state can carry (direction accuracy A, RSE, shape error S);
- pulse: the same while the command acts;
- direct-input share: during the command, the response energy of the model's
  output terms that use the command directly, over that of those terms plus the
  terms that use a latent state (the true device has no direct term);
- split, for models whose output has its own equation: output-only terms
  against the true relaxation -d_y y, the other terms against the true drive G(z).

References come from integrating each device's private specification, which
reproduces the saved held-out records; models are rolled out with the compiled
candidate's restricted evaluator, so no model text is executed. Per seed: mean A
and S, median RSE; a failed rollout gives A = 0, S = 1, RSE = inf.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import statistics
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.expressions.parser import grammar_parser
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting.simulation import trajectory_forcing
from autoformalism.rebuttal.basin_algebra import expanded
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.schemas import CandidateModel
from scripts.score_declared_mechanisms import names_in, split_terms

TARGET, COMMAND = "v01", "u01"
SECONDS = 45.0


def command(item: dict, time: np.ndarray) -> np.ndarray:
    """The command samples of one suite intervention on the grid."""
    u = np.zeros_like(time)
    for on, off, value in item.get("segments", []):
        u += np.where((time >= on - 1e-9) & (time <= off + 1e-9), value, 0.0)
    if "sine" in item:
        s = item["sine"]
        end = s["start"] + s["cycles"] * s["period"]
        active = (time >= s["start"] - 1e-9) & (time <= end + 1e-9)
        u += np.where(
            active,
            s["amplitude"] * np.sin(2 * np.pi * (time - s["start"]) / s["period"]),
            0.0,
        )
    return u


def windows(u: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pulse window (zero sample before to zero sample after) and memory window."""
    active = np.flatnonzero(np.abs(u) > 1e-12)
    if not len(active):
        return np.zeros(len(time), bool), np.zeros(len(time), bool)
    start, stop = max(active[0] - 1, 0), min(active[-1] + 1, len(time) - 1)
    pulse = (np.arange(len(time)) >= start) & (np.arange(len(time)) <= stop)
    return pulse, np.arange(len(time)) > stop


def part(name: str, time: np.ndarray, u: np.ndarray, y0: float) -> dict:
    """A public-shaped trajectory: the command, and the initial output reading."""
    return {
        "trajectory_id": name,
        "time": time.tolist(),
        "external_inputs": {COMMAND: u.tolist()},
        "targets": {TARGET: [y0] + [0.0] * (len(time) - 1)},
        "auxiliaries": {},
    }


def drive(spec: dict, z: np.ndarray) -> float:
    total = sum(
        t["coefficient"] * np.tanh(t["scale"] * z[t["source"]])
        for t in spec["output_terms"]
    )
    return total + sum(
        t["coefficient"]
        * np.tanh(t["scale_1"] * z[t["source_1"]])
        * np.tanh(t["scale_2"] * z[t["source_2"]])
        for t in spec["output_product_terms"]
    )


def truth(spec: dict, record: dict) -> dict:
    """The output, internal drive G(z) and relaxation -d_y y on one trajectory."""
    n = int(spec["n_latent"])
    time = np.asarray(record["time"], float)
    u = np.asarray(record["external_inputs"][COMMAND], float)
    decay, skew = np.asarray(spec["decay"], float), np.asarray(spec["skew"], float)

    def rhs(s, x):
        z = x[:n]
        dz = -decay * z + skew @ z
        for i, terms in enumerate(spec["tanh_terms"]):
            for k in terms:
                dz[i] += k["coefficient"] * np.tanh(
                    k["scale"] * z[k["source"]] + k["bias"]
                )
        for i, terms in enumerate(spec["product_terms"]):
            for k in terms:
                dz[i] += (
                    k["coefficient"]
                    * np.tanh(k["scale_1"] * z[k["source_1"]])
                    * np.tanh(k["scale_2"] * z[k["source_2"]])
                )
        dz += np.asarray(spec["input_vector"], float) * np.tanh(
            spec["input_scale"] * np.interp(s, time, u)
        )
        return np.concatenate([dz, [-spec["output_decay"] * x[n] + drive(spec, z)]])

    x0 = np.zeros(n + 1)
    x0[n] = record["targets"][TARGET][0]
    sol = solve_ivp(
        rhs,
        (time[0], time[-1]),
        x0,
        method="LSODA",
        t_eval=time,
        rtol=1e-10,
        atol=1e-12,
        max_step=0.05,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    z, y = sol.y[:n], sol.y[n]
    return {
        "y": y,
        "G": np.array([drive(spec, z[:, k]) for k in range(len(time))]),
        "R": -spec["output_decay"] * y,
    }


def output_terms(row: dict) -> tuple[bool, list[tuple[int, str, str]]]:
    """Whether the output has its own equation, and its terms grouped by names."""
    candidate = CandidateModel.model_validate(row["candidate"])
    processes = {p.name: p.expression for p in candidate.processes}
    equations = {e.state: e.rhs for e in candidate.state_equations}
    dynamic = TARGET in equations
    text = (
        equations[TARGET]
        if dynamic
        else next(
            o.expression for o in candidate.observation_mappings if o.channel == TARGET
        )
    )
    latent = {s.name for s in candidate.states if s.name != TARGET}
    tree = expanded(text, processes, "default")
    groups = []
    for sign, node, _ in split_terms(tree):
        names = names_in(node) - set(row["parameters"])
        group = (
            "memory"
            if names & latent
            else ("direct_input" if COMMAND in names else "output_only")
        )
        groups.append((sign, ast.unparse(node), group))
    return dynamic, groups


def rollout(row: dict, record: dict, groups: list) -> dict:
    """The model's output and grouped output terms along one trajectory."""
    candidate = CandidateModel.model_validate(row["candidate"])
    context = ValidationContext.model_validate(row["context"])
    compiled = compile_candidate(candidate, context)
    time = np.asarray(record["time"], float)
    tr = Trajectory(
        trajectory_id=record["trajectory_id"],
        time=time,
        targets={TARGET: np.asarray(record["targets"][TARGET], float)},
        auxiliaries={},
        external_inputs={
            COMMAND: np.asarray(record["external_inputs"][COMMAND], float)
        },
        fixed_covariates={},
        derivatives={},
    )
    sim = simulate_trajectory(
        compiled,
        tr,
        row["parameters"],
        row["initials"],
        FitConfig(
            integration_method="Radau",
            relative_tolerance=1e-8,
            absolute_tolerance=1e-10,
            allow_derivative_regression=False,
        ),
        reset_observed_states=False,
        deadline=monotonic() + SECONDS,
    )
    if not sim.success:
        raise ValueError(sim.message)
    parser = grammar_parser("default")
    forcing = trajectory_forcing(compiled, tr)
    out = {g: np.zeros(len(time)) for g in ("memory", "direct_input", "output_only")}
    for sign, text, group in groups:
        expr = parser.parse(text, location="output term")
        out[group] += np.array(
            [
                sign
                * compiled.evaluate_expression(
                    expr, t, sim.states[:, k], row["parameters"], forcing
                )
                for k, t in enumerate(time)
            ]
        )
    return {"y": np.asarray(sim.predictions[TARGET], float), **out}


def measures(pred: np.ndarray, true: np.ndarray) -> dict | None:
    """Direction accuracy A, RSE and shape error S of a response, if there is one."""
    den = float(np.sum(true**2))
    if den == 0 or not len(true):
        return None
    norm, dot = float(np.sum(pred**2)), float(np.sum(pred * true))
    cos = dot / math.sqrt(norm * den) if norm > 0 else 0.0
    return {
        "A": float(
            np.sum(np.abs(true) * (np.sign(pred) == np.sign(true)))
            / np.sum(np.abs(true))
        ),
        "RSE": float(np.sum((pred - true) ** 2) / den),
        "S": 1 - max(0.0, cos) ** 2,
    }


FAILED = {"A": 0.0, "RSE": math.inf, "S": 1.0}


def trajectories(suite: dict, saved: list[dict]) -> list[dict]:
    """Every intervention with its control: the suite's and the saved pulses."""
    time = np.arange(0.0, suite["horizon"] + 1e-9, suite["dt"])
    control = part("no_input", time, np.zeros_like(time), 0.0)
    items = [
        {
            "name": it["name"],
            "run": part(it["name"], time, command(it, time), 0.0),
            "control": control,
            "memory_window": it.get("memory_window", True),
            "source": "suite",
        }
        for it in suite["interventions"]
    ]
    for record in saved:
        if record["case"] in suite["saved_held_out"]:
            items.append(
                {
                    "name": record["case"],
                    "run": record["perturbed"],
                    "control": record["control"],
                    "memory_window": True,
                    "source": "saved",
                }
            )
    return items


def score_device(
    device: str, spec: dict, rows: list[dict], items: list[dict]
) -> list[dict]:
    """Per model and intervention: memory, pulse, direct-input share and split."""
    truths = {}
    for it in items:
        for key in ("run", "control"):
            name = it[key]["trajectory_id"] + (
                "" if it["source"] == "suite" else "_" + key
            )
            if name not in truths:
                truths[name] = truth(spec, it[key])
    out = []
    for row in rows:
        entry = {
            "device": device,
            "method": row["method"],
            "repetition": row["repetition"],
            "status": row["status"],
        }
        if row["status"] != "ready":
            out.append(entry)
            continue
        dynamic, groups = output_terms(row)
        entry.update(
            dynamic_output=dynamic,
            terms=[{"sign": s, "expression": e, "group": g} for s, e, g in groups],
            items=[],
        )
        cache = {}
        for it in items:
            u = np.asarray(it["run"]["external_inputs"][COMMAND], float)
            time = np.asarray(it["run"]["time"], float)
            pulse, memory = windows(u, time)
            keys = {
                k: it[k]["trajectory_id"] + ("" if it["source"] == "suite" else "_" + k)
                for k in ("run", "control")
            }
            true = {k: truths[keys[k]] for k in keys}
            try:
                model = {}
                for k in ("run", "control"):
                    if keys[k] not in cache:
                        cache[keys[k]] = rollout(row, it[k], groups)
                    model[k] = cache[keys[k]]
            except (ValueError, ArithmeticError, RuntimeError) as exc:
                entry["items"].append(
                    {
                        "name": it["name"],
                        "status": f"failed: {exc}"[:200],
                        "memory": FAILED if it["memory_window"] else None,
                        "pulse": FAILED,
                    }
                )
                continue
            d_true = true["run"]["y"] - true["control"]["y"]
            d_model = model["run"]["y"] - model["control"]["y"]
            response = {
                g: model["run"][g] - model["control"][g]
                for g in ("memory", "direct_input", "output_only")
            }
            direct, memory_e = (
                float(np.sum(response["direct_input"][pulse] ** 2)),
                float(np.sum(response["memory"][pulse] ** 2)),
            )
            item = {
                "name": it["name"],
                "status": "complete",
                "memory": measures(d_model[memory], d_true[memory])
                if it["memory_window"]
                else None,
                "pulse": measures(d_model[pulse], d_true[pulse]),
                "direct_input_share": direct / (direct + memory_e)
                if direct + memory_e > 0
                else None,
            }
            if dynamic:
                item["split"] = {
                    "relaxation": measures(
                        response["output_only"], true["run"]["R"] - true["control"]["R"]
                    ),
                    "drive": measures(
                        response["memory"] + response["direct_input"],
                        true["run"]["G"] - true["control"]["G"],
                    ),
                }
            entry["items"].append(item)
        out.append(entry)
    return out


def seed_summary(entry: dict) -> dict | None:
    """Mean A and S, median RSE over the interventions, per measure."""
    if entry["status"] != "ready":
        return None
    result = {}
    for key in ("memory", "pulse"):
        values = [i[key] for i in entry["items"] if i.get(key) is not None]
        if values:
            result[key] = (
                float(np.mean([v["A"] for v in values])),
                statistics.median(v["RSE"] for v in values),
                float(np.mean([v["S"] for v in values])),
            )
    shares = [
        i["direct_input_share"]
        for i in entry["items"]
        if i.get("direct_input_share") is not None
    ]
    result["direct_input_share"] = float(np.mean(shares)) if shares else None
    if entry["dynamic_output"]:
        for part_name in ("relaxation", "drive"):
            values = [
                i["split"][part_name]
                for i in entry["items"]
                if i.get("split") and i["split"][part_name]
            ]
            if values:
                result[part_name] = (
                    float(np.mean([v["A"] for v in values])),
                    statistics.median(v["RSE"] for v in values),
                    float(np.mean([v["S"] for v in values])),
                )
    result["failed"] = sum(i["status"] != "complete" for i in entry["items"])
    return result


def main(stress: Path, suite_path: Path, out: Path) -> dict:
    suite = json.loads(suite_path.read_text())
    if suite.get("status") != "frozen_before_references":
        raise ValueError("the suite is not frozen")
    bundle = sealed_read(stress / "frozen/frozen_models.json")
    devices = sorted(
        {
            r["benchmark_id"]
            for r in bundle["rows"]
            if str(r["benchmark_id"]).startswith("device_")
        }
    )
    entries = []
    for device in devices:
        spec = json.loads(
            (stress / f"design/evaluator/{device}/system.json").read_text()
        )["value"]["equations"]
        saved = json.loads(
            (stress / f"evaluation/references/{device}.json").read_text()
        )["value"]["records"]
        items = trajectories(suite, saved)
        for record in saved:
            for key in ("control", "perturbed"):
                gap = float(
                    np.max(
                        np.abs(
                            truth(spec, record[key])["y"]
                            - np.asarray(record[key]["targets"][TARGET])
                        )
                    )
                )
                if gap > 1e-6:
                    raise ValueError(
                        f"{device}: the equations miss {record['case']} ({gap:.2e})"
                    )
        rows = [r for r in bundle["rows"] if r["benchmark_id"] == device]
        entries += score_device(device, spec, rows, items)
    for e in entries:
        e["summary"] = seed_summary(e)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "protocol": "alien-device-memory-scores-1",
        "suite": suite["suite"],
        "suite_sha256": __import__("hashlib")
        .sha256(suite_path.read_bytes())
        .hexdigest(),
        "stress_bundle_sha256": bundle["artifact_sha256"],
        "entries": entries,
        "private_reference_opened": True,
        "test_data_opened": False,
    }
    (out / "alien_memory.json").write_text(json.dumps(record, indent=1, default=float))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stress", type=Path, required=True, help="the stress study's delivered packet"
    )
    parser.add_argument(
        "--suite", type=Path, default=Path("configs/alien_device_memory_suite_v1.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/alien-memory-stress-devices-v1")
    )
    a = parser.parse_args()
    result = main(a.stress, a.suite, a.out)
    print(len(result["entries"]), "models scored")
