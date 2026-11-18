"""Mechanism scores on the Phase B nine-case glucose and alien cells.

Each cell's six held-out test protocols are scored against matched controls:
the same protocol without its meals and insulin (glucose) or its command
(alien), from the same initial shift. References come from the Phase B generator
with the cell's dynamics and input contract, projected to the cell's public
channel names (so the obfuscated cells read the same physiology). A model is
rolled out on each run and control with its own semantics (ODE, native one-step
map, hybrid map, latent replay) and its target equation's additive terms are
evaluated along its own rollout:

glucose (Gp target)  meal pathway against the true appearance Ra; insulin pathway
                     (T2) on the strong-orthogonal test against a counterfactual
                     with the same meal and no insulin, where every change of the
                     true dGp/dt is caused by the insulin; the remainder.
alien (v01 target)   memory response after the command ends, pulse response,
                     direct-command share of the response, output relaxation and
                     drive against -d_y y and G(z).

Per contrast: direction accuracy A, RSE, shape error S of the response. Per seed:
mean A and S, median RSE; a failed rollout gives A = 0, S = 1, RSE = inf. Test
NMSE: per trajectory, mean over targets of MSE over the training variance.
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

from autoformalism.baselines.d3_native import _evaluate
from autoformalism.baselines.d3_rollout import _NUMPY
from autoformalism.baselines.d3_rollout import predict as native_rollout
from autoformalism.baselines.hybrid_networks import numpy_outputs
from autoformalism.baselines.models import BaselineDevelopmentResult
from autoformalism.benchmarks import phase_b_protocols, simulate_phase_b
from autoformalism.benchmarks.phase_b_public import _private_array, phase_b_public_spec
from autoformalism.benchmarks.phase_c_release import load_public_cell
from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.expressions.parser import grammar_parser
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting.simulation import trajectory_forcing
from autoformalism.rebuttal import phase_b_nine_case as nine
from autoformalism.rebuttal import phase_c_baseline_mechanisms as mechanisms
from autoformalism.rebuttal.basin_algebra import expanded
from autoformalism.rebuttal.dalla_man import DallaManParameters
from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.schemas import CandidateModel
from scripts import score_baseline_interventions as scoring
from scripts.score_declared_mechanisms import names_in, split_terms

CELLS = {
    "phase_b_dalla_man_t1_canonical_named_easy": (
        "dalla_man",
        "T1",
        "easy",
        "named",
        "canonical",
    ),
    "phase_b_anonymous_system_t1_canonical_obfuscated_easy": (
        "dalla_man",
        "T1",
        "easy",
        "obfuscated",
        "canonical",
    ),
    "phase_b_dalla_man_t1_perturbed_named_easy": (
        "dalla_man",
        "T1",
        "easy",
        "named",
        "perturbed",
    ),
    "phase_b_anonymous_system_t1_perturbed_obfuscated_easy": (
        "dalla_man",
        "T1",
        "easy",
        "obfuscated",
        "perturbed",
    ),
    "phase_b_dalla_man_t1_canonical_named_hard": (
        "dalla_man",
        "T1",
        "hard",
        "named",
        "canonical",
    ),
    "phase_b_dalla_man_t2_canonical_named_easy": (
        "dalla_man",
        "T2",
        "easy",
        "named",
        "canonical",
    ),
    "phase_b_dalla_man_t2_canonical_named_hard": (
        "dalla_man",
        "T2",
        "hard",
        "named",
        "canonical",
    ),
    "phase_b_alien_device_unknown_device_mechanism_canonical_functional_easy": (
        "alien_device",
        None,
        "easy",
        "functional",
        "canonical",
    ),
}
#: Phase C versions of the glucose cells, for models searched under the current protocol
#: (continuous meal rates); scored on the same test protocols in their contract.
PHASE_C = {
    f"phase_c_{name}_rates_v1": spec
    for name, spec in {
        "dalla_man_t1_canonical_named_easy": (
            "dalla_man",
            "T1",
            "easy",
            "named",
            "canonical",
        ),
        "anonymous_system_t1_canonical_obfuscated_easy": (
            "dalla_man",
            "T1",
            "easy",
            "obfuscated",
            "canonical",
        ),
        "dalla_man_t1_perturbed_named_easy": (
            "dalla_man",
            "T1",
            "easy",
            "named",
            "perturbed",
        ),
        "anonymous_system_t1_perturbed_obfuscated_easy": (
            "dalla_man",
            "T1",
            "easy",
            "obfuscated",
            "perturbed",
        ),
        "dalla_man_t1_canonical_named_hard": (
            "dalla_man",
            "T1",
            "hard",
            "named",
            "canonical",
        ),
        "dalla_man_t2_canonical_named_easy": (
            "dalla_man",
            "T2",
            "easy",
            "named",
            "canonical",
        ),
        "dalla_man_t2_canonical_named_hard": (
            "dalla_man",
            "T2",
            "hard",
            "named",
            "canonical",
        ),
    }.items()
}
CONTRACT = {
    **dict.fromkeys(CELLS, "legacy-events-1"),
    **dict.fromkeys(PHASE_C, "continuous-rates-1"),
}
CELLS = {**CELLS, **PHASE_C}
PROTOCOLS = {
    "d3": "phase-b-nine-case-d3-family-1",
    "hdtwingen": "phase-b-nine-case-d3-family-1",
    "spec_sindy": "phase-b-nine-case-spec-sindy-1",
}
SIGNS = {"EGP": 1.0, "Uii": -1.0, "E": -1.0}
SECONDS = 45.0
FAILED = {"A": 0.0, "RSE": math.inf, "S": 1.0}
PHASE_C_RELEASE = Path(
    "/Users/yuanzhangxiao/Projects/autoformalism/artifacts/phase-c-development-final-v2"
)


# --- references --------------------------------------------------------------


def control_of(protocol, *, keep_meals: bool = False):
    """The same protocol without its inputs (or only without insulin), same shift."""
    spec = dict(protocol.specification)
    if protocol.family == "dalla_man":
        if not keep_meals:
            spec["meals"] = []
        spec.pop("insulin", None)
        for key in ("insulin_pulses", "insulin_infusions"):
            spec.pop(key, None)
    else:
        spec = {
            "kind": "constant",
            "value": 0.0,
            **(
                {"initial_shift": spec["initial_shift"]}
                if "initial_shift" in spec
                else {}
            ),
        }
    suffix = "_counterfactual" if keep_meals else "_control"
    return protocol.model_copy(
        update={"protocol_id": protocol.protocol_id + suffix, "specification": spec}
    )


def public_trajectory(public, private, name):
    """The private trajectory under the cell's public channel names and roles."""
    roles = {"target": {}, "auxiliary": {}, "external_input": {}}
    for channel in public.channels:
        roles[channel.role][channel.public_name] = np.asarray(
            _private_array(private, channel.private_source), float
        )
    return Trajectory(
        trajectory_id=name,
        time=np.asarray(private.time, float),
        targets=roles["target"],
        auxiliaries=roles["auxiliary"],
        external_inputs=roles["external_input"],
        fixed_covariates={},
        derivatives={},
    )


def references(cell: str, data_root: Path) -> dict:
    """Run, control (and T2's counterfactual) of every test protocol, with truths."""
    family, task, tier, variant, dynamics = CELLS[cell]
    contract = CONTRACT[cell]
    public = phase_b_public_spec(
        family,
        tier,
        variant,
        task=task,
        dynamics=dynamics,
        data_root=data_root,
        input_contract=contract,
    )
    expected = cell.replace("phase_c_", "phase_b_", 1) if cell in PHASE_C else cell
    if public.benchmark_id != expected:
        raise ValueError(
            f"{cell}: the public specification renders {public.benchmark_id}"
        )
    names = {c.public_name: c.private_source for c in public.channels}
    every = (
        phase_b_protocols(family, task=task, input_contract=contract)
        if task
        else phase_b_protocols(family, input_contract=contract)
    )
    protocols = [p for p in every if p.split == "test"]
    out = {"public": public, "names": names, "cases": []}
    for p in protocols:
        case = {"name": p.protocol_id}
        for key, proto in (("run", p), ("control", control_of(p))):
            private = simulate_phase_b(proto, dynamics=dynamics, data_root=data_root)
            case[key] = {
                "private": private,
                "public": public_trajectory(public, private, proto.protocol_id),
            }
        if (
            family == "dalla_man"
            and task == "T2"
            and any(
                p.specification.get(k)
                for k in ("insulin", "insulin_pulses", "insulin_infusions")
            )
        ):
            proto = control_of(p, keep_meals=True)
            private = simulate_phase_b(proto, dynamics=dynamics, data_root=data_root)
            case["counterfactual"] = {
                "private": private,
                "public": public_trajectory(public, private, proto.protocol_id),
            }
        out["cases"].append(case)
    return out


def insulin_terms(private) -> np.ndarray:
    """The reference's insulin terms along a trajectory, as glucose fluxes.

    Production's suppression by delayed plasma insulin (the Id term of EGP) and
    the insulin-dependent part of utilization (the X term of Uid), each as its
    contribution at the trajectory's own state (clipping included). Glucose
    feedback through insulin-independent terms and portal insulin is excluded.
    """
    p = DallaManParameters()
    state = {
        n: np.asarray(private.states[:, i], float)
        for i, n in enumerate(private.state_names)
    }
    gp, gt, ipo = (np.maximum(state[n], 0.0) for n in ("Gp", "Gt", "Ipo"))
    base = p.kp1 - p.kp2 * gp - p.kp4 * ipo
    production = np.maximum(base - p.kp3 * state["Id"], 0.0) - np.maximum(base, 0.0)
    saturation = gt / (p.Km0 + gt)
    utilization = np.maximum(
        (p.Vm0 + p.Vmx * state["X"]) * saturation, 0.0
    ) - np.maximum(p.Vm0 * saturation, 0.0)
    return production - utilization


def glucose_truth(private, supplied: list[str]) -> dict:
    derived = private.derived
    meal = np.asarray(derived["Ra"], float)
    flux = sum(
        (SIGNS[c] * np.asarray(derived[c], float) for c in supplied),
        np.zeros_like(meal),
    )
    total = np.asarray(private.derivatives["Gp"], float)
    return {
        "meal": meal,
        "supplied": flux,
        "remainder": total - meal - flux,
        "total": total,
    }


def alien_truth(private, spec: dict) -> dict:
    y = np.asarray(private.states[:, private.state_names.index("y")], float)
    relax = -float(spec["output_decay"]) * y
    return {
        "y": y,
        "R": relax,
        "G": np.asarray(private.derivatives["y"], float) - relax,
    }


# --- models ------------------------------------------------------------------


def nine_case_rows(method: str, run_root: Path, inputs_root: Path) -> list[dict]:
    """The nine-case run's pick per Phase B cell and seed, failures included."""
    plan = sealed_read(run_root / "plan.json")
    if (
        plan["protocol"] != PROTOCOLS[method]
        or plan.get("test_data_opened") is not False
    ):
        raise ValueError(f"{run_root} is not a sealed {method} nine-case run")
    rows = []
    for source in plan["rows"]:
        index, name = int(source["index"]), source["benchmark_id"]
        if name not in CELLS:
            continue
        cell = nine.load_cell(
            inputs_root,
            name,
            context="baseline" if method == "spec_sindy" else "raw_agent",
        )
        if cell.identity != source["public_identity"]:
            raise ValueError(f"{method} task {index} read other public files")
        row = {
            **mechanisms._row(method, name, source["tier"], int(source["repetition"])),
            "task": index,
        }
        path = run_root / "results" / str(index) / "result.json"
        result = sealed_read(path)
        if (
            result["plan_sha256"] != plan["artifact_sha256"]
            or result.get("test_data_opened") is not False
        ):
            raise ValueError(f"{path} belongs to another plan or opened test data")
        sources = [mechanisms._file(run_root / "plan.json"), mechanisms._file(path)]
        if result["status"] in scoring.SERVICE_STATUSES:
            raise ValueError(f"{row['row_id']} ended {result['status']}: rerun it")
        if result["status"] != "complete":
            rows.append(
                mechanisms._failure(row, sources, result["status"], result.get("error"))
            )
            continue
        context = cell.context
        if method == "spec_sindy":
            model = mechanisms._model(
                equation_candidate("llm_feature_sindy", result["equations"], context),
                {},
                context,
                "continuous_time",
            )
        else:
            sealed = sealed_read(path.with_name("native-selection.json"))
            payload = BaselineDevelopmentResult.model_validate(
                sealed["selection"]
            ).selection_payload
            step = float(np.median(np.diff(cell.dataset.train.trajectories[0].time)))
            model = mechanisms._model(
                CandidateModel.model_validate(payload["candidate"]),
                payload["parameters"],
                context,
                "native_increment",
                increment_step=step,
                networks=list(payload.get("networks") or ()),
                network_weights=payload.get("network_weights"),
            )
        rows.append(mechanisms._ready(row, sources, model))
    return rows


def our_rows(
    bundle_path: Path, cell: str, label: str, benchmark_ids: set[str]
) -> list[dict]:
    """Our frozen candidates as rows the scorer reads, with the default grammar."""
    bundle = sealed_read(bundle_path)
    rows = []
    for r in bundle["rows"]:
        if r["benchmark_id"] not in benchmark_ids:
            continue
        row = {
            "method": label
            if r.get("method") in (None, "full")
            else f"{label}:{r['method']}",
            "benchmark_id": cell,
            "repetition": int(r["repetition"]),
            "status": r["status"],
        }
        if r["status"] == "ready":
            row["model"] = {
                "candidate": r["candidate"],
                "context": r["context"],
                "parameters": r["parameters"],
                "initials": r.get("initials") or {},
                "semantics": "continuous_time",
                "grammar": "default",
            }
        rows.append(row)
    return rows


def submitted_rows(
    models_path: Path, inputs_root: Path, methods: tuple[str, ...] = ("sindy", "pysr")
) -> list[dict]:
    """The submitted SINDy and PySR fits (numeric right-hand sides), Phase B cells."""
    rows, contexts = [], {}
    for line in models_path.read_text().splitlines():
        r = json.loads(line)
        name = r["benchmark_id"]
        if r["method"] not in methods or name not in CELLS:
            continue
        row = {
            "method": r["method"],
            "benchmark_id": name,
            "repetition": int(r["repetition"]),
        }
        if r["adapter_status"] != "adapted" or not r["equations"]:
            rows.append({**row, "status": r["terminal_status"] or r["adapter_status"]})
            continue
        if name not in contexts:
            contexts[name] = nine.load_cell(
                inputs_root, name, context="baseline"
            ).context
        context = contexts[name]
        model = mechanisms._model(
            equation_candidate(r["method"], r["equations"], context),
            {},
            context,
            "continuous_time",
        )
        rows.append({**row, "status": "ready", "model": model})
    return rows


def historical_rows(plan_path: Path, label: str) -> list[dict]:
    """Archived Full models, exported with their fitted shared initial values."""
    plan = sealed_read(plan_path)
    rows = []
    for r in plan["rows"]:
        if r["cell"] not in CELLS:
            continue
        row = {
            "method": label,
            "benchmark_id": r["cell"],
            "repetition": int(r["seed"]),
            "status": r["status"],
        }
        if r["status"] == "ready":
            row["model"] = {
                "candidate": r["candidate"],
                "context": r["context"],
                "parameters": r["parameters"],
                "initials": r.get("initials") or {},
                "semantics": "continuous_time",
                "grammar": "default",
            }
        rows.append(row)
    return rows


def expand_names(
    names: set[str],
    processes: dict[str, str],
    networks: dict[str, list[str]],
    grammar: str,
) -> set[str]:
    """Names of a term, processes inlined and networks replaced by their inputs."""
    out, todo, seen = set(), list(names), set()
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        if n in processes:
            todo += list(names_in(expanded(processes[n], processes, grammar)))
        elif n in networks:
            todo += list(networks[n])
        else:
            out.add(n)
    return out


def terms_of(model: dict, target: str) -> tuple[ast.AST, dict, dict, str]:
    """The target equation with processes inlined, the processes, network inputs."""
    candidate = CandidateModel.model_validate(model["candidate"])
    grammar = model.get("grammar", "baseline")
    processes = {p.name: p.expression for p in candidate.processes}
    networks = {}
    for spec in model.get("networks") or ():
        outputs = spec.get("outputs") or [spec.get("name")]
        for o in outputs:
            networks[o] = list(spec.get("inputs") or ())
    equations = {e.state: e.rhs for e in candidate.state_equations}
    text = (
        equations[target]
        if target in equations
        else next(
            o.expression for o in candidate.observation_mappings if o.channel == target
        )
    )
    grammar_parser(grammar).parse(text, location="target equation")
    return expanded(text, processes, grammar), processes, networks, text


def group_curves(
    row: dict, tr, target: str, groups: list[tuple[int, str, str]]
) -> dict:
    """The model's targets and grouped terms (per time unit) along its own rollout.

    "y" is the mechanism target; "pred" holds every target channel of the model.
    """
    model = row["model"]
    targets = list(model["context"]["targets"])
    time = np.asarray(tr.time, float)
    out = {g: np.zeros(len(time)) for g in {g for _, _, g in groups}}
    if model["semantics"] == "native_increment":
        native = scoring._map(model)
        values = native_rollout(native, tr, teacher_forced=False, seconds=SECONDS)
        parsed = [
            grammar_parser("baseline").parse(t, location="term").tree.body
            for _, t, _ in groups
        ]
        for k in range(len(time)):
            env = dict(zip(native.states, values[k], strict=True))
            env["t"] = float(time[k])
            env.update({n: v[k] for n, v in tr.external_inputs.items()})
            env.update({n: v[k] for n, v in tr.auxiliaries.items()})
            env.update(native.parameters)
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                if native.networks:
                    env.update(
                        numpy_outputs(
                            native.networks,
                            native.network_inputs,
                            native.network_weights,
                            env,
                            _evaluate,
                            _NUMPY,
                        )
                    )
                for name, node in native.processes:
                    env[name] = _evaluate(node, env, _NUMPY)
                for (sign, _, g), node in zip(groups, parsed, strict=True):
                    out[g][k] += sign * _evaluate(node, env, _NUMPY)
        step = float(model["increment_step"])
        pred = {n: values[:, native.states.index(n)] for n in targets}
        return {
            "y": pred[target],
            "pred": pred,
            **{g: v / step for g, v in out.items()},
        }
    if model["semantics"] in scoring.REPLAYS:
        curves = scoring.predict(row, tr, "Radau")
        pred = {n: np.asarray(curves[n], float) for n in targets}
        return {"y": pred[target], "pred": pred}
    candidate = CandidateModel.model_validate(model["candidate"])
    context = ValidationContext.model_validate(model["context"])
    grammar = model.get("grammar", "baseline")
    compiled = compile_candidate(
        candidate,
        context,
        validator=scoring.baseline_validator() if grammar == "baseline" else None,
    )
    sim = simulate_trajectory(
        compiled,
        tr,
        model["parameters"],
        model.get("initials", {}),
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
    parser = grammar_parser(grammar)
    forcing = trajectory_forcing(compiled, tr)
    for sign, text, g in groups:
        expr = parser.parse(text, location="term")
        out[g] += np.array(
            [
                sign
                * compiled.evaluate_expression(
                    expr, t, sim.states[:, k], model["parameters"], forcing
                )
                for k, t in enumerate(time)
            ]
        )
    pred = {n: np.asarray(sim.predictions[n], float) for n in targets}
    return {"y": pred[target], "pred": pred, **out}


# --- measures ----------------------------------------------------------------


def measures(pred, true) -> dict | None:
    true = np.asarray(true, float)
    pred = np.asarray(pred, float)
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


def windows(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = np.flatnonzero(np.abs(u) > 1e-12)
    n = len(u)
    if not len(active):
        return np.zeros(n, bool), np.zeros(n, bool)
    start, stop = max(active[0] - 1, 0), min(active[-1] + 1, n - 1)
    idx = np.arange(n)
    return (idx >= start) & (idx <= stop), idx > stop


def score_row(row: dict, refs: dict, train_var: dict, device: dict | None) -> dict:
    """One pick on every test protocol of its cell."""
    cell = row["benchmark_id"]
    family, task = CELLS[cell][0], CELLS[cell][1]
    entry = {k: row[k] for k in ("method", "benchmark_id", "repetition", "status")}
    if row["status"] != "ready":
        return entry
    names = refs["names"]
    model = row["model"]
    context = ValidationContext.model_validate(model["context"])
    target = next(
        n for n, s in names.items() if s == ("Gp" if family == "dalla_man" else "y")
    )
    if model.get("semantics") in scoring.REPLAYS:
        # a latent replay has no readable terms: only its outputs are scored
        groups = []
        entry["readable_terms"] = False
    else:
        try:
            tree, processes, networks, _ = terms_of(model, target)
            latent = {s["name"] for s in model["candidate"]["states"]} - {target}
            aux = set(context.auxiliaries)
            inputs = set(context.external_inputs)
            grammar = model.get("grammar", "baseline")
            equations = {
                e["state"]: e["rhs"] for e in model["candidate"]["state_equations"]
            }
            observations = {
                o["channel"]: o["expression"]
                for o in model["candidate"]["observation_mappings"]
            }
            # time and parameters are shared by a run and its control: no pathway
            drop = set(model["parameters"]) | {context.time_symbol}

            def used_by(text: str) -> set[str]:
                """Private names an equation uses (processes inlined)."""
                raw_uses = expand_names(
                    names_in(expanded(text, processes, grammar)),
                    processes,
                    networks,
                    grammar,
                )
                return {names.get(n, n) for n in raw_uses} - drop

            uses = {
                s: used_by(equations[s])
                for s in latent - aux - set(context.targets)
                if s in equations
            }
            modeled = {
                names.get(s, s): used_by(equations.get(s, observations.get(s, "0")))
                for s in (set(equations) | set(observations)) - {target}
            }

            def driven(sources: set[str]) -> set[str]:
                """Latent states driven only by these sources and such states."""
                found: set[str] = set()
                changed = True
                while changed:
                    changed = False
                    for state, used in uses.items():
                        channels = used & ({*names.values(), *uses} | sources)
                        if (
                            state not in found
                            and channels & (sources | found)
                            and channels <= sources | found | {state}
                        ):
                            found.add(state)
                            changed = True
                return found

            def reaching(sources: set[str]) -> set[str]:
                """Other modeled states and channels using these, transitively."""
                found: set[str] = set()
                changed = True
                while changed:
                    changed = False
                    for key, used in modeled.items():
                        if key not in found | sources and used & (sources | found):
                            found.add(key)
                            changed = True
                return found

            meal_sources = {"meal_event_g", "meal_rate_g_per_min"}
            insulin_sources = {"I", "insulin_pmol_per_kg_min"}
            meal_states, insulin_states = driven(meal_sources), driven(insulin_sources)
            # e.g. a modeled disposal U, or tissue glucose driven by insulin action
            carriers = (
                reaching(insulin_sources | insulin_states) - meal_states
                if task == "T2"
                else set()
            )
            entry["meal_states"], entry["insulin_states"] = (
                sorted(meal_states),
                sorted(insulin_states),
            )
            entry["insulin_carriers"] = sorted(carriers)

            def classify(node: ast.AST) -> str | None:
                """The pathway of a term by the names it uses; None for constants."""
                raw = expand_names(names_in(node), processes, networks, grammar) - drop
                if not raw:
                    return None
                if family != "dalla_man":
                    return (
                        "memory"
                        if raw & (latent | aux)
                        else ("direct_input" if raw & inputs else "output_only")
                    )
                private = {names.get(n, n) for n in raw}
                meal = bool(private & (meal_sources | meal_states))
                insulin = bool(private & (insulin_sources | insulin_states | carriers))
                rest = (
                    private
                    - meal_sources
                    - insulin_sources
                    - meal_states
                    - insulin_states
                    - carriers
                )
                if meal and (insulin or rest - {"Gp"}):
                    return "mixed"
                if meal:
                    return "meal"
                if insulin and not (rest - {"Gp"}):
                    return "insulin"
                return "supplied" if private <= set(SIGNS) else "remainder"

            # a sum in a term (a deviation like I - Ib) splits only across pathways
            constant = "remainder" if family == "dalla_man" else "output_only"
            groups = [
                (sign, ast.unparse(node), classify(node) or constant)
                for sign, node, _ in split_terms(tree, classify)
            ]
        except (ModelValidationError, ValueError, StopIteration, KeyError) as exc:
            entry["status"] = f"unreadable: {exc}"[:200]
            return entry
        entry["groups"] = [
            {"sign": s, "expression": t[:160], "group": g} for s, t, g in groups
        ]
    present = {g for _, _, g in groups}
    if family == "dalla_man" and entry.get("readable_terms", True):
        required = ["meal", "insulin"] if task == "T2" else ["meal"]
        entry["missing"] = [p for p in required if p not in present]
    supplied = [names[c] for c in context.auxiliaries if names.get(c) in SIGNS]
    targets = list(model["context"]["targets"])
    entry["cases"] = []
    for case in refs["cases"]:
        item = {"name": case["name"]}
        run_pub, ctl_pub = case["run"]["public"], case["control"]["public"]
        u = np.asarray(next(iter(run_pub.external_inputs.values())), float)
        pulse, after = windows(u)
        dy_true = np.asarray(run_pub.targets[target], float) - np.asarray(
            ctl_pub.targets[target], float
        )
        truth = {"target": dy_true}
        if family == "dalla_man":
            t_run, t_ctl = (
                glucose_truth(case[k]["private"], supplied) for k in ("run", "control")
            )
            truth.update(
                {
                    p: t_run[p] - t_ctl[p]
                    for p in ("meal", "supplied", "remainder")
                    if p in present
                }
            )
            if "counterfactual" in case and "insulin" in present:
                truth["insulin_effect"] = insulin_terms(
                    case["run"]["private"]
                ) - insulin_terms(case["counterfactual"]["private"])
        else:
            t_run, t_ctl = (
                alien_truth(case[k]["private"], device) for k in ("run", "control")
            )
            truth.update({"memory": dy_true[after], "pulse": dy_true[pulse]})
            if "output_only" in present:
                truth["relaxation"] = t_run["R"] - t_ctl["R"]
        # a contrast is defined only where the true response is not zero
        truth = {k: v for k, v in truth.items() if len(v) and float(np.sum(v**2)) > 0}
        try:
            curves = {
                k: group_curves(row, case[k]["public"], target, groups)
                for k in ("run", "control", "counterfactual")
                if k in case
            }
        except (
            ValueError,
            ArithmeticError,
            FloatingPointError,
            RuntimeError,
            TimeoutError,
        ) as exc:
            item["status"] = f"failed: {exc}"[:160]
            item.update(dict.fromkeys(truth, FAILED))
            entry["cases"].append(item)
            continue
        item["status"] = "complete"
        item["nmse"] = float(
            np.mean(
                [
                    np.mean((curves["run"]["pred"][n] - run_pub.targets[n]) ** 2)
                    / train_var[n]
                    for n in targets
                ]
            )
        )
        dy_model = curves["run"]["y"] - curves["control"]["y"]
        terms = {g: curves["run"][g] - curves["control"][g] for g in present}
        response = {"target": dy_model}
        if family == "dalla_man":
            response.update(
                {p: terms[p] for p in ("meal", "supplied", "remainder") if p in terms}
            )
            if "counterfactual" in curves and "insulin" in present:
                response["insulin_effect"] = (
                    curves["run"]["insulin"] - curves["counterfactual"]["insulin"]
                )
        else:
            # the memory measure is the output itself after the command; the term
            # groups (memory, direct_input, output_only) enter the share and relaxation
            response.update({"memory": dy_model[after], "pulse": dy_model[pulse]})
            if "output_only" in terms:
                response["relaxation"] = terms["output_only"]
            if pulse.any():
                zero = np.zeros(len(u))
                de = float(np.sum(terms.get("direct_input", zero)[pulse] ** 2))
                me = float(np.sum(terms.get("memory", zero)[pulse] ** 2))
                item["direct_input_share"] = de / (de + me) if de + me > 0 else None
        for key, true in truth.items():
            item[key] = measures(response[key], true)
        entry["cases"].append(item)
    return entry


KEYS = (
    "target",
    "meal",
    "insulin_effect",
    "supplied",
    "remainder",
    "memory",
    "pulse",
    "relaxation",
)


def summary(entry: dict) -> dict | None:
    """Per seed: mean NMSE over the test trajectories (the official test score), and
    per contrast mean A and S and median RSE over the cases that define it."""
    if entry["status"] != "ready" or "cases" not in entry:
        return None
    cases = entry["cases"]
    out = {"failed": sum(c["status"] != "complete" for c in cases)}
    out["test_nmse"] = float(
        np.mean([c["nmse"] if c["status"] == "complete" else math.inf for c in cases])
    )
    for key in KEYS:
        vals = [c[key] for c in cases if c.get(key)]
        if vals:
            out[key] = (
                float(np.mean([v["A"] for v in vals])),
                statistics.median(v["RSE"] for v in vals),
                float(np.mean([v["S"] for v in vals])),
            )
    shares = [
        c["direct_input_share"]
        for c in cases
        if c.get("direct_input_share") is not None
    ]
    if shares:
        out["direct_input_share"] = float(np.mean(shares))
    return out


def main(
    methods: dict[str, Path],
    inputs_root: Path,
    data_root: Path,
    out: Path,
    extra_rows: list[dict] | None = None,
) -> dict:
    rows = [
        r for m, p in methods.items() for r in nine_case_rows(m, p, inputs_root)
    ] + list(extra_rows or [])
    entries = []
    device = json.loads(
        (
            data_root / "benchmark6_alien_device/private/selected_system_spec.json"
        ).read_text()
    )
    for cell in CELLS:
        cell_rows = [r for r in rows if r["benchmark_id"] == cell]
        if not cell_rows:
            continue
        refs = references(cell, data_root)
        if cell in PHASE_C:
            _, training, _ = load_public_cell(PHASE_C_RELEASE / "public" / cell)
            train_var = {
                n: float(np.var(np.concatenate([r.targets[n] for r in training.rows])))
                for n in training.rows[0].targets
            }
        else:
            loaded = nine.load_cell(inputs_root, cell, context="baseline")
            train = loaded.dataset.train.trajectories
            train_var = {
                n: float(np.var(np.concatenate([t.targets[n] for t in train])))
                for n in loaded.context.targets
            }
        for r in cell_rows:
            e = score_row(r, refs, train_var, device)
            e["summary"] = summary(e)
            entries.append(e)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "protocol": "phase-b-mechanism-scores-1",
        "entries": entries,
        "test_data_opened": True,
        "private_reference_opened": True,
    }
    (out / "phase_b_mechanisms.json").write_text(
        json.dumps(record, indent=1, default=float)
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="METHOD=PATH of a sealed nine-case run",
    )
    parser.add_argument(
        "--ours",
        action="append",
        default=[],
        help="LABEL=CELL=BENCHMARK_ID=PATH of a frozen bundle of our models",
    )
    parser.add_argument(
        "--historical",
        action="append",
        default=[],
        help="LABEL=PATH of an exported plan of archived Full models",
    )
    parser.add_argument(
        "--submitted",
        type=Path,
        help="models.jsonl of the submitted SINDy and PySR fits",
    )
    parser.add_argument(
        "--latent",
        action="append",
        default=[],
        help="METHOD=PATH of a sealed latent-model selection (e.g. latent_ode)",
    )
    parser.add_argument(
        "--inputs", type=Path, required=True, help="the staged nine-case inputs"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_raw"))
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    runs = dict(item.split("=", 1) for item in a.run)
    extra = []
    for item in a.ours:
        label, cell, benchmark, path = item.split("=", 3)
        extra += our_rows(Path(path), cell, label, {benchmark})
    for item in a.historical:
        label, path = item.split("=", 1)
        extra += historical_rows(Path(path), label)
    if a.submitted:
        extra += submitted_rows(a.submitted, a.inputs)
    for item in a.latent:
        method, path = item.split("=", 1)
        extra += scoring.latent_rows(method, Path(path))
    result = main(
        {m: Path(p) for m, p in runs.items()},
        a.inputs,
        a.data_root,
        a.out,
        extra_rows=extra,
    )
    print(len(result["entries"]), "picks scored")
