"""Score frozen generated components against private hidden test trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.execution import ExecutionArguments, _context
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.fitting.simulation import trajectory_forcing
from autoformalism.rebuttal.hidden import hidden_mechanism_nmse
from autoformalism.schemas import CandidateModel

PRIVATE_REFERENCES = {
    "original_b1": (
        "benchmark1_original_dalla_man/benchmarks/B1_meal_appearance",
        ("Ra",),
        "hidden_{split}.csv",
    ),
    "perturbed_b1": (
        "benchmark2_perturbed_dalla_man/B1_meal_appearance",
        ("Ra",),
        "hidden_{split}.csv",
    ),
    "obfuscated_original_case01": (
        "benchmark3_obfuscated_dalla_man/private/case_01/hidden_ground_truth",
        ("Ra",),
        "hidden_{split}_semantic.csv",
    ),
    "obfuscated_perturbed_case01": (
        "benchmark4_obfuscated_perturbed_dalla_man/private/case_01/hidden_ground_truth",
        ("Ra",),
        "hidden_{split}_semantic.csv",
    ),
    "benchmark5": (
        "benchmark5_anonymous_nonlinear_process/private",
        ("C", "Tj"),
        "hidden_{split}.csv",
    ),
    "benchmark6": (
        "benchmark6_alien_device/private",
        ("z1", "z2", "z3", "z4", "z5"),
        "hidden_{split}.csv",
    ),
}


def _reference_values(
    path: Path, columns: tuple[str, ...], trajectory_ids: tuple[str, ...]
) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    pieces = {column: [] for column in columns}
    if "trajectory_id" in frame:
        grouped = {str(key): value for key, value in frame.groupby("trajectory_id")}
        for identifier in trajectory_ids:
            group = grouped[identifier]
            for column in columns:
                pieces[column].append(group[column].to_numpy(dtype=float))
    else:
        if len(trajectory_ids) != 1:
            raise ValueError(f"reference {path} has no trajectory identifiers")
        for column in columns:
            pieces[column].append(frame[column].to_numpy(dtype=float))
    return {key: np.concatenate(value) for key, value in pieces.items()}


def _generated_values(
    model: object,
    split: object,
    parameters: dict[str, float],
    global_initials: dict[str, float],
    local_initials: dict[str, dict[str, float]],
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {
        state.name: []
        for state in model.validated.candidate.states
        if state.kind.value == "latent"
    }
    pieces.update({name: [] for name in model.validated.process_order})
    settings = FitConfig(integration_backend="solve_ivp")
    for trajectory in split.trajectories:
        initials = local_initials.get(
            f"{prefix}:{trajectory.trajectory_id}",
            local_initials.get(trajectory.trajectory_id, {}),
        )
        simulation = simulate_trajectory(
            model,
            trajectory,
            parameters,
            {**global_initials, **initials},
            settings,
        )
        if not simulation.success or simulation.states is None:
            raise RuntimeError(simulation.message or "hidden simulation failed")
        for state in model.validated.candidate.states:
            if state.kind.value == "latent":
                pieces[state.name].append(
                    simulation.states[model.state_names.index(state.name)]
                )
        for process_name in model.validated.process_order:
            expression = model.validated.process_expressions[process_name]
            values = []
            for index, time in enumerate(trajectory.time):
                forcing = trajectory_forcing(
                    model, trajectory, causal_index=max(0, index - 1)
                )
                values.append(
                    model.evaluate_expression(
                        expression,
                        float(time),
                        simulation.states[:, index],
                        parameters,
                        forcing,
                    )
                )
            pieces[process_name].append(np.asarray(values, dtype=float))
    return {key: np.concatenate(value) for key, value in pieces.items()}


def _teacher_forced_process_values(
    model: object,
    split: object,
    parameters: dict[str, float],
) -> dict[str, np.ndarray]:
    """Evaluate algebraic processes on the observed D3 state trajectory."""
    pieces: dict[str, list[np.ndarray]] = {
        name: [] for name in model.validated.process_order
    }
    for trajectory in split.trajectories:
        channels = {**trajectory.targets, **trajectory.auxiliaries}
        missing = set(model.state_names) - set(channels)
        if missing:
            raise ValueError(
                f"teacher-forced states are unavailable: {sorted(missing)}"
            )
        states = np.vstack([channels[name] for name in model.state_names])
        for process_name in model.validated.process_order:
            expression = model.validated.process_expressions[process_name]
            values = []
            for index, time in enumerate(trajectory.time):
                forcing = trajectory_forcing(
                    model, trajectory, causal_index=max(0, index - 1)
                )
                values.append(
                    model.evaluate_expression(
                        expression,
                        float(time),
                        states[:, index],
                        parameters,
                        forcing,
                    )
                )
            pieces[process_name].append(np.asarray(values, dtype=float))
    return {key: np.concatenate(value) for key, value in pieces.items()}


def _d3_candidate(path: Path, payload: dict) -> CandidateModel:
    checkpoint = json.loads(
        (path.parent / "d3_checkpoint.json").read_text(encoding="utf-8")
    )
    generation = int(payload["selected_hyperparameters"]["selected_generation"])
    record = next(
        item for item in checkpoint["records"] if int(item["generation"]) == generation
    )
    candidate = record["candidate"]
    fitted = json.loads(payload["selected_hyperparameters"]["selected_parameters"])
    # Native D3's unconstrained Adam fit can leave the proposal's declared
    # interval. Expand metadata solely so the frozen expression can be replayed.
    for parameter in candidate["parameters"]:
        value = float(fitted[parameter["name"]])
        parameter["bounds"]["lower"] = min(
            float(parameter["bounds"]["lower"]), value
        )
        parameter["bounds"]["upper"] = max(
            float(parameter["bounds"]["upper"]), value
        )
    return CandidateModel.model_validate(candidate)


def _score(
    train_generated: dict[str, np.ndarray],
    test_generated: dict[str, np.ndarray],
    train_reference: dict[str, np.ndarray],
    test_reference: dict[str, np.ndarray],
) -> tuple[float, dict[str, str], dict[str, float]]:
    if not train_generated:
        return np.nan, {}, {}
    selected: dict[str, str] = {}
    scores: dict[str, float] = {}
    for reference_name, train_truth in train_reference.items():
        training_scores = {}
        for generated_name, train_value in train_generated.items():
            metric = hidden_mechanism_nmse(
                train_value,
                train_truth,
                train_value,
                train_truth,
                allow_signed_scale=True,
            )
            training_scores[generated_name] = metric.test_nmse
        chosen = min(training_scores, key=training_scores.get)
        selected[reference_name] = chosen
        scores[reference_name] = hidden_mechanism_nmse(
            train_generated[chosen],
            train_truth,
            test_generated[chosen],
            test_reference[reference_name],
            allow_signed_scale=True,
        ).test_nmse
    return float(np.mean(list(scores.values()))), selected, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="*")
    parser.add_argument("--methods", nargs="*")
    args = parser.parse_args()
    runs = pd.read_csv(args.runs)
    runs = runs[
        runs.method.isin(("full", "nojudge", "d3_native_no_tools"))
        & runs.test_mse.notna()
    ]
    if args.benchmarks:
        runs = runs[runs.benchmark.isin(args.benchmarks)]
    if args.methods:
        runs = runs[runs.method.isin(args.methods)]
    loader = BenchmarkLoader(BenchmarkRegistry())
    rows = []
    for row in runs.itertuples(index=False):
        error = None
        try:
            payload = json.loads(Path(row.source).read_text(encoding="utf-8"))
            is_d3 = row.method == "d3_native_no_tools"
            candidate = (
                _d3_candidate(Path(row.source), payload)
                if is_d3
                else CandidateModel.model_validate(payload["frozen"]["candidate"])
            )
            config = DataConfig(
                root=args.data_root.resolve(),
                benchmark_id=row.benchmark,
                tier=row.tier,
            )
            development = loader.load_development(config)
            test = loader.load_test(config)
            arguments = ExecutionArguments(
                data_root=config.root,
                benchmark_id=row.benchmark,
                tier=row.tier,
                seed=int(row.seed),
                proposer_model=None,
                judge_model=None,
                iteration_budget=1,
                beam_size=1,
                output_root=args.output.parent,
                resume=False,
                dry_run=False,
                mock_llm=True,
                use_clean_observations=False,
            )
            model = compile_candidate(candidate, _context(arguments, development))
            if is_d3:
                parameters = json.loads(
                    payload["selected_hyperparameters"]["selected_parameters"]
                )
                train_generated = _teacher_forced_process_values(
                    model, development.train, parameters
                )
                test_generated = _teacher_forced_process_values(model, test, parameters)
            else:
                fit = payload["final_fit"]
                parameters = dict(fit["global_parameters"])
                global_initials = dict(fit["global_initial_conditions"])
                train_generated = _generated_values(
                    model,
                    development.train,
                    parameters,
                    global_initials,
                    dict(fit["training_trajectory_initial_conditions"]),
                    prefix="train",
                )
                test_generated = _generated_values(
                    model,
                    test,
                    parameters,
                    global_initials,
                    dict(payload.get("test_initials", {})),
                    prefix="test",
                )
            relative, columns, template = PRIVATE_REFERENCES[row.benchmark]
            reference_root = args.data_root / relative
            train_reference = _reference_values(
                reference_root / template.format(split="train"),
                columns,
                tuple(item.trajectory_id for item in development.train.trajectories),
            )
            test_reference = _reference_values(
                reference_root / template.format(split="test"),
                columns,
                tuple(item.trajectory_id for item in test.trajectories),
            )
            score, matching, per_reference = _score(
                train_generated, test_generated, train_reference, test_reference
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            score, matching, per_reference = np.nan, {}, {}
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "method": row.method,
                "benchmark": row.benchmark,
                "tier": row.tier,
                "seed": row.seed,
                "hidden_mse": score,
                "train_selected_matching": json.dumps(matching, sort_keys=True),
                "per_reference_hidden_mse": json.dumps(
                    per_reference, sort_keys=True
                ),
                "error": error,
                "source": row.source,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
