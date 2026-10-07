"""M16: coupled linear rollout profiling, including shared latent initial values."""

from copy import deepcopy
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import generic_recovery as common
from autoformalism.fitting import identifiable_cases as controls
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_fit
from autoformalism.fitting.bounded_screening import TrainingOnlySplit
from autoformalism.fitting.coordinates import training_coordinates
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

PROTOCOL = "phase-c-coupled-profile-1"
INPUT_PROTOCOL = "phase-c-coupled-profile-inputs-1"
ARMS = ("rollout_only", "coupled_profiled_rollout")
CASES = ("coupled_linear", "coupled_fast_slow")
CoupledPolicy = common.RecoveryPolicy


def export_inputs(output: Path) -> dict:
    """Generate standalone development controls, leaving benchmark files untouched."""
    if output.exists():
        data = read_seal(output)
        bases(data)
        return {"identity": public.content_sha256(data), "starts": 6}
    template = controls.make_inputs()["cases"]["linear"]
    cases, commons = {}, {}
    for name, coefficients in zip(
        CASES,
        ({"a": 0.8, "b": 1.3, "c": 1.6}, {"a": 18.0, "b": 2.0, "c": 2.0}),
        strict=True,
    ):
        truth = {**coefficients, "init_z_value": 0.4}
        case = deepcopy(template)
        for split in ("training", "validation"):
            states = [controls.reference(row, truth) for row in case[split]["rows"]]
            for row, x in zip(case[split]["rows"], states, strict=True):
                row["targets"]["y"] = x[:, 0].tolist()
            case[split]["fingerprint"] = public.content_sha256(case[split]["rows"])
            if split == "training":
                case["identifiability"] = controls.excitation(
                    case[split]["rows"], states, nonlinear=False
                )
        if not case["identifiability"]["passed"]:
            raise ValueError("coupled control lacks excitation")
        case["reference_parameters"] = truth
        cases[name] = case
        training = public.unpack_split(PublicSplit.model_validate(case["training"]))
        for seed in range(3):
            # Same three generic requests for both timescales and both optimizers.
            request = controls.request("linear", seed)
            model, start, _ = public._lower(request)
            commons[f"{name}_s{seed}"] = {
                "case": name,
                "seed": seed,
                "request": request.model_dump(mode="json"),
                "coordinates": training_coordinates(model, training, start).model_dump(
                    mode="json"
                ),
                "nodes": {},
                "start": start,
            }
    data = {
        "protocol": INPUT_PROTOCOL,
        "cases": cases,
        "commons": commons,
        "config": {
            "cstr_nmse": 1e-6,
            "parameter_relative": 0.01,
            "initial_absolute": 0.001,
        },
        "test_data_opened": False,
        "benchmark_release_modified": False,
        "noise": "none",
        "shared_latent_initial": True,
    }
    bases(data)
    seal(output, data)
    return {"identity": public.content_sha256(data), "starts": 6}


def bases(data: dict) -> dict:
    """Strict roster and training allowlist: no truth or validation in fit payloads."""
    if data["protocol"] != INPUT_PROTOCOL or data["test_data_opened"] is not False:
        raise ValueError("wrong coupled control input contract")
    expected = {f"{case}_s{i}" for case in CASES for i in range(3)}
    if set(data["commons"]) != expected or set(data["cases"]) != set(CASES):
        raise ValueError("require all six coupled control starts")
    result = {}
    for key, entry in data["commons"].items():
        if entry["case"] not in CASES or key != f"{entry['case']}_s{entry['seed']}":
            raise ValueError("coupled case/seed identity differs")
        request = PublicFitRequest.model_validate(entry["request"])
        if entry["start"] != public._lower(request)[1]:
            raise ValueError("generic request/start mismatch")
        training = data["cases"][entry["case"]]["training"]
        TrainingOnlySplit.model_validate(training)
        result[key] = {
            k: deepcopy(entry[k]) for k in ("request", "coordinates", "nodes", "start")
        }
        result[key]["training"] = deepcopy(training)
    return result


def prepare(root: Path, inputs: Path, policy: CoupledPolicy) -> dict:
    return common.prepare(
        root, inputs, policy, protocol=PROTOCOL, arms=ARMS, base_factory=bases
    )


def verify(root: Path, *, runtime=True):
    return common.verify(root, runtime=runtime, protocol=PROTOCOL)


def fit(base: dict, arm: str, policy: dict, folder: Path) -> dict:
    if arm not in ARMS:
        raise ValueError("unknown coupled profiling arm")
    return recovery_fit.fit(
        base,
        "rollout_only",
        policy,
        folder,
        rollout_mode="recovery_rollout" if arm == "rollout_only" else arm,
    )


def evaluation_data(data: dict, task: dict) -> dict:
    """Resolve held-out/reference data only after the selected fit has been sealed."""
    case = data["commons"][task["common"]]["case"]
    return {"case": data["cases"][case], "case_name": case, "config": data["config"]}


def run_task(root: Path, index: int) -> dict:
    return common.run_task(
        root,
        index,
        protocol=PROTOCOL,
        fitter=fit,
        base_factory=bases,
        evaluation_data=evaluation_data,
    )


def report(root: Path) -> dict:
    result = common.report(root, protocol=PROTOCOL, arms=ARMS)
    _, data = verify(root, runtime=False)
    for row in result["rows"]:
        entry = data["commons"][row["common"]]
        row.update(case=entry["case"], seed=entry["seed"])
        path = root / "results" / row["task_id"] / "backend.json"
        if path.exists() and row["status"] != "missing":
            value = (read_seal(path).get("rollout") or {}).get("value") or {}
            row["optimizer"] = {
                k: value.get(k)
                for k in (
                    "stop_reason",
                    "actual_residual_calls",
                    "accounting",
                    "profiled_output",
                    "selected_inner",
                )
            }
    result["limitation"] = (
        "Two noiseless correct-skeleton linear controls; three generic starts each. "
        "One identity output and one shared latent initial. Conditional bounded LS "
        "is exact up to integration/solver tolerance; no outer global guarantee. "
        "Ideal continuous identifiability does not certify sample conditioning. "
        "No nonlinear-block qualification or production promotion. Independent "
        "replay and coefficient errors are reported separately."
    )
    with public._lock(root / "report-lock"):
        public._write(root / "summary.json", result)
    return result
