"""M27: immutable development observations and training-only generic starts."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import nonlinear_comparison_campaign as historical
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.coordinates import training_coordinates
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

PROTOCOL = "phase-c-broad-scaling-inputs-1"
CASES = ("alien_hard", "basin_coupled", "cstr_hard")
STARTS = 5


def generic_request(name: str, payload: dict, seed: int) -> PublicFitRequest:
    """Perturb supplied generic guesses, without reference values or fitted points.

    Seeds 0..2 preserve the M3/M6 rules. Seeds 3..4 use a wider log window.
    Concentration is nonnegative; temperature initials use an additive window.
    """
    payload = deepcopy(payload)
    cstr = name == "cstr_hard"
    rng = np.random.default_rng((20260930 if cstr else 20261002) + seed)
    width = (0.7 if cstr else (0.5, 1.0, 1.5)[seed]) if seed < 3 else 1.5
    payload["parameter_guesses"] = {
        n: float(v * np.exp(rng.uniform(-width, width)))
        for n, v in payload["parameter_guesses"].items()
    }
    for state, rule in payload["initialization_plan"]["rules"].items():
        initial = rule["initial"]
        if initial["mode"] != "value":
            continue
        if cstr:
            initial["guess"] = (
                float(initial["guess"] + rng.uniform(-3, 3) * width / 0.7)
                if state == "Tj"
                else float(initial["guess"] * np.exp(rng.uniform(-width, width)))
            )
            if state == "C":
                initial["role"] = "nonnegative_coefficient"
        else:
            initial["guess"] = float(rng.uniform(-width, width))
    return PublicFitRequest.model_validate(payload)


def common(name: str, case: dict, seed: int) -> dict:
    """Build numerical coordinates from training and generic initial guesses."""
    request = generic_request(name, case["request"], seed)
    model, start, _ = public._lower(request)
    train = public.unpack_split(PublicSplit.model_validate(case["training"]))
    return {
        "case": name,
        "seed": seed,
        "request": request.model_dump(mode="json"),
        "start": start,
        "coordinates": training_coordinates(
            model, train, start, state_channels=case.get("state_channel_proxies")
        ).model_dump(mode="json"),
    }


def bases(data: dict) -> dict[str, checks.TrainingProblem]:
    """Pass only an explicit training allowlist to the fitting controller."""
    expected = {f"{n}_s{s}" for n in CASES for s in range(STARTS)}
    if (
        data["protocol"] != PROTOCOL
        or data["test_data_opened"] is not False
        or set(data["cases"]) != set(CASES)
        or set(data["commons"]) != expected
    ):
        raise ValueError("broad scaling requires exactly three cases and five starts")
    result = {}
    for key, item in data["commons"].items():
        name = item["case"]
        if key != f"{name}_s{item['seed']}":
            raise ValueError("case/start identity differs")
        request = PublicFitRequest.model_validate(item["request"])
        if public._lower(request)[1] != item["start"]:
            raise ValueError("generic start differs from request")
        schema = (
            checks.PiecewiseTrainingProblem
            if name in {"basin_coupled", "cstr_hard"}
            else checks.TrainingProblem
        )
        result[key] = schema.model_validate(
            {k: item[k] for k in ("request", "coordinates", "start")}
            | {"incumbent": item["start"], "training": data["cases"][name]["training"]}
        )
    return result


def export_inputs(m26: Path, challenging: Path, strategies: Path, output: Path) -> dict:
    """Copy existing sealed arrays; never simulate, refit, or inspect test data."""
    old, hard, cstr = map(read_seal, (m26, challenging, strategies))
    historical.bases(old)  # Pin the exact original M26 source, not a warm endpoint.
    if (
        hard["protocol"] != "phase-c-challenging-fitting-inputs-1"
        or cstr["protocol"] != "phase-c-fitting-strategy-inputs-1"
        or hard["test_data_opened"] is not False
        or cstr["test_data_opened"] is not False
        or old["case"] != hard["cases"]["alien_hard"]
    ):
        raise ValueError("development input provenance differs")
    cases = {
        "alien_hard": hard["cases"]["alien_hard"],
        "basin_coupled": hard["cases"]["basin_coupled"],
        "cstr_hard": cstr["cases"]["cstr_hard"],
    }
    commons = {}
    for name, case in cases.items():
        for seed in range(STARTS):
            key = f"{name}_s{seed}"
            if name == "alien_hard" and seed < 3:
                item = old["commons"][key]
                commons[key] = {
                    k: item[k]
                    for k in ("case", "seed", "request", "start", "coordinates")
                }
            else:
                commons[key] = common(name, case, seed)
            public._check_data(
                PublicFitRequest.model_validate(commons[key]["request"]),
                PublicSplit.model_validate(case["training"]),
                PublicSplit.model_validate(case["validation"]),
            )
    data = {
        "protocol": PROTOCOL,
        "cases": cases,
        "commons": commons,
        "source_digests": {
            k: public.content_sha256(v)
            for k, v in (("m26", old), ("m6", hard), ("m3", cstr))
        },
        "evaluation_thresholds": {
            "cstr_nmse": 1e-8,
            "parameter_relative": 0.01,
            "initial_absolute": 1e-3,
        },
        "historical_fits_used": False,
        "test_data_opened": False,
    }
    bases(data)
    seal(output, data)
    return {"inputs_sha256": public.content_sha256(data), "cases": 3, "starts": 15}
