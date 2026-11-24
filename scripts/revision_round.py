"""One round of bounded model revision from three saved scientific models."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
from hashlib import sha256
from pathlib import Path
from time import monotonic

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import revision_round_requests as prompts
from scripts.revision_round_provider import CASES, SEEDS, digest, load_stage, save

REPO = Path(__file__).resolve().parents[1]


def adapter(case: str):
    return importlib.import_module(f"scripts.revision_round_{case}")


def write_stage(root: Path, stage: str, requests: dict) -> None:
    path = root / f"{stage}.json"
    record = {"stage": stage, "requests": requests, "sha256": digest(requests)}
    if path.exists() and load_stage(path) != record:
        raise ValueError("Request stage changed, use a separate experiment directory")
    save(path, record)


def prepare(root: Path) -> dict:
    """Freeze train-only packets, the algorithm candidates, and all generation seeds."""
    packets = {case: adapter(case).packet() for case in CASES}
    requests = {
        f"{case}-{seed}": prompts.generation(packets[case], seed)
        for case in CASES
        for seed in SEEDS
    }
    plan = sealed_write(
        root / "plan.json",
        {
            "protocol": "bounded-one-round-1",
            "packets": packets,
            "seeds": list(SEEDS),
            "generation_sha256": digest(requests),
            "selection": (
                "lowest selection_score, default training NMSE, exact tie keeps parent"
            ),
            "pools": ["algorithm", "20b_first_three", "20b_all_nine"],
        },
    )
    write_stage(root, "generate", requests)
    return plan


def reply(root: Path, stage: str, label: str, request: dict) -> tuple[object, dict]:
    path = root / f"{stage}-responses" / f"{label}.json"
    receipt = json.loads(path.read_text())
    if receipt.get("request_sha256") != digest(request):
        raise ValueError("Provider receipt differs from the frozen request")
    if receipt["status"] == "reserved":
        raise RuntimeError(f"Unresolved provider reservation for {stage}/{label}")
    cost = {k: receipt[k] for k in ("status", "seconds") if k in receipt}
    response = receipt.get("response", {})
    cost["usage"] = response.get("usage")
    if receipt["status"] != "responded":
        return None, cost
    try:
        message = response["choices"][0]["message"]["content"]
        answer = json.loads(message)
        digest(answer)  # Reject nonfinite numbers, including overflowed exponents.
    except (KeyError, IndexError, TypeError, ValueError):
        answer = None
    return answer, cost


def identity(edit: object) -> str:
    """Canonical JSON identity with equal numeric literals grouped."""

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if type(value) is float and value.is_integer():
            return int(value)
        return value

    return digest(clean(edit))


def proposed(root: Path, case: str, generated: dict) -> tuple[list, dict]:
    options, costs = [], {}
    for seed in SEEDS:
        label = f"{case}-{seed}"
        result, costs[label] = reply(root, "generate", label, generated[label])
        edits = result.get("alternatives") if isinstance(result, dict) else None
        options.extend(
            edits if isinstance(edits, list) and len(edits) == 3 else [None] * 3
        )
    return options, costs


def metric(record: dict) -> float:
    return record.get("selection_score", record["train_nmse"])


def fit_execution(case: str, packet: dict) -> str:
    """Bind cached scores to their evidence, evaluation code and package versions."""
    from importlib.metadata import version

    from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash

    paths = [
        "revision_round.py",
        f"revision_round_{case}.py",
        "thermal_balance_trial.py",
        "thermal_forecast_continuation.py",
        "basin_balance_inference.py",
        "basin_family_stability.py",
        "basin_function_family.py",
        "trajectory_functions.py",
        "fit_t2_initial_disposal_order.py",
        "positive_insulin_refit_kernel.py",
        "t2_disposal_coupled_replay.py",
    ]
    return digest(
        [
            packet,
            runtime_source_hash(),
            {p: sha256((REPO / "scripts" / p).read_bytes()).hexdigest() for p in paths},
            {p: version(p) for p in ("numpy", "scipy", "casadi")},
        ]
    )


def prepare_screens(root: Path) -> dict:
    """Freeze training-derived shortlists before inspecting proposer draws."""
    plan = sealed_read(root / "plan.json")
    screens = {}
    for case, packet in plan["packets"].items():
        module = adapter(case)
        if module.packet() != packet:
            raise ValueError("Training evidence changed since generation")
        screens[case] = (
            module.screen()
            if case != "basin"
            else {
                "edits": packet["algorithm_edits"],
                "screened": 24,
                "criterion": "Rank nonparent power pairs using measured flow residuals",
            }
        )
    return sealed_write(
        root / "screens.json",
        {
            "plan": plan["artifact_sha256"],
            "screens": screens,
            "policy": "Three edits per batch, three independent 20B batches. "
            "Include the parent for all selectors. Rank on the blinded union.",
        },
    )


def fit_case(root: Path, case: str, packet: dict, draws: list, screen: dict) -> dict:
    """Charge slots for repeats and invalid edits, fit each distinct edit once."""
    module = adapter(case)
    baseline = packet["parent"]
    arms = {
        "algorithm": packet["algorithm_edits"],
        "algorithm_screen": screen["edits"],
        "20b_all_nine": draws,
    }
    candidates = {identity(baseline): baseline}
    slots = {}
    for arm, edits in arms.items():
        keys = []
        for edit in edits:
            if not isinstance(edit, dict):
                keys.append(None)
                continue
            key = identity(edit)
            candidates.setdefault(key, edit)
            keys.append(key)
        slots[arm] = keys
    for i, name in enumerate(
        ("20b_first_three", "20b_seed101_three", "20b_seed102_three")
    ):
        slots[name] = slots["20b_all_nine"][3 * i : 3 * i + 3]
    execution = fit_execution(case, packet)
    fits = {}
    for key, edit in candidates.items():
        path = root / "fits" / case / f"{key}.json"
        if path.is_file():
            record = sealed_read(path)
        else:
            start = monotonic()
            result = module.score(edit)
            record = sealed_write(
                path,
                {
                    "edit": edit,
                    "result": result,
                    "fit_execution": execution,
                    "seconds": monotonic() - start,
                },
            )
        if identity(record["edit"]) != key or record.get("fit_execution") != execution:
            raise ValueError("Cached edit differs")
        result = record["result"]
        if result["eligible"] and not math.isfinite(metric(result)):
            raise ValueError("Eligible edit needs a finite training selection metric")
        fits[key] = result
    parent = identity(baseline)
    eligible = {k: e for k, e in candidates.items() if fits[k]["eligible"]}
    if not eligible:
        raise ValueError(f"No eligible models for {case}")
    return {
        "parent": parent,
        "slots": slots,
        "eligible": eligible,
        "fits": fits,
        "screen": screen,
        "execution": execution,
    }


def shortlist(root: Path) -> dict:
    """Fit all unique edits and create blinded ranking requests before development."""
    plan = sealed_read(root / "plan.json")
    design = sealed_read(root / "screens.json")
    if design["plan"] != plan["artifact_sha256"]:
        raise ValueError("Shortlists differ from generation plan")
    requests = load_stage(root / "generate.json")
    if requests["sha256"] != plan["generation_sha256"]:
        raise ValueError("Generation plan differs")
    cases, ranking_requests, maps, costs = {}, {}, {}, {}
    for case in CASES:
        packet = plan["packets"][case]
        draws, cost = proposed(root, case, requests["requests"])
        costs.update(cost)
        table = cases[case] = fit_case(
            root, case, packet, draws, design["screens"][case]
        )
        for seed in SEEDS:
            label = f"{case}-{seed}"
            ranking_requests[label], maps[label] = prompts.ranking(
                packet, table["eligible"], case, seed
            )
    result = sealed_write(
        root / "training.json",
        {
            "plan": plan["artifact_sha256"],
            "screens": design["artifact_sha256"],
            "cases": cases,
            "label_maps": maps,
            "generation_cost": costs,
            "rank_sha256": digest(ranking_requests),
        },
    )
    write_stage(root, "rank", ranking_requests)
    return result


def selections(table: dict, judge_orders: dict) -> dict:
    """Compare selectors on the same eligible pool, always allowing the refit parent."""
    rows = {}
    parent = table["parent"]
    for arm, keys in table["slots"].items():
        pool = [k for k in dict.fromkeys([parent, *keys]) if k in table["eligible"]]
        rows[arm] = {
            "algorithm": min(
                pool, key=lambda k: (metric(table["fits"][k]), k != parent, k)
            )
            if pool
            else None,
            "20b": {
                s: next((k for k in order if k in pool), None)
                for s, order in judge_orders.items()
            },
            "uniform": {
                str(s): random.Random(digest([s, sorted(pool)])).choice(sorted(pool))
                if pool
                else None
                for s in SEEDS
            },
            "eligible_slots": sum(k in table["eligible"] for k in keys),
            "distinct_eligible_edits": len(set(keys) & (set(pool) - {parent})),
            "slots": len(keys),
        }
    return rows


def freeze(root: Path) -> dict:
    """Freeze all decisions before the first development observation is accessed."""
    training = sealed_read(root / "training.json")
    requests = load_stage(root / "rank.json")
    if requests["sha256"] != training["rank_sha256"]:
        raise ValueError("Ranking plan differs")
    choices, costs = {}, {}
    for case, table in training["cases"].items():
        orders = {}
        for seed in SEEDS:
            label = f"{case}-{seed}"
            answer, cost = reply(root, "rank", label, requests["requests"][label])
            labels = training["label_maps"][label]
            rank = answer.get("rank") if isinstance(answer, dict) else None
            valid = (
                isinstance(rank, list)
                and all(isinstance(v, str) for v in rank)
                and len(rank) == len(labels)
                and set(rank) == set(labels)
            )
            costs[label] = {**cost, "permutation_valid": valid}
            orders[str(seed)] = [labels[v] for v in rank] if valid else []
        choices[case] = selections(table, orders)
    return sealed_write(
        root / "frozen.json",
        {
            "training": training["artifact_sha256"],
            "choices": choices,
            "rank_cost": costs,
        },
    )


def validate(root: Path) -> dict:
    training, frozen = (
        sealed_read(root / "training.json"),
        sealed_read(root / "frozen.json"),
    )
    if frozen["training"] != training["artifact_sha256"]:
        raise ValueError("Freeze differs")
    scores = {}
    for case, rows in frozen["choices"].items():
        table = training["cases"][case]
        packet = sealed_read(root / "plan.json")["packets"][case]
        if table["execution"] != fit_execution(case, packet):
            raise ValueError("Validation executable differs from the training freeze")
        keys = {table["parent"]}
        for decision in rows.values():
            keys.update(
                [
                    decision["algorithm"],
                    *decision["20b"].values(),
                    *decision["uniform"].values(),
                ]
            )
        scores[case] = {}
        for key in sorted(keys & table["eligible"].keys()):
            path = root / "development" / case / f"{key}.json"
            if not path.exists():
                sealed_write(
                    path,
                    {
                        "result": adapter(case).validate(
                            table["eligible"][key], table["fits"][key]
                        ),
                        "freeze": frozen["artifact_sha256"],
                    },
                )
            record = sealed_read(path)
            if record["freeze"] != frozen["artifact_sha256"]:
                raise ValueError(
                    "Development receipt differs from the selection freeze"
                )
            scores[case][key] = record["result"]
    return sealed_write(
        root / "development.json",
        {
            "freeze": frozen["artifact_sha256"],
            "cases": scores,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = {
        "prepare": prepare,
        "screens": prepare_screens,
        "shortlist": shortlist,
        "freeze": freeze,
        "validate": validate,
    }
    parser.add_argument("stage", choices=stages)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    with _lock(args.root / "lock"):
        result = stages[args.stage](args.root)
    print(args.stage, result["artifact_sha256"])
