"""Compare fitted equation wiring and public behavior across archived seed pairs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from itertools import permutations, product
from math import perm
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

from autoformalism.expressions.parser import APPROVED_FUNCTION_ARITY
from autoformalism.fitting.public_fitting import _lock, content_sha256
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import CandidateModel
from autoformalism.search.identity import candidate_identity
from scripts.iclr_active_history import graph_for, overlap, reachable
from scripts.iclr_submission_evidence import digest, table

PHASE_B_INVENTORY = "ba5a433bbd32ff23e43faf07ff2750b1616c17af3be1bac93df4d9f7a292b97b"
PHASE_B_HISTORY = "2e9f583f4df7e9a6b0e35a630a69aa3721d95ba9c34ec5570f5c889ae96b49d7"
PHASE_C_PLAN = "315ca6edb7c6dac776f00fb450e8a9f0cc584e7a9c0cbc1fbea5e0c47fd92780"
Edge = tuple[str, str]


@dataclass
class Wiring:
    """Fitted one-equation dependencies with shared public ports."""

    states: dict[str, str]
    edges: set[Edge]


def fitted_wiring(model: dict) -> Wiring:
    """Inline algebraic names and retain differential states upstream of a target."""
    candidate = model["candidate"]
    context = model["lowered_context"]
    graph = graph_for(candidate, model["parameters"])
    states = {s["name"]: s["kind"] for s in candidate["states"]}
    processes = {p["name"] for p in candidate["processes"]}
    targets = set(context["targets"])
    if {m["channel"] for m in candidate["observation_mappings"]} != targets:
        raise ValueError("Incomplete public target mapping")
    supplied = set().union(
        *(
            context.get(name, [])
            for name in (
                "external_inputs",
                "auxiliaries",
                "fixed_covariates",
                "lagged_targets",
            )
        )
    )
    supplied.add(context.get("time_symbol", "t"))

    @cache
    def sources(name: str, trail: frozenset[str]) -> frozenset[str]:
        if name in states:
            return frozenset(["state:" + name])
        if name in trail:
            raise ValueError("Algebraic cycle in fitted model")
        if name in processes:
            return frozenset().union(*(sources(n, trail | {name}) for n in graph[name]))
        if name in supplied:
            return frozenset(["public:" + name])
        if name in APPROVED_FUNCTION_ARITY:
            return frozenset()
        raise ValueError(f"Undeclared dependency {name}")

    edges = set()
    active = {}
    queue = [("output:" + t, "output:" + t) for t in sorted(targets)]
    while queue:
        owner, dest = queue.pop()
        for name in graph[owner]:
            for src in sources(name, frozenset()):
                edges.add((src, dest))
                if src.startswith("state:") and src not in active:
                    state = src.removeprefix("state:")
                    active[src] = states[state]
                    queue.append((state, src))
    return Wiring(active, edges)


def _assignments(left: list[str], right: list[str]):
    """Relabel right states onto the left, including all unequal-size matches."""
    if len(right) <= len(left):
        return (
            dict(zip(right, matched, strict=True))
            for matched in permutations(left, len(right))
        )
    return (
        dict(zip(matched, left, strict=True))
        for matched in permutations(right, len(left))
    )


def best_wiring(a: Wiring, b: Wiring) -> tuple[float | None, int]:
    """Maximize direct-edge Jaccard over all same-kind one-to-one alignments."""
    kinds = sorted(set(a.states.values()) | set(b.states.values()))
    groups = [
        (
            sorted(k for k, v in a.states.items() if v == kind),
            sorted(k for k, v in b.states.items() if v == kind),
        )
        for kind in kinds
    ]
    count = 1
    for lnames, rnames in groups:
        count *= perm(max(len(lnames), len(rnames)), min(len(lnames), len(rnames)))
    if count > 1_000_000:
        raise ValueError("Exact state alignment exceeds the enumeration limit")
    best = 0
    for maps in product(*(_assignments(left, right) for left, right in groups)):
        # Unmatched right states must differ even from left states of the same name.
        names = {n: "unmatched:" + n for n in b.states}
        for mapping in maps:
            names.update(mapping)
        shared = sum((names.get(s, s), names.get(t, t)) in a.edges for s, t in b.edges)
        best = max(best, shared)
    union = len(a.edges) + len(b.edges) - best
    return (best / union if union else None), count


def describe(model: dict, checkpoint: str) -> dict:
    """Compute comparable structural projections of an authenticated fitted model."""
    if content_sha256(model["candidate"]) != checkpoint:
        raise ValueError("Fitted candidate digest differs")
    view = reachable(model)
    identity = candidate_identity(CandidateModel.model_validate(model["candidate"]))
    return {
        "model_digest": checkpoint,
        "active_latents": view["active_latents"],
        "active_dimension": len(view["active_states"]),
        "reach": view["public_edges"],
        "signed_topology": identity.topology_sha256,
        "functions": identity.functional_sha256,
        "wiring": fitted_wiring(model),
    }


def phase_b(history: Path, inventory: Path) -> list[dict]:
    """Use every planned row at the submitted six-visit cutoff."""
    receipt = sealed_read(history / "historical_summary.json")
    per_seed = history / "historical_per_seed.csv"
    data = sealed_read(inventory)
    if (
        receipt["artifact_sha256"] != PHASE_B_HISTORY
        or receipt["source_inventory"] != PHASE_B_INVENTORY
        or data["artifact_sha256"] != PHASE_B_INVENTORY
        or digest(per_seed) != receipt["table_digests"]["historical_per_seed"]
    ):
        raise ValueError("Wrong historical source")
    with per_seed.open() as f:
        raw = list(csv.DictReader(f))
    rows = []
    for row in raw:
        available = row["structure_status"] == "available"
        model = data["models"][row["fit_model_id"]] if available else None
        features = describe(model, row["lowered_model_sha256"]) if available else {}
        roles = json.loads(row["role_status"]) if available else None
        if available and (
            features["active_latents"] != json.loads(row["active_latents"])
            or features["reach"] != json.loads(row["public_edges"])
        ):
            raise ValueError("Different fit or different reachability definition")
        rows.append(
            {
                "cohort": "submitted_six_visits",
                "cell": row["cell"],
                "prompt": row["prompt"],
                "arm": row["configuration"],
                "seed": int(row["seed"]),
                "task": row["task"],
                "status": row["structure_status"],
                "validation": (
                    float(row["validation_nmse"]) if row["validation_nmse"] else None
                ),
                "roles": roles,
                "public": (
                    row["training_sha256"],
                    row["validation_sha256"],
                    row["prompt_sha256"],
                ),
                **features,
            }
        )
    return rows


def phase_c(root: Path) -> list[dict]:
    """Inspect verified constructions from the September one-visit eight-cell pilot."""
    summary = json.loads((root / "summary.json").read_text())
    plan = sealed_read(root / "plan.json")
    if (
        summary["plan_sha256"] != PHASE_C_PLAN
        or plan["artifact_sha256"] != PHASE_C_PLAN
    ):
        raise ValueError("Wrong Phase C pilot")
    planned = {t["task_id"]: t for t in plan["tasks"]}
    if set(planned) != {r["task_id"] for r in summary["rows"]} or len(
        summary["rows"]
    ) != len(planned):
        raise ValueError("Incomplete Phase C roster")
    rows = []
    for row in summary["rows"]:
        task = planned[row["task_id"]]
        if any(row[k] != task[k] for k in ("benchmark_id", "arm", "seed")):
            raise ValueError("Different Phase C scientific arm")
        folder = root / "results" / row["task_id"]
        proposal, fit_row, assessment = (
            sealed_read(folder / f"{name}.json")
            for name in ("proposal", "fitted", "assessment")
        )
        if len({r["identity"] for r in (proposal, fit_row, assessment)}) != 1:
            raise ValueError("Crossed Phase C identities")
        fit = fit_row.get("fit") or {}
        if fit_row["status"] != "complete":
            features, roles = {}, None
        else:
            init = proposal["bundle"]["initialization"]
            model = {
                "candidate": init["candidate"],
                "parameters": fit["parameters"],
                "lowered_context": init["context"],
            }
            assessed = assessment["model"]
            if (
                assessment["status"] != "assessed"
                or assessed["candidate"] != model["candidate"]
                or assessed["parameters"] != model["parameters"]
                or fit["validation"]["normalized_mse"] != row["validation_nmse"]
            ):
                raise ValueError("Fit and tested candidate differ")
            features = describe(model, fit["lowered_candidate_sha256"])
            roles = {m["id"]: m["status"] for m in assessment["findings"]}
        rows.append(
            {
                "cohort": "phase_c_one_visit",
                "cell": row["benchmark_id"],
                "prompt": row["arm"],
                "arm": "one_visit",
                "seed": row["seed"],
                "task": row["task_id"],
                "status": fit_row["status"],
                "validation": row["validation_nmse"],
                "roles": roles,
                "public": (
                    fit.get("training_content_sha256"),
                    fit.get("validation_content_sha256"),
                ),
                **features,
            }
        )
    return rows


def paired(rows: list[dict]) -> list[dict]:
    """Keep missing seed pairs and keep pass, fail and unresolved distinct."""
    groups = defaultdict(dict)
    for r in rows:
        key = r["cohort"], r["cell"], r["prompt"], r["arm"]
        if r["seed"] in groups[key]:
            raise ValueError("Duplicate seed in an experimental condition")
        groups[key][r["seed"]] = r
    out = []
    for key, group in sorted(groups.items()):
        if set(group) != {0, 1}:
            raise ValueError("Missing planned seed")
        a, b = group[0], group[1]
        item = dict(zip(("cohort", "cell", "prompt", "arm"), key, strict=True))
        for r in (a, b):
            i = r["seed"]
            for k in (
                "task",
                "status",
                "validation",
                "model_digest",
                "active_dimension",
                "active_latents",
                "reach",
                "roles",
            ):
                v = r.get(k)
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, sort_keys=True, ensure_ascii=False)
                item[f"seed{i}_{k}"] = v
        if a["validation"] is not None and b["validation"] is not None:
            item["absolute_nmse_gap"] = abs(a["validation"] - b["validation"])
        item["comparable"] = "wiring" in a and "wiring" in b
        if item["comparable"]:
            if a["public"] != b["public"] or set(a["roles"]) != set(b["roles"]):
                raise ValueError("Different public evidence or functional tests")
            for r in (a, b):
                if not r["roles"] or set(r["roles"].values()) - {
                    "pass",
                    "fail",
                    "unresolved",
                }:
                    raise ValueError("Incomplete functional evidence")
            wiring, alignments = best_wiring(a["wiring"], b["wiring"])
            fractions = [
                list(r["roles"].values()).count("pass") / len(r["roles"])
                for r in (a, b)
            ]
            item.update(
                direct_wiring_jaccard=wiring,
                state_alignments=alignments,
                reach_jaccard=overlap(a["reach"], b["reach"]),
                same_signed_topology=a["signed_topology"] == b["signed_topology"],
                same_functions=a["functions"] == b["functions"],
                same_active_dimension=a["active_dimension"] == b["active_dimension"],
                same_latent_count=len(a["active_latents"]) == len(b["active_latents"]),
                same_test_outcomes=a["roles"] == b["roles"],
                seed0_confirmed_fraction=fractions[0],
                seed1_confirmed_fraction=fractions[1],
                absolute_compliance_gap_pp=100 * abs(fractions[0] - fractions[1]),
            )
        out.append(item)
    return out


def spread(values: list[float]) -> dict:
    """Unscaled MAD and median across the available matched pairs."""
    if not values:
        return {"n": 0, "median": None, "mad": None}
    m = median(values)
    return {"n": len(values), "median": m, "mad": median(abs(v - m) for v in values)}


def summarize(pairs: list[dict]) -> dict:
    eligible = [p for p in pairs if p["comparable"]]
    counts = {
        k: sum(p[k] for p in eligible)
        for k in (
            "same_active_dimension",
            "same_latent_count",
            "same_signed_topology",
            "same_functions",
            "same_test_outcomes",
        )
    }

    def stats(items):
        return {
            k: spread([p[k] for p in items if p.get(k) is not None])
            for k in (
                "direct_wiring_jaccard",
                "reach_jaccard",
                "absolute_nmse_gap",
                "absolute_compliance_gap_pp",
            )
        }

    same = [p for p in eligible if p["reach_jaccard"] == 1]
    different = [
        p for p in eligible if p["reach_jaccard"] is not None and p["reach_jaccard"] < 1
    ]
    return {
        "planned": len(pairs),
        "comparable": len(eligible),
        "counts": counts,
        "equal_public_reach": len(same),
        "test_vector_agreement_with_equal_reach": sum(
            p["same_test_outcomes"] for p in same
        ),
        "test_vector_agreement_with_changed_reach": sum(
            p["same_test_outcomes"] for p in different
        ),
        "equal_direct_wiring": sum(p["direct_wiring_jaccard"] == 1 for p in eligible),
        "paired_scores_available": stats(pairs)["absolute_nmse_gap"],
        "all_structurally_comparable": stats(eligible),
        "identical_public_reach": stats(same),
        "different_public_reach": stats(different),
    }


def run(history: Path, inventory: Path, pilot: Path, output: Path) -> dict:
    """Reconstruct both source cohorts and report each on its own denominator."""
    pairs = paired(phase_b(history, inventory) + phase_c(pilot))
    output.mkdir(parents=True, exist_ok=True)
    result = {
        cohort: summarize([p for p in pairs if p["cohort"] == cohort])
        for cohort in sorted({p["cohort"] for p in pairs})
    }
    with _lock(output / "audit-lock"), TemporaryDirectory(dir=output) as temp:
        dest = output / "seed_pairs.csv"
        staged = Path(temp) / dest.name
        table(staged, pairs)
        signature = digest(staged)
        if dest.exists() and digest(dest) != signature:
            raise ValueError("Existing seed table differs")
        summary = sealed_write(
            output / "seed_summary.json",
            {
                "protocol": "iclr-archived-seed-wiring-1",
                "cohorts": result,
                "source_sha256": digest(Path(__file__)),
                "table_sha256": signature,
                "historical_inventory": PHASE_B_INVENTORY,
                "pilot_plan": PHASE_C_PLAN,
                "new_fits": 0,
                "new_provider_calls": 0,
                "test_data_opened": False,
            },
        )
        staged.replace(dest)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(run(args.history, args.inventory, args.pilot, args.output), indent=2)
    )
